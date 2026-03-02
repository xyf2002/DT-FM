#include <gtest/gtest.h>
#include <torch/torch.h>

#include <future>
#include <memory>
#include <string>
#include <vector>

#include "scheduler/gpu_worker.h"
#include "utils/logger.h"

// Helper: build a GemmArgs for a single column-major SGEMM
// C(m,n) = alpha * op(A)(m,k) * op(B)(k,n) + beta * C(m,n)
// where op(X) = X if trans='N', X^T if trans='T'
static std::shared_ptr<GemmArgs> MakeGemmArgs(char transa, char transb, int m,
                                              int n, int k, float alpha,
                                              const float* a, int lda,
                                              const float* b, int ldb,
                                              float beta, float* c, int ldc) {
  auto args = std::make_shared<GemmArgs>();
  args->group_size = 1;
  args->transa[0] = transa;
  args->transb[0] = transb;
  args->m[0] = m;
  args->n[0] = n;
  args->k[0] = k;
  args->alpha[0] = alpha;
  args->a[0] = a;
  args->lda[0] = lda;
  args->b[0] = b;
  args->ldb[0] = ldb;
  args->beta[0] = beta;
  args->c[0] = c;
  args->ldc[0] = ldc;
  return args;
}

// Compute reference GEMM using torch::mm
// cublas column-major: C = alpha * op(A) * op(B) + beta * C
//
// Column-major matrix M(rows, cols) with leading dim ld is stored
// as M[row + col * ld]. In torch row-major, this same memory is
// M^T[col + row * ld], i.e. torch::from_blob(data, {cols, ld})
// gives M^T when we take the first `rows` columns.
//
// Strategy: read all matrices as their transposes in torch,
// then use the identity:
//   C = alpha * op(A) * op(B) + beta * C  (column-major)
//   C^T = alpha * op(B)^T * op(A)^T + beta * C^T  (row-major)
//
// Returns the result as [m, n] (row-major view of column-major C).
static torch::Tensor ReferenceGemm(char transa, char transb, int m, int n,
                                   int k, float alpha, const float* a_data,
                                   int lda, const float* b_data, int ldb,
                                   float beta, const float* c_data, int ldc) {
  // Read A's transpose from column-major memory
  // If transa='N': A is (m x k) col-major, lda >= m
  //   memory = lda * k floats, torch sees as [k, lda], take [:, :m]
  //   => A^T [k, m], so op(A) = A => op(A)^T = A^T [k, m]
  // If transa='T': A is (k x m) col-major, lda >= k
  //   memory = lda * m floats, torch sees as [m, lda], take [:, :k]
  //   => A^T [m, k], op(A) = A^T => op(A)^T = A [k, m]... wait
  //   Actually op(A)^T = (A^T)^T = A [k, m] in col-major
  //   But in torch row-major view: [m, k] slice is A^T,
  //   transposing gives [k, m] = A. So op(A)^T = [m, k].slice.t()

  // Let's just construct op(A) directly as [m, k] torch tensor:
  torch::Tensor opA;
  if (transa == 'N' || transa == 'n') {
    // A col-major (m, k) with lda: memory [k, lda] row-major
    opA = torch::from_blob(const_cast<float*>(a_data), {k, lda}, torch::kFloat)
              .slice(1, 0, m)  // [k, m]
              .t()             // [m, k]
              .contiguous();
  } else {
    // A col-major (k, m) with lda: memory [m, lda] row-major
    // op(A) = A^T which is [m, k]
    opA = torch::from_blob(const_cast<float*>(a_data), {m, lda}, torch::kFloat)
              .slice(1, 0, k)  // [m, k]
              .contiguous();
  }

  torch::Tensor opB;
  if (transb == 'N' || transb == 'n') {
    // B col-major (k, n) with ldb: memory [n, ldb] row-major
    opB = torch::from_blob(const_cast<float*>(b_data), {n, ldb}, torch::kFloat)
              .slice(1, 0, k)  // [n, k]
              .t()             // [k, n]
              .contiguous();
  } else {
    // B col-major (n, k) with ldb: memory [k, ldb] row-major
    // op(B) = B^T which is [k, n]
    opB = torch::from_blob(const_cast<float*>(b_data), {k, ldb}, torch::kFloat)
              .slice(1, 0, n)  // [k, n]
              .contiguous();
  }

  // C col-major (m, n) with ldc: memory [n, ldc] row-major
  auto C_orig =
      torch::from_blob(const_cast<float*>(c_data), {n, ldc}, torch::kFloat)
          .slice(1, 0, m)  // [n, m]
          .t()             // [m, n]
          .contiguous();

  // result = alpha * op(A) * op(B) + beta * C
  return alpha * torch::mm(opA, opB) + beta * C_orig;
}

class XtGemmWorkerTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() { InitLogger(); }

  void SetUp() override {
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    if (device_count == 0) {
      GTEST_SKIP() << "No CUDA devices available";
    }
  }

  // Create a single worker on GPU 0 with 1 partition
  std::shared_ptr<XtGemmWorker> CreateSingleWorker(
      size_t buffer_size = 256_MB) {
    auto w = std::make_shared<XtGemmWorker>(0, 1, 0, buffer_size);
    return w;
  }

  // Run a simple NN GEMM (no transpose) and compare with torch
  void RunAndVerify(XtGemmWorker& worker, int m, int n, int k,
                    float alpha = 1.0f, float beta = 0.0f, char transa = 'N',
                    char transb = 'N', float atol = 1e-3f, float rtol = 1e-3f) {
    // Column-major layout: lda >= m for N, lda >= k for T
    int lda = (transa == 'N' || transa == 'n') ? m : k;
    int ldb = (transb == 'N' || transb == 'n') ? k : n;
    int ldc = m;

    // Size in elements for column-major storage
    size_t a_elems = (transa == 'N' || transa == 'n') ? lda * k : lda * m;
    size_t b_elems = (transb == 'N' || transb == 'n') ? ldb * n : ldb * k;
    size_t c_elems = ldc * n;

    // Allocate host memory (pinned for async memcpy)
    float *h_A, *h_B, *h_C;
    cudaHostAlloc(reinterpret_cast<void**>(&h_A), a_elems * sizeof(float),
                  cudaHostAllocDefault);
    cudaHostAlloc(reinterpret_cast<void**>(&h_B), b_elems * sizeof(float),
                  cudaHostAllocDefault);
    cudaHostAlloc(reinterpret_cast<void**>(&h_C), c_elems * sizeof(float),
                  cudaHostAllocDefault);

    // Fill with random data
    auto t_A = torch::rand({static_cast<long>(a_elems)});
    auto t_B = torch::rand({static_cast<long>(b_elems)});
    auto t_C = torch::rand({static_cast<long>(c_elems)});
    memcpy(h_A, t_A.data_ptr<float>(), a_elems * sizeof(float));
    memcpy(h_B, t_B.data_ptr<float>(), b_elems * sizeof(float));
    memcpy(h_C, t_C.data_ptr<float>(), c_elems * sizeof(float));

    // Compute reference
    auto ref = ReferenceGemm(transa, transb, m, n, k, alpha, h_A, lda, h_B, ldb,
                             beta, h_C, ldc);

    // Build GemmArgs and enqueue via worker thread (green ctx is there)
    auto args = MakeGemmArgs(transa, transb, m, n, k, alpha, h_A, lda, h_B, ldb,
                             beta, h_C, ldc);
    auto args_copy = args;  // prevent capture issues
    worker.AddTask("verify_gemm",
                   [&worker, args_copy]() { worker.RunXtGemm(args_copy); });
    worker.WaitTaskDone("verify_gemm");

    // Extract result from h_C (column-major) as [m, n]
    auto result = torch::from_blob(h_C, {n, ldc}, torch::kFloat)
                      .slice(1, 0, m)
                      .t()
                      .contiguous();

    EXPECT_TRUE(torch::allclose(result, ref, atol, rtol))
        << "GEMM mismatch: m=" << m << " n=" << n << " k=" << k
        << " transa=" << transa << " transb=" << transb << " alpha=" << alpha
        << " beta=" << beta
        << "\nmax diff=" << (result - ref).abs().max().item<float>();

    cudaFreeHost(h_A);
    cudaFreeHost(h_B);
    cudaFreeHost(h_C);
  }
};

TEST_F(XtGemmWorkerTest, SingleWorker_SmallGemm) {
  auto worker = CreateSingleWorker();
  RunAndVerify(*worker, 128, 128, 64);
  worker->Stop();
}

TEST_F(XtGemmWorkerTest, SingleWorker_LargeGemm) {
  auto worker = CreateSingleWorker(512_MB);
  RunAndVerify(*worker, 1024, 2048, 512);
  worker->Stop();
}

TEST_F(XtGemmWorkerTest, SingleWorker_TransposeA) {
  auto worker = CreateSingleWorker();
  RunAndVerify(*worker, 256, 128, 64, 1.0f, 0.0f, 'T', 'N');
  worker->Stop();
}

TEST_F(XtGemmWorkerTest, SingleWorker_TransposeB) {
  auto worker = CreateSingleWorker();
  RunAndVerify(*worker, 128, 256, 64, 1.0f, 0.0f, 'N', 'T');
  worker->Stop();
}

TEST_F(XtGemmWorkerTest, SingleWorker_NonSquare) {
  auto worker = CreateSingleWorker();
  RunAndVerify(*worker, 137, 251, 73);
  worker->Stop();
}

TEST_F(XtGemmWorkerTest, SingleWorker_AlphaBeta) {
  auto worker = CreateSingleWorker();
  RunAndVerify(*worker, 256, 256, 128, 2.5f, 0.7f);
  worker->Stop();
}

TEST_F(XtGemmWorkerTest, MultiWorker_Concurrent) {
  // Create 2 workers on GPU 0, each with half the SMs
  auto w0 = std::make_shared<XtGemmWorker>(0, 2, 0, 256_MB);
  auto w1 = std::make_shared<XtGemmWorker>(0, 2, 1, 256_MB);

  const int m = 512, n = 512, k = 256;
  const int lda = m, ldb = k, ldc = m;
  const size_t a_elems = lda * k;
  const size_t b_elems = ldb * n;
  const size_t c_elems = ldc * n;

  // Allocate host memory for two independent GEMMs
  float *h_A0, *h_B0, *h_C0;
  float *h_A1, *h_B1, *h_C1;
  cudaHostAlloc(reinterpret_cast<void**>(&h_A0), a_elems * sizeof(float),
                cudaHostAllocDefault);
  cudaHostAlloc(reinterpret_cast<void**>(&h_B0), b_elems * sizeof(float),
                cudaHostAllocDefault);
  cudaHostAlloc(reinterpret_cast<void**>(&h_C0), c_elems * sizeof(float),
                cudaHostAllocDefault);
  cudaHostAlloc(reinterpret_cast<void**>(&h_A1), a_elems * sizeof(float),
                cudaHostAllocDefault);
  cudaHostAlloc(reinterpret_cast<void**>(&h_B1), b_elems * sizeof(float),
                cudaHostAllocDefault);
  cudaHostAlloc(reinterpret_cast<void**>(&h_C1), c_elems * sizeof(float),
                cudaHostAllocDefault);

  auto t_A0 = torch::rand({static_cast<long>(a_elems)});
  auto t_B0 = torch::rand({static_cast<long>(b_elems)});
  auto t_A1 = torch::rand({static_cast<long>(a_elems)});
  auto t_B1 = torch::rand({static_cast<long>(b_elems)});
  memset(h_C0, 0, c_elems * sizeof(float));
  memset(h_C1, 0, c_elems * sizeof(float));
  memcpy(h_A0, t_A0.data_ptr<float>(), a_elems * sizeof(float));
  memcpy(h_B0, t_B0.data_ptr<float>(), b_elems * sizeof(float));
  memcpy(h_A1, t_A1.data_ptr<float>(), a_elems * sizeof(float));
  memcpy(h_B1, t_B1.data_ptr<float>(), b_elems * sizeof(float));

  // Compute reference
  auto ref0 = ReferenceGemm('N', 'N', m, n, k, 1.0f, h_A0, lda, h_B0, ldb, 0.0f,
                            h_C0, ldc);
  auto ref1 = ReferenceGemm('N', 'N', m, n, k, 1.0f, h_A1, lda, h_B1, ldb, 0.0f,
                            h_C1, ldc);

  // Enqueue both GEMMs concurrently via the worker task queues
  auto args0 = MakeGemmArgs('N', 'N', m, n, k, 1.0f, h_A0, lda, h_B0, ldb, 0.0f,
                            h_C0, ldc);
  auto args1 = MakeGemmArgs('N', 'N', m, n, k, 1.0f, h_A1, lda, h_B1, ldb, 0.0f,
                            h_C1, ldc);

  w0->AddTask("gemm0", [&]() { w0->RunXtGemm(args0); });
  w1->AddTask("gemm1", [&]() { w1->RunXtGemm(args1); });

  w0->WaitTaskDone("gemm0");
  w1->WaitTaskDone("gemm1");

  // Verify results
  auto res0 = torch::from_blob(h_C0, {n, ldc}, torch::kFloat)
                  .slice(1, 0, m)
                  .t()
                  .contiguous();
  auto res1 = torch::from_blob(h_C1, {n, ldc}, torch::kFloat)
                  .slice(1, 0, m)
                  .t()
                  .contiguous();

  EXPECT_TRUE(torch::allclose(res0, ref0, 1e-3, 1e-3))
      << "Worker 0 GEMM mismatch, max diff="
      << (res0 - ref0).abs().max().item<float>();
  EXPECT_TRUE(torch::allclose(res1, ref1, 1e-3, 1e-3))
      << "Worker 1 GEMM mismatch, max diff="
      << (res1 - ref1).abs().max().item<float>();

  w0->Stop();
  w1->Stop();

  cudaFreeHost(h_A0);
  cudaFreeHost(h_B0);
  cudaFreeHost(h_C0);
  cudaFreeHost(h_A1);
  cudaFreeHost(h_B1);
  cudaFreeHost(h_C1);
}

TEST_F(XtGemmWorkerTest, WorkerPool_RoundRobin) {
  const int workers_per_gpu = 2;
  const size_t buffer = 256_MB;
  XtGemmWorkerPool pool(workers_per_gpu, buffer,
                        WorkerSchedulingPolicy::kRoundRobinGemm);

  const int num_tasks = 4;
  const int m = 256, n = 256, k = 128;
  const int lda = m, ldb = k, ldc = m;
  const size_t a_elems = lda * k;
  const size_t b_elems = ldb * n;
  const size_t c_elems = ldc * n;

  struct TaskData {
    float* h_A;
    float* h_B;
    float* h_C;
    torch::Tensor ref;
  };
  std::vector<TaskData> tasks(num_tasks);

  for (int i = 0; i < num_tasks; i++) {
    cudaHostAlloc(reinterpret_cast<void**>(&tasks[i].h_A),
                  a_elems * sizeof(float), cudaHostAllocDefault);
    cudaHostAlloc(reinterpret_cast<void**>(&tasks[i].h_B),
                  b_elems * sizeof(float), cudaHostAllocDefault);
    cudaHostAlloc(reinterpret_cast<void**>(&tasks[i].h_C),
                  c_elems * sizeof(float), cudaHostAllocDefault);

    auto t_A = torch::rand({static_cast<long>(a_elems)});
    auto t_B = torch::rand({static_cast<long>(b_elems)});
    memcpy(tasks[i].h_A, t_A.data_ptr<float>(), a_elems * sizeof(float));
    memcpy(tasks[i].h_B, t_B.data_ptr<float>(), b_elems * sizeof(float));
    memset(tasks[i].h_C, 0, c_elems * sizeof(float));

    tasks[i].ref = ReferenceGemm('N', 'N', m, n, k, 1.0f, tasks[i].h_A, lda,
                                 tasks[i].h_B, ldb, 0.0f, tasks[i].h_C, ldc);

    auto args = MakeGemmArgs('N', 'N', m, n, k, 1.0f, tasks[i].h_A, lda,
                             tasks[i].h_B, ldb, 0.0f, tasks[i].h_C, ldc);
    pool.EnqueueGemm("task_" + std::to_string(i), args);
  }

  pool.WaitAll();

  for (int i = 0; i < num_tasks; i++) {
    auto result = torch::from_blob(tasks[i].h_C, {n, ldc}, torch::kFloat)
                      .slice(1, 0, m)
                      .t()
                      .contiguous();
    EXPECT_TRUE(torch::allclose(result, tasks[i].ref, 1e-3, 1e-3))
        << "Pool task " << i << " GEMM mismatch, max diff="
        << (result - tasks[i].ref).abs().max().item<float>();
  }

  for (int i = 0; i < num_tasks; i++) {
    cudaFreeHost(tasks[i].h_A);
    cudaFreeHost(tasks[i].h_B);
    cudaFreeHost(tasks[i].h_C);
  }
}
