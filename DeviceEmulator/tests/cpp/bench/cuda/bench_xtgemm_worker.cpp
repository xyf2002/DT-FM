#include <benchmark/benchmark.h>

#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "scheduler/gpu_worker.h"
#include "utils/logger.h"

// Helper: build a GemmArgs for a single NN SGEMM (no transpose)
static std::shared_ptr<GemmArgs> MakeNNGemmArgs(int m, int n, int k, float* a,
                                                float* b, float* c) {
  auto args = std::make_shared<GemmArgs>();
  args->group_size = 1;
  args->transa[0] = 'N';
  args->transb[0] = 'N';
  args->m[0] = m;
  args->n[0] = n;
  args->k[0] = k;
  args->alpha[0] = 1.0f;
  args->a[0] = a;
  args->lda[0] = m;
  args->b[0] = b;
  args->ldb[0] = k;
  args->beta[0] = 0.0f;
  args->c[0] = c;
  args->ldc[0] = m;
  return args;
}

// Pinned host memory RAII wrapper
struct PinnedBuffer {
  float* ptr = nullptr;
  size_t bytes = 0;

  PinnedBuffer() = default;

  explicit PinnedBuffer(size_t elems) : bytes(elems * sizeof(float)) {
    if (elems > 0) {
      cudaHostAlloc(reinterpret_cast<void**>(&ptr), bytes,
                    cudaHostAllocDefault);
      for (size_t i = 0; i < elems; i++) {
        ptr[i] = static_cast<float>(i % 1000) * 0.001f;
      }
    }
  }
  ~PinnedBuffer() {
    if (ptr) cudaFreeHost(ptr);
  }
  PinnedBuffer(const PinnedBuffer&) = delete;
  PinnedBuffer& operator=(const PinnedBuffer&) = delete;
  PinnedBuffer(PinnedBuffer&& o) noexcept : ptr(o.ptr), bytes(o.bytes) {
    o.ptr = nullptr;
    o.bytes = 0;
  }
  PinnedBuffer& operator=(PinnedBuffer&& o) noexcept {
    if (this != &o) {
      if (ptr) cudaFreeHost(ptr);
      ptr = o.ptr;
      bytes = o.bytes;
      o.ptr = nullptr;
      o.bytes = 0;
    }
    return *this;
  }
};

// ---------------------------------------------------------------------------
// BM_SingleWorker_Gemm: Latency of a single GEMM at various sizes
// ---------------------------------------------------------------------------
static void BM_SingleWorker_Gemm(benchmark::State& state) {
  static std::once_flag init_flag;
  std::call_once(init_flag, []() { InitLogger(); });

  int dim = state.range(0);
  int m = dim, n = dim, k = dim;
  size_t a_elems = m * k;
  size_t b_elems = k * n;
  size_t c_elems = m * n;

  PinnedBuffer h_A(a_elems);
  PinnedBuffer h_B(b_elems);
  PinnedBuffer h_C(c_elems);

  auto worker = std::make_shared<XtGemmWorker>(0, 1, 0, 512_MB);
  auto args = MakeNNGemmArgs(m, n, k, h_A.ptr, h_B.ptr, h_C.ptr);

  // Warmup (enqueue via worker thread where green ctx lives)
  worker->AddTask("warmup", [&]() { worker->RunXtGemm(args); });
  worker->WaitTaskDone("warmup");

  int iter = 0;
  for (auto _ : state) {
    std::string tid = "iter_" + std::to_string(iter++);
    worker->AddTask(tid, [&]() { worker->RunXtGemm(args); });
    worker->WaitTaskDone(tid);
  }

  // Report GFLOPS: 2*M*N*K / time
  double flops = 2.0 * m * n * k;
  state.counters["GFLOPS"] =
      benchmark::Counter(flops, benchmark::Counter::kIsIterationInvariantRate,
                         benchmark::Counter::kIs1000);

  worker->Stop();
}

BENCHMARK(BM_SingleWorker_Gemm)
    ->Arg(128)
    ->Arg(512)
    ->Arg(1024)
    ->Arg(2048)
    ->Unit(benchmark::kMicrosecond);

// ---------------------------------------------------------------------------
// BM_MultiWorker_Throughput: Total throughput with N concurrent workers
// ---------------------------------------------------------------------------
static void BM_MultiWorker_Throughput(benchmark::State& state) {
  static std::once_flag init_flag;
  std::call_once(init_flag, []() { InitLogger(); });

  int num_workers = state.range(0);
  const int dim = 512;
  const int m = dim, n = dim, k = dim;
  const size_t a_elems = m * k;
  const size_t b_elems = k * n;
  const size_t c_elems = m * n;

  // Create pinned buffers per worker
  struct WorkerData {
    PinnedBuffer h_A{0};
    PinnedBuffer h_B{0};
    PinnedBuffer h_C{0};
    std::shared_ptr<XtGemmWorker> worker;
    std::shared_ptr<GemmArgs> args;
  };
  std::vector<std::unique_ptr<WorkerData>> data;
  data.reserve(num_workers);

  for (int i = 0; i < num_workers; i++) {
    auto d = std::make_unique<WorkerData>();
    d->h_A = PinnedBuffer(a_elems);
    d->h_B = PinnedBuffer(b_elems);
    d->h_C = PinnedBuffer(c_elems);
    d->worker = std::make_shared<XtGemmWorker>(0, num_workers, i, 256_MB);
    d->args = MakeNNGemmArgs(m, n, k, d->h_A.ptr, d->h_B.ptr, d->h_C.ptr);
    data.push_back(std::move(d));
  }

  // Warmup (via task queue on worker thread)
  for (int i = 0; i < num_workers; i++) {
    data[i]->worker->AddTask("warmup_" + std::to_string(i), [&d = data[i]]() {
      d->worker->RunXtGemm(d->args);
    });
  }
  for (auto& d : data) {
    d->worker->WaitTaskDone();
  }

  int iter = 0;
  for (auto _ : state) {
    // Enqueue all workers concurrently
    for (int i = 0; i < num_workers; i++) {
      std::string tid =
          "iter" + std::to_string(iter) + "_w" + std::to_string(i);
      data[i]->worker->AddTask(
          tid, [&d = data[i]]() { d->worker->RunXtGemm(d->args); });
    }
    // Wait for all to complete
    for (auto& d : data) {
      d->worker->WaitTaskDone();
    }
    iter++;
  }

  // Report aggregate GFLOPS: num_workers * 2*M*N*K / time
  double total_flops = static_cast<double>(num_workers) * 2.0 * m * n * k;
  state.counters["GFLOPS"] = benchmark::Counter(
      total_flops, benchmark::Counter::kIsIterationInvariantRate,
      benchmark::Counter::kIs1000);

  for (auto& d : data) {
    d->worker->Stop();
  }
}

BENCHMARK(BM_MultiWorker_Throughput)
    ->Arg(1)
    ->Arg(2)
    ->Arg(4)
    ->Unit(benchmark::kMicrosecond);

// ---------------------------------------------------------------------------
// BM_StreamSync_vs_DeviceSync: Compare stream sync vs device sync
// ---------------------------------------------------------------------------
static void BM_StreamSync(benchmark::State& state) {
  static std::once_flag init_flag;
  std::call_once(init_flag, []() { InitLogger(); });

  const int dim = 512;
  PinnedBuffer h_A(dim * dim);
  PinnedBuffer h_B(dim * dim);
  PinnedBuffer h_C(dim * dim);

  auto worker = std::make_shared<XtGemmWorker>(0, 1, 0, 256_MB);
  auto args = MakeNNGemmArgs(dim, dim, dim, h_A.ptr, h_B.ptr, h_C.ptr);

  // Warmup via task queue
  worker->AddTask("warmup", [&]() { worker->RunXtGemm(args); });
  worker->WaitTaskDone("warmup");

  int iter = 0;
  for (auto _ : state) {
    std::string tid = "ss_" + std::to_string(iter++);
    worker->AddTask(tid, [&]() { worker->RunXtGemm(args); });
    worker->WaitTaskDone(tid);
  }

  state.counters["GFLOPS"] = benchmark::Counter(
      2.0 * dim * dim * dim, benchmark::Counter::kIsIterationInvariantRate,
      benchmark::Counter::kIs1000);

  worker->Stop();
}

BENCHMARK(BM_StreamSync)->Unit(benchmark::kMicrosecond);

static void BM_DeviceSync(benchmark::State& state) {
  static std::once_flag init_flag;
  std::call_once(init_flag, []() { InitLogger(); });

  const int dim = 512;
  PinnedBuffer h_A(dim * dim);
  PinnedBuffer h_B(dim * dim);
  PinnedBuffer h_C(dim * dim);

  auto worker = std::make_shared<XtGemmWorker>(0, 1, 0, 256_MB);
  auto args = MakeNNGemmArgs(dim, dim, dim, h_A.ptr, h_B.ptr, h_C.ptr);

  // Warmup via task queue
  worker->AddTask("warmup", [&]() { worker->RunXtGemm(args); });
  worker->WaitTaskDone("warmup");

  int iter = 0;
  for (auto _ : state) {
    std::string tid = "ds_" + std::to_string(iter++);
    worker->AddTask(tid, [&]() {
      worker->RunXtGemm(args);
      cudaDeviceSynchronize();  // Additional device-wide sync
    });
    worker->WaitTaskDone(tid);
  }

  state.counters["GFLOPS"] = benchmark::Counter(
      2.0 * dim * dim * dim, benchmark::Counter::kIsIterationInvariantRate,
      benchmark::Counter::kIs1000);

  worker->Stop();
}

BENCHMARK(BM_DeviceSync)->Unit(benchmark::kMicrosecond);
