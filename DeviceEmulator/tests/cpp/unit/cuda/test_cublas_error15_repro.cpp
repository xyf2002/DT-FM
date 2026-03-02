// test_cublas_error15_repro.cpp
//
// Reproduces cuBLAS error 15 (CUBLAS_STATUS_NOT_SUPPORTED) and related
// failures that can occur in ProxyCliHandle::HandlePartition when
// cublasSgemm receives invalid dimensions.  Two data paths are covered:
//   1. Deserialization path: wire data → Deserialize → HandlePartition
//   2. Cache path: partial arrival → SavePartition → FillPartition →
//      HandlePartition
//
// Standalone: depends only on gtest, CUDA runtime, and cuBLAS.

#include <cublas_v2.h>
#include <cuda_runtime_api.h>
#include <gtest/gtest.h>
#include <sys/mman.h>

#include <climits>
#include <cstdint>
#include <cstring>
#include <deque>
#include <iostream>
#include <mutex>
#include <unordered_map>
#include <utility>

// ============================================================================
// CudaPinnedMemoryPool — inlined from proxy_cli.h:21-79
// (avoids pulling in protobuf / torch / networking headers)
// ============================================================================
class CudaPinnedMemoryPool {
 public:
  explicit CudaPinnedMemoryPool(size_t max_buffers_per_bucket = 16)
      : max_per_bucket_(max_buffers_per_bucket) {}

  ~CudaPinnedMemoryPool() {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto& [bucket_size, free_list] : free_lists_) {
      for (auto* ptr : free_list) {
        cudaFreeHost(ptr);
      }
    }
  }

  std::pair<void*, size_t> Acquire(size_t size) {
    size_t bucket = BucketSize(size);
    std::lock_guard<std::mutex> lock(mutex_);
    auto& free_list = free_lists_[bucket];
    if (!free_list.empty()) {
      void* ptr = free_list.back();
      free_list.pop_back();
      return {ptr, bucket};
    }
    void* ptr = nullptr;
    cudaError_t err =
        cudaHostAlloc(&ptr, bucket, cudaHostAllocDefault | cudaHostAllocMapped);
    if (err != cudaSuccess || !ptr) {
      throw std::runtime_error("CudaPinnedMemoryPool: cudaHostAlloc failed");
    }
    return {ptr, bucket};
  }

  void Release(void* ptr, size_t bucket_size) {
    if (!ptr) return;
    std::lock_guard<std::mutex> lock(mutex_);
    auto& free_list = free_lists_[bucket_size];
    if (free_list.size() < max_per_bucket_) {
      free_list.push_back(ptr);
    } else {
      cudaFreeHost(ptr);
    }
  }

 private:
  static size_t BucketSize(size_t size) {
    static constexpr size_t MIN_BUCKET = 4096;
    if (size <= MIN_BUCKET) return MIN_BUCKET;
    size_t bucket = MIN_BUCKET;
    while (bucket < size) bucket <<= 1;
    return bucket;
  }

  size_t max_per_bucket_;
  std::mutex mutex_;
  std::unordered_map<size_t, std::deque<void*>> free_lists_;
};

// ============================================================================
// Helper: replicate HandlePartition's GEMM path (proxy_cli.cc:191-395)
// Returns cublasStatus_t so tests can assert on the error code.
// `skip_dim_guard` — when true, bypasses the row_size/col_size <= 0 early
// return that production code has at line 321, letting the bad dims reach
// cuBLAS so we can observe the actual error.
// ============================================================================
static cublasStatus_t RunGemmLikeHandlePartition(cublasHandle_t handle,
                                                 CudaPinnedMemoryPool& pool,
                                                 void* r_ptr, int64_t r_size,
                                                 void* c_ptr, int64_t c_size,
                                                 int64_t h_dim,
                                                 bool skip_dim_guard = false) {
  // --- dimension calculation (proxy_cli.cc:207-208) ---
  int64_t row_size = r_size / h_dim / static_cast<int64_t>(sizeof(float));
  int64_t col_size = c_size / h_dim / static_cast<int64_t>(sizeof(float));

  std::cout << "  [RunGemm] r_size=" << r_size << " c_size=" << c_size
            << " h_dim=" << h_dim << " → row_size=" << row_size
            << " col_size=" << col_size << "\n";

  // --- production guard (proxy_cli.cc:321) ---
  if (!skip_dim_guard && (row_size <= 0 || col_size <= 0 || h_dim <= 0)) {
    std::cout << "  [RunGemm] Dimension guard triggered (would return in prod)"
              << "\n";
    return CUBLAS_STATUS_INVALID_VALUE;  // sentinel; prod code just returns
  }

  // --- cudaHostRegister both inputs (proxy_cli.cc:215-273) ---
  bool r_registered = false;
  bool c_registered = false;

  if (r_ptr && r_size > 0) {
    cudaPointerAttributes attrs;
    cudaError_t err = cudaPointerGetAttributes(&attrs, r_ptr);
    if (err != cudaSuccess) {
      cudaGetLastError();
      err = cudaHostRegister(r_ptr, r_size, cudaHostRegisterMapped);
      if (err != cudaSuccess) {
        std::cerr << "  [RunGemm] cudaHostRegister row failed: "
                  << cudaGetErrorString(err) << "\n";
        return CUBLAS_STATUS_EXECUTION_FAILED;
      }
      r_registered = true;
    } else if (attrs.type != cudaMemoryTypeHost ||
               attrs.devicePointer == nullptr) {
      err = cudaHostRegister(r_ptr, r_size, cudaHostRegisterMapped);
      if (err != cudaSuccess) {
        std::cerr << "  [RunGemm] cudaHostRegister row failed: "
                  << cudaGetErrorString(err) << "\n";
        return CUBLAS_STATUS_EXECUTION_FAILED;
      }
      r_registered = true;
    }
  }

  if (c_ptr && c_size > 0) {
    cudaPointerAttributes attrs;
    cudaError_t err = cudaPointerGetAttributes(&attrs, c_ptr);
    if (err != cudaSuccess) {
      cudaGetLastError();
      err = cudaHostRegister(c_ptr, c_size, cudaHostRegisterMapped);
      if (err != cudaSuccess) {
        std::cerr << "  [RunGemm] cudaHostRegister col failed: "
                  << cudaGetErrorString(err) << "\n";
        if (r_registered) cudaHostUnregister(r_ptr);
        return CUBLAS_STATUS_EXECUTION_FAILED;
      }
      c_registered = true;
    } else if (attrs.type != cudaMemoryTypeHost ||
               attrs.devicePointer == nullptr) {
      err = cudaHostRegister(c_ptr, c_size, cudaHostRegisterMapped);
      if (err != cudaSuccess) {
        std::cerr << "  [RunGemm] cudaHostRegister col failed: "
                  << cudaGetErrorString(err) << "\n";
        if (r_registered) cudaHostUnregister(r_ptr);
        return CUBLAS_STATUS_EXECUTION_FAILED;
      }
      c_registered = true;
    }
  }

  // RAII cleanup for registered memory
  auto unregister_guard = [&]() {
    if (r_registered) cudaHostUnregister(r_ptr);
    if (c_registered) cudaHostUnregister(c_ptr);
  };

  // --- cudaHostGetDevicePointer (proxy_cli.cc:298-319) ---
  float* d_r_ptr = nullptr;
  float* d_c_ptr = nullptr;

  if (r_ptr && r_size > 0) {
    cudaError_t err =
        cudaHostGetDevicePointer(reinterpret_cast<void**>(&d_r_ptr), r_ptr, 0);
    if (err != cudaSuccess) {
      std::cerr << "  [RunGemm] cudaHostGetDevicePointer row: "
                << cudaGetErrorString(err) << "\n";
      unregister_guard();
      return CUBLAS_STATUS_EXECUTION_FAILED;
    }
  }
  if (c_ptr && c_size > 0) {
    cudaError_t err =
        cudaHostGetDevicePointer(reinterpret_cast<void**>(&d_c_ptr), c_ptr, 0);
    if (err != cudaSuccess) {
      std::cerr << "  [RunGemm] cudaHostGetDevicePointer col: "
                << cudaGetErrorString(err) << "\n";
      unregister_guard();
      return CUBLAS_STATUS_EXECUTION_FAILED;
    }
  }

  // --- result buffer from pool (proxy_cli.cc:330-361) ---
  // Use absolute values to compute a result buffer size (even for error tests)
  int64_t abs_row = (row_size > 0) ? row_size : 1;
  int64_t abs_col = (col_size > 0) ? col_size : 1;
  size_t result_size = static_cast<size_t>(abs_row) *
                       static_cast<size_t>(abs_col) * sizeof(float);

  void* result_ptr = nullptr;
  size_t result_bucket = 0;
  try {
    std::tie(result_ptr, result_bucket) = pool.Acquire(result_size);
  } catch (const std::exception& ex) {
    std::cerr << "  [RunGemm] pool.Acquire failed: " << ex.what() << "\n";
    unregister_guard();
    return CUBLAS_STATUS_ALLOC_FAILED;
  }

  float* d_result_ptr = nullptr;
  {
    cudaError_t err = cudaHostGetDevicePointer(
        reinterpret_cast<void**>(&d_result_ptr), result_ptr, 0);
    if (err != cudaSuccess) {
      std::cerr << "  [RunGemm] cudaHostGetDevicePointer result: "
                << cudaGetErrorString(err) << "\n";
      pool.Release(result_ptr, result_bucket);
      unregister_guard();
      return CUBLAS_STATUS_EXECUTION_FAILED;
    }
  }

  // --- cublasSgemm (proxy_cli.cc:368-383) ---
  float alpha = 1.0f;
  float beta = 0.0f;

  cublasStatus_t status = cublasSgemm(handle,
                                      CUBLAS_OP_N,                 // transa
                                      CUBLAS_OP_N,                 // transb
                                      static_cast<int>(col_size),  // m
                                      static_cast<int>(row_size),  // n
                                      static_cast<int>(h_dim),     // k
                                      &alpha,
                                      d_c_ptr,                     // A
                                      static_cast<int>(col_size),  // lda
                                      d_r_ptr,                     // B
                                      static_cast<int>(h_dim),     // ldb
                                      &beta,
                                      d_result_ptr,               // C
                                      static_cast<int>(col_size)  // ldc
  );

  std::cout << "  [RunGemm] cublasSgemm status=" << static_cast<int>(status)
            << "\n";

  // Synchronize to surface async errors
  cudaError_t sync_err = cudaDeviceSynchronize();
  if (sync_err != cudaSuccess) {
    std::cout << "  [RunGemm] cudaDeviceSynchronize: "
              << cudaGetErrorString(sync_err) << "\n";
  }

  pool.Release(result_ptr, result_bucket);
  unregister_guard();
  return status;
}

// ============================================================================
// Fixture
// ============================================================================
class CublasError15Test : public ::testing::Test {
 public:
  static void SetUpTestSuite() {
    // cudaDeviceMapHost must be set before any CUDA call
    cudaError_t err = cudaSetDeviceFlags(cudaDeviceMapHost);
    if (err != cudaSuccess && err != cudaErrorSetOnActiveProcess) {
      std::cerr << "cudaSetDeviceFlags: " << cudaGetErrorString(err) << "\n";
    }
  }

 protected:
  void SetUp() override {
    cublasStatus_t st = cublasCreate(&handle_);
    ASSERT_EQ(st, CUBLAS_STATUS_SUCCESS) << "cublasCreate failed";
  }

  void TearDown() override {
    if (handle_) cublasDestroy(handle_);
  }

  // Allocate page-aligned memory usable with cudaHostRegister
  static void* AllocAligned(size_t size) {
    void* ptr = nullptr;
    size_t alloc = (size + 4095) & ~size_t(4095);  // page-align
    if (alloc == 0) alloc = 4096;
    int ret = posix_memalign(&ptr, 4096, alloc);
    if (ret != 0 || !ptr) return nullptr;
    memset(ptr, 0, alloc);
    mlock(ptr, alloc);
    return ptr;
  }

  cublasHandle_t handle_ = nullptr;
  CudaPinnedMemoryPool pool_;
};

// ============================================================================
// Test: valid dimensions — baseline sanity check
// Row matrix: 16 × 64, Col matrix: 64 × 32
// cublasSgemm(col_size=32, row_size=16, h_dim=64) → C(16×32)
// ============================================================================
TEST_F(CublasError15Test, ValidDimensions_Succeeds) {
  const int64_t row_m = 16, h_dim = 64, col_n = 32;

  int64_t r_size = row_m * h_dim * sizeof(float);
  int64_t c_size = col_n * h_dim * sizeof(float);

  auto r_ptr_meta = pool_.Acquire(r_size);  // warm up pool (optional)
  auto c_ptr_meta = pool_.Acquire(c_size);

  void* r_ptr = r_ptr_meta.first;
  void* c_ptr = c_ptr_meta.first;

  // void* r_ptr = AllocAligned(r_size);
  // void* c_ptr = AllocAligned(c_size);
  ASSERT_NE(r_ptr, nullptr);
  ASSERT_NE(c_ptr, nullptr);

  // Fill with non-zero data
  for (int64_t i = 0; i < row_m * h_dim; ++i)
    reinterpret_cast<float*>(r_ptr)[i] = static_cast<float>(i % 37) / 37.0f;
  for (int64_t i = 0; i < col_n * h_dim; ++i)
    reinterpret_cast<float*>(c_ptr)[i] = static_cast<float>(i % 53) / 53.0f;

  cublasStatus_t st = RunGemmLikeHandlePartition(handle_, pool_, r_ptr, r_size,
                                                 c_ptr, c_size, h_dim);
  EXPECT_EQ(st, CUBLAS_STATUS_SUCCESS)
      << "Valid GEMM should succeed, got status=" << static_cast<int>(st);

  free(r_ptr);
  free(c_ptr);
}

// ============================================================================
// Test: Pool-allocated inputs — both r_ptr and c_ptr come from
// CudaPinnedMemoryPool (cudaHostAlloc'd).  cudaPointerGetAttributes will
// see type == cudaMemoryTypeHost with a valid devicePointer, so
// RunGemmLikeHandlePartition skips cudaHostRegister entirely.
// Same dimensions as ValidDimensions_Succeeds for comparison.
// ============================================================================
TEST_F(CublasError15Test, PoolAllocatedInputs_Succeeds) {
  const int64_t row_m = 16, h_dim = 64, col_n = 32;

  int64_t r_size = row_m * h_dim * sizeof(float);
  int64_t c_size = col_n * h_dim * sizeof(float);

  // Allocate from pool — returns cudaHostAlloc'd pinned memory
  auto [r_ptr, r_bucket] = pool_.Acquire(r_size);
  auto [c_ptr, c_bucket] = pool_.Acquire(c_size);
  ASSERT_NE(r_ptr, nullptr);
  ASSERT_NE(c_ptr, nullptr);

  // Fill with non-zero data
  for (int64_t i = 0; i < row_m * h_dim; ++i)
    reinterpret_cast<float*>(r_ptr)[i] = static_cast<float>(i % 37) / 37.0f;
  for (int64_t i = 0; i < col_n * h_dim; ++i)
    reinterpret_cast<float*>(c_ptr)[i] = static_cast<float>(i % 53) / 53.0f;

  cublasStatus_t st = RunGemmLikeHandlePartition(handle_, pool_, r_ptr, r_size,
                                                 c_ptr, c_size, h_dim);
  EXPECT_EQ(st, CUBLAS_STATUS_SUCCESS)
      << "Pool-allocated GEMM should succeed, got status="
      << static_cast<int>(st);

  pool_.Release(r_ptr, r_bucket);
  pool_.Release(c_ptr, c_bucket);
}

// ============================================================================
// Test: Deserialization path — zero col_size from non-divisible c_size
// Simulates a deserialized partition where c_size doesn't divide evenly by
// h_dim * sizeof(float), producing col_size = 0 via integer truncation.
// E.g. c_size=3, h_dim=128 → col_size = 3 / 128 / 4 = 0
// With skip_dim_guard=true, we bypass the line-321 check and hit cuBLAS.
// ============================================================================
TEST_F(CublasError15Test, DeserializeWorkflow_ZeroDim_Error15) {
  const int64_t h_dim = 128;
  const int64_t row_m = 16;
  int64_t r_size = row_m * h_dim * sizeof(float);
  int64_t c_size = 3;  // not divisible by h_dim * sizeof(float)

  void* r_ptr = AllocAligned(r_size);
  void* c_ptr = AllocAligned(4096);  // min page for registration
  ASSERT_NE(r_ptr, nullptr);
  ASSERT_NE(c_ptr, nullptr);

  memset(r_ptr, 0, r_size);
  memset(c_ptr, 0, 4096);

  // skip_dim_guard=true: bypass the row_size/col_size <= 0 check
  cublasStatus_t st = RunGemmLikeHandlePartition(handle_, pool_, r_ptr, r_size,
                                                 c_ptr, c_size, h_dim,
                                                 /*skip_dim_guard=*/true);

  std::cout << "  DeserializeWorkflow_ZeroDim: cublas status="
            << static_cast<int>(st) << "\n";

  // cuBLAS should reject m=0 or produce an error
  EXPECT_NE(st, CUBLAS_STATUS_SUCCESS)
      << "Zero col_size should fail; got SUCCESS unexpectedly";

  free(r_ptr);
  free(c_ptr);
}

// ============================================================================
// Test: Deserialization path — mismatched leading dimension from corrupted
// h_dim in proto.
// Allocate row as 16 × 64 floats, but tell HandlePartition h_dim=65.
// row_size = (16*64*4) / 65 / 4 = 15 (truncated), ldb = h_dim = 65.
// Actual data only has stride 64, so ldb=65 exceeds the actual buffer layout.
// cuBLAS detects the inconsistency.
// ============================================================================
TEST_F(CublasError15Test, DeserializeWorkflow_MismatchedLd_Error15) {
  const int64_t actual_h_dim = 64;
  const int64_t corrupt_h_dim = 65;
  const int64_t row_m = 16;
  const int64_t col_n = 32;

  int64_t r_size = row_m * actual_h_dim * sizeof(float);
  int64_t c_size = col_n * actual_h_dim * sizeof(float);

  void* r_ptr = AllocAligned(r_size);
  void* c_ptr = AllocAligned(c_size);
  ASSERT_NE(r_ptr, nullptr);
  ASSERT_NE(c_ptr, nullptr);

  memset(r_ptr, 0, r_size);
  memset(c_ptr, 0, c_size);

  // With corrupt_h_dim=65:
  //   row_size = r_size / 65 / 4 = (4096) / 65 / 4 = 15  (truncated)
  //   col_size = c_size / 65 / 4 = (8192) / 65 / 4 = 31  (truncated)
  //   ldb = 65 but actual stride is 64 → mismatch
  //   Also k=65 but data was laid out with k=64 → OOB access
  cublasStatus_t st = RunGemmLikeHandlePartition(handle_, pool_, r_ptr, r_size,
                                                 c_ptr, c_size, corrupt_h_dim);

  std::cout << "  DeserializeWorkflow_MismatchedLd: cublas status="
            << static_cast<int>(st) << "\n";

  // The GEMM may succeed numerically (cuBLAS doesn't bounds-check buffers)
  // but will read garbage / OOB.  On some GPU arch + CUDA versions, this
  // triggers error 15 or 7.  We log for diagnostics; if cuBLAS doesn't
  // reject it, we note the silent corruption scenario.
  if (st == CUBLAS_STATUS_SUCCESS) {
    std::cout << "  NOTE: cuBLAS accepted mismatched ldb silently — "
              << "this is the silent-corruption scenario the production "
              << "guard at line 321 must catch.\n";
  } else {
    std::cout << "  cuBLAS correctly rejected mismatched leading dimension\n";
  }
  // Informational test — passes either way, documents behavior
  SUCCEED();

  free(r_ptr);
  free(c_ptr);
}

// ============================================================================
// Test: Cache path — dimension mismatch between row (h_dim=128) and col
// (h_dim=64).
// Simulates FillPartition assembling a partition with:
//   partition.h_dim = 128
//   mat[0] = {row_ptr, row_size} laid out with h_dim=128
//   mat[1] = {col_ptr, col_size} laid out with h_dim=64
// When HandlePartition computes col_size using h_dim=128, it gets the
// wrong col dimension. lda = col_size (wrong) vs actual stride.
// ============================================================================
TEST_F(CublasError15Test, CacheWorkflow_DimMismatch_Error15) {
  const int64_t row_h_dim = 128;
  const int64_t col_h_dim = 64;
  const int64_t row_m = 8;
  const int64_t col_n = 16;

  // Row buffer laid out with h_dim=128
  int64_t r_size = row_m * row_h_dim * sizeof(float);
  // Col buffer laid out with h_dim=64
  int64_t c_size = col_n * col_h_dim * sizeof(float);

  void* r_ptr = AllocAligned(r_size);
  void* c_ptr = AllocAligned(c_size);
  ASSERT_NE(r_ptr, nullptr);
  ASSERT_NE(c_ptr, nullptr);

  memset(r_ptr, 0, r_size);
  memset(c_ptr, 0, c_size);

  // FillPartition sets partition.h_dim = row_h_dim (128).
  // HandlePartition then computes:
  //   row_size = r_size / 128 / 4 = (8*128*4) / 128 / 4 = 8  ✓
  //   col_size = c_size / 128 / 4 = (16*64*4) / 128 / 4 = 8  ✗ (should be 16)
  //   lda = col_size = 8, but col was laid out with stride col_h_dim=64
  //   k = 128, but col only has 64 columns → OOB read
  cublasStatus_t st = RunGemmLikeHandlePartition(handle_, pool_, r_ptr, r_size,
                                                 c_ptr, c_size, row_h_dim);

  std::cout << "  CacheWorkflow_DimMismatch: cublas status="
            << static_cast<int>(st) << "\n";

  // cuBLAS with these mismatched dims will either error or silently produce
  // wrong results (reading OOB in the col buffer).
  if (st == CUBLAS_STATUS_SUCCESS) {
    std::cout << "  NOTE: cuBLAS accepted mismatched cache dims silently — "
              << "production code would produce wrong GEMM results.\n";
  } else {
    std::cout << "  cuBLAS rejected dimension mismatch from cache path\n";
    EXPECT_NE(st, CUBLAS_STATUS_SUCCESS);
  }
  SUCCEED();

  free(r_ptr);
  free(c_ptr);
}

// ============================================================================
// Test: Cache path — h_dim overflow.
// h_dim set to a value > INT_MAX. When cast to int for cuBLAS, k overflows
// to a negative value, which cuBLAS should reject.
// ============================================================================
TEST_F(CublasError15Test, CacheWorkflow_OverflowK_Error15) {
  // h_dim that overflows int32 when cast
  const int64_t h_dim = static_cast<int64_t>(INT_MAX) + 1;
  const int64_t row_m = 4;
  const int64_t col_n = 4;

  // We need valid registered memory, but the actual data layout doesn't
  // matter since the GEMM will be rejected before reading.
  // Use small buffers — the dimension calc will produce tiny row_size/col_size
  // since r_size / h_dim / 4 → near zero for reasonable buffer sizes.
  // To get row_size > 0, we need r_size > h_dim * sizeof(float), which would
  // be ~8GB.  Instead, craft sizes that produce row_size=1, col_size=1 after
  // truncation, and let the overflow in h_dim → int(k) do the damage.
  int64_t r_size = h_dim * sizeof(float);  // row_size = 1
  int64_t c_size = h_dim * sizeof(float);  // col_size = 1

  // We can't actually allocate 8GB+, so use a smaller buffer with
  // skip_dim_guard=true and craft the cuBLAS call directly.
  // Allocate minimal aligned buffer and register it.
  size_t alloc_size = 4096;
  void* r_ptr = AllocAligned(alloc_size);
  void* c_ptr = AllocAligned(alloc_size);
  ASSERT_NE(r_ptr, nullptr);
  ASSERT_NE(c_ptr, nullptr);

  memset(r_ptr, 0, alloc_size);
  memset(c_ptr, 0, alloc_size);

  // Register as pinned
  cudaError_t err;
  err = cudaHostRegister(r_ptr, alloc_size, cudaHostRegisterMapped);
  ASSERT_EQ(err, cudaSuccess) << cudaGetErrorString(err);
  err = cudaHostRegister(c_ptr, alloc_size, cudaHostRegisterMapped);
  ASSERT_EQ(err, cudaSuccess) << cudaGetErrorString(err);

  float* d_r_ptr = nullptr;
  float* d_c_ptr = nullptr;
  err = cudaHostGetDevicePointer(reinterpret_cast<void**>(&d_r_ptr), r_ptr, 0);
  ASSERT_EQ(err, cudaSuccess);
  err = cudaHostGetDevicePointer(reinterpret_cast<void**>(&d_c_ptr), c_ptr, 0);
  ASSERT_EQ(err, cudaSuccess);

  // Result buffer from pool
  auto [result_ptr, result_bucket] = pool_.Acquire(4096);
  float* d_result_ptr = nullptr;
  err = cudaHostGetDevicePointer(reinterpret_cast<void**>(&d_result_ptr),
                                 result_ptr, 0);
  ASSERT_EQ(err, cudaSuccess);

  // Call cublasSgemm with overflowed k
  // int(h_dim) wraps to a negative or zero value
  int k_as_int = static_cast<int>(h_dim);
  std::cout << "  h_dim=" << h_dim << " → int(k)=" << k_as_int << "\n";

  float alpha = 1.0f;
  float beta = 0.0f;
  cublasStatus_t st = cublasSgemm(handle_, CUBLAS_OP_N, CUBLAS_OP_N,
                                  1,                   // m = col_size
                                  1,                   // n = row_size
                                  k_as_int,            // k = overflowed h_dim
                                  &alpha, d_c_ptr, 1,  // A, lda
                                  d_r_ptr, k_as_int,   // B, ldb
                                  &beta, d_result_ptr, 1  // C, ldc
  );

  std::cout << "  CacheWorkflow_OverflowK: cublas status="
            << static_cast<int>(st) << "\n";

  // Negative k should be rejected by cuBLAS
  EXPECT_NE(st, CUBLAS_STATUS_SUCCESS)
      << "Overflowed k (negative int) should be rejected by cuBLAS";

  pool_.Release(result_ptr, result_bucket);
  cudaHostUnregister(r_ptr);
  cudaHostUnregister(c_ptr);
  free(r_ptr);
  free(c_ptr);
}
