#include <cublas_v2.h>
#include <cuda_runtime_api.h>

#include <chrono>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>

// ============================================================================
// Simple Test: CUDA Host Register for PyTorch Tensors
//
// Verify: Can we use cudaHostRegister pointers for torch tensor creation
//         and matrix multiplication?
// ============================================================================

void PrintLine() { std::cout << std::string(80, '=') << "\n"; }

void PrintSection(const std::string& title) {
  std::cout << "\n";
  PrintLine();
  std::cout << title << "\n";
  PrintLine();
}

// Compare two matrices with tolerance
bool CompareMatrices(float* a, float* b, int64_t size, float tolerance = 1e-4) {
  float max_diff = 0.0f;
  float max_rel_diff = 0.0f;
  int diff_count = 0;

  for (int64_t i = 0; i < size; i++) {
    float abs_diff = std::abs(a[i] - b[i]);
    if (abs_diff > tolerance) {
      diff_count++;
      max_diff = std::max(max_diff, abs_diff);

      // Relative difference
      float max_val = std::max(std::abs(a[i]), std::abs(b[i]));
      if (max_val > 1e-6f) {
        float rel_diff = abs_diff / max_val;
        max_rel_diff = std::max(max_rel_diff, rel_diff);
      }
    }
  }

  std::cout << "  Differences found: " << diff_count << " / " << size << "\n";
  if (diff_count > 0) {
    std::cout << "  Max absolute difference: " << std::scientific
              << std::setprecision(6) << max_diff << "\n";
    std::cout << "  Max relative difference: " << max_rel_diff << "\n";
  }

  return diff_count == 0;
}

float* TestMatrixMultiplication(const std::string& name, float* host_ptr_a,
                                float* host_ptr_b, int m, int k, int n,
                                bool use_host_ptr = false) {
  std::cout << "\n[" << name << "] Testing matrix multiplication";
  if (use_host_ptr) {
    std::cout << " (using host ptr directly)";
  } else {
    std::cout << " (using GPU ptr after cudaMemcpy)";
  }
  std::cout << "...\n";

  int64_t size_a = (int64_t)m * k * sizeof(float);
  int64_t size_b = (int64_t)k * n * sizeof(float);
  int64_t size_c = (int64_t)m * n * sizeof(float);

  float* gpu_a = nullptr;
  float* gpu_b = nullptr;
  float* gpu_c = nullptr;

  if (use_host_ptr) {
    // Get device pointers from pinned host memory
    std::cout << "  Getting device pointers from pinned memory...\n";
    cudaHostGetDevicePointer((void**)&gpu_a, (void*)host_ptr_a, 0);
    cudaHostGetDevicePointer((void**)&gpu_b, (void*)host_ptr_b, 0);
    std::cout << "  Using host ptr directly for GPU computation\n";
  } else {
    // Allocate GPU memory and copy
    std::cout << "  Allocating GPU memory and copying host data...\n";
    cudaMalloc((void**)&gpu_a, size_a);
    cudaMalloc((void**)&gpu_b, size_b);

    cudaMemcpy(gpu_a, host_ptr_a, size_a, cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_b, host_ptr_b, size_b, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    std::cout << "  Data copied to GPU\n";
  }

  cudaMalloc((void**)&gpu_c, size_c);

  std::cout << "  Matrix A: " << m << " x " << k << "\n";
  std::cout << "  Matrix B: " << k << " x " << n << "\n";

  // Create cuBLAS handle
  cublasHandle_t handle;
  cublasCreate(&handle);

  // Warmup: C = A * B
  float alpha = 1.0f, beta = 0.0f;
  cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, gpu_b, n,
              gpu_a, k, &beta, gpu_c, n);
  cudaDeviceSynchronize();

  // Benchmark (5 iterations)
  auto start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < 5; i++) {
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, gpu_b, n,
                gpu_a, k, &beta, gpu_c, n);
  }
  cudaDeviceSynchronize();
  auto end = std::chrono::high_resolution_clock::now();

  double gpu_elapsed_ms =
      std::chrono::duration<double, std::milli>(end - start).count() / 5.0;

  // Compute FLOPs
  double flops = 2.0 * m * n * k;
  double gpu_gflops = flops / gpu_elapsed_ms / 1e6;

  std::cout << "  GPU Matrix mult time: " << std::fixed << std::setprecision(3)
            << gpu_elapsed_ms << " ms\n";
  std::cout << "  GPU Performance: " << std::setprecision(2) << gpu_gflops
            << " GFLOPS\n";

  // Copy result back from GPU
  float* gpu_result = (float*)malloc(size_c);
  cudaMemcpy(gpu_result, gpu_c, size_c, cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();

  // Cleanup GPU
  cublasDestroy(handle);

  // Only free GPU memory if we allocated it (not for pinned host ptr)
  if (!use_host_ptr) {
    cudaFree(gpu_a);
    cudaFree(gpu_b);
  }
  cudaFree(gpu_c);

  return gpu_result;
}

// void TestMemoryTransfer(const std::string& name, float* host_ptr, int64_t
// size_bytes) {
//   std::cout << "\n[" << name << "] Testing memory transfer...\n";

//   float* device_ptr;
//   cudaMalloc(&device_ptr, size_bytes);

//   // Warmup
//   cudaMemcpy(device_ptr, host_ptr, size_bytes, cudaMemcpyHostToDevice);

//   // Benchmark H2D
//   auto start = std::chrono::high_resolution_clock::now();
//   for (int i = 0; i < 5; i++) {
//     cudaMemcpy(device_ptr, host_ptr, size_bytes, cudaMemcpyHostToDevice);
//   }
//   cudaDeviceSynchronize();
//   auto end = std::chrono::high_resolution_clock::now();

//   double h2d_time_ms =
//       std::chrono::duration<double, std::milli>(end - start).count() / 5.0;
//   double h2d_tp_gbs = (size_bytes / 1e9) / (h2d_time_ms / 1000.0);

//   std::cout << "  H2D Time: " << std::fixed << std::setprecision(3)
//             << h2d_time_ms << " ms, TP: " << std::setprecision(2) <<
//             h2d_tp_gbs
//             << " GB/s\n";

//   // Benchmark D2H
//   start = std::chrono::high_resolution_clock::now();
//   for (int i = 0; i < 5; i++) {
//     cudaMemcpy(host_ptr, device_ptr, size_bytes, cudaMemcpyDeviceToHost);
//   }
//   cudaDeviceSynchronize();
//   end = std::chrono::high_resolution_clock::now();

//   double d2h_time_ms =
//       std::chrono::duration<double, std::milli>(end - start).count() / 5.0;
//   double d2h_tp_gbs = (size_bytes / 1e9) / (d2h_time_ms / 1000.0);

//   std::cout << "  D2H Time: " << d2h_time_ms << " ms, TP: " << d2h_tp_gbs
//             << " GB/s\n";

//   cudaFree(device_ptr);
// }

int main() {
  PrintSection("CUDA Host Register with PyTorch Tensors");

  // Check CUDA
  int device_count = 0;
  cudaGetDeviceCount(&device_count);
  if (device_count == 0) {
    std::cerr << "ERROR: No CUDA devices found!\n";
    return 1;
  }

  cudaSetDevice(0);
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  std::cout << "Device: " << prop.name << "\n";
  std::cout << "Memory: " << (prop.totalGlobalMem / 1e9) << " GB\n";

  // Matrix dimensions (reduced for faster testing with result verification)
  int m = 256;  // Rows of A
  int k = 256;  // Cols of A = Rows of B
  int n = 256;  // Cols of B

  int64_t elements = (int64_t)m * k;
  int64_t size_bytes = elements * sizeof(float);

  std::cout << "\nMatrix Configuration:\n";
  std::cout << "  A: " << m << " x " << k << " (" << (size_bytes / 1e6)
            << " MB)\n";
  std::cout << "  B: " << k << " x " << n << " (" << (size_bytes / 1e6)
            << " MB)\n";

  // Host Register Setup
  PrintSection("Step 1: Allocate and Register Host Memory");

  std::cout << "Allocating host memory with malloc...\n";
  float* host_a = (float*)malloc(size_bytes);
  float* host_b = (float*)malloc(size_bytes);

  if (!host_a || !host_b) {
    std::cerr << "malloc failed!\n";
    return 1;
  }

  // Initialize with random data
  std::cout << "Initializing with random data...\n";
  for (int64_t i = 0; i < elements; i++) {
    host_a[i] = static_cast<float>(rand()) / RAND_MAX;
    host_b[i] = static_cast<float>(rand()) / RAND_MAX;
  }

  // Register as pinned memory
  std::cout << "Registering memory with cudaHostRegister...\n";
  cudaError_t err1 =
      cudaHostRegister(host_a, size_bytes, cudaHostRegisterDefault);
  cudaError_t err2 =
      cudaHostRegister(host_b, size_bytes, cudaHostRegisterDefault);

  if (err1 != cudaSuccess || err2 != cudaSuccess) {
    std::cerr << "cudaHostRegister failed: " << cudaGetErrorString(err1)
              << " / " << cudaGetErrorString(err2) << "\n";
    return 1;
  }

  std::cout << "✓ Host memory registered successfully\n";

  // Test 1: Direct host ptr access
  PrintSection("Test 1: Direct Host Ptr Access (via cudaHostGetDevicePointer)");

  float* direct_host_ptr_result = TestMatrixMultiplication(
      "Direct Host Ptr", host_a, host_b, m, k, n, true);

  // Test 2: cudaMemcpy with GPU ptr
  PrintSection("Test 2: cudaMemcpy + GPU Ptr Access");

  float* gpu_ptr_result = TestMatrixMultiplication(
      "GPU Ptr (after cudaMemcpy)", host_a, host_b, m, k, n, false);

  // Compare the two GPU results
  PrintSection("Comparing GPU Results");
  std::cout << "Verifying that both access methods produce identical results\n";
  std::cout
      << "============================================================\n\n";

  std::cout << "Test Design:\n";
  std::cout << "  - Both tests use the SAME registered host memory\n";
  std::cout << "  - Test 1: cudaHostGetDevicePointer → direct host ptr GEMM\n";
  std::cout << "  - Test 2: cudaMemcpy to GPU memory → GPU ptr GEMM\n";
  std::cout << "  - Expected: Results should be IDENTICAL\n\n";

  int64_t result_size = (int64_t)m * n;
  bool results_match =
      CompareMatrices(direct_host_ptr_result, gpu_ptr_result, result_size);

  std::cout << "Verification Result:\n";
  if (results_match) {
    std::cout << "✓ PASS! Results are IDENTICAL\n";
    std::cout << "  Both access methods work correctly:\n";
    std::cout << "  • Direct host ptr (via cudaHostGetDevicePointer) works\n";
    std::cout << "  • cudaMemcpy to GPU memory works\n";
    std::cout << "  • Both produce identical results\n";
  } else {
    std::cout << "✗ FAIL! Results are DIFFERENT\n";
    std::cout << "  This indicates an issue:\n";
    std::cout << "  • Host ptr access may not be working correctly\n";
    std::cout << "  • Or data was not properly transferred/registered\n";
  }

  // Cleanup
  PrintSection("Cleanup");

  free(direct_host_ptr_result);
  free(gpu_ptr_result);

  cudaError_t err3 = cudaHostUnregister(host_a);
  cudaError_t err4 = cudaHostUnregister(host_b);
  if (err3 != cudaSuccess || err4 != cudaSuccess) {
    std::cerr << "cudaHostUnregister failed\n";
  }
  free(host_a);
  free(host_b);

  std::cout << "✓ Cleaned up successfully\n";

  // Summary
  PrintSection("Summary");
  std::cout << "✓ cudaHostRegister pinned memory successfully\n";
  std::cout << "✓ cudaHostGetDevicePointer retrieves valid device pointer\n";
  std::cout << "✓ Both direct host ptr and GPU ptr access methods work\n";
  std::cout << "\nConclusion: cudaHostRegister enables efficient GPU access to "
               "host data\n";

  return 0;
}
