#include <cublas_v2.h>
#include <cuda_runtime_api.h>

#include <cmath>
#include <iostream>
#include <string>
#include <vector>

static void PrintCudaError(const std::string& prefix, cudaError_t err) {
  if (err == cudaSuccess) {
    std::cout << prefix << " OK" << std::endl;
    return;
  }
  std::cout << prefix << " failed: " << cudaGetErrorString(err) << std::endl;
}

static void PrintCublasStatus(const std::string& prefix, cublasStatus_t st) {
  std::cout << prefix << " status=" << static_cast<int>(st) << std::endl;
}

static void PrintPtrAttrs(const std::string& name, const void* ptr) {
  cudaPointerAttributes attr;
  cudaError_t err = cudaPointerGetAttributes(&attr, ptr);
  if (err != cudaSuccess) {
    std::cout << "  " << name
              << " cudaPointerGetAttributes failed: " << cudaGetErrorString(err)
              << std::endl;
    cudaGetLastError();
    return;
  }
#if CUDART_VERSION >= 10000
  std::cout << "  " << name << " type=" << attr.type
            << " device=" << attr.device << std::endl;
#else
  std::cout << "  " << name << " memoryType=" << attr.memoryType
            << " device=" << attr.device << std::endl;
#endif
}

static void FillMatrix(float* ptr, int64_t count) {
  for (int64_t i = 0; i < count; ++i) {
    ptr[i] = static_cast<float>(i % 97) / 97.0f;
  }
}

static void RunGemm(const std::string& tag, float* A, float* B, float* C, int m,
                    int k, int n) {
  std::cout << "\n[" << tag << "]\n";
  PrintPtrAttrs("A", A);
  PrintPtrAttrs("B", B);
  PrintPtrAttrs("C", C);

  cublasHandle_t handle;
  cublasStatus_t st = cublasCreate(&handle);
  if (st != CUBLAS_STATUS_SUCCESS) {
    PrintCublasStatus("cublasCreate", st);
    return;
  }

  float alpha = 1.0f;
  float beta = 0.0f;
  st = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, B, n, A,
                   k, &beta, C, n);
  PrintCublasStatus("cublasSgemm", st);

  cudaError_t sync_err = cudaDeviceSynchronize();
  PrintCudaError("cudaDeviceSynchronize", sync_err);

  cublasDestroy(handle);
}

int main() {
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  std::cout << "Device: " << prop.name
            << " canMapHostMemory=" << prop.canMapHostMemory << std::endl;

  // Test 0: host alloc default, pass host pointer directly
  const int m = 256;
  const int k = 256;
  const int n = 256;
  const size_t size_a = static_cast<size_t>(m) * k * sizeof(float);
  const size_t size_b = static_cast<size_t>(k) * n * sizeof(float);
  const size_t size_c = static_cast<size_t>(m) * n * sizeof(float);

  float* hA = nullptr;
  float* hB = nullptr;
  float* hC = nullptr;

  cudaError_t err = cudaHostAlloc(reinterpret_cast<void**>(&hA), size_a,
                                  cudaHostAllocDefault);
  PrintCudaError("cudaHostAlloc A", err);
  err = cudaHostAlloc(reinterpret_cast<void**>(&hB), size_b,
                      cudaHostAllocDefault);
  PrintCudaError("cudaHostAlloc B", err);
  err = cudaHostAlloc(reinterpret_cast<void**>(&hC), size_c,
                      cudaHostAllocDefault);
  PrintCudaError("cudaHostAlloc C", err);

  FillMatrix(hA, static_cast<int64_t>(m) * k);
  FillMatrix(hB, static_cast<int64_t>(k) * n);
  RunGemm("HostAllocDefault + host ptr", hA, hB, hC, m, k, n);

  // Test 1: host alloc mapped + cudaHostGetDevicePointer
  cudaDeviceReset();
  cudaSetDeviceFlags(cudaDeviceMapHost);
  cudaGetDeviceProperties(&prop, 0);
  std::cout << "\nAfter cudaSetDeviceFlags: canMapHostMemory="
            << prop.canMapHostMemory << std::endl;

  float* hA2 = nullptr;
  float* hB2 = nullptr;
  float* hC2 = nullptr;
  err = cudaHostAlloc(reinterpret_cast<void**>(&hA2), size_a,
                      cudaHostAllocMapped);
  PrintCudaError("cudaHostAllocMapped A", err);
  err = cudaHostAlloc(reinterpret_cast<void**>(&hB2), size_b,
                      cudaHostAllocMapped);
  PrintCudaError("cudaHostAllocMapped B", err);
  err = cudaHostAlloc(reinterpret_cast<void**>(&hC2), size_c,
                      cudaHostAllocMapped);
  PrintCudaError("cudaHostAllocMapped C", err);

  FillMatrix(hA2, static_cast<int64_t>(m) * k);
  FillMatrix(hB2, static_cast<int64_t>(k) * n);

  float* dA = nullptr;
  float* dB = nullptr;
  float* dC = nullptr;
  err = cudaHostGetDevicePointer(reinterpret_cast<void**>(&dA), hA2, 0);
  PrintCudaError("cudaHostGetDevicePointer A", err);
  err = cudaHostGetDevicePointer(reinterpret_cast<void**>(&dB), hB2, 0);
  PrintCudaError("cudaHostGetDevicePointer B", err);
  err = cudaHostGetDevicePointer(reinterpret_cast<void**>(&dC), hC2, 0);
  PrintCudaError("cudaHostGetDevicePointer C", err);

  RunGemm("HostAllocMapped + device ptr", dA, dB, dC, m, k, n);

  cudaFreeHost(hA);
  cudaFreeHost(hB);
  cudaFreeHost(hC);
  cudaFreeHost(hA2);
  cudaFreeHost(hB2);
  cudaFreeHost(hC2);

  return 0;
}