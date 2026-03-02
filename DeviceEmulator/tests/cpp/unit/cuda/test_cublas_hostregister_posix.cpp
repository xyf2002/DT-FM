#include <cublas_v2.h>
#include <cuda_runtime_api.h>
#include <sys/mman.h>

#include <cstdlib>
#include <iostream>
#include <string>

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
    ptr[i] = static_cast<float>(i % 131) / 131.0f;
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

  const int m = 256;
  const int k = 256;
  const int n = 256;
  const size_t size_a = static_cast<size_t>(m) * k * sizeof(float);
  const size_t size_b = static_cast<size_t>(k) * n * sizeof(float);
  const size_t size_c = static_cast<size_t>(m) * n * sizeof(float);

  // posix_memalign path (similar to AlignedBufferPool)
  float* hA = nullptr;
  float* hB = nullptr;
  float* hC = nullptr;
  if (posix_memalign(reinterpret_cast<void**>(&hA), 4096, size_a) != 0 ||
      posix_memalign(reinterpret_cast<void**>(&hB), 4096, size_b) != 0 ||
      posix_memalign(reinterpret_cast<void**>(&hC), 4096, size_c) != 0) {
    std::cerr << "posix_memalign failed" << std::endl;
    return 1;
  }

  FillMatrix(hA, static_cast<int64_t>(m) * k);
  FillMatrix(hB, static_cast<int64_t>(k) * n);

  std::cout << "\nmlock buffers..." << std::endl;
  if (mlock(hA, size_a) != 0 || mlock(hB, size_b) != 0 ||
      mlock(hC, size_c) != 0) {
    std::cout << "mlock failed (non-fatal)" << std::endl;
  }

  // Register default (not mapped)
  cudaError_t err = cudaHostRegister(hA, size_a, cudaHostRegisterDefault);
  PrintCudaError("cudaHostRegisterDefault A", err);
  err = cudaHostRegister(hB, size_b, cudaHostRegisterDefault);
  PrintCudaError("cudaHostRegisterDefault B", err);
  err = cudaHostRegister(hC, size_c, cudaHostRegisterDefault);
  PrintCudaError("cudaHostRegisterDefault C", err);

  RunGemm("HostRegisterDefault + host ptr", hA, hB, hC, m, k, n);

  cudaHostUnregister(hA);
  cudaHostUnregister(hB);
  cudaHostUnregister(hC);

  // Register mapped + cudaHostGetDevicePointer
  cudaDeviceReset();
  cudaSetDeviceFlags(cudaDeviceMapHost);
  cudaGetDeviceProperties(&prop, 0);
  std::cout << "\nAfter cudaSetDeviceFlags: canMapHostMemory="
            << prop.canMapHostMemory << std::endl;

  err = cudaHostRegister(hA, size_a, cudaHostRegisterMapped);
  PrintCudaError("cudaHostRegisterMapped A", err);
  err = cudaHostRegister(hB, size_b, cudaHostRegisterMapped);
  PrintCudaError("cudaHostRegisterMapped B", err);
  err = cudaHostRegister(hC, size_c, cudaHostRegisterMapped);
  PrintCudaError("cudaHostRegisterMapped C", err);

  float* dA = nullptr;
  float* dB = nullptr;
  float* dC = nullptr;
  err = cudaHostGetDevicePointer(reinterpret_cast<void**>(&dA), hA, 0);
  PrintCudaError("cudaHostGetDevicePointer A", err);
  err = cudaHostGetDevicePointer(reinterpret_cast<void**>(&dB), hB, 0);
  PrintCudaError("cudaHostGetDevicePointer B", err);
  err = cudaHostGetDevicePointer(reinterpret_cast<void**>(&dC), hC, 0);
  PrintCudaError("cudaHostGetDevicePointer C", err);

  RunGemm("HostRegisterMapped + device ptr", dA, dB, dC, m, k, n);

  cudaHostUnregister(hA);
  cudaHostUnregister(hB);
  cudaHostUnregister(hC);

  munlock(hA, size_a);
  munlock(hB, size_b);
  munlock(hC, size_c);
  free(hA);
  free(hB);
  free(hC);

  return 0;
}