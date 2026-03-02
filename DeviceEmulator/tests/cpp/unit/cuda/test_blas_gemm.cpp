#include <cublas_v2.h>
#include <cuda_runtime_api.h>
#include <mkl.h>
#include <torch/torch.h>

#include <chrono>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

static uint32_t A_SIZE = 1024 * 512;
static uint32_t B_SIZE = 512 * 2048;
static uint32_t C_SIZE = 1024 * 2048;

int main() {
  torch::Tensor tensor_a = torch::rand({1024, 512});
  torch::Tensor tensor_b = torch::rand({512, 2048});

  torch::Tensor tensor_c = torch::zeros({1024, 2048});

  tensor_c = tensor_a.mm(tensor_b);

  float* matrix_a = tensor_a.data_ptr<float>();
  float* matrix_b = tensor_b.data_ptr<float>();
  float* matrix_c = (float*)malloc(1024 * 2048 * sizeof(float));

  float alpha = 1.0f;
  float beta = 0.0f;

  // compute matrix multiplication using mkl
  auto start = std::chrono::high_resolution_clock::now();
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 1024, 2048, 512, alpha,
              matrix_a, 512, matrix_b, 2048, beta, matrix_c, 2048);
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed_seconds = end - start;
  std::cout << "MKL sgemm took " << elapsed_seconds.count() << " seconds."
            << std::endl;

  // check the result
  torch::Tensor tensor_c_check = torch::from_blob(matrix_c, {1024, 2048});
  if (torch::allclose(tensor_c, tensor_c_check)) {
    std::cout << "MKL sgemm result is correct." << std::endl;
  } else {
    std::cout << "MKL sgemm result is incorrect." << std::endl;
  }

  // compute matrix multiplication using cublas
  cublasHandle_t handle;
  cublasCreate(&handle);

  void* d_A;
  void* d_B;
  void* d_C;

  cudaMalloc(&d_A, 1024 * 512 * sizeof(float));
  cudaMalloc(&d_B, 512 * 2048 * sizeof(float));
  cudaMalloc(&d_C, 1024 * 2048 * sizeof(float));

  // cublasSetMatrix(512, 1024, sizeof(float), matrix_a, 1024, d_A, 1024);
  // cublasSetMatrix(2048, 512, sizeof(float), matrix_b, 2048, d_B, 2048);

  cudaMemcpy(d_A, matrix_a, 1024 * 512 * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, matrix_b, 512 * 2048 * sizeof(float), cudaMemcpyHostToDevice);

  start = std::chrono::high_resolution_clock::now();
  /*
  cublasStatus_t cublasSgemm(cublasHandle_t handle,
                         cublasOperation_t transa, cublasOperation_t transb,
                         int m, int n, int k,
                         const float           *alpha,
                         const float           *A, int lda,
                         const float           *B, int ldb,
                         const float           *beta,
                         float           *C, int ldc)
  */
  // cublas is column-major, while torch is row-major, perform C = A * B
  cublasSgemm_v2(handle, CUBLAS_OP_N, CUBLAS_OP_N, 2048, 1024, 512, &alpha,
                 (float*)d_B, 2048, (float*)d_A, 512, &beta, (float*)d_C, 2048);
  end = std::chrono::high_resolution_clock::now();
  elapsed_seconds = end - start;
  std::cout << "cuBLAS sgemm took " << elapsed_seconds.count() << " seconds."
            << std::endl;

  cudaMemcpy(matrix_c, d_C, 1024 * 2048 * sizeof(float),
             cudaMemcpyDeviceToHost);

  // check the result
  tensor_c_check = torch::from_blob(matrix_c, {1024, 2048});
  if (torch::allclose(tensor_c, tensor_c_check)) {
    std::cout << "cuBLAS sgemm result is correct." << std::endl;
  } else {
    std::cout << "cuBLAS sgemm result is incorrect." << std::endl;
  }
}
