// test copy speed with multiple async streams

#include <cuda_runtime_api.h>

#include <chrono>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#define MEM_SIZE 1024ULL * 1024 * 1024 * 4

int main() {
  // use malloc tyo allocate memory
  void* src1_ptr = malloc(MEM_SIZE);
  void* src2_ptr = malloc(MEM_SIZE);

  // allocate cuda memory
  void* cuda_ptr1 = nullptr;
  void* cuda_ptr2 = nullptr;

  cudaSetDevice(0);
  cudaMalloc(&cuda_ptr1, MEM_SIZE);
  cudaMalloc(&cuda_ptr2, MEM_SIZE);

  // test memory copy speed using two streams
  cudaStream_t stream1, stream2;
  cudaStreamCreateWithFlags(&stream1, cudaStreamNonBlocking);
  cudaStreamCreateWithFlags(&stream2, cudaStreamNonBlocking);

  std::vector<float> src1(MEM_SIZE / sizeof(float));
  std::vector<float> src2(MEM_SIZE / sizeof(float));

  for (size_t i = 0; i < src1.size(); i++) {
    src1[i] = i;
    src2[i] = i;
  }

  cudaMemset(cuda_ptr1, 0, MEM_SIZE);
  cudaMemset(cuda_ptr2, 0, MEM_SIZE);

  auto start = std::chrono::high_resolution_clock::now();
  cudaMemcpyAsync(cuda_ptr1, src1.data(), MEM_SIZE, cudaMemcpyHostToDevice,
                  stream1);
  cudaMemcpyAsync(cuda_ptr2, src2.data(), MEM_SIZE, cudaMemcpyHostToDevice,
                  stream2);

  cudaStreamSynchronize(stream1);
  cudaStreamSynchronize(stream2);

  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = end - start;
  std::cout << "Elapsed time: " << elapsed.count() << " s\n";
  // convert to GB/s
  std::cout << "Bandwidth: " << MEM_SIZE * 2 / 1e9 / elapsed.count()
            << " GB/s\n";

  // free memory
  free(src1_ptr);
  free(src2_ptr);

  cudaFree(cuda_ptr1);
  cudaFree(cuda_ptr2);

  cudaStreamDestroy(stream1);
  cudaStreamDestroy(stream2);

  return 0;
}
