// test  abdnwidth copy from shared memory to pin memory and back

#include <cuda_runtime_api.h>
#include <numaif.h>
#include <sys/mman.h>
#include <sys/shm.h>

#include <chrono>
#include <cstring>
#include <iostream>
#include <vector>

#include "memory/simd_copy.cpp"

#define MEM_SIZE 1024ULL * 1024 * 1024 * 4

int main() {
  int shm_id = shmget(IPC_PRIVATE, MEM_SIZE, IPC_CREAT | 0666);
  if (shm_id == -1) {
    std::cerr << "shmget failed" << std::endl;
    return 1;
  }

  void* shm_ptr = shmat(shm_id, nullptr, 0);
  if (shm_ptr == (void*)-1) {
    std::cerr << "shmat failed" << std::endl;
    return 1;
  }

  int numa_node = 0;  // Replace with the desired NUMA node number
  unsigned long nodemask = 1UL << numa_node;
  if (mbind(shm_ptr, MEM_SIZE, MPOL_BIND, &nodemask, sizeof(nodemask) * 8, 0) !=
      0) {
    std::cerr << "mbind failed" << std::endl;
    return 1;
  }

  void* pin_ptr;
  cudaError_t err = cudaHostAlloc(&pin_ptr, MEM_SIZE, cudaHostAllocDefault);
  if (err != cudaSuccess) {
    std::cerr << "cudaHostAlloc failed: " << cudaGetErrorString(err)
              << std::endl;
    return 1;
  }

  if (mbind(pin_ptr, MEM_SIZE, MPOL_BIND, &nodemask, sizeof(nodemask) * 8, 0) !=
      0) {
    std::cerr << "mbind failed" << std::endl;
    return 1;
  }

  // auto start = std::chrono::high_resolution_clock::now();
  // // cudaMemcpy(pin_ptr, shm_ptr, MEM_SIZE, cudaMemcpyHostToHost);
  // // memcpy(pin_ptr, shm_ptr, MEM_SIZE);
  // helper_mempcy_8((float*)pin_ptr, (float*)shm_ptr, MEM_SIZE /
  // sizeof(float)); auto end = std::chrono::high_resolution_clock::now();
  // std::chrono::duration<double> elapsed = end - start;
  // std::cout << "Elapsed time: " << elapsed.count() << " s\n";
  // std::cout << "Bandwidth: " << MEM_SIZE / 1e9 / elapsed.count() << "
  // GB/s\n";
  for (size_t i = 0; i < 10; i++) {
    auto start = std::chrono::high_resolution_clock::now();
    // cudaMemcpy(shm_ptr, pin_ptr, MEM_SIZE, cudaMemcpyHostToHost);
    // memcpy(shm_ptr, pin_ptr, MEM_SIZE);
    helper_memcpy_8((float*)shm_ptr, (float*)pin_ptr, MEM_SIZE / sizeof(float));
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Elapsed time: " << elapsed.count() << " s\n";
    std::cout << "Bandwidth: " << MEM_SIZE / 1e9 / elapsed.count() << " GB/s\n";
  }
}
