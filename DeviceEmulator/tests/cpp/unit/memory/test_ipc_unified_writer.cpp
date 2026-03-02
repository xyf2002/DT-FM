#include <cuda_runtime.h>

#include <fstream>
#include <iostream>

#define CUDA_CHECK(call)                                               \
  do {                                                                 \
    cudaError_t err = call;                                            \
    if (err != cudaSuccess) {                                          \
      std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at " \
                << __FILE__ << ":" << __LINE__ << std::endl;           \
      std::exit(err);                                                  \
    }                                                                  \
  } while (0)

void writeIpcHandleToFile(const cudaIpcMemHandle_t& ipcHandle,
                          const std::string& filename) {
  std::ofstream file(filename, std::ios::binary);
  if (!file) {
    std::cerr << "Failed to open file for writing IPC handle." << std::endl;
    std::exit(EXIT_FAILURE);
  }
  file.write(reinterpret_cast<const char*>(&ipcHandle),
             sizeof(cudaIpcMemHandle_t));
  file.close();
}

int main() {
  const size_t size = 1024 * sizeof(int);
  int* unifiedMem;
  cudaIpcMemHandle_t ipcHandle;

  // Allocate Unified Memory
  CUDA_CHECK(cudaMallocManaged(&unifiedMem, size));

  // Initialize the memory
  for (int i = 0; i < 1024; ++i) {
    unifiedMem[i] = i;
  }

  // Create an IPC handle for the Unified Memory
  CUDA_CHECK(cudaIpcGetMemHandle(&ipcHandle, unifiedMem));

  // Write the IPC handle to a file
  writeIpcHandleToFile(ipcHandle, "ipc_handle.dat");

  // Wait for user input to simulate that the process is waiting for the other
  // process to complete
  std::cout << "IPC handle created and shared via file, press Enter to exit..."
            << std::endl;
  std::cin.get();

  // Clean up
  CUDA_CHECK(cudaFree(unifiedMem));

  return 0;
}
