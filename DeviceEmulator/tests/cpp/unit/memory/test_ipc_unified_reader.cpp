#include <cuda_runtime_api.h>

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

void readIpcHandleFromFile(cudaIpcMemHandle_t& ipcHandle,
                           const std::string& filename) {
  std::ifstream file(filename, std::ios::binary);
  if (!file) {
    std::cerr << "Failed to open file for reading IPC handle." << std::endl;
    std::exit(EXIT_FAILURE);
  }
  file.read(reinterpret_cast<char*>(&ipcHandle), sizeof(cudaIpcMemHandle_t));
  file.close();
}

int main() {
  int* sharedMem;
  cudaIpcMemHandle_t ipcHandle;

  // Read the IPC handle from the file
  readIpcHandleFromFile(ipcHandle, "ipc_handle.dat");

  // Map the IPC memory handle to the current process's address space
  CUDA_CHECK(cudaIpcOpenMemHandle((void**)&sharedMem, ipcHandle,
                                  cudaIpcMemLazyEnablePeerAccess));

  // Use the shared memory
  for (int i = 0; i < 1024; ++i) {
    std::cout << "sharedMem[" << i << "] = " << sharedMem[i] << std::endl;
  }

  // Clean up
  CUDA_CHECK(cudaIpcCloseMemHandle(sharedMem));

  return 0;
}
