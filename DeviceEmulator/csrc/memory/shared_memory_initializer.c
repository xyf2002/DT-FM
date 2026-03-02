#include "/home/eren/DeviceEmulator/csrc/memory/shared_memory_initializer.h"

#include <stdio.h>
#include <stdlib.h>

int main(int argc, char* argv[]) {
  if (argc != 2) {
    fprintf(stderr, "Usage: %s <shm_size>\n", argv[0]);
    return EXIT_FAILURE;
  }

  // Parse the shared memory size from the command-line argument
  size_t shm_size = atol(argv[1]);
  if (shm_size == 0) {
    fprintf(stderr, "Invalid shared memory size.\n");
    return EXIT_FAILURE;
  }

  // Initialize shared memory
  shared_memory_t* shared_mem_ptr = initialize_shared_memory(shm_size);
  if (shared_mem_ptr == NULL) {
    fprintf(stderr, "Failed to initialize shared memory.\n");
    return EXIT_FAILURE;
  }

  printf("Shared memory initialized successfully. Size: %zu bytes.\n",
         shm_size);
  return EXIT_SUCCESS;
}
