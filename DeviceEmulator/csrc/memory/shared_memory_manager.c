#include "/home/eren/DeviceEmulator/csrc/memory/shared_memory_manager.h"  // Include the correct header file

#include <fcntl.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/sysinfo.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

#define SHM_NAME "/sgemm_shm"
#define LOG_FILE \
  "/home/eren/DeviceEmulator/csrc/intercept/logs/shared_memory.log"
shared_memory_t* shared_mem_ptr = NULL;  // Central shared memory pointer

// Central shared memory pointer (actual definition)

size_t get_shared_memory_size() {
  struct sysinfo info;
  sysinfo(&info);
  return info.totalram / 2;  // Return half of the total system RAM
}

void log_message(const char* message) {
  FILE* log_file = fopen(LOG_FILE, "a");
  if (log_file) {
    time_t now = time(NULL);
    struct tm* local = localtime(&now);
    char timestamp[20];
    strftime(timestamp, sizeof(timestamp), "%Y-%m-%d %H:%M:%S", local);
    fprintf(log_file, "[%s] %s\n", timestamp, message);
    fclose(log_file);
  } else {
    perror("Failed to open log file");
  }
}

void init_task_queue(task_queue_t* task_queue) {
  atomic_store(&task_queue->head, 0);
  atomic_store(&task_queue->tail, 0);
  atomic_store(&task_queue->size, 0);

  log_message("Task queue initialized. Head, tail, and size set to 0.");
}

shared_memory_t* initialize_shared_memory(size_t shm_size) {
  log_message("Attempting to initialize shared memory.");

  if (shared_mem_ptr != NULL) {
    log_message("Shared memory already initialized.");
    return shared_mem_ptr;
  }

  char log_buffer[256];

  // Create or open the shared memory object
  int fd = shm_open(SHM_NAME, O_RDWR | O_CREAT, 0666);
  if (fd == -1) {
    perror("shm_open");
    log_message("Error: Failed to open shared memory.");
    return NULL;
  } else {
    snprintf(log_buffer, sizeof(log_buffer),
             "Shared memory object opened with file descriptor: %d", fd);
    log_message(log_buffer);
  }

  // Set the size of the shared memory object
  if (ftruncate(fd, shm_size) == -1) {
    perror("ftruncate");
    log_message("Error: Failed to set the size of shared memory.");
    close(fd);
    return NULL;
  } else {
    snprintf(log_buffer, sizeof(log_buffer),
             "Shared memory size set to: %zu bytes.", shm_size);
    log_message(log_buffer);
  }

  // Map the shared memory object into the process's address space
  shared_mem_ptr =
      mmap(NULL, shm_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  if (shared_mem_ptr == MAP_FAILED) {
    perror("mmap");
    log_message("Error: Failed to map shared memory.");
    close(fd);
    return NULL;
  } else {
    snprintf(log_buffer, sizeof(log_buffer),
             "Shared memory mapped at address: %p", (void*)shared_mem_ptr);
    log_message(log_buffer);
  }

  close(fd);
  log_message("Shared memory initialization complete.");

  init_task_queue(&shared_mem_ptr->task_queue);
  shared_mem_ptr->current_offset = sizeof(shared_memory_t);

  return shared_mem_ptr;
}

shared_memory_t* attach_shared_memory(size_t shm_size) {
  log_message("Attempting to attach to existing shared memory.");

  // Open the shared memory object (do not create if it doesn't exist)
  int fd = shm_open(SHM_NAME, O_RDWR, 0666);
  if (fd == -1) {
    perror("shm_open");
    log_message("Error: Failed to open shared memory.");
    return NULL;
  }

  log_message("Shared memory object opened successfully.");

  // Map the shared memory object into the address space
  shared_mem_ptr =
      mmap(NULL, shm_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  if (shared_mem_ptr == MAP_FAILED) {
    perror("mmap");
    log_message("Error: Failed to map shared memory.");
    close(fd);
    return NULL;
  }

  close(fd);

  char log_buffer[256];
  snprintf(log_buffer, sizeof(log_buffer),
           "Shared memory attached at address: %p", (void*)shared_mem_ptr);
  log_message(log_buffer);

  return shared_mem_ptr;
}

void destroy_shared_memory(shared_memory_t* shared_mem_ptr, size_t shm_size) {
  if (munmap(shared_mem_ptr, shm_size) == -1) {
    perror("munmap");
    log_message("Failed to unmap shared memory.");
  } else {
    log_message("Shared memory unmapped successfully.");
  }

  if (shm_unlink(SHM_NAME) == -1) {
    perror("shm_unlink");
    log_message("Failed to unlink shared memory.");
  } else {
    log_message("Shared memory unlinked successfully.");
  }
}
