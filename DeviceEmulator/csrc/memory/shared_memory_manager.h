#ifndef SHARED_MEMORY_MANAGER_H
#define SHARED_MEMORY_MANAGER_H

#include <pthread.h>
#include <stdatomic.h>
#include <stddef.h>
#include <stdint.h>

#define MAX_TASKS 1024
#define QUEUE_SIZE 1024
#define SHM_NAME "/sgemm_shm"  // Shared memory name is now a constant

typedef struct {
  atomic_int head;
  atomic_int tail;
  atomic_int size;
  atomic_int initialization_flag;
  int queue[QUEUE_SIZE];
} task_queue_t;

typedef struct {
  atomic_int flag;
  size_t offset;
} task_meta_data_t;

typedef struct {
  pthread_mutex_t offset_mutex;
  size_t current_offset;
  task_queue_t task_queue;
  task_meta_data_t meta_data[MAX_TASKS];
} shared_memory_t;

extern shared_memory_t* shared_mem_ptr;

// Function declarations
shared_memory_t* initialize_shared_memory(size_t shm_size);
shared_memory_t* attach_shared_memory(size_t shm_size);  // Attach shared memory
void destroy_shared_memory(shared_memory_t* shared_mem_ptr, size_t shm_size);
size_t get_shared_memory_size();
void log_message(const char* message);
void init_task_queue(task_queue_t* task_queue);
int dequeue_task(task_queue_t* task_queue);

#endif  // SHARED_MEMORY_MANAGER_H
