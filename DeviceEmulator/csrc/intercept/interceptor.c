#if 0
#include "interceptor.h"
sgemm_type orig_sgemm = NULL;
shared_memory_t* shared_mem_ptr = NULL;  // Central shared memory pointer

FILE* log_file;
void initialize_logging() {
  if (log_file == NULL) {
    char log_filename[256];
    snprintf(log_filename, sizeof(log_filename), "%sprocess_%d_sgemm_log.txt",
             LOG_DIR, getpid());
    log_file = fopen(log_filename, "a");
    if (log_file == NULL) {
      perror("fopen");
      exit(EXIT_FAILURE);
    }
  }
}

void log_matrix_column_major(const char* matrix_name, const float* matrix,
                             int rows, int cols) {
  char message[1024];
  snprintf(message, sizeof(message),
           "Matrix %s in column-major order:", matrix_name);
  log_message(message);

  for (int j = 0; j < cols; ++j) {
    char column_values[1024] = "";
    for (int i = 0; i < rows; ++i) {
      char value_str[32];
      snprintf(value_str, sizeof(value_str), "%f ",
               matrix[j * rows + i]);  // Access element in column-major order
      strncat(column_values, value_str,
              sizeof(column_values) - strlen(column_values) - 1);
    }
    log_message(column_values);
  }
}

void log_matrix_with_transpose(const char* matrix_name, const float* matrix,
                               int rows, int cols, bool is_transposed) {
  char message[1024];
  snprintf(message, sizeof(message),
           "Matrix %s in column-major order:", matrix_name);
  log_message(message);

  // When matrix is transposed, swap rows and columns for printing
  if (is_transposed) {
    for (int i = 0; i < cols; ++i) {
      char row_values[1024] = "";
      for (int j = 0; j < rows; ++j) {
        char value_str[32];
        snprintf(value_str, sizeof(value_str), "%f ",
                 matrix[j * cols + i]);  // Access as transposed
        strncat(row_values, value_str,
                sizeof(row_values) - strlen(row_values) - 1);
      }
      log_message(row_values);
    }
  } else {
    for (int j = 0; j < cols; ++j) {
      char column_values[1024] = "";
      for (int i = 0; i < rows; ++i) {
        char value_str[32];
        snprintf(value_str, sizeof(value_str), "%f ",
                 matrix[j * rows + i]);  // Access as is
        strncat(column_values, value_str,
                sizeof(column_values) - strlen(column_values) - 1);
      }
      log_message(column_values);
    }
  }
}

void get_timestamp(char* buffer, size_t size) {
  time_t now = time(NULL);
  struct tm* local = localtime(&now);
  strftime(buffer, size, "%Y-%m-%d %H:%M:%S", local);
}

void log_message(const char* message) {
  if (log_file == NULL) {
    initialize_logging();
  }
  char timestamp[20];
  get_timestamp(timestamp, sizeof(timestamp));
  fprintf(log_file, "[%s] %s\n", timestamp, message);
  fflush(log_file);
}

void log_device_pointer(const char* matrix_name, const float* device_ptr) {
  char message[256];
  snprintf(message, sizeof(message), "Device pointer for %s: %p", matrix_name,
           (void*)device_ptr);
  log_message(message);
}

size_t get_shared_memory_size() {
  struct sysinfo info;
  sysinfo(&info);
  return info.totalram / 2;
}

int enqueue_task(task_queue_t* task_queue, int task_index) {
  int current_size = atomic_load(&task_queue->size);
  if (current_size >= QUEUE_SIZE) {
    return -1;
  }

  int tail = atomic_load(&task_queue->tail);
  task_queue->queue[tail] = task_index;

  atomic_store(&task_queue->tail, (tail + 1) % QUEUE_SIZE);
  atomic_fetch_add(&task_queue->size, 1);

  char message[200];
  snprintf(message, sizeof(message),
           "Process %d: Task enqueued successfully at queue position %d with "
           "task index %d. Queue size before enqueue: %d, after enqueue: %d.",
           getpid(), tail, task_index, current_size, current_size + 1);
  log_message(message);

  return 0;
}

void print_first_10_values(const char* matrix_name, const float* matrix,
                           int size) {
  char message[1024];

  snprintf(message, sizeof(message),
           "First 10 values of matrix %s (at %p): ", matrix_name,
           (void*)matrix);

  for (int i = 0; i < 20 && i < size; i++) {
    char value_str[32];
    snprintf(value_str, sizeof(value_str), "%f ", matrix[i]);
    strncat(message, value_str, sizeof(message) - strlen(message) - 1);
  }

  log_message(message);
}

void free_task(sgemm_args_t* task) {
  free((void*)task->a);
  free((void*)task->b);

  free(task->c);
  free(task);
}

int dummy_dequeue_task(task_queue_t* task_queue,
                       shared_memory_t* shared_mem_ptr) {
  int current_size = atomic_load(&task_queue->size);
  if (current_size == 0) {
    char message[100];
    snprintf(message, sizeof(message),
             "Process %d: Queue is empty. No task to dequeue.", getpid());
    log_message(message);
    return -1;
  }

  int head = atomic_load(&task_queue->head);
  int task_index_from_queue = task_queue->queue[head];

  size_t actual_offset =
      shared_mem_ptr->meta_data[task_index_from_queue].offset;

  char debug_message[256];
  snprintf(debug_message, sizeof(debug_message),
           "Process %d: Dequeue - Index from queue: %d, Actual offset in "
           "shared memory: %zu.",
           getpid(), task_index_from_queue, actual_offset);
  log_message(debug_message);

  if (actual_offset >= sizeof(shared_memory_t) &&
      actual_offset + sizeof(sgemm_args_t) <= get_shared_memory_size()) {
    sgemm_args_t* task = (sgemm_args_t*)((char*)shared_mem_ptr + actual_offset);

    if (task && task_index_from_queue >= 0 &&
        task_index_from_queue < MAX_TASKS) {
      char message[512];
      snprintf(message, sizeof(message),
               "Process %d: Dummy dequeued task at queue position %d with task "
               "index %d. "
               "Parameters: transa=%c, transb=%c, m=%d, n=%d, k=%d, "
               "alpha=%f, a=%p, lda=%d, b=%p, ldb=%d, beta=%f, c=%p, ldc=%d",
               getpid(), head, task_index_from_queue, task->transa,
               task->transb, task->m, task->n, task->k, task->alpha,
               (void*)task->a, task->lda, (void*)task->b, task->ldb, task->beta,
               (void*)task->c, task->ldc);
      log_message(message);

      log_device_pointer("Matrix A", task->a);
      log_device_pointer("Matrix B", task->b);
      log_device_pointer("Matrix C", task->c);

    } else {
      char error_message[256];
      snprintf(error_message, sizeof(error_message),
               "Process %d: Error - Invalid task data or task index %d at "
               "queue position %d.",
               getpid(), task_index_from_queue, head);
      log_message(error_message);
      return -1;
    }

  } else {
    // Log an error if the offset is out of bounds
    char error_message[256];
    snprintf(error_message, sizeof(error_message),
             "Process %d: Error - actual offset %zu out of bounds for task "
             "index %d.",
             getpid(), actual_offset, task_index_from_queue);
    log_message(error_message);
    return -1;
  }

  // Update the queue
  atomic_store(&task_queue->head, (head + 1) % QUEUE_SIZE);
  atomic_fetch_sub(&task_queue->size, 1);

  return 0;
}

void print_first_10_values_column_major(const char* label, float* matrix,
                                        int rows, int cols) {
  printf("%s:\n", label);

  int count = 0;
  for (int j = 0; j < cols; ++j) {    // Iterate over columns
    for (int i = 0; i < rows; ++i) {  // Iterate over rows within each column
      printf("%f ",
             matrix[j * rows + i]);  // Access element in column-major order
      count++;
      if (count >= 10) {
        printf("\n");
        return;
      }
    }
  }
  printf("\n");
}

void load_sgemm() {
  if (!orig_sgemm) {
    void* handle_lib = dlopen("libmkl_rt.so", RTLD_LAZY);
    if (!handle_lib) {
      char message[256];
      snprintf(message, sizeof(message), "Error loading MKL library: %s",
               dlerror());
      log_message(message);
      return;
    }
    orig_sgemm = (sgemm_type)dlsym(handle_lib, "sgemm_");
    if (!orig_sgemm) {
      char message[256];
      snprintf(message, sizeof(message), "Error loading original sgemm_: %s",
               dlerror());
      log_message(message);
      return;
    }
  }
}

bool ensure_shared_memory_initialized(size_t* shm_size) {
  if (!shared_mem_ptr) {
    *shm_size = get_shared_memory_size();
    shared_mem_ptr = attach_shared_memory(*shm_size);

    if (!shared_mem_ptr) {
      log_message("Error: Shared memory is not initialized.");
      return false;
    }
  } else {
    *shm_size = get_shared_memory_size();
  }

  return true;
}

bool get_cuda_memory_info(size_t* freeMem, size_t* totalMem) {
  cudaError_t cudaStatus = cudaMemGetInfo(freeMem, totalMem);

  if (cudaStatus != cudaSuccess) {
    log_message("Error: Unable to retrieve CUDA memory information.");
    return false;
  }

  return true;
}

bool sgemm_task_size_and_offset(const char* transa, const char* transb,
                                const int* m, const int* n, const int* k,
                                const int* lda, const int* ldb, const int* ldc,
                                size_t shm_size, size_t* task_size,
                                size_t* offset, size_t* size_a, size_t* size_b,
                                size_t* size_c) {
  *size_a = (*transa == 'N' || *transa == 'n') ? (*lda) * (*k) * sizeof(float)
                                               : (*lda) * (*m) * sizeof(float);

  *size_b = (*transb == 'N' || *transb == 'n') ? (*ldb) * (*n) * sizeof(float)
                                               : (*ldb) * (*k) * sizeof(float);

  *size_c = (*ldc) * (*n) * sizeof(float);

  *task_size = sizeof(sgemm_args_t) + *size_a + *size_b + *size_c;

  *offset = atomic_fetch_add(&shared_mem_ptr->current_offset, *task_size);

  if (*offset + *task_size > shm_size) {
    log_message("Not enough shared memory for task. Skipping task.");
    return false;
  }

  return true;
}

void sgemm_write_task_to_shared_memory(
    const char* transa, const char* transb, const int* m, const int* n,
    const int* k, const float* alpha, const float* a, const int* lda,
    const float* b, const int* ldb, const float* beta, float* c, const int* ldc,
    size_t offset, size_t size_a, size_t size_b, size_t size_c, size_t index) {
  char param_log[512];
  snprintf(param_log, sizeof(param_log),
           "Process %d: Parameters before writing to shared memory - "
           "transa=%c, transb=%c, m=%d, n=%d, k=%d, alpha=%f, a=%p, lda=%d, "
           "b=%p, ldb=%d, beta=%f, c=%p, ldc=%d",
           getpid(), *transa, *transb, *m, *n, *k, *alpha, (void*)a, *lda,
           (void*)b, *ldb, *beta, (void*)c, *ldc);

  char timestamp[20];
  get_timestamp(timestamp, sizeof(timestamp));
  fprintf(log_file, "[%s] %s\n", timestamp, param_log);
  fflush(log_file);

  sgemm_args_t* task_ptr = (sgemm_args_t*)((char*)shared_mem_ptr + offset);
  task_ptr->transa = *transa;
  task_ptr->transb = *transb;
  task_ptr->m = *m;
  task_ptr->n = *n;
  task_ptr->k = *k;
  task_ptr->alpha = *alpha;

  task_ptr->a = (float*)((char*)task_ptr + sizeof(sgemm_args_t));
  for (int col = 0; col < (*transa == 'N' || *transa == 'n' ? *k : *m); col++) {
    memcpy((float*)(task_ptr->a + col * (*lda)), a + col * (*lda),
           (*transa == 'N' || *transa == 'n' ? *m : *k) * sizeof(float));
  }
  task_ptr->lda = *lda;

  task_ptr->b = (float*)((char*)task_ptr->a + size_a);
  for (int col = 0; col < (*transb == 'N' || *transb == 'n' ? *n : *k); col++) {
    memcpy((float*)(task_ptr->b + col * (*ldb)), b + col * (*ldb),
           (*transb == 'N' || *transb == 'n' ? *k : *n) * sizeof(float));
  }
  task_ptr->ldb = *ldb;

  task_ptr->c = (float*)((char*)task_ptr->b + size_b);
  for (int col = 0; col < *n; col++) {
    memcpy((float*)(task_ptr->c + col * (*ldc)), c + col * (*ldc),
           (*m) * sizeof(float));
  }
  task_ptr->ldc = *ldc;

  shared_mem_ptr->meta_data[index].offset = offset;
  atomic_store(&shared_mem_ptr->meta_data[index].flag, WRITTEN);
}

bool enqueue_and_wait_for_completion(size_t index) {
  if (enqueue_task(&shared_mem_ptr->task_queue, index) == -1) {
    char message[200];
    snprintf(
        message, sizeof(message),
        "Process %d: Task queue is full. Task at index %zu was not enqueued.",
        getpid(), index);
    log_message(message);
    return false;
  }

  log_message("Task enqueued successfully.");

  while (atomic_load(&shared_mem_ptr->meta_data[index].flag) != COMPLETE) {
    usleep(1000);
  }

  log_message("Task is COMPLETE.");
  return true;
}

bool sgemm_write_back_results_to_matrix(float* c, const int* m, const int* n,
                                        const int* ldc, size_t size_a,
                                        size_t size_b, size_t shm_size,
                                        size_t index) {
  sgemm_args_t* task_ptr =
      (sgemm_args_t*)((char*)shared_mem_ptr +
                      shared_mem_ptr->meta_data[index].offset);
  task_ptr->a = (float*)((char*)task_ptr + sizeof(sgemm_args_t));
  task_ptr->b = (float*)((char*)task_ptr->a + size_a);
  task_ptr->c = (float*)((char*)task_ptr->b + size_b);

  if ((char*)task_ptr->c < (char*)shared_mem_ptr ||
      (char*)task_ptr->c + (*m) * (*n) * sizeof(float) >
          (char*)shared_mem_ptr + shm_size) {
    log_message("Error: task_ptr->c is outside of shared memory bounds.");
    return false;
  }

  log_message("Copying result from shared memory to original matrix C.");
  for (int col = 0; col < *n; col++) {
    memcpy(c + col * (*ldc), task_ptr->c + col * (*ldc), (*m) * sizeof(float));
  }

  atomic_store(&shared_mem_ptr->meta_data[index].flag, EMPTY);

  return true;
}

bool find_and_enqueue_task(const char* transa, const char* transb, const int* m,
                           const int* n, const int* k, const float* alpha,
                           const float* a, const int* lda, const float* b,
                           const int* ldb, const float* beta, float* c,
                           const int* ldc, size_t offset, size_t size_a,
                           size_t size_b, size_t size_c, size_t shm_size) {
  int slot_found = 0;
  for (size_t index = 0; index < MAX_TASKS; index++) {
    int expected = EMPTY;

    if (atomic_compare_exchange_strong(&shared_mem_ptr->meta_data[index].flag,
                                       &expected, WRITING)) {
      sgemm_write_task_to_shared_memory(transa, transb, m, n, k, alpha, a, lda,
                                        b, ldb, beta, c, ldc, offset, size_a,
                                        size_b, size_c, index);

      if (!enqueue_and_wait_for_completion(index)) {
        atomic_store(&shared_mem_ptr->meta_data[index].flag, EMPTY);
        return false;
      }

      if (!sgemm_write_back_results_to_matrix(c, m, n, ldc, size_a, size_b,
                                              shm_size, index)) {
        atomic_store(&shared_mem_ptr->meta_data[index].flag, EMPTY);
        return false;
      }
      slot_found = 1;
      break;
    }
  }

  if (!slot_found) {
    char message[100];
    snprintf(message, sizeof(message),
             "Process %d: No available slot found for intercepted SGEMM.",
             getpid());
    log_message(message);
    return false;
  }

  return true;
}

void sgemm_(const char* transa, const char* transb, const int* m, const int* n,
            const int* k, const float* alpha, const float* a, const int* lda,
            const float* b, const int* ldb, const float* beta, float* c,
            const int* ldc) {
  load_sgemm();
  if (!orig_sgemm) {
    return;
  }

  size_t shm_size;
  if (!ensure_shared_memory_initialized(&shm_size)) {
    return;
  }

  size_t freeMem, totalMem;
  if (!get_cuda_memory_info(&freeMem, &totalMem)) {
    return;
  }

  size_t task_size, offset;
  size_t size_a, size_b, size_c;
  if (!sgemm_task_size_and_offset(transa, transb, m, n, k, lda, ldb, ldc,
                                  shm_size, &task_size, &offset, &size_a,
                                  &size_b, &size_c)) {
    return;
  }

  if (!find_and_enqueue_task(transa, transb, m, n, k, alpha, a, lda, b, ldb,
                             beta, c, ldc, offset, size_a, size_b, size_c,
                             shm_size)) {
    return;
  }
}

int main() {
  initialize_logging();

  char message[256];
  snprintf(message, sizeof(message), "Process %d: Started processing.",
           getpid());
  log_message(message);

  size_t shm_size = get_shared_memory_size();
  snprintf(message, sizeof(message),
           "Process %d: Calculated shared memory size: %zu bytes.", getpid(),
           shm_size);
  log_message(message);

  shared_mem_ptr = attach_shared_memory(shm_size);
  if (!shared_mem_ptr) {
    log_message("Error: Shared memory attachment failed.");
    return 1;
  }

  snprintf(message, sizeof(message),
           "Process %d: Shared memory attached successfully.", getpid());
  log_message(message);

  destroy_shared_memory(shared_mem_ptr, shm_size);

  snprintf(message, sizeof(message),
           "Process %d: Shared memory resources freed.", getpid());
  log_message(message);

  snprintf(message, sizeof(message), "Process %d: Finished processing.",
           getpid());
  log_message(message);

  fclose(log_file);
  return 0;
}
#endif
