#ifndef GPU_PROCESS_H
#define GPU_PROCESS_H

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <fcntl.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/sysinfo.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

#include "../memory/initialize_memory.h"

#define SHM_NAME "/sgemm_shm"
#define POLLING_INTERVAL 100000
#define LOG_FILE "/home/eren/DeviceEmulator/csrc/intercept/logs/gpu_process.log"
#define MAX_TASKS 1024

#define EMPTY 0
#define WRITING 1
#define WRITTEN 2
#define READING 3
#define EXECUTING 4
#define COMPLETE 5

const char* cublasGetErrorString(cublasStatus_t error);

typedef struct {
  char transa;
  char transb;
  int m;
  int n;
  int k;
  float alpha;
  const float* a;
  int lda;
  const float* b;
  int ldb;
  float beta;
  float* c;
  int ldc;
} sgemm_args_t;
#define MAX_GROUP_SIZE 1024

typedef struct {
  int group_count;
  int group_size;
  char transa_array[MAX_GROUP_SIZE];
  char transb_array[MAX_GROUP_SIZE];
  int m_array[MAX_GROUP_SIZE];
  int n_array[MAX_GROUP_SIZE];
  int k_array[MAX_GROUP_SIZE];
  float alpha_array[MAX_GROUP_SIZE];
  float* a_array[MAX_GROUP_SIZE];
  int lda_array[MAX_GROUP_SIZE];
  float* b_array[MAX_GROUP_SIZE];
  int ldb_array[MAX_GROUP_SIZE];
  float beta_array[MAX_GROUP_SIZE];
  float* c_array[MAX_GROUP_SIZE];
  int ldc_array[MAX_GROUP_SIZE];
} sgemm_batch_args_t;

char* get_timestamp();
int dequeue_task(task_queue_t* task_queue);
void log_message(const char* message);
size_t get_shared_memory_size();
void dummy_execute(sgemm_args_t* task, size_t offset, int task_index);
void real_execute(sgemm_args_t* task, size_t offset, int task_index);
void* gpu_process(void* arg);

#endif
