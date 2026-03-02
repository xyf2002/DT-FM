#pragma once

// #include "client.h"
#include "memory/caching_allocator.h"
#include "utils/logger.h"

extern "C" {

typedef void (*sgemm_type)(const char*, const char*, const int*, const int*,
                           const int*, const float*, const float*, const int*,
                           const float*, const int*, const float*, float*,
                           const int*);
extern sgemm_type orig_sgemm;
void sgemm_(const char* transa, const char* transb, const int* m, const int* n,
            const int* k, const float* alpha, const float* a, const int* lda,
            const float* b, const int* ldb, const float* beta, float* c,
            const int* ldc);
}
#if 0
#define DECLARE_INTERCEPTOR_TYPE(name, T)                                     \
  typedef void (*name##_type)(const char*, const char*, const int*,           \
                              const int*, const int*, const float*, const T*, \
                              const int*, const T*, const int*, const float*, \
                              T*, const int*)
#define DECLARE_INTERCEPTOR_PTR(name) extern name##_type orig_##name
#define DECLARE_INTERCEPTOR_FUNC(name, T)                                     \
  void name##_(const char* transa, const char* transb, const int* m,          \
               const int* n, const int* k, const float* alpha, const T* a,    \
               const int* lda, const T* b, const int* ldb, const float* beta, \
               T* c, const int* ldc)

#define DECLARE_INTERCEPTOR(name, T) \
  DECLARE_INTERCEPTOR_TYPE(name, T); \
  DECLARE_INTERCEPTOR_PTR(name);     \
  DECLARE_INTERCEPTOR_FUNC(name, T);

#define IMPL_INTERCEPTOR_PTR(name) name##_type orig_##name = NULL
#define IMPL_INTERCEPTOR_FUNC(name, T, LIB)                                    \
  void name##_(const char* transa, const char* transb, const int* m,           \
               const int* n, const int* k, const float* alpha, const T* a,     \
               const int* lda, const T* b, const int* ldb, const float* beta,  \
               T* c, const int* ldc) {                                         \
    if (!orig_##name) {                                                        \
      void* handle_lib = dlopen(LIB, RTLD_LAZY);                               \
      LOG_FATAL_IF(!handle_lib) << "Error loading MKL library: " << dlerror(); \
      orig_##name = (name##_type)dlsym(handle_lib, #name "_");                 \
      LOG_FATAL_IF(!orig_##name)                                               \
          << "Error loading original " << #name << "_: " << dlerror();         \
    }                                                                          \
    LOG_DEBUG(                                                                 \
        "Intercepted {}; args transa: {}, transb: {}, m: {}, n: {}, k: "       \
        "{}, alpha: {}, lda: {}, ldb: {}, beta: {}, ldc: {}",                  \
        #name, *transa, *transb, *m, *n, *k, *alpha, *lda, *ldb, *beta, *ldc); \
    GemmArgs args = {.group_size = 1};                                         \
    args.transa = *transa;                                                     \
    args.transb = *transb;                                                     \
    args.m = *m;                                                               \
    args.n = *n;                                                               \
    args.k = *k;                                                               \
    args.alpha = *alpha;                                                       \
    args.a = a;                                                                \
    args.lda = *lda;                                                           \
    args.b = b;                                                                \
    args.ldb = *ldb;                                                           \
    args.beta = *beta;                                                         \
    args.c = c;                                                                \
    args.ldc = *ldc;                                                           \
    TaskExecution(args);                                                       \
  }

#define IMPL_INTERCEPTOR(name, T, LIB) \
  IMPL_INTERCEPTOR_PTR(name);          \
  IMPL_INTERCEPTOR_FUNC(name, T, LIB);
#endif

// enum class CPUDataType { kFloat, kHalf };
// enum class GPUDataType { kFloat, kHalf, kBFloat };

#define MAX_GEMM_DIM 5  // FIXME: very unlikely the dimension will exceed 5
struct GemmArgs {
  char transa[MAX_GEMM_DIM];
  char transb[MAX_GEMM_DIM];
  int m[MAX_GEMM_DIM];
  int n[MAX_GEMM_DIM];
  int k[MAX_GEMM_DIM];
  float alpha[MAX_GEMM_DIM];
  const float* a[MAX_GEMM_DIM];  // FIXME: assumes CPU always use float, GPU use
                                 // different types
  int lda[MAX_GEMM_DIM];
  const float* b[MAX_GEMM_DIM];
  int ldb[MAX_GEMM_DIM];
  float beta[MAX_GEMM_DIM];
  float* c[MAX_GEMM_DIM];
  int ldc[MAX_GEMM_DIM];
  int group_size;  // FIXME: we do not deal with group with different sizes

  std::string DebugString() const {
    std::string str = "GemmArgs: ";
    for (int i = 0; i < group_size; i++) {
      str += "transa: " + std::string(1, transa[i]) + ", ";
      str += "transb: " + std::string(1, transb[i]) + ", ";
      str += "m: " + std::to_string(m[i]) + ", ";
      str += "n: " + std::to_string(n[i]) + ", ";
      str += "k: " + std::to_string(k[i]) + ", ";
      str += "alpha: " + std::to_string(alpha[i]) + ", ";
      str += "lda: " + std::to_string(lda[i]) + ", ";
      str += "ldb: " + std::to_string(ldb[i]) + ", ";
      str += "beta: " + std::to_string(beta[i]) + ", ";
      str += "ldc: " + std::to_string(ldc[i]) + ", ";
    }
    return str;
  }
};

typedef std::unique_ptr<GemmArgs> GemmArgsPtr;

inline std::tuple<size_t, size_t, size_t> CalculateTaskSizes(
    const GemmArgs* args) {
  if (args->group_size == 1) {
    auto size_a = (args->transa[0] == 'N' || args->transa[0] == 'n')
                      ? (args->lda[0]) * (args->k[0]) * sizeof(float)
                      : (args->lda[0]) * (args->m[0]) * sizeof(float);
    auto size_b = (args->transb[0] == 'N' || args->transb[0] == 'n')
                      ? (args->ldb[0]) * (args->n[0]) * sizeof(float)
                      : (args->ldb[0]) * (args->k[0]) * sizeof(float);
    auto size_c = (args->ldc[0]) * (args->n[0]) * sizeof(float);
    LOG_FATAL_IF(size_a == 0 || size_b == 0 || size_c == 0)
        << "Invalid task sizes: A: " << size_a << ", B: " << size_b
        << ", C: " << size_c;
    return {size_a, size_b, size_c};
  }
  LOG_FATAL << "Grouped gemm not supported yet, group size: "
            << args->group_size;
  return {0, 0, 0};
}

// bool CheckBufferOffloaded(const void* buffer, size_t size);

void TaskExecution(const GemmArgsPtr& args);

// extern "C" {
// DECLARE_INTERCEPTOR(sgemm, float)
// }

#if 0
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <dlfcn.h>
#include <fcntl.h>
#include <math.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdbool.h>
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

// Updated includes for shared memory management
#include "/home/eren/DeviceEmulator/csrc/memory/shared_memory_initializer.h"
#include "/home/eren/DeviceEmulator/csrc/memory/shared_memory_manager.h"

#define SHM_NAME "/sgemm_shm"
#define LOG_DIR "/home/eren/DeviceEmulator/csrc/intercept/logs/"
#define EMPTY 0
#define WRITING 1
#define WRITTEN 2
#define READING 3
#define EXECUTING 4
#define COMPLETE 5
#define MAX_TASKS 1024
#define QUEUE_SIZE 1024

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

typedef void (*sgemm_type)(const char*, const char*, const int*, const int*,
                           const int*, const float*, const float*, const int*,
                           const float*, const int*, const float*, float*,
                           const int*);

typedef void (*sgemm_batch_type)(
    const char* transa_array, const char* transb_array, const int* m_array,
    const int* n_array, const int* k_array, const float* alpha_array,
    const float* a_array[], const int* lda_array, const float* b_array[],
    const int* ldb_array, const float* beta_array, float* c_array[],
    const int* ldc_array, const int* group_count, const int* group_size);

extern FILE* log_file;
extern pthread_mutex_t shm_mutex;
extern sgemm_type orig_sgemm;

// Function declarations
void initialize_logging();
void get_timestamp(char* buffer, size_t size);
void log_message(const char* message);
void lock_memory();
void unlock_memory();
size_t calculate_matrix_size(char trans, int rows, int cols, int leading_dim);

size_t get_shared_memory_size();
int enqueue_task(task_queue_t* task_queue, int task_index);
void sgemm_(const char* transa, const char* transb, const int* m, const int* n,
            const int* k, const float* alpha, const float* a, const int* lda,
            const float* b, const int* ldb, const float* beta, float* c,
            const int* ldc);
void process_task_from_queue(shared_memory_t* shared_mem_ptr);
#endif
