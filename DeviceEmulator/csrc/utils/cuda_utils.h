// Copyright (c) TorchMoE.
// SPDX-License-Identifier: Apache-2.0

// TorchMoE Team

#pragma once

#include <cublas_v2.h>
#include <cuda.h>  // CUDA driver API types (CUresult, etc.)
#include <cuda_runtime_api.h>

#include <cstdint>

#define CUDA_CHECK(call)                                               \
  do {                                                                 \
    cudaError_t err = call;                                            \
    if (err != cudaSuccess) {                                          \
      std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at " \
                << __FILE__ << ":" << __LINE__ << std::endl;           \
      std::exit(err);                                                  \
    }                                                                  \
  } while (0)

#define CHECK_CUBLAS_ERROR(call)                                    \
  {                                                                 \
    cublasStatus_t err = (call);                                    \
    LOG_FATAL_IF(err != CUBLAS_STATUS_SUCCESS)                      \
        << "CUBLAS error. message: " << cublasGetStatusString(err); \
  }

#define CHECK_CUDA_ERROR(call)                                 \
  {                                                            \
    cudaError_t err = (call);                                  \
    LOG_FATAL_IF(err != cudaSuccess)                           \
        << "CUDA error. message: " << cudaGetErrorString(err); \
  }

#define CHECK_CU_RESULT(call)                                             \
  {                                                                       \
    CUresult err = (call);                                                \
    if (err != CUDA_SUCCESS) {                                            \
      const char* err_str = nullptr;                                      \
      cuGetErrorString(err, &err_str);                                    \
      LOG_FATAL << "CUDA driver error: " << (err_str ? err_str : "???"); \
    }                                                                     \
  }

bool LogCudaError(cudaError_t status, const char* context);
bool LogCublasError(cublasStatus_t status, const char* context);

bool IsDevicePointer(const void* ptr);
int GetDeviceCount();
std::size_t GetTotalDeviceMemory(int device_id);
std::size_t GetFreeDeviceMemory(int device_id);

#define DEVICE_CACHE_LIMIT(gid) GetTotalDeviceMemory(gid) * 0.7
#define NUM_DEVICES GetDeviceCount()
