// Copyright (c) TorchMoE.
// SPDX-License-Identifier: Apache-2.0

// TorchMoE Team

#pragma once

#include <cstdint>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>

#include "base/log_file.h"
#include "base/logging.h"
#include "common/types_and_defs.h"  // for TensorKey

#define PRINT_ARGS(...)                \
  do {                                 \
    std::cout << #__VA_ARGS__ << ": "; \
    print(__VA_ARGS__);                \
  } while (0)

// Helper function to print each argument
inline void print() { std::cout << std::endl; }  // Base case to end recursion

template <typename T, typename... Args>
inline void print(T first, Args... args) {
  std::cout << first;
  if constexpr (sizeof...(args) > 0) {
    std::cout << ", ";
    print(args...);  // Recursive call
  } else {
    std::cout << std::endl;
  }
}

int str2level(const char* level);
std::string level2str(int level);
std::string formatstr();

enum LogLevel { kFatal, kDebug, kInfo, kWarn, kError };

extern std::once_flag kLoggerFlag;
extern int kLogLevel;
extern std::mutex kLogMutex;
extern std::unique_ptr<base::LogFile> g_tee_log_file;

extern void InitLogger();

// Use base/logging.h macros: LOG_DEBUG, LOG_INFO, LOG_WARN, LOG_ERROR,
// LOG_FATAL These use streaming interface: LOG_INFO << "message" << variable;

#define LOG_FATAL_IF(cond) \
  if (cond) LOG_FATAL

#define LOG_ERROR_IF(cond) \
  if (cond) LOG_ERROR

#define LOG_WARN_IF(cond) \
  if (cond) LOG_WARN

#define LOG_INFO_IF(cond) \
  if (cond) LOG_INFO

void TeeOutput(const char* msg, int len);
void TeeFlush();

// Streaming operators for common types used in logging
template <typename T>
inline base::LogStream& operator<<(base::LogStream& stream,
                                   const std::vector<T>& v) {
  stream << "[";
  for (size_t i = 0; i < v.size(); i++) {
    if (i > 0) stream << ",";
    stream << v[i];
  }
  stream << "]";
  return stream;
}

inline base::LogStream& operator<<(base::LogStream& stream,
                                   const TensorKey& key) {
  stream << "[" << std::get<0>(key) << ":" << std::get<1>(key) << ":"
         << std::get<2>(key) << ":" << std::get<3>(key) << "]";
  return stream;
}

// Device log tag helpers (moved from utils/device_log_tag.h)
inline std::string DevLogTag(int64_t dev_id, int64_t gemm_id) {
  return "[dev:" + std::to_string(dev_id) + "|gemm:" + std::to_string(gemm_id) +
         "] ";
}

inline std::string DevLogTag(int64_t dev_id, std::string part_key) {
  return "[dev:" + std::to_string(dev_id) + "|part:" + part_key + "] ";
}

inline std::string DevLogTagDev(int64_t dev_id) {
  return "[dev:" + std::to_string(dev_id) + "] ";
}

#define DEV_TAG(dev_id, gemm_id) DevLogTag(dev_id, gemm_id)
#define DEV_TAG_PART(dev_id, part_key) DevLogTag(dev_id, part_key)
#define DEV_TAG_DEV(dev_id) DevLogTagDev(dev_id)
