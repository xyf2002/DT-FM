#pragma once
#include <string>

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
