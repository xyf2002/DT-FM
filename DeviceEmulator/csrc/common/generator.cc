#include "generator.h"

#include <atomic>

uint32_t NumGenerator::ctx_id_ = 1;
std::mutex NumGenerator::mutex_;

// 0是一个特殊的id，必须保证永远不会生成0这个id
uint32_t NumGenerator::ctx_id() {
  std::lock_guard g(mutex_);
  uint32_t ret = ctx_id_++;
  if (ret == 0) ret = ctx_id_++;
  return ret;
}

uint32_t NumGenerator::flowno() {
  static std::atomic<uint32_t> flowno(1024);
  return flowno++;
}
