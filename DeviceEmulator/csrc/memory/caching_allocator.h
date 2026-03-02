#pragma once

#include <cuda_runtime_api.h>

#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <unordered_map>

#include "common/types_and_defs.h"
#include "memory/shared_memory.h"
#include "utils/logger.h"
#include "utils/noncopyable.h"

#define MEMORY_TYPE_VALUES(X, EnumType) \
  X(SHM, EnumType)                      \
  X(PIN, EnumType)                      \
  X(CUDA, EnumType)                     \
  X(PIN_SHM, EnumType)

DEFINE_ENUM_CLASS(MemoryType, MEMORY_TYPE_VALUES)

class CachingAllocator;
extern std::unique_ptr<CachingAllocator> kCachingAllocator;

struct TorchCtx {
  void* ptr;
  size_t size;
};

extern "C" {
void* TorchAllocate(size_t bytes);
void TorchFree(void* ptr);
void TorchFreeCtx(void* ctx);
}

// the caching allocator that supports CPU and CUDA memory
// work as an offset manager for the memory pool
class CachingAllocator : public noncopyable {
 public:
  explicit CachingAllocator(size_t bytes, MemoryType type, int device_id = -1);
  virtual ~CachingAllocator();

  virtual void* Allocate(const size_t bytes);
  virtual void Free(void* ptr);

  bool IsAllocated(void* ptr) {
    std::lock_guard<std::mutex> guard(mutex_);
    return allocation_map_.find(ptr) != allocation_map_.end();
  }

  ShmMeta FindShmMetaByRange(void* ptr);
  void InsertShmMeta(ShmMeta meta);

  MemoryType GetType() const { return type_; }

  size_t GetMaxBytes() const { return max_bytes_; }
  size_t GetAllocatedBytes() const { return allocated_bytes_; }
  size_t GetUsedBytes() const { return used_bytes_; }

 private:
  void* AllocateAndCache(const size_t bytes);
  void FreeCached();

  void* AllocateMemory(size_t bytes);

  void* AllocCudaMemory(size_t bytes);
  void* AllocPinMemory(size_t bytes);
  void* AllocShmMemory(size_t bytes);
  void* AllocPinShmMemory(size_t bytes);

  void FreeMemory(void* ptr);
  void FreeCudaMemory(void* ptr);
  void FreePinMemory(void* ptr);
  void FreeShmMemory(void* ptr);
  void FreePinShmMemory(void* ptr);

 protected:
  int device_id_;
  MemoryType type_;
  const size_t max_bytes_;
  size_t allocated_bytes_;
  size_t used_bytes_;

  std::unordered_map<size_t, std::deque<void*>> available_map_;
  std::unordered_map<void*, size_t> allocation_map_;
  std::mutex mutex_;
  std::unordered_map<void*, ShmMeta> shm_id_map_;
};

extern std::once_flag kInitCachingAllocatorFlag;

static inline void InitCachingAllocator(MemoryType type, int device_id = -1) {
  std::call_once(kInitCachingAllocatorFlag, [&]() {
    size_t bytes = 0;
    LOG_DEBUG << "InitCachingAllocator: type: " << MemoryTypeToString(type)
              << " device_id: " << device_id;
    if (type == MemoryType::CUDA) {
      LOG_FATAL_IF(device_id < 0) << "Invalid device id";
      // Get environment variable MORPHLING_SHM_SIZE
      const char* size = std::getenv("MORPHLING_GPU_SIZE");
      LOG_FATAL_IF(size == nullptr) << "MORPHLING_GPU_SIZE is not set";
      bytes = std::stoull(size);
    } else if (type == MemoryType::SHM) {
      // Get environment variable MORPHLING_SHM_SIZE
      const char* size = std::getenv("MORPHLING_SHM_SIZE");
      LOG_FATAL_IF(size == nullptr) << "MORPHLING_SHM_SIZE is not set";
      bytes = std::stoull(size);
    } else if (type == MemoryType::PIN or type == MemoryType::PIN_SHM) {
      const char* size = std::getenv("MORPHLING_PIN_SIZE");
      LOG_FATAL_IF(size == nullptr) << "MORPHLING_PIN_SIZE is not set";
      bytes = std::stoull(size);
    } else {
      LOG_FATAL << "Unknown memory type";
    }
    LOG_FATAL_IF(kCachingAllocator != nullptr)
        << "Caching allocator is already initialized";

    kCachingAllocator =
        std::make_unique<CachingAllocator>(bytes, type, device_id);
    LOG_INFO << "Caching allocator initialized with" << bytes / GB
             << "GB, type:" << MemoryTypeToString(type);
  });
}
