#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <cstring>
#include <deque>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <vector>

// Inline the CudaPinnedMemoryPool class to avoid pulling in proxy_cli.h
// which transitively includes torch and protobuf (version conflicts).
// This is a direct copy from csrc/backend/proxy_cli.h.
class CudaPinnedMemoryPool {
 public:
  explicit CudaPinnedMemoryPool(size_t max_buffers_per_bucket = 16)
      : max_per_bucket_(max_buffers_per_bucket) {}

  ~CudaPinnedMemoryPool() {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto& [bucket_size, free_list] : free_lists_) {
      for (auto* ptr : free_list) {
        cudaFreeHost(ptr);
      }
    }
  }

  std::pair<void*, size_t> Acquire(size_t size) {
    size_t bucket = BucketSize(size);
    std::lock_guard<std::mutex> lock(mutex_);
    auto& free_list = free_lists_[bucket];
    if (!free_list.empty()) {
      void* ptr = free_list.back();
      free_list.pop_back();
      return {ptr, bucket};
    }
    void* ptr = nullptr;
    cudaError_t err = cudaHostAlloc(&ptr, bucket, cudaHostAllocDefault);
    if (err != cudaSuccess || !ptr) {
      throw std::runtime_error("CudaPinnedMemoryPool: cudaHostAlloc failed");
    }
    return {ptr, bucket};
  }

  void Release(void* ptr, size_t bucket_size) {
    if (!ptr) return;
    std::lock_guard<std::mutex> lock(mutex_);
    auto& free_list = free_lists_[bucket_size];
    if (free_list.size() < max_per_bucket_) {
      free_list.push_back(ptr);
    } else {
      cudaFreeHost(ptr);
    }
  }

 private:
  static size_t BucketSize(size_t size) {
    static constexpr size_t MIN_BUCKET = 4096;
    if (size <= MIN_BUCKET) return MIN_BUCKET;
    size_t bucket = MIN_BUCKET;
    while (bucket < size) bucket <<= 1;
    return bucket;
  }

  size_t max_per_bucket_;
  std::mutex mutex_;
  std::unordered_map<size_t, std::deque<void*>> free_lists_;
};

class CudaPinnedMemoryPoolTest : public ::testing::Test {
 protected:
  CudaPinnedMemoryPool pool_{16};
};

TEST_F(CudaPinnedMemoryPoolTest, Acquire_ReturnsNonNull) {
  auto [ptr, bucket] = pool_.Acquire(4096);
  ASSERT_NE(ptr, nullptr);
  pool_.Release(ptr, bucket);
}

TEST_F(CudaPinnedMemoryPoolTest, BucketSizing_Small) {
  auto [ptr, bucket] = pool_.Acquire(100);
  EXPECT_EQ(bucket, 4096u);
  pool_.Release(ptr, bucket);
}

TEST_F(CudaPinnedMemoryPoolTest, BucketSizing_RoundsUp) {
  auto [ptr, bucket] = pool_.Acquire(5000);
  EXPECT_EQ(bucket, 8192u);
  pool_.Release(ptr, bucket);
}

TEST_F(CudaPinnedMemoryPoolTest, Acquire_Release_Reuse) {
  auto [ptr1, bucket1] = pool_.Acquire(4096);
  pool_.Release(ptr1, bucket1);
  auto [ptr2, bucket2] = pool_.Acquire(4096);
  EXPECT_EQ(ptr1, ptr2);
  pool_.Release(ptr2, bucket2);
}

TEST_F(CudaPinnedMemoryPoolTest, PoolFull_Behavior) {
  std::vector<std::pair<void*, size_t>> acquired;
  for (size_t i = 0; i < 18; ++i) {
    auto [ptr, bucket] = pool_.Acquire(4096);
    ASSERT_NE(ptr, nullptr);
    acquired.push_back({ptr, bucket});
  }
  for (auto& [ptr, bucket] : acquired) {
    pool_.Release(ptr, bucket);
  }
}

TEST_F(CudaPinnedMemoryPoolTest, Buffer_IsWritable) {
  auto [ptr, bucket] = pool_.Acquire(4096);
  ASSERT_NE(ptr, nullptr);
  memset(ptr, 0xAB, 4096);
  EXPECT_EQ(static_cast<uint8_t*>(ptr)[0], 0xAB);
  pool_.Release(ptr, bucket);
}

TEST_F(CudaPinnedMemoryPoolTest, ThreadSafety_ConcurrentAcquireRelease) {
  constexpr int kNumThreads = 4;
  constexpr int kOpsPerThread = 50;

  auto worker = [this]() {
    for (int i = 0; i < kOpsPerThread; ++i) {
      auto [ptr, bucket] = pool_.Acquire(4096);
      ASSERT_NE(ptr, nullptr);
      memset(ptr, 0, 4096);
      pool_.Release(ptr, bucket);
    }
  };

  std::vector<std::thread> threads;
  for (int i = 0; i < kNumThreads; ++i) {
    threads.emplace_back(worker);
  }
  for (auto& t : threads) {
    t.join();
  }
}
