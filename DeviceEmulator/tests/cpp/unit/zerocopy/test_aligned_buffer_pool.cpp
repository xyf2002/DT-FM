#include <gtest/gtest.h>

#include <atomic>
#include <thread>
#include <vector>

#include "backend/server_base.h"

class AlignedBufferPoolTest : public ::testing::Test {
 protected:
  AlignedBufferPool pool_;
};

TEST_F(AlignedBufferPoolTest, BucketSize_BelowPageSize_ReturnsPageSize) {
  auto [ptr, bucket] = pool_.Acquire(100);
  EXPECT_EQ(bucket, AlignedBufferPool::PAGE_SIZE);
  pool_.Release(ptr, bucket);
}

TEST_F(AlignedBufferPoolTest, BucketSize_ExactPageSize_ReturnsPageSize) {
  auto [ptr, bucket] = pool_.Acquire(AlignedBufferPool::PAGE_SIZE);
  EXPECT_EQ(bucket, AlignedBufferPool::PAGE_SIZE);
  pool_.Release(ptr, bucket);
}

TEST_F(AlignedBufferPoolTest, BucketSize_RoundsUpToPowerOf2) {
  auto [ptr, bucket] = pool_.Acquire(5000);
  EXPECT_EQ(bucket, 8192u);
  pool_.Release(ptr, bucket);
}

TEST_F(AlignedBufferPoolTest, BucketSize_ExactPowerOf2_NoRounding) {
  auto [ptr, bucket] = pool_.Acquire(8192);
  EXPECT_EQ(bucket, 8192u);
  pool_.Release(ptr, bucket);
}

TEST_F(AlignedBufferPoolTest, BucketSize_LargeSize) {
  size_t size = 1024 * 1024 + 1;  // 1MB + 1
  auto [ptr, bucket] = pool_.Acquire(size);
  EXPECT_EQ(bucket, 2 * 1024 * 1024u);
  pool_.Release(ptr, bucket);
}

TEST_F(AlignedBufferPoolTest, Acquire_ReturnsPageAlignedNonNull) {
  auto [ptr, bucket] = pool_.Acquire(4096);
  ASSERT_NE(ptr, nullptr);
  EXPECT_EQ(reinterpret_cast<uintptr_t>(ptr) % AlignedBufferPool::PAGE_SIZE,
            0u);
  pool_.Release(ptr, bucket);
}

TEST_F(AlignedBufferPoolTest, Release_Reacquire_ReusesSameBuffer) {
  auto [ptr1, bucket1] = pool_.Acquire(4096);
  pool_.Release(ptr1, bucket1);
  auto [ptr2, bucket2] = pool_.Acquire(4096);
  EXPECT_EQ(ptr1, ptr2);
  EXPECT_EQ(bucket1, bucket2);
  pool_.Release(ptr2, bucket2);
}

TEST_F(AlignedBufferPoolTest, PoolFull_FreesBuffer) {
  std::vector<uint8_t*> ptrs;
  size_t bucket = AlignedBufferPool::PAGE_SIZE;

  // Acquire MAX_BUFFERS_PER_BUCKET + 1 buffers
  for (size_t i = 0; i <= AlignedBufferPool::MAX_BUFFERS_PER_BUCKET; ++i) {
    auto [ptr, b] = pool_.Acquire(bucket);
    ASSERT_NE(ptr, nullptr);
    ptrs.push_back(ptr);
  }

  // Release all — the last one should be freed (pool full), no crash
  for (auto* ptr : ptrs) {
    pool_.Release(ptr, bucket);
  }
}

TEST_F(AlignedBufferPoolTest, MultiBucket_Independence) {
  auto [ptr_4k, bucket_4k] = pool_.Acquire(4096);
  auto [ptr_8k, bucket_8k] = pool_.Acquire(8192);
  EXPECT_NE(bucket_4k, bucket_8k);

  pool_.Release(ptr_4k, bucket_4k);

  // Acquiring 8k should NOT return the 4k buffer
  auto [ptr_8k_2, bucket_8k_2] = pool_.Acquire(8192);
  EXPECT_NE(ptr_8k_2, ptr_4k);

  pool_.Release(ptr_8k, bucket_8k);
  pool_.Release(ptr_8k_2, bucket_8k_2);
}

TEST_F(AlignedBufferPoolTest, NullRelease_IsNoOp) {
  EXPECT_NO_THROW(pool_.Release(nullptr, 4096));
  EXPECT_NO_THROW(pool_.Release(nullptr, 0));
}

TEST_F(AlignedBufferPoolTest, ThreadSafety_ConcurrentAcquireRelease) {
  constexpr int kNumThreads = 8;
  constexpr int kOpsPerThread = 100;

  auto worker = [this]() {
    for (int i = 0; i < kOpsPerThread; ++i) {
      auto [ptr, bucket] = pool_.Acquire(4096);
      ASSERT_NE(ptr, nullptr);
      // Touch the memory to verify it's accessible
      memset(ptr, 0xAB, 4096);
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

TEST_F(AlignedBufferPoolTest, Acquire_BufferIsWritable) {
  auto [ptr, bucket] = pool_.Acquire(4096);
  ASSERT_NE(ptr, nullptr);
  // Write pattern and verify
  for (size_t i = 0; i < 4096; ++i) {
    ptr[i] = static_cast<uint8_t>(i & 0xFF);
  }
  for (size_t i = 0; i < 4096; ++i) {
    EXPECT_EQ(ptr[i], static_cast<uint8_t>(i & 0xFF));
  }
  pool_.Release(ptr, bucket);
}

// ---------------------------------------------------------------------------
// Custom pin/unpin callback tests
// ---------------------------------------------------------------------------

static std::atomic<int> g_pin_count{0};
static std::atomic<int> g_unpin_count{0};

static int TestPin(void* /*ptr*/, size_t /*size*/) {
  g_pin_count.fetch_add(1, std::memory_order_relaxed);
  return 0;
}

static void TestUnpin(void* /*ptr*/, size_t /*size*/) {
  g_unpin_count.fetch_add(1, std::memory_order_relaxed);
}

TEST(AlignedBufferPoolCustomPin, PinCalledOnAcquire) {
  g_pin_count = 0;
  g_unpin_count = 0;

  AlignedBufferPool pool;
  pool.SetPinFunctions(TestPin, TestUnpin);

  auto [ptr, bucket] = pool.Acquire(4096);
  ASSERT_NE(ptr, nullptr);
  EXPECT_EQ(g_pin_count.load(), 1);

  // Second acquire forces a new allocation (pool is empty)
  auto [ptr2, bucket2] = pool.Acquire(4096);
  EXPECT_EQ(g_pin_count.load(), 2);

  pool.Release(ptr, bucket);
  pool.Release(ptr2, bucket2);
}

TEST(AlignedBufferPoolCustomPin, UnpinCalledOnOverflowFree) {
  g_pin_count = 0;
  g_unpin_count = 0;

  AlignedBufferPool pool;
  pool.SetPinFunctions(TestPin, TestUnpin);

  size_t bucket = AlignedBufferPool::PAGE_SIZE;
  std::vector<uint8_t*> ptrs;

  // Acquire MAX_BUFFERS_PER_BUCKET + 1 buffers
  for (size_t i = 0; i <= AlignedBufferPool::MAX_BUFFERS_PER_BUCKET; ++i) {
    auto [ptr, b] = pool.Acquire(bucket);
    ASSERT_NE(ptr, nullptr);
    ptrs.push_back(ptr);
  }

  int unpin_before = g_unpin_count.load();

  // Release all — the last release overflows the pool and triggers unpin
  for (auto* ptr : ptrs) {
    pool.Release(ptr, bucket);
  }

  EXPECT_EQ(g_unpin_count.load(), unpin_before + 1);
}

TEST(AlignedBufferPoolCustomPin, UnpinCalledOnDestruction) {
  g_pin_count = 0;
  g_unpin_count = 0;

  {
    AlignedBufferPool pool;
    pool.SetPinFunctions(TestPin, TestUnpin);

    auto [ptr1, b1] = pool.Acquire(4096);
    auto [ptr2, b2] = pool.Acquire(8192);
    pool.Release(ptr1, b1);
    pool.Release(ptr2, b2);
    // 2 buffers sitting in the pool
    EXPECT_EQ(g_unpin_count.load(), 0);
  }
  // Destructor should unpin both
  EXPECT_EQ(g_unpin_count.load(), 2);
}
