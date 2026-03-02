#include <gtest/gtest.h>

#include "backend/server_base.h"

class ScatterGatherBufferTest : public ::testing::Test {
 protected:
  AlignedBufferPool pool_;
};

TEST_F(ScatterGatherBufferTest, DefaultConstructor_Empty) {
  ScatterGatherBuffer sg;
  EXPECT_EQ(sg.GetSegments().size(), 0u);
  EXPECT_EQ(sg.GetTotalSize(), 0u);
}

TEST_F(ScatterGatherBufferTest, ConstructorWithPool_Empty) {
  ScatterGatherBuffer sg(pool_);
  EXPECT_EQ(sg.GetSegments().size(), 0u);
  EXPECT_EQ(sg.GetTotalSize(), 0u);
}

TEST_F(ScatterGatherBufferTest, AddReferenceSegment_Normal) {
  ScatterGatherBuffer sg(pool_);
  uint8_t data[100] = {};
  sg.AddReferenceSegment(data, sizeof(data));

  EXPECT_EQ(sg.GetSegments().size(), 1u);
  EXPECT_EQ(sg.GetSegments()[0].data, data);
  EXPECT_EQ(sg.GetSegments()[0].size, sizeof(data));
  EXPECT_FALSE(sg.GetSegments()[0].owned);
  EXPECT_EQ(sg.GetTotalSize(), sizeof(data));
}

TEST_F(ScatterGatherBufferTest, AddReferenceSegment_NullIgnored) {
  ScatterGatherBuffer sg(pool_);
  sg.AddReferenceSegment(nullptr, 100);
  EXPECT_EQ(sg.GetSegments().size(), 0u);
}

TEST_F(ScatterGatherBufferTest, AddReferenceSegment_ZeroSizeIgnored) {
  ScatterGatherBuffer sg(pool_);
  uint8_t data[1] = {};
  sg.AddReferenceSegment(data, 0);
  EXPECT_EQ(sg.GetSegments().size(), 0u);
}

TEST_F(ScatterGatherBufferTest, AddOwnedSegment_Tracked) {
  ScatterGatherBuffer sg(pool_);
  auto [ptr, bucket] = pool_.Acquire(4096);
  sg.AddOwnedSegment(ptr, 4096, bucket);

  EXPECT_EQ(sg.GetSegments().size(), 1u);
  EXPECT_EQ(sg.GetSegments()[0].data, ptr);
  EXPECT_EQ(sg.GetSegments()[0].size, 4096u);
  EXPECT_TRUE(sg.GetSegments()[0].owned);
  EXPECT_EQ(sg.GetTotalSize(), 4096u);
  // sg destructor will release ptr back to pool_
}

TEST_F(ScatterGatherBufferTest, MixedSegments_TotalSize) {
  ScatterGatherBuffer sg(pool_);

  auto [ptr, bucket] = pool_.Acquire(4096);
  sg.AddOwnedSegment(ptr, 4096, bucket);

  uint8_t ref_data[256] = {};
  sg.AddReferenceSegment(ref_data, sizeof(ref_data));

  EXPECT_EQ(sg.GetSegments().size(), 2u);
  EXPECT_EQ(sg.GetTotalSize(), 4096u + 256u);
}

TEST_F(ScatterGatherBufferTest, Destructor_ReleasesOwnedToPool) {
  uint8_t* ptr_from_pool = nullptr;
  {
    ScatterGatherBuffer sg(pool_);
    auto [ptr, bucket] = pool_.Acquire(4096);
    ptr_from_pool = ptr;
    sg.AddOwnedSegment(ptr, 4096, bucket);
    // sg goes out of scope, should release ptr to pool_
  }

  // Re-acquire should return same pointer
  auto [ptr2, bucket2] = pool_.Acquire(4096);
  EXPECT_EQ(ptr2, ptr_from_pool);
  pool_.Release(ptr2, bucket2);
}

TEST_F(ScatterGatherBufferTest, Destructor_IgnoresReferenceSegments) {
  uint8_t ref_data[64] = {0xAA};
  {
    ScatterGatherBuffer sg(pool_);
    sg.AddReferenceSegment(ref_data, sizeof(ref_data));
    // sg destructor should NOT free ref_data
  }
  // ref_data should still be valid
  EXPECT_EQ(ref_data[0], 0xAA);
}

TEST_F(ScatterGatherBufferTest, MoveConstructor_TransfersOwnership) {
  ScatterGatherBuffer sg(pool_);
  auto [ptr, bucket] = pool_.Acquire(4096);
  sg.AddOwnedSegment(ptr, 4096, bucket);

  ScatterGatherBuffer moved(std::move(sg));
  EXPECT_EQ(moved.GetSegments().size(), 1u);
  EXPECT_EQ(moved.GetSegments()[0].data, ptr);
  EXPECT_EQ(moved.GetTotalSize(), 4096u);

  // Source should be empty
  EXPECT_EQ(sg.GetSegments().size(), 0u);
  EXPECT_EQ(sg.GetTotalSize(), 0u);
}

TEST_F(ScatterGatherBufferTest, MoveAssignment_CleansUpOldAndTransfers) {
  // Create first sg with owned segment
  ScatterGatherBuffer sg1(pool_);
  auto [ptr1, bucket1] = pool_.Acquire(4096);
  sg1.AddOwnedSegment(ptr1, 4096, bucket1);

  // Create second sg with different owned segment
  ScatterGatherBuffer sg2(pool_);
  auto [ptr2, bucket2] = pool_.Acquire(8192);
  sg2.AddOwnedSegment(ptr2, 8192, bucket2);

  // Move assign sg2 into sg1 — sg1's old segment should be released
  sg1 = std::move(sg2);

  EXPECT_EQ(sg1.GetSegments().size(), 1u);
  EXPECT_EQ(sg1.GetSegments()[0].data, ptr2);

  // ptr1 should have been released back to pool
  auto [ptr_reused, bucket_reused] = pool_.Acquire(4096);
  EXPECT_EQ(ptr_reused, ptr1);
  pool_.Release(ptr_reused, bucket_reused);
}
