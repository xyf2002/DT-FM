#include <gtest/gtest.h>

#include "backend/server_base.h"

class SerializationBufferTest : public ::testing::Test {
 protected:
  AlignedBufferPool pool_;
};

TEST_F(SerializationBufferTest, DefaultConstructor_AllZero) {
  SerializationBuffer buf;
  EXPECT_EQ(buf.GetBuffer(), nullptr);
  EXPECT_EQ(buf.GetSize(), 0u);
  EXPECT_EQ(buf.GetOffset(), 0u);
}

TEST_F(SerializationBufferTest, Allocate_SetsUpBuffer) {
  SerializationBuffer buf;
  buf.Allocate(1024, pool_);
  ASSERT_NE(buf.GetBuffer(), nullptr);
  EXPECT_GE(buf.GetSize(), 1024u);
  EXPECT_EQ(buf.GetOffset(), 0u);
  // Page-aligned
  EXPECT_EQ(reinterpret_cast<uintptr_t>(buf.GetBuffer()) %
                AlignedBufferPool::PAGE_SIZE,
            0u);
}

TEST_F(SerializationBufferTest, WriteReadUInt32_NativeOrder) {
  SerializationBuffer buf;
  buf.Allocate(64, pool_);

  uint32_t value = 0xDEADBEEF;
  buf.WriteUInt32(value, false);
  buf.SeekTo(0);
  EXPECT_EQ(buf.ReadUInt32(false), value);
}

TEST_F(SerializationBufferTest, WriteReadUInt32_NetworkOrder) {
  SerializationBuffer buf;
  buf.Allocate(64, pool_);

  uint32_t value = 0x12345678;
  buf.WriteUInt32(value, true);
  buf.SeekTo(0);
  EXPECT_EQ(buf.ReadUInt32(true), value);
}

TEST_F(SerializationBufferTest, WriteReadUInt64_Roundtrip) {
  SerializationBuffer buf;
  buf.Allocate(64, pool_);

  uint64_t value = 0xDEADBEEFCAFEBABEull;
  buf.WriteUInt64(value);
  buf.SeekTo(0);
  EXPECT_EQ(buf.ReadUInt64(), value);
}

TEST_F(SerializationBufferTest, WriteReadInt64_Roundtrip) {
  SerializationBuffer buf;
  buf.Allocate(64, pool_);

  int64_t value = -123456789012345LL;
  buf.WriteInt64(value);
  buf.SeekTo(0);
  EXPECT_EQ(buf.ReadInt64(), value);
}

TEST_F(SerializationBufferTest, WriteReadBytes_Roundtrip) {
  SerializationBuffer buf;
  buf.Allocate(256, pool_);

  std::vector<uint8_t> data(128);
  for (size_t i = 0; i < data.size(); ++i) {
    data[i] = static_cast<uint8_t>(i);
  }

  buf.WriteBytes(data.data(), data.size());
  buf.SeekTo(0);

  std::vector<uint8_t> readback(128);
  buf.ReadBytes(readback.data(), readback.size());
  EXPECT_EQ(data, readback);
}

TEST_F(SerializationBufferTest, MixedSequentialReadWrite) {
  SerializationBuffer buf;
  buf.Allocate(256, pool_);

  uint32_t u32 = 42;
  uint64_t u64 = 123456789;
  int64_t i64 = -987654321;
  uint8_t bytes[] = {0xAA, 0xBB, 0xCC, 0xDD};

  buf.WriteUInt32(u32, false);
  buf.WriteUInt64(u64);
  buf.WriteInt64(i64);
  buf.WriteBytes(bytes, sizeof(bytes));

  buf.SeekTo(0);

  EXPECT_EQ(buf.ReadUInt32(false), u32);
  EXPECT_EQ(buf.ReadUInt64(), u64);
  EXPECT_EQ(buf.ReadInt64(), i64);

  uint8_t readback[4];
  buf.ReadBytes(readback, sizeof(readback));
  EXPECT_EQ(memcmp(bytes, readback, sizeof(bytes)), 0);
}

TEST_F(SerializationBufferTest, SeekTo_CanRead_ValidateSize) {
  SerializationBuffer buf;
  buf.Allocate(64, pool_);

  buf.WriteUInt32(0, false);
  EXPECT_EQ(buf.GetOffset(), sizeof(uint32_t));

  buf.SeekTo(0);
  EXPECT_EQ(buf.GetOffset(), 0u);

  EXPECT_TRUE(buf.CanRead(64));
  EXPECT_FALSE(buf.CanRead(65));

  EXPECT_NO_THROW(buf.ValidateSize(64));
  EXPECT_THROW(buf.ValidateSize(65), std::runtime_error);
}

TEST_F(SerializationBufferTest, MoveConstructor_TransfersOwnership) {
  SerializationBuffer buf;
  buf.Allocate(1024, pool_);
  void* original_ptr = buf.GetBuffer();

  SerializationBuffer moved(std::move(buf));
  EXPECT_EQ(moved.GetBuffer(), original_ptr);
  EXPECT_EQ(moved.GetSize(), 1024u);

  // Source should be zeroed
  EXPECT_EQ(buf.GetBuffer(), nullptr);
  EXPECT_EQ(buf.GetSize(), 0u);
}

TEST_F(SerializationBufferTest, MoveAssignment_TransfersOwnership) {
  SerializationBuffer buf;
  buf.Allocate(1024, pool_);
  void* original_ptr = buf.GetBuffer();

  SerializationBuffer moved;
  moved = std::move(buf);
  EXPECT_EQ(moved.GetBuffer(), original_ptr);
  EXPECT_EQ(moved.GetSize(), 1024u);

  EXPECT_EQ(buf.GetBuffer(), nullptr);
  EXPECT_EQ(buf.GetSize(), 0u);
}

TEST_F(SerializationBufferTest, NonOwningBuffer_DoesNotFree) {
  uint8_t data[64] = {};
  {
    SerializationBuffer buf(data, sizeof(data), false);
    EXPECT_EQ(buf.GetBuffer(), data);
    EXPECT_EQ(buf.GetSize(), sizeof(data));
    // Should not free on destruction since take_ownership = false
  }
  // data is still valid on stack — no crash
  data[0] = 0xFF;
  EXPECT_EQ(data[0], 0xFF);
}

TEST_F(SerializationBufferTest, PoolBasedFree_ReturnsToPool) {
  uint8_t* ptr_from_pool = nullptr;
  {
    SerializationBuffer buf;
    buf.Allocate(4096, pool_);
    ptr_from_pool = static_cast<uint8_t*>(buf.GetBuffer());
    // buf goes out of scope, should return buffer to pool_
  }

  // Re-acquire from pool should get same pointer back
  auto [ptr, bucket] = pool_.Acquire(4096);
  EXPECT_EQ(ptr, ptr_from_pool);
  pool_.Release(ptr, bucket);
}

TEST_F(SerializationBufferTest, GetCurrentPtr_AdvancesWithOffset) {
  SerializationBuffer buf;
  buf.Allocate(64, pool_);
  auto* base = static_cast<const uint8_t*>(buf.GetBuffer());

  EXPECT_EQ(buf.GetCurrentPtr(), base);

  buf.WriteUInt32(0, false);
  EXPECT_EQ(buf.GetCurrentPtr(), base + sizeof(uint32_t));
}

TEST_F(SerializationBufferTest, HexString_ReturnsCorrectLength) {
  SerializationBuffer buf;
  buf.Allocate(64, pool_);
  memset(buf.GetBuffer(), 0xAB, 64);

  std::string hex = buf.HexString(4);
  EXPECT_EQ(hex.size(), 8u);  // 4 bytes -> 8 hex chars
}
