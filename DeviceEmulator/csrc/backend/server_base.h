#pragma once

#include <sys/mman.h>
#include <torch/torch.h>

#include <charconv>
#include <cstring>
#include <deque>
#include <functional>
#include <future>
#include <mutex>
#include <unordered_map>

#include "common/types_and_defs.h"
#include "global_api.pb.h"
#include "morphling.pb.h"
#include "network/uevent.h"

// ============================================================================
// AlignedBufferPool: Thread-safe pool of page-aligned, mlocked buffers
// ============================================================================

class AlignedBufferPool {
 public:
  static constexpr size_t PAGE_SIZE = 4096;
  static constexpr size_t MAX_BUFFERS_PER_BUCKET = 32;

  // Pluggable pin/unpin callbacks. Return 0 on success.
  using PinFn = int (*)(void* ptr, size_t size);
  using UnpinFn = void (*)(void* ptr, size_t size);

  static AlignedBufferPool& instance() {
    static AlignedBufferPool pool;
    return pool;
  }

  // Override the default mlock/munlock pinning strategy.
  // Must be called before any Acquire() (e.g. at process startup).
  void SetPinFunctions(PinFn pin, UnpinFn unpin) {
    pin_fn_ = pin;
    unpin_fn_ = unpin;
  }

  // Acquire a buffer of at least `size` bytes (page-aligned, pinned)
  // Returns {pointer, actual_bucket_size}
  std::pair<uint8_t*, size_t> Acquire(size_t size) {
    size_t bucket = BucketSize(size);
    std::lock_guard<std::mutex> lock(mutex_);
    auto& free_list = free_lists_[bucket];
    if (!free_list.empty()) {
      uint8_t* ptr = free_list.back();
      free_list.pop_back();
      return {ptr, bucket};
    }
    // Allocate new buffer
    return {AllocateNewBuffer(bucket), bucket};
  }

  // Release a buffer back to the pool
  void Release(uint8_t* ptr, size_t bucket_size) {
    if (!ptr) return;
    std::lock_guard<std::mutex> lock(mutex_);
    auto& free_list = free_lists_[bucket_size];
    if (free_list.size() < MAX_BUFFERS_PER_BUCKET) {
      free_list.push_back(ptr);
    } else {
      // Pool is full, free the buffer
      unpin_fn_(ptr, bucket_size);
      free(ptr);
    }
  }

  ~AlignedBufferPool() {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto& [bucket_size, free_list] : free_lists_) {
      for (auto* ptr : free_list) {
        unpin_fn_(ptr, bucket_size);
        free(ptr);
      }
    }
    free_lists_.clear();
  }

  // Public for testing. Production code should use instance().
  AlignedBufferPool() = default;

 private:
  AlignedBufferPool(const AlignedBufferPool&) = delete;
  AlignedBufferPool& operator=(const AlignedBufferPool&) = delete;

  static int DefaultPin(void* ptr, size_t size) { return mlock(ptr, size); }

  static void DefaultUnpin(void* ptr, size_t size) { munlock(ptr, size); }

  // Round up to the next power-of-two bucket size (min 4KB)
  static size_t BucketSize(size_t size) {
    if (size <= PAGE_SIZE) return PAGE_SIZE;
    // Round up to next power of 2
    size_t bucket = PAGE_SIZE;
    while (bucket < size) bucket <<= 1;
    return bucket;
  }

  uint8_t* AllocateNewBuffer(size_t size) {
    uint8_t* ptr = nullptr;
    int ret = posix_memalign(reinterpret_cast<void**>(&ptr), PAGE_SIZE, size);
    if (ret != 0 || !ptr) {
      throw std::runtime_error("AlignedBufferPool: posix_memalign failed");
    }
    ret = pin_fn_(ptr, size);
    if (ret != 0) {
      // Pin failure is non-fatal, continue without pinning
      throw std::runtime_error("AlignedBufferPool: pin_fn_ failed");
    }
    return ptr;
  }

  PinFn pin_fn_ = DefaultPin;
  UnpinFn unpin_fn_ = DefaultUnpin;
  std::mutex mutex_;
  std::unordered_map<size_t, std::deque<uint8_t*>> free_lists_;
};

// Common message handler signature
// All message handlers follow this signature: (connection, payload, size)
using MessageHandlerSignature = void(const uevent::ConnectionUeventPtr&,
                                     const void*, size_t);
using MessageHandler = std::function<MessageHandlerSignature>;

inline std::string BinaryToHex(const unsigned char* data, size_t length) {
  std::string result(length * 2, '0');

  for (size_t i = 0; i < length; ++i) {
    std::to_chars(result.data() + i * 2, result.data() + i * 2 + 2, data[i],
                  16);
  }

  return result;
}

template <typename T>
std::vector<std::vector<T>> CartesianProduct(const std::vector<T>& list) {
  std::vector<std::vector<T>> ranges;
  for (auto i : list) {
    std::vector<T> range;
    for (T j = 0; j < i; ++j) {
      range.push_back(j);
    }
    ranges.push_back(range);
  }

  std::vector<std::vector<T>> result;
  if (ranges.empty()) return result;

  // Initialize with the first list
  result.push_back({});
  for (const auto& list : ranges) {
    std::vector<std::vector<T>> temp;
    for (const auto& res : result) {
      for (const auto& elem : list) {
        std::vector<T> new_combination = res;
        new_combination.push_back(elem);
        temp.push_back(new_combination);
      }
    }
    result = std::move(temp);
  }
  return result;
}

torch::Tensor CreateOutputMatrix(const torch::Tensor& mat_a,
                                 const torch::Tensor& mat_b);

void UpdateMatrixBlock(torch::Tensor& target, torch::Tensor& mat, int64_t r,
                       int64_t c, int64_t pivot, int64_t block_size);
void IndexPutMatrixBlock(torch::Tensor& target, torch::Tensor& mat, int64_t r,
                         int64_t c, int64_t pivot, int64_t block_size);

typedef std::tuple<void*, int64_t> PtrData;

// define hash function for PtrData
namespace std {
template <>
struct hash<PtrData> {
  std::size_t operator()(const PtrData& p) const {
    return std::hash<void*>{}(std::get<0>(p)) ^
           std::hash<int64_t>{}(std::get<1>(p));
  }
};
}  // namespace std

typedef std::tuple<uint64_t, int64_t, int64_t, bool>
    TensorKey;  // version, pivot, r/c, is_row

// create a spdlog fmt for TensorKey

// define custom hash function for TensorKey
namespace std {
template <>
struct hash<TensorKey> {
  std::size_t operator()(const TensorKey& k) const {
    return std::hash<uint64_t>{}(std::get<0>(k)) ^
           std::hash<int64_t>{}(std::get<1>(k)) ^
           std::hash<int64_t>{}(std::get<2>(k)) ^
           std::hash<bool>{}(std::get<3>(k));
  }
};
}  // namespace std

// ============================================================================
// Generic Serialization Framework
// ============================================================================

// Forward declarations
class SerializationBuffer;
typedef std::shared_ptr<SerializationBuffer> SerializationBufferPtr;

enum class SerializationFormat {
  PROTOBUF  // Protobuf-based format (default)
};

// Base interface for serializable messages
class ISerializable {
 public:
  virtual ~ISerializable() = default;

  // Serialize to binary format
  virtual SerializationBufferPtr Serialize(
      SerializationFormat format = SerializationFormat::PROTOBUF) const = 0;

  // Deserialize from binary format
  virtual void Deserialize(
      const void* data, size_t size,
      SerializationFormat format = SerializationFormat::PROTOBUF) = 0;

  // Get the message type
  virtual int32_t GetMessageType() const = 0;

  // Get debug string representation
  virtual std::string DebugString() const = 0;
};

// Buffer handler for reading/writing binary data
class SerializationBuffer {
 public:
  SerializationBuffer();
  explicit SerializationBuffer(const void* data, size_t size,
                               bool take_ownership = false);
  ~SerializationBuffer();

  // Non-copyable, movable
  SerializationBuffer(const SerializationBuffer&) = delete;
  SerializationBuffer& operator=(const SerializationBuffer&) = delete;
  SerializationBuffer(SerializationBuffer&& other) noexcept;
  SerializationBuffer& operator=(SerializationBuffer&& other) noexcept;

  // Allocate new buffer (uses AlignedBufferPool)
  void Allocate(size_t size);
  // Allocate using a specific pool (for testing)
  void Allocate(size_t size, AlignedBufferPool& pool);

  // Write methods
  void WriteUInt32(uint32_t value, bool network_order = false);
  void WriteUInt64(uint64_t value);
  void WriteInt64(int64_t value);
  void WriteBytes(const void* data, size_t size);

  // Read methods
  uint32_t ReadUInt32(bool network_order = false);
  uint64_t ReadUInt64();
  int64_t ReadInt64();
  void ReadBytes(void* dest, size_t size);
  const void* GetCurrentPtr() const;

  // Position management
  void SeekTo(size_t offset);
  size_t GetOffset() const { return offset_; }
  void* GetBuffer() const { return buffer_; }
  size_t GetSize() const { return size_; }

  // Validation
  bool CanRead(size_t bytes) const;
  void ValidateSize(size_t min_size) const;

  std::string HexString(size_t length) const;

 private:
  void FreeBuffer();

  uint8_t* buffer_;
  size_t size_;
  size_t offset_;
  bool owns_buffer_;
  size_t
      pool_bucket_size_;  // 0 = not from pool, >0 = bucket size for pool return
  AlignedBufferPool* pool_ = nullptr;  // nullptr = use singleton
};

// ============================================================================
// ScatterGatherBuffer: Zero-copy serialization with multiple segments
// ============================================================================

struct BufferSegment {
  const void* data;
  size_t size;
  bool owned;  // if true, data is freed on destruction

  BufferSegment(const void* d, size_t s, bool o) : data(d), size(s), owned(o) {}
};

class ScatterGatherBuffer {
 public:
  ScatterGatherBuffer() : pool_(&AlignedBufferPool::instance()) {}
  explicit ScatterGatherBuffer(AlignedBufferPool& pool) : pool_(&pool) {}
  ~ScatterGatherBuffer();

  ScatterGatherBuffer(ScatterGatherBuffer&& other) noexcept;
  ScatterGatherBuffer& operator=(ScatterGatherBuffer&& other) noexcept;
  ScatterGatherBuffer(const ScatterGatherBuffer&) = delete;
  ScatterGatherBuffer& operator=(const ScatterGatherBuffer&) = delete;

  // Add an owned segment (will be freed on destruction via pool)
  void AddOwnedSegment(uint8_t* data, size_t size, size_t pool_bucket);

  // Add a reference segment (not owned, caller must ensure lifetime)
  void AddReferenceSegment(const void* data, size_t size);

  size_t GetTotalSize() const;
  const std::vector<BufferSegment>& GetSegments() const { return segments_; }

 private:
  std::vector<BufferSegment> segments_;
  // Track pool bucket sizes for owned segments to return them
  std::vector<std::pair<uint8_t*, size_t>> owned_pool_entries_;
  AlignedBufferPool* pool_;
};

typedef std::shared_ptr<ScatterGatherBuffer> ScatterGatherBufferPtr;

// ============================================================================
// Message Structs
// ============================================================================

struct MatrixPartition : public ISerializable {
  uint64_t version;
  int64_t row;
  int64_t col;
  int64_t pivot;             // leading dimension
  int64_t h_dim;             // hidden dimension
  int64_t dev_id;            // device id
  int64_t oid;               // output matrix id for parallel matmul
  int64_t gemm_id;           // global gemm operation id
  uint64_t timestamp;        // timestamp on sending the message
  std::vector<PtrData> mat;  // if ptr is null and size is 0, means that this
                             // entry need to be fetched from cache first mat is
                             // row block, second mat is col block
  void* ptr_ = nullptr;      // pointer to the data
  size_t size_ = 0;          // size of the data

  // ISerializable interface
  SerializationBufferPtr Serialize(
      SerializationFormat format =
          SerializationFormat::PROTOBUF) const override;
  void Deserialize(
      const void* data, size_t size,
      SerializationFormat format = SerializationFormat::PROTOBUF) override;
  int32_t GetMessageType() const override;

  // Public helper methods
  TensorKey GetRowKey() const {
    return std::make_tuple(version, pivot, row, true);
  }

  TensorKey GetColKey() const {
    return std::make_tuple(version, pivot, col, false);
  }

  int64_t Size() const {
    int64_t size = 0;
    for (const auto& m : mat) {
      size += std::get<1>(m);
    }
    return size;
  }

  std::string GetPartitionKey() const {
    // [version:pivot:row:col]
    return "[" + std::to_string(version) + ":" + std::to_string(pivot) + ":" +
           std::to_string(row) + ":" + std::to_string(col) + "]";
  }

  std::string DebugString() const;

  // Zero-copy scatter-gather serialization (avoids tensor memcpy)
  ScatterGatherBufferPtr SerializeZeroCopy() const;

 private:
  // Format-specific implementations
  SerializationBufferPtr SerializeProto() const;
  void DeserializeProto(const void* data, size_t size);

  // Helper methods for metadata
  void WriteMetadataToBuffer(SerializationBuffer& buffer) const;
  void ReadMetadataFromBuffer(SerializationBuffer& buffer);
  void ReadMatricesData(SerializationBuffer& buffer, size_t end_offset);
};

typedef std::shared_ptr<MatrixPartition> MatrixPartitionPtr;

// Device registration request (server -> client)

// Device registration request (server -> client)
struct DeviceRegisterRequest : public ISerializable {
  // Empty request

  // ISerializable interface
  SerializationBufferPtr Serialize(
      SerializationFormat format =
          SerializationFormat::PROTOBUF) const override;
  void Deserialize(
      const void* data, size_t size,
      SerializationFormat format = SerializationFormat::PROTOBUF) override;
  int32_t GetMessageType() const override;
  std::string DebugString() const override { return "DeviceRegisterRequest"; }

 private:
  SerializationBufferPtr SerializeProto() const;
  void DeserializeProto(const void* data, size_t size);
};
typedef std::shared_ptr<DeviceRegisterRequest> DeviceRegisterRequestPtr;

// Device profile data (client -> server)
// Contains device performance characteristics and capabilities
struct DeviceProfileData : public ISerializable {
  uint64_t uuid;
  uint64_t flops;
  uint64_t memory;
  uint64_t ul_bw;   // upload bandwidth
  uint64_t dl_bw;   // download bandwidth
  uint64_t ul_lat;  // upload latency
  uint64_t dl_lat;  // download latency

  // ISerializable interface
  SerializationBufferPtr Serialize(
      SerializationFormat format =
          SerializationFormat::PROTOBUF) const override;
  void Deserialize(
      const void* data, size_t size,
      SerializationFormat format = SerializationFormat::PROTOBUF) override;
  int32_t GetMessageType() const override;
  std::string DebugString() const;

 private:
  SerializationBufferPtr SerializeProto() const;
  void DeserializeProto(const void* data, size_t size);
};
typedef std::shared_ptr<DeviceProfileData> DeviceProfileDataPtr;

// Legacy aliases for backward compatibility
typedef DeviceProfileData DeviceRegisterResponse;
typedef std::shared_ptr<DeviceRegisterResponse> DeviceRegisterResponsePtr;
typedef DeviceProfileData DevicePerf;
typedef std::shared_ptr<DevicePerf> DevicePerfPtr;

// enum TimerType { kTimerGet, kTimerPut };

// struct Timer {
//   uint8_t type;
//   uint64_t time;
// };

MatrixPartitionPtr CalculateMatrixPartition(const torch::Tensor& mat_a,
                                            const torch::Tensor& mat_b,
                                            int64_t r, int64_t c, int64_t pivot,
                                            int64_t block_size);

// The computation must be AB^T
// Every matrix is partitioned into blocks on rows since torch is row major
std::vector<MatrixPartitionPtr> PartitionMatrices(const torch::Tensor& mat_a,
                                                  const torch::Tensor& mat_b,
                                                  int64_t block_size);
