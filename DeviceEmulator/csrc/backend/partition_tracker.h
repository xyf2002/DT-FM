#pragma once

#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "common/pytorch_defs.h"
#include "server_base.h"

// Convenient macro for accessing the PartitionTracker singleton
#define PARTITION_TRACKER morphling::backend::PartitionTracker::GetInstance()

namespace morphling {
namespace backend {

// Forward declarations
class PartitionSchedulingPolicy;
typedef std::shared_ptr<PartitionSchedulingPolicy> PartitionSchedulingPolicyPtr;

// Partition execution state
enum class PartitionState {
  IDLE = 0,     // Created but not yet sent to device
  RUNNING = 1,  // Sent to device, waiting for response
  // FAILED = 2,   // Device failed while processing
  FINISHED = 3  // Response received, computation complete
};

// Partition tracking structure with ownership and OID tracking
struct PartitionInfo {
  std::string key;
  int64_t oid;  // Operation ID to track which MatMul this partition belongs to
  int64_t owner_device_id;       // Device that owns this partition
  PartitionState state;          // Current execution state of the partition
  MatrixPartitionPtr partition;  // Shared pointer to partition data

  PartitionInfo() : oid(-1), owner_device_id(-1), state(PartitionState::IDLE) {}
  PartitionInfo(const std::string& k, int64_t o, int64_t owner,
                MatrixPartitionPtr p)
      : key(k),
        oid(o),
        owner_device_id(owner),
        state(PartitionState::IDLE),
        partition(p) {}
};
typedef std::shared_ptr<PartitionInfo> PartitionInfoPtr;

// Partition tracker - manages partition lifecycle and assignments
// Thread-safe singleton component for tracking:
// 1. Partition assignments per device
// 2. Partition state transitions
// 3. Partition redistribution on device failure
// 4. Device tensor cache for cache-aware scheduling
class PartitionTracker {
 public:
  static PartitionTracker& GetInstance();

  // Disable copy and move
  PartitionTracker(const PartitionTracker&) = delete;
  PartitionTracker& operator=(const PartitionTracker&) = delete;
  PartitionTracker(PartitionTracker&&) = delete;
  PartitionTracker& operator=(PartitionTracker&&) = delete;

  // Partition lifecycle management
  void AddPartition(int64_t device_id, const std::string& partition_key,
                    int64_t oid, MatrixPartitionPtr partition);
  void RemovePartition(int64_t device_id, const std::string& partition_key);
  void RemovePartitionByKey(const std::string& partition_key);

  // Mark all partitions owned by a device as failed (ownership removed)
  void MarkDevicePartitionsFailed(int64_t device_id);

  // State management for partitions
  void MarkPartitionRunning(const std::string& partition_key);
  void MarkPartitionFinished(const std::string& partition_key);
  void MarkPartitionFailed(const std::string& partition_key);
  void MarkPartitionIdle(const std::string& partition_key);
  void MarkDevicePartitionsRunning(
      int64_t device_id);  // Mark all IDLE partitions as RUNNING

  // Partition queries
  std::vector<PartitionInfoPtr> GetDevicePartitions(int64_t device_id) const;
  size_t GetDevicePartitionCount(int64_t device_id) const;
  bool HasPendingPartitions(int64_t device_id) const;
  std::vector<PartitionInfoPtr> GetIdlePartitions() const;

  // Statistics for a specific OID on a device (for debugging)
  struct DeviceOidStats {
    size_t idle_count = 0;
    size_t running_count = 0;
    size_t finished_count = 0;
    std::vector<std::string> partition_keys;  // For detailed inspection
  };
  DeviceOidStats GetDeviceOidStats(int64_t device_id, int64_t oid) const;

  // Partition redistribution on device failure
  void RedistributeFailedDevicePartitions(
      int64_t failed_device_id, PartitionSchedulingPolicyPtr policy = nullptr);

  // Device cleanup on disconnect
  void ClearDevicePartitions(int64_t device_id);

  // Tensor cache management
  void InitializeDeviceTensors(size_t device_count);
  void AddTensorToDevice(int64_t device_id, const TensorKey& tensor_key);
  const std::unordered_set<TensorKey>& GetDeviceTensors(
      int64_t device_id) const;
  void ClearDeviceTensors(int64_t device_id);
  void ClearAllDeviceTensors();

  // Debug and monitoring
  std::string DebugString() const;
  void DumpState() const;

  // Clear all state (for testing)
  void Reset();

 private:
  PartitionTracker();
  ~PartitionTracker() = default;

  // State
  mutable std::mutex mutex_;

  // Partition tracking: device_id -> [PartitionInfo, ...]
  std::unordered_set<PartitionInfoPtr> partitions_set_;
  std::unordered_map<std::string, PartitionInfoPtr> partition_map_;
  std::unordered_map<int64_t, std::vector<PartitionInfoPtr>> device_partitions_;

  // Reverse index: partition_key -> device_id (for fast lookup)
  std::unordered_map<std::string, int64_t> partition_to_device_;

  // Device tensor cache tracking: device_id -> set of tensor keys
  std::unordered_map<int64_t, std::unordered_set<TensorKey>> device_tensors_;
};

}  // namespace backend
}  // namespace morphling
