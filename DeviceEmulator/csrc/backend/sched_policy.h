#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "common/pytorch_defs.h"
#include "server_base.h"

namespace morphling {
namespace backend {

// Forward declarations
struct PartitionInfo;
typedef std::shared_ptr<PartitionInfo> PartitionInfoPtr;

// Scheduling policy interface for partition assignment
class PartitionSchedulingPolicy {
 public:
  PartitionSchedulingPolicy(int block_size = 0, bool enable_cache = true)
      : block_size_(block_size), enable_cache_(enable_cache) {}
  virtual ~PartitionSchedulingPolicy() = default;

  // Assign partitions to devices during initial dispatch
  // Returns device_id for each partition (by index)
  virtual std::vector<int64_t> AssignPartitionsToDevices(
      const std::vector<MatrixPartitionPtr>& partitions,
      const std::unordered_set<int64_t>& excluded_devices = {}) = 0;

  // Redistribute failed partitions to available devices
  // Returns mapping of partition -> target_device_id
  virtual std::unordered_map<std::string, int64_t> RedistributePartitions(
      const std::vector<PartitionInfoPtr>& partitions) = 0;

 protected:
  int block_size_;
  bool enable_cache_;
};

typedef std::shared_ptr<PartitionSchedulingPolicy> PartitionSchedulingPolicyPtr;

// Round-robin scheduling policy
class RoundRobinSchedulingPolicy : public PartitionSchedulingPolicy {
 public:
  RoundRobinSchedulingPolicy(int block_size = 0, bool enable_cache = true)
      : PartitionSchedulingPolicy(block_size, enable_cache) {}
  ~RoundRobinSchedulingPolicy() override = default;

  std::vector<int64_t> AssignPartitionsToDevices(
      const std::vector<MatrixPartitionPtr>& partitions,
      const std::unordered_set<int64_t>& excluded_devices = {}) override;

  std::unordered_map<std::string, int64_t> RedistributePartitions(
      const std::vector<PartitionInfoPtr>& partitions) override;

 private:
  size_t next_device_idx_ = 0;
};

// Greedy scheduling policy (existing behavior with cost calculation)
class GreedySchedulingPolicy : public PartitionSchedulingPolicy {
 public:
  explicit GreedySchedulingPolicy(int block_size, bool enable_cache = true)
      : PartitionSchedulingPolicy(block_size, enable_cache) {}
  ~GreedySchedulingPolicy() override = default;

  std::vector<int64_t> AssignPartitionsToDevices(
      const std::vector<MatrixPartitionPtr>& partitions,
      const std::unordered_set<int64_t>& excluded_devices = {}) override;

  std::unordered_map<std::string, int64_t> RedistributePartitions(
      const std::vector<PartitionInfoPtr>& partitions) override;
};

// Load-balanced scheduling policy (considers current device load)
class LoadBalancedSchedulingPolicy : public PartitionSchedulingPolicy {
 public:
  LoadBalancedSchedulingPolicy(int block_size = 0, bool enable_cache = true)
      : PartitionSchedulingPolicy(block_size, enable_cache) {}
  ~LoadBalancedSchedulingPolicy() override = default;

  std::vector<int64_t> AssignPartitionsToDevices(
      const std::vector<MatrixPartitionPtr>& partitions,
      const std::unordered_set<int64_t>& excluded_devices = {}) override;

  std::unordered_map<std::string, int64_t> RedistributePartitions(
      const std::vector<PartitionInfoPtr>& partitions) override;
};

}  // namespace backend
}  // namespace morphling
