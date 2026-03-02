#include "partition_tracker.h"

#include <algorithm>
#include <sstream>

#include "sched_policy.h"
#include "utils/logger.h"

namespace morphling {
namespace backend {

PartitionTracker& PartitionTracker::GetInstance() {
  static PartitionTracker instance;
  return instance;
}

PartitionTracker::PartitionTracker() {}

void PartitionTracker::AddPartition(int64_t device_id,
                                    const std::string& partition_key,
                                    int64_t oid, MatrixPartitionPtr partition) {
  std::lock_guard<std::mutex> lock(mutex_);

  auto info =
      std::make_shared<PartitionInfo>(partition_key, oid, device_id, partition);
  device_partitions_[device_id].push_back(info);
  partitions_set_.insert(info);
  partition_map_[partition_key] = info;
  partition_to_device_[partition_key] = device_id;

  LOG_DEBUG << "[PartitionTracker::AddPartition] Added partition "
            << partition_key << " (oid=" << oid << ") to device " << device_id
            << ", partition->dev_id=" << partition->dev_id;
}

void PartitionTracker::RemovePartition(int64_t device_id,
                                       const std::string& partition_key) {
  std::lock_guard<std::mutex> lock(mutex_);

  auto it = device_partitions_.find(device_id);
  if (it != device_partitions_.end()) {
    auto& parts = it->second;
    auto part_it = std::find_if(parts.begin(), parts.end(),
                                [&partition_key](const PartitionInfoPtr& p) {
                                  return p->key == partition_key;
                                });

    if (part_it != parts.end()) {
      partitions_set_.erase(*part_it);
      parts.erase(part_it);
      partition_map_.erase(partition_key);
      partition_to_device_.erase(partition_key);
      LOG_DEBUG << "[PartitionTracker] Removed partition " << partition_key
                << " from device " << device_id;

      // Clean up empty partition list
      if (parts.empty()) {
        device_partitions_.erase(it);
      }
    }
  }
}

void PartitionTracker::RemovePartitionByKey(const std::string& partition_key) {
  std::lock_guard<std::mutex> lock(mutex_);

  // Find device that has this partition
  auto rev_it = partition_to_device_.find(partition_key);
  if (rev_it == partition_to_device_.end()) {
    LOG_WARN << "[PartitionTracker::RemovePartitionByKey] Partition "
             << partition_key << " not found in any device";
    return;
  }

  int64_t device_id = rev_it->second;
  partition_to_device_.erase(rev_it);

  // Remove from partition_map_
  auto map_it = partition_map_.find(partition_key);
  if (map_it != partition_map_.end()) {
    partitions_set_.erase(map_it->second);
    partition_map_.erase(map_it);
  }

  // Remove from device's partition list
  auto it = device_partitions_.find(device_id);
  if (it != device_partitions_.end()) {
    auto& parts = it->second;
    auto part_it = std::find_if(parts.begin(), parts.end(),
                                [&partition_key](const PartitionInfoPtr& p) {
                                  return p->key == partition_key;
                                });

    if (part_it != parts.end()) {
      parts.erase(part_it);
      LOG_DEBUG << "[PartitionTracker::RemovePartitionByKey] Removed partition "
                << partition_key << " from device " << device_id
                << " (by key lookup)";

      if (parts.empty()) {
        device_partitions_.erase(it);
        LOG_DEBUG << "[PartitionTracker::RemovePartitionByKey] Device "
                  << device_id << " has no more partitions, removed from map";
      }
    }
  }
}

void PartitionTracker::MarkDevicePartitionsFailed(int64_t device_id) {
  std::lock_guard<std::mutex> lock(mutex_);

  auto it = device_partitions_.find(device_id);
  if (it == device_partitions_.end()) {
    LOG_DEBUG << "[PartitionTracker] No partitions found for device "
              << device_id;
    return;
  }

  size_t marked_count = 0;
  for (auto& part_info : it->second) {
    // Only mark RUNNING partitions as FAILED (not IDLE or already FINISHED)
    if (part_info->state == PartitionState::RUNNING) {
      part_info->state = PartitionState::IDLE;
      // Clear ownership - partition is now orphaned
      part_info->owner_device_id = -1;
      marked_count++;
      LOG_DEBUG << "[PartitionTracker] Marked partition " << part_info->key
                << " as FAILED (oid=" << part_info->oid << ")";
    }
  }

  LOG_INFO << "[PartitionTracker] Marked " << marked_count
           << " RUNNING partitions as FAILED for device " << device_id;
}

void PartitionTracker::MarkPartitionRunning(const std::string& partition_key) {
  std::lock_guard<std::mutex> lock(mutex_);

  auto it = partition_map_.find(partition_key);
  if (it == partition_map_.end()) {
    LOG_WARN << "[PartitionTracker] Cannot mark partition " << partition_key
             << " as RUNNING: not found";
    return;
  }

  it->second->state = PartitionState::RUNNING;
  LOG_DEBUG << "[PartitionTracker] Partition " << partition_key
            << " marked as RUNNING";
}

void PartitionTracker::MarkPartitionFinished(const std::string& partition_key) {
  std::lock_guard<std::mutex> lock(mutex_);

  auto it = partition_map_.find(partition_key);
  if (it == partition_map_.end()) {
    LOG_WARN << "[PartitionTracker] Cannot mark partition " << partition_key
             << " as FINISHED: not found";
    return;
  }

  it->second->state = PartitionState::FINISHED;
  LOG_DEBUG << "[PartitionTracker] Partition " << partition_key
            << " marked as FINISHED";
}

void PartitionTracker::MarkPartitionFailed(const std::string& partition_key) {
  std::lock_guard<std::mutex> lock(mutex_);

  auto it = partition_map_.find(partition_key);
  if (it == partition_map_.end()) {
    LOG_WARN << "[PartitionTracker] Cannot mark partition " << partition_key
             << " as FAILED: not found";
    return;
  }

  it->second->state = PartitionState::IDLE;
  LOG_DEBUG << "[PartitionTracker] Partition " << partition_key
            << " marked as FAILED";
}

void PartitionTracker::MarkPartitionIdle(const std::string& partition_key) {
  std::lock_guard<std::mutex> lock(mutex_);

  auto it = partition_map_.find(partition_key);
  if (it == partition_map_.end()) {
    LOG_WARN << "[PartitionTracker] Cannot mark partition " << partition_key
             << " as IDLE: not found";
    return;
  }

  it->second->state = PartitionState::IDLE;
  LOG_DEBUG << "[PartitionTracker] Partition " << partition_key
            << " marked as IDLE";
}

void PartitionTracker::MarkDevicePartitionsRunning(int64_t device_id) {
  std::lock_guard<std::mutex> lock(mutex_);

  auto it = device_partitions_.find(device_id);
  if (it == device_partitions_.end()) {
    return;
  }

  size_t marked_count = 0;
  for (auto& part : it->second) {
    if (part->state == PartitionState::IDLE) {
      part->state = PartitionState::RUNNING;
      marked_count++;
    }
  }

  LOG_DEBUG << "[PartitionTracker] Marked " << marked_count
            << " IDLE partitions as RUNNING for device " << device_id;
}

std::vector<PartitionInfoPtr> PartitionTracker::GetDevicePartitions(
    int64_t device_id) const {
  std::lock_guard<std::mutex> lock(mutex_);

  auto it = device_partitions_.find(device_id);
  if (it != device_partitions_.end()) {
    return it->second;
  }
  return {};
}

size_t PartitionTracker::GetDevicePartitionCount(int64_t device_id) const {
  std::lock_guard<std::mutex> lock(mutex_);

  auto it = device_partitions_.find(device_id);
  return (it != device_partitions_.end()) ? it->second.size() : 0;
}

bool PartitionTracker::HasPendingPartitions(int64_t device_id) const {
  std::lock_guard<std::mutex> lock(mutex_);

  auto it = device_partitions_.find(device_id);
  return it != device_partitions_.end() && !it->second.empty();
}

std::vector<PartitionInfoPtr> PartitionTracker::GetIdlePartitions() const {
  std::lock_guard<std::mutex> lock(mutex_);

  std::vector<PartitionInfoPtr> idle_partitions;
  for (const auto& part_info : partitions_set_) {
    if (part_info->state == PartitionState::IDLE) {
      idle_partitions.push_back(part_info);
    }
  }
  return idle_partitions;
}

PartitionTracker::DeviceOidStats PartitionTracker::GetDeviceOidStats(
    int64_t device_id, int64_t oid) const {
  std::lock_guard<std::mutex> lock(mutex_);

  DeviceOidStats stats;
  auto it = device_partitions_.find(device_id);
  if (it == device_partitions_.end()) {
    return stats;
  }

  for (const auto& part : it->second) {
    if (part->oid == oid) {
      switch (part->state) {
        case PartitionState::IDLE:
          stats.idle_count++;
          break;
        case PartitionState::RUNNING:
          stats.running_count++;
          break;
        case PartitionState::FINISHED:
          stats.finished_count++;
          break;
      }
      stats.partition_keys.push_back(part->key);
    }
  }
  return stats;
}

void PartitionTracker::RedistributeFailedDevicePartitions(
    int64_t failed_device_id, PartitionSchedulingPolicyPtr policy) {
  std::lock_guard<std::mutex> lock(mutex_);

  // Get failed device's partitions
  auto it = device_partitions_.find(failed_device_id);
  if (it == device_partitions_.end() || it->second.empty()) {
    LOG_INFO << "[PartitionTracker] Device " << failed_device_id
             << " has no pending partitions to redistribute";
    return;
  }

  // Count FAILED partitions (only redistribute those that were running when
  // device failed)
  std::vector<PartitionInfoPtr> partitions_to_redistribute;
  std::unordered_map<int64_t, size_t> oid_counts;
  for (const auto& part : it->second) {
    if (part->state == PartitionState::IDLE) {
      partitions_to_redistribute.push_back(part);
      oid_counts[part->oid]++;
    }
  }

  if (partitions_to_redistribute.empty()) {
    LOG_INFO << "[PartitionTracker] No FAILED partitions to redistribute "
                "from device "
             << failed_device_id;
    return;
  }

  LOG_INFO << "[PartitionTracker] Redistributing "
           << partitions_to_redistribute.size() << " partitions from device "
           << failed_device_id << " (" << oid_counts.size() << " OIDs)";

  // Use provided policy or default to round-robin
  if (!policy) {
    policy = std::make_shared<RoundRobinSchedulingPolicy>();
  }

  // Get redistribution mapping from policy
  auto redistribution_map =
      policy->RedistributePartitions(partitions_to_redistribute);

  // Apply the redistribution
  for (auto& part : partitions_to_redistribute) {
    auto map_it = redistribution_map.find(part->key);
    if (map_it == redistribution_map.end()) {
      LOG_ERROR << "[PartitionTracker] No target device found for partition "
                << part->key;
      continue;
    }

    int64_t target_device_id = map_it->second;

    // Update ownership and reset state to IDLE
    part->owner_device_id = target_device_id;
    part->state = PartitionState::IDLE;

    // Add to target device's partition list
    device_partitions_[target_device_id].push_back(part);

    // Update reverse index
    partition_to_device_[part->key] = target_device_id;

    LOG_DEBUG << "[PartitionTracker] Redistributed partition " << part->key
              << " (oid=" << part->oid << ") from device " << failed_device_id
              << " to device " << target_device_id;
  }

  // Remove failed device's partitions
  device_partitions_.erase(failed_device_id);

  LOG_INFO << "[PartitionTracker] Redistribution complete";
}

void PartitionTracker::ClearDevicePartitions(int64_t device_id) {
  std::lock_guard<std::mutex> lock(mutex_);

  auto it = device_partitions_.find(device_id);
  if (it == device_partitions_.end()) {
    return;
  }

  // Remove all partitions from sets and maps
  for (const auto& part : it->second) {
    partitions_set_.erase(part);
    partition_map_.erase(part->key);
    partition_to_device_.erase(part->key);
  }

  device_partitions_.erase(device_id);
  LOG_DEBUG << "[PartitionTracker] Cleared all partitions for device "
            << device_id;
}

void PartitionTracker::InitializeDeviceTensors(size_t device_count) {
  std::lock_guard<std::mutex> lock(mutex_);
  device_tensors_.clear();
  // Note: device_tensors_ will be populated on-demand as devices are used
}

void PartitionTracker::AddTensorToDevice(int64_t device_id,
                                         const TensorKey& tensor_key) {
  std::lock_guard<std::mutex> lock(mutex_);
  device_tensors_[device_id].insert(tensor_key);
}

const std::unordered_set<TensorKey>& PartitionTracker::GetDeviceTensors(
    int64_t device_id) const {
  std::lock_guard<std::mutex> lock(mutex_);
  static const std::unordered_set<TensorKey> empty_set;
  auto it = device_tensors_.find(device_id);
  return (it != device_tensors_.end()) ? it->second : empty_set;
}

void PartitionTracker::ClearDeviceTensors(int64_t device_id) {
  std::lock_guard<std::mutex> lock(mutex_);
  device_tensors_.erase(device_id);
}

void PartitionTracker::ClearAllDeviceTensors() {
  std::lock_guard<std::mutex> lock(mutex_);
  device_tensors_.clear();
}

std::string PartitionTracker::DebugString() const {
  std::lock_guard<std::mutex> lock(mutex_);

  std::ostringstream oss;
  oss << "PartitionTracker{\n";
  oss << "  Total partitions: " << partitions_set_.size() << "\n";
  oss << "  Devices with partitions: " << device_partitions_.size() << "\n";

  // Count by state
  size_t idle = 0, running = 0, finished = 0;
  for (const auto& part : partitions_set_) {
    switch (part->state) {
      case PartitionState::IDLE:
        idle++;
        break;
      case PartitionState::RUNNING:
        running++;
        break;
      case PartitionState::FINISHED:
        finished++;
        break;
    }
  }

  oss << "  By state: IDLE=" << idle << ", RUNNING=" << running
      << ", FINISHED=" << finished << "\n";
  oss << "}";
  return oss.str();
}

void PartitionTracker::DumpState() const {
  LOG_INFO << "[PartitionTracker] " << DebugString();
}

void PartitionTracker::Reset() {
  std::lock_guard<std::mutex> lock(mutex_);

  LOG_INFO << "[PartitionTracker] Resetting all state";
  device_partitions_.clear();
  partitions_set_.clear();
  partition_map_.clear();
  partition_to_device_.clear();
  device_tensors_.clear();
}

}  // namespace backend
}  // namespace morphling
