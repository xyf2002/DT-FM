#include "sched_policy.h"

#include <algorithm>
#include <limits>

#include "device_tracker.h"
#include "partition_tracker.h"
#include "utils/logger.h"

namespace morphling {
namespace backend {

// ============================================================================
// RoundRobinSchedulingPolicy
// ============================================================================

std::vector<int64_t> RoundRobinSchedulingPolicy::AssignPartitionsToDevices(
    const std::vector<MatrixPartitionPtr>& partitions,
    const std::unordered_set<int64_t>& excluded_devices) {
  std::vector<int64_t> assignments;
  assignments.reserve(partitions.size());

  // Get connected devices from tracker
  auto& tracker = DEVICE_TRACKER;
  std::vector<int64_t> connected_devices = tracker.GetConnectedDevices();

  // Filter out excluded devices
  std::vector<int64_t> eligible_devices;
  for (int64_t device_id : connected_devices) {
    if (excluded_devices.find(device_id) == excluded_devices.end()) {
      eligible_devices.push_back(device_id);
    }
  }

  if (eligible_devices.empty()) {
    LOG_ERROR << "[RoundRobinScheduling] No eligible devices available";
    return assignments;
  }

  // Initialize tensor cache for eligible devices
  PARTITION_TRACKER.ClearAllDeviceTensors();

  // Assign partitions round-robin
  for (size_t i = 0; i < partitions.size(); ++i) {
    int device_idx = next_device_idx_;
    int64_t device_id = eligible_devices[device_idx];
    assignments.push_back(device_id);

    // Track tensor keys for cache awareness
    auto tensor_key_row = partitions[i]->GetRowKey();
    auto tensor_key_col = partitions[i]->GetColKey();
    PARTITION_TRACKER.AddTensorToDevice(device_id, tensor_key_row);
    PARTITION_TRACKER.AddTensorToDevice(device_id, tensor_key_col);

    next_device_idx_ = (next_device_idx_ + 1) % eligible_devices.size();
  }

  LOG_INFO << "[RoundRobinScheduling] Assigned " << partitions.size()
           << " partitions across " << eligible_devices.size() << " devices";

  return assignments;
}

std::unordered_map<std::string, int64_t>
RoundRobinSchedulingPolicy::RedistributePartitions(
    const std::vector<PartitionInfoPtr>& partitions) {
  std::unordered_map<std::string, int64_t> redistribution;

  // Get connected devices from tracker
  auto& tracker = DEVICE_TRACKER;
  std::vector<int64_t> connected_devices = tracker.GetConnectedDevices();

  if (connected_devices.empty()) {
    LOG_ERROR << "[RoundRobinScheduling] No available devices for "
                 "redistribution";
    return redistribution;
  }

  size_t device_idx = 0;
  for (const auto& part : partitions) {
    int64_t target_device_id = connected_devices[device_idx];
    redistribution[part->key] = target_device_id;
    part->owner_device_id = target_device_id;
    device_idx = (device_idx + 1) % connected_devices.size();
  }

  LOG_INFO << "[RoundRobinScheduling] Redistributed " << partitions.size()
           << " partitions across " << connected_devices.size() << " devices";

  return redistribution;
}

// ============================================================================
// GreedySchedulingPolicy
// ============================================================================

std::vector<int64_t> GreedySchedulingPolicy::AssignPartitionsToDevices(
    const std::vector<MatrixPartitionPtr>& partitions,
    const std::unordered_set<int64_t>& excluded_devices) {
  std::vector<int64_t> assignments;
  assignments.reserve(partitions.size());

  // Get connected devices from tracker
  auto& tracker = DEVICE_TRACKER;
  std::vector<int64_t> connected_devices = tracker.GetConnectedDevices();

  // Filter out excluded devices
  std::vector<int64_t> eligible_devices;
  for (int64_t device_id : connected_devices) {
    if (excluded_devices.find(device_id) == excluded_devices.end()) {
      eligible_devices.push_back(device_id);
    }
  }

  if (eligible_devices.empty()) {
    LOG_ERROR << "[GreedyScheduling] No eligible devices available";
    return assignments;
  }

  // Initialize tensor cache for eligible devices
  PARTITION_TRACKER.ClearAllDeviceTensors();

  int actual_num_devices = static_cast<int>(eligible_devices.size());
  std::vector<float> device_time(actual_num_devices, 0);

  // Greedy algorithm to select the device with minimal time
  for (const auto& partition : partitions) {
    float min_time = std::numeric_limits<float>::max();
    int min_device_idx = 0;

    auto tensor_key_row = partition->GetRowKey();
    auto tensor_key_col = partition->GetColKey();

    for (int i = 0; i < actual_num_devices; i++) {
      int64_t device_id = eligible_devices[i];
      const auto& tensors = PARTITION_TRACKER.GetDeviceTensors(device_id);

      bool r_cached = tensors.find(tensor_key_row) != tensors.end();
      bool c_cached = tensors.find(tensor_key_col) != tensors.end();

      auto r_size = std::get<1>(partition->mat[0]);
      auto c_size = std::get<1>(partition->mat[1]);
      auto cached_r_size = (r_cached) ? 0 : r_size;
      auto cached_c_size = (c_cached) ? 0 : c_size;

      int64_t num_rows = r_size / partition->h_dim / sizeof(float);
      int64_t num_cols = c_size / partition->h_dim / sizeof(float);

      float ul_time = (float)(num_rows * num_cols) * sizeof(float) / MB;
      float dl_time = (float)(cached_r_size + cached_c_size) / MB;
      float flops = (float)2.0 * num_rows * num_cols * partition->h_dim / TB;

      float time = std::max(std::max(ul_time, dl_time), flops) + device_time[i];
      if (time < min_time) {
        min_time = time;
        min_device_idx = i;
      }
    }

    device_time[min_device_idx] = min_time;
    int64_t assigned_device_id = eligible_devices[min_device_idx];
    assignments.push_back(assigned_device_id);
    PARTITION_TRACKER.AddTensorToDevice(assigned_device_id, tensor_key_row);
    PARTITION_TRACKER.AddTensorToDevice(assigned_device_id, tensor_key_col);
  }

  LOG_INFO << "[GreedyScheduling] Assigned " << partitions.size()
           << " partitions across " << eligible_devices.size()
           << " devices using greedy algorithm";

  return assignments;
}

std::unordered_map<std::string, int64_t>
GreedySchedulingPolicy::RedistributePartitions(
    const std::vector<PartitionInfoPtr>& partitions) {
  // For redistribution, fall back to round-robin as we don't have
  // partition details to compute costs
  RoundRobinSchedulingPolicy rr_policy;
  return rr_policy.RedistributePartitions(partitions);
}

// ============================================================================
// LoadBalancedSchedulingPolicy
// ============================================================================

std::vector<int64_t> LoadBalancedSchedulingPolicy::AssignPartitionsToDevices(
    const std::vector<MatrixPartitionPtr>& partitions,
    const std::unordered_set<int64_t>& excluded_devices) {
  std::vector<int64_t> assignments;
  assignments.reserve(partitions.size());

  // Get connected devices from tracker
  auto& tracker = DEVICE_TRACKER;
  std::vector<int64_t> connected_devices = tracker.GetConnectedDevices();

  // Filter out excluded devices
  std::vector<int64_t> eligible_devices;
  for (int64_t device_id : connected_devices) {
    if (excluded_devices.find(device_id) == excluded_devices.end()) {
      eligible_devices.push_back(device_id);
    }
  }

  if (eligible_devices.empty()) {
    LOG_ERROR << "[LoadBalancedScheduling] No eligible devices available";
    return assignments;
  }

  // Assign each partition to the device with the least current load
  for (const auto& partition : partitions) {
    int64_t best_device = eligible_devices[0];
    size_t min_partitions =
        PARTITION_TRACKER.GetDevicePartitionCount(best_device);

    for (size_t i = 1; i < eligible_devices.size(); ++i) {
      int64_t device_id = eligible_devices[i];
      size_t partition_count =
          PARTITION_TRACKER.GetDevicePartitionCount(device_id);

      if (partition_count < min_partitions) {
        min_partitions = partition_count;
        best_device = device_id;
      }
    }

    assignments.push_back(best_device);
  }

  LOG_INFO << "[LoadBalancedScheduling] Assigned " << partitions.size()
           << " partitions across " << eligible_devices.size()
           << " devices using load balancing";

  return assignments;
}

std::unordered_map<std::string, int64_t>
LoadBalancedSchedulingPolicy::RedistributePartitions(
    const std::vector<PartitionInfoPtr>& partitions) {
  std::unordered_map<std::string, int64_t> redistribution;

  // Get connected devices from tracker
  auto& tracker = DEVICE_TRACKER;
  std::vector<int64_t> connected_devices = tracker.GetConnectedDevices();

  if (connected_devices.empty()) {
    LOG_ERROR << "[LoadBalancedScheduling] No available devices for "
                 "redistribution";
    return redistribution;
  }

  // Track partition counts for load balancing during redistribution
  std::unordered_map<int64_t, size_t> current_loads;
  for (int64_t device_id : connected_devices) {
    current_loads[device_id] =
        PARTITION_TRACKER.GetDevicePartitionCount(device_id);
  }

  // Assign each partition to the device with minimum load
  for (const auto& part : partitions) {
    int64_t best_device = connected_devices[0];
    size_t min_load = current_loads[best_device];

    for (size_t i = 1; i < connected_devices.size(); ++i) {
      int64_t device_id = connected_devices[i];
      if (current_loads[device_id] < min_load) {
        min_load = current_loads[device_id];
        best_device = device_id;
      }
    }

    redistribution[part->key] = best_device;
    part->owner_device_id = best_device;
    current_loads[best_device]++;  // Update load for next iteration
  }

  LOG_INFO << "[LoadBalancedScheduling] Redistributed " << partitions.size()
           << " partitions across " << connected_devices.size()
           << " devices using load balancing";

  return redistribution;
}

}  // namespace backend
}  // namespace morphling
