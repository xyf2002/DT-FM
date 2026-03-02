#pragma once

#include <atomic>
#include <chrono>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "base/log_file.h"
#include "morphling.pb.h"
#include "network/uevent.h"
#include "partition_tracker.h"

// Convenient macro for accessing the DevicePartitionTracker singleton
#define DEVICE_TRACKER morphling::backend::DevicePartitionTracker::GetInstance()

namespace morphling {
namespace backend {
struct DeviceLiveness {
  int64_t device_id;
  std::string conn_addr;
  bool is_connected;
  std::chrono::steady_clock::time_point last_seen;
  std::chrono::steady_clock::time_point connected_at;
  std::chrono::steady_clock::time_point
      stats_start_time;  // Time when stats recording started
  uint64_t total_partitions_processed;
  uint64_t total_bytes_sent;
  uint64_t total_bytes_received;
  DeviceProfileData profile;

  // Per-packet throughput tracking
  std::chrono::steady_clock::time_point
      last_packet_time;           // Time of last packet
  uint64_t last_packet_size;      // Size of last packet
  double last_packet_throughput;  // Throughput of last packet (B/s)

  // Package count
  uint64_t total_packets_sent;
  uint64_t total_packets_received;

  // Epoch timestamps (microseconds)
  uint64_t
      last_packet_start_epoch_us;  // When last packet started (us since epoch)
  uint64_t last_packet_end_epoch_us;  // When last packet ended (us since epoch)

  DeviceLiveness()
      : device_id(-1),
        is_connected(false),
        stats_start_time(std::chrono::steady_clock::now()),
        total_partitions_processed(0),
        total_bytes_sent(0),
        total_bytes_received(0),
        last_packet_time(std::chrono::steady_clock::now()),
        last_packet_size(0),
        last_packet_throughput(0.0),
        total_packets_sent(0),
        total_packets_received(0),
        last_packet_start_epoch_us(0),
        last_packet_end_epoch_us(0) {}

  std::string DebugString() const;
};
typedef std::shared_ptr<DeviceLiveness> DeviceLivenessPtr;

// Device and partition tracker - manages device lifecycle and partition
// assignments Thread-safe singleton component for tracking:
// 1. Device liveness (connections, disconnections, health)
// 2. Partition assignments per device
// 3. Device metrics and statistics
class DevicePartitionTracker {
 public:
  static DevicePartitionTracker& GetInstance();

  // Disable copy and move
  DevicePartitionTracker(const DevicePartitionTracker&) = delete;
  DevicePartitionTracker& operator=(const DevicePartitionTracker&) = delete;
  DevicePartitionTracker(DevicePartitionTracker&&) = delete;
  DevicePartitionTracker& operator=(DevicePartitionTracker&&) = delete;

  // Device lifecycle management
  int64_t RegisterDevice(const std::string& conn_addr,
                         const DeviceProfileData& profile);
  void UnregisterDevice(int64_t device_id);
  //   void MarkDeviceConnected(int64_t device_id, const std::string&
  //   conn_addr); void MarkDeviceDisconnected(int64_t device_id);
  void UpdateDeviceLastSeen(int64_t device_id);

  // Device queries
  bool IsDeviceConnected(int64_t device_id) const;
  bool HasDevice(int64_t device_id) const;
  int64_t GetDeviceIdByAddr(const std::string& conn_addr) const;
  std::string GetDeviceAddr(int64_t device_id) const;
  DeviceLiveness GetDeviceLiveness(int64_t device_id) const;
  std::vector<int64_t> GetConnectedDevices() const;
  std::vector<int64_t> GetAllDevices() const;
  size_t GetConnectedDeviceCount() const;
  size_t GetTotalDeviceCount() const;

  // Statistics
  void RecordPartitionProcessed(int64_t device_id);
  void RecordBytesSent(int64_t device_id, uint64_t bytes);
  void RecordBytesReceived(int64_t device_id, uint64_t bytes);

  // Throughput calculation (bytes per second)
  double GetUploadThroughput(int64_t device_id) const;
  double GetDownloadThroughput(int64_t device_id) const;
  double GetLastPacketThroughput(
      int64_t device_id) const;  // Throughput of last packet
  double GetAveragePacketThroughput(
      int64_t device_id) const;  // Average throughput across all packets

  // Get packet epoch timestamps for a device
  void GetLastPacketEpochTimestamps(int64_t device_id, uint64_t& start_us,
                                    uint64_t& end_us) const;

  // Server-level aggregated statistics
  uint64_t GetServerTotalBytesSent() const;
  uint64_t GetServerTotalBytesReceived() const;
  double GetServerAggregatedThroughput()
      const;  // All bytes / total time for all connected devices

  // Performance logging to file
  void LogThroughputToFile(int64_t device_id, int64_t gemm_id,
                           const std::string& direction, uint64_t bytes,
                           double throughput, uint64_t epoch_start_us,
                           uint64_t epoch_end_us) const;

  // Virtual time logging to file (separate log entry with virtual time info)
  void LogVirtualTimeEvent(int64_t device_id, int64_t gemm_id,
                           const std::string& phase, const std::string& event,
                           uint64_t vt_start_us, uint64_t vt_end_us) const;

  void InitPerfLog(const std::string& log_path = "./perf.log");

  // Initialize separate performance logs for Server and each Device
  // This avoids multi-process race conditions by giving each entity its own log
  // file log_dir: directory to store logs, e.g., "./logs" entity_type: "server"
  // or "device" entity_id: device_id for devices, ignored for server
  void InitSeparatePerfLog(const std::string& log_dir,
                           const std::string& entity_type,
                           int64_t entity_id = -1);
  std::string GetPerfLogPath() const;

  // Connection management
  void SetDeviceConnection(int64_t device_id,
                           const uevent::ConnectionUeventPtr& conn);
  uevent::ConnectionUeventPtr GetDeviceConnection(int64_t device_id) const;
  void RemoveDeviceConnection(int64_t device_id);

  // Debug and monitoring
  std::string DebugString() const;
  void DumpState() const;

  // Clear all state (for testing)
  void Reset();

 private:
  DevicePartitionTracker();
  ~DevicePartitionTracker() = default;

  // Internal helpers
  int64_t AllocateDeviceId();

  // State
  mutable std::mutex mutex_;
  mutable std::mutex perf_log_mutex_;

  // Device liveness and ID management
  std::unordered_set<DeviceLivenessPtr> devices_set_;
  std::unordered_map<int64_t, DeviceLivenessPtr> devices_map_;

  int64_t next_device_id_;
  std::unordered_map<std::string, int64_t> addr_to_device_id_;
  std::unordered_map<int64_t, std::string> device_id_to_addr_;

  // Connection management
  std::unordered_map<int64_t, uevent::ConnectionUeventPtr> device_conn_;

  // Performance log file using LogFile class
  mutable std::unique_ptr<base::LogFile> perf_log_file_;
};

}  // namespace backend
}  // namespace morphling
