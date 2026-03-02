#include "device_tracker.h"

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <sstream>

#include "utils/logger.h"

namespace morphling {
namespace backend {

std::string DeviceLiveness::DebugString() const {
  std::ostringstream oss;

  // Calculate elapsed time since stats started
  auto now = std::chrono::steady_clock::now();
  auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
      now - stats_start_time);
  double elapsed_seconds = elapsed.count() / 1000.0;

  // Calculate throughputs
  double upload_throughput = 0.0;
  double download_throughput = 0.0;
  double total_throughput = 0.0;

  if (elapsed_seconds > 0) {
    upload_throughput = total_bytes_sent / elapsed_seconds;
    download_throughput = total_bytes_received / elapsed_seconds;
    total_throughput = upload_throughput + download_throughput;
  }

  oss << "DeviceLiveness{device_id=" << device_id << ", addr=" << conn_addr
      << ", connected=" << is_connected << ", bytes_sent=" << total_bytes_sent
      << ", bytes_received=" << total_bytes_received
      << ", upload_throughput=" << upload_throughput << " B/s"
      << ", download_throughput=" << download_throughput << " B/s"
      << ", total_throughput=" << total_throughput << " B/s"
      << ", elapsed=" << elapsed_seconds << "s}";
  return oss.str();
}

DevicePartitionTracker& DevicePartitionTracker::GetInstance() {
  static DevicePartitionTracker instance;
  return instance;
}

DevicePartitionTracker::DevicePartitionTracker() : next_device_id_(0) {}

int64_t DevicePartitionTracker::AllocateDeviceId() { return next_device_id_++; }

int64_t DevicePartitionTracker::RegisterDevice(
    const std::string& conn_addr, const DeviceProfileData& profile) {
  std::lock_guard<std::mutex> lock(mutex_);

  // Check if device already exists (reconnection case)
  auto it = addr_to_device_id_.find(conn_addr);
  if (it != addr_to_device_id_.end()) {
    int64_t device_id = it->second;
    LOG_INFO << "[DeviceTracker] Reconnection from " << conn_addr
             << ", reusing device_id=" << device_id;

    // Update device state
    auto dev_it = devices_map_.find(device_id);
    if (dev_it != devices_map_.end()) {
      auto& device = dev_it->second;
      device->is_connected = true;
      device->last_seen = std::chrono::steady_clock::now();
      device->connected_at = std::chrono::steady_clock::now();
      device->conn_addr = conn_addr;
      device->profile = profile;
    }

    return device_id;
  }

  // New device - allocate ID
  int64_t device_id = AllocateDeviceId();
  LOG_INFO << "[DeviceTracker] New device registered: " << conn_addr
           << ", assigned device_id=" << device_id;

  // Create device liveness record
  auto liveness = std::make_shared<DeviceLiveness>();
  liveness->device_id = device_id;
  liveness->conn_addr = conn_addr;
  liveness->is_connected = true;
  liveness->last_seen = std::chrono::steady_clock::now();
  liveness->connected_at = std::chrono::steady_clock::now();
  liveness->stats_start_time = std::chrono::steady_clock::now();
  liveness->profile = profile;

  // Update mappings
  addr_to_device_id_[conn_addr] = device_id;
  device_id_to_addr_[device_id] = conn_addr;
  devices_map_[device_id] = liveness;
  devices_set_.insert(liveness);

  return device_id;
}

void DevicePartitionTracker::UnregisterDevice(int64_t device_id) {
  std::lock_guard<std::mutex> lock(mutex_);

  auto it = devices_map_.find(device_id);
  if (it == devices_map_.end()) {
    LOG_WARN << "[DeviceTracker] Cannot unregister unknown device "
             << device_id;
    return;
  }

  std::string conn_addr = it->second->conn_addr;
  LOG_INFO << "[DeviceTracker] Unregistering device " << device_id << " ("
           << conn_addr << ")";

  // Remove from all mappings
  devices_set_.erase(it->second);
  addr_to_device_id_.erase(conn_addr);
  device_id_to_addr_.erase(device_id);
  devices_map_.erase(device_id);
}

// void DevicePartitionTracker::MarkDeviceConnected(int64_t device_id,
//                                                  const std::string&
//                                                  conn_addr) {
//   std::lock_guard<std::mutex> lock(mutex_);

//   auto it = devices_map_.find(device_id);
//   if (it != devices_map_.end()) {
//     it->second->is_connected = true;
//     it->second->conn_addr = conn_addr;
//     it->second->connected_at = std::chrono::steady_clock::now();
//     it->second->last_seen = std::chrono::steady_clock::now();
//     LOG_INFO << "[DeviceTracker] Device " << device_id << " marked
//     connected";
//   }
// }

// void DevicePartitionTracker::MarkDeviceDisconnected(int64_t device_id) {
//   std::lock_guard<std::mutex> lock(mutex_);

//   auto it = devices_map_.find(device_id);
//   if (it != devices_map_.end()) {
//     it->second->is_connected = false;
//     LOG_INFO << "[DeviceTracker] Device " << device_id
//              << " marked disconnected";
//   }
// }

void DevicePartitionTracker::UpdateDeviceLastSeen(int64_t device_id) {
  std::lock_guard<std::mutex> lock(mutex_);

  auto it = devices_map_.find(device_id);
  if (it != devices_map_.end()) {
    it->second->last_seen = std::chrono::steady_clock::now();
  }
}

bool DevicePartitionTracker::IsDeviceConnected(int64_t device_id) const {
  std::lock_guard<std::mutex> lock(mutex_);

  auto it = devices_map_.find(device_id);
  return it != devices_map_.end() && it->second->is_connected;
}

bool DevicePartitionTracker::HasDevice(int64_t device_id) const {
  std::lock_guard<std::mutex> lock(mutex_);
  return devices_map_.find(device_id) != devices_map_.end();
}

int64_t DevicePartitionTracker::GetDeviceIdByAddr(
    const std::string& conn_addr) const {
  std::lock_guard<std::mutex> lock(mutex_);

  auto it = addr_to_device_id_.find(conn_addr);
  return (it != addr_to_device_id_.end()) ? it->second : -1;
}

std::string DevicePartitionTracker::GetDeviceAddr(int64_t device_id) const {
  std::lock_guard<std::mutex> lock(mutex_);

  auto it = device_id_to_addr_.find(device_id);
  return (it != device_id_to_addr_.end()) ? it->second : "";
}

DeviceLiveness DevicePartitionTracker::GetDeviceLiveness(
    int64_t device_id) const {
  std::lock_guard<std::mutex> lock(mutex_);

  auto it = devices_map_.find(device_id);
  if (it != devices_map_.end()) {
    return *it->second;
  }
  return DeviceLiveness();
}

std::vector<int64_t> DevicePartitionTracker::GetConnectedDevices() const {
  std::lock_guard<std::mutex> lock(mutex_);

  std::vector<int64_t> connected;
  for (const auto& [device_id, liveness] : devices_map_) {
    if (liveness->is_connected) {
      connected.push_back(device_id);
    }
  }
  return connected;
}

std::vector<int64_t> DevicePartitionTracker::GetAllDevices() const {
  std::lock_guard<std::mutex> lock(mutex_);

  std::vector<int64_t> all_devices;
  for (const auto& [device_id, _] : devices_map_) {
    all_devices.push_back(device_id);
  }
  return all_devices;
}

size_t DevicePartitionTracker::GetConnectedDeviceCount() const {
  std::lock_guard<std::mutex> lock(mutex_);

  size_t count = 0;
  for (const auto& [_, liveness] : devices_map_) {
    if (liveness->is_connected) {
      count++;
    }
  }
  return count;
}

size_t DevicePartitionTracker::GetTotalDeviceCount() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return devices_map_.size();
}

double DevicePartitionTracker::GetUploadThroughput(int64_t device_id) const {
  std::lock_guard<std::mutex> lock(mutex_);

  auto it = devices_map_.find(device_id);
  if (it == devices_map_.end()) {
    return 0.0;
  }

  auto now = std::chrono::steady_clock::now();
  auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
      now - it->second->stats_start_time);

  if (elapsed.count() == 0) {
    return 0.0;
  }

  // Calculate throughput in bytes per second
  double elapsed_seconds = elapsed.count() / 1000.0;
  double throughput = it->second->total_bytes_sent / elapsed_seconds;
  return throughput;
}

double DevicePartitionTracker::GetDownloadThroughput(int64_t device_id) const {
  std::lock_guard<std::mutex> lock(mutex_);

  auto it = devices_map_.find(device_id);
  if (it == devices_map_.end()) {
    return 0.0;
  }

  auto now = std::chrono::steady_clock::now();
  auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
      now - it->second->stats_start_time);

  if (elapsed.count() == 0) {
    return 0.0;
  }

  // Calculate throughput in bytes per second
  double elapsed_seconds = elapsed.count() / 1000.0;
  double throughput = it->second->total_bytes_received / elapsed_seconds;
  return throughput;
}

double DevicePartitionTracker::GetLastPacketThroughput(
    int64_t device_id) const {
  std::lock_guard<std::mutex> lock(mutex_);

  auto it = devices_map_.find(device_id);
  if (it == devices_map_.end()) {
    return 0.0;
  }

  return it->second->last_packet_throughput;
}

double DevicePartitionTracker::GetAveragePacketThroughput(
    int64_t device_id) const {
  std::lock_guard<std::mutex> lock(mutex_);

  auto it = devices_map_.find(device_id);
  if (it == devices_map_.end()) {
    return 0.0;
  }

  auto now = std::chrono::steady_clock::now();
  auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
      now - it->second->stats_start_time);

  if (elapsed.count() == 0) {
    return 0.0;
  }

  uint64_t total_packets =
      it->second->total_packets_sent + it->second->total_packets_received;
  if (total_packets == 0) {
    return 0.0;
  }

  // Calculate average throughput: total_bytes / elapsed_time
  // This is the same as overall throughput but normalized by packet count for
  // understanding
  double elapsed_seconds = elapsed.count() / 1000.0;
  uint64_t total_bytes =
      it->second->total_bytes_sent + it->second->total_bytes_received;
  double throughput = total_bytes / elapsed_seconds;
  return throughput;
}

void DevicePartitionTracker::GetLastPacketEpochTimestamps(
    int64_t device_id, uint64_t& start_us, uint64_t& end_us) const {
  std::lock_guard<std::mutex> lock(mutex_);

  start_us = 0;
  end_us = 0;

  auto it = devices_map_.find(device_id);
  if (it != devices_map_.end()) {
    start_us = it->second->last_packet_start_epoch_us;
    end_us = it->second->last_packet_end_epoch_us;
  }
}

uint64_t DevicePartitionTracker::GetServerTotalBytesSent() const {
  std::lock_guard<std::mutex> lock(mutex_);

  uint64_t total = 0;
  for (const auto& [_, liveness] : devices_map_) {
    total += liveness->total_bytes_sent;
  }
  return total;
}

uint64_t DevicePartitionTracker::GetServerTotalBytesReceived() const {
  std::lock_guard<std::mutex> lock(mutex_);

  uint64_t total = 0;
  for (const auto& [_, liveness] : devices_map_) {
    total += liveness->total_bytes_received;
  }
  return total;
}

double DevicePartitionTracker::GetServerAggregatedThroughput() const {
  std::lock_guard<std::mutex> lock(mutex_);

  if (devices_map_.empty()) {
    return 0.0;
  }

  // Sum up the last packet throughput from all devices
  double total_server_tp = 0.0;
  for (const auto& [_, liveness] : devices_map_) {
    total_server_tp += liveness->last_packet_throughput;
  }

  return total_server_tp;
}

void DevicePartitionTracker::RecordPartitionProcessed(int64_t device_id) {
  std::lock_guard<std::mutex> lock(mutex_);

  auto it = devices_map_.find(device_id);
  if (it != devices_map_.end()) {
    it->second->total_partitions_processed++;
  }
}

void DevicePartitionTracker::RecordBytesSent(int64_t device_id,
                                             uint64_t bytes) {
  std::lock_guard<std::mutex> lock(mutex_);

  auto it = devices_map_.find(device_id);
  if (it != devices_map_.end()) {
    auto now = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        now - it->second->last_packet_time);

    // Calculate throughput for this packet: bytes / elapsed_time
    double elapsed_seconds =
        (elapsed.count() == 0) ? 0.001 : (elapsed.count() / 1000.0);
    it->second->last_packet_throughput = bytes / elapsed_seconds;

    // Record epoch timestamps (microseconds since epoch)
    auto now_epoch = std::chrono::system_clock::now();
    uint64_t current_epoch_us =
        std::chrono::duration_cast<std::chrono::microseconds>(
            now_epoch.time_since_epoch())
            .count();

    it->second->last_packet_start_epoch_us =
        it->second->last_packet_end_epoch_us;
    it->second->last_packet_end_epoch_us = current_epoch_us;

    it->second->total_bytes_sent += bytes;
    it->second->total_packets_sent++;
    it->second->last_packet_time = now;
    it->second->last_packet_size = bytes;
  }
}

void DevicePartitionTracker::RecordBytesReceived(int64_t device_id,
                                                 uint64_t bytes) {
  std::lock_guard<std::mutex> lock(mutex_);

  auto it = devices_map_.find(device_id);
  if (it != devices_map_.end()) {
    auto now = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        now - it->second->last_packet_time);

    // Calculate throughput for this packet: bytes / elapsed_time
    double elapsed_seconds =
        (elapsed.count() == 0) ? 0.001 : (elapsed.count() / 1000.0);
    it->second->last_packet_throughput = bytes / elapsed_seconds;

    // Record epoch timestamps (microseconds since epoch)
    auto now_epoch = std::chrono::system_clock::now();
    uint64_t current_epoch_us =
        std::chrono::duration_cast<std::chrono::microseconds>(
            now_epoch.time_since_epoch())
            .count();

    it->second->last_packet_start_epoch_us =
        it->second->last_packet_end_epoch_us;
    it->second->last_packet_end_epoch_us = current_epoch_us;

    it->second->total_bytes_received += bytes;
    it->second->total_packets_received++;
    it->second->last_packet_time = now;
    it->second->last_packet_size = bytes;
  }
}

void DevicePartitionTracker::SetDeviceConnection(
    int64_t device_id, const uevent::ConnectionUeventPtr& conn) {
  std::lock_guard<std::mutex> lock(mutex_);

  if (devices_map_.find(device_id) == devices_map_.end()) {
    LOG_WARN << "[DeviceTracker] Cannot set connection for unknown device "
             << device_id;
    return;
  }

  device_conn_[device_id] = conn;
  LOG_DEBUG << "[DeviceTracker] Set connection for device_id " << device_id;
}

uevent::ConnectionUeventPtr DevicePartitionTracker::GetDeviceConnection(
    int64_t device_id) const {
  std::lock_guard<std::mutex> lock(mutex_);

  auto it = device_conn_.find(device_id);
  if (it != device_conn_.end()) {
    return it->second;
  }
  return nullptr;
}

void DevicePartitionTracker::RemoveDeviceConnection(int64_t device_id) {
  std::lock_guard<std::mutex> lock(mutex_);

  device_conn_.erase(device_id);
  LOG_DEBUG << "[DeviceTracker] Removed connection for device_id " << device_id;
}

std::string DevicePartitionTracker::DebugString() const {
  std::lock_guard<std::mutex> lock(mutex_);

  std::ostringstream oss;
  oss << "DevicePartitionTracker{\n";
  oss << "  Total devices: " << devices_map_.size() << "\n";
  oss << "  Connected devices: ";

  size_t connected = 0;
  for (const auto& [_, liveness] : devices_map_) {
    if (liveness->is_connected) connected++;
  }
  oss << connected << "\n";

  oss << "  Devices:\n";
  for (const auto& [device_id, liveness] : devices_map_) {
    oss << "    " << liveness->DebugString() << "\n";
  }

  oss << "}";
  return oss.str();
}

void DevicePartitionTracker::DumpState() const {
  LOG_INFO << "[DeviceTracker] " << DebugString();
}

void DevicePartitionTracker::InitPerfLog(const std::string& log_path) {
  std::lock_guard<std::mutex> lock(perf_log_mutex_);

  // Only initialize once - check if file is already set
  if (perf_log_file_) {
    LOG_DEBUG << "[DeviceTracker] Performance log already initialized";
    return;
  }

  // Create LogFile with rollSize of 512MB, thread-safe, flush every 3 seconds
  perf_log_file_ =
      std::make_unique<base::LogFile>(log_path, 512 * 1024 * 1024, true, 3);

  // Write header for the log file
  const char* header =
      "timestamp_us,device_id,direction,bytes,throughput_b_s,"
      "epoch_start_us,epoch_end_us,packet_duration_us\n";
  perf_log_file_->append(header, strlen(header));
  perf_log_file_->flush();

  LOG_INFO << "[DeviceTracker] Performance log initialized at: " << log_path;
}

void DevicePartitionTracker::InitSeparatePerfLog(const std::string& log_dir,
                                                 const std::string& entity_type,
                                                 int64_t entity_id) {
  std::lock_guard<std::mutex> lock(perf_log_mutex_);

  // Only initialize once
  if (perf_log_file_) {
    LOG_DEBUG << "[DeviceTracker] Performance log already initialized";
    return;
  }

  // Generate file path based on entity type
  std::string log_path;
  if (entity_type == "server") {
    log_path = log_dir + "/perf_server.log";
  } else if (entity_type == "device") {
    log_path = log_dir + "/perf_device_" + std::to_string(entity_id) + ".log";
  } else {
    LOG_ERROR << "[DeviceTracker] Unknown entity_type: " << entity_type;
    return;
  }

  // Create log directory if it doesn't exist
  int ret = system(("mkdir -p " + log_dir).c_str());
  if (ret != 0) {
    LOG_WARN << "[DeviceTracker] Failed to create log directory: " << log_dir;
  }

  // Create LogFile with rollSize of 512MB, thread-safe, flush every 3 seconds
  perf_log_file_ =
      std::make_unique<base::LogFile>(log_path, 512 * 1024 * 1024, true, 3);

  // Write headers for the log file
  std::string vtime_header =
      "# VTIME format: "
      "VTIME,timestamp_us,device_id,gemm_id,phase,event,vt_start_us,vt_end_us,"
      "vt_duration_us\n";
  std::string throughput_header =
      "# Throughput format: "
      "timestamp_us,device_id,gemm_id,direction,bytes,throughput_b_s,epoch_"
      "start_us,epoch_end_us,packet_duration_us\n";

  perf_log_file_->append(vtime_header.c_str(), vtime_header.length());
  perf_log_file_->append(throughput_header.c_str(), throughput_header.length());

  // Write a comment line indicating this is a separate log
  std::string comment = "# Separate performance log for " + entity_type;
  if (entity_type == "device") {
    comment += " " + std::to_string(entity_id);
  }
  comment += "\n";
  perf_log_file_->append(comment.c_str(), comment.length());
  perf_log_file_->flush();

  LOG_INFO << "[DeviceTracker] Separate performance log initialized at: "
           << log_path;
}

std::string DevicePartitionTracker::GetPerfLogPath() const {
  // perf_log_path_ removed; returning empty string for compatibility
  return "";
}

void DevicePartitionTracker::LogThroughputToFile(
    int64_t device_id, int64_t gemm_id, const std::string& direction,
    uint64_t bytes, double throughput, uint64_t epoch_start_us,
    uint64_t epoch_end_us) const {
  if (!perf_log_file_) {
    return;  // Log file not initialized
  }

  // Get current timestamp
  auto now = std::chrono::system_clock::now();
  uint64_t now_us = std::chrono::duration_cast<std::chrono::microseconds>(
                        now.time_since_epoch())
                        .count();

  // Calculate packet duration
  uint64_t packet_duration_us =
      (epoch_end_us > epoch_start_us) ? (epoch_end_us - epoch_start_us) : 0;

  // Format and append to log file
  // Format:
  // timestamp_us,device_id,gemm_id,direction,bytes,throughput,epoch_start_us,epoch_end_us,packet_duration_us
  char buf[256];
  int len =
      snprintf(buf, sizeof(buf), "%lu,%ld,%ld,%s,%lu,%.2f,%lu,%lu,%lu\n",
               now_us, device_id, gemm_id, direction.c_str(), bytes, throughput,
               epoch_start_us, epoch_end_us, packet_duration_us);
  if (len > 0 && len < (int)sizeof(buf)) {
    perf_log_file_->append(buf, len);
  }
}

void DevicePartitionTracker::LogVirtualTimeEvent(
    int64_t device_id, int64_t gemm_id, const std::string& phase,
    const std::string& event, uint64_t vt_start_us, uint64_t vt_end_us) const {
  if (!perf_log_file_) {
    LOG_DEBUG
        << "[LogVirtualTimeEvent] perf_log_file_ not initialized, skipping";
    return;
  }

  // Get current system timestamp
  auto now = std::chrono::system_clock::now();
  uint64_t now_us = std::chrono::duration_cast<std::chrono::microseconds>(
                        now.time_since_epoch())
                        .count();

  uint64_t vt_duration_us =
      (vt_end_us > vt_start_us) ? (vt_end_us - vt_start_us) : 0;

  // Format and append to log file
  // Format:
  // VTIME,timestamp_us,device_id,gemm_id,phase,event,vt_start_us,vt_end_us,vt_duration_us
  char buf[256];
  int len = snprintf(buf, sizeof(buf), "VTIME,%lu,%ld,%ld,%s,%s,%lu,%lu,%lu\n",
                     now_us, device_id, gemm_id, phase.c_str(), event.c_str(),
                     vt_start_us, vt_end_us, vt_duration_us);
  if (len > 0 && len < (int)sizeof(buf)) {
    perf_log_file_->append(buf, len);
    LOG_DEBUG << "[LogVirtualTimeEvent] Appended VTIME event for device "
              << device_id << " gemm_id " << gemm_id;
  } else {
    LOG_WARN << "[LogVirtualTimeEvent] Format error, len=" << len;
  }
}

void DevicePartitionTracker::Reset() {
  std::lock_guard<std::mutex> lock(mutex_);

  LOG_INFO << "[DeviceTracker] Resetting all state";
  next_device_id_ = 0;
  addr_to_device_id_.clear();
  device_id_to_addr_.clear();
  devices_map_.clear();
  devices_set_.clear();
  device_conn_.clear();
}

}  // namespace backend
}  // namespace morphling
