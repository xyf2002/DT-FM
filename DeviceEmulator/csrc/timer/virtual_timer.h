#pragma once

#include <cstdint>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <vector>

#include "utils/noncopyable.h"

struct DeviceConfig {
  int device_id;
  int num_threads;
};

// The virtual timer for each device
struct DeviceVirtualTimer {
  uint64_t current_timestamp;
};

class VirtualTimeManager : public noncopyable {
 public:
  explicit VirtualTimeManager(int num_devices);

 private:
  std::unordered_map<int, DeviceVirtualTimer> device_timers_;
  std::unordered_map<int, std::vector<std::pair<uint64_t, uint64_t>>>
      device_time_intervals_;  // start, end
};

extern std::unique_ptr<VirtualTimeManager> virtual_time_manager;
extern std::once_flag kInitVirtualTimeManagerFlag;

static void InitVirtualTimeManager(int num_devices) {
  std::call_once(kInitVirtualTimeManagerFlag, [&]() {
    virtual_time_manager = std::make_unique<VirtualTimeManager>(num_devices);
  });
}
