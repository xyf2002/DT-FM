#include "virtual_clock.h"

#include <chrono>
#include <mutex>

namespace base {

// Helper function to get current system time in microseconds
static uint64_t GetSystemTimeMicros() {
  return std::chrono::duration_cast<std::chrono::microseconds>(
             std::chrono::system_clock::now().time_since_epoch())
      .count();
}

// Static getter for singleton instance
VirtualClock& VirtualClock::instance() {
  static VirtualClock instance_;
  return instance_;
}

// Constructor
VirtualClock::VirtualClock()
    : virtual_time_us_(0),
      start_virtual_time_us_(0),
      start_system_time_us_(0),
      time_scale_factor_(1.0),
      is_paused_(false),
      pause_virtual_time_us_(0) {}

// Initialize with system time as base
void VirtualClock::Initialize() {
  std::lock_guard<std::mutex> lock(mutex_);
  start_system_time_us_ = GetSystemTimeMicros();
  start_virtual_time_us_ = 0;
  virtual_time_us_ = 0;
  time_scale_factor_ = 1.0;
  is_paused_ = false;
}

// Get current virtual time
uint64_t VirtualClock::GetCurrentTime() const {
  std::lock_guard<std::mutex> lock(mutex_);

  if (is_paused_) {
    return pause_virtual_time_us_;
  }

  // Calculate elapsed system time
  uint64_t current_system_time_us = GetSystemTimeMicros();
  uint64_t elapsed_system_time_us =
      current_system_time_us - start_system_time_us_;

  // Apply time scale factor
  uint64_t elapsed_virtual_time_us =
      static_cast<uint64_t>(elapsed_system_time_us * time_scale_factor_);

  // Return virtual time from start + elapsed
  return start_virtual_time_us_ + elapsed_virtual_time_us;
}

// Manually advance virtual time
void VirtualClock::Advance(uint64_t delta_us) {
  std::lock_guard<std::mutex> lock(mutex_);

  if (is_paused_) {
    pause_virtual_time_us_ += delta_us;
  } else {
    // In non-paused mode, we adjust the start times to simulate advancement
    // This keeps the clock "ahead" while maintaining system time relationship
    virtual_time_us_ += delta_us;
    // Adjust start time to maintain the relationship: vt = svt + (st - sst) *
    // scale We want: vt_new = vt_old + delta So: start_virtual_time = vt_new -
    // (st_now - sst) * scale
    uint64_t current_system_time_us = GetSystemTimeMicros();
    uint64_t elapsed_system_time_us =
        current_system_time_us - start_system_time_us_;
    uint64_t elapsed_virtual_time_us =
        static_cast<uint64_t>(elapsed_system_time_us * time_scale_factor_);
    start_virtual_time_us_ =
        virtual_time_us_ + delta_us - elapsed_virtual_time_us;
  }
}

// Set time scale factor
void VirtualClock::SetTimeScale(double scale) {
  std::lock_guard<std::mutex> lock(mutex_);

  if (scale < 0.0) {
    return;  // Ignore invalid scales
  }

  // To change scale without jumping time:
  // Get current virtual time first
  uint64_t current_system_time_us = GetSystemTimeMicros();
  uint64_t elapsed_system_time_us =
      current_system_time_us - start_system_time_us_;
  uint64_t current_virtual_time_us =
      start_virtual_time_us_ +
      static_cast<uint64_t>(elapsed_system_time_us * time_scale_factor_);

  // Update scale and recalibrate start times
  time_scale_factor_ = scale;
  start_system_time_us_ = current_system_time_us;
  start_virtual_time_us_ = current_virtual_time_us;
}

// Get current time scale
double VirtualClock::GetTimeScale() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return time_scale_factor_;
}

// Pause virtual time
void VirtualClock::Pause() {
  std::lock_guard<std::mutex> lock(mutex_);

  if (is_paused_) {
    return;  // Already paused
  }

  // Capture current virtual time
  uint64_t current_system_time_us = GetSystemTimeMicros();
  uint64_t elapsed_system_time_us =
      current_system_time_us - start_system_time_us_;
  pause_virtual_time_us_ =
      start_virtual_time_us_ +
      static_cast<uint64_t>(elapsed_system_time_us * time_scale_factor_);
  is_paused_ = true;
}

// Resume virtual time
void VirtualClock::Resume() {
  std::lock_guard<std::mutex> lock(mutex_);

  if (!is_paused_) {
    return;  // Not paused
  }

  // Resume from paused state
  is_paused_ = false;
  start_system_time_us_ = GetSystemTimeMicros();
  start_virtual_time_us_ = pause_virtual_time_us_;
}

// Check if paused
bool VirtualClock::IsPaused() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return is_paused_;
}

// Reset virtual clock
void VirtualClock::Reset(uint64_t initial_time_us) {
  std::lock_guard<std::mutex> lock(mutex_);
  start_system_time_us_ = GetSystemTimeMicros();
  start_virtual_time_us_ = initial_time_us;
  virtual_time_us_ = initial_time_us;
  time_scale_factor_ = 1.0;
  is_paused_ = false;
  pause_virtual_time_us_ = 0;
}

// Get elapsed time since initialization
uint64_t VirtualClock::GetElapsedTime() const {
  std::lock_guard<std::mutex> lock(mutex_);

  uint64_t current_time = GetCurrentTime();
  return (current_time >= start_virtual_time_us_)
             ? (current_time - start_virtual_time_us_)
             : 0;
}

}  // namespace base
