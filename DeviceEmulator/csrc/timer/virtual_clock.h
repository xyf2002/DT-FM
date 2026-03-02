#ifndef MUDUO_TIMER_VIRTUAL_CLOCK_H
#define MUDUO_TIMER_VIRTUAL_CLOCK_H

#include <atomic>
#include <chrono>
#include <cstdint>
#include <mutex>

namespace base {

///
/// Virtual Clock - Global timing system for distributed emulation
///
/// Provides virtual time that can be independently controlled from system time.
/// Supports:
/// - Getting current virtual time
/// - Manual time advancement (for testing)
/// - Time scaling (acceleration/deceleration)
/// - Pause/resume capability
///
/// Thread-safe singleton implementation.
///
class VirtualClock {
 public:
  /// Get the singleton instance
  static VirtualClock& instance();

  /// Get current virtual time in microseconds
  uint64_t GetCurrentTime() const;

  /// Advance virtual time by delta_us microseconds
  /// Used in step-by-step simulation mode
  void Advance(uint64_t delta_us);

  /// Set the time scale factor (1.0 = real-time, 2.0 = 2x speed, 0.5 = half
  /// speed)
  void SetTimeScale(double scale);

  /// Get current time scale factor
  double GetTimeScale() const;

  /// Pause virtual time progression
  void Pause();

  /// Resume virtual time progression
  void Resume();

  /// Check if virtual clock is paused
  bool IsPaused() const;

  /// Reset virtual clock to zero with optional initial time
  void Reset(uint64_t initial_time_us = 0);

  /// Initialize virtual clock with system time as base
  /// Call this at application startup
  void Initialize();

  /// Get the virtual time relative to start (useful for normalized logs)
  uint64_t GetElapsedTime() const;

 private:
  VirtualClock();
  ~VirtualClock() = default;

  // Prevent copy and move
  VirtualClock(const VirtualClock&) = delete;
  VirtualClock& operator=(const VirtualClock&) = delete;

  // Synchronization
  mutable std::mutex mutex_;

  // Virtual time state
  uint64_t virtual_time_us_;        // Current virtual time in microseconds
  uint64_t start_virtual_time_us_;  // Virtual time at initialization
  uint64_t start_system_time_us_;   // System time at initialization
  double time_scale_factor_;        // 1.0 = real-time
  bool is_paused_;                  // Pause flag
  int64_t pause_virtual_time_us_;   // Virtual time when paused
};

}  // namespace base

#endif  // MUDUO_TIMER_VIRTUAL_CLOCK_H
