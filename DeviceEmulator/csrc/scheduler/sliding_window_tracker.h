#pragma once

#include <array>
#include <chrono>
#include <cstdint>

// Header-only sliding window duration tracker.
// Tracks recent task durations and computes a running average.
// NOT thread-safe — intended to be accessed from a single event loop thread.
template <size_t WindowSize = 64>
class SlidingWindowDurationTracker {
 public:
  using Clock = std::chrono::steady_clock;
  using TimePoint = Clock::time_point;

  explicit SlidingWindowDurationTracker(int64_t default_us = 0)
      : default_us_(default_us) {}

  void RecordDuration(int64_t duration_us) {
    samples_[write_idx_ % WindowSize] = duration_us;
    write_idx_++;
  }

  int64_t GetAverageDurationUs() const {
    size_t count = SampleCount();
    if (count == 0) return default_us_;
    int64_t sum = 0;
    size_t start =
        (write_idx_ > WindowSize) ? write_idx_ - WindowSize : 0;
    for (size_t i = start; i < write_idx_; i++)
      sum += samples_[i % WindowSize];
    return sum / static_cast<int64_t>(count);
  }

  size_t SampleCount() const {
    return (write_idx_ < WindowSize) ? write_idx_ : WindowSize;
  }

  static TimePoint Now() { return Clock::now(); }

  static int64_t ElapsedUs(TimePoint start) {
    return std::chrono::duration_cast<std::chrono::microseconds>(
               Clock::now() - start)
        .count();
  }

 private:
  std::array<int64_t, WindowSize> samples_{};
  size_t write_idx_ = 0;
  int64_t default_us_;
};
