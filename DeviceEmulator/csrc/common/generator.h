#pragma once

#include <uuid/uuid.h>

#include <bitset>
#include <chrono>
#include <iomanip>
#include <mutex>
#include <random>
#include <sstream>
#include <string>

class NumGenerator {
 public:
  static uint32_t ctx_id();
  static uint32_t flowno();

 private:
  static std::mutex mutex_;
  static uint32_t ctx_id_;
};

inline std::string GenUUID() {
  uuid_t uuid;
  uuid_generate(uuid);
  char uuid_str[37];
  uuid_unparse(uuid, uuid_str);
  return std::string(uuid_str);
}

inline uint64_t GenUUID64() {
  static std::random_device rd;
  static std::mt19937_64 eng(rd());
  static std::uniform_int_distribution<uint64_t> distr;

  std::bitset<64> uuid;
  uuid = std::chrono::high_resolution_clock::now().time_since_epoch().count();
  uuid ^= distr(eng);

  return uuid.to_ullong();
}

inline std::string CurrentTimeString() {
  // Get current time as time_point
  auto now = std::chrono::system_clock::now();

  // Convert time_point to system time for breaking down into components
  auto now_c = std::chrono::system_clock::to_time_t(now);
  auto now_tm = *std::localtime(&now_c);

  // Get the current time as milliseconds
  auto now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                    now.time_since_epoch()) %
                1000;

  // Use stringstream to format the time
  std::ostringstream oss;
  oss << std::put_time(&now_tm, "%Y-%m-%d %H:%M:%S");
  oss << '.' << std::setfill('0') << std::setw(3) << now_ms.count();

  return oss.str();
}

// constexpr microseconds since epoch
inline uint64_t CurrentTimeMicros() {
  return std::chrono::duration_cast<std::chrono::microseconds>(
             std::chrono::system_clock::now().time_since_epoch())
      .count();
}

// Virtual clock functions (convenience wrappers)
#include "timer/virtual_clock.h"

inline uint64_t VirtualClockNow() {
  return base::VirtualClock::instance().GetCurrentTime();
}

inline uint64_t VirtualClockElapsed() {
  return base::VirtualClock::instance().GetElapsedTime();
}
