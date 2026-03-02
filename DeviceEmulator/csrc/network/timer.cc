#include "timer.h"

namespace uevent {

std::atomic<int64_t> Timer::created_num_;

void Timer::Restart(base::Timestamp now) {
  if (repeat_) {
    expiration_ = base::addTime(now, interval_);
  } else {
    expiration_ = base::Timestamp::invalid();
  }
}

}  // namespace uevent
