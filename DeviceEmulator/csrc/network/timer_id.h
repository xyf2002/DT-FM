#ifndef UEVENT_TIMERID_H_
#define UEVENT_TIMERID_H_

#include "base/copyable.h"

namespace uevent {

class Timer;

// An opaque identifier, for canceling Timer.
class TimerId : public base::copyable {
 public:
  TimerId() : timer_(NULL), sequence_(0) {}
  int64_t GetId() const { return sequence_; }
  TimerId(Timer* timer, int64_t seq) : timer_(timer), sequence_(seq) {}
  // default copy-ctor, dtor and assignment are okay
  friend class TimerQueueUevent;

 private:
  Timer* timer_;
  int64_t sequence_;
};

}  // namespace uevent

#endif  // UEVENT_TIMERID_H_
