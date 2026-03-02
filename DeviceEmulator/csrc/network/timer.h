#ifndef UEVENT_TIMER_H_
#define UEVENT_TIMER_H_

#include "atomic"
#include "base/timestamp.h"
#include "callbacks.h"

namespace uevent {

// Internal class for timer event.
class Timer : base::noncopyable {
 public:
  Timer(TimerCb cb, base::Timestamp when, double interval)
      : callback_(std::move(cb)),
        expiration_(when),
        interval_(interval),
        repeat_(interval > 0.0),
        sequence_(created_num_++) {}

  void Run() const { callback_(); }

  base::Timestamp expiration() const { return expiration_; }
  bool repeat() const { return repeat_; }
  int64_t sequence() const { return sequence_; }

  void Restart(base::Timestamp now);

  static int64_t GetCreatedNum() { return created_num_.load(); }

 private:
  const TimerCb callback_;
  base::Timestamp expiration_;
  const double interval_;
  const bool repeat_;
  const int64_t sequence_;
  // 不同线程中的timer共用, 所以使用原子变量
  static std::atomic<int64_t> created_num_;
};

}  // namespace uevent

#endif
