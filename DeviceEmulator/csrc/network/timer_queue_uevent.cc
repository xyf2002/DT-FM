
#ifndef __STDC_LIMIT_MACROS
#define __STDC_LIMIT_MACROS
#endif

#include "timer_queue_uevent.h"

#include <sys/timerfd.h>
#include <unistd.h>

#include "base/logging.h"
#include "timer.h"
#include "timer_id.h"
#include "uevent.h"

using base::Timestamp;

namespace uevent {

namespace detail {

int CreateTimerfd() {
  int timerfd = ::timerfd_create(CLOCK_MONOTONIC, TFD_NONBLOCK | TFD_CLOEXEC);
  if (timerfd < 0) {
    LOG_SYSFATAL << "Failed in timerfd_create";
  }
  return timerfd;
}

struct timespec HowMuchTimeFromNow(Timestamp when) {
  int64_t microseconds =
      when.microSecondsSinceEpoch() - Timestamp::now().microSecondsSinceEpoch();
  if (microseconds < 100) {  // 定时精度为100 us
    microseconds = 100;
  }
  struct timespec ts;
  ts.tv_sec =
      static_cast<time_t>(microseconds / Timestamp::kMicroSecondsPerSecond);
  ts.tv_nsec = static_cast<long>(
      (microseconds % Timestamp::kMicroSecondsPerSecond) * 1000);
  return ts;
}

void ReadTimerfd(int timerfd, Timestamp now) {
  uint64_t howmany;
  ssize_t n = ::read(timerfd, &howmany, sizeof howmany);
  if (n != sizeof howmany) {
    LOG_ERROR << "read " << n << " bytes instead of 8";
  }
  // LOG_TRACE << "timerfd expired " << howmany
  //           <<" times since last read or timerfd_settime, at"
  //           << now.toString();
}

void ResetTimerfd(int timerfd, Timestamp expiration) {
  // wake up loop by timerfd_settime()
  struct itimerspec new_value;
  struct itimerspec old_value;
  bzero(&new_value, sizeof new_value);
  bzero(&old_value, sizeof old_value);
  // 这里使用的是相对定时，计算定时距离当前的时间
  new_value.it_value = HowMuchTimeFromNow(expiration);
  int ret = ::timerfd_settime(timerfd, 0, &new_value, &old_value);
  if (ret) {
    LOG_SYSERR << "timerfd_settime()";
  }
}

}  // namespace detail

}  // namespace uevent

using namespace uevent;
using namespace uevent::detail;

TimerQueueUevent::TimerQueueUevent(UeventLoop* loop)
    : timerfd_(CreateTimerfd()),
      loop_(loop),
      timers_(),
      calling_expired_timers_(false) {}

TimerQueueUevent::~TimerQueueUevent() {
  ::close(timerfd_);
  // do not remove channel, since we're in EventLoop::dtor();
  for (TimerList::iterator it = timers_.begin(); it != timers_.end(); ++it) {
    delete it->second;
  }
}

// 这里传入的时间是绝对时间
TimerId TimerQueueUevent::AddTimer(TimerCb cb, Timestamp when,
                                   double interval) {
  Timer* timer = new Timer(std::move(cb), when, interval);
  loop_->RunInLoop(std::bind(&TimerQueueUevent::AddTimerInLoop, this, timer));
  return TimerId(timer, timer->sequence());
}

void TimerQueueUevent::Cancel(const TimerId& timer_id) {
  loop_->RunInLoop(std::bind(&TimerQueueUevent::CancelInLoop, this, timer_id));
}

void TimerQueueUevent::AddTimerInLoop(Timer* timer) {
  loop_->AssertInLoopThread();
  bool earliest_changed = Insert(timer);
  if (earliest_changed) {
    ResetTimerfd(timerfd_, timer->expiration());
  }
}

void TimerQueueUevent::CancelInLoop(const TimerId& timer_id) {
  loop_->AssertInLoopThread();
  assert(timers_.size() == active_timers_.size());
  ActiveTimer timer(timer_id.timer_, timer_id.sequence_);
  ActiveTimerSet::iterator it = active_timers_.find(timer_id.sequence_);
  if (it != active_timers_.end()) {
    size_t n =
        timers_.erase(Entry(it->second.first->expiration(), it->second.first));
    assert(n == 1);
    (void)n;
    delete it->second.first;
    active_timers_.erase(it);
  } else if (calling_expired_timers_) {
    // 如果找不到，且正在调用定时回调，则需要记录下来，否则这次cancel 就丢失了
    // 执行完定时回调后，有可能又加入进去了（周期执行的定时器）
    canceling_timers_.insert(std::make_pair(timer_id.sequence_, timer));
  }
  assert(timers_.size() == active_timers_.size());
}

void TimerQueueUevent::TimerfdReadCb() {
  loop_->AssertInLoopThread();
  Timestamp now(Timestamp::now());
  ReadTimerfd(timerfd_, now);
  std::vector<Entry> expired = GetExpired(now);
  // 如果在定时回调中调用了 cancel,
  calling_expired_timers_ = true;
  canceling_timers_.clear();
  // safe to callback outside critical section
  for (std::vector<Entry>::iterator it = expired.begin(); it != expired.end();
       ++it) {
    // 在同一批超时的回调中被cancel了不执行了
    if (canceling_timers_.find(it->second->sequence()) ==
        canceling_timers_.end()) {
      it->second->Run();
    }
  }
  calling_expired_timers_ = false;
  Reset(expired, now);
}

std::vector<TimerQueueUevent::Entry> TimerQueueUevent::GetExpired(
    Timestamp now) {
  assert(timers_.size() == active_timers_.size());
  std::vector<Entry> expired;
  Entry sentry(now, reinterpret_cast<Timer*>(UINTPTR_MAX));
  TimerList::iterator end = timers_.lower_bound(sentry);
  // 全部超时，或者部分超时
  assert(end == timers_.end() || now < end->first);
  std::copy(timers_.begin(), end, back_inserter(expired));
  timers_.erase(timers_.begin(), end);

  for (std::vector<Entry>::iterator it = expired.begin(); it != expired.end();
       ++it) {
    size_t n = active_timers_.erase(it->second->sequence());
    assert(n == 1);
    (void)n;
  }

  assert(timers_.size() == active_timers_.size());
  return expired;
}

void TimerQueueUevent::Reset(const std::vector<Entry>& expired, Timestamp now) {
  Timestamp next_expire;
  for (std::vector<Entry>::const_iterator it = expired.begin();
       it != expired.end(); ++it) {
    // 如果是周期定时，且没有被取消则restart后重新加回
    if (it->second->repeat() &&
        canceling_timers_.find(it->second->sequence()) ==
            canceling_timers_.end()) {
      it->second->Restart(now);  // 重新设置定时
      Insert(it->second);
    } else {
      delete it->second;
    }
  }
  if (!timers_.empty()) {
    next_expire = timers_.begin()->second->expiration();
  }
  if (next_expire.valid()) {
    ResetTimerfd(timerfd_, next_expire);
  }
}

bool TimerQueueUevent::Insert(Timer* timer) {
  loop_->AssertInLoopThread();
  assert(timers_.size() == active_timers_.size());
  bool earliest_changed = false;
  Timestamp when = timer->expiration();
  TimerList::iterator it = timers_.begin();
  if (it == timers_.end() || when < it->first) {
    earliest_changed = true;
  }
  {
    std::pair<TimerList::iterator, bool> result =
        timers_.insert(Entry(when, timer));
    assert(result.second);
    (void)result;
  }
  {
    std::pair<ActiveTimerSet::iterator, bool> result =
        active_timers_.insert(std::make_pair(
            timer->sequence(), ActiveTimer(timer, timer->sequence())));
    assert(result.second);
    (void)result;
  }

  assert(timers_.size() == active_timers_.size());
  return earliest_changed;
}
