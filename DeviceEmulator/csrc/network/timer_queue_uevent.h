// Copyright 2010, Shuo Chen.  All rights reserved.
// http://code.google.com/p/muduo/
//
// Use of this source code is governed by a BSD-style license
// that can be found in the License file.

// Author: Shuo Chen (chenshuo at chenshuo dot com)
//
// This is an internal header file, you should not include this.

#ifndef UEVENT_TIMER_QUEUE_H_
#define UEVENT_TIMER_QUEUE_H_

#include <set>
#include <unordered_map>
#include <vector>

#include "base/mutex.h"
#include "base/timestamp.h"
#include "callbacks.h"

namespace uevent {

class UeventLoop;
class Timer;
class TimerId;

/// A best efforts timer queue.
/// No guarantee that the callback will be on time.

class TimerQueueUevent : base::noncopyable {
 public:
  explicit TimerQueueUevent(UeventLoop* loop);
  ~TimerQueueUevent();
  virtual int BindWithLoop() = 0;

  /// Schedules the callback to be run at given time,
  /// repeats if @c interval > 0.0.
  /// Must be thread safe. Usually be called from other threads.
  TimerId AddTimer(TimerCb cb, base::Timestamp when, double interval);

  void Cancel(const TimerId& timerId);

 protected:
  void TimerfdReadCb();
  const int timerfd_;

 private:
  typedef std::pair<base::Timestamp, Timer*> Entry;
  typedef std::set<Entry> TimerList;
  typedef std::pair<Timer*, int64_t> ActiveTimer;
  typedef std::unordered_map<int64_t, ActiveTimer> ActiveTimerSet;

  void AddTimerInLoop(Timer* timer);
  void CancelInLoop(const TimerId& timer_id);
  // move out all expired timers
  std::vector<Entry> GetExpired(base::Timestamp now);
  void Reset(const std::vector<Entry>& expired, base::Timestamp now);
  bool Insert(Timer* timer);

  UeventLoop* loop_;
  // Timer list sorted by expiration
  TimerList timers_;

  // 为了删除的时候能通过TimerId 快速找到对应的定时器
  //  TODO(henry) 这个可以使用unordermap 提到效率
  ActiveTimerSet active_timers_;

  bool calling_expired_timers_;
  ActiveTimerSet canceling_timers_;
};

}  // namespace uevent

#endif  // UEVENT_TIMERQUEUE_H_
