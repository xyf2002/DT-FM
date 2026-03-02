#ifndef UEVENT_TIMER_QUEUE_LIBEVENT_H_
#define UEVENT_TIMER_QUEUE_LIBEVENT_H_

#include "timer_queue_uevent.h"

struct event;

namespace uevent {

class UeventLoop;
class EventLoopLibevent;

class TimerQueueLibevent : public TimerQueueUevent {
 public:
  explicit TimerQueueLibevent(EventLoopLibevent* loop)
      : TimerQueueUevent(reinterpret_cast<UeventLoop*>(loop)),
        loop_(loop),
        timer_event_(NULL) {}
  virtual ~TimerQueueLibevent();
  virtual int BindWithLoop();
  static void TimerfdReadCbWrapper(int fd, short event, void* arg);

 private:
  EventLoopLibevent* loop_;
  struct event* timer_event_;
};

}  // namespace uevent

#endif
