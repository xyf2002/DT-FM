#include "timer_queue_libevent.h"

#include <event2/event.h>

#include "eventloop_libevent.h"
#include "uevent.h"

namespace uevent {

TimerQueueLibevent::~TimerQueueLibevent() { event_free(timer_event_); }
int TimerQueueLibevent::BindWithLoop() {
  struct event_base* base = loop_->GetInnerBase();
  timer_event_ = event_new(base, timerfd_, EV_READ | EV_PERSIST,
                           TimerfdReadCbWrapper, this);
  event_add(timer_event_, NULL);
  return 0;
}

void TimerQueueLibevent::TimerfdReadCbWrapper(int fd, short event, void* arg) {
  TimerQueueLibevent* instance = reinterpret_cast<TimerQueueLibevent*>(arg);
  if (event & EV_READ) {
    instance->TimerfdReadCb();
  }
}

}  // namespace uevent
