#include "eventloop_libevent.h"
// #include "aio_libevent.h"
#include "timer_queue_libevent.h"

namespace uevent {

EventLoopLibevent::EventLoopLibevent(const std::string& thread_name,
                                     CreateLoopHandleCb cb)
    : UeventLoop(thread_name), base_(event_base_new()) {
  wakeup_event_ = event_new(base_, wakeup_fd_, EV_READ | EV_PERSIST,
                            WakeupReadCbWrapper, this);
  event_add(wakeup_event_, NULL);
  // aio_ptr_.reset(new AioLibevent());
  // aio_ptr_->BindWithLoop(this);
  timer_queue_ptr_.reset(new TimerQueueLibevent(this));
  timer_queue_ptr_->BindWithLoop();
  if (cb) {
    loop_handle_ = cb(this);
  }
}

EventLoopLibevent::~EventLoopLibevent() {
  event_free(wakeup_event_);
  if (base_ != NULL) {
    Quit();
  }
}

struct event_base* EventLoopLibevent::GetInnerBase() {
  return base_;
}

void EventLoopLibevent::WakeupReadCbWrapper(int fd, short event, void* arg) {
  if (event & EV_READ) {
    EventLoopLibevent* ev_loop = reinterpret_cast<EventLoopLibevent*>(arg);
    ev_loop->WakeupReadCb();
  }
}

}  // namespace uevent
