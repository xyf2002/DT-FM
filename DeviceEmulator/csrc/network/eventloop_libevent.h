#ifndef EVENT_LOOP_LIBEVENT_H
#define EVENT_LOOP_LIBEVENT_H

#include <event2/event.h>

#include "uevent.h"

namespace uevent {

class EventLoopLibevent : public UeventLoop {
 public:
  EventLoopLibevent(const std::string& thread_name,
                    CreateLoopHandleCb cb = CreateLoopHandleCb());
  virtual ~EventLoopLibevent();

  struct event_base* GetInnerBase();

  virtual void Start() {
    if (started_ == true) {
      return;
    }
    started_ = true;
    event_base_dispatch(base_);
  }

  virtual void Quit() {
    started_ = false;
    event_base_loopbreak(base_);
    event_base_free(base_);
    base_ = NULL;
  }

 private:
  static void WakeupReadCbWrapper(int fd, short event, void* arg);
  struct event* wakeup_event_;
  struct event_base* base_;
};

}  // namespace uevent

#endif
