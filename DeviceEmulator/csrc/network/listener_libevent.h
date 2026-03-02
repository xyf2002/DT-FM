#ifndef LISTENER_LIBEVENT_H
#define LISTENER_LIBEVENT_H

#include <event2/listener.h>

#include <string>

#include "uevent.h"

namespace uevent {

class ListenerLibevent : public ListenerUevent {
 public:
  ListenerLibevent(UeventLoop* loop, const UsockAddress& listen_addr,
                   const std::string& name, const Option& option = Option());
  virtual ~ListenerLibevent() {}
  static void AcceptCb(struct evconnlistener* listener, evutil_socket_t sockfd,
                       struct sockaddr* addr, int len, void* arg);
  static void AcceptErrorCb(struct evconnlistener* listener, void* arg);

 private:
  struct evconnlistener* evlistener_;
};

}  // namespace uevent

#endif
