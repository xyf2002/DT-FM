#ifndef CONNECTOR_LIBEVENT_H
#define CONNECTOR_LIBEVENT_H

#include <string>

#include "uevent.h"

namespace uevent {

class EventLoopLibevent;

class ConnectorLibevent : public ConnectorUevent {
 public:
  ConnectorLibevent(UeventLoop* loop, const UsockAddress& peer_addr,
                    const std::string& name);

  virtual ~ConnectorLibevent();
  virtual int Connect();

 private:
  EventLoopLibevent* loop_;
};

}  // namespace uevent

#endif
