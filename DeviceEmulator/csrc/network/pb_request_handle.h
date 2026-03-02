#if 0

#ifndef PROTOBUF_REQUEST_HANDLE_H
#define PROTOBUF_REQUEST_HANDLE_H

#include "ucloud_message.h"
#include "uevent.h"

namespace uevent {

class PbRequestHandle:
    public std::enable_shared_from_this<PbRequestHandle> {
 public:
  virtual ~PbRequestHandle() {}
  virtual void EntryInit(const ConnectionUeventPtr& conn, ucloud::UMessage* um) = 0;
};

}
#endif

#endif
