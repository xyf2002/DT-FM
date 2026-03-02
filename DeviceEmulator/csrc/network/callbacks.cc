#include "callbacks.h"

#include "loop_handle.h"
#include "uevent.h"

namespace uevent {

// TODO(yeheng) 没有注册回调,就使用默认的回调，打印一些可用的信息

void DefaultConnectionSuccessCbImpl(const ConnectionUeventPtr& conn) {}
void DefaultConnectionClosedCbImpl(const ConnectionUeventPtr& conn) {}
void DefaultMessageReadCbImpl(const ConnectionUeventPtr& conn) {}
void DefaultMessageWriteCbImpl(const ConnectionUeventPtr& conn) {}

ConnectionSuccessCb DefaultConnectionSuccessCb =
    &DefaultConnectionSuccessCbImpl;
ConnectionClosedCb DefaultConnectionClosedCb = &DefaultConnectionClosedCbImpl;
MessageReadCb DefaultMessageReadCb = &DefaultMessageReadCbImpl;
MessageWriteCb DefaultMessageWriteCb = &DefaultMessageWriteCbImpl;

}  // namespace uevent
