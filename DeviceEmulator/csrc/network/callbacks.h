#ifndef CALLBACKS_H
#define CALLBACKS_H

#include <stdint.h>

#include <functional>
#include <memory>

namespace ucloud {
class UMessage;
}

namespace uevent {

// All client visible callbacks go here.
class ConnectionUevent;
class ConnectorUevent;
class UeventLoop;
// class AioParam;
// class CrossAioParam;
class LoopHandle;
class PbRequestHandle;

typedef std::shared_ptr<ConnectionUevent> ConnectionUeventPtr;
typedef std::shared_ptr<ConnectorUevent> ConnectorUeventPtr;

typedef std::function<PbRequestHandle*(UeventLoop*)> PbRequestCb;
typedef std::function<void(ucloud::UMessage*)> PbResponseCb;
typedef std::function<void()> TimerCb;

typedef std::function<void(const ConnectionUeventPtr&)> ConnectionSuccessCb;
typedef std::function<void(const ConnectionUeventPtr&)> ConnectionClosedCb;
typedef std::function<void(const ConnectionUeventPtr&)> MessageReadCb;
typedef std::function<void(const ConnectionUeventPtr&)> MessageWriteCb;
// typedef std::function<void(AioParam*)> AioResponseCb;
// typedef std::function<void(CrossAioParam*)> CrossAioResponseCb;

typedef std::function<void(UeventLoop*)> ThreadInitCb;
typedef std::function<LoopHandle*(UeventLoop*)> CreateLoopHandleCb;

extern ConnectionSuccessCb DefaultConnectionSuccessCb;
extern ConnectionClosedCb DefaultConnectionClosedCb;
extern MessageReadCb DefaultMessageReadCb;
extern MessageWriteCb DefaultMessageWriteCb;

void DefaultConnectionSuccessCbImpl(const ConnectionUeventPtr& conn);
void DefaultConnectionClosedCbImpl(const ConnectionUeventPtr& conn);
void DefaultMessageReadCbImpl(const ConnectionUeventPtr& conn);
void DefaultMessageWriteCbImpl(const ConnectionUeventPtr& conn);

typedef int (*IoFinishFunc)(void* arg);
// typedef int (*TransAioParamFunc) (void *arg, int *io_type, int *fd, void
// **buf, uint64_t *offset, uint64_t *size);
typedef int (*LoopArgFunc)(void* arg);

}  // namespace uevent

#endif
