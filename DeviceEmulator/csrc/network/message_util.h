#if 0

#ifndef UEVENT_MESSAGE_UTIL_H_
#define UEVENT_MESSAGE_UTIL_H_

#include "callbacks.h"
#include "pb_request_handle.h"
#include "umessage_common.h"

#define REGISTE_PROTO_HANDLER(loop, message_type, ClassName)         \
  do {                                                               \
    (loop)->GetProtobufDispatcher()->RegisterRequestCb(              \
        message_type,                                                \
        std::bind(&ClassName::CreateMyself, std::placeholders::_1)); \
  } while (0);

#define MYSELF_CREATE(ClassName)                                           \
  static uevent::PbRequestHandle* CreateMyself(uevent::UeventLoop* loop) { \
    return reinterpret_cast<PbRequestHandle*>(new ClassName(loop));        \
  }

namespace uevent {


class MessageUtil {
 public:
  MessageUtil();
  ~MessageUtil();
  static unsigned Flowno();
  static unsigned ObjId();
  static int SendPbRequest(
              const ConnectionUeventPtr& conn,
              const ucloud::UMessage& message,
              PbResponseCb res_cb = PbResponseCb(),
              TimerCb timeout_cb = TimerCb(),
              double time = 0.0);
    //对于pb应答消息不需要注册应答回调和超时回调
  static int SendPbResponse(
              const ConnectionUeventPtr& conn,
              const ucloud::UMessage& message);

  static void ProtobufReadCallBack(const ConnectionUeventPtr& conn);

 private:
   // 如果是不关心response的request，后三个参数不设。
  static int SendPbRequestInLoop(
              const ConnectionUeventPtr& conn,
              const ucloud::UMessage& message,
              PbResponseCb res_cb,
              TimerCb timeout_cb,
              double time);
  static int SendPbResponseInLoop(
              const ConnectionUeventPtr& conn,
              const ucloud::UMessage& message);
};

} // namespace uevent

#endif

#endif
