#if 0

#include "message_util.h"

#include <atomic>
//#include "evbuffer_zero_copy_stream.h"
//#include "manager_thread.h"
#include "logging.h"

using namespace std;
using namespace uevent;

MessageUtil::MessageUtil() {}

MessageUtil::~MessageUtil() {}

unsigned MessageUtil::Flowno() {
  static atomic<unsigned> flow_no(1024);
  unsigned ret = flow_no++;
  return ret;
}

unsigned MessageUtil::ObjId() {
  static atomic<unsigned> obj_id(1024);
  unsigned ret = obj_id++;
  return ret;
}

void MessageUtil::ProtobufReadCallBack(const ConnectionUeventPtr& conn) {
  conn->GetLoop()->GetProtobufDispatcher()->ProtobufDecodeCb(conn);
}

int MessageUtil::SendPbRequest(
              const ConnectionUeventPtr& conn,
              const ucloud::UMessage& message,
              PbResponseCb res_cb,
              TimerCb timeout_cb,
              double time) {
  conn->GetLoop()->RunInLoop(std::bind(&MessageUtil::SendPbRequestInLoop, conn, message, res_cb, timeout_cb, time));
  return 0;
}

// 如果是不关心response的request，后三个参数不设。
int MessageUtil::SendPbRequestInLoop(
              const ConnectionUeventPtr& conn,
              const ucloud::UMessage& message,
              PbResponseCb res_cb,
              TimerCb timeout_cb,
              double time) {
  uint32_t size = message.ByteSize();
  void* buf = malloc(size);
  message.SerializeToArray(buf, size);
  uint32_t nl_size = htonl(size);
  conn->SendData(&nl_size, sizeof(nl_size));
  conn->SendData(buf, size);
  free(buf);
  buf = NULL;
  const ucloud::Head& head = message.head();
  uint32_t obj_id = head.source_entity();
  std::string call_purpose = head.call_purpose();
  std::string session_no = head.session_no();
  std::string timeout_desc = "obj_id:" + std::to_string(obj_id) + ",sesssion_no:" + session_no
      + ",call_purpose:" + call_purpose;
  if (res_cb && timeout_cb && time) {
    conn->GetLoop()->GetProtobufDispatcher()-> RegisterResponseEntry(obj_id,
        res_cb, timeout_cb, timeout_desc, time);
  }
  return 0;
}

//对于pb应答消息不需要注册应答回调和超时回调
int MessageUtil::SendPbResponse(
              const ConnectionUeventPtr& conn,
              const ucloud::UMessage& message) {
  conn->GetLoop()->RunInLoop(std::bind(&MessageUtil::SendPbResponseInLoop, conn, message));
  return 0;
}

int MessageUtil::SendPbResponseInLoop(
              const ConnectionUeventPtr& conn,
              const ucloud::UMessage& message) {
  uint32_t size = message.ByteSize();
  void* buf = malloc(size);
  message.SerializeToArray(buf, size);
  uint32_t nl_size = htonl(size);
  conn->SendData(&nl_size, sizeof(nl_size));
  conn->SendData(buf, size);
  free(buf);
  buf = NULL;
  return 0;
}

#endif
