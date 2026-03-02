#if 0

#ifndef UEVENT_PROTOBUF_DISPATCHER_H_
#define UEVENT_PROTOBUF_DISPATCHER_H_

#include <memory>
#include <unordered_map>

#include "callbacks.h"
#include "logging.h"
#include "pb_request_handle.h"
#include "timer_id.h"

namespace uevent {

class ConnectionUevent;
class UeventLoop;
class PbResponseCbEntry;

class ProtobufDispatcher {
 public:
  explicit ProtobufDispatcher(UeventLoop* loop);

  //接收到一个request请求，根据 message_type找到处理的回调函数
  void RegisterRequestCb(int message_type, PbRequestCb cb);
  //接收到一个response，通过 obj_id找到发送对象
  void RegisterResponseEntry(uint32_t obj_id,
                             PbResponseCb res_cb,
                             TimerCb timeout_cb,
                             const std::string& timeout_desc,
                             double time);
  void UnregisterResponseCbEntry(uint32_t obj_id);
  void dispatch(const ConnectionUeventPtr& conn, ucloud::UMessage* um);
  void ProtobufDecodeCb(ConnectionUeventPtr conn);
  int DecodeMessage(::ucloud::UMessage* pMessage, const char* pData, unsigned iSize);

 private:
  UeventLoop* loop_;
  std::unordered_map<int, PbRequestCb> map_request_cb_;
  std::unordered_map<uint32_t, PbResponseCbEntry*> map_response_cb_entry_;
};

struct PbResponseCbEntry {
  void TimeoutCbWrapper() {
    LOG_ERROR << timeout_desc << " timeout";
    timeout_cb();
    // 这个函数最后调用，因为会析构自己
    dispatcher->UnregisterResponseCbEntry(obj_id);
  }
  uint64_t obj_id;
  PbResponseCb res_cb;
  TimerCb timeout_cb;
  TimerId timer_id;
  double time;
  std::string timeout_desc;
  std::shared_ptr<ProtobufDispatcher> dispatcher;
};

} // nanespace uevent

#endif

#endif
