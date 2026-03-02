#if 0

#include "protobuf_dispatcher.h"

#include "logging.h"
#include "message_plug.h"
#include "ucloud_message.h"
#include "uevent.h"

namespace uevent {

ProtobufDispatcher::ProtobufDispatcher(UeventLoop* loop)
    : loop_(loop) {
}

void ProtobufDispatcher::RegisterRequestCb(int message_type, PbRequestCb cb) {
  if (map_request_cb_.find(message_type) == map_request_cb_.end()) {
    map_request_cb_[message_type] = cb;
  }
}

void ProtobufDispatcher::RegisterResponseEntry(uint32_t obj_id,
                                               PbResponseCb res_cb,
                                               TimerCb timeout_cb,
                                               const std::string& timeout_desc,
                                               double time) {
  PbResponseCbEntry*  entry = new PbResponseCbEntry();
  entry->obj_id = obj_id;
  entry->res_cb = std::move(res_cb);
  entry->timeout_cb = timeout_cb;
  entry->timeout_desc = timeout_desc;
  entry->timer_id = loop_->RunAfter(time,
      std::bind(&PbResponseCbEntry::TimeoutCbWrapper, entry));
  entry->dispatcher = loop_->GetProtobufDispatcher();
  if (map_response_cb_entry_.find(entry->obj_id) ==
          map_response_cb_entry_.end()) {
    map_response_cb_entry_[entry->obj_id] = entry;
  } else {
    LOG_FATAL << "the obj_id duplicated, impossible";
  }
}

void ProtobufDispatcher::UnregisterResponseCbEntry(uint32_t obj_id) {
  auto it = map_response_cb_entry_.find(obj_id);
  if (it == map_response_cb_entry_.end()) {
    LOG_ERROR << "response cb entry has been unregister";
    return;
  }
  delete it->second; // 析构entry
  map_response_cb_entry_.erase(obj_id); // 触发后需要注销这个对象的回调
}

int ProtobufDispatcher::DecodeMessage(::ucloud::UMessage* pMessage, const char *pData, unsigned iSize) {
  if ( iSize < sizeof(unsigned) ) {
    return 0;
  }
  unsigned iDataSize = ntohl(*(unsigned *)pData);
  if ( iSize < iDataSize + sizeof(unsigned)) {
    return 0;
  }

  if (!pMessage->ParseFromArray(pData + sizeof(unsigned), iDataSize)) {
    return -1;
  }

  return iDataSize + sizeof(unsigned);
}

// 这里不能用引用，因为在消息处理中可能会释放ctor及其中的connection
// 这样导致继续解消息时会core掉
void ProtobufDispatcher::ProtobufDecodeCb(ConnectionUeventPtr conn) {
  while(1) { // 如果一次来多个消息要一次处理完
    ucloud::UMessage um;
    ssize_t readable = conn->ReadableLength();
    if (readable <= 0) {
      LOG_TRACE << "connection closed or no more data, readable:" << readable;
      break;
    }
    char* data = new char[readable];
    conn->ReceiveData(data, readable);
    int res = DecodeMessage(&um, data, readable);
    delete[] data;
    if (res == -1) {
      LOG_ERROR << "decode pb message error, close the connection";
      conn->ForceClose();
      break;
    } else if (res == 0) {
      LOG_TRACE << "pb message length not enough";
      break;
    } else { // res > 0
      dispatch(conn, &um); //按消息类型调用回调
      conn->DrainData(res); //传引用的话这行会core
    }
  }
}

void ProtobufDispatcher::dispatch(const ConnectionUeventPtr& conn,
                                  ucloud::UMessage* um) {
  uint32_t obj_id = um->head().dest_entity();
  int type = um->head().message_type();
  if (obj_id != 0) {  //说明是response
    auto it = map_response_cb_entry_.find(obj_id);
    if (it == map_response_cb_entry_.end()) {
      LOG_ERROR << "can't find the object for pb response, obj_id: "
                << obj_id << "message type: "
                << type << " maybe has timeout";
      return;
    }
    loop_->CancelTimer(it->second->timer_id); // 取消超时定时
    // 调用response的回调
    it->second->res_cb(um);
    UnregisterResponseCbEntry(obj_id); // 触发后需要注销这个对象的回调
  } else {  // 接收到request
    auto it = map_request_cb_.find(type);
    if (it == map_request_cb_.end()) {
      LOG_ERROR << "can't find the pb request callback, request type: "
                << type;
      return;
    }
    //用智能指针管理状态机，引用为0时会自动释放
    std::shared_ptr<PbRequestHandle> pb_req_handle(it->second(loop_)); //创建对应的处理类
    pb_req_handle->EntryInit(conn, um);
  }
}

} // namespace uevent

#endif
