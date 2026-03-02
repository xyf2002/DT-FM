#ifndef CONNECTION_LIBEVENT_H
#define CONNECTION_LIBEVENT_H

#include "base/logging.h"
#include "event2/buffer.h"
#include "event2/bufferevent.h"
#include "event2/event.h"
#include "uevent.h"

// BEV_EVENT_READING   0x01    /**< 读取过程中遇到的错误 */
// BEV_EVENT_WRITING   0x02    /**< 写数据过程中遇到错误 */
// BEV_EVENT_EOF       0x10    /**< 读到了文件结束符EOF */
// BEV_EVENT_ERROR     0x20    /**< 遇到了不可恢复的错误 */
// BEV_EVENT_TIMEOUT   0x40    /**< 表示计时器的时间到了 */
// BEV_EVENT_CONNECTED 0x80    /**< 表示connect操作完成 */

namespace uevent {

class EventLoopLibevent;

class BuffereventWrapper {
 public:
  BuffereventWrapper(struct event_base* base, int fd,
                     const ConnectionUeventPtr& conn);
  ~BuffereventWrapper();
  inline struct bufferevent* bev() const { return bev_; }
  inline ssize_t ReadableLength() { return evbuffer_get_length(inbuf_); }
  inline int SendData(const void* data, size_t data_len) {
    return evbuffer_add(outbuf_, data, data_len);
  }
  inline int SendDataZeroCopy(const void* data, size_t data_len,
                              void (*cleanup_cb)(const void*, size_t, void*),
                              void* cleanup_arg) {
    return evbuffer_add_reference(outbuf_, data, data_len, cleanup_cb,
                                  cleanup_arg);
  }
  inline unsigned char* PullupData(size_t len) {
    return evbuffer_pullup(inbuf_, static_cast<ev_ssize_t>(len));
  }
  inline int ReceiveData(void* data, size_t data_len) {
    return evbuffer_copyout(inbuf_, data, data_len);
  }
  inline int DrainData(size_t len) { return evbuffer_drain(inbuf_, len); }
  int RemoveData(void* data, size_t data_len) {
    return evbuffer_remove(inbuf_, data, data_len);
  }
  void enable() { bufferevent_enable(bev_, EV_READ | EV_WRITE | EV_PERSIST); }

 private:
  inline static void ReadCbWrapper(struct bufferevent* bev, void* arg);
  inline static void WriteCbWrapper(struct bufferevent* bev, void* arg);
  static void EventCbWrapper(struct bufferevent* bev, short event, void* arg);
  struct evbuffer* inbuf_;
  struct evbuffer* outbuf_;
  struct bufferevent* bev_;
  ConnectionUeventPtr conn_;
};

class ConnectionLibevent : public ConnectionUevent {
 public:
  // @brief 通过accept 产生的连接，此时连接已经建立成功
  // 在 listener 中分发连接时，会调用连接成功的回调.
  // 用connect建立的连接，会产生连接成功事件触发回调
  ConnectionLibevent(EventLoopLibevent* loop, int fd, int64_t conn_id,
                     const std::string& name, const UsockAddress& peer_addr);
  virtual ~ConnectionLibevent();
  virtual int Init();
  virtual void SetFd();  // connector 确认连接成功后调用
  virtual ssize_t ReadableLength();
  virtual int SendData(const void* data, size_t data_len);
  virtual int SendDataZeroCopy(const void* data, size_t data_len,
                               void (*cleanup_cb)(const void*, size_t, void*),
                               void* cleanup_arg);
  virtual unsigned char* PullupData(size_t len);
  virtual int ReceiveData(void* data, size_t data_len);
  virtual int DrainData(size_t len);
  virtual int RemoveData(void* data, size_t data_len);
  virtual void ConnectionEstablished();
  virtual void ForceClose();
  virtual void ForceCloseInLoop();
  virtual void ConnectionEnable();
  virtual bool IsClosed();

  struct bufferevent* GetInnerBev() {
    if (bev_wrapper_ == NULL) {
      LOG_ERROR << "bev is null";
      return NULL;
    }
    return bev_wrapper_->bev();
  }

 private:
  BuffereventWrapper* bev_wrapper_;
};

}  // namespace uevent
#endif
