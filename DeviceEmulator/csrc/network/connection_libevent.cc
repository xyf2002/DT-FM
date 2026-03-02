#include "connection_libevent.h"

#include "base/logging.h"
#include "eventloop_libevent.h"

namespace uevent {

BuffereventWrapper::BuffereventWrapper(struct event_base* base, int fd,
                                       const ConnectionUeventPtr& conn) {
  conn_ = conn;
  bev_ = bufferevent_socket_new(base, fd, BEV_OPT_CLOSE_ON_FREE);
  inbuf_ = bufferevent_get_input(bev_);
  outbuf_ = bufferevent_get_output(bev_);
  bufferevent_setcb(bev_, ReadCbWrapper, WriteCbWrapper, EventCbWrapper, this);
}

BuffereventWrapper::~BuffereventWrapper() { bufferevent_free(bev_); }

inline void BuffereventWrapper::ReadCbWrapper(struct bufferevent* bev,
                                              void* arg) {
  BuffereventWrapper* pobj = reinterpret_cast<BuffereventWrapper*>(arg);
  pobj->conn_->message_read_cb_(pobj->conn_);
}

inline void BuffereventWrapper::WriteCbWrapper(struct bufferevent* bev,
                                               void* arg) {
  BuffereventWrapper* pobj = reinterpret_cast<BuffereventWrapper*>(arg);
  pobj->conn_->message_write_cb_(pobj->conn_);
}

void BuffereventWrapper::EventCbWrapper(struct bufferevent* bev, short event,
                                        void* arg) {
  BuffereventWrapper* pobj = reinterpret_cast<BuffereventWrapper*>(arg);
  if (event & BEV_EVENT_CONNECTED) {
    LOG_DEBUG << "connect success event:" << event;
    pobj->conn_->SetState(ConnectionUevent::kConnected);
    pobj->conn_->SetTcpNoDelay(true);  // 默认开启tcpnodelay
    pobj->conn_->connection_success_cb_(pobj->conn_);
  } else {
    LOG_DEBUG << "connect fail event:" << event;
    pobj->conn_->ForceClose();
  }
}

ConnectionLibevent::ConnectionLibevent(EventLoopLibevent* loop, int fd,
                                       int64_t conn_id, const std::string& name,
                                       const UsockAddress& peer_addr)
    : ConnectionUevent(reinterpret_cast<UeventLoop*>(loop), fd, conn_id, name,
                       peer_addr) {}

// 如果连接没有释放，关闭连接，触发
// connection_closed_cb_, 必须在本线程中调用析构函数,
// 析构函数不是线程安全的.
ConnectionLibevent::~ConnectionLibevent() {
  LOG_DEBUG << "connection deconstruct, conn_name:" << name_
            << " id: " << conn_id_ << " peer address:" << peer_addr_.ToString();
  // 主动释放连接的入口只有RemoveConnection和Destroy, 这两个地方都会保证
  // 将任然连接的conn关闭, 即保证bev_wrapper_已经析构
  if (!IsClosed()) {
    LOG_FATAL << "connection not closed when deconstruct";
  }
}
// 由于不能在构造函数中使用shared_from_this,需要调用Init
int ConnectionLibevent::Init() {
  struct event_base* base =
      static_cast<EventLoopLibevent*>(loop_)->GetInnerBase();
  std::shared_ptr<ConnectionLibevent> ptr =
      std::dynamic_pointer_cast<ConnectionLibevent>(shared_from_this());
  bev_wrapper_ = new BuffereventWrapper(base, fd_, ptr);
  return 0;
}

void ConnectionLibevent::SetFd() {
  fd_ = bufferevent_getfd(GetInnerBev());
  assert(fd_ != -1);
}

ssize_t ConnectionLibevent::ReadableLength() {
  if (bev_wrapper_ == NULL) {
    LOG_WARN << "the connection has been closed";
    return -1;
  }
  return bev_wrapper_->ReadableLength();
}

int ConnectionLibevent::SendData(const void* data, size_t data_len) {
  if (bev_wrapper_ == NULL) {
    LOG_WARN << "the connection has been closed";
    return -1;
  }
  return bev_wrapper_->SendData(data, data_len);
}

int ConnectionLibevent::SendDataZeroCopy(const void* data, size_t data_len,
                                         void (*cleanup_cb)(const void*, size_t,
                                                            void*),
                                         void* cleanup_arg) {
  if (bev_wrapper_ == NULL) {
    LOG_WARN << "the connection has been closed";
    return -1;
  }
  return bev_wrapper_->SendDataZeroCopy(data, data_len, cleanup_cb,
                                        cleanup_arg);
}

unsigned char* ConnectionLibevent::PullupData(size_t len) {
  if (bev_wrapper_ == NULL) {
    LOG_WARN << "the connection has been closed";
    return nullptr;
  }
  return bev_wrapper_->PullupData(len);
}

int ConnectionLibevent::ReceiveData(void* data, size_t data_len) {
  if (bev_wrapper_ == NULL) {
    LOG_WARN << "the connection has been closed";
    return -1;
  }
  return bev_wrapper_->ReceiveData(data, data_len);
}

int ConnectionLibevent::DrainData(size_t len) {
  if (bev_wrapper_ == NULL) {
    LOG_WARN << "the connection has been closed";
    return -1;
  }
  return bev_wrapper_->DrainData(len);
}

int ConnectionLibevent::RemoveData(void* data, size_t data_len) {
  if (bev_wrapper_ == NULL) {
    LOG_WARN << "the connection has been closed";
    return -1;
  }
  return bev_wrapper_->RemoveData(data, data_len);
}

void ConnectionLibevent::ConnectionEstablished() {
  ConnectionEnable();
  SetState(kConnected);
  SetTcpNoDelay(true);  // 默认开启tcpnodelay
  std::shared_ptr<ConnectionLibevent> ptr =
      std::dynamic_pointer_cast<ConnectionLibevent>(shared_from_this());
  connection_success_cb_(ptr);
}

void ConnectionLibevent::ForceClose() {
  std::shared_ptr<ConnectionLibevent> ptr =
      std::dynamic_pointer_cast<ConnectionLibevent>(shared_from_this());
  loop_->RunInLoop(std::bind(&ConnectionLibevent::ForceCloseInLoop, ptr));
}

void ConnectionLibevent::ForceCloseInLoop() {
  loop_->AssertInLoopThread();
  if (bev_wrapper_ == NULL) {
    LOG_DEBUG << "the connection has been closed, conn_name:" << name_
              << "connection id: " << conn_id_;
    return;
  }
  LOG_DEBUG << "close connection, conn_name:" << name_
            << " connection id: " << conn_id_
            << " peer addr:" << peer_addr_.ToString();
  delete bev_wrapper_;
  bev_wrapper_ = NULL;
  SetState(kDisconnected);
  std::shared_ptr<ConnectionLibevent> ptr =
      std::dynamic_pointer_cast<ConnectionLibevent>(shared_from_this());
  connection_closed_cb_(ptr);
}

void ConnectionLibevent::ConnectionEnable() {
  if (bev_wrapper_ == NULL) {
    LOG_ERROR << "the connection has been closed";
    return;
  }
  bev_wrapper_->enable();
}

bool ConnectionLibevent::IsClosed() { return bev_wrapper_ == NULL; }

}  // namespace uevent
