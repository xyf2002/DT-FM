#include "listener_libevent.h"

#include <unistd.h>

#include "base/logging.h"
#include "connection_libevent.h"
#include "eventloop_libevent.h"
#include "ueventloop_thread_pool.h"

namespace uevent {

ListenerLibevent::ListenerLibevent(UeventLoop* loop,
                                   const UsockAddress& listen_addr,
                                   const std::string& name,
                                   const Option& option)
    : ListenerUevent(loop, listen_addr, name, option), evlistener_(NULL) {
  uint32_t flag = LEV_OPT_CLOSE_ON_FREE;
  if (option.reuse_port) {
    flag |= LEV_OPT_REUSEABLE;
  }
  evlistener_ = evconnlistener_new_bind(
      static_cast<EventLoopLibevent*>(loop)->GetInnerBase(), AcceptCb, this,
      flag, -1, listen_addr_.GetSockAddr(), listen_addr_.GetSockAddrLen());
  if (!evlistener_) {
    LOG_SYSFATAL << "create listener failed at: " << listen_addr.ToString();
  }
  evconnlistener_set_error_cb(evlistener_, AcceptErrorCb);
}

void ListenerLibevent::AcceptCb(struct evconnlistener* listener,
                                evutil_socket_t sockfd, struct sockaddr* addr,
                                int len, void* arg) {
  ListenerLibevent* pobj = reinterpret_cast<ListenerLibevent*>(arg);
  struct sockaddr_storage* p = reinterpret_cast<struct sockaddr_storage*>(addr);
  UsockAddress peer_addr(*p);
  LOG_DEBUG << "new connection from client:" << peer_addr.ToString()
            << ", and the fd:" << sockfd;
  EventLoopLibevent* io_loop = NULL;
  if (pobj->option_.loop_strategy == Option::kRoundRobin) {
    io_loop =
        reinterpret_cast<EventLoopLibevent*>(pobj->thread_pool_->GetNextLoop());
  } else if (pobj->option_.loop_strategy == Option::kEmptyOne) {
    io_loop = reinterpret_cast<EventLoopLibevent*>(
        pobj->thread_pool_->GetEmptyLoop());
  } else if (pobj->option_.loop_strategy == Option::kLightestOne) {
    io_loop = reinterpret_cast<EventLoopLibevent*>(
        pobj->thread_pool_->GetLightestLoop());
  }
  if (io_loop == NULL) {
    ::close(sockfd);
    LOG_ERROR << "can't get an available loop for new connection, fd:"
              << sockfd;
  }
  LoopHandle* loop_handle = io_loop->GetLoopHandle();
  if (loop_handle != NULL) {  // 为loop的引用计数加一
    loop_handle->IncRefs();
  }
  int64_t conn_id = pobj->next_conn_id_++;
  ConnectionUeventPtr conn(
      new ConnectionLibevent(io_loop, sockfd, conn_id, "", peer_addr));
  conn->Init();
  LOG_DEBUG << "accept new connection, conn_id:" << conn_id << " fd:" << sockfd
            << " perr address:" << peer_addr.ToString();
  pobj->connections_[conn_id] = conn;  // 保存在这里，退出后不会析构
  conn->SetConnectionSuccessCb(pobj->connection_success_cb_);
  conn->SetConnectionClosedCb(pobj->connection_closed_cb_);
  conn->SetMessageReadCb(pobj->message_read_cb_);
  conn->SetMessageWriteCb(pobj->message_write_cb_);
  io_loop->RunInLoop(std::bind(&ConnectionUevent::ConnectionEstablished, conn));
}

void ListenerLibevent::AcceptErrorCb(struct evconnlistener* listener,
                                     void* arg) {
  ListenerLibevent* pobj = reinterpret_cast<ListenerLibevent*>(arg);
  LOG_SYSFATAL << "get an error on libevent listener, name:" << pobj->GetName();
}

}  // namespace uevent
