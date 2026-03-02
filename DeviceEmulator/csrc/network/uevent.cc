#include <netinet/tcp.h>
#include <signal.h>
#include <sys/eventfd.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>

#include "base/logging.h"
#include "ueventloop_thread_pool.h"
// #include "aio_loop_thread_pool.h"
#include "uevent.h"
// #include "aio_uevent.h"
#include "timer_queue_uevent.h"

using base::MutexLockGuard;
using base::Timestamp;

namespace uevent {

namespace {

__thread UeventLoop* loop_in_this_thread = 0;

#pragma GCC diagnostic ignored "-Wold-style-cast"
class IgnoreSigPipe {
 public:
  IgnoreSigPipe() {
    ::signal(SIGPIPE, SIG_IGN);
    // LOG_TRACE << "Ignore SIGPIPE";
  }
};

IgnoreSigPipe init_obj;

}  // namespace

int CreateEventfd() {
  int evtfd = ::eventfd(0, EFD_NONBLOCK | EFD_CLOEXEC);
  if (evtfd < 0) {
    LOG_SYSERR << "Failed in eventfd";
    abort();
  }
  return evtfd;
}

UeventLoopBase::UeventLoopBase()
    : started_(false), thread_id_(base::CurrentThread::tid()) {}

UeventLoopBase::~UeventLoopBase() {}

void UeventLoopBase::AssertInLoopThread() {
  if (!IsInLoopThread()) {
    LOG_FATAL << "UeventLoop::AssertInLoopThread " << this
              << " was created in thread id: = " << thread_id_
              << ", current thread id: = " << base::CurrentThread::tid();
  }
}

//=======================================UeventLoop============================

UeventLoop::UeventLoop(const std::string& thread_name)
    : UeventLoopBase(),
      thread_name_(thread_name),
      loog_arg_func_(NULL),
      wakeup_fd_(CreateEventfd()),
      timer_fd_(-1),
      loop_handle_(NULL) {
  // LOG_INFO << "UeventLoop created " << this << " in thread id: " <<
  // thread_id_;
  if (loop_in_this_thread) {
    LOG_FATAL << "Another UeventLoop " << loop_in_this_thread
              << " exists in this thread id: " << thread_id_;
  } else {
    loop_in_this_thread = this;
  }

  // init protobuf dispatcher
  // protobuf_dispatcher_ptr_.reset(new ProtobufDispatcher(this));
}

UeventLoop::~UeventLoop() { ::close(wakeup_fd_); }

int UeventLoop::Wakeup() {
  uint64_t one = 1;
  ssize_t n = ::write(wakeup_fd_, &one, sizeof one);
  if (n != sizeof one) {
    LOG_ERROR << "UeventLoop::Wakeup() writes " << n << " bytes instead of 8";
    return -1;
  }
  return 0;
}

int UeventLoop::WakeupReadCb() {
  uint64_t one = 1;
  ssize_t n = ::read(wakeup_fd_, &one, sizeof one);
  if (n != sizeof one) {
    LOG_ERROR << "UeventLoop::WakeupReadCb reads " << n
              << " bytes instead of 8";
    return -1;
  }
  DoPendingFunctors();
  return 0;
}

void UeventLoop::DoPendingFunctors() {
  std::vector<Functor> functors;
  std::vector<void*> tasks;
  {
    base::MutexLockGuard lock(mutex_);
    // 使用swap减少了加锁时间，也防止嵌套调用QueueInLoop死锁
    functors.swap(pending_functors_);
    tasks.swap(pending_tasks_);
  }
  for (size_t i = 0; i < functors.size(); ++i) {
    functors[i]();
  }
  for (size_t i = 0; i < tasks.size(); ++i) {
    loog_arg_func_(tasks[i]);
  }
}

void UeventLoop::RunInLoop(Functor cb) {
  if (IsInLoopThread()) {
    cb();
  } else {
    QueueInLoop(std::move(cb));
  }
}

int UeventLoop::QueueInLoop(Functor cb) {
  bool empty = false;
  {
    base::MutexLockGuard lock(mutex_);
    empty = pending_functors_.empty();
    pending_functors_.push_back(std::move(cb));
  }
  if (empty) return Wakeup();
  return 0;
}

void UeventLoop::RunInLoop(void* arg) {
  if (IsInLoopThread()) {
    loog_arg_func_(arg);
  } else {
    QueueInLoop(arg);
  }
}

int UeventLoop::QueueInLoop(void* arg) {
  bool empty = false;
  {
    base::MutexLockGuard lock(mutex_);
    empty = pending_tasks_.empty();
    pending_tasks_.push_back(arg);
  }
  // TODO 这里使用event base 的循环无法修改，每次QueueInLoop
  // 都需要唤醒，多次唤醒会触发几次可读事件回调 ？？？
  //  if (fake_pending_wake_ && fake_wake_count_ < 100) {
  //    ++fake_wake_count_;
  //  } else {
  //    fake_wake_count_ = 0;
  //  }
  if (empty) return Wakeup();
  return 0;
}

TimerId UeventLoop::RunAt(base::Timestamp time, TimerCb cb) {
  return timer_queue_ptr_->AddTimer(std::move(cb), time, 0.0);
}

TimerId UeventLoop::RunAfter(double delay, TimerCb cb) {
  Timestamp time(base::addTime(Timestamp::now(), delay));
  return RunAt(time, std::move(cb));
}

TimerId UeventLoop::RunEvery(double interval, TimerCb cb) {
  Timestamp time(base::addTime(Timestamp::now(), interval));
  return timer_queue_ptr_->AddTimer(std::move(cb), time, interval);
}
void UeventLoop::CancelTimer(const TimerId& timer_id) {
  timer_queue_ptr_->Cancel(timer_id);
}

//===================================ConnectionUevent==========================

ConnectionUevent::ConnectionUevent(UeventLoop* loop, int fd, int64_t conn_id,
                                   const std::string& name,
                                   const UsockAddress& peer_addr)
    : loop_(loop),
      fd_(fd),
      conn_id_(conn_id),
      name_(name),
      state_(kDisconnected),
      peer_addr_(peer_addr) {
  if (fd_ == -1) {
    state_ = kConnecting;
  }
}

int ConnectionUevent::SetTcpNoDelay(bool on) {
  assert(fd_ != -1);  // 连接必须已经建立成功
  int optval = on ? 1 : 0;
  int ret = ::setsockopt(fd_, IPPROTO_TCP, TCP_NODELAY, &optval,
                         static_cast<socklen_t>(sizeof optval));
  if (ret == -1) {
    LOG_WARN << "set tcp nodelay return false, maybe unix socket, conn_id:"
             << conn_id_ << " name:" << name_
             << " peer address:" << peer_addr_.ToString();
  }
  return ret;
}

//===================================ListenerUevent============================

ListenerUevent::ListenerUevent(UeventLoop* loop,
                               const UsockAddress& listen_addr,
                               const std::string& name, const Option& option)
    : loop_(loop),
      listen_addr_(listen_addr),
      name_(name),
      next_conn_id_(0),
      option_(option),
      thread_pool_(new UeventLoopThreadPool(loop, name_)),
      // aio_pool_(new AioLoopThreadPool("aio")),
      connection_success_cb_(DefaultConnectionSuccessCb),
      connection_closed_cb_(DefaultConnectionClosedCb),
      message_read_cb_(DefaultMessageReadCb),
      message_write_cb_(DefaultMessageWriteCb),
      started_(false) {}

void ListenerUevent::RemoveConnection(const ConnectionUeventPtr& conn) {
  loop_->RunInLoop(
      std::bind(&ListenerUevent::RemoveConnectionInLoop, this, conn));
}

void ListenerUevent::RemoveConnectionInLoop(const ConnectionUeventPtr& conn) {
  loop_->AssertInLoopThread();
  auto it = connections_.find(conn->GetId());
  if (it == connections_.end()) {
    LOG_INFO << "Invalid Connection, conn id:" << conn->GetId()
             << " peer addr:" << conn->GetPeerAddress().ToString();
    return;
  }
  LOG_DEBUG << "RemoveConnectionInLoop, listener name:" << name_
            << "connection name:" << it->second->GetName()
            << " connection id:" << it->second->GetId();
  // 必须放在ForceClose之前,否则在同一个线程中时直接进入ClosedCb,
  // 会又触发remove, 会触发core
  connections_.erase(it);
  conn->ForceClose();
}

void ListenerUevent::RemoveConnectionByName(const std::string& name) {
  loop_->RunInLoop(
      std::bind(&ListenerUevent::RemoveConnectionByNameInLoop, this, name));
}

void ListenerUevent::RemoveConnectionByNameInLoop(const std::string& name) {
  for (auto it = connections_.begin(); it != connections_.end();) {
    ConnectionUeventPtr conn = it->second;
    if (it->second->GetName() == name) {
      LOG_DEBUG << "RemoveConnectionInLoop, listener name:" << name_
                << "connection name:" << conn->GetName()
                << " connection id:" << conn->GetId();
      connections_.erase(it++);
      conn->ForceClose();
    } else {
      it++;
    }
  }
}

void ListenerUevent::SetThreadNum(int num) {
  assert(num >= 0);
  thread_pool_->set_thread_num(num);
  // if (aio_pool_) aio_pool_->set_thread_num(num);
}

void ListenerUevent::Start() {
  started_ = true;
  thread_pool_->Start(create_loop_handle_cb_, thread_init_cb_);
  loop_->Start();
}

void ListenerUevent::StartServer() {
  started_ = true;
  loop_->Start();
}

void ListenerUevent::StartPrimaryLoop() {
  thread_pool_->Start(create_loop_handle_cb_, thread_init_cb_);
}

// void ListenerUevent::StartCrossAioLoop() {
//   aio_pool_->set_name("aio");
//   aio_pool_->Start();
// }

std::vector<UeventLoop*> ListenerUevent::GetPrimayLoops() {
  return thread_pool_->GetAllLoops();
}

// std::vector<UeventLoop*> ListenerUevent::GetAioLoops() {
//   return aio_pool_->loops();
// }

// void ListenerUevent::BindNetLoopEqCrossAioLoop(const
// std::vector<AioUeventPtr>& aio_uevents) {
//   std::vector<UeventLoop*> primary_loops = thread_pool_->GetAllLoops();
//   std::vector<UeventLoop*> aio_loops = aio_pool_->loops();
//   assert(primary_loops.size() == aio_loops.size());
//   assert(primary_loops.size() == aio_uevents.size());
//   for (size_t i = 0; i < aio_loops.size(); ++i) {
//     UeventLoop *aio_loop = aio_loops[i];
//     aio_uevents[i]->BindWithLoop(aio_loop);
//     aio_uevents[i]->set_cross_loop(primary_loops[i]);
//     primary_loops[i]->set_aio_ptr(aio_uevents[i]);
//   }
// }

//===============================ConnectorUevent===============================

ConnectorUevent::ConnectorUevent(UeventLoop* loop,
                                 const UsockAddress& peer_addr,
                                 const std::string& name)
    : loop_(loop),
      peer_addr_(peer_addr),
      name_(name),
      conn_(NULL),
      connection_success_cb_(DefaultConnectionSuccessCb),
      connection_closed_cb_(DefaultConnectionClosedCb),
      message_read_cb_(DefaultMessageReadCb),
      message_write_cb_(DefaultMessageWriteCb) {}

int ConnectorUevent::ReConnect() {
  DestroyConnection();
  Connect();
  return 0;
}

void ConnectorUevent::DestroyConnection() {
  if (!conn_) {
    LOG_DEBUG << "connection has been destroyed";
    return;
  }
  LOG_DEBUG << "connection will be destroy, peer addr:"
            << peer_addr_.ToString();
  // 主动关闭连接时,需要关闭，并触发回调.
  // 如果是被动关闭，回调已经触发过了，只用释放conn就行
  if (conn_->IsClosed() == false) {
    conn_->ForceClose();
  }
  // 释放这个conn,当ForceClose结束后，引用归0, conn会析构
  conn_.reset();
}

// 判断是否有可用的连接，没有会尝试新建一次
bool ConnectorUevent::HasAvailableConnection() {
  if (!conn_) {  // conn_ 未初始化，创建ctor未调用Connect
    Connect();
  } else if (conn_->IsClosed() == true) {  // 已经关闭需要重连
    ReConnect();
  } else {  // 有正常的连接
    return true;
  }
  if (conn_->IsClosed() == true) {  // 任然是关闭状态的连接
    return false;
  }
  return true;
}

}  // namespace uevent
