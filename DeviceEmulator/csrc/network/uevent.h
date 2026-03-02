#ifndef UEVENT_H
#define UEVENT_H

#include <functional>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "base/condition.h"
#include "base/current_thread.h"
#include "base/mutex.h"
#include "base/timestamp.h"
#include "callbacks.h"
#include "forward_desc.h"
#include "loop_handle.h"
#include "protobuf_dispatcher.h"
#include "timer_id.h"
#include "uevent.h"
#include "usock_address.h"

namespace uevent {

// class AioUevent;
class TimerQueueUevent;
// class ProtobufDispatcher;
class LoopHandle;

int CreateEventfd();

struct Option {
  enum LoopSelectStrategy {
    kRoundRobin,
    kEmptyOne,
    kLightestOne,
  };

  Option() : loop_strategy(kRoundRobin), reuse_port(true) {}

  LoopSelectStrategy loop_strategy;
  bool reuse_port;
};

class UeventLoopBase {
 public:
  typedef std::function<void()> Functor;

  UeventLoopBase();
  virtual ~UeventLoopBase();

  bool IsInLoopThread() const {
    return thread_id_ == base::CurrentThread::tid();
  }

  virtual void Start() = 0;
  virtual void Quit() = 0;

  virtual int QueueInLoop(Functor cb) = 0;
  virtual void RunInLoop(Functor cb) = 0;

  virtual int QueueInLoop(void* arg) = 0;
  virtual void RunInLoop(void* arg) = 0;

  virtual int Wakeup() = 0;

  void AssertInLoopThread();

 protected:
  bool started_;
  const pid_t thread_id_;
};

//================================UeventLoop===================================
class UeventLoop : public UeventLoopBase {
 public:
  typedef std::function<void()> Functor;

  explicit UeventLoop(const std::string& thread_name);
  virtual ~UeventLoop();

  inline LoopHandle* GetLoopHandle() { return loop_handle_; }
  const std::string& thread_name() const { return thread_name_; }

  virtual int QueueInLoop(Functor cb);
  virtual void RunInLoop(Functor cb);

  virtual int QueueInLoop(void* arg);
  virtual void RunInLoop(void* arg);

  virtual int Wakeup();

  //@brief Runs callback at 'time'.
  // Safe to call from other threads.
  TimerId RunAt(base::Timestamp time, TimerCb cb);
  //@brief Runs callback after @c delay seconds.
  // Safe to call from other threads.
  TimerId RunAfter(double delay, TimerCb cb);
  //@brief Runs callback every @c interval seconds.
  // Safe to call from other threads.
  TimerId RunEvery(double interval, TimerCb cb);
  //@brief Cancels the timer.
  // Safe to call from other threads.
  void CancelTimer(const TimerId& timer_id);

  // inline const std::shared_ptr<AioUevent>& GetAio() const {
  //   return aio_ptr_;
  // }
  // inline const std::shared_ptr<ProtobufDispatcher>& GetProtobufDispatcher() {
  //   return protobuf_dispatcher_ptr_;
  // }

  virtual void Start() = 0;
  virtual void Quit() = 0;

  // const std::shared_ptr<AioUevent>& aio_ptr() const {
  //   return aio_ptr_;
  // }

  // void set_aio_ptr(const std::shared_ptr<AioUevent>& aio_ptr) {
  //   aio_ptr_ = aio_ptr;
  // }

  LoopArgFunc loog_arg_func() const { return loog_arg_func_; }

  void set_loog_arg_func(LoopArgFunc loop_arg_func) {
    loog_arg_func_ = loop_arg_func;
  }

 protected:
  int WakeupReadCb();  // waked up
  virtual void DoPendingFunctors();

  std::string thread_name_;
  LoopArgFunc loog_arg_func_;
  int wakeup_fd_;
  int timer_fd_;
  // std::shared_ptr<AioUevent> aio_ptr_;
  std::unique_ptr<TimerQueueUevent> timer_queue_ptr_;
  // std::shared_ptr<ProtobufDispatcher> protobuf_dispatcher_ptr_;
  mutable base::MutexLock mutex_;
  std::vector<Functor> pending_functors_;  // @GuardedBy mutex_
  std::vector<void*> pending_tasks_;       // @GuardedBy mutex_
  LoopHandle* loop_handle_;
};

//==========================ConnectionUevent===================================
class ConnectionUevent : public std::enable_shared_from_this<ConnectionUevent> {
 public:
  ConnectionUevent(UeventLoop* loop, int fd, int64_t conn_id,
                   const std::string& name, const UsockAddress& peer_addr);
  enum StateE { kDisconnected, kConnecting, kConnected, kDisconnecting };

  int SetTcpNoDelay(bool on);

  virtual int Init() = 0;
  virtual void SetFd() = 0;

  virtual ~ConnectionUevent() {}

  //@brief 获取可读数据的长度
  virtual ssize_t ReadableLength() = 0;

  //@brief 向发送缓冲区发送数据
  //@param data 要发送的数据指针
  //@param data_len 要发送的数据长度
  //@return 0 on success, -1 on failure.
  virtual int SendData(const void* data, size_t data_len) = 0;

  // Zero-copy send: data is referenced, not copied. cleanup_cb is called when
  // libevent is done with the data. Caller must ensure data lives until
  // cleanup.
  virtual int SendDataZeroCopy(const void* data, size_t data_len,
                               void (*cleanup_cb)(const void*, size_t, void*),
                               void* cleanup_arg) = 0;

  // Zero-copy receive: returns a contiguous pointer to data in the input
  // buffer. Data remains in buffer until DrainData() is called. Returns nullptr
  // on failure.
  virtual unsigned char* PullupData(size_t len) = 0;

  //@brief 从接收缓冲区拷贝数据，数据仍保留在缓冲区中
  //@param data 接收数据的地址
  //@param data_len 接收数据的长度
  //@return 成功返回实际接收数据的长度，失败返回 -1
  virtual int ReceiveData(void* data, size_t data_len) = 0;

  //@brief 从接收缓冲区头部开始移除指定长度数据
  //@param len 指定移除的数据长度
  //@return 成功返回实际移除的数据的长度，失败返回 -1
  virtual int DrainData(size_t len) = 0;

  //@brief 从接收缓冲区拷贝数据，并且从缓冲区中移除
  //@param data 接收数据的地址
  //@param data_len 接收数据的长度
  //@return 成功返回实际接收数据的长度，失败返回 -1
  virtual int RemoveData(void* data, size_t data_len) = 0;

  //@brief 使能连接并且触发连接成功回调，用于通过accept
  //       新建连接的时候调用
  virtual void ConnectionEstablished() = 0;

  //@brief 强制关闭连接，触发连接关闭回调 thread safe
  virtual void ForceClose() = 0;
  //@brief 强制关闭连接，触发连接关闭回调，not thread safe
  virtual void ForceCloseInLoop() = 0;

  //@brief 使能事件监听，not thread safe
  virtual void ConnectionEnable() = 0;

  //@brief 内部的通信渠道已经关闭，不会有回调再触发
  // 对于libevent 就是bufferevent已经释放
  virtual bool IsClosed() = 0;

  inline const int64_t GetId() const { return conn_id_; }

  inline const UsockAddress& GetPeerAddress() const { return peer_addr_; }
  inline const std::string& GetName() const { return name_; }

  inline void SetName(const std::string& name) { name_ = name; }

  inline UeventLoop* GetLoop() { return loop_; }

  inline void SetState(StateE state) { state_ = state; }
  inline int GetFd() const { return fd_; }
  void SetConnectionSuccessCb(const ConnectionSuccessCb& cb) {
    connection_success_cb_ = cb;
  }
  void SetConnectionClosedCb(const ConnectionClosedCb& cb) {
    connection_closed_cb_ = cb;
  }
  void SetMessageReadCb(const MessageReadCb& cb) { message_read_cb_ = cb; }
  void SetMessageWriteCb(const MessageWriteCb& cb) { message_write_cb_ = cb; }

  ConnectionSuccessCb connection_success_cb_;
  ConnectionClosedCb connection_closed_cb_;
  MessageReadCb message_read_cb_;
  MessageWriteCb message_write_cb_;

 protected:
  UeventLoop* loop_;
  int fd_;
  int64_t conn_id_;
  std::string name_;
  StateE state_;
  UsockAddress peer_addr_;
};

//=========================ListenerUevent======================================

class UeventLoopThreadPool;
class AioLoopThreadPool;

class ListenerUevent {
 public:
  ListenerUevent(UeventLoop* loop, const UsockAddress& listen_addr,
                 const std::string& name, const Option& option);
  virtual ~ListenerUevent() {}
  void Start();
  void StartServer();
  void StartPrimaryLoop();
  // void StartCrossAioLoop();
  // void BindNetLoopEqCrossAioLoop(const std::vector<AioUeventPtr>&
  // aio_uevents);

  std::vector<UeventLoop*> GetPrimayLoops();
  // std::vector<UeventLoop*> GetAioLoops();

  // Thread safe.
  void RemoveConnection(const ConnectionUeventPtr& conn);
  void RemoveConnectionInLoop(const ConnectionUeventPtr& conn);
  void RemoveConnectionByName(const std::string& name);
  void RemoveConnectionByNameInLoop(const std::string& name);

  void SetConnectionSuccessCb(const ConnectionSuccessCb& cb) {
    connection_success_cb_ = cb;
  }
  void SetConnectionClosedCb(const ConnectionClosedCb& cb) {
    connection_closed_cb_ = cb;
  }
  void SetMessageReadCb(const MessageReadCb& cb) { message_read_cb_ = cb; }
  void SetMessageWriteCb(const MessageWriteCb& cb) { message_write_cb_ = cb; }
  const std::string& GetName() const { return name_; }
  UeventLoop* GetLoop() const { return loop_; }
  void SetCreateLoopHandleCb(CreateLoopHandleCb cb) {
    create_loop_handle_cb_ = cb;
  }
  std::shared_ptr<UeventLoopThreadPool> GetThreadPool() const {
    return thread_pool_;
  }
  /// Set the number of threads for handling input.
  /// Always accepts new connection in loop's thread.
  /// Must be called before @c start
  /// @param numThreads
  /// - 0 means all I/O in loop's thread, no thread will created.
  ///   this is the default value.
  /// - 1 means all I/O in another thread.
  /// - N means a thread pool with N threads, new connections
  ///   are assigned on a round-robin basis.
  void SetThreadNum(int thread_num);
  void SetThreadInitCb(const ThreadInitCb& cb) { thread_init_cb_ = cb; }
  std::shared_ptr<UeventLoopThreadPool> GetThreadPool() { return thread_pool_; }
  // std::shared_ptr<AioLoopThreadPool>& aio_pool() {
  //   return aio_pool_;
  // }
  // void set_aio_pool(const std::shared_ptr<AioLoopThreadPool>& aio_pool) {
  //   aio_pool_ = aio_pool;
  // }

 protected:
  typedef std::map<int64_t, ConnectionUeventPtr> ConnectionMap;

  UeventLoop* loop_;  // the acceptor loop
  UsockAddress listen_addr_;
  std::string name_;
  int64_t next_conn_id_;
  Option option_;
  std::shared_ptr<UeventLoopThreadPool> thread_pool_;
  // std::shared_ptr<AioLoopThreadPool> aio_pool_;

  ConnectionSuccessCb connection_success_cb_;
  ConnectionClosedCb connection_closed_cb_;
  MessageReadCb message_read_cb_;
  MessageWriteCb message_write_cb_;
  CreateLoopHandleCb create_loop_handle_cb_;
  ThreadInitCb thread_init_cb_;
  bool started_;
  ConnectionMap connections_;
};

//============================ConnectorUevent=======================

class ConnectorUevent {
 public:
  ConnectorUevent(UeventLoop* loop, const UsockAddress& peer_addr,
                  const std::string& name);
  virtual ~ConnectorUevent() {}
  virtual int Connect() = 0;
  int ReConnect();
  void DestroyConnection();
  bool HasAvailableConnection();
  inline UsockAddress GetPeerAddress() const { return peer_addr_; }
  inline const ConnectionUeventPtr& GetConnection() const { return conn_; }
  inline std::string GetName() const { return name_; }
  void SetConnectionSuccessCb(ConnectionSuccessCb cb) {
    connection_success_cb_ = cb;
  }
  void SetConnectionClosedCb(ConnectionClosedCb cb) {
    connection_closed_cb_ = cb;
  }
  void SetMessageReadCb(MessageReadCb cb) { message_read_cb_ = cb; }
  void SetMessageWriteCb(MessageWriteCb cb) { message_write_cb_ = cb; }

 protected:
  UeventLoop* loop_;
  UsockAddress peer_addr_;
  std::string name_;
  ConnectionUeventPtr conn_;
  ConnectionSuccessCb connection_success_cb_;
  ConnectionClosedCb connection_closed_cb_;
  MessageReadCb message_read_cb_;
  MessageWriteCb message_write_cb_;
};

}  // namespace uevent

#endif
