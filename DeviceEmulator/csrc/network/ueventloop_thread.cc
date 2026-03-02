#include "ueventloop_thread.h"

#include "eventloop_libevent.h"
#include "uevent.h"

namespace uevent {

UeventLoopThread::UeventLoopThread(const CreateLoopHandleCb& cb1,
                                   const ThreadInitCb& cb2,
                                   const std::string& name)
    : name_(name),
      loop_(NULL),
      thread_(std::bind(&UeventLoopThread::ThreadFunc, this), name),
      mutex_(),
      cond_(mutex_),
      create_loop_handle_cb_(cb1),
      thread_init_cb_(cb2) {}

UeventLoopThread::~UeventLoopThread() {
  if (loop_ != NULL) {
    // not 100% race-free, eg. ThreadFunc could be running callback_.
    // still a tiny chance to call destructed object, if ThreadFunc exits just
    // now. but when EventLoopThread destructs, usually programming is exiting
    // anyway.
    loop_->Quit();
    thread_.join();
  }
}

UeventLoop* UeventLoopThread::StartLoop() {
  assert(!thread_.started());
  thread_.start();
  {
    base::MutexLockGuard lock(mutex_);
    while (loop_ == NULL) {
      cond_.wait();
    }
  }
  return loop_;
}

void UeventLoopThread::ThreadFunc() {
  // 将来使用RDMA 这里进行替换即可
  UeventLoop* loop = new EventLoopLibevent(name_, create_loop_handle_cb_);
  if (thread_init_cb_) {
    thread_init_cb_(loop);
  }
  {
    base::MutexLockGuard lock(mutex_);
    loop_ = loop;
    cond_.notify();
  }
  loop_->Start();  // 开始循环
  loop_ = NULL;
}

}  // namespace uevent
