#include "ueventloop_thread_pool.h"

#include <assert.h>
#include <stdio.h>
#include <unistd.h>

#include "base/logging.h"
#include "uevent.h"
#include "ueventloop_thread.h"

using base::implicit_cast;

namespace uevent {

UeventLoopThreadPool::UeventLoopThreadPool(UeventLoop* base_loop,
                                           const std::string& name)
    : base_loop_(base_loop),
      name_(name),
      started_(false),
      thread_num_(0),
      next_(0) {}

UeventLoopThreadPool::~UeventLoopThreadPool() {}

void UeventLoopThreadPool::Start(const CreateLoopHandleCb& cb1,
                                 const ThreadInitCb& cb2) {
  // assert(!started_);
  base_loop_->AssertInLoopThread();
  started_ = true;
  for (int i = 0; i < thread_num_; ++i) {
    char buf[name_.size() + 32];
    snprintf(buf, sizeof buf, "%s%d", name_.c_str(), i);
    UeventLoopThread* t = new UeventLoopThread(cb1, cb2, buf);
    threads_.push_back(std::unique_ptr<UeventLoopThread>(t));
    loops_.push_back(t->StartLoop());
  }
  if (thread_num_ == 0 && cb2) {
    cb2(base_loop_);
  }
  LOG_INFO << "thread_pool loop size: " << loops_.size();
}

UeventLoop* UeventLoopThreadPool::GetNextLoop() {
  base_loop_->AssertInLoopThread();
  UeventLoop* loop = base_loop_;

  if (!loops_.empty()) {
    // round-robin
    loop = loops_[next_];
    ++next_;
    if (implicit_cast<size_t>(next_) >= loops_.size()) {
      next_ = 0;
    }
  }
  return loop;
}

UeventLoop* UeventLoopThreadPool::GetEmptyLoop() {
  base_loop_->AssertInLoopThread();
  assert(started_);
  UeventLoop* loop = base_loop_;
  for (uint32_t i = 0; i < loops_.size(); i++) {
    if (loops_[i]->GetLoopHandle()->GetRefs() == 0) {
      LOG_INFO << "loop index: " << i << " will be used";
      return loops_[i];
    }
  }
  if (!loops_.empty()) {
    // 有线程池且没有找到可用的loop返回 NULL
    return NULL;
  } else {  // 没有线程池需要检查base_loop是否为空
    if (loop->GetLoopHandle()->GetRefs() == 0) {
      return loop;
    } else {
      return NULL;
    }
  }
}
// TODO
UeventLoop* UeventLoopThreadPool::GetLightestLoop() {
  base_loop_->AssertInLoopThread();
  assert(started_);
  UeventLoop* loop = base_loop_;

  if (!loops_.empty()) {
    // round-robin
    loop = loops_[next_];
    ++next_;
    if (implicit_cast<size_t>(next_) >= loops_.size()) {
      next_ = 0;
    }
  }
  return loop;
}

UeventLoop* UeventLoopThreadPool::GetLoopForHash(size_t hashCode) {
  base_loop_->AssertInLoopThread();
  UeventLoop* loop = base_loop_;

  if (!loops_.empty()) {
    loop = loops_[hashCode % loops_.size()];
  }
  return loop;
}

std::vector<UeventLoop*> UeventLoopThreadPool::GetAllLoops() {
  // base_loop_->AssertInLoopThread();
  // assert(started_);
  if (thread_num_ == 0) {
    return std::vector<UeventLoop*>(1, base_loop_);
  } else {
    while ((int)loops_.size() < thread_num_) {
      usleep(100);
    }
    return loops_;
  }
}
}  // namespace uevent
