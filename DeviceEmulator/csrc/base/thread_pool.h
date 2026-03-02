// Use of this source code is governed by a BSD-style license
// that can be found in the License file.
//
// Author: Shuo Chen (chenshuo at chenshuo dot com)

#ifndef MUDUO_BASE_THREADPOOL_H
#define MUDUO_BASE_THREADPOOL_H

#include <deque>
#include <vector>

#include "condition.h"
#include "mutex.h"
#include "thread.h"
#include "types.h"

namespace base {

class ThreadPool : noncopyable {
 public:
  typedef std::function<void()> Task;

  explicit ThreadPool(const std::string& nameArg = std::string("ThreadPool"));
  ~ThreadPool();

  // Must be called before start().
  void setMaxQueueSize(int maxSize) { maxQueueSize_ = maxSize; }
  void setThreadInitCallback(const Task& cb) { threadInitCallback_ = cb; }

  void start(int numThreads);
  void stop();

  const std::string& name() const { return name_; }

  size_t queueSize() const;

  // Could block if maxQueueSize > 0
  void run(Task f);

 private:
  bool isFull() const;
  void runInThread();
  Task take();

  mutable MutexLock mutex_;
  Condition notEmpty_;
  Condition notFull_;
  std::string name_;
  Task threadInitCallback_;
  std::vector<std::unique_ptr<base::Thread>> threads_;
  std::deque<Task> queue_;
  size_t maxQueueSize_;
  bool running_;
};

}  // namespace base

#endif
