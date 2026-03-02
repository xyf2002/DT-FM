#pragma once

#include <atomic>
#include <condition_variable>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_set>

#include "utils/noncopyable.h"

// Async completion callback type
using TaskCallback = std::function<void(const std::string& task_id)>;

// Per-task completion state, shared between producer and consumer
struct TaskState {
  std::mutex mutex;
  std::condition_variable cv;
  bool completed = false;
  TaskCallback callback;
  std::string task_id;

  void Wait() {
    std::unique_lock<std::mutex> lk(mutex);
    cv.wait(lk, [this] { return completed; });
  }

  bool IsComplete() {
    std::lock_guard<std::mutex> lk(mutex);
    return completed;
  }

  void OnComplete(TaskCallback cb) {
    bool call_now = false;
    std::string id;
    {
      std::lock_guard<std::mutex> lk(mutex);
      if (completed) {
        call_now = true;
        id = task_id;
      } else {
        callback = std::move(cb);
        return;
      }
    }
    if (call_now && cb) {
      cb(id);
    }
  }
};

using TaskHandle = std::shared_ptr<TaskState>;

class WorkerBase : public noncopyable {
 public:
  typedef std::function<void()> Task;

 public:
  virtual ~WorkerBase() = default;

  virtual void Start() {
    worker_ = std::thread([this] { Run(); });
  }

  virtual void Stop() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      quit_ = true;
    }
    cv_.notify_all();
    worker_.join();
  }

  virtual TaskHandle AddTask(const std::string& task_id, Task&& t,
                             TaskCallback callback = nullptr) {
    auto state = std::make_shared<TaskState>();
    state->task_id = task_id;
    if (callback) {
      state->callback = std::move(callback);
    }
    {
      std::lock_guard<std::mutex> lock(mutex_);
      tasks_.emplace_back(task_id, std::move(t), state);
      task_count_++;
      task_ids_.insert(task_id);
    }
    cv_.notify_all();
    return state;
  }

  virtual void Run() {
    while (true) {
      std::unique_lock<std::mutex> lock(mutex_);
      cv_.wait(lock, [this] { return !tasks_.empty() || quit_; });
      if (quit_) {
        break;
      }

      auto [task_id, task, state] = std::move(tasks_.front());
      tasks_.pop_front();
      lock.unlock();

      task();
      task_count_--;

      // Mark TaskState completed and capture callback
      TaskCallback cb;
      std::string id;
      if (state) {
        {
          std::lock_guard<std::mutex> lk(state->mutex);
          state->completed = true;
          cb = std::move(state->callback);
          id = state->task_id;
        }
        state->cv.notify_all();
      }

      lock.lock();
      task_ids_.erase(task_id);
      lock.unlock();

      cv_.notify_all();

      // Invoke callback outside all locks
      if (cb) {
        cb(id);
      }
    }
  }

  int GetTaskCount() const {
    return task_count_.load(std::memory_order_relaxed);
  }

  void WaitTaskDone() {
    std::unique_lock<std::mutex> lock(mutex_);
    cv_.wait(lock,
             [this] { return (tasks_.empty() && task_count_ == 0) || quit_; });
  }

  void WaitTaskDone(const std::string& task_id) {
    std::unique_lock<std::mutex> lock(mutex_);
    cv_.wait(lock, [this, &task_id] {
      return (task_ids_.count(task_id) == 0) || quit_;
    });
  }

 protected:
  std::mutex mutex_;
  std::condition_variable cv_;
  bool quit_ = false;
  std::thread worker_;
  std::deque<std::tuple<std::string, Task, TaskHandle>> tasks_;
  std::unordered_set<std::string> task_ids_;
  std::atomic_int task_count_{0};
};

class WorkerPool : public noncopyable {
 public:
  WorkerPool(int num_workers) : num_workers_(num_workers) {
    for (int i = 0; i < num_workers_; i++) {
      auto worker = std::make_unique<WorkerBase>();
      worker->Start();
      workers_.emplace_back(std::move(worker));
    }
  }

  ~WorkerPool() {
    for (auto& worker : workers_) {
      worker->Stop();
    }
  }

  TaskHandle AddTask(const std::string& task_id, WorkerBase::Task&& t,
                     TaskCallback callback = nullptr) {
    auto& worker = workers_[next_worker_];
    auto handle = worker->AddTask(task_id, std::move(t), std::move(callback));
    next_worker_ = (next_worker_ + 1) % num_workers_;
    return handle;
  }

  void WaitTaskDone() {
    for (auto& worker : workers_) {
      worker->WaitTaskDone();
    }
  }

  void WaitTaskDone(const std::string& task_id) {
    for (auto& worker : workers_) {
      worker->WaitTaskDone(task_id);
    }
  }

 private:
  int num_workers_;
  int next_worker_ = 0;
  std::vector<std::unique_ptr<WorkerBase>> workers_;
};
