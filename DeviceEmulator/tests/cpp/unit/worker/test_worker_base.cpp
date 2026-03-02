#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "scheduler/worker_base.h"

// Simple latch (C++17 doesn't have std::latch)
class SimpleLatch {
 public:
  explicit SimpleLatch(int count) : count_(count) {}
  void CountDown() {
    std::lock_guard<std::mutex> lk(mu_);
    if (--count_ <= 0) cv_.notify_all();
  }
  void Wait() {
    std::unique_lock<std::mutex> lk(mu_);
    cv_.wait(lk, [this] { return count_ <= 0; });
  }

 private:
  std::mutex mu_;
  std::condition_variable cv_;
  int count_;
};

class WorkerBaseTest : public ::testing::Test {
 protected:
  void SetUp() override {
    worker_ = std::make_unique<WorkerBase>();
    worker_->Start();
  }

  void TearDown() override {
    if (worker_) {
      worker_->Stop();
    }
  }

  std::unique_ptr<WorkerBase> worker_;
};

// 1. SingleTask_HandleWait
TEST_F(WorkerBaseTest, SingleTask_HandleWait) {
  std::atomic<bool> executed{false};
  auto handle = worker_->AddTask("t1", [&] { executed = true; });
  handle->Wait();
  EXPECT_TRUE(executed.load());
  EXPECT_TRUE(handle->IsComplete());
}

// 2. SingleTask_Callback
TEST_F(WorkerBaseTest, SingleTask_Callback) {
  std::atomic<int> cb_count{0};
  std::string cb_id;
  std::mutex cb_mu;

  auto handle = worker_->AddTask(
      "t1", [] {},
      [&](const std::string& id) {
        std::lock_guard<std::mutex> lk(cb_mu);
        cb_count++;
        cb_id = id;
      });

  handle->Wait();
  // Give callback a moment to fire (it runs after state
  // completion, outside locks)
  std::this_thread::sleep_for(std::chrono::milliseconds(50));

  std::lock_guard<std::mutex> lk(cb_mu);
  EXPECT_EQ(cb_count.load(), 1);
  EXPECT_EQ(cb_id, "t1");
}

// 3. OnComplete_BeforeCompletion
TEST_F(WorkerBaseTest, OnComplete_BeforeCompletion) {
  SimpleLatch latch(1);
  std::atomic<bool> cb_fired{false};

  auto handle =
      worker_->AddTask("t1", [&] { latch.Wait(); });

  // Register callback before task completes
  handle->OnComplete([&](const std::string& id) {
    EXPECT_EQ(id, "t1");
    cb_fired = true;
  });

  EXPECT_FALSE(cb_fired.load());
  latch.CountDown();
  handle->Wait();
  // Give callback time to fire
  std::this_thread::sleep_for(std::chrono::milliseconds(50));
  EXPECT_TRUE(cb_fired.load());
}

// 4. OnComplete_AfterCompletion
TEST_F(WorkerBaseTest, OnComplete_AfterCompletion) {
  auto handle =
      worker_->AddTask("t1", [] {});
  handle->Wait();

  std::atomic<bool> cb_fired{false};
  handle->OnComplete([&](const std::string& id) {
    EXPECT_EQ(id, "t1");
    cb_fired = true;
  });

  EXPECT_TRUE(cb_fired.load());
}

// 5. IsComplete_Poll
TEST_F(WorkerBaseTest, IsComplete_Poll) {
  SimpleLatch latch(1);
  auto handle =
      worker_->AddTask("t1", [&] { latch.Wait(); });

  EXPECT_FALSE(handle->IsComplete());
  latch.CountDown();
  handle->Wait();
  EXPECT_TRUE(handle->IsComplete());
}

// 6. MultipleTasksSerial
TEST_F(WorkerBaseTest, MultipleTasksSerial) {
  const int N = 50;
  std::atomic<int> counter{0};

  for (int i = 0; i < N; i++) {
    worker_->AddTask(
        "t" + std::to_string(i), [&] { counter++; });
  }
  worker_->WaitTaskDone();
  EXPECT_EQ(counter.load(), N);
}

// 7. ConcurrentProducers
TEST_F(WorkerBaseTest, ConcurrentProducers) {
  const int M = 8;   // producer threads
  const int N = 50;  // tasks per producer
  std::atomic<int> counter{0};
  std::vector<std::thread> producers;

  for (int m = 0; m < M; m++) {
    producers.emplace_back([&, m] {
      for (int n = 0; n < N; n++) {
        std::string id = "p" + std::to_string(m) + "_t" +
                         std::to_string(n);
        worker_->AddTask(id, [&] { counter++; });
      }
    });
  }

  for (auto& t : producers) t.join();
  worker_->WaitTaskDone();
  EXPECT_EQ(counter.load(), M * N);
}

// 8. WorkerPool_RoundRobin
TEST(WorkerPoolTest, RoundRobin) {
  WorkerPool pool(4);
  const int N = 100;
  std::atomic<int> counter{0};
  std::vector<TaskHandle> handles;

  for (int i = 0; i < N; i++) {
    handles.push_back(pool.AddTask(
        "t" + std::to_string(i), [&] { counter++; }));
  }
  pool.WaitTaskDone();
  EXPECT_EQ(counter.load(), N);
  for (auto& h : handles) {
    EXPECT_TRUE(h->IsComplete());
  }
}

// 9. WorkerPool_HandleWait
TEST(WorkerPoolTest, HandleWait) {
  WorkerPool pool(4);
  const int N = 20;
  std::vector<TaskHandle> handles;

  std::atomic<int> counter{0};
  for (int i = 0; i < N; i++) {
    handles.push_back(pool.AddTask(
        "t" + std::to_string(i), [&] { counter++; }));
  }

  for (auto& h : handles) {
    h->Wait();
    EXPECT_TRUE(h->IsComplete());
  }
  EXPECT_EQ(counter.load(), N);
}

// 10. WorkerPool_ConcurrentAddTask
TEST(WorkerPoolTest, ConcurrentAddTask) {
  WorkerPool pool(4);
  const int M = 8;
  const int N = 50;
  std::atomic<int> counter{0};
  std::vector<std::thread> producers;

  for (int m = 0; m < M; m++) {
    producers.emplace_back([&, m] {
      for (int n = 0; n < N; n++) {
        std::string id = "p" + std::to_string(m) + "_t" +
                         std::to_string(n);
        pool.AddTask(id, [&] { counter++; });
      }
    });
  }

  for (auto& t : producers) t.join();
  pool.WaitTaskDone();
  EXPECT_EQ(counter.load(), M * N);
}

// 11. StopDuringExecution
TEST(WorkerBaseStopTest, StopDuringExecution) {
  auto worker = std::make_unique<WorkerBase>();
  worker->Start();

  SimpleLatch task_started(1);
  SimpleLatch allow_finish(1);

  worker->AddTask("long_task", [&] {
    task_started.CountDown();
    allow_finish.Wait();
  });

  task_started.Wait();

  // Stop from another thread with timeout detection
  std::atomic<bool> stopped{false};
  std::thread stopper([&] {
    allow_finish.CountDown();
    worker->Stop();
    stopped = true;
  });

  // Wait up to 5 seconds for clean shutdown
  auto deadline = std::chrono::steady_clock::now() +
                  std::chrono::seconds(5);
  while (!stopped.load() &&
         std::chrono::steady_clock::now() < deadline) {
    std::this_thread::sleep_for(
        std::chrono::milliseconds(10));
  }
  EXPECT_TRUE(stopped.load()) << "Stop() deadlocked";
  stopper.join();
  worker.reset();  // prevent TearDown double-stop
}

// 12. WaitTaskDone_ById
TEST_F(WorkerBaseTest, WaitTaskDone_ById) {
  SimpleLatch latch1(1);
  SimpleLatch latch2(1);
  std::atomic<bool> task1_done{false};
  std::atomic<bool> task2_done{false};

  worker_->AddTask("id_A", [&] {
    latch1.Wait();
    task1_done = true;
  });
  worker_->AddTask("id_B", [&] {
    latch2.Wait();
    task2_done = true;
  });

  // Release task A, wait specifically for it
  latch1.CountDown();
  worker_->WaitTaskDone("id_A");
  EXPECT_TRUE(task1_done.load());

  // Task B should still be pending or running
  // (single worker processes sequentially, so B may
  // not have started yet)

  // Release task B and wait
  latch2.CountDown();
  worker_->WaitTaskDone("id_B");
  EXPECT_TRUE(task2_done.load());
}
