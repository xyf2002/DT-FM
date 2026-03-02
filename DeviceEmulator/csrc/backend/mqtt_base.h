#pragma once

#include <mosquitto.h>
#include <stdint.h>  // uint32_t

#include <atomic>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "common/generator.h"
#include "common/types_and_defs.h"
#include "redis_base.h"
#include "utils/logger.h"

#define MQTT_COMPUTE_TOPIC_REQ "/morphling/comp/req/"
#define MQTT_COMPUTE_TOPIC_RSP "/morphling/comp/rsp/"
// #define MQTT_TIMER_TOPIC_REQ "/morphling/timer/req/"
// #define MQTT_TIMER_TOPIC_RSP "/morphling/timer/rsp/"

class MQTTBase {
 public:
  struct PubTask {
    std::string topic;
    void* payload;
    int payloadlen;
  };

  void Publish(int idx, std::string& topic, void* payload, int payloadlen) {
    // mosquitto_publish(mosq_, NULL, topic.c_str(), payloadlen, payload, 0,
    // false);
    auto task = PubTask{topic, payload, payloadlen};
    // auto idx = publish_count_++ % num_mosq_;
    idx = idx % num_mosq_;
    {
      std::lock_guard<std::mutex> lock(pub_mutex_[idx]);
      // if (!pub_ready_[idx]) {
      //   pub_ready_[idx] = true;
      // }
      pub_queue_[idx].push_back(task);
    }
    pub_cv_[idx].notify_one();
    // auto iter = mosq_.begin();
    // std::advance(iter, (publish_count_++) % num_mosq_);
    // mosquitto_publish(iter->second, NULL, topic.c_str(), payloadlen, payload,
    // 0,
    //                   false);
    // // get current time til microseconds
    // auto now = CurrentTimeString();
    // fprintf(stderr, "%s Published message %s by %p \n", now.c_str(),
    //         topic.c_str(), payload);
  }

  virtual void OnMessage(struct mosquitto* mosq, void* obj,
                         const struct mosquitto_message* message) {}

  void OnPublish(struct mosquitto* mosq, void* obj, int mid) {
    pub_cb_count_--;
    // fprintf(stderr, "Published message %d\n", mid);
    if (pub_cb_count_ == 0) {
      LOG_DEBUG << "All messages published, clearing buffer";
      for (auto* ptr : pub_buffer_) {
        free(ptr);
      }
      pub_buffer_.clear();
    }

    // auto this_id = std::this_thread::get_id();
    // find mosq in mosq_
    // auto iter = mosq_.begin();
    // int idx = 0;
    // for (; iter != mosq_.end(); iter++, idx++) {
    //   if (iter->second == mosq) {
    //     break;
    //   }
    // }
    // {
    //   std::lock_guard<std::mutex> lock(pub_mutex_[idx]);
    //   pub_ready_[idx] = true;
    // }
    // pub_cv_[idx].notify_one();
  }

  explicit MQTTBase(const std::string& host = "localhost", int port = 1883,
                    bool clean_session = true, int keepalive = 60)
      : host_(host),
        port_(port),
        running_(true),
        clean_session_(clean_session),
        keepalive_(keepalive) {
    mosquitto_lib_init();

    num_mosq_ = std::stoi(GETENV("MORPHLING_NUM_CONN", "1"));
    LOG_DEBUG << "Number of connections: " << num_mosq_;
    pub_mutex_ = std::move(std::vector<std::mutex>(num_mosq_));
    pub_cv_ = std::move(std::vector<std::condition_variable>(num_mosq_));
    for (int i = 0; i < num_mosq_; i++) {
      pub_queue_.push_back(std::deque<PubTask>());
      pub_threads_.push_back(std::thread(&MQTTBase::RunPublishThread, this, i));

      // get tid of pub thread
      // pub_ready_[i] = false;
    }
  }

  ~MQTTBase() {
    // mosquitto_destroy(mosq_);
    for (auto m : mosq_) {
      mosquitto_destroy(m.second);
    }
    mosquitto_lib_cleanup();
  }

  void Start() {
    // mosquitto_connect(mosq_, host_.c_str(), port_, keepalive_);
    // mosquitto_loop_start(mosq_);
    for (auto m : mosq_) {
      mosquitto_connect(m.second, host_.c_str(), port_, keepalive_);
      mosquitto_loop_start(m.second);
    }
  }
  void Stop() {
    // mosquitto_disconnect(mosq_);
    // mosquitto_loop_stop(mosq_, true);
    running_ = false;
    for (auto m : mosq_) {
      mosquitto_disconnect(m.second);
      mosquitto_loop_stop(m.second, true);
    }
  }

 private:
  void RunPublishThread(int idx) {
    // auto this_id = std::this_thread::get_id();
    while (running_) {
      PubTask task;
      {
        std::unique_lock<std::mutex> lock(pub_mutex_[idx]);
        pub_cv_[idx].wait(lock, [this, idx] {
          return !pub_queue_[idx].empty();  // && pub_ready_[idx];
        });
        task = pub_queue_[idx].front();
        pub_queue_[idx].pop_front();
        // pub_ready_[idx] = false;
      }
      // LOG_DEBUG("Publishing message {} by {:p} size {}", task.topic,
      //           task.payload, task.payloadlen);
      // fprintf(stderr, "Publishing message %s, ptr %p, size %d\n",
      //         task.topic.c_str(), task.payload, task.payloadlen);
      int ret = mosquitto_publish(mosq_[idx], NULL, task.topic.c_str(),
                                  task.payloadlen, task.payload, 0, false);
      if (ret != MOSQ_ERR_SUCCESS) {
        LOG_ERROR << "Failed to publish message " << task.topic
                  << ", error code " << ret;
      }
    }
  }

 protected:
  std::unordered_map<uint32_t, struct mosquitto*> mosq_;
  // std::atomic_ullong publish_count_{0};
  // struct mosquitto* mosq_;
  int num_mosq_;
  std::string host_;
  int port_;
  bool clean_session_;
  int keepalive_;
  bool running_;

  std::atomic_ullong pub_count_{0};
  std::atomic_ullong pub_cb_count_{0};
  std::vector<void*> pub_buffer_;

  std::atomic_ullong rsp_cb_count_{0};

 private:
  std::vector<std::thread> pub_threads_;
  // std::unordered_map<int, bool> pub_ready_;
  std::vector<std::deque<PubTask>> pub_queue_;
  std::vector<std::condition_variable> pub_cv_;
  std::vector<std::mutex> pub_mutex_;
};
