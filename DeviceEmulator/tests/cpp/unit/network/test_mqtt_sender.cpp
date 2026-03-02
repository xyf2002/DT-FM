// test mqtt server performance

#include <mosquitto.h>
#include <sys/wait.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <iostream>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <vector>

// const int MSG_SIZE_SEND = 2097152 * 4;    // Message size for requests
// const int MSG_SIZE_RESP = 512 * 512 * 4;  // Message size for responses
const int MSG_SIZE_SEND = 1024 * 4;
const int MSG_SIZE_RESP = 1024 * 4;
// const int MSG_SIZE_SEND = 838 * 4;    // Message size for requests
// const int MSG_SIZE_RESP = 512 * 4;  // Message size for responses
#define TOPIC_REQUEST "/topic/request/"
#define TOPIC_RESPONSE "/topic/response/"
const int NUM_RECEIVERS = 64;  // Number of receiver processes
const int NUM_MOSQ = 10;       // Number of Mosquitto instances

#define MILLISECONDS_EPOCH                                 \
  std::chrono::duration_cast<std::chrono::milliseconds>(   \
      std::chrono::system_clock::now().time_since_epoch()) \
      .count()

class MQTTBase {
 public:
  struct PubTask {
    std::string topic;
    void* payload;
    int payloadlen;
  };

  void Publish(int idx, std::string& topic, void* payload, int payloadlen) {
    auto task = PubTask{topic, payload, payloadlen};
    idx = idx % num_mosq_;
    {
      std::lock_guard<std::mutex> lock(pub_mutex_[idx]);
      pub_queue_[idx].push_back(task);
    }
    pub_cv_[idx].notify_one();
  }

  virtual void OnMessage(struct mosquitto* mosq, void* obj,
                         const struct mosquitto_message* message) {}

  void OnPublish(struct mosquitto* mosq, void* obj, int mid) {
    pub_cb_count_--;
    // fprintf(stderr, "Published message %d\n", mid);
    if (pub_cb_count_ == 0) {
      for (auto* ptr : pub_buffer_) {
        free(ptr);
      }
      pub_buffer_.clear();
    }
  }

  explicit MQTTBase(const std::string& host = "localhost", int port = 1883,
                    bool clean_session = true, int keepalive = 60,
                    int num_mosq = 1)
      : host_(host),
        port_(port),
        running_(true),
        clean_session_(clean_session),
        keepalive_(keepalive) {
    // mosquitto_lib_init();

    num_mosq_ = num_mosq;
    pub_mutex_ = std::move(std::vector<std::mutex>(num_mosq_));
    pub_cv_ = std::move(std::vector<std::condition_variable>(num_mosq_));
    for (int i = 0; i < num_mosq_; i++) {
      pub_queue_.push_back(std::deque<PubTask>());
      pub_threads_.push_back(std::thread(&MQTTBase::RunPublishThread, this, i));
    }
  }

  ~MQTTBase() {
    // mosquitto_destroy(mosq_);
    for (auto m : in_mosq_) {
      mosquitto_destroy(m.second);
    }
    for (auto m : out_mosq_) {
      mosquitto_destroy(m.second);
    }
    mosquitto_lib_cleanup();
  }

  void Start() {
    for (auto m : out_mosq_) {
      mosquitto_connect(m.second, host_.c_str(), port_, keepalive_);
      mosquitto_loop_start(m.second);
    }

    for (auto m : in_mosq_) {
      mosquitto_connect(m.second, host_.c_str(), port_, keepalive_);
      mosquitto_loop_start(m.second);
    }
  }
  void Stop() {
    running_ = false;
    for (auto m : out_mosq_) {
      mosquitto_disconnect(m.second);
      mosquitto_loop_stop(m.second, true);
    }

    for (auto m : in_mosq_) {
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
        pub_cv_[idx].wait(lock,
                          [this, idx] { return !pub_queue_[idx].empty(); });
        task = pub_queue_[idx].front();
        pub_queue_[idx].pop_front();
      }
      int ret = mosquitto_publish(out_mosq_[idx], NULL, task.topic.c_str(),
                                  task.payloadlen, task.payload, 0, false);
      if (ret != MOSQ_ERR_SUCCESS) {
        std::cerr << "Failed to publish message, error code: " << ret
                  << std::endl;
      }
    }
  }

 protected:
  std::unordered_map<uint32_t, struct mosquitto*> in_mosq_;
  std::unordered_map<uint32_t, struct mosquitto*> out_mosq_;
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

class MQTTServer : public MQTTBase {
 public:
  MQTTServer(int num_mosq) : MQTTBase("localhost", 1883, true, 60, num_mosq) {
    for (int i = 0; i < num_mosq_; i++) {
      struct mosquitto* mosq = mosquitto_new(nullptr, true, this);
      in_mosq_[i] = mosq;
      SetUpInMosq(mosq);
    }
    for (int i = 0; i < num_mosq_; i++) {
      struct mosquitto* mosq = mosquitto_new(nullptr, true, this);
      out_mosq_[i] = mosq;
      SetUpOutMosq(mosq);
    }
    Start();
    std::cout << "Server created with " << in_mosq_.size()
              << " Mosquitto instances" << std::endl;
  }

  void Run() {
    pub_buffer_ = std::vector<void*>(2 * NUM_RECEIVERS, nullptr);
    pub_cb_count_ = 2 * NUM_RECEIVERS;
    for (int i = 0; i < 2 * NUM_RECEIVERS; i++) {
      auto topic =
          std::string(TOPIC_REQUEST) + std::to_string(i % NUM_RECEIVERS);
      pub_buffer_[i] = malloc(MSG_SIZE_SEND);
      Publish(i, topic, pub_buffer_[i], MSG_SIZE_SEND);
      std::cout << MILLISECONDS_EPOCH
                << " Server Published message to topic: " << topic << std::endl;
    }
    std::cout << "Server published " << 2 * NUM_RECEIVERS * MSG_SIZE_SEND / 1e6
              << "MB messages" << std::endl;
  }

 private:
  void OnMessage(struct mosquitto* mosq, void* obj,
                 const struct mosquitto_message* message) override {
    std::cout << MILLISECONDS_EPOCH << " Server Received message" << std::endl;
  }

  void OnConnect(struct mosquitto* mosq, void* userdata, int result) {
    if (result == 0) {
      std::cout << "Server connected " << in_mosq_.size() << std::endl;
      for (size_t i = 0; i < NUM_RECEIVERS; i++) {
        auto topic = std::string(TOPIC_RESPONSE) + std::to_string(i);
        if (in_mosq_[i % num_mosq_] == mosq) {
          int ret = mosquitto_subscribe(in_mosq_[i % num_mosq_], NULL,
                                        topic.c_str(), 0);
          if (ret != MOSQ_ERR_SUCCESS) {
            std::cerr << "Failed to subscribe to topic, error code: " << ret
                      << std::endl;
          }
          fprintf(stderr, "Server Subscribed to topic %s\n", topic.c_str());
        }
      }
    } else {
      std::cerr << "Connect failed with code " << result << std::endl;
    }
  }

  void SetUpInMosq(struct mosquitto* mosq) {
    mosquitto_connect_callback_set(
        mosq, [](struct mosquitto* mosq, void* obj, int result) {
          static_cast<MQTTServer*>(obj)->OnConnect(mosq, obj, result);
        });
    mosquitto_message_callback_set(
        mosq, [](struct mosquitto* mosq, void* obj,
                 const struct mosquitto_message* message) {
          static_cast<MQTTServer*>(obj)->OnMessage(mosq, obj, message);
        });
  }

  void SetUpOutMosq(struct mosquitto* mosq) {
    mosquitto_publish_callback_set(
        mosq, [](struct mosquitto* mosq, void* obj, int mid) {
          static_cast<MQTTServer*>(obj)->OnPublish(mosq, obj, mid);
        });
  }
};

class MQTTWorker : public MQTTBase {
 public:
  MQTTWorker(const std::string& uuid, int num_mosq)
      : MQTTBase("localhost", 1883, true, 60, num_mosq) {
    for (int i = 0; i < num_mosq_; i++) {
      struct mosquitto* mosq = mosquitto_new(nullptr, true, this);
      in_mosq_[i] = mosq;
      SetUpInMosq(mosq);
    }
    for (int i = 0; i < num_mosq_; i++) {
      struct mosquitto* mosq = mosquitto_new(nullptr, true, this);
      out_mosq_[i] = mosq;
      SetUpOutMosq(mosq);
    }

    uuid_ = uuid;
    Start();
    std::cout << "Worker starting with " << num_mosq_ << " Mosquitto instances"
              << std::endl;
  }

 private:
  void OnMessage(struct mosquitto* mosq, void* obj,
                 const struct mosquitto_message* message) override {
    std::string topic = std::string(TOPIC_RESPONSE) + uuid_;
    pub_cb_count_++;
    pub_buffer_.push_back(malloc(MSG_SIZE_RESP));
    Publish(0, topic, pub_buffer_.back(), MSG_SIZE_RESP);
    std::cout << MILLISECONDS_EPOCH
              << " Worker published message to topic: " << topic << std::endl;
  }

  void OnConnect(struct mosquitto* mosq, void* userdata, int result) {
    if (result == 0) {
      std::cout << "Worker connected" << std::endl;
      int ret = 0;
      auto topic = std::string(TOPIC_REQUEST) + uuid_;
      std::cout << "Worker subscribing to topic: " << topic << std::endl;
      ret = mosquitto_subscribe(mosq, NULL, topic.c_str(), 0);
      if (ret != MOSQ_ERR_SUCCESS) {
        std::cerr << "Failed to subscribe to topic, error code: " << ret
                  << std::endl;
      }
      fprintf(stderr, "Worker Subscribed to topic %s\n", topic.c_str());
    } else {
      std::cerr << "Connect failed with code " << result << std::endl;
    }
  }

  void SetUpInMosq(struct mosquitto* mosq) {
    mosquitto_connect_callback_set(
        mosq, [](struct mosquitto* mosq, void* obj, int result) {
          static_cast<MQTTWorker*>(obj)->OnConnect(mosq, obj, result);
        });
    mosquitto_message_callback_set(
        mosq, [](struct mosquitto* mosq, void* obj,
                 const struct mosquitto_message* message) {
          static_cast<MQTTWorker*>(obj)->OnMessage(mosq, obj, message);
        });
  }

  void SetUpOutMosq(struct mosquitto* mosq) {
    mosquitto_publish_callback_set(
        mosq, [](struct mosquitto* mosq, void* obj, int mid) {
          static_cast<MQTTWorker*>(obj)->OnPublish(mosq, obj, mid);
        });
  }

 private:
  std::string uuid_;
};

int main() {
  pid_t pid;
  std::vector<pid_t> receivers;

  mosquitto_lib_init();

  // Create receiver child processes
  for (int i = 0; i < NUM_RECEIVERS; i++) {
    pid = fork();
    if (pid == 0) {  // Child process
      MQTTWorker worker(std::to_string(i), 1);
      while (true) {
        std::this_thread::sleep_for(std::chrono::seconds(1));
      }
      return 0;
    } else if (pid > 0) {  // Parent process
      receivers.push_back(pid);
    } else {
      std::cerr << "Failed to fork" << std::endl;
      exit(1);
    }
  }
  std::cout << "Created " << NUM_RECEIVERS << " receiver processes"
            << std::endl;

  MQTTServer server(NUM_MOSQ);
  server.Run();

  // Wait for all children to exit
  for (auto& receiver : receivers) {
    waitpid(receiver, nullptr, 0);
  }

  return 0;
}
