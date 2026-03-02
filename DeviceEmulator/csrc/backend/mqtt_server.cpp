#include "mqtt_server.h"

#include <iostream>

#include "common/types_and_defs.h"
#include "utils/logger.h"

void MQTTServer::OnMessage(struct mosquitto* mosq, void* obj,
                           const struct mosquitto_message* message) {
  // std::cout << "Received message: " << (char*)message->payload << std::endl;
  auto topic = std::string(message->topic);
  if (topic.find(MQTT_COMPUTE_TOPIC_RSP) != std::string::npos) {
    HandleMatMul(message);
  }

  // if (topic.find(MQTT_TIMER_TOPIC_RSP) != std::string::npos) {
  //   HandleTimer(message);
  // }

  // // last number after / is the device id
  // std::string device_id = topic.substr(topic.find_last_of("/") + 1);
  // bool set = redis_->expire(device_id, 10);  // refresh the key, means device
  // is alive LOG_FATAL_IF(!set, "Failed to refresh key in redis for device:
  // {}", device_id);
}

void MQTTServer::HandleTimer(const struct mosquitto_message* message) {}

void MQTTServer::HandleMatMul(const struct mosquitto_message* message) {
  auto start = std::chrono::high_resolution_clock::now();
  MatrixPartition partition;
  partition.Deserialize(message->payload, message->payloadlen);
  auto part_key = partition.GetPartitionKey();
  auto end = std::chrono::high_resolution_clock::now();
  LOG_DEBUG << part_key << " RSP Deserialization time: "
            << std::chrono::duration_cast<std::chrono::microseconds>(end -
                                                                     start)
                   .count()
            << "us";

  auto [o_ptr, o_size] = partition.mat[0];
  int64_t row_size = o_size / partition.h_dim / sizeof(float);
  int64_t col_size = partition.h_dim;

  uint64_t ul_overhead = CurrentTimeMicros() - partition.timestamp;
  // fprintf(stderr, "UL overhead: %ldus\n", ul_overhead);
  // std::string key = std::to_string(partition.oid);
  // update

  LOG_DEBUG << part_key << " partition: " << partition.DebugString();

  start = std::chrono::high_resolution_clock::now();
  auto output = torch::from_blob(o_ptr, {row_size, col_size},
                                 FLOAT32_TENSOR_OPTIONS(torch::kCPU));
  {
    // std::lock_guard<std::mutex> lock(outputs_mutex_[partition.oid]);
    IndexPutMatrixBlock(outputs_[partition.oid], output, partition.row,
                        partition.col, partition.pivot, block_size_);
  }
  end = std::chrono::high_resolution_clock::now();
  LOG_DEBUG << "UpdateMatrixBlock time: "
            << std::chrono::duration_cast<std::chrono::microseconds>(end -
                                                                     start)
                   .count()
            << "us";

  std::string uuid = std::to_string(partition.dev_id);
  // redis_->expire(uuid, 5);  // refresh the key, means device is alive
  // int nan_count = torch::isnan(output_).sum().item<int64_t>();
  // if (nan_count == 0) {
  //   output_cv_.notify_one();
  // }
  // if (rsp_cb_count_.fetch_sub(1) == 1) {
  //   output_cv_.notify_all();
  // }
  rsp_cb_counts_[partition.oid]--;
  LOG_DEBUG << "Number of responses left: "
            << rsp_cb_counts_[partition.oid].load();
  // // count the number of nans in the output
  // LOG_DEBUG("Number of NaNs in output: {}, output size: {}", nan_count,
  //           output.numel());
}

/*
torch::Tensor MQTTServer::DispatchMatMul(torch::Tensor& mat_a,
                                         torch::Tensor& mat_b) {
  output_ = CreateOutputMatrix(mat_a, mat_b);
  auto partitions = PartitionMatrices(mat_a, mat_b, block_size_);
  LOG_DEBUG << "Number of partitions: " << partitions.size();

  std::vector<std::future<void>> futures;
  // int64_t num_devices = std::stoi(GETENV("NUM_DEVICES", "1"));
  pub_buffer_.resize(partitions.size());
  pub_cb_count_ = partitions.size();
  rsp_cb_count_ = partitions.size();
  // pub_count_ = 0;
  int64_t count = 0;
  auto start = std::chrono::high_resolution_clock::now();
  for (auto& partition : partitions) {
    // auto future = std::async(std::launch::async, [this, &partition, count] {
    auto buffer = partition.Serialize();
    auto topic =
        std::string("/morphling/req/") + std::to_string(count % num_devices_);
    Publish(topic, buffer->GetBuffer(), buffer->GetSize());
    // LOG_DEBUG("Published message to topic {}, count {}", topic, count);
    pub_buffer_[count] = buffer->GetBuffer();
    // });
    // pub_count_ = (pub_count_++) % num_devices_;
    count++;
    // futures.push_back(std::move(future));
  }
  auto end = std::chrono::high_resolution_clock::now();
  LOG_DEBUG << "Publish time: "
            << std::chrono::duration_cast<std::chrono::microseconds>(end -
start) .count() << "us";

  // auto start = std::chrono::high_resolution_clock::now();
  // for (auto& f : futures) {
  //   f.wait();
  // }
  // auto end = std::chrono::high_resolution_clock::now();
  // std::cerr << "DispatchMatMul time: "
  //           << std::chrono::duration_cast<std::chrono::milliseconds>(end -
  //                                                                    start)
  //                  .count()
  //           << "ms" << std::endl;
  // output needs to be all non-nan
  // while (std::isnan(output_.sum().item<float>())) {
  //   std::this_thread::sleep_for(std::chrono::milliseconds(100));
  // }
  start = std::chrono::high_resolution_clock::now();
  // std::unique_lock<std::mutex> lock(output_mutex_);
  // output_cv_.wait(lock, [this] {
  //   bool all_non_nan = !torch::isnan(output_).any().item<bool>();
  //   int64_t nan_count = torch::isnan(output_).sum().item<int64_t>();
  //   LOG_DEBUG("All non-nan: {}, number of NaNs: {}", all_non_nan, nan_count);
  //   return all_non_nan;
  // });
  while (rsp_cb_count_ > 0) {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
  end = std::chrono::high_resolution_clock::now();
  LOG_DEBUG << "Waiting time: "
            << std::chrono::duration_cast<std::chrono::microseconds>(end -
start) .count() << "us"; return outputs_[0];
}
*/

torch::Tensor MQTTServer::WaitMatMul(int oid) {
  auto start = std::chrono::high_resolution_clock::now();
  while (rsp_cb_counts_[oid] > 0) {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
  auto end = std::chrono::high_resolution_clock::now();
  auto shape = outputs_[oid].sizes().vec();
  auto wait_time =
      std::chrono::duration_cast<std::chrono::microseconds>(end - start)
          .count();
  LOG_DEBUG << "Waiting time: " << wait_time << "us for oid: " << oid
            << ", shape: " << shape;

  uint64_t max_time = 0;
  for (int i = 0; i < num_devices_; i++) {
    std::string key = std::to_string(i);
    auto logical_time = redis_->hget(key, "logical_time");
    if (!logical_time) {
      LOG_FATAL << "Failed to get logical time from redis for key: " << key;
    }

    // convert to uint64_t
    uint64_t time = std::stoull(*logical_time);
    max_time = std::max(max_time, time);
  }
  // set max time to all devices
  for (int i = 0; i < num_devices_; i++) {
    std::string key = std::to_string(i);
    redis_->hset(key, "logical_time", std::to_string(max_time));
  }
  logical_time_ = max_time;
  real_time_ += wait_time;
  LOG_INFO << "Real time: " << real_time_.load()
           << "us, Logical time: " << logical_time_.load() << "us";
  mm_count_--;
  return outputs_[oid];
}

void MQTTServer::PublishPartition(MatrixPartitionPtr& partition, int64_t oid,
                                  int count) {
  partition->oid = oid;
  auto buffer = partition->Serialize();
  auto topic =
      std::string(MQTT_COMPUTE_TOPIC_REQ) + std::to_string(partition->dev_id);
  Publish(partition->dev_id, topic, buffer->GetBuffer(), buffer->GetSize());
  // LOG_DEBUG("Published message to topic {}, count {}", topic, count);
  pub_buffer_[count] = buffer->GetBuffer();
}

void MQTTServer::DispatchMatMulAsync(torch::Tensor& mat_a,
                                     torch::Tensor& mat_b) {
  outputs_[mm_count_].set_data(CreateOutputMatrix(mat_a, mat_b));
  auto partitions = PartitionMatrices(mat_a, mat_b, block_size_);
  auto a_shape = mat_a.sizes().vec();
  auto b_shape = mat_b.sizes().vec();

  auto cur_ver = partitions[0]->version;
  LOG_INFO << "[" << cur_ver << "] Number of partitions: " << partitions.size()
           << " for A: " << a_shape << " and B: " << b_shape;

  RephrasePartitions(partitions);

  int64_t total_bytes = 0;
  for (auto& partition : partitions) {
    total_bytes +=
        std::get<1>(partition->mat[0]) + std::get<1>(partition->mat[1]);
  }
  LOG_INFO << "Total bytes: " << (total_bytes / 1e6) << "MB";

  // int64_t num_devices = std::stoi(GETENV("NUM_DEVICES", "1"));
  int64_t count = pub_buffer_.size();
  pub_buffer_.resize(pub_buffer_.size() + partitions.size());
  pub_cb_count_ += partitions.size();
  rsp_cb_counts_[mm_count_] = partitions.size();

  auto start = std::chrono::high_resolution_clock::now();
  std::vector<std::future<void>> futures;
  for (auto& partition : partitions) {
    // auto publish_func = std::bind(&MQTTServer::PublishPartition, this,
    //                               std::ref(partition), mm_count_.load(),
    //                               count);
    // futures.push_back(std::async(std::launch::async, publish_func));
    // count++;
    partition->oid = mm_count_;
    auto buffer = partition->Serialize();
    auto topic =
        std::string(MQTT_COMPUTE_TOPIC_REQ) + std::to_string(partition->dev_id);
    Publish(partition->dev_id, topic, buffer->GetBuffer(), buffer->GetSize());
    // LOG_DEBUG("Published message to topic {}, count {}", topic, count);
    pub_buffer_[count] = buffer->GetBuffer();
    // });
    // pub_count_ = (pub_count_++) % num_devices_;
    count++;
    // futures.push_back(std::move(future));
  }
  for (auto& f : futures) {
    f.wait();
  }
  auto end = std::chrono::high_resolution_clock::now();
  LOG_INFO << "Publish time: "
           << std::chrono::duration_cast<std::chrono::microseconds>(end - start)
                  .count()
           << "us";
  mm_count_++;
}

void MQTTServer::OnConnect(struct mosquitto* mosq, void* userdata, int result) {
  if (result == 0) {
    // Connection successful
    for (size_t i = 0; i < num_devices_; i++) {
      auto topic = std::string(MQTT_COMPUTE_TOPIC_RSP) + std::to_string(i);
      if (mosq == mosq_[i % num_mosq_]) {
        int ret = mosquitto_subscribe(mosq, NULL, topic.c_str(), 0);
        LOG_FATAL_IF(ret != MOSQ_ERR_SUCCESS)
            << "Failed to subscribe to topic, error code: " << ret;
        fprintf(stderr, "Subscribed to topic %s\n", topic.c_str());
      }
    }
  } else {
    LOG_FATAL << "Connect failed with code " << result;
  }
}

MQTTServer::MQTTServer(int64_t block_size)
    : MQTTBase(),
      block_size_(block_size),
      outputs_mutex_(5),
      rsp_cb_counts_(5) {
  // num_devices_ = std::stoi(GETENV("NUM_DEVICES", "1"));

  for (int i = 0; i < num_mosq_; i++) {
    // auto key = GenUUID();
    struct mosquitto* mosq = mosquitto_new(NULL, clean_session_, this);
    mosq_[i] = mosq;
    SetUpMosq(mosq);
  }

  // Start();
  InitLogger();

  // no more than 5 MAtMul in parallel
  outputs_ = std::move(std::vector<torch::Tensor>(5));
  // rsp_cb_counts_ = std::move(std::vector<std::atomic_ullong>(5));
  // outputs_mutex_ = std::move(std::vector<std::mutex>(5));
  for (int i = 0; i < 5; i++) {
    outputs_[i] = torch::empty({0, 0});
    rsp_cb_counts_[i] = 0;
  }

  redis_ = GetRedisConnection();

  // get the number of keys from redis
  num_devices_ = GetNumKeys(redis_);
  LOG_INFO << "Number of devices registered: " << num_devices_;
}

// void MQTTServer::OnPublish(struct mosquitto* mosq, void* obj, int mid) {
//   pub_cb_count_--;
//   fprintf(stderr, "Published message %d\n", mid);
//   if (pub_cb_count_ == 0) {
//     std::cout << "All messages published" << std::endl;
//     // for (auto* ptr : pub_buffer_) {
//     //   free(ptr);
//     // }
//     pub_buffer_.clear();
//   }
// }

void MQTTServer::SetUpMosq(struct mosquitto* mosq) {
  mosquitto_publish_callback_set(
      mosq, [](struct mosquitto* mosq, void* obj, int mid) {
        static_cast<MQTTServer*>(obj)->OnPublish(mosq, obj, mid);
      });
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

void MQTTServer::RephrasePartitions(
    std::vector<MatrixPartitionPtr>& partitions) {
  std::vector<float> device_time(num_devices_, 0);
  std::vector<float> device_ul_bw(num_devices_, 0);
  std::vector<float> device_dl_bw(num_devices_, 0);
  std::vector<float> device_flops(num_devices_, 0);
  std::vector<std::unordered_set<TensorKey>> device_tensors(num_devices_);

  for (int i = 0; i < num_devices_; i++) {
    std::string uuid = std::to_string(i);
    auto d_info = GetDeviceInfo(redis_, uuid);
    device_ul_bw[i] = std::stof(d_info["ul_bw"]);
    device_dl_bw[i] = std::stof(d_info["dl_bw"]);
    device_flops[i] = std::stof(d_info["flops"]);
  }

  // LOG_INFO("Device info: ul_bw: {}, dl_bw: {}, flops: {}", device_ul_bw,
  //           device_dl_bw, device_flops);
  // shuffle the partitions
  std::random_shuffle(partitions.begin(), partitions.end());

  // greedy algorithm to select the minimal time
  for (auto& partition : partitions) {
    float min_time = std::numeric_limits<float>::max();
    int min_device = 0;
    auto version = partition->version;
    auto tensor_key_row = partition->GetRowKey();
    auto tensor_key_col = partition->GetColKey();
    bool min_r_cached = false;
    bool min_c_cached = false;
    for (int i = 0; i < num_devices_; i++) {
      auto& tensors = device_tensors[i];

      bool r_cached = tensors.find(tensor_key_row) != tensors.end();
      bool c_cached = tensors.find(tensor_key_col) != tensors.end();

      auto r_size = std::get<1>(partition->mat[0]);
      auto c_size = std::get<1>(partition->mat[1]);
      auto cached_r_size = (r_cached) ? 0 : r_size;
      auto cached_c_size = (c_cached) ? 0 : c_size;

      int64_t num_rows = r_size / partition->h_dim / sizeof(float);
      int64_t num_cols = c_size / partition->h_dim / sizeof(float);

      float ul_time =
          (float)(num_rows * num_cols) * sizeof(float) / device_ul_bw[i];
      float dl_time = (float)(cached_r_size + cached_c_size) / device_dl_bw[i];
      float flops =
          (float)2.0 * num_rows * num_cols * partition->h_dim / device_flops[i];

      float time = std::max(std::max(ul_time, dl_time), flops) + device_time[i];
      // ul_time + dl_time + flops + device_time[i];
      if (time < min_time) {
        min_time = time;
        min_device = i;
        min_r_cached = r_cached;
        min_c_cached = c_cached;
        // fprintf(stderr, "Device: %d, UL time: %f, DL time: %f, FLOPS: %f,
        // Time: %f\n", i, ul_time, dl_time, flops, time);
      }
    }
    assert(min_time != std::numeric_limits<float>::max());
    // update the time for the device
    device_time[min_device] = min_time;
    partition->dev_id = min_device;
    device_tensors[min_device].insert(tensor_key_row);
    device_tensors[min_device].insert(tensor_key_col);

    if (min_r_cached) {
      partition->mat[0] = {nullptr, 0};
    }
    if (min_c_cached) {
      partition->mat[1] = {nullptr, 0};
    }

    // print device time
    // std::string time_str = "[";
    // for (int i = 0; i < num_devices_; i++) {
    //   time_str += std::to_string(device_time[i]) + ",";
    // }
    // time_str += "]";
    // fprintf(stderr, "Device time: %s\n", time_str.c_str());
  }
  LOG_INFO << "Device time: " << device_time;
}
