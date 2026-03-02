#include "mqtt_worker.h"

#include <torch/torch.h>

#include <iostream>

#include "common/pytorch_defs.h"
#include "server_base.h"
#include "utils/logger.h"

void MQTTWorker::OnMessage(struct mosquitto* mosq, void* obj,
                           const struct mosquitto_message* message) {
  auto start = std::chrono::high_resolution_clock::now();
  auto topic = std::string(message->topic);
  if (topic.find(MQTT_COMPUTE_TOPIC_REQ) != std::string::npos) {
    HandleMatMul(message);
  }
  auto end = std::chrono::high_resolution_clock::now();
  LOG_DEBUG << "Handle message time: "
            << std::chrono::duration_cast<std::chrono::microseconds>(end -
                                                                     start)
                   .count()
            << "us";

  // if (topic.find(MQTT_TIMER_TOPIC_REQ) != std::string::npos) {
  //   HandleTimer(message);
  // }
}

void MQTTWorker::HandleTimer(const struct mosquitto_message* message) {}

void MQTTWorker::HandleMatMul(const struct mosquitto_message* message) {
  // std::cerr << "Handling matmul" << std::endl;
  at::InferenceMode infer_guard(true);
  auto start = std::chrono::high_resolution_clock::now();
  MatrixPartition partition;
  partition.Deserialize(message->payload, message->payloadlen);
  auto part_key = partition.GetPartitionKey();
  auto end = std::chrono::high_resolution_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  LOG_DEBUG << part_key << " REQ Deserialization time: " << duration.count()
            << "us";

  LOG_DEBUG << part_key << " partition: " << partition.DebugString();

  // create tensor from partition
  auto [r_ptr, r_size] = partition.mat[0];
  auto [c_ptr, c_size] = partition.mat[1];
  int64_t row_size = r_size / partition.h_dim / sizeof(float);
  int64_t col_size = c_size / partition.h_dim / sizeof(float);

  assert(row_size * partition.h_dim * sizeof(float) == r_size);
  assert(col_size * partition.h_dim * sizeof(float) == c_size);

  auto tensor_key_row = partition.GetRowKey();
  auto tensor_key_col = partition.GetColKey();

  {
    std::lock_guard<std::mutex> lock(cache_mutex_);
    if (r_size > 0) {
      CacheTensor(tensor_key_row, r_ptr, r_size, partition.h_dim);
    }

    if (c_size > 0) {
      CacheTensor(tensor_key_col, c_ptr, c_size, partition.h_dim);
    }

    auto r_cached = cached_tensors_.Exist(tensor_key_row);
    auto c_cached = cached_tensors_.Exist(tensor_key_col);

    LOG_DEBUG << part_key << " Row cached: " << r_cached
              << ", row size: " << row_size << ", Col cached: " << c_cached
              << ", col size: " << col_size;

    FillPartition(partition);

    if (r_size == 0 && !r_cached) {
      LOG_WARN << part_key << " Row not cached, saving for next msg";
      SavePartition(partition);
      return;
    }

    if (c_size == 0 && !c_cached) {
      LOG_WARN << part_key << " Col not cached, saving for next msg";
      SavePartition(partition);
      return;
    }
  }

  LOG_DEBUG << part_key << " Handle partition immediately";
  HandlePartition(partition);

  std::vector<std::string> keys;
  for (auto& c_part : cached_partitions_) {
    FillPartition(c_part);
    r_size = std::get<1>(c_part.mat[0]);
    c_size = std::get<1>(c_part.mat[1]);
    if (r_size > 0 && c_size > 0) {
      auto key = c_part.GetPartitionKey();
      LOG_DEBUG << key << " Handle partition from cache";
      HandlePartition(c_part);
      keys.push_back(key);
    } else {
      LOG_WARN << c_part.GetPartitionKey()
               << " Partition is not ready, r_size: " << r_size
               << ", c_size: " << c_size;
    }
  }

  for (auto it = cached_partitions_.begin(); it != cached_partitions_.end();) {
    if (std::find(keys.begin(), keys.end(), it->GetPartitionKey()) !=
        keys.end()) {
      it = cached_partitions_.erase(it);
    } else {
      ++it;
    }
  }
}

void MQTTWorker::FillPartition(MatrixPartition& partition) {
  auto r_size = std::get<1>(partition.mat[0]);
  auto c_size = std::get<1>(partition.mat[1]);
  auto tensor_key_row = partition.GetRowKey();
  auto tensor_key_col = partition.GetColKey();
  auto r_cached = cached_tensors_.Exist(tensor_key_row);
  auto c_cached = cached_tensors_.Exist(tensor_key_col);
  if (r_size == 0 && r_cached) {
    auto cached_tensor = cached_tensors_.Get(tensor_key_row);
    partition.mat[0] = {cached_tensor.data_ptr(),
                        cached_tensor.numel() * sizeof(float)};
  }

  if (c_size == 0 && c_cached) {
    auto cached_tensor = cached_tensors_.Get(tensor_key_col);
    partition.mat[1] = {cached_tensor.data_ptr(),
                        cached_tensor.numel() * sizeof(float)};
  }
}

void MQTTWorker::HandlePartition(const MatrixPartition& partition) {
  auto part_key = partition.GetPartitionKey();
  // create tensors from partition
  auto [r_ptr, r_size] = partition.mat[0];
  auto [c_ptr, c_size] = partition.mat[1];
  int64_t row_size = r_size / partition.h_dim / sizeof(float);
  int64_t col_size = c_size / partition.h_dim / sizeof(float);

  auto start = std::chrono::high_resolution_clock::now();
  auto row = torch::from_blob(r_ptr, {row_size, partition.h_dim},
                              FLOAT32_TENSOR_OPTIONS(torch::kCPU))
                 .to(torch::kCUDA, 0);
  auto col = torch::from_blob(c_ptr, {col_size, partition.h_dim},
                              FLOAT32_TENSOR_OPTIONS(torch::kCPU))
                 .to(torch::kCUDA, 0);

  LOG_DEBUG << part_key << " Row: " << row.sizes().vec()
            << ", Col: " << col.sizes().vec();

  auto result = torch::mm(row, col.transpose(0, 1)).to(torch::kCPU);
  uint64_t mm_time = (double)row_size * partition.h_dim * col_size * 2 /
                     device_info_["flops"] * 1e6;  // in microseconds
  // logical_time_ += (double)row_size * partition.h_dim * col_size * 2 /
  //                  device_info_["flops"] * 1e6;  // in microseconds

  // auto zeros = torch::zeros_like(result);
  // LOG_FATAL_IF(torch::allclose(result, zeros),
  //              "Result is all zero, something went wrong, worker cannot be "
  //              "launched as child process");

  auto end = std::chrono::high_resolution_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  LOG_DEBUG << part_key << " Matmul real time: " << duration.count()
            << "us, Matmul logical time: " << mm_time << "us";
  mm_time = duration.count();
  // logical_time_ += duration.count();

  start = std::chrono::high_resolution_clock::now();
  MatrixPartition response = partition;
  response.h_dim = result.size(1);
  response.timestamp = CurrentTimeMicros();
  response.mat.clear();
  response.mat.push_back({result.data_ptr(), result.numel() * sizeof(float)});

  auto buffer = response.Serialize();

  pub_cb_count_++;
  pub_buffer_.push_back(buffer->GetBuffer());

  // replace req with rsp
  std::string topic = MQTT_COMPUTE_TOPIC_RSP + uuid_;
  end = std::chrono::high_resolution_clock::now();
  duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  LOG_DEBUG << part_key << " RSP Serialization time: " << duration.count()
            << "us";

  Publish(partition.dev_id, topic, buffer->GetBuffer(), buffer->GetSize());
  uint64_t ul_time =
      buffer->GetSize() / device_info_["ul_bw"] * 1e6;  // in microseconds

  // std::vector<OptionalString> vals;
  // auto overhead = redis.hget(uuid_, "r_dl_overhead");

  uint64_t dl_time =
      (double)partition.size_ / device_info_["dl_bw"] * 1e6;  // in microseconds
  uint64_t dl_overhead = CurrentTimeMicros() - partition.timestamp;

  std::unordered_map<std::string, std::string> time_map;
  time_map["v_dl_time"] = std::to_string(dl_time);
  time_map["v_mm_time"] = std::to_string(mm_time);
  time_map["v_ul_time"] = std::to_string(ul_time);
  time_map["r_dl_overhead"] = std::to_string(dl_overhead);

  std::lock_guard<std::mutex> lock(redis_mutex_);
  // get logical time from redis
  auto time = redis_->hget(uuid_, "logical_time");
  LOG_FATAL_IF(!time) << "Failed to get logical time from redis, key: "
                      << uuid_;
  int64_t logical_time = std::stoull(*time);
  logical_time += std::max(dl_time, std::max(mm_time, ul_time));
  // fprintf(stderr, "DL overhead: %ldus\n", dl_overhead);
  // logical_time_ = std::stoull(*time);
  // logical_time_ += (double)message->payloadlen / device_info_["dl_bw"] *
  //                  1e6;  // in microseconds

  // logical_time_ += std::max(dl_time, std::max(mm_time, ul_time));
  // logical_time_ +=
  //     (double)size / device_info_["ul_bw"] * 1e6;  // in microseconds
  // LOG_INFO("Logical time: {}us", logical_time_.load());
  redis_->hmset(uuid_, time_map.begin(), time_map.end());
  redis_->hset(uuid_, "logical_time", std::to_string(logical_time));
}

void MQTTWorker::CacheTensor(const TensorKey& key, void* ptr, int64_t size,
                             int64_t h_dim) {
  if (cached_tensors_.Exist(key)) {
    return;
  }
  void* cpy_ptr = kCachingAllocator->Allocate(size);
  int64_t ld_size = size / h_dim / sizeof(float);
  std::memcpy(cpy_ptr, ptr, size);
  cached_tensors_.Put(key,
                      torch::from_blob(cpy_ptr, {ld_size, h_dim},
                                       FLOAT32_TENSOR_OPTIONS(torch::kCPU)),
                      size);
}

void MQTTWorker::SavePartition(MatrixPartition& partition) {
  // void* ptr = malloc(partition.size_);
  // std::memcpy(ptr, partition.ptr_, partition.size_);
  // partition.Deserialize(ptr, partition.size_);
  for (auto& mat : partition.mat) {
    mat = {nullptr, 0};
  }
  cached_partitions_.push_back(partition);
}

// void MQTTWorker::SaveMessage(const struct mosquitto_message* message) {
//   int64_t msg_struct_size = sizeof(struct mosquitto_message);
//   int64_t size = message->payloadlen + msg_struct_size;

//   auto cached_key = std::make_tuple((void*)message, size);
//   if (cached_msgs_.find(cached_key) != cached_msgs_.end()) {
//     return;
//   }

//   void* cpy_ptr = malloc(size);
//   int64_t offset = 0;
//   std::memcpy(cpy_ptr, message, msg_struct_size);
//   offset += msg_struct_size;
//   std::memcpy((char*)cpy_ptr + offset, message->payload,
//   message->payloadlen); reinterpret_cast<struct
//   mosquitto_message*>(cpy_ptr)->payload =
//       (void*)((char*)cpy_ptr + msg_struct_size);

//   cached_msgs_.push_back({cpy_ptr, size});
// }

void MQTTWorker::OnConnect(struct mosquitto* mosq, void* userdata, int result) {
  if (result == 0) {
    // Connection successful
    int ret = 0;
    ret = mosquitto_subscribe(mosq, NULL, comp_topic_.c_str(), 0);
    LOG_FATAL_IF(ret != MOSQ_ERR_SUCCESS)
        << "Failed to subscribe to topic, error code: " << ret;
    fprintf(stderr, "Subscribed to topic %s\n", comp_topic_.c_str());
    // ret = mosquitto_subscribe(mosq, NULL, timer_topic_.c_str(), 0);
    // LOG_FATAL_IF(ret != MOSQ_ERR_SUCCESS,
    //              "Failed to subscribe to topic, error code: {}", ret);
    // fprintf(stderr, "Subscribed to topic %s\n", timer_topic_.c_str());
  } else {
    LOG_FATAL << "Connect failed with code " << result;
  }
}

void MQTTWorker::SetUpMosq(struct mosquitto* mosq) {
  mosquitto_publish_callback_set(
      mosq, [](struct mosquitto* mosq, void* obj, int mid) {
        static_cast<MQTTWorker*>(obj)->OnPublish(mosq, obj, mid);
      });
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

MQTTWorker::MQTTWorker(const std::string& uuid)
    : MQTTBase(), cached_tensors_(0) {
  for (int i = 0; i < num_mosq_; i++) {
    // auto key = GenUUID();
    struct mosquitto* mosq = mosquitto_new(NULL, clean_session_, this);
    mosq_[i] = mosq;
    SetUpMosq(mosq);
  }
  // Start();
  InitLogger();

  uuid_ = uuid;

  comp_topic_ = std::string(MQTT_COMPUTE_TOPIC_REQ) + uuid;
  // timer_topic_ =
  //     std::string(MQTT_TIMER_TOPIC_REQ) +
  //     std::to_string(device_info_.at("id"));

  // CUDA context warmup and do random matmul
  torch::Tensor warmup_a = torch::rand({128, 4096}).to(torch::kCUDA, 0);
  torch::Tensor warmup_b = torch::rand({4096, 128}).to(torch::kCUDA, 0);

  torch::mm(warmup_a, warmup_b);

  redis_ = GetRedisConnection();
  std::unordered_map<std::string, std::string> info;
  redis_->hgetall(uuid, std::inserter(info, info.begin()));
  // convert to uint64_t
  for (auto& [key, value] : info) {
    device_info_[key] = std::stoull(value);
    // std::cerr << key << " " << value << std::endl;
  }

  // set env variable MORPHLING_PIN_SIZE
  std::string pin_size = std::to_string(device_info_["memory"]);
  setenv("MORPHLING_PIN_SIZE", pin_size.c_str(), 1);
  InitCachingAllocator(MemoryType::PIN);
  cached_tensors_ = FixSizeLRUCache<TensorKey, torch::Tensor>(
      device_info_["memory"],
      [](const TensorKey& key, const torch::Tensor& tensor) {
        kCachingAllocator->Free(tensor.data_ptr());
      });

  // // create a thread that refreshes the key in redis every 1 second
  // std::thread t([this]() {
  //   while (running_) {
  //     std::this_thread::sleep_for(std::chrono::seconds(1));
  //     auto set =
  //         redis_->expire(uuid_, 5);  // refresh the key, means device is
  //         alive
  //     if (!set) {
  //       LOG_WARN("Failed to refresh key in redis for device: {}", uuid_);
  //       // reset device info
  //       std::unordered_map<std::string, std::string> info;
  //       for (auto& [key, value] : device_info_) {
  //         info[key] = std::to_string(value);
  //       }
  //       redis_->hmset(uuid_, info.begin(), info.end());
  //       redis_->hset(uuid_, "logical_time",
  //       std::to_string(logical_time_.load())); redis_->expire(uuid_, 5);
  //     }
  //   }
  // });
  // t.detach();
}
