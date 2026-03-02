#pragma once

#include "common/generator.h"
#include "common/pytorch_defs.h"
#include "memory/caching_allocator.h"
#include "mqtt_base.h"
#include "server_base.h"

class MQTTWorker : public MQTTBase {
 public:
  // MQTTWorker(const std::unordered_map<std::string, uint64_t>& device_info);
  MQTTWorker(const std::string& uuid);
  ~MQTTWorker() { Stop(); }
  void OnMessage(struct mosquitto* mosq, void* obj,
                 const struct mosquitto_message* message) override;

  void SetUpMosq(struct mosquitto* mosq);
  void OnConnect(struct mosquitto* mosq, void* userdata, int result);

 private:
  void HandleMatMul(const struct mosquitto_message* message);
  void HandleTimer(const struct mosquitto_message* message);
  void CacheTensor(const TensorKey& key, void* ptr, int64_t size,
                   int64_t h_dim);
  // void SaveMessage(const struct mosquitto_message* message);
  void SavePartition(MatrixPartition& partition);
  void HandlePartition(const MatrixPartition& partition);
  void FillPartition(MatrixPartition& partition);

 private:
  std::string comp_topic_;
  std::string timer_topic_;
  std::unordered_map<std::string, uint64_t> device_info_;
  std::string uuid_;

  std::vector<MatrixPartition> cached_partitions_;
  std::unordered_set<PtrData> cached_msgs_;

  FixSizeLRUCache<TensorKey, torch::Tensor> cached_tensors_;
  std::mutex cache_mutex_;
  std::mutex redis_mutex_;

  CachingAllocator* allocator_;

  // std::atomic_ullong logical_time_{0};
  sw::redis::Redis* redis_;
};
