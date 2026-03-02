#pragma once

#include "common/generator.h"
#include "common/pytorch_defs.h"
#include "mqtt_base.h"
#include "server_base.h"

class MQTTServer : public MQTTBase {
 public:
  MQTTServer(int64_t block_size = 32);
  ~MQTTServer() = default;

  // torch::Tensor DispatchMatMul(torch::Tensor& mat_a, torch::Tensor& mat_b);

  void DispatchMatMulAsync(torch::Tensor& mat_a, torch::Tensor& mat_b);
  torch::Tensor WaitMatMul(int oid);

 private:
  void OnMessage(struct mosquitto* mosq, void* obj,
                 const struct mosquitto_message* message) override;
  // void OnPublish(struct mosquitto* mosq, void* obj, int mid);
  void OnConnect(struct mosquitto* mosq, void* userdata, int result);
  void SetUpMosq(struct mosquitto* mosq);

  void HandleMatMul(const struct mosquitto_message* message);
  void HandleTimer(const struct mosquitto_message* message);

  void RephrasePartitions(std::vector<MatrixPartitionPtr>& partitions);
  void PublishPartition(MatrixPartitionPtr& partition, int64_t oid, int count);

 private:
  int64_t block_size_;
  int num_devices_;

  std::atomic_int mm_count_{0};
  std::vector<torch::Tensor> outputs_;
  std::vector<std::mutex> outputs_mutex_;
  std::vector<std::atomic_ullong> rsp_cb_counts_;

  torch::Tensor output_;
  std::mutex output_mutex_;
  std::condition_variable output_cv_;

  sw::redis::Redis* redis_;
  std::atomic_ullong logical_time_{0};
  std::atomic_ullong real_time_{0};
};

// class GreedySchedulingPolicy {
//   public:
//     GreedySchedulingPolicy(MQTTServer& server) : server_(server) {}
//     ~GreedySchedulingPolicy() = default;

//     void RephrasePartitions(std::vector<MatrixPartition>& partitions);
//     void Refresh();

//   private:
//     MQTTServer& server_;
// };
