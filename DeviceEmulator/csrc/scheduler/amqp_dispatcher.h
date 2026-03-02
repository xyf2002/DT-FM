#pragma once

#include <torch/torch.h>

#include <atomic>
#include <condition_variable>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "amqp_base.h"
#include "common/amqp.h"
#include "common/generator.h"
#include "common/pytorch_defs.h"
#include "utils/logger.h"

template <typename T>
std::vector<std::vector<T>> cartesian_product(const std::vector<T>& list) {
  std::vector<std::vector<T>> ranges;
  for (auto i : list) {
    std::vector<T> range;
    for (T j = 0; j < i; ++j) {
      range.push_back(j);
    }
    ranges.push_back(range);
  }

  std::vector<std::vector<T>> result;
  if (ranges.empty()) return result;

  // Initialize with the first list
  result.push_back({});
  for (const auto& list : ranges) {
    std::vector<std::vector<T>> temp;
    for (const auto& res : result) {
      for (const auto& elem : list) {
        std::vector<T> new_combination = res;
        new_combination.push_back(elem);
        temp.push_back(new_combination);
      }
    }
    result = std::move(temp);
  }
  return result;
}

class AMQPBackend : public AMQPBase {
 public:
  explicit AMQPBackend(const std::string& host = "localhost",
                       int64_t block_size = 32)
      : AMQPBase(host, block_size) {
    amqp_basic_consume(conn_, rsp_channel_,
                       amqp_cstring_bytes("mm_response_queue"),
                       amqp_empty_bytes, 0, 1, 0, amqp_empty_table);
    // create thread equal to number of cores
    // for (int i = 0; i < std::thread::hardware_concurrency(); i++) {
    //   rsp_threads_.emplace_back([this]() { HandleRsp(); });
    // }
    rsp_threads_.emplace_back([this]() { HandleRsp(); });
  }
  ~AMQPBackend() = default;

  // void disconnect() {
  //   if (conn_) {
  //     amqp_channel_close(conn_, channel_, AMQP_REPLY_SUCCESS);
  //     amqp_connection_close(conn_, AMQP_REPLY_SUCCESS);
  //     amqp_destroy_connection(conn_);
  //     conn_ = nullptr;
  //   }
  // }

  void DispatchMatMulBlock(torch::Tensor mat_a, torch::Tensor mat_b, int64_t r,
                           int64_t c, const std::vector<int64_t>& ld);

  torch::Tensor DispatchMatMul(torch::Tensor& mat_a, torch::Tensor& mat_b);

 private:
  // void Connect();

  // void DeclareQueues();
  void HandleRsp() override;

  void CallMatMulBlock(torch::Tensor& mat_a, torch::Tensor& mat_b, int64_t r,
                       int64_t c);

  void CreateOutputMatrix(const torch::Tensor& mat_a,
                          const torch::Tensor& mat_b);

 private:
  // std::string host_;
  // int64_t block_size_;
  // amqp_connection_state_t conn_;
  // amqp_socket_t* socket_;
  // int channel_;
  // int rsp_channel_;
  // std::atomic<bool> should_continue_;
  std::mutex publish_mutex_;
  // std::vector<std::thread> consume_threads_;
  torch::Tensor output_matrix_;

  std::condition_variable cv_;
  std::mutex mutex_;

  std::unordered_map<uint64_t, std::shared_ptr<MatMulRequestMessage>>
      request_map_;
};
