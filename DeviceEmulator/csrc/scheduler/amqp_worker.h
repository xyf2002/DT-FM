#pragma once

#include "amqp_base.h"
#include "common/amqp.h"
#include "common/pytorch_defs.h"

class AMQPWorker : public AMQPBase {
 public:
  explicit AMQPWorker(const std::string& host = "localhost",
                      int64_t block_size = 32)
      : AMQPBase(host, block_size) {}
  ~AMQPWorker() = default;
  void HandleReq() override;
  void HandleRsp() override;
};
