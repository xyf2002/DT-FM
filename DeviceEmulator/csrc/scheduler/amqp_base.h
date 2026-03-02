#pragma once

#include "common/amqp.h"
#include "common/pytorch_defs.h"
#include "utils/logger.h"

class AMQPBase {
 public:
  // channel 0 is reserved, DO NOT use it
  explicit AMQPBase(const std::string& host = "localhost",
                    int64_t block_size = 32)
      : host_(host),
        block_size_(block_size),
        conn_(nullptr),
        req_channel_(100),
        rsp_channel_(101),
        should_continue_(true) {
    Connect();
  }
  ~AMQPBase() {
    should_continue_ = false;
    Disconnect();

    for (auto& t : req_threads_) {
      t.join();
    }

    for (auto& t : rsp_threads_) {
      t.join();
    }
  }

 protected:
  void Connect() {
    conn_ = amqp_new_connection();
    socket_ = amqp_tcp_socket_new(conn_);
    if (!socket_) {
      throw std::runtime_error("Failed to create TCP socket");
    }

    int status = amqp_socket_open(socket_, host_.c_str(), 5672);
    if (status) {
      LOG_ERROR << "Failed to open TCP socket, status: " << status;
      throw std::runtime_error("Failed to open TCP socket");
    }

    LOG_DEBUG << "Opened TCP socket to AMQP server";
    {
      auto r = amqp_login(conn_, "/", 0, 131072, 0, AMQP_SASL_METHOD_PLAIN,
                          "guest", "guest");
      die_on_amqp_error(r, "Logging in to AMQP server");
      LOG_DEBUG << "Logged in to AMQP server";
    }

    {
      auto* r = amqp_channel_open(conn_, req_channel_);
      die_on_amqp_error(amqp_get_rpc_reply(conn_),
                        "Opening channel 0 for requests");
    }

    {
      auto* r = amqp_channel_open(conn_, rsp_channel_);
      die_on_amqp_error(amqp_get_rpc_reply(conn_),
                        "Opening channel 1 for responses");
    }

    LOG_DEBUG << "Opened channel to AMQP server";
    DeclareQueues();

    LOG_DEBUG << "Connected to AMQP server";
    // for (int i = 0; i < 10; i++) {
    //   consume_threads_.emplace_back([this]() { ConsumeResponse(); });
    // }
  }
  void DeclareQueues() {
    // Declare exchanges and queues
    {
      auto req_queuename = amqp_cstring_bytes("mm_request_queue");
      amqp_queue_declare_ok_t* r = amqp_queue_declare(
          conn_, req_channel_, req_queuename, 0, 0, 0, 0, amqp_empty_table);
      die_on_amqp_error(amqp_get_rpc_reply(conn_),
                        "Declaring queue mm_request_queue");
      LOG_DEBUG << "Declared request queue";

      // amqp_queue_bind(conn_, req_channel_, req_queuename,
      // amqp_cstring_bytes(""),
      //                 req_queuename, amqp_empty_table);
      // die_on_amqp_error(amqp_get_rpc_reply(conn_),
      //                   "Binding queue mm_request_queue");
    }

    {
      auto rsp_queuename = amqp_cstring_bytes("mm_response_queue");
      amqp_queue_declare_ok_t* r = amqp_queue_declare(
          conn_, rsp_channel_, rsp_queuename, 0, 0, 0, 0, amqp_empty_table);
      die_on_amqp_error(amqp_get_rpc_reply(conn_),
                        "Declaring queue mm_response_queue");
      LOG_DEBUG << "Declared response queue";

      // amqp_queue_bind(conn_, rsp_channel_, rsp_queuename,
      // amqp_cstring_bytes(""),
      //                 rsp_queuename, amqp_empty_table);
      // die_on_amqp_error(amqp_get_rpc_reply(conn_),
      //                   "Binding queue mm_response_queue");
    }
  }
  void Disconnect() {
    if (conn_) {
      amqp_channel_close(conn_, req_channel_, AMQP_REPLY_SUCCESS);
      amqp_channel_close(conn_, rsp_channel_, AMQP_REPLY_SUCCESS);
      amqp_connection_close(conn_, AMQP_REPLY_SUCCESS);
      amqp_destroy_connection(conn_);
      conn_ = nullptr;
    }
  }

  virtual void HandleReq() { LOG_FATAL << "HandleReq not implemented"; }
  virtual void HandleRsp() { LOG_FATAL << "HandleRsp not implemented"; }

 protected:
  amqp_connection_state_t conn_;
  amqp_socket_t* socket_;
  int64_t block_size_;
  int req_channel_;
  int rsp_channel_;
  std::string host_;
  bool should_continue_;
  std::vector<std::thread> req_threads_;
  std::vector<std::thread> rsp_threads_;
};
