#include "amqp_dispatcher.h"

#include <rabbitmq-c/framing.h>

#include <future>

void AMQPBackend::CallMatMulBlock(torch::Tensor& mat_a, torch::Tensor& mat_b,
                                  int64_t r, int64_t c) {
  // check matrix number of dimensions:
  // 1) mat_a and mat_b are 2D matrices
  // 2) mat_a > 2D and mat_b is 2D
  // 3) mat_a > 2D and mat_b > 2D

  if (mat_a.dim() == 2 && mat_b.dim() == 2) {
    // print("call 2D-2D", mat_a.sizes(), mat_b.sizes())
    LOG_DEBUG << "Calling 2D-2D matmul block";
    DispatchMatMulBlock(mat_a, mat_b, r, c, {});
  } else if (mat_a.dim() > 2 && mat_b.dim() == 2) {
    LOG_DEBUG << "Calling >2D-2D matmul block";
    auto ld = mat_a.sizes().slice(0, mat_a.dim() - 2);
    auto ld_combinations = cartesian_product(ld.vec());
    for (const auto ld : ld_combinations) {
      // convert to torch::tensor
      torch::Tensor ld_vec = torch::tensor(ld);
      DispatchMatMulBlock(mat_a.index({ld_vec, "..."}), mat_b, r, c, ld);
    }
  } else {
    auto ld = mat_a.sizes().slice(0, mat_a.dim() - 2);
    // get all combinations of indices for the leading dimensions
    LOG_DEBUG << "Calling >2D->2D matmul block";
    auto ld_combinations = cartesian_product(ld.vec());
    for (const auto ld : ld_combinations) {
      // convert to torch::tensor
      torch::Tensor ld_vec = torch::tensor(ld);
      DispatchMatMulBlock(mat_a.index({ld_vec}), mat_b.index({ld_vec}), r, c,
                          ld);
    }
  }
}

torch::Tensor AMQPBackend::DispatchMatMul(torch::Tensor& mat_a,
                                          torch::Tensor& mat_b) {
  auto a_shape = mat_a.sizes().vec();
  auto b_shape = mat_b.sizes().vec();

  int64_t in_dim = a_shape[a_shape.size() - 2];
  int64_t h_dim = a_shape[a_shape.size() - 1];
  int64_t out_dim = b_shape[b_shape.size() - 1];
  bool padded = false;

  CreateOutputMatrix(mat_a, mat_b);

  // if (!(in_dim % block_size_ == 0 && out_dim % block_size_ == 0)) {
  //   int64_t in_dim_padded = in_dim + block_size_ - in_dim % block_size_;
  //   int64_t out_dim_padded = out_dim + block_size_ - out_dim % block_size_;
  //   LOG_DEBUG("Padded input and output dimensions to {}x{}", in_dim,
  //   out_dim);

  //   padded = true;

  //   std::vector<int64_t> ld;
  //   if (a_shape.size() > 2) {
  //     ld = mat_a.sizes().slice(0, a_shape.size() - 2).vec();
  //   }

  //   // pad the matrices
  //   std::vector<int64_t> a_padded_sizes = ld;
  //   a_padded_sizes.push_back(in_dim_padded - in_dim);
  //   a_padded_sizes.push_back(h_dim);

  //   std::vector<int64_t> b_padded_sizes = ld;
  //   b_padded_sizes.push_back(h_dim);
  //   b_padded_sizes.push_back(out_dim_padded - out_dim);

  //   mat_a = torch::cat({mat_a, torch::zeros(a_padded_sizes)}, -2);
  //   mat_b = torch::cat({mat_b, torch::zeros(b_padded_sizes)}, -1);
  // }

  std::vector<std::future<void>> futures;
  for (int r = 0; r < in_dim / block_size_; ++r) {
    for (int c = 0; c < out_dim / block_size_; ++c) {
      auto func = [this, &mat_a, &mat_b, r, c]() {
        CallMatMulBlock(mat_a, mat_b, r, c);
      };
      LOG_DEBUG << "Dispatching block " << r << ", " << c;
      futures.push_back(
          std::async(std::launch::async, std::forward<decltype(func)>(func)));
    }
  }

  LOG_DEBUG << "Waiting for futures to finish";
  for (auto& future : futures) {
    future.wait();
  }

  // wait for output matrix to be filled
  LOG_DEBUG << "Waiting for output matrix to be filled";
  // std::unique_lock<std::mutex> lock(mutex_);
  // cv_.wait(lock, [this] { return request_map_.empty(); });
  while (!request_map_.empty()) {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }

  LOG_DEBUG << "Output matrix filled";
  return output_matrix_;
}

void AMQPBackend::CreateOutputMatrix(const torch::Tensor& mat_a,
                                     const torch::Tensor& mat_b) {
  auto a_shape = mat_a.sizes().vec();
  auto b_shape = mat_b.sizes().vec();

  LOG_DEBUG << "Creating output matrix, A shape: " << a_shape
            << ", B shape: " << b_shape;

  int64_t in_dim = a_shape[a_shape.size() - 2];
  int64_t out_dim = b_shape[b_shape.size() - 1];

  std::vector<int64_t> c_shape;
  if (a_shape.size() == 2 && b_shape.size() == 2) {
    c_shape = {in_dim, out_dim};
  } else if (a_shape.size() > 2 && b_shape.size() == 2) {
    c_shape.insert(c_shape.end(), a_shape.begin(), a_shape.end() - 2);
    c_shape.push_back(in_dim);
    c_shape.push_back(out_dim);
  } else {
    auto lda_shape = std::vector<int64_t>(a_shape.begin(), a_shape.end() - 2);
    auto ldb_shape = std::vector<int64_t>(b_shape.begin(), b_shape.end() - 2);
    if (lda_shape != ldb_shape) {
      throw std::runtime_error("Input dimensions must be the same");
    }
    c_shape = lda_shape;
    c_shape.push_back(in_dim);
    c_shape.push_back(out_dim);
  }

  output_matrix_ = torch::empty(c_shape);
  // fill with nan
  output_matrix_.fill_(std::nan(""));
  LOG_DEBUG << "Output matrix shape: " << output_matrix_.sizes().vec();
}

void AMQPBackend::HandleRsp() {
  // Consuming responses in a separate thread
  while (should_continue_) {
    amqp_rpc_reply_t res;
    amqp_envelope_t envelope;

    {
      // std::lock_guard<std::mutex> lock(mutex_);
      amqp_maybe_release_buffers(conn_);
      res = amqp_consume_message(conn_, &envelope, NULL, 0);

      if (AMQP_RESPONSE_NORMAL != res.reply_type) {
        continue;
      }
    }

    int channel = envelope.channel;
    std::string routing_key(reinterpret_cast<char*>(envelope.routing_key.bytes),
                            envelope.routing_key.len);

    assert(channel == req_channel_);
    assert(routing_key == "mm_response_queue");

    std::string response(static_cast<char*>(envelope.message.body.bytes),
                         envelope.message.body.len);
    MatMulResponseMessage response_message;
    response_message.Deserialize(response);

    LOG_DEBUG << "Received response, row: " << response_message.row
              << ", col: " << response_message.col;

    // Update output matrix
    int64_t r = response_message.row;
    int64_t c = response_message.col;
    auto ld = response_message.ld;

    int64_t offset_r = r * block_size_;
    int64_t offset_c = c * block_size_;

    auto out_shape = output_matrix_.sizes().vec();

    int64_t end_r =
        std::min(offset_r + block_size_, out_shape[out_shape.size() - 2]);
    int64_t end_c =
        std::min(offset_c + block_size_, out_shape[out_shape.size() - 1]);

    LOG_DEBUG << "Updating output matrix, row: " << r << ", col: " << c
              << ", with mat size " << response_message.mat.sizes().vec();

    {
      // std::lock_guard<std::mutex> lock(mutex_);
      LOG_DEBUG << "Updating output matrix, row: " << r << ", col: " << c;
      if (ld.size() > 0) {
        torch::Tensor ld_vec = torch::tensor(ld);
        // output_matrix_.index_put_(
        //     {ld_vec, torch::indexing::Slice(offset_r, end_r),
        //      torch::indexing::Slice(offset_c, end_c)},
        //     response_message.mat);
        LOG_DEBUG << "Updated output matrix, row: " << r << ", col: " << c;
      } else {
        // output_matrix_.index_put_({torch::indexing::Slice(offset_r, end_r),
        //                            torch::indexing::Slice(offset_c, end_c)},
        //                           response_message.mat);
        LOG_DEBUG << "Updated output matrix, row: " << r << ", col: " << c;
      }

      uint64_t req_key = (r << 32) | c;
      request_map_.erase(req_key);
      LOG_DEBUG << "Erased request, row: " << r << ", col: " << c;
      // amqp_basic_ack(conn_, rsp_channel_, envelope.delivery_tag, 0);
      amqp_destroy_envelope(&envelope);
    }

    // lock.unlock();
    cv_.notify_all();

    // lock.lock();
    // ack message

    // lock.unlock();
    LOG_DEBUG << "Acked message, row: " << r << ", col: " << c;
  }
}

void AMQPBackend::DispatchMatMulBlock(torch::Tensor mat_a, torch::Tensor mat_b,
                                      int64_t r, int64_t c,
                                      const std::vector<int64_t>& ld) {
  int64_t offset_r = r * block_size_;
  int64_t offset_c = c * block_size_;

  int64_t end_r = std::min(offset_r + block_size_, mat_a.size(0));
  int64_t end_c = std::min(offset_c + block_size_, mat_b.size(1));

  torch::Tensor a_rows =
      mat_a.index({torch::indexing::Slice(offset_r, end_r), "..."});
  torch::Tensor b_cols =
      mat_b.index({"...", torch::indexing::Slice(offset_c, end_c)});

  // dim of a_rows and b_cols should be 2, only keeps the last 2 dims
  if (a_rows.dim() > 2) {
    auto sizes = a_rows.sizes();
    a_rows = a_rows.reshape({sizes[sizes.size() - 2], sizes[sizes.size() - 1]});
  }

  if (b_cols.dim() > 2) {
    auto sizes = b_cols.sizes();
    b_cols = b_cols.reshape({sizes[sizes.size() - 2], sizes[sizes.size() - 1]});
  }

  uint64_t req_key = (r << 32) | c;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    request_map_[req_key] = std::make_shared<MatMulRequestMessage>();
    request_map_[req_key]->row = r;
    request_map_[req_key]->col = c;
    request_map_[req_key]->ld = ld;
    request_map_[req_key]->mat.emplace_back(a_rows);
    request_map_[req_key]->mat.emplace_back(b_cols);
    request_map_[req_key]->Serialize();

    LOG_DEBUG << "Dispatching message, row: " << r << ", col: " << c;
  }

  // Generate UUID
  std::string corr_id = GenUUID();

  // Prepare message properties
  amqp_basic_properties_t props;
  props._flags = AMQP_BASIC_CONTENT_TYPE_FLAG | AMQP_BASIC_DELIVERY_MODE_FLAG |
                 AMQP_BASIC_CORRELATION_ID_FLAG | AMQP_BASIC_REPLY_TO_FLAG;
  props.content_type = amqp_cstring_bytes("application/octet-stream");
  props.delivery_mode = 1;  // Non-persistent
  props.correlation_id = amqp_cstring_bytes(corr_id.c_str());
  props.reply_to = amqp_cstring_bytes("mm_response_queue");

  // Publish message
  std::lock_guard<std::mutex> lock(mutex_);
  std::string& message = request_map_[req_key]->serialized;
  amqp_bytes_t message_bytes;
  message_bytes.len = message.size();
  message_bytes.bytes = (void*)message.data();

  amqp_basic_publish(conn_, req_channel_, amqp_cstring_bytes(""),
                     amqp_cstring_bytes("mm_request_queue"), 0, 0, &props,
                     message_bytes);
  LOG_DEBUG << "Published message, row: " << r << ", col: " << c;
}

// //
// AMQPBackend::AMQPBackend(const std::string& host, int64_t block_size)
//     : host_(host),
//       block_size_(block_size),
//       conn_(nullptr),
//       channel_(100),
//       rsp_channel_(101),
//       should_continue_(true) {
//   Connect();
// }

// AMQPBackend::~AMQPBackend() {
//   should_continue_ = false;
//   //   if (consume_thread_.joinable()) {
//   //     consume_thread_.join();
//   //   }
//   disconnect();
// }

// void AMQPBackend::Connect() {
//   conn_ = amqp_new_connection();
//   socket_ = amqp_tcp_socket_new(conn_);
//   if (!socket_) {
//     throw std::runtime_error("Failed to create TCP socket");
//   }

//   int status = amqp_socket_open(socket_, host_.c_str(), 5672);
//   if (status) {
//     LOG_ERROR("Failed to open TCP socket, status: {}", status);
//     throw std::runtime_error("Failed to open TCP socket");
//   }

//   LOG_DEBUG("Opened TCP socket to AMQP server");
//   {
//     auto r = amqp_login(conn_, "/", 0, 131072, 0, AMQP_SASL_METHOD_PLAIN,
//                         "guest", "guest");
//     die_on_amqp_error(r, "Logging in to AMQP server");
//     LOG_DEBUG("Logged in to AMQP server");
//   }

//   {
//     auto* r = amqp_channel_open(conn_, channel_);
//     die_on_amqp_error(amqp_get_rpc_reply(conn_),
//                       "Opening channel 0 for requests");
//   }

//   {
//     auto* r = amqp_channel_open(conn_, rsp_channel_);
//     die_on_amqp_error(amqp_get_rpc_reply(conn_),
//                       "Opening channel 1 for responses");
//   }

//   // amqp_channel_open(conn_, channel_);
//   // amqp_get_rpc_reply(conn_);

//   // amqp_channel_open(conn_, rsp_channel_);
//   // amqp_get_rpc_reply(conn_);

//   LOG_DEBUG("Opened channel to AMQP server");
//   DeclareQueues();

//   LOG_DEBUG("Connected to AMQP server");
//   for (int i = 0; i < 10; i++) {
//     consume_threads_.emplace_back([this]() { ConsumeResponse(); });
//   }
// }

// void AMQPBackend::DeclareQueues() {
//   // Declare exchanges and queues
//   {
//     amqp_queue_declare_ok_t* r = amqp_queue_declare(
//         conn_, channel_, amqp_cstring_bytes("mm_request_queue"), 0, 0, 0, 0,
//         amqp_empty_table);
//     die_on_amqp_error(amqp_get_rpc_reply(conn_),
//                       "Declaring queue mm_request_queue");
//     // queuename = amqp_bytes_malloc_dup(r->queue);
//     // if (queuename.bytes == NULL) {
//     //   fprintf(stderr, "Out of memory while copying queue name");
//     // }
//     LOG_DEBUG("Declared request queue");
//   }

//   {
//     amqp_queue_declare_ok_t* r = amqp_queue_declare(
//         conn_, rsp_channel_, amqp_cstring_bytes("mm_response_queue"), 0, 0,
//         0, 0, amqp_empty_table);
//     die_on_amqp_error(amqp_get_rpc_reply(conn_),
//                       "Declaring queue mm_response_queue");
//     // queuename = amqp_bytes_malloc_dup(r->queue);
//     // if (queuename.bytes == NULL) {
//     //   fprintf(stderr, "Out of memory while copying queue name");
//     //   return 1;
//     // }
//     LOG_DEBUG("Declared response queue");
//   }
// }
