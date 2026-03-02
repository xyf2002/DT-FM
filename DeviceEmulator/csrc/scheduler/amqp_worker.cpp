#include "amqp_worker.h"

void AMQPWorker::HandleReq() {
  amqp_basic_consume(conn_, req_channel_,
                     amqp_cstring_bytes("mm_request_queue"), amqp_empty_bytes,
                     0, 1, 0, amqp_empty_table);
  while (should_continue_) {
    amqp_envelope_t envelope;
    amqp_maybe_release_buffers(conn_);

    amqp_rpc_reply_t res = amqp_consume_message(conn_, &envelope, nullptr, 0);
    if (AMQP_RESPONSE_NORMAL != res.reply_type) {
      // LOG_ERROR("Failed to consume message {}", (int)res.reply_type);
      continue;
    }
    std::string request(static_cast<char*>(envelope.message.body.bytes),
                        envelope.message.body.len);

    int channel = envelope.channel;
    std::string routing_key(reinterpret_cast<char*>(envelope.routing_key.bytes),
                            envelope.routing_key.len);

    assert(channel == req_channel_);
    assert(routing_key == "mm_request_queue");

    // get req_props form message
    amqp_basic_properties_t req_props = envelope.message.properties;

    LOG_DEBUG << "Received request: " << request.size();

    MatMulRequestMessage request_message;
    request_message.Deserialize(request);

    // call matmul block
    auto& mat_a = request_message.mat[0];
    auto& mat_b = request_message.mat[1];

    LOG_DEBUG << "Calling matmul block for row: " << request_message.row
              << ", col: " << request_message.col << ", mat sizes: a "
              << mat_a.sizes().vec() << ", b " << mat_b.sizes().vec();

    auto result =
        torch::mm(mat_a.to(DEFAULT_CUDA_DEVICE), mat_b.to(DEFAULT_CUDA_DEVICE))
            .to(CPU_DEVICE);
    {
      MatMulResponseMessage response_message;
      response_message.row = request_message.row;
      response_message.col = request_message.col;
      response_message.ld = request_message.ld;
      response_message.mat = result;

      LOG_DEBUG << "Sending response for row: " << request_message.row
                << ", col: " << request_message.col;

      std::string message_body = response_message.Serialize();
      amqp_bytes_t message_bytes;
      message_bytes.len = message_body.size();
      message_bytes.bytes = (void*)message_body.c_str();

      //   std::string req_corr_id =
      //       std::string(reinterpret_cast<char*>(req_props.correlation_id.bytes),
      //                   req_props.correlation_id.len);
      //   std::string req_reply_to =
      //       std::string(reinterpret_cast<char*>(req_props.reply_to.bytes),
      //                   req_props.reply_to.len);

      // Publish response back to the broker
      amqp_basic_properties_t props;
      props._flags =
          AMQP_BASIC_CORRELATION_ID_FLAG | AMQP_BASIC_DELIVERY_MODE_FLAG;
      props.correlation_id = req_props.correlation_id;
      props.delivery_mode = 1;  // non-persistent delivery mode

      amqp_basic_publish(conn_, rsp_channel_, amqp_cstring_bytes(""),
                         req_props.reply_to, 0, 0, &props, message_bytes);

      LOG_DEBUG << "Published response for row: " << request_message.row
                << ", col: " << request_message.col;
    }

    // ack message
    amqp_basic_ack(conn_, req_channel_, envelope.delivery_tag, 0);
    amqp_destroy_envelope(&envelope);
    LOG_DEBUG << "Acknowledged message for row: " << request_message.row
              << ", col: " << request_message.col;
  }
}
void AMQPWorker::HandleRsp() {}
