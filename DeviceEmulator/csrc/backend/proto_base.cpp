#include "proto_base.h"

#include <cstring>

#include "morphling.pb.h"
#include "utils/logger.h"

// SerializationBufferPtr MessageFormat::ReconstructWireFormat(
//     const void* proto_data, size_t proto_size, const void* tensor_data,
//     size_t tensor_size) {
//   size_t total_size = HEADER_SIZE + proto_size + tensor_size;

//   SerializationBuffer ser_buffer;
//   ser_buffer.Allocate(total_size);

//   uint32_t payload_size = proto_size + tensor_size;
//   ser_buffer.WriteUInt32(payload_size, true);  // network byte order
//   ser_buffer.WriteUInt32(proto_size, false);
//   ser_buffer.WriteUInt64(tensor_size);
//   ser_buffer.WriteBytes(proto_data, proto_size);
//   if (tensor_size > 0 && tensor_data) {
//     ser_buffer.WriteBytes(tensor_data, tensor_size);
//   }

//   return std::make_shared<SerializationBuffer>(std::move(ser_buffer));
// }

int32_t GetMessageType(const void* payload, size_t size) {
  if (payload == nullptr || size < 16) {
    return -1;
  }

  SerializationBuffer buffer(payload, size, false);

  // Read wire format header
  uint32_t payload_size = buffer.ReadUInt32(true);  // network byte order
  uint32_t proto_size = buffer.ReadUInt32(false);
  uint64_t tensor_size = buffer.ReadUInt64();

  // Validate proto size
  if (proto_size == 0 || proto_size > 100 * 1024 * 1024) {
    return -1;
  }

  // Parse protobuf message
  morphling::UMessage umsg;
  if (!umsg.ParseFromArray(buffer.GetCurrentPtr(), proto_size)) {
    return -1;
  }

  // Extract and return message type from header
  return umsg.head().message_type();
}
