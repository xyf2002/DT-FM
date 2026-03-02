#pragma once

#include <cstddef>
#include <cstdint>

#include "server_base.h"

// Forward declaration to avoid including protobuf headers
namespace morphling {
class UMessage;
}

class SerializationBuffer;

// Message format constants and utilities
struct MessageFormat {
  static constexpr size_t PAYLOAD_SIZE_OFFSET = 0;
  static constexpr size_t PAYLOAD_SIZE_LENGTH = 4;
  static constexpr size_t PROTO_SIZE_OFFSET = 4;
  static constexpr size_t PROTO_SIZE_LENGTH = 4;
  static constexpr size_t TENSOR_SIZE_OFFSET = 8;
  static constexpr size_t TENSOR_SIZE_LENGTH = 8;
  static constexpr size_t HEADER_SIZE = 16;  // 4 + 4 + 8

  // Helper to reconstruct full wire format from separated proto and tensor data
  // Returns SerializationBufferPtr containing the reconstructed wire format
  // static SerializationBufferPtr ReconstructWireFormat(const void* proto_data,
  //                                                  size_t proto_size,
  //                                                  const void* tensor_data,
  //                                                  size_t tensor_size);
};

// Get the message type from a serialized UMessage payload
// This function parses the wire format and extracts the message_type field
// from the UMessage header.
//
// Wire format:
// [4 bytes: payload_size (network byte order)]
// [4 bytes: proto_size]
// [8 bytes: tensor_size]
// [proto_size bytes: protobuf UMessage]
// [tensor_size bytes: tensor data (if any)]
//
// Returns: message_type from UMessage.head().message_type(), or -1 on error
int32_t GetMessageType(const void* payload, size_t size);
