#include "server_base.h"

#include <arpa/inet.h>
#include <sys/mman.h>

#include <cerrno>
#include <chrono>
#include <cstring>

#include "common/generator.h"
#include "global_api.pb.h"
#include "utils/logger.h"

// ============================================================================
// SerializationBuffer Implementation
// ============================================================================

SerializationBuffer::SerializationBuffer()
    : buffer_(nullptr),
      size_(0),
      offset_(0),
      owns_buffer_(false),
      pool_bucket_size_(0) {}

SerializationBuffer::SerializationBuffer(const void* data, size_t size,
                                         bool take_ownership)
    : buffer_(const_cast<uint8_t*>(static_cast<const uint8_t*>(data))),
      size_(size),
      offset_(0),
      owns_buffer_(take_ownership),
      pool_bucket_size_(0) {}

SerializationBuffer::SerializationBuffer(SerializationBuffer&& other) noexcept
    : buffer_(other.buffer_),
      size_(other.size_),
      offset_(other.offset_),
      owns_buffer_(other.owns_buffer_),
      pool_bucket_size_(other.pool_bucket_size_),
      pool_(other.pool_) {
  other.buffer_ = nullptr;
  other.size_ = 0;
  other.offset_ = 0;
  other.owns_buffer_ = false;
  other.pool_bucket_size_ = 0;
  other.pool_ = nullptr;
}

SerializationBuffer& SerializationBuffer::operator=(
    SerializationBuffer&& other) noexcept {
  if (this != &other) {
    FreeBuffer();
    buffer_ = other.buffer_;
    size_ = other.size_;
    offset_ = other.offset_;
    owns_buffer_ = other.owns_buffer_;
    pool_bucket_size_ = other.pool_bucket_size_;
    pool_ = other.pool_;
    other.buffer_ = nullptr;
    other.size_ = 0;
    other.offset_ = 0;
    other.owns_buffer_ = false;
    other.pool_bucket_size_ = 0;
    other.pool_ = nullptr;
  }
  return *this;
}

SerializationBuffer::~SerializationBuffer() { FreeBuffer(); }

void SerializationBuffer::FreeBuffer() {
  if (owns_buffer_ && buffer_) {
    if (pool_bucket_size_ > 0) {
      // Return to injected pool or singleton
      if (pool_) {
        pool_->Release(buffer_, pool_bucket_size_);
      } else {
        AlignedBufferPool::instance().Release(buffer_, pool_bucket_size_);
      }
    } else {
      free(buffer_);
    }
    buffer_ = nullptr;
    owns_buffer_ = false;
    pool_bucket_size_ = 0;
  }
}

void SerializationBuffer::Allocate(size_t size) {
  FreeBuffer();

  // Acquire from pool (page-aligned, mlocked)
  auto [ptr, bucket] = AlignedBufferPool::instance().Acquire(size);
  buffer_ = ptr;
  size_ = size;
  offset_ = 0;
  owns_buffer_ = true;
  pool_bucket_size_ = bucket;
}

void SerializationBuffer::Allocate(size_t size, AlignedBufferPool& pool) {
  FreeBuffer();

  auto [ptr, bucket] = pool.Acquire(size);
  buffer_ = ptr;
  size_ = size;
  offset_ = 0;
  owns_buffer_ = true;
  pool_bucket_size_ = bucket;
  pool_ = &pool;
}

void SerializationBuffer::WriteUInt32(uint32_t value, bool network_order) {
  if (network_order) {
    value = htonl(value);
  }
  memcpy(buffer_ + offset_, &value, sizeof(uint32_t));
  offset_ += sizeof(uint32_t);
}

void SerializationBuffer::WriteUInt64(uint64_t value) {
  memcpy(buffer_ + offset_, &value, sizeof(uint64_t));
  offset_ += sizeof(uint64_t);
}

void SerializationBuffer::WriteInt64(int64_t value) {
  memcpy(buffer_ + offset_, &value, sizeof(int64_t));
  offset_ += sizeof(int64_t);
}

void SerializationBuffer::WriteBytes(const void* data, size_t size) {
  if (size > 0 && data != nullptr) {
    // Use memcpy with optimization hints for large copies
    // For large buffers, memcpy should use SIMD instructions
    // Compiler will optimize this based on -O3 and -march=native flags
    memcpy(buffer_ + offset_, data, size);

    // Optional: Force memory to be loaded into cache for large copies
    // This helps with subsequent operations on the copied data
    // if (size > 1024 * 1024) {  // > 1 MB
    //   // Clflush hint to cache (compiler may optimize this away)
    //   // In practice, memcpy already does optimal caching
    // }
  }
  offset_ += size;
}

uint32_t SerializationBuffer::ReadUInt32(bool network_order) {
  uint32_t value;
  memcpy(&value, buffer_ + offset_, sizeof(uint32_t));
  offset_ += sizeof(uint32_t);
  return network_order ? ntohl(value) : value;
}

uint64_t SerializationBuffer::ReadUInt64() {
  uint64_t value;
  memcpy(&value, buffer_ + offset_, sizeof(uint64_t));
  offset_ += sizeof(uint64_t);
  return value;
}

int64_t SerializationBuffer::ReadInt64() {
  int64_t value;
  memcpy(&value, buffer_ + offset_, sizeof(int64_t));
  offset_ += sizeof(int64_t);
  return value;
}

void SerializationBuffer::ReadBytes(void* dest, size_t size) {
  memcpy(dest, buffer_ + offset_, size);
  offset_ += size;
}

const void* SerializationBuffer::GetCurrentPtr() const {
  return buffer_ + offset_;
}

void SerializationBuffer::SeekTo(size_t offset) { offset_ = offset; }

bool SerializationBuffer::CanRead(size_t bytes) const {
  return offset_ + bytes <= size_;
}

void SerializationBuffer::ValidateSize(size_t min_size) const {
  if (size_ < min_size) {
    throw std::runtime_error("Buffer size too small: " + std::to_string(size_) +
                             " < " + std::to_string(min_size));
  }
}

std::string SerializationBuffer::HexString(size_t length) const {
  size_t read_length = std::min(length, size_);
  return BinaryToHex(buffer_, read_length);
  // std::stringstream ss;
  // for (size_t i = 0; i < read_length; ++i) {
  //   ss << std::hex << std::setw(2) << std::setfill('0')
  //      << static_cast<int>(buffer_[i]);
  // }
  // return ss.str();
}

// ============================================================================
// Helper Functions
// ============================================================================

static void CreateMessageHeader(morphling::UMessage& umsg,
                                int32_t message_type) {
  auto* head = umsg.mutable_head();
  head->set_version(1);
  head->set_magic_flag(0x12340987);
  head->set_random_num(0);
  head->set_flow_no(0);
  head->set_session_no("");
  head->set_message_type(message_type);
}

// ============================================================================
// Matrix Operations
// ============================================================================

void IndexPutMatrixBlock(torch::Tensor& target, torch::Tensor& mat, int64_t r,
                         int64_t c, int64_t pivot, int64_t block_size) {
  // implement torch::index_put_ using memory copy
  void* target_ptr = target.data_ptr();
  void* mat_ptr = mat.data_ptr();

  auto mat_shape = mat.sizes().vec();
  auto target_shape = target.sizes().vec();

  int64_t mat_n_rows = mat_shape[mat_shape.size() - 2];
  int64_t mat_n_cols = mat_shape[mat_shape.size() - 1];

  int64_t elem_size = mat.element_size();

  int64_t in_dim = target_shape[target_shape.size() - 2];
  int64_t out_dim = target_shape[target_shape.size() - 1];

  int64_t offset_r = r * block_size * out_dim * elem_size;
  int64_t offset_c = c * block_size * elem_size;
  int64_t target_offset =
      pivot * in_dim * out_dim * elem_size + offset_r + offset_c;

  for (int64_t i = 0; i < mat_n_rows; ++i) {
    int64_t mat_row_offset = i * mat_n_cols * elem_size;
    int64_t target_row_offset = target_offset + i * out_dim * elem_size;
    // LOG_DEBUG("IndexPutMatrixBlock, target_row_offset: {}, mat_row_offset:
    // {}",
    //           target_row_offset, mat_row_offset);
    memcpy((char*)target_ptr + target_row_offset,
           (char*)mat_ptr + mat_row_offset, mat_n_cols * elem_size);
  }
}

torch::Tensor CreateOutputMatrix(const torch::Tensor& mat_a,
                                 const torch::Tensor& mat_b) {
  auto a_shape = mat_a.sizes().vec();
  auto b_shape = mat_b.sizes().vec();

  // assume b needs to be transposed
  int64_t in_dim = a_shape[a_shape.size() - 2];
  int64_t out_dim = b_shape[b_shape.size() - 2];

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

  // LOG_DEBUG(
  //     "Creating output matrix, A shape: {}, B shape: {}, Output shape: {}",
  //     a_shape, b_shape, c_shape);

  auto output_matrix = torch::empty(c_shape);
  // fill with nan
  output_matrix.fill_(std::nan(""));

  return output_matrix;
}

void UpdateMatrixBlock(torch::Tensor& target, torch::Tensor& mat, int64_t r,
                       int64_t c, int64_t pivot, int64_t block_size) {
  auto target_shape = target.sizes().vec();
  auto mat_shape = mat.sizes().vec();

  // LOG_DEBUG("Updating matrix block, target shape: {}, mat shape: {}",
  //           target_shape, mat_shape);

  IndexPutMatrixBlock(target, mat, r, c, pivot, block_size);

  // if (target_shape.size() == 2) {
  //   // no need to reshape
  //   auto offset_r = r * block_size;
  //   auto offset_c = c * block_size;

  //   auto end_r = std::min(offset_r + block_size, target_shape[0]);
  //   auto end_c = std::min(offset_c + block_size, target_shape[1]);

  //   target.index_put_({torch::indexing::Slice(offset_r, end_r),
  //                      torch::indexing::Slice(offset_c, end_c)},
  //                     mat);
  // } else {
  //   auto num_ld = target_shape.size() - 2;
  //   auto offset_r = r * block_size;
  //   auto offset_c = c * block_size;

  //   auto end_r =
  //       std::min(offset_r + block_size, target_shape[target_shape.size() -
  //       2]);
  //   auto end_c =
  //       std::min(offset_c + block_size, target_shape[target_shape.size() -
  //       1]);

  //   if (num_ld == 1) {
  //     LOG_DEBUG(
  //         "Updating matrix block, offset_r: {}, end_r: {}, offset_c: {}, "
  //         "end_c: {}",
  //         offset_r, end_r, offset_c, end_c);
  //     target.index_put_({pivot, torch::indexing::Slice(offset_r, end_r),
  //                        torch::indexing::Slice(offset_c, end_c)},
  //                       mat);
  //   } else {
  //     auto ld_combinations = CartesianProduct(
  //         std::vector<int64_t>(target_shape.begin(), target_shape.end() -
  //         2));
  //     auto ld_vec = torch::tensor(ld_combinations[pivot]);
  //     target.index_put_({ld_vec, torch::indexing::Slice(offset_r, end_r),
  //                        torch::indexing::Slice(offset_c, end_c)},
  //                       mat);
  //   }
  // }
  // LOG_DEBUG("Updated matrix block, r: {}, c: {}, pivot: {}", r, c, pivot);
}

MatrixPartitionPtr CalculateMatrixPartition(const torch::Tensor& mat_a,
                                            const torch::Tensor& mat_b,
                                            int64_t r, int64_t c, int64_t pivot,
                                            int64_t block_size) {
  auto a_shape = mat_a.sizes().vec();
  auto b_shape = mat_b.sizes().vec();

  int64_t in_dim = a_shape[a_shape.size() - 2];
  int64_t h_dim = a_shape[a_shape.size() - 1];
  int64_t out_dim = b_shape[b_shape.size() - 2];

  int64_t offset_r =
      (r * block_size + pivot * in_dim) * h_dim * mat_a.element_size();
  int64_t offset_c = (c * block_size +
                      (mat_b.sizes().vec().size() > 2 ? pivot : 0) * out_dim) *
                     h_dim * mat_b.element_size();

  // fprintf(stderr, "offset_r: %ld, offset_c: %ld, r: %ld, c: %ld, pivot:
  // %ld\n",
  //         offset_r, offset_c, r, c, pivot);
  void* offset_r_ptr = (char*)mat_a.data_ptr() + offset_r;
  void* offset_c_ptr = (char*)mat_b.data_ptr() + offset_c;

  // int64_t a_bytes = in_dim * h_dim * mat_a.element_size();
  // int64_t b_bytes = h_dim * out_dim * mat_b.element_size();

  int64_t size_r = std::min(block_size, in_dim - r * block_size) * h_dim *
                   mat_a.element_size();
  int64_t size_c = std::min(block_size, out_dim - c * block_size) * h_dim *
                   mat_b.element_size();

  // fprintf(stderr, "size_r: %ld, size_c: %ld, a_bytes: %ld, b_bytes: %ld\n",
  // size_r, size_c, a_bytes, b_bytes);

  auto partition = std::make_shared<MatrixPartition>();
  // partition->version = 0;  // need to set version
  // partition->oid = -1;     // need to set oid
  partition->row = r;
  partition->col = c;
  partition->h_dim = h_dim;
  partition->pivot = pivot;
  // partition->dev_id = -1;
  partition->timestamp = CurrentTimeMicros();
  partition->mat.push_back({offset_r_ptr, size_r});
  partition->mat.push_back({offset_c_ptr, size_c});
  // partition->block_size = block_size;

  return partition;
}

std::vector<MatrixPartitionPtr> PartitionMatrices(const torch::Tensor& mat_a,
                                                  const torch::Tensor& mat_b,
                                                  int64_t block_size) {
  auto a_shape = mat_a.sizes().vec();
  auto b_shape = mat_b.sizes().vec();

  assert(a_shape.size() >= b_shape.size());

  int64_t in_dim = a_shape[a_shape.size() - 2];
  int64_t h_dim = a_shape[a_shape.size() - 1];
  int64_t out_dim = b_shape[b_shape.size() - 2];

  std::vector<MatrixPartitionPtr> partitions;
  auto uuid64 = GenUUID64();

  int64_t num_block_rows = in_dim / block_size + (in_dim % block_size != 0);
  int64_t num_block_cols = out_dim / block_size + (out_dim % block_size != 0);

  for (int r = 0; r < num_block_rows; ++r) {
    for (int c = 0; c < num_block_cols; ++c) {
      int64_t pivot = 0;
      if (mat_a.dim() == 2 && mat_b.dim() == 2) {
        auto partition =
            CalculateMatrixPartition(mat_a, mat_b, r, c, pivot, block_size);
        partition->version = uuid64;
        partitions.push_back(partition);
      } else {
        auto ld_combinations = CartesianProduct(
            std::vector<int64_t>(a_shape.begin(), a_shape.end() - 2));
        for (const auto ld : ld_combinations) {
          auto partition =
              CalculateMatrixPartition(mat_a, mat_b, r, c, pivot, block_size);
          partition->version = uuid64;
          partitions.push_back(partition);
          // fprintf(stderr, "r: %d, c: %d, pivot: %ld\n", r, c, pivot);
          pivot++;
        }
      }
    }
  }

  return partitions;
}

// ============================================================================
// MatrixPartition Implementation
// ============================================================================

int32_t MatrixPartition::GetMessageType() const {
  return morphling::global_api::COMPUTE_GEMM_DATA;
}

SerializationBufferPtr MatrixPartition::Serialize(
    SerializationFormat format) const {
  switch (format) {
    case SerializationFormat::PROTOBUF:
      return SerializeProto();
    default:
      throw std::runtime_error("Unsupported serialization format");
  }
}

void MatrixPartition::Deserialize(const void* data, size_t size,
                                  SerializationFormat format) {
  switch (format) {
    case SerializationFormat::PROTOBUF:
      DeserializeProto(data, size);
      break;
    default:
      throw std::runtime_error("Unsupported serialization format");
  }
}

void MatrixPartition::WriteMetadataToBuffer(SerializationBuffer& buffer) const {
  buffer.WriteUInt64(version);
  buffer.WriteInt64(row);
  buffer.WriteInt64(col);
  buffer.WriteInt64(pivot);
  buffer.WriteInt64(h_dim);
  buffer.WriteInt64(dev_id);
  buffer.WriteInt64(oid);
  buffer.WriteUInt64(timestamp);
}

void MatrixPartition::ReadMetadataFromBuffer(SerializationBuffer& buffer) {
  version = buffer.ReadUInt64();
  row = buffer.ReadInt64();
  col = buffer.ReadInt64();
  pivot = buffer.ReadInt64();
  h_dim = buffer.ReadInt64();
  dev_id = buffer.ReadInt64();
  oid = buffer.ReadInt64();
  timestamp = buffer.ReadUInt64();
}

void MatrixPartition::ReadMatricesData(SerializationBuffer& buffer,
                                       size_t end_offset) {
  int mat_count = 0;
  while (buffer.GetOffset() < end_offset) {
    if (!buffer.CanRead(sizeof(int64_t))) {
      LOG_ERROR << "Not enough data for mat_size at offset="
                << buffer.GetOffset();
      break;
    }

    int64_t mat_size = buffer.ReadInt64();

    if (mat_size == 0) {
      mat.push_back({nullptr, 0});
      mat_count++;
      continue;
    }

    if (mat_size < 0 || !buffer.CanRead(mat_size)) {
      LOG_ERROR << "Invalid mat_size=" << mat_size;
      break;
    }

    mat.push_back({const_cast<void*>(buffer.GetCurrentPtr()),
                   static_cast<size_t>(mat_size)});
    buffer.SeekTo(buffer.GetOffset() + mat_size);
    mat_count++;
  }
}

SerializationBufferPtr MatrixPartition::SerializeProto() const {
  auto start_total = std::chrono::high_resolution_clock::now();

  // ========== Stage 1: Create and populate protobuf message ==========
  auto start_stage1 = std::chrono::high_resolution_clock::now();

  morphling::UMessage umsg;
  CreateMessageHeader(umsg, morphling::global_api::COMPUTE_GEMM_DATA);

  auto* body = umsg.mutable_body();
  auto* gemm_data =
      body->MutableExtension(morphling::global_api::compute_gemm_data);

  gemm_data->set_version(version);
  gemm_data->set_row(row);
  gemm_data->set_col(col);
  gemm_data->set_pivot(pivot);
  gemm_data->set_h_dim(h_dim);
  gemm_data->set_dev_id(dev_id);
  gemm_data->set_oid(oid);
  gemm_data->set_gemm_id(gemm_id);
  gemm_data->set_timestamp(timestamp);

  for (const auto& m : mat) {
    auto* mat_payload = gemm_data->add_matrices();
    mat_payload->set_offset(0);
    mat_payload->set_size(std::get<1>(m));
  }

  auto end_stage1 = std::chrono::high_resolution_clock::now();
  auto duration_stage1 = std::chrono::duration_cast<std::chrono::microseconds>(
                             end_stage1 - start_stage1)
                             .count();
  LOG_DEBUG << "[Stage 1] Create protobuf message: " << duration_stage1
            << " us";

  // ========== Stage 2: Serialize protobuf and calculate sizes ==========
  auto start_stage2 = std::chrono::high_resolution_clock::now();

  std::string proto_str = umsg.SerializeAsString();
  uint32_t proto_size = proto_str.size();

  uint64_t tensor_size = 0;
  for (const auto& m : mat) {
    tensor_size += std::get<1>(m);
  }

  uint32_t payload_size =
      sizeof(proto_size) + sizeof(tensor_size) + proto_size + tensor_size;
  uint64_t total_size = sizeof(payload_size) + payload_size;

  auto end_stage2 = std::chrono::high_resolution_clock::now();
  auto duration_stage2 = std::chrono::duration_cast<std::chrono::microseconds>(
                             end_stage2 - start_stage2)
                             .count();
  LOG_DEBUG << "[Stage 2] Serialize protobuf and calculate sizes: "
            << duration_stage2 << " us (proto_size=" << proto_size
            << ", tensor_size=" << tensor_size << ")";

  // ========== Stage 3: Allocate buffer ==========
  auto start_stage3 = std::chrono::high_resolution_clock::now();

  SerializationBufferPtr buffer = std::make_shared<SerializationBuffer>();
  buffer->Allocate(total_size);

  auto end_stage3 = std::chrono::high_resolution_clock::now();
  auto duration_stage3 = std::chrono::duration_cast<std::chrono::microseconds>(
                             end_stage3 - start_stage3)
                             .count();
  LOG_DEBUG << "[Stage 3] Allocate buffer: " << duration_stage3
            << " us (total_size=" << total_size << ")";

  // ========== Stage 4: Write data to buffer ==========
  auto start_stage4 = std::chrono::high_resolution_clock::now();

  // Write headers
  auto t_header_start = std::chrono::high_resolution_clock::now();
  buffer->WriteUInt32(payload_size, true);  // network byte order
  buffer->WriteUInt32(proto_size, false);
  buffer->WriteUInt64(tensor_size);
  auto t_header_end = std::chrono::high_resolution_clock::now();
  auto header_time = std::chrono::duration_cast<std::chrono::microseconds>(
                         t_header_end - t_header_start)
                         .count();

  // Write proto data
  auto t_proto_start = std::chrono::high_resolution_clock::now();
  buffer->WriteBytes(proto_str.data(), proto_size);
  auto t_proto_end = std::chrono::high_resolution_clock::now();
  auto proto_copy_time = std::chrono::duration_cast<std::chrono::microseconds>(
                             t_proto_end - t_proto_start)
                             .count();
  double proto_throughput =
      proto_size > 0
          ? (proto_size / static_cast<double>(proto_copy_time) * 1e6 / 1e9)
          : 0;  // GB/s

  // Write tensor data - optimized with direct memcpy
  auto t_tensor_start = std::chrono::high_resolution_clock::now();
  uint64_t total_tensor_copied = 0;
  uint8_t* dest_ptr =
      static_cast<uint8_t*>(buffer->GetBuffer()) + buffer->GetOffset();

  int mat_idx = 0;
  for (const auto& m : mat) {
    const size_t size = std::get<1>(m);
    const void* src = std::get<0>(m);

    // Direct memcpy without function call overhead
    if (size > 0 && src != nullptr) {
      auto copy_start = std::chrono::high_resolution_clock::now();
      memcpy(dest_ptr, src, size);
      auto copy_end = std::chrono::high_resolution_clock::now();
      auto copy_time = std::chrono::duration_cast<std::chrono::microseconds>(
                           copy_end - copy_start)
                           .count();
      double copy_throughput =
          size > 0 ? (size / static_cast<double>(copy_time) * 1e6 / 1e9)
                   : 0;  // GB/s

      LOG_DEBUG << "[Memcpy #" << mat_idx << "] size=" << size << " bytes ("
                << (size / 1024.0) << " KB), time=" << copy_time
                << " us, TP=" << copy_throughput << " GB/s, src=" << src
                << " -> dest=" << (void*)dest_ptr;

      dest_ptr += size;
      total_tensor_copied += size;
    }
    mat_idx++;
  }

  // Update buffer offset once at the end
  buffer->SeekTo(buffer->GetOffset() + total_tensor_copied);
  auto t_tensor_end = std::chrono::high_resolution_clock::now();
  auto tensor_copy_time = std::chrono::duration_cast<std::chrono::microseconds>(
                              t_tensor_end - t_tensor_start)
                              .count();
  double tensor_throughput =
      total_tensor_copied > 0
          ? (total_tensor_copied / static_cast<double>(tensor_copy_time) * 1e6 /
             1e9)
          : 0;  // GB/s

  auto end_stage4 = std::chrono::high_resolution_clock::now();
  auto duration_stage4 = std::chrono::duration_cast<std::chrono::microseconds>(
                             end_stage4 - start_stage4)
                             .count();

  LOG_DEBUG << "[Stage 4.1] Write headers: " << header_time << " us";
  LOG_DEBUG << "[Stage 4.2] Write proto data: " << proto_copy_time << " us ("
            << proto_size << " bytes, TP: " << proto_throughput << " GB/s)";
  LOG_DEBUG << "[Stage 4.3] Write tensor data: " << tensor_copy_time << " us ("
            << total_tensor_copied << " bytes, TP: " << tensor_throughput
            << " GB/s)";
  LOG_DEBUG << "[Stage 4] Write data to buffer: " << duration_stage4 << " us";

  auto end_total = std::chrono::high_resolution_clock::now();
  auto duration_total = std::chrono::duration_cast<std::chrono::microseconds>(
                            end_total - start_total)
                            .count();
  double total_throughput =
      total_size > 0
          ? (total_size / static_cast<double>(duration_total) * 1e6 / 1e9)
          : 0;  // GB/s
  LOG_DEBUG << "[SerializeProto] Total time: " << duration_total
            << " us (Stage1=" << duration_stage1
            << ", Stage2=" << duration_stage2 << ", Stage3=" << duration_stage3
            << ", Stage4=" << duration_stage4
            << ") | Overall TP: " << total_throughput << " GB/s";

  return buffer;
}

void MatrixPartition::DeserializeProto(const void* data, size_t size) {
  LOG_INFO << "DeserializeProto called: data=" << data << ", size=" << size;

  if (data == nullptr) {
    LOG_ERROR << "DeserializeProto: data pointer is nullptr!";
    throw std::runtime_error("DeserializeProto: data pointer is nullptr");
  }

  if (size < 16) {
    LOG_ERROR << "DeserializeProto: size too small, size=" << size;
    throw std::runtime_error("DeserializeProto: size too small");
  }

  SerializationBuffer buffer(data, size, false);

  // Read header
  uint32_t payload_size = buffer.ReadUInt32(true);  // network byte order
  uint32_t proto_size = buffer.ReadUInt32(false);
  uint64_t tensor_size = buffer.ReadUInt64();

  LOG_DEBUG << "Read header: proto_size=" << proto_size
            << ", tensor_size=" << tensor_size;

  // Validate sizes
  if (proto_size == 0 || proto_size > 100 * 1024 * 1024) {
    LOG_FATAL << "Invalid proto_size=" << proto_size;
  }

  if (tensor_size > 1ull * 1024 * 1024 * 1024) {
    LOG_FATAL << "Invalid tensor_size=" << tensor_size;
  }

  if (buffer.GetOffset() + proto_size + tensor_size > size) {
    LOG_ERROR << "Total size mismatch: "
              << (buffer.GetOffset() + proto_size + tensor_size) << " > "
              << size;
    throw std::runtime_error("Total size exceeds provided buffer");
  }

  // Parse proto
  LOG_DEBUG << "Parsing protobuf, proto_size=" << proto_size;
  morphling::UMessage umsg;
  if (!umsg.ParseFromArray(buffer.GetCurrentPtr(), proto_size)) {
    LOG_ERROR << "Failed to parse UMessage from proto";
    throw std::runtime_error("Failed to parse UMessage from proto");
  }
  buffer.SeekTo(buffer.GetOffset() + proto_size);
  LOG_DEBUG << "Protobuf parsed successfully";

  // Extract message
  const auto& body = umsg.body();
  if (!body.HasExtension(morphling::global_api::compute_gemm_data)) {
    LOG_ERROR << "UMessage does not contain compute_gemm_data";
    throw std::runtime_error("UMessage does not contain compute_gemm_data");
  }

  const auto& gemm_data =
      body.GetExtension(morphling::global_api::compute_gemm_data);
  LOG_DEBUG << "ComputeGemmData extracted";

  // Fill fields
  version = gemm_data.version();
  row = gemm_data.row();
  col = gemm_data.col();
  pivot = gemm_data.pivot();
  h_dim = gemm_data.h_dim();
  dev_id = gemm_data.dev_id();
  oid = gemm_data.oid();
  gemm_id = gemm_data.gemm_id();
  timestamp = gemm_data.timestamp();

  LOG_INFO << "Partition fields: version=" << version << ", row=" << row
           << ", col=" << col << ", pivot=" << pivot << ", h_dim=" << h_dim
           << ", gemm_id=" << gemm_id;

  // Extract tensor data pointers
  mat.clear();
  for (const auto& payload : gemm_data.matrices()) {
    LOG_DEBUG << "Matrix payload: size=" << payload.size();
    mat.push_back({const_cast<void*>(buffer.GetCurrentPtr()),
                   static_cast<size_t>(payload.size())});
    buffer.SeekTo(buffer.GetOffset() + payload.size());
  }

  ptr_ = buffer.GetBuffer();
  size_ = size;

  LOG_INFO << "DeserializeProto completed: parsed " << mat.size()
           << " matrices";
}

// ============================================================================
// DeviceRegisterRequest Implementation
// ============================================================================

int32_t DeviceRegisterRequest::GetMessageType() const {
  return morphling::global_api::DEVICE_REGISTER_REQUEST;
}

SerializationBufferPtr DeviceRegisterRequest::Serialize(
    SerializationFormat format) const {
  if (format != SerializationFormat::PROTOBUF) {
    throw std::runtime_error(
        "DeviceRegisterRequest only supports PROTOBUF format");
  }
  return SerializeProto();
}

void DeviceRegisterRequest::Deserialize(const void* data, size_t size,
                                        SerializationFormat format) {
  if (format != SerializationFormat::PROTOBUF) {
    throw std::runtime_error(
        "DeviceRegisterRequest only supports PROTOBUF format");
  }
  DeserializeProto(data, size);
}

SerializationBufferPtr DeviceRegisterRequest::SerializeProto() const {
  // Create UMessage
  morphling::UMessage umsg;
  CreateMessageHeader(umsg, morphling::global_api::DEVICE_REGISTER_REQUEST);

  auto* body = umsg.mutable_body();
  auto* request_msg =
      body->MutableExtension(morphling::global_api::device_resgister_request);
  // Request is empty

  // Serialize (no tensor data for DeviceRegisterRequest)
  std::string proto_str = umsg.SerializeAsString();
  uint32_t proto_size = proto_str.size();
  uint64_t tensor_size = 0;

  uint32_t payload_size = sizeof(proto_size) + sizeof(tensor_size) + proto_size;
  uint64_t total_size = sizeof(payload_size) + payload_size;

  SerializationBufferPtr buffer = std::make_shared<SerializationBuffer>();
  buffer->Allocate(total_size);

  buffer->WriteUInt32(payload_size, true);
  buffer->WriteUInt32(proto_size, false);
  buffer->WriteUInt64(tensor_size);
  buffer->WriteBytes(proto_str.data(), proto_size);

  LOG_DEBUG << "DeviceRegisterRequest SerializeProto completed: total_size="
            << total_size << ", payload_size=" << payload_size
            << ", proto_size=" << proto_size << ", tensor_size=" << tensor_size;

  LOG_DEBUG << "Serialized DeviceRegisterRequest (full): "
            << BinaryToHex(static_cast<const uint8_t*>(buffer->GetBuffer()),
                           buffer->GetSize());

  return buffer;
}

void DeviceRegisterRequest::DeserializeProto(const void* data, size_t size) {
  if (data == nullptr || size < 16) {
    throw std::runtime_error(
        "DeviceRegisterRequest::DeserializeProto: invalid input");
  }

  SerializationBuffer buffer(data, size, false);

  uint32_t payload_size = buffer.ReadUInt32(true);
  uint32_t proto_size = buffer.ReadUInt32(false);
  uint64_t tensor_size = buffer.ReadUInt64();

  if (proto_size == 0 || proto_size > 100 * 1024 * 1024) {
    LOG_FATAL << "Invalid proto_size=" << proto_size;
  }

  morphling::UMessage umsg;
  if (!umsg.ParseFromArray(buffer.GetCurrentPtr(), proto_size)) {
    throw std::runtime_error("Failed to parse UMessage");
  }

  const auto& body = umsg.body();
  if (!body.HasExtension(morphling::global_api::device_resgister_request)) {
    throw std::runtime_error(
        "UMessage does not contain device_resgister_request");
  }

  // Request is empty, nothing to extract
}

// ============================================================================
// DeviceProfileData Implementation
// ============================================================================

int32_t DeviceProfileData::GetMessageType() const {
  return morphling::global_api::DEVICE_PROFILE_DATA;
}

std::string DeviceProfileData::DebugString() const {
  std::ostringstream oss;
  oss << "uuid: " << uuid << ", flops: " << flops << ", memory: " << memory
      << ", ul_bw: " << ul_bw << ", dl_bw: " << dl_bw << ", ul_lat: " << ul_lat
      << ", dl_lat: " << dl_lat;
  return oss.str();
}

SerializationBufferPtr DeviceProfileData::Serialize(
    SerializationFormat format) const {
  if (format != SerializationFormat::PROTOBUF) {
    throw std::runtime_error("DeviceProfileData only supports PROTOBUF format");
  }
  return SerializeProto();
}

void DeviceProfileData::Deserialize(const void* data, size_t size,
                                    SerializationFormat format) {
  if (format != SerializationFormat::PROTOBUF) {
    throw std::runtime_error("DeviceProfileData only supports PROTOBUF format");
  }
  DeserializeProto(data, size);
}

SerializationBufferPtr DeviceProfileData::SerializeProto() const {
  // Create UMessage
  morphling::UMessage umsg;
  CreateMessageHeader(umsg, morphling::global_api::DEVICE_PROFILE_DATA);

  auto* body = umsg.mutable_body();
  auto* profile_msg =
      body->MutableExtension(morphling::global_api::device_profile_data);

  profile_msg->set_uuid(uuid);
  profile_msg->set_flops(flops);
  profile_msg->set_memory(memory);
  profile_msg->set_ul_bw(ul_bw);
  profile_msg->set_dl_bw(dl_bw);
  profile_msg->set_ul_lat(ul_lat);
  profile_msg->set_dl_lat(dl_lat);

  // Serialize (no tensor data for DeviceProfileData)
  std::string proto_str = umsg.SerializeAsString();
  uint32_t proto_size = proto_str.size();
  uint64_t tensor_size = 0;

  uint32_t payload_size = sizeof(proto_size) + sizeof(tensor_size) + proto_size;
  uint64_t total_size = sizeof(payload_size) + payload_size;

  SerializationBufferPtr buffer = std::make_shared<SerializationBuffer>();
  buffer->Allocate(total_size);

  buffer->WriteUInt32(payload_size, true);
  buffer->WriteUInt32(proto_size, false);
  buffer->WriteUInt64(tensor_size);
  buffer->WriteBytes(proto_str.data(), proto_size);

  return buffer;
}

void DeviceProfileData::DeserializeProto(const void* data, size_t size) {
  if (data == nullptr || size < 16) {
    throw std::runtime_error(
        "DeviceProfileData::DeserializeProto: invalid input");
  }

  SerializationBuffer buffer(data, size, false);

  uint32_t payload_size = buffer.ReadUInt32(true);
  uint32_t proto_size = buffer.ReadUInt32(false);
  uint64_t tensor_size = buffer.ReadUInt64();

  if (proto_size == 0 || proto_size > 100 * 1024 * 1024) {
    LOG_FATAL << "Invalid proto_size=" << proto_size;
  }

  morphling::UMessage umsg;
  if (!umsg.ParseFromArray(buffer.GetCurrentPtr(), proto_size)) {
    throw std::runtime_error("Failed to parse UMessage");
  }

  const auto& body = umsg.body();
  if (!body.HasExtension(morphling::global_api::device_profile_data)) {
    throw std::runtime_error("UMessage does not contain device_profile_data");
  }

  const auto& profile_msg =
      body.GetExtension(morphling::global_api::device_profile_data);

  uuid = profile_msg.uuid();
  flops = profile_msg.flops();
  memory = profile_msg.memory();
  ul_bw = profile_msg.ul_bw();
  dl_bw = profile_msg.dl_bw();
  ul_lat = profile_msg.ul_lat();
  dl_lat = profile_msg.dl_lat();
}

// ============================================================================
// Utility Functions
// ============================================================================

// ============================================================================
// Other MatrixPartition methods
// ============================================================================

std::string MatrixPartition::DebugString() const {
  std::ostringstream oss;
  oss << "v: " << version << ", r: " << row << ", c: " << col
      << ", p: " << pivot << ", h: " << h_dim << ", dev_id: " << dev_id
      << ", oid: " << oid;

  for (const auto& mat_data : mat) {
    oss << ", m_size: " << std::get<1>(mat_data);
  }
  return oss.str();
}

// ============================================================================
// ScatterGatherBuffer Implementation
// ============================================================================

ScatterGatherBuffer::~ScatterGatherBuffer() {
  for (auto& [ptr, bucket] : owned_pool_entries_) {
    pool_->Release(ptr, bucket);
  }
}

ScatterGatherBuffer::ScatterGatherBuffer(ScatterGatherBuffer&& other) noexcept
    : segments_(std::move(other.segments_)),
      owned_pool_entries_(std::move(other.owned_pool_entries_)),
      pool_(other.pool_) {
  other.pool_ = &AlignedBufferPool::instance();
}

ScatterGatherBuffer& ScatterGatherBuffer::operator=(
    ScatterGatherBuffer&& other) noexcept {
  if (this != &other) {
    // Free current owned entries
    for (auto& [ptr, bucket] : owned_pool_entries_) {
      pool_->Release(ptr, bucket);
    }
    segments_ = std::move(other.segments_);
    owned_pool_entries_ = std::move(other.owned_pool_entries_);
    pool_ = other.pool_;
    other.pool_ = &AlignedBufferPool::instance();
  }
  return *this;
}

void ScatterGatherBuffer::AddOwnedSegment(uint8_t* data, size_t size,
                                          size_t pool_bucket) {
  segments_.emplace_back(data, size, true);
  owned_pool_entries_.emplace_back(data, pool_bucket);
}

void ScatterGatherBuffer::AddReferenceSegment(const void* data, size_t size) {
  if (data && size > 0) {
    segments_.emplace_back(data, size, false);
  }
}

size_t ScatterGatherBuffer::GetTotalSize() const {
  size_t total = 0;
  for (const auto& seg : segments_) {
    total += seg.size;
  }
  return total;
}

// ============================================================================
// MatrixPartition::SerializeZeroCopy Implementation
// ============================================================================

ScatterGatherBufferPtr MatrixPartition::SerializeZeroCopy() const {
  // Create protobuf message (same as SerializeProto)
  morphling::UMessage umsg;
  CreateMessageHeader(umsg, morphling::global_api::COMPUTE_GEMM_DATA);

  auto* body = umsg.mutable_body();
  auto* gemm_data =
      body->MutableExtension(morphling::global_api::compute_gemm_data);

  gemm_data->set_version(version);
  gemm_data->set_row(row);
  gemm_data->set_col(col);
  gemm_data->set_pivot(pivot);
  gemm_data->set_h_dim(h_dim);
  gemm_data->set_dev_id(dev_id);
  gemm_data->set_oid(oid);
  gemm_data->set_gemm_id(gemm_id);
  gemm_data->set_timestamp(timestamp);

  for (const auto& m : mat) {
    auto* mat_payload = gemm_data->add_matrices();
    mat_payload->set_offset(0);
    mat_payload->set_size(std::get<1>(m));
  }

  std::string proto_str = umsg.SerializeAsString();
  uint32_t proto_size = proto_str.size();

  uint64_t tensor_size = 0;
  for (const auto& m : mat) {
    tensor_size += std::get<1>(m);
  }

  uint32_t payload_size =
      sizeof(proto_size) + sizeof(tensor_size) + proto_size + tensor_size;

  // Header = 4 (payload_size) + 4 (proto_size) + 8 (tensor_size) + proto_data
  size_t header_size = sizeof(payload_size) + sizeof(proto_size) +
                       sizeof(tensor_size) + proto_size;

  // Allocate header buffer from pool
  auto [header_ptr, header_bucket] =
      AlignedBufferPool::instance().Acquire(header_size);

  // Write header into owned buffer
  size_t off = 0;
  uint32_t payload_size_n = htonl(payload_size);
  memcpy(header_ptr + off, &payload_size_n, sizeof(uint32_t));
  off += sizeof(uint32_t);
  memcpy(header_ptr + off, &proto_size, sizeof(uint32_t));
  off += sizeof(uint32_t);
  memcpy(header_ptr + off, &tensor_size, sizeof(uint64_t));
  off += sizeof(uint64_t);
  memcpy(header_ptr + off, proto_str.data(), proto_size);

  auto sg = std::make_shared<ScatterGatherBuffer>();

  // Segment 1: header + proto (owned, from pool)
  sg->AddOwnedSegment(header_ptr, header_size, header_bucket);

  // Segments 2+: tensor data (referenced in-place, NOT copied)
  for (const auto& m : mat) {
    auto* ptr = std::get<0>(m);
    auto sz = std::get<1>(m);
    if (ptr && sz > 0) {
      sg->AddReferenceSegment(ptr, sz);
    }
  }

  return sg;
}
