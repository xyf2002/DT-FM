// Copyright (c) TorchMoE.
// SPDX-License-Identifier: Apache-2.0

// TorchMoE Team

#include "archer_tensor_handle.h"

#include <cuda_runtime_api.h>
#include <torch/script.h>

#include "utils/logger.h"

const int c_block_size = 128 * 1024;
const int c_io_queue_depth = 8;

const char* ARCHER_PARAM_NAME = "archer_param";
const char* ARCHER_IHDEX_NAME = "archer_index";

ArcherTensorHandle::ArcherTensorHandle(const std::string& prefix)
    : prefix_(prefix), prio_aio_handle_(prefix), file_id_(0), file_offset_(0) {
  InitLogger();
  // google::InitGoogleLogging("morphling");

  if (prefix_.back() != '/') {
    prefix_ += '/';
  }

  struct stat st;
  if (stat(prefix_.c_str(), &st) != -1 && !S_ISDIR(st.st_mode)) {
    LOG_FATAL << "Invalid prefix: " << prefix_ << " is not a directory";
  }
  if (stat(prefix_.c_str(), &st) == -1) {
    LOG_WARN << "Invalid prefix: " << prefix_ << " does not exist, creating";
    mkdir(prefix_.c_str(), 0777);
  }

  auto ckpt_index_path = prefix_ + std::string(ARCHER_IHDEX_NAME);
  if (access(ckpt_index_path.c_str(), F_OK) != -1) {
    LOG_INFO << "Loading index file from " << ckpt_index_path;
    tensor_index_->Deserialize(ckpt_index_path.c_str());
    is_serialized_ = true;
  } else {
    LOG_INFO << "Index file size " << ckpt_index_path
             << " does not exist, creating";
  }
  LOG_INFO << "Index file size " << tensor_index_->size();
}

std::uint64_t ArcherTensorHandle::OffloadTensor(torch::Tensor& tensor,
                                                const std::uint32_t tensor_id) {
  auto offset = StoreTensor(tensor_id, tensor);

  auto ckpt_index_path = prefix_ + std::string(ARCHER_IHDEX_NAME);

  std::unique_lock<std::mutex> lock(mutex_);
  tensor_index_->Serialize(ckpt_index_path.c_str());

  return offset;
}

bool ArcherTensorHandle::IsTensorOffloaded(const std::uint32_t tensor_id) {
  std::unique_lock<std::mutex> lock(mutex_);
  auto it = tensor_index_->find(tensor_id);
  // ARCHER_LOG_DEBUG("Check tensor {} {}", tensor_id, it ==
  // tensor_index_->end());
  bool is_offloaded = it != tensor_index_->end();
  if (is_offloaded) {
    it->second.id = tensor_id;
  }
  return is_offloaded;
}

std::uint64_t ArcherTensorHandle::StoreTensor(const std::uint32_t tensor_id,
                                              torch::Tensor& buffer) {
  auto it = tensor_index_->find(tensor_id);
  bool tensor_exists = (it != tensor_index_->end());

  std::unique_lock<std::mutex> lock(mutex_);
  TensorStorageMeta tensor_meta{file_id_, file_offset_, buffer.nbytes(),
                                buffer.sizes().vec()};
  tensor_meta.options = buffer.options();
  tensor_meta.id = tensor_id;

  auto num_bytes = buffer.nbytes();
  std::int64_t num_bytes_aligned =
      (num_bytes + kAioAlignment - 1) & ~(kAioAlignment - 1);

  if (tensor_exists) {
    // size must be the same if found
    if (it->second.size != buffer.nbytes()) {
      LOG_FATAL << "Tensor " << tensor_id << " size mismatch "
                << it->second.size << " != " << buffer.nbytes();
    }
    tensor_meta = it->second;
  }

  file_offset_ += tensor_exists ? 0 : num_bytes_aligned;

  tensor_index_->insert(std::make_pair(tensor_id, tensor_meta));

  auto filename = GetIndexFileName(tensor_meta.file_id);

  lock.unlock();
  prio_aio_handle_.Write(filename, buffer.data_ptr(), false, tensor_meta.size,
                         tensor_meta.offset);

  return tensor_meta.offset;
}

std::string ArcherTensorHandle::GetIndexFileName(
    const std::uint32_t file_id) const {
  return prefix_ + std::string(ARCHER_PARAM_NAME) + "_" +
         std::to_string(file_id);
}

void ArcherTensorHandle::ReadTensor(const uint32_t tensor_id, void* memory_ptr,
                                    bool on_demand) {
  auto it = tensor_index_->find(tensor_id);
  LOG_FATAL_IF(it == tensor_index_->end()) << "Tensor not found " << tensor_id;

  auto tensor_meta = it->second;
  auto filename = GetIndexFileName(tensor_meta.file_id);

  prio_aio_handle_.Read(filename, memory_ptr, on_demand, tensor_meta.size,
                        tensor_meta.offset);
}
