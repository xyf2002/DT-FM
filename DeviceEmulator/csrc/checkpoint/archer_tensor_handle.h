// Copyright (c) TorchMoE.
// SPDX-License-Identifier: Apache-2.0

// TorchMoE Team

#pragma once

#include <torch/torch.h>

#include "archer_prio_aio_handle.h"
#include "archer_tensor_index.h"
#include "utils/noncopyable.h"

extern const char* ARCHER_PARAM_NAME;
extern const char* ARCHER_IHDEX_NAME;

class ArcherTensorHandle : public noncopyable {
 public:
  explicit ArcherTensorHandle(const std::string& prefix);
  ~ArcherTensorHandle() = default;

  bool IsTensorOffloaded(const std::uint32_t tensor_id);
  std::uint64_t OffloadTensor(torch::Tensor& tensor,
                              const std::uint32_t tensor_id);
  bool IsTensorIndexInitialized() const { return is_serialized_; }

 private:
  std::uint64_t StoreTensor(const std::uint32_t tensor_id,
                            torch::Tensor& buffer);
  std::string GetIndexFileName(const std::uint32_t file_id) const;

  void ReadTensor(const std::uint32_t tensor_id, void* memory_ptr,
                  bool on_demand = false);

 private:
  std::string prefix_;
  ArcherPrioAioHandle prio_aio_handle_;
  std::uint32_t file_id_;
  std::int64_t file_offset_;
  std::unordered_map<void*, std::uint32_t> tensor_to_id_;

  std::mutex mutex_;
  ArcherTensorIndex* tensor_index_ = new ArcherTensorIndex();
  bool is_serialized_ = false;
};
