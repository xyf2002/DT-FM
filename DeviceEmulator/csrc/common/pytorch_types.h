#pragma once

#include <torch/torch.h>

#define CPU_DEVICE torch::Device(torch::kCPU)
#define CUDA_DEVICE(index) torch::Device(torch::kCUDA, index)
#define DISK_DEVICE torch::Device(torch::kMeta)
#define DEFAULT_CUDA_DEVICE torch::Device(torch::kCUDA, 0)

#define FLOAT32_TENSOR_OPTIONS(target) \
  torch::TensorOptions().dtype(torch::kFloat32).device(target)
#define FLOAT16_TENSOR_OPTIONS(target) \
  torch::TensorOptions().dtype(torch::kFloat16).device(target)
#define FAKE_TENSOR_SIZES torch::IntArrayRef({1})
