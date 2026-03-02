#pragma once

#include <cuda_runtime_api.h>

#include "pytorch_types.h"
#include "types_and_defs.h"

enum MatMulMessageType {
  kMatMulOrdinary = 0,
  kMatMulAttnFusedFwd = 1,
  kMatMulAttnFusedBwd = 2,
  kMatMulMLPFusedFwd = 3,
  kMatMulMLPFusedBwd = 4,
};

inline int64_t write_options(std::string& bytes,
                             const torch::TensorOptions& obj) {
  bool pinned_memory = obj.pinned_memory();
  bool requires_grad = obj.requires_grad();
  std::int8_t dtype = static_cast<std::int8_t>(obj.dtype().toScalarType());
  std::int8_t device_index = static_cast<std::int8_t>(obj.device().index());
  std::int8_t device_type = static_cast<std::int8_t>(obj.device().type());
  std::int8_t layout = static_cast<std::int8_t>(obj.layout());

  bytes.append(reinterpret_cast<char*>(&pinned_memory), sizeof(pinned_memory));
  bytes.append(reinterpret_cast<char*>(&requires_grad), sizeof(requires_grad));
  bytes.append(reinterpret_cast<char*>(&dtype), sizeof(dtype));
  bytes.append(reinterpret_cast<char*>(&device_index), sizeof(device_index));
  bytes.append(reinterpret_cast<char*>(&device_type), sizeof(device_type));
  bytes.append(reinterpret_cast<char*>(&layout), sizeof(layout));

  return sizeof(pinned_memory) + sizeof(requires_grad) + sizeof(dtype) +
         sizeof(device_index) + sizeof(device_type) + sizeof(layout);
}

inline void read_options(std::string& bytes, int64_t& offset,
                         torch::TensorOptions& obj) {
  bool pinned_memory = obj.pinned_memory();
  bool requires_grad = obj.requires_grad();
  std::int8_t dtype = static_cast<std::int8_t>(obj.dtype().toScalarType());
  std::int8_t device_index = static_cast<std::int8_t>(obj.device().index());
  std::int8_t device_type = static_cast<std::int8_t>(obj.device().type());
  std::int8_t layout = static_cast<std::int8_t>(obj.layout());

  std::memcpy(&pinned_memory, bytes.data() + offset, sizeof(pinned_memory));
  offset += sizeof(pinned_memory);
  std::memcpy(&requires_grad, bytes.data() + offset, sizeof(requires_grad));
  offset += sizeof(requires_grad);
  std::memcpy(&dtype, bytes.data() + offset, sizeof(dtype));
  offset += sizeof(dtype);
  std::memcpy(&device_index, bytes.data() + offset, sizeof(device_index));
  offset += sizeof(device_index);
  std::memcpy(&device_type, bytes.data() + offset, sizeof(device_type));
  offset += sizeof(device_type);
  std::memcpy(&layout, bytes.data() + offset, sizeof(layout));
  offset += sizeof(layout);

  obj =
      obj.dtype(static_cast<c10::ScalarType>(dtype))
          .device(torch::Device(static_cast<torch::DeviceType>(device_type),
                                static_cast<torch::DeviceIndex>(device_index)))
          .layout(static_cast<c10::Layout>(layout))
          .requires_grad(requires_grad)
          .pinned_memory(pinned_memory);
}

struct MatMulRequestMessage {
  int64_t row;
  int64_t col;
  std::vector<int64_t> ld;
  std::vector<torch::Tensor> mat;
  std::vector<void*> mat_ptr;
  std::vector<std::tuple<int64_t, int64_t>> mat_shape;

  std::string serialized;

  void SetMat(torch::Tensor& mat_a, torch::Tensor& mat_b, int64_t block_size) {
    int64_t offset_r = row * block_size;
    int64_t offset_c = col * block_size;

    auto a_shape = mat_a.sizes().vec();
    auto b_shape = mat_b.sizes().vec();

    int64_t end_r =
        std::min(offset_r + block_size, a_shape[a_shape.size() - 2]);
    int64_t end_c =
        std::min(offset_c + block_size, b_shape[b_shape.size() - 1]);

    void* offset_r_ptr =
        mat_a.data_ptr() + offset_r * mat_a.strides()[mat_a.dim() - 2];
    void* offset_c_ptr =
        mat_b.data_ptr() + offset_c * mat_b.strides()[mat_b.dim() - 1];

    mat_ptr.push_back(offset_r_ptr);
    mat_ptr.push_back(offset_c_ptr);

    mat_shape.push_back(
        std::make_tuple(end_r - offset_r, a_shape[a_shape.size() - 1]));
    mat_shape.push_back(
        std::make_tuple(b_shape[b_shape.size() - 2], end_c - offset_c));

    mat.resize(2);
  }

  // Serialize to bytes
  std::string Serialize() {
    // std::string serialized;
    serialized.clear();
    serialized.append(reinterpret_cast<char*>(&row), sizeof(row));
    serialized.append(reinterpret_cast<char*>(&col), sizeof(col));
    // fprintf(stderr, "Serializing row: %ld, col: %ld\n", row, col);
    int64_t ld_size = ld.size();
    serialized.append(reinterpret_cast<char*>(&ld_size), sizeof(ld_size));
    for (auto l : ld) {
      serialized.append(reinterpret_cast<char*>(&l), sizeof(l));
    }
    // fprintf(stderr, "Serialized ld\n");

    int64_t mat_size = mat.size();
    serialized.append(reinterpret_cast<char*>(&mat_size), sizeof(mat_size));
    // fprintf(stderr, "Serialized mat_size: %ld\n", mat_size);

    for (int64_t i = 0; i < mat_size; i++) {
      auto sizes = mat_shape[i];
      int64_t sizes_size = 2;
      serialized.append(reinterpret_cast<char*>(&sizes_size),
                        sizeof(sizes_size));
      serialized.append(reinterpret_cast<char*>(&std::get<0>(sizes)),
                        sizeof(std::get<0>(sizes)));
      serialized.append(reinterpret_cast<char*>(&std::get<1>(sizes)),
                        sizeof(std::get<1>(sizes)));
      void* data = mat_ptr[i];
      serialized.append(
          reinterpret_cast<char*>(data),
          std::get<0>(sizes) * std::get<1>(sizes) * sizeof(float));
    }

    // for (auto m : mat) {
    //   // serialize tensor options
    //   auto options = m.options();
    //   write_options(serialized, options);
    //   // serialize tensor sizes
    //   std::vector<int64_t> sizes = m.sizes().vec();
    //   int64_t sizes_size = sizes.size();
    //   serialized.append(reinterpret_cast<char*>(&sizes_size),
    //                     sizeof(sizes_size));
    //   // fprintf(stderr, "Serialized sizes_size: %ld\n", sizes_size);
    //   for (int64_t s : sizes) {
    //     serialized.append(reinterpret_cast<char*>(&s), sizeof(s));
    //   }
    //   void* data = m.data_ptr();
    //   serialized.append(reinterpret_cast<char*>(data),
    //                     m.numel() * sizeof(float));
    //   // fprintf(stderr, "Serialized tensor\n");
    // }
    return serialized;
  }

  void Deserialize(std::string& bytes) {
    serialized.reserve(bytes.size());
    serialized.assign(bytes);
    // fprintf(stderr, "Deserializing request\n");
    int64_t offset = 0;
    std::memcpy(&row, serialized.data() + offset, sizeof(row));
    // fprintf(stderr, "Deserialized row: %ld\n", row);
    offset += sizeof(row);
    std::memcpy(&col, serialized.data() + offset, sizeof(col));
    // fprintf(stderr, "Deserialized col: %ld\n", col);
    offset += sizeof(col);
    int64_t ld_size;
    std::memcpy(&ld_size, serialized.data() + offset, sizeof(ld_size));
    // fprintf(stderr, "Deserialized ld_size: %ld\n", ld_size);
    offset += sizeof(ld_size);
    ld.resize(ld_size);
    for (int64_t i = 0; i < ld_size; i++) {
      std::memcpy(&ld[i], serialized.data() + offset, sizeof(ld[i]));
      offset += sizeof(ld[i]);
    }

    int64_t mat_size;
    std::memcpy(&mat_size, serialized.data() + offset, sizeof(mat_size));
    offset += sizeof(mat_size);
    // fprintf(stderr, "Deserialized mat_size: %ld\n", mat_size);

    for (int64_t i = 0; i < mat_size; i++) {
      int64_t sizes_size;
      std::memcpy(&sizes_size, serialized.data() + offset, sizeof(sizes_size));
      // fprintf(stderr, "Deserialized sizes_size: %ld\n", sizes_size);
      offset += sizeof(sizes_size);
      std::vector<int64_t> sizes(sizes_size);
      for (int64_t j = 0; j < sizes_size; j++) {
        std::memcpy(&sizes[j], serialized.data() + offset, sizeof(sizes[j]));
        // fprintf(stderr, "Deserialized size: %ld\n", sizes[j]);
        offset += sizeof(sizes[j]);
      }
      mat_shape.emplace_back(std::make_tuple(sizes[0], sizes[1]));
      mat_ptr.emplace_back(serialized.data() + offset);
      offset += sizes[0] * sizes[1] * sizeof(float);
    }
    assert(offset == serialized.size());

    // mat.resize(mat_size);
    // // mat_ptr.resize(mat_size);
    // for (int64_t i = 0; i < mat_size; i++) {
    //   // deserialize tensor options
    //   torch::TensorOptions options;
    //   read_options(serialized, offset, options);
    //   // deserialize tensor sizes
    //   int64_t sizes_size;
    //   std::memcpy(&sizes_size, serialized.data() + offset,
    //   sizeof(sizes_size));
    //   // fprintf(stderr, "Deserialized sizes_size: %ld\n", sizes_size);
    //   offset += sizeof(sizes_size);
    //   std::vector<int64_t> sizes(sizes_size);
    //   int64_t num_ele = 1;
    //   for (int64_t j = 0; j < sizes_size; j++) {
    //     std::memcpy(&sizes[j], serialized.data() + offset, sizeof(sizes[j]));
    //     num_ele *= sizes[j];
    //     // fprintf(stderr, "Deserialized size: %ld\n", sizes[j]);
    //     offset += sizeof(sizes[j]);
    //   }
    //   // deserialize tensor data
    //   void* data = serialized.data() + offset;
    //   // mat[i] = torch::from_blob(mat_ptr[i], sizes,
    //   // options.device(torch::kCUDA, 0));
    //   // mat[i].set_data(torch::from_blob(data, sizes, options));
    //   mat[i] = torch::zeros(sizes, options);
    //   std::memcpy(mat[i].data_ptr(), data, num_ele * sizeof(float));
    //   offset += mat[i].numel() * sizeof(float);
    //   // fprintf(stderr, "Deserialized tensor %ld\n", mat[i].numel());
    // }
  }
};

struct MatMulResponseMessage {
  int64_t row;
  int64_t col;
  std::vector<int64_t> ld;
  torch::Tensor mat;

  std::string serialized;

  // Serialize to bytes
  std::string Serialize() {
    serialized.clear();
    serialized.append(reinterpret_cast<char*>(&row), sizeof(row));
    serialized.append(reinterpret_cast<char*>(&col), sizeof(col));
    int64_t ld_size = ld.size();
    serialized.append(reinterpret_cast<char*>(&ld_size), sizeof(ld_size));
    for (auto l : ld) {
      serialized.append(reinterpret_cast<char*>(&l), sizeof(l));
    }

    // serialize tensor options
    auto options = mat.options();
    write_options(serialized, options);
    // serialize tensor sizes
    auto sizes = mat.sizes().vec();
    int64_t sizes_size = sizes.size();
    serialized.append(reinterpret_cast<char*>(&sizes_size), sizeof(sizes_size));
    for (auto s : sizes) {
      serialized.append(reinterpret_cast<char*>(&s), sizeof(s));
    }
    // serialize tensor data

    void* data = mat.data_ptr();
    serialized.append(reinterpret_cast<char*>(data),
                      mat.numel() * sizeof(float));
    return serialized;
  }

  // Deserialize from bytes

  void Deserialize(std::string& bytes) {
    serialized.reserve(bytes.size());
    serialized.assign(bytes);
    int64_t offset = 0;
    std::memcpy(&row, serialized.data() + offset, sizeof(row));
    offset += sizeof(row);
    std::memcpy(&col, serialized.data() + offset, sizeof(col));
    offset += sizeof(col);
    int64_t ld_size;
    std::memcpy(&ld_size, serialized.data() + offset, sizeof(ld_size));
    offset += sizeof(ld_size);
    ld.resize(ld_size);
    for (int64_t i = 0; i < ld_size; i++) {
      std::memcpy(&ld[i], serialized.data() + offset, sizeof(ld[i]));
      offset += sizeof(ld[i]);
    }

    // deserialize tensor options
    torch::TensorOptions options;
    read_options(serialized, offset, options);
    // deserialize tensor sizes
    int64_t sizes_size;
    std::memcpy(&sizes_size, serialized.data() + offset, sizeof(sizes_size));
    offset += sizeof(sizes_size);
    std::vector<int64_t> sizes(sizes_size);
    for (int64_t j = 0; j < sizes_size; j++) {
      std::memcpy(&sizes[j], serialized.data() + offset, sizeof(sizes[j]));
      offset += sizeof(sizes[j]);
    }
    // deserialize tensor data
    void* data = serialized.data() + offset;
    offset += sizes_size * sizeof(float);
    // mat.set_data(torch::from_blob(data, sizes, options));
    mat = torch::zeros(sizes, options);
    std::memcpy(mat.data_ptr(), data, mat.numel() * sizeof(float));
  }
};
