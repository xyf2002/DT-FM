#include "checkpoint_handle.h"

#include <cuda_runtime_api.h>

#include "archer_tensor_handle.h"
#include "common/types_and_defs.h"
#include "utils/json_reader.h"
// #include "utils/logger.h"
#include "utils/progress_bar.h"

CheckpointHandle::CheckpointHandle(const std::filesystem::path& prefix)
    : prefix_(prefix), prio_aio_handle_(prefix), allocator_(nullptr) {
  InitCachingAllocator(MemoryType::PIN_SHM);
  buffer_ = nullptr;
  auto param_meta_map_file = prefix / PARAM_META_FILE;
  auto json_reader = JsonReader<ParamMeta>(param_meta_map_file.string());
  param_meta_map_ = json_reader.ParseIntoMap();

  LOG_DEBUG << "param_meta_map_file: " << param_meta_map_file.c_str()
            << ", size " << param_meta_map_.size();
}

std::filesystem::path CheckpointHandle::GetFilePathByID(
    uint32_t file_id) const {
  return prefix_ /
         (std::string(ARCHER_PARAM_NAME) + "_" + std::to_string(file_id));
}

void CheckpointHandle::ReadCheckpoint() {
  int file_id = 0;  // FIXME: hard code only one file
  auto param_filename = GetFilePathByID(file_id);
  auto index_filename = prefix_ / std::string(ARCHER_IHDEX_NAME);

  LOG_DEBUG << "param_filename: " << param_filename.c_str();

  // get param_filename file size
  struct stat st;
  if (stat(param_filename.c_str(), &st) == -1) {
    LOG_FATAL << "Invalid prefix: " << param_filename.c_str()
              << " does not exist";
  }
  auto file_size = st.st_size;

  auto [pin_mem_size, pin_mem_offsets] = ComputePinOffsets();

  LOG_WARN_IF(pin_mem_size != file_size)
      << "Pin memory size " << pin_mem_size << " != file size " << file_size;
  // LOG_FATAL_IF(buffer_ != nullptr, "Buffer is not null, should only load
  // once");

  // std::call_once(flag_, [&]() {
  //   allocator_ =
  //       std::make_unique<CachingAllocator>(pin_mem_size,
  //       MemoryType::PIN_SHM);
  //   // buffer_ = allocator_->Allocate(pin_mem_size);
  // });

  tensor_index_.Deserialize(index_filename.c_str());

  std::unordered_map<std::string, size_t> name_id_map;
  for (const auto& [param_name, param_meta] : param_meta_map_) {
    name_id_map[param_name] = param_meta.id;
  }

  float count = 0;
  std::unordered_map<std::string, int> filenames;
  for (auto& [name, buffer_offset] : pin_mem_offsets) {
    auto id = name_id_map[name];
    auto tensor_meta = tensor_index_[id];
    auto file_id = tensor_meta.file_id;

    auto file_offset = tensor_meta.offset;
    auto num_bytes = tensor_meta.size;

    param_filename = GetFilePathByID(file_id);

    if (filenames.find(param_filename) == filenames.end()) {
      int fd = open(param_filename.c_str(), O_RDONLY);
      filenames[param_filename] = fd;
    }
    int fd = filenames[param_filename];

    size_t aligned_bytes = (num_bytes + 4095) & ~4095;

    void* buffer = kCachingAllocator->Allocate(num_bytes);
    auto shm_meta = kCachingAllocator->FindShmMetaByRange(buffer);
    auto shm_name = shm_meta.name;
    param_shm_map_[name] = {.id = -1 /* not used */,
                            .ptr = buffer,
                            .size = num_bytes,
                            .name = shm_name};

    void* temp_buffer = aligned_alloc(4096, aligned_bytes);
    // read using pread
    int ret = pread(fd, temp_buffer, aligned_bytes, file_offset);
    LOG_FATAL_IF(ret == -1)
        << "pread failed: errno " << errno << ", message " << strerror(errno);
    // check if temp_buffer contains all zeros
    bool is_zero = true;
    for (size_t i = 0; i < num_bytes; i++) {
      if (((char*)temp_buffer)[i] != 0) {
        is_zero = false;
        break;
      }
    }
    LOG_FATAL_IF(
        is_zero,
        "Read all zeros, file: {}, offset: {}, size: {}, aligned_size: {}",
        param_filename.c_str(), file_offset, num_bytes, aligned_bytes);
    memcpy(buffer, temp_buffer, num_bytes);

    // prio_aio_handle_.Read(param_filename, buffer, false, num_bytes,
    //                       file_offset);
    count += num_bytes;
    showProgressBar(count / pin_mem_size, "Reading checkpoint ");
    // LOG_DEBUG("Read param: {}, id: {}, size: {}, file_id: {}, file_offset:
    // {}",
    //           name, id, num_bytes, file_id, file_offset);
  }
}

std::vector<uint32_t> CheckpointHandle::FindIDsSameSize(size_t size) {
  std::vector<uint32_t> ids;
  for (const auto& [param_name, param_meta] : param_meta_map_) {
    if (param_meta.size == size) {
      ids.push_back(param_meta.id);
    }
  }
  return ids;
}

std::tuple<uint64_t, std::unordered_map<size_t, size_t>>
CheckpointHandle::ComputeShmOffsets() {
  size_t shm_mem_size = 0;
  std::unordered_map<size_t, size_t> shm_mem_offsets;
  std::unordered_map<size_t, size_t> unique_sizes_counter;

  for (const auto& [param_name, param_meta] : param_meta_map_) {
    if (unique_sizes_counter.find(param_meta.size) ==
        unique_sizes_counter.end()) {
      unique_sizes_counter[param_meta.size] = 0;
    }
    unique_sizes_counter[param_meta.size]++;
  }

  for (const auto& [size, count] : unique_sizes_counter) {
    shm_mem_size += size + 4 * count;
    shm_mem_offsets[size] = 0;
  }

  // map ordering is only guaranteed after keys are fixed
  size_t shm_offset = 0;
  for (const auto& [size, offset] : shm_mem_offsets) {
    shm_mem_offsets[size] = shm_offset;
    shm_offset += size + 4 * unique_sizes_counter[size];
  }

  return {shm_mem_size, shm_mem_offsets};
}

std::tuple<uint64_t, std::unordered_map<std::string, size_t>>
CheckpointHandle::ComputePinOffsets() {
  std::unordered_map<std::string, size_t> pin_mem_offsets;
  size_t pin_mem_size = 0;
  // size_t current_offset = 0;

  for (const auto& [param_name, param_meta] : param_meta_map_) {
    pin_mem_offsets[param_name] = pin_mem_size;
    // current_offset += param_meta.size;
    pin_mem_size += param_meta.size;
    LOG_DEBUG << "param_name: " << param_name << ", size: " << param_meta.size
              << ", offset: " << pin_mem_offsets[param_name];
  }

  return {pin_mem_size, pin_mem_offsets};
}
