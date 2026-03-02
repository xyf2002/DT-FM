#pragma once

#include <filesystem>
#include <string>
#include <unordered_map>

#include "archer_prio_aio_handle.h"
#include "archer_tensor_index.h"
#include "memory/caching_allocator.h"
#include "utils/json_reader.h"
#include "utils/noncopyable.h"

// /* Specialize fmt::formatter for ParamMeta with rttr support */
// template <>
// struct fmt::formatter<ParamMeta> {
//   constexpr auto parse(format_parse_context& ctx) { return ctx.begin(); }

//   template <typename FormatContext>
//   auto format(const ParamMeta& p, FormatContext& ctx) {
//     // use rttr to get the property names
//     auto prop_names = rttr::type::get<ParamMeta>().get_properties();
//     std::string prop_str;
//     for (auto& prop : prop_names) {
//       prop_str += fmt::format("{}={}, ", prop.get_name().to_string(),
//                               prop.get_value(p).to_string());
//     }
//     return format_to(ctx.out(), "{{{}}}", prop_str);
//   }
// };

typedef std::unordered_map<std::string, ParamMeta> ParamMetaMap;

class CheckpointHandle : public noncopyable {
 public:
  explicit CheckpointHandle(const std::filesystem::path& prefix);
  ~CheckpointHandle() = default;

  void ReadCheckpoint();
  ParamShmMap GetParamShmMap() const { return param_shm_map_; }

 private:
  std::filesystem::path GetFilePathByID(uint32_t file_id) const;

  std::vector<uint32_t> FindIDsSameSize(size_t size);
  std::tuple<uint64_t, std::unordered_map<size_t, size_t>> ComputeShmOffsets();
  std::tuple<uint64_t, std::unordered_map<std::string, size_t>>
  ComputePinOffsets();

 private:
  void* buffer_;
  std::filesystem::path prefix_;
  ArcherTensorIndex tensor_index_;
  ArcherPrioAioHandle prio_aio_handle_;
  ParamMetaMap param_meta_map_;
  std::unique_ptr<CachingAllocator> allocator_;
  ParamShmMap param_shm_map_;
  std::once_flag flag_;
};
