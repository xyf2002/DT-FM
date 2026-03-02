#pragma once
#include <rapidjson/document.h>

#include <cstdint>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "lru.h"
// #include "rttr_registration.h"

#define PARAM_META_FILE "param_meta_map.json"
#define BUFFER_SHM_NAME "/emulator_shm_buffer"
#define PARAM_SHM_NAME "/emulator_shm_param"
#define ALIGNMENT 512

typedef std::uint32_t TensorID;
typedef std::size_t HashID;
typedef std::size_t NodeID;
typedef std::uint64_t GraphID;
typedef std::uint64_t RequestID;

struct ShmMeta {
  int id;
  void* ptr;
  size_t size;
  std::string name;
  bool is_remote = false;
};

struct ParamMeta {
  uint32_t id;
  size_t size;
  // size_t shm_offset;
  size_t file_offset;

  void FromJson(const rapidjson::Value& json_obj) {
    if (json_obj.HasMember("id")) {
      id = json_obj["id"].GetUint();
    }
    if (json_obj.HasMember("size")) {
      size = json_obj["size"].GetUint64();
    }
    if (json_obj.HasMember("file_offset")) {
      file_offset = json_obj["file_offset"].GetUint64();
    }
  }
};

typedef std::unordered_map<std::string, ShmMeta> ParamShmMap;

// #define KB 1024
// #define MB (KB * KB)
// #define GB (KB * KB * KB)

constexpr size_t KB = 1024;
constexpr size_t MB = KB * KB;
constexpr size_t GB = KB * KB * KB;
constexpr size_t TB = KB * KB * KB * KB;

typedef std::tuple<uint64_t, int64_t, int64_t, bool>
    TensorKey;  // version, pivot, r/c, is_row

template <typename T>
struct DoNothingDeleter {
  void operator()(T* ptr) const {}
};

template <typename T>
struct CudaHostDeleter {
  void operator()(T* ptr) const {
    if (ptr) {
      cudaFreeHost(ptr);
    }
  }
};

#define GETENV(name, default_value) \
  (std::getenv(name) ? std::getenv(name) : default_value)

constexpr size_t operator"" _KB(unsigned long long val) { return val * KB; }
constexpr size_t operator"" _MB(unsigned long long val) { return val * MB; }
constexpr size_t operator"" _GB(unsigned long long val) { return val * GB; }

// A constexpr function to convert any pointer (const or non-const) to void*
template <typename T>
constexpr void* pointer_to_void(T* ptr) {
  return const_cast<void*>(reinterpret_cast<const void*>(
      ptr));  // Cast to void* while preserving constness
}

// // Overload for const pointers
// template <typename T>
// constexpr const void* pointer_to_void(const T* ptr) {
//   return static_cast<const void*>(ptr);  // Cast to const void*
// }

// A constexpr function to convert any const T* pointer to void*
template <typename T>
constexpr void* pointer_to_void(const T* ptr) {
  return const_cast<void*>(reinterpret_cast<const void*>(
      ptr));  // Cast to void* while preserving constness
}

// Helper macros to generate enum and string mappings
#define ENUM_ENTRY_COMMA(value, EnumType) value,
#define ENUM_CASE(value, EnumType) \
  case EnumType::value:            \
    return #value;
#define STRING_CASE(value, EnumType) \
  if (s == #value) return EnumType::value;

// Macro to define enum class, enum to string, and string to enum functions
#define DEFINE_ENUM_CLASS(EnumType, ENUM_VALUES)                           \
  enum class EnumType { ENUM_VALUES(ENUM_ENTRY_COMMA, EnumType) Unknown }; \
                                                                           \
  /* Enum to string function */                                            \
  inline const char* EnumType##ToString(EnumType v) {                      \
    switch (v) {                                                           \
      ENUM_VALUES(ENUM_CASE, EnumType)                                     \
      default:                                                             \
        return "Unknown";                                                  \
    }                                                                      \
  }                                                                        \
                                                                           \
  /* String to enum function */                                            \
  inline EnumType StringTo##EnumType(const std::string& s) {               \
    ENUM_VALUES(STRING_CASE, EnumType)                                     \
    return EnumType::Unknown;                                              \
  }
