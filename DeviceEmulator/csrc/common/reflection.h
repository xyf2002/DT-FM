#pragma once

#include <functional>
#include <string>
#include <unordered_map>

// Macro to define struct members and map them to string keys for reflection
#define REFLECTABLE(...)                                                       \
  std::unordered_map<std::string,                                              \
                     std::function<void(void*, const std::string&)>>           \
      setters = {__VA_ARGS__};                                                 \
  std::unordered_map<std::string, std::function<std::string(void*)>> getters = \
      {__VA_ARGS__};

#define SETTER(MyStruct, field, type)                                            \
  {                                                                              \
#field,                                                                      \
        [](void* obj,                                                            \
           const std::string&                                                    \
               value) { static_cast<MyStruct*>(obj)->field = std::stoi(value); } \
  }

#define GETTER(MyStruct, field, type)                          \
  {                                                            \
#field, [](void* obj) -> std::string {                       \
     return std::to_string(static_cast<MyStruct*>(obj)->field); \
   } \
  }
