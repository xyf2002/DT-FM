#pragma once

#include <rapidjson/document.h>
#include <rapidjson/istreamwrapper.h>

#include <fstream>
#include <iostream>
#include <unordered_map>

#include "utils/logger.h"

template <typename T>
class JsonReader {
 public:
  JsonReader(const std::string& json_file) {
    json_file_ = json_file;
    std::ifstream ifs(json_file_);
    rapidjson::IStreamWrapper isw(ifs);
    d_.ParseStream(isw);
    LOG_DEBUG << "Read json file: " << json_file_;
  }

  std::unordered_map<std::string, T> ParseIntoMap() {
    std::unordered_map<std::string, T> result;
    for (auto& m : d_.GetObject()) {
      T obj;
      obj.FromJson(m.value);
      result[m.name.GetString()] = obj;
      // LOG_DEBUG("Read object: {}", m.name.GetString());
    }
    return result;
  }

 private:
  rapidjson::Document d_;
  std::string json_file_;
};
