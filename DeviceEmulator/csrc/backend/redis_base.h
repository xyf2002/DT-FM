#pragma once

#include <sw/redis++/redis++.h>

inline sw::redis::Redis* GetRedisConnection() {
  sw::redis::ConnectionOptions connection_options;
  connection_options.host = "localhost";
  connection_options.port = 6379;
  return new sw::redis::Redis(connection_options);
}

inline int GetNumKeys(sw::redis::Redis* redis) {
  sw::redis::Cursor cursor = 0;
  auto pattern = "*";
  auto count = 5;
  std::unordered_set<std::string> keys;
  while (true) {
    cursor =
        redis->scan(cursor, pattern, count, std::inserter(keys, keys.begin()));
    // Default pattern is "*", and default count is 10
    // cursor = redis.scan(cursor, std::inserter(keys, keys.begin()));

    if (cursor == 0) {
      break;
    }
  }
  return keys.size();
}

inline std::unordered_map<std::string, std::string> GetDeviceInfo(
    sw::redis::Redis* redis, const std::string& uuid) {
  std::unordered_map<std::string, std::string> info;
  redis->hgetall(uuid, std::inserter(info, info.begin()));
  return std::move(info);
}
