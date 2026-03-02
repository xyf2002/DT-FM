// Copyright (c) TorchMoE.
// SPDX-License-Identifier: Apache-2.0

// TorchMoE Team

#include "logger.h"

#include <stdio.h>
#include <string.h>

#include <cstdlib>
#include <iostream>

std::unique_ptr<base::LogFile> g_tee_log_file;

std::once_flag kLoggerFlag;
int kLogLevel = -1;
std::mutex kLogMutex;

int str2level(const char* level) {
  if (strcmp(level, "info") == 0) {
    return kInfo;
  } else if (strcmp(level, "error") == 0) {
    return kError;
  } else if (strcmp(level, "warn") == 0) {
    return kWarn;
  } else if (strcmp(level, "debug") == 0) {
    return kDebug;
  } else if (strcmp(level, "fatal") == 0) {
    return kFatal;
  } else {
    return -1;
  }
}

std::string level2str(int level) {
  switch (level) {
    case kInfo:
      return "INFO";
    case kError:
      return "ERROR";
    case kWarn:
      return "WARN";
    case kDebug:
      return "DEBUG";
    case kFatal:
      return "FATAL";
    default:
      return "UNKNOWN";
  }
}

std::string formatstr() {
  // get actual values in the format
  auto time = std::chrono::system_clock::now();
  auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                time.time_since_epoch()) %
            1000;
  auto timer = std::chrono::system_clock::to_time_t(time);
  auto tm = *std::localtime(&timer);

  auto year = tm.tm_year + 1900;
  auto month = tm.tm_mon + 1;
  auto day = tm.tm_mday;
  auto hour = tm.tm_hour;
  auto min = tm.tm_min;
  auto sec = tm.tm_sec;
  auto msec = ms.count();

  char buf[128];
  sprintf(buf, "%04d-%02d-%02d %02d:%02d:%02d.%03ld", year, month, day, hour,
          min, sec, msec);
  return std::string(buf) + " ";
}

void InitLogger() {
  std::call_once(kLoggerFlag, []() {
    // Set log level from environment variable
    const char* log_level_env = getenv("LOG_LEVEL");
    if (log_level_env) {
      base::Logger::setLogLevel(std::string(log_level_env));
    } else {
      base::Logger::setLogLevel(base::Logger::INFO);
    }

    LOG_INFO << "Logger initialized with level: "
             << (log_level_env ? log_level_env : "INFO");
  });
}

void TeeOutput(const char* msg, int len) {
  ::fwrite(msg, 1, len, stdout);
  if (g_tee_log_file) {
    g_tee_log_file->append(msg, len);
  }
}

void TeeFlush() {
  ::fflush(stdout);
  if (g_tee_log_file) {
    g_tee_log_file->flush();
  }
}
