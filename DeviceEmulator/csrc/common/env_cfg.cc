#include "env_cfg.h"

#include "backend/sched_policy.h"
#include "base/ini_config.h"
#include "base/logging.h"

#define PARSE_INT_ENVCFG(cfg, section, key, required, default_value, value) \
  {                                                                         \
    if (cfg.Count(section, key) == 0) {                                     \
      if (!required)                                                        \
        value = default_value;                                              \
      else {                                                                \
        LOG_ERROR << "missing " << section << "." << key;                   \
        return -1;                                                          \
      }                                                                     \
    } else {                                                                \
      value = cfg.IntValue(section, key);                                   \
    }                                                                       \
  }

#define PARSE_STR_ENVCFG(cfg, section, key, required, default_value, value) \
  {                                                                         \
    if (cfg.Count(section, key) == 0) {                                     \
      if (!required)                                                        \
        value = default_value;                                              \
      else {                                                                \
        LOG_ERROR << "missing " << section << "." << key;                   \
        return -1;                                                          \
      }                                                                     \
    } else {                                                                \
      value = cfg.GetValue(section, key);                                   \
    }                                                                       \
  }

ProxyEnvCfg::ProxyEnvCfg() = default;
ProxyEnvCfg::~ProxyEnvCfg() = default;

int ProxyEnvCfg::Initialize(const std::string& cfg_file) {
  base::IniConfig parser;
  if (parser.LoadFromFile(cfg_file.c_str())) {
    return -1;
  }
  PARSE_STR_ENVCFG(parser, "log", "log_path", false, "/root/log", log_path);
  PARSE_STR_ENVCFG(parser, "log", "log_level", false, "info", log_level);
  PARSE_STR_ENVCFG(parser, "log", "log_type", false, "terminal", log_type);

  PARSE_INT_ENVCFG(parser, "network", "listen_port", true, 0, listen_port);
  PARSE_STR_ENVCFG(parser, "network", "listen_ip", true, "", listen_ip);

  // Allow environment variables to override config file settings
  const char* env_listen_ip = std::getenv("MORPHLING_PROXY_HOST");
  const char* env_listen_port = std::getenv("MORPHLING_PROXY_PORT");

  if (env_listen_ip != nullptr && strlen(env_listen_ip) > 0) {
    listen_ip = std::string(env_listen_ip);
    LOG_INFO << "Overriding listen_ip from environment: " << listen_ip;
  }

  if (env_listen_port != nullptr && strlen(env_listen_port) > 0) {
    listen_port = std::atoi(env_listen_port);
    LOG_INFO << "Overriding listen_port from environment: " << listen_port;
  }

  PARSE_INT_ENVCFG(parser, "worker", "thread", false, 2, thread);
  PARSE_INT_ENVCFG(parser, "worker", "max_inflight", false, 5, max_inflight);
  PARSE_INT_ENVCFG(parser, "worker", "block_size", false, 32, block_size);
  PARSE_INT_ENVCFG(parser, "worker", "num_device", false, 32, num_device);
  PARSE_STR_ENVCFG(parser, "worker", "pool_mode", false, "gpu", pool_mode);

  PARSE_INT_ENVCFG(parser, "internal", "cleanup_wait", false, 60, cleanup_wait);
  PARSE_INT_ENVCFG(parser, "internal", "tcp_timeout", false, 5, tcp_timeout);
  PARSE_INT_ENVCFG(parser, "internal", "enable_cli_cache", false, 0,
                   enable_cli_cache);

  // Parse scheduling policy type (default to ROUND_ROBIN)
  int sched_policy_int = 0;
  PARSE_INT_ENVCFG(parser, "worker", "sched_policy", false, 0,
                   sched_policy_int);
  sched_policy_type = static_cast<SchedulingPolicyType>(sched_policy_int);

  // Initialize scheduling policy based on type
  switch (sched_policy_type) {
    case SchedulingPolicyType::ROUND_ROBIN:
      sched_policy =
          std::make_unique<morphling::backend::RoundRobinSchedulingPolicy>(
              block_size, enable_cli_cache);
      break;
    case SchedulingPolicyType::GREEDY:
      sched_policy =
          std::make_unique<morphling::backend::GreedySchedulingPolicy>(
              block_size, enable_cli_cache);
      break;
    case SchedulingPolicyType::LOAD_BALANCED:
      sched_policy =
          std::make_unique<morphling::backend::LoadBalancedSchedulingPolicy>(
              block_size, enable_cli_cache);
      break;
    default:
      throw std::invalid_argument("Invalid scheduling policy type");
      break;
  }

  base::Logger::setLogLevel(log_level);

  return 0;
}
