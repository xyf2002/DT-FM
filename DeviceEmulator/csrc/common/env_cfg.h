#pragma once

#include <memory>
#include <string>

namespace morphling {
namespace backend {
class PartitionSchedulingPolicy;
}
}  // namespace morphling

enum class SchedulingPolicyType {
  ROUND_ROBIN = 0,
  GREEDY = 1,
  LOAD_BALANCED = 2
};

struct ProxyEnvCfg {
  ProxyEnvCfg();
  ~ProxyEnvCfg();

  std::string listen_ip;
  unsigned listen_port;

  // worker
  int thread;
  int64_t block_size;
  int64_t num_device;
  int max_inflight;
  std::string pool_mode;  // "gpu", "cpu", or "both"

  // log
  std::string log_path;
  std::string log_level;
  std::string log_type;

  // internal
  int cleanup_wait;
  int64_t tcp_timeout;
  int enable_cli_cache;

  void* instance;

  // Scheduling policy configuration
  SchedulingPolicyType sched_policy_type;
  std::unique_ptr<morphling::backend::PartitionSchedulingPolicy> sched_policy;

  // Get scheduling policy with type-based reinterpretation
  template <typename T>
  T* GetSchedPolicy() {
    return dynamic_cast<T*>(sched_policy.get());
  }

  int Initialize(const std::string& cfg_file);
};
