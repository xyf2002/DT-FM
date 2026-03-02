#pragma once

#include <sched.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <vector>

#include "base/logging.h"

namespace morphling {

// Pin the calling thread to a single CPU core.
inline void PinThreadToCore(int core_id) {
  pid_t tid = syscall(SYS_gettid);
  cpu_set_t cpuset;
  CPU_ZERO(&cpuset);
  CPU_SET(core_id, &cpuset);

  int ret =
      syscall(SYS_sched_setaffinity, tid, sizeof(cpu_set_t), &cpuset);
  if (ret != 0) {
    LOG_WARN << "Failed to pin thread " << tid << " to core "
             << core_id << " (errno: " << errno << ")";
  }
}

// Pin the calling thread to a set of CPU cores.
inline void PinThreadToCores(const cpu_set_t& cpuset) {
  pid_t tid = syscall(SYS_gettid);
  int ret =
      syscall(SYS_sched_setaffinity, tid, sizeof(cpu_set_t), &cpuset);
  if (ret != 0) {
    LOG_WARN << "Failed to set thread " << tid
             << " affinity (errno: " << errno << ")";
  }
}

// Return the number of online CPU cores.
inline int GetOnlineCoreCount() {
  int n = sysconf(_SC_NPROCESSORS_ONLN);
  if (n <= 0) {
    LOG_WARN << "sysconf(_SC_NPROCESSORS_ONLN) returned " << n
             << ", defaulting to 1";
    return 1;
  }
  return n;
}

// Return a vector of core IDs [0, N) for all online cores.
inline std::vector<int> GetAllOnlineCores() {
  int n = GetOnlineCoreCount();
  std::vector<int> cores(n);
  for (int i = 0; i < n; i++) {
    cores[i] = i;
  }
  return cores;
}

}  // namespace morphling
