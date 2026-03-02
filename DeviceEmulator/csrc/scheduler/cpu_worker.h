#pragma once

#include <cblas.h>
#include <sched.h>

#include <algorithm>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "intercept/interceptor.h"
#include "memory/caching_allocator.h"
#include "scheduling_policy.h"
#include "utils/thread_affinity.h"
#include "worker_base.h"

// RAII wrapper for a CPU core-affinity bitmask at a specific core
// count.  Movable, not copyable.
struct CpuAffinitySlot {
  int core_count = 0;
  std::vector<int> core_ids;
  cpu_set_t cpuset{};

  CpuAffinitySlot() { CPU_ZERO(&cpuset); }

  CpuAffinitySlot(int count, std::vector<int> ids)
      : core_count(count), core_ids(std::move(ids)) {
    CPU_ZERO(&cpuset);
    for (int id : core_ids) {
      CPU_SET(id, &cpuset);
    }
  }

  // Apply this affinity to the calling thread.
  void Apply() const { morphling::PinThreadToCores(cpuset); }

  // Move semantics
  CpuAffinitySlot(CpuAffinitySlot&& other) noexcept
      : core_count(other.core_count),
        core_ids(std::move(other.core_ids)),
        cpuset(other.cpuset) {
    other.core_count = 0;
    CPU_ZERO(&other.cpuset);
  }

  CpuAffinitySlot& operator=(CpuAffinitySlot&& other) noexcept {
    if (this != &other) {
      core_count = other.core_count;
      core_ids = std::move(other.core_ids);
      cpuset = other.cpuset;
      other.core_count = 0;
      CPU_ZERO(&other.cpuset);
    }
    return *this;
  }

  CpuAffinitySlot(const CpuAffinitySlot&) = delete;
  CpuAffinitySlot& operator=(const CpuAffinitySlot&) = delete;
};

// One CpuWorker per logical partition of CPU cores.
// Pre-creates affinity slots at every valid core count within its
// partition.  Tasks choose cores dynamically via
// SwitchAffinity(num_cores).
class CpuWorker : public WorkerBase,
                  public std::enable_shared_from_this<CpuWorker> {
 public:
  // assigned_cores: the set of CPU core IDs this worker owns
  // partition_idx: this worker's partition index (for logging)
  // buffer_size: CachingAllocator pool size per worker
  CpuWorker(std::vector<int> assigned_cores, int partition_idx,
            size_t buffer_size = 0);
  ~CpuWorker();

  DELETE_COPY_AND_ASSIGN(CpuWorker);

  // Switch to a preset affinity slot with exactly `num_cores` cores.
  // Returns false if no such slot exists.
  bool SwitchAffinity(int num_cores);

  // Switch to an arbitrary subset of cores (must be within assigned
  // set). Returns false if any core is not in the assigned set.
  bool SwitchAffinity(const std::vector<int>& cores);

  // MKL cblas_sgemm: CPU-side GEMM mirroring XtGemmWorker::RunXtGemm.
  void RunMklGemm(std::shared_ptr<GemmArgs> args);

  int GetActiveCoreCount() const;
  std::vector<int> GetActiveCores() const;
  const std::vector<int>& GetAssignedCores() const {
    return assigned_cores_;
  }
  std::vector<int> GetAvailableCoreCounts() const;
  int GetPartitionIdx() const { return partition_idx_; }

  CachingAllocator* GetAllocator() const {
    return allocator_.get();
  }

 private:
  void Run() override;
  void InitAllAffinitySlots();
  CpuAffinitySlot CreateAffinitySlot(int num_cores);

  std::vector<int> assigned_cores_;
  int partition_idx_;
  size_t buffer_size_;

  std::unordered_map<int, CpuAffinitySlot> affinity_slots_;
  CpuAffinitySlot* active_slot_ = nullptr;

  // Per-worker pinned memory pool
  std::unique_ptr<CachingAllocator> allocator_;
};

// Pool of CpuWorkers, partitioning CPU cores among workers.
class CpuWorkerPool : public noncopyable {
 public:
  // num_workers: number of CPU workers to create
  // assignable_cores: full set of cores to partition among workers
  // policy: scheduling policy for task distribution
  // buffer_size: CachingAllocator pool size per worker
  CpuWorkerPool(int num_workers, std::vector<int> assignable_cores,
                WorkerSchedulingPolicy policy, size_t buffer_size = 0);
  ~CpuWorkerPool();

  DELETE_COPY_AND_ASSIGN(CpuWorkerPool);

  TaskHandle EnqueueTask(const std::string& task_id,
                         WorkerBase::Task task,
                         TaskCallback callback = nullptr);
  TaskHandle EnqueueGemm(const std::string& task_id,
                         std::shared_ptr<GemmArgs> args,
                         TaskCallback callback = nullptr);
  void WaitAll();
  void Wait(const std::string& task_id);

  size_t GetWorkerCount() const { return workers_.size(); }

  int GetPendingTaskCount() const {
    int total = 0;
    for (const auto& w : workers_) total += w->GetTaskCount();
    return total;
  }

  std::shared_ptr<CpuWorker> GetWorker(size_t idx) {
    return workers_.at(idx);
  }

 private:
  std::vector<std::shared_ptr<CpuWorker>> workers_;
  std::unique_ptr<SchedulingPolicy> scheduler_;
};
