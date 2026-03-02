#pragma once

#include <cublas_v2.h>
#include <cublasXt.h>
#include <cuda.h>  // CUDA driver API (green contexts)
#include <cuda_runtime_api.h>

#include <condition_variable>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "intercept/interceptor.h"
#include "memory/caching_allocator.h"
#include "scheduling_policy.h"
#include "utils/cuda_utils.h"
#include "worker_base.h"

// One entry in an SM switching schedule file.
struct SmScheduleEntry {
  int64_t time_offset_us;
  int num_sms;
  int64_t duration_us;
};

#define CUDA_MEMCPY_ASYNC_LOOP(trans, dst, src, ld, m, k, mode, stream)      \
  for (int col = 0; col < ((trans == 'N' || trans == 'n') ? k : m); col++) { \
    size_t num_elements = ((trans == 'N' || trans == 'n') ? m : k);          \
    size_t offset = col * ld;                                                \
    CHECK_CUDA_ERROR(cudaMemcpyAsync(dst + offset, src + offset,             \
                                     num_elements * sizeof(float), mode,     \
                                     stream))                                \
  }

#define CUDA_TRANS_OP(trans) \
  (trans == 'N' || trans == 'n') ? CUBLAS_OP_N : CUBLAS_OP_T

// RAII wrapper for a green context + stream + cublasXt handle at a
// specific SM count.  Movable, not copyable.
struct ContextSlot {
  int sm_count = 0;
  CUdevResourceDesc resource_desc = nullptr;
  CUgreenCtx green_ctx = nullptr;
  CUcontext cuda_ctx = nullptr;
  cudaStream_t stream = nullptr;
  cublasXtHandle_t xt_handle = nullptr;

  ContextSlot() = default;
  ~ContextSlot();
  ContextSlot(ContextSlot&& other) noexcept;
  ContextSlot& operator=(ContextSlot&& other) noexcept;
  ContextSlot(const ContextSlot&) = delete;
  ContextSlot& operator=(const ContextSlot&) = delete;
};

// One XtGemmWorker per logical partition on a GPU.
// Pre-creates green contexts at every valid SM granularity within its
// partition.  Tasks choose SMs dynamically via SwitchContext(num_sms).
// Uses cublasXt so callers pass host pointers; H2D/D2H is automatic.
class XtGemmWorker : public WorkerBase,
                     public std::enable_shared_from_this<XtGemmWorker> {
 public:
  // gpu_id: physical GPU index
  // num_partitions: how many workers share this GPU (for SM partitioning)
  // partition_idx: this worker's partition index [0, num_partitions)
  // buffer_size: CachingAllocator pool size per worker
  XtGemmWorker(int gpu_id, int num_partitions, int partition_idx,
               size_t buffer_size);
  ~XtGemmWorker();

  DELETE_COPY_AND_ASSIGN(XtGemmWorker);

  // cublasXt-style API: host pointers in, host pointers out.
  // cublasXt handles all H2D/D2H internally.
  void RunXtGemm(std::shared_ptr<GemmArgs> args);

  // Switch to the green context with exactly `num_sms` SMs.
  // Returns false if no such context exists.
  bool SwitchContext(int num_sms);

  cudaStream_t GetStream() const {
    return active_slot_ ? active_slot_->stream : nullptr;
  }
  int GetGpuId() const { return gpu_id_; }
  int GetPartitionIdx() const { return partition_idx_; }
  int GetSmStep() const { return sm_step_; }
  int GetPartitionSmCount() const { return partition_sm_count_; }
  int GetActiveSmCount() const;
  std::vector<int> GetAvailableSmCounts() const;

  CachingAllocator* GetAllocator() const {
    return allocator_.get();
  }

  // Load an SM switching schedule from a text file.
  // Returns false on parse error or invalid SM counts.
  // Must be called after contexts are initialized (use warmup task pattern).
  bool LoadSmSchedule(const std::string& path);

  // Execute the loaded schedule synchronously (blocking).
  // Must run on the worker thread — enqueue via AddTask().
  void RunSmSchedule();

  const std::vector<SmScheduleEntry>& GetSmSchedule() const {
    return sm_schedule_;
  }

 private:
  void Run() override;  // Thread entry: set device, init contexts, run loop
  void InitAllContexts();
  ContextSlot CreateContextSlot(CUdevResource* groups, int num_groups,
                                int sm_count);

  int gpu_id_;
  int num_partitions_;
  int partition_idx_;
  size_t buffer_size_;

  CUdevice cu_device_ = 0;

  // Multi-context state (populated by InitAllContexts on worker thread)
  std::unordered_map<int, ContextSlot> context_slots_;
  ContextSlot* active_slot_ = nullptr;
  int sm_step_ = 0;             // hardware SM granularity
  int partition_sm_count_ = 0;  // total SMs for this partition
  std::vector<CUdevResource> sm_groups_;

  // Per-worker CUDA memory pool
  std::unique_ptr<CachingAllocator> allocator_;

  // SM switching schedule (loaded from file)
  std::vector<SmScheduleEntry> sm_schedule_;
};

// Pool of XtGemmWorkers, potentially multiple per GPU
class XtGemmWorkerPool : public noncopyable {
 public:
  // workers_per_gpu: number of green context partitions per GPU
  // buffer_size: CachingAllocator pool size per worker
  // policy: scheduling policy for task distribution
  XtGemmWorkerPool(int workers_per_gpu, size_t buffer_size,
                   WorkerSchedulingPolicy policy);
  ~XtGemmWorkerPool();

  DELETE_COPY_AND_ASSIGN(XtGemmWorkerPool);

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

 private:
  std::vector<std::shared_ptr<XtGemmWorker>> workers_;
  std::unique_ptr<SchedulingPolicy> scheduler_;
};
