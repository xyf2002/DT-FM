#pragma once

#include <atomic>
#include <memory>

#include "common/env_cfg.h"
#include "common/lru.h"
#include "common/pytorch_defs.h"
#include "morphling.pb.h"
#include "network/connector_libevent.h"
#include "network/uevent.h"
#include "network/ueventloop_thread.h"
#include "scheduler/sliding_window_tracker.h"
#include "server_base.h"

// ============================================================================
// CudaPinnedMemoryPool: Pool of cudaHostAlloc'd buffers for GEMM results
// ============================================================================

class CudaPinnedMemoryPool {
 public:
  static CudaPinnedMemoryPool& Instance() {
    static CudaPinnedMemoryPool pool;
    return pool;
  }

  explicit CudaPinnedMemoryPool(size_t max_buffers_per_bucket = 16)
      : max_per_bucket_(max_buffers_per_bucket) {}

  ~CudaPinnedMemoryPool() {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto& [bucket_size, free_list] : free_lists_) {
      for (auto* ptr : free_list) {
        cudaFreeHost(ptr);
      }
    }
  }

  // Acquire a pinned buffer of at least `size` bytes
  // Returns {pointer, actual_bucket_size}
  std::pair<void*, size_t> Acquire(size_t size) {
    size_t bucket = BucketSize(size);
    std::lock_guard<std::mutex> lock(mutex_);
    auto& free_list = free_lists_[bucket];
    if (!free_list.empty()) {
      void* ptr = free_list.back();
      free_list.pop_back();
      return {ptr, bucket};
    }
    // Allocate new pinned buffer
    void* ptr = nullptr;
    cudaError_t err =
        cudaHostAlloc(&ptr, bucket, cudaHostAllocDefault | cudaHostAllocMapped);
    if (err != cudaSuccess || !ptr) {
      throw std::runtime_error("CudaPinnedMemoryPool: cudaHostAlloc failed");
    }
    return {ptr, bucket};
  }

  // Release a buffer back to the pool
  void Release(void* ptr, size_t bucket_size) {
    if (!ptr) return;
    std::lock_guard<std::mutex> lock(mutex_);
    auto& free_list = free_lists_[bucket_size];
    if (free_list.size() < max_per_bucket_) {
      free_list.push_back(ptr);
    } else {
      cudaFreeHost(ptr);
    }
  }

 private:
  static size_t BucketSize(size_t size) {
    static constexpr size_t MIN_BUCKET = 4096;
    if (size <= MIN_BUCKET) return MIN_BUCKET;
    size_t bucket = MIN_BUCKET;
    while (bucket < size) bucket <<= 1;
    return bucket;
  }

  size_t max_per_bucket_;
  std::mutex mutex_;
  std::unordered_map<size_t, std::deque<void*>> free_lists_;
};

// Forward declarations for worker pools
struct GemmArgs;
class XtGemmWorkerPool;
class CpuWorkerPool;

namespace morphling {
namespace backend {

class ProxyCliHandle : public uevent::LoopHandle {
 public:
  ProxyCliHandle(ProxyEnvCfg& ctx, uevent::UeventLoop* loop,
                 int64_t device_id = 0,
                 XtGemmWorkerPool* gpu_pool = nullptr,
                 CpuWorkerPool* cpu_pool = nullptr);
  ~ProxyCliHandle();

  static void ThreadInit(uevent::UeventLoop* loop);
  static uevent::LoopHandle* CreateMyself(ProxyEnvCfg& ctx, int64_t device_id,
                                          XtGemmWorkerPool* gpu_pool,
                                          CpuWorkerPool* cpu_pool,
                                          uevent::UeventLoop* loop) {
    return new ProxyCliHandle(ctx, loop, device_id, gpu_pool, cpu_pool);
  }

 public:
  // void PbRequestCb(const uevent::ConnectionUeventPtr& conn);
  void ConnectionSuccessCb(const uevent::ConnectionUeventPtr& conn);
  void ConnectionClosedCb(const uevent::ConnectionUeventPtr& conn);

  void ResponseToCaller(const uevent::ConnectionUeventPtr& conn,
                        const MatrixPartition& partition,
                        void* deferred_cuda_ptr = nullptr,
                        void* deferred_host_ptr = nullptr);
  void HandlePartition(const uevent::ConnectionUeventPtr& conn,
                       const MatrixPartition& partition);

 private:
  bool ShouldUseGpu() const;

  void SubmitToGpuPool(const uevent::ConnectionUeventPtr& conn,
                       const MatrixPartition& partition,
                       float* out_ptr, int64_t out_bytes,
                       const float* row_ptr, int64_t row_size,
                       const float* col_ptr, int64_t col_size,
                       int64_t h_dim, uint64_t vt_compute_start,
                       bool is_host_alloc,
                       SlidingWindowDurationTracker<>::TimePoint
                           task_enqueue_time);
  void SubmitToCpuPool(const uevent::ConnectionUeventPtr& conn,
                       const MatrixPartition& partition,
                       float* out_ptr, int64_t out_bytes,
                       const float* row_ptr, int64_t row_size,
                       const float* col_ptr, int64_t col_size,
                       int64_t h_dim, uint64_t vt_compute_start,
                       bool is_host_alloc,
                       SlidingWindowDurationTracker<>::TimePoint
                           task_enqueue_time);
  static std::shared_ptr<GemmArgs> BuildGemmArgs(
      const float* col_ptr, int64_t col_size,
      const float* row_ptr, int64_t row_size,
      int64_t h_dim, float* out_ptr);
  void OnComputeComplete(const uevent::ConnectionUeventPtr& conn,
                         const MatrixPartition& partition,
                         float* out_ptr, int64_t out_bytes,
                         int64_t col_size, uint64_t vt_compute_start,
                         bool is_host_alloc, bool is_gpu_task,
                         SlidingWindowDurationTracker<>::TimePoint
                             task_enqueue_time);

  ProxyEnvCfg& ctx_;
  uevent::UeventLoop* loop_;
  int64_t device_id_;
  XtGemmWorkerPool* gpu_pool_;
  CpuWorkerPool* cpu_pool_;
  std::atomic<uint64_t> task_counter_{0};

  // Sliding window trackers for dispatch estimation
  SlidingWindowDurationTracker<64> gpu_duration_tracker_{1000};  // 1ms default
  SlidingWindowDurationTracker<64> cpu_duration_tracker_{5000};  // 5ms default
};

struct CachedTensor {
  void* data = nullptr;
  int64_t rows = 0;
  int64_t cols = 0;
  int64_t bytes = 0;
};

class ProxyCliImpl : public std::enable_shared_from_this<ProxyCliImpl> {
 public:
  ProxyCliImpl(ProxyEnvCfg& context, int64_t device_id);
  void Initialize(uevent::UeventLoop* loop);

 private:
  void ConnectionSuccessCb(const uevent::ConnectionUeventPtr& conn);
  void ConnectionClosedCb(const uevent::ConnectionUeventPtr& conn);
  void RequestCb(const uevent::ConnectionUeventPtr& conn);

  void DecodeAndDispatch(const uevent::ConnectionUeventPtr& conn,
                         const void* payload, size_t size);

  // Message handlers following MessageHandler interface
  MessageHandlerSignature HandleRegisterRequest;
  MessageHandlerSignature HandleMatMulRequest;

  void HandleMatMul(const uevent::ConnectionUeventPtr& conn,
                    MatrixPartition& partition);
  void SendRegisterResponse(const uevent::ConnectionUeventPtr& conn);

  void SendPbRequest(const std::string& topic, const std::string& payload);
  void ReceivePbRequest(const std::string& topic,
                        const google::protobuf::Message& pb);

 private:
  MatrixPartition DecodeRequest(const void* payload, size_t size);
  void FillPartition(MatrixPartition& partition);
  void CacheTensor(const TensorKey& key, void* ptr, int64_t size,
                   int64_t h_dim);
  void SavePartition(MatrixPartition& partition);
  void HandlePartition(const uevent::ConnectionUeventPtr& conn,
                       const MatrixPartition& partition);
  void CheckCachedPartition(const uevent::ConnectionUeventPtr& conn);

 private:
  ProxyEnvCfg& ctx_;
  int64_t device_id_;
  std::shared_ptr<uevent::ConnectorLibevent> connector_;

  // sw::redis::Redis* redis_;
  // std::unordered_map<std::string, uint64_t> device_info_;

  std::string uuid_;
  std::vector<MatrixPartition> cached_partitions_;
  std::unordered_set<PtrData> cached_msgs_;
  FixSizeLRUCache<TensorKey, CachedTensor> cached_tensors_;
};

typedef std::shared_ptr<ProxyCliImpl> ProxyCliImplPtr;

class ProxyCli {
 public:
  class Status {
   public:
    enum StatusType {
      kOK = 0,
      kFatal,
      kUnknown,
    };

   public:
    Status() : status_(kOK), err_() {}
    const std::string& err() { return err_; }
    bool OK() { return status_ == kOK; }
    void SetError(StatusType status, const std::string& err) {
      status_ = status;
      if (status_str_.find(status) == status_str_.end()) status_ = kUnknown;
      err_ = status_str_.at(status_) + err_;
    }

   private:
    StatusType status_;
    std::string err_;
    static const std::map<StatusType, std::string> status_str_;
  };

 public:
  ProxyCli();
  ~ProxyCli();
  void Initialize(const std::string& cfg_file, int64_t device_id = 0);
  void Start();
  void Send(const torch::Tensor& tensor,
            std::optional<int64_t> rank = std::nullopt);
  void Receive(torch::Tensor& tensor,
               std::optional<int64_t> rank = std::nullopt);
  void AsyncSend(const torch::Tensor& tensor,
                 std::optional<int64_t> rank = std::nullopt);
  void AsyncReceive(torch::Tensor& tensor,
                    std::optional<int64_t> rank = std::nullopt);

 private:
  ProxyCliImplPtr svr_;
  ProxyEnvCfg context_;
  // Worker pools declared before loop_thread_ for correct destruction order:
  // loop_thread_ is destroyed first (draining callbacks), then pools.
  std::unique_ptr<XtGemmWorkerPool> gpu_pool_;
  std::unique_ptr<CpuWorkerPool> cpu_pool_;
  std::shared_ptr<uevent::UeventLoopThread> loop_thread_;
};

}  // namespace backend
}  // namespace morphling
