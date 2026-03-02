#include "gpu_worker.h"

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <thread>

#include "sliding_window_tracker.h"
#include "utils/logger.h"

namespace {
constexpr size_t kDefaultPoolBytes = 256ull * 1024 * 1024;  // 256 MB

size_t ResolvePoolBytes(size_t buffer_size) {
  if (const char* v = std::getenv("MORPHLING_WORKER_POOL_SIZE")) {
    return std::stoull(v);
  }
  if (buffer_size > 0) return buffer_size;
  return kDefaultPoolBytes;
}
}  // namespace

// ---------------------------------------------------------------------------
// ContextSlot RAII
// ---------------------------------------------------------------------------

ContextSlot::~ContextSlot() {
  if (cuda_ctx) {
    cuCtxSetCurrent(cuda_ctx);
  }
  if (xt_handle) {
    cublasXtDestroy(xt_handle);
    xt_handle = nullptr;
  }
  if (stream) {
    cudaStreamDestroy(stream);
    stream = nullptr;
  }
  if (cuda_ctx) {
    cuCtxDestroy(cuda_ctx);
    cuda_ctx = nullptr;
  }
  if (green_ctx) {
    cuGreenCtxDestroy(green_ctx);
    green_ctx = nullptr;
  }
}

ContextSlot::ContextSlot(ContextSlot&& other) noexcept
    : sm_count(other.sm_count),
      resource_desc(other.resource_desc),
      green_ctx(other.green_ctx),
      cuda_ctx(other.cuda_ctx),
      stream(other.stream),
      xt_handle(other.xt_handle) {
  other.sm_count = 0;
  other.resource_desc = nullptr;
  other.green_ctx = nullptr;
  other.cuda_ctx = nullptr;
  other.stream = nullptr;
  other.xt_handle = nullptr;
}

ContextSlot& ContextSlot::operator=(ContextSlot&& other) noexcept {
  if (this != &other) {
    // Destroy current resources
    this->~ContextSlot();
    // Move from other
    sm_count = other.sm_count;
    resource_desc = other.resource_desc;
    green_ctx = other.green_ctx;
    cuda_ctx = other.cuda_ctx;
    stream = other.stream;
    xt_handle = other.xt_handle;
    // Null out other
    other.sm_count = 0;
    other.resource_desc = nullptr;
    other.green_ctx = nullptr;
    other.cuda_ctx = nullptr;
    other.stream = nullptr;
    other.xt_handle = nullptr;
  }
  return *this;
}

// ---------------------------------------------------------------------------
// XtGemmWorker
// ---------------------------------------------------------------------------

XtGemmWorker::XtGemmWorker(int gpu_id, int num_partitions, int partition_idx,
                           size_t buffer_size)
    : gpu_id_(gpu_id),
      num_partitions_(num_partitions),
      partition_idx_(partition_idx),
      buffer_size_(buffer_size) {
  worker_ = std::thread([this] { Run(); });
  LOG_DEBUG << "XtGemmWorker created: gpu=" << gpu_id_
            << " partition=" << partition_idx_ << "/" << num_partitions_;
}

XtGemmWorker::~XtGemmWorker() {
  if (worker_.joinable()) {
    Stop();
  }
  active_slot_ = nullptr;
  allocator_.reset();       // Free CUDA memory while contexts still exist
  context_slots_.clear();   // Then destroy contexts
  cudaSetDevice(gpu_id_);   // Restore primary context after green ctx cleanup
  LOG_DEBUG << "XtGemmWorker destroyed: gpu=" << gpu_id_
            << " partition=" << partition_idx_;
}

void XtGemmWorker::InitAllContexts() {
  CHECK_CU_RESULT(cuDeviceGet(&cu_device_, gpu_id_));

  // Query total SM count
  int sm_count = 0;
  CHECK_CU_RESULT(cuDeviceGetAttribute(
      &sm_count, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, cu_device_));
  LOG_INFO << "GPU " << gpu_id_ << " has " << sm_count << " SMs, "
           << "partitioning " << num_partitions_ << " ways for partition "
           << partition_idx_;

  // Split into finest-grained groups (minCount=2)
  CUdevResource device_sm_resource = {};
  CHECK_CU_RESULT(cuDeviceGetDevResource(cu_device_, &device_sm_resource,
                                         CU_DEV_RESOURCE_TYPE_SM));

  unsigned int nb_groups = 0;
  CHECK_CU_RESULT(cuDevSmResourceSplitByCount(
      nullptr, &nb_groups, &device_sm_resource, nullptr,
      CU_DEV_SM_RESOURCE_SPLIT_IGNORE_SM_COSCHEDULING, 2));
  LOG_FATAL_IF(nb_groups == 0) << "Cannot split SMs into groups";

  sm_groups_.resize(nb_groups);
  CUdevResource remaining = {};
  CHECK_CU_RESULT(cuDevSmResourceSplitByCount(
      sm_groups_.data(), &nb_groups, &device_sm_resource, &remaining,
      CU_DEV_SM_RESOURCE_SPLIT_IGNORE_SM_COSCHEDULING, 2));

  sm_step_ = static_cast<int>(sm_groups_[0].sm.smCount);
  LOG_INFO << "SM step size: " << sm_step_
           << " (" << nb_groups << " groups total)";

  // Divide groups among partitions
  unsigned int groups_per_partition = nb_groups / num_partitions_;
  LOG_FATAL_IF(groups_per_partition == 0)
      << "Not enough SM groups (" << nb_groups << ") for "
      << num_partitions_ << " partitions";

  partition_sm_count_ = static_cast<int>(groups_per_partition) * sm_step_;
  unsigned int base_offset = partition_idx_ * groups_per_partition;

  LOG_INFO << "Partition " << partition_idx_ << ": "
           << groups_per_partition << " groups, "
           << partition_sm_count_ << " SMs (offset=" << base_offset << ")";

  // Create a context slot for each valid SM count (1 group, 2 groups, ...)
  for (unsigned int n = 1; n <= groups_per_partition; n++) {
    int slot_sm_count = static_cast<int>(n) * sm_step_;
    auto slot = CreateContextSlot(
        &sm_groups_[base_offset], static_cast<int>(n), slot_sm_count);
    context_slots_.emplace(slot_sm_count, std::move(slot));
    LOG_INFO << "  Created context slot: " << slot_sm_count << " SMs"
             << " (" << n << " groups)";
  }

  // Default: use all partition SMs
  active_slot_ = &context_slots_.at(partition_sm_count_);
  CHECK_CU_RESULT(cuCtxSetCurrent(active_slot_->cuda_ctx));

  // Initialize per-worker CUDA memory pool
  size_t pool_bytes = ResolvePoolBytes(buffer_size_);
  allocator_ = std::make_unique<CachingAllocator>(
      pool_bytes, MemoryType::CUDA, gpu_id_);
  LOG_INFO << "XtGemmWorker gpu=" << gpu_id_
           << " partition=" << partition_idx_
           << " allocator initialized: " << pool_bytes << " bytes";

  LOG_INFO << "XtGemmWorker initialized: " << context_slots_.size()
           << " context slots, active=" << partition_sm_count_ << " SMs";
}

ContextSlot XtGemmWorker::CreateContextSlot(CUdevResource* groups,
                                            int num_groups,
                                            int sm_count) {
  ContextSlot slot;
  slot.sm_count = sm_count;

  // Combine groups into a resource descriptor
  CHECK_CU_RESULT(cuDevResourceGenerateDesc(
      &slot.resource_desc, groups, num_groups));

  // Create green context
  CHECK_CU_RESULT(cuGreenCtxCreate(
      &slot.green_ctx, slot.resource_desc, cu_device_,
      CU_GREEN_CTX_DEFAULT_STREAM));
  CHECK_CU_RESULT(cuCtxFromGreenCtx(&slot.cuda_ctx, slot.green_ctx));
  CHECK_CU_RESULT(cuCtxSetCurrent(slot.cuda_ctx));

  // Create stream within the green context
  CUstream cu_stream = nullptr;
  CHECK_CU_RESULT(cuGreenCtxStreamCreate(
      &cu_stream, slot.green_ctx, CU_STREAM_NON_BLOCKING, 0));
  slot.stream = cu_stream;

  // Create cublasXt handle — manages H2D/D2H internally
  CHECK_CUBLAS_ERROR(cublasXtCreate(&slot.xt_handle));
  int device_id = gpu_id_;
  CHECK_CUBLAS_ERROR(
      cublasXtDeviceSelect(slot.xt_handle, 1, &device_id));

  return slot;
}

bool XtGemmWorker::SwitchContext(int num_sms) {
  auto it = context_slots_.find(num_sms);
  if (it == context_slots_.end()) return false;
  active_slot_ = &it->second;
  CHECK_CU_RESULT(cuCtxSetCurrent(active_slot_->cuda_ctx));
  return true;
}

int XtGemmWorker::GetActiveSmCount() const {
  return active_slot_ ? active_slot_->sm_count : 0;
}

std::vector<int> XtGemmWorker::GetAvailableSmCounts() const {
  std::vector<int> counts;
  counts.reserve(context_slots_.size());
  for (const auto& [sm_count, _] : context_slots_) {
    counts.push_back(sm_count);
  }
  std::sort(counts.begin(), counts.end());
  return counts;
}

void XtGemmWorker::Run() {
  cudaSetDevice(gpu_id_);
  InitAllContexts();

  LOG_INFO << "XtGemmWorker ready: gpu=" << gpu_id_
           << " partition=" << partition_idx_;

  // Enter the WorkerBase task loop
  WorkerBase::Run();
}

void XtGemmWorker::RunXtGemm(std::shared_ptr<GemmArgs> args) {
  LOG_DEBUG << "RunXtGemm on gpu=" << gpu_id_
            << " partition=" << partition_idx_
            << " sms=" << (active_slot_ ? active_slot_->sm_count : 0)
            << " " << args->DebugString();

  cublasXtHandle_t handle = active_slot_->xt_handle;

  cublasOperation_t transa = CUDA_TRANS_OP(args->transa[0]);
  cublasOperation_t transb = CUDA_TRANS_OP(args->transb[0]);

  CHECK_CUBLAS_ERROR(cublasXtSgemm(
      handle, transa, transb,
      args->m[0], args->n[0], args->k[0],
      args->alpha, args->a[0], args->lda[0],
      args->b[0], args->ldb[0],
      args->beta, args->c[0], args->ldc[0]));

  CHECK_CUDA_ERROR(cudaDeviceSynchronize());

  LOG_DEBUG << "RunXtGemm completed on gpu=" << gpu_id_
            << " partition=" << partition_idx_;
}

bool XtGemmWorker::LoadSmSchedule(const std::string& path) {
  std::ifstream file(path);
  if (!file.is_open()) {
    LOG_ERROR << "Cannot open SM schedule file: " << path;
    return false;
  }

  std::vector<SmScheduleEntry> schedule;
  std::string line;
  int line_num = 0;
  while (std::getline(file, line)) {
    line_num++;
    // Skip comments and blank lines
    size_t first = line.find_first_not_of(" \t");
    if (first == std::string::npos || line[first] == '#') continue;

    std::istringstream iss(line);
    SmScheduleEntry entry;
    if (!(iss >> entry.time_offset_us >> entry.num_sms >> entry.duration_us)) {
      LOG_ERROR << "Parse error at line " << line_num << ": " << line;
      return false;
    }

    // Validate monotonic time offsets
    if (!schedule.empty() &&
        entry.time_offset_us < schedule.back().time_offset_us) {
      LOG_ERROR << "Non-monotonic time_offset_us at line " << line_num
                << ": " << entry.time_offset_us
                << " < " << schedule.back().time_offset_us;
      return false;
    }

    // Validate SM count exists
    if (context_slots_.find(entry.num_sms) == context_slots_.end()) {
      LOG_ERROR << "Invalid num_sms=" << entry.num_sms << " at line "
                << line_num << " (not in context_slots_)";
      return false;
    }

    schedule.push_back(entry);
  }

  sm_schedule_ = std::move(schedule);
  LOG_INFO << "Loaded SM schedule: " << sm_schedule_.size()
           << " entries from " << path;
  return true;
}

void XtGemmWorker::RunSmSchedule() {
  if (sm_schedule_.empty()) return;

  auto start = SlidingWindowDurationTracker<>::Now();

  for (const auto& entry : sm_schedule_) {
    // Spin-wait until this entry's time offset
    while (SlidingWindowDurationTracker<>::ElapsedUs(start) <
           entry.time_offset_us) {
      std::this_thread::yield();
    }

    SwitchContext(entry.num_sms);
    LOG_DEBUG << "SM schedule: switched to " << entry.num_sms << " SMs at t="
              << SlidingWindowDurationTracker<>::ElapsedUs(start) << "us";
  }

  // Hold until last entry's duration expires
  const auto& last = sm_schedule_.back();
  int64_t end_us = last.time_offset_us + last.duration_us;
  while (SlidingWindowDurationTracker<>::ElapsedUs(start) < end_us) {
    std::this_thread::yield();
  }

  LOG_DEBUG << "SM schedule completed: total "
            << SlidingWindowDurationTracker<>::ElapsedUs(start) << "us";
}

// ---------------------------------------------------------------------------
// XtGemmWorkerPool
// ---------------------------------------------------------------------------

XtGemmWorkerPool::XtGemmWorkerPool(int workers_per_gpu, size_t buffer_size,
                                   WorkerSchedulingPolicy policy) {
  int device_count = 0;
  CHECK_CUDA_ERROR(cudaGetDeviceCount(&device_count));

  int total_workers = workers_per_gpu * device_count;

  for (int gpu = 0; gpu < device_count; gpu++) {
    for (int p = 0; p < workers_per_gpu; p++) {
      workers_.emplace_back(
          std::make_shared<XtGemmWorker>(gpu, workers_per_gpu, p, buffer_size));
    }
  }

  switch (policy) {
    case WorkerSchedulingPolicy::kRoundRobinGemm:
      scheduler_ = std::make_unique<RoundRobinGemmPolicy>(total_workers);
      break;
    default:
      LOG_FATAL << "Unsupported scheduling policy: "
                << WorkerSchedulingPolicyToString(policy);
  }

  LOG_INFO << "XtGemmWorkerPool created: " << total_workers << " workers ("
           << workers_per_gpu << " per GPU, " << device_count
           << " GPUs), policy=" << WorkerSchedulingPolicyToString(policy);
}

XtGemmWorkerPool::~XtGemmWorkerPool() {
  for (auto& worker : workers_) {
    worker->Stop();
  }
}

TaskHandle XtGemmWorkerPool::EnqueueGemm(
    const std::string& task_id,
    std::shared_ptr<GemmArgs> args,
    TaskCallback callback) {
  auto [worker_idx, priority] = scheduler_->Schedule(args.get());
  auto task = std::bind(&XtGemmWorker::RunXtGemm, workers_[worker_idx], args);
  return workers_[worker_idx]->AddTask(
      task_id, std::move(task), std::move(callback));
}

void XtGemmWorkerPool::WaitAll() {
  for (auto& worker : workers_) {
    worker->WaitTaskDone();
  }
}

void XtGemmWorkerPool::Wait(const std::string& task_id) {
  for (auto& worker : workers_) {
    worker->WaitTaskDone(task_id);
  }
}
