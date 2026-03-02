#include <benchmark/benchmark.h>

#include <algorithm>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "common/types_and_defs.h"
#include "scheduler/cpu_worker.h"
#include "scheduler/gpu_worker.h"
#include "scheduler/sliding_window_tracker.h"
#include "utils/logger.h"
#include "utils/thread_affinity.h"

// ---- RAII helpers --------------------------------------------------------

// Pinned host memory (reused from bench_xtgemm_worker.cpp)
struct PinnedBuffer {
  float* ptr = nullptr;
  size_t bytes = 0;

  PinnedBuffer() = default;

  explicit PinnedBuffer(size_t elems) : bytes(elems * sizeof(float)) {
    if (elems > 0) {
      cudaHostAlloc(reinterpret_cast<void**>(&ptr), bytes,
                    cudaHostAllocDefault);
      for (size_t i = 0; i < elems; i++) {
        ptr[i] = static_cast<float>(i % 1000) * 0.001f;
      }
    }
  }
  ~PinnedBuffer() {
    if (ptr) cudaFreeHost(ptr);
  }
  PinnedBuffer(const PinnedBuffer&) = delete;
  PinnedBuffer& operator=(const PinnedBuffer&) = delete;
  PinnedBuffer(PinnedBuffer&& o) noexcept : ptr(o.ptr), bytes(o.bytes) {
    o.ptr = nullptr;
    o.bytes = 0;
  }
  PinnedBuffer& operator=(PinnedBuffer&& o) noexcept {
    if (this != &o) {
      if (ptr) cudaFreeHost(ptr);
      ptr = o.ptr;
      bytes = o.bytes;
      o.ptr = nullptr;
      o.bytes = 0;
    }
    return *this;
  }
};

// Heap-allocated host memory for CPU pool tasks
struct HostBuffer {
  float* ptr = nullptr;
  size_t bytes = 0;

  HostBuffer() = default;

  explicit HostBuffer(size_t elems) : bytes(elems * sizeof(float)) {
    if (elems > 0) {
      ptr = static_cast<float*>(std::malloc(bytes));
      for (size_t i = 0; i < elems; i++) {
        ptr[i] = static_cast<float>(i % 1000) * 0.001f;
      }
    }
  }
  ~HostBuffer() { std::free(ptr); }
  HostBuffer(const HostBuffer&) = delete;
  HostBuffer& operator=(const HostBuffer&) = delete;
  HostBuffer(HostBuffer&& o) noexcept : ptr(o.ptr), bytes(o.bytes) {
    o.ptr = nullptr;
    o.bytes = 0;
  }
  HostBuffer& operator=(HostBuffer&& o) noexcept {
    if (this != &o) {
      std::free(ptr);
      ptr = o.ptr;
      bytes = o.bytes;
      o.ptr = nullptr;
      o.bytes = 0;
    }
    return *this;
  }
};

// ---- GemmArgs builders ---------------------------------------------------

// TN layout matching proxy_cli.cc:BuildGemmArgs (transa='T', transb='N')
static std::shared_ptr<GemmArgs> MakeTNGemmArgs(int m, int n, int k,
                                                 const float* a,
                                                 const float* b, float* c) {
  auto args = std::make_shared<GemmArgs>();
  args->group_size = 1;
  args->transa[0] = 'T';
  args->transb[0] = 'N';
  args->m[0] = m;
  args->n[0] = n;
  args->k[0] = k;
  args->alpha[0] = 1.0f;
  args->beta[0] = 0.0f;
  args->a[0] = a;
  args->lda[0] = k;  // leading dim = k for transposed A
  args->b[0] = b;
  args->ldb[0] = k;  // leading dim = k for non-transposed B
  args->c[0] = c;
  args->ldc[0] = m;
  return args;
}

// ---- GPU worker helper ---------------------------------------------------

// Create a single XtGemmWorker on GPU 0 (avoids multi-GPU pool overhead).
// Matches the pattern used by bench_xtgemm_worker.cpp.
static std::shared_ptr<XtGemmWorker> MakeGpuWorker() {
  return std::make_shared<XtGemmWorker>(0, 1, 0, 256_MB);
}

// ---- CPU pool config helper ----------------------------------------------

// Build a safe CpuWorkerPool: 2 workers with cores auto-detected.
static std::unique_ptr<CpuWorkerPool> MakeCpuPool() {
  int online = morphling::GetOnlineCoreCount();
  // Need at least 2 cores (1 per worker) to avoid LOG_FATAL
  int num_workers = std::min(2, online);
  std::vector<int> cores;
  // Reserve core 0 for OS; use cores [1..online) if possible
  int start = (online > num_workers) ? 1 : 0;
  for (int i = start; cores.size() < static_cast<size_t>(num_workers); i++) {
    cores.push_back(i);
  }
  return std::make_unique<CpuWorkerPool>(
      num_workers, std::move(cores),
      WorkerSchedulingPolicy::kRoundRobinCpu);
}

// =========================================================================
// BM_GpuPool_Latency: single GEMM latency through XtGemmWorker on GPU 0
// =========================================================================
static void BM_GpuPool_Latency(benchmark::State& state) {
  static std::once_flag init_flag;
  std::call_once(init_flag, []() { InitLogger(); });

  int dim = state.range(0);
  size_t elems = static_cast<size_t>(dim) * dim;

  PinnedBuffer h_A(elems), h_B(elems), h_C(elems);
  auto worker = MakeGpuWorker();
  auto args = MakeTNGemmArgs(dim, dim, dim, h_A.ptr, h_B.ptr, h_C.ptr);

  // Warmup
  worker->AddTask("warmup", [&]() { worker->RunXtGemm(args); });
  worker->WaitTaskDone("warmup");

  int iter = 0;
  for (auto _ : state) {
    std::string tid = "gpu_lat_" + std::to_string(iter++);
    worker->AddTask(tid, [&]() { worker->RunXtGemm(args); });
    worker->WaitTaskDone(tid);
  }

  double flops = 2.0 * dim * dim * dim;
  state.counters["GFLOPS"] =
      benchmark::Counter(flops, benchmark::Counter::kIsIterationInvariantRate,
                         benchmark::Counter::kIs1000);
  worker->Stop();
}

BENCHMARK(BM_GpuPool_Latency)
    ->Arg(128)
    ->Arg(512)
    ->Arg(1024)
    ->Arg(2048)
    ->Unit(benchmark::kMicrosecond);

// =========================================================================
// BM_CpuPool_Latency: single GEMM latency through CpuWorkerPool
// =========================================================================
static void BM_CpuPool_Latency(benchmark::State& state) {
  static std::once_flag init_flag;
  std::call_once(init_flag, []() { InitLogger(); });

  int dim = state.range(0);
  size_t elems = static_cast<size_t>(dim) * dim;

  HostBuffer h_A(elems), h_B(elems), h_C(elems);
  auto cpu_pool = MakeCpuPool();
  auto args = MakeTNGemmArgs(dim, dim, dim, h_A.ptr, h_B.ptr, h_C.ptr);

  // Warmup
  cpu_pool->EnqueueGemm("warmup", args);
  cpu_pool->WaitAll();

  int iter = 0;
  for (auto _ : state) {
    std::string tid = "cpu_lat_" + std::to_string(iter++);
    cpu_pool->EnqueueGemm(tid, args);
    cpu_pool->Wait(tid);
  }

  double flops = 2.0 * dim * dim * dim;
  state.counters["GFLOPS"] =
      benchmark::Counter(flops, benchmark::Counter::kIsIterationInvariantRate,
                         benchmark::Counter::kIs1000);
}

BENCHMARK(BM_CpuPool_Latency)
    ->Arg(128)
    ->Arg(512)
    ->Arg(1024)
    ->Arg(2048)
    ->Unit(benchmark::kMicrosecond);

// =========================================================================
// BM_GpuPool_Throughput: burst of N GEMMs (dim=512), GPU only
// =========================================================================
static void BM_GpuPool_Throughput(benchmark::State& state) {
  static std::once_flag init_flag;
  std::call_once(init_flag, []() { InitLogger(); });

  int num_tasks = state.range(0);
  const int dim = 512;
  size_t elems = static_cast<size_t>(dim) * dim;

  // Per-task buffers (pinned)
  std::vector<PinnedBuffer> bufs_a, bufs_b, bufs_c;
  bufs_a.reserve(num_tasks);
  bufs_b.reserve(num_tasks);
  bufs_c.reserve(num_tasks);
  std::vector<std::shared_ptr<GemmArgs>> args_vec;
  args_vec.reserve(num_tasks);

  for (int i = 0; i < num_tasks; i++) {
    bufs_a.emplace_back(elems);
    bufs_b.emplace_back(elems);
    bufs_c.emplace_back(elems);
    args_vec.push_back(MakeTNGemmArgs(dim, dim, dim, bufs_a[i].ptr,
                                       bufs_b[i].ptr, bufs_c[i].ptr));
  }

  auto worker = MakeGpuWorker();

  // Warmup
  worker->AddTask("warmup", [&]() { worker->RunXtGemm(args_vec[0]); });
  worker->WaitTaskDone("warmup");

  int iter = 0;
  for (auto _ : state) {
    for (int i = 0; i < num_tasks; i++) {
      std::string tid =
          "gpu_tp_" + std::to_string(iter) + "_" + std::to_string(i);
      auto& a = args_vec[i];
      worker->AddTask(tid, [&worker, a]() { worker->RunXtGemm(a); });
    }
    worker->WaitTaskDone();
    iter++;
  }

  double total_flops = static_cast<double>(num_tasks) * 2.0 * dim * dim * dim;
  state.counters["GFLOPS"] = benchmark::Counter(
      total_flops, benchmark::Counter::kIsIterationInvariantRate,
      benchmark::Counter::kIs1000);
  state.counters["tasks/s"] = benchmark::Counter(
      static_cast<double>(num_tasks),
      benchmark::Counter::kIsIterationInvariantRate);

  worker->Stop();
}

BENCHMARK(BM_GpuPool_Throughput)
    ->Arg(4)
    ->Arg(8)
    ->Arg(16)
    ->Arg(32)
    ->Unit(benchmark::kMicrosecond);

// =========================================================================
// BM_CpuPool_Throughput: burst of N GEMMs (dim=512), CPU only
// =========================================================================
static void BM_CpuPool_Throughput(benchmark::State& state) {
  static std::once_flag init_flag;
  std::call_once(init_flag, []() { InitLogger(); });

  int num_tasks = state.range(0);
  const int dim = 512;
  size_t elems = static_cast<size_t>(dim) * dim;

  std::vector<HostBuffer> bufs_a, bufs_b, bufs_c;
  bufs_a.reserve(num_tasks);
  bufs_b.reserve(num_tasks);
  bufs_c.reserve(num_tasks);
  std::vector<std::shared_ptr<GemmArgs>> args_vec;
  args_vec.reserve(num_tasks);

  for (int i = 0; i < num_tasks; i++) {
    bufs_a.emplace_back(elems);
    bufs_b.emplace_back(elems);
    bufs_c.emplace_back(elems);
    args_vec.push_back(MakeTNGemmArgs(dim, dim, dim, bufs_a[i].ptr,
                                       bufs_b[i].ptr, bufs_c[i].ptr));
  }

  auto cpu_pool = MakeCpuPool();

  // Warmup
  cpu_pool->EnqueueGemm("warmup", args_vec[0]);
  cpu_pool->WaitAll();

  int iter = 0;
  for (auto _ : state) {
    for (int i = 0; i < num_tasks; i++) {
      std::string tid =
          "cpu_tp_" + std::to_string(iter) + "_" + std::to_string(i);
      cpu_pool->EnqueueGemm(tid, args_vec[i]);
    }
    cpu_pool->WaitAll();
    iter++;
  }

  double total_flops = static_cast<double>(num_tasks) * 2.0 * dim * dim * dim;
  state.counters["GFLOPS"] = benchmark::Counter(
      total_flops, benchmark::Counter::kIsIterationInvariantRate,
      benchmark::Counter::kIs1000);
  state.counters["tasks/s"] = benchmark::Counter(
      static_cast<double>(num_tasks),
      benchmark::Counter::kIsIterationInvariantRate);
}

BENCHMARK(BM_CpuPool_Throughput)
    ->Arg(4)
    ->Arg(8)
    ->Arg(16)
    ->Arg(32)
    ->Unit(benchmark::kMicrosecond);

// =========================================================================
// BM_Hybrid_RoundRobin: even/odd round-robin dispatch to GPU+CPU
// =========================================================================
static void BM_Hybrid_RoundRobin(benchmark::State& state) {
  static std::once_flag init_flag;
  std::call_once(init_flag, []() { InitLogger(); });

  int num_tasks = state.range(0);
  const int dim = 512;
  size_t elems = static_cast<size_t>(dim) * dim;

  // Over-provision: allocate num_tasks of each since dispatch is interleaved
  std::vector<PinnedBuffer> gpu_a, gpu_b, gpu_c;
  std::vector<HostBuffer> cpu_a, cpu_b, cpu_c;
  gpu_a.reserve(num_tasks);
  gpu_b.reserve(num_tasks);
  gpu_c.reserve(num_tasks);
  cpu_a.reserve(num_tasks);
  cpu_b.reserve(num_tasks);
  cpu_c.reserve(num_tasks);

  std::vector<std::shared_ptr<GemmArgs>> gpu_args, cpu_args;
  gpu_args.reserve(num_tasks);
  cpu_args.reserve(num_tasks);

  for (int i = 0; i < num_tasks; i++) {
    gpu_a.emplace_back(elems);
    gpu_b.emplace_back(elems);
    gpu_c.emplace_back(elems);
    gpu_args.push_back(MakeTNGemmArgs(dim, dim, dim, gpu_a[i].ptr,
                                       gpu_b[i].ptr, gpu_c[i].ptr));
    cpu_a.emplace_back(elems);
    cpu_b.emplace_back(elems);
    cpu_c.emplace_back(elems);
    cpu_args.push_back(MakeTNGemmArgs(dim, dim, dim, cpu_a[i].ptr,
                                       cpu_b[i].ptr, cpu_c[i].ptr));
  }

  auto gpu_worker = MakeGpuWorker();
  auto cpu_pool = MakeCpuPool();

  // Warmup both
  gpu_worker->AddTask("warmup_gpu",
                      [&]() { gpu_worker->RunXtGemm(gpu_args[0]); });
  cpu_pool->EnqueueGemm("warmup_cpu", cpu_args[0]);
  gpu_worker->WaitTaskDone("warmup_gpu");
  cpu_pool->WaitAll();

  int iter = 0;
  int64_t gpu_task_count = 0;
  int64_t cpu_task_count = 0;

  for (auto _ : state) {
    int gpu_idx = 0, cpu_idx = 0;
    for (int i = 0; i < num_tasks; i++) {
      std::string tid =
          "rr_" + std::to_string(iter) + "_" + std::to_string(i);
      if (i % 2 == 0) {
        // Even tasks -> GPU
        auto& a = gpu_args[gpu_idx++];
        gpu_worker->AddTask(tid,
                            [&gpu_worker, a]() { gpu_worker->RunXtGemm(a); });
        gpu_task_count++;
      } else {
        // Odd tasks -> CPU
        cpu_pool->EnqueueGemm(tid, cpu_args[cpu_idx++]);
        cpu_task_count++;
      }
    }
    gpu_worker->WaitTaskDone();
    cpu_pool->WaitAll();
    iter++;
  }

  double total_flops = static_cast<double>(num_tasks) * 2.0 * dim * dim * dim;
  state.counters["GFLOPS"] = benchmark::Counter(
      total_flops, benchmark::Counter::kIsIterationInvariantRate,
      benchmark::Counter::kIs1000);
  state.counters["tasks/s"] = benchmark::Counter(
      static_cast<double>(num_tasks),
      benchmark::Counter::kIsIterationInvariantRate);
  state.counters["gpu_tasks"] = static_cast<double>(gpu_task_count);
  state.counters["cpu_tasks"] = static_cast<double>(cpu_task_count);

  gpu_worker->Stop();
}

BENCHMARK(BM_Hybrid_RoundRobin)
    ->Arg(8)
    ->Arg(16)
    ->Arg(32)
    ->Arg(64)
    ->Unit(benchmark::kMicrosecond);

// =========================================================================
// BM_Hybrid_Adaptive: ShouldUseGpu() wait-estimation dispatch
// Replicates the logic from proxy_cli.cc:ShouldUseGpu()
// =========================================================================
static void BM_Hybrid_Adaptive(benchmark::State& state) {
  static std::once_flag init_flag;
  std::call_once(init_flag, []() { InitLogger(); });

  int num_tasks = state.range(0);
  const int dim = 512;
  size_t elems = static_cast<size_t>(dim) * dim;

  // Over-provision buffers for both targets since dispatch is dynamic
  std::vector<PinnedBuffer> gpu_a, gpu_b, gpu_c;
  std::vector<HostBuffer> cpu_a, cpu_b, cpu_c;
  gpu_a.reserve(num_tasks);
  gpu_b.reserve(num_tasks);
  gpu_c.reserve(num_tasks);
  cpu_a.reserve(num_tasks);
  cpu_b.reserve(num_tasks);
  cpu_c.reserve(num_tasks);

  std::vector<std::shared_ptr<GemmArgs>> gpu_args, cpu_args;
  gpu_args.reserve(num_tasks);
  cpu_args.reserve(num_tasks);

  for (int i = 0; i < num_tasks; i++) {
    gpu_a.emplace_back(elems);
    gpu_b.emplace_back(elems);
    gpu_c.emplace_back(elems);
    gpu_args.push_back(MakeTNGemmArgs(dim, dim, dim, gpu_a[i].ptr,
                                       gpu_b[i].ptr, gpu_c[i].ptr));
    cpu_a.emplace_back(elems);
    cpu_b.emplace_back(elems);
    cpu_c.emplace_back(elems);
    cpu_args.push_back(MakeTNGemmArgs(dim, dim, dim, cpu_a[i].ptr,
                                       cpu_b[i].ptr, cpu_c[i].ptr));
  }

  auto gpu_worker = MakeGpuWorker();
  auto cpu_pool = MakeCpuPool();

  // Duration trackers with same defaults as proxy_cli
  // gpu default=1000us, cpu default=5000us
  SlidingWindowDurationTracker<64> gpu_tracker(1000);
  SlidingWindowDurationTracker<64> cpu_tracker(5000);

  // Warmup both and seed trackers
  {
    auto t0 = SlidingWindowDurationTracker<64>::Now();
    gpu_worker->AddTask("warmup_gpu",
                        [&]() { gpu_worker->RunXtGemm(gpu_args[0]); });
    gpu_worker->WaitTaskDone("warmup_gpu");
    gpu_tracker.RecordDuration(
        SlidingWindowDurationTracker<64>::ElapsedUs(t0));
  }
  {
    auto t0 = SlidingWindowDurationTracker<64>::Now();
    cpu_pool->EnqueueGemm("warmup_cpu", cpu_args[0]);
    cpu_pool->WaitAll();
    cpu_tracker.RecordDuration(
        SlidingWindowDurationTracker<64>::ElapsedUs(t0));
  }

  int iter = 0;
  int64_t gpu_task_count = 0;
  int64_t cpu_task_count = 0;

  for (auto _ : state) {
    int gpu_idx = 0, cpu_idx = 0;

    for (int i = 0; i < num_tasks; i++) {
      std::string tid =
          "adap_" + std::to_string(iter) + "_" + std::to_string(i);

      // Replicate ShouldUseGpu() logic:
      // gpu_wait = pending_gpu * avg_gpu_duration
      // cpu_wait = pending_cpu * avg_cpu_duration
      // prefer GPU on tie (<=)
      int64_t gpu_wait =
          static_cast<int64_t>(gpu_worker->GetTaskCount()) *
          gpu_tracker.GetAverageDurationUs();
      int64_t cpu_wait =
          static_cast<int64_t>(cpu_pool->GetPendingTaskCount()) *
          cpu_tracker.GetAverageDurationUs();
      bool use_gpu = (gpu_wait <= cpu_wait);

      if (use_gpu) {
        auto t0 = SlidingWindowDurationTracker<64>::Now();
        auto& a = gpu_args[gpu_idx++];
        gpu_worker->AddTask(
            tid, [&gpu_worker, a]() { gpu_worker->RunXtGemm(a); },
            [&gpu_tracker, t0](const std::string&) {
              gpu_tracker.RecordDuration(
                  SlidingWindowDurationTracker<64>::ElapsedUs(t0));
            });
        gpu_task_count++;
      } else {
        auto t0 = SlidingWindowDurationTracker<64>::Now();
        cpu_pool->EnqueueGemm(
            tid, cpu_args[cpu_idx++],
            [&cpu_tracker, t0](const std::string&) {
              cpu_tracker.RecordDuration(
                  SlidingWindowDurationTracker<64>::ElapsedUs(t0));
            });
        cpu_task_count++;
      }
    }
    gpu_worker->WaitTaskDone();
    cpu_pool->WaitAll();
    iter++;
  }

  double total_flops = static_cast<double>(num_tasks) * 2.0 * dim * dim * dim;
  state.counters["GFLOPS"] = benchmark::Counter(
      total_flops, benchmark::Counter::kIsIterationInvariantRate,
      benchmark::Counter::kIs1000);
  state.counters["tasks/s"] = benchmark::Counter(
      static_cast<double>(num_tasks),
      benchmark::Counter::kIsIterationInvariantRate);
  state.counters["gpu_tasks"] = static_cast<double>(gpu_task_count);
  state.counters["cpu_tasks"] = static_cast<double>(cpu_task_count);

  gpu_worker->Stop();
}

BENCHMARK(BM_Hybrid_Adaptive)
    ->Arg(8)
    ->Arg(16)
    ->Arg(32)
    ->Arg(64)
    ->Unit(benchmark::kMicrosecond);
