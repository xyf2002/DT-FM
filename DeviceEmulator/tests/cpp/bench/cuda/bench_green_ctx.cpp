#include <benchmark/benchmark.h>
#include <cuda.h>

#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "bench_cuda_utils.h"

// Helper: run one full green context lifecycle (split -> create -> destroy)
static void WarmupGreenCtxLifecycle(CUdevice dev, unsigned int min_sm) {
  CUdevResource sm = {};
  CHECK_CU_RESULT(
      cuDeviceGetDevResource(dev, &sm, CU_DEV_RESOURCE_TYPE_SM));
  unsigned int n = 0;
  CHECK_CU_RESULT(cuDevSmResourceSplitByCount(
      nullptr, &n, &sm, nullptr,
      CU_DEV_SM_RESOURCE_SPLIT_IGNORE_SM_COSCHEDULING, min_sm));
  std::vector<CUdevResource> splits(n);
  CUdevResource rem = {};
  CHECK_CU_RESULT(cuDevSmResourceSplitByCount(
      splits.data(), &n, &sm, &rem,
      CU_DEV_SM_RESOURCE_SPLIT_IGNORE_SM_COSCHEDULING, min_sm));
  CUdevResourceDesc desc = nullptr;
  CHECK_CU_RESULT(cuDevResourceGenerateDesc(&desc, &splits[0], 1));
  CUgreenCtx gc = nullptr;
  CHECK_CU_RESULT(
      cuGreenCtxCreate(&gc, desc, dev, CU_GREEN_CTX_DEFAULT_STREAM));
  CUcontext ctx = nullptr;
  CHECK_CU_RESULT(cuCtxFromGreenCtx(&ctx, gc));
  cuCtxDestroy(ctx);
  cuGreenCtxDestroy(gc);
}

// ============================================================================
// Group 1: Green Context Create/Destroy Overhead
// ============================================================================

// --- Full lifecycle fixture ---
class GreenCtxLifecycle : public benchmark::Fixture {
 public:
  void SetUp(benchmark::State& state) override {
    EnsureLoggerInit();
    EnsureDriverInit();
    sm_count_ = GetSmCount();
    num_partitions_ = state.range(0);
    min_sm_ = static_cast<unsigned int>(sm_count_ / num_partitions_);
    if (min_sm_ == 0) {
      state.SkipWithMessage("Not enough SMs for requested partitions");
      return;
    }
    CHECK_CU_RESULT(cuDeviceGet(&dev_, 0));
    WarmupGreenCtxLifecycle(dev_, min_sm_);
  }

 protected:
  CUdevice dev_ = 0;
  int sm_count_ = 0;
  int num_partitions_ = 1;
  unsigned int min_sm_ = 0;
};

BENCHMARK_DEFINE_F(GreenCtxLifecycle, FullLifecycle)
(benchmark::State& state) {
  for (auto _ : state) {
    CUdevResource device_sm = {};
    CHECK_CU_RESULT(
        cuDeviceGetDevResource(dev_, &device_sm, CU_DEV_RESOURCE_TYPE_SM));

    unsigned int nb_groups = 0;
    CHECK_CU_RESULT(cuDevSmResourceSplitByCount(
        nullptr, &nb_groups, &device_sm, nullptr,
        CU_DEV_SM_RESOURCE_SPLIT_IGNORE_SM_COSCHEDULING, min_sm_));

    std::vector<CUdevResource> splits(nb_groups);
    CUdevResource remaining = {};
    CHECK_CU_RESULT(cuDevSmResourceSplitByCount(
        splits.data(), &nb_groups, &device_sm, &remaining,
        CU_DEV_SM_RESOURCE_SPLIT_IGNORE_SM_COSCHEDULING, min_sm_));

    CUdevResourceDesc desc = nullptr;
    CHECK_CU_RESULT(cuDevResourceGenerateDesc(&desc, &splits[0], 1));

    CUgreenCtx green_ctx = nullptr;
    CHECK_CU_RESULT(cuGreenCtxCreate(&green_ctx, desc, dev_,
                                     CU_GREEN_CTX_DEFAULT_STREAM));
    CUcontext cuda_ctx = nullptr;
    CHECK_CU_RESULT(cuCtxFromGreenCtx(&cuda_ctx, green_ctx));

    cuCtxDestroy(cuda_ctx);
    cuGreenCtxDestroy(green_ctx);
  }

  state.SetLabel(std::to_string(num_partitions_) + " partitions");
  state.counters["SMs_total"] = sm_count_;
  state.counters["SMs_per_partition"] = sm_count_ / num_partitions_;
}

BENCHMARK_REGISTER_F(GreenCtxLifecycle, FullLifecycle)
    ->Arg(1)
    ->Arg(2)
    ->Arg(4)
    ->Arg(8)
    ->Unit(benchmark::kMicrosecond);

// --- Split only fixture ---
class GreenCtxSplit : public benchmark::Fixture {
 public:
  void SetUp(benchmark::State& state) override {
    EnsureLoggerInit();
    EnsureDriverInit();
    int sm_count = GetSmCount();
    int num_partitions = state.range(0);
    min_sm_ = static_cast<unsigned int>(sm_count / num_partitions);
    if (min_sm_ == 0) {
      state.SkipWithMessage("Not enough SMs for requested partitions");
      return;
    }
    CHECK_CU_RESULT(cuDeviceGet(&dev_, 0));

    // Warmup: one split cycle
    CUdevResource sm = {};
    CHECK_CU_RESULT(
        cuDeviceGetDevResource(dev_, &sm, CU_DEV_RESOURCE_TYPE_SM));
    unsigned int n = 0;
    CHECK_CU_RESULT(cuDevSmResourceSplitByCount(
        nullptr, &n, &sm, nullptr,
        CU_DEV_SM_RESOURCE_SPLIT_IGNORE_SM_COSCHEDULING, min_sm_));
    std::vector<CUdevResource> splits(n);
    CUdevResource rem = {};
    CHECK_CU_RESULT(cuDevSmResourceSplitByCount(
        splits.data(), &n, &sm, &rem,
        CU_DEV_SM_RESOURCE_SPLIT_IGNORE_SM_COSCHEDULING, min_sm_));
  }

 protected:
  CUdevice dev_ = 0;
  unsigned int min_sm_ = 0;
};

BENCHMARK_DEFINE_F(GreenCtxSplit, SplitOnly)(benchmark::State& state) {
  for (auto _ : state) {
    CUdevResource device_sm = {};
    CHECK_CU_RESULT(
        cuDeviceGetDevResource(dev_, &device_sm, CU_DEV_RESOURCE_TYPE_SM));

    unsigned int nb_groups = 0;
    CHECK_CU_RESULT(cuDevSmResourceSplitByCount(
        nullptr, &nb_groups, &device_sm, nullptr,
        CU_DEV_SM_RESOURCE_SPLIT_IGNORE_SM_COSCHEDULING, min_sm_));

    std::vector<CUdevResource> splits(nb_groups);
    CUdevResource remaining = {};
    CHECK_CU_RESULT(cuDevSmResourceSplitByCount(
        splits.data(), &nb_groups, &device_sm, &remaining,
        CU_DEV_SM_RESOURCE_SPLIT_IGNORE_SM_COSCHEDULING, min_sm_));
  }
}

BENCHMARK_REGISTER_F(GreenCtxSplit, SplitOnly)
    ->Arg(1)
    ->Arg(2)
    ->Arg(4)
    ->Arg(8)
    ->Unit(benchmark::kMicrosecond);

// --- Create only fixture (pre-computed split/desc, warmup create+destroy) ---
class GreenCtxCreate : public benchmark::Fixture {
 public:
  void SetUp(benchmark::State& state) override {
    EnsureLoggerInit();
    EnsureDriverInit();
    int sm_count = GetSmCount();
    int num_partitions = state.range(0);
    unsigned int min_sm =
        static_cast<unsigned int>(sm_count / num_partitions);
    if (min_sm == 0) {
      state.SkipWithMessage("Not enough SMs for requested partitions");
      return;
    }
    CHECK_CU_RESULT(cuDeviceGet(&dev_, 0));

    // Pre-compute split and descriptor
    CUdevResource device_sm = {};
    CHECK_CU_RESULT(
        cuDeviceGetDevResource(dev_, &device_sm, CU_DEV_RESOURCE_TYPE_SM));
    unsigned int nb_groups = 0;
    CHECK_CU_RESULT(cuDevSmResourceSplitByCount(
        nullptr, &nb_groups, &device_sm, nullptr,
        CU_DEV_SM_RESOURCE_SPLIT_IGNORE_SM_COSCHEDULING, min_sm));
    splits_.resize(nb_groups);
    CUdevResource remaining = {};
    CHECK_CU_RESULT(cuDevSmResourceSplitByCount(
        splits_.data(), &nb_groups, &device_sm, &remaining,
        CU_DEV_SM_RESOURCE_SPLIT_IGNORE_SM_COSCHEDULING, min_sm));
    desc_ = nullptr;
    CHECK_CU_RESULT(cuDevResourceGenerateDesc(&desc_, &splits_[0], 1));

    // Warmup: one create/destroy cycle
    CUgreenCtx gc = nullptr;
    CHECK_CU_RESULT(cuGreenCtxCreate(&gc, desc_, dev_,
                                     CU_GREEN_CTX_DEFAULT_STREAM));
    CUcontext ctx = nullptr;
    CHECK_CU_RESULT(cuCtxFromGreenCtx(&ctx, gc));
    cuCtxDestroy(ctx);
    cuGreenCtxDestroy(gc);
  }

 protected:
  CUdevice dev_ = 0;
  std::vector<CUdevResource> splits_;
  CUdevResourceDesc desc_ = nullptr;
};

BENCHMARK_DEFINE_F(GreenCtxCreate, CreateOnly)(benchmark::State& state) {
  for (auto _ : state) {
    CUgreenCtx green_ctx = nullptr;
    CHECK_CU_RESULT(cuGreenCtxCreate(&green_ctx, desc_, dev_,
                                     CU_GREEN_CTX_DEFAULT_STREAM));
    CUcontext cuda_ctx = nullptr;
    CHECK_CU_RESULT(cuCtxFromGreenCtx(&cuda_ctx, green_ctx));

    cuCtxDestroy(cuda_ctx);
    cuGreenCtxDestroy(green_ctx);
  }
}

BENCHMARK_REGISTER_F(GreenCtxCreate, CreateOnly)
    ->Arg(1)
    ->Arg(2)
    ->Arg(4)
    ->Arg(8)
    ->Unit(benchmark::kMicrosecond);

// --- Destroy only fixture (pre-computed desc, warmup, PauseTiming for create)
class GreenCtxDestroy : public benchmark::Fixture {
 public:
  void SetUp(benchmark::State& state) override {
    EnsureLoggerInit();
    EnsureDriverInit();
    int sm_count = GetSmCount();
    CHECK_CU_RESULT(cuDeviceGet(&dev_, 0));

    // Pre-compute split and descriptor (1 partition = all SMs)
    CUdevResource device_sm = {};
    CHECK_CU_RESULT(
        cuDeviceGetDevResource(dev_, &device_sm, CU_DEV_RESOURCE_TYPE_SM));
    unsigned int min_sm = static_cast<unsigned int>(sm_count);
    unsigned int nb_groups = 0;
    CHECK_CU_RESULT(cuDevSmResourceSplitByCount(
        nullptr, &nb_groups, &device_sm, nullptr,
        CU_DEV_SM_RESOURCE_SPLIT_IGNORE_SM_COSCHEDULING, min_sm));
    splits_.resize(nb_groups);
    CUdevResource remaining = {};
    CHECK_CU_RESULT(cuDevSmResourceSplitByCount(
        splits_.data(), &nb_groups, &device_sm, &remaining,
        CU_DEV_SM_RESOURCE_SPLIT_IGNORE_SM_COSCHEDULING, min_sm));
    desc_ = nullptr;
    CHECK_CU_RESULT(cuDevResourceGenerateDesc(&desc_, &splits_[0], 1));

    // Warmup: one create/destroy cycle
    CUgreenCtx gc = nullptr;
    CHECK_CU_RESULT(cuGreenCtxCreate(&gc, desc_, dev_,
                                     CU_GREEN_CTX_DEFAULT_STREAM));
    CUcontext ctx = nullptr;
    CHECK_CU_RESULT(cuCtxFromGreenCtx(&ctx, gc));
    cuCtxDestroy(ctx);
    cuGreenCtxDestroy(gc);
  }

 protected:
  CUdevice dev_ = 0;
  std::vector<CUdevResource> splits_;
  CUdevResourceDesc desc_ = nullptr;
};

BENCHMARK_DEFINE_F(GreenCtxDestroy, DestroyOnly)
(benchmark::State& state) {
  for (auto _ : state) {
    state.PauseTiming();
    CUgreenCtx green_ctx = nullptr;
    CHECK_CU_RESULT(cuGreenCtxCreate(&green_ctx, desc_, dev_,
                                     CU_GREEN_CTX_DEFAULT_STREAM));
    CUcontext cuda_ctx = nullptr;
    CHECK_CU_RESULT(cuCtxFromGreenCtx(&cuda_ctx, green_ctx));
    state.ResumeTiming();

    cuCtxDestroy(cuda_ctx);
    cuGreenCtxDestroy(green_ctx);
  }
}

BENCHMARK_REGISTER_F(GreenCtxDestroy, DestroyOnly)
    ->Unit(benchmark::kMicrosecond);

// --- Regular context baseline fixture ---
class RegularCtx : public benchmark::Fixture {
 public:
  void SetUp(benchmark::State& state) override {
    EnsureDriverInit();
    CHECK_CU_RESULT(cuDeviceGet(&dev_, 0));

    // Warmup: one create/destroy cycle
    CUcontext ctx = nullptr;
    CHECK_CU_RESULT(cuCtxCreate(&ctx, 0, dev_));
    cuCtxDestroy(ctx);
  }

 protected:
  CUdevice dev_ = 0;
};

BENCHMARK_DEFINE_F(RegularCtx, Baseline)(benchmark::State& state) {
  for (auto _ : state) {
    CUcontext ctx = nullptr;
    CHECK_CU_RESULT(cuCtxCreate(&ctx, 0, dev_));
    cuCtxDestroy(ctx);
  }
}

BENCHMARK_REGISTER_F(RegularCtx, Baseline)->Unit(benchmark::kMicrosecond);

// ============================================================================
// Group 2: SM Count vs GEMM Performance
// ============================================================================

// --- Single-partition GEMM perf fixture ---
class GreenCtxGemmPerf : public benchmark::Fixture {
 public:
  void SetUp(benchmark::State& state) override {
    EnsureLoggerInit();
    num_partitions_ = state.range(0);
    dim_ = state.range(1);
    size_t elems = static_cast<size_t>(dim_) * dim_;

    h_A_ = PinnedBuffer(elems);
    h_B_ = PinnedBuffer(elems);
    h_C_ = PinnedBuffer(elems);

    worker_ = std::make_shared<XtGemmWorker>(
        0, num_partitions_, 0, 512_MB);
    args_ = MakeNNGemmArgs(dim_, dim_, dim_,
                           h_A_.ptr, h_B_.ptr, h_C_.ptr);

    // Warmup GEMM
    worker_->AddTask("warmup", [this]() { worker_->RunXtGemm(args_); });
    worker_->WaitTaskDone("warmup");
  }

  void TearDown(benchmark::State& state) override {
    worker_->Stop();
    worker_.reset();
    args_.reset();
    h_A_ = PinnedBuffer();
    h_B_ = PinnedBuffer();
    h_C_ = PinnedBuffer();
  }

 protected:
  int num_partitions_ = 1;
  int dim_ = 0;
  PinnedBuffer h_A_, h_B_, h_C_;
  std::shared_ptr<XtGemmWorker> worker_;
  std::shared_ptr<GemmArgs> args_;
};

BENCHMARK_DEFINE_F(GreenCtxGemmPerf, GemmPerf)
(benchmark::State& state) {
  int iter = 0;
  for (auto _ : state) {
    std::string tid = "iter_" + std::to_string(iter++);
    worker_->AddTask(tid, [this]() { worker_->RunXtGemm(args_); });
    worker_->WaitTaskDone(tid);
  }

  double flops = 2.0 * dim_ * dim_ * dim_;
  state.counters["GFLOPS"] =
      benchmark::Counter(flops, benchmark::Counter::kIsIterationInvariantRate,
                         benchmark::Counter::kIs1000);
  state.counters["SMs_assigned"] = GetSmCount() / num_partitions_;
}

BENCHMARK_REGISTER_F(GreenCtxGemmPerf, GemmPerf)
    ->Args({1, 512})
    ->Args({1, 1024})
    ->Args({1, 2048})
    ->Args({1, 4096})
    ->Args({2, 512})
    ->Args({2, 1024})
    ->Args({2, 2048})
    ->Args({2, 4096})
    ->Args({4, 512})
    ->Args({4, 1024})
    ->Args({4, 2048})
    ->Args({4, 4096})
    ->Unit(benchmark::kMicrosecond);

// --- Aggregate scaling fixture ---
class GreenCtxGemmScaling : public benchmark::Fixture {
 public:
  void SetUp(benchmark::State& state) override {
    EnsureLoggerInit();
    num_partitions_ = state.range(0);
    const size_t elems = static_cast<size_t>(kDim) * kDim;

    data_.clear();
    data_.reserve(num_partitions_);
    for (int i = 0; i < num_partitions_; i++) {
      auto d = std::make_unique<WorkerData>();
      d->h_A = PinnedBuffer(elems);
      d->h_B = PinnedBuffer(elems);
      d->h_C = PinnedBuffer(elems);
      d->worker = std::make_shared<XtGemmWorker>(
          0, num_partitions_, i, 512_MB);
      d->args = MakeNNGemmArgs(kDim, kDim, kDim,
                               d->h_A.ptr, d->h_B.ptr, d->h_C.ptr);
      data_.push_back(std::move(d));
    }

    // Warmup all workers
    for (int i = 0; i < num_partitions_; i++) {
      data_[i]->worker->AddTask(
          "warmup_" + std::to_string(i),
          [&d = data_[i]]() { d->worker->RunXtGemm(d->args); });
    }
    for (auto& d : data_) {
      d->worker->WaitTaskDone();
    }
  }

  void TearDown(benchmark::State& state) override {
    for (auto& d : data_) {
      d->worker->Stop();
    }
    data_.clear();
  }

 protected:
  static constexpr int kDim = 1024;

  struct WorkerData {
    PinnedBuffer h_A{0};
    PinnedBuffer h_B{0};
    PinnedBuffer h_C{0};
    std::shared_ptr<XtGemmWorker> worker;
    std::shared_ptr<GemmArgs> args;
  };

  int num_partitions_ = 1;
  std::vector<std::unique_ptr<WorkerData>> data_;
};

BENCHMARK_DEFINE_F(GreenCtxGemmScaling, GemmScaling)
(benchmark::State& state) {
  int iter = 0;
  for (auto _ : state) {
    for (int i = 0; i < num_partitions_; i++) {
      std::string tid =
          "iter" + std::to_string(iter) + "_w" + std::to_string(i);
      data_[i]->worker->AddTask(
          tid, [&d = data_[i]]() { d->worker->RunXtGemm(d->args); });
    }
    for (auto& d : data_) {
      d->worker->WaitTaskDone();
    }
    iter++;
  }

  double total_flops =
      static_cast<double>(num_partitions_) * 2.0 * kDim * kDim * kDim;
  state.counters["Aggregate_GFLOPS"] = benchmark::Counter(
      total_flops, benchmark::Counter::kIsIterationInvariantRate,
      benchmark::Counter::kIs1000);
  state.counters["Partitions"] = num_partitions_;
}

BENCHMARK_REGISTER_F(GreenCtxGemmScaling, GemmScaling)
    ->Arg(1)
    ->Arg(2)
    ->Arg(4)
    ->Unit(benchmark::kMicrosecond);

// ============================================================================
// Group 3: Non-Uniform SM Split Benchmarks
// ============================================================================

// NonUniformSplit and GreenCtxSlot are defined in bench_cuda_utils.h

// --- Experiment 1: MultiGreenCtxCoexist ---
// Create multiple green contexts simultaneously with different SM counts
// (geometric: 1, 2, 4, 8 base groups) and run concurrent GEMMs.

class MultiGreenCtxCoexist : public benchmark::Fixture {
 public:
  void SetUp(benchmark::State& state) override {
    EnsureLoggerInit();
    dim_ = state.range(0);
    size_t elems = static_cast<size_t>(dim_) * dim_;
    size_t bytes = elems * sizeof(float);

    if (!split_.Init()) {
      state.SkipWithMessage("NonUniformSplit failed");
      return;
    }

    // Geometric allocation: slot i gets 2^i groups => need 1+2+4+8=15 groups
    static constexpr int kSlotGroups[] = {1, 2, 4, 8};
    static constexpr int kNumSlots = 4;
    static constexpr int kTotalGroupsNeeded = 15;

    if (split_.nb_groups < kTotalGroupsNeeded) {
      state.SkipWithMessage(
          "Not enough groups (" + std::to_string(split_.nb_groups) +
          ") for 15 required");
      return;
    }

    slots_.resize(kNumSlots);
    unsigned int offset = 0;
    for (int i = 0; i < kNumSlots; i++) {
      auto& s = slots_[i];
      s.num_groups = kSlotGroups[i];
      s.sm_count = s.num_groups * static_cast<int>(split_.group_sm_count);

      CHECK_CU_RESULT(cuDevResourceGenerateDesc(
          &s.desc, &split_.all_groups[offset], s.num_groups));
      offset += s.num_groups;

      CHECK_CU_RESULT(cuGreenCtxCreate(&s.green_ctx, s.desc, split_.dev,
                                       CU_GREEN_CTX_DEFAULT_STREAM));
      CHECK_CU_RESULT(cuCtxFromGreenCtx(&s.cuda_ctx, s.green_ctx));

      // Create stream and cuBLAS within this green context
      CHECK_CU_RESULT(cuCtxSetCurrent(s.cuda_ctx));
      CHECK_CU_RESULT(cuGreenCtxStreamCreate(&s.cu_stream, s.green_ctx,
                                             CU_STREAM_NON_BLOCKING, 0));
      s.stream = s.cu_stream;

      CHECK_CUBLAS_ERROR(cublasCreate(&s.cublas_handle));
      CHECK_CUBLAS_ERROR(cublasSetStream(s.cublas_handle, s.stream));

      // Allocate device memory in this context
      CHECK_CUDA_ERROR(
          cudaMalloc(reinterpret_cast<void**>(&s.d_A), bytes));
      CHECK_CUDA_ERROR(
          cudaMalloc(reinterpret_cast<void**>(&s.d_B), bytes));
      CHECK_CUDA_ERROR(
          cudaMalloc(reinterpret_cast<void**>(&s.d_C), bytes));

      // Pinned host buffers per slot
      s.h_A = PinnedBuffer(elems);
      s.h_B = PinnedBuffer(elems);
      s.h_C = PinnedBuffer(elems);

      s.args = MakeNNGemmArgs(dim_, dim_, dim_,
                              s.h_A.ptr, s.h_B.ptr, s.h_C.ptr);
    }

    // Warmup: run one GEMM per slot
    for (auto& s : slots_) {
      CHECK_CU_RESULT(cuCtxSetCurrent(s.cuda_ctx));
      RunGemmDirect(s.cublas_handle, s.stream, s.d_A, s.d_B, s.d_C,
                    s.args.get());
      CHECK_CUDA_ERROR(cudaStreamSynchronize(s.stream));
    }
  }

  void TearDown(benchmark::State&) override {
    for (auto& s : slots_) {
      if (s.cuda_ctx) {
        CHECK_CU_RESULT(cuCtxSetCurrent(s.cuda_ctx));
      }
      if (s.stream) cudaStreamSynchronize(s.stream);
      if (s.cublas_handle) cublasDestroy(s.cublas_handle);
      if (s.stream) cudaStreamDestroy(s.stream);
      if (s.d_A) cudaFree(s.d_A);
      if (s.d_B) cudaFree(s.d_B);
      if (s.d_C) cudaFree(s.d_C);
      if (s.cuda_ctx) cuCtxDestroy(s.cuda_ctx);
      if (s.green_ctx) cuGreenCtxDestroy(s.green_ctx);
    }
    slots_.clear();
  }

 protected:
  int dim_ = 0;
  NonUniformSplit split_;
  std::vector<GreenCtxSlot> slots_;
};

BENCHMARK_DEFINE_F(MultiGreenCtxCoexist, ConcurrentGemm)
(benchmark::State& state) {
  for (auto _ : state) {
    // Launch GEMM on all slots
    for (auto& s : slots_) {
      CHECK_CU_RESULT(cuCtxSetCurrent(s.cuda_ctx));
      RunGemmDirect(s.cublas_handle, s.stream, s.d_A, s.d_B, s.d_C,
                    s.args.get());
    }
    // Sync all
    for (auto& s : slots_) {
      CHECK_CUDA_ERROR(cudaStreamSynchronize(s.stream));
    }
  }

  double total_flops = 0;
  for (size_t i = 0; i < slots_.size(); i++) {
    double flops = 2.0 * dim_ * dim_ * dim_;
    total_flops += flops;
    state.counters["SMs_slot" + std::to_string(i)] = slots_[i].sm_count;
  }
  state.counters["Aggregate_GFLOPS"] = benchmark::Counter(
      total_flops, benchmark::Counter::kIsIterationInvariantRate,
      benchmark::Counter::kIs1000);
  state.counters["num_contexts"] = static_cast<double>(slots_.size());
}

BENCHMARK_REGISTER_F(MultiGreenCtxCoexist, ConcurrentGemm)
    ->Arg(512)
    ->Arg(1024)
    ->Arg(2048)
    ->Unit(benchmark::kMicrosecond);

// Measure cuCtxSetCurrent overhead when round-robin switching between
// non-uniform green contexts (1, 2, 4, 8 groups of SMs).
BENCHMARK_DEFINE_F(MultiGreenCtxCoexist, CtxSwitchOverhead)
(benchmark::State& state) {
  int idx = 0;
  int n = static_cast<int>(slots_.size());
  for (auto _ : state) {
    CHECK_CU_RESULT(cuCtxSetCurrent(slots_[idx].cuda_ctx));
    idx = (idx + 1) % n;
  }
  state.counters["num_contexts"] = static_cast<double>(n);
  for (int i = 0; i < n; i++) {
    state.counters["SMs_slot" + std::to_string(i)] = slots_[i].sm_count;
  }
}

BENCHMARK_REGISTER_F(MultiGreenCtxCoexist, CtxSwitchOverhead)
    ->Arg(512)
    ->Unit(benchmark::kNanosecond);

// --- Experiment 2: GreenCtxImmutability ---
// Verify green context SM resources are immutable after creation.

// Sub-bench A: DestroyRecreate — alternates between small (1 group) and
// large (4 groups) descriptors. Measures destroy+recreate cycle cost.
class GreenCtxImmutability : public benchmark::Fixture {
 public:
  void SetUp(benchmark::State& state) override {
    EnsureLoggerInit();
    if (!split_.Init()) {
      state.SkipWithMessage("NonUniformSplit failed");
      return;
    }
    if (split_.nb_groups < 4) {
      state.SkipWithMessage(
          "Not enough groups (" + std::to_string(split_.nb_groups) +
          ") for immutability test");
      return;
    }

    // Pre-build small (1 group) and large (4 groups) descriptors
    CHECK_CU_RESULT(
        cuDevResourceGenerateDesc(&small_desc_, &split_.all_groups[0], 1));
    CHECK_CU_RESULT(
        cuDevResourceGenerateDesc(&large_desc_, &split_.all_groups[0], 4));

    // Warmup: one create/destroy cycle with each
    for (auto desc : {small_desc_, large_desc_}) {
      CUgreenCtx gc = nullptr;
      CHECK_CU_RESULT(cuGreenCtxCreate(&gc, desc, split_.dev,
                                       CU_GREEN_CTX_DEFAULT_STREAM));
      CUcontext ctx = nullptr;
      CHECK_CU_RESULT(cuCtxFromGreenCtx(&ctx, gc));
      cuCtxDestroy(ctx);
      cuGreenCtxDestroy(gc);
    }
  }

 protected:
  NonUniformSplit split_;
  CUdevResourceDesc small_desc_ = nullptr;
  CUdevResourceDesc large_desc_ = nullptr;
};

BENCHMARK_DEFINE_F(GreenCtxImmutability, DestroyRecreate)
(benchmark::State& state) {
  bool use_small = true;
  for (auto _ : state) {
    CUdevResourceDesc desc = use_small ? small_desc_ : large_desc_;
    use_small = !use_small;

    CUgreenCtx gc = nullptr;
    CHECK_CU_RESULT(cuGreenCtxCreate(&gc, desc, split_.dev,
                                     CU_GREEN_CTX_DEFAULT_STREAM));
    CUcontext ctx = nullptr;
    CHECK_CU_RESULT(cuCtxFromGreenCtx(&ctx, gc));

    cuCtxDestroy(ctx);
    cuGreenCtxDestroy(gc);
  }

  state.counters["small_SMs"] =
      1 * static_cast<double>(split_.group_sm_count);
  state.counters["large_SMs"] =
      4 * static_cast<double>(split_.group_sm_count);
}

BENCHMARK_REGISTER_F(GreenCtxImmutability, DestroyRecreate)
    ->Unit(benchmark::kMicrosecond);

BENCHMARK_DEFINE_F(GreenCtxImmutability, VerifyBoundSMs)
(benchmark::State& state) {
  // Create a green ctx with 4 groups and verify SM count is stable
  CUgreenCtx gc = nullptr;
  CHECK_CU_RESULT(cuGreenCtxCreate(&gc, large_desc_, split_.dev,
                                   CU_GREEN_CTX_DEFAULT_STREAM));
  CUcontext ctx = nullptr;
  CHECK_CU_RESULT(cuCtxFromGreenCtx(&ctx, gc));

  unsigned int expected_sm =
      4 * static_cast<unsigned int>(split_.group_sm_count);
  int queries = 0;

  for (auto _ : state) {
    CUdevResource queried = {};
    CHECK_CU_RESULT(
        cuGreenCtxGetDevResource(gc, &queried, CU_DEV_RESOURCE_TYPE_SM));
    if (queried.sm.smCount != expected_sm) {
      state.SkipWithMessage(
          "SM count mismatch: expected " + std::to_string(expected_sm) +
          " got " + std::to_string(queried.sm.smCount));
      break;
    }
    queries++;
  }

  cuCtxDestroy(ctx);
  cuGreenCtxDestroy(gc);

  state.counters["expected_SMs"] = static_cast<double>(expected_sm);
  state.counters["queries_ok"] = queries;
}

BENCHMARK_REGISTER_F(GreenCtxImmutability, VerifyBoundSMs)
    ->Unit(benchmark::kNanosecond);

// --- Experiment 3: RoundRobinGreenCtxStreams ---
// Pre-create green contexts with geometric SM counts, round-robin GEMM tasks.

class RoundRobinGreenCtxStreams : public benchmark::Fixture {
 public:
  void SetUp(benchmark::State& state) override {
    EnsureLoggerInit();
    dim_ = state.range(0);
    total_tasks_ = state.range(1);
    size_t elems = static_cast<size_t>(dim_) * dim_;
    size_t bytes = elems * sizeof(float);

    if (!split_.Init()) {
      state.SkipWithMessage("NonUniformSplit failed");
      return;
    }

    // Geometric allocation: slot i gets 2^i groups.
    // Max slots where 2^0 + ... + 2^(N-1) = 2^N - 1 <= nb_groups
    num_slots_ = 0;
    unsigned int sum = 0;
    while (sum + (1u << num_slots_) <= split_.nb_groups) {
      sum += (1u << num_slots_);
      num_slots_++;
    }
    if (num_slots_ == 0) {
      state.SkipWithMessage("Not enough groups for any slot");
      return;
    }

    slots_.resize(num_slots_);
    unsigned int offset = 0;
    for (int i = 0; i < num_slots_; i++) {
      auto& s = slots_[i];
      s.num_groups = (1 << i);
      s.sm_count = s.num_groups * static_cast<int>(split_.group_sm_count);

      CHECK_CU_RESULT(cuDevResourceGenerateDesc(
          &s.desc, &split_.all_groups[offset], s.num_groups));
      offset += s.num_groups;

      CHECK_CU_RESULT(cuGreenCtxCreate(&s.green_ctx, s.desc, split_.dev,
                                       CU_GREEN_CTX_DEFAULT_STREAM));
      CHECK_CU_RESULT(cuCtxFromGreenCtx(&s.cuda_ctx, s.green_ctx));

      CHECK_CU_RESULT(cuCtxSetCurrent(s.cuda_ctx));
      CHECK_CU_RESULT(cuGreenCtxStreamCreate(&s.cu_stream, s.green_ctx,
                                             CU_STREAM_NON_BLOCKING, 0));
      s.stream = s.cu_stream;

      CHECK_CUBLAS_ERROR(cublasCreate(&s.cublas_handle));
      CHECK_CUBLAS_ERROR(cublasSetStream(s.cublas_handle, s.stream));

      CHECK_CUDA_ERROR(
          cudaMalloc(reinterpret_cast<void**>(&s.d_A), bytes));
      CHECK_CUDA_ERROR(
          cudaMalloc(reinterpret_cast<void**>(&s.d_B), bytes));
      CHECK_CUDA_ERROR(
          cudaMalloc(reinterpret_cast<void**>(&s.d_C), bytes));

      s.h_A = PinnedBuffer(elems);
      s.h_B = PinnedBuffer(elems);
      s.h_C = PinnedBuffer(elems);

      s.args = MakeNNGemmArgs(dim_, dim_, dim_,
                              s.h_A.ptr, s.h_B.ptr, s.h_C.ptr);
    }

    // Warmup
    for (auto& s : slots_) {
      CHECK_CU_RESULT(cuCtxSetCurrent(s.cuda_ctx));
      RunGemmDirect(s.cublas_handle, s.stream, s.d_A, s.d_B, s.d_C,
                    s.args.get());
      CHECK_CUDA_ERROR(cudaStreamSynchronize(s.stream));
    }
  }

  void TearDown(benchmark::State&) override {
    for (auto& s : slots_) {
      if (s.cuda_ctx) {
        CHECK_CU_RESULT(cuCtxSetCurrent(s.cuda_ctx));
      }
      if (s.stream) cudaStreamSynchronize(s.stream);
      if (s.cublas_handle) cublasDestroy(s.cublas_handle);
      if (s.stream) cudaStreamDestroy(s.stream);
      if (s.d_A) cudaFree(s.d_A);
      if (s.d_B) cudaFree(s.d_B);
      if (s.d_C) cudaFree(s.d_C);
      if (s.cuda_ctx) cuCtxDestroy(s.cuda_ctx);
      if (s.green_ctx) cuGreenCtxDestroy(s.green_ctx);
    }
    slots_.clear();
  }

 protected:
  int dim_ = 0;
  int total_tasks_ = 0;
  int num_slots_ = 0;
  NonUniformSplit split_;
  std::vector<GreenCtxSlot> slots_;
};

BENCHMARK_DEFINE_F(RoundRobinGreenCtxStreams, RoundRobin)
(benchmark::State& state) {
  for (auto _ : state) {
    for (int t = 0; t < total_tasks_; t++) {
      int slot_idx = t % num_slots_;
      auto& s = slots_[slot_idx];

      // Sync this slot's stream before reusing it
      CHECK_CUDA_ERROR(cudaStreamSynchronize(s.stream));
      CHECK_CU_RESULT(cuCtxSetCurrent(s.cuda_ctx));
      RunGemmDirect(s.cublas_handle, s.stream, s.d_A, s.d_B, s.d_C,
                    s.args.get());
    }
    // Sync all at end of iteration
    for (auto& s : slots_) {
      CHECK_CUDA_ERROR(cudaStreamSynchronize(s.stream));
    }
  }

  double total_flops = static_cast<double>(total_tasks_) * 2.0 *
                       dim_ * dim_ * dim_;
  state.counters["Aggregate_GFLOPS"] = benchmark::Counter(
      total_flops, benchmark::Counter::kIsIterationInvariantRate,
      benchmark::Counter::kIs1000);
  state.counters["tasks_per_iter"] = total_tasks_;
  state.counters["num_slots"] = num_slots_;
  for (int i = 0; i < num_slots_; i++) {
    state.counters["SMs_slot" + std::to_string(i)] = slots_[i].sm_count;
  }
}

BENCHMARK_REGISTER_F(RoundRobinGreenCtxStreams, RoundRobin)
    ->Args({512, 8})
    ->Args({512, 16})
    ->Args({1024, 8})
    ->Args({1024, 16})
    ->Args({2048, 8})
    ->Args({2048, 16})
    ->Unit(benchmark::kMicrosecond);

// ============================================================================
// Group 4: Green Context Memory Consumption
// ============================================================================
// Measures GPU memory consumed per green context at three resource levels:
//   - Bare:   green ctx + CUDA ctx only
//   - Stream: + CUDA stream
//   - Full:   + CUDA stream + cuBLAS handle  (matches ContextSlot)
//
// arg(0) = number of contexts to create simultaneously.
// Memory delta is sampled with cuMemGetInfo_v2 from a stable primary
// context to avoid per-context accounting skew.

class GreenCtxMemory : public benchmark::Fixture {
 public:
  void SetUp(benchmark::State& state) override {
    EnsureLoggerInit();
    EnsureDriverInit();
    CHECK_CU_RESULT(cuDeviceGet(&dev_, 0));

    sm_count_ = GetSmCount();
    num_ctx_ = state.range(0);

    // Create a primary CUDA context that stays alive for memory queries
    CHECK_CU_RESULT(cuCtxCreate(&primary_ctx_, 0, dev_));
    CHECK_CU_RESULT(cuCtxSetCurrent(primary_ctx_));

    // Prime cuBLAS + stream runtime in the primary context so lazy init
    // doesn't pollute measurements
    {
      cublasHandle_t h;
      CHECK_CUBLAS_ERROR(cublasCreate(&h));
      cudaStream_t s;
      cudaStreamCreate(&s);
      cudaStreamDestroy(s);
      cublasDestroy(h);
    }

    // Split SMs into finest groups (minCount=2)
    CUdevResource device_sm = {};
    CHECK_CU_RESULT(
        cuDeviceGetDevResource(dev_, &device_sm, CU_DEV_RESOURCE_TYPE_SM));
    unsigned int nb = 0;
    CHECK_CU_RESULT(cuDevSmResourceSplitByCount(
        nullptr, &nb, &device_sm, nullptr,
        CU_DEV_SM_RESOURCE_SPLIT_IGNORE_SM_COSCHEDULING, 2));
    if (nb == 0) {
      state.SkipWithMessage("GPU does not support SM splitting");
      return;
    }
    groups_.resize(nb);
    CUdevResource rem = {};
    CHECK_CU_RESULT(cuDevSmResourceSplitByCount(
        groups_.data(), &nb, &device_sm, &rem,
        CU_DEV_SM_RESOURCE_SPLIT_IGNORE_SM_COSCHEDULING, 2));
    group_sm_ = groups_[0].sm.smCount;

    if (static_cast<unsigned int>(num_ctx_) > nb) {
      state.SkipWithMessage(
          "Requested " + std::to_string(num_ctx_) + " contexts but only " +
          std::to_string(nb) + " SM groups available");
      return;
    }

    // Pre-build one descriptor per context (each gets 1 SM group)
    descs_.resize(num_ctx_);
    for (int i = 0; i < num_ctx_; i++) {
      CHECK_CU_RESULT(
          cuDevResourceGenerateDesc(&descs_[i], &groups_[i], 1));
    }

    // Warmup: one green ctx create/destroy cycle
    WarmupGreenCtxLifecycle(dev_, 2);

    // Stabilize
    CHECK_CU_RESULT(cuCtxSetCurrent(primary_ctx_));
    cudaDeviceSynchronize();
  }

  void TearDown(benchmark::State&) override {
    if (primary_ctx_) {
      cuCtxDestroy(primary_ctx_);
      primary_ctx_ = nullptr;
    }
  }

 protected:
  // Query device free memory from the primary context
  size_t QueryFreeBytes() {
    CHECK_CU_RESULT(cuCtxSetCurrent(primary_ctx_));
    cudaDeviceSynchronize();
    size_t free_bytes = 0, total_bytes = 0;
    cuMemGetInfo_v2(&free_bytes, &total_bytes);
    return free_bytes;
  }

  CUdevice dev_ = 0;
  CUcontext primary_ctx_ = nullptr;
  int sm_count_ = 0;
  int num_ctx_ = 0;
  unsigned int group_sm_ = 0;
  std::vector<CUdevResource> groups_;
  std::vector<CUdevResourceDesc> descs_;
};

// --- Bare: green ctx + CUDA ctx only ---
BENCHMARK_DEFINE_F(GreenCtxMemory, BareCtx)(benchmark::State& state) {
  for (auto _ : state) {
    state.PauseTiming();
    size_t free_before = QueryFreeBytes();
    state.ResumeTiming();

    std::vector<CUgreenCtx> gcs(num_ctx_);
    std::vector<CUcontext> ctxs(num_ctx_);
    for (int i = 0; i < num_ctx_; i++) {
      CHECK_CU_RESULT(cuGreenCtxCreate(&gcs[i], descs_[i], dev_,
                                       CU_GREEN_CTX_DEFAULT_STREAM));
      CHECK_CU_RESULT(cuCtxFromGreenCtx(&ctxs[i], gcs[i]));
    }

    state.PauseTiming();
    size_t free_after = QueryFreeBytes();
    int64_t consumed = static_cast<int64_t>(free_before) -
                       static_cast<int64_t>(free_after);
    state.counters["total_MB"] =
        static_cast<double>(consumed) / (1024.0 * 1024.0);
    state.counters["per_ctx_KB"] =
        static_cast<double>(consumed) / (1024.0 * num_ctx_);

    for (int i = num_ctx_ - 1; i >= 0; i--) {
      cuCtxDestroy(ctxs[i]);
      cuGreenCtxDestroy(gcs[i]);
    }
    CHECK_CU_RESULT(cuCtxSetCurrent(primary_ctx_));
    cudaDeviceSynchronize();
    state.ResumeTiming();
  }

  state.counters["num_contexts"] = num_ctx_;
  state.counters["SMs_per_ctx"] = static_cast<double>(group_sm_);
}

BENCHMARK_REGISTER_F(GreenCtxMemory, BareCtx)
    ->Arg(1)
    ->Arg(2)
    ->Arg(4)
    ->Arg(8)
    ->Arg(16)
    ->Iterations(5)
    ->Unit(benchmark::kMicrosecond);

// --- With stream: green ctx + CUDA ctx + stream ---
BENCHMARK_DEFINE_F(GreenCtxMemory, WithStream)(benchmark::State& state) {
  for (auto _ : state) {
    state.PauseTiming();
    size_t free_before = QueryFreeBytes();
    state.ResumeTiming();

    std::vector<CUgreenCtx> gcs(num_ctx_);
    std::vector<CUcontext> ctxs(num_ctx_);
    std::vector<cudaStream_t> streams(num_ctx_);
    for (int i = 0; i < num_ctx_; i++) {
      CHECK_CU_RESULT(cuGreenCtxCreate(&gcs[i], descs_[i], dev_,
                                       CU_GREEN_CTX_DEFAULT_STREAM));
      CHECK_CU_RESULT(cuCtxFromGreenCtx(&ctxs[i], gcs[i]));
      CHECK_CU_RESULT(cuCtxSetCurrent(ctxs[i]));
      CUstream s;
      CHECK_CU_RESULT(cuGreenCtxStreamCreate(
          &s, gcs[i], CU_STREAM_NON_BLOCKING, 0));
      streams[i] = s;
    }

    state.PauseTiming();
    size_t free_after = QueryFreeBytes();
    int64_t consumed = static_cast<int64_t>(free_before) -
                       static_cast<int64_t>(free_after);
    state.counters["total_MB"] =
        static_cast<double>(consumed) / (1024.0 * 1024.0);
    state.counters["per_ctx_KB"] =
        static_cast<double>(consumed) / (1024.0 * num_ctx_);

    for (int i = num_ctx_ - 1; i >= 0; i--) {
      CHECK_CU_RESULT(cuCtxSetCurrent(ctxs[i]));
      cudaStreamDestroy(streams[i]);
      cuCtxDestroy(ctxs[i]);
      cuGreenCtxDestroy(gcs[i]);
    }
    CHECK_CU_RESULT(cuCtxSetCurrent(primary_ctx_));
    cudaDeviceSynchronize();
    state.ResumeTiming();
  }

  state.counters["num_contexts"] = num_ctx_;
  state.counters["SMs_per_ctx"] = static_cast<double>(group_sm_);
}

BENCHMARK_REGISTER_F(GreenCtxMemory, WithStream)
    ->Arg(1)
    ->Arg(2)
    ->Arg(4)
    ->Arg(8)
    ->Arg(16)
    ->Iterations(5)
    ->Unit(benchmark::kMicrosecond);

// --- Full ContextSlot: green ctx + CUDA ctx + stream + cuBLAS handle ---
BENCHMARK_DEFINE_F(GreenCtxMemory, FullSlot)(benchmark::State& state) {
  for (auto _ : state) {
    state.PauseTiming();
    size_t free_before = QueryFreeBytes();
    state.ResumeTiming();

    std::vector<CUgreenCtx> gcs(num_ctx_);
    std::vector<CUcontext> ctxs(num_ctx_);
    std::vector<cudaStream_t> streams(num_ctx_);
    std::vector<cublasHandle_t> handles(num_ctx_);
    for (int i = 0; i < num_ctx_; i++) {
      CHECK_CU_RESULT(cuGreenCtxCreate(&gcs[i], descs_[i], dev_,
                                       CU_GREEN_CTX_DEFAULT_STREAM));
      CHECK_CU_RESULT(cuCtxFromGreenCtx(&ctxs[i], gcs[i]));
      CHECK_CU_RESULT(cuCtxSetCurrent(ctxs[i]));
      CUstream s;
      CHECK_CU_RESULT(cuGreenCtxStreamCreate(
          &s, gcs[i], CU_STREAM_NON_BLOCKING, 0));
      streams[i] = s;
      CHECK_CUBLAS_ERROR(cublasCreate(&handles[i]));
      CHECK_CUBLAS_ERROR(cublasSetStream(handles[i], streams[i]));
    }

    state.PauseTiming();
    size_t free_after = QueryFreeBytes();
    int64_t consumed = static_cast<int64_t>(free_before) -
                       static_cast<int64_t>(free_after);
    state.counters["total_MB"] =
        static_cast<double>(consumed) / (1024.0 * 1024.0);
    state.counters["per_ctx_KB"] =
        static_cast<double>(consumed) / (1024.0 * num_ctx_);

    for (int i = num_ctx_ - 1; i >= 0; i--) {
      CHECK_CU_RESULT(cuCtxSetCurrent(ctxs[i]));
      cublasDestroy(handles[i]);
      cudaStreamDestroy(streams[i]);
      cuCtxDestroy(ctxs[i]);
      cuGreenCtxDestroy(gcs[i]);
    }
    CHECK_CU_RESULT(cuCtxSetCurrent(primary_ctx_));
    cudaDeviceSynchronize();
    state.ResumeTiming();
  }

  state.counters["num_contexts"] = num_ctx_;
  state.counters["SMs_per_ctx"] = static_cast<double>(group_sm_);
}

BENCHMARK_REGISTER_F(GreenCtxMemory, FullSlot)
    ->Arg(1)
    ->Arg(2)
    ->Arg(4)
    ->Arg(8)
    ->Arg(16)
    ->Iterations(5)
    ->Unit(benchmark::kMicrosecond);

// ============================================================================
// Group 5: Green Context Stream Switch + Kernel Launch Overhead
// ============================================================================
// Answers: if launching a GEMM on green ctx A's stream takes X µs, does
// launching the same GEMM on green ctx B's stream (after cuCtxSetCurrent
// switch) have the same latency, or more?
//
// Three benchmarks, all use cublasSgemm (not cublasXt) so the measurement
// captures H2D + compute + D2H on the green ctx stream:
//
//   SameCtx:        launch+sync on ctx A every iteration          (baseline)
//   AlternateCtx:   alternate A→B→A→B… each iteration             (2-ctx switch)
//   RoundRobinCtx:  round-robin across N contexts (arg = N)       (N-ctx switch)
//
// arg(0) = GEMM dim (M=N=K), arg(1) = num_contexts (ignored for SameCtx)

class GreenCtxKernelSwitch : public benchmark::Fixture {
 public:
  void SetUp(benchmark::State& state) override {
    EnsureLoggerInit();
    dim_ = state.range(0);
    num_ctx_ = state.range(1);
    size_t elems = static_cast<size_t>(dim_) * dim_;
    size_t bytes = elems * sizeof(float);

    if (!split_.Init()) {
      state.SkipWithMessage("NonUniformSplit failed");
      return;
    }

    // Each context gets 1 SM group (equal-sized partitions)
    if (static_cast<unsigned int>(num_ctx_) > split_.nb_groups) {
      state.SkipWithMessage(
          "Not enough SM groups (" + std::to_string(split_.nb_groups) +
          ") for " + std::to_string(num_ctx_) + " contexts");
      return;
    }

    slots_.resize(num_ctx_);
    for (int i = 0; i < num_ctx_; i++) {
      auto& s = slots_[i];
      s.num_groups = 1;
      s.sm_count = static_cast<int>(split_.group_sm_count);

      CHECK_CU_RESULT(cuDevResourceGenerateDesc(
          &s.desc, &split_.all_groups[i], 1));
      CHECK_CU_RESULT(cuGreenCtxCreate(&s.green_ctx, s.desc, split_.dev,
                                       CU_GREEN_CTX_DEFAULT_STREAM));
      CHECK_CU_RESULT(cuCtxFromGreenCtx(&s.cuda_ctx, s.green_ctx));

      CHECK_CU_RESULT(cuCtxSetCurrent(s.cuda_ctx));
      CHECK_CU_RESULT(cuGreenCtxStreamCreate(&s.cu_stream, s.green_ctx,
                                             CU_STREAM_NON_BLOCKING, 0));
      s.stream = s.cu_stream;

      CHECK_CUBLAS_ERROR(cublasCreate(&s.cublas_handle));
      CHECK_CUBLAS_ERROR(cublasSetStream(s.cublas_handle, s.stream));

      CHECK_CUDA_ERROR(
          cudaMalloc(reinterpret_cast<void**>(&s.d_A), bytes));
      CHECK_CUDA_ERROR(
          cudaMalloc(reinterpret_cast<void**>(&s.d_B), bytes));
      CHECK_CUDA_ERROR(
          cudaMalloc(reinterpret_cast<void**>(&s.d_C), bytes));

      s.h_A = PinnedBuffer(elems);
      s.h_B = PinnedBuffer(elems);
      s.h_C = PinnedBuffer(elems);

      s.args = MakeNNGemmArgs(dim_, dim_, dim_,
                              s.h_A.ptr, s.h_B.ptr, s.h_C.ptr);
    }

    // Warmup: run one GEMM per slot
    for (auto& s : slots_) {
      CHECK_CU_RESULT(cuCtxSetCurrent(s.cuda_ctx));
      RunGemmDirect(s.cublas_handle, s.stream, s.d_A, s.d_B, s.d_C,
                    s.args.get());
      CHECK_CUDA_ERROR(cudaStreamSynchronize(s.stream));
    }
  }

  void TearDown(benchmark::State&) override {
    for (auto& s : slots_) {
      if (s.cuda_ctx) {
        CHECK_CU_RESULT(cuCtxSetCurrent(s.cuda_ctx));
      }
      if (s.stream) cudaStreamSynchronize(s.stream);
      if (s.cublas_handle) cublasDestroy(s.cublas_handle);
      if (s.stream) cudaStreamDestroy(s.stream);
      if (s.d_A) cudaFree(s.d_A);
      if (s.d_B) cudaFree(s.d_B);
      if (s.d_C) cudaFree(s.d_C);
      if (s.cuda_ctx) cuCtxDestroy(s.cuda_ctx);
      if (s.green_ctx) cuGreenCtxDestroy(s.green_ctx);
    }
    slots_.clear();
  }

 protected:
  int dim_ = 0;
  int num_ctx_ = 1;
  NonUniformSplit split_;
  std::vector<GreenCtxSlot> slots_;
};

// Baseline: always launch on context 0, never switch
BENCHMARK_DEFINE_F(GreenCtxKernelSwitch, SameCtx)
(benchmark::State& state) {
  auto& s = slots_[0];
  CHECK_CU_RESULT(cuCtxSetCurrent(s.cuda_ctx));

  for (auto _ : state) {
    RunGemmDirect(s.cublas_handle, s.stream, s.d_A, s.d_B, s.d_C,
                  s.args.get());
    CHECK_CUDA_ERROR(cudaStreamSynchronize(s.stream));
  }

  double flops = 2.0 * dim_ * dim_ * dim_;
  state.counters["GFLOPS"] =
      benchmark::Counter(flops, benchmark::Counter::kIsIterationInvariantRate,
                         benchmark::Counter::kIs1000);
  state.counters["SMs"] = slots_[0].sm_count;
}

BENCHMARK_REGISTER_F(GreenCtxKernelSwitch, SameCtx)
    ->Args({512, 1})
    ->Args({1024, 1})
    ->Args({2048, 1})
    ->Unit(benchmark::kMicrosecond);

// Alternate between ctx 0 and ctx 1 every iteration
BENCHMARK_DEFINE_F(GreenCtxKernelSwitch, AlternateCtx)
(benchmark::State& state) {
  int idx = 0;
  for (auto _ : state) {
    auto& s = slots_[idx];
    CHECK_CU_RESULT(cuCtxSetCurrent(s.cuda_ctx));
    RunGemmDirect(s.cublas_handle, s.stream, s.d_A, s.d_B, s.d_C,
                  s.args.get());
    CHECK_CUDA_ERROR(cudaStreamSynchronize(s.stream));
    idx = 1 - idx;
  }

  double flops = 2.0 * dim_ * dim_ * dim_;
  state.counters["GFLOPS"] =
      benchmark::Counter(flops, benchmark::Counter::kIsIterationInvariantRate,
                         benchmark::Counter::kIs1000);
  state.counters["SMs_per_ctx"] = slots_[0].sm_count;
  state.counters["num_contexts"] = 2;
}

BENCHMARK_REGISTER_F(GreenCtxKernelSwitch, AlternateCtx)
    ->Args({512, 2})
    ->Args({1024, 2})
    ->Args({2048, 2})
    ->Unit(benchmark::kMicrosecond);

// Round-robin across N contexts
BENCHMARK_DEFINE_F(GreenCtxKernelSwitch, RoundRobinCtx)
(benchmark::State& state) {
  int n = static_cast<int>(slots_.size());
  int idx = 0;
  for (auto _ : state) {
    auto& s = slots_[idx];
    CHECK_CU_RESULT(cuCtxSetCurrent(s.cuda_ctx));
    RunGemmDirect(s.cublas_handle, s.stream, s.d_A, s.d_B, s.d_C,
                  s.args.get());
    CHECK_CUDA_ERROR(cudaStreamSynchronize(s.stream));
    idx = (idx + 1) % n;
  }

  double flops = 2.0 * dim_ * dim_ * dim_;
  state.counters["GFLOPS"] =
      benchmark::Counter(flops, benchmark::Counter::kIsIterationInvariantRate,
                         benchmark::Counter::kIs1000);
  state.counters["SMs_per_ctx"] = slots_[0].sm_count;
  state.counters["num_contexts"] = n;
}

BENCHMARK_REGISTER_F(GreenCtxKernelSwitch, RoundRobinCtx)
    ->Args({512, 2})
    ->Args({512, 4})
    ->Args({512, 8})
    ->Args({1024, 2})
    ->Args({1024, 4})
    ->Args({1024, 8})
    ->Args({2048, 2})
    ->Args({2048, 4})
    ->Args({2048, 8})
    ->Unit(benchmark::kMicrosecond);
