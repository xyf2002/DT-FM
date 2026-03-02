#include <benchmark/benchmark.h>
#include <sys/mman.h>

#include <thread>
#include <vector>

#include "backend/server_base.h"

// Warm pool: acquire and release (buffer comes from free list)
static void BM_PoolAcquireRelease(benchmark::State& state) {
  AlignedBufferPool pool;
  size_t size = state.range(0);

  // Warm the pool with one buffer
  auto [warmup_ptr, warmup_bucket] = pool.Acquire(size);
  pool.Release(warmup_ptr, warmup_bucket);

  for (auto _ : state) {
    auto [ptr, bucket] = pool.Acquire(size);
    benchmark::DoNotOptimize(ptr);
    pool.Release(ptr, bucket);
  }
  state.SetBytesProcessed(state.iterations() * size);
}
BENCHMARK(BM_PoolAcquireRelease)->RangeMultiplier(4)->Range(4096, 16 << 20);

// Cold pool: acquire from empty pool (fresh allocation)
static void BM_PoolColdAllocation(benchmark::State& state) {
  size_t size = state.range(0);

  for (auto _ : state) {
    AlignedBufferPool pool;
    auto [ptr, bucket] = pool.Acquire(size);
    benchmark::DoNotOptimize(ptr);
    pool.Release(ptr, bucket);
    // Pool destructor frees the buffer
  }
  state.SetBytesProcessed(state.iterations() * size);
}
BENCHMARK(BM_PoolColdAllocation)->RangeMultiplier(4)->Range(4096, 16 << 20);

// Raw posix_memalign + mlock + munlock + free (baseline comparison)
static void BM_RawPosixMemalign(benchmark::State& state) {
  size_t size = state.range(0);

  for (auto _ : state) {
    void* ptr = nullptr;
    int ret = posix_memalign(&ptr, 4096, size);
    benchmark::DoNotOptimize(ret);
    if (ptr) {
      mlock(ptr, size);
      benchmark::DoNotOptimize(ptr);
      munlock(ptr, size);
      free(ptr);
    }
  }
  state.SetBytesProcessed(state.iterations() * size);
}
BENCHMARK(BM_RawPosixMemalign)->RangeMultiplier(4)->Range(4096, 16 << 20);

// Multi-thread contention on shared pool
static void BM_MultiThread_PoolContention(benchmark::State& state) {
  static AlignedBufferPool shared_pool;
  size_t size = 4096;

  for (auto _ : state) {
    auto [ptr, bucket] = shared_pool.Acquire(size);
    benchmark::DoNotOptimize(ptr);
    shared_pool.Release(ptr, bucket);
  }
  state.SetBytesProcessed(state.iterations() * size);
}
BENCHMARK(BM_MultiThread_PoolContention)
    ->Threads(1)
    ->Threads(2)
    ->Threads(4)
    ->Threads(8)
    ->Threads(16);
