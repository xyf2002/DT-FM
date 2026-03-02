#include <benchmark/benchmark.h>

#include <cstring>
#include <vector>

#include "backend/server_base.h"

// Helper: create a MatrixPartition with given tensor sizes
static MatrixPartition CreatePartition(size_t tensor_bytes) {
  MatrixPartition partition;
  partition.version = 1;
  partition.row = 0;
  partition.col = 0;
  partition.pivot = 0;
  partition.h_dim = 1;
  partition.dev_id = 0;
  partition.oid = 0;
  partition.gemm_id = 0;
  partition.timestamp = 1000;
  return partition;
}

// Benchmark: SerializeProto throughput
static void BM_SerializeProto(benchmark::State& state) {
  size_t tensor_bytes = state.range(0);

  // Create persistent tensor data
  std::vector<uint8_t> tensor_a(tensor_bytes / 2);
  std::vector<uint8_t> tensor_b(tensor_bytes / 2);
  std::fill(tensor_a.begin(), tensor_a.end(), 0xAA);
  std::fill(tensor_b.begin(), tensor_b.end(), 0xBB);

  MatrixPartition partition = CreatePartition(tensor_bytes);
  partition.mat.push_back({tensor_a.data(), (int64_t)tensor_a.size()});
  partition.mat.push_back({tensor_b.data(), (int64_t)tensor_b.size()});

  for (auto _ : state) {
    auto buf = partition.Serialize();
    benchmark::DoNotOptimize(buf);
  }
  state.SetBytesProcessed(state.iterations() * tensor_bytes);
}
BENCHMARK(BM_SerializeProto)->RangeMultiplier(4)->Range(1 << 10, 16 << 20);

// Benchmark: SerializeZeroCopy throughput
static void BM_SerializeZeroCopy(benchmark::State& state) {
  size_t tensor_bytes = state.range(0);

  std::vector<uint8_t> tensor_a(tensor_bytes / 2);
  std::vector<uint8_t> tensor_b(tensor_bytes / 2);
  std::fill(tensor_a.begin(), tensor_a.end(), 0xAA);
  std::fill(tensor_b.begin(), tensor_b.end(), 0xBB);

  MatrixPartition partition = CreatePartition(tensor_bytes);
  partition.mat.push_back({tensor_a.data(), (int64_t)tensor_a.size()});
  partition.mat.push_back({tensor_b.data(), (int64_t)tensor_b.size()});

  for (auto _ : state) {
    auto sg = partition.SerializeZeroCopy();
    benchmark::DoNotOptimize(sg);
  }
  state.SetBytesProcessed(state.iterations() * tensor_bytes);
}
BENCHMARK(BM_SerializeZeroCopy)->RangeMultiplier(4)->Range(1 << 10, 16 << 20);

// Benchmark: Deserialization throughput
static void BM_Deserialization(benchmark::State& state) {
  size_t tensor_bytes = state.range(0);

  std::vector<uint8_t> tensor_a(tensor_bytes / 2);
  std::vector<uint8_t> tensor_b(tensor_bytes / 2);
  std::fill(tensor_a.begin(), tensor_a.end(), 0xAA);
  std::fill(tensor_b.begin(), tensor_b.end(), 0xBB);

  MatrixPartition partition = CreatePartition(tensor_bytes);
  partition.mat.push_back({tensor_a.data(), (int64_t)tensor_a.size()});
  partition.mat.push_back({tensor_b.data(), (int64_t)tensor_b.size()});

  // Pre-serialize
  auto serialized = partition.Serialize();

  for (auto _ : state) {
    MatrixPartition deserialized;
    deserialized.Deserialize(serialized->GetBuffer(), serialized->GetSize());
    benchmark::DoNotOptimize(deserialized);
  }
  state.SetBytesProcessed(state.iterations() * tensor_bytes);
}
BENCHMARK(BM_Deserialization)->RangeMultiplier(4)->Range(1 << 10, 16 << 20);

// Benchmark: WriteBytes throughput at various chunk sizes
static void BM_WriteBytes(benchmark::State& state) {
  size_t chunk_size = state.range(0);

  std::vector<uint8_t> data(chunk_size);
  std::fill(data.begin(), data.end(), 0xCC);

  SerializationBuffer buf;
  buf.Allocate(chunk_size + 64);

  for (auto _ : state) {
    buf.SeekTo(0);
    buf.WriteBytes(data.data(), data.size());
    benchmark::DoNotOptimize(buf.GetBuffer());
  }
  state.SetBytesProcessed(state.iterations() * chunk_size);
}
BENCHMARK(BM_WriteBytes)->RangeMultiplier(4)->Range(1 << 10, 16 << 20);
