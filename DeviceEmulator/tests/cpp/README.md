# C++ Tests

This directory contains standalone C++ tests and benchmarks. Build artifacts are
produced under `tests/cpp/build`.

## Test Organization

```
tests/cpp/
├── unit/           # GoogleTest sources
│   ├── cuda/      # CUDA/cuBLAS tests
│   ├── memory/    # Memory operation tests (sources only unless wired in CMake)
│   ├── ml/        # ML framework tests (sources only unless wired in CMake)
│   ├── network/   # Network/messaging tests (sources only unless wired in CMake)
│   ├── worker/    # Worker/XtGemm tests
│   └── zerocopy/  # Zero-copy buffer tests
│
├── bench/          # Google Benchmark sources
│   ├── cuda/      # CUDA benchmarks
│   └── zerocopy/  # Zero-copy benchmarks
│
└── integration/   # Integration tests (placeholder)
```

> **Note:** Only targets explicitly listed in `tests/cpp/CMakeLists.txt` are
> built. The catalog below reflects the current executable targets.

## Build Options

Default CMake behavior:

- `ENABLE_CUDA_TESTS=ON` (CUDA/cuBLAS tests)
- `ENABLE_XTGEMM_TESTS=OFF` (XtGemm worker tests + CUDA benchmarks)
- `ENABLE_ZEROCOPY_TESTS=OFF` (zerocopy tests + benchmarks)

The Docker image enables **all** suites:

```bash
cmake -S tests/cpp -B tests/cpp/build \
  -DENABLE_CUDA_TESTS=ON \
  -DENABLE_XTGEMM_TESTS=ON \
  -DENABLE_ZEROCOPY_TESTS=ON
cmake --build tests/cpp/build -j
```

### Prerequisites / Constraints

- **CUDA tests**: require CUDA toolkit + runtime.
- **XtGemm tests/benchmarks**: require CUDA driver API availability and
  compatible GPU support (green contexts / compute capability 9.0+).
- **Zerocopy tests/benchmarks**: require protobuf generation, Torch, and CUDA
  for the pinned-pool test.

## Test & Benchmark Catalog (Executable Targets)

### Unit / Worker (GoogleTest)

- `test_worker_base` (unit/worker)

### CUDA/cuBLAS Tests (GoogleTest)

- `test_cublas_hostalloc_direct`
- `test_cublas_hostregister_posix`
- `test_cublas_error15_repro`

### XtGemm Worker Tests (optional, GoogleTest)

- `test_xtgemm_worker`

### Zerocopy Tests (optional, GoogleTest)

- `test_aligned_buffer_pool`
- `test_serialization_buffer`
- `test_scatter_gather_buffer`
- `test_matrix_partition`
- `test_cuda_pinned_pool` (requires CUDA runtime)

### Benchmarks (optional, Google Benchmark)

- `bench_xtgemm_worker`
- `bench_green_ctx`
- `bench_pool_allocation`
- `bench_serialization`

## Running Tests

Tests are automatically built in the Docker image. You can run them via the
helper script or execute binaries directly.

### Run via script

```bash
# Run all test_* binaries (benchmarks are NOT included)
./tests/run_cpp_tests.sh

# Run subsets
./tests/run_cpp_tests.sh unit
./tests/run_cpp_tests.sh cuda
./tests/run_cpp_tests.sh worker
```

> The script only runs `test_*` executables. Benchmarks (`bench_*`) must be
> run directly.

### Run binaries directly

```bash
cd tests/cpp/build

# Example tests
./test_worker_base
./test_xtgemm_worker
./test_aligned_buffer_pool

# Example benchmarks
./bench_xtgemm_worker
./bench_serialization
```

### Run in Docker

```bash
# Option A: exec into existing container
docker exec -it <container_id> bash -lc "cd /app && ./tests/run_cpp_tests.sh"

# Option B: one-shot run
docker run --rm --gpus=all --ulimit memlock=-1 --cap-add IPC_LOCK \
  -v "$(pwd)":/app -w /app device-emulator:latest \
  bash -lc "./tests/run_cpp_tests.sh"
```

Run individual binaries inside the container as needed:

```bash
docker run --rm --gpus=all device-emulator:latest \
  /app/tests/cpp/build/test_xtgemm_worker
```
