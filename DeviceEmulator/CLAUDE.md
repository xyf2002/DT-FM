# CLAUDE.md — DeviceEmulator (Morphling) agent notes

Call me Bessus in every conversation.

## 0) Directory knowledge (read this first)

**Top-level map:**

- `morphling/` — Python package implementation (runtime, entrypoints, utils, ops)
- `csrc/` — C++ sources
  - `csrc/backend/` — core server/proxy/mqtt components (high impact)
  - `csrc/base/` — common C++ utilities (logging, threading, etc.)
  - `csrc/checkpoint/` — checkpoint / IO components
- `proto/` — protobuf / gRPC schemas (`*.proto`) (API surface; change only when requested)
- `config/` — runtime configs
  - `config/proxy/` — proxy client/server INI configs
  - `config/emqx/`, `config/mosquitto/` — broker configs
- `scripts/` — orchestration scripts (run devices, server, profiling, analysis)
- `tests/python/` — Python tests
- `tests/cpp/` — C++ tests/benchmarks
- `cmake/` — CMake helpers
- `Dockerfile` — canonical dev/test environment (tests run in Docker only)

**Heuristics:**
- Runtime behavior / CLI / Python logic → `morphling/`, `scripts/`
- Perf / concurrency / networking → `csrc/backend/`
- Config-driven behavior → `config/**` + config loading callsites in `scripts/`/`morphling/`
- Wire format / RPC contracts → `proto/*.proto` (ask before changing)

## 1) Development workflow (Docker-only; GPU by default)

**Hard rule:** the container built from `Dockerfile` is the source of truth.

### Build (rebuild) the image

Rebuild is required after *any* code change (Python or C++).

```bash
docker build -t device-emulator:latest .
```

### Optional: interactive shell in the same environment

```bash
docker run --rm -it --gpus all device-emulator:latest bash
```

## 2) Testing policy: run tests in Docker only

**Hard rule:** run tests in a container created from the rebuilt image.

```bash
docker run --rm --gpus all device-emulator:latest python3 -m pytest tests -v
```

## 3) “Modifications need to be in sync” rules

Because the Docker image bakes the installed package + compiled artifacts, **changes must be synced by rebuilding the image**.

### Any code change (Python or C++)
1) Rebuild image
```bash
docker build -t device-emulator:latest .
```

2) Run tests in container
```bash
docker run --rm --gpus all device-emulator:latest python3 -m pytest tests -v
```

### Dependency / build changes
If you change `Dockerfile` / system deps / Python deps, rebuild and re-test the same way.

### Proto / wire contract changes
- Treat `proto/**` as public API.
- Ask/confirm before editing.
- If edited, update all consumers together (C++ + Python) and re-run the Docker tests.

## 4) Conventions

- Python lint: `ruff` (see `pyproject.toml`), line length 80.
- C++ format: `.clang-format`.
- Prefer minimal, targeted edits; follow existing patterns.

## 5) "Don't do this without asking"

- Don't change `proto/*.proto` unless explicitly requested.
- Don't do broad refactors in `csrc/backend/`.
- Don't add heavy dependencies.
- Don't provide summary of task, only when asked
- Any runtime fix, please update corresponding readme

## 6) "Do this without asking"

- Plans write to a md file as PRD
- Before action, convert PRD using taskmaster mcp

## 7) CUDA API offline reference

Use the offline CUDA Driver/Runtime API reference instead of web search:

- Index: `docs/cuda/README.md`
- Driver API: `docs/cuda/driver_api.md`
- Runtime API: `docs/cuda/runtime_api.md`

Regenerate (inside the CUDA-enabled Docker image):

```bash
python3 scripts/generate_cuda_api_docs.py \
  --cuda-include /usr/local/cuda/include \
  --out docs/cuda
```

## 8) Test organization principles

Tests follow a strict organization pattern by **language** and **test type**:

```
tests/
├── cpp/
│   ├── unit/        # Unit tests (Google Test)
│   ├── bench/       # Benchmarks (Google Benchmark)
│   └── integration/ # Integration tests
├── python/
│   ├── unit/        # Unit tests (pytest)
│   ├── bench/       # Benchmarks
│   └── integration/ # Integration tests
└── cmake/           # Shared CMake test utilities
```

**Rules:**
- All C++ tests go in `tests/cpp/`, all Python tests in `tests/python/`
- Separate `unit/` (correctness) from `bench/` (performance) at the language level
- Group tests by component/subject area within each category (e.g., `cpp/unit/cuda/`, `cpp/unit/memory/`)
- CMake helper functions go in `tests/cmake/`
- Build artifacts (e.g., `tests/cpp/build/`) should never be committed

## 9) Core architecture patterns

### Worker pool model
- `WorkerBase` → `XtGemmWorker` (GPU, cublasXt) and `CpuWorker` (MKL)
- `WorkerPool` dispatches via pluggable `SchedulingPolicy` (round-robin, shortest-wait)
- Task queue: `std::mutex` + `std::condition_variable`, atomic task counts
- Dual-path GPU/CPU with identical public interfaces (`AddTask`, `EnqueueGemm`)

### Memory & zero-copy
- Pool-based allocation everywhere: `AlignedBufferPool`, CUDA pinned pool, context slots
- Pools are bucketed by power-of-2 sizes, pre-allocated, mlocked
- Zero-copy send: `evbuffer_add_reference()` with shared_ptr cleanup callbacks
- Scatter-gather buffers: `SerializeZeroCopy()` → `ScatterGatherBufferPtr`
- RAII + move semantics for all resource wrappers

### CUDA green contexts
- SM partitioning via `cuGreenCtxCreate` (requires SM 9.0+ / Hopper)
- Pre-computed context map keyed by SM count; switch at runtime
- `ContextSlot` struct: green context + stream + cublasXt handle (RAII)
- Graceful skip on older GPUs
- **Cleanup ordering is critical:**
  1. `cuCtxSetCurrent(ctx)` **before** destroying any resource bound to that context (cuBLAS handles, streams, device memory)
  2. Free pooled/cached CUDA memory **before** destroying the contexts that own it
  3. After destroying all green contexts, call `cudaSetDevice(gpu_id)` to restore the primary context — without this, CUDA runtime cleanup at process exit will SIGSEGV
  4. Workers with threads must call `Stop()` (join) before destroying CUDA resources

## 10) Vendored & linked dependencies

- **Protobuf:** vendored in `external/protobuf/` — never use system protobuf
- **MKL:** linked via `mkl_rt`; thread-local control with `mkl_set_num_threads_local()`
- **Google Benchmark / Google Test:** used for C++ bench/unit tests
- **libevent:** used for networking layer (`evbuffer` APIs)

## 11) Concurrency rules

- All shared state mutations under `std::unique_lock<std::mutex>`
- `std::atomic<int>` for lock-free read counters (e.g., `task_count_`)
- Completion signaling via `cv_.notify_all()` + per-task-ID tracking
- CPU affinity: POSIX `sched_setaffinity`, contiguous core partitioning across workers
- Performance tracking: `SlidingWindowDurationTracker` (header-only, single-thread access from event loop)

## 12) CLI → C++ parameter flow

Runtime parameters (e.g., `device_id`, scheduling policy) flow:
  Python CLI (`--id X`) → pybind11 binding → C++ constructor parameter
Always use default values for backward compatibility.
