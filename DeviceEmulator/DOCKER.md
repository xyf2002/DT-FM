# DeviceEmulator Docker Deployment

This document describes how to deploy DeviceEmulator using Docker.

## Requirements

### Base requirements

- Docker 20.10+
- Docker Compose 2.0+
- At least 8 GB RAM
- At least 10 GB disk space

### GPU support (optional)

If you need GPU support:

- NVIDIA GPU (compute capability 7.0+)
- NVIDIA Docker support
- `nvidia-docker2` or Docker with `nvidia-container-toolkit`

## ccache Build Caching

The Docker build uses ccache to cache C++/CUDA compilation results across
rebuilds. This requires BuildKit (enabled by default in Docker 23+, or set
`DOCKER_BUILDKIT=1`).

### How it works

- `setup.py` detects `ccache` on `$PATH` and passes
  `-DCMAKE_<LANG>_COMPILER_LAUNCHER=ccache` to CMake.
- The Dockerfile uses `--mount=type=cache,target=/ccache` so the cache persists
  across layer rebuilds.
- The cache is capped at 5 GB (`ccache -M 5G`).

### Build behavior

- First build (cold cache): normal build time, cache is populated.
- Subsequent builds: ccache hits on unchanged translation units, even when
  `COPY . /app/` invalidates the layer.

### Check cache stats

```bash
docker run --rm device-emulator:latest ccache -s
```

### Run a container with GPU

```bash
docker run --rm -itd --gpus=all --ulimit memlock=-1:-1 --cap-add IPC_LOCK device-emulator bash
```
