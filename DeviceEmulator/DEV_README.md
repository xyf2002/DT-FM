# Morphling Development Workflow

This workflow lets you iterate on Morphling without rebuilding the Docker image
on every edit. Source code is bind-mounted into the container, and build output
is cached in a Docker volume.

## Quick Start

### 1. Build the base image (one-time)

```bash
./dev.sh build
```

### 2. Start the dev container

```bash
./dev.sh start
```

### 3. Enter the container

```bash
./dev.sh shell
```

### 4. Test your changes

```bash
# If you changed C++ code, rebuild
./scripts/dev_build.sh

# Or rebuild from the host
./dev.sh rebuild

# Run a test workload
python3 scripts/run_devices.py \
  --num_devices 4 \
  --model_name facebook/opt-125m \
  --backend proxy \
  --seq_length 128 \
  --batch_size 1 \
  --cfg config/proxy/svr.ini
```

## Why this workflow

- Source code is mounted into the container, so edits apply immediately.
- Incremental C++ builds use a Docker volume cache.
- The dev container provides a consistent environment.

## Available Commands

| Command | Description |
| --- | --- |
| `./dev.sh build` | Build the Docker image |
| `./dev.sh start` | Start the dev container |
| `./dev.sh stop` | Stop the container |
| `./dev.sh restart` | Restart the container |
| `./dev.sh shell` | Open a shell in the container |
| `./dev.sh rebuild` | Rebuild inside the container |
| `./dev.sh run <cmd>` | Run a command in the container |
| `./dev.sh logs` | View container logs |
| `./dev.sh clean` | Remove containers and volumes |

## Common Development Flows

### Edit C++ code

1. Modify files under `csrc/`.
2. Run `./dev.sh rebuild`.
3. Test with `./dev.sh run python3 scripts/run_devices.py ...`.

### Edit Python code

1. Modify files under `morphling/`.
2. Run your tests directly (changes take effect immediately).

### Debugging

```bash
./dev.sh shell

export SPDLOG_LEVEL=debug

SPDLOG_LEVEL=debug python run_devices.py \
  --num_devices 4 \
  --model_name facebook/opt-125m \
  --backend proxy \
  --seq_length 128 \
  --batch_size 1 \
  --cfg ../config/proxy/svr.ini 2>&1 | tee server.log
```

## Bind Mounts

These host directories are mounted into the container:

- `./csrc` -> `/app/csrc`
- `./morphling` -> `/app/morphling`
- `./scripts` -> `/app/scripts`
- `./config` -> `/app/config`
- `./logs` -> `/app/logs`
- `morphling_build` volume -> `/app/build`

## Environment Variables

- `MORPHLING_DEV_MODE=1`: enable dev mode
- `SPDLOG_LEVEL=debug`: set log verbosity

## Notes

1. The first build can take a while.
2. Build output is cached in a Docker volume; deleting the volume clears cache.
3. If you hit permission errors, ensure Docker can access the repo.
4. Ensure ports 8080, 443, 39000, and 28516 are available.

## Troubleshooting

### Build fails

```bash
./dev.sh clean
./dev.sh build
```

### Permission errors

```bash
ls -la scripts/
chmod +x scripts/*.sh
```

### Port conflicts

```bash
lsof -i :28516
# Or update port mappings in docker-compose.yml
```
