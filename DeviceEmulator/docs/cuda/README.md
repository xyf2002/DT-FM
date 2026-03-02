# CUDA API Offline Reference

This directory contains auto-generated Markdown references for the CUDA
Driver API and Runtime API. It is intended for offline search by agents
without relying on web access.

## Contents

- [CUDA Driver API](driver_api.md) — 594 functions (CUDA_VERSION=12060)
- [CUDA Runtime API](runtime_api.md) — 306 functions (CUDART_VERSION=12060)

## Regenerate

Run inside the CUDA-enabled environment (Docker image) so the headers
match the CUDA version:

```bash
python3 scripts/generate_cuda_api_docs.py \
  --cuda-include /usr/local/cuda/include \
  --out docs/cuda
```
## Notes

- The content is derived from NVIDIA CUDA headers; ensure your usage
  complies with the CUDA license agreement.
- Use repository search (e.g., `cuInit`, `cudaMalloc`) for quick lookup.