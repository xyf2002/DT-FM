# ============================================================================
# Asteroid Edge Training Container
# ============================================================================
# Multi-stage build supporting both x86_64 (CUDA) and ARM64 (Jetson) architectures.
#
# Build examples:
#   # x86_64 with CUDA 12.x
#   docker build -t asteroid:latest .
#
#   # ARM64 Jetson (requires different base image)
#   docker build -t asteroid:jetson --build-arg BASE_IMAGE=nvcr.io/nvidia/l4t-pytorch:r35.2.1-pth2.0-py3 .
#
# Usage:
#   # Interactive (for debugging)
#   docker run --gpus all -it asteroid:latest bash
#
#   # K8s will inject env vars and mount hpp_plan.json
# ============================================================================

# Build argument for base image selection
ARG BASE_IMAGE=pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime

FROM ${BASE_IMAGE}

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# ============================================================================
# System Dependencies
# ============================================================================
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    python3-dev \
    git \
    curl \
    wget \
    iperf3 \
    net-tools \
    iproute2 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# ============================================================================
# Working Directory
# ============================================================================
WORKDIR /app

# ============================================================================
# Python Dependencies (installed directly from pyproject.toml)
# ============================================================================

# Upgrade pip and install build tools
RUN pip3 install --no-cache-dir --upgrade pip setuptools wheel build

# Copy project metadata and package structure first (for layer caching)
COPY pyproject.toml .
COPY asteroid/ ./asteroid/

# Install project dependencies from pyproject.toml
# --no-build-isolation allows using already-installed setuptools
# Base image already has PyTorch with CUDA, so we skip reinstalling it
RUN pip3 install --no-cache-dir --no-build-isolation . && \
    pip3 cache purge || true

# ============================================================================
# Application Code
# ============================================================================

# Copy worker scripts (these are not part of the installable package)
COPY worker.py .
COPY run_planner.py .
COPY profile_node.py .

# Create necessary directories
RUN mkdir -p /config /cache/huggingface /opt/asteroid/profiles

# ============================================================================
# Pre-download Resources (Optional - reduces startup time)
# ============================================================================

# Pre-download HuggingFace tokenizer (GPT-2)
# This is optional and can be commented out to reduce image size
RUN python3 -c "from transformers import GPT2Tokenizer; GPT2Tokenizer.from_pretrained('gpt2')" 2>/dev/null || true

# ============================================================================
# Environment Configuration
# ============================================================================

# Python configuration
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# PyTorch distributed configuration
ENV TORCH_NCCL_BLOCKING_WAIT=1
ENV NCCL_DEBUG=WARN

# HuggingFace cache
ENV HF_HOME=/cache/huggingface
ENV TRANSFORMERS_CACHE=/cache/huggingface/transformers

# Default HPP plan path (mounted via ConfigMap in K8s)
ENV HPP_PLAN_PATH=/config/hpp_plan.json

# ============================================================================
# Health Check (Optional)
# ============================================================================
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python3 -c "import torch; assert torch.cuda.is_available()" || exit 1

# ============================================================================
# Entry Point
# ============================================================================

# Default command runs the worker
ENTRYPOINT ["python3", "-u", "worker.py"]

# Default arguments (can be overridden)
CMD []
