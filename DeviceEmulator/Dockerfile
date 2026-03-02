# DeviceEmulator Dockerfile
FROM pytorch/pytorch:2.9.1-cuda12.8-cudnn9-devel

RUN sed -i 's|http://security.ubuntu.com/ubuntu|http://archive.ubuntu.com/ubuntu|g' /etc/apt/sources.list || true

RUN apt-get update && apt-get upgrade -y

RUN apt-get install -y \
    curl \
    wget \
    git \
    build-essential \
    pkg-config

# Install Nsight Systems (nsys) and Nsight Compute (ncu)
RUN apt-get update && apt-get install -y \
    nsight-systems-2025.5.2 \
    nsight-compute-2025.4.1

RUN apt-get install -y \
    cmake \
    ninja-build \
    ccache

RUN apt-get install -y \
    libssl-dev \
    protobuf-compiler \
    libgrpc-dev \
    libgrpc++-dev

RUN apt-get install -y \
    libfmt-dev \
    libspdlog-dev

RUN apt-get install -y \
    libevent-dev

RUN apt-get install -y \
    libxml2-dev \
    xsltproc \
    uuid-dev \
    libmosquitto-dev \
    libhiredis-dev \
    rapidjson-dev \
    libxslt1-dev

RUN apt-get install -y \
    gdb \
    vim \
    nano \
    psmisc \
    tmux \
    iputils-ping \
    lsof \
    net-tools

RUN apt-get install -y libopenblas-dev

# 创建工作目录
WORKDIR /app

# copy requirements.txt first to leverage Docker cache
COPY requirements.txt /app/
RUN pip install --no-cache -r /app/requirements.txt

# Copy the rest of the project files
COPY . /app/

# Remove old .so files that would shadow the newly built package
RUN rm -f /app/morphling/*.so

# ccache: persist compiler cache across Docker builds (requires BuildKit)
ARG USE_CCACHE=1
ENV CCACHE_DIR=/ccache
RUN if [ "$USE_CCACHE" = "1" ]; then ccache -M 5G; fi

# 构建和安装项目（使用系统 python）with BuildKit cache mount for ccache
RUN --mount=type=cache,target=/ccache \
    if [ "$USE_CCACHE" = "1" ]; then export PATH="/usr/lib/ccache:$PATH"; fi && \
    pip install --no-build-isolation --verbose ./

# Build standalone C++ tests (enable all optional suites)
RUN --mount=type=cache,target=/ccache \
    if [ "$USE_CCACHE" = "1" ]; then export PATH="/usr/lib/ccache:$PATH"; fi && \
    cmake -S tests/cpp -B tests/cpp/build \
    -DENABLE_CUDA_TESTS=ON \
    -DENABLE_XTGEMM_TESTS=ON \
    -DENABLE_ZEROCOPY_TESTS=ON && \
    cmake --build tests/cpp/build -j

# 创建必要的目录
RUN mkdir -p /app/logs /app/data /app/config

# 设置权限
RUN chmod +x /app/scripts/*.sh || true && chmod +x /app/csrc/*.sh || true

# 暴露端口
EXPOSE 39000

# 设置环境变量用于运行时
ENV SPDLOG_LEVEL=debug
ENV MORPHLING_HOME=/app
ENV PYTHONPATH=/opt/conda/lib/python3.11/site-packages
