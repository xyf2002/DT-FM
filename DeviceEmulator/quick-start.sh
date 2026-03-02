#!/bin/bash

# DeviceEmulator 快速启动脚本

set -e

echo "=== DeviceEmulator Quick Start Script ==="

# 检查Docker是否安装
if ! command -v docker &> /dev/null; then
    echo "ERROR: Docker is not installed. Please install Docker first."
    exit 1
fi

# 检查Docker Compose是否安装
if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
    echo "ERROR: Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

# 检查NVIDIA Docker支持
if command -v nvidia-docker &> /dev/null || docker info | grep -q nvidia; then
    echo "NVIDIA Docker support detected"
    USE_GPU=true
else
    echo "WARNING: NVIDIA Docker support not detected. Running in CPU-only mode."
    USE_GPU=false
fi

# 构建镜像
echo "Building DeviceEmulator Docker image..."
docker build -t device-emulator:latest .

# 启动服务前，自动循环移除所有 redis 和 morphling 相关容器，直到全部清理干净
while :; do
    REDIS_CONTAINERS=$(docker ps -aq -f name=redis)
    MORPHLING_CONTAINERS=$(docker ps -aq -f name=morphling)
    DEVICEEMU_CONTAINERS=$(docker ps -aq -f name=deviceemulator-device-emulator)
    MORPHLING_REDIS_CONTAINERS=$(docker ps -aq -f name=morphling-redis)
    if [ -n "$REDIS_CONTAINERS" ]; then
        echo "Stopping and removing existing redis containers..."
        docker rm -f $REDIS_CONTAINERS
    fi
    if [ -n "$MORPHLING_CONTAINERS" ]; then
        echo "Stopping and removing existing morphling containers..."
        docker rm -f $MORPHLING_CONTAINERS
    fi
    if [ -n "$DEVICEEMU_CONTAINERS" ]; then
        echo "Stopping and removing existing deviceemulator-device-emulator containers..."
        docker rm -f $DEVICEEMU_CONTAINERS
    fi
    if [ -n "$MORPHLING_REDIS_CONTAINERS" ]; then
        echo "Stopping and removing existing morphling-redis containers..."
        docker rm -f $MORPHLING_REDIS_CONTAINERS
    fi
    # 如果都为空则跳出循环
    if [ -z "$REDIS_CONTAINERS" ] && [ -z "$MORPHLING_CONTAINERS" ] && [ -z "$DEVICEEMU_CONTAINERS" ] && [ -z "$MORPHLING_REDIS_CONTAINERS" ]; then
        break
    fi
    sleep 1
done

echo "Starting services..."
if [ "$USE_GPU" = true ]; then
    docker-compose up -d
else
    # 修改compose文件以移除GPU要求
    echo "Starting in CPU-only mode..."
    docker-compose -f docker-compose.yml -f docker-compose.cpu.yml up -d
fi

# 等待服务启动
echo "Waiting for services to start..."
sleep 10

# 检查服务状态
echo "Service status:"
docker-compose ps

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Available commands:"
echo "  docker-compose exec device-emulator bash                    # 进入容器"
echo "  docker-compose exec device-emulator morphling_emulator --help  # 查看帮助"
echo "  docker-compose logs device-emulator                         # 查看日志"
echo "  docker-compose down                                         # 停止服务"
echo ""
echo "Redis is running on: localhost:6379"
echo "DeviceEmulator is running in container: morphling-emulator"
