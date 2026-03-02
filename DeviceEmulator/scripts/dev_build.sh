#!/bin/bash
# 开发模式下的快速构建脚本
#
# Usage: ./scripts/dev_build.sh
#
# Examples:
#   ./scripts/dev_build.sh
#   PYTHON_EXECUTABLE=/usr/bin/python3.11 ./scripts/dev_build.sh

set -e

echo "=== Morphling Development Build Script ==="

# 检查是否在容器内
if [ ! -f /.dockerenv ]; then
    echo "Warning: This script is designed to run inside Docker container"
    echo "Run: docker-compose exec device-emulator bash scripts/dev_build.sh"
    exit 1
fi

# 设置环境变量
PYTHON_ROOT_DIR=${PYTHON_ROOT_DIR:-/usr}
PYTHON_EXECUTABLE=${PYTHON_EXECUTABLE:-/usr/bin/python3.10}
PYTHON_INCLUDE_DIR=${PYTHON_INCLUDE_DIR:-/usr/include/python3.10}
PYTHON_LIBRARY=${PYTHON_LIBRARY:-/usr/lib/x86_64-linux-gnu/libpython3.10.so}

export Python3_ROOT_DIR="${PYTHON_ROOT_DIR}"
export Python3_EXECUTABLE="${PYTHON_EXECUTABLE}"
export MORPHLING_PYTHON_EXECUTABLE="${PYTHON_EXECUTABLE}"
export LDFLAGS="-L/usr/lib/x86_64-linux-gnu"
export CPPFLAGS="-I${PYTHON_INCLUDE_DIR}"
export CMAKE_ARGS="-DPython3_EXECUTABLE=${PYTHON_EXECUTABLE} -DPython3_LIBRARY=${PYTHON_LIBRARY} -DPython3_INCLUDE_DIR=${PYTHON_INCLUDE_DIR} -DCMAKE_PREFIX_PATH=${PYTHON_ROOT_DIR}"

# 进入项目目录
PROJECT_ROOT=${PROJECT_ROOT:-/app}
cd "${PROJECT_ROOT}"

echo "=== Building morphling in development mode ==="

# 使用 pip install -e . 进行开发模式安装（可编辑安装）
python3 -m pip install --no-build-isolation --no-cache-dir --verbose -e .

echo "=== Build completed ==="
echo "You can now run your Python scripts to test the changes"
echo "Example: python3 scripts/run_devices.py --num_devices 4 --model_name facebook/opt-125m --backend proxy --seq_length 128 --batch_size 1 --cfg config/proxy/svr.ini"
