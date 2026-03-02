#!/bin/bash
unset LD_PRELOAD
export CUDA_VISIBLE_DEVICES=5
# Navigate to the project root directory (one level up from csrc)
cd "$(dirname "$0")/.."
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# Set environment variables for MKL and CUDA
export MKLROOT=/opt/intel/oneapi/mkl/latest
export CUDA_HOME=/usr/local/cuda-12.2
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Additional library paths
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$(pwd)/csrc/memory:$LD_LIBRARY_PATH

# Set the log directory and clean up old log files
LOG_DIR="$(pwd)/csrc/intercept/logs"
echo "Cleaning up old log files..."

if [ ! -d "$LOG_DIR" ]; then
    mkdir -p "$LOG_DIR"
fi

rm -f "$LOG_DIR"/process_*.txt
rm -f "$LOG_DIR"/gpu_process.log
rm -f "$LOG_DIR"/shared_memory.log

# Set shared memory size and name
total_ram=$(grep MemTotal /proc/meminfo | awk '{print $2}')
shm_size=$((total_ram * 1024 / 2))
shm_name="/sgemm_shm"

# Clean up any existing shared memory
if [ -e "/dev/shm$shm_name" ]; then
    rm "/dev/shm$shm_name"
fi

# Create shared memory segment and set permissions
dd if=/dev/zero of="/dev/shm$shm_name" bs=$shm_size count=1
chmod 666 "/dev/shm$shm_name"

# Compile the shared memory library
gcc -g -fPIC -shared -o "csrc/memory/shared_memory.so" \
    "csrc/memory/shared_memory_manager.c" \
    -I"${MKLROOT}/include" -I"${CUDA_HOME}/include" \
    -I"csrc/memory" \
    -L"${MKLROOT}/lib/intel64" -L"${CUDA_HOME}/lib64" \
    -lmkl_rt -lcublas -lcudart -ldl -lpthread
if [ $? -ne 0 ]; then
    echo "Failed to compile shared_memory.so. Exiting."
    exit 1
fi
sleep 1

# Compile the shared memory initializer
gcc -o "csrc/memory/shared_memory_initializer" \
    "csrc/memory/shared_memory_initializer.c" \
    "csrc/memory/shared_memory_manager.c" \
    -I"csrc/memory" -lpthread
if [ $? -ne 0 ]; then
    echo "Failed to compile shared_memory_initializer. Exiting."
    exit 1
fi
sleep 1

# Initialize shared memory
echo "Initializing shared memory..."
"./csrc/memory/shared_memory_initializer" $shm_size
if [ $? -ne 0 ]; then
    echo "Failed to initialize shared memory. Exiting."
    exit 1
fi
sleep 2

# Compile the interceptor library
gcc -g -fPIC -shared -o "csrc/intercept/interceptor.so" \
    "csrc/intercept/interceptor.c" \
    "csrc/memory/shared_memory.so" \
    -I"csrc/memory" \
    -I"${MKLROOT}/include" -I"${CUDA_HOME}/include" \
    -L"${MKLROOT}/lib/intel64" -L"${CUDA_HOME}/lib64" \
    -Wl,-rpath="$(pwd)/csrc/memory" \
    -lmkl_rt -lcublas -lcudart -ldl -lpthread -lspdlog
if [ $? -ne 0 ]; then
    echo "Failed to compile interceptor.so. Exiting."
    exit 1
fi
sleep 1

# Compile the worker process
gcc -g -o "csrc/intercept/gpu_process" "csrc/worker/worker.c" \
    -I"csrc/memory" \
    -I"${MKLROOT}/include" -I"${CUDA_HOME}/include" \
    "csrc/memory/shared_memory.so" \
    -L"${MKLROOT}/lib/intel64" -L"${CUDA_HOME}/lib64" \
    -Wl,-rpath="$(pwd)/csrc/memory" \
    -lmkl_rt -lcublas -lcudart -ldl -lpthread
if [ $? -ne 0 ]; then
    echo "Failed to compile gpu_process. Exiting."
    exit 1
fi
sleep 5

# Start GPU process
echo "Starting GPU process..."
"./csrc/intercept/gpu_process" &
GPU_PID=$!

# Ensure the shared memory initializer has finished before running interceptors
sleep 2

# Start Python script(s) with the interceptor
for i in $(seq 1 50); do
    echo "Starting Python script $i with interceptor..."
    LD_PRELOAD="$(pwd)/csrc/intercept/interceptor.so" \
    python3 "/home/eren/DeviceEmulator/tests/cpp/interception_tests/sgemm_dimensions.py" &
done
sleep 1

# Wait for all processes to complete
wait

# Clean up any remaining processes
ps aux | grep '[p]ython3.*sgemm_dimensions.py\|[g]pu_process' | awk '{print $2}' | xargs -r kill -9

echo "All processes have completed. Logs are available in the '$LOG_DIR' directory."


g++ -g -fPIC -shared -o -std=c++17 "csrc/intercept/interceptor.so" \
    "csrc/intercept/interceptor.cpp" \
    "csrc/utils/logger.cpp" \
    "csrc/memory/caching_allocator.cpp" \
    "build/temp.linux-x86_64-cpython-39/morphling.pb.cc" \
    -D_GLIBCXX_USE_CXX11_ABI=0 \
    -I"csrc" \
    -I"${CUDA_HOME}/include" \
    -I"build/temp.linux-x86_64-cpython-39/_deps/grpc-src/third_party/protobuf/src" \
    -I"build/temp.linux-x86_64-cpython-39/" \
    -I"build/temp.linux-x86_64-cpython-39/_deps/grpc-src/include" \
    -I"build/temp.linux-x86_64-cpython-39/_deps/grpc-src/third_party/abseil-cpp/" \
    -L"${CUDA_HOME}/lib64" \
    -L"/mnt/data/xly/.conda/envs/emulator/lib" \
    -L"build/temp.linux-x86_64-cpython-39/_deps/grpc-build/build/temp.linux-x86_64-cpython-39" \
    -L"build/temp.linux-x86_64-cpython-39/_deps/grpc-build/third_party/protobuf/build/temp.linux-x86_64-cpython-39" \
    -lmkl_rt -lgrpc++ -lgrpc++_reflection -lcublas -lcudart -lspdlog
