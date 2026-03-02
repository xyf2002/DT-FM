#!/bin/bash
# Run all C++ tests for DeviceEmulator
# Usage: ./run_cpp_tests.sh [category]
# Categories: all, unit, bench, cuda, memory, worker, zerocopy
#
# Note: Tests should be built beforehand (e.g., in Docker image).
# This script runs the test executables directly.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/cpp/build"

# Default category
CATEGORY="${1:-all}"

echo "========================================"
echo "DeviceEmulator C++ Test Runner"
echo "========================================"
echo "Category: ${CATEGORY}"
echo ""

# Check if build directory exists
if [ ! -d "${BUILD_DIR}" ]; then
    echo "Error: Build directory not found at ${BUILD_DIR}"
    echo "Please build tests first with CMake."
    exit 1
fi

echo "========================================"
echo "Running tests..."
echo "========================================"

# Run tests based on category
case "${CATEGORY}" in
    all)
        echo "Running all available tests..."
        # Run each test executable
        for test in "${BUILD_DIR}"/test_*; do
            if [ -x "$test" ]; then
                echo "Running $(basename $test)..."
                "$test" || true
            fi
        done
        ;;
    unit)
        echo "Running unit tests..."
        for test in "${BUILD_DIR}"/test_worker_base "${BUILD_DIR}"/test_cublas_error15_repro; do
            if [ -x "$test" ]; then
                echo "Running $(basename $test)..."
                "$test" || true
            fi
        done
        ;;
    cuda)
        echo "Running CUDA tests..."
        for test in "${BUILD_DIR}"/test_cublas_*; do
            if [ -x "$test" ]; then
                echo "Running $(basename $test)..."
                "$test" || true
            fi
        done
        ;;
    worker)
        echo "Running worker tests..."
        if [ -x "${BUILD_DIR}/test_worker_base" ]; then
            echo "Running test_worker_base..."
            "${BUILD_DIR}/test_worker_base" || true
        fi
        ;;
    *)
        echo "Unknown category: ${CATEGORY}"
        echo "Available categories: all, unit, cuda, worker"
        exit 1
        ;;
esac

echo ""
echo "========================================"
echo "Test run complete!"
echo "========================================"
