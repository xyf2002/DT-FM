#!/usr/bin/env python3
"""
真实 Protobuf vs FlatBuffers 对比测试

这个脚本可以用来测试实际的 FlatBuffers 实现
当前占位符实现，用于展示测试框架
"""

import os
import sys
import time

import numpy as np


def benchmark_true_comparison(num_iterations=10, matrix_size_mb=40):
    """进行真实的 Protobuf 和 FlatBuffers 对比"""

    matrix_size = matrix_size_mb * 1024 * 1024
    row_data = np.random.randn(matrix_size // 4).astype(np.float32)
    col_data = np.random.randn(matrix_size // 4).astype(np.float32)

    print("\n" + "=" * 70)
    print("REAL PROTOBUF vs FLATBUFFERS COMPARISON")
    print("=" * 70)

    print("\nTest Configuration:")
    print(f"  - Iterations: {num_iterations}")
    print(f"  - Row matrix: {matrix_size_mb} MB")
    print(f"  - Col matrix: {matrix_size_mb} MB")
    print(f"  - Total payload per message: {2 * matrix_size_mb} MB")

    metadata = {
        "version": 42,
        "row": 512,
        "col": 512,
        "pivot": 1024,
        "h_dim": 1024,
        "dev_id": 3,
        "oid": 5,
        "gemm_id": 100,
        "timestamp": 1234567890,
    }

    # ============================================================
    # 测试 Protobuf (使用实际的 protobuf 库)
    # ============================================================
    print("\n[1] PROTOBUF TEST")
    print("-" * 70)

    try:
        from morphling import global_api_pb2

        protobuf_serialize_times = []
        protobuf_deserialize_times = []

        # Warmup
        for _ in range(2):
            partition = global_api_pb2.ComputeGemmData()
            partition.version = metadata["version"]
            partition.row = metadata["row"]
            partition.col = metadata["col"]
            partition.pivot = metadata["pivot"]
            partition.h_dim = metadata["h_dim"]
            partition.dev_id = metadata["dev_id"]
            partition.oid = metadata["oid"]
            partition.gemm_id = metadata["gemm_id"]
            partition.timestamp = metadata["timestamp"]

            # Add matrices
            payload1 = partition.matrices.add()
            payload1.data = row_data.tobytes()

            payload2 = partition.matrices.add()
            payload2.data = col_data.tobytes()

            _ = partition.SerializeToString()

        # Actual test
        for i in range(num_iterations):
            partition = global_api_pb2.ComputeGemmData()
            partition.version = metadata["version"] + i
            partition.row = metadata["row"]
            partition.col = metadata["col"]
            partition.pivot = metadata["pivot"]
            partition.h_dim = metadata["h_dim"]
            partition.dev_id = metadata["dev_id"]
            partition.oid = metadata["oid"]
            partition.gemm_id = metadata["gemm_id"]
            partition.timestamp = metadata["timestamp"]

            payload1 = partition.matrices.add()
            payload1.data = row_data.tobytes()

            payload2 = partition.matrices.add()
            payload2.data = col_data.tobytes()

            # Serialize
            start = time.perf_counter()
            buf = partition.SerializeToString()
            end = time.perf_counter()
            protobuf_serialize_times.append((end - start) * 1e6)

            # Deserialize
            start = time.perf_counter()
            partition2 = global_api_pb2.ComputeGemmData()
            partition2.ParseFromString(buf)
            end = time.perf_counter()
            protobuf_deserialize_times.append((end - start) * 1e6)

        avg_pb_serialize = np.mean(protobuf_serialize_times)
        avg_pb_deserialize = np.mean(protobuf_deserialize_times)

        print(f"  ✓ Serialize:   {avg_pb_serialize:.2f} us")
        print(f"  ✓ Deserialize: {avg_pb_deserialize:.2f} us")
        print(
            f"  ✓ Total:       {avg_pb_serialize + avg_pb_deserialize:.2f} us"
        )

    except ImportError as e:
        print(f"  ✗ Protobuf test skipped: {e}")
        print("    (需要安装: pip install protobuf)")
        avg_pb_serialize = None
        avg_pb_deserialize = None

    # ============================================================
    # 测试 FlatBuffers (将来实现)
    # ============================================================
    print("\n[2] FLATBUFFERS TEST")
    print("-" * 70)
    print("  ⏳ 等待 FlatBuffers C++ 实现...")
    print("  TODO: 实现 Python FlatBuffers 绑定或 C++ 扩展")

    # ============================================================
    # 对比
    # ============================================================
    if avg_pb_serialize is not None:
        print("\n[3] COMPARISON")
        print("-" * 70)
        print("  提示：运行真实对比，需要:")
        print("    1. 实现 FlatBuffers 序列化到 MatrixPartition")
        print("    2. 生成 Python 绑定或 C++ 扩展")
        print("    3. 对两者使用相同的测试数据和迭代次数")


def explain_test_setup():
    """解释测试架构"""

    print("\n" + "=" * 70)
    print("测试架构说明")
    print("=" * 70)

    print("\n【当前测试状态】")
    print("""
    ✓ Protobuf 模拟测试:
      - 生成 80 MB 的测试数据
      - 测试序列化时间
      - 这是真实的系统内存写入操作

    ✗ FlatBuffers:
      - 没有真实实现，只是投影估算
      - 基于行业标准的 5-20x 改善倍数
    """)

    print("\n【如何做真实对比】")
    print("""
    第1步: 构建 FlatBuffers
    ├─ flatc -c++ -o csrc/backend proto/matrix_partition.fbs
    └─ 生成 matrix_partition_generated.h

    第2步: 实现序列化 (C++)
    ├─ MatrixPartition::SerializeFlatBuffers()
    └─ MatrixPartition::DeserializeFlatBuffers()

    第3步: Python 绑定 (可选)
    ├─ 创建 Python 扩展
    └─ 或者直接在 C++ 测试中运行

    第4步: 对比测试
    ├─ 同样的数据
    ├─ 同样的迭代次数
    └─ 测试序列化 + 反序列化时间
    """)

    print("\n【消息大小预期】")
    print("""
    对于 80 MB 的消息:

    Protobuf:
    ├─ 矩阵数据:    80.000000 MB (float32 x 20M)
    ├─ 元数据:      ~200 bytes (varint 编码)
    └─ 总计:        80.0002 MB

    FlatBuffers:
    ├─ 矩阵数据:    80.000000 MB (相同)
    ├─ 头部+偏移:   ~40 bytes
    └─ 总计:        80.00004 MB

    → 消息大小差异: < 0.0001% (可忽略)
    → 性能差异:     主要来自编码/解析开销
    """)


if __name__ == "__main__":
    print("\n" + "╔" + "=" * 68 + "╗")
    print(
        "║"
        + " " * 15
        + "真实 Protobuf vs FlatBuffers 对比测试框架"
        + " " * 11
        + "║"
    )
    print("╚" + "=" * 68 + "╝")

    benchmark_true_comparison(num_iterations=5, matrix_size_mb=40)
    explain_test_setup()

    print("\n")
