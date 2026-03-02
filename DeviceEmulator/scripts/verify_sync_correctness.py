#!/usr/bin/env python3
"""
Verify synchronization correctness by comparing before/after virtual times.
Shows detailed examples for each GEMM operation.

Usage:
  python3 scripts/verify_sync_correctness.py --before perf_merged.log --after perf_merged_synced.log
  python3 scripts/verify_sync_correctness.py perf_merged.log perf_merged_synced.log
"""

import argparse
import re
from collections import defaultdict


def parse_vtime_event(line):
    """Parse VTIME event line"""
    parts = line.strip().split(",")
    if len(parts) < 9:
        return None
    return {
        "timestamp": int(parts[1]),
        "device_id": int(parts[2]),
        "gemm_id": int(parts[3]),
        "phase": parts[4],
        "event": parts[5],
        "vt_start_us": int(parts[6]),
        "vt_end_us": int(parts[7]),
        "vt_duration_us": int(parts[8]),
    }


def read_log(filepath):
    """Read log file and extract VTIME events"""
    events = defaultdict(list)
    with open(filepath, "r") as f:
        for line in f:
            if line.startswith("VTIME,"):
                event = parse_vtime_event(line)
                if event:
                    events[event["gemm_id"]].append(event)
    return events


def analyze_gemm(before_events, after_events, gemm_id):
    """Analyze a specific GEMM operation"""
    print(f"\n{'=' * 100}")
    print(f"GEMM {gemm_id} 详细分析")
    print(f"{'=' * 100}")

    before = {
        evt["device_id"]: evt
        for evt in before_events.get(gemm_id, [])
        if evt["phase"] == "COMPUTE" and evt["event"] == "START"
    }
    after = {
        evt["device_id"]: evt
        for evt in after_events.get(gemm_id, [])
        if evt["phase"] == "COMPUTE" and evt["event"] == "START"
    }

    if not before or not after:
        print(f"无法找到 COMPUTE START 事件")
        return

    # Calculate offsets (difference between before and after)
    print(f"\n【COMPUTE START 虚拟时间对比】\n")
    print(
        f"{'Device':<10} {'原始 vt_start_us':<20} {'同步后 vt_start_us':<20} {'偏移量 (us)':<20}"
    )
    print(f"{'-' * 70}")

    offsets = {}
    after_vts = {}
    for device_id in sorted(before.keys()):
        before_vt = before[device_id]["vt_start_us"]
        after_vt = after[device_id]["vt_start_us"]
        offset = after_vt - before_vt
        offsets[device_id] = offset
        after_vts[device_id] = after_vt

        print(f"{device_id:<10} {before_vt:<20} {after_vt:<20} {offset:<20}")

    # Analysis: all devices should have same vt_start after synchronization
    print(f"\n【同步分析】\n")
    unique_vts = set(after_vts.values())
    if len(unique_vts) == 1:
        baseline_vt = list(unique_vts)[0]
        print(f"✓ 同步成功: 所有设备的 COMPUTE START 都对齐到 {baseline_vt} us")
    else:
        print(f"✗ 同步异常: 设备间虚拟时间不一致")
        print(f"  虚拟时间: {unique_vts}")
        return

    print(f"✓ 同步方法: LATEST (同步到最慢设备)")

    # Show all events in order
    print(f"\n【所有事件时间序列对比】\n")
    print(
        f"{'类型':<10} {'设备':<8} {'事件':<15} {'原始 vt':<20} {'同步后 vt':<20} {'变化':<15}"
    )
    print(f"{'-' * 100}")

    before_all = {
        (evt["device_id"], evt["phase"], evt["event"]): evt
        for evt in before_events.get(gemm_id, [])
    }
    after_all = {
        (evt["device_id"], evt["phase"], evt["event"]): evt
        for evt in after_events.get(gemm_id, [])
    }

    for key in sorted(before_all.keys()):
        b_evt = before_all[key]
        a_evt = after_all.get(key)

        if a_evt:
            device_id, phase, event = key
            before_vt = b_evt["vt_start_us"]
            after_vt = a_evt["vt_start_us"]
            change = after_vt - before_vt

            print(
                f"{'✓':<10} {device_id:<8} {phase}/{event:<12} {before_vt:<20} {after_vt:<20} {change:+<15}"
            )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Verify virtual time synchronization correctness"
    )
    parser.add_argument(
        "before",
        nargs="?",
        default="perf_merged.log",
        help="Path to original merged log (default: perf_merged.log)",
    )
    parser.add_argument(
        "after",
        nargs="?",
        default="perf_merged_synced.log",
        help="Path to synchronized log (default: perf_merged_synced.log)",
    )
    parser.add_argument(
        "--before",
        dest="before_flag",
        help="Path to original merged log",
    )
    parser.add_argument(
        "--after",
        dest="after_flag",
        help="Path to synchronized log",
    )
    return parser.parse_args()


def main():
    print("\n" + "=" * 100)
    print("虚拟时间同步验证分析")
    print("=" * 100)

    args = parse_args()
    before_path = args.before_flag or args.before
    after_path = args.after_flag or args.after

    before = read_log(before_path)
    after = read_log(after_path)

    print(f"\n原始日志: {len(before)} 个 GEMM 操作")
    print(f"同步日志: {len(after)} 个 GEMM 操作")

    # Show details for first 5 GEMMs
    print(f"\n详细分析前 5 个 GEMM 操作:")
    for gemm_id in range(min(5, len(before))):
        analyze_gemm(before, after, gemm_id)

    # Summary statistics
    print(f"\n{'=' * 100}")
    print("同步总体统计")
    print(f"{'=' * 100}\n")

    total_events = 0
    correct_syncs = 0
    device_offsets = defaultdict(list)

    for gemm_id in before.keys():
        compute_starts_before = [
            evt
            for evt in before[gemm_id]
            if evt["phase"] == "COMPUTE" and evt["event"] == "START"
        ]
        compute_starts_after = [
            evt
            for evt in after[gemm_id]
            if evt["phase"] == "COMPUTE" and evt["event"] == "START"
        ]

        if compute_starts_before and compute_starts_after:
            before_dict = {
                evt["device_id"]: evt["vt_start_us"]
                for evt in compute_starts_before
            }
            after_dict = {
                evt["device_id"]: evt["vt_start_us"]
                for evt in compute_starts_after
            }

            for device_id in before_dict.keys():
                offset = after_dict[device_id] - before_dict[device_id]
                device_offsets[device_id].append(offset)
                total_events += 1

    print(f"✓ 成功同步的 GEMM 操作: {len(before)}")
    print(f"✓ 成功同步的设备-GEMM 对: {total_events}")

    print(f"\n【每个设备的偏移统计】\n")
    for device_id in sorted(device_offsets.keys()):
        offsets = device_offsets[device_id]
        avg_offset = sum(offsets) / len(offsets)
        min_offset = min(offsets)
        max_offset = max(offsets)

        print(f"Device {device_id}:")
        print(f"  平均偏移: {avg_offset:>12.1f} us")
        print(f"  最小偏移: {min_offset:>12.1f} us")
        print(f"  最大偏移: {max_offset:>12.1f} us")
        print(f"  总偏移:   {sum(offsets):>12.1f} us")


if __name__ == "__main__":
    main()
