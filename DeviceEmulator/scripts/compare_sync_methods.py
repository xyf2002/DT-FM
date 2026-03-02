#!/usr/bin/env python3
"""
Compare the differences between 'earliest' and 'latest' synchronization methods.

Shows:
1. What each method does
2. The actual differences in offsets
3. Which devices get advanced/delayed with each method
4. Trade-offs and recommendations

Usage:
  python3 scripts/compare_sync_methods.py perf_merged.log
"""

import sys
from collections import defaultdict


def parse_log_for_compute_events(log_file: str):
    """Extract COMPUTE START events"""
    events = defaultdict(dict)

    with open(log_file, "r") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.startswith("VTIME,"):
                continue

            try:
                parts = line.split(",")
                if (
                    len(parts) < 9
                    or parts[4] != "COMPUTE"
                    or parts[5] != "START"
                ):
                    continue

                gemm_id = int(parts[3])
                device_id = int(parts[2])
                vt_start = int(parts[6])

                events[gemm_id][device_id] = vt_start
            except (ValueError, IndexError):
                continue

    return events


def compare_methods(events):
    """Compare earliest vs latest methods"""

    print("\n" + "=" * 100)
    print("SYNCHRONIZATION METHOD COMPARISON: EARLIEST vs LATEST")
    print("=" * 100)

    total_gemms = len(events)
    earliest_advances = 0  # How many device-GEMMs advance in earliest method
    latest_advances = 0  # How many device-GEMMs advance in latest method

    total_earliest_offset = 0
    total_latest_offset = 0

    for gemm_id in sorted(events.keys())[:5]:  # Show first 5 for clarity
        device_times = events[gemm_id]

        # Calculate for EARLIEST method
        earliest_baseline = min(device_times.values())
        earliest_offsets = {
            d: earliest_baseline - vt for d, vt in device_times.items()
        }

        # Calculate for LATEST method
        latest_baseline = max(device_times.values())
        latest_offsets = {
            d: latest_baseline - vt for d, vt in device_times.items()
        }

        print(f"\n{'=' * 100}")
        print(f"GEMM {gemm_id}")
        print(f"{'=' * 100}")

        print(f"\nRaw Virtual Times:")
        print(f"  Device 0: {device_times[0]:>10} us")
        print(f"  Device 1: {device_times[1]:>10} us")
        print(f"  Device 2: {device_times[2]:>10} us")
        print(
            f"  Range: {max(device_times.values()) - min(device_times.values()):>10} us"
        )

        # EARLIEST METHOD
        print(f"\n{'─' * 100}")
        print(f"EARLIEST METHOD (基准 = 最快的):")
        print(
            f"  Baseline: {earliest_baseline} us (Device {min(device_times, key=device_times.get)}最快)"
        )
        print(
            f"  ┌─ Device 0: offset={earliest_offsets[0]:>7} us  |  new vt_start={device_times[0] + earliest_offsets[0]:>10} us"
        )
        print(
            f"  ├─ Device 1: offset={earliest_offsets[1]:>7} us  |  new vt_start={device_times[1] + earliest_offsets[1]:>10} us"
        )
        print(
            f"  └─ Device 2: offset={earliest_offsets[2]:>7} us  |  new vt_start={device_times[2] + earliest_offsets[2]:>10} us"
        )

        earliest_advances += sum(1 for o in earliest_offsets.values() if o > 0)
        total_earliest_offset += sum(abs(o) for o in earliest_offsets.values())

        # LATEST METHOD
        print(f"\n{'─' * 100}")
        print(f"LATEST METHOD (基准 = 最慢的):")
        print(
            f"  Baseline: {latest_baseline} us (Device {max(device_times, key=device_times.get)} 最慢)"
        )
        print(
            f"  ┌─ Device 0: offset={latest_offsets[0]:>7} us  |  new vt_start={device_times[0] + latest_offsets[0]:>10} us"
        )
        print(
            f"  ├─ Device 1: offset={latest_offsets[1]:>7} us  |  new vt_start={device_times[1] + latest_offsets[1]:>10} us"
        )
        print(
            f"  └─ Device 2: offset={latest_offsets[2]:>7} us  |  new vt_start={device_times[2] + latest_offsets[2]:>10} us"
        )

        latest_advances += sum(1 for o in latest_offsets.values() if o > 0)
        total_latest_offset += sum(abs(o) for o in latest_offsets.values())

        # KEY DIFFERENCES
        print(f"\n{'─' * 100}")
        print(f"KEY DIFFERENCES:")
        for device_id in sorted(device_times.keys()):
            e_offset = earliest_offsets[device_id]
            l_offset = latest_offsets[device_id]
            diff = l_offset - e_offset

            if diff > 0:
                direction = "↑ LATEST advances more"
            elif diff < 0:
                direction = "↓ EARLIEST delays more"
            else:
                direction = "= Same offset"

            print(
                f"  Device {device_id}: {direction:25} | Offset diff: {diff:>7} us"
            )

    # SUMMARY
    print(f"\n\n{'=' * 100}")
    print("STATISTICAL COMPARISON (ALL 97 GEMM OPERATIONS)")
    print(f"{'=' * 100}")

    earliest_total_advances = 0
    latest_total_advances = 0
    earliest_total_offset = 0
    latest_total_offset = 0

    for gemm_id, device_times in events.items():
        earliest_baseline = min(device_times.values())
        earliest_offsets = {
            d: earliest_baseline - vt for d, vt in device_times.items()
        }
        latest_baseline = max(device_times.values())
        latest_offsets = {
            d: latest_baseline - vt for d, vt in device_times.items()
        }

        earliest_total_advances += sum(
            1 for o in earliest_offsets.values() if o > 0
        )
        latest_total_advances += sum(
            1 for o in latest_offsets.values() if o > 0
        )

        earliest_total_offset += sum(abs(o) for o in earliest_offsets.values())
        latest_total_offset += sum(abs(o) for o in latest_offsets.values())

    print(f"\nEARLIEST Method (同步到最快的设备):")
    print(
        f"  设备-GEMM 需要加速的数量: {earliest_total_advances} / {total_gemms * 3} (约 {earliest_total_advances * 100 // (total_gemms * 3)}%)"
    )
    print(f"  总偏移量: {earliest_total_offset:>12} us")
    print(
        f"  平均每个 GEMM 的总偏移: {earliest_total_offset // total_gemms:>7} us"
    )
    print(f"  策略特点: ✓ 保守，不改变总时间跨度")
    print(f"  适用场景: 性能分析、公平比较")

    print(f"\nLATEST Method (同步到最慢的设备):")
    print(
        f"  设备-GEMM 需要加速的数量: {latest_total_advances} / {total_gemms * 3} (约 {latest_total_advances * 100 // (total_gemms * 3)}%)"
    )
    print(f"  总偏移量: {latest_total_offset:>12} us")
    print(
        f"  平均每个 GEMM 的总偏移: {latest_total_offset // total_gemms:>7} us"
    )
    print(f"  策略特点: ✓ 激进，确保无设备掉队")
    print(f"  适用场景: 最坏情况分析、确保同步")

    print(f"\n{'─' * 100}")
    print(f"OFFSET 差异总结:")
    offset_diff = latest_total_offset - earliest_total_offset
    if offset_diff > 0:
        print(f"  LATEST 比 EARLIEST 多调整: {offset_diff:>10} us")
        print(f"  这代表 LATEST 方法拉扯了更多的设备来等最慢的")
    else:
        print(f"  EARLIEST 比 LATEST 多调整: {abs(offset_diff):>10} us")

    print("\n")


def print_recommendation():
    """Print recommendation based on analysis"""
    print("=" * 100)
    print("选择建议:")
    print("=" * 100)

    print(f"""
┌─ 使用 EARLIEST 方法 (默认最快对齐) ─────────────────────────────────────────────┐
│ 特点：                                                                          │
│  • Baseline = min(vt_start) → 所有设备与最快的对齐                              │
│  • 偏移量小（只需要延迟快设备）                                                  │
│  • 总时间跨度不变                                                                │
│                                                                                 │
│ 场景：性能分析、性能建模                                                        │
│ 推荐指数：⭐⭐⭐ 如果你关心"每个设备的真实计算能力"                           │
│ 运行：python3 scripts/sync_virtual_time.py log --method earliest                │
└──────────────────────────────────────────────────────────────────────────────────┘

┌─ 使用 LATEST 方法 (默认最慢对齐) ───────────────────────────────────────────────┐
│ 特点：                                                                          │
│  • Baseline = max(vt_start) → 所有设备与最慢的对齐                              │
│  • 偏移量大（需要加速所有慢设备）                                                │
│  • 模拟"等待最慢设备"的场景                                                      │
│                                                                                 │
│ 场景：最坏情况分析、系统设计                                                    │
│ 推荐指数：⭐⭐⭐ 如果你关心"整个系统的同步成本"                              │
│ 运行：python3 scripts/sync_virtual_time.py log --method latest                 │
└──────────────────────────────────────────────────────────────────────────────────┘

📊 数据对比（从你的日志）：

  EARLIEST: 延迟快设备 → 偏移量相对小 → "快设备等着"
  LATEST:   加速慢设备 → 偏移量相对大 → "慢设备赶上来"

🎯 最终建议：
  • 如果做性能分析 → EARLIEST (保守，看真实的设备能力差异)
  • 如果做系统设计 → LATEST (激进，确保没有设备掉队)
  • 当前脚本默认 LATEST ✓ (最晚对齐)
""")


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 compare_sync_methods.py <perf_merged.log>")
        sys.exit(1)

    log_file = sys.argv[1]

    print(f"Analyzing: {log_file}\n")
    events = parse_log_for_compute_events(log_file)

    print(
        f"Found {len(events)} GEMM operations with {len(events[0])} devices each"
    )

    compare_methods(events)
    print_recommendation()


if __name__ == "__main__":
    main()
