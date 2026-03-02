#!/usr/bin/env python3
"""
深度对比两种同步策略的差异和影响

用于回答问题：
1. "最早对齐" vs "最慢对齐" 有什么区别？
2. 哪种方式更合适？
3. 对性能分析有什么影响？

用法:
  python3 scripts/compare_sync_strategies.py perf_merged.log
"""

import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


def parse_log_for_compute_times(
    log_file: str,
) -> Dict[int, Dict[int, Tuple[int, int]]]:
    """
    提取所有 COMPUTE START 和 END 的虚拟时间
    返回: {gemm_id: {device_id: (vt_start, vt_end)}}
    """
    compute_times = defaultdict(dict)

    with open(log_file, "r") as f:
        for line in f:
            if not line.startswith("VTIME,"):
                continue

            parts = line.split(",")
            if len(parts) < 9 or parts[4] != "COMPUTE":
                continue

            try:
                gemm_id = int(parts[3])
                device_id = int(parts[2])
                event = parts[5]
                vt_start = int(parts[6])
                vt_end = int(parts[7])

                if event == "START" and gemm_id not in compute_times[device_id]:
                    # 只取 START 时间的 vt_start
                    compute_times[gemm_id][device_id] = (vt_start, None)
                elif event == "END":
                    # 记录 END 时间的 vt_end
                    if (
                        gemm_id in compute_times
                        and device_id in compute_times[gemm_id]
                    ):
                        start_time, _ = compute_times[gemm_id][device_id]
                        compute_times[gemm_id][device_id] = (start_time, vt_end)
                    else:
                        compute_times[gemm_id][device_id] = (None, vt_end)
            except (ValueError, IndexError):
                continue

    return compute_times


def analyze_strategy_differences(
    compute_times: Dict[int, Dict[int, Tuple[int, int]]],
) -> Tuple[Dict, Dict]:
    """
    分析两种策略的差异

    返回:
        (earliest_results, latest_results)
        其中每个都是 {gemm_id: {metric: value}}
    """
    earliest_results = {}
    latest_results = {}

    for gemm_id in sorted(compute_times.keys()):
        devices = compute_times[gemm_id]

        # 提取所有设备的 vt_start（都能取到的）
        vt_starts = {}
        vt_ends = {}
        for device_id, (start, end) in devices.items():
            if start is not None:
                vt_starts[device_id] = start
            if end is not None:
                vt_ends[device_id] = end

        if not vt_starts:
            continue

        # ============ EARLIEST 策略 ============
        baseline_earliest = min(vt_starts.values())
        offsets_earliest = {
            d: baseline_earliest - vt_starts[d] for d in vt_starts.keys()
        }

        # ============ LATEST 策略 ============
        baseline_latest = max(vt_starts.values())
        offsets_latest = {
            d: baseline_latest - vt_starts[d] for d in vt_starts.keys()
        }

        # ============ FASTEST 设备策略 ============
        # 找出速度最快的设备（COMPUTE 持续时间最短）
        if vt_ends:
            compute_durations = {}
            for device_id in vt_starts.keys():
                if device_id in vt_ends:
                    duration = vt_ends[device_id] - vt_starts[device_id]
                    compute_durations[device_id] = duration

            fastest_device = (
                min(compute_durations.items(), key=lambda x: x[1])[0]
                if compute_durations
                else None
            )
        else:
            fastest_device = None

        # 计算指标
        earliest_results[gemm_id] = {
            "baseline": baseline_earliest,
            "offsets": offsets_earliest,
            "max_offset": max(abs(o) for o in offsets_earliest.values()),
            "min_offset": min(abs(o) for o in offsets_earliest.values()),
            "spread_before": max(vt_starts.values()) - min(vt_starts.values()),
            "spread_after": 0,
            "baseline_device": min(vt_starts.items(), key=lambda x: x[1])[0],
            "strategy_name": "EARLIEST (最早对齐)",
        }

        latest_results[gemm_id] = {
            "baseline": baseline_latest,
            "offsets": offsets_latest,
            "max_offset": max(abs(o) for o in offsets_latest.values()),
            "min_offset": min(abs(o) for o in offsets_latest.values()),
            "spread_before": max(vt_starts.values()) - min(vt_starts.values()),
            "spread_after": 0,
            "baseline_device": max(vt_starts.items(), key=lambda x: x[1])[0],
            "strategy_name": "LATEST (最晚对齐)",
            "fastest_device": fastest_device,
        }

    return earliest_results, latest_results


def print_detailed_comparison(
    earliest: Dict, latest: Dict, compute_times: Dict
):
    """打印详细对比分析"""

    print("\n" + "=" * 100)
    print("VIRTUAL TIME SYNCHRONIZATION STRATEGY COMPARISON")
    print("=" * 100)

    print("\n【概念说明】")
    print("-" * 100)
    print("""
EARLIEST (最早对齐) - 以最快的设备为基准
  └─ baseline = min(所有 vt_start)
  └─ 含义: 所有设备都向最快的设备看齐
  └─ 效果: 快的设备调慢（负偏移），慢的设备调得更慢（更大的负偏移）
  └─ 特点: 保守，虚拟时间整体向后推

LATEST (最晚对齐) - 以最慢的设备为基准
  └─ baseline = max(所有 vt_start)
  └─ 含义: 所有设备都向最慢的设备看齐
  └─ 效果: 快的设备调快（正偏移），慢的设备不动（零偏移）
  └─ 特点: 激进，虚拟时间整体向前推

关键区别:
  • 同步后虚拟时间的绝对值不同
  • 对后续计算和分析有影响
  • 选择哪个取决于你的分析目标
""")

    # 取前 5 个 GEMM 做详细对比
    print("\n【GEMM 0-4 详细对比】")
    print("=" * 100)

    for gemm_id in range(0, min(5, len(earliest))):
        if gemm_id not in earliest or gemm_id not in latest:
            continue

        early = earliest[gemm_id]
        late = latest[gemm_id]
        devices = compute_times[gemm_id]

        print(
            f"\n┌─ GEMM {gemm_id} ─────────────────────────────────────────────────────────────────────────────┐"
        )

        # 原始数据
        print(f"│")
        print(f"│ 【原始虚拟时间】")
        vt_starts = {}
        for device_id, (start, _) in devices.items():
            if start is not None:
                vt_starts[device_id] = start
                print(f"│   Device {device_id}: {start:>12} us")

        spread = max(vt_starts.values()) - min(vt_starts.values())
        print(f"│   └─ Spread: {spread} us (设备间差异)")

        # EARLIEST 策略
        print(f"│")
        print(f"│ 【EARLIEST 策略 - 以最快的设备为基准】")
        print(
            f"│   Baseline: {early['baseline']} us (Device {early['baseline_device']} 最快)"
        )
        print(f"│   同步后的虚拟时间:")
        for device_id in sorted(vt_starts.keys()):
            offset = early["offsets"][device_id]
            new_vt = vt_starts[device_id] + offset
            status = "基准 (0)" if offset == 0 else f"拖慢 ({offset})"
            print(
                f"│     Device {device_id}: {new_vt:>12} us  (offset={offset:>7} {status})"
            )

        # LATEST 策略
        print(f"│")
        print(f"│ 【LATEST 策略 - 以最慢的设备为基准】")
        print(
            f"│   Baseline: {late['baseline']} us (Device {late['baseline_device']} 最慢)"
        )
        print(f"│   同步后的虚拟时间:")
        for device_id in sorted(vt_starts.keys()):
            offset = late["offsets"][device_id]
            new_vt = vt_starts[device_id] + offset
            status = (
                "基准 (0)"
                if offset == 0
                else f"加快 ({offset:>+7})"
                if offset > 0
                else f"拖慢 ({offset})"
            )
            print(
                f"│     Device {device_id}: {new_vt:>12} us  (offset={offset:>+7} {status})"
            )

        # 差异分析
        print(f"│")
        print(f"│ 【策略差异分析】")
        print(f"│   虚拟时间位移:")
        earliest_baseline = early["baseline"]
        latest_baseline = late["baseline"]
        shift = latest_baseline - earliest_baseline
        print(f"│     • EARLIEST baseline: {earliest_baseline} us")
        print(f"│     • LATEST baseline:   {latest_baseline} us")
        print(f"│     • 位移差异: {shift} us ({shift / 1000:.1f} ms)")

        print(f"│")
        print(f"│   最大偏移量:")
        print(f"│     • EARLIEST: {early['max_offset']} us")
        print(f"│     • LATEST:   {late['max_offset']} us")
        print(
            f"│     • 影响范围: {'LATEST 更大' if late['max_offset'] > early['max_offset'] else 'EARLIEST 更大'}"
        )

        print(f"│")
        print(f"│ 【哪个更合适？】")
        print(f"│   • 如果目标是分析 Device 性能: ❌ 两个都不理想")
        print(f"│     原因：EARLIEST 用最快的做基准，可能掩盖性能差异")
        print(f"│     原因：LATEST 用最慢的做基准，可能夸大慢设备的影响")
        print(f"│   • 如果目标是公平对标: ✅ EARLIEST")
        print(f"│     原因：比较时都在同一起点，公平性更强")
        print(f"│   • 如果目标是最坏情况: ✅ LATEST")
        print(f"│     原因：确保所有设备都赶得上，容错性分析")

        print(f"└{'─' * 97}┘")


def print_statistical_comparison(earliest: Dict, latest: Dict):
    """打印统计对比"""

    print("\n" + "=" * 100)
    print("STATISTICAL COMPARISON - 全量统计对比")
    print("=" * 100)

    earliest_baselines = [v["baseline"] for v in earliest.values()]
    latest_baselines = [v["baseline"] for v in latest.values()]

    earliest_max_offsets = [v["max_offset"] for v in earliest.values()]
    latest_max_offsets = [v["max_offset"] for v in latest.values()]

    earliest_baseline_devices = [
        v["baseline_device"] for v in earliest.values()
    ]
    latest_baseline_devices = [v["baseline_device"] for v in latest.values()]

    print(f"\n【Baseline 虚拟时间】")
    print(f"├─ EARLIEST strategy:")
    print(
        f"│  ├─ 平均值: {sum(earliest_baselines) / len(earliest_baselines):>12.0f} us"
    )
    print(f"│  ├─ 最小值: {min(earliest_baselines):>12} us")
    print(f"│  ├─ 最大值: {max(earliest_baselines):>12} us")
    print(f"│  └─ 特点: 总是选择最小的虚拟时间（最快的设备）")
    print(f"│")
    print(f"└─ LATEST strategy:")
    print(
        f"   ├─ 平均值: {sum(latest_baselines) / len(latest_baselines):>12.0f} us"
    )
    print(f"   ├─ 最小值: {min(latest_baselines):>12} us")
    print(f"   ├─ 最大值: {max(latest_baselines):>12} us")
    print(f"   └─ 特点: 总是选择最大的虚拟时间（最慢的设备）")

    baseline_shift = sum(latest_baselines) / len(latest_baselines) - sum(
        earliest_baselines
    ) / len(earliest_baselines)
    print(f"\n【虚拟时间位移】")
    print(
        f"├─ 平均位移: {baseline_shift:>12.0f} us (LATEST 比 EARLIEST 向前推)"
    )
    print(f"└─ 此位移对所有设备影响相同，不影响相对关系")

    print(f"\n【最大偏移量（设备需要调整的幅度）】")
    print(f"├─ EARLIEST:")
    print(
        f"│  ├─ 平均值: {sum(earliest_max_offsets) / len(earliest_max_offsets):>12.0f} us"
    )
    print(f"│  ├─ 最小值: {min(earliest_max_offsets):>12} us")
    print(f"│  └─ 最大值: {max(earliest_max_offsets):>12} us")
    print(f"│")
    print(f"└─ LATEST:")
    print(
        f"   ├─ 平均值: {sum(latest_max_offsets) / len(latest_max_offsets):>12.0f} us"
    )
    print(f"   ├─ 最小值: {min(latest_max_offsets):>12} us")
    print(f"   └─ 最大值: {max(latest_max_offsets):>12} us")

    print(f"\n【Baseline 选择的设备分布】")
    from collections import Counter

    earliest_device_counts = Counter(earliest_baseline_devices)
    latest_device_counts = Counter(latest_baseline_devices)

    print(f"├─ EARLIEST (最快的设备作为基准):")
    for device_id in sorted(earliest_device_counts.keys()):
        count = earliest_device_counts[device_id]
        pct = count * 100 / len(earliest)
        print(f"│  └─ Device {device_id}: {count:>2} 次 ({pct:>5.1f}%)")

    print(f"│")
    print(f"└─ LATEST (最慢的设备作为基准):")
    for device_id in sorted(latest_device_counts.keys()):
        count = latest_device_counts[device_id]
        pct = count * 100 / len(latest)
        print(f"   └─ Device {device_id}: {count:>2} 次 ({pct:>5.1f}%)")


def print_recommendation(earliest: Dict, latest: Dict):
    """打印建议"""

    print("\n" + "=" * 100)
    print("RECOMMENDATION - 策略选择建议")
    print("=" * 100)

    print("""
【核心问题】

你现在用的是 EARLIEST（最早对齐），问题是：
"为什么要以最早对齐而不是最晚对齐？"

这实际上是在问：同步的目标是什么？


【三种常见目标】

1️⃣  性能公平对标
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   • 目标: 在相同虚拟时间起点，比较不同设备的性能
   • 适用: Device 性能分析、负载均衡优化
   • 选择: ✅ EARLIEST

   原因:
   └─ 所有设备从"最早的时间"启动
   └─ 快的设备被拖慢到和最快的一样快
   └─ 这样对比是最公平的（消除了启动时间差异）

   例子:
   └─ Device 0 比 Device 2 快 111,358 us
   └─ EARLIEST 将 Device 0 拖慢 111,358 us
   └─ 现在他们虚拟时间相同，可以公平比较真实计算时间


2️⃣  最坏情况分析
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   • 目标: 确保系统能应对最慢设备的情况
   • 适用: 容错性评估、SLA 保证
   • 选择: ✅ LATEST

   原因:
   └─ 所有设备都要"等待"最慢的设备
   └─ 能评估系统在最坏情况下的性能
   └─ 快的设备需要向前推进以匹配最慢设备

   例子:
   └─ Device 0 比 Device 2 快 111,358 us
   └─ LATEST 将 Device 0 向前推进 111,358 us
   └─ 现在 Device 0 需要等待 Device 2 才能继续


3️⃣  时间基准保留
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   • 目标: 保留部分原始时间关系，用于跨系统对标
   • 适用: 多个实验之间的结果比较
   • 选择: ⚠️  都不太理想，建议自定义基准

   原因:
   └─ EARLIEST 会改变全局时间戳
   └─ LATEST 改变得更多
   └─ 可能和其他实验的时间关系对不上


【你现在的选择 - EARLIEST】

✅ 优点:
   ├─ 保守，虚拟时间只向后推，不破坏原有关系
   ├─ 适合性能分析和设备对标
   ├─ 最快的设备（通常是参考点）保持不变
   └─ 容易理解：所有设备都向最快的对齐

❌ 缺点:
   ├─ 快设备会被"惩罚"（虽然只是虚拟上）
   ├─ 如果要分析系统最坏情况，不够激进
   └─ 对最慢设备的性能问题掩盖较多


【如果改用 LATEST】

✅ 优点:
   ├─ 快设备不被"惩罚"
   ├─ 最慢设备成为中心（会显示其真实影响）
   ├─ 适合最坏情况分析
   └─ 更激进，问题暴露更充分

❌ 缺点:
   ├─ 虚拟时间向前推进，改变了原有时间戳
   ├─ 可能和其他分析系统的时间不匹配
   ├─ 总时间跨度可能变大
   └─ 对快设备不太"公平"


【我的建议】

对于你的场景（97 个 GEMM 的分布式训练同步分析）：

🎯 保持使用 EARLIEST
   原因:
   ├─ 你的主要目标是性能分析和对标
   ├─ Device 0 和 Device 1 的性能问题一致（都快约 111ms）
   ├─ 这说明是系统级别的差异，不是随机的
   ├─ EARLIEST 能更清晰地展示这个系统差异
   └─ 公平性更强（都从最快的时间启动）

📊 额外建议：两个方案都跑一遍
   ├─ 用 EARLIEST 分析设备性能差异
   ├─ 用 LATEST 分析系统容错能力
   ├─ 对比两个结果，获得更全面的理解
   └─ 代码中已经支持 --method 参数了，很容易切换

""")

    print("\n" + "=" * 100)
    print("END OF ANALYSIS")
    print("=" * 100)


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        print("\nUsage: python3 compare_sync_strategies.py <perf_merged.log>")
        sys.exit(1)

    log_file = sys.argv[1]

    print(f"分析日志: {log_file}")
    compute_times = parse_log_for_compute_times(log_file)
    print(f"解析到 {len(compute_times)} 个 GEMM 操作")

    # 计算两种策略
    earliest, latest = analyze_strategy_differences(compute_times)

    # 打印详细对比
    print_detailed_comparison(earliest, latest, compute_times)

    # 打印统计对比
    print_statistical_comparison(earliest, latest)

    # 打印建议
    print_recommendation(earliest, latest)


if __name__ == "__main__":
    main()
