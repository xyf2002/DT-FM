#!/usr/bin/env python3
"""
Analyze virtual time synchronization before and after.

Compares original and synchronized logs to show:
- How much each device was adjusted
- Whether devices are now synchronized at COMPUTE START
- Timeline visualization for each GEMM operation

Usage: python3 analyze_sync.py <original_log> <synced_log>

Examples:
  python3 scripts/analyze_sync.py perf_merged.log perf_merged_synced.log
  python3 scripts/analyze_sync.py logs/original.log logs/synced.log
"""

import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


@dataclass
class ComputeEvent:
    """Compute event info"""

    gemm_id: int
    device_id: int
    vt_start_us: int
    vt_end_us: int


def parse_log_for_compute_events(log_file: str) -> List[ComputeEvent]:
    """Extract all COMPUTE START/END events"""
    events = []

    with open(log_file, "r") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.startswith("VTIME,"):
                continue

            try:
                parts = line.split(",")
                if len(parts) < 9:
                    continue

                if parts[4] == "COMPUTE":  # phase
                    gemm_id = int(parts[3])
                    device_id = int(parts[2])
                    vt_start = int(parts[6])
                    vt_end = int(parts[7])

                    events.append(
                        ComputeEvent(
                            gemm_id=gemm_id,
                            device_id=device_id,
                            vt_start_us=vt_start,
                            vt_end_us=vt_end,
                        )
                    )
            except (ValueError, IndexError):
                continue

    return events


def organize_by_gemm(
    events: List[ComputeEvent],
) -> Dict[int, List[ComputeEvent]]:
    """Group events by GEMM ID"""
    result = defaultdict(list)
    for event in events:
        result[event.gemm_id].append(event)
    return result


def calculate_sync_metrics(
    original_events: Dict[int, List[ComputeEvent]],
    synced_events: Dict[int, List[ComputeEvent]],
) -> Dict[int, Dict]:
    """
    Calculate synchronization quality metrics for each GEMM operation.

    Returns:
        Dict[gemm_id] = {
            'devices': [device_ids],
            'original_vt_starts': {device_id: vt_start},
            'synced_vt_starts': {device_id: vt_start},
            'original_spread_us': max - min,
            'synced_spread_us': max - min,
            'improvement_us': reduction in spread,
            'improvement_pct': reduction percentage,
            'is_synchronized': spread < threshold
        }
    """
    SYNC_THRESHOLD_US = 100  # Consider synchronized if spread < 100 us

    metrics = {}

    for gemm_id in original_events.keys():
        if gemm_id not in synced_events:
            continue

        orig_events = original_events[gemm_id]
        sync_events = synced_events[gemm_id]

        # Group by device
        orig_by_device = {e.device_id: e for e in orig_events}
        sync_by_device = {e.device_id: e for e in sync_events}

        devices = sorted(orig_by_device.keys())

        # Get VT starts
        orig_vt_starts = {d: orig_by_device[d].vt_start_us for d in devices}
        sync_vt_starts = {d: sync_by_device[d].vt_start_us for d in devices}

        # Calculate spread
        orig_spread = max(orig_vt_starts.values()) - min(
            orig_vt_starts.values()
        )
        sync_spread = max(sync_vt_starts.values()) - min(
            sync_vt_starts.values()
        )
        improvement = orig_spread - sync_spread
        improvement_pct = (
            (improvement / orig_spread * 100) if orig_spread > 0 else 0
        )

        metrics[gemm_id] = {
            "devices": devices,
            "original_vt_starts": orig_vt_starts,
            "synced_vt_starts": sync_vt_starts,
            "original_spread_us": orig_spread,
            "synced_spread_us": sync_spread,
            "improvement_us": improvement,
            "improvement_pct": improvement_pct,
            "is_synchronized": sync_spread < SYNC_THRESHOLD_US,
            "sync_threshold_us": SYNC_THRESHOLD_US,
        }

    return metrics


def print_comparison(metrics: Dict[int, Dict]):
    """Print detailed comparison report"""
    print("\n" + "=" * 90)
    print("VIRTUAL TIME SYNCHRONIZATION ANALYSIS")
    print("=" * 90)

    # Overall statistics
    total_gemm = len(metrics)
    synced_count = sum(1 for m in metrics.values() if m["is_synchronized"])

    print(f"\nOVERALL RESULTS:")
    print(f"  Total GEMM operations: {total_gemm}")
    print(
        f"  Successfully synchronized: {synced_count} ({synced_count * 100 // total_gemm}%)"
    )
    print(f"  Synchronization threshold: <100 us")

    # Per-GEMM details
    print(f"\nDETAILED SYNCHRONIZATION METRICS BY GEMM OPERATION:")
    print("-" * 90)

    for gemm_id in sorted(metrics.keys()):
        m = metrics[gemm_id]

        print(f"\nGEMM {gemm_id}:")
        print(f"  Devices: {m['devices']}")

        # Original state
        print(f"  ┌─ BEFORE SYNC:")
        print(f"  │  Spread (max-min): {m['original_spread_us']:>6} us")
        for dev_id in m["devices"]:
            vt = m["original_vt_starts"][dev_id]
            print(f"  │    Device {dev_id}: {vt:>12} us")

        # Synchronized state
        print(f"  └─ AFTER SYNC:")
        print(f"     Spread (max-min): {m['synced_spread_us']:>6} us")
        for dev_id in m["devices"]:
            vt = m["synced_vt_starts"][dev_id]
            print(f"       Device {dev_id}: {vt:>12} us")

        # Improvement
        status = (
            "✓ SYNCHRONIZED" if m["is_synchronized"] else "✗ NOT SYNCHRONIZED"
        )
        print(f"  Status: {status}")
        print(
            f"  Improvement: {m['improvement_us']:>6} us ({m['improvement_pct']:>6.1f}%)"
        )


def print_timeline(metrics: Dict[int, Dict]):
    """Print ASCII timeline visualization"""
    print("\n" + "=" * 90)
    print("TIMELINE VISUALIZATION")
    print("=" * 90)

    for gemm_id in sorted(metrics.keys())[:5]:  # Show first 5 GEMM ops
        m = metrics[gemm_id]

        print(f"\nGEMM {gemm_id} - BEFORE SYNC:")
        orig_times = m["original_vt_starts"]
        min_time = min(orig_times.values())

        for dev_id in m["devices"]:
            vt = orig_times[dev_id]
            offset_from_min = vt - min_time

            # Scale: 20 chars per 1000 us
            bar_length = max(1, offset_from_min // 50)
            bar = "─" * bar_length + "●"
            print(
                f"  Device {dev_id}: {bar} {vt} us (offset: +{offset_from_min} us)"
            )

        print(f"GEMM {gemm_id} - AFTER SYNC:")
        sync_times = m["synced_vt_starts"]
        min_time = min(sync_times.values())

        for dev_id in m["devices"]:
            vt = sync_times[dev_id]
            offset_from_min = vt - min_time

            bar_length = max(1, offset_from_min // 50)
            bar = "─" * bar_length + "●"
            status = "✓" if offset_from_min < 100 else "✗"
            print(
                f"  Device {dev_id}: {bar} {vt} us (offset: +{offset_from_min} us) {status}"
            )


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)

    original_log = sys.argv[1]
    synced_log = sys.argv[2]

    print(f"Analyzing synchronization...")
    print(f"  Original log: {original_log}")
    print(f"  Synced log:   {synced_log}")

    # Parse logs
    print(f"\nParsing original log...")
    orig_events = parse_log_for_compute_events(original_log)
    orig_by_gemm = organize_by_gemm(orig_events)

    print(f"Parsing synchronized log...")
    sync_events = parse_log_for_compute_events(synced_log)
    sync_by_gemm = organize_by_gemm(sync_events)

    print(f"  Found {len(orig_events)} COMPUTE events in original")
    print(f"  Found {len(sync_events)} COMPUTE events in synced")

    # Calculate metrics
    metrics = calculate_sync_metrics(orig_by_gemm, sync_by_gemm)

    # Print reports
    print_comparison(metrics)
    print_timeline(metrics)

    print("\n" + "=" * 90)
    print("Analysis complete!")
    print("=" * 90)


if __name__ == "__main__":
    main()
