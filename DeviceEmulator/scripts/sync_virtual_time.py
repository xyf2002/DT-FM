#!/usr/bin/env python3
"""
Synchronize virtual time across devices for each GEMM operation.

For each GEMM operation (identified by gemm_id), ensure all devices' COMPUTE operations
start at synchronized virtual time by applying calibration offsets (fx).

The virtual times are then normalized to start from 0 for easier analysis.

Usage: python3 sync_virtual_time.py <perf_merged.log> [--method earliest|latest]

Examples:
  python3 scripts/sync_virtual_time.py perf_merged.log --method earliest
  python3 scripts/sync_virtual_time.py perf_merged.log --method latest --output perf_merged_synced.log

Synchronization strategy:
- latest: Use the latest (slowest) COMPUTE START time as baseline, advance others (default, ensures no device lags)
- earliest: Use the earliest (fastest) COMPUTE START time as baseline, delay others (conservative approach)

Normalization:
- All virtual times are shifted so the minimum vt_start_us becomes 0
- This makes it easier to analyze the trace from the beginning
"""

import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


@dataclass
class VTEvent:
    """Virtual time event"""

    line: str
    timestamp_us: int
    device_id: int
    gemm_id: int
    phase: str
    event: str
    vt_start_us: int
    vt_end_us: int
    vt_duration_us: int


@dataclass
class ThroughputEvent:
    """Throughput event"""

    line: str
    timestamp_us: int
    device_id: int
    gemm_id: int
    direction: str
    bytes: int
    throughput_b_s: float
    epoch_start_us: int
    epoch_end_us: int
    packet_duration_us: int


def parse_vtime_event(line: str) -> Optional[VTEvent]:
    """Parse a VTIME event line"""
    if not line.startswith("VTIME,"):
        return None

    try:
        parts = line.strip().split(",")
        if len(parts) < 9:
            return None

        return VTEvent(
            line=line,
            timestamp_us=int(parts[1]),
            device_id=int(parts[2]),
            gemm_id=int(parts[3]),
            phase=parts[4],
            event=parts[5],
            vt_start_us=int(parts[6]),
            vt_end_us=int(parts[7]),
            vt_duration_us=int(parts[8]),
        )
    except (ValueError, IndexError):
        return None


def parse_throughput_event(line: str) -> Optional[ThroughputEvent]:
    """Parse a throughput event line"""
    if (
        line.startswith("VTIME,")
        or line.startswith("#")
        or not line[0].isdigit()
    ):
        return None

    try:
        parts = line.strip().split(",")
        if len(parts) < 9:
            return None

        return ThroughputEvent(
            line=line,
            timestamp_us=int(parts[0]),
            device_id=int(parts[1]),
            gemm_id=int(parts[2]),
            direction=parts[3],
            bytes=int(parts[4]),
            throughput_b_s=float(parts[5]),
            epoch_start_us=int(parts[6]),
            epoch_end_us=int(parts[7]),
            packet_duration_us=int(parts[8]),
        )
    except (ValueError, IndexError):
        return None


def read_log(
    log_file: str,
) -> Tuple[List[VTEvent], List[ThroughputEvent], List[str]]:
    """Read and parse the merged log file"""
    vtime_events = []
    throughput_events = []
    other_lines = []

    with open(log_file, "r") as f:
        for line in f:
            line = line.rstrip("\n")

            # Try parsing as VTIME event
            vtime_event = parse_vtime_event(line)
            if vtime_event:
                vtime_events.append(vtime_event)
                continue

            # Try parsing as throughput event
            throughput_event = parse_throughput_event(line)
            if throughput_event:
                throughput_events.append(throughput_event)
                continue

            # Keep headers and comments
            other_lines.append(line)

    return vtime_events, throughput_events, other_lines


def find_compute_start_times(
    vtime_events: List[VTEvent],
) -> Dict[int, Dict[int, int]]:
    """
    Find COMPUTE START virtual time for each (gemm_id, device_id) pair.

    Returns:
        Dict[gemm_id][device_id] = vt_start_us
    """
    compute_starts = defaultdict(dict)

    for event in vtime_events:
        if event.phase == "COMPUTE" and event.event == "START":
            compute_starts[event.gemm_id][event.device_id] = event.vt_start_us

    return compute_starts


def calculate_offsets(
    compute_starts: Dict[int, Dict[int, int]], method: str = "earliest"
) -> Dict[int, Dict[int, int]]:
    """
    Calculate calibration offsets (fx) for each device in each GEMM operation.

    For each gemm_id:
        - If method='earliest': baseline = min(all vt_start_us), offset[device] = baseline - vt_start_us
        - If method='latest': baseline = max(all vt_start_us), offset[device] = baseline - vt_start_us

    Positive offset means advance the device's virtual time (slow device).
    Negative offset means delay the device's virtual time (fast device).

    Args:
        compute_starts: Dict[gemm_id][device_id] = vt_start_us
        method: 'earliest' (conservative) or 'latest' (aggressive)

    Returns:
        Dict[gemm_id][device_id] = offset (to be added to all virtual times for that device)
    """
    offsets = defaultdict(dict)

    for gemm_id, device_times in compute_starts.items():
        if not device_times:
            continue

        if method == "earliest":
            baseline = min(device_times.values())
            sync_method_name = "Earliest (conservative)"
        elif method == "latest":
            baseline = max(device_times.values())
            sync_method_name = "Latest (aggressive)"
        else:
            raise ValueError(f"Unknown method: {method}")

        for device_id, vt_start in device_times.items():
            offset = baseline - vt_start
            offsets[gemm_id][device_id] = offset

        # Log synchronization details
        if len(device_times) > 1:
            print(f"\nGEMM {gemm_id} - {sync_method_name} synchronization:")
            print(f"  Baseline virtual time: {baseline} us")
            for device_id in sorted(device_times.keys()):
                vt = device_times[device_id]
                offset = offsets[gemm_id][device_id]
                status = (
                    "ahead"
                    if offset < 0
                    else "behind"
                    if offset > 0
                    else "aligned"
                )
                print(
                    f"  Device {device_id}: vt_start={vt:>10} us, offset={offset:>6} us ({status})"
                )

    return offsets


def apply_offsets(
    vtime_events: List[VTEvent],
    throughput_events: List[ThroughputEvent],
    offsets: Dict[int, Dict[int, int]],
) -> Tuple[List[str], List[str]]:
    """
    Apply calibration offsets to all virtual times.

    Returns:
        (synchronized_vtime_lines, synchronized_throughput_lines)
    """
    synced_vtime_lines = []
    synced_throughput_lines = []

    # First pass: apply offsets and track all virtual times
    adjusted_vtime_events = []
    adjusted_throughput_events = []

    # Apply offsets to VTIME events
    for event in vtime_events:
        offset = offsets.get(event.gemm_id, {}).get(event.device_id, 0)

        # Apply offset to virtual times
        new_vt_start = event.vt_start_us + offset
        new_vt_end = event.vt_end_us + offset
        new_vt_duration = event.vt_duration_us  # Duration unchanged

        # Store adjusted event
        adjusted_event = VTEvent(
            line=event.line,
            timestamp_us=event.timestamp_us,
            device_id=event.device_id,
            gemm_id=event.gemm_id,
            phase=event.phase,
            event=event.event,
            vt_start_us=new_vt_start,
            vt_end_us=new_vt_end,
            vt_duration_us=new_vt_duration,
        )
        adjusted_vtime_events.append(adjusted_event)

    # Apply offsets to throughput events (epoch times also adjusted)
    for event in throughput_events:
        offset = offsets.get(event.gemm_id, {}).get(event.device_id, 0)

        # Apply offset to epoch times (if they're valid)
        new_epoch_start = (
            event.epoch_start_us + offset
            if event.epoch_start_us > 0
            else event.epoch_start_us
        )
        new_epoch_end = (
            event.epoch_end_us + offset
            if event.epoch_end_us > 0
            else event.epoch_end_us
        )

        # Store adjusted event
        adjusted_event = ThroughputEvent(
            line=event.line,
            timestamp_us=event.timestamp_us,
            device_id=event.device_id,
            gemm_id=event.gemm_id,
            direction=event.direction,
            bytes=event.bytes,
            throughput_b_s=event.throughput_b_s,
            epoch_start_us=new_epoch_start,
            epoch_end_us=new_epoch_end,
            packet_duration_us=event.packet_duration_us,
        )
        adjusted_throughput_events.append(adjusted_event)

    # Second pass: find minimum virtual time and normalize to start from 0
    min_vt = float("inf")

    for event in adjusted_vtime_events:
        if event.vt_start_us > 0:
            min_vt = min(min_vt, event.vt_start_us)

    for event in adjusted_throughput_events:
        if event.epoch_start_us > 0:
            min_vt = min(min_vt, event.epoch_start_us)

    if min_vt == float("inf"):
        min_vt = 0

    # Third pass: normalize to start from 0
    for event in adjusted_vtime_events:
        new_vt_start = event.vt_start_us - min_vt
        new_vt_end = event.vt_end_us - min_vt

        new_line = (
            f"VTIME,{event.timestamp_us},{event.device_id},{event.gemm_id},"
            f"{event.phase},{event.event},{new_vt_start},{new_vt_end},{event.vt_duration_us}"
        )
        synced_vtime_lines.append(new_line)

    for event in adjusted_throughput_events:
        new_epoch_start = (
            event.epoch_start_us - min_vt
            if event.epoch_start_us > 0
            else event.epoch_start_us
        )
        new_epoch_end = (
            event.epoch_end_us - min_vt
            if event.epoch_end_us > 0
            else event.epoch_end_us
        )

        new_line = (
            f"{event.timestamp_us},{event.device_id},{event.gemm_id},"
            f"{event.direction},{event.bytes},{event.throughput_b_s:.2f},"
            f"{new_epoch_start},{new_epoch_end},{event.packet_duration_us}"
        )
        synced_throughput_lines.append(new_line)

    return synced_vtime_lines, synced_throughput_lines


def write_synchronized_log(
    output_file: str,
    other_lines: List[str],
    synced_vtime_lines: List[str],
    synced_throughput_lines: List[str],
    vtime_events: List[VTEvent],
    throughput_events: List[ThroughputEvent],
):
    """Write synchronized log to file, maintaining chronological order"""

    # Create merged list of all events with timestamps
    all_events = []

    for line in other_lines:
        all_events.append((0, "header", line))

    for line in synced_vtime_lines:
        parts = line.split(",")
        timestamp_us = int(parts[1])
        all_events.append((timestamp_us, "vtime", line))

    for line in synced_throughput_lines:
        parts = line.split(",")
        timestamp_us = int(parts[0])
        all_events.append((timestamp_us, "throughput", line))

    # Sort by timestamp (headers go first)
    all_events.sort(key=lambda x: (x[0], 0 if x[1] == "header" else 1))

    # Write to file
    with open(output_file, "w") as f:
        for _, _, line in all_events:
            f.write(line + "\n")


def print_statistics(
    vtime_events: List[VTEvent],
    throughput_events: List[ThroughputEvent],
    offsets: Dict[int, Dict[int, int]],
):
    """Print synchronization statistics"""
    print("\n" + "=" * 70)
    print("VIRTUAL TIME SYNCHRONIZATION STATISTICS")
    print("=" * 70)

    # Count GEMM operations per device
    gemm_counts = defaultdict(set)
    for event in vtime_events:
        if event.phase == "COMPUTE" and event.event == "START":
            gemm_counts[event.device_id].add(event.gemm_id)

    print(f"\nTotal VTIME events: {len(vtime_events)}")
    print(f"Total throughput events: {len(throughput_events)}")
    print(f"Total GEMM operations synchronized: {len(offsets)}")

    print(f"\nDevices:")
    for device_id in sorted(gemm_counts.keys()):
        count = len(gemm_counts[device_id])
        print(f"  Device {device_id}: {count} GEMM operations")

    # Calculate total offset applied
    total_adjusted = 0
    for gemm_id, device_offsets in offsets.items():
        for device_id, offset in device_offsets.items():
            if offset != 0:
                total_adjusted += 1

    print(f"\nOffsets applied:")
    print(f"  Devices with non-zero offsets: {total_adjusted}")

    # Show offset range
    all_offsets = []
    for gemm_id, device_offsets in offsets.items():
        all_offsets.extend(device_offsets.values())

    if all_offsets:
        min_offset = min(all_offsets)
        max_offset = max(all_offsets)
        print(f"  Offset range: {min_offset} to {max_offset} us")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    log_file = sys.argv[1]
    method = "latest"  # Default method - synchronize to the slowest device

    # Parse arguments
    if "--method" in sys.argv:
        idx = sys.argv.index("--method")
        if idx + 1 < len(sys.argv):
            method = sys.argv[idx + 1]

    # Determine output file
    log_path = Path(log_file)
    output_file = log_path.parent / f"{log_path.stem}_synced.log"

    if "--output" in sys.argv:
        idx = sys.argv.index("--output")
        if idx + 1 < len(sys.argv):
            output_file = sys.argv[idx + 1]

    print(f"Reading log file: {log_file}")
    vtime_events, throughput_events, other_lines = read_log(log_file)

    print(f"Parsed {len(vtime_events)} VTIME events")
    print(f"Parsed {len(throughput_events)} throughput events")

    print(f"\nFinding COMPUTE START times for synchronization...")
    compute_starts = find_compute_start_times(vtime_events)

    print(f"Found {len(compute_starts)} GEMM operations")

    print(f"\nCalculating calibration offsets (method: {method})...")
    offsets = calculate_offsets(compute_starts, method=method)

    print(f"\nApplying offsets to virtual times...")
    synced_vtime_lines, synced_throughput_lines = apply_offsets(
        vtime_events, throughput_events, offsets
    )

    print(f"Writing synchronized log to: {output_file}")
    write_synchronized_log(
        str(output_file),
        other_lines,
        synced_vtime_lines,
        synced_throughput_lines,
        vtime_events,
        throughput_events,
    )

    print_statistics(vtime_events, throughput_events, offsets)

    print(f"\n✓ Synchronization complete!")
    print(f"Output: {output_file}")
    print("=" * 70)


if __name__ == "__main__":
    main()
