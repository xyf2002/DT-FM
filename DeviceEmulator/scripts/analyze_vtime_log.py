#!/usr/bin/env python3
"""
Virtual Time Log Parser and Analyzer

Parses perf.log and extracts virtual time events for analysis.
Displays duration with appropriate units (microseconds or milliseconds).

Usage:
  python3 scripts/analyze_vtime_log.py ./perf.log
  python3 scripts/analyze_vtime_log.py ./logs/perf_merged.log
"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


def format_duration(us):
    """Format duration with appropriate units (us or ms)."""
    if us >= 1000:
        return f"{us / 1000:.2f} ms"
    else:
        return f"{us:.0f} us"


def parse_perf_log(log_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Parse perf.log and separate regular metrics from virtual time events.

    Args:
        log_path: Path to perf.log file

    Returns:
        Tuple of (regular_df, vtime_df)
    """
    log_file = Path(log_path)

    if not log_file.exists():
        print(f"Error: Log file not found: {log_path}")
        sys.exit(1)

    # Read raw data
    raw_data = []
    vtime_data = []

    with open(log_file, "r") as f:
        headers = None
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue

            # Skip header line
            if i == 0:
                headers = line.split(",")
                continue

            # Parse data lines
            if line.startswith("VTIME"):
                # Virtual time event format:
                # VTIME,timestamp_us,device_id,phase,event,vt_start_us,vt_end_us,vt_duration_us
                parts = line.split(",")
                vtime_data.append(
                    {
                        "type": "VTIME",
                        "timestamp_us": int(parts[1]),
                        "device_id": int(parts[2]),
                        "phase": parts[3],
                        "event": parts[4],
                        "vt_start_us": int(parts[5]),
                        "vt_end_us": int(parts[6]),
                        "vt_duration_us": int(parts[7]),
                    }
                )
            else:
                # Regular throughput metrics
                try:
                    parts = line.split(",")
                    raw_data.append(
                        {
                            "timestamp_us": int(parts[0]),
                            "device_id": int(parts[1]),
                            "direction": parts[2],
                            "bytes": int(parts[3]),
                            "throughput_b_s": float(parts[4]),
                            "epoch_start_us": int(parts[5]),
                            "epoch_end_us": int(parts[6]),
                            "packet_duration_us": int(parts[7]),
                        }
                    )
                except (ValueError, IndexError) as e:
                    print(f"Warning: Failed to parse line {i}: {line}")
                    continue

    regular_df = pd.DataFrame(raw_data) if raw_data else pd.DataFrame()
    vtime_df = pd.DataFrame(vtime_data) if vtime_data else pd.DataFrame()

    return regular_df, vtime_df


def print_vtime_summary(vtime_df: pd.DataFrame) -> None:
    """Print summary of virtual time events."""

    if vtime_df.empty:
        print("No virtual time events found in log.")
        return

    print("\n" + "=" * 80)
    print("VIRTUAL TIME EVENTS SUMMARY")
    print("=" * 80)

    # Group by device
    for device_id in sorted(vtime_df["device_id"].unique()):
        device_events = vtime_df[vtime_df["device_id"] == device_id]
        print(f"\nDevice ID: {device_id}")
        print("-" * 80)

        # Group by phase
        for phase in ["SEND", "COMPUTE", "RECEIVE"]:
            phase_events = device_events[device_events["phase"] == phase]
            if phase_events.empty:
                continue

            print(f"\n  {phase} Events:")
            print(f"    Total events: {len(phase_events)}")

            if len(phase_events) > 0:
                total_duration = phase_events["vt_duration_us"].sum()
                avg_duration = phase_events["vt_duration_us"].mean()
                min_duration = phase_events["vt_duration_us"].min()
                max_duration = phase_events["vt_duration_us"].max()

                # Format durations with appropriate units
                def format_duration(us):
                    if us >= 1000:
                        return f"{us / 1000:.2f} ms"
                    else:
                        return f"{us:.0f} us"

                print(
                    f"    Total duration: {format_duration(total_duration)} "
                    f"({total_duration:,} us)"
                )
                print(f"    Avg duration:   {format_duration(avg_duration)}")
                print(f"    Min duration:   {format_duration(min_duration)}")
                print(f"    Max duration:   {format_duration(max_duration)}")

                # Show first few events
                print(f"    First 3 events:")
                for idx, row in phase_events.head(3).iterrows():
                    duration_str = format_duration(row["vt_duration_us"])
                    print(
                        f"      [{row['vt_start_us']:15d} -> {row['vt_end_us']:15d}] "
                        f"{duration_str:>10s}"
                    )


def print_timeline(vtime_df: pd.DataFrame) -> None:
    """Print a timeline view of virtual time events."""

    if vtime_df.empty:
        print("No virtual time events for timeline.")
        return

    print("\n" + "=" * 80)
    print("VIRTUAL TIME TIMELINE (Visual Bar Chart)")
    print("=" * 80)

    # Sort by vt_start_us
    sorted_df = vtime_df.sort_values("vt_start_us")
    first_time = sorted_df.iloc[0]["vt_start_us"]

    for idx, row in sorted_df.iterrows():
        device_id = row["device_id"]
        phase = row["phase"]
        vt_start = row["vt_start_us"]
        duration = row["vt_duration_us"]

        # Create a visual timeline bar (relative to first event)
        relative_start = (vt_start - first_time) // 1000  # Convert to ms

        # Format duration appropriately
        if duration >= 1000:
            duration_str = f"{duration / 1000:.2f} ms"
            bar_length = duration // 1000  # Scale: 1ms = 1 char
        else:
            duration_str = f"{duration:.0f} us"
            bar_length = duration // 100  # Scale: 100us = 1 char

        bar = "█" * min(bar_length, 50)

        print(
            f"[D{device_id:2d}] {phase:8s} {relative_start:6d}ms {bar:50s} {duration_str:>10s}"
        )


def print_device_timeline(vtime_df: pd.DataFrame) -> None:
    """Print per-device timeline of events."""

    if vtime_df.empty:
        return

    print("\n" + "=" * 80)
    print("PER-DEVICE TIMELINE (with proper unit scaling)")
    print("=" * 80)

    for device_id in sorted(vtime_df["device_id"].unique()):
        device_events = vtime_df[
            vtime_df["device_id"] == device_id
        ].sort_values("vt_start_us")

        if device_events.empty:
            continue

        print(f"\nDevice {device_id}:")
        print("-" * 80)
        print(
            f"  {'Relative Time':>12s} {'Duration':>12s} {'Phase':>10s} {'Virtual Time Span':>35s}"
        )
        print("-" * 80)

        first_time = device_events.iloc[0]["vt_start_us"]

        for idx, row in device_events.iterrows():
            phase = row["phase"]
            vt_start = row["vt_start_us"]
            vt_end = row["vt_end_us"]
            duration = row["vt_duration_us"]

            relative_start = (vt_start - first_time) // 1000

            # Format duration with appropriate unit
            if duration >= 1000:
                duration_str = f"{duration / 1000:.2f} ms"
            else:
                duration_str = f"{duration:.0f} us"

            relative_str = f"{relative_start} ms"

            print(
                f"  {relative_str:>12s} {duration_str:>12s} {phase:>10s} "
                f"[{vt_start:12d} -> {vt_end:12d}]"
            )


def print_microsecond_precision_timeline(vtime_df: pd.DataFrame) -> None:
    """Print timeline with microsecond precision for detailed analysis."""

    if vtime_df.empty:
        return

    print("\n" + "=" * 80)
    print("MICROSECOND PRECISION TIMELINE (First 20 events)")
    print("=" * 80)
    print(
        f"  {'#':>3s} {'Device':>3s} {'Phase':>8s} {'Start (us)':>15s} {'End (us)':>15s} "
        f"{'Duration (us)':>15s}"
    )
    print("-" * 80)

    sorted_df = vtime_df.sort_values("vt_start_us").head(20)

    for count, (idx, row) in enumerate(sorted_df.iterrows(), 1):
        device_id = row["device_id"]
        phase = row["phase"]
        vt_start = row["vt_start_us"]
        vt_end = row["vt_end_us"]
        duration = row["vt_duration_us"]

        print(
            f"  {count:3d} {device_id:3d} {phase:>8s} {vt_start:15d} {vt_end:15d} {duration:15d}"
        )


def main():
    """Main function."""

    if len(sys.argv) < 2:
        log_path = "./perf.log"
    else:
        log_path = sys.argv[1]

    print(f"Parsing log file: {log_path}")

    # Parse logs
    regular_df, vtime_df = parse_perf_log(log_path)

    # Print summaries
    if not regular_df.empty:
        print(f"\nFound {len(regular_df)} regular throughput records")

    if not vtime_df.empty:
        print(f"Found {len(vtime_df)} virtual time events")
        print_vtime_summary(vtime_df)
        print_timeline(vtime_df)
        print_microsecond_precision_timeline(vtime_df)
        print_device_timeline(vtime_df)
    else:
        print("No virtual time events found in log.")

    print("\n" + "=" * 80)
    print("Analysis complete.")
    print("=" * 80)


if __name__ == "__main__":
    main()
