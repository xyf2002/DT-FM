#!/usr/bin/env python3
"""
Generate Gantt charts for device event timeline visualization.
Each device gets its own chart, with events displayed as horizontal bars.
X-axis: virtual time, Y-axis: event phases

Usage:
  python3 scripts/gantt_plot.py perf_merged.log . --gemm-range 0:3
  python3 scripts/gantt_plot.py perf_merged_synced.log output_dir
"""

import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend
import os
import sys
from collections import defaultdict

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# Color mapping for different event types
EVENT_COLORS = {
    "COMPUTE": "#2E86AB",  # Blue
    "DOWNLOAD": "#A23B72",  # Purple
    "UPLOAD": "#F18F01",  # Orange
    "SEND": "#C73E1D",  # Red
    "RECEIVE": "#6A994E",  # Green
}


def parse_vtime_event(line):
    """Parse VTIME event"""
    if not line.startswith("VTIME,"):
        return None

    try:
        parts = line.strip().split(",")
        if len(parts) < 9:
            return None

        return {
            "timestamp_us": int(parts[1]),
            "device_id": int(parts[2]),
            "gemm_id": int(parts[3]),
            "phase": parts[4],
            "event": parts[5],
            "vt_start_us": int(parts[6]),
            "vt_end_us": int(parts[7]),
            "vt_duration_us": int(parts[8]),
        }
    except (ValueError, IndexError):
        return None


def read_log(log_file):
    """Read log and extract VTIME events"""
    vtime_events = []

    with open(log_file, "r") as f:
        for line in f:
            line = line.rstrip("\n")
            vtime_event = parse_vtime_event(line)
            if vtime_event:
                vtime_events.append(vtime_event)

    return vtime_events


def parse_gemm_range(range_str):
    """Parse gemm range string like '0:5' or '3'"""
    if ":" in range_str:
        start, end = range_str.split(":")
        return int(start), int(end)
    else:
        gemm_id = int(range_str)
        return gemm_id, gemm_id


def extract_gantt_events(vtime_events, gemm_start=None, gemm_end=None):
    """Extract events for gantt chart, organized by device and phase"""
    device_events = defaultdict(list)

    for evt in vtime_events:
        device_id = evt["device_id"]
        gemm_id = evt["gemm_id"]

        # Filter by gemm range
        if gemm_start is not None:
            if gemm_id < gemm_start or gemm_id > gemm_end:
                continue

        # Only process END events to get complete information
        if evt["event"] != "END":
            continue

        start = evt["vt_start_us"]
        end = evt["vt_end_us"]
        duration = end - start

        # Skip anomalies
        if duration < 0 or duration > 10000000:
            continue

        device_events[device_id].append(
            {
                "gemm_id": gemm_id,
                "phase": evt["phase"],
                "start": start,
                "end": end,
                "duration": duration,
            }
        )

    return device_events


def plot_gantt_chart(device_id, events, output_dir, time_range=None):
    """Plot gantt chart for a single device"""

    if not events:
        print(f"Warning: No events for device {device_id}")
        return

    # Get unique phases and sort them
    phases = sorted(set(evt["phase"] for evt in events))
    phase_to_y = {phase: i for i, phase in enumerate(phases)}

    # Create figure - make it wider to accommodate large time ranges
    fig, ax = plt.subplots(figsize=(20, 8))

    # Get time range (use actual virtual time values)
    min_time = min(evt["start"] for evt in events)
    max_time = max(evt["end"] for evt in events)
    actual_span = max_time - min_time

    # Use provided time_range for consistent scaling with timeline_plot
    if time_range is None:
        time_range = (min_time, max_time)

    start_time, end_time = time_range
    time_span = end_time - start_time

    # Add padding to time range
    time_padding = time_span * 0.02

    # Plot events using ABSOLUTE virtual time
    for evt in events:
        y = phase_to_y[evt["phase"]]
        start = evt["start"]
        duration = evt["duration"]
        end = evt["end"]

        # Skip events outside time range
        if end < start_time or start > end_time:
            continue

        color = EVENT_COLORS.get(evt["phase"], "#808080")

        # Add horizontal bar - thicker for better visibility
        bar_height = 0.7
        ax.barh(
            y,
            duration,
            left=start,
            height=bar_height,
            color=color,
            edgecolor="black",
            linewidth=1.5,
            alpha=0.85,
        )

        # Add GEMM ID label in the middle of the bar
        bar_center = start + duration / 2
        # Only show label if bar is wide enough
        if duration > 50000:  # Only if wider than 50k μs
            ax.text(
                bar_center,
                y,
                f"G{evt['gemm_id']}",
                ha="center",
                va="center",
                fontsize=9,
                fontweight="bold",
                color="white",
                zorder=5,
            )

        # Add duration label above the bar
        if duration > 50000:
            duration_label = f"{duration}μs"
            ax.text(
                bar_center,
                y + bar_height / 2 + 0.15,
                duration_label,
                ha="center",
                va="bottom",
                fontsize=7,
                fontweight="bold",
                color="#333333",
            )

        # Add start time on the left side
        start_label = f"{evt['start']}"
        ax.text(
            start,
            y - bar_height / 2 - 0.15,
            start_label,
            ha="right",
            va="top",
            fontsize=7,
            fontweight="bold",
            color="#555555",
        )

        # Add end time on the right side
        end_label = f"{evt['end']}"
        ax.text(
            end,
            y - bar_height / 2 - 0.15,
            end_label,
            ha="left",
            va="top",
            fontsize=7,
            fontweight="bold",
            color="#555555",
        )

    # Configure axes
    ax.set_yticks(range(len(phases)))
    ax.set_yticklabels(phases, fontsize=12, fontweight="bold")
    ax.set_xlabel("Virtual Time (μs)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Event Phase", fontsize=13, fontweight="bold")
    ax.set_title(
        f"Device {device_id} Event Timeline (Gantt Chart)",
        fontsize=15,
        fontweight="bold",
        pad=20,
    )

    # Set x-axis limits - use the same scale as timeline_plot
    ax.set_xlim(start_time - time_padding, end_time + time_padding)

    # Format x-axis to show time values - same as timeline_plot
    def format_time(x, pos):
        if x >= 1e6:
            return f"{x / 1e6:.2f}ms"
        else:
            return f"{x / 1e3:.0f}k"

    ax.xaxis.set_major_formatter(FuncFormatter(format_time))
    ax.locator_params(axis="x", nbins=18)  # More tick marks like timeline_plot
    ax.grid(axis="x", alpha=0.5, linestyle="-", linewidth=0.7, color="#cccccc")
    ax.set_axisbelow(True)
    plt.xticks(rotation=45, ha="right", fontsize=11)

    # Add legend
    legend_patches = [
        mpatches.Patch(color=EVENT_COLORS.get(phase, "#808080"), label=phase)
        for phase in phases
    ]
    ax.legend(
        handles=legend_patches,
        loc="upper right",
        fontsize=11,
        framealpha=0.95,
        edgecolor="black",
        fancybox=True,
    )

    # Add statistics box
    stats_text = f"Events: {len(events)}\nTime Span: {time_span / 1e3:.0f}k μs"
    ax.text(
        0.02,
        0.98,
        stats_text,
        transform=ax.transAxes,
        fontsize=10,
        fontweight="bold",
        verticalalignment="top",
        bbox=dict(
            boxstyle="round",
            facecolor="#ffffcc",
            alpha=0.9,
            edgecolor="black",
            linewidth=1,
        ),
    )

    plt.tight_layout()

    # Save figure
    output_file = os.path.join(output_dir, f"gantt_device_{device_id}.png")
    plt.savefig(output_file, dpi=100, bbox_inches="tight")
    print(f"Saved: {output_file}")
    plt.close()


def main():
    if len(sys.argv) < 2:
        print(
            "Usage: python3 gantt_plot.py <log_file> [output_dir] [--gemm-range START:END]"
        )
        print(
            "Example: python3 gantt_plot.py perf_merged.log . --gemm-range 0:3"
        )
        sys.exit(1)

    log_file = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "."

    # Parse optional parameters
    gemm_start = None
    gemm_end = None

    i = 3
    while i < len(sys.argv):
        if sys.argv[i] == "--gemm-range" and i + 1 < len(sys.argv):
            gemm_start, gemm_end = parse_gemm_range(sys.argv[i + 1])
            i += 2
        else:
            i += 1

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Read and process log
    print(f"Reading log file: {log_file}")
    vtime_events = read_log(log_file)
    print(f"Parsed {len(vtime_events)} VTIME events")

    # Extract events
    print(
        f"Extracting events{'for gemm range ' + str(gemm_start) + ':' + str(gemm_end) if gemm_start else ''}..."
    )
    device_events = extract_gantt_events(vtime_events, gemm_start, gemm_end)

    # Calculate time range for consistent scaling (same as timeline_plot)
    min_time = float("inf")
    max_time = 0
    for device_id in device_events:
        for evt in device_events[device_id]:
            min_time = min(min_time, evt["start"])
            max_time = max(max_time, evt["end"])

    if min_time == float("inf"):
        min_time = 0
        max_time = 1000

    # Add padding like timeline_plot does
    time_span = max(max_time - min_time, 100000)
    time_range = (
        max(0, min_time - time_span * 0.05),
        max_time + time_span * 0.05,
    )
    print(f"Using time range: {time_range[0]:.0f} - {time_range[1]:.0f} μs")

    # Generate charts for each device
    print(f"Generating gantt charts for {len(device_events)} device(s)...")
    for device_id in sorted(device_events.keys()):
        events = device_events[device_id]
        print(f"  Device {device_id}: {len(events)} events")
        plot_gantt_chart(device_id, events, output_dir, time_range=time_range)

    print("Done!")


if __name__ == "__main__":
    main()
