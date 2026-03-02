#!/usr/bin/env python3
"""
Visualize virtual time timeline for devices.
Shows COMPUTE, DOWNLOAD, UPLOAD events as horizontal bars.
Supports both real time (timestamp_us) and virtual time (vt_start_us, vt_end_us).

Usage:
  python3 scripts/timeline_plot_backup.py perf_merged.log . --gemm-range 0:4
  python3 scripts/timeline_plot_backup.py perf_merged_synced.log output_dir
"""

import sys
from collections import defaultdict

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

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


def parse_throughput_event(line):
    """Parse throughput event"""
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

        return {
            "timestamp_us": int(parts[0]),
            "device_id": int(parts[1]),
            "gemm_id": int(parts[2]),
            "direction": parts[3],  # DOWNLOAD or UPLOAD
            "bytes": int(parts[4]),
            "throughput_b_s": float(parts[5]),
            "epoch_start_us": int(parts[6]),
            "epoch_end_us": int(parts[7]),
            "packet_duration_us": int(parts[8]),
        }
    except (ValueError, IndexError):
        return None


def read_log(log_file):
    """Read log and extract events"""
    vtime_events = []
    throughput_events = []

    with open(log_file, "r") as f:
        for line in f:
            line = line.rstrip("\n")

            # Parse VTIME events
            vtime_event = parse_vtime_event(line)
            if vtime_event:
                vtime_events.append(vtime_event)
                continue

            # Parse throughput events
            throughput_event = parse_throughput_event(line)
            if throughput_event:
                throughput_events.append(throughput_event)

    return vtime_events, throughput_events


def extract_timeline_events(vtime_events, throughput_events, use_vtime=True):
    """
    Extract timeline events for plotting.

    Returns:
        Dict[device_id] -> List of (start_time, duration, phase, event_type)
    """
    timeline = defaultdict(list)

    if use_vtime:
        # Use virtual time from VTIME events
        for evt in vtime_events:
            device_id = evt["device_id"]
            phase = evt["phase"]
            start = evt["vt_start_us"]
            end = evt["vt_end_us"]
            duration = max(1, end - start)  # Ensure visible duration

            # Combine phase and event type for label
            event_type = f"{phase}/{evt['event']}"

            timeline[device_id].append(
                {
                    "start": start,
                    "duration": duration,
                    "phase": phase,
                    "event_type": event_type,
                    "gemm_id": evt["gemm_id"],
                }
            )

        # Add throughput events with epoch times
        for evt in throughput_events:
            device_id = evt["device_id"]
            direction = evt["direction"]

            # Only use if epoch times are valid
            if evt["epoch_start_us"] > 0 and evt["epoch_end_us"] > 0:
                start = evt["epoch_start_us"]
                end = evt["epoch_end_us"]
                duration = max(1, end - start)

                timeline[device_id].append(
                    {
                        "start": start,
                        "duration": duration,
                        "phase": direction,
                        "event_type": direction,
                        "gemm_id": evt["gemm_id"],
                    }
                )
    else:
        # Use real time (timestamp_us)
        for evt in vtime_events:
            device_id = evt["device_id"]
            phase = evt["phase"]

            # Use timestamp as reference point, estimate 1ms duration for visibility
            start = evt["timestamp_us"]
            duration = 1000  # 1ms for visibility in real time view

            event_type = f"{phase}/{evt['event']}"

            timeline[device_id].append(
                {
                    "start": start,
                    "duration": duration,
                    "phase": phase,
                    "event_type": event_type,
                    "gemm_id": evt["gemm_id"],
                }
            )

        # Add throughput events
        for evt in throughput_events:
            device_id = evt["device_id"]
            direction = evt["direction"]
            start = evt["timestamp_us"]
            duration = 1000  # 1ms for visibility

            timeline[device_id].append(
                {
                    "start": start,
                    "duration": duration,
                    "phase": direction,
                    "event_type": direction,
                    "gemm_id": evt["gemm_id"],
                }
            )

    # Sort events by start time
    for device_id in timeline:
        timeline[device_id].sort(key=lambda x: x["start"])

    return timeline


def plot_timeline(
    timeline,
    devices,
    use_vtime=True,
    title="Timeline",
    filename=None,
    time_range=None,
):
    """
    Plot timeline with devices as horizontal rows and time flowing left-to-right.

    Args:
        time_range: tuple of (start_time, end_time) to limit the view. If None, show all.
    """
    # Larger figure size for horizontal layout - wider is better for timeline
    fig, ax = plt.subplots(figsize=(24, 6))

    # Y position for each device (horizontal layout)
    y_positions = {device: i for i, device in enumerate(devices)}

    # Find time range if not specified
    if time_range is None:
        all_times = []
        for device_id in devices:
            if device_id in timeline:
                for event in timeline[device_id]:
                    all_times.append(event["start"])
                    all_times.append(event["start"] + event["duration"])

        if all_times:
            time_range = (min(all_times), max(all_times))
        else:
            time_range = (0, 1000)

    start_time, end_time = time_range

    # Plot events for each device
    for device_id in devices:
        if device_id not in timeline:
            continue

        y = y_positions[device_id]
        events = timeline[device_id]

        for event in events:
            event_start = event["start"]
            event_end = event["start"] + event["duration"]

            # Skip events outside time range
            if event_end < start_time or event_start > end_time:
                continue

            # Clip event to time range
            clipped_start = max(event_start, start_time)
            clipped_end = min(event_end, end_time)
            clipped_duration = clipped_end - clipped_start

            phase = event["phase"]
            color = EVENT_COLORS.get(phase, "#999999")

            # Draw horizontal bar with time on x-axis
            ax.barh(
                y,
                clipped_duration,
                left=clipped_start,
                height=0.6,
                color=color,
                edgecolor="black",
                linewidth=1.2,
                alpha=0.85,
            )

            # Add event label
            if (
                clipped_duration > 3000
            ):  # Only label if visible duration > 3000 us
                label_x = clipped_start + clipped_duration / 2
                ax.text(
                    label_x,
                    y,
                    phase,
                    ha="center",
                    va="center",
                    fontsize=8,
                    fontweight="bold",
                    color="white",
                    bbox=dict(
                        boxstyle="round,pad=0.3", facecolor="black", alpha=0.4
                    ),
                )

    # Set y-axis (devices)
    ax.set_yticks(range(len(devices)))
    ax.set_yticklabels(
        [f"Device {d}" for d in devices], fontsize=13, fontweight="bold"
    )
    ax.set_ylim(-0.5, len(devices) - 0.5)

    # Set x-axis (time - more detailed formatting)
    x_label = "Virtual Time (μs)" if use_vtime else "Real Time (μs)"
    ax.set_xlabel(x_label, fontsize=14, fontweight="bold", labelpad=10)
    ax.set_ylabel("Device", fontsize=14, fontweight="bold")
    ax.set_title(title, fontsize=16, fontweight="bold", pad=20)

    # Set x-axis limits
    ax.set_xlim(start_time, end_time)

    # Format x-axis with detailed time ticks
    time_span = end_time - start_time

    # More granular tick intervals for better clarity
    if time_span < 5000:  # < 5ms
        tick_interval = 500  # 0.5ms
        time_format = lambda x: f"{x / 1000:.2f}ms"
    elif time_span < 10000:  # < 10ms
        tick_interval = 1000  # 1ms
        time_format = lambda x: f"{x / 1000:.1f}ms"
    elif time_span < 50000:  # < 50ms
        tick_interval = 5000  # 5ms
        time_format = lambda x: f"{x / 1000:.1f}ms"
    elif time_span < 100000:  # < 100ms
        tick_interval = 10000  # 10ms
        time_format = lambda x: f"{x / 1000:.0f}ms"
    elif time_span < 1000000:  # < 1s
        tick_interval = 50000  # 50ms
        time_format = lambda x: f"{x / 1000:.0f}ms"
    else:
        tick_interval = 500000  # 500ms
        time_format = lambda x: f"{x / 1e6:.2f}s"

    # Generate custom tick positions with more ticks
    tick_positions = []
    tick_labels = []
    tick_pos = int((start_time // tick_interval) * tick_interval)
    while tick_pos <= end_time:
        if tick_pos >= start_time:
            tick_positions.append(tick_pos)
            tick_labels.append(time_format(tick_pos))
        tick_pos += tick_interval

    ax.set_xticks(tick_positions, minor=False)
    ax.set_xticklabels(
        tick_labels, rotation=45, ha="right", fontsize=11, fontweight="bold"
    )

    # Add minor ticks for better readability
    minor_interval = tick_interval // 5
    if minor_interval > 100:
        minor_positions = []
        minor_pos = int((start_time // minor_interval) * minor_interval)
        while minor_pos <= end_time:
            if minor_pos >= start_time:
                minor_positions.append(minor_pos)
            minor_pos += minor_interval
        ax.set_xticks(minor_positions, minor=True)
        ax.tick_params(axis="x", which="minor", length=3, width=0.5)

    # Add grid with more visible lines
    ax.grid(
        True,
        axis="x",
        which="major",
        alpha=0.6,
        linestyle="-",
        linewidth=1.0,
        color="gray",
    )
    ax.grid(
        True,
        axis="x",
        which="minor",
        alpha=0.2,
        linestyle=":",
        linewidth=0.5,
        color="lightgray",
    )
    ax.grid(
        True, axis="y", alpha=0.3, linestyle="--", linewidth=0.8, color="gray"
    )
    ax.set_axisbelow(True)

    # Create legend
    legend_elements = [
        mpatches.Patch(
            facecolor=EVENT_COLORS["COMPUTE"],
            label="COMPUTE",
            edgecolor="black",
            linewidth=1,
        ),
        mpatches.Patch(
            facecolor=EVENT_COLORS["DOWNLOAD"],
            label="DOWNLOAD",
            edgecolor="black",
            linewidth=1,
        ),
        mpatches.Patch(
            facecolor=EVENT_COLORS["UPLOAD"],
            label="UPLOAD",
            edgecolor="black",
            linewidth=1,
        ),
        mpatches.Patch(
            facecolor=EVENT_COLORS["SEND"],
            label="SEND",
            edgecolor="black",
            linewidth=1,
        ),
        mpatches.Patch(
            facecolor=EVENT_COLORS["RECEIVE"],
            label="RECEIVE",
            edgecolor="black",
            linewidth=1,
        ),
    ]
    ax.legend(
        handles=legend_elements,
        loc="upper right",
        fontsize=12,
        framealpha=0.98,
        edgecolor="black",
        title="Event Types",
        title_fontsize=12,
        frameon=True,
        fancybox=True,
    )

    # Format tick labels
    ax.tick_params(axis="both", labelsize=11)
    ax.tick_params(axis="x", which="major", length=8, width=1.2)
    ax.tick_params(axis="y", length=6, width=1.0)

    # Add time range info with detailed breakdown
    time_span = end_time - start_time
    if time_span < 1000000:
        span_str = f"{time_span / 1000:.2f} ms"
        start_str = f"{start_time / 1000:.2f} ms"
        end_str = f"{end_time / 1000:.2f} ms"
    else:
        span_str = f"{time_span / 1e6:.3f} s"
        start_str = f"{start_time / 1e6:.3f} s"
        end_str = f"{end_time / 1e6:.3f} s"

    info_text = f"Range: {start_str} → {end_str}  |  Span: {time_span:,.0f} μs ({span_str})"
    ax.text(
        0.98,
        0.02,
        info_text,
        transform=ax.transAxes,
        fontsize=12,
        ha="right",
        va="bottom",
        fontweight="bold",
        bbox=dict(
            boxstyle="round,pad=0.7",
            facecolor="lightyellow",
            alpha=0.95,
            edgecolor="black",
            linewidth=1.5,
        ),
    )

    plt.tight_layout()

    if filename:
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        print(f"✓ Saved: {filename}")

    plt.show()


def main():
    if len(sys.argv) < 2:
        print(
            "Usage: python3 timeline_plot.py <log_file> [output_dir] [--gemm-range START:END]"
        )
        print("\nExample:")
        print("  python3 timeline_plot.py perf_merged.log")
        print(
            "  python3 timeline_plot.py perf_merged_synced.log . --gemm-range 0:5"
        )
        print(
            "  python3 timeline_plot.py perf_merged_synced.log . --gemm-range 5:10"
        )
        print("\nThis will generate:")
        print("  - timeline_vtime_<logname>.png (Virtual Time)")
        print("  - timeline_realtime_<logname>.png (Real Time)")
        sys.exit(1)

    log_file = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "."

    # Parse gemm range if provided
    gemm_range = None
    if "--gemm-range" in sys.argv:
        idx = sys.argv.index("--gemm-range")
        if idx + 1 < len(sys.argv):
            range_str = sys.argv[idx + 1]
            try:
                start, end = map(int, range_str.split(":"))
                gemm_range = (start, end)
            except:
                print(f"Invalid gemm range format: {range_str}. Use START:END")
                sys.exit(1)

    # Get log name for output files
    import os

    log_name = os.path.splitext(os.path.basename(log_file))[0]

    print(f"Reading log file: {log_file}")
    vtime_events, throughput_events = read_log(log_file)

    print(f"Parsed {len(vtime_events)} VTIME events")
    print(f"Parsed {len(throughput_events)} throughput events")

    # Select devices (0 and 1)
    devices = [0, 1]

    # Extract timeline for virtual time
    print("\nExtracting virtual time timeline...")
    timeline_vtime = extract_timeline_events(
        vtime_events, throughput_events, use_vtime=True
    )

    # Extract timeline for real time
    print("Extracting real time timeline...")
    timeline_realtime = extract_timeline_events(
        vtime_events, throughput_events, use_vtime=False
    )

    # Determine time range based on GEMM range if specified
    vtime_range = None
    realtime_range = None

    if gemm_range:
        start_gemm, end_gemm = gemm_range
        print(f"\nFocusing on GEMM {start_gemm} to {end_gemm}...")

        # Find time range for these GEMMs in virtual time
        min_vt = float("inf")
        max_vt = 0
        min_rt = float("inf")
        max_rt = 0

        for device_id in devices:
            if device_id in timeline_vtime:
                for event in timeline_vtime[device_id]:
                    if start_gemm <= event.get("gemm_id", -1) <= end_gemm:
                        min_vt = min(min_vt, event["start"])
                        max_vt = max(max_vt, event["start"] + event["duration"])

            if device_id in timeline_realtime:
                for event in timeline_realtime[device_id]:
                    if start_gemm <= event.get("gemm_id", -1) <= end_gemm:
                        min_rt = min(min_rt, event["start"])
                        max_rt = max(max_rt, event["start"] + event["duration"])

        if min_vt != float("inf"):
            # Add padding
            vt_span = max_vt - min_vt
            vtime_range = (
                max(0, min_vt - vt_span * 0.05),
                max_vt + vt_span * 0.05,
            )

        if min_rt != float("inf"):
            rt_span = max_rt - min_rt
            realtime_range = (
                max(0, min_rt - rt_span * 0.05),
                max_rt + rt_span * 0.05,
            )
    else:
        # Auto-select first 5 GEMMs for better visibility
        print("\nAuto-selecting first 5 GEMMs for better visibility...")
        gemm_range = (0, 4)

        min_vt = float("inf")
        max_vt = 0
        min_rt = float("inf")
        max_rt = 0

        for device_id in devices:
            if device_id in timeline_vtime:
                for event in timeline_vtime[device_id]:
                    if 0 <= event.get("gemm_id", -1) <= 4:
                        min_vt = min(min_vt, event["start"])
                        max_vt = max(max_vt, event["start"] + event["duration"])

            if device_id in timeline_realtime:
                for event in timeline_realtime[device_id]:
                    if 0 <= event.get("gemm_id", -1) <= 4:
                        min_rt = min(min_rt, event["start"])
                        max_rt = max(max_rt, event["start"] + event["duration"])

        if min_vt != float("inf"):
            vt_span = max_vt - min_vt
            vtime_range = (
                max(0, min_vt - vt_span * 0.05),
                max_vt + vt_span * 0.05,
            )

        if min_rt != float("inf"):
            rt_span = max_rt - min_rt
            realtime_range = (
                max(0, min_rt - rt_span * 0.05),
                max_rt + rt_span * 0.05,
            )

    # Plot virtual time
    print("\nPlotting virtual time timeline...")
    vtime_file = f"{output_dir}/timeline_vtime_{log_name}_gemm{gemm_range[0]}-{gemm_range[1]}.png"
    plot_timeline(
        timeline_vtime,
        devices,
        use_vtime=True,
        title=f"Virtual Time Timeline - {log_name} (GEMM {gemm_range[0]}-{gemm_range[1]})",
        filename=vtime_file,
        time_range=vtime_range,
    )

    # Plot real time
    print("Plotting real time timeline...")
    realtime_file = f"{output_dir}/timeline_realtime_{log_name}_gemm{gemm_range[0]}-{gemm_range[1]}.png"
    plot_timeline(
        timeline_realtime,
        devices,
        use_vtime=False,
        title=f"Real Time Timeline - {log_name} (GEMM {gemm_range[0]}-{gemm_range[1]})",
        filename=realtime_file,
        time_range=realtime_range,
    )

    print("\n✓ Done!")
    print(f"Output files:")
    print(f"  {vtime_file}")
    print(f"  {realtime_file}")


if __name__ == "__main__":
    main()
