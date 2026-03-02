#!/usr/bin/env python3
"""
Visualize virtual time timeline for devices - SIMPLIFIED VERSION
Shows COMPUTE, DOWNLOAD, UPLOAD events as horizontal bars.
Supports both real time (timestamp_us) and virtual time (vt_start_us, vt_end_us).

Usage:
  python3 scripts/timeline_plot.py perf_merged.log . --gemm-range 0:2
  python3 scripts/timeline_plot.py perf_merged_synced.log output_dir
"""

import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend
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
            "direction": parts[3],
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
    """Extract timeline events for plotting"""
    timeline = defaultdict(list)

    if use_vtime:
        # Use virtual time from VTIME events - only process END events to avoid duplication
        # Group by (device_id, gemm_id, phase) and take the END event
        seen_ops = {}
        MIN_DURATION = 10  # Minimum duration in microseconds to display

        for evt in vtime_events:
            device_id = evt["device_id"]
            phase = evt["phase"]
            gemm_id = evt["gemm_id"]
            event_type = evt["event"]  # START or END

            # Only process END events - they contain the full duration
            if event_type != "END":
                continue

            key = (device_id, gemm_id, phase)
            if key in seen_ops:
                continue  # Skip duplicates, keep first END event

            start = evt["vt_start_us"]
            end = evt["vt_end_us"]
            duration = end - start

            # Filter: skip events with very small durations or anomalies
            if duration < MIN_DURATION or duration > 10000000:
                continue

            timeline[device_id].append(
                {
                    "start": start,
                    "duration": duration,
                    "phase": phase,
                    "gemm_id": gemm_id,
                }
            )
            seen_ops[key] = True

        # Add throughput events (DOWNLOAD, UPLOAD) to virtual time timeline
        # These don't have vtime info, so use actual packet duration for visibility
        for evt in throughput_events:
            device_id = evt["device_id"]
            phase = evt["direction"]  # DOWNLOAD or UPLOAD
            gemm_id = evt["gemm_id"]
            # Use timestamp_us as start position (more reliable than epoch_end_us)
            start = evt["timestamp_us"]
            # Use packet_duration for bar width, but cap it for visibility
            duration = min(max(evt["packet_duration_us"], 10), 200000)

            timeline[device_id].append(
                {
                    "start": start,
                    "duration": duration,
                    "phase": phase,
                    "gemm_id": gemm_id,
                }
            )
    else:
        # Use real time (timestamp_us) for both start and duration
        # This gives accurate chronological ordering based on real time
        seen_ops = {}
        MIN_DURATION = 10
        start_times = defaultdict(dict)  # Store START event timestamps
        vt_durations = defaultdict(
            dict
        )  # Store virtual time durations as fallback

        # First pass: collect START event timestamps and virtual time durations
        for evt in vtime_events:
            device_id = evt["device_id"]
            phase = evt["phase"]
            gemm_id = evt["gemm_id"]
            event_type = evt["event"]

            key = (device_id, gemm_id, phase)

            if event_type == "START":
                start_times[device_id][key] = evt["timestamp_us"]

            # Store virtual time duration for all END events (to use as fallback)
            if event_type == "END":
                vt_duration = evt["vt_end_us"] - evt["vt_start_us"]
                if device_id not in vt_durations:
                    vt_durations[device_id] = {}
                vt_durations[device_id][key] = vt_duration

        # Second pass: process END events with real timestamps
        for evt in vtime_events:
            device_id = evt["device_id"]
            phase = evt["phase"]
            gemm_id = evt["gemm_id"]
            event_type = evt["event"]

            # Only process END events
            if event_type != "END":
                continue

            key = (device_id, gemm_id, phase)
            if key in seen_ops:
                continue

            # Get start time from START event if available
            if key in start_times.get(device_id, {}):
                start = start_times[device_id][key]
                end = evt["timestamp_us"]
                duration = end - start
            else:
                # For events without START (like SEND), use virtual time duration
                # but position them at the END timestamp
                start = evt["timestamp_us"]
                vt_duration = evt["vt_end_us"] - evt["vt_start_us"]
                duration = vt_duration

            # Filter by duration
            if duration < MIN_DURATION or duration > 10000000:
                continue

            timeline[device_id].append(
                {
                    "start": start,
                    "duration": duration,
                    "phase": phase,
                    "gemm_id": gemm_id,
                }
            )
            seen_ops[key] = True

        # Add throughput events to real time timeline
        for evt in throughput_events:
            device_id = evt["device_id"]
            phase = evt["direction"]
            gemm_id = evt["gemm_id"]
            # Use timestamp_us as position
            start = evt["timestamp_us"]
            duration = min(max(evt["packet_duration_us"], 10), 200000)

            timeline[device_id].append(
                {
                    "start": start,
                    "duration": duration,
                    "phase": phase,
                    "gemm_id": gemm_id,
                }
            )

    # Sort events by start time
    for device_id in timeline:
        timeline[device_id].sort(key=lambda x: x["start"])

    # Find minimum start time across all devices and events
    min_start_time = float("inf")
    for device_id in timeline:
        for event in timeline[device_id]:
            min_start_time = min(min_start_time, event["start"])

    if min_start_time == float("inf"):
        min_start_time = 0

    return timeline, min_start_time


def plot_gantt(
    timeline,
    devices,
    use_vtime=True,
    title="Gantt Chart",
    filename=None,
    time_range=None,
    gemm_range=(0, 0),
    offset_time=0,
):
    """Plot gantt chart with events grouped by phase - one chart per device

    Args:
        offset_time: Time offset to subtract from all timestamps (for starting at 0)
    """

    # Define desired phase order (reversed for top-to-bottom display)
    PHASE_ORDER = ["RECEIVE", "UPLOAD", "COMPUTE", "DOWNLOAD", "SEND"]

    for device_id in devices:
        if device_id not in timeline:
            continue

        print(f"  Creating gantt chart for device {device_id}...")

        # Get unique phases for this device, ordered according to PHASE_ORDER
        available_phases = set(event["phase"] for event in timeline[device_id])
        phases = [p for p in PHASE_ORDER if p in available_phases]
        phase_to_y = {phase: i for i, phase in enumerate(phases)}

        # Create figure
        fig, ax = plt.subplots(figsize=(20, 8))

        # Find time range if not specified
        if time_range is None:
            all_times = []
            for event in timeline[device_id]:
                all_times.append(event["start"])
                all_times.append(event["start"] + event["duration"])

            if all_times:
                actual_range = (min(all_times), max(all_times))
                time_span = actual_range[1] - actual_range[0]
                time_padding = time_span * 0.02
                time_range_to_use = (
                    max(0, actual_range[0] - time_padding),
                    actual_range[1] + time_padding,
                )
            else:
                time_range_to_use = (0, 1000)
        else:
            time_range_to_use = time_range

        start_time, end_time = time_range_to_use
        time_span = end_time - start_time

        print(f"    Time range: {start_time:.0f} - {end_time:.0f} μs")

        # Plot events for this device
        for event in timeline[device_id]:
            # Filter by GEMM range
            gemm_id = event.get("gemm_id", -1)
            if gemm_id < gemm_range[0] or gemm_id > gemm_range[1]:
                continue

            phase = event["phase"]
            y = phase_to_y[phase]

            # Apply offset to normalize time to start from 0
            event_start = event["start"] - offset_time
            event_end = event_start + event["duration"]

            # Skip events outside time range
            if event_end < start_time or event_start > end_time:
                continue

            # Clip event to time range
            clipped_start = max(event_start, start_time)
            clipped_end = min(event_end, end_time)
            clipped_duration = clipped_end - clipped_start

            color = EVENT_COLORS.get(phase, "#999999")

            # Draw horizontal bar
            ax.barh(
                y,
                clipped_duration,
                left=clipped_start,
                height=0.7,
                color=color,
                edgecolor="black",
                linewidth=1.2,
                alpha=0.85,
            )

            # Add GEMM ID label in the middle
            label_x = clipped_start + clipped_duration / 2
            if clipped_duration > 50000:  # Only if wide enough
                ax.text(
                    label_x,
                    y,
                    f"G{gemm_id}",
                    ha="center",
                    va="center",
                    fontsize=9,
                    fontweight="bold",
                    color="white",
                    zorder=10,
                )

            # Add duration label
            if clipped_duration > 100000:
                duration_text = f"{clipped_duration / 1e3:.0f}k"
                ax.text(
                    label_x,
                    y + 0.38,
                    duration_text,
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    color="#333333",
                    fontweight="bold",
                )

        # Configure axes
        ax.set_yticks(range(len(phases)))
        ax.set_yticklabels(phases, fontsize=12, fontweight="bold")
        ax.set_ylim(-0.5, len(phases) - 0.5)

        x_label = "Virtual Time (ms)" if use_vtime else "Real Time (ms)"
        ax.set_xlabel(x_label, fontsize=13, fontweight="bold")
        ax.set_ylabel("Event Phase", fontsize=13, fontweight="bold")
        ax.set_title(
            f"Device {device_id} Gantt Chart (GEMM {gemm_range[0]}-{gemm_range[1]})",
            fontsize=15,
            fontweight="bold",
            pad=20,
        )

        # Set x-axis limits
        time_padding = (end_time - start_time) * 0.02
        ax.set_xlim(start_time - time_padding, end_time + time_padding)

        # Format x-axis (convert to ms, 1 ms = 1000 us)
        def time_formatter(x, p):
            return f"{x / 1e3:.2f}ms"

        ax.xaxis.set_major_formatter(FuncFormatter(time_formatter))
        ax.locator_params(axis="x", nbins=18)
        ax.grid(
            True,
            axis="x",
            alpha=0.5,
            linestyle="-",
            linewidth=0.7,
            color="#cccccc",
        )
        ax.set_axisbelow(True)

        # Add legend - order from SEND to RECEIVE
        legend_order = ["SEND", "DOWNLOAD", "COMPUTE", "UPLOAD", "RECEIVE"]
        legend_elements = [
            mpatches.Patch(
                facecolor=EVENT_COLORS.get(phase, "#999999"), label=phase
            )
            for phase in legend_order
            if phase in phases
        ]
        ax.legend(
            handles=legend_elements,
            loc="upper right",
            fontsize=11,
            framealpha=0.95,
            edgecolor="#333333",
            fancybox=True,
        )

        # Rotate x labels
        plt.xticks(rotation=45, ha="right", fontsize=11)

        # Add time span info
        info_text = f"Time Span: {time_span / 1e3:.2f} ms"

        ax.text(
            0.98,
            0.02,
            info_text,
            transform=ax.transAxes,
            fontsize=11,
            ha="right",
            va="bottom",
            fontweight="bold",
            bbox=dict(
                boxstyle="round",
                facecolor="#ffffcc",
                alpha=0.95,
                edgecolor="#333333",
                linewidth=1.5,
            ),
        )

        plt.tight_layout()

        # Save gantt chart - include device_id in filename
        if filename:
            base_filename = filename.replace(".png", "")
            gantt_filename = f"{base_filename}_device{device_id}_gantt.png"
        else:
            gantt_filename = f"gantt_device_{device_id}.png"
        print(f"    Saving: {gantt_filename}")
        plt.savefig(gantt_filename, dpi=150, bbox_inches="tight")
        plt.close()


def plot_timeline(
    timeline,
    devices,
    use_vtime=True,
    title="Timeline",
    filename=None,
    time_range=None,
    gemm_range=(0, 0),
    offset_time=0,
):
    """Plot timeline with horizontal bars (横向时间线)

    Args:
        offset_time: Time offset to subtract from all timestamps (for starting at 0)
    """

    print(f"  Creating figure...")
    # 设置更宽的图形，时间在横向，更高的纵向空间以显示更粗的柱子
    fig, ax = plt.subplots(figsize=(32, 10))

    # Y position for each device
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

    print(
        f"  Plotting events (time range: {start_time:.0f} - {end_time:.0f} μs)..."
    )

    # Plot events for each device - filter by GEMM range
    for device_id in devices:
        if device_id not in timeline:
            continue

        y = y_positions[device_id]
        events = timeline[device_id]

        for event in events:
            # Filter events by GEMM ID (only show events in the specified GEMM range)
            gemm_id = event.get("gemm_id", -1)
            if gemm_id < gemm_range[0] or gemm_id > gemm_range[1]:
                continue

            # Apply offset to normalize time to start from 0
            event_start = event["start"] - offset_time
            event_end = event_start + event["duration"]

            # Skip events outside time range
            if event_end < start_time or event_start > end_time:
                continue

            # Clip event to time range
            clipped_start = max(event_start, start_time)
            clipped_end = min(event_end, end_time)
            clipped_duration = clipped_end - clipped_start

            phase = event["phase"]
            color = EVENT_COLORS.get(phase, "#999999")

            # Draw horizontal bar - much thicker for better visibility
            ax.barh(
                y,
                clipped_duration,
                left=clipped_start,
                height=0.6,
                color=color,
                edgecolor="#333333",
                linewidth=2,
                alpha=0.88,
            )

            # Add event label in the middle
            label_x = clipped_start + clipped_duration / 2
            ax.text(
                label_x,
                y,
                phase,
                ha="center",
                va="center",
                fontsize=11,
                fontweight="bold",
                color="white",
                zorder=10,
            )

            # Add start time on the left
            time_label_start = f"{clipped_start / 1e3:.2f}ms"
            ax.text(
                clipped_start,
                y - 0.38,
                time_label_start,
                ha="right",
                va="top",
                fontsize=9,
                color="#333333",
                fontweight="bold",
            )

            # Add end time on the right
            time_label_end = f"{clipped_end / 1e3:.2f}ms"
            ax.text(
                clipped_end,
                y - 0.38,
                time_label_end,
                ha="left",
                va="top",
                fontsize=9,
                color="#333333",
                fontweight="bold",
            )

    print(f"  Formatting axes...")

    # Set y-axis with more spacing
    ax.set_yticks(range(len(devices)))
    ax.set_yticklabels(
        [f"Device {d}" for d in devices], fontsize=15, fontweight="bold"
    )
    ax.set_ylim(-0.5, len(devices) - 0.5)

    # Set x-axis with better formatting
    x_label = "Virtual Time (ms)" if use_vtime else "Real Time (ms)"
    ax.set_xlabel(x_label, fontsize=14, fontweight="bold")
    ax.set_ylabel("Device", fontsize=14, fontweight="bold")
    ax.set_title(title, fontsize=17, fontweight="bold", pad=20)

    # Set x-axis limits with some padding
    time_padding = (end_time - start_time) * 0.02
    ax.set_xlim(start_time - time_padding, end_time + time_padding)

    # Format x-axis to show time more clearly - show more tick marks (convert to ms, 1 ms = 1000 us)
    def time_formatter(x, p):
        return f"{x / 1e3:.2f}ms"

    ax.xaxis.set_major_formatter(FuncFormatter(time_formatter))
    ax.locator_params(axis="x", nbins=18)  # More tick marks

    # Add grid - more visible
    ax.grid(
        True, axis="x", alpha=0.5, linestyle="-", linewidth=0.7, color="#cccccc"
    )
    ax.set_axisbelow(True)

    # Create legend - only show the event types that are actually displayed
    legend_elements = [
        mpatches.Patch(facecolor=EVENT_COLORS["SEND"], label="SEND"),
        mpatches.Patch(facecolor=EVENT_COLORS["DOWNLOAD"], label="DOWNLOAD"),
        mpatches.Patch(facecolor=EVENT_COLORS["COMPUTE"], label="COMPUTE"),
        mpatches.Patch(facecolor=EVENT_COLORS["UPLOAD"], label="UPLOAD"),
        mpatches.Patch(facecolor=EVENT_COLORS["RECEIVE"], label="RECEIVE"),
    ]
    ax.legend(
        handles=legend_elements,
        loc="upper right",
        fontsize=13,
        framealpha=0.95,
        edgecolor="#333333",
        fancybox=True,
    )

    # Rotate x labels for better readability
    plt.xticks(rotation=45, ha="right", fontsize=12)

    # Add time range info
    time_span = end_time - start_time
    info_text = f"Time Span: {time_span / 1e3:.2f} ms"

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
            boxstyle="round",
            facecolor="#ffffcc",
            alpha=0.95,
            edgecolor="#333333",
            linewidth=2,
        ),
    )

    plt.tight_layout()

    print(f"  Saving figure...")
    if filename:
        plt.savefig(filename, dpi=150, bbox_inches="tight")
        print(f"✓ Saved: {filename}")

    plt.close()  # 关闭图形释放内存


def main():
    if len(sys.argv) < 2:
        print(
            "Usage: python3 timeline_plot.py <log_file> [output_dir] [--gemm-range START:END]"
        )
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
    timeline_vtime, min_start_vtime = extract_timeline_events(
        vtime_events, throughput_events, use_vtime=True
    )

    # Extract timeline for real time
    print("Extracting real time timeline...")
    timeline_realtime, min_start_realtime = extract_timeline_events(
        vtime_events, throughput_events, use_vtime=False
    )

    print(f"  Virtual time starts at: {min_start_vtime:.0f} μs")
    print(f"  Real time starts at: {min_start_realtime:.0f} μs")

    # Determine time range based on GEMM range if specified
    vtime_range = None
    realtime_range = None

    # Always use first 5 GEMMs for clarity
    if not gemm_range:
        gemm_range = (0, 0)

    start_gemm, end_gemm = gemm_range
    print(f"\nFocusing on GEMM {start_gemm} to {end_gemm}...")

    # Find time range for these GEMMs
    min_vt = float("inf")
    max_vt = 0
    min_rt = float("inf")
    max_rt = 0

    for device_id in devices:
        if device_id in timeline_vtime:
            for idx, event in enumerate(timeline_vtime[device_id]):
                # Use event index if gemm_id not available
                current_gemm = event.get(
                    "gemm_id", idx // 6
                )  # Roughly 6 events per GEMM
                if start_gemm <= current_gemm <= end_gemm:
                    # Apply offset for normalized time
                    min_vt = min(min_vt, event["start"] - min_start_vtime)
                    max_vt = max(
                        max_vt,
                        event["start"] - min_start_vtime + event["duration"],
                    )

        if device_id in timeline_realtime:
            for idx, event in enumerate(timeline_realtime[device_id]):
                current_gemm = event.get("gemm_id", idx // 6)
                if start_gemm <= current_gemm <= end_gemm:
                    # Apply offset for normalized time starting at 0
                    min_rt = min(min_rt, event["start"] - min_start_realtime)
                    max_rt = max(
                        max_rt,
                        event["start"] - min_start_realtime + event["duration"],
                    )

    vtime_range = None
    realtime_range = None

    # Handle virtual time range - filter out anomalies
    if (
        min_vt != float("inf") and max_vt > 0 and max_vt < 100000000
    ):  # Max 100 seconds
        vt_span = max(max_vt - min_vt, 100000)  # At least 100ms span
        vtime_range = (max(0, min_vt - vt_span * 0.05), max_vt + vt_span * 0.05)
        print(
            f"  VTime range: {min_vt / 1e3:.2f} - {max_vt / 1e3:.2f} ms (span: {vt_span / 1e3:.2f} ms)"
        )

    # Handle real time range
    if min_rt != float("inf") and max_rt > 0:
        rt_span = max(max_rt - min_rt, 100000)  # At least 100ms span
        realtime_range = (
            max(0, min_rt - rt_span * 0.05),
            max_rt + rt_span * 0.05,
        )
        print(
            f"  RTime range: {min_rt / 1e3:.2f} - {max_rt / 1e3:.2f} ms (span: {rt_span / 1e3:.2f} ms)"
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
        gemm_range=gemm_range,
        offset_time=int(min_start_vtime),
    )

    # Plot gantt chart for virtual time
    print("\nPlotting virtual time gantt charts...")
    gantt_vtime_file = f"{output_dir}/gantt_vtime_{log_name}_gemm{gemm_range[0]}-{gemm_range[1]}.png"
    plot_gantt(
        timeline_vtime,
        devices,
        use_vtime=True,
        title=f"Virtual Time Gantt - {log_name} (GEMM {gemm_range[0]}-{gemm_range[1]})",
        filename=gantt_vtime_file,
        time_range=vtime_range,
        gemm_range=gemm_range,
        offset_time=int(min_start_vtime),
    )

    # Plot real time - start from 0
    print("Plotting real time timeline...")
    realtime_file = f"{output_dir}/timeline_realtime_{log_name}_gemm{gemm_range[0]}-{gemm_range[1]}.png"
    plot_timeline(
        timeline_realtime,
        devices,
        use_vtime=False,
        title=f"Real Time Timeline - {log_name} (GEMM {gemm_range[0]}-{gemm_range[1]}) [Normalized to 0]",
        filename=realtime_file,
        time_range=realtime_range,
        gemm_range=gemm_range,
        offset_time=int(min_start_realtime),
    )

    # Plot gantt chart for real time - start from 0
    print("Plotting real time gantt charts...")
    gantt_realtime_file = f"{output_dir}/gantt_realtime_{log_name}_gemm{gemm_range[0]}-{gemm_range[1]}.png"
    plot_gantt(
        timeline_realtime,
        devices,
        use_vtime=False,
        title=f"Real Time Gantt - {log_name} (GEMM {gemm_range[0]}-{gemm_range[1]}) [Normalized to 0]",
        filename=gantt_realtime_file,
        time_range=realtime_range,
        gemm_range=gemm_range,
        offset_time=int(min_start_realtime),
    )

    print("\n✓ Done!")
    print(f"Output files:")
    print(f"  Timeline plots:")
    print(f"    {vtime_file}")
    print(f"    {realtime_file}")
    print(f"  Gantt charts:")
    for device in devices:
        print(
            f"    gantt_vtime_{log_name}_gemm{gemm_range[0]}-{gemm_range[1]}_gantt_device_{device}.png"
        )
        print(
            f"    gantt_realtime_{log_name}_gemm{gemm_range[0]}-{gemm_range[1]}_gantt_device_{device}.png"
        )


if __name__ == "__main__":
    main()
