#!/usr/bin/env python3
"""
Performance log analyzer - parses perf.log and provides statistics
Correctly calculates server-level throughput and aligns timestamps across devices

Usage:
  python3 scripts/analyze_perf.py ./perf.log --skip-warmup 5 --devices 0,1
  python3 scripts/analyze_perf.py ./perf.log --direction upload --output perf_summary.json
"""

import argparse
import csv
import json
import sys
from collections import defaultdict
from statistics import mean, median, stdev


def calculate_peak_throughput(stats, direction):
    """
    Calculate peak concurrent throughput considering packet timing overlaps.

    Uses a sweep line algorithm:
    1. Create events for each packet start/end
    2. Sweep through time, tracking active packets
    3. At each moment, sum the throughput of overlapping packets
    4. Return the maximum concurrent throughput

    Args:
        stats: Dictionary of device stats with timestamps and bytes
        direction: 'upload' or 'download'

    Returns:
        Tuple of (peak_throughput_b_s, timestamp_when_peak_occurred)
    """

    # Collect all packet events (start and end times)
    events = []  # List of (time_us, event_type, device_id, throughput)

    for device_id, device_stats in stats.items():
        if not device_stats["timestamps"]:
            continue

        # Get the appropriate throughput list and epoch times
        if direction == "upload":
            tp_list = device_stats["upload"]
            epoch_starts = device_stats.get("epoch_starts", [])
            epoch_ends = device_stats.get("epoch_ends", [])
        else:
            tp_list = device_stats["download"]
            epoch_starts = device_stats.get("epoch_starts", [])
            epoch_ends = device_stats.get("epoch_ends", [])

        if not tp_list:
            continue

        # Create start/end events for each packet
        for i, tp in enumerate(tp_list):
            if i < len(epoch_starts) and i < len(epoch_ends):
                start_us = epoch_starts[i]
                end_us = epoch_ends[i]

                # Avoid division by zero
                duration_s = (end_us - start_us) / 1_000_000.0
                if duration_s <= 0:
                    duration_s = 0.001  # Minimum 1ms

                # Calculate throughput for this packet
                packet_size = 0
                if direction == "upload" and i < len(
                    device_stats["upload_bytes"]
                ):
                    packet_size = device_stats["upload_bytes"][i]
                elif direction == "download" and i < len(
                    device_stats["download_bytes"]
                ):
                    packet_size = device_stats["download_bytes"][i]

                if packet_size > 0:
                    tp_b_s = packet_size / duration_s

                    # Add start and end events
                    events.append(
                        (start_us, "start", device_id, tp_b_s, duration_s)
                    )
                    events.append(
                        (end_us, "end", device_id, tp_b_s, duration_s)
                    )

    if not events:
        return 0.0, 0.0

    # Sort events by time, with ends before starts at the same time
    # This ensures we count the maximum overlap correctly
    events.sort(
        key=lambda x: (x[0], x[1] == "start")
    )  # 'end' (False) comes before 'start' (True)

    # Sweep through events and track active packets
    active_packets = {}  # device_id -> (throughput, duration)
    peak_tp = 0.0
    peak_time_us = 0

    for event_time_us, event_type, device_id, tp_b_s, duration_s in events:
        if event_type == "start":
            # Packet starts: add to active set
            active_packets[device_id] = (tp_b_s, duration_s)
        else:
            # Packet ends: remove from active set
            if device_id in active_packets:
                del active_packets[device_id]

        # Calculate current throughput (sum of all active packets)
        current_tp = sum(tp for tp, _ in active_packets.values())

        # Track peak
        if current_tp > peak_tp:
            peak_tp = current_tp
            peak_time_us = event_time_us

    # Convert timestamp to seconds from global start if needed
    # For now, just return the peak throughput and when it occurred
    return peak_tp, peak_time_us / 1_000_000.0  # Convert to seconds


def analyze_perf_log(
    log_file, skip_warmup_seconds=0, target_devices=None, direction_filter=None
):
    """Analyze performance log file with proper aggregation"""

    if not log_file:
        print("Error: No log file specified")
        print("Usage: python3 analyze_perf.py <log_file>")
        return None

    try:
        with open(log_file, "r") as f:
            reader = csv.DictReader(f)

            # Collect statistics by device and direction
            stats = defaultdict(
                lambda: {
                    "upload": [],
                    "download": [],
                    "upload_bytes": [],
                    "download_bytes": [],
                    "packet_durations": [],
                    "timestamps": [],
                    "epoch_starts": [],
                    "epoch_ends": [],
                }
            )

            total_rows = 0
            for row in reader:
                total_rows += 1
                try:
                    device_id = str(row["device_id"]).strip()
                    direction = row["direction"].lower()
                    timestamp = int(row["timestamp_us"])
                    throughput = float(row["throughput_b_s"])
                    bytes_val = int(row["bytes"])
                    packet_duration = int(row["packet_duration_us"])
                    epoch_start = int(row["epoch_start_us"])
                    epoch_end = int(row["epoch_end_us"])

                    # Filter by device if specified
                    if target_devices and device_id not in target_devices:
                        continue

                    # Filter by direction if specified
                    if direction_filter and direction != direction_filter:
                        continue

                    stats[device_id]["timestamps"].append(timestamp)
                    stats[device_id]["packet_durations"].append(packet_duration)
                    stats[device_id]["epoch_starts"].append(epoch_start)
                    stats[device_id]["epoch_ends"].append(epoch_end)

                    if direction == "upload":
                        stats[device_id]["upload"].append(throughput)
                        stats[device_id]["upload_bytes"].append(bytes_val)
                    elif direction == "download":
                        stats[device_id]["download"].append(throughput)
                        stats[device_id]["download_bytes"].append(bytes_val)
                except (ValueError, KeyError):
                    continue

            if total_rows == 0 or len(stats) == 0:
                print("No data found in log file")
                return None

            # Calculate global time alignment
            all_timestamps = []
            for device_stats in stats.values():
                all_timestamps.extend(device_stats["timestamps"])

            if not all_timestamps:
                print("No valid timestamps found")
                return None

            global_start_us = min(all_timestamps)
            global_end_us = max(all_timestamps)
            global_duration_s = (global_end_us - global_start_us) / 1_000_000.0

            # Skip warmup if specified
            if skip_warmup_seconds > 0:
                warmup_us = skip_warmup_seconds * 1_000_000
                filtered_stats = defaultdict(
                    lambda: {
                        "upload": [],
                        "download": [],
                        "upload_bytes": [],
                        "download_bytes": [],
                        "packet_durations": [],
                        "timestamps": [],
                        "epoch_starts": [],
                        "epoch_ends": [],
                    }
                )

                for device_id, device_stats in stats.items():
                    for i, ts in enumerate(device_stats["timestamps"]):
                        if ts >= global_start_us + warmup_us:
                            if device_stats["upload"]:
                                filtered_stats[device_id]["upload"].append(
                                    device_stats["upload"][i]
                                )
                                filtered_stats[device_id][
                                    "upload_bytes"
                                ].append(device_stats["upload_bytes"][i])
                            if device_stats["download"]:
                                filtered_stats[device_id]["download"].append(
                                    device_stats["download"][i]
                                )
                                filtered_stats[device_id][
                                    "download_bytes"
                                ].append(device_stats["download_bytes"][i])
                            filtered_stats[device_id]["timestamps"].append(ts)
                            filtered_stats[device_id][
                                "packet_durations"
                            ].append(device_stats["packet_durations"][i])
                            filtered_stats[device_id]["epoch_starts"].append(
                                device_stats["epoch_starts"][i]
                            )
                            filtered_stats[device_id]["epoch_ends"].append(
                                device_stats["epoch_ends"][i]
                            )

                stats = filtered_stats
                global_start_us += warmup_us
                global_duration_s = (
                    global_end_us - global_start_us
                ) / 1_000_000.0

            # Print summary statistics
            print(f"\n{'=' * 80}")
            print(f"Performance Log Analysis: {log_file}")
            print(f"{'=' * 80}\n")
            print(f"Total entries processed: {total_rows}")
            print(f"Measurement window: {global_duration_s:.2f} seconds")
            print(f"Global start: {global_start_us} (us)")
            print(f"Global end: {global_end_us} (us)\n")

            # Per-device analysis
            device_summary = {}
            for device_id in sorted(stats.keys(), key=lambda x: int(x)):
                device_stats = stats[device_id]
                print(f"\nDevice {device_id}:")
                print(f"  {'-' * 60}")

                # Calculate per-device duration
                if device_stats["timestamps"]:
                    device_start = min(device_stats["timestamps"])
                    device_end = max(device_stats["timestamps"])
                    device_duration_s = (
                        device_end - device_start
                    ) / 1_000_000.0
                    print(f"  Runtime: {device_duration_s:.2f} seconds")
                else:
                    device_duration_s = 0

                device_info = {
                    "upload_bytes": 0,
                    "download_bytes": 0,
                    "runtime_s": device_duration_s,
                }

                # Upload statistics
                if device_stats["upload"]:
                    upload_tp = device_stats["upload"]
                    upload_bytes_total = sum(device_stats["upload_bytes"])
                    device_info["upload_bytes"] = upload_bytes_total

                    if device_duration_s > 0:
                        upload_rate = upload_bytes_total / device_duration_s
                        print(f"  Upload:")
                        print(
                            f"    Total: {upload_bytes_total / 1024 / 1024:.2f} MB"
                        )
                        print(f"    Packets: {len(upload_tp)}")
                        print(f"    Rate: {upload_rate / 1024 / 1024:.2f} MB/s")
                        print(
                            f"    Per-packet TP - Min: {min(upload_tp) / 1024 / 1024:.2f} MB/s, Max: {max(upload_tp) / 1024 / 1024:.2f} MB/s"
                        )
                        print(
                            f"    Per-packet TP - Mean: {mean(upload_tp) / 1024 / 1024:.2f} MB/s, Median: {median(upload_tp) / 1024 / 1024:.2f} MB/s"
                        )
                        if len(upload_tp) > 1:
                            print(
                                f"    Per-packet TP - StdDev: {stdev(upload_tp) / 1024 / 1024:.2f} MB/s"
                            )

                # Download statistics
                if device_stats["download"]:
                    download_tp = device_stats["download"]
                    download_bytes_total = sum(device_stats["download_bytes"])
                    device_info["download_bytes"] = download_bytes_total

                    if device_duration_s > 0:
                        download_rate = download_bytes_total / device_duration_s
                        print(f"  Download:")
                        print(
                            f"    Total: {download_bytes_total / 1024 / 1024:.2f} MB"
                        )
                        print(f"    Packets: {len(download_tp)}")
                        print(
                            f"    Rate: {download_rate / 1024 / 1024:.2f} MB/s"
                        )
                        print(
                            f"    Per-packet TP - Min: {min(download_tp) / 1024 / 1024:.2f} MB/s, Max: {max(download_tp) / 1024 / 1024:.2f} MB/s"
                        )
                        print(
                            f"    Per-packet TP - Mean: {mean(download_tp) / 1024 / 1024:.2f} MB/s, Median: {median(download_tp) / 1024 / 1024:.2f} MB/s"
                        )
                        if len(download_tp) > 1:
                            print(
                                f"    Per-packet TP - StdDev: {stdev(download_tp) / 1024 / 1024:.2f} MB/s"
                            )

                # Packet duration statistics
                if device_stats["packet_durations"]:
                    durations = device_stats["packet_durations"]
                    print(f"  Packet Duration (microseconds):")
                    print(f"    Min: {min(durations)}, Max: {max(durations)}")
                    print(
                        f"    Mean: {mean(durations):.2f}, Median: {median(durations):.2f}"
                    )
                    if len(durations) > 1:
                        print(f"    StdDev: {stdev(durations):.2f}")

                device_summary[device_id] = device_info

            # Server aggregated statistics (CORRECT CALCULATION)
            print(f"\n{'=' * 80}")
            print("Server Aggregated Statistics:")
            print(f"{'=' * 80}")

            total_upload_bytes = sum(
                info["upload_bytes"] for info in device_summary.values()
            )
            total_download_bytes = sum(
                info["download_bytes"] for info in device_summary.values()
            )
            total_bytes = total_upload_bytes + total_download_bytes

            print(f"\nMeasurement Window: {global_duration_s:.2f} seconds\n")

            if total_upload_bytes > 0:
                server_upload_tp = total_upload_bytes / global_duration_s
                print(f"Upload:")
                print(
                    f"  Total Bytes: {total_upload_bytes / 1024 / 1024:.2f} MB"
                )
                print(
                    f"  Average TP: {server_upload_tp / 1024 / 1024:.2f} MB/s"
                )

            if total_download_bytes > 0:
                server_download_tp = total_download_bytes / global_duration_s
                print(f"Download:")
                print(
                    f"  Total Bytes: {total_download_bytes / 1024 / 1024:.2f} MB"
                )
                print(
                    f"  Average TP: {server_download_tp / 1024 / 1024:.2f} MB/s"
                )

            server_total_tp = 0.0
            if total_bytes > 0:
                server_total_tp = total_bytes / global_duration_s
                print(f"\nTotal (Upload + Download):")
                print(f"  Total Bytes: {total_bytes / 1024 / 1024:.2f} MB")
                print(f"  Average TP: {server_total_tp / 1024 / 1024:.2f} MB/s")

            # Calculate peak concurrent throughput (considering packet timing overlaps)
            print(f"\n{'=' * 80}")
            print(
                "Peak Concurrent Throughput (considering packet timing overlaps):"
            )
            print(f"{'=' * 80}\n")

            peak_upload_tp, peak_upload_time = calculate_peak_throughput(
                stats, "upload"
            )
            peak_download_tp, peak_download_time = calculate_peak_throughput(
                stats, "download"
            )
            peak_total_tp = peak_upload_tp + peak_download_tp

            print(
                f"Peak Upload TP: {peak_upload_tp / 1024 / 1024:.2f} MB/s at {peak_upload_time:.6f}s"
            )
            print(
                f"Peak Download TP: {peak_download_tp / 1024 / 1024:.2f} MB/s at {peak_download_time:.6f}s"
            )
            print(f"Peak Total TP: {peak_total_tp / 1024 / 1024:.2f} MB/s")

            if total_bytes > 0:
                peak_ratio = (
                    peak_total_tp / server_total_tp
                    if server_total_tp > 0
                    else 0
                )
                print(f"\nPeak/Average Ratio: {peak_ratio:.2f}x")

            # Device comparison table
            print(f"\n{'=' * 80}")
            print("Device Comparison Table:")
            print(f"{'=' * 80}\n")
            print(
                f"{'Device':<8} {'Upload(MB)':<12} {'Download(MB)':<14} {'Runtime(s)':<12} {'UL Rate(MB/s)':<15} {'DL Rate(MB/s)':<15}"
            )
            print("-" * 100)

            for device_id in sorted(
                device_summary.keys(), key=lambda x: int(x)
            ):
                info = device_summary[device_id]
                runtime = info["runtime_s"]
                ul_mb = info["upload_bytes"] / 1024 / 1024
                dl_mb = info["download_bytes"] / 1024 / 1024

                if runtime > 0:
                    ul_rate = info["upload_bytes"] / runtime / 1024 / 1024
                    dl_rate = info["download_bytes"] / runtime / 1024 / 1024
                else:
                    ul_rate = dl_rate = 0

                print(
                    f"{device_id:<8} {ul_mb:<12.2f} {dl_mb:<14.2f} {runtime:<12.2f} {ul_rate:<15.2f} {dl_rate:<15.2f}"
                )

            print(f"\n{'=' * 80}\n")

            return {
                "total_rows": total_rows,
                "global_duration_s": global_duration_s,
                "total_upload_bytes": total_upload_bytes,
                "total_download_bytes": total_download_bytes,
                "total_bytes": total_bytes,
                "device_summary": device_summary,
            }

    except FileNotFoundError:
        print(f"Error: Log file not found: {log_file}")
        return None
    except Exception as e:
        print(f"Error analyzing log file: {e}")
        import traceback

        traceback.print_exc()
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze performance logs")
    parser.add_argument(
        "log_file",
        nargs="?",
        default="./perf.log",
        help="Path to perf.log file",
    )
    parser.add_argument(
        "--skip-warmup",
        type=float,
        default=0,
        help="Skip first N seconds (warmup period)",
    )
    parser.add_argument(
        "--devices",
        type=str,
        help='Analyze only specific devices (comma-separated, e.g., "0,1,2")',
    )
    parser.add_argument(
        "--direction",
        choices=["upload", "download"],
        help="Filter by direction",
    )
    parser.add_argument(
        "--output", type=str, help="Output results to JSON file"
    )

    args = parser.parse_args()

    target_devices = set(args.devices.split(",")) if args.devices else None

    result = analyze_perf_log(
        args.log_file,
        skip_warmup_seconds=args.skip_warmup,
        target_devices=target_devices,
        direction_filter=args.direction,
    )

    if result and args.output:
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Results saved to {args.output}")
