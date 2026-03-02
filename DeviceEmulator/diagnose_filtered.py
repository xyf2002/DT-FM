#!/usr/bin/env python3
"""诊断实际提取的虚拟时间事件"""

from collections import defaultdict


def parse_vtime_event(line):
    if not line.startswith("VTIME,"):
        return None
    try:
        parts = line.strip().split(",")
        if len(parts) < 9:
            return None
        return {
            "device_id": int(parts[2]),
            "gemm_id": int(parts[3]),
            "phase": parts[4],
            "event": parts[5],
            "vt_start_us": int(parts[6]),
            "vt_end_us": int(parts[7]),
        }
    except:
        return None


# Read VTIME events for GEMM 0
gemm0_events = []
all_events = []
with open("perf_merged_synced.log", "r") as f:
    for line in f:
        evt = parse_vtime_event(line.rstrip("\n"))
        if evt:
            all_events.append(evt)
            if evt["gemm_id"] == 0:
                gemm0_events.append(evt)

print(f"Total VTIME events: {len(all_events)}")
print(f"GEMM 0 VTIME events: {len(gemm0_events)}")

# Count GEMM 0 by event type
start_count = sum(1 for e in gemm0_events if e["event"] == "START")
end_count = sum(1 for e in gemm0_events if e["event"] == "END")
print(f"\nGEMM 0: {start_count} START, {end_count} END")

# Filter: only END events with duration >= 10
filtered = []
MIN_DURATION = 10
seen_ops = {}

for evt in gemm0_events:
    if evt["event"] != "END":
        continue

    key = (evt["device_id"], evt["gemm_id"], evt["phase"])
    if key in seen_ops:
        continue

    duration = evt["vt_end_us"] - evt["vt_start_us"]
    if duration < MIN_DURATION or duration > 10000000:
        continue

    filtered.append(evt)
    seen_ops[key] = True

print(
    f"\nFiltered GEMM 0 events (END only, duration >= {MIN_DURATION}): {len(filtered)}"
)
for evt in filtered:
    duration = evt["vt_end_us"] - evt["vt_start_us"]
    print(f"  Device {evt['device_id']}, {evt['phase']}: {duration}μs")
