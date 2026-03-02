#!/usr/bin/env python3
"""诊断虚拟时间事件"""

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


# Read only GEMM 0 events
gemm0_events = []
with open("perf_merged_synced.log", "r") as f:
    for line in f:
        evt = parse_vtime_event(line.rstrip("\n"))
        if evt and evt["gemm_id"] == 0:
            gemm0_events.append(evt)

print(f"GEMM 0 VTIME events: {len(gemm0_events)}")

# Group by device and phase, filter only END events
by_device_phase = defaultdict(lambda: defaultdict(list))
for evt in gemm0_events:
    if evt["event"] == "END":
        by_device_phase[evt["device_id"]][evt["phase"]].append(evt)

print("\nGEMM 0 END events by device and phase:")
for device in sorted(by_device_phase.keys()):
    for phase in sorted(by_device_phase[device].keys()):
        events = by_device_phase[device][phase]
        print(f"Device {device}, {phase}: {len(events)} events")
        for e in events:
            duration = e["vt_end_us"] - e["vt_start_us"]
            print(
                f"  vt_start={e['vt_start_us']}, vt_end={e['vt_end_us']}, duration={duration}"
            )
