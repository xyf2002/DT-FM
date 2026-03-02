#!/usr/bin/env python3
from collections import defaultdict


def parse_vtime_event(line):
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
    except:
        return None


# Read and count
vtime_events = []
with open("perf_merged_synced.log", "r") as f:
    for line in f:
        evt = parse_vtime_event(line.rstrip("\n"))
        if evt:
            vtime_events.append(evt)

print(f"Total VTIME events: {len(vtime_events)}")

# Count by phase and event type
by_phase = defaultdict(lambda: defaultdict(int))
for evt in vtime_events:
    by_phase[evt["phase"]][evt["event"]] += 1

print(f"\nBreakdown by phase and event type:")
for phase in sorted(by_phase.keys()):
    print(f"  {phase}:")
    for event_type in sorted(by_phase[phase].keys()):
        count = by_phase[phase][event_type]
        print(f"    {event_type}: {count}")

# Check GEMM 0 events only
gemm0_events = [
    e for e in vtime_events if e["gemm_id"] == 0 and e["event"] == "END"
]
print(f"\nGEMM 0 END events: {len(gemm0_events)}")

# Group by device and phase
by_device_phase = defaultdict(lambda: defaultdict(list))
for evt in gemm0_events:
    by_device_phase[evt["device_id"]][evt["phase"]].append(evt)

print(f"\nGEMM 0 breakdown by device and phase:")
for device in sorted(by_device_phase.keys()):
    print(f"  Device {device}:")
    for phase in sorted(by_device_phase[device].keys()):
        events = by_device_phase[device][phase]
        print(f"    {phase}: {len(events)} events")
        if len(events) <= 5:
            for e in events:
                print(
                    f"      vt: {e['vt_start_us']}-{e['vt_end_us']} (duration: {e['vt_duration_us']})"
                )
