"""
Parse emulator logs for real/logical time and partition counts.

Usage:
  python3 scripts/log_parser.py --log_file ./logs/device_0.log
"""

import argparse
import re

parser = argparse.ArgumentParser(description="Morphling Emulator Interface")
parser.add_argument(
    "--log_file",
    type=str,
    help="The log file to parse",
)


# [11/29/24 14:25:19.991] [ info  ] [thread 2873140] Real time: 774523853us, Logical time: 109578857us
time_pattern = re.compile(r"Real time: (\d+)us, Logical time: (\d+)us")

# [11/29/24 14:26:27.935] [ info  ] [thread 2873140] [16902096931142170827] Number of partitions: 512 for A: [16384,2048] and B: [8192,2048]
partition_pattern = re.compile(r"Number of partitions: (\d+)")

args = parser.parse_args()
print(args.log_file)

real_times = []
logical_times = []
partition_counts = []
with open(args.log_file, "r") as f:
    for line in f:
        time_match = time_pattern.search(line)
        if time_match:
            print(time_match.groups())
            real_times.append(int(time_match.groups()[0]))
            logical_times.append(int(time_match.groups()[1]))
        partition_match = partition_pattern.search(line)
        if partition_match:
            print(partition_match.groups())
            partition_counts.append(int(partition_match.groups()[0]))

print("Real times", real_times)
print("Logical times", logical_times)
print("Partition counts", partition_counts)
