# read input file and parse time from example "Publish time: 6314us"

import argparse
import re


def parse_time(input_file, pattern):
    all_time = []
    with open(input_file, "r") as f:
        for line in f:
            match = re.search(pattern, line)
            if match:
                all_time.append(int(match.group(1)))
    return all_time


def main():
    parser = argparse.ArgumentParser(description="Parse time from input file")
    parser.add_argument("--file", type=str, help="input file")
    args = parser.parse_args()
    print(f"Input file: {args.file}")
    push_time = parse_time(args.file, r"Publish time: (\d+)us")
    mm_time = parse_time(args.file, r"Matmul time: (\d+)us")
    wait_time = parse_time(args.file, r"Waiting time: (\d+)us")

    # sum all time
    sum_time = sum(push_time) / 1e6
    print(f"Sum time: {sum_time}s")

    sum_time = sum(mm_time) / 1e6
    print(f"Sum time: {sum_time}s")

    sum_time = sum(wait_time) / 1e6
    print(f"Sum time: {sum_time}s")


if __name__ == "__main__":
    main()
