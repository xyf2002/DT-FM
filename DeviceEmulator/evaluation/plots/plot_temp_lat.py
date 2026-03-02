import os

import cudf as gd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from filterpy.kalman import KalmanFilter
from joblib import Parallel, delayed
from scipy import stats
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm

plt.style.use("classic")
matplotlib.rcParams.update({"font.size": 28, "lines.linewidth": 3})
matplotlib.rcParams["axes.linewidth"] = 2
matplotlib.rcParams["axes.edgecolor"] = "black"

this_dir = os.path.join(os.path.dirname(__file__))

cpu_lat_file = os.path.join(this_dir, "cpu_workload_latency.csv")
gpu_lat_file = os.path.join(this_dir, "gpu_workload_latency.csv")
cpu_temp_file = os.path.join(this_dir, "cpu_workload_temperature.csv")
gpu_temp_file = os.path.join(this_dir, "gpu_workload_temperature.csv")


df_cpu_lat = gd.read_csv(cpu_lat_file)
df_gpu_lat = gd.read_csv(gpu_lat_file)
df_cpu_temp = gd.read_csv(cpu_temp_file)
df_gpu_temp = gd.read_csv(gpu_temp_file)


def remove_zeros(df, col):
    # remove zero rows from the dataframe on column col
    df = df[df[col] != 0]
    # sort by k in descending order
    df = df.sort_values(by="k", ascending=False)
    df["Time"] = df["Time"] / 1e3  # convert to s
    df["InSize_Full"] = (
        (df["m"] * df["k"] + df["k"] * df["n"]) * 4 / (1024 * 1024)
    )  # convert to MB
    df["OurSize_Full"] = (df["m"] * df["n"]) * 4 / (1024 * 1024)
    return df


df_cpu_lat = remove_zeros(df_cpu_lat, "Latency-ms_overhead")
df_gpu_lat = remove_zeros(df_gpu_lat, "Latency-ms_overhead")

df_cpu_lat[["Latency-ms_overhead", "Latency-ms_compute"]] = (
    df_cpu_lat[["Latency-ms_overhead", "Latency-ms_compute"]] / 1e6
)  # convert to ms
df_gpu_lat[["Latency-ms_overhead", "Latency-ms_compute"]] = (
    df_gpu_lat[["Latency-ms_overhead", "Latency-ms_compute"]] / 1e6
)  # convert to ms

df_cpu_temp = remove_zeros(df_cpu_temp, "cpuss-0")
df_gpu_temp = remove_zeros(df_gpu_temp, "gpuss-1")

print(df_cpu_lat)
print(df_gpu_lat)
print(df_cpu_temp)
print(df_gpu_temp)

# plot cpu temperature
grouped_cpu_temp = df_cpu_temp.groupby(["m", "n", "k"])
grouped_cpu_temp = {name: group for name, group in grouped_cpu_temp}

grouped_cpu_lat = df_cpu_lat.groupby(["m", "n", "k"])
grouped_cpu_lat = {name: group for name, group in grouped_cpu_lat}

grouped_gpu_temp = df_gpu_temp.groupby(["m", "n", "k"])
grouped_gpu_temp = {name: group for name, group in grouped_gpu_temp}

grouped_gpu_lat = df_gpu_lat.groupby(["m", "n", "k"])
grouped_gpu_lat = {name: group for name, group in grouped_gpu_lat}

print(f"{len(grouped_cpu_temp.keys())=}")
print(f"{len(grouped_cpu_lat.keys())=}")
print(f"{len(grouped_gpu_temp.keys())=}")
print(f"{len(grouped_gpu_lat.keys())=}")


def gaussian_filter(data):
    return gaussian_filter1d(data, sigma=1, radius=7)


def kalman_filter(data):
    kf = KalmanFilter(dim_x=1, dim_z=1)
    kf.x = np.zeros((1, 1))  # state
    kf.P *= 1000.0  # covariance matrix
    kf.R = 5  # measurement noise
    kf.Q = 0.1  # process noise
    kf.H = np.array([[1.0]])  # measurement function
    kf.F = np.array([[1.0]])  # state transition matrix
    filtered_data = []
    for measurement in data:
        kf.predict()
        kf.update(measurement)
        filtered_data.append(float(kf.x.item()))
    return np.array(filtered_data)


for name in tqdm(grouped_cpu_temp):
    if not name in grouped_cpu_lat or not name in grouped_cpu_temp:
        print(f"Skipping {name}")

    cpu_temp = grouped_cpu_temp[name]
    cpu_lat = grouped_cpu_lat[name]

    cpu_lat = cpu_lat.sort_values(by="Time")
    cpu_temp = cpu_temp.sort_values(by="Time")

    # plot dual y-axis
    fig, ax1 = plt.subplots(figsize=(9, 6))
    ax2 = ax1.twinx()

    # apply gaussian filter
    cpu_temp["cpuss-0"] = gaussian_filter(cpu_temp["cpuss-0"].values.get())
    cpu_lat["Latency-ms_compute"] = gaussian_filter(
        cpu_lat["Latency-ms_compute"].values.get()
    )

    ax1.plot(
        cpu_temp["Time"],
        cpu_temp["cpuss-0"],
        label="CPU Temp",
        color="red",
        alpha=0.7,
    )
    ax2.plot(
        cpu_lat["Time"],
        cpu_lat["Latency-ms_compute"],
        label="Compute Latency",
        color="blue",
        alpha=0.7,
    )

    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel(r"Temperature ($^\circ$C)", color="red")

    ax2.set_ylabel("Latency (ms)", color="blue")

    ax1.grid()
    # ax1.legend(loc='upper left')
    # ax2.legend(loc='upper right')

    output_path = os.path.join(this_dir, "figures", f"cpu_temp_lat_{name}.pdf")
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


for name in tqdm(grouped_gpu_temp):
    if not name in grouped_gpu_lat or not name in grouped_gpu_temp:
        print(f"Skipping {name}")
        continue

    gpu_temp = grouped_gpu_temp[name]
    gpu_lat = grouped_gpu_lat[name]

    gpu_lat = gpu_lat.sort_values(by="Time")
    gpu_temp = gpu_temp.sort_values(by="Time")

    # plot dual y-axis
    fig, ax1 = plt.subplots(figsize=(9, 6))
    ax2 = ax1.twinx()

    # apply gaussian filter
    gpu_temp["gpuss-1"] = gaussian_filter(gpu_temp["gpuss-1"].values.get())
    gpu_lat["Latency-ms_compute"] = gaussian_filter(
        gpu_lat["Latency-ms_compute"].values.get()
    )

    ax1.plot(
        gpu_temp["Time"],
        gpu_temp["gpuss-1"],
        label="GPU Temp",
        color="red",
        alpha=0.7,
    )
    ax2.plot(
        gpu_lat["Time"],
        gpu_lat["Latency-ms_compute"],
        label="Compute Latency",
        color="blue",
        alpha=0.7,
    )

    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel(r"Temperature ($^\circ$C)", color="red")

    ax2.set_ylabel("Latency (ms)", color="blue")

    ax1.grid()
    # ax1.legend(loc='upper left')
    # ax2.legend(loc='upper right')

    output_path = os.path.join(this_dir, "figures", f"gpu_temp_lat_{name}.pdf")
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


# # convert to list
# grouped_cpu_temp = [(name, group) for name, group in grouped_cpu_temp]
# grouped_cpu_temp = grouped_cpu_temp[::-1]
# print(f"{len(grouped_cpu_temp)=}")
# for name, group in grouped_cpu_temp:
#     group = group.sort_values(by='Time')
#     print(name)
#     print(group)
#     plt.figure(figsize=(10, 6))
#     plt.plot(group['Time'], group['cpuss-0'], label='CPU Temp')
#     plt.xlabel('Time (s)')
#     plt.ylabel(r'Temperature ($^\circ$C)')
#     # plt.title('CPU Temperature Over Time')
#     # plt.legend()
#     plt.grid()
#     plt.savefig(os.path.join(this_dir, 'cpu_temp_over_time.png'), bbox_inches='tight')
#     plt.close()
#     break

# grouped_cpu_lat = df_cpu_lat.groupby(['m', 'n', 'k'])
# # convert to list
# grouped_cpu_lat = [(name, group) for name, group in grouped_cpu_lat]
# grouped_cpu_lat = grouped_cpu_lat[::-1]
# for name, group in grouped_cpu_lat:
#     group = group.sort_values(by='Time')
#     print(name)
#     print(group)
#     plt.figure(figsize=(10, 6))
#     m, n, k = name
#     ul_latrency = group['OurSize_Full'].values[0] * 8 / 50 * 1000
#     dl_latency = group['InSize_Full'].values[0] * 8 / 250 * 1000 / 2 * (m+n) / (m*n)
#     print(ul_latrency, dl_latency)
#     # plt.plot(group['Time'], group['Latency-ms_overhead'], label='Overhead')
#     # plt.axhline(y=ul_latrency, color='red', linestyle='--', label='Uplink Latency')
#     # plt.axhline(y=dl_latency, color='red', linestyle='-.', label='Downlink Latency')
#     plt.plot(group['Time'], group['Latency-ms_compute'], label='Compute')
#     plt.xlabel('Time (s)')
#     plt.ylabel('Latency (ms)')
#     # plt.yscale('log')
#     # plt.title('CPU Latency Over Time')
#     plt.legend()
#     plt.grid()
#     plt.savefig(os.path.join(this_dir, 'cpu_lat_over_time.png'), bbox_inches='tight')
#     plt.close()
#     break

exit()


def setup_kalman_filter(dim_x=1, dim_z=1):
    kf = KalmanFilter(dim_x=dim_x, dim_z=dim_z)
    kf.x = np.zeros((dim_x, 1))  # state
    kf.P *= 1000.0  # covariance matrix
    kf.R = 5  # measurement noise
    kf.Q = 0.1  # process noise
    kf.H = np.array([[1.0]])  # measurement function
    kf.F = np.array([[1.0]])  # state transition matrix
    return kf


def apply_kalman_filter(data):
    kf = setup_kalman_filter()
    filtered_data = []
    for measurement in data:
        kf.predict()
        kf.update(measurement)
        filtered_data.append(float(kf.x.item()))
    return np.array(filtered_data)


def read_workload_file(filename, n_std=1, use_median=True):
    timestamps = []
    cpu_temps = []
    gpu_temps = []
    battery_temps = []

    with open(filename, "r") as f:
        header = next(f).strip().split(",")
        header = [h.strip() for h in header]
        for line in f:
            parts = [p.strip() for p in line.strip().split(",")]
            if len(parts) < 4:
                continue
            try:
                time_val = int(float(parts[1]) / 1000.0)
                battery_temp = float(parts[2])
                cpu_temp = float(parts[3])
                gpu_temp = float(parts[4])
            except ValueError:
                continue

            timestamps.append(time_val)
            cpu_temps.append(cpu_temp)
            gpu_temps.append(gpu_temp)
            battery_temps.append(battery_temp)

    return (
        np.array(timestamps),
        np.array(cpu_temps),
        np.array(gpu_temps),
        np.array(battery_temps),
    )


def read_latency_file(filename):
    data = []
    with open(filename, "r") as f:
        next(f)  # skip header
        for line in f:
            parts = [p.strip() for p in line.strip().split(",")]
            if len(parts) < 3:
                continue
            size_str = parts[0]
            try:
                time_val = float(parts[1]) / 1_000_000.0  # convert to s from ns
            except ValueError:
                continue
            # latency is (6828489, 172285885) (overhead, compute)
            latency_str = parts[2].strip().strip("()")
            latency_parts = latency_str.split(",")
            try:
                # add both
                latency = sum([int(p.strip()) for p in latency_parts])
            except ValueError:
                continue
            if latency <= 0:
                continue  # skip 0 values
            data.append([time_val, latency, size_str])
    data = np.array(data, dtype=object)
    data[:, 0] = data[:, 0].astype(float)
    data[:, 1] = data[:, 1].astype(float)
    return data


def find_throttling_states(temp_data, threshold=5):
    throttling_timesteps = []
    for i in range(1, len(temp_data)):
        if temp_data[i - 1] == 0:  # ignore 0 values
            continue
        if (temp_data[i - 1] - temp_data[i]) > threshold:
            throttling_timesteps.append(i)
    return throttling_timesteps


# def plot_temperature_over_time(timestamps, cpu_temps, gpu_temps, battery_temps, workload_type):
#     if workload_type == 'CPU':
#         sensor_temp = cpu_temps
#     elif workload_type == 'GPU':
#         sensor_temp = gpu_temps
#     else:
#         print(f"Unknown workload type: {workload_type}. Using CPU temperatures by default.")
#         sensor_temp = cpu_temps

#     temperature_by_second = {}
#     for t, temp in zip(timestamps, sensor_temp):
#         sec = int(np.floor(t))
#         if sec not in temperature_by_second:
#             temperature_by_second[sec] = []
#         temperature_by_second[sec].append(temp)

#     seconds_sorted = sorted(temperature_by_second.keys())
#     box_data = [temperature_by_second[s] for s in seconds_sorted]
#     positions = seconds_sorted  # Each second is a distinct x-axis value.

#     plt.figure(figsize=(10, 6))
#     plt.boxplot(box_data, positions=positions, widths=0.8)
#     plt.xlabel('Time (s)')
#     plt.ylabel('Temperature (°C)')
#     plt.title(f'Temperature Distribution Over Time ({workload_type}) - Granular (per second)')
#     plt.savefig(f'plots/{workload_type}_temp_over_time.png')
#     plt.show()
"""
ABBOVE IS TOO SLOW BECAUSE GRANULARITY IS TOO HIGH
"""


def plot_temperature_over_time(
    timestamps, cpu_temps, gpu_temps, workload_type, threshold=5
):
    if workload_type == "CPU":
        sensor_temp = cpu_temps
    elif workload_type == "GPU":
        sensor_temp = gpu_temps
    else:
        print(
            f"Unknown workload type: {workload_type}. Using CPU temperatures by default."
        )
        sensor_temp = cpu_temps

    num_bins = 50  # number of bins for time
    time_bins = np.linspace(timestamps.min(), timestamps.max(), num_bins)

    box_data = []
    positions = []
    for i in range(len(time_bins) - 1):
        mask = (timestamps >= time_bins[i]) & (timestamps < time_bins[i + 1])
        bin_data = sensor_temp[mask]
        if bin_data.size > 0:
            box_data.append(bin_data)
            positions.append((time_bins[i] + time_bins[i + 1]) / 2)

    # hardcoded throttling state from visually looking at graph (finding jump was too finicky)
    throttling_indices = (
        [21] if workload_type == "CPU" else []
    )  # find_throttling_states([np.mean(b) for b in box_data], threshold=threshold)

    plt.figure(figsize=(10, 6))

    plt.axvline(
        x=180,
        color="black",
        linestyle="--",
        alpha=0.5,
        label="Cooling State Started",
    )
    for idx in throttling_indices:
        plt.axvline(
            x=idx,
            color="red",
            linestyle="--",
            alpha=0.5,
            label="Throttling State Started",
        )

    plt.boxplot(
        box_data,
        positions=positions,
        widths=(time_bins[1] - time_bins[0]) * 0.8,
    )
    plt.xlabel("Time (s)")
    plt.ylabel("Temperature (°C)")
    plt.xticks(positions, [f"{int(p)}" for p in positions], rotation=45)
    plt.title(f"Temperature Distribution Over Time ({workload_type})")
    plt.legend()
    plt.savefig(f"Temperature/plots/{workload_type}_temp_over_time.png")
    plt.show()


def plot_temperature_over_latency(
    timestamps, cpu_temps, gpu_temps, battery_temps, latency_data, workload_type
):
    nearest_cpu = []
    nearest_gpu = []
    nearest_battery = []
    for row in latency_data:
        t_lat = row[0]
        idx = np.argmin(np.abs(timestamps - t_lat))
        nearest_cpu.append(cpu_temps[idx])
        nearest_gpu.append(gpu_temps[idx])
        nearest_battery.append(battery_temps[idx])
    latency_vals = latency_data[:, 1].astype(float)

    plt.figure(figsize=(10, 6))
    plt.scatter(latency_vals, nearest_cpu, color="red", label="CPU Temp")
    plt.scatter(latency_vals, nearest_gpu, color="blue", label="GPU Temp")
    plt.scatter(
        latency_vals, nearest_battery, color="green", label="Battery Temp"
    )
    plt.xlabel("Latency (ms)")
    plt.ylabel("Temperature (C)")
    plt.title(f"Temperature vs Latency ({workload_type})")
    plt.legend()
    plt.savefig(f"Temperature/plots/{workload_type}_temp_vs_latency.png")
    plt.show()


def plot_latency_over_size(
    timestamps, cpu_temps, gpu_temps, battery_temps, latency_data, workload_type
):
    if workload_type == "CPU":
        temp = cpu_temps
    elif workload_type == "GPU":
        temp = gpu_temps
    else:
        print("Unknown workload type")
        return

    throttling_indices = find_throttling_states(temp, threshold=5)
    intervals = []
    start_time = timestamps[0]
    for idx in throttling_indices:
        intervals.append((start_time, timestamps[idx]))
        start_time = timestamps[idx]
    intervals.append((start_time, timestamps[-1]))

    plt.figure(figsize=(10, 6))
    colors = ["red", "blue", "green", "orange", "purple"]
    for i, (start, end) in enumerate(intervals):
        mask = (latency_data[:, 0] >= start) & (latency_data[:, 0] <= end)
        interval_data = latency_data[mask]
        if interval_data.size == 0:
            continue
        sizes = []
        for row in interval_data:
            size_str = row[2]
            size_vals = size_str.strip("()").split()
            if len(size_vals) >= 2:
                size_val = float(size_vals[1])
            else:
                size_val = float(size_vals[0])
            sizes.append(size_val)
        plt.scatter(
            sizes,
            interval_data[:, 1].astype(float),
            color=colors[i % len(colors)],
            label=f"State {i + 1} ({start:.1f}-{end:.1f}s)",
        )
    plt.xlabel("Input Size (middle value)")
    plt.ylabel("Latency (ms)")
    plt.title(f"Latency vs Input Size ({workload_type})")
    plt.legend()
    plt.savefig(f"Temperature/plots/{workload_type}_latency_vs_size.png")
    plt.show()


def main():
    timestamps_cpu, cpu_temps_cpu, gpu_temps_cpu, battery_temps_cpu = (
        read_workload_file(
            "Temperature/data/cpu_workload_temprature.txt",
            n_std=1,
            use_median=False,
        )
    )
    timestamps_gpu, cpu_temps_gpu, gpu_temps_gpu, battery_temps_gpu = (
        read_workload_file(
            "Temperature/data/gpu_workload_temprature.txt",
            n_std=1,
            use_median=False,
        )
    )

    cpu_latency = read_latency_file("Temperature/data/cpu_workload_latency.txt")
    gpu_latency = read_latency_file("Temperature/data/gpu_workload_latency.txt")

    plot_temperature_over_time(
        timestamps_cpu, cpu_temps_cpu, gpu_temps_cpu, "CPU", threshold=5
    )
    plot_temperature_over_time(
        timestamps_gpu, cpu_temps_gpu, gpu_temps_gpu, "GPU", threshold=3
    )

    # plot_temperature_over_latency(timestamps_cpu, cpu_temps_cpu, gpu_temps_cpu, battery_temps_cpu, cpu_latency, 'CPU')
    # plot_temperature_over_latency(timestamps_gpu, cpu_temps_gpu, gpu_temps_gpu, battery_temps_gpu, gpu_latency, 'GPU')

    plot_latency_over_size(
        timestamps_cpu,
        cpu_temps_cpu,
        gpu_temps_cpu,
        battery_temps_cpu,
        cpu_latency,
        "CPU",
    )
    plot_latency_over_size(
        timestamps_gpu,
        cpu_temps_gpu,
        gpu_temps_gpu,
        battery_temps_gpu,
        gpu_latency,
        "GPU",
    )


if __name__ == "__main__":
    main()
