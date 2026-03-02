import os
import re

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.style.use("classic")
matplotlib.rcParams.update({"font.size": 28, "lines.linewidth": 3})
matplotlib.rcParams["axes.linewidth"] = 2
matplotlib.rcParams["axes.edgecolor"] = "black"

this_dir = os.path.join(os.path.dirname(__file__))

byte_size = 4
df = pd.read_csv(os.path.join(this_dir, "mergedClientServerLogs.csv"))
df[["rowReadMicrosec", "colReadMicrosec"]] = (
    df[["rowReadMicrosec", "colReadMicrosec"]] / 1e3
)  # convert to ms

df = df[["rowReadMicrosec", "colReadMicrosec", "m", "n"]]


# if m > n, then swap values
df["m"], df["n"] = np.where(
    df["m"] > df["n"], (df["n"], df["m"]), (df["m"], df["n"])
)

# group by m, n and take the mean
df = df.groupby(["m", "n"]).mean().reset_index()
df["size"] = df["m"] * df["n"] * byte_size / (1024 * 1024)

# sort by size
df = df.sort_values(by="size")

# group by size and take the mean
df = df.groupby("size").mean().reset_index()

print(df)

sizes = df["size"].values

# each rowReadMicrosec and colReadMicrosec needs to scale by n/m
df["rowReadMicrosec"] = df["rowReadMicrosec"] * df["n"] / df["m"]
df["colReadMicrosec"] = df["colReadMicrosec"] * df["n"] / df["m"]

print(df)

# RTT in ms
local_RTT = 0.3 / 2
edge_RTT = 25 / 2
cloud_RTT = 80 / 2
cross_RTT = 332 / 2

plt.figure(figsize=(9, 6))

# draw the lines
plt.axhline(y=local_RTT, color="k", linestyle="--", linewidth=2)
plt.axhline(y=edge_RTT, color="k", linestyle="--", linewidth=2)
plt.axhline(y=cloud_RTT, color="k", linestyle="--", linewidth=2)
plt.axhline(y=cross_RTT, color="k", linestyle="--", linewidth=2)

# annotate text at the end of the lines
x_offset = 3e1  # 2e-4
plt.text(
    s="LAN",
    x=x_offset,
    y=local_RTT + 0.1,
    fontsize=24,
    color="k",
    fontdict={"weight": "bold"},
)
plt.text(
    s="CDN",
    x=x_offset,
    y=edge_RTT + 0.2,
    fontsize=24,
    color="k",
    fontdict={"weight": "bold"},
)
plt.text(
    s="Cloud",
    x=x_offset,
    y=cloud_RTT + 0.2,
    fontsize=24,
    color="k",
    fontdict={"weight": "bold"},
)
plt.text(
    s="Continent",
    x=x_offset,
    y=cross_RTT + 0.3,
    fontsize=24,
    color="k",
    fontdict={"weight": "bold"},
)

# apply color palette according to sizes for each marker
colors = plt.cm.plasma(np.linspace(0, 1, len(sizes)))

# scatter plot


plt.scatter(
    sizes,
    df["rowReadMicrosec"],
    color="b",
    marker="x",
    s=120,
    alpha=0.7,
    label="Row-Layout",
)
plt.scatter(
    sizes,
    df["colReadMicrosec"],
    color="b",
    marker="s",
    s=120,
    alpha=0.7,
    label="Col-Layout",
)


plt.xlabel("Volume (MB)")
plt.ylabel("Latency (ms)")
plt.yscale("log")
plt.xscale("log")
plt.legend(ncol=2, loc="upper left", bbox_to_anchor=(-0.05, 1.2), fontsize=26)
plt.grid(True)

plt.savefig(os.path.join(this_dir, "rtt_scatter_plot.pdf"), bbox_inches="tight")
plt.savefig(os.path.join(this_dir, "rtt_scatter_plot.png"), bbox_inches="tight")

exit()


def load_and_preprocess(csv_path):
    """Load CSV, compute timestamps, corrections, and return the DataFrame."""
    df = pd.read_csv(csv_path)
    df["T1_time"] = df["T1_sec"] + df["T1_usec"] / 1e6
    df["T2_time"] = df["t2Sec"] + df["t2Usec"] / 1e6
    df["T3_time"] = df["t3Sec"] + df["t3Usec"] / 1e6
    df["T4_time"] = df["T4_sec"] + df["T4_usec"] / 1e6
    df["downlink_corr"] = df["T2_time"] - df["T1_time"]
    df["uplink_corr"] = df["T4_time"] - df["T3_time"]
    df["RTT"] = df["T4_time"] - df["T1_time"]
    df["matrix_size_MB"] = (df["m"] * df["n"] * 2) / (1024.0 * 1024.0)
    df["rowRTT"] = df["RTT"] + df["rowTime_s"]
    df["colRTT"] = df["RTT"] + df["colTime_s"]
    df["uplink_plus_row"] = df["uplink_corr"] + df["rowTime_s"]
    df["uplink_plus_col"] = df["uplink_corr"] + df["colTime_s"]
    return df


def remove_outliers(df, col):
    """Remove outliers from the DataFrame, < 5% and > 95%."""
    Q1 = df[col].quantile(0.05)
    Q3 = df[col].quantile(0.95)
    mask = (df[col] > Q1) & (df[col] < Q3)
    return df[mask]


def plot_all_metrics_together(df, out_dir):
    """Plot downlink RTT, uplink RTT, rowRTT, and colRTT together."""
    plt.figure(figsize=(9, 6))

    df_dl = remove_outliers(df, "downlink_corr")
    df_ul = remove_outliers(df, "uplink_corr")
    df_row = remove_outliers(df, "rowRTT")
    df_col = remove_outliers(df, "colRTT")

    plt.scatter(
        df_dl["matrix_size_MB"],
        df_dl["downlink_corr"],
        color="#3366ff",
        marker="x",
        s=150,
        alpha=0.5,
        label="DL Recv",
    )
    plt.scatter(
        df_ul["matrix_size_MB"],
        df_ul["uplink_corr"],
        color="#3366ff",
        marker="+",
        s=150,
        alpha=0.5,
        label="UL Send",
    )
    plt.scatter(
        df_row["matrix_size_MB"],
        df_row["rowRTT"],
        color="#ff3333",
        marker="^",
        s=150,
        alpha=0.5,
        label="Row-Read",
    )
    plt.scatter(
        df_col["matrix_size_MB"],
        df_col["colRTT"],
        color="#ff3333",
        marker="s",
        s=150,
        alpha=0.5,
        label="Col-Read",
    )
    plt.xlabel("Volume (MB)")
    plt.ylabel("Latency (ms)")
    plt.grid(True)
    plt.xscale("log")
    plt.yscale("log")
    plt.legend(
        loc="upper center", bbox_to_anchor=(0.45, 1.3), ncol=2, fontsize=26
    )

    plt.savefig(
        os.path.join(out_dir, "rtt_scatter_plot.pdf"), bbox_inches="tight"
    )


cols = ["downlink_corr", "uplink_corr", "rowRTT", "colRTT"]
df = load_and_preprocess(os.path.join(this_dir, "mergedLogs.csv"))
df[cols] = df[cols] * 1000

df["downlink_corr"] = df["downlink_corr"] * 15 * 8 / 250
df["uplink_corr"] = df["uplink_corr"] * 15 * 8 / 50

# df = df.sample(frac=0.2, random_state=42)

plot_all_metrics_together(df, this_dir)

exit()

# file_path = os.path.join(this_dir, 'iphone15_single_native.txt')

# matrix_sizes_mb = []
# row_reads = []
# col_reads = []

# with open(file_path, 'r') as f:
#     lines = f.readlines()

# i = 0
# while i < len(lines):
#     line = lines[i].strip()
#     shape_match = re.search(r'\[LOG\] DONE shape A=(\d+)x(\d+), B=(\d+)x(\d+)', line)
#     if shape_match:
#         A_rows = int(shape_match.group(1))
#         A_cols = int(shape_match.group(2))
#         B_rows = int(shape_match.group(3))
#         B_cols = int(shape_match.group(4))
#         total_bytes = (A_rows * A_cols + B_rows * B_cols) * 4
#         total_mb = total_bytes / (1024 * 1024)


#         while i < len(lines) and "Accelerate:" not in lines[i]:
#             i += 1
#         if i < len(lines) and "Accelerate:" in lines[i]:
#             if i + 2 < len(lines):
#                 row_line = lines[i+1].strip()
#                 col_line = lines[i+2].strip()
#                 row_match = re.search(r'row-read=([\d.eE+-]+)', row_line)
#                 col_match = re.search(r'col-read=([\d.eE+-]+)', col_line)
#                 if row_match and col_match:
#                     row_read_val = float(row_match.group(1))
#                     col_read_val = float(col_match.group(1))
#                     matrix_sizes_mb.append(total_mb)
#                     row_reads.append(row_read_val)
#                     col_reads.append(col_read_val)
#             i += 3
#             continue
#     i += 1

# row_reads = np.array(row_reads) * 1000
# col_reads = np.array(col_reads) * 1000

# plt.figure(figsize=(9, 6))
# plt.scatter(matrix_sizes_mb, col_reads, color='#3366ff', label='Col-Read', marker='x', s=150, alpha=0.5)
# plt.scatter(matrix_sizes_mb, row_reads, color='#3366ff', label='Row-Read', marker='^', s=150, alpha=0.5)
# plt.xlabel('Communication Volume (MB)')
# plt.ylabel('Communication Time (ms)')
# plt.yscale('log')
# plt.xscale('log')
# plt.legend(ncol=2, loc='upper left', bbox_to_anchor=(-0.05, 1.2), fontsize=26)
# plt.grid(True)


# output_path = os.path.join(this_dir, 'rtt_scatter_plot.pdf')
# plt.savefig(output_path, bbox_inches='tight')
