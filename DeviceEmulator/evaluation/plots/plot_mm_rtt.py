import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.style.use("classic")
matplotlib.rcParams.update({"font.size": 28, "lines.linewidth": 3})
matplotlib.rcParams["axes.linewidth"] = 2
matplotlib.rcParams["axes.edgecolor"] = "black"

# set scatter plot marker edge color to black
matplotlib.rcParams["scatter.edgecolors"] = "none"

this_dir = os.path.join(os.path.dirname(__file__))

s24_gemm_df = pd.read_csv(os.path.join(this_dir, "gemm_data_buffer.csv"))
s24_gemv_df = pd.read_csv(os.path.join(this_dir, "gemv_data_buffer.csv"))

iphone_gemm_df = pd.read_csv(os.path.join(this_dir, "iphone_gemm_data.csv"))
iphone_gemv_df = pd.read_csv(os.path.join(this_dir, "iphone_gemv_data.csv"))

iphone_gemm_df[["CPU_avg_s", "GPU_avg_s"]] = (
    iphone_gemm_df[["CPU_avg_s", "GPU_avg_s"]] * 1000
)
iphone_gemv_df[["CPU_time_s", "GPU_time_s"]] = (
    iphone_gemv_df[["CPU_time_s", "GPU_time_s"]] * 1000
)

s24_gemm_df["DL_size"] = (
    (s24_gemm_df["m"] * s24_gemm_df["k"] + s24_gemm_df["k"] * s24_gemm_df["n"])
    * 2
    / 1024
    / 1024
)
s24_gemv_df["DL_size"] = (
    (s24_gemv_df["m"] * s24_gemv_df["k"] + s24_gemv_df["k"] * s24_gemv_df["n"])
    * 2
    / 1024
    / 1024
)

s24_gemm_df["UL_size"] = (s24_gemm_df["m"] * s24_gemm_df["n"]) * 2 / 1024 / 1024
s24_gemv_df["UL_size"] = (s24_gemv_df["m"] * s24_gemv_df["n"]) * 2 / 1024 / 1024

iphone_gemm_df["DL_size"] = (
    (
        iphone_gemm_df["M"] * iphone_gemm_df["K"]
        + iphone_gemm_df["K"] * iphone_gemm_df["N"]
    )
    * 2
    / 1024
    / 1024
)
iphone_gemv_df["DL_size"] = (
    (
        iphone_gemv_df["M"] * iphone_gemv_df["K"]
        + iphone_gemv_df["K"] * iphone_gemv_df["N"]
    )
    * 2
    / 1024
    / 1024
)

iphone_gemm_df["UL_size"] = (
    (iphone_gemm_df["M"] * iphone_gemm_df["N"]) * 2 / 1024 / 1024
)
iphone_gemv_df["UL_size"] = (
    (iphone_gemv_df["M"] * iphone_gemv_df["N"]) * 2 / 1024 / 1024
)


print(s24_gemm_df)
print(s24_gemv_df)
print(iphone_gemm_df)
print(iphone_gemv_df)

plt.figure(figsize=(9, 8))
plt.scatter(
    s24_gemm_df["DL_size"] / 250 * 8 * 1000 / 2,
    s24_gemm_df["CPU Comp Time"] + s24_gemm_df["UL_size"] / 50 * 8 * 1000,
    alpha=0.5,
    s=100,
    label="Android CPU",
    color="red",
    edgecolors="none",
    marker="^",
)
plt.scatter(
    s24_gemm_df["DL_size"] / 250 * 8 * 1000 / 2,
    s24_gemm_df["GPU Comp Time"] + s24_gemm_df["UL_size"] / 50 * 8 * 1000,
    alpha=0.5,
    s=100,
    label="Android GPU",
    color="blue",
    edgecolors="none",
    marker="s",
)
plt.scatter(
    iphone_gemm_df["DL_size"] / 250 * 8 * 1000 / 2,
    iphone_gemm_df["CPU_avg_s"] + iphone_gemm_df["UL_size"] / 50 * 8 * 1000,
    alpha=0.5,
    s=100,
    label="iPhone CPU",
    color="green",
    edgecolors="none",
    marker="D",
)
plt.scatter(
    iphone_gemm_df["DL_size"] / 250 * 8 * 1000 / 2,
    iphone_gemm_df["GPU_avg_s"] + iphone_gemm_df["UL_size"] / 50 * 8 * 1000,
    alpha=0.5,
    s=100,
    label="iPhone GPU",
    color="orange",
    edgecolors="none",
    marker="o",
)

# draw y=x line
x = np.linspace(0, 1200, 100)
y = x
plt.plot(x, y, color="black", linestyle="--")

plt.xlabel("DL Time (ms)")
plt.ylabel("Exec + UL Time (ms)")
plt.xlim(0.1, 1200)
plt.ylim(0.1, 1200)
plt.xscale("log")
plt.yscale("log")
# plt.legend(ncols=2, loc='upper left', fontsize=26, markerscale=1.2, bbox_to_anchor=(-0.15, 1.3))
plt.grid()
plt.tight_layout()
plt.savefig(os.path.join(this_dir, "gemm_rtt.pdf"), bbox_inches="tight")
plt.close()


plt.figure(figsize=(9, 8))
plt.scatter(
    s24_gemv_df["DL_size"] / 200 * 8 * 1000 / 2,
    s24_gemv_df["CPU Comp Time"] + s24_gemv_df["UL_size"] / 50 * 8 * 1000,
    alpha=0.5,
    s=100,
    label="Android CPU",
    color="red",
    edgecolors="none",
    marker="^",
)
plt.scatter(
    s24_gemv_df["DL_size"] / 200 * 8 * 1000 / 2,
    s24_gemv_df["GPU Comp Time"] + s24_gemv_df["UL_size"] / 50 * 8 * 1000,
    alpha=0.5,
    s=100,
    label="Android GPU",
    color="blue",
    edgecolors="none",
    marker="s",
)
plt.scatter(
    iphone_gemv_df["DL_size"] / 200 * 8 * 1000 / 2,
    iphone_gemv_df["CPU_time_s"] + iphone_gemv_df["UL_size"] / 50 * 8 * 1000,
    alpha=0.5,
    s=100,
    label="iPhone CPU",
    color="green",
    edgecolors="none",
    marker="D",
)
plt.scatter(
    iphone_gemv_df["DL_size"] / 200 * 8 * 1000 / 2,
    iphone_gemv_df["GPU_time_s"] + iphone_gemv_df["UL_size"] / 50 * 8 * 1000,
    alpha=0.5,
    s=100,
    label="iPhone GPU",
    color="orange",
    edgecolors="none",
    marker="o",
)

# draw y=x line
x = np.linspace(0, 1200, 100)
y = x
plt.plot(x, y, color="black", linestyle="--")

plt.xlabel("DL Time (ms)")
plt.ylabel("Exec + UL Time (ms)")
plt.xlim(0.1, 1200)
plt.ylim(0.1, 1200)
plt.xscale("log")
plt.yscale("log")
# plt.legend(ncols=2, loc='upper left', fontsize=26, markerscale=1.2, bbox_to_anchor=(-0.15, 1.3))
plt.grid()
plt.tight_layout()
plt.savefig(os.path.join(this_dir, "gemv_rtt.pdf"), bbox_inches="tight")
plt.close()


def export_legend(legend, filename="plots/legend.pdf", expand=[-5, -5, 5, 5]):
    fig = legend.figure
    fig.canvas.draw()
    bbox = legend.get_window_extent()
    bbox = bbox.from_extents(*(bbox.extents + np.array(expand)))
    bbox = bbox.transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(filename, dpi="figure", bbox_inches=bbox)


# export legend as pdf
fig = plt.figure(figsize=(9, 9))
plt.scatter(
    [0],
    [0],
    alpha=0.5,
    s=100,
    label="Android CPU",
    color="red",
    edgecolors="none",
    marker="s",
)
plt.scatter(
    [0],
    [0],
    alpha=0.5,
    s=100,
    label="Android GPU",
    color="blue",
    edgecolors="none",
    marker="D",
)
plt.scatter(
    [0],
    [0],
    alpha=0.5,
    s=100,
    label="iPhone CPU",
    color="green",
    edgecolors="none",
    marker="s",
)
plt.scatter(
    [0],
    [0],
    alpha=0.5,
    s=100,
    label="iPhone GPU",
    color="orange",
    edgecolors="none",
    marker="D",
)

legend = plt.legend(
    bbox_to_anchor=[100, 0, 1, 1], loc="upper right", ncol=4, fontsize=26
)

plt.axis("off")
export_legend(
    legend,
    filename=os.path.join(this_dir, "gemm_rtt_legend.pdf"),
    expand=[-5, -5, 5, 5],
)
plt.close()
