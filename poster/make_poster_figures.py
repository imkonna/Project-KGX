from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm


OUT = Path(__file__).parent / "assets"

TEAL = "#1F7A8C"
TEAL_M = "#5AA2B2"
TEAL_L = "#C5E0E5"
TEAL_XL = "#E8F4F6"
ORANGE = "#E9633B"
ORANGE_M = "#EC8862"
ORANGE_L = "#F5C9B8"
ORANGE_XL = "#FDF0EA"
PANEL = "#FFFFFF"
BG = "#F5F2EC"
INK = "#1E2D36"
SOFT = "#46555E"
LINE = "#DAD2C2"


def rc(size=18):
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": size,
            "axes.titlesize": size + 2,
            "axes.labelsize": size,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "figure.facecolor": PANEL,
            "axes.facecolor": BG,
            "axes.edgecolor": LINE,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": LINE,
            "grid.linewidth": 0.65,
            "savefig.dpi": 270,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.08,
            "text.color": INK,
            "axes.labelcolor": INK,
            "xtick.color": SOFT,
            "ytick.color": INK,
        }
    )


def save(fig, name):
    fig.savefig(OUT / name, facecolor=PANEL)
    plt.close(fig)
    print(name)


def main_task_delta():
    rc(18)
    rows = [
        ("DILI\nn=475, 45% pos.", 0.043, "GIN 0.870", "RF 0.827"),
        ("ClinTox\nn=1484, 8% pos.", 0.020, "GIN 0.820", "XGB 0.800"),
        ("AMES\nn=7255, 50% pos.", -0.012, "GIN 0.780", "RF 0.792"),
        ("hERG\nn=13445, 40% pos.", -0.018, "GINE 0.780", "RF 0.798"),
        ("LD50 R²\nn=7385, regr.", -0.039, "SAGE -0.009", "RF 0.030"),
    ]
    labels = [r[0] for r in rows]
    gaps = np.array([r[1] for r in rows])
    gnn = [r[2] for r in rows]
    classical = [r[3] for r in rows]
    y = np.arange(len(rows))
    colors = [TEAL if v > 0 else ORANGE for v in gaps]

    fig, ax = plt.subplots(figsize=(13.5, 8.4))
    ax.barh(y, gaps, color=colors, height=0.66, edgecolor=PANEL, linewidth=1.6)
    ax.axvline(0, color=INK, linewidth=1.8)
    ax.axhline(1.5, color=SOFT, linewidth=1.4, linestyle="--", alpha=0.45)

    for i, value in enumerate(gaps):
        if value > 0:
            txt = f"+{value:.3f}   {gnn[i]}"
            ax.text(value + 0.003, i, txt, va="center", ha="left", fontsize=14, color=TEAL, fontweight="bold")
        else:
            txt = f"{value:.3f}   {classical[i]}"
            ax.text(value - 0.003, i, txt, va="center", ha="right", fontsize=14, color=ORANGE, fontweight="bold")

    ax.text(0.24, 1.07, "Classical wins", transform=ax.transAxes, ha="center", fontsize=16, color=ORANGE, fontweight="bold")
    ax.text(0.80, 1.07, "GNN wins", transform=ax.transAxes, ha="center", fontsize=16, color=TEAL, fontweight="bold")
    ax.annotate("", xy=(0.42, 1.04), xytext=(0.06, 1.04), xycoords="axes fraction", arrowprops=dict(arrowstyle="<-", color=ORANGE, lw=1.8))
    ax.annotate("", xy=(0.96, 1.04), xytext=(0.60, 1.04), xycoords="axes fraction", arrowprops=dict(arrowstyle="->", color=TEAL, lw=1.8))

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=14)
    ax.set_xlabel("Graph-model lead over best fingerprint baseline   (AUROC; R² for LD50)", fontsize=14, color=SOFT)
    ax.set_xlim(-0.106, 0.112)
    ax.set_ylim(-0.55, len(rows) - 0.25)
    ax.invert_yaxis()
    ax.tick_params(left=False)
    ax.spines["left"].set_visible(False)
    ax.grid(axis="x")
    fig.subplots_adjust(left=0.22, right=0.97, top=0.82, bottom=0.18)
    save(fig, "main_task_delta_v03.png")


def benchmark_heatmap():
    rc(18)
    methods = ["RF", "XGBoost", "ChemBERTa", "GCN", "GIN", "GAT", "GraphSAGE", "GINE"]
    datasets = ["AMES", "ClinTox", "DILI", "hERG", "LD50 R²"]
    data = np.array(
        [
            [0.792, 0.733, 0.827, 0.798, 0.030],
            [0.757, 0.800, 0.815, 0.794, -0.049],
            [0.708, 0.625, 0.701, 0.688, np.nan],
            [0.718, 0.691, 0.845, 0.692, -0.073],
            [0.780, 0.820, 0.870, 0.778, -0.523],
            [0.749, 0.745, 0.836, 0.703, -0.020],
            [0.774, 0.815, 0.840, 0.716, -0.009],
            [0.758, 0.810, 0.860, 0.780, -0.755],
        ]
    )
    best = [int(np.nanargmax(data[:, j])) for j in range(data.shape[1])]
    cmap = LinearSegmentedColormap.from_list("poster_heat", [ORANGE, ORANGE_L, PANEL, TEAL_L, TEAL], N=256)
    ncls = TwoSlopeNorm(vmin=0.60, vcenter=0.76, vmax=0.88)
    nreg = TwoSlopeNorm(vmin=-0.80, vcenter=-0.05, vmax=0.08)

    fig, ax = plt.subplots(figsize=(13.5, 6.6))
    ax.set_facecolor(PANEL)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            value = data[i, j]
            if np.isnan(value):
                face = LINE
                text = "NA"
                color = SOFT
            else:
                normed = float((ncls if j < 4 else nreg)(value))
                face = cmap(np.clip(normed, 0, 1))
                luminance = 0.299 * face[0] + 0.587 * face[1] + 0.114 * face[2]
                color = "white" if luminance < 0.48 else INK
                text = f"{value:.3f}"
            ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor=face, edgecolor=PANEL, linewidth=1.5))
            weight = "bold" if i == best[j] and not np.isnan(value) else "normal"
            ax.text(j, i, text, ha="center", va="center", fontsize=16.5, color=color, fontweight=weight)

    ax.plot([-0.5, len(datasets) - 0.5], [2.5, 2.5], color=SOFT, linewidth=1.5, linestyle="--")
    ax.plot([3.5, 3.5], [-0.5, len(methods) - 0.5], color=SOFT, linewidth=1.1, linestyle=":")
    ax.set_xlim(-0.5, len(datasets) - 0.5)
    ax.set_ylim(-0.5, len(methods) - 0.5)
    ax.set_xticks(range(len(datasets)))
    ax.set_xticklabels(datasets, fontsize=18)
    ax.xaxis.tick_top()
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels(methods, fontsize=18)
    ax.text(-1.15, 1.0, "Classical", va="center", ha="right", fontsize=16.5, color=ORANGE, fontweight="bold")
    ax.text(-1.15, 5.5, "GNN", va="center", ha="right", fontsize=16.5, color=TEAL, fontweight="bold")
    ax.plot([-0.45, 3.45], [-1.18, -1.18], color=SOFT, lw=1.2, clip_on=False)
    ax.plot([3.55, 4.45], [-1.18, -1.18], color=SOFT, lw=1.2, clip_on=False)
    ax.text(1.5, -1.34, "Classification (AUROC)", ha="center", va="bottom", fontsize=14.5, color=SOFT, fontweight="bold", clip_on=False)
    ax.text(4.0, -1.34, "Regression (R²)", ha="center", va="bottom", fontsize=14.5, color=SOFT, fontweight="bold", clip_on=False)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(False)
    ax.invert_yaxis()
    save(fig, "benchmark_heatmap_v03.png")


def ld50_split():
    """Dumbbell chart: each model's R2 drops from random split to scaffold split."""
    rc(18)
    methods = ["RF", "XGB", "GCN", "GIN", "GAT", "SAGE", "GINE"]
    random_split = np.array([0.58, 0.60, 0.34, 0.49, 0.40, 0.42, 0.44])
    scaffold = np.array([0.030, -0.049, -0.073, -0.523, -0.020, -0.009, -0.755])
    order = np.argsort(scaffold)[::-1]
    methods = [methods[i] for i in order]
    random_split = random_split[order]
    scaffold = scaffold[order]
    y = np.arange(len(methods))

    fig, ax = plt.subplots(figsize=(9.6, 6.2))
    ax.axvspan(-1.05, 0, color=ORANGE, alpha=0.06, zorder=0)
    ax.axvline(0, color=INK, linewidth=1.4, zorder=2)
    for i in range(len(methods)):
        ax.plot([scaffold[i], random_split[i]], [y[i], y[i]], color="#CBC1AB", lw=3.2,
                solid_capstyle="round", zorder=2)
    ax.scatter(random_split, y, s=240, color=TEAL_M, edgecolor=TEAL, linewidth=1.6, zorder=4, label="Random split")
    ax.scatter(scaffold, y, s=240, color=ORANGE_M, edgecolor=ORANGE, linewidth=1.6, zorder=4, label="Scaffold split")
    ax.set_yticks(y)
    ax.set_yticklabels(methods, fontsize=15)
    ax.invert_yaxis()
    ax.set_xlabel("R²  on LD50 regression", fontsize=16)
    ax.set_xlim(-1.05, 0.78)
    ax.set_ylim(len(methods) - 0.4, -0.7)
    ax.legend(frameon=False, fontsize=14, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.10))
    ax.grid(axis="x")
    ax.text(-0.62, 1.25, "below 0 = worse than\npredicting the mean",
            fontsize=12.5, color=ORANGE, ha="center", va="center", style="italic")
    save(fig, "ld50_split_leakage_v02.png")


def schnet_vs_gine():
    rc(18)
    datasets = ["ClinTox", "AMES"]
    schnet = np.array([0.711, 0.756])
    gine = np.array([0.810, 0.773])
    schnet_err = np.array([0.0, 0.006])
    gine_err = np.array([0.028, 0.008])
    x = np.arange(len(datasets))
    width = 0.34
    fig, ax = plt.subplots(figsize=(9.6, 6.2))
    ax.bar(x - width / 2, schnet, width, color=ORANGE_M, edgecolor=ORANGE, linewidth=1.25, yerr=schnet_err, capsize=6, label="SchNet 3D")
    ax.bar(x + width / 2, gine, width, color=TEAL_M, edgecolor=TEAL, linewidth=1.25, yerr=gine_err, capsize=6, label="GINE 2D+bonds")
    for i, value in enumerate(schnet):
        ax.text(i - width / 2, value + schnet_err[i] + 0.015, f"{value:.3f}", ha="center", fontsize=16, color=ORANGE, fontweight="bold")
    for i, value in enumerate(gine):
        ax.text(i + width / 2, value + gine_err[i] + 0.015, f"{value:.3f}", ha="center", fontsize=16, color=TEAL, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, fontsize=16)
    ax.set_ylabel("AUROC", fontsize=16)
    ax.set_ylim(0.63, 0.89)
    ax.legend(frameon=False, fontsize=14, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.14))
    ax.grid(axis="y")
    save(fig, "schnet_vs_gine_v03.png")


def tox21_transfer():
    rc(18)
    endpoints = ["AR", "AR-LBD", "AhR", "Arom.", "ER", "ER-LBD", "PPARg", "ARE", "ATAD5", "HSE", "MMP", "p53"]
    delta = np.array([-0.037, 0.082, 0.001, -0.031, 0.007, -0.028, -0.069, -0.020, 0.039, 0.007, -0.002, -0.044])
    order = np.argsort(delta)
    endpoints = [endpoints[i] for i in order]
    delta = delta[order]
    colors = [TEAL if value >= 0 else ORANGE for value in delta]
    y = np.arange(len(delta))
    fig, ax = plt.subplots(figsize=(9.6, 6.2))
    ax.barh(y, delta, color=colors, edgecolor=PANEL, linewidth=1.0)
    ax.axvline(0, color=INK, linewidth=1.35)
    ax.axvline(-0.024, color=SOFT, linewidth=1.2, linestyle="--")
    ax.text(-0.024, len(delta) - 0.1, "macro avg", ha="right", va="bottom", fontsize=13, color=SOFT)
    ax.set_yticks(y)
    ax.set_yticklabels(endpoints, fontsize=15)
    ax.set_xlabel("Change in AUROC from multitask sharing  (positive = gain)", fontsize=15)
    ax.set_xlim(-0.085, 0.095)
    ax.grid(axis="x")
    ax.text(0.052, len(delta) - 1.0, "multitask gains", color=TEAL, fontsize=14, fontweight="bold", ha="center")
    ax.text(-0.055, 1.8, "negative\ntransfer", color=ORANGE, fontsize=13, ha="center", alpha=0.85)
    save(fig, "tox21_transfer_v03.png")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    main_task_delta()
    benchmark_heatmap()
    ld50_split()
    schnet_vs_gine()
    tox21_transfer()
