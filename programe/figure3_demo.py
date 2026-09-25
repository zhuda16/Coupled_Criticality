"""Reproduce the final quantitative Figure 3B--E panels."""
from __future__ import annotations

from pathlib import Path
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
from demo_common import DATA, OUTPUT


FIG3 = DATA / "figure3"
LBI_VALUES = np.linspace(1.0445, 1.13, 20)
ORDER = ("coupled_homo", "uncoupled_hete", "coupled_hete")
LABELS = ("Coupled\nHomo.", "Uncoupled\nHete.", "Coupled\nHete.")
COLORS = ("#C44E52", "#4C72B0", "#DD8452")
SIGNAL_COLORS = ("tab:blue", "tab:red", "tab:orange", "tab:green", "tab:pink", "tab:gray")

mpl.rcParams.update({
    "font.family": "Arial", "font.size": 8, "axes.labelsize": 8,
    "axes.titlesize": 9, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 7, "pdf.fonttype": 42, "ps.fonttype": 42,
    "svg.fonttype": "none", "axes.linewidth": 0.8,
    "xtick.major.width": 0.7, "ytick.major.width": 0.7,
})


def load_data():
    b = pd.read_csv(FIG3 / "figure3b_network_values.csv")
    c = pd.read_csv(FIG3 / "figure3c_projection_points.csv")
    d = pd.read_csv(FIG3 / "figure3d_summary.csv")
    e = pd.read_csv(FIG3 / "figure3e_network_curves.csv", index_col="network")
    if tuple(b["condition"].drop_duplicates()) != ORDER or len(b) != 60:
        raise ValueError("Figure 3B data must contain three conditions x 20 networks")
    if len(c) != 1200 or len(d) != 20 or e.shape != (20, 20):
        raise ValueError("Unexpected Figure 3C/D/E data dimensions")
    return b, c, d, e


def plot_b(ax, data, panel_label=True):
    groups = [data.loc[data["condition"] == condition, "accuracy"].to_numpy() for condition in ORDER]
    parts = ax.violinplot(groups, showmeans=True, showextrema=True)
    for body, color in zip(parts["bodies"], COLORS):
        body.set_facecolor(color); body.set_edgecolor("black"); body.set_alpha(0.45)
    for key in ("cmeans", "cbars", "cmaxes", "cmins"):
        parts[key].set_color("black"); parts[key].set_linewidth(0.9)
    rng = np.random.default_rng(20260919)
    for position, (values, color) in enumerate(zip(groups, COLORS), start=1):
        ax.scatter(position + rng.uniform(-0.12, 0.12, len(values)), values, s=10,
                   color=color, alpha=0.55, edgecolor="white", linewidth=0.25, zorder=3)
    ax.set_xticks((1, 2, 3), LABELS); ax.set_ylabel("Accuracy")
    ax.set_ylim(0.58, 0.88); ax.set_yticks((0.6, 0.7, 0.8))
    ax.spines[["top", "right"]].set_visible(False)
    if panel_label:
        ax.text(-0.20, 1.04, "B", transform=ax.transAxes, fontsize=13, fontweight="bold")


def _style_3d(ax, title):
    ax.set_title(title, pad=1); ax.set_xlabel(""); ax.set_ylabel(""); ax.set_zlabel("")
    ax.text2D(0.23, 0.08, "PC1", transform=ax.transAxes, fontsize=7)
    ax.text2D(0.70, 0.08, "PC2", transform=ax.transAxes, fontsize=7)
    ax.text2D(0.86, 0.50, "PC3", transform=ax.transAxes, fontsize=7)
    ax.set_xticks(np.linspace(*ax.get_xlim(), 4)); ax.set_yticks(np.linspace(*ax.get_ylim(), 4))
    ax.set_zticks(np.linspace(*ax.get_zlim(), 4))
    ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([])
    ax.tick_params(axis="both", which="major", length=0, pad=-2)
    ax.xaxis.set_pane_color((1, 1, 1, 1)); ax.yaxis.set_pane_color((1, 1, 1, 1))
    ax.zaxis.set_pane_color((1, 1, 1, 1)); ax.grid(True)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis._axinfo["grid"].update(color=(0.82, 0.82, 0.82, 0.72), linewidth=0.55)
    ax.set_box_aspect((1, 1, 0.82)); ax.view_init(elev=30, azim=-60)


def plot_c(axes, data, panel_label=True):
    for ax, condition, title in zip(axes, ("uncoupled", "coupled"), ("Uncoupled Hete.", "Coupled Hete.")):
        selected = data[data["condition"] == condition]
        for signal in range(6):
            points = selected[selected["signal"] == signal][["pc1", "pc2", "pc3"]].to_numpy()
            ax.plot(*points.T, ".", color=SIGNAL_COLORS[signal], markersize=4, alpha=0.70, rasterized=True)
        _style_3d(ax, title)
    if panel_label:
        axes[0].text2D(-0.05, 1.03, "C", transform=axes[0].transAxes, fontsize=13, fontweight="bold")


def plot_d(ax, data, panel_label=True):
    x = data["lbi"].to_numpy(); blue, red = "#377EB8", "#E41A1C"
    sil = data["silhouette_mean"].to_numpy(); sil_sd = data["silhouette_sd"].to_numpy()
    rel = data["reliability_mean"].to_numpy(); rel_sd = data["reliability_sd"].to_numpy()
    ax.plot(x, sil, color=blue, linewidth=1.5)
    ax.fill_between(x, sil - sil_sd, sil + sil_sd, color=blue, alpha=0.18, linewidth=0)
    ax.set(xlabel=r"$g_{EI}^{scale}$", ylabel="Sil. coefficient", xlim=(1.044, 1.131), ylim=(-0.12, 0.62))
    ax.set_xticks((1.06, 1.08, 1.10, 1.12)); ax.set_yticks((-0.10, 0, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60))
    ax.tick_params(axis="y", colors=blue); ax.yaxis.label.set_color(blue); ax.spines["top"].set_visible(False)
    twin = ax.twinx(); twin.plot(x, rel, color=red, linewidth=1.5)
    twin.fill_between(x, rel - rel_sd, rel + rel_sd, color=red, alpha=0.16, linewidth=0)
    twin.set(ylabel="Ava. reliability", ylim=(-0.005, 0.24)); twin.set_yticks((0, 0.05, 0.10, 0.15, 0.20))
    twin.tick_params(axis="y", colors=red); twin.tick_params(axis="x", bottom=False, labelbottom=False)
    twin.spines["top"].set_visible(False); ax.axvline(LBI_VALUES[11], color="0.25", linestyle="-.", linewidth=0.9)
    if panel_label:
        ax.text(-0.18, 1.04, "D", transform=ax.transAxes, fontsize=13, fontweight="bold")
    return twin


def plot_e(ax, data, panel_label=True):
    curves = data.to_numpy(); mean = curves.mean(axis=0); sd = curves.std(axis=0, ddof=0)
    ax.fill_between(LBI_VALUES, mean - sd, mean + sd, color="0.75", alpha=0.45, linewidth=0)
    ax.plot(LBI_VALUES, mean, color="black", linewidth=1.8)
    ax.axvline(LBI_VALUES[11], color="black", linestyle="-.", linewidth=0.9)
    ax.set(xlabel=r"$g_{EI}^{scale}$", ylabel="Accuracy", xlim=(1.044, 1.131), ylim=(0.56, 0.88))
    ax.set_xticks((1.06, 1.08, 1.10, 1.12)); ax.set_yticks((0.6, 0.7, 0.8))
    ax.spines[["top", "right"]].set_visible(False)
    if panel_label:
        ax.text(-0.18, 1.04, "E", transform=ax.transAxes, fontsize=13, fontweight="bold")


def build_figure():
    """Build a compact layout containing only the quantitative panels B--E."""
    b, c, d, e = load_data()
    fig = plt.figure(figsize=(7.5, 5.0))
    outer = fig.add_gridspec(2, 1, height_ratios=(1.08, 1.0), hspace=0.38)
    top = outer[0].subgridspec(1, 3, width_ratios=(0.88, 1.08, 1.08), wspace=0.02)
    bottom = outer[1].subgridspec(1, 2, width_ratios=(1.08, 1.0), wspace=0.62)

    plot_b(fig.add_subplot(top[0, 0]), b)
    plot_c((fig.add_subplot(top[0, 1], projection="3d"),
            fig.add_subplot(top[0, 2], projection="3d")), c)
    plot_d(fig.add_subplot(bottom[0, 0]), d)
    plot_e(fig.add_subplot(bottom[0, 1]), e)
    fig.subplots_adjust(left=0.09, right=0.94, bottom=0.11, top=0.95)
    return fig


def main():
    OUTPUT.mkdir(exist_ok=True)
    fig = build_figure()
    for ext in ("png", "pdf", "svg"):
        fig.savefig(OUTPUT / f"Figure3.{ext}", dpi=500)
    plt.close(fig)


if __name__ == "__main__":
    main()
