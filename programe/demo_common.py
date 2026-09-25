"""Portable paths and manuscript plot styling; no research-tree dependency."""
from pathlib import Path

import matplotlib as mpl

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'
OUTPUT = ROOT / 'output'
mpl.rcParams.update({'font.family': 'Arial', 'font.size': 8, 'axes.labelsize': 8,
                     'axes.titlesize': 9, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
                     'legend.fontsize': 7, 'pdf.fonttype': 42, 'svg.fonttype': 'none',
                     'axes.linewidth': .8})


def save_figure(fig, name):
    OUTPUT.mkdir(exist_ok=True)
    for ext in ('png', 'pdf'):
        fig.savefig(OUTPUT / f'{name}.{ext}', dpi=300)


def panel(ax, label):
    text = ax.text2D if hasattr(ax, 'text2D') else ax.text
    text(0, .98 if hasattr(ax, 'text2D') else 1.02, label,
         transform=ax.transAxes, weight='bold', fontsize=11)


def representation_axis(ax, data, colors, dims=(0, 1, 2)):
    from analysis_core import plot_3d_points
    plot_3d_points(ax, data, colors, dims=dims, s=4, alpha=.7)
    # Put the z label within its own panel to avoid neighboring axes.
    ax.set_xlabel(''); ax.set_ylabel(''); ax.set_zlabel('')
    ax.text2D(.23, .08, 'PC1', transform=ax.transAxes, fontsize=7)
    ax.text2D(.70, .08, 'PC2', transform=ax.transAxes, fontsize=7)
    ax.text2D(.86, .50, 'PC3', transform=ax.transAxes, fontsize=7)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
