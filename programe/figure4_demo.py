"""Original PCA/PC selection applied to characteristic avalanches before/after rewiring."""
import numpy as np
import matplotlib.pyplot as plt
from analysis_core import low_dim_presentation_2026
from demo_common import DATA, panel, representation_axis


def compute_representations():
    result = {}
    for case in ['no', 'spatial_mixture']:
        with np.load(DATA / 'figure4' / f'{case}_neuronsets.npz') as d:
            for region in range(2):
                mats = [d[f'region{region}_signal{i}'].astype(float) for i in range(4)]
                result[f'{case}_region{region}'] = low_dim_presentation_2026(mats, signal_num=4, ana_dim=3)
    return result


def build_figure(representations=None):
    representations = compute_representations() if representations is None else representations
    fig = plt.figure(figsize=(6.8, 5.6), constrained_layout=True)
    gs = fig.add_gridspec(2, 2)
    for row, case in enumerate(['no', 'spatial_mixture']):
        for col in range(2):
            ax = fig.add_subplot(gs[row, col], projection='3d')
            result = representations[f'{case}_region{col}']
            representation_axis(ax, result[0], result[1])
            ax.set_title(f"Network {'AB'[col]} — {'Before' if row == 0 else 'After'}")
            if col == 0: panel(ax, 'B' if row == 0 else 'E')
    return fig
