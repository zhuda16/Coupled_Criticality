"""Critical panels from coupledcriticality_figure_tools.py, with portable inputs."""
import numpy as np
import matplotlib.pyplot as plt
from analysis_core import loglog_pdf, plot_noise_relation, clean_axis, meanfield_couple
from demo_common import DATA, panel


def load_critical():
    with np.load(DATA / 'figure2/critical_avalanches.npz') as d:
        result = {name: np.concatenate([d[f'duration_{i}_{r}'] for i in range(20)])
                  for name, r in [('duration_net_a', 0), ('duration_net_b', 1), ('duration_coupled', 2)]}
        result['size_coupled'] = np.concatenate([d[f'size_{i}_2'] for i in range(20)])
        result['a_theory'] = np.array([d[f'a_theory_{i}'][2] for i in range(20)])
        sigma = np.stack([d[f'sigma_{i}'] for i in range(50)])
    return result, sigma


def meanfield_point(sigma=None):
    if sigma is None:
        _, sigma = load_critical()
    sigma_e = np.nanmean(sigma[:, :, 0], axis=0)
    sigma_i = np.nanmean(sigma[:, :, 1], axis=0)
    lbi = float(np.linspace(1.01, 1.21, 20)[8])
    raw = meanfield_couple(sigma_e, sigma_i, LBI=lbi, GBE=2)
    if not np.isfinite(raw):
        raise ValueError('Mean-field calculation did not return a finite eigenvalue.')
    return dict(g_EI_scale=lbi, sigma_e=sigma_e.tolist(), sigma_i=sigma_i.tolist(),
                raw_max_real_eigenvalue=raw, source_plot_offset=.04,
                source_plotted_eigenvalue=raw + .04)


def build_figure(avalanche=None):
    if avalanche is None:
        avalanche, _ = load_critical()
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 3.2), constrained_layout=True)
    ax = axes[0]
    for key, color, label in [('duration_net_a', '#BB5566', 'Net A'),
                              ('duration_net_b', '#DDAA33', 'Net B'),
                              ('duration_coupled', 'k', 'Coupled Net')]:
        loglog_pdf(ax, avalanche[key], color=color, label=label, marker='', linewidth=2)
    ax.set(xlabel='Avalanche Duration', ylabel='Prob.')
    ax.set_xlim(1, 1.05 * max(np.max(avalanche[k]) for k in
                             ['duration_net_a', 'duration_net_b', 'duration_coupled']))
    ax.set_xticks([1, 10, 100]); ax.set_yticks([1e-1, 1e-3, 1e-5])
    ax.minorticks_off()
    ax.legend(frameon=False, loc='upper center', bbox_to_anchor=(.5, -.24), ncol=3)
    clean_axis(ax); panel(ax, 'D')
    ax = axes[1]
    exponent = float(np.nanmean(avalanche['a_theory']))
    plot_noise_relation(ax, avalanche['duration_coupled'], avalanche['size_coupled'],
                        line_slope=exponent, label_text=False)
    ax.text(.08, .90, rf'$\frac{{\alpha-1}}{{\tau-1}}={exponent:.2f}$',
            transform=ax.transAxes, va='top')
    ax.set_xlabel(r'Avalanche Duration $T$')
    ax.set_xlim(1, 1.05 * np.max(avalanche['duration_coupled']))
    ax.set_ylim(.8, 400); ax.set_xticks([1, 10, 100]); ax.set_yticks([1, 10, 100])
    ax.minorticks_off(); panel(ax, 'E')
    return fig
