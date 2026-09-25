"""Figure 1 adapter: original sampling, fitting and plotting functions.
Only input loading and output paths changed from figure1_keyresult_local_rebuild.py.
"""
import contextlib
import csv
import io
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FixedLocator, LogFormatterMathtext, NullFormatter
from demo_common import DATA, OUTPUT, save_figure
COLORS = {'local': '#111111', 'duration': '#111111', 'size': '#111111',
          'sub': '#DDAA33', 'sup': '#BB5566', 'global': '#242424'}

def _draw_original_schematic(ax, key):
    _hide_schematic_axis(ax)
    ax.imshow(plt.imread(DATA / 'figure1' / f'schematic_{key}.png'))

def load_events():
    return {name: dict(np.load(DATA / 'figure1' / f'{name}_avalanches.npz'))
            for name in ('local', 'sub', 'sup', 'merged')}

def build_figure(events=None, save=False):
    events = load_events() if events is None else events
    local_av, sub_av, sup_av, merged_av = [events[k] for k in ('local','sub','sup','merged')]
    with (DATA / 'figure1/local_fit_summary.csv').open(encoding='utf-8') as h:
        local_row = next(csv.DictReader(h))
    duration_xmin = _to_float(local_row.get('fig2_duration_fit_xmin_mean'))
    duration_xmax = _to_float(local_row.get('fig2_duration_xmax_mean'))
    gamma_local_hat = _gamma_from_events(local_av['sizes'], local_av['durations_bins'])
    gamma_local_pred, local_size_alpha, local_duration_alpha = _summary_gamma_pred(local_row)
    sampling_seed = 20260625
    standard_critical_event_count = int(min(local_av["sizes"].size, local_av["durations_bins"].size))
    rng = np.random.default_rng(sampling_seed)
    local_av = _sample_avalanche_events(local_av, standard_critical_event_count, rng)
    sub_av = _sample_avalanche_events(sub_av, standard_critical_event_count, rng)
    sup_av = _sample_avalanche_events(sup_av, standard_critical_event_count, rng)
    merged_av = _sample_avalanche_events(merged_av, standard_critical_event_count, rng)
    event_counts = {
        "local_source": local_av["source_event_count"],
        "local_sampled": local_av["sampled_event_count"],
        "sub_source": sub_av["source_event_count"],
        "sub_sampled": sub_av["sampled_event_count"],
        "sup_source": sup_av["source_event_count"],
        "sup_sampled": sup_av["sampled_event_count"],
        "merged_source": merged_av["source_event_count"],
        "merged_sampled": merged_av["sampled_event_count"],
    }

    merged_sizes = merged_av["sizes"]
    merged_durations = merged_av["durations_bins"]
    merged_gamma_hat = _gamma_from_events(merged_sizes, merged_durations)
    merged_size_alpha, merged_size_xmin, _ = _fit_powerlaw_alpha(merged_sizes, xmin_range=(0, 20))
    merged_duration_alpha, merged_duration_xmin, merged_duration_xmax = _fit_powerlaw_alpha(
        merged_durations,
        xmin_range=(0, 10),
    )
    merged_gamma_pred = _gamma_pred_from_alphas(merged_size_alpha, merged_duration_alpha)

    fig = plt.figure(figsize=(7.2, 4.4), dpi=300)
    gs = GridSpec(
        4,
        3,
        figure=fig,
        width_ratios=[1.22, 1.0, 1.0],
        height_ratios=[1, 1, 1, 1],
        hspace=0.50,
        wspace=0.46,
        left=0.07,
        right=0.98,
        bottom=0.09,
        top=0.96,
    )
    pdf_axes = []
    scaling_axes = []

    ax = fig.add_subplot(gs[0:2, 0])
    _panel_label(ax, "A")
    _draw_original_schematic(ax, "local")

    ax = fig.add_subplot(gs[0, 1])
    _plot_pdf(
        ax,
        [
            (local_av["durations_bins"], COLORS["duration"], "Duration", 2.4),
        ],
        "Avalanche Duration",
        label_positions=[],
    )
    pdf_axes.append(ax)

    ax = fig.add_subplot(gs[1, 1])
    _plot_pdf(
        ax,
        [
            (local_av["sizes"], COLORS["size"], "Size", 2.4),
        ],
        "Avalanche size",
        label_positions=[],
    )
    pdf_axes.append(ax)

    ax = fig.add_subplot(gs[0:2, 2])
    _plot_scaling(
        ax,
        local_av["sizes"],
        local_av["durations_bins"],
        gamma=gamma_local_pred,
        xmin=duration_xmin,
        xmax=duration_xmax,
        label_color=COLORS["local"],
    )
    scaling_axes.append(ax)

    ax = fig.add_subplot(gs[2:4, 0])
    _panel_label(ax, "B")
    _draw_original_schematic(ax, "global")

    ax = fig.add_subplot(gs[2, 1])
    _plot_pdf(
        ax,
        [
            (sub_av["durations_bins"], COLORS["sub"], "Sub", 2.2),
            (sup_av["durations_bins"], COLORS["sup"], "Sup", 2.2),
            (merged_durations, COLORS["global"], "Global", 2.5),
        ],
        "Avalanche Duration",
        label_positions=[],
    )
    pdf_axes.append(ax)

    ax = fig.add_subplot(gs[3, 1])
    _plot_pdf(
        ax,
        [
            (sub_av["sizes"], COLORS["sub"], "Sub", 2.2),
            (sup_av["sizes"], COLORS["sup"], "Sup", 2.2),
            (merged_sizes, COLORS["global"], "Global", 2.5),
        ],
        "Avalanche size",
        label_positions=[(0.58, 0.72), (0.70, 0.72), (0.82, 0.72)],
    )
    pdf_axes.append(ax)

    ax = fig.add_subplot(gs[2:4, 2])
    _plot_scaling(
        ax,
        merged_sizes,
        merged_durations,
        gamma=merged_gamma_pred,
        xmin=merged_duration_xmin,
        xmax=merged_duration_xmax,
        label_color=COLORS["global"],
    )
    scaling_axes.append(ax)

    fig.align_labels()
    fig.align_ylabels(pdf_axes)
    fig.align_ylabels(scaling_axes)
    fig.canvas.draw()
    _match_pdf_pair_height_to_reference(pdf_axes[0], pdf_axes[1], scaling_axes[0], gap_fraction=0.28)
    _match_pdf_pair_height_to_reference(pdf_axes[2], pdf_axes[3], scaling_axes[1], gap_fraction=0.36)
    pdf_axes[0].tick_params(axis="x", which="both", labelbottom=False)
    for axis in pdf_axes:
        axis.yaxis.set_label_coords(-0.30, 0.5)
        axis.xaxis.labelpad = 1.0
        axis.tick_params(axis="both", which="major", pad=1)
    for axis in scaling_axes:
        axis.yaxis.set_label_coords(-0.28, 0.5)
    fig.canvas.draw()

    fig.figure1_metadata = dict(event_counts, gamma_local=gamma_local_pred,
                                gamma_merged=merged_gamma_pred,
                                merged_size_alpha=merged_size_alpha,
                                merged_duration_alpha=merged_duration_alpha,
                                sampling_seed=sampling_seed)
    if save:
        save_figure(fig, 'Figure1')
    return fig


def _to_float(value, default=np.nan):
    try:
        value = float(value)
    except Exception:
        return default
    return value if np.isfinite(value) else default

def _sample_avalanche_events(avalanche, target_count, rng):
    sizes = np.asarray(avalanche["sizes"], dtype=int)
    durations = np.asarray(avalanche["durations_bins"], dtype=int)
    event_count = int(min(sizes.size, durations.size))
    sizes = sizes[:event_count]
    durations = durations[:event_count]
    target_count = int(target_count) if target_count is not None else event_count
    if event_count > target_count > 0:
        keep = np.sort(rng.choice(event_count, size=target_count, replace=False))
        sizes = sizes[keep]
        durations = durations[keep]
    return {
        "sizes": sizes,
        "durations_bins": durations,
        "bin_ms": float(avalanche.get("bin_ms", np.nan)),
        "source_event_count": event_count,
        "sampled_event_count": int(sizes.size),
    }

def _gamma_from_events(sizes, durations):
    x, y, _ = _mean_size_by_duration(sizes, durations)
    if x.size < 2:
        return np.nan
    return float(np.polyfit(np.log10(x), np.log10(y), 1)[0])

def _gamma_pred_from_alphas(size_alpha, duration_alpha):
    size_alpha = _to_float(size_alpha)
    duration_alpha = _to_float(duration_alpha)
    if not np.isfinite(size_alpha) or not np.isfinite(duration_alpha):
        return np.nan
    if abs(size_alpha - 1.0) < np.finfo(float).eps:
        return np.nan
    return float((duration_alpha - 1.0) / (size_alpha - 1.0))

def _summary_gamma_pred(row):
    size_alpha = _to_float(row.get("fig2_size_alpha_mean"))
    duration_alpha = _to_float(row.get("fig2_duration_alpha_mean"))
    gamma_pred = _to_float(row.get("fig2_gamma_pred_from_scaling_relation_mean"))
    if not np.isfinite(gamma_pred):
        gamma_pred = _gamma_pred_from_alphas(size_alpha, duration_alpha)
    return gamma_pred, size_alpha, duration_alpha

def _fit_powerlaw_alpha(values, xmin_range=(0, 20), xmax=None):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values) & (values > 0)]
    if values.size < 10:
        return np.nan, np.nan, np.nan

    import powerlaw

    data = values.astype(int)
    data = data[data > 0]
    if data.size < 10:
        return np.nan, np.nan, np.nan
    if xmax is None:
        xmax = int(np.max(data))
    with contextlib.redirect_stdout(io.StringIO()):
        fit = powerlaw.Fit(
            data,
            discrete=True,
            xmin=xmin_range,
            xmax=int(xmax),
            verbose=False,
        )
    fit_xmax = getattr(fit, "xmax", xmax)
    if fit_xmax is None:
        fit_xmax = xmax
    return float(fit.alpha), float(fit.xmin), float(fit_xmax)

def _distribution(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values) & (values > 0)]
    x, counts = np.unique(values.astype(int), return_counts=True)
    p = counts.astype(float) / float(np.sum(counts))
    return x.astype(float), p

def _mean_size_by_duration(sizes, durations, min_count=20):
    sizes = np.asarray(sizes, dtype=float)
    durations = np.asarray(durations, dtype=float)
    keep = np.isfinite(sizes) & np.isfinite(durations) & (sizes > 0) & (durations > 0)
    sizes = sizes[keep]
    durations = durations[keep].astype(int)
    rows = []
    for duration in np.unique(durations):
        mask = durations == duration
        if int(np.sum(mask)) >= min_count:
            rows.append((float(duration), float(np.mean(sizes[mask])), int(np.sum(mask))))
    if not rows:
        return np.asarray([]), np.asarray([]), np.asarray([])
    x = np.asarray([row[0] for row in rows], dtype=float)
    y = np.asarray([row[1] for row in rows], dtype=float)
    n = np.asarray([row[2] for row in rows], dtype=float)
    return x, y, n

def _avalanche_from_spike_times(spike_times_ms, burn_in_ms=200.0):
    spike_times = np.sort(np.asarray(spike_times_ms, dtype=float))
    spike_times = spike_times[np.isfinite(spike_times) & (spike_times > float(burn_in_ms))]
    if spike_times.size < 3:
        return {"sizes": np.asarray([]), "durations_bins": np.asarray([]), "bin_ms": np.nan}

    rel = spike_times - float(burn_in_ms)
    isi = np.diff(rel)
    isi = isi[isi >= 0]
    bin_ms = float(np.mean(isi)) if isi.size else np.nan
    if not np.isfinite(bin_ms) or bin_ms <= 0:
        raise ValueError("Cannot estimate a positive bin width from reconstructed spike times")

    bin_index = np.floor(rel / bin_ms).astype(int)
    bin_index = bin_index[bin_index >= 0]
    counts = np.bincount(bin_index, minlength=int(np.max(bin_index)) + 1)

    sizes = []
    durations = []
    idx = 0
    n_bins = counts.size
    while idx < n_bins:
        if counts[idx] <= 0:
            idx += 1
            continue
        start = idx
        while idx < n_bins and counts[idx] > 0:
            idx += 1
        segment = counts[start:idx]
        durations.append(int(segment.size))
        sizes.append(int(np.sum(segment)))

    return {
        "sizes": np.asarray(sizes, dtype=int),
        "durations_bins": np.asarray(durations, dtype=int),
        "bin_ms": bin_ms,
        "counts": counts,
    }

def _fit_intercept_for_slope(x, y, slope, xmin=None, xmax=None):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    keep = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    if xmin is not None:
        keep &= x >= float(xmin)
    if xmax is not None:
        keep &= x <= float(xmax)
    if np.sum(keep) < 2:
        keep = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    intercept = np.median(np.log10(y[keep]) - float(slope) * np.log10(x[keep]))
    return float(intercept)

def _panel_label(ax, label):
    ax.text(
        -0.16,
        1.08,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=11,
        fontweight="bold",
        clip_on=False,
    )

def _clean_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", length=3, width=0.7)

def _hide_schematic_axis(ax):
    ax.set_axis_off()
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)
    ax.set_aspect("equal", adjustable="box")

def _style_log_axis(ax, max_y_ticks=None):
    def ticks_for(lim, max_ticks=None):
        lo, hi = sorted([float(lim[0]), float(lim[1])])
        lo = max(lo, np.finfo(float).tiny)
        exponents = range(int(np.floor(np.log10(lo))), int(np.ceil(np.log10(hi))) + 1)
        ticks = [10.0 ** exp for exp in exponents if lo <= 10.0 ** exp <= hi]
        if max_ticks is not None and len(ticks) > max_ticks:
            stride = int(np.ceil(len(ticks) / max_ticks))
            ticks = ticks[::stride]
        return ticks if ticks else [lo, hi]

    ax.xaxis.set_major_locator(FixedLocator(ticks_for(ax.get_xlim())))
    ax.yaxis.set_major_locator(FixedLocator(ticks_for(ax.get_ylim(), max_y_ticks)))
    ax.xaxis.set_major_formatter(LogFormatterMathtext(base=10))
    ax.yaxis.set_major_formatter(LogFormatterMathtext(base=10))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_minor_formatter(NullFormatter())
    _clean_axis(ax)

def _plot_pdf(ax, series, xlabel, label_positions=None):
    all_x = []
    all_p = []
    for values, color, label, marker_size in series:
        x, p = _distribution(values)
        if not x.size:
            continue
        all_x.append(x)
        all_p.append(p)
        ax.loglog(
            x,
            p,
            "-o",
            ms=max(1.0, marker_size * 0.6),
            color=color,
            alpha=0.92,
            lw=1.0,
            markeredgewidth=0,
            label=label,
        )
    if not all_x or not all_p:
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Prob.")
        _clean_axis(ax)
        return
    x_concat = np.concatenate(all_x)
    p_concat = np.concatenate(all_p)
    ax.set_xlim(0.9, max(2.0, float(np.nanmax(x_concat)) * 1.15))
    ax.set_ylim(max(1e-6, float(np.nanmin(p_concat[p_concat > 0])) * 0.7), 0.7)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Prob.")
    if label_positions is None:
        label_positions = [(0.64, 0.84 - 0.18 * idx) for idx in range(len(series))]
    for (_, color, label, _), (xpos, ypos) in zip(series, label_positions):
        ax.text(xpos, ypos, label, transform=ax.transAxes, color=color, fontsize=6.0, fontweight="bold")
    _style_log_axis(ax, max_y_ticks=2)

def _plot_scaling(ax, sizes, durations, gamma, xmin, xmax, label_color="0.35"):
    x, y, n = _mean_size_by_duration(sizes, durations)
    if x.size:
        ax.loglog(x, y, "o", ms=2.4, color=label_color, alpha=0.55, markeredgewidth=0)
        intercept = _fit_intercept_for_slope(x, y, gamma, xmin=xmin, xmax=xmax)
        line_x = np.geomspace(max(np.nanmin(x), 1), max(np.nanmax(x), 2), 100)
        line_y = 10 ** intercept * line_x ** float(gamma)
        ax.loglog(line_x, line_y, "-", color="k", lw=1.2)
        ax.set_xlim(max(0.9, float(np.nanmin(x)) * 0.85), float(np.nanmax(x)) * 1.15)
        ax.set_ylim(max(0.8, float(np.nanmin(y)) * 0.8), float(np.nanmax(y)) * 1.25)
    ax.set_xlabel(r"Avalanche Duration $T$")
    ax.set_ylabel(r"$<S>(T)$")
    ax.text(
        0.08,
        0.88,
        rf"$\frac{{\alpha-1}}{{\tau-1}}={float(gamma):.2f}$",
        transform=ax.transAxes,
        fontsize=9,
        va="top",
    )
    _style_log_axis(ax)
    ax.set_box_aspect(0.55)

def _match_pdf_pair_height_to_reference(top_ax, bottom_ax, reference_ax, gap_fraction=0.10):
    """Place two PDF axes so their combined height matches the reference axis."""
    ref_pos = reference_ax.get_position()
    top_pos = top_ax.get_position()
    bottom_pos = bottom_ax.get_position()
    left = min(top_pos.x0, bottom_pos.x0)
    right = max(top_pos.x1, bottom_pos.x1)
    gap = ref_pos.height * gap_fraction
    pdf_height = (ref_pos.height - gap) / 2.0
    bottom_ax.set_position([left, ref_pos.y0, right - left, pdf_height])
    top_ax.set_position([left, ref_pos.y0 + pdf_height + gap, right - left, pdf_height])