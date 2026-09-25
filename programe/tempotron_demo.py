"""Portable spike input and prediction adapter around the original Tempotron."""
import contextlib
import io
import json
import random

import matplotlib.pyplot as plt
import numpy as np

from demo_common import DATA, OUTPUT
from tempotron_original import Tempotron, VmaxANDtmaxfun, spiketrain


def load_spike_trains():
    meta = json.loads((DATA / 'tempotron_cri/metadata.json').read_text())
    rng = random.Random(meta['neuron_sampling_seed'])
    selected = [rng.sample(range(1600), 800) for _ in range(2)]
    trains = []
    with np.load(DATA / 'tempotron_cri/spikes.npz', allow_pickle=False) as d:
        for signal in meta['signals']:
            trials = []
            for trial in range(100):
                combined = []
                for region in range(2):
                    prefix = f'signal{signal}_region{region}'
                    begin, end = d[prefix + '_trial_offsets'][trial:trial+2]
                    t = d[prefix + '_time_ms'][begin:end]
                    idx = d[prefix + '_neuron_id'][begin:end]
                    mask = (t >= 50) & (t <= 150)
                    grouped = spiketrain(t[mask], idx[mask], neuronnum=1600)
                    combined.extend(grouped[i] for i in selected[region])
                trials.append(combined)
            trains.append(trials)
    return trains, np.asarray(selected), meta


def train_and_predict(trains, selected, meta):
    # Preserve caller RNG state while making original random initialization repeatable.
    previous = np.random.get_state()
    log = io.StringIO()
    try:
        np.random.seed(meta['weight_seed'])
        with contextlib.redirect_stdout(log):
            accuracy, weights, last_step, peak_times = Tempotron(trains[0], trains[1])
    finally:
        np.random.set_state(previous)
    peak_voltage = np.empty((2, 80))
    example_voltage = np.empty((2, 100))
    for label in range(2):
        for position, trial in enumerate(range(20, 100)):
            _, peak_voltage[label, position], trace = VmaxANDtmaxfun(weights, trains[label][trial])
            if trial == 20:
                example_voltage[label] = trace
    # The original score uses strict > / < inequalities for the two error counts.
    errors = np.count_nonzero(peak_voltage[0] < 10) + np.count_nonzero(peak_voltage[1] > 10)
    assert accuracy == 1 - errors / 160
    result = dict(accuracy=np.asarray(accuracy), weights=weights,
                  updates=np.asarray(last_step + 1), peak_times_signal_positive=peak_times[20:],
                  peak_voltage=peak_voltage, example_voltage=example_voltage,
                  selected_neuron_ids=selected, signals=np.asarray(meta['signals']),
                  threshold=np.asarray(10.), time_ms=np.arange(50., 150.),
                  test_trial_ids=np.arange(20, 100),
                  threshold_ties=np.asarray(np.count_nonzero(peak_voltage == 10)))
    OUTPUT.mkdir(exist_ok=True)
    np.savez_compressed(OUTPUT / 'Tempotron_cri_result.npz', **result)
    (OUTPUT / 'Tempotron_cri_training.log').write_text(log.getvalue(), encoding='utf-8')
    return result


def build_figure(result=None):
    if result is None:
        with np.load(OUTPUT / 'Tempotron_cri_result.npz', allow_pickle=False) as d:
            result = dict(d)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1), constrained_layout=True)
    colors = ['tab:blue', 'tab:orange']
    for label, color in enumerate(colors):
        name = f"Signal {int(result['signals'][label])} ({'positive' if label == 0 else 'negative'})"
        axes[0].plot(result['time_ms'], result['example_voltage'][label], color=color, label=name)
        axes[1].plot(result['test_trial_ids'], result['peak_voltage'][label], '.', color=color, markersize=4)
    for ax in axes:
        ax.axhline(float(result['threshold']), ls='--', color='.4', lw=1)
        ax.spines[['top', 'right']].set_visible(False)
    axes[0].set(xlabel='Time (ms)', ylabel='Readout voltage (a.u.)', title='Held-out trial 20')
    axes[1].set(xlabel='Held-out trial', ylabel='Peak readout voltage (a.u.)',
                title=f"Test accuracy: {float(result['accuracy']):.1%}")
    axes[0].set_xlim(50, 149); axes[0].set_xticks([50, 75, 100, 125, 149])
    axes[1].set_xlim(19, 100); axes[1].set_xticks([20, 40, 60, 80, 99])
    for ax in axes:
        lo, hi = ax.get_ylim()
        ax.set_yticks([tick for tick in ax.get_yticks() if lo <= tick <= hi])
    axes[0].legend(frameon=False, loc='upper center', bbox_to_anchor=(.5, -.24), fontsize=7)
    return fig
