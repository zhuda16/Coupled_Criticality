# Coupled criticality: main-figure demos

Four main-figure notebooks reproduce the selected quantitative panels, and a fifth notebook trains a critical-state Tempotron directly from spikes. They use the manuscript's existing analysis functions and a compact, self-contained data release. Start with **Python 3.11**.

```text
data/                       numerical inputs and provenance manifest
  figure1/                  avalanche size/duration pairs, fit summaries, schematics
  figure2/                  critical avalanches and mean-field sigma inputs
  figure3/                  compact inputs for the finalized N=20/S6 Figure 3
  figure4/                  neuron membership matrices before/after rewiring
  tempotron_cri/             recorded spikes for a critical-state binary task
programe/                   five notebooks and their local Python modules
output/                     reproduced PNG/PDF figures
qa/                         figure-layout audit reports
```

## Run

Create an environment, install the supplied requirements, and open any notebook in `programe/`. Run its cells from top to bottom. Paths are relative to this release; no parent research repository, Brian2, cluster connection, or network simulation is needed.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
python -m jupyterlab
```

| Notebook | Reproduced panels | Data and computation |
|---|---|---|
| [Demo_Figure1](programe/Demo_Figure1.ipynb) | Figure 1A/B | All 20 trials; local, subcritical, supercritical and combined avalanche arrays; original seeded sampling, PDF and scaling plots. Original schematic images are included. |
| [Demo_Figure2](programe/Demo_Figure2.ipynb) | Figure 2D/E; one F point | Critical avalanches from 20 trials; F uses sigma inputs from 50 trials and runs one mean-field calculation. |
| [Demo_Figure3](programe/Demo_Figure3.ipynb) | Figure 3B–E | Reproduces only the finalized quantitative panels; B includes network points, C fixes network 17, and D/E use across-network SD bands. |
| [Demo_Figure4](programe/Demo_Figure4.ipynb) | Figure 4B/E | Recomputes standardization, PCA and dimension selection for four signals in regions A/B, before and after rewiring. |
| [Demo_Tempotron_cri](programe/Demo_Tempotron_cri.ipynb) | Critical-state readout training demo | Starts from spikes, trains the original Tempotron, predicts held-out trials and saves weights/voltages. |

Execution was checked on a copy outside the research repository. See [validation_environment.json](validation_environment.json) for package versions and measured notebook runtimes. Timing depends on hardware and BLAS thread configuration. Set `OPENBLAS_NUM_THREADS=1` and `OMP_NUM_THREADS=1` before starting Jupyter if excessive threading slows down the small PCA problems.

## Data definitions and original programs

The NPZ files contain ordinary numerical/string arrays and load with `numpy.load(..., allow_pickle=False)`. No original pickle objects are required by the notebooks. [data/manifest.json](data/manifest.json) records the precise original input filenames, extraction descriptions and SHA-256 checksums. Source paths in that file are provenance, not runtime dependencies.

- **Figure 1:** `Response_CB/Make_Figure/figure1_keyresult_local_rebuild.py`, using `build_all_trials_figure`. For each condition, `sizes` is spike count and `durations_bins` is the number of consecutively occupied bins. `trial_offsets` separates trials 0–19; `trial_bin_ms` records each trial's mean-ISI bin width. The combined avalanches were detected after merging sub/supercritical spike times, not by joining the two avalanche lists. The notebook retains the original seed 20260625 and event downsampling. The local reference exponent comes from the original per-condition fit summary; the combined-system exponent is refitted from the supplied events. Source counts are local **877,691**, subcritical **1,042,518**, supercritical **559,093**, combined **1,871,602**. Larger sets are sampled down to 877,691 for plotting, as in the source. Data export used NumPy 1.26.4: NumPy 2.4.6 changed a few boundary assignments and the counts by one, so the former is pinned.
- **Figure 2:** `Response_CB/Make_Figure/coupledcriticality_figure_tools.py`, functions `figure2_critical_avalanches`, `plot_noise_relation` and `meanfield_couple`. Region indices 0/1/2 mean A/B/combined. All size/duration pairs from trials 0–19 at critical index 8 are included, with **1,040,369** combined events. Per-trial fitted scaling exponents are retained from the original analysis. `sigma_<trial>` contains two rows (A/B), two columns (E/I); all 50 inputs are supplied. The operating point is **1.0942105263157895**. F returns the raw maximum real eigenvalue and, separately, the original display value including its **+0.04** offset. This offset is a plotting convention in the source, not part of the eigenvalue calculation. The original mean-size reference-line formula is retained unchanged for reproduction.
- **Figure 3:** [figure3_demo.py](programe/figure3_demo.py) is the sole portable builder for the finalized N=20-network, six-signal quantitative panels B–E. Its compact inputs are in `data/figure3/`: B shows subsampled direction-peak accuracy averaged across all 15 signal pairs within each network; C uses network 17 and the best three-PC combination among the first ten PCs; D and E display mean ± SD across 20 networks. Projection coordinates and aggregate values are supplied so the public demo does not require the large raw cluster pickle archive. Schematic/demo panel A is intentionally excluded.
- **Figure 4B/E:** `figure4_rewiring_representation_2026` in the same source module. The release matrices were recomputed from all 100 trials in `Analysis_CoupledCri_2026/RewiringReliabilityData`, using the original `activity_mask_2026`, `avalanche_stats_from_spikes` and `figure4_rewiring_reliable_neuronset_2026` functions. Selection uses 0–250 ms, avalanche threshold 20 and attractor 40. The original rule removes empty/full-response columns before PCA. Included conditions are `no` and `spatial_mixture`, the latter at noise 0.2. The notebook starts from these selected binary matrices and reruns the full dimensional analysis. It includes both regions before/after, as in the existing Figure 4 builder. Surviving sample counts are 387/371 before and 376/377 after; signal lengths can differ due to the source filtering.

## Critical-state Tempotron demo

[Demo_Tempotron_cri.ipynb](programe/Demo_Tempotron_cri.ipynb) uses network realization 0, LBI index 10 (**g_EI_scale=1.0895**, the Figure 3C working point), and signals **9/4**. The signal pair comes from archived readout replicate 0. [data/tempotron_cri/spikes.npz](data/tempotron_cri/spikes.npz) contains all recorded excitatory spike times and original neuron IDs in both regions, for all 100 trials per signal, before readout subsampling. Times are in ms. Each `signal<S>_region<R>_trial_offsets` array separates the corresponding `time_ms` and `neuron_id` arrays into trials 0–99. Original inhibitory arrays were empty and are not needed by the excitatory readout.

The loader selects 800 neurons from each region using seed 20260909 and applies the original 50–150 ms input window. Training uses the first 20 trials per signal and testing uses the remaining 80. The original cluster `Tempotron` function is preserved verbatim in [tempotron_original.py](programe/tempotron_original.py), with only its import context minimized. Its 50–149 ms voltage grid matches the `subsample` simulation data; the older standalone `Figure_Temptron.py` example uses a different window and is not used here. Weight initialization is seeded with 20260909. No test results select the seed, weights or neuron subset.

The notebook trains from scratch; `output/Tempotron_cri_result.npz` and `Tempotron_cri_training.log` are outputs, not required inputs. The result includes selected neuron IDs, weights, held-out peak voltages and peak times, and a fixed trial-20 voltage example. The original error formula counts positive voltages below 10 and negative voltages above 10 as errors; exact threshold ties are reported. The verified seeded run made **12 updates** and achieved **0.80625 accuracy (80.625%, 129/160)**. This is a new training demonstration, not the exact historical Figure 3E replicate: the historical neuron choices and initialization seed were not saved. Its old accuracy is recorded separately in metadata for provenance and is not used in training.

[metadata.json](data/tempotron_cri/metadata.json) records the parameters and hashes of all 100 original spike files. `original_project/` in the manifest denotes the original project root, whose data were copied into this release; it is not a runtime dependency. The verification compares every exported spike and neuron ID to source, checks that the original training function is unchanged, executes the notebook outside the research repository, and confirms a repeated seeded fit is identical.

`analysis_core.py` contains verbatim original function bodies with minimal imports. `figure1_demo.py` retains original fitting/sampling/plotting helpers and layout, with portable input loading. The other small adapters retain the source calculations and arrange only the requested panels. The authors' code is released under the [MIT License](LICENSE). `programe/powerlaw.py` retains Jeff Alstott's original MIT copyright/license notice.

## Validation

All five notebooks and their scripts were execution-checked. Figures are exported as PNG and PDF; the Figure 3 builder also exports SVG. Figure 1 counts and fitted exponents, Figure 2 pooled event count, Figure 3 summary values, and Figure 4 sample counts/PC choices/silhouette values were compared with the existing manuscript analysis outputs. See [reproduction_checks.json](reproduction_checks.json) and the reports in `qa/` for the results and any intentional layout warnings.

Final figure audit: **zero errors**. Figures 1, 2, 4 and the Tempotron demo have zero warnings. Figure 3 has the expected panel-D `twinx` overlap warning and one benign panel-label association warning caused by the 3D axes; the rendered labels are aligned and were visually checked.
