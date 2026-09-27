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


