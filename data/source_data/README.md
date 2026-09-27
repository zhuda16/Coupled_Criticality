# Main-figure plotting data

`main_figure_plot_data.npz` contains numerical inputs for the quantitative
panels of Figures 1-4. It contains plotted coordinates and the replicate values
used for error bands and violins, not the complete raw spike archive. Schematic
artwork is excluded. Load with NumPy only; no pickle objects are required.

```python
import numpy as np
import matplotlib.pyplot as plt

with np.load("main_figure_plot_data.npz", allow_pickle=False) as d:
    x = d["fig3e_gEI"]
    y = d["fig3e_accuracy"]  # network x coupling value
    mean, sd = y.mean(axis=0), y.std(axis=0)
    plt.plot(x, mean, color="black")
    plt.fill_between(x, mean - sd, mean + sd, alpha=0.2)
    plt.xlabel("Local inhibition scale")
    plt.ylabel("Accuracy")
plt.show()
```

| Keys | Contents and array dimensions |
|---|---|
| `fig1_{local,sub,sup,merged}_{duration,size}_pdf_{x,y}` | Distribution coordinates. `local` belongs to A; the other three to B. Duration is in mean-ISI bins, size in spikes. Original seed 20260625 and event downsampling are retained. |
| `fig1_{local,merged}_scaling_{x,y,count}` | Duration, mean avalanche size and event count for each plotted point; only durations with at least 20 events are shown. `reference_x/y` gives the fitted reference line. |
| `fig2_gEI`, `fig2_trial_id` | Coupling coordinates (20) and simulation trial IDs (50). |
| `fig2_{b_duration,b_size,c_dcc}_values` | Underlying numerical results, shape (20 coupling values, 50 trials). The B values are the saved within-trial mean KS p-values; C is absolute scaling-exponent discrepancy. |
| `fig2_{b_duration,b_size,c_dcc}_{mean,sd,lower,upper}` | Plotted curves and band edges. Original Gaussian smoothing (sigma=1) of the mean and population SD is retained; lower/upper clipping follows the source plot. |
| `fig2d_region{0,1,2}_duration_pdf_{x,y}` | Pooled duration distributions from 20 critical-state trials. Regions 0/1/2 mean A/B/combined. |
| `fig2e_duration`, `fig2e_mean_size`, `fig2e_reference_y` | Mean-size scaling points and original reference-line values. |
| `fig2f_eigenvalue_raw`, `fig2f_eigenvalue_display` | Mean-field eigenvalues across the 20 coupling values. Display values retain the original +0.04 plotting offset. |
| `fig3b_conditions`, `fig3b_accuracy` | Three conditions in the stored order; shape (3, 20 networks). Neuron subsampling, direction peak, 10 samples averaged per signal pair, then all 15 pairs averaged per network. |
| `fig3c_{uncoupled,coupled}_{xyz,signal,trial_within_signal}` | Display coordinates (600, 3) and signal labels for network 17; six signals, 100 trials each. Trial IDs follow the within-signal CSV order. |
| `fig3d_lbi`, `fig3d_{silhouette,reliability}_values` | Coupling coordinates and underlying values (20 networks, 20 coupling values). Silhouette is averaged over all fifteen 4-of-6 signal subsets within each network. |
| `fig3d_{silhouette,reliability}_{mean,sd,sem}` | Across-network summaries. Final D bands use sample SD (`ddof=1`), not SEM. |
| `fig3e_gEI`, `fig3e_accuracy`, `fig3e_mean`, `fig3e_sd` | Accuracy (20 networks, 20 coupling values), with the same aggregation as B. E uses population SD (`ddof=0`). Critical index is 11 (zero-based). |
| `fig4{b,e}_region{0,1}_{xyz,signal,pc_indices_zero_based}` | Before/after-rewiring display coordinates, signal labels and selected PC indices. Rows are retained response samples; filtering gives different sample counts across panels. |
| `fig4f_groups`, `fig4f_values` | Six groups (region then dynamical state) with 20 displayed values per group. The original seeded selection from 50 saved scores is retained. |
| `fig4f_rewiring_id`, `fig4f_resample_id` | IDs of the selected scores. These 20 values are resamples nested within rewiring realizations, not 20 independent networks. |
| `fig4g_region{0,1,2}_size_pdf_{x,y}` | Avalanche-size distributions after rewiring, pooled across 10 saved trials. |
| `fig4g_duration`, `fig4g_mean_size`, `fig4g_reference_y` | Combined-network scaling points and the reference curve using the original fit and intercept convention. |

All indices are zero-based. `metadata_json` contains source-file SHA-256
checksums and an array-shape inventory. Figure 3 uses the finalized 20-network,
six-signal data, rather than the previous legacy decoding results. Other panels
follow the existing manuscript plotting programs and their saved numerical
inputs. This compact NPZ is for plotting; journal Source Data may additionally
require Excel or text-format tables.
