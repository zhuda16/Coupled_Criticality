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


