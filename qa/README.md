# Figure layout audit

Audited using the research repository's `.codex/skills/paper-figure-qa/scripts/mpl_figure_audit.py`, calling each adapter's `build_figure` function in the recorded Python environment.

| Figure | Errors | Warnings |
|---|---:|---:|
| 1 | 0 | 0 |
| 2D/E | 0 | 0 |
| Figure 3B–E | 0 | 2 |
| 4B/E | 0 | 0 |
| Critical-state Tempotron demo | 0 | 0 |

Figure 3 retains the expected `axes_overlap` warning because panel D uses overlaid `twinx` axes. Its other warning is a benign panel-label association caused by the 3D axes; the rendered labels are aligned. The figure was inspected for clipping and collisions.
