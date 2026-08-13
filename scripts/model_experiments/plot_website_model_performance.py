"""Plot the leakage-safe website model performance (baseline + three runs).

Reads website_run_metrics.json (captured from the notebook's ls_results) and
renders in the same seaborn whitegrid style as the other website figures, so
the three sit consistently together. Deliberately clean: no floating arrow
annotation, and the chance line is a subtle grey reference, not red clutter.

Run: python plot_website_model_performance.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import numpy as np
import seaborn as sns

sns.set_theme(style="whitegrid", context="notebook", palette="muted")


def lighten(hex_c, f=0.5):
    r, g, b = mcolors.to_rgb(hex_c)
    return (r + (1 - r) * f, g + (1 - g) * f, b + (1 - b) * f)

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
DATA = json.load(open(HERE / "website_run_metrics.json"))
OUT = REPO / "outputs" / "eda_website" / "website_model_performance.png"
OUT.parent.mkdir(parents=True, exist_ok=True)

# hue = model (baseline grey, each run its own colour); shade = metric
GROUP_COLORS = ["#8C8C8C", "#4C72B0", "#DD8452", "#55A868"]  # baseline, run1, run2, run3

labels = ["Baseline\n(random forest)", "Run 1\nfull",
          "Run 2\nlanding hidden", "Run 3\ndirect arrivals\nexcluded"]
runs = DATA["runs"]
test = [r["test AUC"] for r in runs]
cv = [r["CV AUC"] for r in runs]
cv_sd = [r["CV AUC sd"] for r in runs]

x = np.arange(len(labels))
w = 0.38

fig, ax = plt.subplots(figsize=(10, 6))
for xi, t, c, sd, gc in zip(x, test, cv, cv_sd, GROUP_COLORS):
    ax.bar(xi - w / 2, t, w, color=lighten(gc, 0.55))          # test = lighter
    ax.bar(xi + w / 2, c, w, yerr=sd, capsize=4, color=gc)      # CV = full colour
    ax.text(xi + w / 2, c + sd + 0.02, f"{c:.2f}", ha="center", va="bottom", fontsize=9)

# chance reference line (red dashed); labelled in the key, not on the chart
ax.axhline(0.5, color="red", linestyle="--", linewidth=1)

ax.set_ylim(0, 1.05)
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylabel("ROC AUC")
ax.set_title("Website conversion-propensity model: predictive performance\n"
             "(leakage-safe, three progressively stricter runs)")

# key: shade convention plus the chance reference, bottom left
key = [Patch(facecolor="#C9C9C9", label="Test-set AUC (lighter bar)"),
       Patch(facecolor="#6E6E6E", label="Cross-validated AUC (darker bar)"),
       Line2D([0], [0], color="red", linestyle="--", linewidth=1, label="chance = 0.5")]
ax.legend(handles=key, loc="lower left", framealpha=0.9)

fig.tight_layout()
fig.savefig(OUT, dpi=150, bbox_inches="tight")
print("saved", OUT)
