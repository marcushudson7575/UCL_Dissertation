"""Plot the demo honest leakage-safe model coefficients.

Reads demo_honest_coefs.csv (written by demo_experiment.py) and renders a clean
coefficient chart in the same seaborn whitegrid style as the other figures.
Significance is shown as a tidy p-value column on the right, not scattered on
each bar. The n and censoring detail belong in the caption, not the title.

Run: python plot_demo_honest_coefs.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid", context="notebook", palette="muted")

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
df = pd.read_csv(HERE / "demo_honest_coefs.csv")
OUT = REPO / "outputs" / "eda_demo" / "demo_honest_coefs.png"
OUT.parent.mkdir(parents=True, exist_ok=True)

LABELS = {
    "rageclick_count": "Rageclicks",
    "total_pv": "Total pageviews",
    "mean_dur": "Mean session duration",
    "engagement_depth": "Engagement depth",
    "total_clicks": "Total clicks",
    "max_dur": "Longest session duration",
    "has_referrer": "Arrived with a referrer",
    "n_sessions": "Number of sessions",
}
GREEN, RED = "#55A868", "#C44E52"

df["label"] = df["feature"].map(LABELS).fillna(df["feature"])
df = df.sort_values("std_coef")  # ascending -> positive at top, negative at bottom in barh
colors = [GREEN if v > 0 else RED for v in df["std_coef"]]


def stars(p):
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."


fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(df["label"], df["std_coef"], color=colors)
ax.axvline(0, color="0.4", linewidth=1)
ax.set_xlabel("Standardised logistic coefficient")
ax.set_title("Demo sign-up drivers")

# significance stars just past each bar's tip; exact p-values live in the caption
for y, (coef, p) in enumerate(zip(df["std_coef"], df["MW_p"])):
    if coef >= 0:
        ax.text(coef + 0.02, y, stars(p), va="center", ha="left", fontsize=11, color="0.35")
    else:
        ax.text(coef - 0.02, y, stars(p), va="center", ha="right", fontsize=11, color="0.35")
ax.set_xlim(df["std_coef"].min() - 0.28, df["std_coef"].max() + 0.22)

fig.tight_layout()
fig.savefig(OUT, dpi=150, bbox_inches="tight")
print("saved", OUT)
