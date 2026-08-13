"""Plot the elastic-net logistic odds ratios saved by website_experiment.py.

Reads website_logit_odds_ratios.csv and renders a driver chart in the same
seaborn whitegrid style as the notebook figures (e.g. the Run 3 importance
chart), so the two sit consistently together in the dissertation.

Run: python plot_website_odds_ratios.py
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
CSV = HERE / "website_logit_odds_ratios.csv"
OUT = REPO / "outputs" / "eda_website" / "website_odds_ratios.png"
OUT.parent.mkdir(parents=True, exist_ok=True)

# readable labels for the features shown, matching the existing figure
LABELS = {
    "num__landed_on_get_started": "Landed directly on sign-up page",
    "num__pre_clicks": "Pre-conversion clicks (volume of interaction)",
    "num__pre_uc_breadth": "Use-case breadth (distinct use-case pages)",
    "cat__country_group_Switzerland": "Country: Switzerland",
    "cat__landing_group_commercial": "Landing page: commercial (pricing/platform)",
    "cat__landing_group_home": "Landing page: home",
    "cat__browser_group_other": "Browser: other/uncommon",
    "cat__referrer_group_google.com": "Referrer: google.com",
    "cat__landing_group_other": "Landing page: other",
    "num__pre_uc_pageviews": "Use-case pageviews (pre-conversion)",
    "num__pre_duration_s": "Pre-conversion time on site (seconds)",
    "cat__browser_group_Firefox": "Browser: Firefox",
    "cat__utm_bucket_other": "UTM source: other/untagged",
    "cat__country_group_France": "Country: France",
}

GREEN = "#55A868"   # matches the notebook importance charts
RED = "#C44E52"     # seaborn muted red for OR < 1

df = pd.read_csv(CSV)
df = df[df["feature"].isin(LABELS)].copy()
df["label"] = df["feature"].map(LABELS)
df = df.sort_values("OR")  # smallest OR first so largest sits at the top in barh

colors = [GREEN if v > 1 else RED for v in df["OR"]]

fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(df["label"], df["OR"], color=colors)
ax.set_xscale("log")
ax.axvline(1.0, color="0.4", linestyle="--", linewidth=1)
ax.set_title("Website conversion drivers: elastic-net logistic odds ratios")
ax.set_xlabel("Odds ratio (log scale). Values above 1 raise the odds of conversion")
ax.set_ylabel("")

for y, v in enumerate(df["OR"]):
    # always place the value just to the right of the bar end
    ax.text(v * 1.06, y, f"{v:.2f}", va="center", ha="left", fontsize=9)

fig.tight_layout()
fig.savefig(OUT, dpi=150, bbox_inches="tight")
print("saved", OUT)
