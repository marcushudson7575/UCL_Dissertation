"""Compact refinements of the platform Cox model, honestly CV'd."""
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from sklearn.model_selection import RepeatedKFold

warnings.filterwarnings("ignore")
SCRATCH = Path(__file__).parent
RNG = 42

dfe = pd.read_pickle(SCRATCH / "platform_enriched_df.pkl")

def cv_cindex(df, feats, penalizer=0.05, label=""):
    rkf = RepeatedKFold(n_splits=5, n_repeats=10, random_state=RNG)
    scores = []
    for tr, te in rkf.split(df):
        try:
            m = CoxPHFitter(penalizer=penalizer).fit(
                df.iloc[tr][["days", "returned"] + feats], "days", "returned")
            risk = m.predict_partial_hazard(df.iloc[te][feats])
            scores.append(concordance_index(df.iloc[te]["days"], -risk, df.iloc[te]["returned"]))
        except Exception:
            continue
    s = np.array(scores)
    print(f"{label}: CV C = {s.mean():.3f} ± {s.std():.3f}")

VARIANTS = {
    "V0 baseline [launched, navi, team]": ["launched_agents", "used_Navi", "team_org"],
    "V1 launch intensity [launch_log, navi, team]": ["launch_clicks_log", "used_Navi", "team_org"],
    "V2 V1 + rageclicks": ["launch_clicks_log", "used_Navi", "team_org", "fs_rageclicks"],
    "V3 V2 + used_Results": ["launch_clicks_log", "used_Navi", "team_org", "fs_rageclicks", "used_Results"],
    "V4 V1 + used_Results": ["launch_clicks_log", "used_Navi", "team_org", "used_Results"],
}
for label, feats in VARIANTS.items():
    cv_cindex(dfe, feats, label=label)

# inference for the best-by-CV variant gets printed after inspection; run all summaries
for label, feats in VARIANTS.items():
    m = CoxPHFitter(penalizer=0.05).fit(dfe[["days", "returned"] + feats], "days", "returned")
    hrs = np.exp(m.params_).round(2).to_dict()
    ps = m.summary["p"].round(3).to_dict()
    print(f"\n{label}: in-sample C={m.concordance_index_:.3f}")
    print("  HR:", hrs)
    print("  p :", ps)
print("DONE")
