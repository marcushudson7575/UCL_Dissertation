"""Platform retention (time-to-second-session) experiment.

P0: reproduce the notebook's 3-feature Cox (in-sample C ~= 0.646) and give it
    an HONEST repeated-CV concordance.
P1: enriched first-session features -> penalized Cox (ridge grid via CV).
P2: XGBoost survival:cox on enriched features, CV concordance.
Bootstrap HRs for the best Cox variant.
"""
from pathlib import Path
import pickle
import warnings

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from sklearn.model_selection import RepeatedKFold
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")
SCRATCH = Path(__file__).parent
RAW = Path("/Users/marcushudson/Documents/GitHub/UCL Dissertation/raw data/main platform files 1")
RNG = 42

d = pickle.load(open(SCRATCH / "platform_dfs.pkl", "rb"))
surv = d["surv"].reset_index(drop=True)
sessions = d["sessions"]
persons = d["persons"]
events = d["events"]
COX_FEATURES = d["COX_FEATURES"]

# ------------------------------------------------ P0 baseline reproduction
cph = CoxPHFitter().fit(surv[["days", "returned"] + COX_FEATURES], "days", "returned")
print(f"P0 in-sample C-index = {cph.concordance_index_:.3f} (notebook ~0.646)")
print(cph.summary[["exp(coef)", "exp(coef) lower 95%", "exp(coef) upper 95%", "p"]].round(3).to_string())

def cv_cindex(df, feats, penalizer=0.05, n_splits=5, n_repeats=10, label=""):
    rkf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=RNG)
    scores = []
    for tr, te in rkf.split(df):
        try:
            m = CoxPHFitter(penalizer=penalizer).fit(
                df.iloc[tr][["days", "returned"] + feats], "days", "returned")
            risk = m.predict_partial_hazard(df.iloc[te][feats])
            scores.append(concordance_index(df.iloc[te]["days"], -risk, df.iloc[te]["returned"]))
        except Exception:
            continue
    scores = np.array(scores)
    print(f"{label}: CV C-index = {scores.mean():.3f} ± {scores.std():.3f} (n={len(scores)} folds)")
    return scores.mean()

cv_cindex(surv, COX_FEATURES, penalizer=0.05, label="P0 Cox 3-feature (honest CV)")

# ------------------------------------- rebuild user table w/ session order
sess = sessions.dropna(subset=["person_id"]).sort_values(["person_id", "start_timestamp"]).copy()
first = sess.groupby("person_id").head(1).set_index("person_id")
sec = sess.groupby("person_id")["start_timestamp"].apply(lambda s: s.iloc[1] if len(s) > 1 else pd.NaT)
EXPORT_END = pd.Timestamp("2026-06-10", tz="UTC")
days = ((sec.fillna(EXPORT_END) - first["start_timestamp"]).dt.total_seconds() / 86400).clip(lower=1e-4)
returned = sec.notna().astype(int)
print(f"\nrebuilt: N={len(first)}, returned={returned.sum()} (notebook: 106/69)")

# ------------------------------------------- enriched first-session features
fs_ids = set(first["session_id"])
ev1 = events[events["session_id"].isin(fs_ids)]
pv1 = ev1[ev1["event"] == "$pageview"]
areas1 = pv1.groupby("session_id")["path_area"].agg(lambda s: set(s.dropna()))
rage1 = ev1[ev1["event"] == "$rageclick"].groupby("session_id").size()
exc1 = ev1[ev1["event"] == "$exception"].groupby("session_id").size()
navi_pv1 = pv1[pv1["path_area"] == "Navi"].groupby("session_id").size()

# launch clicks from raw csv (element_text not in pickled events)
launch = pd.read_csv(
    RAW / "events_master.csv",
    usecols=["session_id", "event_type", "element_text"],
    dtype={"session_id": "string", "event_type": "string", "element_text": "string"},
)
launch = launch[(launch["event_type"] == "click")
                & launch["element_text"].str.contains("Launch", case=False, na=False)]
launch_n = launch.groupby("session_id").size()
print("launch-click sessions total:", launch_n.shape[0])

E = pd.DataFrame(index=first.index)
E["fs_duration_log"] = np.log1p(first["session_duration_seconds"].fillna(0))
E["fs_pageviews_log"] = np.log1p(first["pageview_count"].fillna(0))
E["fs_clicks_log"] = np.log1p(first["autocapture_count"].fillna(0))
E["fs_is_bounce"] = first["is_bounce"].fillna(0).astype(int)
E["fs_meaningful"] = first["meaningful"].fillna(0).astype(int)
_a = first["session_id"].map(areas1)
for area in ["Documents", "Agents", "Results", "Factbook", "Navi"]:
    E[f"used_{area}"] = _a.apply(lambda s: int(isinstance(s, set) and area in s))
E["fs_breadth"] = _a.apply(lambda s: len(s) if isinstance(s, set) else 0)
E["launch_clicks"] = first["session_id"].map(launch_n).fillna(0)
E["launched_agents"] = (E["launch_clicks"] > 0).astype(int)
E["launch_clicks_log"] = np.log1p(E["launch_clicks"])
E["navi_pv_log"] = np.log1p(first["session_id"].map(navi_pv1).fillna(0))
E["fs_rageclicks"] = first["session_id"].map(rage1).fillna(0).clip(0, 10)
E["fs_exceptions"] = (first["session_id"].map(exc1).fillna(0) > 0).astype(int)

org_size = persons.groupby("auth0_org_id")["person_id"].nunique()
p2org = persons.set_index("person_id")["auth0_org_id"]
E["team_org"] = [int(org_size.get(p2org.get(pid), 1) > 1) for pid in E.index]
created = persons.set_index("person_id")["created_at"]
created = pd.to_datetime(created, utc=True, errors="coerce")
lag = (first["start_timestamp"] - created.reindex(E.index)).dt.total_seconds() / 86400
E["signup_lag_log"] = np.log1p(lag.clip(lower=0)).fillna(0)
E = E.drop(columns=["launch_clicks"])

# modelling frame = rebuilt cohort (validated in aggregate against surv)
df = E.copy()
df["days"] = days
df["returned"] = returned
print("\naggregate validation vs notebook surv frame:")
for f in COX_FEATURES:
    print(f"  {f}: rebuilt sum={int(df[f].sum())} vs notebook={int(surv[f].sum())}")
cph_check = CoxPHFitter().fit(df[["days", "returned"] + COX_FEATURES], "days", "returned")
print(f"  3-feature Cox on rebuilt frame: C={cph_check.concordance_index_:.3f} "
      f"HRs={np.exp(cph_check.params_).round(2).to_dict()}")

ENRICHED = ["fs_duration_log", "fs_pageviews_log", "fs_clicks_log", "fs_is_bounce",
            "fs_meaningful", "used_Documents", "used_Agents", "used_Results",
            "used_Factbook", "used_Navi", "fs_breadth", "launched_agents",
            "launch_clicks_log", "navi_pv_log", "fs_rageclicks", "fs_exceptions",
            "team_org", "signup_lag_log"]
dfe = df[["days", "returned"] + ENRICHED].astype(float)

for pen in [0.05, 0.1, 0.3, 0.5]:
    cv_cindex(dfe, ENRICHED, penalizer=pen, label=f"P1 Cox enriched (pen={pen})")

# ------------------------------------------------ P2 XGBoost survival:cox
def cv_xgb_cox(dfx, feats, n_splits=5, n_repeats=10):
    rkf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=RNG)
    scores = []
    X = dfx[feats].to_numpy()
    t = dfx["days"].to_numpy()
    e = dfx["returned"].to_numpy()
    ylab = np.where(e == 1, t, -t)  # xgb survival:cox: negative = censored
    for tr, te in rkf.split(X):
        m = XGBRegressor(objective="survival:cox", n_estimators=200, max_depth=2,
                         learning_rate=0.05, subsample=0.9, colsample_bytree=0.9,
                         min_child_weight=5, reg_lambda=2.0, tree_method="hist",
                         n_jobs=1, random_state=RNG)
        m.fit(X[tr], ylab[tr])
        risk = m.predict(X[te])  # predicted hazard ratio; higher = sooner return
        scores.append(concordance_index(t[te], -risk, e[te]))
    scores = np.array(scores)
    print(f"P2 XGB survival:cox: CV C-index = {scores.mean():.3f} ± {scores.std():.3f}")

cv_xgb_cox(dfe, ENRICHED)

# ------------------------------------ inference for best enriched Cox model
best_pen = 0.1
m = CoxPHFitter(penalizer=best_pen).fit(dfe[["days", "returned"] + ENRICHED], "days", "returned")
print(f"\nP1 enriched Cox (pen={best_pen}) in-sample C={m.concordance_index_:.3f}")
print(m.summary[["exp(coef)", "exp(coef) lower 95%", "exp(coef) upper 95%", "p"]].round(3).to_string())

rng = np.random.default_rng(RNG)
boot = {f: [] for f in ENRICHED}
for b in range(500):
    samp = dfe.iloc[rng.integers(0, len(dfe), len(dfe))]
    try:
        mb = CoxPHFitter(penalizer=best_pen).fit(samp[["days", "returned"] + ENRICHED], "days", "returned")
        for f in ENRICHED:
            boot[f].append(float(np.exp(mb.params_[f])))
    except Exception:
        continue
rows = []
for f in ENRICHED:
    a = np.array(boot[f])
    rows.append({"feature": f, "HR_med": np.median(a),
                 "lo": np.percentile(a, 2.5), "hi": np.percentile(a, 97.5),
                 "pct_gt1": (a > 1).mean()})
bs = pd.DataFrame(rows).sort_values("HR_med", ascending=False)
print(f"\nBootstrap HRs ({len(boot[ENRICHED[0]])} resamples):")
print(bs.round(3).to_string(index=False))
bs.to_csv(SCRATCH / "platform_bootstrap_hrs.csv", index=False)
dfe.to_pickle(SCRATCH / "platform_enriched_df.pkl")
print("DONE")
