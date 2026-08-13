"""Demo conversion model experiment (5 positives / 207 visitors).

B0: reproduce the notebook's synthetic-trained XGBoost -> real-validation AUC.
H1: honest small-N pipeline — leakage-safe features (censored at first CTA
    click), L2 logistic, leave-one-out CV, permutation-test p-value,
    bootstrap CI, precision@k.
H2: same honest pipeline but with the ORIGINAL (uncensored) features, to
    quantify how much apparent signal is post-outcome contamination.
"""
from pathlib import Path
import pickle
import warnings

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import LeaveOneOut, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
SCRATCH = Path(__file__).parent
RNG = 42

d = pickle.load(open(SCRATCH / "demo_dfs.pkl", "rb"))
vdf = d["_vdf"]
combined = d["_combined"]
de = d["demo_events"]
ds = d["demo_sessions"]
cta = d["cta_clicks"]
signup_ids = d["_signup_ids"]

print("vdf index sample:", list(vdf.index[:3]))
print("event types in demo_events:", de["event"].value_counts().to_dict())
visitors = vdf.index
y = vdf["clicked_signup"].astype(int)
print(f"N={len(visitors)}, positives={y.sum()}")

# ------------------------------------------------ B0: notebook reproduction
_FEAT_COLS = [
    "project_name", "demo_vertical", "n_sessions", "mean_dur", "max_dur",
    "total_pv", "total_clicks", "all_bounce", "any_bounce", "has_referrer",
    "reached_results", "reached_result_detail", "reached_agents", "reached_agent_run",
    "reached_documents", "reached_doc_detail", "engagement_depth", "rageclick_count",
    "device_type", "country", "visit_hour", "visit_dow",
]
_CAT = ["project_name", "demo_vertical", "device_type", "country"]
enc = combined[_FEAT_COLS + ["clicked_signup", "_source"]].copy()
for c in _CAT:
    enc[c] = LabelEncoder().fit_transform(enc[c].fillna("Unknown").astype(str))
num = [c for c in _FEAT_COLS if c not in _CAT]
enc[num] = enc[num].fillna(enc[num].median())
X_syn = enc.loc[enc["_source"] == "synthetic", _FEAT_COLS].astype("float32")
y_syn = enc.loc[enc["_source"] == "synthetic", "clicked_signup"].astype(int)
X_real = enc.loc[enc["_source"] == "real", _FEAT_COLS].astype("float32")
y_real = enc.loc[enc["_source"] == "real", "clicked_signup"].astype(int)
X_fit, X_hold, y_fit, y_hold = train_test_split(X_syn, y_syn, test_size=0.25, random_state=RNG, stratify=y_syn)
xgb = XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.8,
                    colsample_bytree=0.8, scale_pos_weight=17, eval_metric="logloss",
                    tree_method="hist", n_jobs=1, random_state=RNG)
xgb.fit(X_fit, y_fit)
print(f"\nB0 notebook repro: synthetic holdout AUC={roc_auc_score(y_hold, xgb.predict_proba(X_hold)[:,1]):.3f}, "
      f"REAL validation AUC={roc_auc_score(y_real, xgb.predict_proba(X_real)[:,1]):.3f}")

# ------------------------------------- leakage-safe (pre-CTA) feature build
cutoff = cta.groupby("distinct_id")["event_timestamp"].min()

de = de.copy()
de["cutoff"] = de["distinct_id"].map(cutoff)
pre_ev = de[de["cutoff"].isna() | (de["event_timestamp"] < de["cutoff"])]

AREA_FLAGS = {
    "reached_results": ["/projects/{id}/results"],
    "reached_result_detail": ["/projects/{id}/results/{result_id}"],
    "reached_agents": ["/projects/{id}/agents"],
    "reached_agent_run": ["/projects/{id}/agents/{agent_id}/{run_id}", "/projects/{id}/agents/{agent_id}"],
    "reached_documents": ["/projects/{id}/documents"],
    "reached_doc_detail": ["/projects/{id}/documents/{doc_id}"],
}

ss = ds.copy()
ss["cutoff"] = ss["distinct_id"].map(cutoff)
ss = ss[ss["cutoff"].isna() | (ss["start_timestamp"] < ss["cutoff"])]
end_eff = ss[["end_timestamp", "cutoff"]].min(axis=1)
ss["dur_pre"] = (end_eff - ss["start_timestamp"]).dt.total_seconds().clip(lower=0)

pv_pre = pre_ev[pre_ev["event"] == "$pageview"].groupby("distinct_id").size()
clicks_pre = pre_ev[(pre_ev["event"] == "$autocapture") & (pre_ev["event_type"] == "click")].groupby("distinct_id").size()
rage_pre = pre_ev[pre_ev["event"] == "$rageclick"].groupby("distinct_id").size()

feat = pd.DataFrame(index=visitors)
feat["n_sessions"] = ss.groupby("distinct_id").size().reindex(visitors).fillna(0)
feat["mean_dur"] = ss.groupby("distinct_id")["dur_pre"].mean().reindex(visitors).fillna(0)
feat["max_dur"] = ss.groupby("distinct_id")["dur_pre"].max().reindex(visitors).fillna(0)
feat["total_pv"] = pv_pre.reindex(visitors).fillna(0)
feat["total_clicks"] = clicks_pre.reindex(visitors).fillna(0)
feat["rageclick_count"] = rage_pre.reindex(visitors).fillna(0)
areas_pre = pre_ev.groupby("distinct_id")["page_type"].agg(lambda s: set(s.dropna()))
depth = areas_pre.apply(lambda s: sum(any(p in s for p in paths) for paths in AREA_FLAGS.values()))
feat["engagement_depth"] = depth.reindex(visitors).fillna(0)
feat["has_referrer"] = vdf["has_referrer"].fillna(0).astype(int)

HONEST_FEATS = list(feat.columns)
print("\nleakage-safe features:", HONEST_FEATS)
print("\nconverters pre-CTA vs original (means):")
cmp = pd.DataFrame({
    "orig_conv": vdf.loc[y == 1, ["n_sessions", "mean_dur", "max_dur", "total_pv", "total_clicks", "engagement_depth"]].mean(),
    "pre_conv": feat.loc[y == 1, ["n_sessions", "mean_dur", "max_dur", "total_pv", "total_clicks", "engagement_depth"]].mean(),
    "nonconv": feat.loc[y == 0, ["n_sessions", "mean_dur", "max_dur", "total_pv", "total_clicks", "engagement_depth"]].mean(),
})
print(cmp.round(1).to_string())

# ------------------------------------------------------- honest LOOCV eval
def loocv_scores(X, yv, seed=RNG):
    Xm = X.to_numpy(dtype=float)
    yv = np.asarray(yv)
    oof = np.zeros(len(yv))
    for tr, te in LeaveOneOut().split(Xm):
        sc = StandardScaler().fit(Xm[tr])
        clf = LogisticRegression(C=0.5, class_weight="balanced", max_iter=2000, random_state=seed)
        clf.fit(sc.transform(Xm[tr]), yv[tr])
        oof[te] = clf.predict_proba(sc.transform(Xm[te]))[:, 1]
    return oof

def report(oof, yv, label):
    auc = roc_auc_score(yv, oof)
    ap = average_precision_score(yv, oof)
    u, p_mw = mannwhitneyu(oof[yv == 1], oof[yv == 0], alternative="greater")
    order = np.argsort(-oof)
    p10 = int(yv[order[:10]].sum()); p20 = int(yv[order[:20]].sum())
    rng = np.random.default_rng(RNG)
    aucs = []
    for _ in range(2000):
        b = rng.integers(0, len(yv), len(yv))
        if yv[b].sum() in (0, len(yv)):
            continue
        aucs.append(roc_auc_score(yv[b], oof[b]))
    lo, hi = np.percentile(aucs, [2.5, 97.5])
    print(f"{label}: LOOCV AUC={auc:.3f} [boot {lo:.3f},{hi:.3f}] AP={ap:.3f} "
          f"MW p={p_mw:.4f} | top10 contains {p10}/5, top20 contains {p20}/5")
    return auc

yv = y.to_numpy()
oof_h = loocv_scores(feat[HONEST_FEATS], yv)
auc_h = report(oof_h, yv, "H1 honest (pre-CTA censored)")

orig_num = vdf[["n_sessions", "mean_dur", "max_dur", "total_pv", "total_clicks",
                "rageclick_count", "engagement_depth", "has_referrer"]].astype(float).fillna(0)
oof_o = loocv_scores(orig_num, yv)
auc_o = report(oof_o, yv, "H2 honest eval, UNCENSORED features")

# label-permutation test for H1 (LOOCV inside each permutation)
rng = np.random.default_rng(RNG)
n_perm = 200
null_aucs = []
for i in range(n_perm):
    yp = rng.permutation(yv)
    if yp.sum() == 0:
        continue
    null_aucs.append(roc_auc_score(yp, loocv_scores(feat[HONEST_FEATS], yp)))
null_aucs = np.array(null_aucs)
p_perm = (1 + (null_aucs >= auc_h).sum()) / (1 + len(null_aucs))
print(f"\nPermutation test (n={len(null_aucs)}): null AUC mean={null_aucs.mean():.3f} "
      f"sd={null_aucs.std():.3f}, p(perm >= observed)={p_perm:.3f}")

# interpretation: full-data standardized coefficients + univariate tests
sc = StandardScaler().fit(feat[HONEST_FEATS])
clf = LogisticRegression(C=0.5, class_weight="balanced", max_iter=2000, random_state=RNG)
clf.fit(sc.transform(feat[HONEST_FEATS]), yv)
coefs = pd.DataFrame({"feature": HONEST_FEATS, "std_coef": clf.coef_[0]})
uni = []
for c in HONEST_FEATS:
    u, p = mannwhitneyu(feat.loc[y == 1, c], feat.loc[y == 0, c], alternative="two-sided")
    uni.append(p)
coefs["MW_p"] = uni
coefs = coefs.reindex(coefs["std_coef"].abs().sort_values(ascending=False).index)
print("\nStandardized L2-logistic coefficients (full data) + univariate Mann-Whitney p:")
print(coefs.to_string(index=False))
coefs.to_csv(SCRATCH / "demo_honest_coefs.csv", index=False)

pd.DataFrame({"oof_honest": oof_h, "oof_uncensored": oof_o, "y": yv},
             index=visitors).to_csv(SCRATCH / "demo_oof_scores.csv")
print("DONE")
