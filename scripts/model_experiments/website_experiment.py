"""Website conversion model experiment.

M0 = faithful reproduction of the notebook's RF propensity model.
M1 = same RF, leakage-safe enriched features (acquisition + pre-form behaviour).
M2 = XGBoost on M1 features.
M3 = elastic-net logistic on M1 features (interpretable).

All models evaluated on the SAME 30% test visitors (split reproduces the
notebook's train_test_split(random_state=42, stratify=y)), plus repeated
stratified CV and paired bootstrap CIs for the delta vs M0.
"""
from pathlib import Path
import pickle
import warnings

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
SCRATCH = Path(__file__).parent
RNG = 42

d = pickle.load(open(SCRATCH / "website_dfs.pkl", "rb"))
prop = d["propensity_data"].reset_index(drop=True)
we = d["web_events"]
first = d["first_sessions"]
pv = d["pageviews"]
USE_CASE_PATHS = d["USE_CASE_PATHS"]
COMMERCIAL_PATHS = d["COMMERCIAL_PATHS"]
USE_CASE_MAP = d["USE_CASE_MAP"]

y = prop["converted"].astype(int)
print(f"visitors={len(prop)}, positives={y.sum()} ({y.mean()*100:.1f}%)")

# ---------------------------------------------------------------- M0 baseline
excluded = {
    "landing_pathname", "duration_seconds", "pageview_count", "unique_urls_visited",
    "is_bounce", "first_session_measured_seconds", "first_session_home_views",
    "first_session_home_seconds", "first_session_nav_hub_pageviews",
    "first_session_solutions_hub_views", "first_session_solutions_hub_seconds",
}
m0_cols = [c for c in prop.columns if c not in {"distinct_id", "converted", "cluster_id"} and c not in excluded]
X0 = prop[m0_cols]

def rf_pipeline(num_cols, cat_cols):
    return Pipeline([
        ("prep", ColumnTransformer([
            ("num", SimpleImputer(strategy="median"), num_cols),
            ("cat", Pipeline([
                ("imp", SimpleImputer(strategy="most_frequent")),
                ("ohe", OneHotEncoder(handle_unknown="ignore")),
            ]), cat_cols),
        ])),
        ("clf", RandomForestClassifier(
            n_estimators=400, min_samples_leaf=10, class_weight="balanced_subsample",
            random_state=RNG, n_jobs=-1)),
    ])

# ------------------------------------------------- M1 feature construction
# Cutoff = first /get-started pageview inside the first session (arriving at
# the form page is the outcome's doorstep; everything after it is post-funnel).
fs = first[["distinct_id", "session_id", "session_start", "session_end",
            "landing_pathname", "channel_type", "referring_domain",
            "device_type", "browser", "country"]].copy()
fs_ids = set(fs["session_id"])

ev1 = we[we["session_id"].isin(fs_ids)].copy()
ev1 = ev1.merge(fs[["session_id", "distinct_id", "session_start"]], on=["session_id", "distinct_id"], how="inner")

gs_pv = ev1[(ev1["event"] == "$pageview") & (ev1["pathname"] == "/get-started")]
cutoff = gs_pv.groupby("distinct_id")["event_timestamp"].min().rename("cutoff")
ev1 = ev1.merge(cutoff, on="distinct_id", how="left")
pre = ev1[ev1["cutoff"].isna() | (ev1["event_timestamp"] < ev1["cutoff"])].copy()

# dwell: seconds to next event in the same session, capped at 30 min
pre = pre.sort_values(["session_id", "event_timestamp"])
pre["next_ts"] = pre.groupby("session_id")["event_timestamp"].shift(-1)
pre["dwell_s"] = (pre["next_ts"] - pre["event_timestamp"]).dt.total_seconds().clip(0, 1800).fillna(0)

ppv = pre[pre["event"] == "$pageview"].copy()
ppv["is_uc"] = ppv["pathname"].isin(USE_CASE_PATHS)
ppv["is_comm"] = ppv["pathname"].isin(COMMERCIAL_PATHS)
ppv["uc_group"] = ppv["pathname"].map(USE_CASE_MAP)

def per_visitor(g):
    return pd.Series({
        "pre_pageviews": len(g),
        "pre_unique_paths": g["pathname"].nunique(),
        "pre_home_views": (g["pathname"] == "/").sum(),
        "pre_solutions_views": (g["pathname"] == "/solutions").sum(),
        "pre_platform_views": (g["pathname"] == "/platform").sum(),
        "pre_pricing_views": (g["pathname"] == "/pricing").sum(),
        "pre_api_views": (g["pathname"] == "/api").sum(),
        "pre_labs_views": (g["pathname"] == "/labs").sum(),
        "pre_about_views": (g["pathname"] == "/about-us").sum(),
        "pre_uc_pageviews": g["is_uc"].sum(),
        "pre_uc_breadth": g.loc[g["is_uc"], "pathname"].nunique(),
        "pre_uc_group_breadth": g["uc_group"].dropna().nunique(),
        "pre_comm_pageviews": g["is_comm"].sum(),
        "pre_comm_breadth": g.loc[g["is_comm"], "pathname"].nunique(),
        "pre_cross_journey": int(g["is_uc"].any() and g["is_comm"].any()),
        "pre_home_seconds": g.loc[g["pathname"] == "/", "dwell_s"].sum(),
        "pre_comm_seconds": g.loc[g["is_comm"], "dwell_s"].sum(),
        "pre_uc_seconds": g.loc[g["is_uc"], "dwell_s"].sum(),
    })

beh = ppv.groupby("distinct_id").apply(per_visitor, include_groups=False)

evagg = pre.groupby("distinct_id").agg(
    pre_events=("event", "size"),
    pre_clicks=("ac_event_type", lambda s: (s == "click").sum()),
    pre_dead_clicks=("event", lambda s: (s == "$dead_click").sum()),
    pre_rageclicks=("event", lambda s: (s == "$rageclick").sum()),
    last_pre_ts=("event_timestamp", "max"),
)

base = fs.set_index("distinct_id")
feat = pd.DataFrame(index=base.index)
feat = feat.join(beh).join(evagg)
feat["pre_duration_s"] = (feat["last_pre_ts"] - base["session_start"]).dt.total_seconds().clip(lower=0)
feat = feat.drop(columns=["last_pre_ts"])
num_cols_m1 = list(feat.columns)
feat[num_cols_m1] = feat[num_cols_m1].fillna(0)

# acquisition / context
feat["landed_on_get_started"] = (base["landing_pathname"] == "/get-started").astype(int)

def landing_group(p):
    if p == "/get-started": return "get-started"
    if p == "/": return "home"
    if p in USE_CASE_PATHS: return "use-case"
    if p in COMMERCIAL_PATHS: return "commercial"
    if p in ("/solutions", "/labs", "/about-us"): return "nav-hub"
    return "other"

feat["landing_group"] = base["landing_pathname"].map(landing_group).fillna("other")
feat["channel_type"] = base["channel_type"].fillna("Unknown")
rd = base["referring_domain"].fillna("$direct").astype(str).str.replace("www.", "", regex=False)
top_rd = rd.value_counts().head(8).index
feat["referrer_group"] = np.where(rd.isin(top_rd), rd, "other")
feat["device_type"] = base["device_type"].fillna("Unknown")
br = base["browser"].fillna("Unknown")
feat["browser_group"] = np.where(br.isin(br.value_counts().head(5).index), br, "other")
co = base["country"].fillna("Unknown")
feat["country_group"] = np.where(co.isin(co.value_counts().head(8).index), co, "other")
feat["visit_hour"] = base["session_start"].dt.hour
feat["visit_dow"] = base["session_start"].dt.dayofweek

# utm via person map
p2 = pd.read_parquet(SCRATCH / "parquet" / "website__persons_2.parquet")[
    ["person_id", "utm_source", "utm_medium"]]
idmap = we[["distinct_id", "person_id"]].dropna().drop_duplicates("distinct_id")
utm = idmap.merge(p2, on="person_id", how="left").set_index("distinct_id")

def utm_bucket(row):
    if pd.isna(row["utm_source"]): return "none"
    if str(row["utm_medium"]) == "newsletter": return "newsletter"
    return "other"

feat["utm_bucket"] = utm.apply(utm_bucket, axis=1).reindex(feat.index).fillna("none")

feat = feat.reset_index()
X1full = prop[["distinct_id"]].merge(feat, on="distinct_id", how="left")
assert len(X1full) == len(prop)
cat_cols_m1 = ["landing_group", "channel_type", "referrer_group", "device_type",
               "browser_group", "country_group", "utm_bucket"]
num_cols_all = [c for c in X1full.columns if c not in cat_cols_m1 + ["distinct_id"]]
X1 = X1full[num_cols_all + cat_cols_m1]
print(f"M1 features: {len(num_cols_all)} numeric + {len(cat_cols_m1)} categorical")
print("utm buckets:", feat["utm_bucket"].value_counts().to_dict())

# ------------------------------------------------------------------ models
def xgb_pipeline(num_cols, cat_cols, spw):
    return Pipeline([
        ("prep", ColumnTransformer([
            ("num", SimpleImputer(strategy="median"), num_cols),
            ("cat", Pipeline([
                ("imp", SimpleImputer(strategy="most_frequent")),
                ("ohe", OneHotEncoder(handle_unknown="ignore")),
            ]), cat_cols),
        ])),
        ("clf", XGBClassifier(
            n_estimators=500, learning_rate=0.05, max_depth=4, subsample=0.9,
            colsample_bytree=0.8, min_child_weight=5, reg_lambda=1.0,
            scale_pos_weight=spw, tree_method="hist", n_jobs=1,
            random_state=RNG, eval_metric="logloss")),
    ])

def logit_pipeline(num_cols, cat_cols):
    return Pipeline([
        ("prep", ColumnTransformer([
            ("num", Pipeline([
                ("imp", SimpleImputer(strategy="median")),
                ("sc", StandardScaler()),
            ]), num_cols),
            ("cat", Pipeline([
                ("imp", SimpleImputer(strategy="most_frequent")),
                ("ohe", OneHotEncoder(handle_unknown="ignore")),
            ]), cat_cols),
        ])),
        ("clf", LogisticRegression(
            penalty="elasticnet", solver="saga", l1_ratio=0.3, C=0.5,
            class_weight="balanced", max_iter=5000, random_state=RNG)),
    ])

spw = (len(y) - y.sum()) / y.sum()
models = {
    "M0 baseline RF (notebook features)": (X0, rf_pipeline(m0_cols, [])),
    "M1 enriched RF": (X1, rf_pipeline(num_cols_all, cat_cols_m1)),
    "M2 enriched XGBoost": (X1, xgb_pipeline(num_cols_all, cat_cols_m1, spw)),
    "M3 enriched logistic (elastic net)": (X1, logit_pipeline(num_cols_all, cat_cols_m1)),
}

# identical test visitors across models: split indices via y (same call signature)
idx = np.arange(len(y))
tr_idx, te_idx, _, _ = train_test_split(idx, y, test_size=0.30, random_state=RNG, stratify=y)

results, test_probas = [], {}
for name, (X, pipe) in models.items():
    pipe.fit(X.iloc[tr_idx], y.iloc[tr_idx])
    p = pipe.predict_proba(X.iloc[te_idx])[:, 1]
    test_probas[name] = p
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=RNG)
    cvres = cross_validate(pipe, X, y, cv=cv, scoring=["roc_auc", "average_precision"], n_jobs=4)
    results.append({
        "model": name,
        "test_AUC": roc_auc_score(y.iloc[te_idx], p),
        "test_AP": average_precision_score(y.iloc[te_idx], p),
        "test_Brier": brier_score_loss(y.iloc[te_idx], p),
        "cv_AUC_mean": cvres["test_roc_auc"].mean(),
        "cv_AUC_sd": cvres["test_roc_auc"].std(),
        "cv_AP_mean": cvres["test_average_precision"].mean(),
        "cv_AP_sd": cvres["test_average_precision"].std(),
    })
    r = results[-1]
    print(f"{name}: test AUC={r['test_AUC']:.3f} AP={r['test_AP']:.3f} | "
          f"CV AUC={r['cv_AUC_mean']:.3f}±{r['cv_AUC_sd']:.3f} AP={r['cv_AP_mean']:.3f}±{r['cv_AP_sd']:.3f}")

res_df = pd.DataFrame(results)
res_df.to_csv(SCRATCH / "website_model_results.csv", index=False)

# paired bootstrap for delta vs M0 on the same test visitors
rng = np.random.default_rng(RNG)
yte = y.iloc[te_idx].to_numpy()
base_p = test_probas["M0 baseline RF (notebook features)"]
print("\nPaired bootstrap (1000) deltas vs M0 on identical test visitors:")
for name, p in test_probas.items():
    if name.startswith("M0"):
        continue
    d_auc, d_ap = [], []
    for _ in range(1000):
        b = rng.integers(0, len(yte), len(yte))
        if yte[b].sum() in (0, len(yte)):
            continue
        d_auc.append(roc_auc_score(yte[b], p[b]) - roc_auc_score(yte[b], base_p[b]))
        d_ap.append(average_precision_score(yte[b], p[b]) - average_precision_score(yte[b], base_p[b]))
    d_auc, d_ap = np.array(d_auc), np.array(d_ap)
    print(f"  {name}: dAUC={d_auc.mean():+.3f} [{np.percentile(d_auc,2.5):+.3f},{np.percentile(d_auc,97.5):+.3f}] "
          f"dAP={d_ap.mean():+.3f} [{np.percentile(d_ap,2.5):+.3f},{np.percentile(d_ap,97.5):+.3f}]")

# interpretation: permutation importance for best model + logistic odds ratios
from sklearn.inspection import permutation_importance
best_name = max((r for r in results if not r["model"].startswith("M0")), key=lambda r: r["cv_AP_mean"])["model"]
Xb, pipe_b = models[best_name]
pipe_b.fit(Xb.iloc[tr_idx], y.iloc[tr_idx])
perm = permutation_importance(pipe_b, Xb.iloc[te_idx], y.iloc[te_idx],
                              n_repeats=10, random_state=RNG, scoring="average_precision", n_jobs=4)
imp = pd.DataFrame({"feature": Xb.columns, "importance": perm.importances_mean}).sort_values("importance", ascending=False)
print(f"\nPermutation importance (AP) — {best_name}, top 15:")
print(imp.head(15).to_string(index=False))
imp.to_csv(SCRATCH / "website_best_perm_importance.csv", index=False)

logit = models["M3 enriched logistic (elastic net)"][1]
logit.fit(X1.iloc[tr_idx], y.iloc[tr_idx])
names = logit.named_steps["prep"].get_feature_names_out()
coefs = logit.named_steps["clf"].coef_[0]
ors = pd.DataFrame({"feature": names, "coef": coefs, "OR": np.exp(coefs)})
ors = ors.reindex(ors["coef"].abs().sort_values(ascending=False).index)
print("\nLogistic odds ratios (per SD for numeric), top 20:")
print(ors.head(20).to_string(index=False))
ors.to_csv(SCRATCH / "website_logit_odds_ratios.csv", index=False)

# funnel decomposition readout
reached_ever = we[(we["event"] == "$pageview") & (we["pathname"] == "/get-started")]["distinct_id"].unique()
r1 = prop["distinct_id"].isin(set(reached_ever)).astype(int)
print(f"\nFunnel: reached /get-started ever = {r1.sum()} ({r1.mean()*100:.1f}%), "
      f"converted | reached = {y[r1 == 1].mean()*100:.1f}%, converted overall = {y.mean()*100:.1f}%")
print("DONE")
