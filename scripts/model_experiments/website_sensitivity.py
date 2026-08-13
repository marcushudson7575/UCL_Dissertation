"""Sensitivity: does the website lift survive removing direct-to-form landing info?
S1: drop landed_on_get_started + collapse landing_group 'get-started' -> 'other'.
S2: additionally EXCLUDE visitors who landed on /get-started entirely.
"""
from pathlib import Path
import pickle
import warnings

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
SCRATCH = Path(__file__).parent
RNG = 42

ns = {"__file__": str(SCRATCH / "website_experiment.py")}
exec(open(SCRATCH / "website_experiment.py").read().split("# ------------------------------------------------------------------ models")[0],
     ns)
X1, y, prop = ns["X1"], ns["y"], ns["prop"]
num_cols_all, cat_cols_m1 = ns["num_cols_all"], ns["cat_cols_m1"]

print("landing_group counts:", X1["landing_group"].value_counts().to_dict())
print("landed_on_get_started=1:", int(X1["landed_on_get_started"].sum()),
      "of whom converted:", int(y[X1["landed_on_get_started"] == 1].sum()))

def xgb_pipe(num_cols, cat_cols, spw):
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

def evaluate(X, yv, label):
    spw = (len(yv) - yv.sum()) / yv.sum()
    num = [c for c in X.columns if c not in cat_cols_m1]
    cat = [c for c in X.columns if c in cat_cols_m1]
    pipe = xgb_pipe(num, cat, spw)
    idx = np.arange(len(yv))
    tr, te, _, _ = train_test_split(idx, yv, test_size=0.30, random_state=RNG, stratify=yv)
    pipe.fit(X.iloc[tr], yv.iloc[tr])
    p = pipe.predict_proba(X.iloc[te])[:, 1]
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=RNG)
    cvres = cross_validate(pipe, X, yv, cv=cv, scoring=["roc_auc", "average_precision"], n_jobs=4)
    print(f"{label}: N={len(yv)} pos={int(yv.sum())} | test AUC={roc_auc_score(yv.iloc[te], p):.3f} "
          f"AP={average_precision_score(yv.iloc[te], p):.3f} | "
          f"CV AUC={cvres['test_roc_auc'].mean():.3f}±{cvres['test_roc_auc'].std():.3f} "
          f"AP={cvres['test_average_precision'].mean():.3f}±{cvres['test_average_precision'].std():.3f}")

# S1: hide direct-to-form info
XS1 = X1.drop(columns=["landed_on_get_started"]).copy()
XS1["landing_group"] = XS1["landing_group"].replace("get-started", "other")
evaluate(XS1, y, "S1 XGB no get-started landing info")

# S2: exclude direct-to-form landers entirely
mask = X1["landed_on_get_started"] == 0
XS2 = XS1.loc[mask].reset_index(drop=True)
yS2 = y.loc[mask].reset_index(drop=True)
evaluate(XS2, yS2, "S2 XGB excluding direct-to-form landers")
print("DONE")
