# Model-strengthening experiments (July 2026)

Standalone experiments that reproduce each notebook's predictive baseline on
identical data and then demonstrate stronger, leakage-safe alternatives.
Tied to the two research objectives:

1. **Conversion** — what causes user conversion on the website and demo.
2. **Stickiness** — what makes users return to the main platform.

## How to regenerate

All scripts expect two pickle/parquet caches produced first (paths are
relative to this directory):

```bash
# 1. Parquet cache of the website/demo Excel exports (creates ./parquet/)
.venv/bin/python convert_to_parquet.py

# 2. Execute each notebook headlessly and pickle its exact modelling frames
.venv/bin/python extract_dataframes.py "notebooks/Website Notebook.ipynb" \
    scripts/model_experiments/website_dfs.pkl \
    propensity_data web_events web_sessions first_sessions pageviews \
    USE_CASE_MAP USE_CASE_PATHS COMMERCIAL_PATHS
.venv/bin/python extract_dataframes.py "notebooks/Demo Notebook.ipynb" \
    scripts/model_experiments/demo_dfs.pkl \
    _vdf demo_events demo_ac demo_sessions cta_clicks _signup_ids _combined PROJECT_NAMES
.venv/bin/python extract_dataframes.py "notebooks/Main Platform Notebook.ipynb" \
    scripts/model_experiments/platform_dfs.pkl \
    mdf surv sessions persons events FEATURES COX_FEATURES

# 3. Run the experiments
.venv/bin/python website_experiment.py     # M0 baseline vs M1-M3 enriched
.venv/bin/python website_sensitivity.py    # S1/S2: lift survives w/o get-started info
.venv/bin/python demo_experiment.py        # B0 repro vs honest LOOCV pipeline
.venv/bin/python platform_experiment.py    # P0 repro + CV, enriched Cox, XGB-Cox
.venv/bin/python platform_refine.py        # compact Cox variants V0-V4
```

(Note: `convert_to_parquet.py` and `extract_dataframes.py` were written with
scratchpad output paths; point their output at this directory when
regenerating, or edit the `SCRATCH`/pickle paths at the top of each
experiment script.)

## Headline results

| Surface | Notebook baseline | Improved (honest) | Notes |
|---|---|---|---|
| Website | RF AUC 0.576 / AP 0.079 | XGB **AUC 0.936 / AP 0.596** (CV 0.926±0.012) | leakage-safe pre-form features + acquisition context |
| Website (hard mode) | — | AUC 0.885 / CV 0.842 | excluding all 1,104 direct-to-form landers |
| Demo | synthetic-trained XGB, real AUC 0.38–0.40 (below chance) | L2 logistic LOOCV **AUC 0.775**, perm p=0.005 | pre-CTA censored features; CI wide (5 positives) |
| Platform | Cox C 0.646 (in-sample only) | honest CV C 0.622±0.080; best refinement V2 CV C **0.635** | enrichment beyond ~4 features hurts at N=106 |

## Key findings

- **Demo notebook bug (cell 46)**: `reached_*` flags compare `page_type`
  (path templates like `/projects/{id}/results`) against display labels
  ("Results"); they never match, so all six flags and `engagement_depth`
  are constant 0/False for all 207 visitors. The "area flags have zero SHAP
  impact" conclusion is an artifact of this bug.
- **Website**: 75% of converters (395/527) land directly on /get-started —
  conversion is acquisition-led. Funnel: 18.3% reach the form page; 36.2%
  of reachers engage the form.
- **Demo leakage**: original features counted the sign-up click itself and
  post-click sessions; censored features lose little signal (LOOCV 0.775 vs
  0.790 uncensored), so the synthetic-training approach — not leakage — was
  the main problem.
- **Platform**: `launch_clicks_log` HR 1.63 [1.04, 3.14] (intensity beats
  the binary flag), first-session rageclicks HR 1.16 [1.01, 1.60], team_org
  HR 1.80 [1.05, 3.06], used_Navi HR 0.62 [0.36, 1.06]. `signup_lag_log` is
  degenerate (drop it).
