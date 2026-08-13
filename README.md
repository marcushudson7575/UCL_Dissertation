# UCL Dissertation — Parsewise User-Behaviour Analytics

MSc Business Analytics dissertation (UCL) analysing user behaviour for **Parsewise**, an
early-stage agentic-AI B2B company. The project follows three stages of the customer journey —
the marketing **website**, the interactive **demo**, and the live **platform** — to identify
what drives conversion and retention, using event data from PostHog.

## What's in this repo

- **`notebooks/`** — the three analysis notebooks, one per dataset:
  - `Website Notebook.ipynb` — acquisition, engagement and conversion modelling
  - `Demo Notebook.ipynb` — demo routing, engagement and sign-up prediction
  - `Main Platform Notebook.ipynb` — platform usage and retention (Kaplan–Meier / Cox survival analysis)
- **`files/Diss/SQL queries used to extract data/`** — the SQL used to pull the raw data from PostHog
- **`scripts/model_experiments/`** — supplementary model code that backs some figures not built
  inside the notebooks (e.g. the website elastic-net odds ratios and the refined platform Cox model)
- **`outputs (visualisations etc)/`** — all generated figures, tables and CSV summaries
- **`requirements.txt`** — the Python packages needed to run the project

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Python 3.12 recommended. Key packages: `pandas`, `scikit-learn`, `xgboost`, `shap`, `sdv`,
`lifelines`, `plotly`, `matplotlib`, `seaborn` (full list in `requirements.txt`).
On macOS, `xgboost` also needs `libomp` (`brew install libomp`).

## Running

Open the notebooks in Jupyter and run them top to bottom. Generated figures are written to the
outputs folder.

## Data

The raw event data is Parsewise's private, first-party data and is **not included** in this
repository. The committed notebooks, scripts and outputs therefore document the analysis and its
results, but the notebooks cannot be re-run end to end without the underlying PostHog exports.
