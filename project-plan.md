# AI-Based Urban Water-Quality Monitoring Framework
### Repository Structure, Branching Strategy & Team Task Allocation

---

## 1. Repository Structure

```
water-quality-ai/
│
├── data/
│   ├── raw/                     # Original, untouched datasets
│   ├── interim/                 # Partially cleaned data
│   ├── processed/               # Final model-ready data
│   └── external/                # Geo boundaries, shapefiles, reference data
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_model_training.ipynb
│   ├── 04_spatial_hotspots.ipynb
│   ├── 05_anomaly_detection.ipynb
│   ├── 06_explainability.ipynb
│   └── 07_temporal_uncertainty.ipynb
│
├── src/
│   ├── __init__.py
│   ├── config.py                # Paths, constants, thresholds
│   │
│   ├── data/
│   │   ├── preprocessing.py     # Cleaning, missing-value handling
│   │   └── split.py             # Train/test split, CV folds
│   │
│   ├── models/
│   │   ├── train.py             # RF, XGBoost, TabNet training
│   │   ├── evaluate.py          # R², RMSE, MAE, model comparison
│   │   ├── select_best.py
│   │   └── predict.py
│   │
│   ├── spatial/
│   │   ├── hotspot_classifier.py   # Risk-level classification
│   │   ├── geo_utils.py            # Lat/long handling, geocoding
│   │   └── mapping.py              # Map generation (folium/plotly)
│   │
│   ├── anomaly/
│   │   ├── residuals.py         # Observed − predicted
│   │   ├── threshold.py         # Anomaly threshold logic
│   │   └── flagging.py          # Flag + report abnormal events
│   │
│   ├── explainability/
│   │   ├── shap_analysis.py
│   │   └── feature_ranking.py
│   │
│   └── temporal/
│       ├── time_grouping.py     # Group by year/season/month
│       ├── trend_tracking.py    # Persistent/emerging hotspots
│       └── uncertainty.py       # Error/uncertainty over time
│
├── app/ or dashboard/           # Optional: Streamlit/Dash app for maps + alerts
│   ├── app.py
│   └── components/
│
├── reports/
│   ├── figures/
│   └── final_report.md
│
├── models/                      # Saved model artifacts (.pkl, .json)
│
├── tests/                       # Unit tests per module
│
├── requirements.txt
├── environment.yml
├── README.md
└── .gitignore
```

---

## 2. Branching Strategy

```
main                → stable, demo-ready code only
 └── develop         → integration branch, merged via PRs

 ├── feature/prediction-model      (Member 1)
 ├── feature/spatial-temporal      (Member 2)
 └── feature/anomaly-explainability (Member 3)
```

**Workflow:**
- Each member branches off `develop`, never off `main`.
- Merge into `develop` via pull request + at least one teammate review.
- `main` is updated only at milestones (e.g., after integration testing).
- Naming convention: `feature/<short-task-name>`, e.g. `feature/rf-xgboost-cv`, `feature/hotspot-map`, `feature/shap-ranking`.

---

## 3. Feature → Member Allocation

The 5 features aren't equal in size, and several depend on the trained model, so grouping them by *dependency chain* rather than splitting evenly keeps handoffs simple.

### 🧑‍💻 Member 1 — Prediction Core
**Branch:** `feature/prediction-model`
**Owns: Feature 1 (Prediction Model) + Feature 4 (Explainability)**

| Task |
|---|
| Preprocess dataset, handle missing values |
| Train/test split, 10-fold CV |
| Train RF, XGBoost, TabNet (or similar) |
| Compare R², RMSE, MAE → select best model |
| Produce final prediction pipeline (`predict.py`) used by everyone else |
| SHAP / feature-importance analysis on the selected model |
| Rank & display top influencing parameters per prediction |

*Why paired:* Explainability is applied directly to the trained model, so the person who built it is best placed to interpret it — no handoff delay.

---

### 🧑‍💻 Member 2 — Spatial Intelligence
**Branch:** `feature/spatial-temporal`
**Owns: Feature 2 (Spatial Hotspot Detection) + the spatial half of Feature 5**

| Task |
|---|
| Use lat/long or site metadata to geolocate predictions |
| Feed in Member 1's predicted pollution values |
| Classify locations into risk levels (low/medium/high) |
| Build interactive map (folium/plotly/kepler.gl) |
| Add time slider to show hotspots evolving |
| Group observations by time period (year/season) |
| Track pollution/prediction trends over time |
| Detect persistent vs. emerging hotspots |

*Why paired:* Hotspot mapping and hotspot persistence-over-time are the same visualization problem with a time axis added — natural single owner.

---

### 🧑‍💻 Member 3 — Anomaly & Reliability
**Branch:** `feature/anomaly-explainability`
**Owns: Feature 3 (Anomaly Detection) + the uncertainty half of Feature 5**

| Task |
|---|
| Use Member 1's predictions to compute residuals (observed − predicted) |
| Define anomaly threshold (statistical or domain-based) |
| Flag abnormal observations, generate pollution-event report |
| Measure prediction uncertainty/error (e.g., CV error spread, prediction intervals) |
| Check whether flagged anomalies stay stable across years |
| Feed anomaly stability results back into Member 2's temporal map (optional overlay) |

*Why paired:* Residual-based anomaly logic and uncertainty quantification both come from the same error analysis — one person keeps it consistent.

---

## 4. Dependency Order (recommended timeline)

```
Week 1–2:  Member 1 builds preprocessing + baseline models     (others start scaffolding/EDA)
Week 2–3:  Member 1 finalizes best model + predict.py           → shared with team
Week 3–5:  Member 2 & 3 build on top of predictions in parallel
Week 4–5:  Member 1 runs SHAP on finalized model
Week 5–6:  Integration on `develop`, cross-testing, dashboard assembly
Week 6:    Merge to `main`, final report + demo
```

Because Features 2, 3, 4, and 5 all consume Member 1's model output, it's worth Member 1 shipping a minimal working `predict.py` early (even before full model tuning is done) so Members 2 and 3 aren't blocked.

---

## 5. Integration Notes

- Agree on a shared prediction output schema early (e.g., columns: `site_id, lat, lon, timestamp, observed, predicted, model_version`) — everything downstream depends on this contract.
- Keep `config.py` shared and version-controlled so thresholds (anomaly cutoff, risk-level bins) aren't hardcoded differently across branches.
- Weekly sync to merge into `develop` and catch schema drift early.
