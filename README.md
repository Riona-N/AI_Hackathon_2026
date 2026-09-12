# AI-Based Urban Water-Quality Monitoring Framework — Integrated Build

This is the three team branches (prediction/explainability, spatial/temporal,
anomaly/uncertainty) merged into one working repo, per `project-plan.md`'s
integration contract.

## Run it

```bash
pip install -r requirements.txt
python run_pipeline.py
```

Everything (predictions, map, reports) is written to `outputs/` and the
trained model to `models/`. No manual steps or notebook-clicking required.

## What's in here

```
data/raw/water_quality.csv     - the sample dataset (see note below)
src/config.py                  - shared paths, target column, thresholds
src/data/                      - cleaning/pivoting + robust split helpers
src/models/                    - candidate models, CV, training, predict.py
src/explainability/            - feature ranking + SHAP
src/spatial/                   - risk classification, hotspot map
src/temporal/                  - time grouping, hotspot evolution tracking
src/anomaly/                   - residuals, thresholds, flagging, stability
run_pipeline.py                - orchestrates all of the above, end to end
notebooks/                     - original member notebooks, kept for reference
```

## What changed to make the three pieces fit together

1. **Removed** `anomaly-explainability` branch's old `src/models/train.py`.
   It trained an unrelated SMOTE + SVM *classifier* for a "water-potability"
   task, but Member 1's actual model is a *regressor* predicting River
   Pollution Index — a different problem entirely. It also depended on
   `imbalanced-learn`, which nothing else in the project needs. It's been
   replaced by a real `src/models/train.py` (Random Forest / Gradient
   Boosting / Linear Regression candidates, extracted and generalized from
   `feat1and4.ipynb`).
2. **Added `src/config.py`** — a single place for the target column,
   metadata columns, the shared prediction-output schema, and the anomaly /
   risk thresholds, so the three branches can't silently drift out of sync
   (this was called out as a to-do in `project-plan.md` section 5).
3. **Added `src/data/`, `src/models/`, `src/explainability/`** as real,
   importable modules (pulled out of `feat1and4.ipynb`, which only worked as
   a Colab notebook with `!pip install` and `files.upload()` — not something
   the other branches could import from).
4. **Made the small-data path safe instead of crashing.** The bundled
   sample CSV only has 2 site/date readings with complete (non-placeholder)
   values, which breaks the "80/20 split + 10-fold CV" assumptions in the
   original plan. `src/data/split.py` now falls back to smaller CV folds /
   an in-sample holdout when there isn't enough data, and always says so in
   its output rather than silently producing a misleading number or
   throwing. Drop a larger CSV into `data/raw/water_quality.csv` and it'll
   automatically use the full 80/20 + 10-fold path.
5. **Fixed a stale docstring** in `src/anomaly/flagging.py` that referenced
   a `final_svm` model that doesn't exist in this pipeline; updated to
   describe the actual regressor handoff.
6. **Dropped `imbalanced-learn`, `xgboost`, `catboost`** from requirements
   (only used by the removed/unused code paths) in favor of `scikit-learn`
   models already exercised by the pipeline, so `pip install -r
   requirements.txt` doesn't pull in extra weight for no benefit.
7. Kept both original notebooks under `notebooks/` for reference/history,
   but they are no longer where the "real" code lives — `run_pipeline.py`
   plus the modules under `src/` are.

## About accuracy

The sample dataset only has 2 usable labeled rows (the rest are placeholder
`-` values for all measurements), so R²/MAE/RMSE numbers here are not
meaningful — this is a plumbing demo, not a tuned model. The pipeline is
written so none of that breaks anything: it clearly labels degraded-data
situations in its printed output and in `outputs/final_report.md` rather
than pretending the numbers are trustworthy. Swap in a bigger CSV with the
same columns and every stage (CV, holdout split, SHAP, hotspot evolution,
anomaly stability) scales up automatically.
