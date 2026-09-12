"""Train candidate regressors for River Pollution Index and pick the best one.

This replaces the anomaly branch's old `train.py`, which trained an unrelated
SMOTE + SVM *classifier* for a "water-potability" task that Member 1 never
actually built (Member 1's real model is a regressor predicting RPI). Kept
dependency-light (sklearn only) so the pipeline runs anywhere without extra
installs like xgboost/catboost/imbalanced-learn.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline

from src.config import RANDOM_STATE
from src.data.split import safe_cv


def build_candidate_models(random_state: int = RANDOM_STATE) -> dict[str, Pipeline]:
    models = {
        "Random Forest": RandomForestRegressor(
            n_estimators=200, max_depth=6, min_samples_leaf=1,
            random_state=random_state, n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=100, learning_rate=0.05, max_depth=2,
            random_state=random_state,
        ),
        "Linear Regression": LinearRegression(),
    }
    return {
        name: Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", model)])
        for name, model in models.items()
    }


def cross_validate_models(X, y, n_splits: int = 10, random_state: int = RANDOM_STATE):
    """Compare candidate models with CV, degrading gracefully on tiny data."""
    cv, cv_note = safe_cv(len(X), n_splits=n_splits, random_state=random_state)
    rows = []
    for name, pipeline in build_candidate_models(random_state).items():
        if cv is None:
            rows.append({"Model": name, "R2": np.nan, "RMSE": np.nan, "MAE": np.nan})
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            scores = cross_validate(
                pipeline, X, y, cv=cv,
                scoring={
                    "R2": "r2",
                    "RMSE": "neg_root_mean_squared_error",
                    "MAE": "neg_mean_absolute_error",
                },
                n_jobs=-1,
                error_score=np.nan,
            )
        rows.append({
            "Model": name,
            "R2": np.nanmean(scores["test_R2"]),
            "RMSE": -np.nanmean(scores["test_RMSE"]),
            "MAE": -np.nanmean(scores["test_MAE"]),
        })
    cv_results_df = pd.DataFrame(rows).sort_values("R2", ascending=False, na_position="last")
    return cv_results_df.reset_index(drop=True), cv_note


def select_best_model_name(cv_results_df: pd.DataFrame) -> str:
    ranked = cv_results_df.dropna(subset=["R2"])
    if ranked.empty:
        # No CV signal available (too little data) - default to the most
        # stable/least overfit-prone option.
        return "Random Forest"
    return ranked.iloc[0]["Model"]


def train_final_model(X, y, model_name: str, random_state: int = RANDOM_STATE) -> Pipeline:
    candidates = build_candidate_models(random_state)
    pipeline = candidates[model_name]
    pipeline.fit(X, y)
    return pipeline


def evaluate_holdout(model: Pipeline, X_test, y_test) -> dict:
    """Compute holdout metrics, guarding against the degenerate 1-row case
    where R2 is undefined."""
    y_pred = model.predict(X_test)
    metrics = {
        "mae": float(mean_absolute_error(y_test, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
    }
    try:
        metrics["r2"] = float(r2_score(y_test, y_pred)) if len(y_test) >= 2 else None
    except ValueError:
        metrics["r2"] = None
    return metrics, y_pred
