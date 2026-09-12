"""Load the trained model and produce predictions in the shared schema.

Downstream branches (spatial, temporal, anomaly) all consume the output of
`predict_dataframe`, so its column names must match `config.PREDICTION_SCHEMA`.
"""

from __future__ import annotations

import joblib
import pandas as pd

from src.config import MODELS_DIR


def load_model(model_path=None):
    model_path = model_path or (MODELS_DIR / "final_model.pkl")
    return joblib.load(model_path)


def predict(data, model_path=None):
    model = load_model(model_path)
    return model.predict(data)


def build_prediction_output(
    quality_df: pd.DataFrame,
    row_index,
    X_subset,
    observed,
    predicted,
    model_version: str,
) -> pd.DataFrame:
    """Assemble the standardized prediction table for a given subset of rows.

    `row_index` are the original `quality_df` index labels that `X_subset`
    corresponds to, so metadata (site, lat/lon, timestamp) can be reattached.
    """
    output = quality_df.loc[row_index].copy()
    output["observed"] = pd.Series(observed).values
    output["predicted"] = pd.Series(predicted).values
    output["residual"] = output["observed"] - output["predicted"]
    output["model_version"] = model_version

    output = output.rename(columns={
        "siteid": "site_id",
        "twd97lat": "lat",
        "twd97lon": "lon",
        "sampledate": "timestamp",
    })

    schema_cols = ["site_id", "lat", "lon", "timestamp", "observed", "predicted",
                   "residual", "model_version"]
    ordered = schema_cols + [c for c in output.columns if c not in schema_cols]
    return output[ordered]
