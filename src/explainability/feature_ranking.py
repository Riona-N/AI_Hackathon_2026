"""Rank features by the fitted model's own importance/coefficient signal.

Used as the always-available fallback under (and alongside) SHAP, since SHAP
can fail on some model types or extremely small samples.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def rank_features(pipeline, feature_columns: list[str]) -> pd.DataFrame:
    model = pipeline.named_steps["model"]

    if hasattr(model, "feature_importances_"):
        importance = np.asarray(model.feature_importances_)
    elif hasattr(model, "coef_"):
        importance = np.abs(np.asarray(model.coef_)).ravel()
    else:
        raise AttributeError(
            f"{type(model).__name__} exposes neither feature_importances_ nor coef_"
        )

    return (
        pd.DataFrame({"Feature": feature_columns, "Importance": importance})
        .sort_values("Importance", ascending=False)
        .reset_index(drop=True)
    )
