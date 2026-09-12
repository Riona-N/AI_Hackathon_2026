"""SHAP value computation for the trained model.

Extracted from Member 1's `feat1and4.ipynb` (cells 27-28). Wrapped in a
try/except: SHAP's TreeExplainer needs a supported tree model and at least
one row, and this repo's demo dataset is tiny, so failures here should never
take down the rest of the pipeline - callers get `None` plus a reason instead
of an exception.
"""

from __future__ import annotations

import pandas as pd


def compute_shap_values(pipeline, X: pd.DataFrame):
    """Return (shap_values, X_processed, note). shap_values is None if SHAP
    could not be computed for this model/data (note explains why)."""
    try:
        import shap
    except ImportError:
        return None, None, "shap is not installed; skipping SHAP analysis."

    model = pipeline.named_steps["model"]
    imputer = pipeline.named_steps["imputer"]
    X_processed = pd.DataFrame(imputer.transform(X), columns=X.columns, index=X.index)

    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_processed)
        return shap_values, X_processed, "SHAP TreeExplainer succeeded."
    except Exception as exc:  # model type unsupported, too little data, etc.
        return None, X_processed, f"SHAP TreeExplainer failed ({exc}); skipping SHAP analysis."
