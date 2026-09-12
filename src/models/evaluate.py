"""R2/RMSE/MAE comparison across candidate models (see train.py for the
actual implementation - kept as a separate module name to match
project-plan.md's repository layout)."""

from src.models.train import cross_validate_models, evaluate_holdout  # noqa: F401
