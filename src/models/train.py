"""Leakage-safe model training with SMOTE for water-potability data."""

from __future__ import annotations

from typing import Any

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def build_svm_pipeline(random_state: int = 42) -> Pipeline:
	"""Build the member 1 SVM pipeline with SMOTE applied during fitting.

	SMOTE is deliberately inside the imbalanced-learn pipeline. During cross-
	validation, it is fitted only on each fold's training data and is never
	applied to validation or test observations.
	"""
	return Pipeline(
		steps=[
			("imputer", SimpleImputer(strategy="median")),
			("smote", SMOTE(random_state=random_state)),
			("scaler", StandardScaler()),
			("model", SVC(kernel="rbf", probability=True, random_state=random_state)),
		]
	)


def train_svm(
	features: Any,
	target: Any,
	random_state: int = 42,
) -> Pipeline:
	"""Fit and return an SVM pipeline using SMOTE on the training data."""
	pipeline = build_svm_pipeline(random_state)
	pipeline.fit(features, target)
	return pipeline


def cross_validate_svm(
	features: Any,
	target: Any,
	n_splits: int = 10,
	random_state: int = 42,
) -> dict[str, float]:
	"""Evaluate the SMOTE-enabled SVM with stratified cross-validation."""
	if n_splits < 2:
		raise ValueError("n_splits must be at least 2")
	cv = StratifiedKFold(
		n_splits=n_splits,
		shuffle=True,
		random_state=random_state,
	)
	scores = cross_validate(
		build_svm_pipeline(random_state),
		features,
		target,
		cv=cv,
		scoring={
			"accuracy": "accuracy",
			"precision": "precision",
			"recall": "recall",
			"f1": "f1",
		},
		n_jobs=-1,
	)
	return {
		metric: float(scores[f"test_{metric}"].mean())
		for metric in ("accuracy", "precision", "recall", "f1")
	}