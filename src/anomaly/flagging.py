"""Flag and report abnormal water-quality observations."""

from __future__ import annotations

from collections import defaultdict
from math import isfinite
from typing import Any, Iterable, Mapping

from .residuals import compute_residuals
from .threshold import calculate_threshold


def flag_anomalies(
	residuals: Iterable[float],
	threshold: float | None = None,
	method: str = "mad",
	multiplier: float = 3.0,
) -> list[dict[str, Any]]:
	"""Return one flag record per residual, including anomaly direction."""
	values = [float(value) for value in residuals]
	if not values or any(not isfinite(value) for value in values):
		raise ValueError("residuals must contain at least one finite value")
	if threshold is None:
		threshold = calculate_threshold(values, method, multiplier)
	if threshold < 0 or not isfinite(float(threshold)):
		raise ValueError("threshold must be a finite, non-negative number")
	return [
		{
			"index": index,
			"residual": value,
			"absolute_residual": abs(value),
			"is_anomaly": abs(value) > threshold,
			"direction": "high" if value > threshold else "low" if value < -threshold else "normal",
			"threshold": float(threshold),
		}
		for index, value in enumerate(values)
	]


def build_event_report(
	records: Iterable[Mapping[str, Any]],
	observed_key: str = "observed",
	predicted_key: str = "predicted",
	threshold: float | None = None,
	method: str = "mad",
	multiplier: float = 3.0,
) -> list[dict[str, Any]]:
	"""Add residual and anomaly fields to prediction records."""
	source = [dict(record) for record in records]
	if not source:
		return []
	residuals = compute_residuals(
		(record[observed_key] for record in source),
		(record[predicted_key] for record in source),
	)
	flags = flag_anomalies(residuals, threshold, method, multiplier)
	return [{**record, **flag} for record, flag in zip(source, flags)]


def build_model_event_report(
	model: Any,
	features: Any,
	observed: Iterable[Any],
	metadata: Iterable[Mapping[str, Any]] | None = None,
	threshold: float = 0.5,
) -> list[dict[str, Any]]:
	"""Build an anomaly report from member 1's fitted classifier.

	The prediction-model notebook exposes a fitted ``final_svm`` with a
	 scikit-learn ``predict`` method, plus ``X_test`` and ``y_test``. This
	 adapter keeps that handoff independent of pandas and preserves optional
	 site or timestamp metadata in each report row.
	"""
	predicted = list(model.predict(features))
	observed_values = list(observed)
	if len(predicted) != len(observed_values):
		raise ValueError("model predictions and observed values must have the same length")
	if metadata is None:
		metadata_values = [{} for _ in observed_values]
	else:
		metadata_values = [dict(record) for record in metadata]
		if len(metadata_values) != len(observed_values):
			raise ValueError("metadata and observed values must have the same length")
	records = [
		{**record, "observed": actual, "predicted": estimate}
		for record, actual, estimate in zip(metadata_values, observed_values, predicted)
	]
	return build_event_report(records, threshold=threshold)


def anomaly_stability(
	records: Iterable[Mapping[str, Any]],
	year_key: str = "year",
	anomaly_key: str = "is_anomaly",
) -> list[dict[str, Any]]:
	"""Summarize anomaly counts and rates by year for temporal analysis."""
	grouped: dict[Any, list[Mapping[str, Any]]] = defaultdict(list)
	for record in records:
		if year_key not in record or anomaly_key not in record:
			raise KeyError(f"records must contain '{year_key}' and '{anomaly_key}'")
		grouped[record[year_key]].append(record)
	return [
		{
			"year": year,
			"observations": len(items),
			"anomalies": sum(bool(item[anomaly_key]) for item in items),
			"anomaly_rate": sum(bool(item[anomaly_key]) for item in items) / len(items),
		}
		for year, items in sorted(grouped.items(), key=lambda item: item[0])
	]
