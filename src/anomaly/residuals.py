"""Residual and prediction-error utilities."""

from __future__ import annotations

from math import isfinite, sqrt
from statistics import mean, stdev
from typing import Iterable


def compute_residuals(observed: Iterable[float], predicted: Iterable[float]) -> list[float]:
	"""Return observed minus predicted for each paired observation."""
	observed_values = [float(value) for value in observed]
	predicted_values = [float(value) for value in predicted]
	if len(observed_values) != len(predicted_values):
		raise ValueError("observed and predicted must have the same length")
	if not observed_values or any(
		not isfinite(value) for value in observed_values + predicted_values
	):
		raise ValueError("observed and predicted must contain finite values")
	return [actual - estimate for actual, estimate in zip(observed_values, predicted_values)]


def error_metrics(residuals: Iterable[float]) -> dict[str, float]:
	"""Summarize bias and common prediction-error metrics."""
	values = [float(value) for value in residuals]
	if not values or any(not isfinite(value) for value in values):
		raise ValueError("residuals must contain at least one finite value")
	absolute = [abs(value) for value in values]
	return {
		"count": float(len(values)),
		"mean_error": mean(values),
		"mae": mean(absolute),
		"rmse": sqrt(mean(value * value for value in values)),
		"max_absolute_error": max(absolute),
	}


def prediction_interval(
	residuals: Iterable[float], confidence: float = 0.95
) -> tuple[float, float]:
	"""Return a normal-approximation interval for future residuals."""
	values = [float(value) for value in residuals]
	if len(values) < 2:
		raise ValueError("at least two residuals are required for an interval")
	if not 0 < confidence < 1:
		raise ValueError("confidence must be between 0 and 1")
	z_score = 1.96 if confidence == 0.95 else 2.576 if confidence >= 0.99 else 1.645
	margin = z_score * stdev(values)
	center = mean(values)
	return center - margin, center + margin


def uncertainty_summary(
	residuals: Iterable[float], confidence: float = 0.95
) -> dict[str, float | tuple[float, float]]:
	"""Return error metrics and a prediction interval from residuals."""
	values = list(residuals)
	metrics = error_metrics(values)
	metrics["residual_std"] = stdev(values) if len(values) > 1 else 0.0
	metrics["prediction_interval"] = prediction_interval(values, confidence)
	return metrics
