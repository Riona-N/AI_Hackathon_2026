"""Threshold strategies for residual-based anomaly detection."""

from __future__ import annotations

from math import isfinite
from statistics import median, stdev
from typing import Iterable


def calculate_threshold(
	residuals: Iterable[float], method: str = "mad", multiplier: float = 3.0
) -> float:
	"""Calculate a non-negative absolute residual threshold."""
	values = [float(value) for value in residuals]
	if not values or any(not isfinite(value) for value in values):
		raise ValueError("residuals must contain at least one finite value")
	if multiplier <= 0:
		raise ValueError("multiplier must be positive")
	if method == "mad":
		center = median(values)
		deviation = median(abs(value - center) for value in values)
		return max(0.0, multiplier * 1.4826 * deviation)
	if method == "std":
		spread = stdev(values) if len(values) > 1 else 0.0
		return max(0.0, multiplier * spread)
	if method == "quantile":
		if multiplier >= 1:
			raise ValueError("quantile multiplier must be less than 1")
		absolute = sorted(abs(value) for value in values)
		position = multiplier * (len(absolute) - 1)
		lower = int(position)
		upper = min(lower + 1, len(absolute) - 1)
		fraction = position - lower
		return absolute[lower] + fraction * (absolute[upper] - absolute[lower])
	raise ValueError("method must be 'mad', 'std', or 'quantile'")
