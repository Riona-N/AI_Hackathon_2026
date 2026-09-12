"""Train/test split and CV-fold helpers.

The full pipeline in project-plan.md assumes a dataset large enough for an
80/20 holdout and 10-fold CV. The demo dataset shipped in this repo only has
a handful of site/date rows, so these helpers fall back to something sane
(and clearly labelled) instead of crashing.
"""

from __future__ import annotations

from sklearn.model_selection import KFold, train_test_split

MIN_ROWS_FOR_HOLDOUT = 8
MIN_ROWS_FOR_CV = 8


def safe_train_test_split(X, y, test_size=0.2, random_state=42):
    """Same as sklearn's train_test_split, but falls back to a 1-row holdout
    (or, for truly tiny data, reuses the training rows as the test set) so a
    handful of rows doesn't raise a ValueError.

    Returns (X_train, X_test, y_train, y_test, note) where `note` documents
    which behavior was used, for transparent reporting downstream.
    """
    n = len(X)
    if n < 3:
        return X, X, y, y, (
            f"Only {n} labeled rows available - training and evaluating on "
            "the same rows (in-sample only). Metrics will look optimistic."
        )
    if n < MIN_ROWS_FOR_HOLDOUT:
        # Still split, but hold out just one row so both sets are non-empty.
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=1, random_state=random_state
        )
        return X_train, X_test, y_train, y_test, (
            f"Only {n} labeled rows available - using a 1-row holdout instead "
            "of the standard 80/20 split."
        )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    return X_train, X_test, y_train, y_test, "Standard 80/20 holdout split."


def safe_cv(n_samples, n_splits=10, random_state=42):
    """Return a KFold with a feasible number of splits, and a note describing
    any adjustment, instead of raising when n_samples < n_splits."""
    if n_samples < 2:
        return None, "Not enough rows for cross-validation; skipped."
    effective_splits = min(n_splits, n_samples)
    if effective_splits < n_splits:
        note = (
            f"Requested {n_splits}-fold CV but only {n_samples} rows are "
            f"available - using {effective_splits}-fold CV instead."
        )
    else:
        note = f"{n_splits}-fold cross-validation."
    return KFold(n_splits=effective_splits, shuffle=True, random_state=random_state), note
