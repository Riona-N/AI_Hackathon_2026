"""Cleaning and pivoting for the raw long-format water-quality readings.

Extracted from Member 1's `feat1and4.ipynb` (cells 7-16) and made reusable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import METADATA_COLUMNS, TARGET


def load_raw(path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    return df


def clean_item_values(df: pd.DataFrame) -> pd.DataFrame:
    """Convert '<0.0003' / '>10' style qualifier strings to numeric values."""
    x = df.copy()
    x["itemvalue_numeric"] = (
        x["itemvalue"]
        .astype(str)
        .str.strip()
        .str.replace("<", "", regex=False)
        .str.replace(">", "", regex=False)
    )
    x["itemvalue_numeric"] = pd.to_numeric(x["itemvalue_numeric"], errors="coerce")
    x["sampledate"] = pd.to_datetime(x["sampledate"], errors="coerce")
    return x


def pivot_to_model_ready(df: pd.DataFrame) -> pd.DataFrame:
    """Pivot long (one row per site/date/parameter) into wide model-ready rows
    (one row per site/date, one column per parameter)."""
    x = clean_item_values(df)
    quality_df = x.pivot_table(
        index=[c for c in METADATA_COLUMNS if c in x.columns],
        columns="itemengname",
        values="itemvalue_numeric",
        aggfunc="mean",
    ).reset_index()
    quality_df.columns.name = None
    return quality_df


def build_features_and_target(
    quality_df: pd.DataFrame,
    target: str = TARGET,
    max_missing_ratio: float = 0.70,
):
    """Drop rows with a missing target and high-missing feature columns.

    Returns (X, y, feature_columns).
    """
    df = quality_df[quality_df[target].notna()].copy().reset_index(drop=True)

    feature_columns = [c for c in df.columns if c not in METADATA_COLUMNS and c != target]
    X = df[feature_columns].apply(pd.to_numeric, errors="coerce")
    y = pd.to_numeric(df[target], errors="coerce")

    missing_ratio = X.isnull().mean()
    high_missing = missing_ratio[missing_ratio > max_missing_ratio].index.tolist()
    X = X.drop(columns=high_missing)
    feature_columns = X.columns.tolist()

    return X, y, feature_columns, df
