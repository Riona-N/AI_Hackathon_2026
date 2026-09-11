import numpy as np
import pandas as pd

RPI_NAME = "River Pollution Index"

def numeric_item_values(df):
    x = df.copy()
    raw = x["itemvalue"].astype(str).str.strip()
    x["value_numeric"] = pd.to_numeric(
        raw.str.replace("<", "", regex=False)
           .str.replace(">", "", regex=False)
           .replace({"-": np.nan, "": np.nan}),
        errors="coerce"
    )
    return x

def build_water_quality_matrix(df):
    x = numeric_item_values(df)
    idx = ["siteid", "siteengname", "sampledate", "twd97lat", "twd97lon"]
    value = x.pivot_table(
        index=idx, columns="itemengname", values="value_numeric",
        aggfunc="mean"
    ).reset_index()
    return value

def classify_risk(df):
    x = df.copy()
    if RPI_NAME in x.columns:
        rpi = pd.to_numeric(x[RPI_NAME], errors="coerce")
        x["risk_score"] = rpi
        # Dataset RPI values observed here are around 4.5. We use
        # transparent relative bins when no official threshold table is supplied.
        q1, q2 = rpi.quantile([0.33, 0.67])
        x["risk_level"] = np.select(
            [rpi <= q1, rpi <= q2],
            ["LOW", "MEDIUM"], default="HIGH"
        )
        x["risk_basis"] = "River Pollution Index"
    else:
        # Fallback: percentile score over available numeric indicators.
        numeric = x.select_dtypes(include="number").drop(
            columns=[c for c in ["siteid","twd97lat","twd97lon"] if c in x], errors="ignore"
        )
        score = numeric.rank(pct=True).mean(axis=1)
        x["risk_score"] = score
        x["risk_level"] = pd.cut(
            score, [-np.inf, .33, .67, np.inf],
            labels=["LOW","MEDIUM","HIGH"]
        ).astype(str)
        x["risk_basis"] = "relative multi-indicator percentile"
    return x
