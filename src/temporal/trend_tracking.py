import pandas as pd

def site_trends(df):
    cols = ["siteid", "siteengname", "sampledate", "risk_score", "risk_level"]
    x = df[[c for c in cols if c in df.columns]].copy()
    x["sampledate"] = pd.to_datetime(x["sampledate"], errors="coerce")
    return x.sort_values(["siteengname", "sampledate"])

def classify_hotspot_evolution(df):
    rows = []
    for site, g in df.groupby("siteengname"):
        g = g.sort_values("sampledate")
        levels = g["risk_level"].astype(str).tolist()
        scores = pd.to_numeric(g["risk_score"], errors="coerce")
        if len(levels) < 2:
            status = "SINGLE_OBSERVATION"
        elif levels[-1] == "HIGH" and "HIGH" in levels[:-1]:
            status = "PERSISTENT"
        elif levels[-1] == "HIGH":
            status = "EMERGING"
        elif levels[0] == "HIGH" and levels[-1] != "HIGH":
            status = "IMPROVING"
        else:
            status = "STABLE/LOWER_RISK"
        rows.append({
            "siteengname": site,
            "observations": len(g),
            "first_risk": levels[0] if levels else None,
            "latest_risk": levels[-1] if levels else None,
            "mean_score": scores.mean(),
            "hotspot_status": status
        })
    return pd.DataFrame(rows)
