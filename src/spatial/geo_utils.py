import pandas as pd

LAT = "twd97lat"
LON = "twd97lon"

def validate_coordinates(df):
    x = df.copy()
    x[LAT] = pd.to_numeric(x[LAT], errors="coerce")
    x[LON] = pd.to_numeric(x[LON], errors="coerce")
    return x.dropna(subset=[LAT, LON])

def prepare_locations(df):
    x = validate_coordinates(df)
    cols = ["siteid", "siteengname", "countyen", "townshipen",
            "basinen", "riveren", LAT, LON]
    cols = [c for c in cols if c in x.columns]
    return x[cols].drop_duplicates()
