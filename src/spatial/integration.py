import pandas as pd

REQUIRED = ["site_id", "lat", "lon", "timestamp"]

def normalize_prediction_output(predictions):
    x = predictions.copy()
    rename = {"siteid":"site_id", "twd97lat":"lat", "twd97lon":"lon",
              "sampledate":"timestamp", "observed":"observed",
              "predicted":"predicted"}
    x = x.rename(columns={k:v for k,v in rename.items() if k in x.columns})
    return x

def merge_predictions_with_spatial(predictions, spatial_df):
    p = normalize_prediction_output(predictions)
    s = spatial_df.rename(columns={
        "siteid":"site_id", "twd97lat":"lat", "twd97lon":"lon",
        "sampledate":"timestamp"
    })
    keys = [k for k in ["site_id","timestamp"] if k in p.columns and k in s.columns]
    return p.merge(s, on=keys, how="left", suffixes=("_pred", "_water"))
