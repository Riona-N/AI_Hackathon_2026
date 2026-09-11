import pandas as pd

def add_time_groups(df):
    x = df.copy()
    x["sampledate"] = pd.to_datetime(x["sampledate"], errors="coerce")
    x["year"] = x["sampledate"].dt.year
    x["month"] = x["sampledate"].dt.month
    x["date_only"] = x["sampledate"].dt.date
    x["season"] = x["month"].map({
        12:"Winter", 1:"Winter", 2:"Winter",
        3:"Spring", 4:"Spring", 5:"Spring",
        6:"Summer", 7:"Summer", 8:"Summer",
        9:"Autumn", 10:"Autumn", 11:"Autumn"
    })
    return x
