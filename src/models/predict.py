import pandas as pd
import joblib

MODEL_PATH = "models/final_svm.pkl"

model = joblib.load(MODEL_PATH)


def predict_water_quality(data):
    """
    Predict water potability for one or more samples.
    """
    predictions = model.predict(data)
    return predictions
