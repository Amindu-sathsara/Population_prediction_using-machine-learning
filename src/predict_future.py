# src/predict_future.py

import pandas as pd
import joblib

def predict_future():
    model = joblib.load("../models/national_model.pkl")

    df = pd.read_csv("../data/processed/sri_lanka_population_monthly.csv", index_col=0)

    last_row = df.iloc[-1:].copy()

    future_preds = []

    for i in range(12):  # next 12 months
        pred = model.predict(last_row)[0]

        new_row = last_row.copy()
        new_row["Population"] = pred
        new_row["Population_Lag_1M"] = pred

        future_preds.append(pred)
        last_row = new_row

    print("Future Predictions:", future_preds)