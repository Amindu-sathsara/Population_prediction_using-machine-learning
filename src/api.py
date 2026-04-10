from datetime import date
import os

import pandas as pd
from fastapi import FastAPI, HTTPException

app = FastAPI()


def _paths():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(base_dir)
    models_dir = os.path.join(project_root, "models")
    processed_dir = os.path.join(project_root, "data", "processed")

    return {
        "project_root": project_root,
        "models_dir": models_dir,
        "processed_dir": processed_dir,
    }


def _load_predictions_and_history():
    paths = _paths()
    models_dir = paths["models_dir"]
    processed_dir = paths["processed_dir"]

    national_path = os.path.join(models_dir, "future_national_population.csv")
    district_path = os.path.join(models_dir, "district_population_predictions.csv")
    district_hist_path = os.path.join(processed_dir, "sri_lanka_district_population_monthly.csv")

    national_df = None
    district_df = None
    district_hist_df = None

    if os.path.exists(national_path):
        national_df = pd.read_csv(national_path, parse_dates=["Date"])
        national_df["Year"] = national_df["Date"].dt.year
        national_df["Month"] = national_df["Date"].dt.month

    if os.path.exists(district_path):
        district_df = pd.read_csv(district_path, parse_dates=["Date"])
        district_df["Year"] = district_df["Date"].dt.year
        district_df["Month"] = district_df["Date"].dt.month
        district_df["District_lower"] = district_df["District"].str.lower()
    if os.path.exists(district_hist_path):
        # Historical monthly district populations (actuals)
        district_hist_df = pd.read_csv(district_hist_path, parse_dates=["Date"])
        district_hist_df["Year"] = district_hist_df["Date"].dt.year
        district_hist_df["Month"] = district_hist_df["Date"].dt.month
        district_hist_df["District_lower"] = district_hist_df["District"].str.lower()

    return national_df, district_df, district_hist_df


NATIONAL_DF, DISTRICT_DF, DISTRICT_HIST_DF = _load_predictions_and_history()


@app.get("/population/national")
def get_national_population(query_date: date):
    """Get predicted national population for the given calendar date.

    We match by (year, month) to the monthly forecast.
    """
    if NATIONAL_DF is None:
        raise HTTPException(status_code=500, detail="National forecast file not found. Run the training pipeline first.")

    year = query_date.year
    month = query_date.month

    row = NATIONAL_DF[(NATIONAL_DF["Year"] == year) & (NATIONAL_DF["Month"] == month)]
    if row.empty:
        raise HTTPException(status_code=404, detail=f"No national prediction available for {query_date}.")

    pred = float(row["Predicted_Population"].iloc[0])
    return {
        "year": year,
        "month": month,
        "date_requested": str(query_date),
        "predicted_population": pred,
    }


@app.get("/population/district")
def get_district_population(district: str, query_date: date):
    """Get predicted population for a specific district and date.

    District matching is case-insensitive; we again match by (year, month).
    """
    if DISTRICT_DF is None and DISTRICT_HIST_DF is None:
        raise HTTPException(status_code=500, detail="No district data available. Run the training pipeline first.")

    year = query_date.year
    month = query_date.month

    d_lower = district.lower()

    row = None

    # 1) Try forecast first (future predictions)
    if DISTRICT_DF is not None:
        row_f = DISTRICT_DF[
            (DISTRICT_DF["District_lower"] == d_lower)
            & (DISTRICT_DF["Year"] == year)
            & (DISTRICT_DF["Month"] == month)
        ]
        if not row_f.empty:
            val = float(row_f["Predicted_Population"].iloc[0])
            return {
                "district": row_f["District"].iloc[0],
                "year": year,
                "month": month,
                "date_requested": str(query_date),
                "value_type": "forecast",
                "predicted_population": val,
            }

    # 2) Fallback to historical actuals if forecast not available
    if DISTRICT_HIST_DF is not None:
        row_h = DISTRICT_HIST_DF[
            (DISTRICT_HIST_DF["District_lower"] == d_lower)
            & (DISTRICT_HIST_DF["Year"] == year)
            & (DISTRICT_HIST_DF["Month"] == month)
        ]
        if not row_h.empty:
            val = float(row_h["Population"].iloc[0])
            return {
                "district": row_h["District"].iloc[0],
                "year": year,
                "month": month,
                "date_requested": str(query_date),
                "value_type": "historical_actual",
                "predicted_population": val,
            }

    # Nothing found
    raise HTTPException(status_code=404, detail=f"No data available for district '{district}' on {query_date}.")


@app.get("/population/national/all")
def get_all_national_predictions():
    """Return the full national forecast table (for debugging / inspection)."""
    if NATIONAL_DF is None:
        raise HTTPException(status_code=500, detail="National forecast file not found. Run the training pipeline first.")
    return NATIONAL_DF.to_dict(orient="records")


@app.get("/population/district/all")
def get_all_district_predictions():
    """Return the full district forecast table (for debugging / inspection)."""
    if DISTRICT_DF is None:
        raise HTTPException(status_code=500, detail="District forecast file not found. Run the training pipeline first.")
    return DISTRICT_DF.drop(columns=["District_lower"]).to_dict(orient="records")