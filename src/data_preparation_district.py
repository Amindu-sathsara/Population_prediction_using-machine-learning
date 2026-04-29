# src/data_preparation_district.py

import pandas as pd
import os
import numpy as np


def prepare_district_dataset():
    # Paths relative to project root so this works via main.py
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(base_dir)
    raw_path = os.path.join(project_root, "data", "raw")

    df = pd.read_csv(os.path.join(raw_path, "sri_lanka_district_population_2014_2024_new.csv"))

    national_processed = os.path.join(project_root, "data", "processed", "sri_lanka_population_monthly.csv")

    df = df.groupby(["District", "Year"])["Total"].sum().reset_index()
    df.rename(columns={"Total": "Population"}, inplace=True)

    df["Date"] = pd.to_datetime(df["Year"].astype(str) + "-12-31")

    df.set_index(["District", "Date"], inplace=True)
    df = df.sort_index()

    all_districts = []

    # Create simple monthly series per district via linear interpolation
    for d, g in df.groupby(level=0):
        g = g.droplevel(0)
        g = g.resample("MS").mean().interpolate(method="linear")
        g["District"] = d
        g["Year"] = g.index.year
        all_districts.append(g)

    df = pd.concat(all_districts)
    df.reset_index(inplace=True)
    df.rename(columns={"index": "Date"}, inplace=True)

    # Add national population as a global driver for district models
    if os.path.exists(national_processed):
        nat = pd.read_csv(national_processed, parse_dates=["Date"])
        if "Population" in nat.columns:
            nat = nat[["Date", "Population"]].rename(columns={"Population": "National_Population"})
            df = df.merge(nat, on="Date", how="left")

    # Feature engineering for district forecasting
    df = df.sort_values(["District", "Date"]).reset_index(drop=True)
    df["Month"] = df["Date"].dt.month
    df["lag_1"] = df.groupby("District")["Population"].shift(1)
    df["lag_12"] = df.groupby("District")["Population"].shift(12)
    df["rolling_mean_3"] = (
        df.groupby("District")["Population"]
        .rolling(window=3, min_periods=3)
        .mean()
        .reset_index(level=0, drop=True)
    )
    df["month_sin"] = np.sin(2 * np.pi * df["Month"] / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * df["Month"] / 12.0)
    if "National_Population" in df.columns:
        df["national_lag_1"] = df["National_Population"].shift(1)
        df["national_lag_12"] = df["National_Population"].shift(12)

    df.dropna(inplace=True)

    processed_dir = os.path.join(project_root, "data", "processed")
    os.makedirs(processed_dir, exist_ok=True)
    out_path = os.path.join(processed_dir, "sri_lanka_district_population_monthly.csv")
    df.to_csv(out_path, index=False)

    print(f"✅ District dataset created at {out_path} with Season feature (Yala/Maha)!")


if __name__ == "__main__":
    prepare_district_dataset()