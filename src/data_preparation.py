# src/data_preparation.py

import pandas as pd
import numpy as np
import os

def get_season(month):
    if month in [5, 6, 7, 8, 9]:
        return "Yala"
    else:
        return "Maha"

def prepare_dataset():
    # Build paths relative to the project root so this works
    # whether called from src/ or via main.py in the root.
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(base_dir)
    raw_path = os.path.join(project_root, "data", "raw")

    # Load datasets
    pop = pd.read_excel(os.path.join(raw_path, "Sri_Lanka_Population_1950_2025.xlsx"))
    birth = pd.read_excel(os.path.join(raw_path, "Sri_Lanka_Birth_Rate_1950_2025.xlsx"))
    urban = pd.read_excel(os.path.join(raw_path, "Sri_Lanka_Urban_Population_1960_2023.xlsx"))
    rural = pd.read_excel(os.path.join(raw_path, "Sri_Lanka_Rural_Population_1960_2023.xlsx"))

    density = pd.read_csv(os.path.join(raw_path, "Sri-Lanka-Population-Density-People-per-Square-KM-2026-03-05-21-40.csv"))
    deaths = pd.read_csv(os.path.join(raw_path, "Year,Deaths per 1000 People.csv"))

    # Rename
    birth.rename(columns={"Birth Rate (per 1000 people)": "Birth_Rate"}, inplace=True)
    urban.rename(columns={"Urban Population": "Urban_Population"}, inplace=True)
    rural.rename(columns={"Rural Population": "Rural_Population"}, inplace=True)
    density.rename(columns={"Population Density": "Population_Density"}, inplace=True)
    deaths.rename(columns={"Deaths per 1000 People": "Deaths_per_1000"}, inplace=True)

    # Merge
    df = pop.merge(birth, on="Year", how="left") \
            .merge(urban, on="Year", how="left") \
            .merge(rural, on="Year", how="left") \
            .merge(density, on="Year", how="left") \
            .merge(deaths, on="Year", how="left")

    # Date
    df["Date"] = pd.to_datetime(df["Year"].astype(str) + "-12-31")
    df.set_index("Date", inplace=True)
    df.sort_index(inplace=True)

    # Monthly interpolation
    df_monthly = df.resample("MS").mean().interpolate(method="spline", order=2)

    # Features
    df_monthly["Year"] = df_monthly.index.year
    df_monthly["Month"] = df_monthly.index.month

    # ==================== SEASON FEATURE ====================
    df_monthly['Season'] = df_monthly['Month'].apply(get_season)
    df_monthly = pd.get_dummies(df_monthly, columns=['Season'])
    # =======================================================

    df_monthly["Population_Lag_1M"] = df_monthly["Population"].shift(1)
    df_monthly["Population_Lag_12M"] = df_monthly["Population"].shift(12)

    df_monthly.dropna(inplace=True)

    # Save
    processed_dir = os.path.join(project_root, "data", "processed")
    os.makedirs(processed_dir, exist_ok=True)
    out_path = os.path.join(processed_dir, "sri_lanka_population_monthly.csv")
    df_monthly.to_csv(out_path)

    print(f"✅ National dataset created at {out_path} with Season feature (Yala/Maha)!")


if __name__ == "__main__":
    prepare_dataset()