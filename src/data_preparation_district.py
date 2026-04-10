# src/data_preparation_district.py

import pandas as pd
import os


def prepare_district_dataset():
    # Paths relative to project root so this works via main.py
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(base_dir)
    raw_path = os.path.join(project_root, "data", "raw")

    df = pd.read_csv(os.path.join(raw_path, "sri_lanka_district_population_2014_2024_new.csv"))

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

    processed_dir = os.path.join(project_root, "data", "processed")
    os.makedirs(processed_dir, exist_ok=True)
    out_path = os.path.join(processed_dir, "sri_lanka_district_population_monthly.csv")
    df.to_csv(out_path, index=False)

    print(f"✅ District dataset created at {out_path} with Season feature (Yala/Maha)!")


if __name__ == "__main__":
    prepare_district_dataset()