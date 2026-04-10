import os
from pathlib import Path

import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
import pickle


DRIVER_COLUMNS = [
    "Birth_Rate",
    "Death_Rate",
    "Population_Density",
    "Rural_Population",
    "Urban_Population",
]


def train_national_driver_models():
    """Train simple ARIMA/SARIMA models for key national drivers.

    These models forecast monthly values for birth rate, death rate,
    population density, rural and urban population. The fitted models
    and their metadata are saved to models/nat_driver_models.pkl.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(base_dir)
    processed_path = os.path.join(project_root, "data", "processed", "national_master_monthly.csv")

    if not os.path.exists(processed_path):
        raise FileNotFoundError(
            f"Processed national dataset not found at {processed_path}. Run dataset_builder.py first."
        )

    df = pd.read_csv(processed_path, parse_dates=["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    df.set_index("Date", inplace=True)

    models_dir = Path(project_root) / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    driver_models = {}

    for col in DRIVER_COLUMNS:
        if col not in df.columns:
            continue

        series = df[col].astype(float)

        # Basic non-seasonal ARIMA(1,1,1). This is intentionally simple
        # and robust for small datasets.
        model = SARIMAX(series, order=(1, 1, 1), seasonal_order=(0, 0, 0, 0))
        res = model.fit(disp=False)

        driver_models[col] = {
            "model": res,
            "last_date": series.index[-1],
        }

    out_path = models_dir / "nat_driver_models.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(driver_models, f)

    print(f"✅ Saved national driver models to {out_path}")


if __name__ == "__main__":
    train_national_driver_models()
