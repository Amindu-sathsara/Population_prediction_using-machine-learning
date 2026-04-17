import os
from datetime import datetime

import pandas as pd


def upsert_national_metric(project_root: str, model_name: str, mae: float, mape: float, train_size: int, test_size: int, notes: str = "") -> str:
    """Create or update models/national_model_metrics.csv with latest model metrics."""
    models_dir = os.path.join(project_root, "models")
    os.makedirs(models_dir, exist_ok=True)
    metrics_path = os.path.join(models_dir, "national_model_metrics.csv")

    row = {
        "Model": model_name,
        "MAE": float(mae),
        "MAPE": float(mape),
        "Train_Size": int(train_size),
        "Test_Size": int(test_size),
        "Notes": notes,
        "Updated_At": datetime.now().isoformat(timespec="seconds"),
    }

    if os.path.exists(metrics_path):
        df = pd.read_csv(metrics_path)
    else:
        df = pd.DataFrame([row])
        df = df.sort_values("Model").reset_index(drop=True)
        df.to_csv(metrics_path, index=False)
        return metrics_path

    if (df["Model"] == model_name).any():
        df.loc[df["Model"] == model_name, list(row.keys())] = [row[k] for k in row.keys()]
    else:
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)

    df = df.sort_values("Model").reset_index(drop=True)
    df.to_csv(metrics_path, index=False)
    return metrics_path
