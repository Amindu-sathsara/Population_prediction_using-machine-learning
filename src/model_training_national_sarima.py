import os

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from statsmodels.tsa.statespace.sarimax import SARIMAX

from national_metrics_utils import upsert_national_metric


def train_national_sarima():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(base_dir)
    raw_path = os.path.join(project_root, "data", "raw")

    pop_path = os.path.join(raw_path, "Sri_Lanka_Population_1950_2025.xlsx")
    if not os.path.exists(pop_path):
        raise FileNotFoundError(f"Yearly population file not found at {pop_path}.")

    df = pd.read_excel(pop_path)[["Year", "Population"]]
    # Use end-of-year dates as index
    df["Date"] = pd.to_datetime(df["Year"].astype(str) + "-12-31")
    df = df.set_index("Date").sort_index()

    series = df["Population"].astype(float)

    split = int(len(series) * 0.8)
    train = series.iloc[:split]
    test = series.iloc[split:]

    # Basic SARIMA model; you can tune (p,d,q) if needed
    model = SARIMAX(train, order=(1, 1, 1), seasonal_order=(0, 0, 0, 0))
    res = model.fit(disp=False)

    # In-sample forecast for test period
    test_forecast = res.get_forecast(steps=len(test))
    test_pred = test_forecast.predicted_mean

    mae = mean_absolute_error(test, test_pred)
    mape = mean_absolute_percentage_error(test, test_pred) * 100

    print("SARIMA YEARLY MAE:", mae)
    print("SARIMA YEARLY MAPE:", mape)

    metrics_path = upsert_national_metric(
        project_root=project_root,
        model_name="SARIMA",
        mae=mae,
        mape=mape,
        train_size=len(train),
        test_size=len(test),
        notes="Yearly univariate SARIMA(1,1,1) with no seasonal component.",
    )

    models_dir = os.path.join(project_root, "models")
    os.makedirs(models_dir, exist_ok=True)

    # Plot actual vs predicted on test period
    plt.figure(figsize=(10, 5))
    plt.plot(test.index, test.values, label="Actual")
    plt.plot(test.index, test_pred.values, label="Predicted")
    plt.xlabel("Year")
    plt.ylabel("Population")
    plt.title("National Population - SARIMA (Yearly)")
    plt.legend()
    plot_path = os.path.join(models_dir, "national_sarima_plot.png")
    plt.savefig(plot_path)

    # Forecast future years
    horizon_years = 15
    full_forecast = res.get_forecast(steps=len(test) + horizon_years)
    full_pred = full_forecast.predicted_mean

    full_index = full_pred.index
    last_obs_date = series.index.max()

    future_mask = full_index > last_obs_date
    future_yearly = pd.DataFrame({
        "Year": full_index[future_mask].year,
        "Predicted_Population": full_pred[future_mask].values,
    })

    yearly_out = os.path.join(models_dir, "future_national_population_yearly_sarima.csv")
    future_yearly.to_csv(yearly_out, index=False)

    # Monthly interpolation for API-style forecast
    all_years = pd.concat([
        df[["Population"]].rename(columns={"Population": "Value"}),
        full_pred[future_mask].to_frame(name="Value"),
    ])
    monthly = all_years.resample("MS").interpolate("linear")

    cutoff_date = last_obs_date + pd.DateOffset(months=1)
    monthly_future = monthly[monthly.index >= cutoff_date].copy()
    monthly_future.rename(columns={"Value": "Predicted_Population"}, inplace=True)
    monthly_future.index.name = "Date"

    monthly_out = os.path.join(models_dir, "future_national_population_sarima.csv")
    monthly_future.to_csv(monthly_out)

    print(f"✅ SARIMA yearly national forecast saved to {yearly_out}")
    print(f"✅ SARIMA monthly national forecast saved to {monthly_out}")
    print(f"✅ National metrics updated at {metrics_path}")


if __name__ == "__main__":
    train_national_sarima()
