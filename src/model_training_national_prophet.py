import os

import matplotlib.pyplot as plt
import pandas as pd
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error


def train_national_prophet():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(base_dir)
    raw_path = os.path.join(project_root, "data", "raw")

    pop_path = os.path.join(raw_path, "Sri_Lanka_Population_1950_2025.xlsx")
    if not os.path.exists(pop_path):
        raise FileNotFoundError(f"Yearly population file not found at {pop_path}.")

    df = pd.read_excel(pop_path)[["Year", "Population"]]

    # Focus on more recent years so the trend reflects
    # current dynamics instead of very old history.
    df = df[df["Year"] >= 1990].copy()

    df["ds"] = pd.to_datetime(df["Year"].astype(str) + "-12-31")
    df["y"] = df["Population"].astype(float)

    # Set a capacity for logistic growth so Prophet can
    # produce a curved (saturating) trajectory instead of
    # a purely straight line.
    cap_value = df["y"].max() * 1.05  # 5% above max observed
    df["cap"] = cap_value

    df = df.sort_values("ds").reset_index(drop=True)

    split = int(len(df) * 0.8)
    train = df.iloc[:split][["ds", "y", "cap"]]
    test = df.iloc[split:]

    # Use logistic growth with a higher changepoint prior so
    # Prophet can follow non-linear trend more closely.
    m = Prophet(
        growth="logistic",
        yearly_seasonality=False,
        weekly_seasonality=False,
        daily_seasonality=False,
        changepoint_prior_scale=0.8,
        n_changepoints=25,
    )

    m.fit(train)

    # Forecast up to end of test period for evaluation
    future_test = m.make_future_dataframe(periods=len(test), freq="Y")
    future_test["cap"] = cap_value
    forecast_test = m.predict(future_test)

    merged = forecast_test[["ds", "yhat"]].merge(test[["ds", "y"]], on="ds", how="inner")
    mae = mean_absolute_error(merged["y"], merged["yhat"])
    mape = mean_absolute_percentage_error(merged["y"], merged["yhat"]) * 100

    print("PROPHET YEARLY MAE:", mae)
    print("PROPHET YEARLY MAPE:", mape)

    models_dir = os.path.join(project_root, "models")
    os.makedirs(models_dir, exist_ok=True)

    # Plot actual vs predicted on test period
    plt.figure(figsize=(10, 5))
    plt.plot(merged["ds"], merged["y"], label="Actual")
    plt.plot(merged["ds"], merged["yhat"], label="Predicted")
    plt.xlabel("Year")
    plt.ylabel("Population")
    plt.title("National Population - Prophet (Yearly)")
    plt.legend()
    plot_path = os.path.join(models_dir, "national_prophet_plot.png")
    plt.savefig(plot_path)

    # Forecast additional future years (e.g. 15 years beyond last observation)
    horizon_years = 15
    future_full = m.make_future_dataframe(periods=horizon_years, freq="Y")
    future_full["cap"] = cap_value
    forecast_full = m.predict(future_full)

    # Extract only the future part
    last_obs_date = df["ds"].max()
    future_rows = forecast_full[forecast_full["ds"] > last_obs_date].copy()
    future_rows["Year"] = future_rows["ds"].dt.year

    yearly_out = os.path.join(models_dir, "future_national_population_yearly_prophet.csv")
    future_rows[["Year", "yhat"]].rename(columns={"yhat": "Predicted_Population"}).to_csv(
        yearly_out, index=False
    )

    # Create monthly forecast via linear interpolation between yearly Prophet outputs
    all_years = pd.concat([
        df[["Year", "Population"]].rename(columns={"Population": "Value"}),
        future_rows[["Year", "yhat"]].rename(columns={"yhat": "Value"}),
    ], ignore_index=True)
    all_years["Date"] = pd.to_datetime(all_years["Year"].astype(str) + "-12-31")
    all_years = all_years.set_index("Date").sort_index()

    monthly = all_years[["Value"]].resample("MS").interpolate("linear")

    cutoff_date = last_obs_date + pd.DateOffset(months=1)
    monthly_future = monthly[monthly.index >= cutoff_date].copy()
    monthly_future.rename(columns={"Value": "Predicted_Population"}, inplace=True)
    monthly_future.index.name = "Date"

    monthly_out = os.path.join(models_dir, "future_national_population_prophet.csv")
    monthly_future.to_csv(monthly_out)

    print(f"✅ Prophet yearly national forecast saved to {yearly_out}")
    print(f"✅ Prophet monthly national forecast saved to {monthly_out}")


if __name__ == "__main__":
    train_national_prophet()
