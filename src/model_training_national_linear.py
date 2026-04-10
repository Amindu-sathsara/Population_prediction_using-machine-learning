import os

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error


def build_yearly_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("Year").reset_index(drop=True)
    df["Year_centered"] = df["Year"] - df["Year"].mean()
    df["Year_centered2"] = df["Year_centered"] ** 2
    return df


def train_national_linear():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(base_dir)
    raw_path = os.path.join(project_root, "data", "raw")

    pop_path = os.path.join(raw_path, "Sri_Lanka_Population_1950_2025.xlsx")
    if not os.path.exists(pop_path):
        raise FileNotFoundError(f"Yearly population file not found at {pop_path}.")

    df = pd.read_excel(pop_path)[["Year", "Population"]]
    df = build_yearly_features(df)

    X = df[["Year_centered", "Year_centered2"]]
    y = df["Population"].astype(float)

    split = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    model = LinearRegression()
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    mape = mean_absolute_percentage_error(y_test, preds) * 100

    print("LINEAR YEARLY MAE:", mae)
    print("LINEAR YEARLY MAPE:", mape)

    models_dir = os.path.join(project_root, "models")
    os.makedirs(models_dir, exist_ok=True)

    # Plot actual vs predicted on test period
    plt.figure(figsize=(10, 5))
    plt.plot(df["Year"].iloc[split:], y_test.values, label="Actual")
    plt.plot(df["Year"].iloc[split:], preds, label="Predicted")
    plt.xlabel("Year")
    plt.ylabel("Population")
    plt.title("National Population - Linear Regression (Yearly)")
    plt.legend()
    plot_path = os.path.join(models_dir, "national_linear_plot.png")
    plt.savefig(plot_path)

    # Forecast future years
    last_year = int(df["Year"].max())
    horizon_years = 15
    future_years = list(range(last_year + 1, last_year + 1 + horizon_years))

    future_df = pd.DataFrame({"Year": future_years})
    future_df["Year_centered"] = future_df["Year"] - df["Year"].mean()
    future_df["Year_centered2"] = future_df["Year_centered"] ** 2

    future_df["Predicted_Population"] = model.predict(
        future_df[["Year_centered", "Year_centered2"]]
    )

    yearly_out = os.path.join(project_root, "models", "future_national_population_yearly_linear.csv")
    future_df[["Year", "Predicted_Population"]].to_csv(yearly_out, index=False)

    # Monthly interpolation from yearly linear trend
    all_years = pd.concat([
        df[["Year", "Population"]].rename(columns={"Population": "Value"}),
        future_df[["Year", "Predicted_Population"]].rename(columns={"Predicted_Population": "Value"}),
    ], ignore_index=True)
    all_years["Date"] = pd.to_datetime(all_years["Year"].astype(str) + "-12-31")
    all_years = all_years.set_index("Date").sort_index()

    monthly = all_years[["Value"]].resample("MS").interpolate("linear")

    cutoff_date = pd.to_datetime(f"{last_year}-12-31") + pd.DateOffset(months=1)
    monthly_future = monthly[monthly.index >= cutoff_date].copy()
    monthly_future.rename(columns={"Value": "Predicted_Population"}, inplace=True)
    monthly_future.index.name = "Date"

    monthly_out = os.path.join(models_dir, "future_national_population_linear.csv")
    monthly_future.to_csv(monthly_out)

    print(f"✅ Linear yearly national forecast saved to {yearly_out}")
    print(f"✅ Linear monthly national forecast saved to {monthly_out}")


if __name__ == "__main__":
    train_national_linear()
