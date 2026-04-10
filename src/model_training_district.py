import pandas as pd
from sklearn.linear_model import LinearRegression
import os


def train_district_model():
    """Train a simple yearly trend model per district and create monthly forecasts.

    We use a polynomial trend in Year for each district (no seasonal or lag columns).
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(base_dir)
    processed_path = os.path.join(project_root, 'data', 'processed', 'sri_lanka_district_population_monthly.csv')

    if not os.path.exists(processed_path):
        raise FileNotFoundError(
            f"Processed district dataset not found at {processed_path}. Run data_preparation_district.py first."
        )

    df = pd.read_csv(processed_path, parse_dates=["Date"])
    df = df.sort_values(["District", "Date"])

    all_monthly_forecasts = []

    for district in df["District"].unique():
        d = df[df["District"] == district].copy()

        # Use December of each year as the yearly observation to keep it simple
        d_dec = d[d["Date"].dt.month == 12].copy().sort_values("Date")
        d_dec["Year"] = d_dec["Date"].dt.year

        # Build simple polynomial trend features
        year_mean = d_dec["Year"].mean()
        d_dec["Year_centered"] = d_dec["Year"] - year_mean
        d_dec["Year_centered2"] = d_dec["Year_centered"] ** 2

        X = d_dec[["Year_centered", "Year_centered2"]]
        y = d_dec["Population"]

        model = LinearRegression()
        model.fit(X, y)

        # Forecast future yearly values (e.g. 15 years ahead)
        last_year = int(d_dec["Year"].max())
        horizon_years = 15
        future_years = list(range(last_year + 1, last_year + 1 + horizon_years))

        future_df = pd.DataFrame({"Year": future_years})
        future_df["Year_centered"] = future_df["Year"] - year_mean
        future_df["Year_centered2"] = future_df["Year_centered"] ** 2

        future_df["Predicted_Population"] = model.predict(
            future_df[["Year_centered", "Year_centered2"]]
        )

        # Combine historical actuals + future predictions for interpolation
        hist_years = d_dec[["Year", "Population"]].rename(columns={"Population": "Value"})
        hist_years["is_future"] = False
        fut_years = future_df[["Year", "Predicted_Population"]].rename(
            columns={"Predicted_Population": "Value"}
        )
        fut_years["is_future"] = True

        full = pd.concat([hist_years, fut_years], ignore_index=True)
        full["Date"] = pd.to_datetime(full["Year"].astype(str) + "-12-31")
        full = full.set_index("Date").sort_index()

        # Build a monthly series and keep only future months after last historical year
        monthly = full[["Value"]].resample("MS").interpolate("linear")

        cutoff_date = pd.to_datetime(f"{last_year}-12-31") + pd.DateOffset(months=1)
        monthly_future = monthly[monthly.index >= cutoff_date].copy()
        monthly_future["District"] = district
        monthly_future.rename(columns={"Value": "Predicted_Population"}, inplace=True)

        all_monthly_forecasts.append(monthly_future)

    future_all = pd.concat(all_monthly_forecasts).reset_index().rename(columns={"index": "Date"})

    models_dir = os.path.join(project_root, 'models')
    os.makedirs(models_dir, exist_ok=True)
    out_path = os.path.join(models_dir, 'district_population_predictions.csv')
    future_all.to_csv(out_path, index=False)
    print(f"✅ District forecast saved to {out_path}")

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    train_district_model()