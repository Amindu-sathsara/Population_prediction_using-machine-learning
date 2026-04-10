import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import joblib
import os

def forecast_future(model, df, features, steps=120):
    last_data = df.iloc[-1:].copy()
    future_preds = []

    for i in range(steps):
        pred = model.predict(last_data[features])[0]
        future_preds.append(pred)

        last_data['Population_Lag_1M'] = pred
        if 'Population_Lag_12M' in last_data.columns:
            last_data['Population_Lag_12M'] = last_data['Population_Lag_1M']

        next_date = last_data.index[0] + pd.DateOffset(months=1)
        last_data.index = [next_date]
        last_data['Year'] = next_date.year
        last_data['Month'] = next_date.month

    return future_preds

def train_population_model():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(base_dir)
    processed_path = os.path.join(project_root, 'data', 'processed', 'sri_lanka_population_monthly.csv')

    if not os.path.exists(processed_path):
        raise FileNotFoundError(f"Processed national dataset not found at {processed_path}. Run data_preparation.py first.")

    df = pd.read_csv(processed_path, index_col=0, parse_dates=True)

    target = 'Population'
    leakage = ['Birth_Rate','Death_Rate','Growth_Rate',
               'Population_Density','Deaths_per_1000',
               'Urban_Population','Rural_Population','Population_Excel']

    features = [c for c in df.columns if c != target and c not in leakage]

    X = df[features]
    y = df[target]

    split = int(len(df)*0.85)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    # More expressive XGBoost model with regularisation + early stopping
    model = XGBRegressor(
        n_estimators=2000,
        learning_rate=0.01,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42,
    )
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_test, y_test)],
        early_stopping_rounds=100,
        verbose=False,
    )

    preds = model.predict(X_test)

    print("MAE:", mean_absolute_error(y_test, preds))
    print("MAPE:", mean_absolute_percentage_error(y_test, preds)*100)

    # 📊 Plot
    plt.figure(figsize=(12,6))
    plt.plot(y_test.index, y_test, label='Actual')
    plt.plot(y_test.index, preds, label='Predicted')
    plt.legend()
    plt.title("National Population Prediction")

    models_dir = os.path.join(project_root, 'models')
    os.makedirs(models_dir, exist_ok=True)
    import pandas as pd
    from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
    from xgboost import XGBRegressor
    import matplotlib.pyplot as plt
    import joblib
    import os


    def build_yearly_features(df: pd.DataFrame) -> pd.DataFrame:
        """Build features on yearly data instead of interpolated monthly data.

        Expects columns: Year, Population.
        """
        df = df.sort_values("Year").reset_index(drop=True)

        # Simple time trend features
        df["Year_centered"] = df["Year"] - df["Year"].mean()
        df["Year_centered2"] = df["Year_centered"] ** 2

        # Lag features (previous 1 and 2 years)
        df["Population_lag1"] = df["Population"].shift(1)
        df["Population_lag2"] = df["Population"].shift(2)

        df = df.dropna().reset_index(drop=True)
        return df


    def train_population_model():
        base_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(base_dir)
        raw_path = os.path.join(project_root, "data", "raw")

        pop_path = os.path.join(raw_path, "Sri_Lanka_Population_1950_2025.xlsx")
        if not os.path.exists(pop_path):
            raise FileNotFoundError(f"Yearly population file not found at {pop_path}.")

        pop_df = pd.read_excel(pop_path)
        pop_df = pop_df[["Year", "Population"]]

        df = build_yearly_features(pop_df)

        feature_cols = ["Year_centered", "Year_centered2", "Population_lag1", "Population_lag2"]
        X = df[feature_cols]
        y = df["Population"]

        split = int(len(df) * 0.8)
        X_train, X_test = X.iloc[:split], X.iloc[split:]
        y_train, y_test = y.iloc[:split], y.iloc[split:]

        model = XGBRegressor(
            n_estimators=1000,
            learning_rate=0.02,
            max_depth=2,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            random_state=42,
        )

        model.fit(
            X_train,
            y_train,
            eval_set=[(X_test, y_test)],
            early_stopping_rounds=50,
            verbose=False,
        )

        preds = model.predict(X_test)
        print("YEARLY MAE:", mean_absolute_error(y_test, preds))
        print("YEARLY MAPE:", mean_absolute_percentage_error(y_test, preds) * 100)

        models_dir = os.path.join(project_root, "models")
        os.makedirs(models_dir, exist_ok=True)

        # Plot yearly actual vs predicted on the test period
        plt.figure(figsize=(10, 5))
        plt.plot(df["Year"].iloc[split:], y_test.values, label="Actual")
        plt.plot(df["Year"].iloc[split:], preds, label="Predicted")
        plt.xlabel("Year")
        plt.ylabel("Population")
        plt.title("National Population Prediction (Yearly Model)")
        plt.legend()
        plot_path = os.path.join(models_dir, "national_plot_yearly.png")
        plt.savefig(plot_path)

        # Save trained yearly model
        model_path = os.path.join(models_dir, "national_model_yearly.pkl")
        joblib.dump(model, model_path)

        # Forecast yearly population forward (e.g. 2026-2040)
        last_year = int(pop_df["Year"].max())
        horizon_years = 15
        future_years = list(range(last_year + 1, last_year + 1 + horizon_years))

        # Build a combined frame of historical + future years with features
        full_years = pd.DataFrame({"Year": list(pop_df["Year"]) + future_years})
        full_years["Year_centered"] = full_years["Year"] - full_years["Year"].mean()
        full_years["Year_centered2"] = full_years["Year_centered"] ** 2

        # Fill lags iteratively for future years
        full_years["Population"] = None
        full_years.loc[full_years["Year"].isin(pop_df["Year"]), "Population"] = pop_df["Population"].values

        for idx in range(len(pop_df), len(full_years)):
            # compute lags from already-filled Population column
            full_years.loc[idx, "Population_lag1"] = full_years.loc[idx - 1, "Population"]
            full_years.loc[idx, "Population_lag2"] = full_years.loc[idx - 2, "Population"]

            row = full_years.loc[idx, ["Year_centered", "Year_centered2", "Population_lag1", "Population_lag2"]]
            pred_val = model.predict(row.to_frame().T)[0]
            full_years.loc[idx, "Population"] = pred_val

        # Save yearly future predictions
        future_mask = full_years["Year"].isin(future_years)
        future_yearly = full_years.loc[future_mask, ["Year", "Population"]].rename(
            columns={"Population": "Predicted_Population"}
        )
        yearly_out = os.path.join(models_dir, "future_national_population_yearly.csv")
        future_yearly.to_csv(yearly_out, index=False)

        # Also create a monthly forecast CSV for the API by linear interpolation
        all_years = full_years[["Year", "Population"]].copy()
        all_years["Date"] = pd.to_datetime(all_years["Year"].astype(str) + "-12-31")
        all_years = all_years.set_index("Date").sort_index()

        monthly = all_years.resample("MS").interpolate("linear")

        # Keep only months after the last historical year
        cutoff_date = pd.to_datetime(f"{last_year}-12-31") + pd.DateOffset(months=1)
        monthly_future = monthly[monthly.index >= cutoff_date]

        future_monthly = monthly_future.rename(columns={"Population": "Predicted_Population"})
        future_monthly.index.name = "Date"

        monthly_out = os.path.join(models_dir, "future_national_population.csv")
        future_monthly.to_csv(monthly_out)

        print(f"✅ Yearly national forecast saved to {yearly_out}")
        print(f"✅ Monthly national forecast (for API) saved to {monthly_out}")
        print(f"✅ Yearly national model saved to {model_path}")