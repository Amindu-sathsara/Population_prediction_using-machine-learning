import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from prophet import Prophet
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
from xgboost import XGBRegressor


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mape(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)


def mase(y_true, y_pred, y_train):
    y_train = np.asarray(y_train, dtype=float)
    if len(y_train) < 2:
        return np.nan

    scale = np.mean(np.abs(y_train[1:] - y_train[:-1]))
    if scale == 0:
        return np.nan

    return float(mean_absolute_error(y_true, y_pred) / scale)


def prepare_common_frame(pop_path: Path, start_year: int) -> pd.DataFrame:
    df = pd.read_excel(pop_path)[["Year", "Population"]].copy()
    df = df[df["Year"] >= start_year].sort_values("Year").reset_index(drop=True)
    df["ds"] = pd.to_datetime(df["Year"].astype(str) + "-12-31")
    return df


def split_frame(df: pd.DataFrame, split_ratio: float):
    split_idx = int(len(df) * split_ratio)
    split_idx = max(1, min(split_idx, len(df) - 1))
    train_df = df.iloc[:split_idx].copy().reset_index(drop=True)
    test_df = df.iloc[split_idx:].copy().reset_index(drop=True)
    return train_df, test_df


def evaluate_linear(train_df: pd.DataFrame, test_df: pd.DataFrame):
    year_mean = train_df["Year"].mean()

    def build_features(frame: pd.DataFrame) -> pd.DataFrame:
        features = frame.copy()
        features["Year_centered"] = features["Year"] - year_mean
        features["Year_centered2"] = features["Year_centered"] ** 2
        return features

    train_features = build_features(train_df)
    test_features = build_features(test_df)

    X_train = train_features[["Year_centered", "Year_centered2"]]
    y_train = train_features["Population"].astype(float)
    X_test = test_features[["Year_centered", "Year_centered2"]]
    y_test = test_features["Population"].astype(float)

    model = LinearRegression()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    return {
        "Model": "LINEAR",
        "RMSE": rmse(y_test, preds),
        "MAE": float(mean_absolute_error(y_test, preds)),
        "MAPE": mape(y_test, preds),
        "MASE": mase(y_test, preds, y_train),
        "Notes": "Yearly polynomial trend with centered year and squared year features.",
    }


def evaluate_xgb(train_df: pd.DataFrame, test_df: pd.DataFrame):
    year_mean = train_df["Year"].mean()
    full_df = pd.concat([train_df, test_df], ignore_index=True).copy()
    full_df["Year_centered"] = full_df["Year"] - year_mean
    full_df["Year_centered2"] = full_df["Year_centered"] ** 2
    full_df["Population_lag1"] = full_df["Population"].shift(1)
    full_df["Population_lag2"] = full_df["Population"].shift(2)

    train_features = full_df.iloc[: len(train_df)].copy().dropna().reset_index(drop=True)

    xgb_model = XGBRegressor(
        n_estimators=1000,
        learning_rate=0.02,
        max_depth=2,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="reg:squarederror",
        random_state=42,
    )

    xgb_model.fit(
        train_features[["Year_centered", "Year_centered2", "Population_lag1", "Population_lag2"]],
        train_features["Population"].astype(float),
    )

    history = list(train_df["Population"].astype(float).values)
    preds = []

    for _, row in test_df.iterrows():
        feature_row = pd.DataFrame([
            {
                "Year_centered": row["Year"] - year_mean,
                "Year_centered2": (row["Year"] - year_mean) ** 2,
                "Population_lag1": history[-1],
                "Population_lag2": history[-2],
            }
        ])
        pred = float(xgb_model.predict(feature_row)[0])
        preds.append(pred)
        history.append(pred)

    y_test = test_df["Population"].astype(float).values
    y_train = train_df["Population"].astype(float).values

    return {
        "Model": "XGB",
        "RMSE": rmse(y_test, preds),
        "MAE": float(mean_absolute_error(y_test, preds)),
        "MAPE": mape(y_test, preds),
        "MASE": mase(y_test, preds, y_train),
        "Notes": "Yearly XGBoost with recursive lagged forecasting.",
    }


def evaluate_sarima(train_df: pd.DataFrame, test_df: pd.DataFrame):
    train_series = train_df.set_index("ds")["Population"].astype(float)
    test_series = test_df["Population"].astype(float).values

    model = SARIMAX(
        train_series,
        order=(1, 1, 1),
        seasonal_order=(0, 0, 0, 0),
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    result = model.fit(disp=False)
    preds = result.get_forecast(steps=len(test_df)).predicted_mean.values

    return {
        "Model": "SARIMA",
        "RMSE": rmse(test_series, preds),
        "MAE": float(mean_absolute_error(test_series, preds)),
        "MAPE": mape(test_series, preds),
        "MASE": mase(test_series, preds, train_series.values),
        "Notes": "Yearly univariate SARIMA(1,1,1) with no seasonal component.",
    }


def evaluate_prophet(train_df: pd.DataFrame, test_df: pd.DataFrame):
    cap_value = float(train_df["Population"].max() * 1.05)

    train_prophet = train_df[["ds", "Population"]].rename(columns={"Population": "y"}).copy()
    train_prophet["cap"] = cap_value

    model = Prophet(
        growth="logistic",
        yearly_seasonality=False,
        weekly_seasonality=False,
        daily_seasonality=False,
        changepoint_prior_scale=0.8,
        n_changepoints=25,
    )
    model.fit(train_prophet)

    future = model.make_future_dataframe(periods=len(test_df), freq="YE")
    future["cap"] = cap_value
    forecast = model.predict(future)

    pred_df = forecast[["ds", "yhat"]].merge(test_df[["ds", "Population"]], on="ds", how="inner")
    preds = pred_df["yhat"].astype(float).values
    y_true = pred_df["Population"].astype(float).values

    return {
        "Model": "PROPHET",
        "RMSE": rmse(y_true, preds),
        "MAE": float(mean_absolute_error(y_true, preds)),
        "MAPE": mape(y_true, preds),
        "MASE": mase(y_true, preds, train_df["Population"].astype(float).values),
        "Notes": "Yearly logistic-growth Prophet model trained on the same split.",
    }


def evaluate_national_models(start_year: int, split_ratio: float, output_name: str):
    base_dir = Path(__file__).resolve().parent
    project_root = base_dir.parent
    pop_path = project_root / "data" / "raw" / "Sri_Lanka_Population_1950_2025.xlsx"

    if not pop_path.exists():
        raise FileNotFoundError(f"Yearly population file not found at {pop_path}.")

    common_df = prepare_common_frame(pop_path, start_year=start_year)
    if len(common_df) < 10:
        raise ValueError("Not enough rows for a fair national evaluation after applying the start year filter.")

    train_df, test_df = split_frame(common_df, split_ratio=split_ratio)

    results = []
    for evaluator in (evaluate_linear, evaluate_xgb, evaluate_sarima, evaluate_prophet):
        results.append(evaluator(train_df, test_df))

    metrics_df = pd.DataFrame(results).sort_values("Model").reset_index(drop=True)
    metrics_df.insert(1, "Train_Start", int(train_df["Year"].min()))
    metrics_df.insert(2, "Train_End", int(train_df["Year"].max()))
    metrics_df.insert(3, "Test_Start", int(test_df["Year"].min()))
    metrics_df.insert(4, "Test_End", int(test_df["Year"].max()))
    metrics_df["Updated_At"] = datetime.now().isoformat(timespec="seconds")

    models_dir = project_root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    output_path = models_dir / output_name
    metrics_df.to_csv(output_path, index=False)

    print(f"Saved fair national metrics to {output_path}")
    print(metrics_df[["Model", "RMSE", "MAE", "MAPE", "MASE"]].to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate national population models on a fair common yearly split.")
    parser.add_argument("--start-year", type=int, default=1990, help="First year to include in the common evaluation window.")
    parser.add_argument("--split-ratio", type=float, default=0.8, help="Chronological train split ratio.")
    parser.add_argument("--output", type=str, default="national_model_metrics_fair.csv", help="Output CSV name inside models/.")
    args = parser.parse_args()

    evaluate_national_models(
        start_year=args.start_year,
        split_ratio=args.split_ratio,
        output_name=args.output,
    )
