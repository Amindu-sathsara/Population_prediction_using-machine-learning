import argparse
import os
import warnings
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from prophet import Prophet
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tools.sm_exceptions import ConvergenceWarning, ValueWarning
from xgboost import XGBRegressor


def configure_warnings(show_warnings: bool):
    if show_warnings:
        return

    warnings.filterwarnings("ignore", category=ValueWarning)
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    warnings.filterwarnings("ignore", category=FutureWarning, module="statsmodels")
    warnings.filterwarnings("ignore", category=UserWarning, module="xgboost")


@contextmanager
def suppress_output(enabled: bool = True):
    if not enabled:
        yield
        return

    buf = StringIO()
    with redirect_stdout(buf), redirect_stderr(buf):
        yield


def resolve_eval_max_year(eval_max_year: int | None) -> int:
    if eval_max_year is not None:
        return int(eval_max_year)
    return int(pd.Timestamp.now().year - 1)


def prepare_eval_frame(world_yearly: pd.DataFrame, start_year: int, split_ratio: float, eval_max_year: int | None):
    cutoff_year = resolve_eval_max_year(eval_max_year)

    df = world_yearly[
        (world_yearly["Year"] >= start_year) & (world_yearly["Year"] <= cutoff_year)
    ].copy().sort_values("Year").reset_index(drop=True)

    if len(df) < 10:
        raise ValueError("Not enough rows after start/eval filters. Use a lower start-year.")

    if "Birth_Rate" not in df.columns:
        df["Birth_Rate"] = df.get("Fertility_Rate", np.nan)
    if "Death_Rate" not in df.columns:
        df["Death_Rate"] = df["Birth_Rate"].rolling(3, min_periods=1).mean() * 0.5

    required = [
        "Yearly_Percent_Change",
        "Yearly_Change",
        "Birth_Rate",
        "Death_Rate",
        "Migrants_net",
        "World_Population",
        "Fertility_Rate",
        "Urban_Population",
        "Population_Density",
        "Median_Age",
    ]

    for c in required:
        if c not in df.columns:
            df[c] = np.nan
        df[c] = pd.to_numeric(df[c], errors="coerce")
        df[c] = df[c].interpolate(method="linear", limit_direction="both").ffill().bfill()

    ym = df["Year"].mean()
    df["Year_centered"] = df["Year"] - ym
    df["Year_centered2"] = df["Year_centered"] ** 2
    df["Population_lag1"] = df["National_Population"].shift(1)
    df["Population_lag2"] = df["National_Population"].shift(2)

    for c in required:
        df[f"{c}_lag1"] = df[c].shift(1)

    df = df.dropna().reset_index(drop=True)

    split = max(3, min(int(len(df) * split_ratio), len(df) - 2))
    train = df.iloc[:split].copy()
    test = df.iloc[split:].copy()

    return train, test, cutoff_year


def predict_all_models(train: pd.DataFrame, test: pd.DataFrame):
    y_train = train["National_Population"].values

    base_features = [
        "Year_centered",
        "Year_centered2",
        "Population_lag1",
        "Population_lag2",
        "Yearly_Percent_Change_lag1",
        "Yearly_Change_lag1",
        "Birth_Rate_lag1",
        "Death_Rate_lag1",
        "Migrants_net_lag1",
        "World_Population_lag1",
        "Fertility_Rate_lag1",
        "Urban_Population_lag1",
        "Population_Density_lag1",
        "Median_Age_lag1",
    ]

    exog_features = [
        "Yearly_Percent_Change_lag1",
        "Yearly_Change_lag1",
        "Birth_Rate_lag1",
        "Death_Rate_lag1",
        "Migrants_net_lag1",
        "World_Population_lag1",
        "Fertility_Rate_lag1",
        "Urban_Population_lag1",
        "Population_Density_lag1",
        "Median_Age_lag1",
    ]

    preds = {}

    # LINEAR
    lin_scaler = StandardScaler()
    X_train_lin = lin_scaler.fit_transform(train[base_features])
    X_test_lin = lin_scaler.transform(test[base_features])

    lin = LinearRegression()
    lin.fit(X_train_lin, y_train)
    preds["LINEAR"] = lin.predict(X_test_lin)

    # XGB
    xgb_model = XGBRegressor(
        n_estimators=1000,
        learning_rate=0.01,
        max_depth=3,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42,
    )
    xgb_model.fit(train[base_features], y_train)

    hist = list(train["National_Population"].values)
    pred_xgb = []
    for _, r in test.iterrows():
        feat = pd.DataFrame([
            {
                "Year_centered": r["Year_centered"],
                "Year_centered2": r["Year_centered2"],
                "Population_lag1": hist[-1],
                "Population_lag2": hist[-2],
                "Yearly_Percent_Change_lag1": r["Yearly_Percent_Change_lag1"],
                "Yearly_Change_lag1": r["Yearly_Change_lag1"],
                "Birth_Rate_lag1": r["Birth_Rate_lag1"],
                "Death_Rate_lag1": r["Death_Rate_lag1"],
                "Migrants_net_lag1": r["Migrants_net_lag1"],
                "World_Population_lag1": r["World_Population_lag1"],
                "Fertility_Rate_lag1": r["Fertility_Rate_lag1"],
                "Urban_Population_lag1": r["Urban_Population_lag1"],
                "Population_Density_lag1": r["Population_Density_lag1"],
                "Median_Age_lag1": r["Median_Age_lag1"],
            }
        ])
        p = float(xgb_model.predict(feat)[0])
        pred_xgb.append(p)
        hist.append(p)
    preds["XGB"] = np.array(pred_xgb)

    # SARIMA
    train_indexed = train.set_index("Date").sort_index()
    test_indexed = test.set_index("Date").sort_index()

    ser = train_indexed["National_Population"].astype(float)
    exog_train = train_indexed[exog_features].astype(float)
    exog_test = test_indexed[exog_features].astype(float)

    sar = sm.tsa.statespace.SARIMAX(
        ser,
        exog=exog_train,
        order=(1, 1, 1),
        seasonal_order=(0, 0, 0, 0),
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    sar_res = sar.fit(disp=False)
    preds["SARIMA"] = sar_res.forecast(steps=len(test), exog=exog_test).values

    # PROPHET
    p_train = train[["Date", "National_Population"] + exog_features].rename(
        columns={"Date": "ds", "National_Population": "y"}
    )
    prophet_scaler = StandardScaler()
    p_train_scaled = p_train.copy()
    p_train_scaled[exog_features] = prophet_scaler.fit_transform(p_train[exog_features])

    pro = Prophet(growth="linear", changepoint_prior_scale=0.1)
    for f_name in exog_features:
        pro.add_regressor(f_name)

    with suppress_output(True):
        pro.fit(p_train_scaled)

    future = test[["Date"] + exog_features].rename(columns={"Date": "ds"})
    future[exog_features] = prophet_scaler.transform(future[exog_features])
    preds["PROPHET"] = pro.predict(future)["yhat"].values

    # HOLT-WINTERS
    hw = ExponentialSmoothing(
        train["National_Population"].astype(float).values,
        trend="add",
        seasonal=None,
        damped_trend=True,
        initialization_method="estimated",
    )
    hw_fit = hw.fit(optimized=True)
    preds["HOLT_WINTERS"] = hw_fit.forecast(len(test))

    return preds


def plot_split(train: pd.DataFrame, test: pd.DataFrame, output_dir: Path):
    plt.figure(figsize=(12, 6))
    plt.plot(train["Year"], train["National_Population"], label="Train", linewidth=2)
    plt.plot(test["Year"], test["National_Population"], label="Test", linewidth=2)

    split_year = int(test["Year"].min())
    plt.axvline(split_year, color="black", linestyle="--", linewidth=1.5, label="Split")
    plt.title("Worldometers National Population: Train vs Test Split")
    plt.xlabel("Year")
    plt.ylabel("Population")
    plt.legend()
    plt.tight_layout()

    out = output_dir / "train_test_split.png"
    plt.savefig(out, dpi=200)
    plt.close()


def plot_model_predictions(test: pd.DataFrame, preds: dict[str, np.ndarray], output_dir: Path):
    actual = test["National_Population"].values
    years = test["Year"].values

    for model_name, model_pred in preds.items():
        plt.figure(figsize=(11, 5))
        plt.plot(years, actual, marker="o", linewidth=2, label="Actual")
        plt.plot(years, model_pred, marker="o", linewidth=2, label=f"{model_name} Predicted")
        plt.title(f"Actual vs Predicted: {model_name}")
        plt.xlabel("Year")
        plt.ylabel("Population")
        plt.legend()
        plt.tight_layout()

        out = output_dir / f"actual_vs_predicted_{model_name.lower()}.png"
        plt.savefig(out, dpi=200)
        plt.close()


def plot_model_comparison(metrics_path: Path, output_dir: Path):
    if metrics_path.exists():
        m = pd.read_csv(metrics_path)
        if "MAPE" not in m.columns:
            raise ValueError("Metrics file does not include MAPE column.")
    else:
        fallback_path = metrics_path.parent / "national_best_model_leaderboard.csv"
        if not fallback_path.exists():
            raise FileNotFoundError(
                f"Metrics file not found at {metrics_path}, and fallback not found at {fallback_path}."
            )
        m_fb = pd.read_csv(fallback_path)
        if "model" not in m_fb.columns:
            raise ValueError("Fallback leaderboard file does not contain 'model' column.")

        mape_col = "test_mape" if "test_mape" in m_fb.columns else "val_mape"
        if mape_col not in m_fb.columns:
            raise ValueError("Fallback leaderboard file does not contain test_mape/val_mape.")

        m = m_fb[["model", mape_col]].rename(columns={"model": "Model", mape_col: "MAPE"})
        print(f"Using fallback metrics from: {fallback_path}")

    m["Accuracy_Percent"] = 100 - m["MAPE"]
    m = m.sort_values("Accuracy_Percent", ascending=False).reset_index(drop=True)

    plt.figure(figsize=(12, 6))
    bars = plt.bar(m["Model"], m["Accuracy_Percent"])
    plt.title("Model Comparison (Higher is Better): Accuracy % = 100 - MAPE")
    plt.xlabel("Model")
    plt.ylabel("Accuracy (%)")
    plt.ylim(0, max(100, float(m["Accuracy_Percent"].max()) + 2))

    for bar, acc in zip(bars, m["Accuracy_Percent"]):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.2,
            f"{acc:.2f}%",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()
    out = output_dir / "model_comparison_accuracy.png"
    plt.savefig(out, dpi=200)
    plt.close()


def main(args):
    configure_warnings(args.show_warnings)

    project_root = Path(__file__).resolve().parent.parent
    os.chdir(project_root)

    yearly_path = project_root / "data" / "processed" / "worldometers_national_yearly.csv"
    if not yearly_path.exists():
        raise FileNotFoundError(
            f"Worldometers yearly file not found: {yearly_path}. "
            "Run src/worldometers_pipeline_compare.py first."
        )

    world_yearly = pd.read_csv(yearly_path)
    if "Date" in world_yearly.columns:
        world_yearly["Date"] = pd.to_datetime(world_yearly["Date"])
    else:
        world_yearly["Date"] = pd.to_datetime(world_yearly["Year"].astype(int).astype(str) + "-12-31")

    output_dir = project_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    train, test, cutoff_year = prepare_eval_frame(
        world_yearly=world_yearly,
        start_year=args.start_year,
        split_ratio=args.split_ratio,
        eval_max_year=args.eval_max_year,
    )

    preds = predict_all_models(train, test)

    plot_split(train, test, output_dir)
    plot_model_predictions(test, preds, output_dir)

    metrics_path = project_root / "models" / "worldometers_national_model_metrics_fair.csv"
    plot_model_comparison(metrics_path, output_dir)

    print("\n✅ Validation plots generated")
    print(f"Output folder: {output_dir}")
    print(f"Evaluation cutoff year: <= {cutoff_year}")
    print(f"Train years: {int(train['Year'].min())}..{int(train['Year'].max())}")
    print(f"Test years:  {int(test['Year'].min())}..{int(test['Year'].max())}")
    print("Generated files:")
    print("- train_test_split.png")
    print("- model_comparison_accuracy.png")
    for model_name in preds.keys():
        print(f"- actual_vs_predicted_{model_name.lower()}.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate train/test split and model validation plots for presentation."
    )
    parser.add_argument("--start-year", type=int, default=1990, help="Evaluation start year")
    parser.add_argument("--split-ratio", type=float, default=0.8, help="Chronological train split ratio")
    parser.add_argument(
        "--eval-max-year",
        type=int,
        default=None,
        help="Maximum year to include in evaluation (default: current year - 1)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models/presentation_plots",
        help="Folder where plot PNG files are written",
    )
    parser.add_argument("--show-warnings", action="store_true", help="Show full library warnings output")
    args = parser.parse_args()

    main(args)
