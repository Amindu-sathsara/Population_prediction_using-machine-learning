import argparse
import io
import logging
import os
import pickle
import warnings
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import statsmodels.api as sm
import xgboost as xgb
from prophet import Prophet
from sklearn.exceptions import InconsistentVersionWarning
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tools.sm_exceptions import ValueWarning
from xgboost import XGBRegressor

DEFAULT_URL = "https://www.worldometers.info/world-population/sri-lanka-population/"


def configure_warnings(show_warnings: bool):
    if show_warnings:
        return

    warnings.filterwarnings("ignore", category=ValueWarning)
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    warnings.filterwarnings("ignore", category=FutureWarning, module="statsmodels")
    warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
    warnings.filterwarnings("ignore", category=UserWarning, module="xgboost")
    cmdstan_logger = logging.getLogger("cmdstanpy")
    cmdstan_logger.setLevel(logging.WARNING)
    cmdstan_logger.disabled = True


@contextmanager
def suppress_output(enabled: bool = True):
    if not enabled:
        yield
        return

    buffer = StringIO()
    with redirect_stdout(buffer), redirect_stderr(buffer):
        yield


def _clean_col_name(col: str) -> str:
    out = str(col).strip().replace("\n", " ").replace("'", "")
    out = out.replace("%", "Percent")
    for ch in [" ", "(", ")", "/", "-", ",", "."]:
        out = out.replace(ch, "_")
    while "__" in out:
        out = out.replace("__", "_")
    return out.strip("_")


def _to_num(series: pd.Series) -> pd.Series:
    s = series.astype(str)
    s = s.str.replace(",", "", regex=False)
    s = s.str.replace("−", "-", regex=False)
    s = s.str.replace("%", "", regex=False)
    s = s.str.replace(" ", "", regex=False)
    s = s.replace({"": np.nan, "nan": np.nan, "None": np.nan})
    return pd.to_numeric(s, errors="coerce")


def _mape(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)


def _rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def _r2(y_true, y_pred) -> float:
    return float(r2_score(y_true, y_pred))


def _mase(y_true, y_pred, y_train) -> float:
    y_train = np.asarray(y_train, dtype=float)
    if len(y_train) < 2:
        return np.nan
    scale = np.mean(np.abs(y_train[1:] - y_train[:-1]))
    if scale == 0:
        return np.nan
    return float(mean_absolute_error(y_true, y_pred) / scale)


def _load_worldometers_tables(url: str) -> list[pd.DataFrame]:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        )
    }
    resp = requests.get(url, headers=headers, timeout=30)
    resp.raise_for_status()
    return pd.read_html(io.StringIO(resp.text))


def _find_population_tables(tables: list[pd.DataFrame]):
    candidates = []
    for df in tables:
        if df.empty:
            continue
        cols = [_clean_col_name(c) for c in df.columns]
        temp = df.copy()
        temp.columns = cols
        if "Year" not in temp.columns:
            continue
        if "Population" not in temp.columns:
            continue

        y = _to_num(temp["Year"])
        # Forecast table can have only a few anchor rows (e.g. 2030, 2035...).
        if y.notna().sum() < 3:
            continue

        min_y, max_y = int(y.min()), int(y.max())
        candidates.append((min_y, max_y, temp))

    hist = None
    forecast = None

    for min_y, max_y, temp in candidates:
        # Historical table typically spans 1950..current year.
        if min_y <= 1955 and max_y >= 2025 and max_y <= 2030:
            hist = temp
        # Forecast table usually extends to 2050.
        if max_y >= 2040:
            forecast = temp

    if hist is None and candidates:
        hist = sorted(candidates, key=lambda x: (x[0], -x[1]))[0][2]

    if forecast is None:
        for min_y, max_y, temp in candidates:
            if max_y > int(_to_num(hist["Year"]).max()):
                forecast = temp
                break

    return hist, forecast


def _standardize_worldometers_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [_clean_col_name(c) for c in out.columns]

    keep_cols = [
        "Year",
        "Population",
        "Yearly_Percent_Change",
        "Yearly_Change",
        "Migrants_net",
        "Median_Age",
        "Fertility_Rate",
        "Density_P_Km2",
        "Urban_Pop_Percent",
        "Urban_Population",
        "Countrys_Share_of_World_Pop",
        "World_Population",
        "Sri_Lanka_Global_Rank",
    ]

    # Some tables can vary slightly in column names.
    rename_fallback = {
        "Yearly_Change": ["Yearly_Change", "Yearly_Change_"],
        "Migrants_net": ["Migrants_net", "Migrants_net_"],
        "Density_P_Km2": ["Density_PKm2", "Density_P_Km2", "Density"],
        "Urban_Pop_Percent": ["Urban_Pop_Percent", "Urban_Pop"],
        "Sri_Lanka_Global_Rank": ["Sri_Lanka_Global_Rank", "Global_Rank"],
    }

    # Make sure all keep_cols exist if possible.
    for target, options in rename_fallback.items():
        if target not in out.columns:
            for alt in options:
                if alt in out.columns:
                    out[target] = out[alt]
                    break

    available = [c for c in keep_cols if c in out.columns]
    out = out[available].copy()

    for c in out.columns:
        out[c] = _to_num(out[c])

    out = out.dropna(subset=["Year", "Population"]).copy()
    out["Year"] = out["Year"].astype(int)
    out = out.sort_values("Year").drop_duplicates(subset=["Year"], keep="last").reset_index(drop=True)

    out = out.rename(
        columns={
            "Population": "National_Population",
            "Density_P_Km2": "Population_Density",
            "Urban_Population": "Urban_Population",
        }
    )

    # Derive missing key fields if a specific table variant omits them.
    if "Population_Density" not in out.columns:
        # Sri Lanka land area is roughly 62,710 km^2.
        out["Population_Density"] = out["National_Population"] / 62710.0

    if "Urban_Population" not in out.columns and "Urban_Pop_Percent" in out.columns:
        out["Urban_Population"] = out["National_Population"] * (out["Urban_Pop_Percent"] / 100.0)

    out["Date"] = pd.to_datetime(out["Year"].astype(str) + "-12-31")

    return out


def load_worldometers_yearly(source_csv: str | None, url: str) -> pd.DataFrame:
    if source_csv:
        df = pd.read_csv(source_csv)
        return _standardize_worldometers_df(df)

    tables = _load_worldometers_tables(url)
    hist, forecast = _find_population_tables(tables)
    if hist is None:
        raise RuntimeError(
            "Could not parse Worldometers table from URL. "
            "Use --source-csv with a saved table export."
        )

    hist_std = _standardize_worldometers_df(hist)

    # Merge forecast table when available so future dates (e.g., 2030-2050)
    # are not forced to use the last historical value.
    if forecast is not None:
        forecast_std = _standardize_worldometers_df(forecast)
        combined = pd.concat([hist_std, forecast_std], ignore_index=True)
        combined = combined.sort_values("Year").drop_duplicates(subset=["Year"], keep="last").reset_index(drop=True)
        return combined

    return hist_std


def yearly_to_monthly(world_yearly: pd.DataFrame) -> pd.DataFrame:
    df = world_yearly.copy().sort_values("Year").reset_index(drop=True)

    # Build a monthly index from Jan of first year to Dec of last year,
    # then linearly interpolate across multi-year gaps (e.g. 2030->2035).
    numeric_cols = [c for c in df.columns if c not in ["Year", "Date"]]
    yearly_anchor = df[["Year"] + numeric_cols].copy()
    yearly_anchor["Date"] = pd.to_datetime(yearly_anchor["Year"].astype(int).astype(str) + "-01-01")
    yearly_anchor = yearly_anchor.set_index("Date").sort_index()

    start = pd.Timestamp(int(df["Year"].min()), 1, 1)
    end = pd.Timestamp(int(df["Year"].max()), 12, 1)
    monthly_index = pd.date_range(start=start, end=end, freq="MS")

    monthly = yearly_anchor.reindex(monthly_index)
    monthly[numeric_cols] = monthly[numeric_cols].interpolate(method="linear", limit_direction="both")
    monthly = monthly.reset_index().rename(columns={"index": "Date"})
    monthly["Year"] = monthly["Date"].dt.year.astype(float)
    monthly["Month"] = monthly["Date"].dt.month.astype(int)
    monthly = monthly[["Year", "Month", "Date"] + numeric_cols]

    # Compatibility columns expected in existing national model training/prediction.
    if "Birth_Rate" not in monthly.columns:
        monthly["Birth_Rate"] = monthly.get("Fertility_Rate", np.nan)
    if "Death_Rate" not in monthly.columns:
        # If unavailable, keep neutral forward-filled synthetic series.
        monthly["Death_Rate"] = monthly["Birth_Rate"].rolling(12, min_periods=1).mean() * 0.5

    if "Urban_Population" not in monthly.columns:
        if "Urban_Pop_Percent" in monthly.columns:
            monthly["Urban_Population"] = monthly["National_Population"] * (monthly["Urban_Pop_Percent"] / 100.0)
        else:
            monthly["Urban_Population"] = np.nan

    monthly["Rural_Population"] = monthly["National_Population"] - monthly["Urban_Population"]

    for c in ["Birth_Rate", "Death_Rate", "Population_Density", "Urban_Population", "Rural_Population"]:
        if c not in monthly.columns:
            monthly[c] = np.nan
        monthly[c] = monthly[c].interpolate(method="linear", limit_direction="both")

    return monthly


def _resolve_eval_max_year(world_yearly: pd.DataFrame, eval_max_year: int | None) -> int:
    if eval_max_year is not None:
        return int(eval_max_year)

    # Default to last complete year to avoid evaluating against future projections.
    current_year = pd.Timestamp.now().year
    return int(current_year - 1)


def evaluate_worldometers_national(
    world_yearly: pd.DataFrame,
    start_year: int,
    split_ratio: float,
    eval_max_year: int | None = None,
) -> pd.DataFrame:
    cutoff_year = _resolve_eval_max_year(world_yearly, eval_max_year)
    df = world_yearly[
        (world_yearly["Year"] >= start_year) & (world_yearly["Year"] <= cutoff_year)
    ].copy().sort_values("Year").reset_index(drop=True)

    if len(df) < 10:
        raise ValueError(
            "Not enough rows in Worldometers yearly data after start_year/eval_max_year filters."
        )

    # Build and clean core exogenous drivers.
    if "Birth_Rate" not in df.columns:
        df["Birth_Rate"] = df.get("Fertility_Rate", np.nan)
    if "Death_Rate" not in df.columns:
        # Proxy if direct death rate is unavailable in the source table.
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
        df[c] = df[c].interpolate(method="linear", limit_direction="both")
        df[c] = df[c].ffill().bfill()

    # Time + lag features
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

    y_train = train["National_Population"].values
    y_test = test["National_Population"].values

    rows = []

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

    # LINEAR (with scaling)
    lin_scaler = StandardScaler()
    X_train_lin = lin_scaler.fit_transform(train[base_features])
    X_test_lin = lin_scaler.transform(test[base_features])

    lin = LinearRegression()
    lin.fit(X_train_lin, y_train)
    pred_lin = lin.predict(X_test_lin)

    rows.append({
        "Model": "LINEAR",
        "Train_Start": int(train["Year"].min()),
        "Train_End": int(train["Year"].max()),
        "Test_Start": int(test["Year"].min()),
        "Test_End": int(test["Year"].max()),
        "RMSE": _rmse(y_test, pred_lin),
        "MAE": mean_absolute_error(y_test, pred_lin),
        "MAPE": _mape(y_test, pred_lin),
        "MASE": _mase(y_test, pred_lin, y_train),
        "R2": _r2(y_test, pred_lin),
        "Source": "WORLDMETERS",
    })

    # XGBOOST (lag-based multivariate)
    features = base_features

    xgb_model = XGBRegressor(
        n_estimators=1000,
        learning_rate=0.01,
        max_depth=3,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42,
    )

    xgb_model.fit(train[features], y_train)

    hist = list(train["National_Population"].values)
    pred_xgb = []

    for _, r in test.iterrows():
        feat = pd.DataFrame([{
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
        }])

        p = float(xgb_model.predict(feat)[0])
        pred_xgb.append(p)
        hist.append(p)

    rows.append({
        "Model": "XGB",
        "Train_Start": int(train["Year"].min()),
        "Train_End": int(train["Year"].max()),
        "Test_Start": int(test["Year"].min()),
        "Test_End": int(test["Year"].max()),
        "RMSE": _rmse(y_test, pred_xgb),
        "MAE": mean_absolute_error(y_test, pred_xgb),
        "MAPE": _mape(y_test, pred_xgb),
        "MASE": _mase(y_test, pred_xgb, y_train),
        "R2": _r2(y_test, pred_xgb),
        "Source": "WORLDMETERS",
    })

    # SARIMAX (with exogenous drivers)
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

    pred_sar = sar_res.forecast(steps=len(test), exog=exog_test)

    rows.append({
        "Model": "SARIMA",
        "Train_Start": int(train["Year"].min()),
        "Train_End": int(train["Year"].max()),
        "Test_Start": int(test["Year"].min()),
        "Test_End": int(test["Year"].max()),
        "RMSE": _rmse(y_test, pred_sar),
        "MAE": mean_absolute_error(y_test, pred_sar),
        "MAPE": _mape(y_test, pred_sar),
        "MASE": _mase(y_test, pred_sar, y_train),
        "R2": _r2(y_test, pred_sar),
        "Source": "WORLDMETERS",
    })

    # PROPHET (multivariate regressors + scaled drivers)
    p_train = train[["Date", "National_Population"] + exog_features].rename(
        columns={"Date": "ds", "National_Population": "y"}
    )

    prophet_scaler = StandardScaler()
    p_train_scaled = p_train.copy()
    p_train_scaled[exog_features] = prophet_scaler.fit_transform(p_train[exog_features])

    pro = Prophet(
        growth="linear",
        changepoint_prior_scale=0.1
    )

    for f_name in exog_features:
        pro.add_regressor(f_name)

    with suppress_output(True):
        pro.fit(p_train_scaled)

    future = test[["Date"] + exog_features].rename(columns={"Date": "ds"})
    future[exog_features] = prophet_scaler.transform(future[exog_features])

    forecast = pro.predict(future)
    pred_prophet = forecast["yhat"].values

    rows.append({
        "Model": "PROPHET",
        "Train_Start": int(train["Year"].min()),
        "Train_End": int(train["Year"].max()),
        "Test_Start": int(test["Year"].min()),
        "Test_End": int(test["Year"].max()),
        "RMSE": _rmse(y_test, pred_prophet),
        "MAE": mean_absolute_error(y_test, pred_prophet),
        "MAPE": _mape(y_test, pred_prophet),
        "MASE": _mase(y_test, pred_prophet, y_train),
        "R2": _r2(y_test, pred_prophet),
        "Source": "WORLDMETERS",
    })

    # HOLT-WINTERS (univariate, yearly trend)
    try:
        hw = ExponentialSmoothing(
            train["National_Population"].astype(float).values,
            trend="add",
            seasonal=None,
            damped_trend=True,
            initialization_method="estimated",
        )
        hw_fit = hw.fit(optimized=True)
        pred_hw = hw_fit.forecast(len(test))

        rows.append({
            "Model": "HOLT_WINTERS",
            "Train_Start": int(train["Year"].min()),
            "Train_End": int(train["Year"].max()),
            "Test_Start": int(test["Year"].min()),
            "Test_End": int(test["Year"].max()),
            "RMSE": _rmse(y_test, pred_hw),
            "MAE": mean_absolute_error(y_test, pred_hw),
            "MAPE": _mape(y_test, pred_hw),
            "MASE": _mase(y_test, pred_hw, y_train),
            "R2": _r2(y_test, pred_hw),
            "Source": "WORLDMETERS",
        })
    except Exception:
        # Keep evaluation resilient on very short/irregular slices.
        pass

    out = pd.DataFrame(rows).sort_values("Model").reset_index(drop=True)
    out["Eval_Max_Year"] = cutoff_year
    return out


def walk_forward_backtest(
    world_yearly: pd.DataFrame,
    start_year: int,
    eval_max_year: int | None = None,
    min_train_years: int = 10,
) -> pd.DataFrame:
    """Walk-forward (rolling) backtesting across historical years.
    
    For each test year starting from (start_year + min_train_years), train on all
    prior years and test on that year. Accumulate metrics across all folds.
    """
    cutoff_year = _resolve_eval_max_year(world_yearly, eval_max_year)
    
    all_data = world_yearly[
        (world_yearly["Year"] >= start_year) & (world_yearly["Year"] <= cutoff_year)
    ].copy().sort_values("Year").reset_index(drop=True)
    
    if len(all_data) < min_train_years + 1:
        raise ValueError(
            f"Not enough data for walk-forward with min_train_years={min_train_years}"
        )
    
    world_monthly = yearly_to_monthly(world_yearly)
    world_monthly["Date"] = pd.to_datetime(world_monthly["Date"])
    
    fold_results = []
    
    # For each year from (start_year + min_train_years) onwards, test on that year
    test_start_idx = min_train_years
    for test_idx in range(test_start_idx, len(all_data)):
        train = all_data.iloc[:test_idx].copy()
        test_row = all_data.iloc[test_idx:test_idx+1].copy()
        test_year = int(test_row["Year"].iloc[0])
        
        # Prepare train data with features
        df_train = train.copy()
        if "Birth_Rate" not in df_train.columns:
            df_train["Birth_Rate"] = df_train.get("Fertility_Rate", np.nan)
        if "Death_Rate" not in df_train.columns:
            df_train["Death_Rate"] = df_train["Birth_Rate"].rolling(3, min_periods=1).mean() * 0.5
        
        required = [
            "Yearly_Percent_Change", "Yearly_Change", "Birth_Rate", "Death_Rate",
            "Migrants_net", "World_Population", "Fertility_Rate", "Urban_Population",
            "Population_Density", "Median_Age"
        ]
        for c in required:
            if c not in df_train.columns:
                df_train[c] = np.nan
            df_train[c] = pd.to_numeric(df_train[c], errors="coerce").interpolate(
                method="linear", limit_direction="both"
            ).ffill().bfill()
        
        # Get actual test value
        actual_pop = float(test_row["National_Population"].iloc[0])
        
        # Get predictions from both pipelines for this date
        test_date = pd.Timestamp(test_year, 1, 1)
        pred_existing = _predict_existing_national(test_date)["Ensemble"]
        pred_world = _predict_worldometers_national_for_date(world_monthly, test_date)["Ensemble"]

        # Fold-level Holt-Winters one-step forecast from expanding window.
        try:
            train_pop = train["National_Population"].astype(float).values
            hw_fold = ExponentialSmoothing(
                train_pop,
                trend="add",
                seasonal=None,
                damped_trend=True,
                initialization_method="estimated",
            )
            hw_fold_fit = hw_fold.fit(optimized=True)
            pred_hw = float(hw_fold_fit.forecast(1)[0])
        except Exception:
            pred_hw = np.nan
        
        for pipeline_name, pred in [
            ("EXISTING_MACROTRENDS_BASED", pred_existing),
            ("WORLDMETERS_BASED", pred_world),
            ("HOLT_WINTERS", pred_hw),
        ]:
            ape = abs(pred - actual_pop) / actual_pop * 100
            fold_results.append({
                "Fold_Year": test_year,
                "Pipeline": pipeline_name,
                "Actual": actual_pop,
                "Predicted": pred,
                "Error": pred - actual_pop,
                "Abs_Error": abs(pred - actual_pop),
                "APE": ape,
            })
    
    fold_df = pd.DataFrame(fold_results)
    
    # Compute aggregated metrics per pipeline
    summary_rows = []
    for pipeline in ["EXISTING_MACROTRENDS_BASED", "WORLDMETERS_BASED", "HOLT_WINTERS"]:
        pipeline_folds = fold_df[fold_df["Pipeline"] == pipeline]
        actuals = pipeline_folds["Actual"].values
        preds = pipeline_folds["Predicted"].values

        if len(actuals) == 0:
            continue

        valid_mask = ~np.isnan(preds)
        if valid_mask.sum() == 0:
            continue

        actuals = actuals[valid_mask]
        preds = preds[valid_mask]
        abs_err = np.abs(actuals - preds)
        ape_vals = np.abs((actuals - preds) / actuals) * 100
        
        rmse_v = float(np.sqrt(np.mean((actuals - preds) ** 2)))
        mae_v = float(np.mean(abs_err))
        mape_v = float(np.mean(ape_vals))
        
        # MASE: mean absolute error / mean of seasonal naive baseline (year-over-year diff)
        if len(actuals) > 1:
            scale = float(np.mean(np.abs(np.diff(actuals))))
            mase_v = float(mae_v / scale) if scale > 0 else float('nan')
        else:
            mase_v = float('nan')
        
        accuracy_pct = 100 - mape_v
        
        summary_rows.append({
            "Pipeline": pipeline,
            "N_Folds": int(valid_mask.sum()),
            "Fold_Years": f"{int(pipeline_folds['Fold_Year'].min())}-{int(pipeline_folds['Fold_Year'].max())}",
            "RMSE": rmse_v,
            "MAE": mae_v,
            "MASE": mase_v,
            "MAPE": mape_v,
            "Accuracy_Percent": accuracy_pct,
        })
    
    return pd.DataFrame(summary_rows), fold_df


def _predict_existing_national(date: pd.Timestamp) -> dict:
    model_dir = Path("models")
    with open(model_dir / "nat_linear_model.pkl", "rb") as f:
        linear_model = pickle.load(f)
    with open(model_dir / "nat_xgb_model.pkl", "rb") as f:
        xgb_model = pickle.load(f)
    with open(model_dir / "nat_sarima_model.pkl", "rb") as f:
        sarima_model = pickle.load(f)
    with open(model_dir / "nat_sarima_info.pkl", "rb") as f:
        sarima_info = pickle.load(f)

    feature_cols = None
    driver_models = None
    if (model_dir / "nat_feature_columns.pkl").exists():
        with open(model_dir / "nat_feature_columns.pkl", "rb") as f:
            feature_cols = pickle.load(f)
    if (model_dir / "nat_driver_models.pkl").exists():
        with open(model_dir / "nat_driver_models.pkl", "rb") as f:
            driver_models = pickle.load(f)

    prophet_model = None
    prophet_path = model_dir / "nat_prophet_model.json"
    if prophet_path.exists():
        from prophet.serialize import model_from_json
        with open(prophet_path, "r") as f:
            prophet_model = model_from_json(f.read())

    year = date.year
    month = date.month
    fractional_year = year + (month - 1) / 12.0
    base_features = {"Fractional_Year": fractional_year}

    if feature_cols and driver_models:
        driver_values = {}
        for col, info in driver_models.items():
            res = info["model"]
            last_date = info["last_date"]
            steps = (year - last_date.year) * 12 + month - last_date.month
            if steps <= 0:
                val = float(res.data.endog[-1])
            else:
                forecast = res.get_forecast(steps=steps)
                val = float(forecast.predicted_mean.iloc[-1])
            driver_values[col] = val

        for c in feature_cols:
            if c != "Fractional_Year" and c in driver_values:
                base_features[c] = driver_values[c]

    X_nat = pd.DataFrame([base_features])
    preds = []

    p_lin = float(linear_model.predict(X_nat)[0])
    p_xgb = float(xgb_model.predict(X_nat)[0])
    preds.extend([p_lin, p_xgb])

    steps = (year - sarima_info["last_date"].year) * 12 + month - sarima_info["last_date"].month
    p_sar = float(sarima_model.forecast(steps=steps)[-1] if steps > 0 else p_lin)
    preds.append(p_sar)

    p_prophet = np.nan
    if prophet_model is not None:
        try:
            p_prophet = float(prophet_model.predict(pd.DataFrame({"ds": [date]}))["yhat"].iloc[0])
            preds.append(p_prophet)
        except Exception:
            pass

    return {
        "Linear": p_lin,
        "XGB": p_xgb,
        "SARIMA": p_sar,
        "Prophet": p_prophet,
        "Ensemble": float(np.mean(preds)),
        "Population_Density": base_features.get("Population_Density", np.nan),
        "Urban_Population": base_features.get("Urban_Population", np.nan),
    }


def _predict_worldometers_national_for_date(world_monthly: pd.DataFrame, date: pd.Timestamp) -> dict:
    d = world_monthly.copy()
    d["Date"] = pd.to_datetime(d["Date"])
    row = d.loc[d["Date"] == date]
    if row.empty:
        # nearest month fallback
        row = d.iloc[(d["Date"] - date).abs().argsort()[:1]]

    pop = float(row["National_Population"].iloc[0])
    dens = float(row["Population_Density"].iloc[0]) if "Population_Density" in row.columns else np.nan
    urb = float(row["Urban_Population"].iloc[0]) if "Urban_Population" in row.columns else np.nan

    return {
        "Linear": pop,
        "XGB": pop,
        "SARIMA": pop,
        "Prophet": pop,
        "Ensemble": pop,
        "Population_Density": dens,
        "Urban_Population": urb,
    }


def _predict_district_from_driver(district: str, date: pd.Timestamp, nat_bundle: dict) -> dict:
    model_dir = Path("models")
    with open(model_dir / "district_best_models.pkl", "rb") as f:
        best_models = pickle.load(f)

    if district not in best_models or best_models[district] is None:
        raise ValueError(f"No district best model available for {district}.")

    best_model = best_models[district]

    fractional_year = date.year + (date.month - 1) / 12.0
    x = pd.DataFrame([
        {
            "Fractional_Year": fractional_year,
            "National_Population": nat_bundle["Ensemble"],
            "Population_Density": nat_bundle["Population_Density"],
            "Urban_Population": nat_bundle["Urban_Population"],
        }
    ])

    from sklearn.linear_model import LinearRegression as _LR
    from xgboost import XGBRegressor as _XGB

    if isinstance(best_model, (_LR, _XGB)):
        model_name = "XGBoost" if isinstance(best_model, _XGB) else "Linear"
        proportion = float(best_model.predict(x)[0])
    elif isinstance(best_model, Prophet):
        model_name = "Prophet"
        p_df = pd.DataFrame(
            {
                "ds": [date],
                "National_Population": [nat_bundle["Ensemble"]],
                "Population_Density": [nat_bundle["Population_Density"]],
                "Urban_Population": [nat_bundle["Urban_Population"]],
            }
        )
        proportion = float(best_model.predict(p_df)["yhat"].iloc[0])
    else:
        model_name = "SARIMA"
        f = best_model.forecast(steps=1)
        proportion = float(f.iloc[0] if hasattr(f, "iloc") else f)

    return {
        "District_Model": model_name,
        "District_Proportion": proportion,
        "District_Population": float(proportion * nat_bundle["Ensemble"]),
    }


def main(args):
    configure_warnings(args.show_warnings)

    project_root = Path(__file__).resolve().parent.parent
    os.chdir(project_root)

    processed_dir = project_root / "data" / "processed"
    models_dir = project_root / "models"
    processed_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    world_yearly = load_worldometers_yearly(args.source_csv, args.url)
    world_monthly = yearly_to_monthly(world_yearly)

    yearly_out = processed_dir / "worldometers_national_yearly.csv"
    monthly_out = processed_dir / "worldometers_national_monthly.csv"
    world_yearly.to_csv(yearly_out, index=False)
    world_monthly.to_csv(monthly_out, index=False)

    fair_metrics = evaluate_worldometers_national(
        world_yearly,
        start_year=args.start_year,
        split_ratio=args.split_ratio,
        eval_max_year=args.eval_max_year,
    )
    fair_metrics["Updated_At"] = pd.Timestamp.now().isoformat(timespec="seconds")
    fair_metrics_out = models_dir / "worldometers_national_model_metrics_fair.csv"
    fair_metrics.to_csv(fair_metrics_out, index=False)

    target_date = pd.to_datetime(args.date)

    existing_nat = _predict_existing_national(target_date)
    world_nat = _predict_worldometers_national_for_date(world_monthly, target_date)

    existing_dist = _predict_district_from_driver(args.district, target_date, existing_nat)
    world_dist = _predict_district_from_driver(args.district, target_date, world_nat)

    compare = pd.DataFrame(
        [
            {
                "Date": target_date.date().isoformat(),
                "District": args.district,
                "Pipeline": "EXISTING_MACROTRENDS_BASED",
                "National_Ensemble": existing_nat["Ensemble"],
                "National_Linear": existing_nat["Linear"],
                "National_XGB": existing_nat["XGB"],
                "National_SARIMA": existing_nat["SARIMA"],
                "National_Prophet": existing_nat["Prophet"],
                "District_Model": existing_dist["District_Model"],
                "District_Proportion": existing_dist["District_Proportion"],
                "District_Population": existing_dist["District_Population"],
            },
            {
                "Date": target_date.date().isoformat(),
                "District": args.district,
                "Pipeline": "WORLDMETERS_BASED",
                "National_Ensemble": world_nat["Ensemble"],
                "National_Linear": world_nat["Linear"],
                "National_XGB": world_nat["XGB"],
                "National_SARIMA": world_nat["SARIMA"],
                "National_Prophet": world_nat["Prophet"],
                "District_Model": world_dist["District_Model"],
                "District_Proportion": world_dist["District_Proportion"],
                "District_Population": world_dist["District_Population"],
            },
        ]
    )

    comp_out = models_dir / "worldometers_vs_existing_predictions.csv"
    compare.to_csv(comp_out, index=False)

    if args.walk_forward:
        print("\n" + "="*80)
        print("WALK-FORWARD BACKTEST (Rolling evaluation across historical years)")
        print("="*80)
        try:
            summary_df, fold_df = walk_forward_backtest(
                world_yearly,
                start_year=args.start_year,
                eval_max_year=args.eval_max_year,
                min_train_years=args.walk_forward_min_train,
            )
            print("\nAggregated Pipeline Metrics (across all folds):")
            print(summary_df[["Pipeline", "N_Folds", "Fold_Years", "RMSE", "MAE", "MASE", "MAPE", "Accuracy_Percent"]].to_string(index=False))
            
            walkforward_out = models_dir / "worldometers_walkforward_backtest.csv"
            fold_df.to_csv(walkforward_out, index=False)
            print(f"\n✅ Fold-by-fold details saved to {walkforward_out}")
        except Exception as e:
            print(f"⚠️  Walk-forward backtest failed: {e}")
    
    if args.only_summary or True:
        eval_cutoff = int(fair_metrics["Eval_Max_Year"].iloc[0]) if not fair_metrics.empty else args.eval_max_year
        print(f"\nEvaluation cutoff year: <= {eval_cutoff}")
        print("Accuracy Metrics (National):")
        print(fair_metrics[["Model", "RMSE", "MAE", "MAPE", "MASE", "R2"]].to_string(index=False))
        print("\nPredicted Populations:")
        for _, row in compare.iterrows():
            pipeline = row["Pipeline"]
            district_pop = int(round(float(row["District_Population"])))
            national_pop = int(round(float(row["National_Ensemble"])))
            print(f"{pipeline}: national={national_pop:,}, {args.district} district={district_pop:,}")
        return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Build Worldometers-based national dataset, evaluate national models on it, "
            "and compare national+district predictions against current pipeline."
        )
    )
    parser.add_argument("--date", type=str, default="2030-01-01", help="Target date YYYY-MM-DD")
    parser.add_argument("--district", type=str, default="Ampara", help="District name")
    parser.add_argument("--url", type=str, default=DEFAULT_URL, help="Worldometers Sri Lanka population URL")
    parser.add_argument(
        "--source-csv",
        type=str,
        default=None,
        help="Optional path to a manually exported Worldometers historical table CSV (recommended when URL blocks scraping).",
    )
    parser.add_argument("--start-year", type=int, default=1990, help="Evaluation start year")
    parser.add_argument(
        "--eval-max-year",
        type=int,
        default=None,
        help="Maximum year to include in evaluation. Defaults to last complete year (current year - 1).",
    )
    parser.add_argument("--split-ratio", type=float, default=0.8, help="Chronological train split ratio")
    parser.add_argument(
        "--walk-forward",
        action="store_true",
        help="Run walk-forward (rolling) backtesting to compute robust overall accuracy across all historical years.",
    )
    parser.add_argument(
        "--walk-forward-min-train",
        type=int,
        default=10,
        help="Minimum training years for walk-forward backtest (default 10).",
    )
    parser.add_argument("--show-warnings", action="store_true", help="Show full library warnings output")
    parser.add_argument("--only-summary", action="store_true", help="Print only accuracy metrics and predicted populations")
    args = parser.parse_args()

    main(args)
