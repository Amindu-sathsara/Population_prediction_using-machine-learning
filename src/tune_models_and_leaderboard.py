import argparse
import json
import warnings
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tools.sm_exceptions import ConvergenceWarning, ValueWarning
from statsmodels.tsa.forecasting.theta import ThetaModel
from statsmodels.tsa.holtwinters import ExponentialSmoothing


def mape(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.where(y_true == 0, np.nan, y_true)
    return float(np.nanmean(np.abs((y_true - y_pred) / denom)) * 100)


def rmse(y_true, y_pred):
    # Guard against occasional exploding forecasts during broad grid-search.
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    err = y_true - y_pred
    if not np.all(np.isfinite(err)):
        return float("inf")
    max_abs_err = np.max(np.abs(err)) if err.size else 0.0
    if max_abs_err > 1e150:
        return float("inf")
    return float(np.sqrt(np.mean(err * err)))


def build_three_way_split(df: pd.DataFrame, train_ratio: float, val_ratio: float):
    n = len(df)
    train_end = max(24, int(n * train_ratio))
    val_end = int(n * (train_ratio + val_ratio))
    val_end = max(train_end + 12, val_end)
    val_end = min(val_end, n - 12)

    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()
    return train_df, val_df, test_df


def add_national_features(df: pd.DataFrame):
    out = df.copy().sort_values("Date").reset_index(drop=True)
    out["Fractional_Year"] = out["Year"] + (out["Month"] - 1) / 12.0
    out["Month_Sin"] = np.sin(2 * np.pi * out["Month"] / 12.0)
    out["Month_Cos"] = np.cos(2 * np.pi * out["Month"] / 12.0)
    out["lag_1"] = out["National_Population"].shift(1)
    out["lag_12"] = out["National_Population"].shift(12)
    out["rolling_mean_3"] = out["National_Population"].rolling(window=3, min_periods=3).mean()
    for col in ["Birth_Rate", "Death_Rate", "Population_Density", "Rural_Population", "Urban_Population"]:
        if col in out.columns:
            out[f"{col}_lag1"] = out[col].shift(1)
    out = out.dropna().reset_index(drop=True)
    return out


def forecast_seasonal_naive(train_series: pd.Series, horizon: int, season_length: int = 12) -> np.ndarray:
    values = train_series.values
    history = list(values)
    preds = []
    for _ in range(horizon):
        if len(history) >= season_length:
            pred = history[-season_length]
        else:
            pred = history[-1]
        preds.append(pred)
        history.append(pred)
    return np.asarray(preds, dtype=float)


def eval_preds(y_train, y_val, y_test, p_train, p_val, p_test):
    return {
        "train_rmse": rmse(y_train, p_train),
        "train_mae": float(mean_absolute_error(y_train, p_train)),
        "train_mape": mape(y_train, p_train),
        "val_rmse": rmse(y_val, p_val),
        "val_mae": float(mean_absolute_error(y_val, p_val)),
        "val_mape": mape(y_val, p_val),
        "test_rmse": rmse(y_test, p_test),
        "test_mae": float(mean_absolute_error(y_test, p_test)),
        "test_mape": mape(y_test, p_test),
    }


def compute_selection_score(train_mape: float, val_mape: float) -> float:
    train_floor = max(float(train_mape), 0.5)
    ratio = float(val_mape) / train_floor
    penalty = max(0.0, ratio - 1.0)
    return float(val_mape) * (1.0 + 0.5 * penalty)


def fit_sarimax_safely(model, maxiter: int = 200):
    """Fit SARIMAX-like models while suppressing warnings and rejecting non-converged fits."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            warnings.simplefilter("ignore", UserWarning)
            res = model.fit(disp=False, maxiter=maxiter)
    except Exception:
        return None

    mle_retvals = getattr(res, "mle_retvals", {}) or {}
    if isinstance(mle_retvals, dict):
        converged = mle_retvals.get("converged", True)
        if not converged:
            return None
    return res


def tune_national(processed_dir: Path, models_dir: Path, train_ratio: float, val_ratio: float):
    df = pd.read_csv(processed_dir / "national_master_monthly.csv")
    df["Date"] = pd.to_datetime(df["Date"])
    df = add_national_features(df)

    # Quick diagnostics for exogenous driver strength and lag effects
    corr_cols = [
        "National_Population",
        "Birth_Rate",
        "Death_Rate",
        "Population_Density",
        "Rural_Population",
        "Urban_Population",
        "Birth_Rate_lag1",
        "Death_Rate_lag1",
        "Population_Density_lag1",
        "Rural_Population_lag1",
        "Urban_Population_lag1",
    ]
    corr_cols = [c for c in corr_cols if c in df.columns]
    if len(corr_cols) > 1:
        corr = df[corr_cols].corr(numeric_only=True)[["National_Population"]].sort_values("National_Population", ascending=False)
        corr.to_csv(models_dir / "national_driver_correlations.csv")

    feature_cols = [
        "Fractional_Year",
        "Month_Sin",
        "Month_Cos",
        "lag_1",
        "lag_12",
        "rolling_mean_3",
        "Birth_Rate",
        "Death_Rate",
        "Population_Density",
        "Rural_Population",
        "Urban_Population",
        "Birth_Rate_lag1",
        "Death_Rate_lag1",
        "Population_Density_lag1",
        "Rural_Population_lag1",
        "Urban_Population_lag1",
    ]

    feature_cols = [c for c in feature_cols if c in df.columns]

    train_df, val_df, test_df = build_three_way_split(df, train_ratio, val_ratio)

    X_train = train_df[feature_cols]
    y_train = train_df["National_Population"].astype(float).values
    X_val = val_df[feature_cols]
    y_val = val_df["National_Population"].astype(float).values
    X_test = test_df[feature_cols]
    y_test = test_df["National_Population"].astype(float).values

    rows = []

    # Linear baseline
    lin = LinearRegression()
    lin.fit(X_train, y_train)
    scores = eval_preds(
        y_train,
        y_val,
        y_test,
        lin.predict(X_train),
        lin.predict(X_val),
        lin.predict(X_test),
    )
    rows.append({"level": "NATIONAL", "model": "LINEAR", "params": "{}", **scores})

    # XGBoost tuning
    xgb_grid = [
        {
            "n_estimators": 800,
            "max_depth": 1,
            "learning_rate": 0.02,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 8,
            "reg_lambda": 8.0,
            "reg_alpha": 1.0,
        },
        {
            "n_estimators": 1200,
            "max_depth": 2,
            "learning_rate": 0.01,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 10,
            "reg_lambda": 10.0,
            "reg_alpha": 1.5,
        },
    ]
    for params in xgb_grid:
        model = xgb.XGBRegressor(
            objective="reg:squarederror",
            random_state=42,
            **params,
        )
        model.fit(X_train, y_train)
        scores = eval_preds(
            y_train,
            y_val,
            y_test,
            model.predict(X_train),
            model.predict(X_val),
            model.predict(X_test),
        )
        rows.append({"level": "NATIONAL", "model": "XGBOOST", "params": json.dumps(params), **scores})

    # Random forest tuning
    rf_grid = [
        {"n_estimators": 200, "max_depth": 4, "min_samples_leaf": 8},
        {"n_estimators": 300, "max_depth": 5, "min_samples_leaf": 10},
    ]
    for params in rf_grid:
        model = RandomForestRegressor(random_state=42, n_jobs=-1, **params)
        model.fit(X_train, y_train)
        scores = eval_preds(
            y_train,
            y_val,
            y_test,
            model.predict(X_train),
            model.predict(X_val),
            model.predict(X_test),
        )
        rows.append({"level": "NATIONAL", "model": "RANDOM_FOREST", "params": json.dumps(params), **scores})

    y_train_series = pd.Series(y_train, index=train_df["Date"]).asfreq("MS")
    y_val_series = pd.Series(y_val, index=val_df["Date"]).asfreq("MS")
    y_test_series = pd.Series(y_test, index=test_df["Date"]).asfreq("MS")
    y_train_series.index = pd.DatetimeIndex(y_train_series.index, freq="MS")
    y_val_series.index = pd.DatetimeIndex(y_val_series.index, freq="MS")
    y_test_series.index = pd.DatetimeIndex(y_test_series.index, freq="MS")

    # ARIMA
    arima = sm.tsa.statespace.SARIMAX(
        y_train_series,
        order=(1, 1, 1),
        seasonal_order=(0, 0, 0, 0),
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    arima_res = fit_sarimax_safely(arima, maxiter=200)
    if arima_res is not None:
        p_train = arima_res.fittedvalues.iloc[1:]
        p_val = arima_res.forecast(steps=len(y_val_series))
        p_test = arima_res.forecast(steps=len(y_val_series) + len(y_test_series)).iloc[len(y_val_series):]
        scores = eval_preds(y_train_series.iloc[1:].values, y_val_series.values, y_test_series.values, p_train.values, p_val.values, p_test.values)
        rows.append({"level": "NATIONAL", "model": "ARIMA", "params": json.dumps({"order": [1, 1, 1]}), **scores})

    # SARIMA grid search: p,d,q in [0..2], P,D,Q in [0..1], seasonal period 12
    p = d = q = range(0, 3)
    P = D = Q = range(0, 2)
    seasonal_periods = [12]
    sarima_param_grid = list(product(p, d, q, P, D, Q, seasonal_periods))

    for p_, d_, q_, P_, D_, Q_, sp in sarima_param_grid:
        try:
            sarima = sm.tsa.statespace.SARIMAX(
                y_train_series,
                order=(p_, d_, q_),
                seasonal_order=(P_, D_, Q_, sp),
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            sarima_res = fit_sarimax_safely(sarima, maxiter=200)
            if sarima_res is None:
                continue

            warmup = max(1, sp + d_ + D_ * sp)
            p_train = sarima_res.fittedvalues.iloc[warmup:]
            p_val = sarima_res.forecast(steps=len(y_val_series))
            p_test = sarima_res.forecast(steps=len(y_val_series) + len(y_test_series)).iloc[len(y_val_series):]

            if len(p_train) == 0:
                continue

            scores = eval_preds(
                y_train_series.iloc[warmup:].values,
                y_val_series.values,
                y_test_series.values,
                p_train.values,
                p_val.values,
                p_test.values,
            )

            train_n = len(y_train_series)
            k = p_ + d_ + q_ + P_ + D_ + Q_ + 1
            aic = float(sarima_res.aic)
            if train_n - k - 1 > 0:
                aicc = aic + (2 * k * (k + 1)) / (train_n - k - 1)
            else:
                aicc = np.nan

            rows.append(
                {
                    "level": "NATIONAL",
                    "model": "SARIMA",
                    "params": json.dumps({"order": [p_, d_, q_], "seasonal_order": [P_, D_, Q_, sp]}),
                    "aic": aic,
                    "aicc": aicc,
                    **scores,
                }
            )
        except Exception:
            continue

    # SARIMAX
    exog_cols = ["Birth_Rate", "Death_Rate", "Population_Density", "Rural_Population", "Urban_Population"]
    sarimax = sm.tsa.statespace.SARIMAX(
        y_train_series,
        exog=train_df.set_index("Date")[exog_cols],
        order=(1, 1, 1),
        seasonal_order=(0, 0, 0, 0),
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    sarimax_res = fit_sarimax_safely(sarimax, maxiter=200)
    if sarimax_res is not None:
        p_train = sarimax_res.fittedvalues.iloc[1:]
        p_val = sarimax_res.forecast(steps=len(y_val_series), exog=val_df.set_index("Date")[exog_cols])
        p_test = sarimax_res.forecast(
            steps=len(y_val_series) + len(y_test_series),
            exog=pd.concat([val_df.set_index("Date")[exog_cols], test_df.set_index("Date")[exog_cols]]),
        ).iloc[len(y_val_series):]
        scores = eval_preds(y_train_series.iloc[1:].values, y_val_series.values, y_test_series.values, p_train.values, p_val.values, p_test.values)
        rows.append({"level": "NATIONAL", "model": "SARIMAX", "params": json.dumps({"order": [1, 1, 1]}), **scores})

    cand = pd.DataFrame(rows)
    cand["val_train_ratio"] = cand["val_mape"] / cand["train_mape"].clip(lower=0.5)
    cand["selection_score"] = cand.apply(
        lambda r: compute_selection_score(r["train_mape"], r["val_mape"]), axis=1
    )
    cand = cand.sort_values(["selection_score", "val_rmse", "test_rmse"]).reset_index(drop=True)
    cand.to_csv(models_dir / "national_tuning_candidates.csv", index=False)

    # Leaderboard: best by model family + overall best
    best_per_model = cand.sort_values(["selection_score", "val_rmse"]).groupby("model", as_index=False).first()
    best_per_model = best_per_model.sort_values(["selection_score", "val_rmse"]).reset_index(drop=True)
    best_per_model["rank_by_val"] = np.arange(1, len(best_per_model) + 1)
    best_per_model.to_csv(models_dir / "national_best_model_leaderboard.csv", index=False)

    return cand, best_per_model


def tune_district(processed_dir: Path, models_dir: Path, train_ratio: float, val_ratio: float):
    df = pd.read_csv(processed_dir / "district_master_monthly.csv")
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(["District", "Date"]).reset_index(drop=True)
    df["Fractional_Year"] = df["Year"] + (df["Month"] - 1) / 12.0
    df["month_sin"] = np.sin(2 * np.pi * df["Month"] / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * df["Month"] / 12.0)
    df["Proportion"] = df["District_Total"] / df["National_Population"]

    if "national_lag_1" not in df.columns:
        df["national_lag_1"] = df["National_Population"].shift(1)
    if "national_lag_12" not in df.columns:
        df["national_lag_12"] = df["National_Population"].shift(12)
    df = df.dropna().reset_index(drop=True)

    feature_cols = [
        "Fractional_Year",
        "National_Population",
        "Population_Density",
        "Urban_Population",
        "month_sin",
        "month_cos",
        "national_lag_1",
        "national_lag_12",
    ]
    feature_cols = [c for c in feature_cols if c in df.columns]
    districts = sorted(df["District"].unique())

    all_rows = []
    best_rows = []

    for district in districts:
        d_df = df[df["District"] == district].copy().reset_index(drop=True)
        if len(d_df) < 48:
            continue

        train_df, val_df, test_df = build_three_way_split(d_df, train_ratio, val_ratio)

        X_train = train_df[feature_cols]
        y_train = train_df["Proportion"].values
        X_val = val_df[feature_cols]
        y_val = val_df["Proportion"].values
        X_test = test_df[feature_cols]
        y_test = test_df["Proportion"].values

        district_rows = []

        lin = LinearRegression()
        lin.fit(X_train, y_train)
        scores = eval_preds(y_train, y_val, y_test, lin.predict(X_train), lin.predict(X_val), lin.predict(X_test))
        district_rows.append({"District": district, "Model": "LINEAR", "params": "{}", **scores})

        # ARIMA / SARIMA / SARIMAX + lightweight baselines
        y_train_s = pd.Series(y_train, index=train_df["Date"]).asfreq("MS")
        y_val_s = pd.Series(y_val, index=val_df["Date"]).asfreq("MS")
        y_test_s = pd.Series(y_test, index=test_df["Date"]).asfreq("MS")
        y_train_s.index = pd.DatetimeIndex(y_train_s.index, freq="MS")
        y_val_s.index = pd.DatetimeIndex(y_val_s.index, freq="MS")
        y_test_s.index = pd.DatetimeIndex(y_test_s.index, freq="MS")

        # Seasonal Naive baseline
        try:
            p_train = forecast_seasonal_naive(y_train_s, len(y_train_s))
            p_val = forecast_seasonal_naive(y_train_s, len(y_val_s))
            p_test = forecast_seasonal_naive(pd.concat([y_train_s, y_val_s]), len(y_test_s))
            scores = eval_preds(y_train_s.values, y_val_s.values, y_test_s.values, p_train, p_val, p_test)
            district_rows.append({"District": district, "Model": "SEASONAL_NAIVE", "params": json.dumps({"season_length": 12}), **scores})
        except Exception:
            pass

        # ETS baseline
        try:
            ets = ExponentialSmoothing(
                y_train_s,
                trend="add",
                seasonal="add",
                seasonal_periods=12,
                initialization_method="estimated",
            ).fit(optimized=True)
            p_train = ets.fittedvalues.values
            p_val = ets.forecast(len(y_val_s)).values
            ets_val_refit = ExponentialSmoothing(
                pd.concat([y_train_s, y_val_s]),
                trend="add",
                seasonal="add",
                seasonal_periods=12,
                initialization_method="estimated",
            ).fit(optimized=True)
            p_test = ets_val_refit.forecast(len(y_test_s)).values
            scores = eval_preds(y_train_s.values, y_val_s.values, y_test_s.values, p_train, p_val, p_test)
            district_rows.append({"District": district, "Model": "ETS", "params": json.dumps({"trend": "add", "seasonal": "add", "seasonal_periods": 12}), **scores})
        except Exception:
            pass

        # Theta method baseline
        try:
            theta = ThetaModel(y_train_s, period=12).fit()
            p_train = y_train_s.values  # in-sample approximation for consistent reporting
            p_val = theta.forecast(len(y_val_s)).values
            theta_val_refit = ThetaModel(pd.concat([y_train_s, y_val_s]), period=12).fit()
            p_test = theta_val_refit.forecast(len(y_test_s)).values
            scores = eval_preds(y_train_s.values, y_val_s.values, y_test_s.values, p_train, p_val, p_test)
            district_rows.append({"District": district, "Model": "THETA", "params": json.dumps({"period": 12}), **scores})
        except Exception:
            pass

        # ARIMA
        try:
            arima = sm.tsa.statespace.SARIMAX(
                y_train_s,
                order=(1, 1, 1),
                seasonal_order=(0, 0, 0, 0),
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            arima_res = fit_sarimax_safely(arima, maxiter=200)
            if arima_res is None:
                raise ValueError("ARIMA did not converge")
            p_train = arima_res.fittedvalues.iloc[1:]
            p_val = arima_res.forecast(steps=len(y_val_s))
            p_test = arima_res.forecast(steps=len(y_val_s) + len(y_test_s)).iloc[len(y_val_s):]
            scores = eval_preds(y_train_s.iloc[1:].values, y_val_s.values, y_test_s.values, p_train.values, p_val.values, p_test.values)
            district_rows.append({"District": district, "Model": "ARIMA", "params": json.dumps({"order": [1, 1, 1]}), **scores})
        except Exception:
            pass

        # SARIMA
        try:
            sarima = sm.tsa.statespace.SARIMAX(
                y_train_s,
                order=(1, 1, 1),
                seasonal_order=(1, 1, 1, 12),
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            sarima_res = fit_sarimax_safely(sarima, maxiter=200)
            if sarima_res is None:
                raise ValueError("SARIMA did not converge")
            p_train = sarima_res.fittedvalues.iloc[13:]
            p_val = sarima_res.forecast(steps=len(y_val_s))
            p_test = sarima_res.forecast(steps=len(y_val_s) + len(y_test_s)).iloc[len(y_val_s):]
            scores = eval_preds(y_train_s.iloc[13:].values, y_val_s.values, y_test_s.values, p_train.values, p_val.values, p_test.values)
            district_rows.append({"District": district, "Model": "SARIMA", "params": json.dumps({"order": [1, 1, 1], "seasonal_order": [1, 1, 1, 12]}), **scores})
        except Exception:
            pass

        # SARIMAX
        exog_cols = ["National_Population", "Population_Density", "Urban_Population"]
        try:
            sarimax = sm.tsa.statespace.SARIMAX(
                y_train_s,
                exog=X_train[exog_cols],
                order=(1, 1, 1),
                seasonal_order=(0, 0, 0, 0),
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            sarimax_res = fit_sarimax_safely(sarimax, maxiter=200)
            if sarimax_res is None:
                raise ValueError("SARIMAX did not converge")
            p_train = sarimax_res.fittedvalues.iloc[1:]
            p_val = sarimax_res.forecast(steps=len(y_val_s), exog=X_val[exog_cols])
            p_test = sarimax_res.forecast(
                steps=len(y_val_s) + len(y_test_s),
                exog=pd.concat([X_val[exog_cols], X_test[exog_cols]], ignore_index=True),
            ).iloc[len(y_val_s):]
            scores = eval_preds(y_train_s.iloc[1:].values, y_val_s.values, y_test_s.values, p_train.values, p_val.values, p_test.values)
            district_rows.append({"District": district, "Model": "SARIMAX", "params": json.dumps({"order": [1, 1, 1]}), **scores})
        except Exception:
            pass

        if district_rows:
            district_df = pd.DataFrame(district_rows)
            district_df["val_train_ratio"] = district_df["val_mape"] / district_df["train_mape"].clip(lower=0.5)
            district_df["selection_score"] = district_df.apply(
                lambda r: compute_selection_score(r["train_mape"], r["val_mape"]), axis=1
            )
            district_df = district_df.sort_values(["selection_score", "val_rmse", "test_rmse"]).reset_index(drop=True)
            all_rows.extend(district_rows)
            best = district_df.iloc[0].to_dict()
            best["rank_by_val"] = 1
            best_rows.append(best)

    cand = pd.DataFrame(all_rows)
    cand["val_train_ratio"] = cand["val_mape"] / cand["train_mape"].clip(lower=0.5)
    cand["selection_score"] = cand.apply(
        lambda r: compute_selection_score(r["train_mape"], r["val_mape"]), axis=1
    )
    cand = cand.sort_values(["selection_score", "val_rmse", "test_rmse"]).reset_index(drop=True)
    cand.to_csv(models_dir / "district_tuning_candidates.csv", index=False)

    leaderboard = pd.DataFrame(best_rows)
    leaderboard["val_train_ratio"] = leaderboard["val_mape"] / leaderboard["train_mape"].clip(lower=0.5)
    leaderboard["selection_score"] = leaderboard.apply(
        lambda r: compute_selection_score(r["train_mape"], r["val_mape"]), axis=1
    )
    leaderboard = leaderboard.sort_values(["selection_score", "val_rmse", "test_rmse"]).reset_index(drop=True)
    leaderboard["overall_rank_by_val"] = np.arange(1, len(leaderboard) + 1)
    leaderboard.to_csv(models_dir / "district_best_model_leaderboard.csv", index=False)
    return cand, leaderboard


def main(args):
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    warnings.filterwarnings("ignore", category=ValueWarning)

    project_root = Path(__file__).resolve().parent.parent
    processed_dir = project_root / "data" / "processed"
    models_dir = project_root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    print("Tuning national models with validation-only model selection...")
    _, nat_board = tune_national(processed_dir, models_dir, args.train_ratio, args.val_ratio)

    print("Tuning district models with validation-only model selection...")
    _, dist_board = tune_district(processed_dir, models_dir, args.train_ratio, args.val_ratio)

    print("\n✅ Tuning complete")
    print("Saved files:")
    print("- models/national_tuning_candidates.csv")
    print("- models/national_best_model_leaderboard.csv")
    print("- models/district_tuning_candidates.csv")
    print("- models/district_best_model_leaderboard.csv")

    if not nat_board.empty:
        top_nat = nat_board.iloc[0]
        print(
            f"Best national model by validation: {top_nat['model']} "
            f"(VAL RMSE={top_nat['val_rmse']:.3f}, TEST RMSE={top_nat['test_rmse']:.3f})"
        )

    if not dist_board.empty:
        print(
            f"District leaderboard rows: {len(dist_board)} "
            f"(best districts ranked by validation RMSE)"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Tune models using validation-only selection and produce national/district leaderboards."
    )
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    args = parser.parse_args()
    main(args)
