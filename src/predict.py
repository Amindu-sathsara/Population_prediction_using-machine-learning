import argparse
import pandas as pd
import pickle
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

# Model imports
from xgboost import XGBRegressor
from prophet import Prophet
from sklearn.linear_model import LinearRegression

def predict(date_str, district=None):
    # Resolve model directory relative to the project root so this works
    # whether called from the project root or from src/.
    base_dir = Path(__file__).resolve().parent
    project_root = base_dir.parent
    model_dir = project_root / "models"
    
    # ====================== 1. NATIONAL POPULATION ======================
    try:
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
        elif (model_dir / "nat_prophet_model.pkl").exists():
            with open(model_dir / "nat_prophet_model.pkl", "rb") as f:
                prophet_model = pickle.load(f)
    except Exception as e:
        print(f"National models error: {e}")
        return

    target_date = pd.to_datetime(date_str)
    year = target_date.year
    month = target_date.month
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

    print(f"\n--- National Population for {date_str} ---")
    valid_preds = []
    X_nat = pd.DataFrame([base_features])

    pred_linear = linear_model.predict(X_nat)[0]
    print(f" Linear   : {int(pred_linear):,}")
    valid_preds.append(pred_linear)

    pred_xgb = xgb_model.predict(X_nat)[0]
    print(f" XGBoost  : {int(pred_xgb):,}")
    valid_preds.append(pred_xgb)

    steps = (year - sarima_info['last_date'].year) * 12 + month - sarima_info['last_date'].month
    if steps > 0:
        pred_sarima = sarima_model.forecast(steps=steps)[-1]
    else:
        pred_sarima = pred_linear
    print(f" SARIMA   : {int(pred_sarima):,}")
    valid_preds.append(pred_sarima)

    pred_prophet = None
    if prophet_model:
        try:
            df_p = pd.DataFrame({'ds': [target_date]})
            pred_prophet_df = prophet_model.predict(df_p)
            pred_prophet = pred_prophet_df['yhat'].iloc[0]
            print(f" Prophet  : {int(pred_prophet):,}")
            valid_preds.append(pred_prophet)
        except:
            pass

    final_nat_pop = sum(valid_preds) / len(valid_preds)
    print(f" National Ensemble : {int(final_nat_pop):,}")

    # ====================== 2. DISTRICT - BEST MODEL ======================
    if not district:
        return

    try:
        with open(model_dir / "district_best_models.pkl", "rb") as f:
            best_models = pickle.load(f)
    except FileNotFoundError:
        print("\nBest district models not found. Run train_district_model.py first.")
        return

    if district not in best_models or best_models[district] is None:
        print(f"\nNo best model available for {district}.")
        return

    print(f"\n--- District {district} Projection ---")
    
    best_model = best_models[district]
    
    future_features = {
        'Fractional_Year': fractional_year,
        'National_Population': final_nat_pop,
        'Population_Density': base_features.get('Population_Density'),
        'Urban_Population': base_features.get('Urban_Population')
    }
    X_future = pd.DataFrame([future_features])

    try:
        if isinstance(best_model, (LinearRegression, XGBRegressor)):
            model_type = "XGBoost" if isinstance(best_model, XGBRegressor) else "Linear"
            pred_prop = float(best_model.predict(X_future)[0])

        elif isinstance(best_model, Prophet):
            model_type = "Prophet"
            prophet_df = pd.DataFrame({
                'ds': [target_date],
                'National_Population': [final_nat_pop],
                'Population_Density': [future_features.get('Population_Density')],
                'Urban_Population': [future_features.get('Urban_Population')]
            })
            forecast = best_model.predict(prophet_df)
            pred_prop = float(forecast['yhat'].iloc[0])

        else:  # SARIMA
            model_type = "SARIMA"
            forecast_result = best_model.forecast(steps=1)
            pred_prop = float(forecast_result.iloc[0] if hasattr(forecast_result, 'iloc') else forecast_result)

        final_dist_pop = pred_prop * final_nat_pop

        print(f" Best Model Used       : {model_type}")
        print(f" Forecasted Proportion : {pred_prop:.4%}")
        print(f" District Population   : {int(final_dist_pop):,}")
        
    except Exception as e:
        print(f"Error during district prediction for {district}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict Population for Sri Lanka.")
    parser.add_argument("--date", type=str, required=True, help="Target Date (YYYY-MM-DD)")
    parser.add_argument("--district", type=str, default=None, help="District name e.g. Ampara")
    args = parser.parse_args()
    predict(args.date, args.district) 