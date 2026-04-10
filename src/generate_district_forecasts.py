import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# Import model types
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from prophet import Prophet

from xgboost import XGBRegressor
from prophet import Prophet

def generate_district_forecasts(start_date: str = "2025-01-01", 
                               end_date: str = "2035-12-01", 
                               national_forecast_csv=None):
    """
    Generate monthly district population forecasts using the BEST model per district.
    """
    model_dir = Path("models")
    processed_dir = Path("data/processed")
    
    print("Loading best district models...")
    try:
        with open(model_dir / "district_best_models.pkl", "rb") as f:
            best_models = pickle.load(f)
    except FileNotFoundError:
        print("Error: district_best_models.pkl not found. Please run train_district_model.py first!")
        return
    
    # Load historical data
    print("Loading historical data for reference...")
    df = pd.read_csv(processed_dir / "district_master_monthly.csv")
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(["District", "Date"]).reset_index(drop=True)
    
    # Create future date range
    future_dates = pd.date_range(start=start_date, end=end_date, freq='MS')
    future_df = pd.DataFrame({'Date': future_dates})
    future_df['Year'] = future_df['Date'].dt.year
    future_df['Month'] = future_df['Date'].dt.month
    future_df['Fractional_Year'] = future_df['Year'] + (future_df['Month'] - 1) / 12.0
    
    # National Population (simple extrapolation for now)
    if national_forecast_csv and Path(national_forecast_csv).exists():
        nat_df = pd.read_csv(national_forecast_csv)
        nat_df['Date'] = pd.to_datetime(nat_df['Date'])
        future_df = future_df.merge(nat_df[['Date', 'National_Population']], on='Date', how='left')
        print("Using provided national forecast CSV.")
    else:
        last_nat = df['National_Population'].iloc[-1]
        last_date = df['Date'].max()
        months_ahead = ((future_df['Date'] - last_date).dt.days / 30.44).values
        future_df['National_Population'] = last_nat * (1 + 0.008 * months_ahead)  # ~0.8% monthly
        print("Using simple national population extrapolation.")
    
    # Other drivers (simple)
    last_density = df['Population_Density'].iloc[-1]
    last_urban_ratio = df['Urban_Population'].iloc[-1] / df['National_Population'].iloc[-1]
    future_df['Population_Density'] = last_density
    future_df['Urban_Population'] = last_urban_ratio * future_df['National_Population']
    
    print(f"Generating forecasts from {start_date} to {end_date} for {len(best_models)} districts...\n")
    
    forecasts = []
    
    for district, model in best_models.items():
        if model is None:
            print(f"  ⚠️ Skipping {district} (no model)")
            continue
            
        print(f"  → Forecasting {district}...")
        dist_future = future_df.copy()
        dist_future['District'] = district
        
        X_future = dist_future[['Fractional_Year', 'National_Population', 
                                'Population_Density', 'Urban_Population']]
        
        try:
            if isinstance(model, (LinearRegression, XGBRegressor)):
                pred_prop = model.predict(X_future)
                
            elif isinstance(model, Prophet):
                prophet_input = pd.DataFrame({
                    'ds': dist_future['Date'],
                    'National_Population': dist_future['National_Population'],
                    'Population_Density': dist_future['Population_Density'],
                    'Urban_Population': dist_future['Urban_Population']
                })
                forecast = model.predict(prophet_input)
                pred_prop = forecast['yhat'].values
                
            else:  # SARIMA
                steps = len(dist_future)
                pred_prop = model.forecast(steps=steps)
                # Clip SARIMA to reasonable range to avoid explosion
                pred_prop = np.clip(pred_prop, 0.01, 0.5)  # proportion between 1% and 50%
            
            # Clean invalid values
            pred_prop = np.nan_to_num(pred_prop, nan=0.05, posinf=0.3, neginf=0.01)
            
            dist_future['Predicted_Proportion'] = pred_prop
            dist_future['Predicted_District_Population'] = (pred_prop * dist_future['National_Population']).round(0)
            
            forecasts.append(dist_future)
            
        except Exception as e:
            print(f"    ⚠️ Error forecasting {district}: {e}")
            continue
    
    if not forecasts:
        print("No forecasts generated.")
        return None
    
    # Combine
    final_forecast = pd.concat(forecasts, ignore_index=True)
    final_forecast = final_forecast[[
        'Date', 'District', 'Year', 'Month', 
        'Predicted_Proportion', 'Predicted_District_Population',
        'National_Population', 'Population_Density', 'Urban_Population'
    ]]
    
    # Final cleaning
    final_forecast['Predicted_Proportion'] = final_forecast['Predicted_Proportion'].round(6)
    final_forecast['Predicted_District_Population'] = final_forecast['Predicted_District_Population'].astype('Int64')  # nullable integer
    
    # Save
    output_path = model_dir / "district_forecasts.csv"
    final_forecast.to_csv(output_path, index=False)
    
    print(f"\n✅ Forecast generation completed successfully!")
    print(f"   → Saved to: models/district_forecasts.csv")
    print(f"   → Total rows: {len(final_forecast):,}")
    
    print("\nPreview (first 5 rows of Colombo if available):")
    colombo_sample = final_forecast[final_forecast['District'] == 'Colombo'].head(5)
    if not colombo_sample.empty:
        print(colombo_sample[['Date', 'Predicted_Proportion', 'Predicted_District_Population']])
    else:
        print(final_forecast.head(5)[['Date', 'District', 'Predicted_District_Population']])
    
    return final_forecast


if __name__ == "__main__":
    generate_district_forecasts(
        start_date="2025-01-01",
        end_date="2035-12-01"
    )