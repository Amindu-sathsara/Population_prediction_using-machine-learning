import os
import pandas as pd

def calculate_crop_consumption():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(base_dir)
    models_dir = os.path.join(project_root, 'models')

    source_path = os.path.join(models_dir, 'future_national_population.csv')
    if not os.path.exists(source_path):
        raise FileNotFoundError(f"Future national population file not found at {source_path}. Run model_training.py first.")

    df = pd.read_csv(source_path)

    # Example: rice consumption per person per year (kg)
    per_capita = 100  

    df['Estimated_Rice_Consumption_kg'] = df['Predicted_Population'] * per_capita

    out_path = os.path.join(models_dir, 'crop_consumption.csv')
    df.to_csv(out_path, index=False)
    print(f"✅ Crop consumption calculated and saved to {out_path}")

if __name__ == "__main__":
    calculate_crop_consumption()