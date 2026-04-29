import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

def generate_report():
    models_dir = Path("models")
    plots_dir = Path("models/evaluation_plots")
    plots_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Data
    try:
        metrics_df = pd.read_csv(models_dir / "metrics.csv")
        preds_df = pd.read_csv(models_dir / "predictions.csv")
    except FileNotFoundError:
        print("Error: Metrics or predictions not found. Run training scripts first.")
        return

    print("Generating Evaluation Report...")

    # 2. Plot Training/Validation/Testing in Same Graph
    model_names = preds_df["Model"].unique()
    fig, axes = plt.subplots(len(model_names), 1, figsize=(15, 6 * len(model_names)))
    if len(model_names) == 1: axes = [axes]

    for idx, model in enumerate(model_names):
        ax = axes[idx]
        df_model = preds_df[preds_df["Model"] == model].sort_values("Year")
        
        # Plot splits
        for split, color in [("TRAIN", "blue"), ("VAL", "orange"), ("TEST", "red")]:
            subset = df_model[df_model["Split"] == split]
            if subset.empty: continue
            
            # Actual as points
            ax.scatter(subset["Year"], subset["Actual"], color=color, label=f"Actual ({split})", alpha=0.5, s=20)
            # Predicted as line
            ax.plot(subset["Year"], subset["Predicted"], color=color, linestyle='--', label=f"Predicted ({split})", linewidth=2)

        ax.set_title(f"National Population: {model} Model Performance", fontsize=16, fontweight='bold')
        ax.set_xlabel("Year")
        ax.set_ylabel("Population")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = plots_dir / "national_performance_splits.png"
    plt.savefig(plot_path)
    print(f"Performance graph saved to: {plot_path}")

    # 3. Overfitting/Underfitting Analysis Table
    print("\nModel Accuracy & Overfitting Analysis:")
    print("-" * 100)
    
    # Pivot metrics for easier comparison
    perf = metrics_df.pivot(index="Model", columns="Split", values="MAPE")
    
    # Calculate Overfitting Ratio (Val MAPE / Train MAPE)
    perf["Overfitting_Ratio"] = perf["VAL"] / perf["TRAIN"]
    
    def diagnose(row):
        if row["TRAIN"] > 5: return "Underfitting (High Bias)"
        if row["Overfitting_Ratio"] > 5: return "Overfitting (High Variance)"
        return "Good Fit"

    perf["Status"] = perf.apply(diagnose, axis=1)
    
    print(perf.to_string())
    print("-" * 100)
    print("NOTE: MAPE values are in percentage (%). Lower is better.")
    print("Good Fit: Low Train error AND Low Validation/Test error.")

    # 4. Save Detailed Metrics
    metrics_styled_path = models_dir / "detailed_evaluation_metrics.csv"
    metrics_df.to_csv(metrics_styled_path, index=False)
    print(f"Detailed metrics saved to: {metrics_styled_path}")

if __name__ == "__main__":
    generate_report()
