import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def generate_simple_plots():
    models_dir = Path("models")
    plots_dir = Path("models/evaluation_plots")
    plots_dir.mkdir(parents=True, exist_ok=True)

    try:
        preds_df = pd.read_csv(models_dir / "predictions.csv")
    except: 
        print("Predictions file not found.")
        return

    # 1. National Performance Splits (Combined view)
    plt.figure(figsize=(15, 8))
    model_names = preds_df["Model"].unique()
    for idx, model in enumerate(model_names):
        plt.subplot(len(model_names), 1, idx+1)
        subset = preds_df[preds_df["Model"] == model].sort_values("Year")
        for split, color in [("TRAIN", "blue"), ("VAL", "orange"), ("TEST", "red")]:
            s = subset[subset["Split"] == split]
            plt.scatter(s["Year"], s["Actual"], color=color, s=20, alpha=0.5)
            plt.plot(s["Year"], s["Predicted"], color=color, linestyle='--', linewidth=2)
        plt.title(f"Model: {model}")
    plt.tight_layout()
    plt.savefig(plots_dir / "national_performance_splits.png")
    plt.close()

    # 2. Primary Confirmation (Linear Model)
    plt.figure(figsize=(12, 6))
    subset = preds_df[preds_df["Model"] == "LINEAR"].sort_values("Year")
    for split, color in [("TRAIN", "blue"), ("VAL", "orange"), ("TEST", "red")]:
        s = subset[subset["Split"] == split]
        plt.scatter(s["Year"], s["Actual"], color=color, label=f"Actual {split}", s=30)
        plt.plot(s["Year"], s["Predicted"], color=color, linestyle='--', linewidth=2)
    plt.title("National Population Prediction: Robustness Confirmation", fontsize=14, fontweight='bold')
    plt.legend()
    plt.savefig(plots_dir / "national_confirmation.png")
    plt.close()

    print("Reverted to simple confirmation plots in models/evaluation_plots/")

if __name__ == "__main__":
    generate_simple_plots()
