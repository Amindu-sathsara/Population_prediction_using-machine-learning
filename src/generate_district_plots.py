import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def generate_master_plot():
    models_dir = Path("models")
    plots_dir = Path("models/evaluation_plots")
    plots_dir.mkdir(parents=True, exist_ok=True)

    try:
        df = pd.read_csv(models_dir / "district_predictions.csv")
    except FileNotFoundError:
        print("Error: district_predictions.csv not found. Run src/models/train_district_model.py first.")
        return

    districts = sorted(df['District'].unique())
    n = len(districts)
    ncols = 5
    nrows = (n + ncols - 1) // ncols
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(25, nrows * 4))
    fig.suptitle('District Proportions: Train / Test Confirmation (All 25 Districts)', fontsize=20, fontweight='bold')
    
    for idx, dist in enumerate(districts):
        ax = axes.flat[idx]
        d = df[df['District'] == dist].sort_values('Year')
        
        for split, color in [('TRAIN', 'blue'), ('TEST', 'red')]:
            s = d[d['Split'] == split]
            if s.empty: continue
            ax.scatter(s['Year'], s['Actual_Proportion'], color=color, s=15, alpha=0.5, label=split if idx == 0 else "")
            ax.plot(s['Year'], s['Predicted_Proportion'], color=color, linestyle='--', linewidth=1.5)
        
        ax.set_title(dist, fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.2)
        ax.tick_params(labelsize=8)

    # Add a legend to the first plot
    axes.flat[0].legend(loc='upper left', fontsize=10)

    for i in range(n, len(axes.flat)): axes.flat[i].axis('off')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    output_path = plots_dir / "districts_master_confirmation.png"
    plt.savefig(output_path, dpi=150)
    print(f"✅ Master district plot saved to: {output_path}")

if __name__ == "__main__":
    generate_master_plot()
