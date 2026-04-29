import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def _save_model_split_bar(metrics: pd.DataFrame, metric_col: str, out_path: Path, title: str):
    pivot = metrics.pivot(index="Model", columns="Split", values=metric_col)
    split_order = [c for c in ["TRAIN", "VAL", "TEST"] if c in pivot.columns]
    pivot = pivot[split_order].sort_index()

    ax = pivot.plot(kind="bar", figsize=(12, 6))
    ax.set_title(title)
    ax.set_xlabel("Model")
    ax.set_ylabel(metric_col)
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def _save_national_predictions(preds: pd.DataFrame, out_dir: Path):
    for model in sorted(preds["Model"].unique()):
        m = preds[preds["Model"] == model].copy()
        if m.empty:
            continue

        plt.figure(figsize=(12, 5))
        for split_name, color in [("TRAIN", "#1f77b4"), ("VAL", "#ff7f0e"), ("TEST", "#2ca02c")]:
            part = m[m["Split"] == split_name]
            if part.empty:
                continue
            plt.plot(part["Date"], part["Actual"], color=color, linewidth=2, alpha=0.5, label=f"{split_name} Actual")
            plt.plot(part["Date"], part["Predicted"], color=color, linewidth=1.5, linestyle="--", label=f"{split_name} Pred")

        plt.title(f"National Actual vs Predicted ({model})")
        plt.xlabel("Date")
        plt.ylabel("Population")
        plt.legend(ncol=2)
        plt.tight_layout()
        plt.savefig(out_dir / f"national_actual_vs_pred_{model.lower()}.png", dpi=180)
        plt.close()


def _save_district_summary(d_metrics: pd.DataFrame, out_dir: Path):
    # Average MAPE by model and split for district-level overview
    g = (
        d_metrics.groupby(["Model", "Split"], dropna=False)["MAPE"]
        .mean()
        .reset_index()
    )
    _save_model_split_bar(
        metrics=g,
        metric_col="MAPE",
        out_path=out_dir / "district_avg_mape_train_val_test.png",
        title="District Models: Average MAPE by Split",
    )

    # Overfitting gap (TEST - VAL) by model
    pivot = g.pivot(index="Model", columns="Split", values="MAPE")
    if "TEST" in pivot.columns and "VAL" in pivot.columns:
        gap = (pivot["TEST"] - pivot["VAL"]).sort_values(ascending=False)
        plt.figure(figsize=(11, 5))
        gap.plot(kind="bar", color="#d62728")
        plt.title("District Overfitting Gap: TEST MAPE - VAL MAPE")
        plt.xlabel("Model")
        plt.ylabel("MAPE Gap")
        plt.axhline(0, color="black", linewidth=1)
        plt.grid(axis="y", alpha=0.25)
        plt.tight_layout()
        plt.savefig(out_dir / "district_overfitting_gap.png", dpi=180)
        plt.close()


def main(args):
    project_root = Path(__file__).resolve().parent.parent
    models_dir = project_root / "models"
    out_dir = project_root / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    n_metrics_path = models_dir / "national_train_val_test_metrics.csv"
    n_preds_path = models_dir / "national_predictions_by_split.csv"
    d_metrics_path = models_dir / "district_model_metrics.csv"

    if not n_metrics_path.exists():
        raise FileNotFoundError(f"Missing file: {n_metrics_path}. Run src/models/train_national_model.py first.")
    if not n_preds_path.exists():
        raise FileNotFoundError(f"Missing file: {n_preds_path}. Run src/models/train_national_model.py first.")
    if not d_metrics_path.exists():
        raise FileNotFoundError(f"Missing file: {d_metrics_path}. Run src/models/train_district_model.py first.")

    n_metrics = pd.read_csv(n_metrics_path)
    n_preds = pd.read_csv(n_preds_path)
    d_metrics = pd.read_csv(d_metrics_path)

    n_preds["Date"] = pd.to_datetime(n_preds["Date"])

    _save_model_split_bar(
        metrics=n_metrics,
        metric_col="MAPE",
        out_path=out_dir / "national_mape_train_val_test.png",
        title="National Models: MAPE by Split",
    )
    _save_model_split_bar(
        metrics=n_metrics,
        metric_col="RMSE",
        out_path=out_dir / "national_rmse_train_val_test.png",
        title="National Models: RMSE by Split",
    )

    _save_national_predictions(n_preds, out_dir)
    _save_district_summary(d_metrics, out_dir)

    # Convenience tables
    n_metrics.to_csv(out_dir / "national_metrics_train_val_test.csv", index=False)
    d_metrics.to_csv(out_dir / "district_metrics_train_val_test.csv", index=False)

    print("\n✅ Overfitting report generated")
    print(f"Output folder: {out_dir}")
    print("Generated files:")
    print("- national_mape_train_val_test.png")
    print("- national_rmse_train_val_test.png")
    print("- district_avg_mape_train_val_test.png")
    print("- district_overfitting_gap.png")
    print("- national_actual_vs_pred_<model>.png (one per model)")
    print("- national_metrics_train_val_test.csv")
    print("- district_metrics_train_val_test.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate train/validation/test overfitting report plots and tables.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models/overfitting_report",
        help="Output directory for plots and metric tables",
    )
    args = parser.parse_args()
    main(args)
