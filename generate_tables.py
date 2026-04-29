import pandas as pd
import numpy as np

# Table 1: Normal Evaluation (Observed Years Only, 2000-2025 split)
eval_df = pd.read_csv('models/worldometers_national_model_metrics_fair.csv')
eval_df['Accuracy_Percent'] = 100 - eval_df['MAPE']
table1 = eval_df[['Model', 'RMSE', 'MAE', 'MAPE', 'MASE', 'Accuracy_Percent', 'R2']].copy()
table1.columns = ['Model', 'RMSE', 'MAE', 'MAPE (%)', 'MASE', 'Accuracy (%)', 'R²']
table1 = table1.sort_values('Accuracy (%)', ascending=False).reset_index(drop=True)

# Format to reasonable precision
for col in ['RMSE', 'MAE', 'MASE', 'R²']:
    table1[col] = table1[col].apply(lambda x: f"{x:,.0f}" if col in ['RMSE', 'MAE'] else f"{x:.4f}")
for col in ['MAPE (%)', 'Accuracy (%)']:
    table1[col] = table1[col].apply(lambda x: f"{x:.2f}")

print("="*100)
print("TABLE 1: National Model Performance (Evaluation Split - Observed Years 2000-2025)")
print("="*100)
print(table1.to_string(index=False))
print("\nNote: Sorted by Accuracy % (highest to lowest)")

# Table 2: Walk-Forward Backtest (6 folds across years 2015-2025)
walkfwd = pd.read_csv('models/worldometers_walkforward_backtest.csv')
pipeline_names = walkfwd['Pipeline'].unique()
wf_rows = []
for pipe in pipeline_names:
    pipe_data = walkfwd[walkfwd['Pipeline'] == pipe]
    actuals = pipe_data['Actual'].values
    preds = pipe_data['Predicted'].values
    
    valid_mask = ~np.isnan(preds)
    if valid_mask.sum() == 0:
        continue
    
    actuals_v = actuals[valid_mask]
    preds_v = preds[valid_mask]
    
    rmse = float(np.sqrt(np.mean((actuals_v - preds_v)**2)))
    mae = float(np.mean(np.abs(actuals_v - preds_v)))
    mape = float(np.mean(np.abs((actuals_v - preds_v) / actuals_v)) * 100)
    scale = float(np.mean(np.abs(np.diff(actuals_v))))
    mase = float(mae / scale) if scale > 0 else np.nan
    acc = 100 - mape
    
    wf_rows.append({
        'Pipeline': pipe,
        'N Folds': int(valid_mask.sum()),
        'Years': f"{int(pipe_data['Fold_Year'].min())}-{int(pipe_data['Fold_Year'].max())}",
        'RMSE': rmse,
        'MAE': mae,
        'MAPE (%)': mape,
        'MASE': mase,
        'Accuracy (%)': acc,
    })

table2 = pd.DataFrame(wf_rows).sort_values('Accuracy (%)', ascending=False).reset_index(drop=True)

# Format precision
for col in ['RMSE', 'MAE', 'MASE']:
    table2[col] = table2[col].apply(lambda x: f"{x:,.0f}" if col in ['RMSE', 'MAE'] else f"{x:.4f}")
for col in ['MAPE (%)', 'Accuracy (%)']:
    table2[col] = table2[col].apply(lambda x: f"{x:.2f}")

print("\n" + "="*100)
print("TABLE 2: Walk-Forward Backtest Summary (Expanding Window, 6 Folds: 2015-2025)")
print("="*100)
print(table2.to_string(index=False))

# Table 3: Markdown format for easy copy-paste
print("\n" + "="*100)
print("TABLE 1 (MARKDOWN FORMAT - Copy directly into Word/Google Docs/Markdown)")
print("="*100)
md_table1 = eval_df[['Model', 'RMSE', 'MAE', 'MAPE', 'MASE', 'Accuracy_Percent', 'R2']].copy()
md_table1['Accuracy %'] = (100 - md_table1['MAPE']).round(2)
md_table1 = md_table1[['Model', 'RMSE', 'MAE', 'MAPE', 'MASE', 'Accuracy %', 'R2']].sort_values('Accuracy %', ascending=False)

# Create markdown table manually
print("| Model | RMSE | MAE | MAPE | MASE | Accuracy % | R² |")
print("|-------|------|-----|------|------|------------|-----|")
for idx, row in md_table1.iterrows():
    print(f"| {row['Model']} | {row['RMSE']:,.0f} | {row['MAE']:,.0f} | {row['MAPE']:.3f} | {row['MASE']:.4f} | {row['Accuracy %']:.2f} | {row['R2']:.4f} |")

print("\n" + "="*100)
print("TABLE 2 (MARKDOWN FORMAT - Copy directly into Word/Google Docs/Markdown)")
print("="*100)
print("| Pipeline | N Folds | Years | RMSE | MAE | MAPE (%) | MASE | Accuracy (%) |")
print("|----------|---------|-------|------|-----|----------|------|--------------|")
for idx, row in table2.iterrows():
    print(f"| {row['Pipeline']} | {row['N Folds']} | {row['Years']} | {row['RMSE']} | {row['MAE']} | {row['MAPE (%)']} | {row['MASE']} | {row['Accuracy (%)']} |")

print("\n" + "="*100)
print("CSV FORMAT (for Excel/Sheets import)")
print("="*100)
print("\nTable 1 CSV:")
csv1 = eval_df[['Model', 'RMSE', 'MAE', 'MAPE', 'MASE', 'Accuracy_Percent', 'R2']].sort_values('Accuracy_Percent', ascending=False)
csv1.columns = ['Model', 'RMSE', 'MAE', 'MAPE', 'MASE', 'Accuracy %', 'R2']
print(csv1.to_csv(index=False))
