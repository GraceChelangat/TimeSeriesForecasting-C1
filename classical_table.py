"""
Print the one-day results of every classical model and ARMAX's hit-rate test.

Run from the project root:
    python classical_table.py

Reads data/predictions_classical.csv and data/forecast_backtest_intermediate.csv.
Writes data/classical_table.csv. It changes nothing else.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from tslab.metrics import diebold_mariano, dm_verdict, directional_accuracy, out_of_sample_r2

bt = pd.read_csv(ROOT / "data" / "forecast_backtest_intermediate.csv", parse_dates=["date"])
pred = pd.read_csv(ROOT / "data" / "predictions_classical.csv", index_col=0, parse_dates=True)

close_today = bt["naive_forecast"].to_numpy()
actual_close = bt["actual_close"].to_numpy()
actual_return = np.log(actual_close / close_today)

rows = []
for name in pred.columns:
    p = pred[name].dropna()
    if len(p) != len(bt):
        print(f"Skipped {name}: {len(p)} predictions, expected {len(bt)}")
        continue
    r = p.to_numpy()
    price = close_today * np.exp(r)
    stat, pval = diebold_mariano(actual_return, r, np.zeros(len(r)))
    # hit rate against 50% (days where the model predicts a direction)
    m = np.abs(r) > 1e-8
    hits = int((np.sign(actual_return[m]) == np.sign(r[m])).sum())
    n = int(m.sum())
    hit_p = stats.binomtest(hits, n, 0.5).pvalue
    rows.append({
        "Model": name,
        "RMSE ($)": np.sqrt(np.mean((actual_close - price) ** 2)),
        "MAE ($)": np.mean(np.abs(actual_close - price)),
        "Directional accuracy": directional_accuracy(actual_return, r),
        "Out-of-sample R2": out_of_sample_r2(actual_return, r),
        "DM p-value vs naive": pval,
        "Verdict": dm_verdict(stat, pval),
        "Hit rate p-value (vs 50%)": hit_p,
    })

naive_rmse = np.sqrt(np.mean((actual_close - close_today) ** 2))
table = pd.DataFrame(rows)
table.to_csv(ROOT / "data" / "classical_table.csv", index=False)
print(f"Naive RMSE: {naive_rmse:.4f} on {len(bt)} test days")
print(table.round(4).to_string(index=False))
