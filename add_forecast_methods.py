"""Add one column per method to data/forecast_intermediate.csv.

Methods in the 63-day forecast file:
  forecast_naive              last close, held flat (the benchmark)
  forecast_drift              last close grown at the average historical daily log return (classical drift)
  forecast_gradient_boosting  Member 6's existing direct 63-day forecast (copied from forecast_close)

The original columns are kept, so the Power BI dashboard keeps working.
Run from the repo folder:  python add_forecast_methods.py
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument("--prices", default=str(ROOT / "data" / "target_MSFT.csv"), help="CSV of MSFT prices")
args = parser.parse_args()

fc_path = ROOT / "data" / "forecast_intermediate.csv"
fc = pd.read_csv(fc_path)
needed = {"date", "horizon_days", "forecast_close", "lower_10", "upper_90", "last_actual_close", "last_actual_date"}
if not needed.issubset(fc.columns):
    raise SystemExit(f"forecast_intermediate.csv is missing columns: {sorted(needed - set(fc.columns))}. Found: {list(fc.columns)}")

# ---- price history
px = pd.read_csv(args.prices)
lower = {c.lower(): c for c in px.columns}
date_col = lower.get("date", px.columns[0])
close_col = next((lower[c] for c in ("close", "adj close", "adj_close") if c in lower), None)
if close_col is None:
    raise SystemExit(f"No close column found in {args.prices}. Columns are: {list(px.columns)}")
px[date_col] = pd.to_datetime(px[date_col])
px = px.sort_values(date_col).set_index(date_col)
last_date = pd.to_datetime(fc["last_actual_date"].iloc[0])
close = px.loc[:last_date, close_col].astype(float)

last_close = float(fc["last_actual_close"].iloc[0])
if abs(float(close.iloc[-1]) - last_close) > 0.01:
    raise SystemExit(f"Price file ends at {close.iloc[-1]:.2f} on {close.index[-1].date()}, but the forecast file says "
                     f"{last_close:.2f} on {last_date.date()}. These must agree.")

mu = float(np.log(close).diff().dropna().mean())
h = fc["horizon_days"].to_numpy()

backup = ROOT / "data" / "forecast_intermediate_gb_only.csv"
if not backup.exists():
    fc.to_csv(backup, index=False)

out = fc.copy()
out["forecast_naive"] = last_close
out["forecast_drift"] = last_close * np.exp(mu * h)
out["forecast_gradient_boosting"] = out["forecast_close"]
out.to_csv(fc_path, index=False)

# long format for Power BI
long = out.melt(id_vars=["date", "horizon_days"],
                value_vars=["forecast_naive", "forecast_drift", "forecast_gradient_boosting"],
                var_name="Method", value_name="Forecast")
long["Method"] = long["Method"].map({"forecast_naive": "Naive (flat)",
                                     "forecast_drift": "Drift",
                                     "forecast_gradient_boosting": "Gradient boosting"})
(ROOT / "data" / "powerbi").mkdir(exist_ok=True)
long.to_csv(ROOT / "data" / "powerbi" / "forecast_long.csv", index=False)

print(f"Average daily log return used for drift: {mu:.6f}")
print(f"Last actual close {last_close:.2f} on {last_date.date()}")
print(out[["date", "horizon_days", "forecast_naive", "forecast_drift", "forecast_gradient_boosting",
           "lower_10", "upper_90"]].iloc[[0, len(out) // 2, -1]].round(2).to_string(index=False))
print(f"Saved {fc_path} (original kept as {backup.name}) and data/powerbi/forecast_long.csv")
