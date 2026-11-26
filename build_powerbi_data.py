"""
Build tidy CSV files for the Power BI dashboard.

Run from the project root (the folder that contains data/ and outputs/):
    python build_powerbi_data.py

Reads (all already in the repo):
    data/forecast_backtest_intermediate.csv   (Member 6: actual close, naive forecast)
    outputs/predictions_ml.csv                (Member 4: Ridge / Gradient boosting returns, q10, q90)
    outputs/model_comparison.csv              (Member 4: RMSE, MAE, MAPE, directional accuracy, R2)
    outputs/dm_results_ml.csv                 (Member 4: Diebold-Mariano results)

Writes to data/powerbi/ :
    backtest_wide.csv   one row per day, one column per method (price forecasts)
    backtest_long.csv   same data, one row per day and method (easy legend/slicer in Power BI)
    metrics.csv         one row per method with the error metrics and the DM test
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))          # so that "from tslab..." works
OUT = ROOT / "data" / "powerbi"
OUT.mkdir(parents=True, exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--classical-file", default=str(ROOT / "data" / "predictions_classical.csv"),
                    help="CSV of Member 3's predicted RETURNS (one column per model, indexed by date)")
parser.add_argument("--classical-column", default=None,
                    help="the column to add to the dashboard as the classical model")
args = parser.parse_args()

# ---------------------------------------------------------------- load
bt = pd.read_csv(ROOT / "data" / "forecast_backtest_intermediate.csv", parse_dates=["date"])
pred = pd.read_csv(ROOT / "outputs" / "predictions_ml.csv", index_col=0, parse_dates=True)
pred = pred.dropna()          # keep only the test days (the first 1000 rows have no prediction)

needed = ["Ridge", "Gradient boosting", "Gradient boosting q10", "Gradient boosting q90"]
missing = [c for c in needed if c not in pred.columns]
if missing:
    raise SystemExit(f"predictions_ml.csv is missing the columns {missing}. Found: {list(pred.columns)}")

# ---------------------------------------------------------------- align
# predictions_ml.csv is indexed by day t (the day the forecast is made).
# The backtest file is indexed by the day being forecast (t + 1), and its
# naive_forecast column is the close of day t.
if len(pred) != len(bt):
    raise SystemExit(f"Row counts differ: predictions have {len(pred)} test days, backtest has {len(bt)}.")
if not (bt["date"].values > pred.index.values).all():
    raise SystemExit("Dates do not line up: a backtest date is not after its prediction date.")

close_today = bt["naive_forecast"].to_numpy()          # today's close = the naive forecast


def to_price(log_return):
    """Predicted log return -> predicted closing price."""
    return close_today * np.exp(np.asarray(log_return))


wide = pd.DataFrame({
    "Date": bt["date"],
    "Actual close": bt["actual_close"],
    "Naive (no change)": bt["naive_forecast"],
    "Ridge": to_price(pred["Ridge"]),
    "Gradient boosting": to_price(pred["Gradient boosting"]),
    "Gradient boosting lower (10%)": to_price(pred["Gradient boosting q10"]),
    "Gradient boosting upper (90%)": to_price(pred["Gradient boosting q90"]),
})

# ---------------------------------------------------------------- safety check
# The RMSE rebuilt from these prices must match Member 4's saved table.
metrics = pd.read_csv(ROOT / "outputs" / "model_comparison.csv", index_col=0)
metrics.index.name = "Model"
for model in ["Ridge", "Gradient boosting"]:
    rebuilt = float(np.sqrt(np.mean((wide["Actual close"] - wide[model]) ** 2)))
    saved = float(metrics.loc[model, "RMSE ($)"])
    if abs(rebuilt - saved) > 0.01:
        raise SystemExit(
            f"Alignment check failed for {model}: rebuilt RMSE {rebuilt:.4f} vs saved {saved:.4f}. "
            "Do not use these files."
        )
    print(f"Check passed for {model}: RMSE {rebuilt:.4f} matches the saved {saved:.4f}")

# ---------------------------------------------------------------- metrics + DM
dm = pd.read_csv(ROOT / "outputs" / "dm_results_ml.csv", index_col=0)
dm.index.name = "Model"
table = metrics.join(dm, how="left")

methods = ["Actual close", "Naive (no change)", "Ridge", "Gradient boosting"]

# ---------------------------------------------------------------- optional: classical model
if args.classical_column:
    from tslab.metrics import diebold_mariano, dm_verdict, directional_accuracy, out_of_sample_r2

    classical = pd.read_csv(args.classical_file, index_col=0, parse_dates=True)
    if args.classical_column not in classical.columns:
        raise SystemExit(f"Column not found. Available columns: {list(classical.columns)}")
    name = args.classical_column
    series = classical[name].dropna()
    if len(series) != len(bt):
        raise SystemExit(f"{name} has {len(series)} predictions but the backtest has {len(bt)} days. "
                         "Member 3 and Member 4 must use the same test days.")
    wide[name] = to_price(series.to_numpy())
    methods.append(name)

    actual_return = np.log(wide["Actual close"].to_numpy() / close_today)
    predicted_return = series.to_numpy()
    statistic, p_value = diebold_mariano(actual_return, predicted_return, np.zeros(len(series)))
    table.loc[name, "RMSE ($)"] = float(np.sqrt(np.mean((wide["Actual close"] - wide[name]) ** 2)))
    table.loc[name, "MAE ($)"] = float(np.mean(np.abs(wide["Actual close"] - wide[name])))
    table.loc[name, "MAPE (%)"] = float(np.mean(np.abs((wide["Actual close"] - wide[name]) / wide["Actual close"])) * 100)
    table.loc[name, "Directional accuracy"] = directional_accuracy(actual_return, predicted_return)
    table.loc[name, "Out-of-sample R2 (returns)"] = out_of_sample_r2(actual_return, predicted_return)
    table.loc[name, "DM statistic vs naive"] = statistic
    table.loc[name, "DM p-value"] = p_value
    table.loc[name, "Verdict"] = dm_verdict(statistic, p_value)
    print(f"Added {name}: RMSE {table.loc[name, 'RMSE ($)']:.4f}. Compare this with Notebook 3's table.")

table = table.reset_index()

# ---------------------------------------------------------------- long format
long = wide.melt(id_vars="Date", value_vars=methods, var_name="Series", value_name="Price")

# ---------------------------------------------------------------- save
wide.to_csv(OUT / "backtest_wide.csv", index=False)
long.to_csv(OUT / "backtest_long.csv", index=False)
table.to_csv(OUT / "metrics.csv", index=False)
print(f"Saved {len(wide)} days to {OUT}")
print(table.round(4).to_string(index=False))
