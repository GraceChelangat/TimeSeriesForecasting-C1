"""
Advanced level: build data/forecast_advanced.csv and refresh the Power BI files so
that they contain the ARMAX model (Member 3) and the GRU network (Member 3).

Run from the project root (the folder that contains data/ and outputs/):
    python build_advanced_data.py

Reads (all already in the repo):
    data/forecast_backtest_intermediate.csv   actual close and naive forecast
    outputs/predictions_ml.csv                Ridge / Gradient boosting returns, q10, q90
    outputs/model_comparison.csv              saved metrics of Ridge, Gradient boosting, Naive
    outputs/dm_results_ml.csv                 saved Diebold-Mariano results
    data/predictions_classical.csv            Member 3's classical predictions (returns)
    data/predictions_nn.csv                   Member 3's GRU predictions (returns)
    data/nn_metrics.csv                       Member 3's saved GRU metrics (used as a safety check)
    data/dm_results_nn.csv                    Member 3's saved GRU Diebold-Mariano results (check)

Writes:
    data/forecast_advanced.csv                one row per test day, one column per method and GRU setting
    data/powerbi/backtest_wide.csv            same idea, price forecasts, for the Power BI line chart
    data/powerbi/backtest_long.csv            one row per day and method
    data/powerbi/metrics.csv                  one row per method, with the Diebold-Mariano test
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))          # so that "from tslab..." works
DATA = ROOT / "data"
OUT = DATA / "powerbi"
OUT.mkdir(parents=True, exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--classical-column", default="ARMAX(1,0), 7 predictors")
parser.add_argument("--nn-file", default=str(DATA / "predictions_nn.csv"))
parser.add_argument("--nn-average-column", default="GRU w20 u64 d0 3-seed average",
                    help="the GRU column that is drawn as a line in the dashboard")
args = parser.parse_args()

from tslab.metrics import diebold_mariano, dm_verdict, directional_accuracy, out_of_sample_r2

# ---------------------------------------------------------------- load
bt = pd.read_csv(DATA / "forecast_backtest_intermediate.csv", parse_dates=["date"])
pred = pd.read_csv(ROOT / "outputs" / "predictions_ml.csv", index_col=0, parse_dates=True).dropna()
classical = pd.read_csv(DATA / "predictions_classical.csv", index_col=0, parse_dates=True)
nn = pd.read_csv(args.nn_file, index_col=0, parse_dates=True).dropna()

for col in ["Ridge", "Gradient boosting", "Gradient boosting q10", "Gradient boosting q90"]:
    if col not in pred.columns:
        raise SystemExit(f"predictions_ml.csv is missing the column {col}. Found: {list(pred.columns)}")
if args.classical_column not in classical.columns:
    raise SystemExit(f"Column not found. Available classical columns: {list(classical.columns)}")
if args.nn_average_column not in nn.columns:
    raise SystemExit(f"Column not found. Available GRU columns: {list(nn.columns)}")
armax = classical[args.classical_column].dropna()

# ---------------------------------------------------------------- align
# All predictions are indexed by day t (the day the forecast is made); the backtest
# file is indexed by the day being forecast (t + 1) and its naive_forecast is the close of day t.
if len(pred) != len(bt):
    raise SystemExit(f"Row counts differ: predictions {len(pred)}, backtest {len(bt)}.")
if not (bt["date"].values > pred.index.values).all():
    raise SystemExit("Dates do not line up: a backtest date is not after its prediction date.")
for label, series_index in [("ARMAX", armax.index), ("GRU", nn.index)]:
    if len(series_index) != len(pred) or not (series_index == pred.index).all():
        raise SystemExit(f"{label} was not forecast on exactly the same test days as Ridge and "
                         f"Gradient boosting ({len(series_index)} vs {len(pred)} days). "
                         "All members must use the same test days.")
print(f"All models cover the same {len(pred)} test days "
      f"({pred.index.min().date()} to {pred.index.max().date()}).")

close_today = bt["naive_forecast"].to_numpy()
actual_close = bt["actual_close"].to_numpy()
actual_return = np.log(actual_close / close_today)


def to_price(log_return):
    """Predicted log return -> predicted closing price."""
    return close_today * np.exp(np.asarray(log_return))


# ---------------------------------------------------------------- one table with everything
wide = pd.DataFrame({
    "Date": bt["date"],
    "Actual close": actual_close,
    "Naive (no change)": close_today,
    "Ridge": to_price(pred["Ridge"]),
    "Gradient boosting": to_price(pred["Gradient boosting"]),
    "Gradient boosting lower (10%)": to_price(pred["Gradient boosting q10"]),
    "Gradient boosting upper (90%)": to_price(pred["Gradient boosting q90"]),
    args.classical_column: to_price(armax.to_numpy()),
})
nn_columns = list(nn.columns)
for col in nn_columns:
    wide[col] = to_price(nn[col].to_numpy())


def rmse(col):
    return float(np.sqrt(np.mean((actual_close - wide[col].to_numpy()) ** 2)))


# ---------------------------------------------------------------- safety checks
saved = pd.read_csv(ROOT / "outputs" / "model_comparison.csv", index_col=0)
for model in ["Ridge", "Gradient boosting"]:
    rebuilt, expected = rmse(model), float(saved.loc[model, "RMSE ($)"])
    if abs(rebuilt - expected) > 0.01:
        raise SystemExit(f"Alignment check failed for {model}: rebuilt {rebuilt:.4f} vs saved {expected:.4f}.")
    print(f"Check passed for {model}: RMSE {rebuilt:.4f} matches the saved {expected:.4f}")

nn_saved_file = DATA / "nn_metrics.csv"
if nn_saved_file.exists():
    nn_saved = pd.read_csv(nn_saved_file, index_col=0)
    for col in nn_columns:
        if col not in nn_saved.index:
            print(f"Note: {col} is not in nn_metrics.csv, skipped the check.")
            continue
        rebuilt, expected = rmse(col), float(nn_saved.loc[col, "RMSE ($)"])
        if abs(rebuilt - expected) > 0.01:
            raise SystemExit(f"Alignment check failed for {col}: rebuilt {rebuilt:.4f} vs saved {expected:.4f}. "
                             "Do not use these files.")
        print(f"Check passed for {col}: RMSE {rebuilt:.4f} matches the saved {expected:.4f}")
else:
    print("Note: data/nn_metrics.csv not found, GRU RMSE was not cross-checked.")

# ---------------------------------------------------------------- metrics + Diebold-Mariano
dm_saved = pd.read_csv(ROOT / "outputs" / "dm_results_ml.csv", index_col=0)
table = saved.join(dm_saved, how="left")
table.index.name = "Model"

extra = {args.classical_column: armax.to_numpy()}
extra.update({col: nn[col].to_numpy() for col in nn_columns})
for name, predicted_return in extra.items():
    price = wide[name].to_numpy()
    statistic, p_value = diebold_mariano(actual_return, predicted_return, np.zeros(len(predicted_return)))
    table.loc[name, "RMSE ($)"] = rmse(name)
    table.loc[name, "MAE ($)"] = float(np.mean(np.abs(actual_close - price)))
    table.loc[name, "MAPE (%)"] = float(np.mean(np.abs((actual_close - price) / actual_close)) * 100)
    table.loc[name, "Directional accuracy"] = directional_accuracy(actual_return, predicted_return)
    table.loc[name, "Out-of-sample R2 (returns)"] = out_of_sample_r2(actual_return, predicted_return)
    table.loc[name, "DM statistic vs naive"] = statistic
    table.loc[name, "DM p-value"] = p_value
    table.loc[name, "Verdict"] = dm_verdict(statistic, p_value)

dm_nn_file = DATA / "dm_results_nn.csv"
if dm_nn_file.exists():
    dm_nn = pd.read_csv(dm_nn_file, index_col=0)
    for col in nn_columns:
        if col in dm_nn.index:
            mine, theirs = float(table.loc[col, "DM p-value"]), float(dm_nn.loc[col, "DM p-value"])
            status = "matches" if abs(mine - theirs) < 0.002 else "DIFFERS from"
            print(f"DM p-value for {col}: {mine:.4f} {status} the saved {theirs:.4f}")

table = table.reset_index()

# ---------------------------------------------------------------- save
forecast_advanced = wide.copy()
forecast_advanced.to_csv(DATA / "forecast_advanced.csv", index=False)

line_methods = ["Actual close", "Naive (no change)", "Ridge", "Gradient boosting",
                args.classical_column, args.nn_average_column]
wide[["Date"] + line_methods + ["Gradient boosting lower (10%)", "Gradient boosting upper (90%)"]] \
    .to_csv(OUT / "backtest_wide.csv", index=False)
wide.melt(id_vars="Date", value_vars=line_methods, var_name="Series", value_name="Price") \
    .to_csv(OUT / "backtest_long.csv", index=False)
table.to_csv(OUT / "metrics.csv", index=False)

print(f"\nSaved {len(forecast_advanced)} days x {forecast_advanced.shape[1] - 1} columns to data/forecast_advanced.csv")
print(f"Refreshed the three Power BI files in {OUT}")
print(table[["Model", "RMSE ($)", "MAE ($)", "Directional accuracy", "DM p-value", "Verdict"]]
      .round(4).to_string(index=False))
