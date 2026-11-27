# Written Verdict: Forecasting Microsoft (MSFT) Closing Prices

**Client question:** Does our forecast beat "tomorrow will be the same as today" on days it never saw, and how much should we trust the 63-day forecast?

## Verdict

**No. None of our models beat the naive forecast, and the neural network is significantly worse.** The naive forecast (tomorrow equals today) has the lowest error on 1,781 test days: an RMSE of $5.79, against $5.82 for the best model (ARMAX with seven predictors), $5.83 for Gradient Boosting, $5.87 for the GRU neural network (average of three seeds) and $5.88 for Ridge regression.

**How we know.** We tested every model with walk-forward validation: each model was trained only on days before the block it was tested on, scalers were fitted on the training block only, and settings were tuned on the first 1,000 days before the test period. A Diebold-Mariano test asks whether a gap in error is bigger than luck. For ARMAX, Gradient Boosting and Ridge it is not (p = 0.41, 0.46 and 0.20), so we cannot say they are better *or* worse than naive. For the GRU it is: the network is worse than naive with all three seeds (p between 0.0001 and 0.004) and on average (p = 0.032). The seeds differ very little from each other, so this is not bad luck with the random start. The hit rates of Ridge, Gradient Boosting and the GRU (49% to 52%) are indistinguishable from a coin toss; ARMAX's 52.8% is the highest, but it did not lead to a lower error.

**Does it make money?** Not reliably. In a long/short back-test, ARMAX beat buy and hold only with no trading costs (Sharpe ratio 1.04 against 0.69). At a cost of 5 basis points per trade its Sharpe ratio fell to 0.65, below buy and hold, and at 10 basis points to 0.27. Gradient Boosting and Ridge never beat buy and hold. This is not a trading recommendation.

**The 63-business-day forecast.** The last close in our data is about $545 (9 November 2026). The forecast for the end of the 63 days is about **$566, with a 10% to 90% range of $471 to $625**. Treat it as a rough guide, not a prediction to act on. At this horizon the model was worse than naive in testing, and its range captured only 55% of real outcomes instead of the intended 80%, so the true risk is larger than the range shows. The central estimate that is safest to use is the naive one, about $545.

**Our honest finding:** with 47 predictors, careful tuning and honest testing, we found no reliable signal in MSFT's daily price movements. That is a valid result and a common one. A forecast that looks good on a chart but was not tested this way should not be trusted either.

*This is a classroom analysis, not investment advice.*

---

# Supporting results

## What we did

- **Stock and data.** MSFT daily data from 2 January 2015 to 9 November 2026 (2,981 trading days), with seven market predictors: the S&P 500, a technology fund, the VIX, the 10-year US yield, the US dollar, Brent crude and gold. Data download date: `[CONFIRM: Member 1 to state the exact date]`.
- **Method.** We forecast the next-day *return* and converted it to a price. Expanding-window walk-forward validation: 29 folds, **1,781 test days** (8 October 2019 to 6 November 2026), fixed random seed 42. The look-ahead test (`check_no_lookahead`) passed on all 47 machine-learning features and caught a deliberately leaky feature, which shows the test works. `[CONFIRM: Member 3 ran the same test on the 7 ARMAX predictors]`.
- **Models.** Naive; classical models (drift, moving average, exponential smoothing, ARIMA, ARMA, ARMAX, SARIMA); Ridge regression and Gradient Boosting with 47 predictors; a GRU neural network.

## One day ahead: all models on the same test days

| Model | RMSE ($) | MAE ($) | Directional accuracy | Diebold-Mariano p-value vs naive | Verdict |
|---|---|---|---|---|---|
| Naive (no change) | 5.79 | 3.99 | n/a | n/a | n/a |
| ARMAX(1,0), 7 predictors | 5.82 | 4.03 | 52.8% | 0.41 | No significant difference |
| Gradient Boosting | 5.83 | 4.03 | 52.0% | 0.46 | No significant difference |
| GRU, 3-seed average | 5.87 | 4.11 | 49.1% | 0.032 | Significantly worse |
| Ridge | 5.88 | 4.07 | 50.7% | 0.20 | No significant difference |
| GRU, single seeds (42, 43, 44) | 5.94 / 5.98 / 5.98 | 4.16 / 4.22 / 4.20 | 50.6% / 49.4% / 49.5% | 0.0001 / 0.0001 / 0.004 | Significantly worse |

`[CONFIRM: test ARMAX's hit rate against 50% the same way as the other models; it is the only one that may stand out.]`

Out-of-sample R² (on returns): ARMAX 0.011, Gradient Boosting -0.007, Ridge -0.017, GRU average -0.028. Only ARMAX is above zero, by a margin that is far inside the noise. `[CONFIRM: add the other classical models' rows from Member 3's table]`

## The neural network

We trained a GRU on windows of the features, with the protocol of Notebook 6: scalers fitted on the training block only, a validation share taken from the end of the training block, early stopping and fixed seeds. We tuned three settings (window length 10, 20 or 40 days; 16, 32 or 64 units; dropout 0 or 0.2) on validation data only. The best setting was a 20-day window, 64 units and no dropout, with a validation loss of 0.752 against 0.801 for the zero-forecast baseline, so on validation the network learned something. On the test days it did not hold up: it was significantly worse than naive. The validation gain was not a real signal. This is an overfitting lesson, and it is why only the test result counts.

**Seed spread.** Across seeds 42, 43 and 44 the RMSE on returns is 0.0194 with a spread (largest minus smallest) of 0.00009, while the gap to naive (0.0189) is about 0.0005, six times larger. The spread is small, so the result is stable: the network is consistently a little worse than naive, not randomly so.

## Back-test with transaction costs

The rule goes long when the predicted return is above zero and short otherwise, and pays 0, 5 or 10 basis points per trade.

| Strategy | Share of days long | Annual return (0 / 5 / 10 bp) | Sharpe ratio (0 / 5 / 10 bp) |
|---|---|---|---|
| ARMAX, 7 predictors | 65.7% | 0.311 / 0.196 / 0.080 | 1.04 / 0.65 / 0.27 |
| Gradient Boosting | 85.5% | 0.185 / 0.141 / 0.098 | 0.62 / 0.47 / 0.33 |
| Ridge | 60.8% | 0.109 / 0.013 / -0.082 | 0.36 / 0.05 / -0.27 |
| Buy and hold | 100% | 0.206 | 0.69 |

This is not a trading recommendation. The ARMAX edge is statistically weak (p = 0.41, R² = 0.011), and it is long on two thirds of the days, so part of its return is exposure to a stock that rose. The test trades at the closing price used for the forecast, charges only a flat cost with no spread or slippage, and covers one stock in one period.

## Which predictors helped?

In the feature-group ablation, the market-wide group and the volatility group were the only ones that helped slightly; removing the technical indicators or the past returns made Ridge marginally better. All differences are below about 0.01 in R², so they are hints, not findings. `[CONFIRM: align with Member 5's own interpretation]`

## How uncertain is it? Longer horizons

A 10% to 90% range from quantile regression should contain about 80% of outcomes. For one day it contained 75.9% in our test, so it is slightly too narrow `[CONFIRM against Member 5's coverage summary]`.

| Horizon | Model RMSE ($) | Naive RMSE ($) | Out-of-sample R² | Outcomes inside the range | Average range width |
|---|---|---|---|---|---|
| 1 day | 5.89 | 5.79 | -0.025 | 77.5% | 3.9% of price |
| 5 days | 12.83 | 12.64 | -0.028 | 72.3% | 7.5% |
| 21 days | 26.41 | 25.50 | -0.036 | 63.5% | 13.1% |
| 63 days | 50.89 | 45.53 | -0.110 | 55.0% | 20.3% |

The model never beats naive at any horizon, and its disadvantage grows with the horizon. The range widens five-fold from 1 to 63 days, yet the share of outcomes it captures falls from 77.5% to 55%. The higher directional accuracy at 63 days (63.4%) is not skill: MSFT rose in 52.9%, 57.9%, 59.9% and 66.3% of the 1-, 5-, 21- and 63-day periods in the test window, so a rule that always says "up" would score higher than the model at every horizon.

**Two Gradient Boosting models.** The one-day table uses the model tuned in Notebook 4 (maximum depth 2, learning rate 0.01, 200 trees). The multi-horizon table uses a separate model with fixed settings (depth 2, learning rate 0.03, 150 trees), trained once per horizon, so their one-day results differ slightly (p = 0.46 for the tuned one, 0.015 for the fixed one). Neither beats naive.

## The 63-business-day forecast in numbers

| Method | Forecast on day 63 | Notes |
|---|---|---|
| Naive | $545.18 | Flat at the last close |
| Drift | $576.25 | Last close grown by the average daily log return (0.088% per day) |
| Gradient Boosting | $565.61 | 10% to 90% range: $470.66 to $624.82, which is 28.3% of the last close |

The range is wider than the 20.3% average seen in the back-test, so the model currently sees more uncertainty than usual. All forecasts are in `data/forecast_intermediate.csv`, and every method's one-day forecasts on the test days are in `data/forecast_advanced.csv`.
