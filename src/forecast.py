"""Shared future forecasting: recursive on log returns, rebuilt by feature NAME.

Each step predicts a return from the simulated history, converts it to the
next close (``last_sim * exp(r_hat)``), appends it, and rebuilds every
Close-derived feature (lagged returns, rolling return stats, Close/MA ratios,
RSI) from the extended simulated series.

Frozen by design (stated, not hidden): Volume-derived features (Vol_Ratio)
and the high-low feature (HL_Pct) stay at their last OBSERVED values because
this model does not predict Volume/High/Low. Prediction intervals are added
in 5c on top of the pooled out-of-sample residuals.
"""

import numpy as np
import pandas as pd

from .indicators import rsi_last

LAG_FEATURES = [1, 2, 3, 5]
ROLL_WINDOWS = [5, 10, 20]


def _feature_row(closes, last_vol_ratio, last_hl_pct, feature_cols):
    """Build one unscaled feature row (ordered as ``feature_cols``).

    ``closes`` is history + simulated closes so far; volumes and High/Low
    stay frozen (see module docstring).
    """
    row = {}
    logret = np.log(np.asarray(closes, dtype=float))
    # rets[j] is the log return AT raw index j+1, so rets[-k] is logret[m-k]:
    # exactly the Ret_Lag_k value row m holds in training.
    rets = np.diff(logret)  # len(closes) - 1 return observations
    last = closes[-1]

    for lag in LAG_FEATURES:
        row[f'Ret_Lag_{lag}'] = rets[-lag]

    for window in ROLL_WINDOWS:
        win = rets[-window:]
        row[f'Ret_MA_{window}'] = float(np.mean(win))
        if f'Ret_Std_{window}' in feature_cols:
            row[f'Ret_Std_{window}'] = float(np.std(win, ddof=1))

    ma20 = float(np.mean(np.asarray(closes[-20:], dtype=float)))
    ma50 = float(np.mean(np.asarray(closes[-50:], dtype=float)))
    row['Close_MA20_Ratio'] = last / ma20 - 1 if ma20 != 0 else 0.0
    row['Close_MA50_Ratio'] = last / ma50 - 1 if ma50 != 0 else 0.0
    row['RSI_14'] = rsi_last(closes)
    row['Vol_Ratio'] = last_vol_ratio
    row['HL_Pct'] = last_hl_pct

    missing = [c for c in feature_cols if c not in row]
    if missing:
        raise ValueError(
            f"Cannot recursively update unknown feature(s): {missing}. "
            "Extend _feature_row or switch to direct multi-step forecasting."
        )
    ordered = [row[c] for c in feature_cols]
    if any(not np.isfinite(v) for v in ordered):
        raise ValueError("Non-finite feature value during recursive forecasting.")
    return ordered


def _last_observed_ratios(data):
    """Frozen volume / high-low inputs from the last observed row."""
    last = data.iloc[-1]
    vol_ma20 = float(data['Volume'].rolling(window=20).mean().iloc[-1])
    vol = float(last['Volume'])
    vol_ratio = vol / vol_ma20 if vol_ma20 != 0 else 1.0
    close = float(last['Close'])
    hl_pct = float((last['High'] - last['Low']) / close) if close != 0 else 0.0
    return vol_ratio, hl_pct


def predict_future_prices(model, data, feature_cols, days):
    """Recursive return forecast.

    ``model`` is a fitted ``Pipeline(StandardScaler, regressor)`` predicting
    log returns. Returns ``{'close': [...], 'return': [...]}`` — lists of
    ``days`` simulated closes and the predicted returns behind them.
    """
    closes = [float(x) for x in data['Close'].tolist()]
    last_vol_ratio, last_hl_pct = _last_observed_ratios(data)

    pred_closes, pred_rets = [], []
    for _ in range(days):
        row = _feature_row(closes + pred_closes, last_vol_ratio, last_hl_pct, feature_cols)
        r_hat = float(model.predict(pd.DataFrame([row], columns=feature_cols))[0])
        if not np.isfinite(r_hat):
            raise ValueError("Model predicted a non-finite return during forecasting.")
        pred_rets.append(r_hat)
        pred_closes.append((closes + pred_closes)[-1] * float(np.exp(r_hat)))

    return {'close': pred_closes, 'return': pred_rets}
