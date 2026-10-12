"""Shared future-price forecasting (recursive, updated by feature NAME).

Each step rebuilds the feature row by name from (history + predictions so
far): Close-derived features (lags, rolling means/stds, Price_Change, and the
legacy MA_20/MA_50/RSI/Bollinger columns) are recomputed from the extended
close series, so no prediction is ever injected into the wrong slot.

Known limitation (stated, not hidden): Volume-derived features and
High/Low-derived features (Volume lags/MAs, High_Low_Ratio, Price_Range) are
frozen at their last known values because this model does not predict
Volume/High/Low. The cleaner long-term fix is direct multi-step models (one
model per horizon) or jointly predicting OHLCV — out of scope for Phase 2.
"""

import numpy as np
import pandas as pd

LAG_FEATURES = [1, 2, 3, 5]
ROLL_WINDOWS = [5, 10, 20]


def _rsi_last(closes):
    """Last RSI-14 value with the same formula as src.indicators."""
    s = pd.Series(np.asarray(closes, dtype=float))
    delta = s.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean().iloc[-1]
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean().iloc[-1]
    if pd.isna(gain) or pd.isna(loss):
        return 50.0
    if gain == 0 and loss == 0:
        return 50.0
    if loss == 0:
        return 100.0
    rs = gain / loss
    return float(100 - (100 / (1 + rs)))


def _feature_row(closes, volumes, last_hl_ratio, last_range, feature_cols):
    """Build one unscaled feature row (ordered as ``feature_cols``)."""
    row = {}
    last, prev = closes[-1], closes[-2]
    row['Price_Change'] = (last - prev) / prev if prev != 0 else 0.0
    row['High_Low_Ratio'] = last_hl_ratio
    row['Price_Range'] = last_range

    for lag in LAG_FEATURES:
        row[f'Close_Lag_{lag}'] = closes[-lag]
        row[f'Volume_Lag_{lag}'] = volumes[-lag]

    for window in ROLL_WINDOWS:
        win = np.asarray(closes[-window:], dtype=float)
        row[f'Close_MA_{window}'] = float(np.mean(win))
        if f'Close_Std_{window}' in feature_cols:
            row[f'Close_Std_{window}'] = float(np.std(win, ddof=1))
        row[f'Volume_MA_{window}'] = float(np.mean(np.asarray(volumes[-window:], dtype=float)))

    # Legacy indicator columns (present in the CLI feature set).
    hist20 = np.asarray(closes[-20:], dtype=float)
    if 'MA_20' in feature_cols:
        row['MA_20'] = float(np.mean(hist20))
    if 'MA_50' in feature_cols:
        row['MA_50'] = float(np.mean(np.asarray(closes[-50:], dtype=float)))
    if 'Volume_MA' in feature_cols:
        row['Volume_MA'] = float(np.mean(np.asarray(volumes[-20:], dtype=float)))
    if 'RSI' in feature_cols:
        row['RSI'] = _rsi_last(closes)
    if 'BB_Middle' in feature_cols or 'BB_Upper' in feature_cols or 'BB_Lower' in feature_cols:
        middle = float(np.mean(hist20))
        std = float(np.std(hist20, ddof=1))
        row['BB_Middle'] = middle
        row['BB_Upper'] = middle + std * 2
        row['BB_Lower'] = middle - std * 2

    missing = [c for c in feature_cols if c not in row]
    if missing:
        raise ValueError(
            f"Cannot recursively update unknown feature(s): {missing}. "
            "Extend _feature_row or switch to direct multi-step forecasting."
        )
    return [row[c] for c in feature_cols]


def predict_future_prices(model, scaler, data, feature_cols, days):
    """Predict future stock prices (recursive, business-day-agnostic values).

    Returns a list of ``days`` predicted closes. Callers generate the
    corresponding business-day dates (``freq='B'``).
    """
    closes = [float(x) for x in data['Close'].tolist()]
    volumes = [float(x) for x in data['Volume'].tolist()]
    last = data.iloc[-1]
    last_low = float(last['Low'])
    last_hl_ratio = float(last['High'] / last['Low']) if last_low != 0 else 1.0
    last_range = float(last['High'] - last['Low'])

    future_predictions = []
    for _ in range(days):
        row = _feature_row(closes + future_predictions, volumes,
                           last_hl_ratio, last_range, feature_cols)
        pred = float(model.predict(scaler.transform([row]))[0])
        future_predictions.append(pred)

    return future_predictions
