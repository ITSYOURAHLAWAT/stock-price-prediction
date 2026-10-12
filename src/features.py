"""Shared feature engineering: scale-free signals + next-period log-return target.

Leakage-free by construction: every feature on row ``t`` is built only from
information available up to ``t-1``. The target ``r[t] = log(Close[t]/Close[t-1])``
is the only quantity that may read ``Close[t]``.

Scale-free by construction: no feature carries the price level (no raw lags,
no raw moving averages, no raw volumes). Multiplying every price by a constant
leaves the feature matrix unchanged, so tree models cannot learn the level.
"""

import numpy as np

from .indicators import rsi_series

TARGET_COL = 'Target'

LAG_FEATURES = [1, 2, 3, 5]
ROLL_WINDOWS = [5, 10, 20]

FEATURE_COLS = (
    [f'Ret_Lag_{lag}' for lag in LAG_FEATURES]
    + [f'Ret_MA_{w}' for w in ROLL_WINDOWS]
    + [f'Ret_Std_{w}' for w in ROLL_WINDOWS]
    + ['Close_MA20_Ratio', 'Close_MA50_Ratio', 'RSI_14', 'Vol_Ratio', 'HL_Pct']
)


def reconstruct_close(prev_close, pred_return):
    """Rebuild predicted closes: ``Close[t-1] * exp(r_hat[t])``.

    Takes the ACTUAL previous close (persistence-compatible), so model
    forecasts are directly comparable with the naive baseline.
    """
    prev_close = np.asarray(prev_close, dtype=float)
    pred_return = np.asarray(pred_return, dtype=float)
    return prev_close * np.exp(pred_return)


def create_features(data):
    """Create leakage-free, scale-free features + log-return target."""
    df = data.copy()
    close = df['Close']
    volume = df['Volume']

    # Next-period log return (the target; the only column allowed to read Close[t]).
    logret = np.log(close / close.shift(1))
    df[TARGET_COL] = logret

    # Lagged returns: row t holds r[t-k], all from closes <= t-1.
    for lag in LAG_FEATURES:
        df[f'Ret_Lag_{lag}'] = logret.shift(lag)

    # Rolling mean/std of returns over windows ending at t-1.
    for window in ROLL_WINDOWS:
        df[f'Ret_MA_{window}'] = logret.rolling(window=window).mean().shift(1)
        df[f'Ret_Std_{window}'] = logret.rolling(window=window).std().shift(1)

    # Distance from trend, as a ratio (shifted to t-1).
    df['Close_MA20_Ratio'] = (close / close.rolling(window=20).mean() - 1).shift(1)
    df['Close_MA50_Ratio'] = (close / close.rolling(window=50).mean() - 1).shift(1)

    # Momentum + activity ratios (shifted to t-1).
    df['RSI_14'] = rsi_series(close).shift(1)
    df['Vol_Ratio'] = (volume / volume.rolling(window=20).mean()).shift(1)
    df['HL_Pct'] = ((df['High'] - df['Low']) / close).shift(1)

    # Degenerate inputs (zero prices/volumes) yield inf — treat as missing.
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Remove warmup rows
    df = df.dropna()

    return df
