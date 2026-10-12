"""Shared technical indicators (display + feature input)."""

from collections.abc import Sequence

import numpy as np
import pandas as pd


def rsi_series(close: pd.Series, window: int = 14) -> pd.Series:
    """RSI (simple rolling-mean variant) as a Series.

    A no-loss streak is genuinely 100; a perfectly flat market is neutral
    (50), not NaN. Warmup NaNs are left untouched.
    """
    values = np.asarray(close, dtype=float)
    delta = np.empty_like(values)
    delta[0] = np.nan
    delta[1:] = values[1:] - values[:-1]
    # Explicit Series construction keeps every step below Series-typed.
    gain = pd.Series(np.where(delta > 0.0, delta, 0.0), index=close.index)
    loss = pd.Series(np.where(delta < 0.0, -delta, 0.0), index=close.index)
    gain_ma = np.asarray(gain.rolling(window=window).mean(), dtype=float)
    loss_ma = np.asarray(loss.rolling(window=window).mean(), dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        rs = gain_ma / loss_ma
        rsi_values = 100 - (100 / (1 + rs))
    rsi = pd.Series(rsi_values, index=close.index)
    flat = pd.Series((gain_ma == 0) & (loss_ma == 0), index=close.index)
    return rsi.mask(flat, 50.0)


def rsi_last(closes: Sequence[float], window: int = 14) -> float:
    """Last RSI value of a close-price sequence (for recursive forecasting).

    Returns neutral 50.0 when history is shorter than the window.
    """
    values = rsi_series(pd.Series(np.asarray(closes, dtype=float)), window=window)
    last = values.iloc[-1]
    return 50.0 if pd.isna(last) else float(last)


def calculate_technical_indicators(data: pd.DataFrame) -> pd.DataFrame:
    """Calculate technical indicators"""
    df = data.copy()

    # Moving averages
    df["MA_20"] = df["Close"].rolling(window=20).mean()
    df["MA_50"] = df["Close"].rolling(window=50).mean()

    # RSI (simple rolling-mean variant, as before)
    df["RSI"] = rsi_series(pd.Series(np.asarray(df["Close"], dtype=float), index=df.index))

    # Bollinger Bands
    df["BB_Middle"] = df["Close"].rolling(window=20).mean()
    bb_std = df["Close"].rolling(window=20).std()
    df["BB_Upper"] = df["BB_Middle"] + (bb_std * 2)
    df["BB_Lower"] = df["BB_Middle"] - (bb_std * 2)

    # Volume moving average
    df["Volume_MA"] = df["Volume"].rolling(window=20).mean()

    return df
