"""Shared technical indicators (display + feature input)."""

import numpy as np
import pandas as pd


def rsi_series(close, window=14):
    """RSI (simple rolling-mean variant) as a Series.

    A no-loss streak is genuinely 100; a perfectly flat market is neutral
    (50), not NaN. Warmup NaNs are left untouched.
    """
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
    return rsi.mask((gain == 0) & (loss == 0), 50.0)


def rsi_last(closes, window=14):
    """Last RSI value of a close-price sequence (for recursive forecasting).

    Returns neutral 50.0 when history is shorter than the window.
    """
    values = rsi_series(pd.Series(np.asarray(closes, dtype=float)), window=window)
    last = values.iloc[-1]
    return 50.0 if pd.isna(last) else float(last)


def calculate_technical_indicators(data):
    """Calculate technical indicators"""
    df = data.copy()

    # Moving averages
    df['MA_20'] = df['Close'].rolling(window=20).mean()
    df['MA_50'] = df['Close'].rolling(window=50).mean()

    # RSI (simple rolling-mean variant, as before)
    df['RSI'] = rsi_series(df['Close'])

    # Bollinger Bands
    df['BB_Middle'] = df['Close'].rolling(window=20).mean()
    bb_std = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
    df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)

    # Volume moving average
    df['Volume_MA'] = df['Volume'].rolling(window=20).mean()

    return df
