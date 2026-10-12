"""Shared technical indicators (display + legacy CLI feature input)."""

import numpy as np


def calculate_technical_indicators(data):
    """Calculate technical indicators"""
    df = data.copy()

    # Moving averages
    df['MA_20'] = df['Close'].rolling(window=20).mean()
    df['MA_50'] = df['Close'].rolling(window=50).mean()

    # RSI (simple rolling-mean variant, as before)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
    # A no-loss streak is genuinely RSI 100; a perfectly flat market is
    # neutral (50), not NaN. Warmup NaNs (first 13 rows) are left untouched.
    df['RSI'] = rsi.mask((gain == 0) & (loss == 0), 50.0)

    # Bollinger Bands
    df['BB_Middle'] = df['Close'].rolling(window=20).mean()
    bb_std = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
    df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)

    # Volume moving average
    df['Volume_MA'] = df['Volume'].rolling(window=20).mean()

    return df
