"""Shared feature engineering.

Leakage-free by construction: every feature on row ``t`` is built only from
information available up to ``t-1``. Same-day-derived quantities (percent
change, High/Low ratios and ranges, rolling means/stds) are ``.shift(1)``-ed,
so predicting ``Close[t]`` never sees ``Close[t]`` / ``High[t]`` / ``Low[t]``
/ ``Volume[t]``.
"""

import numpy as np


def create_features(data, include_close_std=True):
    """Create leakage-free features for machine learning"""
    df = data.copy()

    # Price-based features — shifted so row t only sees data up to t-1.
    df['Price_Change'] = df['Close'].pct_change().shift(1)
    df['High_Low_Ratio'] = (df['High'] / df['Low']).shift(1)
    df['Price_Range'] = (df['High'] - df['Low']).shift(1)

    # Lagged features (already strictly historical)
    for lag in [1, 2, 3, 5]:
        df[f'Close_Lag_{lag}'] = df['Close'].shift(lag)
        df[f'Volume_Lag_{lag}'] = df['Volume'].shift(lag)

    # Rolling statistics over windows ending at t-1
    for window in [5, 10, 20]:
        df[f'Close_MA_{window}'] = df['Close'].rolling(window=window).mean().shift(1)
        if include_close_std:
            df[f'Close_Std_{window}'] = df['Close'].rolling(window=window).std().shift(1)
        df[f'Volume_MA_{window}'] = df['Volume'].rolling(window=window).mean().shift(1)

    # Degenerate inputs (e.g. zero Low) yield inf — treat as missing.
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Remove warmup rows
    df = df.dropna()

    return df
