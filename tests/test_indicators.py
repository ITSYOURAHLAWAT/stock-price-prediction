"""Indicator values on small known series (no randomness)."""
import warnings

import numpy as np
import pandas as pd

from src.indicators import calculate_technical_indicators


def _frame(closes, volume=1000.0):
    closes = np.asarray(closes, dtype=float)
    n = len(closes)
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    return pd.DataFrame({
        "Open": closes, "High": closes + 1, "Low": closes - 1,
        "Close": closes, "Volume": np.full(n, volume),
    }, index=idx)


def test_moving_averages_on_constant_series():
    ind = calculate_technical_indicators(_frame([50.0] * 60))
    assert ind["MA_20"].iloc[-1] == 50.0
    assert ind["MA_50"].iloc[-1] == 50.0
    assert ind["MA_20"].iloc[:19].isna().all()  # warmup stays NaN


def test_bollinger_bands_on_constant_series():
    ind = calculate_technical_indicators(_frame([50.0] * 60))
    last = ind.iloc[-1]
    assert last["BB_Middle"] == 50.0
    assert last["BB_Upper"] == 50.0  # zero volatility -> bands collapse
    assert last["BB_Lower"] == 50.0


def test_rsi_endless_rally_is_100_without_warnings():
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning fails the test
        ind = calculate_technical_indicators(_frame(np.arange(100.0, 160.0)))
    assert (ind["RSI"].iloc[14:] == 100.0).all()


def test_rsi_flat_market_is_neutral_without_warnings(flat_df):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ind = calculate_technical_indicators(flat_df)
    assert (ind["RSI"].iloc[14:] == 50.0).all()
    assert ind["RSI"].iloc[:13].isna().all()  # warmup untouched
