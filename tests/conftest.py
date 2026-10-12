"""Shared fixtures: synthetic OHLCV data and a mocked yfinance (no network).

All randomness uses fixed seeds, and every sklearn model in ``src`` uses a
fixed ``random_state``, so the suite is fully deterministic.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent


def make_ohlcv(seed=11, n=150, drift=0.1, vol=1.2, tz=None):
    """Deterministic business-day OHLCV frame."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2023-01-02", periods=n, freq="B", tz=tz)
    close = 100 + np.cumsum(rng.normal(drift, vol, n))
    return pd.DataFrame({
        "Open": close + rng.normal(0, 0.2, n),
        "High": close + np.abs(rng.normal(0.4, 0.3, n)),
        "Low": close - np.abs(rng.normal(0.4, 0.3, n)),
        "Close": close,
        "Volume": rng.integers(1_000_000, 4_000_000, n).astype(float),
    }, index=idx)


@pytest.fixture
def ohlcv():
    return make_ohlcv()


@pytest.fixture
def short_df():
    """Too few rows to survive feature engineering (< 60 usable)."""
    return make_ohlcv(n=30)


@pytest.fixture
def flat_df():
    """Perfectly flat market (exercises RSI / std edge cases)."""
    df = make_ohlcv(n=40)
    df["Close"] = 100.0
    return df


@pytest.fixture
def yahoo_mock(monkeypatch):
    """Patch ``yfinance.Ticker``. Set ``state['frame']`` per test.

    Records constructor tickers and ``history()`` calls/kwargs so tests can
    assert what (and how often) the app requested.
    """
    import yfinance

    state = {"history_calls": 0, "tickers": [], "kwargs": {}, "frame": None, "error": None}

    class FakeTicker:
        def __init__(self, ticker):
            state["tickers"].append(ticker)

        def history(self, **kwargs):
            state["history_calls"] += 1
            state["kwargs"] = kwargs
            if state["error"] is not None:
                raise state["error"]
            return state["frame"].copy()

    monkeypatch.setattr(yfinance, "Ticker", FakeTicker)
    return state


@pytest.fixture
def clean_caches():
    """Isolate Streamlit caches between AppTest tests (caches are global)."""
    import streamlit as st

    st.cache_data.clear()
    st.cache_resource.clear()
    yield
    st.cache_data.clear()
    st.cache_resource.clear()
