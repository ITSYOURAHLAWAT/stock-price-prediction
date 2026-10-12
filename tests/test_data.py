"""Fetch behavior with mocked yfinance (no network)."""

import time
from datetime import date

import pandas as pd
import pytest

from src.data import fetch_history, fetch_stock_data
from src.models import train_models


def _no_sleep(monkeypatch):
    """Record backoff sleeps without waiting. Returns the recorded delays."""
    sleeps = []
    monkeypatch.setattr(time, "sleep", lambda s: sleeps.append(s))
    return sleeps


def test_empty_ticker_rejected(yahoo_mock):
    assert fetch_stock_data("   ", date(2024, 1, 1), date(2024, 3, 1)) is None
    assert yahoo_mock["history_calls"] == 0  # rejected before any network call


def test_start_after_end_rejected(yahoo_mock):
    assert fetch_stock_data("AAPL", date(2024, 3, 1), date(2024, 1, 1)) is None
    assert yahoo_mock["history_calls"] == 0


def test_requested_range_used_directly(yahoo_mock, ohlcv):
    yahoo_mock["frame"] = ohlcv
    got = fetch_stock_data("aapl ", date(2024, 1, 1), date(2024, 3, 1))
    assert got is not None and len(got) == len(ohlcv)
    assert yahoo_mock["tickers"] == ["AAPL"]  # stripped + uppercased
    assert yahoo_mock["kwargs"].get("start") == date(2024, 1, 1)  # no period-mapping
    assert yahoo_mock["kwargs"].get("end") == date(2024, 3, 1)


def test_empty_result_returns_none_without_substitution(yahoo_mock, monkeypatch):
    yahoo_mock["frame"] = pd.DataFrame()
    sleeps = _no_sleep(monkeypatch)
    assert fetch_stock_data("AAPL", date(2024, 1, 1), date(2024, 3, 1)) is None
    assert yahoo_mock["history_calls"] == 3  # initial + 2 retries, still no 1y swap
    assert sleeps == [1.0, 2.0]


def test_exception_returns_none(yahoo_mock, monkeypatch):
    yahoo_mock["error"] = RuntimeError("boom")
    _no_sleep(monkeypatch)
    assert fetch_stock_data("AAPL", date(2024, 1, 1), date(2024, 3, 1)) is None
    assert yahoo_mock["history_calls"] == 3


def test_multiindex_columns_flattened(yahoo_mock, ohlcv):
    mi = ohlcv.copy()
    mi.columns = pd.MultiIndex.from_product([mi.columns, ["AAPL"]])
    yahoo_mock["frame"] = mi
    got = fetch_stock_data("AAPL", date(2024, 1, 1), date(2024, 3, 1))
    assert got is not None
    assert list(got.columns)[:5] == ["Open", "High", "Low", "Close", "Volume"]


def test_tz_aware_index_accepted(yahoo_mock, ohlcv):
    ohlcv.index = ohlcv.index.tz_localize("America/New_York")
    yahoo_mock["frame"] = ohlcv
    got = fetch_stock_data("AAPL", date(2024, 1, 1), date(2024, 3, 1))
    assert got is not None and len(got) == len(ohlcv)


def test_short_range_fails_at_training_not_fetch(yahoo_mock, short_df):
    """< ~60 usable rows: fetch succeeds, train_models raises (guard lives there)."""
    yahoo_mock["frame"] = short_df
    got = fetch_stock_data("AAPL", date(2024, 1, 1), date(2024, 3, 1))
    assert got is not None
    with pytest.raises(ValueError, match="60"):
        train_models(got)


def test_exception_then_success_recovers(yahoo_mock, ohlcv, monkeypatch):
    yahoo_mock["effects"] = [RuntimeError("boom"), ohlcv]
    sleeps = _no_sleep(monkeypatch)
    got = fetch_stock_data("AAPL", date(2024, 1, 1), date(2024, 3, 1))
    assert got is not None and len(got) == len(ohlcv)
    assert yahoo_mock["history_calls"] == 2
    assert sleeps == [1.0]


def test_empty_then_success_recovers(yahoo_mock, ohlcv, monkeypatch):
    yahoo_mock["effects"] = [pd.DataFrame(), ohlcv]
    sleeps = _no_sleep(monkeypatch)
    got = fetch_stock_data("AAPL", date(2024, 1, 1), date(2024, 3, 1))
    assert got is not None and len(got) == len(ohlcv)
    assert yahoo_mock["history_calls"] == 2
    assert sleeps == [1.0]


def test_rate_limit_message_is_distinct(yahoo_mock, monkeypatch):
    yahoo_mock["error"] = RuntimeError("429 Too Many Requests")
    _no_sleep(monkeypatch)
    msgs = []
    assert (
        fetch_stock_data("AAPL", date(2024, 1, 1), date(2024, 3, 1), on_error=msgs.append) is None
    )
    assert any("rate-limit" in m.lower() for m in msgs)


def test_network_message_suggests_upgrade(yahoo_mock, monkeypatch):
    yahoo_mock["error"] = ConnectionError("connection refused")
    _no_sleep(monkeypatch)
    msgs = []
    assert (
        fetch_stock_data("AAPL", date(2024, 1, 1), date(2024, 3, 1), on_error=msgs.append) is None
    )
    assert any("pip install -U yfinance" in m for m in msgs)


def test_cli_history_reraises_after_retries(yahoo_mock, monkeypatch):
    yahoo_mock["error"] = RuntimeError("boom")
    sleeps = _no_sleep(monkeypatch)
    with pytest.raises(RuntimeError, match="boom"):
        fetch_history("AAPL", period="2y")
    assert yahoo_mock["history_calls"] == 3
    assert sleeps == [1.0, 2.0]


def test_cli_fetch_history_flattens(yahoo_mock, ohlcv):
    mi = ohlcv.copy()
    mi.columns = pd.MultiIndex.from_product([mi.columns, ["AAPL"]])
    yahoo_mock["frame"] = mi
    got = fetch_history("aapl", period="2y")
    assert list(got.columns)[:5] == ["Open", "High", "Low", "Close", "Volume"]
