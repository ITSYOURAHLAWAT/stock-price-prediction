"""Fetch behavior with mocked yfinance (no network)."""
from datetime import date

import pandas as pd
import pytest

from src.data import fetch_history, fetch_stock_data, sanitize_ticker
from src.models import train_models


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


def test_empty_result_returns_none_without_substitution(yahoo_mock):
    yahoo_mock["frame"] = pd.DataFrame()
    assert fetch_stock_data("AAPL", date(2024, 1, 1), date(2024, 3, 1)) is None
    assert yahoo_mock["history_calls"] == 1  # no silent 1y retry


def test_exception_returns_none(yahoo_mock):
    yahoo_mock["error"] = RuntimeError("boom")
    assert fetch_stock_data("AAPL", date(2024, 1, 1), date(2024, 3, 1)) is None


def test_multiindex_columns_flattened(yahoo_mock, ohlcv):
    mi = ohlcv.copy()
    mi.columns = pd.MultiIndex.from_product([mi.columns, ["AAPL"]])
    yahoo_mock["frame"] = mi
    got = fetch_stock_data("AAPL", date(2024, 1, 1), date(2024, 3, 1))
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


def test_cli_fetch_history_flattens(yahoo_mock, ohlcv):
    mi = ohlcv.copy()
    mi.columns = pd.MultiIndex.from_product([mi.columns, ["AAPL"]])
    yahoo_mock["frame"] = mi
    got = fetch_history("aapl", period="2y")
    assert list(got.columns)[:5] == ["Open", "High", "Low", "Close", "Volume"]
