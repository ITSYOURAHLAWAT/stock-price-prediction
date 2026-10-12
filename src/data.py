"""Shared data-fetching helpers.

``fetch_stock_data`` passes the user-selected ``start``/``end`` straight to
yfinance (no period-mapping, no silent 1-year substitution): an empty result
is reported as an error, not quietly replaced. Because no manual date
filtering is done anymore, the old tz-naive vs tz-aware comparison fragility
is gone too.
"""

import logging
from datetime import date, datetime

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


def _default_warn(msg):
    print(msg)


def _default_err(msg):
    print(msg)


def flatten_columns(data):
    """Flatten yfinance MultiIndex columns (``('Close', 'AAPL')`` -> ``'Close'``)."""
    df = data.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


def _as_timestamp(value, end=False):
    if isinstance(value, datetime):
        return value
    if isinstance(value, date):
        return datetime.combine(value, datetime.max.time() if end else datetime.min.time())
    return pd.Timestamp(value)


def fetch_stock_data(ticker, start, end, on_error=None):
    """Fetch OHLCV history for ``[start, end]``. Returns ``None`` on failure."""
    on_error = on_error or _default_err

    clean = (ticker or "").strip().upper()
    if not clean:
        on_error("⚠️ Please select or enter a stock ticker to analyze.")
        return None

    try:
        start_ts = _as_timestamp(start)
        end_ts = _as_timestamp(end, end=True)
    except (TypeError, ValueError) as exc:
        on_error(f"❌ Invalid date range: {exc}")
        return None
    if start_ts >= end_ts:
        on_error("⚠️ Start date must be before end date.")
        return None

    try:
        data = yf.Ticker(clean).history(start=start, end=end)
    except Exception:
        logger.exception("yfinance fetch failed for %s", clean)
        on_error(f"❌ Error fetching data for '{clean}'. Please check:")
        on_error("1. The ticker symbol is correct (e.g., AAPL, GOOGL, MSFT)")
        on_error("2. Your internet connection is working")
        on_error("3. Yahoo Finance API is accessible")
        return None

    data = flatten_columns(data)
    if data.empty:
        on_error(f"❌ No data found for ticker '{clean}' in the selected range. Please check:")
        on_error("1. The ticker symbol is correct (e.g., AAPL, GOOGL, MSFT)")
        on_error("2. The date range contains trading days")
        on_error("3. Yahoo Finance API is accessible")
        return None

    return data


def fetch_history(ticker, period="2y"):
    """Fetch OHLCV history by period (CLI variant). Returns a DataFrame.

    Raises whatever ``yfinance`` raises; the CLI wrapper keeps its
    try/except + print behaviour so output is unchanged.
    """
    clean = (ticker or "").strip().upper()
    stock = yf.Ticker(clean)
    return flatten_columns(stock.history(period=period))
