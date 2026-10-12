"""Shared data-fetching helpers (moved verbatim from app.py / stock_price).

Phase 1 note: logic is intentionally unchanged, including the period-mapping
behaviour in ``fetch_stock_data``. Proper start/end handling, tz-awareness
and retry policy are Phase 2 work.
"""

import yfinance as yf
from datetime import date, datetime


def _default_warn(msg):
    print(msg)


def _default_err(msg):
    print(msg)


def _default_exc(exc):
    print(exc)


def fetch_stock_data(ticker, start, end, on_warning=None, on_error=None,
                     on_exception=None):
    """Fetch stock data with error handling (Streamlit-app variant).

    ``on_warning`` / ``on_error`` / ``on_exception`` are UI callbacks so this
    module does not depend on streamlit. ``app.py`` passes
    ``st.warning`` / ``st.error`` / ``st.exception`` to keep behaviour identical.
    """
    on_warning = on_warning or _default_warn
    on_error = on_error or _default_err
    on_exception = on_exception or _default_exc
    try:
        stock = yf.Ticker(ticker)

        # Try using period first (more reliable)
        try:
            # Calculate period from dates
            if isinstance(start, date):
                start_dt = datetime.combine(start, datetime.min.time())
            else:
                start_dt = start

            if isinstance(end, date):
                end_dt = datetime.combine(end, datetime.max.time())
            else:
                end_dt = end

            days_diff = (end_dt - start_dt).days

            # Use period if reasonable, otherwise use start/end
            if days_diff <= 730:  # 2 years
                if days_diff <= 5:
                    period = "5d"
                elif days_diff <= 30:
                    period = "1mo"
                elif days_diff <= 90:
                    period = "3mo"
                elif days_diff <= 180:
                    period = "6mo"
                elif days_diff <= 365:
                    period = "1y"
                else:
                    period = "2y"

                data = stock.history(period=period)
            else:
                # For longer periods, use start/end dates
                data = stock.history(start=start, end=end)
        except:
            # Fallback to start/end dates
            data = stock.history(start=start, end=end)

        if data.empty or len(data) == 0:
            # Try alternative method with period
            on_warning(f"Trying alternative method to fetch data for {ticker}...")
            data = stock.history(period="1y")

            if data.empty or len(data) == 0:
                on_error(f"❌ No data found for ticker '{ticker}'. Please check:")
                on_error("1. The ticker symbol is correct (e.g., AAPL, GOOGL, MSFT)")
                on_error("2. Your internet connection is working")
                on_error("3. Yahoo Finance API is accessible")
                return None

        # Filter by date range if needed
        if not data.empty and isinstance(start, date) and isinstance(end, date):
            # Convert date to datetime for comparison
            start_dt = datetime.combine(start, datetime.min.time())
            end_dt = datetime.combine(end, datetime.max.time())
            data = data[(data.index >= start_dt) & (data.index <= end_dt)]
            if data.empty:
                on_warning("⚠️ No data in selected date range. Showing available data.")
                data = stock.history(period="1y")

        return data
    except Exception as e:
        on_error(f"❌ Error fetching data: {str(e)}")
        on_error("This might be due to:")
        on_error("• Internet connection issues")
        on_error("• Yahoo Finance API temporarily unavailable")
        on_error("• Invalid ticker symbol")
        on_exception(e)  # Show full error for debugging
        return None


def fetch_history(ticker, period="2y"):
    """Fetch OHLCV history by period (CLI variant). Returns a DataFrame.

    Raises whatever ``yfinance`` raises; the CLI wrapper keeps its
    try/except + print behaviour so output is unchanged.
    """
    stock = yf.Ticker(ticker)
    return stock.history(period=period)
