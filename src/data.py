"""Shared data-fetching helpers.

``fetch_stock_data`` passes the user-selected ``start``/``end`` straight to
yfinance (no period-mapping, no silent 1-year substitution): an empty result
is reported as an error, not quietly replaced. Because no manual date
filtering is done anymore, the old tz-naive vs tz-aware comparison fragility
is gone too.
"""

import logging
import re
import time
from collections.abc import Callable
from datetime import date, datetime

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

# Retry policy: initial attempt + 2 retries with short stdlib-only backoff.
_RETRY_DELAYS = (1.0, 2.0)

# Yahoo tickers: letters, digits, '.', '-', '=' (e.g. EURUSD=X, GC=F) and
# '^' (indices, e.g. ^GSPC), max 12 chars. Anything else is rejected before
# any network call.
_TICKER_RE = re.compile(r"^[A-Z0-9.\-=^]{1,12}$")


def sanitize_ticker(raw: str | None) -> str | None:
    """Strip/uppercase ``raw``; return it if it's a plausible ticker, else None."""
    clean = (raw or "").strip().upper()
    if not clean or not _TICKER_RE.match(clean):
        return None
    return clean


def _default_warn(msg):
    print(msg)


def _default_err(msg):
    print(msg)


def _is_rate_limit_error(exc: BaseException) -> bool:
    """Detect rate limiting by message (yfinance 0.2.51 has no dedicated class)."""
    text = f"{type(exc).__name__}: {exc}".lower()
    return any(
        s in text for s in ("rate limit", "rate-limit", "ratelimit", "429", "too many requests")
    )


def _request_history(
    description: str, make_call: Callable[[], pd.DataFrame]
) -> tuple[pd.DataFrame | None, BaseException | None]:
    """Call ``make_call()`` (returns a DataFrame) with up-to-2 retries.

    Returns ``(frame_or_None, last_exc_or_None)``. Both exceptions and empty
    frames are retried: Yahoo flakes on both. Callers classify the outcome.
    """
    attempts = 1 + len(_RETRY_DELAYS)
    frame, last_exc = None, None
    for attempt in range(attempts):
        try:
            frame, last_exc = make_call(), None
        except Exception as exc:  # yfinance/network surface; classified by caller
            frame, last_exc = None, exc
            logger.warning("%s attempt %d/%d failed: %r", description, attempt + 1, attempts, exc)
        if frame is not None and not frame.empty:
            return frame, None
        if attempt + 1 < attempts:
            time.sleep(_RETRY_DELAYS[attempt])
    return frame, last_exc


def flatten_columns(data: pd.DataFrame) -> pd.DataFrame:
    """Flatten yfinance MultiIndex columns (``('Close', 'AAPL')`` -> ``'Close'``)."""
    df = data.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


def _as_timestamp(value: date | datetime | str | pd.Timestamp, end: bool = False) -> datetime:
    if isinstance(value, datetime):
        return value
    if isinstance(value, date):
        return datetime.combine(value, datetime.max.time() if end else datetime.min.time())
    try:
        stamp = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Unparseable date: {value!r}") from exc
    if not isinstance(stamp, datetime):  # NaT is not a datetime
        raise ValueError(f"Unparseable date: {value!r}")
    return stamp.to_pydatetime()


def fetch_stock_data(
    ticker: str | None,
    start: date | datetime | str | pd.Timestamp,
    end: date | datetime | str | pd.Timestamp,
    on_error: Callable[[str], None] | None = None,
) -> pd.DataFrame | None:
    """Fetch OHLCV history for ``[start, end]``. Returns ``None`` on failure."""
    on_error = on_error or _default_err

    clean = sanitize_ticker(ticker)
    if clean is None:
        on_error(
            "⚠️ Please enter a valid stock ticker (letters, digits, `.`, `-`, "
            "`=` or `^`, max 12 characters)."
        )
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

    data, last_exc = _request_history(
        f"fetch {clean}", lambda: flatten_columns(yf.Ticker(clean).history(start=start, end=end))
    )

    if data is not None and not data.empty:
        return data

    if last_exc is not None:
        logger.exception("fetch for %s failed after retries", clean)
        if _is_rate_limit_error(last_exc):
            on_error(
                f"⏳ Yahoo Finance rate-limited the request for '{clean}'. Please wait "
                "a minute and try again (fewer tickers / wider-spaced runs also help)."
            )
        else:
            on_error(f"❌ Error fetching data for '{clean}'. Please check:")
            on_error("1. Your internet connection is working")
            on_error("2. Yahoo Finance API is accessible (it may be temporarily down)")
            on_error("3. If EVERY ticker fails, try upgrading yfinance: pip install -U yfinance")
        return None

    logger.warning("no data for %s in range after retries", clean)
    on_error(f"❌ No data found for ticker '{clean}' in the selected range. Please check:")
    on_error("1. The ticker symbol is correct (e.g., AAPL, GOOGL, MSFT)")
    on_error("2. The date range contains trading days")
    on_error("3. Yahoo Finance API is accessible")
    return None


def fetch_history(ticker: str | None, period: str = "2y") -> pd.DataFrame:
    """Fetch OHLCV history by period (CLI variant). Returns a DataFrame.

    Retries like :func:`fetch_stock_data`. Re-raises the last exception so
    the CLI wrapper can print the network-style message; an empty frame is
    returned as-is so it prints the no-data message instead.
    """
    clean = sanitize_ticker(ticker)
    stock = yf.Ticker(clean or "")
    frame, last_exc = _request_history(
        f"fetch {clean}", lambda: flatten_columns(stock.history(period=period))
    )
    if last_exc is not None:
        raise last_exc
    return frame if frame is not None else pd.DataFrame()
