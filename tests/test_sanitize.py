"""sanitize_ticker accept/reject lists (parametrized)."""
import pytest

from src.data import sanitize_ticker


@pytest.mark.parametrize("raw,want", [
    ("BRK-B", "BRK-B"),
    ("BTC-USD", "BTC-USD"),
    ("^GSPC", "^GSPC"),
    ("EURUSD=X", "EURUSD=X"),
    ("GC=F", "GC=F"),
    ("RELIANCE.NS", "RELIANCE.NS"),
    ("aapl", "AAPL"),
    ("  googl  ", "GOOGL"),
])
def test_accepted_tickers(raw, want):
    assert sanitize_ticker(raw) == want


@pytest.mark.parametrize("raw", [
    "", "   ", None, "!!!", "A B", "AAPL;ls", "T/ne", "x" * 13, "0123456789012",
])
def test_rejected_tickers(raw):
    assert sanitize_ticker(raw) is None
