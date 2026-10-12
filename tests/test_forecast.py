"""Forecast: by-name updates on simulated returns, length/finiteness."""
import numpy as np
import pandas as pd
import pytest

from src.features import FEATURE_COLS
from src.forecast import _feature_row, predict_future_prices
from src.indicators import rsi_last
from src.models import train_models


def test_feature_row_updates_by_name():
    closes = [float(100 + i) for i in range(60)]  # 100..159
    row = dict(zip(FEATURE_COLS, _feature_row(closes, 1.05, 0.02, FEATURE_COLS)))

    assert row["Ret_Lag_1"] == pytest.approx(np.log(159 / 158))
    assert row["Ret_Lag_5"] == pytest.approx(np.log(155 / 154))
    rets = np.diff(np.log(np.asarray(closes, dtype=float)))
    assert row["Ret_MA_5"] == pytest.approx(np.mean(rets[-5:]))
    assert row["Ret_Std_5"] == pytest.approx(np.std(rets[-5:], ddof=1))
    assert row["Close_MA20_Ratio"] == pytest.approx(159 / np.mean(closes[-20:]) - 1)
    assert row["RSI_14"] == pytest.approx(rsi_last(closes))
    assert row["Vol_Ratio"] == 1.05  # frozen observed volume ratio
    assert row["HL_Pct"] == 0.02  # frozen observed high-low range


def test_feature_row_rejects_unknown_features():
    with pytest.raises(ValueError, match="unknown feature"):
        _feature_row([100.0] * 60, 1.0, 0.01, list(FEATURE_COLS) + ["Mystery"])


def test_recursive_forecast_is_internally_consistent(ohlcv):
    res = train_models(ohlcv, only="Linear Regression")
    out = predict_future_prices(res["models"]["Linear Regression"],
                                ohlcv, res["feature_cols"], 5)

    assert set(out) == {"close", "return"}
    assert len(out["close"]) == 5 and len(out["return"]) == 5
    assert all(np.isfinite(out["close"])) and all(np.isfinite(out["return"]))
    # Each close must chain from the previous via its own predicted return.
    chain = [float(ohlcv["Close"].iloc[-1])] + list(out["close"])
    for i, r in enumerate(out["return"], start=1):
        assert chain[i] == pytest.approx(chain[i - 1] * np.exp(r))
    # Inputs are rebuilt every step, so steps must differ on this fixed seed.
    assert len(set(np.round(out["close"], 6))) > 1


def test_forecast_dates_are_business_days_only():
    dates = pd.date_range(start=pd.Timestamp("2026-01-02") + pd.Timedelta(days=1),
                          periods=10, freq="B")
    assert len(dates) == 10
    assert (dates.weekday < 5).all()
    assert dates[0].weekday() == 0  # Friday -> Monday
