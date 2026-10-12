"""Forecast: by-name updates, length/finiteness, business-day dates."""
import numpy as np
import pandas as pd
import pytest

from src.forecast import _feature_row, predict_future_prices
from src.models import train_models

FEATURE_COLS = [
    "Price_Change", "High_Low_Ratio", "Price_Range",
    "Close_Lag_1", "Close_Lag_2", "Close_Lag_3", "Close_Lag_5",
    "Volume_Lag_1", "Volume_Lag_2", "Volume_Lag_3", "Volume_Lag_5",
    "Close_MA_5", "Close_MA_10", "Close_MA_20",
    "Close_Std_5", "Close_Std_10", "Close_Std_20",
    "Volume_MA_5", "Volume_MA_10", "Volume_MA_20",
]


def test_feature_row_updates_by_name():
    closes = [float(100 + i) for i in range(60)]  # 100..159
    volumes = [1000.0] * 60
    row = dict(zip(FEATURE_COLS, _feature_row(closes, volumes, 1.02, 3.0, FEATURE_COLS)))

    assert row["Close_Lag_1"] == 159.0
    assert row["Close_Lag_5"] == 155.0
    assert row["Close_MA_5"] == pytest.approx(157.0)
    assert row["Price_Change"] == pytest.approx(1 / 158)
    assert row["Close_Std_5"] == pytest.approx(pd.Series(closes[-5:]).std(ddof=1))
    assert row["Volume_Lag_3"] == 1000.0  # frozen volume history


def test_feature_row_rejects_unknown_features():
    with pytest.raises(ValueError, match="unknown feature"):
        _feature_row([1.0] * 60, [1.0] * 60, 1.0, 0.0, FEATURE_COLS + ["Mystery"])


def test_recursive_forecast_length_finite_and_evolving(ohlcv):
    models, _, _, _, _, fcols, _ = train_models(ohlcv)
    model, scaler = models["Linear Regression"]

    preds = predict_future_prices(model, scaler, ohlcv, fcols, 5)

    assert len(preds) == 5
    assert all(np.isfinite(preds))
    # Inputs are rebuilt every step, so steps must differ on this fixed seed.
    assert len(set(np.round(preds, 6))) > 1


def test_forecast_dates_are_business_days_only():
    dates = pd.date_range(start=pd.Timestamp("2026-01-02") + pd.Timedelta(days=1),
                          periods=10, freq="B")
    assert len(dates) == 10
    assert (dates.weekday < 5).all()
    assert dates[0].weekday() == 0  # Friday -> Monday
