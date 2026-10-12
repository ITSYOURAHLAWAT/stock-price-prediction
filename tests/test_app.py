"""Streamlit UI behaviors via AppTest (mocked Yahoo, no network).

Caches are global, so every test uses a distinct ticker AND the
``clean_caches`` fixture. All data uses fixed seeds — no flakiness by design.
"""
import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch

from streamlit.testing.v1 import AppTest

import yfinance
import src.models as models_mod
from src.forecast import predict_future_prices
from src.models import train_models
from tests.conftest import ROOT, make_ohlcv

APP = str(ROOT / "app.py")


def _run_analysis(at, ticker, model=None, days=None):
    at.sidebar.text_input[0].set_value(ticker)
    at.run()
    if model is not None:
        at.sidebar.selectbox[0].set_value(model)
        at.run()
    if days is not None:
        at.sidebar.slider[0].set_value(days)
        at.run()
    at.sidebar.button[0].click()
    at.run()
    assert not at.exception
    return at


def _all_text(at):
    out = []
    for coll in ("markdown", "caption", "error", "warning", "header", "subheader"):
        try:
            out += [e.value for e in getattr(at, coll)]
        except Exception:
            pass
    return "\n".join(str(t) for t in out)


def _future_prices(at):
    tbl = at.dataframe[0].value
    return np.array([float(str(v).replace("$", "")) for v in tbl["Predicted Price"]]), tbl


def test_dashboard_persists_across_widget_changes(clean_caches, yahoo_mock):
    yahoo_mock["frame"] = make_ohlcv(seed=5, tz="America/New_York")
    at = AppTest.from_file(APP)
    at.run()
    assert not at.exception

    _run_analysis(at, "TSTPA")
    price = [m.value for m in at.metric if m.label == "Current Price"][0]
    h0 = yahoo_mock["history_calls"]

    # Slider-only change re-renders from cache: same dashboard, no refetch.
    at.sidebar.slider[0].set_value(10)
    at.run()
    assert not at.exception
    assert [m.value for m in at.metric if m.label == "Current Price"][0] == price
    assert yahoo_mock["history_calls"] == h0
    assert len(at.dataframe[0].value) == 10

    # Ticker edit alone is inert until the next click.
    at.sidebar.text_input[0].set_value("TSTPB")
    at.run()
    assert yahoo_mock["history_calls"] == h0
    assert [m.value for m in at.metric if m.label == "Current Price"][0] == price

    # Clicking analyzes the new ticker.
    at.sidebar.button[0].click()
    at.run()
    assert yahoo_mock["tickers"][-1] == "TSTPB"


def test_forecast_uses_selected_model(clean_caches, yahoo_mock):
    frame = make_ohlcv(seed=5, tz="America/New_York")
    yahoo_mock["frame"] = frame
    models, _, metrics, _, _, fcols, _ = train_models(frame, only="Linear Regression")

    at = AppTest.from_file(APP)
    at.run()
    _run_analysis(at, "TSTFC", model="Linear Regression", days=7)

    expected = np.array(predict_future_prices(models["Linear Regression"][0],
                                              models["Linear Regression"][1],
                                              frame, fcols, 7))
    got, tbl = _future_prices(at)
    assert np.allclose(got, np.round(expected, 2), atol=0.01)
    assert (pd.to_datetime(tbl["Date"]).dt.weekday < 5).all()  # business days only


def test_cache_hit_on_repeat_miss_on_change(clean_caches, yahoo_mock):
    yahoo_mock["frame"] = make_ohlcv(seed=5, tz="America/New_York")
    with patch.object(models_mod, "train_models", MagicMock(wraps=models_mod.train_models)) as spy:
        at = AppTest.from_file(APP)
        at.run()
        _run_analysis(at, "TSTCA")
        assert yahoo_mock["history_calls"] == 1
        assert spy.call_count == 1

        at.sidebar.button[0].click()  # identical inputs
        at.run()
        assert yahoo_mock["history_calls"] == 1
        assert spy.call_count == 1

        at.sidebar.selectbox[0].set_value("Both (Comparison)")  # model change only
        at.run()
        at.sidebar.button[0].click()
        at.run()
        assert yahoo_mock["history_calls"] == 1  # fetch still cached
        assert spy.call_count == 2  # ... but training reruns
        body = _all_text(at)
        assert "Random Forest Performance" in body and "Linear Regression Performance" in body


def test_no_warning_when_model_beats_baseline(clean_caches, yahoo_mock):
    # Seed 5 (offline): LR MAPE 0.864% < baseline 0.868% -> no warning.
    yahoo_mock["frame"] = make_ohlcv(seed=5, tz="America/New_York")
    at = AppTest.from_file(APP)
    at.run()
    _run_analysis(at, "TSTNW", model="Linear Regression")
    assert "does not beat the naive baseline" not in _all_text(at)


def test_warning_when_model_loses_to_baseline(clean_caches, yahoo_mock):
    # Seed 8 (offline): LR MAPE 1.277% > baseline 0.647% -> warning shown.
    yahoo_mock["frame"] = make_ohlcv(seed=8, tz="America/New_York")
    at = AppTest.from_file(APP)
    at.run()
    _run_analysis(at, "TSTWL", model="Linear Regression")
    assert "does not beat the naive baseline" in _all_text(at)
