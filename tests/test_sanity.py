"""Synthetic end-to-end sanity: noise must lose, a real signal must win.

Both use fixed seeds and the real ``train_models`` pipeline. If the AR(1)
test ever fails, the pipeline is broken or too weak — do NOT loosen these
assertions to make them pass.
"""
import numpy as np
import pandas as pd

from src.models import train_models
from tests.conftest import make_ohlcv


def make_ar1(seed=0, n=1500, phi=0.3, vol=0.01):
    """Prices from AR(1) log returns: r[t] = phi*r[t-1] + eps (learnable)."""
    rng = np.random.default_rng(seed)
    eps = rng.normal(0, vol, n)
    rets = np.zeros(n)
    for t in range(1, n):
        rets[t] = phi * rets[t - 1] + eps[t]
    close = 100 * np.exp(np.cumsum(rets))
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    return pd.DataFrame({
        "Open": close, "High": close * 1.005, "Low": close * 0.995, "Close": close,
        "Volume": rng.integers(1_000_000, 4_000_000, n).astype(float),
    }, index=idx)


def test_random_walk_does_not_beat_baseline():
    res = train_models(make_ohlcv(seed=11))
    assert res["verdict"] == {"Linear Regression": False, "Random Forest": False}
    for name, m in res["metrics"].items():
        assert m["r2_oos"] <= 0.05  # small tolerance, never a win


def test_ar1_signal_beats_zero_baseline():
    res = train_models(make_ar1())
    assert res["verdict"] == {"Linear Regression": True, "Random Forest": True}
    for name, m in res["metrics"].items():
        assert m["r2_oos"] > 0
        assert m["dir_acc"] > res["baselines"]["always_up_dir_acc"]
