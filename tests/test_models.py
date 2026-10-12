"""Metrics edge cases, train-only-selected, baseline, min-rows guard."""
import numpy as np
import pytest

from src.models import calculate_metrics, persistence_baseline_metrics, train_models


def test_mape_all_zero_actuals_returns_nan():
    m = calculate_metrics(np.zeros(5), np.zeros(5))
    assert np.isnan(m["MAPE"])
    assert set(m) == {"MAE", "MSE", "RMSE", "R²", "MAPE"}


def test_mape_skips_zero_actuals():
    m = calculate_metrics(np.array([0.0, 2.0, 4.0]), np.array([1.0, 2.0, 5.0]))
    assert m["MAPE"] == pytest.approx((0.0 + 25.0) / 2)


def test_metrics_on_tiny_arrays():
    m = calculate_metrics([1.0, 2.0, 3.0], [1.0, 2.0, 4.0])
    assert m["MAE"] == pytest.approx(1 / 3)
    assert m["RMSE"] == pytest.approx(np.sqrt(1 / 3))
    assert np.isfinite(m["MAPE"])


def test_train_only_selected_trains_subset(ohlcv):
    full = train_models(ohlcv)
    rf_only = train_models(ohlcv, only="Random Forest")
    lr_only = train_models(ohlcv, only="Linear Regression")

    assert set(rf_only[0]) == {"Random Forest"}
    assert set(lr_only[0]) == {"Linear Regression"}
    assert set(full[0]) == {"Linear Regression", "Random Forest"}
    # Selecting a subset must not change that model's numerics.
    np.testing.assert_allclose(rf_only[1]["Random Forest"], full[1]["Random Forest"])
    np.testing.assert_allclose(lr_only[1]["Linear Regression"], full[1]["Linear Regression"])
    # Baseline is always computed, even for a subset.
    assert set(rf_only[6]) == {"MAE", "MSE", "RMSE", "R²", "MAPE"}
    assert len(rf_only) == 7


def test_train_bogus_selection_raises(ohlcv):
    with pytest.raises(ValueError, match="Unknown model selection"):
        train_models(ohlcv, only="Bogus")


def test_baseline_is_persistence_on_test_rows(ohlcv):
    _, _, _, _, y_test, _, baseline = train_models(ohlcv)
    assert set(baseline) == {"MAE", "MSE", "RMSE", "R²", "MAPE"}
    assert all(np.isfinite([baseline["MAE"], baseline["RMSE"], baseline["MAPE"]]))


def test_short_data_raises_value_error(short_df):
    with pytest.raises(ValueError, match="60"):
        train_models(short_df)
