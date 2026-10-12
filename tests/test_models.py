"""Metrics edge cases, train-only-selected, baselines, min-rows guard."""
import numpy as np
import pytest

from src.models import (
    baseline_warning_text,
    directional_accuracy,
    price_metrics,
    r2_vs_zero,
    return_metrics,
    train_models,
)


def test_return_metrics_on_tiny_arrays():
    m = return_metrics([0.01, -0.02, 0.03], [0.0, 0.0, 0.0])
    assert m["MAE"] == pytest.approx(0.02)
    assert m["RMSE"] == pytest.approx(np.sqrt((0.0001 + 0.0004 + 0.0009) / 3))
    assert set(m) == {"MAE", "RMSE"}


def test_price_mape_all_zero_actuals_returns_nan():
    m = price_metrics(np.zeros(5), np.zeros(5))
    assert np.isnan(m["MAPE"])


def test_price_mape_skips_zero_actuals():
    m = price_metrics(np.array([0.0, 200.0, 400.0]), np.array([1.0, 202.0, 405.0]))
    assert m["MAPE"] == pytest.approx((1.0 + 1.25) / 2)


def test_r2_vs_zero_reference_points():
    assert r2_vs_zero([0.01, -0.02], [0.01, -0.02]) == pytest.approx(1.0)
    assert r2_vs_zero([0.01, -0.02], [0.0, 0.0]) == pytest.approx(0.0)
    assert r2_vs_zero([0.01, -0.02], [0.05, 0.05]) < 0.0


def test_directional_accuracy_basic():
    assert directional_accuracy([0.01, -0.02, 0.03], [0.005, -0.01, -0.01]) == pytest.approx(2 / 3)


def test_directional_accuracy_edges():
    assert directional_accuracy([0.01, -0.02], [0.02, -0.01]) == 1.0
    assert directional_accuracy([0.01, -0.02], [-0.02, 0.01]) == 0.0
    assert directional_accuracy([0.0, 0.0], [0.0, 0.0]) == 1.0  # sign(0) == sign(0)
    assert np.isnan(directional_accuracy([], []))


def test_baseline_warning_text_both_branches():
    assert baseline_warning_text("RF", 0.5, 1.0) is None  # winning: silent
    text = baseline_warning_text("RF", 1.5, 1.0)
    assert "does not beat the naive baseline" in text
    assert "1.50%" in text and "1.00%" in text
    assert baseline_warning_text("RF", float("nan"), 1.0) is None


def test_train_result_shape_and_features(ohlcv):
    res = train_models(ohlcv)
    assert set(res) == {"feature_cols", "target_col", "models", "test",
                        "metrics", "baselines", "residuals", "verdict"}
    assert res["target_col"] == "Target"
    assert len(res["feature_cols"]) == 15
    assert "Close_Lag_1" not in res["feature_cols"]  # no price-level features
    assert set(res["models"]) == {"Linear Regression", "Random Forest"}
    for name, m in res["metrics"].items():
        assert set(m) == {"return", "price", "dir_acc", "r2_oos", "folds"}
    assert set(res["baselines"]) == {"zero_return", "always_up_dir_acc"}


def test_train_only_selected_trains_subset(ohlcv):
    full = train_models(ohlcv)
    rf_only = train_models(ohlcv, only="Random Forest")
    assert set(rf_only["models"]) == {"Random Forest"}
    assert set(rf_only["metrics"]) == {"Random Forest"}
    np.testing.assert_allclose(
        rf_only["test"]["pred_ret"]["Random Forest"],
        full["test"]["pred_ret"]["Random Forest"])
    assert set(rf_only["baselines"]) == {"zero_return", "always_up_dir_acc"}


def test_train_bogus_selection_raises(ohlcv):
    with pytest.raises(ValueError, match="Unknown model selection"):
        train_models(ohlcv, only="Bogus")


def test_short_data_raises_value_error(short_df):
    with pytest.raises(ValueError, match="60"):
        train_models(short_df)
