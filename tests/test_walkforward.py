"""Walk-forward validation structure (chronology, gaps, scaler discipline)."""

import numpy as np
from sklearn.model_selection import TimeSeriesSplit

from src.features import TARGET_COL, create_features
from src.models import (
    N_SPLITS,
    WF_GAP,
    beats_baseline,
    directional_accuracy,
    get_feature_columns,
    r2_vs_zero,
    train_models,
)


def test_five_folds_with_gap_and_disjoint_chronological_tests(ohlcv):
    res = train_models(ohlcv, only="Linear Regression")
    folds = res["metrics"]["Linear Regression"]["folds"]
    assert len(folds) == N_SPLITS == 5

    pooled_index = res["test"]["index"]
    # Fold test blocks partition the pooled index in order, disjointly.
    sizes = [b["n"] for b in folds]
    assert sum(sizes) == len(pooled_index)
    assert pooled_index.is_monotonic_increasing
    assert len(set(pooled_index)) == len(pooled_index)

    featured = create_features(ohlcv)
    splitter = TimeSeriesSplit(n_splits=N_SPLITS, gap=WF_GAP)
    expected = list(splitter.split(featured))
    assert len(expected) == N_SPLITS
    offset = 0
    for (train_pos, test_pos), block in zip(expected, folds, strict=True):
        # Gap respected: exactly WF_GAP excluded index/indices sit between the
        # last train row and the first test row (test_start - train_last == 2).
        assert test_pos[0] - train_pos[-1] == WF_GAP + 1
        # Fold covers the matching pooled slice.
        assert block["n"] == len(test_pos)
        sl = pooled_index[offset : offset + len(test_pos)]
        assert (sl == featured.index[test_pos]).all()
        offset += len(test_pos)


def test_scaler_fit_on_train_fold_only(ohlcv):
    """Each fold pipeline's scaler must match its TRAIN slice statistics."""
    featured = create_features(ohlcv)
    cols = get_feature_columns(featured)
    X = featured[cols]
    splitter = TimeSeriesSplit(n_splits=N_SPLITS, gap=WF_GAP)

    # Refit one fold manually and compare scaler statistics.
    from src.models import _fit_pipelines

    for train_pos, _test_pos in splitter.split(X):
        fitted = _fit_pipelines(
            X.iloc[train_pos], featured[TARGET_COL].iloc[train_pos], ["Linear Regression"]
        )
        scaler = fitted["Linear Regression"].named_steps["scaler"]
        np.testing.assert_allclose(scaler.mean_, X.iloc[train_pos].mean().to_numpy())
        np.testing.assert_allclose(scaler.scale_, X.iloc[train_pos].std(ddof=0).to_numpy())
        break  # one fold suffices to pin the wiring


def test_no_test_index_in_its_train_set(ohlcv):
    """Within each fold, train and test rows are disjoint (expanding window)."""
    res = train_models(ohlcv, only="Linear Regression")
    folds = res["metrics"]["Linear Regression"]["folds"]
    last_train_end = -1
    for b in folds:
        train = set(range(0, b["train_end_pos"] + 1))
        test = set(range(b["test_start_pos"], b["test_start_pos"] + b["n"]))
        assert train.isdisjoint(test)
        assert b["train_end_pos"] > last_train_end  # window expands
        last_train_end = b["train_end_pos"]


def test_pooled_metrics_match_concatenated_oos(ohlcv):
    res = train_models(ohlcv, only="Linear Regression")
    m = res["metrics"]["Linear Regression"]
    actual = np.asarray(res["test"]["actual_ret"], dtype=float)
    pred = np.asarray(res["test"]["pred_ret"]["Linear Regression"], dtype=float)
    assert m["r2_oos"] == r2_vs_zero(actual, pred)
    assert m["dir_acc"] == directional_accuracy(actual, pred)


def test_verdict_truth_table():
    assert beats_baseline(0.05, 1.0, 1.5) is True
    assert beats_baseline(-0.01, 1.0, 1.5) is False  # R² alone is not a win
    assert beats_baseline(0.05, 2.0, 1.5) is False  # RMSE alone is not a win
    assert beats_baseline(float("nan"), 1.0, 1.5) is False
    assert beats_baseline(0.05, float("nan"), 1.5) is False


def test_residuals_cover_pooled_oos(ohlcv):
    res = train_models(ohlcv, only="Linear Regression")
    r = res["residuals"]["Linear Regression"]
    assert len(r) == len(res["test"]["index"])
    assert np.all(np.isfinite(r))
