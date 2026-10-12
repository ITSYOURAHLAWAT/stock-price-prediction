"""No-leakage tests: row t of the feature frame must not read OHLCV[t]."""

import numpy as np

from src.features import TARGET_COL, create_features, reconstruct_close
from src.models import get_feature_columns


def test_rows_up_to_perturbed_row_unchanged(ohlcv):
    feats = create_features(ohlcv)
    cols = get_feature_columns(feats)

    bumped = ohlcv.copy()
    bumped.iloc[60, bumped.columns.get_indexer(["Open", "High", "Low", "Close", "Volume"])] *= 2.0
    feats2 = create_features(bumped)

    # Warmup drops the first 50 raw rows (50-day ratio window + t-1 shift),
    # so featured position 10 == raw row 60.
    assert feats[cols].iloc[:11].fillna(0).equals(feats2[cols].iloc[:11].fillna(0))
    # ... while later rows legitimately change (history is actually used).
    assert not feats[cols].iloc[11:].fillna(0).equals(feats2[cols].iloc[11:].fillna(0))


def test_last_row_bump_changes_no_feature_value(ohlcv):
    feats = create_features(ohlcv)
    cols = get_feature_columns(feats)

    bumped = ohlcv.copy()
    bumped.iloc[-1, bumped.columns.get_indexer(["Open", "High", "Low", "Close", "Volume"])] *= 2.0
    feats2 = create_features(bumped)

    # Nothing may read Close[t]/High[t]/Low[t]/Volume[t] (raw OHLCV columns
    # pass through untouched, so compare feature columns only).
    assert feats[cols].fillna(0).equals(feats2[cols].fillna(0))


def test_features_are_scale_free(ohlcv):
    """Multiplying every price by 10 leaves features AND target unchanged."""
    scaled = ohlcv.copy()
    for col in ["Open", "High", "Low", "Close"]:
        scaled[col] = scaled[col] * 10.0

    a = create_features(ohlcv)
    b = create_features(scaled)
    cols = get_feature_columns(a)

    np.testing.assert_allclose(a[cols].to_numpy(), b[cols].to_numpy(), rtol=1e-9)
    np.testing.assert_allclose(a[TARGET_COL].to_numpy(), b[TARGET_COL].to_numpy(), rtol=1e-9)


def test_reconstruct_close_round_trip(ohlcv):
    """prev_close * exp(log-return) recovers the actual close exactly."""
    close = ohlcv["Close"].to_numpy()
    logret = np.log(close[1:] / close[:-1])
    np.testing.assert_allclose(reconstruct_close(close[:-1], logret), close[1:], rtol=1e-12)
