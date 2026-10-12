"""No-leakage tests: row t of the feature frame must not read OHLCV[t]."""
from src.features import create_features
from src.models import get_feature_columns


def test_rows_up_to_perturbed_row_unchanged(ohlcv):
    feats = create_features(ohlcv)
    cols = get_feature_columns(feats)

    bumped = ohlcv.copy()
    bumped.iloc[60, bumped.columns.get_indexer(["Open", "High", "Low", "Close", "Volume"])] *= 2.0
    feats2 = create_features(bumped)

    # Warmup drops the first 20 raw rows, so featured position 40 == raw row 60.
    assert feats[cols].iloc[:41].fillna(0).equals(feats2[cols].iloc[:41].fillna(0))
    # ... while later rows legitimately change (history is actually used).
    assert not feats[cols].iloc[41:].fillna(0).equals(feats2[cols].iloc[41:].fillna(0))


def test_last_row_bump_changes_no_feature_value(ohlcv):
    feats = create_features(ohlcv)
    cols = get_feature_columns(feats)

    bumped = ohlcv.copy()
    bumped.iloc[-1, bumped.columns.get_indexer(["Open", "High", "Low", "Close", "Volume"])] *= 2.0
    feats2 = create_features(bumped)

    # Nothing may read Close[t]/High[t]/Low[t]/Volume[t] (raw OHLCV columns
    # pass through untouched, so compare feature columns only).
    assert feats[cols].fillna(0).equals(feats2[cols].fillna(0))
