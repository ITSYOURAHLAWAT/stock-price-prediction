"""Shared modeling: return-space training, dual-space evaluation, baselines.

Target: next-period log return ``r[t] = log(Close[t]/Close[t-1])`` predicted
from information up to ``t-1``. Evaluation reports metrics on returns AND on
reconstructed prices (``Close[t-1] * exp(r_hat[t])`` with ACTUAL previous
closes, so models stay comparable with the naive baseline).
"""

import numpy as np
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .features import TARGET_COL, FEATURE_COLS, create_features, reconstruct_close

# Minimum usable rows after feature engineering (covers the 50-day window
# plus the t-1 shift, with margin for a meaningful train/test split).
MIN_ROWS_AFTER_FEATURES = 60

# Fixed, modest estimators — deliberately NOT tuned (no search in this repo).
MODEL_SPECS = [
    ('Linear Regression', LinearRegression()),
    ('Random Forest', RandomForestRegressor(
        n_estimators=200, min_samples_leaf=5, max_depth=6,
        random_state=42, n_jobs=-1)),
]


def get_feature_columns(featured_data):
    """Return the scale-free feature list, failing fast if any is missing."""
    missing = [c for c in FEATURE_COLS if c not in featured_data.columns]
    if missing:
        raise ValueError(f"Missing expected feature(s): {missing}")
    return list(FEATURE_COLS)


def return_metrics(y_true, y_pred):
    """MAE/RMSE on log returns."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return {
        'MAE': float(mean_absolute_error(y_true, y_pred)),
        'RMSE': float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }


def price_metrics(y_true, y_pred):
    """MAE/RMSE + zero-safe MAPE on reconstructed prices."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    valid = y_true != 0
    mape = float(np.mean(np.abs((y_true[valid] - y_pred[valid]) / y_true[valid])) * 100) \
        if np.any(valid) else float('nan')
    return {
        'MAE': float(mean_absolute_error(y_true, y_pred)),
        'RMSE': float(np.sqrt(mean_squared_error(y_true, y_pred))),
        'MAPE': mape,
    }


def directional_accuracy(actual_ret, pred_ret):
    """Share of days where sign(predicted return) == sign(actual return)."""
    actual_ret = np.asarray(actual_ret, dtype=float)
    pred_ret = np.asarray(pred_ret, dtype=float)
    if len(actual_ret) == 0:
        return float('nan')
    return float(np.mean(np.sign(pred_ret) == np.sign(actual_ret)))


def r2_vs_zero(actual_ret, pred_ret):
    """Out-of-sample R² against the zero-return baseline: 1 - SSE/SSE_naive.

    Negative means worse than predicting zero every day.
    """
    actual_ret = np.asarray(actual_ret, dtype=float)
    pred_ret = np.asarray(pred_ret, dtype=float)
    sse_naive = float(np.sum(actual_ret ** 2))
    if sse_naive == 0:
        return float('nan')
    return float(1 - np.sum((actual_ret - pred_ret) ** 2) / sse_naive)


def _evaluate_block(actual_ret, prev_close, actual_close, pred_ret):
    """Metric bundle for one out-of-sample block (fold or pooled)."""
    pred_close = reconstruct_close(prev_close, pred_ret)
    return {
        'n': int(len(actual_ret)),
        'return': return_metrics(actual_ret, pred_ret),
        'price': price_metrics(actual_close, pred_close),
        'dir_acc': directional_accuracy(actual_ret, pred_ret),
        'r2_oos': r2_vs_zero(actual_ret, pred_ret),
    }


def _resolve_wanted(only):
    if only in (None, 'Both (Comparison)'):
        return [name for name, _ in MODEL_SPECS]
    if only in ('Linear Regression', 'Random Forest'):
        return [only]
    raise ValueError(f"Unknown model selection: {only!r}")


def _fit_pipelines(X_train, y_train, wanted):
    """Fit one Pipeline(StandardScaler, model) per wanted model (train only)."""
    fitted = {}
    for name, estimator in MODEL_SPECS:
        if name not in wanted:
            continue
        pipe = Pipeline([('scaler', StandardScaler()), ('model', clone(estimator))])
        pipe.fit(X_train, y_train)
        fitted[name] = pipe
    return fitted


def train_models(data, include_close_std=True, only=None):
    """Train return models and evaluate on a chronological 80/20 split.

    ``include_close_std`` is accepted for backward compatibility and ignored
    (single unified scale-free feature set). ``only`` selects
    ``"Random Forest"``, ``"Linear Regression"``, ``"Both (Comparison)"`` or
    ``None`` (both). Baselines are always computed.

    Returns a result dict (see 5b: walk-forward upgrades the evaluation but
    keeps this shape)::

        {
          'feature_cols': [...], 'target_col': 'Target',
          'models': {name: fitted Pipeline on the TRAIN split},
          'test': {'index', 'prev_close', 'actual_ret', 'actual_close',
                   'pred_ret': {name}, 'pred_close': {name}},
          'metrics': {name: {'return', 'price', 'dir_acc', 'r2_oos', 'folds'}},
          'baselines': {'zero_return': {'return', 'price'},
                        'always_up_dir_acc': float},
        }

    Raises ``ValueError`` on too few rows or an unknown ``only`` value.
    """
    featured = create_features(data, include_close_std=include_close_std)

    if len(featured) < MIN_ROWS_AFTER_FEATURES:
        raise ValueError(
            f"Not enough usable rows after feature engineering "
            f"({len(featured)} < {MIN_ROWS_AFTER_FEATURES}). "
            "Use a longer date range."
        )

    feature_cols = get_feature_columns(featured)
    X = featured[feature_cols]
    y = featured[TARGET_COL]
    prev_close = featured['Close'].shift(1).loc[X.index]
    actual_close = featured['Close'].loc[X.index]

    # Single chronological 80/20 split (5b replaces this with walk-forward).
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    wanted = _resolve_wanted(only)
    fitted = _fit_pipelines(X_train, y_train, wanted)

    test_idx = X_test.index
    test_block = {
        'index': test_idx,
        'prev_close': prev_close.loc[test_idx],
        'actual_ret': y_test,
        'actual_close': actual_close.loc[test_idx],
        'pred_ret': {},
        'pred_close': {},
    }
    metrics = {}
    for name, pipe in fitted.items():
        pred_ret = pipe.predict(X_test)
        test_block['pred_ret'][name] = pred_ret
        test_block['pred_close'][name] = reconstruct_close(
            test_block['prev_close'], pred_ret)
        block = _evaluate_block(
            np.asarray(y_test, dtype=float),
            np.asarray(test_block['prev_close'], dtype=float),
            np.asarray(test_block['actual_close'], dtype=float),
            np.asarray(pred_ret, dtype=float))
        metrics[name] = {
            'return': block['return'],
            'price': block['price'],
            'dir_acc': block['dir_acc'],
            'r2_oos': block['r2_oos'],
            'folds': [block],
        }

    zero_ret = np.zeros(len(y_test))
    baselines = {
        'zero_return': {
            'return': return_metrics(y_test, zero_ret),
            'price': price_metrics(
                test_block['actual_close'],
                reconstruct_close(test_block['prev_close'], zero_ret)),
        },
        'always_up_dir_acc': float(np.mean(np.asarray(y_test, dtype=float) > 0)),
    }

    return {
        'feature_cols': feature_cols,
        'target_col': TARGET_COL,
        'models': fitted,
        'test': test_block,
        'metrics': metrics,
        'baselines': baselines,
    }
