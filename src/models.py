"""Shared modeling: return-space training, dual-space evaluation, baselines.

Target: next-period log return ``r[t] = log(Close[t]/Close[t-1])`` predicted
from information up to ``t-1``. Evaluation reports metrics on returns AND on
reconstructed prices (``Close[t-1] * exp(r_hat[t])`` with ACTUAL previous
closes, so models stay comparable with the naive baseline).
"""

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .features import FEATURE_COLS, TARGET_COL, VectorLike, create_features, reconstruct_close

# Minimum usable rows after feature engineering (covers the 50-day window
# plus the t-1 shift, with margin for a meaningful train/test split).
MIN_ROWS_AFTER_FEATURES = 60

# Walk-forward folds: expanding window, one-day gap between train and test.
N_SPLITS = 5
WF_GAP = 1

# Fixed, modest estimators — deliberately NOT tuned (no search in this repo).
MODEL_SPECS = [
    ("Linear Regression", LinearRegression()),
    (
        "Random Forest",
        RandomForestRegressor(
            n_estimators=200, min_samples_leaf=5, max_depth=6, random_state=42, n_jobs=-1
        ),
    ),
]


def get_feature_columns(featured_data: pd.DataFrame) -> list[str]:
    """Return the scale-free feature list, failing fast if any is missing."""
    missing = [c for c in FEATURE_COLS if c not in featured_data.columns]
    if missing:
        raise ValueError(f"Missing expected feature(s): {missing}")
    return list(FEATURE_COLS)


def return_metrics(y_true: VectorLike, y_pred: VectorLike) -> dict[str, float]:
    """MAE/RMSE on log returns."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }


def price_metrics(y_true: VectorLike, y_pred: VectorLike) -> dict[str, float]:
    """MAE/RMSE + zero-safe MAPE on reconstructed prices."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    valid = y_true != 0
    mape = (
        float(np.mean(np.abs((y_true[valid] - y_pred[valid]) / y_true[valid])) * 100)
        if np.any(valid)
        else float("nan")
    )
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAPE": mape,
    }


def directional_accuracy(actual_ret: VectorLike, pred_ret: VectorLike) -> float:
    """Share of days where sign(predicted return) == sign(actual return)."""
    actual_ret = np.asarray(actual_ret, dtype=float)
    pred_ret = np.asarray(pred_ret, dtype=float)
    if len(actual_ret) == 0:
        return float("nan")
    return float(np.mean(np.sign(pred_ret) == np.sign(actual_ret)))


def r2_vs_zero(actual_ret: VectorLike, pred_ret: VectorLike) -> float:
    """Out-of-sample R² against the zero-return baseline: 1 - SSE/SSE_naive.

    Negative means worse than predicting zero every day.
    """
    actual_ret = np.asarray(actual_ret, dtype=float)
    pred_ret = np.asarray(pred_ret, dtype=float)
    sse_naive = float(np.sum(actual_ret**2))
    if sse_naive == 0:
        return float("nan")
    return float(1 - np.sum((actual_ret - pred_ret) ** 2) / sse_naive)


def beats_baseline(r2_oos: float, model_price_rmse: float, naive_price_rmse: float) -> bool:
    """Verdict: beats the baseline only if BOTH hold (never one alone).

    - pooled return-space OOS R² > 0, AND
    - price-space RMSE below the persistence (naive) RMSE.

    Note the two clauses live in different spaces on purpose: same-space
    ``R² > 0`` is algebraically identical to ``RMSE < naive RMSE``, so a
    single-space AND would be vacuous.
    """
    try:
        return bool(r2_oos > 0 and model_price_rmse < naive_price_rmse)
    except TypeError:
        return False


def baseline_warning_text(model_name: str, model_mape: float, baseline_mape: float) -> str | None:
    """Warning text when a forecast loses to the naive baseline, else None."""
    if model_mape > baseline_mape:
        return (
            f"⚠️ The {model_name} forecast (MAPE {model_mape:.2f}%) does not beat "
            f"the naive baseline (MAPE {baseline_mape:.2f}%). Treat it with extra "
            "skepticism — past patterns alone aren't predicting this series."
        )
    return None


def _evaluate_block(actual_ret, prev_close, actual_close, pred_ret):
    """Metric bundle for one out-of-sample block (fold or pooled)."""
    pred_close = reconstruct_close(prev_close, pred_ret)
    return {
        "n": int(len(actual_ret)),
        "return": return_metrics(actual_ret, pred_ret),
        "price": price_metrics(actual_close, pred_close),
        "dir_acc": directional_accuracy(actual_ret, pred_ret),
        "r2_oos": r2_vs_zero(actual_ret, pred_ret),
    }


def _resolve_wanted(only):
    if only in (None, "Both (Comparison)"):
        return [name for name, _ in MODEL_SPECS]
    if only in ("Linear Regression", "Random Forest"):
        return [only]
    raise ValueError(f"Unknown model selection: {only!r}")


def _fit_pipelines(X_train, y_train, wanted):
    """Fit one Pipeline(StandardScaler, model) per wanted model (train only)."""
    fitted = {}
    for name, estimator in MODEL_SPECS:
        if name not in wanted:
            continue
        pipe = Pipeline([("scaler", StandardScaler()), ("model", clone(estimator))])
        pipe.fit(X_train, y_train)
        fitted[name] = pipe
    return fitted


def train_models(data: pd.DataFrame, only: str | None = None) -> dict:
    """Train return models with walk-forward validation.

    ``only`` selects
    ``"Random Forest"``, ``"Linear Regression"``, ``"Both (Comparison)"`` or
    ``None`` (both). Baselines are always computed. No hyperparameter search
    is performed anywhere: estimator settings are fixed in ``MODEL_SPECS``.

    Evaluation is walk-forward (``TimeSeriesSplit`` expanding window, gap of
    one day, pipeline refit per fold). Pooled metrics are computed once over
    the concatenated out-of-sample predictions. ``models`` holds the final
    pipelines refit on ALL available data (used for forecasting).

    Returns a result dict::

        {
          'feature_cols': [...], 'target_col': 'Target',
          'models': {name: final fitted Pipeline},
          'test': {'index', 'prev_close', 'actual_ret', 'actual_close',
                   'pred_ret': {name}, 'pred_close': {name}},  # pooled OOS
          'metrics': {name: {'return', 'price', 'dir_acc', 'r2_oos',
                             'folds': [per-fold blocks]}},
          'baselines': {'zero_return': {'return', 'price'},
                        'always_up_dir_acc': float},  # pooled OOS
          'residuals': {name: pooled OOS return residuals},
          'verdict': {name: bool},  # beats_baseline on pooled metrics
        }

    Raises ``ValueError`` on too few rows or an unknown ``only`` value.
    """
    featured = create_features(data)

    if len(featured) < MIN_ROWS_AFTER_FEATURES:
        raise ValueError(
            f"Not enough usable rows after feature engineering "
            f"({len(featured)} < {MIN_ROWS_AFTER_FEATURES}). "
            "Use a longer date range."
        )

    feature_cols = get_feature_columns(featured)
    X = featured[feature_cols]
    y = featured[TARGET_COL]
    prev_close = featured["Close"].shift(1).loc[X.index]
    actual_close = featured["Close"].loc[X.index]

    wanted = _resolve_wanted(only)

    # Walk-forward: expanding train window, refit per fold, gap of one day.
    splitter = TimeSeriesSplit(n_splits=N_SPLITS, gap=WF_GAP)
    fold_pred_ret = {name: [] for name in wanted}
    fold_blocks = {name: [] for name in wanted}
    pooled_idx = []
    for train_pos, test_pos in splitter.split(X):
        X_train, X_test = X.iloc[train_pos], X.iloc[test_pos]
        y_train, y_test = y.iloc[train_pos], y.iloc[test_pos]
        fitted = _fit_pipelines(X_train, y_train, wanted)
        test_idx = X_test.index
        pooled_idx.append(test_idx)
        for name, pipe in fitted.items():
            pred_ret = np.asarray(pipe.predict(X_test), dtype=float)
            fold_pred_ret[name].append((test_idx, pred_ret))
            block = _evaluate_block(
                np.asarray(y_test, dtype=float),
                np.asarray(prev_close.loc[test_idx], dtype=float),
                np.asarray(actual_close.loc[test_idx], dtype=float),
                pred_ret,
            )
            block["train_size"] = int(len(train_pos))
            block["test_start_pos"] = int(test_pos[0])
            block["train_end_pos"] = int(train_pos[-1])
            fold_blocks[name].append(block)

    # Pooled out-of-sample series (folds are chronological and disjoint).
    pooled_index = pooled_idx[0].append(pooled_idx[1:])
    test_block = {
        "index": pooled_index,
        "prev_close": prev_close.loc[pooled_index],
        "actual_ret": y.loc[pooled_index],
        "actual_close": actual_close.loc[pooled_index],
        "pred_ret": {},
        "pred_close": {},
    }
    metrics = {}
    residuals = {}
    for name in wanted:
        concat_ret = np.concatenate([pred for _, pred in fold_pred_ret[name]])
        test_block["pred_ret"][name] = concat_ret
        test_block["pred_close"][name] = reconstruct_close(test_block["prev_close"], concat_ret)
        pooled = _evaluate_block(
            np.asarray(test_block["actual_ret"], dtype=float),
            np.asarray(test_block["prev_close"], dtype=float),
            np.asarray(test_block["actual_close"], dtype=float),
            concat_ret,
        )
        pooled["folds"] = fold_blocks[name]
        metrics[name] = {
            "return": pooled["return"],
            "price": pooled["price"],
            "dir_acc": pooled["dir_acc"],
            "r2_oos": pooled["r2_oos"],
            "folds": fold_blocks[name],
        }
        residuals[name] = concat_ret - np.asarray(test_block["actual_ret"], dtype=float)

    zero_ret = np.zeros(len(pooled_index))
    naive_price = reconstruct_close(test_block["prev_close"], zero_ret)
    baselines = {
        "zero_return": {
            "return": return_metrics(test_block["actual_ret"], zero_ret),
            "price": price_metrics(test_block["actual_close"], naive_price),
        },
        "always_up_dir_acc": float(np.mean(np.asarray(test_block["actual_ret"], dtype=float) > 0)),
    }

    # Final models refit on ALL available data (used for forecasting).
    final_models = _fit_pipelines(X, y, wanted)

    verdict = {
        name: beats_baseline(
            metrics[name]["r2_oos"],
            metrics[name]["price"]["RMSE"],
            baselines["zero_return"]["price"]["RMSE"],
        )
        for name in wanted
    }

    return {
        "feature_cols": feature_cols,
        "target_col": TARGET_COL,
        "models": final_models,
        "test": test_block,
        "metrics": metrics,
        "baselines": baselines,
        "residuals": residuals,
        "verdict": verdict,
    }
