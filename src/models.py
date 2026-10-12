"""Shared model training + metrics (chronological 80/20 split)."""

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

from .features import create_features

FEATURE_EXCLUDE = ['Close', 'Open', 'High', 'Low', 'Volume', 'Dividends', 'Stock Splits']

# Minimum usable rows after feature engineering (covers the 20-day windows,
# lags and the t-1 shift, with margin for a meaningful train/test split).
MIN_ROWS_AFTER_FEATURES = 60


def get_feature_columns(featured_data):
    """Select feature columns (exclude OHLCV + corporate-action columns)."""
    return [col for col in featured_data.columns if col not in FEATURE_EXCLUDE]


def calculate_metrics(y_true, y_pred):
    """Calculate model performance metrics (zero-safe MAPE)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    # Prices are never zero in practice, but guard the division anyway.
    valid = y_true != 0
    mape = float(np.mean(np.abs((y_true[valid] - y_pred[valid]) / y_true[valid])) * 100) \
        if np.any(valid) else float('nan')

    return {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'R²': r2,
        'MAPE': mape
    }


def persistence_baseline_metrics(featured_data, y_test):
    """Naive baseline: predict Close[t] = actual Close[t-1] (persistence).

    Uses realised history, so it is directly comparable with the model
    metrics on the same test rows. If a model cannot beat this, it has
    learned nothing beyond "tomorrow looks like today".
    """
    naive_pred = featured_data['Close'].shift(1).loc[y_test.index]
    return calculate_metrics(y_test, naive_pred)


def train_models(data, include_close_std=True):
    """Train prediction models.

    Returns ``(models, predictions, metrics, test_index, y_test,
    feature_cols, baseline_metrics)``. Raises ``ValueError`` when too few
    usable rows survive feature engineering.
    """
    # Create features
    featured_data = create_features(data, include_close_std=include_close_std)

    if len(featured_data) < MIN_ROWS_AFTER_FEATURES:
        raise ValueError(
            f"Not enough usable rows after feature engineering "
            f"({len(featured_data)} < {MIN_ROWS_AFTER_FEATURES}). "
            "Use a longer date range."
        )

    # Select feature columns (exclude date and target)
    feature_cols = get_feature_columns(featured_data)

    X = featured_data[feature_cols]
    y = featured_data['Close']

    # Split data (80% train, 20% test) — positional, as before
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {}
    predictions = {}
    metrics = {}

    # Linear Regression
    lr_model = LinearRegression()
    lr_model.fit(X_train_scaled, y_train)
    lr_pred = lr_model.predict(X_test_scaled)

    models['Linear Regression'] = (lr_model, scaler)
    predictions['Linear Regression'] = lr_pred
    metrics['Linear Regression'] = calculate_metrics(y_test, lr_pred)

    # Random Forest
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf_model.fit(X_train_scaled, y_train)
    rf_pred = rf_model.predict(X_test_scaled)

    models['Random Forest'] = (rf_model, scaler)
    predictions['Random Forest'] = rf_pred
    metrics['Random Forest'] = calculate_metrics(y_test, rf_pred)

    baseline = persistence_baseline_metrics(featured_data, y_test)

    return models, predictions, metrics, X_test.index, y_test, feature_cols, baseline
