"""Shared future-price forecasting (moved verbatim from app.py).

Phase 1 note: the simplified ``current_features[0, 0] = pred`` update is
unchanged. ``recursive_update=False`` reproduces the legacy CLI behaviour
(re-predict from the same last row every step). A correct by-name recursive
update is Phase 2 work.
"""

from .features import create_features


def predict_future_prices(model, scaler, data, feature_cols, days,
                          include_close_std=True, recursive_update=True):
    """Predict future stock prices"""
    featured_data = create_features(data, include_close_std=include_close_std)
    last_features = featured_data[feature_cols].iloc[-1:].values
    last_features_scaled = scaler.transform(last_features)

    future_predictions = []

    if not recursive_update:
        # Legacy CLI behaviour: same input row every step.
        for _ in range(days):
            pred = model.predict(last_features_scaled)[0]
            future_predictions.append(pred)
        return future_predictions

    current_features = last_features_scaled.copy()

    for _ in range(days):
        pred = model.predict(current_features)[0]
        future_predictions.append(pred)

        # Update features for next prediction (simplified approach)
        # In practice, you'd want to update all features properly
        current_features[0, 0] = pred  # Assuming first feature is most recent close price

    return future_predictions
