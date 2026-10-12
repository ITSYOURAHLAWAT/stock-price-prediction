"""Shared feature engineering (union of app.py and stock_price versions).

Phase 1 note: ``include_close_std=True`` reproduces the Streamlit app exactly;
``include_close_std=False`` reproduces the legacy CLI exactly (it never
computed ``Close_Std_*``). Leakage fixes are Phase 2 work.
"""


def create_features(data, include_close_std=True):
    """Create features for machine learning"""
    df = data.copy()

    # Price-based features
    df['Price_Change'] = df['Close'].pct_change()
    df['High_Low_Ratio'] = df['High'] / df['Low']
    df['Price_Range'] = df['High'] - df['Low']

    # Lagged features
    for lag in [1, 2, 3, 5]:
        df[f'Close_Lag_{lag}'] = df['Close'].shift(lag)
        df[f'Volume_Lag_{lag}'] = df['Volume'].shift(lag)

    # Rolling statistics
    for window in [5, 10, 20]:
        df[f'Close_MA_{window}'] = df['Close'].rolling(window=window).mean()
        if include_close_std:
            df[f'Close_Std_{window}'] = df['Close'].rolling(window=window).std()
        df[f'Volume_MA_{window}'] = df['Volume'].rolling(window=window).mean()

    # Remove rows with NaN values
    df = df.dropna()

    return df
