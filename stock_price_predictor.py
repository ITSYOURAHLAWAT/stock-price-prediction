import matplotlib.pyplot as plt

# Shared logic lives in src/ (Phase 1 refactor). This file keeps only CLI code.
from src.data import fetch_history
from src.indicators import calculate_technical_indicators as _calculate_indicators
from src.features import create_features as _create_features
from src.models import train_models as _train_models, get_feature_columns as _get_feature_columns
from src.forecast import predict_future_prices as _predict_future


class StockPredictor:
    def __init__(self, ticker, period="2y"):
        self.ticker = ticker
        self.period = period
        self.data = None
        self.models = {}
        self.scaler = None

    def fetch_data(self):
        """Fetch stock data from Yahoo Finance"""
        try:
            self.data = fetch_history(self.ticker, period=self.period)
        except Exception as e:
            print(f"❌ Error fetching data: {e}")
            return False
        if self.data is None or self.data.empty:
            print(f"❌ No data found for ticker '{self.ticker}'. Please check the ticker symbol and your connection.")
            self.data = None
            return False
        print(f"✅ Successfully fetched {len(self.data)} days of data for {self.ticker}")
        return True

    def calculate_technical_indicators(self):
        """Calculate technical indicators"""
        if self.data is None:
            print("❌ No data available. Please fetch data first.")
            return

        # Shared computation; keep the legacy CLI column set (no Bollinger Bands).
        full = _calculate_indicators(self.data)
        for col in ('MA_20', 'MA_50', 'RSI'):
            self.data[col] = full[col]

        # Price change percentage (legacy CLI computed this here, not in features)
        self.data['Price_Change'] = self.data['Close'].pct_change()

        print("✅ Technical indicators calculated")

    def create_features(self):
        """Create features for machine learning"""
        if self.data is None:
            raise ValueError("No data available. Please fetch data first.")
        # Type assertion: after None check, self.data is guaranteed to be a DataFrame
        data = self.data
        assert data is not None, "Data should not be None at this point"
        # Legacy CLI never computed Close_Std_* — preserve that exactly.
        feat = _create_features(data, include_close_std=False)
        # Indicator columns carried over from calculate_technical_indicators()
        # describe day t (unlike src features); shift them so row t only sees
        # information up to t-1. (Price_Change is already shifted inside src.)
        for col in ('MA_20', 'MA_50', 'RSI', 'Volume_MA'):
            if col in feat.columns:
                feat[col] = feat[col].shift(1)
        return feat.dropna()

    def train_models(self):
        """Train prediction models"""
        if self.data is None:
            print("❌ No data available. Please fetch data first.")
            return

        try:
            models, predictions, metrics, test_index, y_test, _feature_cols, baseline = _train_models(
                self.data, include_close_std=False
            )
        except ValueError as e:
            print(f"❌ {e}")
            return

        # Keep a fitted scaler for predict_future (same object the app uses).
        self.scaler = models['Linear Regression'][1]

        results = {}

        for name in ('Linear Regression', 'Random Forest'):
            m = metrics[name]
            results[name] = {
                'model': models[name][0],
                'predictions': predictions[name],
                'mae': m['MAE'],
                'mse': m['MSE'],
                'rmse': m['RMSE'],
                'r2': m['R²'],
                'test_dates': test_index,
                'actual': y_test,
            }

            print(f"✅ {name} trained - R²: {m['R²']:.4f}, RMSE: ${m['RMSE']:.2f}")

        print(f"📏 Naive baseline (prev close) - R²: {baseline['R²']:.4f}, RMSE: ${baseline['RMSE']:.2f}")

        self.models = results
        return results

    def plot_results(self):
        """Plot stock data and predictions"""
        if self.data is None or not self.models:
            print("❌ No data or models available.")
            return

        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'{self.ticker} Stock Analysis', fontsize=16)

        # Plot 1: Stock price with moving averages
        axes[0, 0].plot(self.data.index, self.data['Close'], label='Close Price', linewidth=2)
        axes[0, 0].plot(self.data.index, self.data['MA_20'], label='MA 20', alpha=0.7)
        axes[0, 0].plot(self.data.index, self.data['MA_50'], label='MA 50', alpha=0.7)
        axes[0, 0].set_title('Stock Price & Moving Averages')
        axes[0, 0].set_ylabel('Price ($)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Plot 2: Volume
        axes[0, 1].bar(self.data.index, self.data['Volume'], alpha=0.6, color='orange')
        axes[0, 1].set_title('Trading Volume')
        axes[0, 1].set_ylabel('Volume')
        axes[0, 1].grid(True, alpha=0.3)

        # Plot 3: RSI
        axes[1, 0].plot(self.data.index, self.data['RSI'], color='purple')
        axes[1, 0].axhline(y=70, color='r', linestyle='--', alpha=0.7, label='Overbought (70)')
        axes[1, 0].axhline(y=30, color='g', linestyle='--', alpha=0.7, label='Oversold (30)')
        axes[1, 0].set_title('RSI (Relative Strength Index)')
        axes[1, 0].set_ylabel('RSI')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # Plot 4: Predictions comparison
        best_model = min(self.models.keys(), key=lambda x: self.models[x]['rmse'])
        model_data = self.models[best_model]

        axes[1, 1].plot(model_data['test_dates'], model_data['actual'],
                       label='Actual', linewidth=2, color='blue')
        axes[1, 1].plot(model_data['test_dates'], model_data['predictions'],
                       label=f'{best_model} Predicted', linewidth=2, color='red', linestyle='--')
        axes[1, 1].set_title(f'Actual vs Predicted ({best_model})')
        axes[1, 1].set_ylabel('Price ($)')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def predict_future(self, days=7):
        """Predict future stock prices"""
        if not self.models:
            print("❌ No trained models available.")
            return

        if self.data is None:
            print("❌ No data available.")
            return

        # Use the best performing model
        best_model_name = min(self.models.keys(), key=lambda x: self.models[x]['rmse'])
        best_model = self.models[best_model_name]['model']

        print(f"🔮 Predicting next {days} days using {best_model_name}:")

        # Same feature columns as training; the recursive forecast rebuilds
        # each step's row by feature name (see src.forecast).
        featured_data = self.create_features()
        assert featured_data is not None, "Featured data should not be None"
        feature_cols = _get_feature_columns(featured_data)

        current_price = self.data['Close'].iloc[-1]
        predictions = _predict_future(
            best_model, self.scaler, self.data, feature_cols, days,
        )

        for day, pred in enumerate(predictions, start=1):
            change_pct = ((pred - current_price) / current_price) * 100
            print(f"Day {day}: ${pred:.2f} ({change_pct:+.2f}%)")

        return predictions

    def display_summary(self):
        """Display summary statistics"""
        if self.data is None or len(self.data) < 2:
            print("❌ Not enough data available.")
            return

        current_price = self.data['Close'].iloc[-1]
        price_change = self.data['Close'].iloc[-1] - self.data['Close'].iloc[-2]
        price_change_pct = (price_change / self.data['Close'].iloc[-2]) * 100

        print(f"\n📊 {self.ticker} Summary:")
        print(f"Current Price: ${current_price:.2f}")
        print(f"Daily Change: ${price_change:+.2f} ({price_change_pct:+.2f}%)")
        print(f"Period High: ${self.data['High'].max():.2f}")
        print(f"Period Low: ${self.data['Low'].min():.2f}")
        print(f"Average Volume: {self.data['Volume'].mean():,.0f}")

        if self.models:
            print("\n🤖 Model Performance:")
            for name, results in self.models.items():
                print(f"{name}: R² = {results['r2']:.4f}, RMSE = ${results['rmse']:.2f}")


def main():
    """Main function to run the stock predictor"""
    print("🚀 Advanced Stock Price Predictor")
    print("=" * 40)

    # Get user input
    ticker = input("Enter stock ticker (e.g., AAPL): ").upper()

    if not ticker:
        ticker = "AAPL"
        print("Using default ticker: AAPL")

    # Initialize predictor
    predictor = StockPredictor(ticker)

    # Fetch and process data
    if predictor.fetch_data():
        predictor.calculate_technical_indicators()
        predictor.display_summary()

        # Train models
        print("\n🤖 Training machine learning models...")
        predictor.train_models()

        # Make future predictions
        predictor.predict_future(days=7)

        # Plot results
        print("\n📈 Generating plots...")
        predictor.plot_results()

    else:
        print("Failed to fetch data. Please check the ticker symbol.")


if __name__ == "__main__":
    main()
