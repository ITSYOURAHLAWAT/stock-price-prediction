import matplotlib.pyplot as plt
import numpy as np

# Shared logic lives in src/ (Phase 1 refactor). This file keeps only CLI code.
from src.data import fetch_history, sanitize_ticker
from src.features import create_features as _create_features
from src.forecast import predict_future_prices as _predict_future
from src.indicators import calculate_technical_indicators as _calculate_indicators
from src.models import get_feature_columns as _get_feature_columns
from src.models import train_models as _train_models


class StockPredictor:
    def __init__(self, ticker, period="2y"):
        self.ticker = ticker
        self.period = period
        self.data = None
        self.models = {}
        self.pipelines = {}
        self.feature_cols = None

    def fetch_data(self):
        """Fetch stock data from Yahoo Finance"""
        try:
            self.data = fetch_history(self.ticker, period=self.period)
        except Exception as e:
            print(f"❌ Network/API error fetching data: {e}")
            print("   Check your connection; if EVERY ticker fails, try: pip install -U yfinance")
            return False
        if self.data is None or self.data.empty:
            print(f"❌ No data found for ticker '{self.ticker}'. Please check the ticker symbol.")
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
        for col in ("MA_20", "MA_50", "RSI"):
            self.data[col] = full[col]

        # Price change percentage (legacy CLI computed this here, not in features)
        self.data["Price_Change"] = self.data["Close"].pct_change()

        print("✅ Technical indicators calculated")

    def create_features(self):
        """Create features for machine learning (unified scale-free set)."""
        if self.data is None:
            raise ValueError("No data available. Please fetch data first.")
        # Type assertion: after None check, self.data is guaranteed to be a DataFrame
        data = self.data
        assert data is not None, "Data should not be None at this point"
        # Display/indicator columns (MA_20/MA_50/RSI/Price_Change/...) are for
        # charts only: the return model trains exclusively on the unified
        # scale-free set, so drop them before feature engineering.
        display_cols = [
            c
            for c in (
                "MA_20",
                "MA_50",
                "RSI",
                "Price_Change",
                "Volume_MA",
                "BB_Middle",
                "BB_Upper",
                "BB_Lower",
            )
            if c in data.columns
        ]
        clean = data.drop(columns=display_cols)
        return _create_features(clean)

    def train_models(self):
        """Train return models (plus baselines) and keep plot/forecast state."""
        if self.data is None:
            print("❌ No data available. Please fetch data first.")
            return

        try:
            res = _train_models(self.data)
        except ValueError as e:
            print(f"❌ {e}")
            return

        test, metrics, baselines = res["test"], res["metrics"], res["baselines"]
        results = {}

        for name, m in metrics.items():
            results[name] = {
                "model": res["models"][name],  # fitted Pipeline(StandardScaler, regressor)
                "predictions": np.asarray(test["pred_close"][name]),
                "mae": m["price"]["MAE"],
                "mse": m["price"]["RMSE"] ** 2,
                "rmse": m["price"]["RMSE"],
                "r2": m["r2_oos"],
                "test_dates": test["index"],
                "actual": test["actual_close"],
            }

            print(
                f"✅ {name} trained - OOS R²: {m['r2_oos']:.4f}, "
                f"price RMSE: ${m['price']['RMSE']:.2f}, "
                f"MAPE: {m['price']['MAPE']:.2f}%, dir: {m['dir_acc'] * 100:.1f}%"
            )

        zb = baselines["zero_return"]
        print(
            f"📏 Naive baseline (zero return) - price RMSE: ${zb['price']['RMSE']:.2f}, "
            f"MAPE: {zb['price']['MAPE']:.2f}%, always-up dir: "
            f"{baselines['always_up_dir_acc'] * 100:.1f}%"
        )
        for vname, vbeats in res["verdict"].items():
            print(
                f"{'✅' if vbeats else '⛔'} {vname} verdict: "
                f"{'beats' if vbeats else 'does not beat'} the naive baseline "
                f"(pooled R² {metrics[vname]['r2_oos']:.3f})"
            )

        self.models = results
        # Keep the training feature columns (and the fitted pipelines) so
        # predict_future reuses them instead of rebuilding features for names.
        self.pipelines = res["models"]
        self.feature_cols = res["feature_cols"]
        return results

    def plot_results(self):
        """Plot stock data and predictions"""
        if self.data is None or not self.models:
            print("❌ No data or models available.")
            return

        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f"{self.ticker} Stock Analysis", fontsize=16)

        # Plot 1: Stock price with moving averages
        axes[0, 0].plot(self.data.index, self.data["Close"], label="Close Price", linewidth=2)
        axes[0, 0].plot(self.data.index, self.data["MA_20"], label="MA 20", alpha=0.7)
        axes[0, 0].plot(self.data.index, self.data["MA_50"], label="MA 50", alpha=0.7)
        axes[0, 0].set_title("Stock Price & Moving Averages")
        axes[0, 0].set_ylabel("Price ($)")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Plot 2: Volume
        axes[0, 1].bar(self.data.index, self.data["Volume"], alpha=0.6, color="orange")
        axes[0, 1].set_title("Trading Volume")
        axes[0, 1].set_ylabel("Volume")
        axes[0, 1].grid(True, alpha=0.3)

        # Plot 3: RSI
        axes[1, 0].plot(self.data.index, self.data["RSI"], color="purple")
        axes[1, 0].axhline(y=70, color="r", linestyle="--", alpha=0.7, label="Overbought (70)")
        axes[1, 0].axhline(y=30, color="g", linestyle="--", alpha=0.7, label="Oversold (30)")
        axes[1, 0].set_title("RSI (Relative Strength Index)")
        axes[1, 0].set_ylabel("RSI")
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # Plot 4: Predictions comparison
        best_model = min(self.models.keys(), key=lambda x: self.models[x]["rmse"])
        model_data = self.models[best_model]

        axes[1, 1].plot(
            model_data["test_dates"],
            model_data["actual"],
            label="Actual",
            linewidth=2,
            color="blue",
        )
        axes[1, 1].plot(
            model_data["test_dates"],
            model_data["predictions"],
            label=f"{best_model} Predicted",
            linewidth=2,
            color="red",
            linestyle="--",
        )
        axes[1, 1].set_title(f"Actual vs Predicted ({best_model})")
        axes[1, 1].set_ylabel("Price ($)")
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

        # Use the best performing model (price RMSE on the holdout)
        best_model_name = min(self.models.keys(), key=lambda x: self.models[x]["rmse"])
        best_model = self.pipelines[best_model_name]

        print(f"🔮 Predicting next {days} days using {best_model_name}:")

        # Reuse the training feature columns (fallback: recompute names only
        # if predict_future is called without training first).
        if self.feature_cols is None:
            featured_data = self.create_features()
            assert featured_data is not None, "Featured data should not be None"
            feature_cols = _get_feature_columns(featured_data)
        else:
            feature_cols = self.feature_cols

        current_price = self.data["Close"].iloc[-1]
        predictions = _predict_future(
            best_model,
            self.data,
            feature_cols,
            days,
        )["close"]

        for day, pred in enumerate(predictions, start=1):
            change_pct = ((pred - current_price) / current_price) * 100
            print(f"Day {day}: ${pred:.2f} ({change_pct:+.2f}%)")

        return predictions

    def display_summary(self):
        """Display summary statistics"""
        if self.data is None or len(self.data) < 2:
            print("❌ Not enough data available.")
            return

        current_price = self.data["Close"].iloc[-1]
        price_change = self.data["Close"].iloc[-1] - self.data["Close"].iloc[-2]
        price_change_pct = (price_change / self.data["Close"].iloc[-2]) * 100

        print(f"\n📊 {self.ticker} Summary:")
        print(f"Current Price: ${current_price:.2f}")
        print(f"Daily Change: ${price_change:+.2f} ({price_change_pct:+.2f}%)")
        print(f"Period High: ${self.data['High'].max():.2f}")
        print(f"Period Low: ${self.data['Low'].min():.2f}")
        print(f"Average Volume: {self.data['Volume'].mean():,.0f}")

        if self.models:
            print("\n🤖 Model Performance (out-of-sample):")
            for name, results in self.models.items():
                print(f"{name}: OOS R² = {results['r2']:.4f}, price RMSE = ${results['rmse']:.2f}")


def main():
    """Main function to run the stock predictor"""
    print("🚀 Advanced Stock Price Predictor")
    print("=" * 40)

    # Get user input
    raw = input("Enter stock ticker (e.g., AAPL): ")

    if not raw.strip():
        ticker = "AAPL"
        print("Using default ticker: AAPL")
    else:
        ticker = sanitize_ticker(raw)
        if ticker is None:
            print(
                "❌ Invalid ticker. Use letters, digits, `.`, `-`, `=` or `^` (max 12 characters)."
            )
            return

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
