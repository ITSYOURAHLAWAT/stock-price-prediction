# 📈 Advanced Stock Price Predictor

A stock price analysis application using machine learning and technical analysis.
Fetches historical data from Yahoo Finance, trains Random Forest / Linear
Regression models to predict next-day log returns from scale-free signals,
validates them walk-forward out-of-sample, and shows an interactive dashboard
with a short-horizon recursive forecast and approximate intervals.

## 🚀 Features

### **Machine Learning Models**
- **Random Forest Regressor** - ensemble method
- **Linear Regression** - simple baseline model for comparison
- **Naive persistence baseline** (predict previous close) - the benchmark to beat
- **Feature Engineering** - scale-free signals only (lagged log returns,
  rolling return mean/std, distance-from-trend ratios, RSI, volume and
  high-low ratios); every feature uses only data available up to the
  previous day (no look-ahead); the target is the next-day log return
- **Performance Metrics** - MAE, RMSE, R², MAPE

### **Technical Analysis**
- **Moving Averages** (20-day, 50-day)
- **RSI** (Relative Strength Index)
- **Bollinger Bands**
- **Volume Analysis**

### **Interactive Web App**
- **Historical Stock Data** from Yahoo Finance (cached for 1 hour)
- **Interactive Charts** with Plotly
- **Multi-panel Dashboard** that persists while you tweak display widgets
- **Future Price Predictions** (1-30 business days, illustrative only, with an
  approximate 80% interval band)
- **Model Performance Comparison** with train-only-what-you-select
- **CSV download** of the forecast

## 📦 Installation

1. **Clone the repository:**
```bash
git clone <your-repo-url>
cd Stock-Price-Predictor
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **For development/tests:**
```bash
pip install -r requirements-dev.txt
```

## 🎯 Usage

### **Web Application (Recommended)**
```bash
streamlit run app.py
```
Then open http://localhost:8501 in your browser. Enter a ticker, pick a date
range, click **Analyze Stock**. Changing the ticker box or dates does nothing
until you click Analyze again; display widgets (model, days, indicators)
re-render instantly from the cache.

### **Command Line Version**
```bash
python stock_price_predictor.py
```

### **Tests (offline, Yahoo Finance is mocked)**
```bash
pytest -q
```

## 📊 How It Works

1. **Data Fetching**: Downloads historical OHLCV data from Yahoo Finance for the
   exact start/end range (no silent substitution; empty results are an error)
2. **Feature Engineering**: Creates scale-free signals (return lags, rolling
   return stats, trend ratios, RSI, volume/high-low ratios), all shifted so
   row *t* only sees information up to *t−1*; the target is the next-day log
   return
3. **Model Training**: Trains the selected model(s) as
   `Pipeline(StandardScaler, model)` and evaluates them with walk-forward
   validation (5 expanding folds, 1-day gap, refit per fold), plus a naive
   zero-return baseline and an always-up direction baseline on the same
   out-of-sample rows
4. **Prediction**: Recursively forecasts the next 1-30 business days (each
   step predicts a return, rebuilds features by name from the simulated
   series; volume/high-low inputs stay frozen), with an approximate 80%
   interval from pooled out-of-sample residuals
5. **Visualization**: Interactive charts, actual-vs-predicted plots, per-fold
   tables, verdict vs baseline, and a model comparison table including the
   baseline

## 🛠️ Project Structure

```
Stock-Price-Predictor/
├── app.py                      # Streamlit web application (UI + cache wrappers)
├── stock_price_predictor.py    # Standalone command-line version
├── src/                        # Shared logic imported by both entry points
│   ├── data.py                 # Fetching, ticker sanitizing, column flattening
│   ├── indicators.py           # Technical indicators (MA, RSI, Bollinger Bands)
│   ├── features.py             # Leakage-free feature engineering
│   ├── models.py               # Model training, metrics, naive baseline
│   └── forecast.py             # Recursive by-name forecasting
├── tests/                      # pytest suite (offline; Yahoo Finance mocked)
│   ├── conftest.py             # Synthetic data + mock fixtures
│   ├── test_features.py        # No-leakage tests
│   ├── test_indicators.py      # Indicator values on known series
│   ├── test_models.py          # Metrics edges, model selection, baseline
│   ├── test_forecast.py        # By-name updates, length, business days
│   ├── test_data.py            # Fetch validation, MultiIndex, tz handling
│   ├── test_sanitize.py        # Ticker allow/deny lists
│   └── test_app.py             # UI persistence, forecast selection, caching
├── requirements.txt            # Python dependencies (pinned)
├── requirements-dev.txt        # Test dependencies (pinned)
├── pytest.ini                  # pytest configuration
├── LICENSE                     # MIT License
├── .gitignore                  # Python/venv/editor ignores
└── README.md                   # Project documentation
```

## 🎛️ Configuration Options

- **Stock Selection**: Single ticker input, sanitized (letters, digits, `.`, `-`, `=`, `^`; max 12 characters)
- **Date Range**: Flexible start and end date selection (start must precede end)
- **Model Selection**: Random Forest, Linear Regression, or Both (comparison)
- **Prediction Period**: 1-30 business days of future forecasting
- **Technical Indicators**: Toggle on/off the MA / Bollinger / RSI overlays

## 📊 Performance Metrics

The application evaluates models pooled out-of-sample over 5 walk-forward folds:
- **MAE** (Mean Absolute Error) — on returns and on reconstructed prices
- **RMSE** (Root Mean Square Error) — on returns and on reconstructed prices
- **R²** (Coefficient of Determination) — out-of-sample vs the zero-return baseline
- **MAPE** (Mean Absolute Percentage Error) — on prices only
- **Directional accuracy** — vs an always-up baseline
- **Verdict** — "beats baseline" only with pooled return R² > 0 AND price
  RMSE below the naive baseline; otherwise the app says so plainly

## ⚠️ Known Limitations

- It predicts **next-day log returns** (converted to price levels for display),
  not a trading strategy.
- **Tree models cannot extrapolate** beyond patterns seen in training; the
  scale-free features remove price-level memorization but not regime change.
- **Multi-day recursive forecasts are illustrative**: each step feeds the
  previous prediction back in, so errors compound; volume and high-low inputs
  are frozen at last-known values because the model doesn't predict them.
- **Intervals are approximate**: residual percentiles scaled by √horizon, not
  calibrated probabilities.
- **Walk-forward validation with fixed settings** — no cross-validated tuning,
  no hyperparameter search.
- The **naive zero-return baseline is the benchmark to beat** — on noisy data
  it is often the best "model" in the table, and that is informative.

## 🚨 Disclaimer

**Important**: This application is for educational and research purposes only. It is NOT financial advice. Stock market investments carry risk, and past performance does not guarantee future results. Always consult with qualified financial advisors before making investment decisions.

## 🔧 Technical Requirements

- Python 3.10+
- Internet connection (for fetching stock data; tests run offline with mocks)
- Modern web browser (for Streamlit app)

## 📝 Dependencies

- `streamlit` - Web application framework
- `yfinance` - Yahoo Finance data fetching
- `pandas` - Data manipulation
- `numpy` - Numerical computing
- `scikit-learn` - Machine learning models
- `plotly` - Interactive visualizations
- `matplotlib` - Static plots (for CLI version)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Create a Pull Request

## 📜 License

This project is open source and available under the [MIT License](LICENSE).

## 🎯 Future Enhancements (ideas, not features)

- [ ] LSTM/GRU neural networks
- [ ] Sentiment analysis integration
- [ ] Portfolio optimization features
- [ ] Real-time data streaming
- [ ] More technical indicators
- [ ] Options pricing models
- [ ] Risk management tools

## 📞 Support

If you encounter any issues or have questions:
1. Check the existing issues on GitHub
2. Create a new issue with detailed description
3. Include error messages and system information

---

**Made with ❤️ using Python, Streamlit, and Machine Learning**
