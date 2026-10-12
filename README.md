# 📈 Advanced Stock Price Predictor

A stock price analysis application using machine learning and technical analysis.
Fetches historical data from Yahoo Finance, trains Random Forest / Linear
Regression models on leakage-free lag and rolling features, and shows an
interactive dashboard with a short-horizon recursive forecast.

## 🚀 Features

### **Machine Learning Models**
- **Random Forest Regressor** - ensemble method
- **Linear Regression** - simple baseline model for comparison
- **Naive persistence baseline** (predict previous close) - the benchmark to beat
- **Feature Engineering** - lagged variables (1, 2, 3, 5 days), rolling statistics
  (5, 10, 20 day windows); every feature uses only data available up to the
  previous day (no look-ahead)
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
- **Future Price Predictions** (1-30 business days, illustrative only)
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
2. **Feature Engineering**: Creates lagged variables and rolling statistics,
   all shifted so row *t* only sees information up to *t−1*
3. **Model Training**: Trains the selected model(s) on a chronological 80/20
   train/test split, plus a naive previous-close baseline on the same test rows
4. **Prediction**: Recursively forecasts the next 1-30 business days by
   rebuilding each step's feature row by name
5. **Visualization**: Interactive charts, actual-vs-predicted plots, and a
   model comparison table including the baseline

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

The application provides model evaluation on a chronological holdout:
- **MAE** (Mean Absolute Error)
- **RMSE** (Root Mean Square Error)
- **R²** (Coefficient of Determination)
- **MAPE** (Mean Absolute Percentage Error)
- **Naive baseline** (previous close) shown alongside — if a model can't beat
  it, it hasn't learned anything beyond "tomorrow looks like today"

## ⚠️ Known Limitations

- It predicts **price levels** on a short holdout, not returns or direction.
- **Tree models cannot extrapolate** beyond the price range seen in training;
  forecasts flatten or drift outside it.
- **Multi-day recursive forecasts are illustrative**: each step feeds the
  previous prediction back in, so errors compound; volume and High/Low inputs
  are frozen at last-known values because the model doesn't predict them.
- The **naive previous-close baseline is the benchmark to beat** — on noisy
  data it is often the best "model" in the table, and that is informative.
- No time-series cross-validation or hyperparameter search yet.

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
