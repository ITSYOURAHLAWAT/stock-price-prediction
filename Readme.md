# 📈 Advanced Stock Price Predictor

A comprehensive stock price prediction application using machine learning and technical analysis.

## 🚀 Features

### **Machine Learning Models**
- **Random Forest Regressor** - Advanced ensemble method
- **Linear Regression** - Baseline model for comparison
- **Feature Engineering** - Technical indicators, lagged variables, rolling statistics
- **Performance Metrics** - MAE, RMSE, R², MAPE

### **Technical Analysis**
- **Moving Averages** (20-day, 50-day)
- **RSI** (Relative Strength Index)
- **Bollinger Bands**
- **Volume Analysis**
- **Price Pattern Recognition**

### **Interactive Web App**
- **Real-time Stock Data** from Yahoo Finance
- **Interactive Charts** with Plotly
- **Multi-panel Dashboard**
- **Future Price Predictions** (1-30 days)
- **Model Performance Comparison**

## 📦 Installation

1. **Clone the repository:**
```bash
git clone https://github.com/yourusername/Stock-Price-Predictor.git
cd Stock-Price-Predictor
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

## 🎯 Usage

### **Web Application (Recommended)**
```bash
streamlit run app.py
```
Then open http://localhost:8501 in your browser.

### **Command Line Version**
```bash
python stock_price_predictor.py
```

## 📊 How It Works

1. **Data Fetching**: Downloads historical stock data from Yahoo Finance
2. **Feature Engineering**: Creates technical indicators and lagged variables
3. **Model Training**: Trains multiple ML models with proper validation
4. **Prediction**: Generates future price forecasts
5. **Visualization**: Creates interactive charts and performance metrics

## 🛠️ Project Structure

```
Stock-Price-Predictor/
├── app.py                      # Main Streamlit web application
├── stock_price_predictor.py    # Standalone command-line version
├── requirements.txt            # Python dependencies
└── README.md                  # Project documentation
```

## 📈 Features Overview

### **Web App Interface**
- Stock ticker selection with popular suggestions
- Customizable date ranges
- Model comparison capabilities
- Interactive technical analysis charts
- Future price predictions with confidence intervals

### **Technical Indicators**
- **SMA/EMA**: Simple & Exponential Moving Averages
- **RSI**: Relative Strength Index for momentum
- **Bollinger Bands**: Volatility indicators
- **Volume Analysis**: Trading volume patterns

### **Machine Learning Features**
- Price-based features (High/Low ratios, price ranges)
- Lagged variables (1, 2, 3, 5 days)
- Rolling statistics (5, 10, 20 day windows)
- Proper train/test splitting with time series data

## 🎛️ Configuration Options

- **Stock Selection**: Choose from popular stocks or enter custom ticker
- **Date Range**: Flexible start and end date selection
- **Model Selection**: Compare Random Forest vs Linear Regression
- **Prediction Period**: 1-30 days future forecasting
- **Technical Indicators**: Toggle on/off various indicators

## 📊 Performance Metrics

The application provides comprehensive model evaluation:
- **MAE** (Mean Absolute Error)
- **RMSE** (Root Mean Square Error)
- **R²** (Coefficient of Determination)
- **MAPE** (Mean Absolute Percentage Error)

## 🚨 Disclaimer

**Important**: This application is for educational and research purposes only. It is NOT financial advice. Stock market investments carry risk, and past performance does not guarantee future results. Always consult with qualified financial advisors before making investment decisions.

## 🔧 Technical Requirements

- Python 3.7+
- Internet connection (for fetching stock data)
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

## 🎯 Future Enhancements

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