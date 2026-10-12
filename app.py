import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from datetime import date, datetime
import warnings
warnings.filterwarnings('ignore')

# This is your updated app.py file

# Page configuration
st.set_page_config(
    page_title="Advanced Stock Price Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Main title
st.markdown('<h1 class="main-header">📈 Advanced Stock Price Predictor</h1>', unsafe_allow_html=True)

# Sidebar for inputs
st.sidebar.header("📊 Configuration")

# Stock ticker input with suggestions
popular_stocks = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA", "META", "NVDA", "NFLX", "SPY", "QQQ"]
ticker = st.sidebar.selectbox("Select or enter stock ticker:", [""] + popular_stocks, index=1)
custom_ticker = st.sidebar.text_input("Or enter custom ticker:", "")
final_ticker = custom_ticker.upper() if custom_ticker else ticker

# Date range selection
col1, col2 = st.sidebar.columns(2)
with col1:
    start_date = st.date_input("Start Date", value=date(2020, 1, 1))
with col2:
    end_date = st.date_input("End Date", value=date.today())

# Model selection
model_type = st.sidebar.selectbox(
    "Select Model:",
    ["Random Forest", "Linear Regression", "Both (Comparison)"]
)

# Prediction parameters
prediction_days = st.sidebar.slider("Days to predict:", 1, 30, 7)

# Technical indicators toggle
show_indicators = st.sidebar.checkbox("Show Technical Indicators", True)

def fetch_stock_data(ticker, start, end):
    """Fetch stock data with error handling"""
    try:
        stock = yf.Ticker(ticker)
        
        # Try using period first (more reliable)
        try:
            # Calculate period from dates
            if isinstance(start, date):
                start_dt = datetime.combine(start, datetime.min.time())
            else:
                start_dt = start
            
            if isinstance(end, date):
                end_dt = datetime.combine(end, datetime.max.time())
            else:
                end_dt = end
            
            days_diff = (end_dt - start_dt).days
            
            # Use period if reasonable, otherwise use start/end
            if days_diff <= 730:  # 2 years
                if days_diff <= 5:
                    period = "5d"
                elif days_diff <= 30:
                    period = "1mo"
                elif days_diff <= 90:
                    period = "3mo"
                elif days_diff <= 180:
                    period = "6mo"
                elif days_diff <= 365:
                    period = "1y"
                else:
                    period = "2y"
                
                data = stock.history(period=period)
            else:
                # For longer periods, use start/end dates
                data = stock.history(start=start, end=end)
        except:
            # Fallback to start/end dates
            data = stock.history(start=start, end=end)
        
        if data.empty or len(data) == 0:
            # Try alternative method with period
            st.warning(f"Trying alternative method to fetch data for {ticker}...")
            data = stock.history(period="1y")
            
            if data.empty or len(data) == 0:
                st.error(f"❌ No data found for ticker '{ticker}'. Please check:")
                st.error("1. The ticker symbol is correct (e.g., AAPL, GOOGL, MSFT)")
                st.error("2. Your internet connection is working")
                st.error("3. Yahoo Finance API is accessible")
                return None
        
        # Filter by date range if needed
        if not data.empty and isinstance(start, date) and isinstance(end, date):
            # Convert date to datetime for comparison
            start_dt = datetime.combine(start, datetime.min.time())
            end_dt = datetime.combine(end, datetime.max.time())
            data = data[(data.index >= start_dt) & (data.index <= end_dt)]
            if data.empty:
                st.warning(f"⚠️ No data in selected date range. Showing available data.")
                data = stock.history(period="1y")
            
        return data
    except Exception as e:
        st.error(f"❌ Error fetching data: {str(e)}")
        st.error("This might be due to:")
        st.error("• Internet connection issues")
        st.error("• Yahoo Finance API temporarily unavailable")
        st.error("• Invalid ticker symbol")
        st.exception(e)  # Show full error for debugging
        return None

def calculate_technical_indicators(data):
    """Calculate technical indicators"""
    df = data.copy()
    
    # Moving averages
    df['MA_20'] = df['Close'].rolling(window=20).mean()
    df['MA_50'] = df['Close'].rolling(window=50).mean()
    
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Bollinger Bands
    df['BB_Middle'] = df['Close'].rolling(window=20).mean()
    bb_std = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
    df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
    
    # Volume moving average
    df['Volume_MA'] = df['Volume'].rolling(window=20).mean()
    
    return df

def create_features(data):
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
        df[f'Close_Std_{window}'] = df['Close'].rolling(window=window).std()
        df[f'Volume_MA_{window}'] = df['Volume'].rolling(window=window).mean()
    
    # Remove rows with NaN values
    df = df.dropna()
    
    return df

def train_models(data):
    """Train prediction models"""
    # Create features
    featured_data = create_features(data)
    
    # Select feature columns (exclude date and target)
    feature_cols = [col for col in featured_data.columns if col not in ['Close', 'Open', 'High', 'Low', 'Volume', 'Dividends', 'Stock Splits']]
    
    X = featured_data[feature_cols]
    y = featured_data['Close']
    
    # Split data (80% train, 20% test)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
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
    
    return models, predictions, metrics, X_test.index, y_test, feature_cols

def calculate_metrics(y_true, y_pred):
    """Calculate model performance metrics"""
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    return {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'R²': r2,
        'MAPE': mape
    }

def predict_future_prices(model, scaler, data, feature_cols, days):
    """Predict future stock prices"""
    featured_data = create_features(data)
    last_features = featured_data[feature_cols].iloc[-1:].values
    last_features_scaled = scaler.transform(last_features)
    
    future_predictions = []
    current_features = last_features_scaled.copy()
    
    for _ in range(days):
        pred = model.predict(current_features)[0]
        future_predictions.append(pred)
        
        # Update features for next prediction (simplified approach)
        # In practice, you'd want to update all features properly
        current_features[0, 0] = pred  # Assuming first feature is most recent close price
    
    return future_predictions

# Main app logic
# Show instructions if no analysis has been run
if 'analysis_run' not in st.session_state:
    st.info("👆 Select a stock ticker from the sidebar and click '🚀 Analyze Stock' to get started!")

analyze_button = st.sidebar.button("🚀 Analyze Stock", type="primary")

if analyze_button or st.session_state.get('auto_run', False):
    if not final_ticker or final_ticker == "":
        st.error("⚠️ Please select or enter a stock ticker to analyze.")
    else:
        with st.spinner(f"Fetching data for {final_ticker}..."):
            # yfinance accepts date objects directly, but ensure they're in the right format
            data = fetch_stock_data(final_ticker, start_date, end_date)
        
        if data is not None:
            st.session_state['analysis_run'] = True
            # Display basic stock info
            col1, col2, col3, col4 = st.columns(4)
            
            current_price = data['Close'].iloc[-1]
            price_change = data['Close'].iloc[-1] - data['Close'].iloc[-2]
            price_change_pct = (price_change / data['Close'].iloc[-2]) * 100
            
            with col1:
                st.metric("Current Price", f"${current_price:.2f}", f"{price_change:+.2f} ({price_change_pct:+.2f}%)")
            with col2:
                st.metric("52W High", f"${data['High'].max():.2f}")
            with col3:
                st.metric("52W Low", f"${data['Low'].min():.2f}")
            with col4:
                st.metric("Avg Volume", f"{data['Volume'].mean():,.0f}")
            
            # Calculate technical indicators
            if show_indicators:
                data_with_indicators = calculate_technical_indicators(data)
            else:
                data_with_indicators = data
            
            # Create price chart
            fig = make_subplots(
                rows=3, cols=1,
                subplot_titles=('Stock Price & Technical Indicators', 'Volume', 'RSI'),
                vertical_spacing=0.05,
                row_heights=[0.6, 0.2, 0.2]
            )
            
            # Price and moving averages
            fig.add_trace(
                go.Scatter(x=data.index, y=data['Close'], name='Close Price', line=dict(color='blue', width=2)),
                row=1, col=1
            )
            
            if show_indicators:
                fig.add_trace(
                    go.Scatter(x=data_with_indicators.index, y=data_with_indicators['MA_20'], 
                              name='MA 20', line=dict(color='orange', width=1)),
                    row=1, col=1
                )
                fig.add_trace(
                    go.Scatter(x=data_with_indicators.index, y=data_with_indicators['MA_50'], 
                              name='MA 50', line=dict(color='red', width=1)),
                    row=1, col=1
                )
                
                # Bollinger Bands
                fig.add_trace(
                    go.Scatter(x=data_with_indicators.index, y=data_with_indicators['BB_Upper'], 
                              name='BB Upper', line=dict(color='gray', dash='dash'), opacity=0.5),
                    row=1, col=1
                )
                fig.add_trace(
                    go.Scatter(x=data_with_indicators.index, y=data_with_indicators['BB_Lower'], 
                              name='BB Lower', line=dict(color='gray', dash='dash'), opacity=0.5),
                    row=1, col=1
                )
            
            # Volume
            fig.add_trace(
                go.Bar(x=data.index, y=data['Volume'], name='Volume', marker_color='lightblue'),
                row=2, col=1
            )
            
            # RSI
            if show_indicators:
                fig.add_trace(
                    go.Scatter(x=data_with_indicators.index, y=data_with_indicators['RSI'], 
                              name='RSI', line=dict(color='purple')),
                    row=3, col=1
                )
                fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)  # type: ignore
                fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)  # type: ignore
            
            fig.update_layout(height=800, title_text=f"{final_ticker} Stock Analysis", showlegend=True)
            fig.update_xaxes(title_text="Date", row=3, col=1)
            fig.update_yaxes(title_text="Price ($)", row=1, col=1)
            fig.update_yaxes(title_text="Volume", row=2, col=1)
            fig.update_yaxes(title_text="RSI", row=3, col=1)
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Model training and prediction
            st.subheader("🤖 Machine Learning Predictions")
            
            with st.spinner("Training models..."):
                models, predictions, metrics, test_dates, y_test, feature_cols = train_models(data)
            
            # Display model performance
            if model_type == "Both (Comparison)":
                col1, col2 = st.columns(2)
                
                for i, (model_name, model_metrics) in enumerate(metrics.items()):
                    with col1 if i == 0 else col2:
                        st.write(f"**{model_name} Performance:**")
                        for metric_name, value in model_metrics.items():
                            if metric_name == 'R²':
                                st.metric(metric_name, f"{value:.4f}")
                            elif metric_name == 'MAPE':
                                st.metric(metric_name, f"{value:.2f}%")
                            else:
                                st.metric(metric_name, f"${value:.2f}")
            else:
                selected_model = model_type
                st.write(f"**{selected_model} Performance:**")
                cols = st.columns(5)
                for i, (metric_name, value) in enumerate(metrics[selected_model].items()):
                    with cols[i]:
                        if metric_name == 'R²':
                            st.metric(metric_name, f"{value:.4f}")
                        elif metric_name == 'MAPE':
                            st.metric(metric_name, f"{value:.2f}%")
                        else:
                            st.metric(metric_name, f"${value:.2f}")
            
            # Predictions vs Actual chart
            fig_pred = go.Figure()
            
            if model_type == "Both (Comparison)":
                fig_pred.add_trace(go.Scatter(x=test_dates, y=y_test, name='Actual', line=dict(color='blue', width=2)))
                colors = ['red', 'green']
                for i, (model_name, pred) in enumerate(predictions.items()):
                    fig_pred.add_trace(go.Scatter(x=test_dates, y=pred, name=f'{model_name} Predicted', 
                                                line=dict(color=colors[i], width=1, dash='dash')))
            else:
                fig_pred.add_trace(go.Scatter(x=test_dates, y=y_test, name='Actual', line=dict(color='blue', width=2)))
                fig_pred.add_trace(go.Scatter(x=test_dates, y=predictions[model_type], name='Predicted', 
                                            line=dict(color='red', width=1, dash='dash')))
            
            fig_pred.update_layout(title="Actual vs Predicted Prices (Test Set)", xaxis_title="Date", yaxis_title="Price ($)")
            st.plotly_chart(fig_pred, use_container_width=True)
            
            # Future predictions
            st.subheader(f"🔮 Future Price Predictions ({prediction_days} days)")
            
            best_model_name = min(metrics.keys(), key=lambda x: metrics[x]['MAPE'])
            best_model, best_scaler = models[best_model_name]
            
            future_prices = predict_future_prices(best_model, best_scaler, data, feature_cols, prediction_days)
            future_dates = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=prediction_days)
            
            # Create future predictions chart
            fig_future = go.Figure()
            
            # Historical data (last 30 days)
            recent_data = data.tail(30)
            fig_future.add_trace(go.Scatter(x=recent_data.index, y=recent_data['Close'], 
                                          name='Historical', line=dict(color='blue', width=2)))
            
            # Future predictions
            fig_future.add_trace(go.Scatter(x=future_dates, y=future_prices, 
                                          name='Predicted', line=dict(color='red', width=2, dash='dash')))
            
            fig_future.update_layout(title=f"Future Price Predictions using {best_model_name}", 
                                   xaxis_title="Date", yaxis_title="Price ($)")
            st.plotly_chart(fig_future, use_container_width=True)
            
            # Future predictions table
            future_df = pd.DataFrame({
                'Date': future_dates,
                'Predicted Price': [f"${price:.2f}" for price in future_prices],
                'Change from Current': [f"{((price - current_price) / current_price * 100):+.2f}%" for price in future_prices]
            })
            
            st.dataframe(future_df, use_container_width=True)
            
            # Summary statistics
            st.subheader("📊 Data Summary")
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Price Statistics:**")
                st.dataframe(data[['Open', 'High', 'Low', 'Close', 'Volume']].describe())
            
            with col2:
                st.write("**Model Comparison:**")
                comparison_df = pd.DataFrame(metrics).T
                st.dataframe(comparison_df.round(4))

# Sidebar information
st.sidebar.markdown("---")
st.sidebar.markdown("### ℹ️ About")
st.sidebar.markdown("""
This advanced stock predictor uses:
- **Random Forest & Linear Regression** models
- **Technical indicators** (MA, RSI, Bollinger Bands)
- **Feature engineering** with lagged variables
- **Interactive visualizations** with Plotly

**Disclaimer:** This is for educational purposes only. 
Not financial advice!
""")

# Footer
st.markdown("---")
st.markdown("Made with ❤️ using Streamlit | Data provided by Yahoo Finance")

    