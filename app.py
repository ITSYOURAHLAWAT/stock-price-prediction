import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import date

# Shared logic lives in src/ (Phase 1 refactor). This file keeps only UI code.
from src.data import fetch_stock_data as _fetch_stock_data
from src.indicators import calculate_technical_indicators
from src.models import train_models
from src.forecast import predict_future_prices

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
    """Fetch stock data (thin wrapper over src.data)."""
    return _fetch_stock_data(ticker, start, end, on_error=st.error)

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
            if len(data) < 2:
                st.error("⚠️ Not enough data returned (need at least 2 trading days). Try a longer date range.")
                st.stop()
            st.session_state['analysis_run'] = True
            # Display basic stock info
            col1, col2, col3, col4 = st.columns(4)
            
            current_price = data['Close'].iloc[-1]
            price_change = data['Close'].iloc[-1] - data['Close'].iloc[-2]
            price_change_pct = (price_change / data['Close'].iloc[-2]) * 100
            
            with col1:
                st.metric("Current Price", f"${current_price:.2f}", f"{price_change:+.2f} ({price_change_pct:+.2f}%)")
            with col2:
                st.metric("Period High", f"${data['High'].max():.2f}")
            with col3:
                st.metric("Period Low", f"${data['Low'].min():.2f}")
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
            
            try:
                with st.spinner("Training models..."):
                    models, predictions, metrics, test_dates, y_test, feature_cols, baseline = train_models(data)
            except ValueError as e:
                st.error(f"⚠️ {e}")
                st.stop()
            
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
            future_dates = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=prediction_days, freq='B')
            
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
                st.caption(f"📏 Naive baseline (predict previous close): MAPE {baseline['MAPE']:.2f}% · RMSE ${baseline['RMSE']:.2f} · R² {baseline['R²']:.4f}")
                comparison_df = pd.DataFrame({**metrics, 'Naive (Prev Close)': baseline}).T
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

    