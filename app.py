import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import date

# Shared logic lives in src/ (Phase 1 refactor). This file keeps only UI code.
from src.data import fetch_stock_data as _fetch_stock_data, sanitize_ticker
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

# Single stock ticker input (sanitized; invalid input is rejected below).
ticker_raw = st.sidebar.text_input(
    "Stock ticker:",
    value="GOOGL",
    help="Popular tickers: AAPL, GOOGL, MSFT, AMZN, TSLA, META, NVDA, NFLX, SPY, QQQ",
)
final_ticker = sanitize_ticker(ticker_raw)

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

# Cached wrappers. Keys contain every input that affects the output, so a
# cache hit is only possible for genuinely identical requests.
@st.cache_data(ttl=3600, show_spinner=False)
def _cached_fetch(clean_ticker, start, end):
    """Yahoo fetch cached for 1 hour. Key = (ticker, start, end)."""
    messages = []
    data = _fetch_stock_data(clean_ticker, start, end, on_error=messages.append)
    return data, tuple(messages)


@st.cache_resource(show_spinner=False)
def _cached_train(clean_ticker, start, end, model_type, data):
    """Training cache. Key = (ticker, start, end, model, data): any change
    (including fresh history) recomputes; identical reruns reuse models."""
    only = None if model_type == "Both (Comparison)" else model_type
    return train_models(data, include_close_std=True, only=only)

# Main app logic
# Show instructions if no analysis has been run
if 'analysis' not in st.session_state:
    st.info("👆 Select a stock ticker from the sidebar and click '🚀 Analyze Stock' to get started!")

analyze_button = st.sidebar.button("🚀 Analyze Stock", type="primary")

if analyze_button:
    if not final_ticker:
        st.error("⚠️ Please enter a valid stock ticker (letters, digits, `.`, `-`, `=` or `^`, max 12 characters).")
    else:
        with st.spinner(f"Fetching data for {final_ticker}..."):
            data, fetch_messages = _cached_fetch(final_ticker, start_date, end_date)
        for message in fetch_messages:
            st.error(message)

        if data is not None:
            if len(data) < 2:
                st.error("⚠️ Not enough data returned (need at least 2 trading days). Try a longer date range.")
            else:
                # Remember this analysis. The dashboard below renders from the
                # stored inputs on every rerun, so moving other widgets (model,
                # days, indicators) re-renders without wiping anything, while
                # editing the ticker/dates takes effect only on the next click.
                st.session_state['analysis'] = {
                    'ticker': final_ticker,
                    'start': start_date,
                    'end': end_date,
                }

# Render from the stored analysis — never from the live ticker/date widgets.
if 'analysis' in st.session_state:
    analysis = st.session_state['analysis']
    a_ticker, a_start, a_end = analysis['ticker'], analysis['start'], analysis['end']
    data, _fetch_messages = _cached_fetch(a_ticker, a_start, a_end)  # cache hit, no spinner
    if data is None or len(data) < 2:
        st.error("⚠️ The stored analysis is no longer available (data fetch failed on refresh). Click '🚀 Analyze Stock' again.")
    else:
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

        fig.update_layout(height=800, title_text=f"{a_ticker} Stock Analysis", showlegend=True)
        fig.update_xaxes(title_text="Date", row=3, col=1)
        fig.update_yaxes(title_text="Price ($)", row=1, col=1)
        fig.update_yaxes(title_text="Volume", row=2, col=1)
        fig.update_yaxes(title_text="RSI", row=3, col=1)

        st.plotly_chart(fig, use_container_width=True)

        # Model training and prediction
        st.subheader("🤖 Machine Learning Predictions")

        try:
            with st.spinner("Training models..."):
                models, predictions, metrics, test_dates, y_test, feature_cols, baseline = _cached_train(
                    a_ticker, a_start, a_end, model_type, data
                )
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
            cols = st.columns(len(metrics[selected_model]))
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
        st.caption("⚠️ Illustrative only — these forecasts extrapolate past patterns and are NOT financial advice.")

        # The forecast uses the SELECTED model; only "Both" falls back to lowest MAPE.
        if model_type == "Both (Comparison)":
            forecast_model_name = min(metrics.keys(), key=lambda x: metrics[x]['MAPE'])
        else:
            forecast_model_name = model_type
        forecast_model, forecast_scaler = models[forecast_model_name]

        if metrics[forecast_model_name]['MAPE'] > baseline['MAPE']:
            st.warning(f"⚠️ The {forecast_model_name} forecast (MAPE {metrics[forecast_model_name]['MAPE']:.2f}%) does not beat the naive baseline (MAPE {baseline['MAPE']:.2f}%). Treat it with extra skepticism — past patterns alone aren't predicting this series.")

        future_prices = predict_future_prices(forecast_model, forecast_scaler, data, feature_cols, prediction_days)
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

        fig_future.update_layout(title=f"Future Price Predictions using {forecast_model_name}",
                               xaxis_title="Date", yaxis_title="Price ($)")
        st.plotly_chart(fig_future, use_container_width=True)

        # Future predictions table
        future_df = pd.DataFrame({
            'Date': future_dates,
            'Predicted Price': [f"${price:.2f}" for price in future_prices],
            'Change from Current': [f"{((price - current_price) / current_price * 100):+.2f}%" for price in future_prices]
        })

        st.dataframe(future_df, use_container_width=True)

        future_raw = pd.DataFrame({
            'Date': future_dates,
            'Predicted_Price': future_prices,
            'Change_Pct': [((price - current_price) / current_price * 100) for price in future_prices],
        })
        st.download_button(
            "⬇️ Download predictions as CSV",
            future_raw.to_csv(index=False),
            file_name=f"{a_ticker}_forecast_{prediction_days}d.csv",
            mime="text/csv",
        )

        # Summary statistics
        st.subheader("📊 Data Summary")
        col1, col2 = st.columns(2)

        with col1:
            st.write("**Price Statistics:**")
            st.dataframe(data[['Open', 'High', 'Low', 'Close', 'Volume']].describe())

        with col2:
            st.write("**Model Comparison:**")
            st.caption(f"📏 Naive baseline (predict previous close): MAPE {baseline['MAPE']:.2f}% · RMSE ${baseline['RMSE']:.2f} · R² {baseline['R²']:.4f}. If a model can't beat this baseline, it hasn't learned anything beyond 'tomorrow looks like today'.")
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
