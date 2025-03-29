import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
import time
import requests

# Import custom technical indicators
from technical_indicators import (
    calculate_moving_average,
    calculate_rsi,
    calculate_macd,
    calculate_bollinger_bands,
    calculate_stochastic_oscillator
)

# Set page configuration
st.set_page_config(
    page_title="Stock Market Dashboard",
    page_icon="📈",
    layout="wide"
)

# App title and description
st.title("📈 Real-Time Stock Market Dashboard")
st.markdown("""
This dashboard provides real-time analysis and visualization of stock market data.
* **Data source:** Yahoo Finance
* **Features:** Historical data analysis, performance comparison, and predictive trends
""")

# Sidebar for user inputs
st.sidebar.header("User Input Parameters")

# Select ticker symbols
default_tickers = ["AAPL", "MSFT", "GOOG", "AMZN", "META"]
selected_tickers = st.sidebar.multiselect("Select stocks to analyze:", default_tickers + ["TSLA", "NVDA", "BRK-B", "V", "JNJ", "WMT", "JPM", "PG", "UNH", "HD"], default=default_tickers[:3])

# Date range selection
today = datetime.today()
default_start_date = today - timedelta(days=365)  # 1 year ago
start_date = st.sidebar.date_input("Start date", default_start_date)
end_date = st.sidebar.date_input("End date", today)

# Analysis timeframe
timeframe = st.sidebar.select_slider(
    "Analysis timeframe",
    options=["7D", "1M", "3M", "6M", "YTD", "1Y", "3Y", "5Y", "MAX"],
    value="1Y"
)

# Technical indicators
technical_indicators = st.sidebar.multiselect(
    "Technical Indicators",
    ["Moving Average", "RSI", "MACD", "Bollinger Bands", "Stochastic Oscillator"],
    ["Moving Average"]
)

# Function to load stock data with retries
@st.cache_data(ttl=3600)
def load_data(tickers, start, end, max_retries=3, retry_delay=2):
    for attempt in range(max_retries):
        try:
            data = yf.download(tickers, start=start, end=end)
            return data
        except Exception as e:
            if attempt < max_retries - 1:
                st.warning(f"Attempt {attempt + 1} failed, retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
            else:
                st.error(f"Failed to download data after {max_retries} attempts: {str(e)}")
                return pd.DataFrame()  # Return empty DataFrame on failure

# Function to safely get ticker info with caching
@st.cache_data(ttl=3600)
def get_ticker_info(ticker, max_retries=3, retry_delay=2):
    for attempt in range(max_retries):
        try:
            ticker_obj = yf.Ticker(ticker)
            info = ticker_obj.info
            return info
        except requests.exceptions.HTTPError as e:
            if "429" in str(e) and attempt < max_retries - 1:
                st.warning(f"Rate limit hit for {ticker}, retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
            elif attempt < max_retries - 1:
                st.warning(f"HTTP error for {ticker}, retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                retry_delay *= 2
            else:
                st.error(f"Failed to get info for {ticker} after {max_retries} attempts: {str(e)}")
                # Return minimal info to prevent further errors
                return {"longName": ticker, "sector": "N/A", "industry": "N/A"}
        except Exception as e:
            st.error(f"Error retrieving info for {ticker}: {str(e)}")
            return {"longName": ticker, "sector": "N/A", "industry": "N/A"}

# Display loading message
if selected_tickers:
    with st.spinner('Loading stock data...'):
        # Load stock data
        data = load_data(selected_tickers, start_date, end_date)
        
        # Check if data is empty
        if data.empty:
            st.error("Failed to load stock data. Please try again later or select different stocks.")
            st.stop()
        
        # Main dashboard content
        st.header("Stock Performance Analysis")
        
        # Create tabs for different analyses
        tab1, tab2, tab3, tab4 = st.tabs(["Price Overview", "Performance Comparison", "Technical Analysis", "Predictive Trends"])
        
        with tab1:
            st.subheader("Historical Stock Prices")
            
            # Get closing prices for selected tickers
            if len(selected_tickers) == 1:
                closing_prices = data['Close']
                closing_prices = pd.DataFrame(closing_prices).rename(columns={'Close': selected_tickers[0]})
            else:
                closing_prices = data['Close']
            
            # Plot stock prices
            fig = px.line(
                closing_prices,
                title='Closing Prices of Selected Stocks',
                labels={'value': 'Price (USD)', 'variable': 'Stock', 'date': 'Date'}
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
            
            # Display recent stock data in a table
            st.subheader("Recent Stock Data")
            if len(selected_tickers) == 1:
                recent_data = data.tail(10)
            else:
                recent_data = data['Close'].tail(10)
            st.dataframe(recent_data, use_container_width=True)
            
            # Display stock info with improved error handling
            if len(selected_tickers) <= 3:  # Limit to prevent API overload
                st.subheader("Company Information")
                for i, ticker in enumerate(selected_tickers):
                    # Add delay between API calls to prevent rate limiting
                    if i > 0:
                        time.sleep(1)
                    
                    stock_info = get_ticker_info(ticker)
                    col1, col2 = st.columns([1, 3])
                    
                    with col1:
                        if 'logo_url' in stock_info and stock_info['logo_url']:
                            try:
                                st.image(stock_info['logo_url'], width=100)
                            except:
                                st.write(f"**{ticker}**")
                        else:
                            st.write(f"**{ticker}**")
                    
                    with col2:
                        company_name = stock_info.get('longName', ticker)
                        sector = stock_info.get('sector', 'N/A')
                        industry = stock_info.get('industry', 'N/A')
                        
                        st.write(f"**{company_name}**")
                        st.write(f"Sector: {sector} | Industry: {industry}")
                        
                        if 'longBusinessSummary' in stock_info and stock_info['longBusinessSummary']:
                            with st.expander("Business Summary"):
                                st.write(stock_info['longBusinessSummary'])
                    
                    st.markdown("---")
            
        with tab2:
            st.subheader("Comparative Performance")
            
            # Calculate percentage change from the start
            if len(selected_tickers) == 1:
                perf_data = ((data['Close'] / data['Close'].iloc[0]) - 1) * 100
                perf_data = pd.DataFrame(perf_data).rename(columns={'Close': selected_tickers[0]})
            else:
                perf_data = ((data['Close'] / data['Close'].iloc[0]) - 1) * 100
            
            # Plot performance comparison
            fig = px.line(
                perf_data,
                title='Percentage Change Over Time',
                labels={'value': 'Change (%)', 'variable': 'Stock', 'date': 'Date'}
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
            
            # Display performance metrics in two columns
            st.subheader("Performance Metrics")
            
            col1, col2, col3 = st.columns(3)
            
            for i, ticker in enumerate(selected_tickers):
                col = [col1, col2, col3][i % 3]
                with col:
                    st.subheader(ticker)
                    
                    # For single ticker or multiple tickers
                    if len(selected_tickers) == 1:
                        close_data = data['Close']
                        volume_data = data['Volume'] if 'Volume' in data else None
                    else:
                        close_data = data['Close'][ticker]
                        volume_data = data['Volume'][ticker] if 'Volume' in data else None
                    
                    # Calculate metrics
                    current_price = close_data.iloc[-1]
                    prev_day_price = close_data.iloc[-2]
                    daily_change = ((current_price / prev_day_price) - 1) * 100
                    
                    week_change = ((current_price / close_data.iloc[-5 if len(close_data) > 5 else 0]) - 1) * 100
                    month_change = ((current_price / close_data.iloc[-20 if len(close_data) > 20 else 0]) - 1) * 100
                    
                    # Display metrics
                    st.metric("Current Price", f"${current_price:.2f}", f"{daily_change:.2f}%")
                    st.metric("Weekly Change", f"{week_change:.2f}%")
                    st.metric("Monthly Change", f"{month_change:.2f}%")
                    
                    if volume_data is not None:
                        recent_vol = volume_data.tail(30).mean()
                        st.metric("Avg. 30-Day Volume", f"{int(recent_vol):,}")
            
        with tab3:
            st.subheader("Technical Analysis")
            
            selected_stock = st.selectbox("Select a stock for technical analysis", selected_tickers)
            
            # Get data for the selected stock
            if len(selected_tickers) == 1:
                stock_data = data
            else:
                stock_data = pd.DataFrame({
                    'Open': data['Open'][selected_stock],
                    'High': data['High'][selected_stock],
                    'Low': data['Low'][selected_stock],
                    'Close': data['Close'][selected_stock],
                    'Volume': data['Volume'][selected_stock] if 'Volume' in data else pd.Series(dtype='float64')
                })
            
            # Create tabs for different technical indicators
            tech_tab1, tech_tab2, tech_tab3, tech_tab4 = st.tabs(["Price Charts", "Oscillators", "Moving Averages", "Patterns"])
            
            with tech_tab1:
                # Candlestick chart
                st.subheader(f"{selected_stock} Candlestick Chart")
                fig = go.Figure(data=[go.Candlestick(
                    x=stock_data.index,
                    open=stock_data['Open'],
                    high=stock_data['High'],
                    low=stock_data['Low'],
                    close=stock_data['Close'],
                    name=selected_stock
                )])
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
                
                # Volume chart
                if 'Volume' in stock_data.columns:
                    st.subheader(f"{selected_stock} Volume")
                    fig = px.bar(
                        stock_data,
                        x=stock_data.index,
                        y='Volume',
                        title=f"{selected_stock} Trading Volume"
                    )
                    fig.update_layout(height=300)
                    st.plotly_chart(fig, use_container_width=True)
            
            with tech_tab2:
                # RSI
                if "RSI" in technical_indicators:
                    st.subheader("Relative Strength Index (RSI)")
                    rsi_period = st.slider("RSI Period", 5, 30, 14, key="rsi_period")
                    rsi = calculate_rsi(stock_data['Close'], window=rsi_period)
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=stock_data.index, y=rsi, mode='lines', name='RSI'))
                    
                    # Add overbought/oversold lines
                    fig.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought")
                    fig.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold")
                    
                    fig.update_layout(
                        title=f"{selected_stock} RSI ({rsi_period})",
                        yaxis_title="RSI Value",
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Stochastic Oscillator
                if "Stochastic Oscillator" in technical_indicators:
                    st.subheader("Stochastic Oscillator")
                    k_period = st.slider("K Period", 5, 30, 14, key="k_period")
                    d_period = st.slider("D Period", 3, 15, 3, key="d_period")
                    
                    stoch = calculate_stochastic_oscillator(stock_data, k_period=k_period, d_period=d_period)
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=stock_data.index, y=stoch['K'], mode='lines', name='%K'))
                    fig.add_trace(go.Scatter(x=stock_data.index, y=stoch['D'], mode='lines', name='%D'))
                    
                    # Add overbought/oversold lines
                    fig.add_hline(y=80, line_dash="dash", line_color="red", annotation_text="Overbought")
                    fig.add_hline(y=20, line_dash="dash", line_color="green", annotation_text="Oversold")
                    
                    fig.update_layout(
                        title=f"{selected_stock} Stochastic Oscillator ({k_period}, {d_period})",
                        yaxis_title="Value",
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # MACD
                if "MACD" in technical_indicators:
                    st.subheader("Moving Average Convergence Divergence (MACD)")
                    fast_period = st.slider("Fast Period", 5, 20, 12, key="fast_period")
                    slow_period = st.slider("Slow Period", 10, 40, 26, key="slow_period")
                    signal_period = st.slider("Signal Period", 5, 15, 9, key="signal_period")
                    
                    macd_data = calculate_macd(stock_data['Close'], fast_period=fast_period, slow_period=slow_period, signal_period=signal_period)
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=stock_data.index, y=macd_data['MACD_Line'], mode='lines', name='MACD Line'))
                    fig.add_trace(go.Scatter(x=stock_data.index, y=macd_data['Signal_Line'], mode='lines', name='Signal Line'))
                    
                    # Add MACD histogram
                    fig.add_trace(go.Bar(
                        x=stock_data.index, 
                        y=macd_data['Histogram'],
                        name='Histogram',
                        marker=dict(
                            color=macd_data['Histogram'].apply(lambda x: 'green' if x > 0 else 'red'),
                            line=dict(color='rgba(0,0,0,0)', width=0)
                        )
                    ))
                    
                    fig.update_layout(
                        title=f"{selected_stock} MACD ({fast_period}, {slow_period}, {signal_period})",
                        yaxis_title="Value",
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            with tech_tab3:
                # Moving Average
                if "Moving Average" in technical_indicators:
                    st.subheader("Moving Averages")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        ma1_window = st.number_input("MA 1 Period", 5, 200, 20, step=5)
                    with col2:
                        ma2_window = st.number_input("MA 2 Period", 5, 200, 50, step=5)
                    with col3:
                        ma3_window = st.number_input("MA 3 Period", 5, 200, 200, step=5)
                    
                    ma1 = calculate_moving_average(stock_data['Close'], window=ma1_window)
                    ma2 = calculate_moving_average(stock_data['Close'], window=ma2_window)
                    ma3 = calculate_moving_average(stock_data['Close'], window=ma3_window)
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Close'], mode='lines', name='Close Price'))
                    fig.add_trace(go.Scatter(x=stock_data.index, y=ma1, mode='lines', name=f'{ma1_window}-day MA', line=dict(color='orange')))
                    fig.add_trace(go.Scatter(x=stock_data.index, y=ma2, mode='lines', name=f'{ma2_window}-day MA', line=dict(color='green')))
                    fig.add_trace(go.Scatter(x=stock_data.index, y=ma3, mode='lines', name=f'{ma3_window}-day MA', line=dict(color='purple')))
                    
                    fig.update_layout(
                        title=f"{selected_stock} Price with Moving Averages",
                        xaxis_title="Date",
                        yaxis_title="Price (USD)",
                        height=500
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Bollinger Bands
                if "Bollinger Bands" in technical_indicators:
                    st.subheader("Bollinger Bands")
                    bb_window = st.slider("Bollinger Bands Period", 5, 50, 20, key="bb_period")
                    bb_std = st.slider("Standard Deviation", 1.0, 4.0, 2.0, step=0.5, key="bb_std")
                    
                    bollinger = calculate_bollinger_bands(stock_data['Close'], window=bb_window, num_std=bb_std)
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Close'], mode='lines', name='Close Price'))
                    fig.add_trace(go.Scatter(x=stock_data.index, y=bollinger['MA'], mode='lines', name='Middle Band (MA)', line=dict(color='orange')))
                    fig.add_trace(go.Scatter(x=stock_data.index, y=bollinger['Upper_Band'], mode='lines', name='Upper Band', line=dict(color='green')))
                    fig.add_trace(go.Scatter(x=stock_data.index, y=bollinger['Lower_Band'], mode='lines', name='Lower Band', line=dict(color='red')))
                    
                    fig.update_layout(
                        title=f"{selected_stock} Bollinger Bands ({bb_window}, {bb_std})",
                        xaxis_title="Date",
                        yaxis_title="Price (USD)",
                        height=500
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            with tech_tab4:
                st.subheader("Pattern Recognition")
                st.info("This feature will be available in the next update.")
                
                # Placeholder for future pattern recognition features
                st.image("https://www.investopedia.com/thmb/0MxQkTAKITgG6fN-Ih1t_bJcQFc=/1500x0/filters:no_upscale():max_bytes(150000):strip_icc()/dotdash_Final_Common_Chart_Patterns_in_Technical_Analysis_Aug_2020-01-a64b6c6bd0b24166bad4662699515fa5.jpg", caption="Common chart patterns in technical analysis")
        
        with tab4:
            st.subheader("Predictive Analysis")
            
            selected_stock_pred = st.selectbox("Select a stock for prediction", selected_tickers, key="pred_stock")
            prediction_days = st.slider("Prediction Days", 7, 90, 30)
            
            # Get data for the selected stock
            if len(selected_tickers) == 1:
                stock_data_pred = data['Close']
                stock_data_pred = pd.DataFrame(stock_data_pred).rename(columns={'Close': selected_tickers[0]})
            else:
                stock_data_pred = data['Close'][selected_stock_pred]
                stock_data_pred = pd.DataFrame(stock_data_pred).rename(columns={selected_stock_pred: 'Close'})
            
            # Prepare data for linear regression
            stock_data_pred = stock_data_pred.reset_index()
            stock_data_pred['Date_Numeric'] = (stock_data_pred['Date'] - stock_data_pred['Date'].min()).dt.days
            
            # Linear regression model
            X = stock_data_pred[['Date_Numeric']]
            y = stock_data_pred['Close']
            model = LinearRegression()
            model.fit(X, y)
            
            # Predict future values
            last_date = stock_data_pred['Date'].max()
            future_dates = [last_date + timedelta(days=i) for i in range(1, prediction_days + 1)]
            future_dates_numeric = [(date - stock_data_pred['Date'].min()).days for date in future_dates]
            
            future_predictions = model.predict(np.array(future_dates_numeric).reshape(-1, 1))
            
            # Combine historical and predicted data
            historical_data = stock_data_pred[['Date', 'Close']].rename(columns={'Close': 'Historical'})
            
            prediction_df = pd.DataFrame({
                'Date': future_dates,
                'Predicted': future_predictions
            })
            
            # Prediction metrics
            st.subheader("Prediction Metrics")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                last_price = historical_data['Historical'].iloc[-1]
                pred_price = prediction_df['Predicted'].iloc[-1]
                price_change = ((pred_price / last_price) - 1) * 100
                st.metric(
                    "Predicted End Price", 
                    f"${pred_price:.2f}", 
                    f"{price_change:.2f}%"
                )
            
            with col2:
                min_pred = prediction_df['Predicted'].min()
                max_pred = prediction_df['Predicted'].max()
                st.metric("Predicted Range", f"${min_pred:.2f} - ${max_pred:.2f}")
            
            with col3:
                trend = "Upward" if price_change > 0 else "Downward" if price_change < 0 else "Stable"
                st.metric("Predicted Trend", trend)
            
            # Plot data
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=historical_data['Date'], 
                y=historical_data['Historical'],
                mode='lines',
                name='Historical'
            ))
            fig.add_trace(go.Scatter(
                x=prediction_df['Date'],
                y=prediction_df['Predicted'],
                mode='lines',
                name='Predicted',
                line=dict(dash='dash', color='red')
            ))
            
            # Add confidence interval (simplified)
            if st.checkbox("Show confidence interval", value=True):
                # Create a simple confidence interval (not statistically accurate, just for visualization)
                std_dev = np.std(y)
                upper_bound = prediction_df['Predicted'] + std_dev
                lower_bound = prediction_df['Predicted'] - std_dev
                
                fig.add_trace(go.Scatter(
                    x=prediction_df['Date'],
                    y=upper_bound,
                    fill=None,
                    mode='lines',
                    line=dict(color='rgba(255, 0, 0, 0.1)'),
                    name='Upper Bound'
                ))
                fig.add_trace(go.Scatter(
                    x=prediction_df['Date'],
                    y=lower_bound,
                    fill='tonexty',
                    mode='lines',
                    line=dict(color='rgba(255, 0, 0, 0.1)'),
                    name='Lower Bound'
                ))
            
            fig.update_layout(
                title=f"{selected_stock_pred} Price Prediction ({prediction_days} days)",
                xaxis_title="Date",
                yaxis_title="Price (USD)",
                height=500,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Show disclaimer
            st.info('Note: This is a simplified linear prediction model for demonstration purposes only. Real stock market prediction requires more sophisticated techniques and should not be used as financial advice.')

else:
    st.warning("Please select at least one stock ticker to analyze.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>Built with ❤️ using Streamlit and Yahoo Finance data</p>
    <p>© 2023 Stock Market Dashboard</p>
</div>
""", unsafe_allow_html=True) 