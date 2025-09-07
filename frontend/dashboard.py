"""
Streamlit Dashboard for Unified Multimodal Transformer
Interactive financial forecasting dashboard with explainability
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import requests
import yfinance as yf
from datetime import datetime, timedelta
import time

# Page config
st.set_page_config(
    page_title="Financial Forecasting Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API base URL
API_BASE_URL = "http://localhost:8000"

def check_api_health():
    """Check if API is running"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def get_prediction(symbol, days_back=60, forecast_days=5):
    """Get prediction from API"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/predict",
            json={
                "symbol": symbol,
                "days_back": days_back,
                "forecast_days": forecast_days
            },
            timeout=30
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.status_code}")
            return None
            
    except Exception as e:
        st.error(f"Connection error: {str(e)}")
        return None

def get_explanation(symbol):
    """Get explanation from API"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/explain",
            json={"symbol": symbol},
            timeout=30
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            return None
            
    except Exception as e:
        st.error(f"Explanation error: {str(e)}")
        return None

def get_market_overview():
    """Get market overview from API"""
    try:
        response = requests.get(f"{API_BASE_URL}/market-overview", timeout=15)
        
        if response.status_code == 200:
            return response.json()
        else:
            return None
            
    except Exception as e:
        st.error(f"Market overview error: {str(e)}")
        return None

def get_historical_data(symbol, period="1y"):
    """Get historical data for visualization"""
    try:
        ticker = yf.Ticker(symbol)
        data = ticker.history(period=period)
        return data
    except Exception as e:
        st.error(f"Error fetching historical data: {str(e)}")
        return None

def plot_price_forecast(historical_data, prediction_data):
    """Create price forecast visualization"""
    
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Price Forecast', 'Technical Indicators'),
        vertical_spacing=0.1,
        row_heights=[0.7, 0.3]
    )
    
    # Historical prices
    fig.add_trace(
        go.Scatter(
            x=historical_data.index,
            y=historical_data['Close'],
            mode='lines',
            name='Historical Price',
            line=dict(color='blue', width=2)
        ),
        row=1, col=1
    )
    
    # Current price point
    current_price = prediction_data['current_price']
    last_date = historical_data.index[-1]
    
    fig.add_trace(
        go.Scatter(
            x=[last_date],
            y=[current_price],
            mode='markers',
            name='Current Price',
            marker=dict(color='red', size=10)
        ),
        row=1, col=1
    )
    
    # Forecast
    forecast_dates = pd.to_datetime(prediction_data['forecast_dates'])
    forecast_prices = prediction_data['forecast']
    
    # Connect current to forecast
    extended_dates = [last_date] + forecast_dates.tolist()
    extended_prices = [current_price] + forecast_prices
    
    fig.add_trace(
        go.Scatter(
            x=extended_dates,
            y=extended_prices,
            mode='lines+markers',
            name='Forecast',
            line=dict(color='green', width=3, dash='dash'),
            marker=dict(size=8)
        ),
        row=1, col=1
    )
    
    # Confidence band (simplified)
    volatility = prediction_data['technical_indicators']['volatility']
    upper_band = [p * (1 + volatility) for p in forecast_prices]
    lower_band = [p * (1 - volatility) for p in forecast_prices]
    
    fig.add_trace(
        go.Scatter(
            x=forecast_dates,
            y=upper_band,
            mode='lines',
            name='Upper Confidence',
            line=dict(color='lightgreen', width=1),
            showlegend=False
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=forecast_dates,
            y=lower_band,
            mode='lines',
            name='Lower Confidence',
            line=dict(color='lightgreen', width=1),
            fill='tonexty',
            fillcolor='rgba(0,255,0,0.1)',
            showlegend=False
        ),
        row=1, col=1
    )
    
    # Technical indicators
    fig.add_trace(
        go.Scatter(
            x=historical_data.index[-30:],
            y=historical_data['Volume'][-30:],
            mode='lines',
            name='Volume',
            line=dict(color='orange')
        ),
        row=2, col=1
    )
    
    fig.update_layout(
        title=f"Financial Forecast for {prediction_data['symbol']}",
        xaxis_title="Date",
        yaxis_title="Price",
        height=600,
        showlegend=True
    )
    
    return fig

def plot_feature_importance(explanation_data):
    """Plot feature importance"""
    
    features = list(explanation_data['feature_importance'].keys())
    importance = list(explanation_data['feature_importance'].values())
    
    fig = go.Figure(data=[
        go.Bar(
            x=importance,
            y=features,
            orientation='h',
            marker_color='skyblue'
        )
    ])
    
    fig.update_layout(
        title="Feature Importance Analysis",
        xaxis_title="Importance Score",
        yaxis_title="Features",
        height=400
    )
    
    return fig

def main():
    """Main dashboard application"""
    
    # Header
    st.title("🚀 Unified Multimodal Financial Forecasting")
    st.markdown("*Cross-market AI predictions with explainability*")
    
    # Check API status
    api_status = check_api_health()
    
    if api_status:
        st.success("✅ API Connected")
    else:
        st.error("❌ API Disconnected - Start the API server first")
        st.code("cd api && python main.py")
        return
    
    # Sidebar
    st.sidebar.header("Configuration")
    
    # Symbol selection
    symbol_categories = {
        "US Stocks": ["AAPL", "GOOGL", "MSFT", "TSLA", "NVDA", "AMZN"],
        "Indian Stocks": ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS"],
        "Brazilian Stocks": ["PETR4.SA", "VALE3.SA", "ITUB4.SA"],
        "Crypto": ["BTC-USD", "ETH-USD"],
        "Indices": ["^GSPC", "^NSEI", "^BVSP"]
    }
    
    selected_category = st.sidebar.selectbox("Market Category", list(symbol_categories.keys()))
    selected_symbol = st.sidebar.selectbox("Symbol", symbol_categories[selected_category])
    
    # Prediction parameters
    forecast_days = st.sidebar.slider("Forecast Days", 1, 10, 5)
    lookback_days = st.sidebar.slider("Lookback Days", 30, 120, 60)
    
    # Auto-refresh
    auto_refresh = st.sidebar.checkbox("Auto Refresh (30s)")
    
    if auto_refresh:
        time.sleep(30)
        st.experimental_rerun()
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header(f"📈 Forecast for {selected_symbol}")
        
        # Get prediction
        with st.spinner("Generating AI prediction..."):
            prediction_data = get_prediction(selected_symbol, lookback_days, forecast_days)
        
        if prediction_data:
            # Display key metrics
            metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
            
            with metric_col1:
                st.metric(
                    "Current Price",
                    f"${prediction_data['current_price']:.2f}"
                )
            
            with metric_col2:
                forecast_change = (prediction_data['forecast'][-1] - prediction_data['current_price']) / prediction_data['current_price'] * 100
                st.metric(
                    f"{forecast_days}D Forecast",
                    f"${prediction_data['forecast'][-1]:.2f}",
                    f"{forecast_change:+.2f}%"
                )
            
            with metric_col3:
                st.metric(
                    "Signal",
                    prediction_data['directional_signal'],
                    f"Confidence: {prediction_data['confidence']:.1%}"
                )
            
            with metric_col4:
                anomaly_color = "red" if prediction_data['anomaly_score'] > 0.7 else "green"
                st.metric(
                    "Risk Score",
                    f"{prediction_data['anomaly_score']:.3f}",
                    help="Higher values indicate increased market risk"
                )
            
            # Get historical data for visualization
            historical_data = get_historical_data(selected_symbol, "3mo")
            
            if historical_data is not None:
                # Plot forecast
                forecast_fig = plot_price_forecast(historical_data, prediction_data)
                st.plotly_chart(forecast_fig, use_container_width=True)
            
            # Technical indicators table
            st.subheader("📊 Technical Indicators")
            
            tech_df = pd.DataFrame([
                {"Indicator": "RSI", "Value": f"{prediction_data['technical_indicators']['rsi']:.2f}"},
                {"Indicator": "MACD", "Value": f"{prediction_data['technical_indicators']['macd']:.4f}"},
                {"Indicator": "Volatility", "Value": f"{prediction_data['technical_indicators']['volatility']:.4f}"},
                {"Indicator": "SMA 20", "Value": f"${prediction_data['technical_indicators']['sma_20']:.2f}"},
                {"Indicator": "Volume Ratio", "Value": f"{prediction_data['technical_indicators']['volume_ratio']:.2f}"}
            ])
            
            st.dataframe(tech_df, use_container_width=True)
        
        else:
            st.error("Failed to get prediction. Please try again.")
    
    with col2:
        st.header("🔍 AI Explainability")
        
        if prediction_data:
            # Get explanation
            with st.spinner("Analyzing prediction factors..."):
                explanation_data = get_explanation(selected_symbol)
            
            if explanation_data:
                # Feature importance plot
                importance_fig = plot_feature_importance(explanation_data)
                st.plotly_chart(importance_fig, use_container_width=True)
                
                # Explanation text
                st.subheader("💡 AI Reasoning")
                st.write(explanation_data['explanation_text'])
                
                # Attention heatmap (simplified)
                st.subheader("🎯 Attention Pattern")
                attention_data = explanation_data['attention_weights'][0]
                
                # Show last 20 time steps
                attention_df = pd.DataFrame({
                    'Time Step': range(len(attention_data[-20:])),
                    'Attention Weight': attention_data[-20:]
                })
                
                attention_fig = px.line(
                    attention_df, 
                    x='Time Step', 
                    y='Attention Weight',
                    title="Model Attention Over Time"
                )
                st.plotly_chart(attention_fig, use_container_width=True)
            
            else:
                st.warning("Explanation not available")
    
    # Market Overview Section
    st.header("🌍 Global Market Overview")
    
    market_data = get_market_overview()
    
    if market_data:
        st.write(f"**Market Sentiment**: {market_data['market_sentiment']}")
        st.write(f"**Last Updated**: {market_data['timestamp'][:19]}")
        
        # Market performance
        market_cols = st.columns(len(market_data['markets']))
        
        for i, (market, data) in enumerate(market_data['markets'].items()):
            with market_cols[i]:
                st.subheader(f"{market} Markets")
                
                for symbol, info in data.items():
                    change_color = "green" if info['change'] > 0 else "red"
                    st.write(f"**{symbol}**: ${info['price']:.2f} ({info['change']:+.2f}%)")
        
        # Top predictions
        if market_data['top_predictions']:
            st.subheader("🎯 Top AI Predictions")
            
            pred_df = pd.DataFrame(market_data['top_predictions'])
            pred_df['predicted_change_pct'] = (pred_df['predicted_change'] / pred_df['current_price'] * 100).round(2)
            
            st.dataframe(
                pred_df[['symbol', 'current_price', 'predicted_change_pct']].rename(columns={
                    'symbol': 'Symbol',
                    'current_price': 'Current Price',
                    'predicted_change_pct': 'Predicted Change %'
                }),
                use_container_width=True
            )
    
    # Footer
    st.markdown("---")
    st.markdown(
        "**Unified Multimodal Transformer** | "
        "Cross-market financial forecasting with explainable AI | "
        f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )

if __name__ == "__main__":
    main()