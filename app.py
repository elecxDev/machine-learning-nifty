"""
Simple Stock Analysis App for Presentation
Input: Stock symbol (e.g., AAPL)
Output: BUY/SELL recommendation with risk analysis
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import torch
import torch.nn as nn

# Simple ML Model for Stock Prediction
class SimpleStockPredictor(nn.Module):
    def __init__(self, input_size=10):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.network(x)

def get_stock_data(symbol, period="1y"):
    """Get stock data from Yahoo Finance"""
    try:
        ticker = yf.Ticker(symbol)
        data = ticker.history(period=period)
        info = ticker.info
        return data, info
    except:
        return None, None

def calculate_technical_indicators(data):
    """Calculate technical indicators"""
    # Moving averages
    data['MA_20'] = data['Close'].rolling(20).mean()
    data['MA_50'] = data['Close'].rolling(50).mean()
    
    # RSI
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))
    
    # Volatility
    data['Volatility'] = data['Close'].pct_change().rolling(20).std()
    
    # Price change
    data['Price_Change'] = data['Close'].pct_change()
    
    return data

def analyze_stock(symbol):
    """Main stock analysis function"""
    
    # Get data
    data, info = get_stock_data(symbol)
    
    if data is None:
        return None
    
    # Calculate indicators
    data = calculate_technical_indicators(data)
    
    # Get latest values
    latest = data.iloc[-1]
    current_price = latest['Close']
    
    # Simple scoring system
    score = 0
    reasons = []
    
    # Price vs Moving Averages
    if current_price > latest['MA_20']:
        score += 1
        reasons.append("Price above 20-day MA (Bullish)")
    else:
        score -= 1
        reasons.append("Price below 20-day MA (Bearish)")
    
    if current_price > latest['MA_50']:
        score += 1
        reasons.append("Price above 50-day MA (Bullish)")
    else:
        score -= 1
        reasons.append("Price below 50-day MA (Bearish)")
    
    # RSI Analysis
    rsi = latest['RSI']
    if rsi < 30:
        score += 2
        reasons.append(f"RSI {rsi:.1f} - Oversold (Strong Buy)")
    elif rsi > 70:
        score -= 2
        reasons.append(f"RSI {rsi:.1f} - Overbought (Strong Sell)")
    elif rsi < 50:
        score += 1
        reasons.append(f"RSI {rsi:.1f} - Below 50 (Bullish)")
    else:
        score -= 1
        reasons.append(f"RSI {rsi:.1f} - Above 50 (Bearish)")
    
    # Volatility Risk
    volatility = latest['Volatility']
    if volatility > 0.03:
        risk_level = "HIGH"
        risk_color = "red"
    elif volatility > 0.02:
        risk_level = "MEDIUM"
        risk_color = "orange"
    else:
        risk_level = "LOW"
        risk_color = "green"
    
    # Final recommendation
    if score >= 2:
        recommendation = "STRONG BUY"
        rec_color = "green"
    elif score >= 1:
        recommendation = "BUY"
        rec_color = "lightgreen"
    elif score <= -2:
        recommendation = "STRONG SELL"
        rec_color = "red"
    elif score <= -1:
        recommendation = "SELL"
        rec_color = "lightcoral"
    else:
        recommendation = "HOLD"
        rec_color = "yellow"
    
    # Calculate position size based on risk
    account_balance = 10000  # Assume $10k account
    if risk_level == "LOW":
        position_size = account_balance * 0.1  # 10% of account
    elif risk_level == "MEDIUM":
        position_size = account_balance * 0.05  # 5% of account
    else:
        position_size = account_balance * 0.02  # 2% of account
    
    shares_to_buy = int(position_size / current_price)
    
    return {
        'data': data,
        'current_price': current_price,
        'recommendation': recommendation,
        'rec_color': rec_color,
        'score': score,
        'reasons': reasons,
        'risk_level': risk_level,
        'risk_color': risk_color,
        'volatility': volatility,
        'rsi': rsi,
        'position_size': position_size,
        'shares_to_buy': shares_to_buy,
        'info': info
    }

def create_price_chart(data, symbol):
    """Create price chart with indicators"""
    
    fig = go.Figure()
    
    # Price line
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['Close'],
        mode='lines',
        name='Price',
        line=dict(color='blue', width=2)
    ))
    
    # Moving averages
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['MA_20'],
        mode='lines',
        name='MA 20',
        line=dict(color='orange', width=1)
    ))
    
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['MA_50'],
        mode='lines',
        name='MA 50',
        line=dict(color='red', width=1)
    ))
    
    fig.update_layout(
        title=f'{symbol} Stock Price Analysis',
        xaxis_title='Date',
        yaxis_title='Price ($)',
        height=500
    )
    
    return fig

def main():
    """Main Streamlit app"""
    
    st.set_page_config(
        page_title="Stock Analysis App",
        page_icon="📈",
        layout="wide"
    )
    
    st.title("📈 AI Stock Analysis & Recommendation")
    st.markdown("*Enter a stock symbol to get BUY/SELL recommendation with risk analysis*")
    
    # Input section
    col1, col2 = st.columns([1, 3])
    
    with col1:
        symbol = st.text_input("Stock Symbol", value="AAPL", help="Enter stock symbol (e.g., AAPL, GOOGL, TSLA)")
        analyze_button = st.button("🔍 Analyze Stock", type="primary")
    
    if analyze_button and symbol:
        
        with st.spinner(f"Analyzing {symbol}..."):
            result = analyze_stock(symbol.upper())
        
        if result is None:
            st.error(f"❌ Could not fetch data for {symbol}. Please check the symbol.")
            return
        
        # Display results
        st.success(f"✅ Analysis complete for {symbol.upper()}")
        
        # Main metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Current Price",
                f"${result['current_price']:.2f}"
            )
        
        with col2:
            st.markdown(f"""
            <div style='padding: 10px; border-radius: 5px; background-color: {result['rec_color']}; text-align: center;'>
                <h3 style='margin: 0; color: white;'>{result['recommendation']}</h3>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div style='padding: 10px; border-radius: 5px; background-color: {result['risk_color']}; text-align: center;'>
                <h3 style='margin: 0; color: white;'>RISK: {result['risk_level']}</h3>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.metric(
                "RSI",
                f"{result['rsi']:.1f}",
                help="Relative Strength Index (30=Oversold, 70=Overbought)"
            )
        
        # Investment recommendation
        st.subheader("💰 Investment Recommendation")
        
        if result['recommendation'] in ['BUY', 'STRONG BUY']:
            st.success(f"""
            **Recommendation: {result['recommendation']}**
            
            - **Suggested Position Size**: ${result['position_size']:,.0f} ({result['position_size']/10000*100:.0f}% of $10k account)
            - **Shares to Buy**: {result['shares_to_buy']} shares
            - **Total Investment**: ${result['shares_to_buy'] * result['current_price']:,.0f}
            - **Risk Level**: {result['risk_level']}
            """)
        
        elif result['recommendation'] in ['SELL', 'STRONG SELL']:
            st.error(f"""
            **Recommendation: {result['recommendation']}**
            
            - **Action**: Sell existing positions or avoid buying
            - **Risk Level**: {result['risk_level']}
            - **Volatility**: {result['volatility']*100:.1f}%
            """)
        
        else:
            st.warning(f"""
            **Recommendation: HOLD**
            
            - **Action**: Wait for better entry/exit points
            - **Risk Level**: {result['risk_level']}
            """)
        
        # Analysis reasons
        st.subheader("🔍 Analysis Details")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Technical Analysis:**")
            for reason in result['reasons']:
                if "Bullish" in reason or "Buy" in reason:
                    st.success(f"✅ {reason}")
                elif "Bearish" in reason or "Sell" in reason:
                    st.error(f"❌ {reason}")
                else:
                    st.info(f"ℹ️ {reason}")
        
        with col2:
            st.write("**Key Metrics:**")
            st.write(f"• **Volatility**: {result['volatility']*100:.1f}%")
            st.write(f"• **Analysis Score**: {result['score']}/4")
            st.write(f"• **Risk Level**: {result['risk_level']}")
            
            if result['info']:
                market_cap = result['info'].get('marketCap', 0)
                if market_cap:
                    st.write(f"• **Market Cap**: ${market_cap/1e9:.1f}B")
        
        # Price chart
        st.subheader("📊 Price Chart & Technical Indicators")
        
        chart = create_price_chart(result['data'], symbol.upper())
        st.plotly_chart(chart, use_container_width=True)
        
        # Risk warning
        st.warning("""
        ⚠️ **Disclaimer**: This is a simplified analysis for educational purposes. 
        Always do your own research and consider consulting a financial advisor before making investment decisions.
        """)

if __name__ == "__main__":
    main()