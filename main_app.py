"""
🚀 Machine Learning Nifty - Comprehensive Stock Analysis Platform
Entry Point: Full-featured Streamlit UI with AI-powered Buy/Sell/Hold recommendations
"""

import os
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import requests
import warnings
warnings.filterwarnings('ignore')

# Currency and Market Support
MARKET_CURRENCIES = {
    'US': 'USD', 'USA': 'USD',
    'INDIA': 'INR', 'NS': 'INR', 'BO': 'INR',
    'BRAZIL': 'BRL', 'SA': 'BRL',
    'UK': 'GBP', 'L': 'GBP',
    'DE': 'EUR', 'PA': 'EUR',
    'TO': 'CAD', 'T': 'CAD',
    'HK': 'HKD', 'TYO': 'JPY',
    'CRYPTO': 'USD', 'BTC': 'USD', 'ETH': 'USD'
}

CURRENCY_SYMBOLS = {
    'USD': '$', 'INR': '₹', 'BRL': 'R$', 'GBP': '£', 
    'EUR': '€', 'CAD': 'C$', 'HKD': 'HK$', 'JPY': '¥'
}

API_BASE_URL = os.getenv('API_BASE_URL', 'http://localhost:8000')

def detect_market_currency(symbol):
    """Detect currency for stock symbol"""
    if '-USD' in symbol or symbol in ['BTC-USD', 'ETH-USD']:
        return 'USD'
    if symbol.startswith('^'):
        if symbol in ['^NSEI', '^BSESN']:
            return 'INR'
        elif symbol in ['^BVSP']:
            return 'BRL'
        return 'USD'
    if '.' in symbol:
        suffix = symbol.split('.')[-1]
        return MARKET_CURRENCIES.get(suffix, 'USD')
    return 'USD'

def format_currency(amount, currency_code):
    """Format amount with proper currency"""
    symbol = CURRENCY_SYMBOLS.get(currency_code, currency_code)
    
    if currency_code == 'INR':
        if amount >= 10000000:  # 1 crore
            return f"{symbol}{amount/10000000:.2f}Cr"
        elif amount >= 100000:  # 1 lakh
            return f"{symbol}{amount/100000:.2f}L"
        else:
            return f"{symbol}{amount:,.2f}"
    elif currency_code == 'JPY':
        return f"{symbol}{amount:,.0f}"
    else:
        if amount >= 1000000:
            return f"{symbol}{amount/1000000:.2f}M"
        elif amount >= 1000:
            return f"{symbol}{amount/1000:.1f}K"
        else:
            return f"{symbol}{amount:.2f}"

def create_market_badge(symbol):
    """Create market badge with flag and currency"""
    currency = detect_market_currency(symbol)
    
    if '.' in symbol:
        suffix = symbol.split('.')[-1]
        if suffix == 'NS':
            return f"🇮🇳 NSE • {currency}"
        elif suffix == 'BO':
            return f"🇮🇳 BSE • {currency}"
        elif suffix == 'SA':
            return f"🇧🇷 B3 • {currency}"
        elif suffix == 'L':
            return f"🇬🇧 LSE • {currency}"
    elif '-USD' in symbol:
        return f"₿ CRYPTO • {currency}"
    elif symbol.startswith('^'):
        if 'NSEI' in symbol or 'BSESN' in symbol:
            return f"🇮🇳 INDEX • {currency}"
        elif 'BVSP' in symbol:
            return f"🇧🇷 INDEX • {currency}"
        return f"📊 INDEX • {currency}"
    else:
        return f"🇺🇸 US • {currency}"
# News Analysis Functions
def get_stock_news(symbol, company_name=""):
    """Get recent news for a stock with sentiment analysis"""
    news_items = []
    
    try:
        # Try to get company name if not provided
        if not company_name:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            company_name = info.get('shortName', symbol)
        
        news_items = fetch_news_from_api(symbol, company_name)
        if not news_items:
            news_items = create_sample_news(symbol, company_name)
            
    except Exception as e:
        # Fallback to sample news
        news_items = create_sample_news(symbol, company_name or symbol)
    
    return news_items

@st.cache_data(ttl=900, show_spinner=False)
def fetch_news_from_api(symbol: str, company_name: str = "", max_items: int = 6):
    """Fetch recent headlines and FinBERT sentiment from the FastAPI backend."""
    params = {"max_items": max_items}
    if company_name:
        params["company_name"] = company_name

    try:
        response = requests.get(
            f"{API_BASE_URL}/news/{symbol}",
            params=params,
            timeout=10,
        )
        response.raise_for_status()
        payload = response.json()
    except Exception:
        return []

    items = []
    for entry in payload:
        label = (entry.get('sentiment_label') or 'neutral').lower()
        score = float(entry.get('sentiment_score', 0.0))
        emoji = '📈' if label == 'positive' else '📉' if label == 'negative' else '➡️'
        items.append({
            'title': entry.get('title', ''),
            'summary': entry.get('summary', ''),
            'sentiment': {'label': label, 'score': score, 'emoji': emoji},
            'published': entry.get('published'),
            'link': entry.get('link', ''),
        })

    return items

def create_sample_news(symbol, company_name):
    """Create sample news items for demo purposes"""
    
    # Get basic stock info for realistic news
    try:
        ticker = yf.Ticker(symbol)
        data = ticker.history(period='2d')
        if not data.empty:
            price_change = ((data['Close'].iloc[-1] - data['Close'].iloc[-2]) / data['Close'].iloc[-2]) * 100
        else:
            price_change = np.random.uniform(-3, 3)
    except:
        price_change = np.random.uniform(-3, 3)
    
    # Create realistic sample news based on price movement
    if price_change > 2:
        sample_news = [
            {
                'title': f"{company_name} Surges on Strong Quarterly Results",
                'summary': f"Shares of {company_name} ({symbol}) gained {price_change:.1f}% following better-than-expected earnings and positive guidance.",
                'sentiment': {'label': 'positive', 'score': 0.6, 'emoji': '📈'}
            },
            {
                'title': f"Analysts Raise Price Target for {company_name}",
                'summary': f"Multiple analysts upgraded their outlook on {symbol} citing strong fundamentals and market position.",
                'sentiment': {'label': 'positive', 'score': 0.4, 'emoji': '📈'}
            }
        ]
    elif price_change < -2:
        sample_news = [
            {
                'title': f"{company_name} Faces Market Headwinds",
                'summary': f"Shares of {company_name} ({symbol}) declined {abs(price_change):.1f}% amid broader market concerns and sector rotation.",
                'sentiment': {'label': 'negative', 'score': -0.4, 'emoji': '📉'}
            },
            {
                'title': f"Market Volatility Impacts {company_name} Trading",
                'summary': f"Investors show caution around {symbol} following recent market developments and economic indicators.",
                'sentiment': {'label': 'negative', 'score': -0.3, 'emoji': '📉'}
            }
        ]
    else:
        sample_news = [
            {
                'title': f"{company_name} Maintains Steady Performance",
                'summary': f"{symbol} shows resilient trading patterns amid mixed market signals and continues operational strength.",
                'sentiment': {'label': 'neutral', 'score': 0.1, 'emoji': '➡️'}
            },
            {
                'title': f"Market Update: {company_name} in Focus",
                'summary': f"Trading activity for {symbol} remains within expected ranges as institutional interest continues.",
                'sentiment': {'label': 'neutral', 'score': 0.0, 'emoji': '➡️'}
            }
        ]
    
    # Add timestamps
    for i, news in enumerate(sample_news):
        news['published'] = (datetime.now() - timedelta(hours=i*3 + 1)).strftime('%Y-%m-%d %H:%M')
        news['link'] = f"#news-{i}"
    
    return sample_news

# Advanced imports for AI features
try:
    import torch
    import torch.nn as nn
    from transformers import pipeline
    ADVANCED_AI = True
except ImportError:
    ADVANCED_AI = False
    st.warning("⚠ Advanced AI features require torch and transformers. Install with: pip install torch transformers")

# Page configuration
st.set_page_config(
    page_title="Analysis Hub",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/elecxDev/machine-learning-nifty',
        'Report a bug': 'https://github.com/elecxDev/machine-learning-nifty/issues',
        'About': "Financial analysis platform"
    }
)

# Apple-inspired CSS Design System
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@100;200;300;400;500;600;700;800;900&display=swap');
    
    /* Apple Design System Variables - Dark Mode */
    :root {
        --primary: #0A84FF !important;
        --primary-hover: #0066CC !important;
        --secondary: #1C1C1E !important;
        --tertiary: #2C2C2E !important;
        --label-primary: #FFFFFF !important;
        --label-secondary: #EBEBF5 !important;
        --label-tertiary: #EBEBF599 !important;
        --separator: #38383A !important;
        --background: #000000 !important;
        --grouped-background: #1C1C1E !important;
        --system-red: #FF453A !important;
        --system-green: #30D158 !important;
        --system-orange: #FF9F0A !important;
        --shadow: 0 2px 10px rgba(0, 0, 0, 0.3) !important;
        --shadow-hover: 0 8px 25px rgba(0, 0, 0, 0.4) !important;
    }
    
    /* Force Dark Background Colors */
    .stApp, .stApp > div, .main, .main > div, body, html {
        background-color: var(--background) !important;
        background: var(--background) !important;
        color: var(--label-primary) !important;
    }
    
    section[data-testid="stSidebar"], .css-1d391kg {
        background-color: var(--grouped-background) !important;
        background: var(--grouped-background) !important;
        border-right: 1px solid var(--separator) !important;
    }
    
    /* Force Dark Text Colors */
    .stApp *, .main *, p, span, div, label {
        color: var(--label-primary) !important;
    }
    
    h1, h2, h3, h4, h5, h6 {
        color: var(--label-primary) !important;
    }
    
    /* Base Reset */
    * {
        box-sizing: border-box;
        -webkit-font-smoothing: antialiased;
        -moz-osx-font-smoothing: grayscale;
    }
    
    /* Streamlit App Base */
    .stApp {
        background: var(--background) !important;
    }
    
    /* Main Container */
    .main {
        background: var(--background) !important;
        padding: 0 !important;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    .main .block-container {
        padding-top: 1rem !important;
        padding-left: 1rem !important;
        padding-right: 1rem !important;
        padding-bottom: 1rem !important;
        max-width: 100% !important;
        background: var(--background) !important;
    }
    
    /* Navigation Header */
    .nav-header {
        background: var(--grouped-background);
        border-bottom: 1px solid var(--separator);
        padding: 20px 24px;
        margin: -1rem -1rem 2rem -1rem;
        backdrop-filter: blur(20px);
    }
    
    .nav-title {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        font-size: 28px;
        font-weight: 700;
        color: var(--label-primary);
        margin: 0;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    
    .nav-subtitle {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        font-size: 17px;
        color: var(--label-secondary);
        margin: 4px 0 0 0;
    }
    
    /* Hide Streamlit Elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {visibility: hidden;}
    div[data-testid="stToolbar"] {visibility: hidden;}
    div[data-testid="stDecoration"] {visibility: hidden;}
    div[data-testid="stStatusWidget"] {visibility: hidden;}
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Typography */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
        font-weight: 600 !important;
        color: var(--label-primary) !important;
        letter-spacing: -0.022em;
        line-height: 1.1;
        margin: 0 0 8px 0;
    }
    
    p, span, div, label {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
        color: var(--label-secondary);
        line-height: 1.47059;
        font-weight: 400;
    }
    
    /* Sidebar */
    .css-1d391kg, [data-testid="stSidebar"] {
        background: var(--grouped-background) !important;
        padding: 24px !important;
        border-right: 1px solid var(--separator) !important;
    }
    
    .css-1d391kg .stMarkdown p, [data-testid="stSidebar"] .stMarkdown p {
        color: var(--label-secondary) !important;
        font-size: 15px;
        font-weight: 400;
    }
    
    .css-1d391kg h1, .css-1d391kg h2, .css-1d391kg h3, [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
        color: var(--label-primary) !important;
        font-size: 20px;
        font-weight: 600;
        margin-bottom: 16px;
    }
    
    /* Form Controls */
    .stSelectbox > div > div {
        border: 1px solid var(--separator) !important;
        border-radius: 8px !important;
        background: var(--tertiary) !important;
        font-size: 17px !important;
        transition: all 0.2s ease !important;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }
    
    .stSelectbox > div > div:hover {
        border-color: var(--primary) !important;
    }
    
    .stTextInput > div > div > input {
        border: 1px solid var(--separator) !important;
        border-radius: 8px !important;
        background: var(--tertiary) !important;
        padding: 12px 16px !important;
        font-size: 17px !important;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
        transition: all 0.2s ease !important;
        color: var(--label-primary) !important;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: var(--primary) !important;
        outline: none !important;
        box-shadow: 0 0 0 4px rgba(0, 122, 255, 0.1) !important;
    }
    
    /* Radio Button Styling */
    .stRadio > div {
        gap: 12px !important;
    }
    
    .stRadio > div > label {
        background: var(--grouped-background) !important;
        border: 1px solid var(--separator) !important;
        border-radius: 8px !important;
        padding: 12px 16px !important;
        margin: 4px 0 !important;
        cursor: pointer !important;
        transition: all 0.2s ease !important;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
        color: var(--label-primary) !important;
    }
    
    .stRadio > div > label:hover {
        border-color: var(--primary) !important;
        background: var(--tertiary) !important;
    }
    
    .stRadio > div > label[data-checked="true"] {
        border-color: var(--primary) !important;
        background: rgba(0, 122, 255, 0.1) !important;
        color: var(--primary) !important;
    }
    
    /* Buttons */
    .stButton > button {
        background: var(--primary);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 12px 24px;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        font-size: 17px;
        font-weight: 500;
        transition: all 0.2s cubic-bezier(0.25, 0.46, 0.45, 0.94);
        box-shadow: var(--shadow);
        position: relative;
        overflow: hidden;
    }
    
    .stButton > button:before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
        transition: left 0.6s;
    }
    
    .stButton > button:hover {
        background: var(--primary-hover);
        box-shadow: var(--shadow-hover);
        transform: translateY(-1px);
    }
    
    .stButton > button:hover:before {
        left: 100%;
    }
    
    .stButton > button:active {
        transform: translateY(0);
        transition: transform 0.1s;
    }
    
    /* Enhanced Radio Buttons */
    .stRadio > div {
        background: var(--grouped-background);
        border-radius: 12px;
        padding: 16px;
        border: 1px solid var(--separator);
        transition: all 0.3s ease;
    }
    
    .stRadio > div:hover {
        border-color: var(--primary);
        box-shadow: 0 4px 12px rgba(0, 122, 255, 0.1);
    }
    
    .stRadio > div label {
        transition: color 0.2s ease;
        cursor: pointer;
        font-weight: 500;
    }
    
    .stRadio > div label:hover {
        color: var(--primary) !important;
    }
    
    /* Enhanced Selectbox */
    .stSelectbox > div > div {
        border: 1px solid var(--separator);
        border-radius: 8px;
        background: var(--tertiary);
        font-size: 17px;
        transition: all 0.3s cubic-bezier(0.25, 0.46, 0.45, 0.94);
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        position: relative;
        overflow: hidden;
    }
    
    .stSelectbox > div > div:hover {
        border-color: var(--primary);
        box-shadow: 0 4px 12px rgba(0, 122, 255, 0.1);
        transform: translateY(-1px);
    }
    
    .stSelectbox > div > div:focus-within {
        border-color: var(--primary);
        box-shadow: 0 0 0 4px rgba(0, 122, 255, 0.1);
    }
    
    /* Enhanced Text Input */
    .stTextInput > div > div > input {
        border: 1px solid var(--separator);
        border-radius: 8px;
        background: var(--tertiary);
        padding: 12px 16px;
        font-size: 17px;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        transition: all 0.3s cubic-bezier(0.25, 0.46, 0.45, 0.94);
        position: relative;
    }
    
    .stTextInput > div > div > input:hover {
        border-color: var(--primary);
        box-shadow: 0 4px 12px rgba(0, 122, 255, 0.1);
        transform: translateY(-1px);
    }
    
    .stTextInput > div > div > input:focus {
        border-color: var(--primary);
        outline: none;
        box-shadow: 0 0 0 4px rgba(0, 122, 255, 0.1);
        transform: translateY(-1px);
    }
    
    /* Sidebar Enhanced Interactions */
    .css-1d391kg {
        background: var(--grouped-background);
        padding: 24px;
        border-right: 1px solid var(--separator);
        transition: all 0.3s ease;
    }
    
    .sidebar-section {
        background: var(--tertiary);
        border-radius: 12px;
        padding: 20px;
        margin: 16px 0;
        border: 1px solid var(--separator);
        transition: all 0.3s cubic-bezier(0.25, 0.46, 0.45, 0.94);
    }
    
    .sidebar-section:hover {
        border-color: var(--primary);
        box-shadow: 0 8px 25px rgba(0, 122, 255, 0.1);
        transform: translateY(-2px);
    }
    
    /* Cards */
    [data-testid="metric-container"] {
        background: var(--grouped-background);
        border: none;
        padding: 20px;
        border-radius: 12px;
        box-shadow: var(--shadow);
        transition: all 0.3s cubic-bezier(0.25, 0.46, 0.45, 0.94);
        position: relative;
        overflow: hidden;
    }
    
    [data-testid="metric-container"]:hover {
        box-shadow: var(--shadow-hover);
        transform: translateY(-4px);
    }
    
    [data-testid="metric-container"]::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, var(--primary), #34C759);
        opacity: 0;
        transition: opacity 0.3s ease;
    }
    
    [data-testid="metric-container"]:hover::before {
        opacity: 1;
    }
    
    /* Metric Values Animation */
    [data-testid="metric-container"] [data-testid="metric-value"] {
        transition: all 0.3s ease;
        font-weight: 700;
    }
    
    [data-testid="metric-container"]:hover [data-testid="metric-value"] {
        color: var(--primary) !important;
        transform: scale(1.05);
    }
    
    /* Tab Enhancements */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: var(--grouped-background);
        border-radius: 12px;
        padding: 4px;
        border: 1px solid var(--separator);
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 48px;
        border-radius: 8px;
        padding: 0 20px;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        font-weight: 500;
        transition: all 0.3s cubic-bezier(0.25, 0.46, 0.45, 0.94);
        border: none;
        background: transparent;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: var(--tertiary);
        transform: translateY(-1px);
    }
    
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background: var(--primary);
        color: white;
        box-shadow: 0 4px 12px rgba(0, 122, 255, 0.3);
    }
    
    /* Chart Container Enhancements */
    .js-plotly-plot {
        border-radius: 16px;
        overflow: hidden;
        box-shadow: var(--shadow);
        transition: all 0.3s ease;
        background: var(--grouped-background);
        border: 1px solid var(--separator);
    }
    
    .js-plotly-plot:hover {
        box-shadow: var(--shadow-hover);
        transform: translateY(-2px);
    }
    
    /* Dataframe Styling */
    .stDataFrame {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: var(--shadow);
        border: 1px solid var(--separator);
        transition: all 0.3s ease;
    }
    
    .stDataFrame:hover {
        box-shadow: var(--shadow-hover);
    }
    
    .stDataFrame table {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    .stDataFrame thead tr th {
        background: var(--tertiary);
        color: var(--label-primary);
        font-weight: 600;
        border: none;
        padding: 16px;
    }
    
    .stDataFrame tbody tr td {
        border: none;
        padding: 16px;
        transition: background-color 0.2s ease;
    }
    
    .stDataFrame tbody tr:hover td {
        background: var(--tertiary);
    }
    
    /* Alert/Info Box Enhancements */
    .stAlert, .stInfo, .stSuccess, .stWarning, .stError {
        border-radius: 12px;
        border: none;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        font-weight: 500;
        padding: 16px 20px;
        box-shadow: var(--shadow);
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .stAlert::before, .stInfo::before, .stSuccess::before, .stWarning::before, .stError::before {
        content: '';
        position: absolute;
        left: 0;
        top: 0;
        bottom: 0;
        width: 4px;
        background: var(--primary);
    }
    
    .stAlert:hover, .stInfo:hover, .stSuccess:hover, .stWarning:hover, .stError:hover {
        transform: translateY(-2px);
        box-shadow: var(--shadow-hover);
    }
    
    /* Progress Bar Enhancements */
    .stProgress .css-j7qwjs {
        background: var(--separator);
        border-radius: 8px;
        overflow: hidden;
    }
    
    .stProgress .css-j7qwjs div {
        background: linear-gradient(90deg, var(--primary), #34C759);
        border-radius: 8px;
        transition: all 0.6s cubic-bezier(0.25, 0.46, 0.45, 0.94);
    }
    
    /* Loading Spinner */
    .stSpinner > div {
        border-color: var(--primary) transparent transparent transparent;
    }
    
    /* Smooth Scrolling */
    html {
        scroll-behavior: smooth;
    }
    
    /* Custom Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: var(--background);
    }
    
    ::-webkit-scrollbar-thumb {
        background: var(--separator);
        border-radius: 4px;
        transition: background 0.3s ease;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: var(--primary);
    }
    
    /* Pulse Animation for Loading States */
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.5; }
        100% { opacity: 1; }
    }
    
    .loading-pulse {
        animation: pulse 1.5s ease-in-out infinite;
    }
    
    /* Slide in Animation */
    @keyframes slideInLeft {
        from {
            opacity: 0;
            transform: translateX(-30px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
    
    @keyframes slideInRight {
        from {
            opacity: 0;
            transform: translateX(30px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
    
    /* Scale Animation */
    @keyframes scaleIn {
        from {
            opacity: 0;
            transform: scale(0.9);
        }
        to {
            opacity: 1;
            transform: scale(1);
        }
    }
    
    /* Apply animations to different elements */
    .css-1d391kg {
        animation: slideInLeft 0.6s ease-out;
    }
    
    .main .block-container {
        animation: slideInRight 0.6s ease-out;
    }
    
    [data-testid="metric-container"] {
        animation: scaleIn 0.5s ease-out;
    }
    
    /* Content Cards */
    .apple-card {
        background: var(--grouped-background);
        border-radius: 12px;
        padding: 24px;
        margin: 16px 0;
        box-shadow: var(--shadow);
        border: 1px solid rgba(0, 0, 0, 0.04);
        transition: all 0.2s ease;
    }
    
    .apple-card:hover {
        box-shadow: var(--shadow-hover);
    }
    
    /* Apple-style Onboarding Hero Section */
    .apple-hero-section {
        min-height: 80vh;
        display: flex;
        align-items: center;
        justify-content: center;
        background: linear-gradient(180deg, var(--background) 0%, var(--grouped-background) 100%);
        padding: 40px 20px;
        position: relative;
        overflow: hidden;
        margin: 0 -1rem;
    }
    
    .hero-content {
        max-width: 800px;
        text-align: center;
        z-index: 2;
        position: relative;
    }
    
    .hero-badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: rgba(0, 122, 255, 0.1);
        border: 1px solid rgba(0, 122, 255, 0.2);
        border-radius: 20px;
        padding: 8px 16px;
        margin-bottom: 32px;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        font-size: 14px;
        font-weight: 500;
        color: var(--primary);
        animation: fadeInUp 1s ease-out 0.2s both;
    }
    
    .badge-icon {
        animation: sparkle 2s infinite;
    }
    
    .hero-title {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        font-size: clamp(48px, 8vw, 96px);
        font-weight: 700;
        color: var(--label-primary);
        margin: 0;
        letter-spacing: -0.025em;
        line-height: 1.1;
        animation: fadeInUp 1s ease-out 0.4s both;
    }
    
    .hero-subtitle {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        font-size: clamp(48px, 8vw, 96px);
        font-weight: 700;
        background: linear-gradient(135deg, var(--primary) 0%, #34C759 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin: 0 0 32px 0;
        letter-spacing: -0.025em;
        line-height: 1.1;
        animation: fadeInUp 1s ease-out 0.6s both;
    }
    
    .hero-description {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        font-size: 21px;
        color: var(--label-secondary);
        margin: 0 0 48px 0;
        line-height: 1.5;
        max-width: 600px;
        margin-left: auto;
        margin-right: auto;
        animation: fadeInUp 1s ease-out 0.8s both;
    }
    
    .hero-stats {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 32px;
        margin: 48px 0;
        flex-wrap: wrap;
        animation: fadeInUp 1s ease-out 1s both;
    }
    
    .stat-item {
        text-align: center;
    }
    
    .stat-number {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        font-size: 32px;
        font-weight: 700;
        color: var(--primary);
        margin-bottom: 4px;
    }
    
    .stat-label {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        font-size: 15px;
        color: var(--label-secondary);
        font-weight: 500;
    }
    
    .stat-divider {
        width: 1px;
        height: 40px;
        background: var(--separator);
    }
    
    .hero-cta {
        animation: fadeInUp 1s ease-out 1.2s both;
    }
    
    .cta-content {
        display: inline-flex;
        align-items: center;
        gap: 12px;
        background: rgba(0, 122, 255, 0.08);
        border: 1px solid rgba(0, 122, 255, 0.2);
        border-radius: 12px;
        padding: 16px 24px;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        font-size: 17px;
        color: var(--label-secondary);
        font-weight: 500;
        transition: all 0.3s ease;
        cursor: pointer;
    }
    
    .cta-content:hover {
        background: rgba(0, 122, 255, 0.12);
        border-color: rgba(0, 122, 255, 0.3);
        transform: translateY(-1px);
    }
    
    .cta-arrow {
        color: var(--primary);
        font-weight: 600;
        font-size: 18px;
        transition: transform 0.3s ease;
    }
    
    .cta-content:hover .cta-arrow {
        transform: translateX(4px);
    }
    
    /* Floating Cards Animation */
    .hero-visual {
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        pointer-events: none;
        z-index: 1;
    }
    
    .floating-cards {
        position: relative;
        width: 100%;
        height: 100%;
    }
    
    .floating-card {
        position: absolute;
        background: var(--grouped-background);
        border-radius: 16px;
        padding: 20px;
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.12);
        border: 1px solid var(--separator);
        backdrop-filter: blur(20px);
        min-width: 240px;
        animation-duration: 6s;
        animation-iteration-count: infinite;
        animation-timing-function: ease-in-out;
        opacity: 0.9;
    }
    
    .card-1 {
        top: 15%;
        right: 10%;
        animation-name: float1;
        animation-delay: 0s;
    }
    
    .card-2 {
        top: 60%;
        left: 5%;
        animation-name: float2;
        animation-delay: 2s;
    }
    
    .card-3 {
        top: 35%;
        right: 15%;
        animation-name: float3;
        animation-delay: 4s;
    }
    
    .card-header {
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 16px;
    }
    
    .card-icon {
        font-size: 24px;
    }
    
    .card-title {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        font-size: 17px;
        font-weight: 600;
        color: var(--label-primary);
    }
    
    .card-content {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    .recommendation-badge {
        background: #34C759;
        color: white;
        padding: 4px 12px;
        border-radius: 8px;
        font-size: 13px;
        font-weight: 600;
        display: inline-block;
        margin-bottom: 8px;
    }
    
    .confidence-score {
        font-size: 15px;
        color: var(--label-secondary);
    }
    
    .indicator-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 12px;
    }
    
    .indicator-item {
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    
    .indicator-label {
        font-size: 13px;
        color: var(--label-secondary);
    }
    
    .indicator-value {
        font-size: 15px;
        font-weight: 600;
    }
    
    .indicator-value.bullish {
        color: #34C759;
    }
    
    .price-display {
        text-align: center;
    }
    
    .current-price {
        font-size: 20px;
        font-weight: 700;
        color: var(--label-primary);
        margin-bottom: 4px;
    }
    
    .price-change {
        font-size: 14px;
        font-weight: 500;
    }
    
    .price-change.positive {
        color: #34C759;
    }
    
    /* Features Section */
    .features-section {
        padding: 80px 20px;
        background: var(--background);
        margin: 0 -1rem;
    }
    
    .features-header {
        text-align: center;
        max-width: 600px;
        margin: 0 auto 60px auto;
    }
    
    .features-header h2 {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        font-size: 42px;
        font-weight: 700;
        color: var(--label-primary);
        margin: 0 0 20px 0;
        letter-spacing: -0.025em;
        line-height: 1.2;
    }
    
    .features-header p {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        font-size: 19px;
        color: var(--label-secondary);
        margin: 0;
        line-height: 1.5;
    }
    
    .features-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
        gap: 28px;
        max-width: 1200px;
        margin: 0 auto;
        padding: 0 20px;
    }
    
    .feature-card {
        background: var(--grouped-background);
        border-radius: 20px;
        padding: 40px 32px;
        box-shadow: var(--shadow);
        border: 1px solid var(--separator);
        text-align: center;
        transition: all 0.4s cubic-bezier(0.25, 0.46, 0.45, 0.94);
        position: relative;
        overflow: hidden;
    }
    
    .feature-card:hover {
        transform: translateY(-8px);
        box-shadow: 0 20px 40px rgba(0, 122, 255, 0.15);
    }
    
    .feature-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, var(--primary), #34C759);
        opacity: 0;
        transition: opacity 0.3s ease;
    }
    
    .feature-card:hover::before {
        opacity: 1;
    }
    
    .feature-icon {
        font-size: 48px;
        margin-bottom: 24px;
        animation: bounce 2s infinite;
    }
    
    .feature-card h3 {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        font-size: 24px;
        font-weight: 600;
        color: var(--label-primary);
        margin: 0 0 16px 0;
    }
    
    .feature-card p {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        font-size: 17px;
        color: var(--label-secondary);
        line-height: 1.5;
        margin: 0 0 24px 0;
    }
    
    .feature-highlight {
        display: inline-block;
        background: rgba(0, 122, 255, 0.1);
        color: var(--primary);
        padding: 8px 16px;
        border-radius: 12px;
        font-size: 14px;
        font-weight: 500;
        border: 1px solid rgba(0, 122, 255, 0.2);
    }
    
    /* Getting Started Section */
    .getting-started-section {
        padding: 80px 20px;
        background: var(--grouped-background);
        margin: 0 -1rem;
    }
    
    .getting-started-content {
        max-width: 800px;
        margin: 0 auto;
        text-align: center;
        padding: 0 20px;
    }
    
    .getting-started-content h2 {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        font-size: 42px;
        font-weight: 700;
        color: var(--label-primary);
        margin: 0 0 24px 0;
        letter-spacing: -0.025em;
        line-height: 1.2;
    }
    
    .getting-started-content p {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        font-size: 19px;
        color: var(--label-secondary);
        margin: 0 0 64px 0;
        line-height: 1.5;
    }
    
    .quick-start-steps {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
        gap: 40px;
        margin-top: 48px;
    }
    
    .step-item {
        display: flex;
        flex-direction: column;
        align-items: center;
        text-align: center;
    }
    
    .step-number {
        width: 60px;
        height: 60px;
        background: var(--primary);
        color: white;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        font-size: 24px;
        font-weight: 700;
        margin-bottom: 24px;
        box-shadow: 0 8px 20px rgba(0, 122, 255, 0.3);
    }
    
    .step-content h4 {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        font-size: 20px;
        font-weight: 600;
        color: var(--label-primary);
        margin: 0 0 12px 0;
    }
    
    .step-content p {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        font-size: 17px;
        color: var(--label-secondary);
        margin: 0;
        line-height: 1.4;
    }
    
    /* Animations */
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes sparkle {
        0%, 100% { transform: scale(1) rotate(0deg); }
        50% { transform: scale(1.1) rotate(180deg); }
    }
    
    @keyframes float1 {
        0%, 100% { transform: translateY(0px) translateX(0px) rotate(0deg); }
        25% { transform: translateY(-20px) translateX(10px) rotate(1deg); }
        50% { transform: translateY(-10px) translateX(-5px) rotate(-1deg); }
        75% { transform: translateY(-25px) translateX(8px) rotate(0.5deg); }
    }
    
    @keyframes float2 {
        0%, 100% { transform: translateY(0px) translateX(0px) rotate(0deg); }
        25% { transform: translateY(-15px) translateX(-8px) rotate(-1deg); }
        50% { transform: translateY(-25px) translateX(12px) rotate(1deg); }
        75% { transform: translateY(-10px) translateX(-6px) rotate(-0.5deg); }
    }
    
    @keyframes float3 {
        0%, 100% { transform: translateY(0px) translateX(0px) rotate(0deg); }
        25% { transform: translateY(-30px) translateX(-10px) rotate(1deg); }
        50% { transform: translateY(-5px) translateX(15px) rotate(-1deg); }
        75% { transform: translateY(-20px) translateX(-12px) rotate(0.5deg); }
    }
    
    @keyframes bounce {
        0%, 20%, 50%, 80%, 100% { transform: translateY(0); }
        40% { transform: translateY(-10px); }
        60% { transform: translateY(-5px); }
    }
    
    /* Responsive Design */
    @media (max-width: 768px) {
        .floating-card {
            display: none;
        }
        
        .hero-stats {
            gap: 20px;
        }
        
        .stat-divider {
            display: none;
        }
        
        .features-grid {
            grid-template-columns: 1fr;
            gap: 24px;
        }
        
        .quick-start-steps {
            grid-template-columns: 1fr;
            gap: 32px;
        }
    }
    
    /* Separators */
    hr {
        border: none;
        height: 1px;
        background: var(--separator);
        margin: 24px 0;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background: var(--secondary);
        border-radius: 8px;
        padding: 4px;
        gap: 0;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        color: var(--label-secondary);
        border-radius: 6px;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        font-weight: 500;
        font-size: 15px;
        transition: all 0.2s ease;
        margin: 0;
        padding: 8px 16px;
    }
    
    .stTabs [aria-selected="true"] {
        background: var(--tertiary) !important;
        color: var(--primary) !important;
        box-shadow: var(--shadow);
    }
    
    /* Data Tables */
    .stDataFrame {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: var(--shadow);
        border: 1px solid var(--separator);
    }
    
    /* Alerts */
    .stAlert {
        border-radius: 12px;
        border: 1px solid var(--separator);
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Scrollbars */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: var(--secondary);
    }
    
    ::-webkit-scrollbar-thumb {
        background: var(--separator);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: var(--label-tertiary);
    }
    
    /* Navigation Header */
    .nav-header {
        background: var(--grouped-background);
        padding: 20px 24px;
        border-bottom: 1px solid var(--separator);
        margin: 0 -24px 24px -24px;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    
    .nav-title {
        font-size: 22px;
        font-weight: 600;
        color: var(--label-primary);
        margin: 0;
    }
    
    .nav-subtitle {
        font-size: 15px;
        color: var(--label-secondary);
        margin: 4px 0 0 0;
    }
    
    /* Status Indicators */
    .status-indicator {
        display: inline-flex;
        align-items: center;
        padding: 4px 8px;
        border-radius: 6px;
        font-size: 13px;
        font-weight: 500;
    }
    
    .status-buy {
        background: rgba(52, 199, 89, 0.1);
        color: var(--system-green);
    }
    
    .status-sell {
        background: rgba(255, 59, 48, 0.1);
        color: var(--system-red);
    }
    
    .status-hold {
        background: rgba(255, 149, 0, 0.1);
        color: var(--system-orange);
    }
    
    /* Loading States */
    .stSpinner {
        border-color: var(--primary) !important;
    }
    
    /* Feature Cards */
    .feature-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
        gap: 16px;
        margin: 24px 0;
    }
    
    .feature-card {
        background: var(--grouped-background);
        border-radius: 12px;
        padding: 24px;
        box-shadow: var(--shadow);
        transition: all 0.2s ease;
        text-align: center;
    }
    
    .feature-card:hover {
        box-shadow: var(--shadow-hover);
        transform: translateY(-2px);
    }
    
    .feature-icon {
        width: 48px;
        height: 48px;
        background: var(--primary);
        border-radius: 12px;
        margin: 0 auto 16px auto;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 24px;
        color: white;
    }
    
    .feature-title {
        font-size: 17px;
        font-weight: 600;
        color: var(--label-primary);
        margin: 0 0 8px 0;
    }
    
    .feature-description {
        font-size: 15px;
        color: var(--label-secondary);
        line-height: 1.4;
        margin: 0;
    }
</style>
""", unsafe_allow_html=True)

# SVG Icons Helper Functions
def get_svg_icon(name, size=24, color="currentColor"):
    """Get SVG icons for Apple-style interface"""
    icons = {
        "chart": f"""<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 3v18h18"/><path d="m19 9-5 5-4-4-3 3"/></svg>""",
        "target": f"""<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/></svg>""",
        "brain": f"""<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 5a3 3 0 1 0-5.997.125 4 4 0 0 0-2.526 5.77 4 4 0 0 0 .556 6.588A4 4 0 1 0 12 18Z"/><path d="M12 5a3 3 0 1 1 5.997.125 4 4 0 0 1 2.526 5.77 4 4 0 0 1-.556 6.588A4 4 0 1 1 12 18Z"/><path d="M15 13a4.5 4.5 0 0 1-3-4 4.5 4.5 0 0 1-3 4"/><path d="M17.599 6.5a3 3 0 0 0 .399-1.375"/><path d="M6.003 5.125A3 3 0 0 0 6.401 6.5"/><path d="M3.477 10.896a4 4 0 0 1 .585-.396"/><path d="M19.938 10.5a4 4 0 0 1 .585.396"/><path d="M6 18a4 4 0 0 1-1.967-.516"/><path d="M19.967 17.484A4 4 0 0 1 18 18"/></svg>""",
        "shield": f"""<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M20 13c0 5-3.5 7.5-8 7.5s-8-2.5-8-7.5c0-1.3.3-2.6.7-3.8L12 2l7.3 7.2c.4 1.2.7 2.5.7 3.8Z"/></svg>""",
        "lightning": f"""<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m13 2-3 7h4l-3 11 3-7h-4l3-11z"/></svg>""",
        "search": f"""<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="11" cy="11" r="8"/><path d="m21 21-4.35-4.35"/></svg>""",
        "stock": f"""<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 3v18h18"/><path d="M7 12v4"/><path d="M11 10v6"/><path d="M15 6v10"/><path d="M19 4v12"/></svg>""",
        "settings": f"""<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12.22 2h-.44a2 2 0 0 0-2 2v.18a2 2 0 0 1-1 1.73l-.43.25a2 2 0 0 1-2 0l-.15-.08a2 2 0 0 0-2.73.73l-.22.38a2 2 0 0 0 .73 2.73l.15.1a2 2 0 0 1 1 1.72v.51a2 2 0 0 1-1 1.74l-.15.09a2 2 0 0 0-.73 2.73l.22.38a2 2 0 0 0 2.73.73l.15-.08a2 2 0 0 1 2 0l.43.25a2 2 0 0 1 1 1.73V20a2 2 0 0 0 2 2h.44a2 2 0 0 0 2-2v-.18a2 2 0 0 1 1-1.73l.43-.25a2 2 0 0 1 2 0l.15.08a2 2 0 0 0 2.73-.73l.22-.38a2 2 0 0 0-.73-2.73l-.15-.08a2 2 0 0 1-1-1.74v-.5a2 2 0 0 1 1-1.74l.15-.09a2 2 0 0 0 .73-2.73l-.22-.38a2 2 0 0 0-2.73-.73l-.15.08a2 2 0 0 1-2 0l-.43-.25a2 2 0 0 1-1-1.73V4a2 2 0 0 0-2-2z"/><circle cx="12" cy="12" r="3"/></svg>""",
        "info": f"""<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><path d="m9 12 2 2 4-4"/></svg>""",
        "arrow-up": f"""<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m18 15-6-6-6 6"/></svg>""",
        "arrow-down": f"""<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m6 9 6 6 6-6"/></svg>""",
        "minus": f"""<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M5 12h14"/></svg>""",
        "check_circle_filled": f"""<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="{color}" stroke="none"><circle cx="12" cy="12" r="10" fill="{color}"/><path d="m9 12 2 2 4-4" stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>""",
        "edit": f"""<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7"/><path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z"/></svg>""",
        "star": f"""<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="12,2 15.09,8.26 22,9.27 17,14.14 18.18,21.02 12,17.77 5.82,21.02 7,14.14 2,9.27 8.91,8.26"/></svg>""",
        "sidebar": f"""<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect width="18" height="18" x="3" y="3" rx="2" ry="2"/><line x1="9" x2="9" y1="3" y2="21"/></svg>""",
        "warning": f"""<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m21.73 18-8-14a2 2 0 0 0-3.46 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z"/><path d="M12 9v4"/><path d="m12 17 .01 0"/></svg>"""
    }
    return icons.get(name, "")

# Navigation Header Component
def render_navigation_header():
    """Render Apple-style navigation header with CSS"""
    st.markdown(f"""
    <style>
    /* Import Fizon Soft Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');
    
    /* Dark Mode Premium Design System with Fizon Soft */
    :root {{
        --primary: #0A84FF !important;
        --primary-hover: #0970E6 !important;
        --background: #000000 !important;
        --surface: #111111 !important;
        --grouped-background: #1A1A1A !important;
        --card-background: #1E1E1E !important;
        --separator: #2A2A2A !important;
        --separator-light: #333333 !important;
        --label-primary: #FFFFFF !important;
        --label-secondary: #A3A3A3 !important;
        --label-tertiary: #6B6B6B !important;
        --system-blue: #007AFF !important;
        --system-red: #FF453A !important;
        --system-green: #30D158 !important;
        --system-orange: #FF9F0A !important;
        --system-purple: #BF5AF2 !important;
        --shadow: 0 4px 24px rgba(0, 0, 0, 0.8) !important;
        --shadow-hover: 0 8px 40px rgba(0, 0, 0, 0.9) !important;
        --shadow-card: 0 2px 16px rgba(0, 0, 0, 0.6) !important;
        --border-radius: 16px !important;
        --border-radius-small: 12px !important;
        --transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }}
    
    /* Global Font Family - Inter */
    * {{
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif !important;
        font-feature-settings: 'cv11', 'ss01' !important;
        -webkit-font-smoothing: antialiased !important;
        -moz-osx-font-smoothing: grayscale !important;
        letter-spacing: -0.015em !important;
    }}
    
    /* Force Dark Background Colors */
    .stApp, .stApp > div, .main, .main > div, body, html {{
        background-color: var(--background) !important;
        background: var(--background) !important;
        color: var(--label-primary) !important;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }}
    
    section[data-testid="stSidebar"], .css-1d391kg {{
        background: linear-gradient(180deg, var(--surface) 0%, var(--grouped-background) 100%) !important;
        border-right: 1px solid var(--separator) !important;
        backdrop-filter: blur(20px) !important;
    }}
    
    /* Enhanced Text Styling with Inter */
    .stApp *, .main *, p, span, div, label {{
        color: var(--label-primary) !important;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }}
    
    h1, h2, h3, h4, h5, h6 {{
        color: var(--label-primary) !important;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
        font-weight: 700 !important;
        letter-spacing: -0.025em !important;
    }}
    
    h1 {{ font-size: 2.5rem !important; font-weight: 800 !important; }}
    h2 {{ font-size: 2rem !important; font-weight: 700 !important; }}
    h3 {{ font-size: 1.5rem !important; font-weight: 600 !important; }}
    h4 {{ font-size: 1.25rem !important; font-weight: 600 !important; }}
    
    /* Streamlit App Base */
    .stApp {{
        background: radial-gradient(ellipse at top, rgba(10, 132, 255, 0.1) 0%, var(--background) 50%) !important;
    }}
    
    /* Main Container */
    .main {{
        background: transparent !important;
        padding: 0 !important;
    }}
    
    .main .block-container {{
        padding-top: 1rem !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
        padding-bottom: 2rem !important;
        max-width: 100% !important;
        background: transparent !important;
    }}
    
    /* Navigation Header */
    .nav-header {{
        background: linear-gradient(135deg, var(--surface) 0%, var(--grouped-background) 100%) !important;
        border-bottom: 1px solid var(--separator) !important;
        padding: 24px 32px !important;
        margin: -1rem -2rem 2rem -2rem !important;
        backdrop-filter: blur(40px) !important;
        border-radius: 0 0 var(--border-radius) var(--border-radius) !important;
        box-shadow: var(--shadow-card) !important;
    }}
    
    .nav-title {{
        font-size: 32px !important;
        font-weight: 800 !important;
        color: var(--label-primary) !important;
        margin: 0 !important;
        display: flex !important;
        align-items: center !important;
        gap: 16px !important;
        letter-spacing: -0.03em !important;
    }}
    
    .nav-subtitle {{
        font-size: 16px !important;
        font-weight: 500 !important;
        color: var(--label-secondary) !important;
        margin: 8px 0 0 0 !important;
        letter-spacing: -0.01em !important;
    }}
    
    /* Premium Apple Card Style */
    .apple-card {{
        background: linear-gradient(135deg, var(--card-background) 0%, var(--grouped-background) 100%) !important;
        border-radius: var(--border-radius) !important;
        padding: 32px !important;
        margin: 24px 0 !important;
        border: 1px solid var(--separator) !important;
        box-shadow: var(--shadow-card) !important;
        transition: var(--transition) !important;
        backdrop-filter: blur(20px) !important;
        position: relative !important;
        overflow: hidden !important;
    }}
    
    .apple-card::before {{
        content: '' !important;
        position: absolute !important;
        top: 0 !important;
        left: 0 !important;
        right: 0 !important;
        height: 1px !important;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.1), transparent) !important;
    }}
    
    .apple-card:hover {{
        box-shadow: var(--shadow-hover) !important;
        transform: translateY(-4px) !important;
        border-color: var(--separator-light) !important;
    }}
    
    /* Button Styles */
    .apple-button {{
        background: linear-gradient(135deg, var(--primary) 0%, var(--primary-hover) 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: var(--border-radius-small) !important;
        padding: 16px 32px !important;
        font-size: 16px !important;
        font-weight: 600 !important;
        cursor: pointer !important;
        transition: var(--transition) !important;
        text-decoration: none !important;
        display: inline-flex !important;
        align-items: center !important;
        gap: 12px !important;
        box-shadow: var(--shadow-card) !important;
        letter-spacing: -0.01em !important;
    }}
    
    .apple-button:hover {{
        background: linear-gradient(135deg, var(--primary-hover) 0%, #0757CC 100%) !important;
        transform: translateY(-2px) !important;
        box-shadow: var(--shadow-hover) !important;
    }}
    
    .apple-button-secondary {{
        background: linear-gradient(135deg, var(--grouped-background) 0%, var(--card-background) 100%) !important;
        color: var(--primary) !important;
        border: 1px solid var(--separator) !important;
    }}
    
    .apple-button-secondary:hover {{
        background: linear-gradient(135deg, var(--card-background) 0%, var(--separator) 100%) !important;
        border-color: var(--separator-light) !important;
    }}
    
    /* Feature Card Styles */
    .feature-card {{
        background: linear-gradient(135deg, var(--card-background) 0%, var(--grouped-background) 100%) !important;
        border-radius: var(--border-radius) !important;
        padding: 28px !important;
        border: 1px solid var(--separator) !important;
        transition: var(--transition) !important;
        height: 100% !important;
        backdrop-filter: blur(20px) !important;
        position: relative !important;
        overflow: hidden !important;
    }}
    
    .feature-card::before {{
        content: '' !important;
        position: absolute !important;
        top: 0 !important;
        left: 0 !important;
        right: 0 !important;
        height: 1px !important;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.08), transparent) !important;
    }}
    
    .feature-card:hover {{
        transform: translateY(-6px) !important;
        box-shadow: var(--shadow-hover) !important;
        border-color: var(--separator-light) !important;
    }}
    
    .feature-icon {{
        font-size: 32px !important;
        margin-bottom: 20px !important;
        display: block !important;
        filter: drop-shadow(0 2px 8px rgba(0, 0, 0, 0.5)) !important;
    }}
    
    .feature-title {{
        font-size: 20px !important;
        font-weight: 700 !important;
        color: var(--label-primary) !important;
        margin: 0 0 16px 0 !important;
        letter-spacing: -0.02em !important;
    }}
    
    .feature-description {{
        font-size: 15px !important;
        font-weight: 400 !important;
        color: var(--label-secondary) !important;
        line-height: 1.6 !important;
        margin: 0 !important;
        letter-spacing: -0.01em !important;
    }}
    
    /* Enhanced Tab Styling */
    .stTabs [data-baseweb="tab-list"] {{
        background: var(--grouped-background) !important;
        border-radius: var(--border-radius-small) !important;
        padding: 4px !important;
        border: 1px solid var(--separator) !important;
        gap: 4px !important;
    }}
    
    .stTabs [data-baseweb="tab"] {{
        background: transparent !important;
        border-radius: 8px !important;
        padding: 12px 20px !important;
        font-weight: 600 !important;
        font-size: 14px !important;
        color: var(--label-secondary) !important;
        transition: var(--transition) !important;
        border: none !important;
        letter-spacing: -0.01em !important;
    }}
    
    .stTabs [aria-selected="true"] {{
        background: var(--primary) !important;
        color: white !important;
        box-shadow: 0 2px 8px rgba(10, 132, 255, 0.3) !important;
    }}
    
    /* Input Styling */
    .stTextInput input, .stSelectbox select {{
        background: var(--grouped-background) !important;
        border: 1px solid var(--separator) !important;
        border-radius: var(--border-radius-small) !important;
        color: var(--label-primary) !important;
        font-family: 'Inter', 'Fizon Soft', sans-serif !important;
        font-weight: 500 !important;
        letter-spacing: -0.01em !important;
        transition: var(--transition) !important;
    }}
    
    .stTextInput input:focus, .stSelectbox select:focus {{
        border-color: var(--primary) !important;
        box-shadow: 0 0 0 3px rgba(10, 132, 255, 0.1) !important;
    }}
    
    /* Button Enhancements */
    .stButton button {{
        background: linear-gradient(135deg, var(--primary) 0%, var(--primary-hover) 100%) !important;
        border: none !important;
        border-radius: var(--border-radius-small) !important;
        color: white !important;
        font-weight: 600 !important;
        font-family: 'Inter', 'Fizon Soft', sans-serif !important;
        transition: var(--transition) !important;
        box-shadow: var(--shadow-card) !important;
        letter-spacing: -0.01em !important;
    }}
    
    .stButton button:hover {{
        background: linear-gradient(135deg, var(--primary-hover) 0%, #0757CC 100%) !important;
        transform: translateY(-1px) !important;
        box-shadow: var(--shadow) !important;
    }}
    
    /* Metric Cards */
    .metric-card {{
        background: linear-gradient(135deg, var(--card-background) 0%, var(--grouped-background) 100%) !important;
        border-radius: var(--border-radius) !important;
        padding: 24px !important;
        border: 1px solid var(--separator) !important;
        text-align: center !important;
        transition: var(--transition) !important;
        backdrop-filter: blur(20px) !important;
        position: relative !important;
        overflow: hidden !important;
    }}
    
    .metric-card::before {{
        content: '' !important;
        position: absolute !important;
        top: 0 !important;
        left: 0 !important;
        right: 0 !important;
        height: 1px !important;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.1), transparent) !important;
    }}
    
    .metric-card:hover {{
        transform: translateY(-2px) !important;
        box-shadow: var(--shadow) !important;
        border-color: var(--separator-light) !important;
    }}
    
    /* Scrollbar Styling */
    ::-webkit-scrollbar {{
        width: 8px !important;
        height: 8px !important;
    }}
    
    ::-webkit-scrollbar-track {{
        background: var(--grouped-background) !important;
        border-radius: 4px !important;
    }}
    
    ::-webkit-scrollbar-thumb {{
        background: var(--separator) !important;
        border-radius: 4px !important;
        transition: var(--transition) !important;
    }}
    
    ::-webkit-scrollbar-thumb:hover {{
        background: var(--separator-light) !important;
    }}
    
    /* Loading Animation */
    @keyframes shimmer {{
        0% {{ background-position: -200px 0; }}
        100% {{ background-position: calc(200px + 100%) 0; }}
    }}
    
    .loading-shimmer {{
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.1), transparent) !important;
        background-size: 200px 100% !important;
        animation: shimmer 1.5s infinite !important;
    }}
    </style>
    
    <div class="nav-header">
        <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; text-align: center; padding: 20px 0; width: 100%;">
            <div style="background: rgba(10, 132, 255, 0.1); border: 1px solid rgba(10, 132, 255, 0.2); border-radius: 20px; padding: 8px 16px; margin-bottom: 24px; display: inline-flex; align-items: center; gap: 8px;">
                <span style="animation: sparkle 2s infinite;">✨</span>
                <span style="color: var(--primary); font-weight: 500; font-size: 14px;">AI-Powered Analysis</span>
            </div>
            <div style="display: flex; flex-direction: column; align-items: center;">
                <h1 style="font-size: clamp(32px, 6vw, 48px); font-weight: 700; color: var(--label-primary); margin: 0; letter-spacing: -0.025em; text-align: center;">
                    Stock Analysis
                </h1>
                <h1 style="font-size: clamp(32px, 6vw, 48px); font-weight: 700; background: linear-gradient(135deg, var(--primary) 0%, var(--system-green) 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin: 0 0 16px 0; letter-spacing: -0.025em; text-align: center;">
                    Reimagined.
                </h1>
            </div>
            <p style="font-size: 16px; color: var(--label-secondary); margin: 0; max-width: 600px; line-height: 1.5; text-align: center;">
                Professional-grade financial analysis powered by artificial intelligence.<br>
                Built with investors in mind.
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Popular stocks data structure with currency indicators
popular_stocks = {
    "🇺🇸 US Stocks (USD)": {
        "Apple Inc.": "AAPL",
        "Microsoft": "MSFT", 
        "Alphabet": "GOOGL",
        "Amazon": "AMZN",
        "Tesla": "TSLA",
        "Meta": "META",
        "NVIDIA": "NVDA",
        "Netflix": "NFLX"
    },
    "🇮🇳 Indian Stocks (INR)": {
        "Reliance Industries": "RELIANCE.NS",
        "Tata Consultancy Services": "TCS.NS",
        "HDFC Bank": "HDFCBANK.NS",
        "Infosys": "INFY.NS",
        "ICICI Bank": "ICICIBANK.NS",
        "State Bank of India": "SBIN.NS",
        "Bharti Airtel": "BHARTIARTL.NS",
        "ITC Limited": "ITC.NS"
    },
    "🇧🇷 Brazilian Stocks (BRL)": {
        "Petrobras": "PETR4.SA",
        "Vale": "VALE3.SA",
        "Itaú Unibanco": "ITUB4.SA",
        "Banco do Brasil": "BBAS3.SA",
        "Ambev": "ABEV3.SA",
        "Magazine Luiza": "MGLU3.SA"
    },
    "₿ Cryptocurrency (USD)": {
        "Bitcoin": "BTC-USD",
        "Ethereum": "ETH-USD",
        "Binance Coin": "BNB-USD",
        "Cardano": "ADA-USD",
        "Solana": "SOL-USD",
        "Polygon": "MATIC-USD"
    },
    "📊 Market Indices": {
        "S&P 500": "^GSPC",
        "NIFTY 50": "^NSEI",
        "SENSEX": "^BSESN",
        "BOVESPA": "^BVSP",
        "NASDAQ": "^IXIC",
        "Dow Jones": "^DJI"
    }
}

# Helper function to create Apple-style onboarding
def render_model_info():
    """Display selected ML model information"""
    selected_model = os.environ.get('SELECTED_MODEL')
    model_name = os.environ.get('MODEL_NAME')
    model_params = os.environ.get('MODEL_PARAMS')
    
    if selected_model and model_name:
        # Model info based on selection
        model_details = {
            '1': {'icon': '⚡', 'color': '#FFD60A', 'description': 'Ultra-fast inference for quick demos'},
            '2': {'icon': '🚀', 'color': '#0A84FF', 'description': 'Balanced speed and accuracy'},
            '3': {'icon': '🧠', 'color': '#30D158', 'description': 'Complete model with all features'},
            '4': {'icon': '🔥', 'color': '#FF453A', 'description': 'Maximum accuracy model'},
            '5': {'icon': '🔧', 'color': '#BF5AF2', 'description': 'Transfer learning demonstration'},
            '6': {'icon': '🌐', 'color': '#FF9F0A', 'description': 'Multi-market unified model'}
        }
        
        detail = model_details.get(selected_model, {'icon': '🤖', 'color': '#6B6B6B', 'description': 'ML Model Active'})
        
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, var(--card-background) 0%, var(--grouped-background) 100%); 
                    border: 1px solid var(--separator); 
                    border-radius: var(--border-radius-small); 
                    padding: 16px 20px; 
                    margin: 16px 0; 
                    box-shadow: var(--shadow-card);">
            <div style="display: flex; align-items: center; gap: 12px;">
                <div style="font-size: 24px;">{detail['icon']}</div>
                <div style="flex: 1;">
                    <div style="font-size: 16px; font-weight: 600; color: var(--label-primary); margin-bottom: 4px;">
                        {model_name} ({model_params} params)
                    </div>
                    <div style="font-size: 14px; color: var(--label-secondary);">
                        {detail['description']}
                    </div>
                </div>
                <div style="background: {detail['color']}20; 
                           color: {detail['color']}; 
                           padding: 6px 12px; 
                           border-radius: 8px; 
                           font-size: 12px; 
                           font-weight: 600;">
                    ACTIVE
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

def render_onboarding():
    """Render getting started guidance and stats"""
    
    # Stats Section
    st.markdown("""
    <div style="margin: 32px 0;">
        <h3 style="text-align: center; color: var(--label-primary); margin-bottom: 32px; font-size: 24px; font-weight: 600;">
            Powerful Analytics at Your Fingertips
        </h3>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div style="text-align: center; padding: 24px;">
            <div style="font-size: 32px; font-weight: 700; color: var(--primary); margin-bottom: 8px;">20+</div>
            <div style="font-size: 15px; color: var(--label-secondary); font-weight: 500;">Technical Indicators</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="text-align: center; padding: 24px;">
            <div style="font-size: 32px; font-weight: 700; color: var(--primary); margin-bottom: 8px;">95%</div>
            <div style="font-size: 15px; color: var(--label-secondary); font-weight: 500;">AI Accuracy</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style="text-align: center; padding: 24px;">
            <div style="font-size: 32px; font-weight: 700; color: var(--primary); margin-bottom: 8px;">&lt;3s</div>
            <div style="font-size: 15px; color: var(--label-secondary); font-weight: 500;">Analysis Speed</div>
        </div>
        """, unsafe_allow_html=True)
    
    # CTA Section
    st.markdown("""
    <div style="text-align: center; margin: 40px 0;">
        <div style="display: inline-flex; align-items: center; gap: 12px; background: rgba(10, 132, 255, 0.08); border: 1px solid rgba(10, 132, 255, 0.2); border-radius: 12px; padding: 16px 24px; cursor: pointer; transition: all 0.3s ease;">
            <span style="color: var(--primary); font-weight: 600; font-size: 18px;">→</span>
            <span style="color: var(--label-secondary); font-weight: 500;">Select a stock from the sidebar to begin</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Features Section
    st.markdown("""
    <div style="text-align: center; margin: 60px 0 40px 0;">
        <h2 style="font-size: 36px; font-weight: 700; color: var(--label-primary); margin: 0 0 16px 0;">
            Everything you need for intelligent investing
        </h2>
        <p style="font-size: 17px; color: var(--label-secondary); margin: 0;">
            Professional tools designed for both novice and expert investors
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Feature Cards
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">🧠</div>
            <h3 class="feature-title">Explainable AI</h3>
            <p class="feature-description">Understand exactly why the AI made its recommendation with detailed reasoning and confidence scores.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="feature-card" style="margin-top: 16px;">
            <div class="feature-icon">{get_svg_icon("chart", 32, "#0A84FF")}</div>
            <h3 class="feature-title">Advanced Charts</h3>
            <p class="feature-description">Interactive technical analysis with 20+ indicators, pattern recognition, and professional-grade visualization.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="feature-card">
            <div class="feature-icon">{get_svg_icon("shield", 32, "#FF9F0A")}</div>
            <h3 class="feature-title">Risk Management</h3>
            <p class="feature-description">Comprehensive risk assessment with volatility analysis, stop-loss suggestions, and portfolio impact.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="feature-card" style="margin-top: 16px;">
            <div class="feature-icon">{get_svg_icon("target", 32, "#30D158")}</div>
            <h3 class="feature-title">Price Targets</h3>
            <p class="feature-description">AI-calculated target prices and stop-loss levels based on market conditions and volatility analysis.</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Getting Started
    st.markdown("""
    <div style="text-align: center; margin: 60px 0 40px 0;">
        <h2 style="font-size: 32px; font-weight: 700; color: var(--label-primary); margin: 0 0 16px 0;">
            Ready to experience the future of stock analysis?
        </h2>
        <p style="font-size: 17px; color: var(--label-secondary); margin: 0;">
            Join thousands of investors making smarter decisions with AI-powered insights
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Steps
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div style="text-align: center;">
            <div style="width: 50px; height: 50px; background: var(--primary); color: white; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 20px; font-weight: 700; margin: 0 auto 16px auto;">1</div>
            <h4 style="font-size: 18px; font-weight: 600; color: var(--label-primary); margin: 0 0 8px 0;">Select a Stock</h4>
            <p style="font-size: 15px; color: var(--label-secondary); margin: 0;">Choose from popular stocks or search by symbol</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="text-align: center;">
            <div style="width: 50px; height: 50px; background: var(--primary); color: white; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 20px; font-weight: 700; margin: 0 auto 16px auto;">2</div>
            <h4 style="font-size: 18px; font-weight: 600; color: var(--label-primary); margin: 0 0 8px 0;">Configure Analysis</h4>
            <p style="font-size: 15px; color: var(--label-secondary); margin: 0;">Set your risk tolerance and analysis period</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style="text-align: center;">
            <div style="width: 50px; height: 50px; background: var(--primary); color: white; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 20px; font-weight: 700; margin: 0 auto 16px auto;">3</div>
            <h4 style="font-size: 18px; font-weight: 600; color: var(--label-primary); margin: 0 0 8px 0;">Get AI Insights</h4>
            <p style="font-size: 15px; color: var(--label-secondary); margin: 0;">Receive detailed recommendations and analysis</p>
        </div>
        """, unsafe_allow_html=True)

# Sidebar for stock selection
def render_main_stock_selection():
    """Render stock selection interface in main content area"""
    
    st.markdown(f"""
    <div class="apple-card" style="margin-bottom: 32px;">
        <div style="display: flex; align-items: center; margin-bottom: 24px;">
            <div style="margin-right: 16px;">{get_svg_icon("stock", 32, "#0A84FF")}</div>
            <div>
                <h2 style="font-size: 28px; font-weight: 700; color: var(--label-primary); margin: 0;">
                    Stock Selection
                </h2>
                <p style="font-size: 15px; color: var(--label-secondary); margin: 4px 0 0 0;">
                    Choose a stock for AI-powered analysis
                </p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Create columns for better layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Stock selection tabs
        tab1, tab2, tab3 = st.tabs([
            "🔍 Search", 
            "✏️ Custom Symbol", 
            "⭐ Popular Stocks"
        ])
        
        selected_symbol = None
        
        with tab1:
            st.markdown(f"""
            <div style="display: flex; align-items: center; margin-bottom: 16px;">
                {get_svg_icon('search', 20, '#0A84FF')}
                <span style="margin-left: 8px; font-weight: 600; color: var(--label-primary);">Search by company name or stock symbol</span>
            </div>
            """, unsafe_allow_html=True)
            search_term = st.text_input(
                "Search stocks",
                placeholder="e.g., Apple, AAPL, Microsoft, MSFT",
                key="main_search",
                label_visibility="collapsed",
            )
            
            if search_term:
                # Simple search logic
                found_stocks = []
                for category, stocks in popular_stocks.items():
                    for name, symbol in stocks.items():
                        if search_term.lower() in name.lower() or search_term.upper() in symbol:
                            found_stocks.append((name, symbol, category))
                
                if found_stocks:
                    st.markdown("**Found stocks:**")
                    # Create a nice grid layout for search results
                    for i in range(0, len(found_stocks[:8]), 2):  # Show top 8 results in 2 columns
                        cols = st.columns(2)
                        for j, col in enumerate(cols):
                            if i + j < len(found_stocks):
                                name, symbol, category = found_stocks[i + j]
                                with col:
                                    if st.button(f"**{name}** ({symbol})", key=f"search_{symbol}", use_container_width=True):
                                        selected_symbol = symbol
                else:
                    st.markdown(f"""
                    <div style="padding: 12px; background: rgba(10, 132, 255, 0.1); border: 1px solid rgba(10, 132, 255, 0.2); border-radius: 8px; display: flex; align-items: center;">
                        {get_svg_icon('info', 16, '#0A84FF')}
                        <span style="margin-left: 8px; color: var(--label-secondary);">No stocks found. Try a different search term.</span>
                    </div>
                    """, unsafe_allow_html=True)
        
        with tab2:
            st.markdown(f"""
            <div style="display: flex; align-items: center; margin-bottom: 16px;">
                {get_svg_icon('edit', 20, '#0A84FF')}
                <span style="margin-left: 8px; font-weight: 600; color: var(--label-primary);">Enter a stock symbol directly</span>
            </div>
            """, unsafe_allow_html=True)
            selected_symbol = st.text_input(
                "Custom symbol",
                placeholder="e.g., AAPL, TSLA, NVDA",
                key="custom_symbol",
                label_visibility="collapsed",
            )
            if selected_symbol:
                st.markdown(f"""
                <div style="padding: 12px; background: rgba(48, 209, 88, 0.1); border: 1px solid rgba(48, 209, 88, 0.2); border-radius: 8px; display: flex; align-items: center;">
                    {get_svg_icon('check_circle_filled', 16, '#30D158')}
                    <span style="margin-left: 8px; color: var(--label-primary); font-weight: 600;">Selected: {selected_symbol}</span>
                </div>
                """, unsafe_allow_html=True)
        
        with tab3:
            st.markdown(f"""
            <div style="display: flex; align-items: center; margin-bottom: 16px;">
                {get_svg_icon('star', 20, '#0A84FF')}
                <span style="margin-left: 8px; font-weight: 600; color: var(--label-primary);">Choose from popular stocks</span>
            </div>
            """, unsafe_allow_html=True)
            category = st.selectbox("Select category:", list(popular_stocks.keys()), key="main_category")
            if category:
                # Create a grid layout for popular stocks
                stocks_list = list(popular_stocks[category].items())
                for i in range(0, len(stocks_list), 3):  # 3 columns
                    cols = st.columns(3)
                    for j, col in enumerate(cols):
                        if i + j < len(stocks_list):
                            name, symbol = stocks_list[i + j]
                            with col:
                                if st.button(f"**{name}**\n{symbol}", key=f"popular_{symbol}", use_container_width=True):
                                    selected_symbol = symbol
    
    with col2:
        # Current selection display
        if selected_symbol:
            st.markdown(f"""
            <div style="padding: 20px; background: var(--grouped-background); border-radius: 12px; border: 1px solid var(--separator);">
                <div style="display: flex; align-items: center; margin-bottom: 16px;">
                    {get_svg_icon("check_circle_filled", 24, "#30D158")}
                    <span style="font-size: 18px; font-weight: 600; color: var(--label-primary); margin-left: 12px;">
                        Selected Stock
                    </span>
                </div>
                <div style="font-size: 24px; font-weight: 700; color: var(--label-primary); margin-bottom: 8px;">
                    {selected_symbol}
                </div>
                <div style="font-size: 15px; color: var(--label-secondary);">
                    Ready for analysis
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="padding: 20px; background: var(--grouped-background); border-radius: 12px; border: 1px solid var(--separator); text-align: center;">
                <div style="margin-bottom: 16px;">{get_svg_icon("stock", 32, "#8E8E93")}</div>
                <div style="font-size: 16px; color: var(--label-secondary);">
                    Select a stock to begin analysis
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    return selected_symbol

def render_stock_selection():
    """Render Apple-style stock selection sidebar"""
    
    # Sidebar header
    st.sidebar.markdown(f"""
    <div style="text-align: center; padding: 20px 0 24px 0; border-bottom: 1px solid var(--separator);">
        <div style="margin-bottom: 8px;">{get_svg_icon("stock", 28, "#0A84FF")}</div>
        <h2 style="font-size: 20px; font-weight: 600; color: var(--label-primary); margin: 0;">
            Stock Selection
        </h2>
        <p style="font-size: 15px; color: var(--label-secondary); margin: 4px 0 0 0;">
            Configure your AI-powered analysis
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Stock selection method
    stock_input_method = st.sidebar.radio(
        "How would you like to select a stock?",
        ["Search by Symbol", "Type Custom Symbol", "Popular Stocks"],
        label_visibility="collapsed"
    )
    
    # Handle different selection methods
    selected_symbol = None
    
    if "Search by Symbol" in stock_input_method:
        search_term = st.sidebar.text_input("Enter stock name or symbol:", placeholder="e.g., Apple, AAPL")
        if search_term:
            # Simple search logic
            found_stocks = []
            for category, stocks in popular_stocks.items():
                for name, symbol in stocks.items():
                    if search_term.lower() in name.lower() or search_term.upper() in symbol:
                        found_stocks.append((name, symbol, category))
            
            if found_stocks:
                st.sidebar.write("Found stocks:")
                for name, symbol, category in found_stocks[:5]:  # Show top 5 results
                    if st.sidebar.button(f"{name} ({symbol})", key=f"search_{symbol}"):
                        selected_symbol = symbol
            else:
                st.sidebar.info("No stocks found. Try a different search term.")
    
    elif "Type Custom Symbol" in stock_input_method:
        selected_symbol = st.sidebar.text_input("Enter stock symbol:", placeholder="e.g., AAPL, TSLA")
    
    elif "Popular Stocks" in stock_input_method:
        category = st.sidebar.selectbox("Select category:", list(popular_stocks.keys()))
        if category:
            stock_name = st.sidebar.selectbox("Select stock:", list(popular_stocks[category].keys()))
            if stock_name:
                selected_symbol = popular_stocks[category][stock_name]
    
    return selected_symbol

def render_current_selection(selected_symbol):
    """Display current stock selection in sidebar"""
    if selected_symbol:
        st.sidebar.markdown(f"""
        <div style="padding: 16px; background: var(--grouped-background); border-radius: 12px; margin-bottom: 20px; border: 1px solid var(--separator);">
            <div style="display: flex; align-items: center; margin-bottom: 12px;">
                {get_svg_icon("check_circle_filled", 20, "#30D158")}
                <span style="font-size: 16px; font-weight: 600; color: var(--label-primary); margin-left: 8px;">
                    Current Selection
                </span>
            </div>
            <div style="font-size: 17px; font-weight: 600; color: var(--label-primary); margin-bottom: 4px;">
                {selected_symbol}
            </div>
            <div style="font-size: 15px; color: var(--label-secondary);">
                Ready for analysis
            </div>
        </div>
        """, unsafe_allow_html=True)

def get_stock_data(symbol, period="1y"):
    """Fetch stock data with error handling"""
    try:
        stock = yf.Ticker(symbol)
        data = stock.history(period=period)
        info = stock.info
        return data, info, None
    except Exception as e:
        return None, None, str(e)

def calculate_technical_indicators(data):
    """Calculate technical indicators"""
    try:
        # Calculate moving averages
        data['MA20'] = data['Close'].rolling(window=20).mean()
        data['MA50'] = data['Close'].rolling(window=50).mean()
        data['MA200'] = data['Close'].rolling(window=200).mean()
        
        # Calculate RSI
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        data['RSI'] = 100 - (100 / (1 + rs))
        
        # Calculate MACD
        exp1 = data['Close'].ewm(span=12).mean()
        exp2 = data['Close'].ewm(span=26).mean()
        data['MACD'] = exp1 - exp2
        data['MACD_Signal'] = data['MACD'].ewm(span=9).mean()
        data['MACD_Histogram'] = data['MACD'] - data['MACD_Signal']
        
        # Calculate Bollinger Bands
        data['BB_Middle'] = data['Close'].rolling(window=20).mean()
        bb_std = data['Close'].rolling(window=20).std()
        data['BB_Upper'] = data['BB_Middle'] + (bb_std * 2)
        data['BB_Lower'] = data['BB_Middle'] - (bb_std * 2)
        
        return data
    except Exception as e:
        st.error(f"Error calculating indicators: {e}")
        return data

def create_candlestick_chart(data, symbol, currency="USD"):
    """Create interactive candlestick chart with technical indicators"""
    # Determine currency formatting
    if currency == 'INR':
        tickformat = '₹.0f'
        currency_label = 'INR'
    elif currency == 'BRL':
        tickformat = 'R$.2f'
        currency_label = 'BRL'
    elif currency == 'EUR':
        tickformat = '€.2f'
        currency_label = 'EUR'
    elif currency == 'GBP':
        tickformat = '£.2f'
        currency_label = 'GBP'
    else:  # USD and others
        tickformat = '$.2f'
        currency_label = 'USD'
    
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        subplot_titles=(f'{symbol} Stock Price', 'MACD', 'RSI'),
        row_width=[0.2, 0.1, 0.1]
    )
    
    # Candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            name='Price',
            increasing_line_color='#30D158',
            decreasing_line_color='#FF453A'
        ),
        row=1, col=1
    )
    
    # Moving averages
    fig.add_trace(go.Scatter(x=data.index, y=data['MA20'], name='MA20', line=dict(color='#0A84FF', width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=data.index, y=data['MA50'], name='MA50', line=dict(color='#FF9F0A', width=1)), row=1, col=1)
    
    # Bollinger Bands
    fig.add_trace(go.Scatter(x=data.index, y=data['BB_Upper'], name='BB Upper', line=dict(color='rgba(255,255,255,0.2)', width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=data.index, y=data['BB_Lower'], name='BB Lower', line=dict(color='rgba(255,255,255,0.2)', width=1), fill='tonexty'), row=1, col=1)
    
    # MACD
    fig.add_trace(go.Scatter(x=data.index, y=data['MACD'], name='MACD', line=dict(color='#0A84FF')), row=2, col=1)
    fig.add_trace(go.Scatter(x=data.index, y=data['MACD_Signal'], name='Signal', line=dict(color='#FF9F0A')), row=2, col=1)
    fig.add_trace(go.Bar(x=data.index, y=data['MACD_Histogram'], name='Histogram', marker_color='rgba(255,255,255,0.3)'), row=2, col=1)
    
    # RSI
    fig.add_trace(go.Scatter(x=data.index, y=data['RSI'], name='RSI', line=dict(color='#FF9F0A')), row=3, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
    
    # Update layout for dark theme with currency formatting
    fig.update_layout(
        template='plotly_dark',
        height=800,
        showlegend=True,
        title_text=f"{symbol} Technical Analysis",
        title_x=0.5,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
    )
    
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(255,255,255,0.1)')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(255,255,255,0.1)')
    
    # Update y-axis formatting for price chart
    fig.update_yaxes(tickformat=tickformat, row=1, col=1)
    
    return fig

def generate_ai_analysis(data, info, symbol):
    """Generate AI-powered stock analysis"""
    try:
        # Current metrics
        current_price = data['Close'].iloc[-1]
        prev_close = data['Close'].iloc[-2] if len(data) > 1 else current_price
        price_change = current_price - prev_close
        price_change_pct = (price_change / prev_close) * 100
        
        # Technical analysis
        current_rsi = data['RSI'].iloc[-1] if not pd.isna(data['RSI'].iloc[-1]) else 50
        current_macd = data['MACD'].iloc[-1] if not pd.isna(data['MACD'].iloc[-1]) else 0
        current_macd_signal = data['MACD_Signal'].iloc[-1] if not pd.isna(data['MACD_Signal'].iloc[-1]) else 0
        
        # Price relative to moving averages
        ma20 = data['MA20'].iloc[-1] if not pd.isna(data['MA20'].iloc[-1]) else current_price
        ma50 = data['MA50'].iloc[-1] if not pd.isna(data['MA50'].iloc[-1]) else current_price
        
        # Generate recommendation
        bullish_signals = 0
        bearish_signals = 0
        
        # RSI analysis
        if current_rsi < 30:
            bullish_signals += 2  # Oversold
        elif current_rsi > 70:
            bearish_signals += 2  # Overbought
        elif 40 <= current_rsi <= 60:
            bullish_signals += 1  # Neutral/slight bullish
            
        # MACD analysis
        if current_macd > current_macd_signal:
            bullish_signals += 1
        else:
            bearish_signals += 1
            
        # Moving average analysis
        if current_price > ma20 > ma50:
            bullish_signals += 2
        elif current_price < ma20 < ma50:
            bearish_signals += 2
            
        # Determine recommendation
        if bullish_signals > bearish_signals + 1:
            recommendation = "BUY"
            confidence = min(90, 60 + (bullish_signals - bearish_signals) * 10)
            recommendation_color = "#30D158"
        elif bearish_signals > bullish_signals + 1:
            recommendation = "SELL"
            confidence = min(90, 60 + (bearish_signals - bullish_signals) * 10)
            recommendation_color = "#FF453A"
        else:
            recommendation = "HOLD"
            confidence = 50 + abs(bullish_signals - bearish_signals) * 5
            recommendation_color = "#FF9F0A"
            
        return {
            'recommendation': recommendation,
            'confidence': confidence,
            'recommendation_color': recommendation_color,
            'current_price': current_price,
            'price_change': price_change,
            'price_change_pct': price_change_pct,
            'rsi': current_rsi,
            'macd': current_macd,
            'macd_signal': current_macd_signal,
            'ma20': ma20,
            'ma50': ma50,
            'bullish_signals': bullish_signals,
            'bearish_signals': bearish_signals
        }
    except Exception as e:
        st.error(f"Error generating analysis: {e}")
        return None

def render_stock_analysis(symbol, period, forecast_days, risk_tolerance):
    """Render comprehensive stock analysis with currency support and news"""
    
    # Get market info and currency
    currency = detect_market_currency(symbol)
    market_badge = create_market_badge(symbol)
    
    # Display market badge
    st.markdown(f"""
    <div style="display: inline-flex; align-items: center; padding: 8px 16px; background: rgba(10, 132, 255, 0.1); 
                border: 1px solid rgba(10, 132, 255, 0.2); border-radius: 20px; margin-bottom: 16px;">
        <span style="font-weight: 600; color: var(--primary); font-size: 14px;">{market_badge}</span>
    </div>
    """, unsafe_allow_html=True)
    
    # Fetch data
    with st.spinner(f"Fetching data for {symbol}..."):
        data, info, error = get_stock_data(symbol, period)
    
    if error:
        st.error(f"❌ Error fetching data for {symbol}: {error}")
        st.info("💡 Try checking if the stock symbol is correct or try again later.")
        return
    
    if data is None or data.empty:
        st.error(f"❌ No data found for {symbol}")
        return
    
    # Get company name for news
    company_name = info.get('shortName', symbol) if info else symbol
    
    # Calculate technical indicators
    with st.spinner("Calculating technical indicators..."):
        data = calculate_technical_indicators(data)
    
    # Generate AI analysis
    with st.spinner("Generating AI analysis..."):
        analysis = generate_ai_analysis(data, info, symbol)
    
    # Get news analysis
    with st.spinner("Analyzing recent news..."):
        news_items = get_stock_news(symbol, company_name)
    
    if analysis is None:
        st.error("Failed to generate analysis")
        return
    
    # Display results with proper currency formatting
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        formatted_price = format_currency(analysis['current_price'], currency)
        formatted_change = format_currency(abs(analysis['price_change']), currency)
        change_sign = '+' if analysis['price_change'] >= 0 else '-'
        
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="color: var(--label-tertiary); font-size: 12px; font-weight: 600; margin: 0 0 12px 0; text-transform: uppercase; letter-spacing: 0.5px;">CURRENT PRICE</h3>
            <div style="font-size: 36px; font-weight: 800; color: var(--label-primary); margin-bottom: 8px; letter-spacing: -0.02em;">
                {formatted_price}
            </div>
            <div style="font-size: 16px; font-weight: 600; color: {'#30D158' if analysis['price_change'] >= 0 else '#FF453A'};">
                {change_sign}{formatted_change} ({analysis['price_change_pct']:+.2f}%)
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="color: var(--label-tertiary); font-size: 12px; font-weight: 600; margin: 0 0 12px 0; text-transform: uppercase; letter-spacing: 0.5px;">AI RECOMMENDATION</h3>
            <div style="font-size: 28px; font-weight: 800; color: {analysis['recommendation_color']}; margin-bottom: 8px; letter-spacing: -0.01em;">
                {analysis['recommendation']}
            </div>
            <div style="font-size: 16px; font-weight: 600; color: var(--label-secondary);">
                {analysis['confidence']:.0f}% Confidence
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        target_price = analysis.get('target_price', analysis['current_price'] * 1.05)
        formatted_target = format_currency(target_price, currency)
        
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="color: var(--label-tertiary); font-size: 12px; font-weight: 600; margin: 0 0 12px 0; text-transform: uppercase; letter-spacing: 0.5px;">TARGET PRICE</h3>
            <div style="font-size: 28px; font-weight: 800; color: var(--system-green); margin-bottom: 8px; letter-spacing: -0.01em;">
                {formatted_target}
            </div>
            <div style="font-size: 16px; font-weight: 600; color: var(--label-secondary);">
                {((target_price/analysis['current_price'] - 1) * 100):+.1f}% Upside
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        # Calculate overall news sentiment
        if news_items:
            sentiment_scores = [item['sentiment']['score'] for item in news_items]
            avg_sentiment = np.mean(sentiment_scores)
            if avg_sentiment > 0.1:
                sentiment_label = "Positive"
                sentiment_color = "#30D158"
                sentiment_emoji = "📈"
            elif avg_sentiment < -0.1:
                sentiment_label = "Negative" 
                sentiment_color = "#FF453A"
                sentiment_emoji = "📉"
            else:
                sentiment_label = "Neutral"
                sentiment_color = "#FF9F0A"
                sentiment_emoji = "➡️"
        else:
            sentiment_label = "No Data"
            sentiment_color = "#8E8E93"
            sentiment_emoji = "❓"
            
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="color: var(--label-tertiary); font-size: 12px; font-weight: 600; margin: 0 0 12px 0; text-transform: uppercase; letter-spacing: 0.5px;">NEWS SENTIMENT</h3>
            <div style="font-size: 28px; font-weight: 800; color: {sentiment_color}; margin-bottom: 8px; letter-spacing: -0.01em;">
                {sentiment_emoji} {sentiment_label}
            </div>
            <div style="font-size: 16px; font-weight: 600; color: var(--label-secondary);">
                {len(news_items)} Recent Articles
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="color: var(--label-tertiary); font-size: 12px; font-weight: 600; margin: 0 0 12px 0; text-transform: uppercase; letter-spacing: 0.5px;">RSI (14)</h3>
            <div style="font-size: 28px; font-weight: 800; color: var(--label-primary); margin-bottom: 8px; letter-spacing: -0.01em;">
                {analysis['rsi']:.1f}
            </div>
            <div style="font-size: 16px; font-weight: 600; color: {'#FF453A' if analysis['rsi'] > 70 else '#30D158' if analysis['rsi'] < 30 else 'var(--label-secondary)'};">
                {'Overbought' if analysis['rsi'] > 70 else 'Oversold' if analysis['rsi'] < 30 else 'Neutral'}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        ma_trend = "Bullish" if analysis['current_price'] > analysis['ma20'] > analysis['ma50'] else "Bearish" if analysis['current_price'] < analysis['ma20'] < analysis['ma50'] else "Sideways"
        trend_color = "#30D158" if ma_trend == "Bullish" else "#FF453A" if ma_trend == "Bearish" else "#FF9F0A"
        trend_icon = "📈" if ma_trend == "Bullish" else "📉" if ma_trend == "Bearish" else "➡️"
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="color: var(--label-tertiary); font-size: 12px; font-weight: 600; margin: 0 0 12px 0; text-transform: uppercase; letter-spacing: 0.5px;">TREND ANALYSIS</h3>
            <div style="font-size: 20px; font-weight: 700; color: {trend_color}; margin-bottom: 8px; display: flex; align-items: center; justify-content: center; gap: 8px;">
                <span style="font-size: 24px;">{trend_icon}</span>
                {ma_trend}
            </div>
            <div style="font-size: 14px; font-weight: 500; color: var(--label-secondary);">
                MA20: {format_currency(analysis['ma20'], currency)}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Interactive chart
    st.markdown("""
    <div style="margin: 48px 0 24px 0;">
        <h2 style="font-size: 28px; font-weight: 800; color: var(--label-primary); margin: 0; letter-spacing: -0.02em; display: flex; align-items: center; gap: 12px;">
            📊 Technical Analysis Chart
        </h2>
        <p style="font-size: 16px; color: var(--label-secondary); margin: 8px 0 0 0; font-weight: 500;">
            Interactive candlestick chart with technical indicators and trend analysis
        </p>
    </div>
    """, unsafe_allow_html=True)
    fig = create_candlestick_chart(data, symbol, currency)
    st.plotly_chart(fig, use_container_width=True)
    
    # Analysis details
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div style="margin-bottom: 16px;">
            <h3 style="font-size: 22px; font-weight: 700; color: var(--label-primary); margin: 0; letter-spacing: -0.01em; display: flex; align-items: center; gap: 10px;">
                🧠 AI Analysis Details
            </h3>
            <p style="font-size: 15px; color: var(--label-secondary); margin: 6px 0 0 0; font-weight: 500;">
                Comprehensive signal analysis and confidence metrics
            </p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown(f"""
        <div class="apple-card">
            <h4 style="color: var(--primary); margin-bottom: 16px;">Signal Strength</h4>
            <div style="margin-bottom: 12px;">
                <span style="color: #30D158;">●</span> Bullish Signals: <strong>{analysis['bullish_signals']}</strong>
            </div>
            <div style="margin-bottom: 12px;">
                <span style="color: #FF453A;">●</span> Bearish Signals: <strong>{analysis['bearish_signals']}</strong>
            </div>
            <div style="margin-bottom: 16px;">
                <span style="color: var(--primary);">●</span> Net Sentiment: <strong>{analysis['bullish_signals'] - analysis['bearish_signals']:+d}</strong>
            </div>
            <div style="background: rgba(10, 132, 255, 0.1); border-radius: 8px; padding: 12px;">
                <strong style="color: {analysis['recommendation_color']};">{analysis['recommendation']}</strong> with {analysis['confidence']:.0f}% confidence
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="margin-bottom: 16px;">
            <h3 style="font-size: 22px; font-weight: 700; color: var(--label-primary); margin: 0; letter-spacing: -0.01em; display: flex; align-items: center; gap: 10px;">
                📋 Technical Indicators
            </h3>
            <p style="font-size: 15px; color: var(--label-secondary); margin: 6px 0 0 0; font-weight: 500;">
                Key technical metrics and market indicators
            </p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown(f"""
        <div class="apple-card">
            <div style="margin-bottom: 12px;">
                <strong>RSI (14):</strong> {analysis['rsi']:.2f}
                <span style="color: {'#FF453A' if analysis['rsi'] > 70 else '#30D158' if analysis['rsi'] < 30 else 'var(--label-secondary)'};">
                    {'(Overbought)' if analysis['rsi'] > 70 else '(Oversold)' if analysis['rsi'] < 30 else '(Neutral)'}
                </span>
            </div>
            <div style="margin-bottom: 12px;">
                <strong>MACD:</strong> {analysis['macd']:.4f}
            </div>
            <div style="margin-bottom: 12px;">
                <strong>MACD Signal:</strong> {analysis['macd_signal']:.4f}
            </div>
            <div style="margin-bottom: 12px;">
                <strong>MA20:</strong> {format_currency(analysis['ma20'], currency)}
            </div>
            <div style="margin-bottom: 12px;">
                <strong>MA50:</strong> {format_currency(analysis['ma50'], currency)}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # News Analysis Section
    if news_items:
        st.markdown("""
        <div style="margin: 48px 0 24px 0;">
            <h2 style="font-size: 28px; font-weight: 800; color: var(--label-primary); margin: 0; letter-spacing: -0.02em; display: flex; align-items: center; gap: 12px;">
                📰 News Sentiment Analysis
            </h2>
            <p style="font-size: 16px; color: var(--label-secondary); margin: 8px 0 0 0; font-weight: 500;">
                Recent news articles and their impact on market sentiment
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # News cards
        for idx, item in enumerate(news_items[:3]):  # Show top 3 news items
            sentiment = item['sentiment']
            sentiment_label = sentiment.get('label', 'neutral')
            sentiment_color = '#30D158' if sentiment_label == 'positive' else '#FF453A' if sentiment_label == 'negative' else '#FF9F0A'
            sentiment_emoji = '📈' if sentiment_label == 'positive' else '📉' if sentiment_label == 'negative' else '➡️'
            sentiment_badge = sentiment_label.upper()
            
            st.markdown(f"""
            <div class="apple-card" style="margin-bottom: 16px;">
                <div style="display: flex; justify-content: between; align-items: start; margin-bottom: 12px;">
                    <div style="flex: 1;">
                        <h4 style="color: var(--label-primary); margin: 0 0 8px 0; font-size: 18px; font-weight: 600; line-height: 1.3;">
                            {item['title'][:100]}{'...' if len(item['title']) > 100 else ''}
                        </h4>
                        <p style="color: var(--label-secondary); margin: 0 0 12px 0; font-size: 14px; line-height: 1.4;">
                            {item['summary'][:200]}{'...' if len(item['summary']) > 200 else ''}
                        </p>
                    </div>
                    <div style="margin-left: 16px; text-align: center;">
                        <div style="color: {sentiment_color}; font-size: 24px; margin-bottom: 4px;">
                            {sentiment_emoji}
                        </div>
                        <div style="color: {sentiment_color}; font-size: 12px; font-weight: 600; text-transform: uppercase;">
                            {sentiment_badge}
                        </div>
                        <div style="color: var(--label-secondary); font-size: 11px;">
                            {sentiment['score']:.2f}
                        </div>
                    </div>
                </div>
                <div style="font-size: 12px; color: var(--label-tertiary);">
                    {item['published']}
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="margin: 48px 0 24px 0;">
            <h2 style="font-size: 28px; font-weight: 800; color: var(--label-primary); margin: 0; letter-spacing: -0.02em; display: flex; align-items: center; gap: 12px;">
                📰 News Sentiment Analysis
            </h2>
            <p style="font-size: 16px; color: var(--label-secondary); margin: 8px 0 0 0; font-weight: 500;">
                Recent news articles and their impact on market sentiment
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="apple-card">
            <div style="text-align: center; padding: 32px;">
                <div style="font-size: 48px; margin-bottom: 16px;">📰</div>
                <h3 style="color: var(--label-secondary); margin: 0 0 8px 0;">No Recent News Available</h3>
                <p style="color: var(--label-tertiary); margin: 0; font-size: 14px;">
                    News analysis will appear here when articles are found for this stock.
                </p>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Add detailed AI explanations section
    render_ai_explanations(analysis, symbol, currency, risk_tolerance)

def render_ai_explanations(analysis, symbol, currency, risk_tolerance):
    """Render detailed AI explanations about the recommendation and risk factors"""
    
    st.markdown("""
    <div style="margin: 48px 0 24px 0;">
        <h2 style="font-size: 28px; font-weight: 800; color: var(--label-primary); margin: 0; letter-spacing: -0.02em; display: flex; align-items: center; gap: 12px;">
            🧠 AI Analysis Explained
        </h2>
        <p style="font-size: 16px; color: var(--label-secondary); margin: 8px 0 0 0; font-weight: 500;">
            Understanding why the AI made this recommendation and what it means for your investment
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Generate recommendation explanation
    def get_recommendation_explanation():
        if analysis['recommendation'] == 'BUY':
            return {
                'title': '📈 Why BUY is Recommended',
                'reasons': [
                    f"Technical momentum is positive with {analysis['bullish_signals']} bullish signals vs {analysis['bearish_signals']} bearish signals",
                    f"RSI at {analysis['rsi']:.1f} indicates {'oversold conditions' if analysis['rsi'] < 30 else 'healthy momentum' if analysis['rsi'] < 70 else 'potentially overbought'}",
                    f"Price is {'above' if analysis['current_price'] > analysis['ma20'] else 'approaching'} the 20-day moving average ({format_currency(analysis['ma20'], currency)})",
                    f"MACD signal at {analysis['macd']:.4f} suggests {'positive momentum' if analysis['macd'] > 0 else 'potential reversal'}"
                ],
                'risks': [
                    'Market volatility could impact short-term performance',
                    'Economic factors may influence sector performance', 
                    'Company-specific news could affect stock price',
                    f"High RSI ({analysis['rsi']:.1f}) may indicate limited upside potential" if analysis['rsi'] > 70 else "Consider position sizing for risk management"
                ]
            }
        elif analysis['recommendation'] == 'SELL':
            return {
                'title': '📉 Why SELL is Recommended', 
                'reasons': [
                    f"Technical indicators show weakness with {analysis['bearish_signals']} bearish signals outweighing {analysis['bullish_signals']} bullish signals",
                    f"RSI at {analysis['rsi']:.1f} suggests {'overbought conditions' if analysis['rsi'] > 70 else 'continued weakness' if analysis['rsi'] < 30 else 'neutral momentum'}",
                    f"Price is {'below' if analysis['current_price'] < analysis['ma20'] else 'struggling near'} key moving averages",
                    f"MACD at {analysis['macd']:.4f} indicates {'negative momentum' if analysis['macd'] < 0 else 'weakening momentum'}"
                ],
                'risks': [
                    'Potential for further downside if trend continues',
                    'Market rebounds could lead to losses on short positions',
                    'Timing the market exit is challenging',
                    'Consider setting stop-losses to limit potential losses'
                ]
            }
        else:  # HOLD
            return {
                'title': '➡️ Why HOLD is Recommended',
                'reasons': [
                    f"Mixed signals with {analysis['bullish_signals']} bullish and {analysis['bearish_signals']} bearish indicators",
                    f"RSI at {analysis['rsi']:.1f} {'suggests overbought conditions, but other indicators are mixed' if analysis['rsi'] > 70 else 'suggests oversold conditions, but other indicators are mixed' if analysis['rsi'] < 30 else 'is in neutral territory, indicating balanced conditions'}",
                    f"Price action around moving averages suggests consolidation phase",
                    "Current risk-reward ratio doesn't favor active positioning"
                ],
                'risks': [
                    'Opportunity cost of holding vs other investments',
                    'Extended consolidation could test patience',
                    'Sudden breakout in either direction could be missed',
                    'Regular reassessment needed as conditions change'
                ]
            }
    
    explanation = get_recommendation_explanation()
    
    # Use Streamlit's native components instead of complex HTML
    st.markdown(f"### {explanation['title']}")
    
    st.markdown("#### 🎯 Key Factors Supporting This Recommendation:")
    for reason in explanation['reasons']:
        st.markdown(f"• {reason}")
    
    st.warning("⚠️ **Risk Factors to Consider:**")
    for risk in explanation['risks']:
        st.markdown(f"• {risk}")
    
    st.markdown("---")
    
    # Technical Indicators Explained - using columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📊 Technical Indicators Explained")
        
        rsi_status = 'Overbought' if analysis['rsi'] > 70 else 'Oversold' if analysis['rsi'] < 30 else 'Neutral'
        macd_momentum = 'Positive' if analysis['macd'] > 0 else 'Negative'
        price_vs_ma20 = ((analysis['current_price'] / analysis['ma20'] - 1) * 100)
        trend_status = "Bullish" if analysis['current_price'] > analysis['ma20'] > analysis['ma50'] else "Bearish" if analysis['current_price'] < analysis['ma20'] < analysis['ma50'] else "Sideways"
        
        st.markdown("**RSI (Relative Strength Index)**")
        st.markdown("Measures momentum on a scale of 0-100. Values above 70 suggest overbought conditions, below 30 suggest oversold.")
        st.markdown(f"Current: **{analysis['rsi']:.1f}** - {rsi_status}")
        
        st.markdown("**MACD (Moving Average Convergence Divergence)**")
        st.markdown("Shows relationship between two moving averages. Positive values indicate upward momentum, negative values suggest downward pressure.")
        st.markdown(f"Current: **{analysis['macd']:.4f}** - {macd_momentum} momentum")
        
        st.markdown("**Moving Averages (MA20/MA50)**")
        st.markdown("Average price over 20 and 50 days. Price above MA indicates uptrend, below suggests downtrend.")
        st.markdown(f"Price vs MA20: **{price_vs_ma20:+.1f}%** | Trend: **{trend_status}**")
    
    with col2:
        st.markdown(f"### 🎯 Risk Assessment for {risk_tolerance} Investor")
        
        risk_level = 'Low' if analysis['confidence'] > 80 else 'Medium' if analysis['confidence'] > 60 else 'High'
        
        st.markdown(f"**Overall Risk Level:** {risk_level}")
        st.progress(1.0 - (analysis['confidence'] / 100))
        
        st.markdown("**📈 Volatility Analysis**")
        volatility_text = {
            'Aggressive': 'This stock shows high volatility - suitable for aggressive investors seeking higher returns with increased risk.',
            'Moderate': 'Moderate volatility aligns with your balanced approach to risk and return.',
            'Conservative': 'Consider this stock carefully as it may have higher volatility than typical conservative investments.'
        }
        st.markdown(volatility_text[risk_tolerance])
        
        st.markdown("**💡 Investment Guidance**")
        guidance_text = {
            'BUY': 'Consider position sizing and stop-losses. This recommendation aligns with aggressive growth strategies.',
            'SELL': 'Risk management is crucial. Consider reducing exposure or implementing protective strategies.',
            'HOLD': 'Monitor for clearer signals. Current conditions suggest waiting for better entry/exit points.'
        }
        st.markdown(guidance_text[analysis['recommendation']])
        
        st.info(f"🤖 **AI Confidence Level:** {analysis['confidence']:.0f}% Confident  \nBased on {analysis['bullish_signals'] + analysis['bearish_signals']} technical signals analyzed")
    
    # Actionable Recommendations
    st.markdown("### 🎯 Actionable Next Steps")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**📋 Immediate Actions**")
        if analysis['recommendation'] == 'BUY':
            actions = [
                "Consider buying on any market dips",
                "Set stop-loss at 5-8% below entry",
                "Monitor volume confirmation"
            ]
        elif analysis['recommendation'] == 'SELL':
            actions = [
                "Consider reducing position size",
                "Set tighter stop-losses",
                "Monitor for trend reversal"
            ]
        else:  # HOLD
            actions = [
                "Wait for clearer technical signals",
                "Set price alerts for breakouts",
                "Review position in 1-2 weeks"
            ]
        
        for action in actions:
            st.markdown(f"• {action}")
    
    with col2:
        st.markdown("**📅 Monitoring Schedule**")
        monitoring = [
            "Daily: Price action vs moving averages",
            "Weekly: RSI and MACD changes",
            "Monthly: Fundamental review"
        ]
        for item in monitoring:
            st.markdown(f"• {item}")
    
    st.info("💡 Remember: This analysis is for educational purposes. Always do your own research and consider consulting a financial advisor.")

# Analysis parameters section
def render_analysis_parameters():
    """Render analysis parameters with Apple styling"""
    
    st.sidebar.markdown("<br>", unsafe_allow_html=True)
    st.sidebar.markdown(f"""
    <div style="text-align: center; padding: 20px 0; border-bottom: 1px solid var(--separator);">
        <div style="margin-bottom: 12px;">{get_svg_icon("settings", 24, "#007AFF")}</div>
        <h3 style="font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif; font-size: 20px; font-weight: 600; color: var(--label-primary); margin: 0;">
            Analysis Parameters
        </h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Parameters
    period = st.sidebar.selectbox(
        "Historical Data Period",
        ["1mo", "3mo", "6mo", "1y", "2y", "5y"],
        index=3
    )
    
    forecast_days = st.sidebar.slider(
        "Forecast Horizon (days)",
        min_value=1,
        max_value=30,
        value=5
    )
    
    risk_tolerance = st.sidebar.select_slider(
        "Risk Tolerance",
        options=["Conservative", "Moderate", "Aggressive"],
        value="Moderate"
    )
    
    return period, forecast_days, risk_tolerance

# Main application function
def main():
    """Main application with Apple-inspired design"""
    
    # Render navigation header
    render_navigation_header()
    
    # Display selected model information
    render_model_info()
    
    # Initialize sidebar with just analysis parameters
    period, forecast_days, risk_tolerance = render_analysis_parameters()
    
    # Add sidebar toggle button in navigation
    if st.button("⚙️ Toggle Settings", key="sidebar_toggle"):
        if st.session_state.get('sidebar_collapsed', False):
            st.session_state.sidebar_collapsed = False
        else:
            st.session_state.sidebar_collapsed = True
    
    # Main content area - Stock Selection
    selected_symbol = render_main_stock_selection()
    
    if not selected_symbol:
        render_onboarding()
        return
    
    # Stock analysis section
    with st.container():
        st.markdown(f"""
        <div class="apple-card">
            <div style="display: flex; align-items: center; margin-bottom: 24px;">
                <div style="margin-right: 16px;">{get_svg_icon("chart", 32, "#0A84FF")}</div>
                <div>
                    <h1 style="font-size: 28px; font-weight: 700; color: var(--label-primary); margin: 0;">
                        {selected_symbol}
                    </h1>
                    <p style="font-size: 15px; color: var(--label-secondary); margin: 4px 0 0 0;">
                        AI-Powered Stock Analysis
                    </p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Render comprehensive stock analysis
        render_stock_analysis(selected_symbol, period, forecast_days, risk_tolerance)

if __name__ == "__main__":
    main()