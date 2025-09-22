#!/usr/bin/env python3
"""
Multi-Currency Support Enhancement
Adds proper currency display and conversion for different markets
"""

import yfinance as yf
import streamlit as st
import requests
from datetime import datetime, timedelta

# Currency mappings for different markets
MARKET_CURRENCIES = {
    # US Market
    'US': 'USD',
    'USA': 'USD',
    
    # Indian Market - NSE/BSE stocks end with .NS/.BO
    'INDIA': 'INR',
    'NS': 'INR',  # NSE suffix
    'BO': 'INR',  # BSE suffix
    
    # Brazilian Market - B3 stocks end with .SA
    'BRAZIL': 'BRL',
    'SA': 'BRL',  # B3 suffix
    
    # UK Market
    'UK': 'GBP',
    'L': 'GBP',   # London suffix
    
    # European Markets
    'DE': 'EUR',  # Germany
    'PA': 'EUR',  # Paris
    
    # Other Markets
    'TO': 'CAD',  # Toronto
    'T': 'CAD',   # TSX
    'HK': 'HKD',  # Hong Kong
    'TYO': 'JPY', # Tokyo
    
    # Crypto (always USD)
    'CRYPTO': 'USD',
    'BTC': 'USD',
    'ETH': 'USD'
}

# Currency symbols for display
CURRENCY_SYMBOLS = {
    'USD': '$',
    'INR': '₹',
    'BRL': 'R$',
    'GBP': '£',
    'EUR': '€',
    'CAD': 'C$',
    'HKD': 'HK$',
    'JPY': '¥',
    'CNY': '¥'
}

def detect_market_currency(symbol):
    """Detect the currency for a given stock symbol"""
    
    # Handle crypto currencies
    if '-USD' in symbol or symbol in ['BTC-USD', 'ETH-USD', 'ADA-USD']:
        return 'USD'
    
    # Handle indices
    if symbol.startswith('^'):
        if symbol in ['^NSEI', '^BSESN']:  # Indian indices
            return 'INR'
        elif symbol in ['^BVSP']:  # Brazilian indices
            return 'BRL'
        else:  # Most other indices are USD
            return 'USD'
    
    # Handle market suffixes
    if '.' in symbol:
        suffix = symbol.split('.')[-1]
        return MARKET_CURRENCIES.get(suffix, 'USD')
    
    # Default to USD for US stocks
    return 'USD'

def get_currency_symbol(currency_code):
    """Get currency symbol for display"""
    return CURRENCY_SYMBOLS.get(currency_code, currency_code)

def format_currency(amount, currency_code):
    """Format amount with proper currency symbol and formatting"""
    symbol = get_currency_symbol(currency_code)
    
    if currency_code == 'INR':
        # Indian number formatting (lakhs/crores)
        if amount >= 10000000:  # 1 crore
            return f"{symbol}{amount/10000000:.2f}Cr"
        elif amount >= 100000:  # 1 lakh
            return f"{symbol}{amount/100000:.2f}L"
        else:
            return f"{symbol}{amount:,.2f}"
    
    elif currency_code == 'JPY':
        # Japanese Yen (no decimals)
        return f"{symbol}{amount:,.0f}"
    
    else:
        # Standard formatting for USD, EUR, etc.
        if amount >= 1000000:
            return f"{symbol}{amount/1000000:.2f}M"
        elif amount >= 1000:
            return f"{symbol}{amount/1000:.1f}K"
        else:
            return f"{symbol}{amount:.2f}"

def get_market_info(symbol):
    """Get market and currency information for a symbol"""
    currency = detect_market_currency(symbol)
    
    market_info = {
        'currency': currency,
        'currency_symbol': get_currency_symbol(currency),
        'market_name': get_market_name(symbol),
        'exchange': get_exchange_name(symbol)
    }
    
    return market_info

def get_market_name(symbol):
    """Get human-readable market name"""
    if '.' in symbol:
        suffix = symbol.split('.')[-1]
        market_names = {
            'NS': 'National Stock Exchange (India)',
            'BO': 'Bombay Stock Exchange (India)', 
            'SA': 'B3 (Brazil)',
            'L': 'London Stock Exchange (UK)',
            'TO': 'Toronto Stock Exchange (Canada)',
            'HK': 'Hong Kong Exchange',
            'T': 'Tokyo Stock Exchange',
            'PA': 'Euronext Paris',
            'DE': 'Frankfurt Stock Exchange'
        }
        return market_names.get(suffix, 'International Market')
    
    elif '-USD' in symbol:
        return 'Cryptocurrency Market'
    elif symbol.startswith('^'):
        return 'Market Index'
    else:
        return 'United States Market'

def get_exchange_name(symbol):
    """Get exchange abbreviation"""
    if '.' in symbol:
        suffix = symbol.split('.')[-1]
        return suffix.upper()
    elif '-USD' in symbol:
        return 'CRYPTO'
    elif symbol.startswith('^'):
        return 'INDEX'
    else:
        return 'US'

def display_price_with_currency(price, symbol, label="Price"):
    """Display price with proper currency formatting"""
    market_info = get_market_info(symbol)
    formatted_price = format_currency(price, market_info['currency'])
    
    return f"**{label}**: {formatted_price} ({market_info['currency']})"

def create_market_badge(symbol):
    """Create a market badge for display"""
    market_info = get_market_info(symbol)
    
    # Market flag emojis
    market_flags = {
        'United States Market': '🇺🇸',
        'National Stock Exchange (India)': '🇮🇳',
        'Bombay Stock Exchange (India)': '🇮🇳',
        'B3 (Brazil)': '🇧🇷',
        'London Stock Exchange (UK)': '🇬🇧',
        'Toronto Stock Exchange (Canada)': '🇨🇦',
        'Hong Kong Exchange': '🇭🇰',
        'Tokyo Stock Exchange': '🇯🇵',
        'Euronext Paris': '🇫🇷',
        'Frankfurt Stock Exchange': '🇩🇪',
        'Cryptocurrency Market': '₿',
        'Market Index': '📊'
    }
    
    flag = market_flags.get(market_info['market_name'], '🌍')
    
    return f"{flag} {market_info['exchange']} • {market_info['currency']}"

# Updated popular stocks with proper currency awareness
def get_popular_stocks_by_market():
    """Get popular stocks organized by market with currency info"""
    return {
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

# Enhanced stock analysis with currency awareness
def analyze_stock_with_currency(symbol, data):
    """Analyze stock with proper currency formatting"""
    
    if data is None or data.empty:
        return None
    
    market_info = get_market_info(symbol)
    current_price = data['Close'].iloc[-1]
    
    # Calculate key metrics
    price_change = data['Close'].iloc[-1] - data['Close'].iloc[-2]
    price_change_pct = (price_change / data['Close'].iloc[-2]) * 100
    
    # Volume analysis with currency-aware formatting
    avg_volume = data['Volume'].tail(20).mean()
    current_volume = data['Volume'].iloc[-1]
    
    analysis = {
        'symbol': symbol,
        'market_info': market_info,
        'current_price': current_price,
        'formatted_price': format_currency(current_price, market_info['currency']),
        'price_change': price_change,
        'price_change_pct': price_change_pct,
        'formatted_change': format_currency(abs(price_change), market_info['currency']),
        'volume': current_volume,
        'avg_volume': avg_volume,
        'market_badge': create_market_badge(symbol)
    }
    
    return analysis

def display_currency_info():
    """Display currency information and conversion disclaimer"""
    
    st.markdown("""
    ### 💱 Multi-Currency Support
    
    **Supported Markets & Currencies:**
    - 🇺🇸 **US Stocks**: Displayed in USD ($)
    - 🇮🇳 **Indian Stocks**: Displayed in INR (₹) with Lakh/Crore formatting
    - 🇧🇷 **Brazilian Stocks**: Displayed in BRL (R$)
    - ₿ **Cryptocurrency**: Displayed in USD ($)
    - 📊 **Indices**: Displayed in local currency
    
    **Currency Formatting:**
    - **Indian**: ₹1.5L (1.5 Lakhs), ₹2.3Cr (2.3 Crores)
    - **US/Crypto**: $1.5K, $2.3M, $4.2B
    - **Brazilian**: R$1.5K, R$2.3M
    
    > 📝 **Note**: Prices are shown in original market currency. 
    > Cross-currency analysis uses market-specific formatting.
    """)

# Example usage function
def example_usage():
    """Example of how to use the currency system"""
    
    # Test various symbols
    test_symbols = [
        'AAPL',           # US - USD
        'RELIANCE.NS',    # India - INR
        'PETR4.SA',       # Brazil - BRL
        'BTC-USD',        # Crypto - USD
        '^NSEI'           # Indian Index - INR
    ]
    
    st.markdown("## 🧪 Currency System Test")
    
    for symbol in test_symbols:
        market_info = get_market_info(symbol)
        badge = create_market_badge(symbol)
        
        # Simulate price
        test_price = 1500000 if market_info['currency'] == 'INR' else 150.50
        formatted_price = format_currency(test_price, market_info['currency'])
        
        st.markdown(f"""
        **{symbol}** {badge}
        - Market: {market_info['market_name']}
        - Currency: {market_info['currency']} ({market_info['currency_symbol']})
        - Sample Price: {formatted_price}
        """)

if __name__ == "__main__":
    st.set_page_config(page_title="Multi-Currency Test", page_icon="💱")
    st.title("💱 Multi-Currency Support System")
    
    display_currency_info()
    example_usage()