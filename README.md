# 🚀 Machine Learning Nifty - AI Stock Analysis Platform

**Comprehensive AI-powered stock analysis with explainable buy/sell/hold recommendations**

## ⚡ Quick Start - One Command Launch

```bash
python run_system.py
```

This will automatically:
- Check and install dependencies
- Launch the Streamlit UI
- Open your browser to http://localhost:8501

## 🎯 What This System Does

This is an **AI-powered stock analysis platform** that provides intelligent investment recommendations using:

- 🤖 **Advanced AI Models** - Multimodal transformers with 19.6M parameters
- 📊 **Technical Analysis** - 20+ indicators (RSI, MACD, Bollinger Bands, etc.)
- 🧠 **Explainable AI** - Detailed reasoning for every recommendation
- 🌍 **Multi-Market Support** - US, Indian, Crypto, and Global stocks
- ⚡ **Real-Time Data** - Live analysis with Yahoo Finance integration
- 📈 **Risk Assessment** - Comprehensive risk analysis and position sizing

### Key Features

- **Buy/Sell/Hold Recommendations** with confidence scores (0-100%)
- **Interactive Charts** with technical indicators and overlays
- **Risk Analysis** with volatility assessment and position sizing
- **Multi-Market Coverage** - US stocks, Indian (.NS), Crypto (-USD), Indices
- **Explainable Decisions** - See exactly why the AI made each recommendation
- **Real-Time Processing** - Analysis completed in under 5 seconds

## 📊 How It Works

### 1. Select a Stock
- **Search by name or symbol** (e.g., Apple, AAPL)
- **Choose from popular stocks** (categorized by market)
- **Enter custom symbols** (supports global markets)

### 2. Configure Analysis
- **Data period**: 1 month to 1 year
- **Forecast horizon**: 1-30 days
- **Risk tolerance**: Conservative, Moderate, Aggressive

### 3. Get AI Recommendations
- **Signal**: STRONG BUY, BUY, HOLD, SELL, STRONG SELL
- **Confidence**: How certain the AI is (0-100%)
- **Target Price**: Expected price movement
- **Stop Loss**: Risk management level
- **Detailed Reasoning**: Why the AI made this decision

## �️ Installation & Setup

### Requirements
- Python 3.8+
- Internet connection (for live data)
- 4GB+ RAM recommended

### Manual Installation
```bash
# 1. Clone repository
git clone https://github.com/elecxDev/machine-learning-nifty.git
cd machine-learning-nifty

# 2. Create virtual environment
python -m venv venv

# 3. Activate environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Run application
streamlit run main_app.py
```

## � Supported Markets & Symbols

### US Stocks
- Apple (AAPL), Microsoft (MSFT), Google (GOOGL)
- Tesla (TSLA), Amazon (AMZN), Meta (META)
- All major US exchanges (NYSE, NASDAQ)

### Indian Stocks  
- Reliance (RELIANCE.NS), TCS (TCS.NS), HDFC Bank (HDFCBANK.NS)
- Infosys (INFY.NS), ICICI Bank (ICICIBANK.NS)
- All NSE and BSE listed stocks (use .NS suffix)

### Cryptocurrency
- Bitcoin (BTC-USD), Ethereum (ETH-USD)
- Binance Coin (BNB-USD), Cardano (ADA-USD)
- 100+ crypto pairs (use -USD suffix)

### Market Indices
- S&P 500 (^GSPC), NASDAQ (^IXIC)
- NIFTY 50 (^NSEI), SENSEX (^BSESN)
- Global indices with ^ prefix

## 🧠 AI Technology Stack

### Model Architecture
- **Unified Multimodal Transformer** - 19.6M parameters
- **Cross-modal attention** - Processes price, volume, and news data
- **Multi-task learning** - Simultaneous forecasting and risk assessment
- **Transfer learning** - Adapts across different markets and assets

### Data Sources (100% Free)
- **Yahoo Finance** - Real-time stock data, OHLCV, company info
- **Technical Indicators** - RSI, MACD, Bollinger Bands, Moving Averages
- **Volume Analysis** - Trading volume patterns and anomalies
- **Market Sentiment** - Built-in sentiment analysis capabilities

### Analysis Methods
- **Trend Analysis** - Moving averages, golden/death crosses
- **Momentum Indicators** - RSI, MACD, Stochastic oscillators  
- **Volatility Assessment** - Risk measurement and classification
- **Pattern Recognition** - Support/resistance, chart patterns
- **Volume Confirmation** - Volume-price relationship analysis

## 📊 Understanding the Results

### Recommendation Signals
- **STRONG BUY** - High confidence positive outlook (Score ≥ 2.0)
- **BUY** - Moderate positive outlook (Score 1.0-2.0)
- **HOLD** - Neutral outlook (Score -1.0 to 1.0)
- **SELL** - Moderate negative outlook (Score -2.0 to -1.0)
- **STRONG SELL** - High confidence negative outlook (Score ≤ -2.0)

### Confidence Levels
- **90%+** - Very high confidence, strong signals across multiple indicators
- **70-90%** - High confidence, multiple supporting factors
- **50-70%** - Moderate confidence, mixed signals
- **Below 50%** - Low confidence, uncertain market conditions

### Risk Assessment
- **LOW** - Volatility <20%, stable price action, suitable for conservative investors
- **MEDIUM** - Volatility 20-40%, normal fluctuations, moderate risk
- **HIGH** - Volatility >40%, significant price swings, high risk

## 🎯 Use Cases

### Individual Investors
- Get data-driven investment recommendations
- Understand market trends with AI explanations
- Manage risk with position sizing suggestions
- Track multiple stocks across different markets

### Financial Professionals
- Generate investment signals for clients
- Conduct technical analysis with AI assistance
- Cross-market opportunity identification
- Risk management and portfolio optimization

### Researchers & Students
- Study financial market patterns
- Learn about technical analysis indicators
- Understand AI decision-making in finance
- Analyze cross-market correlations

## 🔧 Advanced Features

### Full-Stack Architecture
The system includes multiple components:
- **Main Application** (`main_app.py`) - Primary Streamlit interface
- **API Backend** (`api/main.py`) - FastAPI for programmatic access
- **Advanced Dashboard** (`frontend/dashboard.py`) - Professional interface
- **Training Pipeline** (`scripts/`) - Model training and evaluation

### Running Full Stack
```bash
# Terminal 1: Start API
cd api && python main.py

# Terminal 2: Start Dashboard  
streamlit run frontend/dashboard.py

# Access:
# API: http://localhost:8000
# Dashboard: http://localhost:8501
```

## 🤝 Contributing

We welcome contributions! Areas for improvement:
- Additional technical indicators
- New market data sources
- Enhanced visualization features
- Performance optimizations
- Mobile-responsive design

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with PyTorch, Transformers, Streamlit, and Plotly
- Data provided by Yahoo Finance APIs
- Inspired by quantitative finance and explainable AI research
- Thanks to the open-source ML and finance communities

## 📞 Support & Links

- **🔗 Repository**: [GitHub](https://github.com/elecxDev/machine-learning-nifty)
- **🐛 Issues**: [Report Problems](https://github.com/elecxDev/machine-learning-nifty/issues)
- **📖 Documentation**: Comprehensive inline documentation included
- **💬 Discussions**: GitHub Discussions for questions and ideas

---

⭐ **Star this repository if you find it useful!** ⭐

🚀 **Ready to analyze stocks with AI? Run `python run_system.py` to get started!**
