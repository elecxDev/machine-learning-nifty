"""
FastAPI Backend for Unified Multimodal Transformer
Real-time financial forecasting API with explainability
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import numpy as np
import pandas as pd
import yfinance as yf
from typing import List, Dict, Optional
import sys
import os
from datetime import datetime, timedelta
import pickle

# Add src to path
current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, current_dir)
sys.path.insert(0, os.path.join(current_dir, 'src'))

try:
    from src.models.unified_transformer import UnifiedMultimodalTransformer, ModelConfig
except ImportError:
    print("Warning: Model not available. Running in demo mode.")
    UnifiedMultimodalTransformer = None
    ModelConfig = None

app = FastAPI(
    title="Unified Multimodal Financial Forecasting API",
    description="Cross-market financial predictions with explainability",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model instance
model = None
config = None

class PredictionRequest(BaseModel):
    symbol: str
    days_back: int = 60
    forecast_days: int = 5

class PredictionResponse(BaseModel):
    symbol: str
    current_price: float
    forecast: List[float]
    forecast_dates: List[str]
    anomaly_score: float
    confidence: float
    directional_signal: str
    technical_indicators: Dict[str, float]

class ExplainabilityRequest(BaseModel):
    symbol: str
    prediction_id: Optional[str] = None

class ExplainabilityResponse(BaseModel):
    symbol: str
    feature_importance: Dict[str, float]
    attention_weights: List[List[float]]
    explanation_text: str

class MarketOverviewResponse(BaseModel):
    timestamp: str
    markets: Dict[str, Dict[str, float]]
    top_predictions: List[Dict[str, float]]
    market_sentiment: str

def load_model():
    """Load the trained model"""
    global model, config
    
    if UnifiedMultimodalTransformer is None:
        print("Model class not available. Running in demo mode.")
        return False
    
    try:
        # Try to load trained model
        model_path = os.path.join(current_dir, 'models', 'unified_transformer_trained.pt')
        
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location='cpu')
            config = checkpoint['config']
            
            # Initialize model
            model = UnifiedMultimodalTransformer(config)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            
            print("Trained model loaded successfully")
            return True
        else:
            # Create demo model
            config = ModelConfig()
            model = UnifiedMultimodalTransformer(config)
            model.eval()
            
            print("Demo model created (no trained weights)")
            return True
        
    except Exception as e:
        print(f"Error loading model: {e}")
        return False

def get_real_time_data(symbol: str, days_back: int = 60):
    """Get real-time market data for a symbol"""
    
    try:
        ticker = yf.Ticker(symbol)
        
        # Get historical data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back + 30)  # Extra buffer
        
        data = ticker.history(start=start_date, end=end_date)
        
        if data.empty:
            raise ValueError(f"No data available for {symbol}")
        
        # Add technical indicators
        data = add_technical_indicators(data)
        
        return data.tail(days_back)  # Return last N days
        
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Error fetching data for {symbol}: {str(e)}")

def add_technical_indicators(df):
    """Add technical indicators to price data"""
    
    # Returns
    df['returns'] = df['Close'].pct_change()
    
    # Moving averages
    df['sma_5'] = df['Close'].rolling(5).mean()
    df['sma_10'] = df['Close'].rolling(10).mean()
    df['sma_20'] = df['Close'].rolling(20).mean()
    df['ema_12'] = df['Close'].ewm(span=12).mean()
    df['ema_26'] = df['Close'].ewm(span=26).mean()
    
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # MACD
    df['macd'] = df['ema_12'] - df['ema_26']
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    
    # Bollinger Bands
    sma_20 = df['Close'].rolling(20).mean()
    std_20 = df['Close'].rolling(20).std()
    df['bb_upper'] = sma_20 + (std_20 * 2)
    df['bb_lower'] = sma_20 - (std_20 * 2)
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / sma_20
    
    # Volatility
    df['volatility'] = df['returns'].rolling(20).std()
    
    # Volume indicators
    df['volume_sma'] = df['Volume'].rolling(20).mean()
    df['volume_ratio'] = df['Volume'] / df['volume_sma']
    
    return df.fillna(method='ffill').dropna()

def prepare_model_input(data):
    """Prepare data for model input"""
    
    # Price features
    price_features = [
        'Open', 'High', 'Low', 'Close', 'Volume', 'returns', 'rsi', 'macd',
        'sma_5', 'sma_10', 'sma_20', 'ema_12', 'ema_26', 'volatility',
        'bb_width', 'volume_ratio'
    ]
    
    # Get last 60 days
    price_data = data[price_features].values[-60:]
    
    # Macro data (simplified - use random for now)
    macro_data = np.random.randn(60, 15)
    
    # Text data (simplified - use random for now)
    text_data = np.random.randn(60, 768)
    
    # Convert to tensors and add batch dimension
    price_tensor = torch.tensor(price_data, dtype=torch.float32).unsqueeze(0)
    macro_tensor = torch.tensor(macro_data, dtype=torch.float32).unsqueeze(0)
    text_tensor = torch.tensor(text_data, dtype=torch.float32).unsqueeze(0)
    
    return price_tensor, macro_tensor, text_tensor

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    success = load_model()
    if not success:
        print("WARNING: Model not loaded. Some endpoints may not work.")

@app.get("/")
async def root():
    """API health check"""
    return {
        "message": "Unified Multimodal Financial Forecasting API",
        "status": "active",
        "model_loaded": model is not None,
        "version": "1.0.0"
    }

@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "api_status": "healthy",
        "model_status": "loaded" if model is not None else "not_loaded",
        "timestamp": datetime.now().isoformat(),
        "supported_markets": ["US", "India", "Brazil", "Crypto"]
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Generate financial forecast for a symbol"""
    
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Get real-time data
        data = get_real_time_data(request.symbol, request.days_back)
        
        # Prepare model input
        price_tensor, macro_tensor, text_tensor = prepare_model_input(data)
        
        # Make prediction
        with torch.no_grad():
            outputs = model(price_tensor, macro_tensor, text_tensor)
        
        # Extract predictions
        forecast = outputs['forecast'].squeeze().numpy()
        anomaly_score = outputs['anomaly_score'].squeeze().item()
        
        # Convert forecast to prices
        current_price = float(data['Close'].iloc[-1])
        forecast_prices = []
        price = current_price
        
        for return_pred in forecast:
            price = price * (1 + return_pred)
            forecast_prices.append(float(price))
        
        # Generate forecast dates
        last_date = data.index[-1]
        forecast_dates = []
        for i in range(1, request.forecast_days + 1):
            future_date = last_date + timedelta(days=i)
            forecast_dates.append(future_date.strftime('%Y-%m-%d'))
        
        # Calculate confidence (simplified)
        volatility = float(data['volatility'].iloc[-1])
        confidence = max(0.5, 1.0 - volatility * 10)  # Higher volatility = lower confidence
        
        # Directional signal
        avg_forecast = np.mean(forecast)
        if avg_forecast > 0.01:
            directional_signal = "BUY"
        elif avg_forecast < -0.01:
            directional_signal = "SELL"
        else:
            directional_signal = "HOLD"
        
        # Technical indicators
        technical_indicators = {
            "rsi": float(data['rsi'].iloc[-1]),
            "macd": float(data['macd'].iloc[-1]),
            "volatility": volatility,
            "sma_20": float(data['sma_20'].iloc[-1]),
            "volume_ratio": float(data['volume_ratio'].iloc[-1])
        }
        
        return PredictionResponse(
            symbol=request.symbol,
            current_price=current_price,
            forecast=forecast_prices,
            forecast_dates=forecast_dates,
            anomaly_score=anomaly_score,
            confidence=confidence,
            directional_signal=directional_signal,
            technical_indicators=technical_indicators
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/explain", response_model=ExplainabilityResponse)
async def explain_prediction(request: ExplainabilityRequest):
    """Generate explanation for a prediction"""
    
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Get data and make prediction with attention
        data = get_real_time_data(request.symbol, 60)
        price_tensor, macro_tensor, text_tensor = prepare_model_input(data)
        
        with torch.no_grad():
            outputs = model(price_tensor, macro_tensor, text_tensor, return_attention=True)
        
        # Extract attention weights
        attention_weights = outputs.get('attention_weights', torch.randn(1, 60, 512))
        attention_summary = attention_weights.mean(dim=-1).squeeze().numpy().tolist()
        
        # Feature importance (simplified)
        feature_importance = {
            "price_trend": 0.25,
            "volume_pattern": 0.15,
            "technical_indicators": 0.20,
            "market_sentiment": 0.18,
            "economic_factors": 0.12,
            "volatility": 0.10
        }
        
        # Generate explanation text
        forecast = outputs['forecast'].squeeze().numpy()
        avg_return = np.mean(forecast)
        
        if avg_return > 0:
            trend = "upward"
            reason = "positive technical indicators and market momentum"
        else:
            trend = "downward" 
            reason = "negative market signals and increased volatility"
        
        explanation_text = f"The model predicts a {trend} trend for {request.symbol} based on {reason}. Key factors include recent price patterns, volume analysis, and technical indicator convergence."
        
        return ExplainabilityResponse(
            symbol=request.symbol,
            feature_importance=feature_importance,
            attention_weights=[attention_summary],
            explanation_text=explanation_text
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Explanation failed: {str(e)}")

@app.get("/market-overview", response_model=MarketOverviewResponse)
async def market_overview():
    """Get overall market overview"""
    
    try:
        # Key market symbols
        market_symbols = {
            "US": ["^GSPC", "^DJI", "^IXIC"],
            "India": ["^NSEI", "^BSESN"],
            "Brazil": ["^BVSP"],
            "Crypto": ["BTC-USD", "ETH-USD"]
        }
        
        markets = {}
        top_predictions = []
        
        for market, symbols in market_symbols.items():
            market_data = {}
            
            for symbol in symbols:
                try:
                    ticker = yf.Ticker(symbol)
                    data = ticker.history(period="2d")
                    
                    if not data.empty:
                        current = float(data['Close'].iloc[-1])
                        previous = float(data['Close'].iloc[-2])
                        change = (current - previous) / previous * 100
                        
                        market_data[symbol] = {
                            "price": current,
                            "change": change
                        }
                        
                        # Add to top predictions if model is available
                        if model is not None and len(top_predictions) < 5:
                            top_predictions.append({
                                "symbol": symbol,
                                "current_price": current,
                                "predicted_change": change * 1.1  # Simplified
                            })
                            
                except:
                    continue
            
            markets[market] = market_data
        
        # Overall market sentiment
        all_changes = []
        for market_data in markets.values():
            for symbol_data in market_data.values():
                all_changes.append(symbol_data["change"])
        
        avg_change = np.mean(all_changes) if all_changes else 0
        
        if avg_change > 1:
            sentiment = "BULLISH"
        elif avg_change < -1:
            sentiment = "BEARISH"
        else:
            sentiment = "NEUTRAL"
        
        return MarketOverviewResponse(
            timestamp=datetime.now().isoformat(),
            markets=markets,
            top_predictions=top_predictions,
            market_sentiment=sentiment
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Market overview failed: {str(e)}")

@app.get("/symbols")
async def get_supported_symbols():
    """Get list of supported symbols"""
    return {
        "US_stocks": ["AAPL", "GOOGL", "MSFT", "TSLA", "NVDA", "AMZN"],
        "Indian_stocks": ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS"],
        "Brazilian_stocks": ["PETR4.SA", "VALE3.SA", "ITUB4.SA"],
        "Crypto": ["BTC-USD", "ETH-USD", "ADA-USD"],
        "Indices": ["^GSPC", "^NSEI", "^BVSP"]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)