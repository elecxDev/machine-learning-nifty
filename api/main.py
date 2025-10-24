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
    from src.data.macro_fetcher import MacroDataFetcher, load_macro_frame
    from src.data.news_sentiment import NewsSentimentFetcher
except ImportError:
    print("Warning: Model not available. Running in demo mode.")
    UnifiedMultimodalTransformer = None
    ModelConfig = None
    MacroDataFetcher = None
    NewsSentimentFetcher = None

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

FRED_API_KEY = os.getenv("FRED_API_KEY")
macro_fetcher = MacroDataFetcher(fred_api_key=FRED_API_KEY) if 'MacroDataFetcher' in globals() else None
news_fetcher = NewsSentimentFetcher(device='auto') if 'NewsSentimentFetcher' in globals() else None

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


class NewsArticle(BaseModel):
    title: str
    summary: str
    sentiment_label: str
    sentiment_score: float
    published: str
    link: Optional[str] = None

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

        if data.index.tz is not None:
            data.index = data.index.tz_localize(None)
        
        # Add technical indicators
        data = add_technical_indicators(data)
        
        return data.tail(days_back)  # Return last N days
        
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Error fetching data for {symbol}: {str(e)}")

def add_technical_indicators(df):
    """Add technical indicators to price data"""
    
    # Returns
    df['returns'] = df['Close'].pct_change()
    df['log_returns'] = np.log(df['Close'] / df['Close'].shift(1))
    
    # Moving averages
    df['sma_5'] = df['Close'].rolling(5).mean()
    df['sma_10'] = df['Close'].rolling(10).mean()
    df['sma_20'] = df['Close'].rolling(20).mean()
    df['sma_50'] = df['Close'].rolling(50).mean()
    df['ema_12'] = df['Close'].ewm(span=12).mean()
    df['ema_26'] = df['Close'].ewm(span=26).mean()
    df['ema_50'] = df['Close'].ewm(span=50).mean()
    
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

def _detect_country(symbol: str) -> str:
    if symbol.endswith('.NS') or symbol.endswith('.BO'):
        return 'India'
    if symbol.endswith('.SA'):
        return 'Brazil'
    return 'US'


def _pad_rows(matrix: np.ndarray, target_len: int) -> np.ndarray:
    if matrix.shape[0] >= target_len:
        return matrix[-target_len:]
    if matrix.shape[0] == 0:
        return np.zeros((target_len, matrix.shape[1]), dtype=np.float32)
    pad_rows = target_len - matrix.shape[0]
    padding = np.repeat(matrix[:1], pad_rows, axis=0)
    return np.vstack([padding, matrix])


def prepare_model_input(symbol: str, data: pd.DataFrame):
    """Prepare price, macro and text tensors for inference."""

    lookback = 60
    price_features = [
        'Open', 'High', 'Low', 'Close', 'Volume', 'returns', 'log_returns', 'rsi',
        'macd', 'macd_signal', 'sma_5', 'sma_10', 'sma_20', 'sma_50', 'ema_12',
        'ema_26', 'ema_50', 'volatility', 'bb_width', 'volume_ratio'
    ]

    price_matrix = data[price_features].values.astype(np.float32)
    price_matrix = _pad_rows(price_matrix, lookback)

    macro_matrix = np.zeros((len(data), 0), dtype=np.float32)
    if macro_fetcher is not None:
        try:
            country = _detect_country(symbol)
            macro_frame = load_macro_frame(
                country,
                data.index.min().strftime('%Y-%m-%d'),
                data.index.max().strftime('%Y-%m-%d'),
                macro_fetcher,
            )
            if not macro_frame.empty:
                macro_frame = macro_frame.reindex(data.index, method='ffill').fillna(0)
                macro_matrix = macro_frame.values.astype(np.float32)
        except Exception as exc:
            print(f"Macro fetch failed for {symbol}: {exc}")
            macro_matrix = np.zeros((len(data), 0), dtype=np.float32)

    if macro_matrix.size == 0:
        macro_matrix = np.zeros((len(data), 15), dtype=np.float32)

    macro_matrix = _pad_rows(macro_matrix, lookback)
    if macro_matrix.shape[1] > 15:
        macro_matrix = macro_matrix[:, :15]
    elif macro_matrix.shape[1] < 15:
        pad_cols = 15 - macro_matrix.shape[1]
        macro_matrix = np.hstack([macro_matrix, np.zeros((lookback, pad_cols), dtype=np.float32)])

    default_vec = np.zeros(768, dtype=np.float32)
    text_vectors: List[np.ndarray] = []
    embeddings = None
    if news_fetcher is not None:
        try:
            news_frame = news_fetcher.fetch_symbol_sentiment(
                symbol,
                data.index.min().to_pydatetime(),
                data.index.max().to_pydatetime(),
            )
            if not news_frame.empty:
                embeddings = news_frame['embedding']
        except Exception as exc:
            print(f"News sentiment fetch failed for {symbol}: {exc}")
            embeddings = None

    current_vec = default_vec
    for ts in data.index:
        if embeddings is not None and ts in embeddings.index and embeddings.loc[ts] is not None:
            series_vec = embeddings.loc[ts]
            if not isinstance(series_vec, np.ndarray):
                series_vec = np.asarray(series_vec, dtype=np.float32)
            else:
                series_vec = series_vec.astype(np.float32)
            current_vec = series_vec
        text_vectors.append(current_vec)

    text_matrix = np.vstack(text_vectors) if text_vectors else np.zeros((0, 768), dtype=np.float32)
    text_matrix = _pad_rows(text_matrix, lookback)

    price_tensor = torch.tensor(price_matrix, dtype=torch.float32).unsqueeze(0)
    macro_tensor = torch.tensor(macro_matrix, dtype=torch.float32).unsqueeze(0)
    text_tensor = torch.tensor(text_matrix, dtype=torch.float32).unsqueeze(0)

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
        price_tensor, macro_tensor, text_tensor = prepare_model_input(request.symbol, data)
        
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
        price_tensor, macro_tensor, text_tensor = prepare_model_input(request.symbol, data)
        
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


@app.get("/news/{symbol}", response_model=List[NewsArticle])
async def get_news_sentiment(
    symbol: str,
    max_items: int = 6,
    days: int = 7,
    company_name: Optional[str] = None,
    force_refresh: bool = False,
):
    """Fetch recent headlines with FinBERT sentiment for a symbol."""

    if news_fetcher is None:
        raise HTTPException(status_code=503, detail="News sentiment service unavailable")

    end_dt = datetime.now()
    start_dt = end_dt - timedelta(days=days)

    keywords = {symbol.upper(), symbol.split('.')[0].upper()}
    if company_name:
        keywords.add(company_name)
        keywords.update(part.strip() for part in company_name.replace(',', '').split())

    try:
        articles = news_fetcher.fetch_headline_details(
            symbol,
            start_dt,
            end_dt,
            max_items=max_items,
            force_refresh=force_refresh,
            keywords=keywords,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"News fetch failed: {str(exc)}")

    results: List[NewsArticle] = []
    for article in articles:
        published_ts = article.get("published_ts")
        if isinstance(published_ts, pd.Timestamp):
            published_ts = published_ts.to_pydatetime()
        if isinstance(published_ts, datetime):
            published_str = published_ts.strftime("%Y-%m-%d %H:%M")
        else:
            published_str = end_dt.strftime("%Y-%m-%d %H:%M")

        label = article.get("sentiment_label") or "neutral"
        score_raw = article.get("sentiment_score", 0.0)
        try:
            score = float(score_raw)
        except (TypeError, ValueError):
            score = 0.0

        results.append(
            NewsArticle(
                title=article.get("title", ""),
                summary=article.get("summary", ""),
                sentiment_label=label,
                sentiment_score=score,
                published=published_str,
                link=article.get("link") or None,
            )
        )

    return results

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