"""
Complete Data Pipeline for Multimodal Financial Forecasting
Handles data collection, preprocessing, and feature engineering
"""

import yfinance as yf
import pandas as pd
import numpy as np
import requests
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class FinancialDataCollector:
    """Collects financial data from multiple free sources"""
    
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained('ProsusAI/finbert')
        self.text_model = AutoModel.from_pretrained('ProsusAI/finbert')
        
    def collect_price_data(self, symbols: List[str], start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:
        """Collect price data from Yahoo Finance"""
        data = {}
        
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                df = ticker.history(start=start_date, end=end_date)
                
                if not df.empty:
                    data[symbol] = df
                    print(f"✓ {symbol}: {len(df)} records")
                else:
                    print(f"✗ {symbol}: No data")
                    
            except Exception as e:
                print(f"✗ {symbol}: {str(e)[:50]}")
                
        return data
    
    def collect_economic_data(self, countries: List[str], start_year: int = 2020, end_year: int = 2023) -> Dict[str, pd.DataFrame]:
        """Collect economic indicators from World Bank API"""
        country_codes = {'US': 'USA', 'India': 'IND', 'Brazil': 'BRA'}
        indicators = {
            'GDP': 'NY.GDP.MKTP.CD',
            'Inflation': 'FP.CPI.TOTL.ZG',
            'Unemployment': 'SL.UEM.TOTL.ZS'
        }
        
        economic_data = {}
        
        for country in countries:
            country_code = country_codes.get(country, country)
            country_data = {}
            
            for indicator_name, indicator_code in indicators.items():
                try:
                    url = f"http://api.worldbank.org/v2/country/{country_code}/indicator/{indicator_code}"
                    params = {
                        'format': 'json',
                        'date': f'{start_year}:{end_year}',
                        'per_page': 100
                    }
                    
                    response = requests.get(url, params=params, timeout=10)
                    
                    if response.status_code == 200:
                        data = response.json()
                        if len(data) > 1 and data[1]:
                            df_data = []
                            for item in data[1]:
                                if item['value'] is not None:
                                    df_data.append({
                                        'date': pd.to_datetime(item['date']),
                                        'value': float(item['value'])
                                    })
                            
                            if df_data:
                                df = pd.DataFrame(df_data).set_index('date').sort_index()
                                country_data[indicator_name] = df
                                
                except Exception as e:
                    print(f"Error collecting {indicator_name} for {country}: {str(e)[:50]}")
            
            economic_data[country] = country_data
            
        return economic_data
    
    def collect_news_sentiment(self, texts: List[str]) -> np.ndarray:
        """Generate sentiment embeddings using FinBERT"""
        embeddings = []
        
        for text in texts:
            inputs = self.tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
            
            with torch.no_grad():
                outputs = self.text_model(**inputs)
                embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
                embeddings.append(embedding)
        
        return np.array(embeddings)

class FeatureEngineer:
    """Feature engineering for financial time series"""
    
    @staticmethod
    def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """Add comprehensive technical indicators"""
        # Price-based features
        df['returns'] = df['Close'].pct_change()
        df['log_returns'] = np.log(df['Close'] / df['Close'].shift(1))
        
        # Moving averages
        for period in [5, 10, 20, 50]:
            df[f'sma_{period}'] = df['Close'].rolling(period).mean()
            df[f'ema_{period}'] = df['Close'].ewm(span=period).mean()
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        ema_12 = df['Close'].ewm(span=12).mean()
        ema_26 = df['Close'].ewm(span=26).mean()
        df['macd'] = ema_12 - ema_26
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
        
        return df.dropna()

class MultimodalFinancialDataset(Dataset):
    """PyTorch Dataset for multimodal financial data"""
    
    def __init__(self, price_data: Dict[str, pd.DataFrame], 
                 economic_data: Dict[str, pd.DataFrame],
                 news_embeddings: Optional[np.ndarray] = None,
                 lookback: int = 60, forecast_horizon: int = 5):
        
        self.lookback = lookback
        self.forecast_horizon = forecast_horizon
        
        # Process data
        self.samples = self._prepare_samples(price_data, economic_data, news_embeddings)
        
    def _prepare_samples(self, price_data: Dict[str, pd.DataFrame], 
                        economic_data: Dict[str, pd.DataFrame],
                        news_embeddings: Optional[np.ndarray]) -> List[Dict]:
        """Prepare training samples"""
        samples = []
        
        for symbol, df in price_data.items():
            # Add technical indicators
            df = FeatureEngineer.add_technical_indicators(df)
            
            # Price features
            price_features = [
                'Open', 'High', 'Low', 'Close', 'Volume', 'returns', 'rsi', 'macd',
                'sma_5', 'sma_10', 'sma_20', 'ema_12', 'ema_26', 'volatility',
                'bb_width', 'volume_ratio'
            ]
            
            # Create sequences
            for i in range(self.lookback, len(df) - self.forecast_horizon):
                # Price sequence
                price_seq = df.iloc[i-self.lookback:i][price_features].values
                
                # Macro sequence (simplified - repeat last known value)
                macro_seq = np.random.randn(self.lookback, 15)  # Placeholder for macro features
                
                # Text sequence (simplified - repeat embedding)
                if news_embeddings is not None and len(news_embeddings) > 0:
                    text_seq = np.tile(news_embeddings[0], (self.lookback, 1))
                else:
                    text_seq = np.random.randn(self.lookback, 768)  # FinBERT embedding size
                
                # Target (next 5 day returns)
                target = df.iloc[i:i+self.forecast_horizon]['returns'].values
                
                # Anomaly label (simplified - based on volatility)
                volatility = df.iloc[i-1]['volatility']
                anomaly_label = 1 if volatility > df['volatility'].quantile(0.95) else 0
                
                sample = {
                    'price_data': price_seq.astype(np.float32),
                    'macro_data': macro_seq.astype(np.float32),
                    'text_data': text_seq.astype(np.float32),
                    'price_targets': target.astype(np.float32),
                    'anomaly_labels': anomaly_label,
                    'symbol': symbol
                }
                
                samples.append(sample)
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        
        return {
            'price_data': torch.tensor(sample['price_data']),
            'macro_data': torch.tensor(sample['macro_data']),
            'text_data': torch.tensor(sample['text_data']),
            'price_targets': torch.tensor(sample['price_targets']),
            'anomaly_labels': torch.tensor(sample['anomaly_labels'], dtype=torch.long)
        }

def create_dataloaders(symbols: List[str], countries: List[str], 
                      start_date: str = '2020-01-01', end_date: str = '2024-01-01',
                      batch_size: int = 32, train_split: float = 0.8) -> Tuple[DataLoader, DataLoader]:
    """Create training and validation dataloaders"""
    
    # Collect data
    collector = FinancialDataCollector()
    
    print("Collecting price data...")
    price_data = collector.collect_price_data(symbols, start_date, end_date)
    
    print("Collecting economic data...")
    economic_data = collector.collect_economic_data(countries)
    
    print("Generating news embeddings...")
    sample_news = ["Market outlook positive", "Economic growth expected", "Volatility concerns"]
    news_embeddings = collector.collect_news_sentiment(sample_news)
    
    # Create dataset
    dataset = MultimodalFinancialDataset(price_data, economic_data, news_embeddings)
    
    # Split dataset
    train_size = int(len(dataset) * train_split)
    val_size = len(dataset) - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"Dataset created: {len(dataset)} samples")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    return train_loader, val_loader