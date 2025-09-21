"""
Complete Training Script with Historical Data
Trains the unified multimodal transformer on real financial data from 2020-2024
"""

import torch
import pandas as pd
import numpy as np
import yfinance as yf
import requests
from datetime import datetime, timedelta
import os
import sys
import pickle
from pathlib import Path
# import ssl

# # SSL Certificate fix for corporate networks and certificate issues
# ssl._create_default_https_context = ssl._create_unverified_context
# os.environ['CURL_CA_BUNDLE'] = ''
# os.environ['REQUESTS_CA_BUNDLE'] = ''

# # Disable SSL warnings
# try:
#     import urllib3
#     urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
#     print("✅ SSL warnings disabled")
# except ImportError:
#     pass

# Fix Python path imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
src_dir = os.path.join(parent_dir, 'src')

# Add both parent and src directories to path
sys.path.insert(0, parent_dir)
sys.path.insert(0, src_dir)

# Now import our modules
try:
    from models.unified_transformer import UnifiedMultimodalTransformer, ModelConfig
    from training.trainer import FinancialTrainer
    print("✅ Successfully imported ML modules")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print(f"Parent dir: {parent_dir}")
    print(f"Src dir: {src_dir}")
    print(f"Current sys.path: {sys.path[:3]}")
    sys.exit(1)

from transformers import AutoTokenizer, AutoModel

class HistoricalDataCollector:
    """Collects and processes historical financial data"""
    
    def __init__(self):
        print("Initializing FinBERT for sentiment analysis...")
        self.tokenizer = AutoTokenizer.from_pretrained('ProsusAI/finbert')
        self.text_model = AutoModel.from_pretrained('ProsusAI/finbert')
        
    def collect_historical_data(self, symbols, start_date='2020-01-01', end_date='2024-12-01'):
        """Collect comprehensive historical data"""
        
        print(f"Collecting historical data from {start_date} to {end_date}")
        print(f"Symbols: {symbols}")
        
        all_data = {}
        
        # 1. Collect price data
        print("\n1. Collecting price data...")
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                data = ticker.history(start=start_date, end=end_date)
                
                if not data.empty:
                    # Add technical indicators
                    data = self.add_technical_indicators(data)
                    all_data[symbol] = data
                    print(f"  SUCCESS: {symbol} - {len(data)} records")
                else:
                    print(f"  FAILED: {symbol} - No data")
                    
            except Exception as e:
                print(f"  ERROR: {symbol} - {str(e)[:50]}")
        
        # 2. Collect economic data
        print("\n2. Collecting economic data...")
        economic_data = self.collect_economic_indicators()
        
        # 3. Generate news sentiment
        print("\n3. Generating news sentiment...")
        news_sentiment = self.generate_news_sentiment()
        
        return all_data, economic_data, news_sentiment
    
    def add_technical_indicators(self, df):
        """Add comprehensive technical indicators"""
        
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
    
    def collect_economic_indicators(self):
        """Collect economic indicators from World Bank"""
        
        countries = {'US': 'USA', 'India': 'IND', 'Brazil': 'BRA'}
        indicators = {
            'GDP': 'NY.GDP.MKTP.CD',
            'Inflation': 'FP.CPI.TOTL.ZG', 
            'Unemployment': 'SL.UEM.TOTL.ZS'
        }
        
        economic_data = {}
        
        for country, country_code in countries.items():
            country_data = {}
            
            for indicator_name, indicator_code in indicators.items():
                try:
                    url = f"http://api.worldbank.org/v2/country/{country_code}/indicator/{indicator_code}"
                    params = {'format': 'json', 'date': '2020:2023', 'per_page': 100}
                    
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
                                print(f"  SUCCESS: {country} {indicator_name}")
                                
                except Exception as e:
                    print(f"  ERROR: {country} {indicator_name} - {str(e)[:30]}")
            
            economic_data[country] = country_data
        
        return economic_data
    
    def generate_news_sentiment(self):
        """Generate news sentiment embeddings"""
        
        # Sample financial news for different periods
        news_samples = [
            "Market shows strong growth with positive economic indicators",
            "Technology stocks lead market gains amid innovation surge", 
            "Economic uncertainty creates market volatility concerns",
            "Central bank policy changes impact global markets",
            "Strong earnings reports boost investor confidence",
            "Geopolitical tensions create market instability",
            "Inflation concerns weigh on market sentiment",
            "Recovery signs emerge in key economic sectors"
        ]
        
        embeddings = []
        
        for text in news_samples:
            inputs = self.tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
            
            with torch.no_grad():
                outputs = self.text_model(**inputs)
                embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
                embeddings.append(embedding)
        
        print(f"  Generated {len(embeddings)} sentiment embeddings")
        return np.array(embeddings)

class HistoricalDataset(torch.utils.data.Dataset):
    """Dataset for historical financial data"""
    
    def __init__(self, price_data, economic_data, news_embeddings, lookback=60, forecast_horizon=5):
        self.lookback = lookback
        self.forecast_horizon = forecast_horizon
        self.samples = []
        
        # Process each symbol
        for symbol, df in price_data.items():
            
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
                
                # Macro sequence (simplified - use random for now)
                macro_seq = np.random.randn(self.lookback, 15)
                
                # Text sequence (repeat random embedding)
                if len(news_embeddings) > 0:
                    text_seq = np.tile(news_embeddings[i % len(news_embeddings)], (self.lookback, 1))
                else:
                    text_seq = np.random.randn(self.lookback, 768)
                
                # Target (next 5 day returns)
                target = df.iloc[i:i+self.forecast_horizon]['returns'].values
                
                # Anomaly label (high volatility periods)
                volatility = df.iloc[i-1]['volatility']
                anomaly_label = 1 if volatility > df['volatility'].quantile(0.95) else 0
                
                self.samples.append({
                    'price_data': price_seq.astype(np.float32),
                    'macro_data': macro_seq.astype(np.float32),
                    'text_data': text_seq.astype(np.float32),
                    'price_targets': target.astype(np.float32),
                    'anomaly_labels': anomaly_label,
                    'symbol': symbol,
                    'date': df.index[i]
                })
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        return {
            'price_data': torch.tensor(sample['price_data']),
            'macro_data': torch.tensor(sample['macro_data']),
            'text_data': torch.tensor(sample['text_data']),
            'price_targets': torch.tensor(sample['price_targets']),
            'anomaly_labels': torch.tensor(sample['anomaly_labels'], dtype=torch.long)
        }

def main():
    """Main training pipeline with historical data"""
    
    print("=" * 60)
    print("UNIFIED MULTIMODAL TRANSFORMER - FULL TRAINING")
    print("=" * 60)
    
    # Multi-country symbols for comprehensive training
    symbols = [
        # US Market
        'AAPL', 'GOOGL', 'MSFT', 'TSLA', 'NVDA', 'AMZN',
        # Indian Market  
        'RELIANCE.NS', 'TCS.NS', 'INFY.NS', 'HDFCBANK.NS',
        # Brazilian Market
        'PETR4.SA', 'VALE3.SA', 'ITUB4.SA',
        # Crypto
        'BTC-USD', 'ETH-USD'
    ]
    
    print(f"Training symbols: {len(symbols)} assets")
    print(f"Markets: US, India, Brazil, Crypto")
    
    # 1. Collect historical data
    print("\nPHASE 1: HISTORICAL DATA COLLECTION")
    print("-" * 40)
    
    collector = HistoricalDataCollector()
    price_data, economic_data, news_embeddings = collector.collect_historical_data(
        symbols, start_date='2020-01-01', end_date='2024-12-01'
    )
    
    print(f"\nData collection summary:")
    print(f"  Price datasets: {len(price_data)}")
    print(f"  Economic indicators: {len(economic_data)}")
    print(f"  News embeddings: {len(news_embeddings)}")
    
    # Save raw data
    os.makedirs('data/raw', exist_ok=True)
    with open('data/raw/price_data.pkl', 'wb') as f:
        pickle.dump(price_data, f)
    with open('data/raw/economic_data.pkl', 'wb') as f:
        pickle.dump(economic_data, f)
    with open('data/raw/news_embeddings.pkl', 'wb') as f:
        pickle.dump(news_embeddings, f)
    
    print("  Raw data saved to data/raw/")
    
    # 2. Create dataset
    print("\nPHASE 2: DATASET CREATION")
    print("-" * 40)
    
    dataset = HistoricalDataset(price_data, economic_data, news_embeddings)
    
    # Split dataset
    train_size = int(len(dataset) * 0.7)
    val_size = int(len(dataset) * 0.15)
    test_size = len(dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    print(f"Dataset created:")
    print(f"  Total samples: {len(dataset):,}")
    print(f"  Training: {len(train_dataset):,}")
    print(f"  Validation: {len(val_dataset):,}")
    print(f"  Testing: {len(test_dataset):,}")
    
    # 3. Initialize model
    print("\nPHASE 3: MODEL INITIALIZATION")
    print("-" * 40)
    
    config = ModelConfig(
        d_model=512,
        n_heads=8,
        n_layers=6,
        dropout=0.1,
        price_features=16,
        macro_features=15,
        text_features=768,
        forecast_horizon=5,
        learning_rate=1e-4,
        weight_decay=0.01
    )
    
    model = UnifiedMultimodalTransformer(config)
    total_params = sum(p.numel() for p in model.parameters())
    
    print(f"Model initialized:")
    print(f"  Parameters: {total_params:,}")
    print(f"  Size: ~{total_params * 4 / 1024 / 1024:.1f} MB")
    
    # 4. Train model
    print("\nPHASE 4: MODEL TRAINING")
    print("-" * 40)
    
    trainer = FinancialTrainer(model, config)
    
    # Train for multiple epochs
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=100,  # Full training
        save_dir='models/checkpoints'
    )
    
    print("Training completed!")
    
    # 5. Evaluate model
    print("\nPHASE 5: MODEL EVALUATION")
    print("-" * 40)
    
    # Test set evaluation
    model.eval()
    test_losses = []
    predictions = []
    targets = []
    
    with torch.no_grad():
        for batch in test_loader:
            outputs = model(
                batch['price_data'],
                batch['macro_data'],
                batch['text_data']
            )
            
            # Compute test loss
            forecast_loss = torch.nn.MSELoss()(outputs['forecast'], batch['price_targets'])
            test_losses.append(forecast_loss.item())
            
            predictions.append(outputs['forecast'].cpu().numpy())
            targets.append(batch['price_targets'].cpu().numpy())
    
    # Calculate metrics
    avg_test_loss = np.mean(test_losses)
    predictions = np.concatenate(predictions, axis=0)
    targets = np.concatenate(targets, axis=0)
    
    # Directional accuracy
    pred_direction = np.sign(predictions[:, 0])
    true_direction = np.sign(targets[:, 0])
    directional_accuracy = np.mean(pred_direction == true_direction)
    
    print(f"Test Results:")
    print(f"  Test Loss (MSE): {avg_test_loss:.4f}")
    print(f"  Directional Accuracy: {directional_accuracy:.3f}")
    print(f"  RMSE: {np.sqrt(avg_test_loss):.4f}")
    
    # 6. Save final model
    print("\nPHASE 6: MODEL SAVING")
    print("-" * 40)
    
    os.makedirs('models', exist_ok=True)
    
    # Save model
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'test_loss': avg_test_loss,
        'directional_accuracy': directional_accuracy,
        'training_history': history
    }, 'models/unified_transformer_trained.pt')
    
    # Save model for HuggingFace
    model_save_path = 'models/unified_transformer_final'
    os.makedirs(model_save_path, exist_ok=True)
    torch.save(model.state_dict(), f'{model_save_path}/pytorch_model.bin')
    
    print(f"Model saved:")
    print(f"  Checkpoint: models/unified_transformer_trained.pt")
    print(f"  HuggingFace format: {model_save_path}/")
    
    # 7. Generate summary report
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE - SUMMARY REPORT")
    print("=" * 60)
    
    print(f"Dataset: {len(symbols)} assets, {len(dataset):,} samples")
    print(f"Training period: 2020-2024 (4+ years)")
    print(f"Model: {total_params:,} parameters")
    print(f"Final test loss: {avg_test_loss:.4f}")
    print(f"Directional accuracy: {directional_accuracy:.1%}")
    
    print(f"\nFiles created:")
    print(f"  data/raw/ - Historical data")
    print(f"  models/unified_transformer_trained.pt - Trained model")
    print(f"  models/checkpoints/ - Training checkpoints")
    
    print(f"\nModel ready for:")
    print(f"  1. Real-time predictions")
    print(f"  2. Cross-market evaluation")
    print(f"  3. Explainability analysis")
    print(f"  4. Production deployment")
    
    return model, history, {
        'test_loss': avg_test_loss,
        'directional_accuracy': directional_accuracy,
        'total_samples': len(dataset)
    }

if __name__ == "__main__":
    model, history, results = main()