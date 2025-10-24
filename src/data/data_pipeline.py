"""End-to-end data pipeline for multimodal financial forecasting."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from torch.utils.data import DataLoader, Dataset
import yfinance as yf

from .macro_fetcher import MacroDataFetcher, load_macro_frame
from .news_sentiment import NewsSentimentFetcher


# ---------------------------------------------------------------------------
# Data collection utilities
# ---------------------------------------------------------------------------


def _detect_country(symbol: str) -> str:
    if symbol.endswith('.NS') or symbol.endswith('.BO'):
        return 'India'
    if symbol.endswith('.SA'):
        return 'Brazil'
    return 'US'


@dataclass
class FinancialDataCollector:
    """Collect price, macro, and news sentiment data."""

    device: str = 'auto'
    fred_api_key: Optional[str] = None

    def __post_init__(self) -> None:
        device = self.device
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)

        self.macro_fetcher = MacroDataFetcher(fred_api_key=self.fred_api_key)
        self.news_fetcher = NewsSentimentFetcher(device=device)

    def collect_price_data(
        self, symbols: List[str], start_date: str, end_date: str
    ) -> Dict[str, pd.DataFrame]:
        frames: Dict[str, pd.DataFrame] = {}
        for symbol in symbols:
            try:
                df = yf.Ticker(symbol).history(start=start_date, end=end_date)
                if df.empty:
                    print(f"✗ {symbol}: no price data")
                    continue
                frames[symbol] = df
                print(f"✓ {symbol}: {len(df)} rows")
            except Exception as exc:  # pragma: no cover - depends on network
                print(f"✗ {symbol}: price download failed ({exc})")
        return frames

    def collect_macro_features(
        self,
        price_data: Dict[str, pd.DataFrame],
        start_date: str,
        end_date: str,
    ) -> Dict[str, pd.DataFrame]:
        macro_frames: Dict[str, pd.DataFrame] = {}
        for symbol, price_df in price_data.items():
            country = _detect_country(symbol)
            frame = load_macro_frame(country, start_date, end_date, self.macro_fetcher)
            if frame.empty:
                macro_frames[symbol] = pd.DataFrame(index=price_df.index)
                continue
            frame = frame.reindex(pd.date_range(start_date, end_date, freq='D')).ffill()
            aligned = frame.reindex(price_df.index, method='ffill')
            macro_frames[symbol] = aligned
        return macro_frames

    def collect_news_embeddings(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str,
    ) -> Dict[str, pd.DataFrame]:
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        news_frames: Dict[str, pd.DataFrame] = {}
        for symbol in symbols:
            frame = self.news_fetcher.fetch_symbol_sentiment(symbol, start_dt, end_dt)
            if frame.empty:
                news_frames[symbol] = pd.DataFrame(columns=['sentiment', 'embedding'])
            else:
                news_frames[symbol] = frame
        return news_frames


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------


class FeatureEngineer:
    """Adds technical indicators to price data."""

    @staticmethod
    def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out['returns'] = out['Close'].pct_change()
        out['log_returns'] = np.log(out['Close'] / out['Close'].shift(1))

        for period in [5, 10, 20, 50]:
            out[f'sma_{period}'] = out['Close'].rolling(period).mean()
            out[f'ema_{period}'] = out['Close'].ewm(span=period).mean()

        delta = out['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        out['rsi'] = 100 - (100 / (1 + rs))

        ema_12 = out['Close'].ewm(span=12).mean()
        ema_26 = out['Close'].ewm(span=26).mean()
        out['ema_12'] = ema_12
        out['ema_26'] = ema_26
        out['macd'] = ema_12 - ema_26
        out['macd_signal'] = out['macd'].ewm(span=9).mean()

        sma_20 = out['Close'].rolling(20).mean()
        std_20 = out['Close'].rolling(20).std()
        out['bb_upper'] = sma_20 + 2 * std_20
        out['bb_lower'] = sma_20 - 2 * std_20
        out['bb_width'] = (out['bb_upper'] - out['bb_lower']) / sma_20

        out['volatility'] = out['returns'].rolling(20).std()
        out['volume_sma'] = out['Volume'].rolling(20).mean()
        out['volume_ratio'] = out['Volume'] / out['volume_sma']
        return out.dropna()


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class MultimodalFinancialDataset(Dataset):
    """Dataset supplying price, macro, and text sequences."""

    def __init__(
        self,
        price_data: Dict[str, pd.DataFrame],
        macro_data: Dict[str, pd.DataFrame],
        news_data: Dict[str, pd.DataFrame],
        lookback: int = 60,
        forecast_horizon: int = 5,
        macro_dim: int = 15,
    ) -> None:
        self.lookback = lookback
        self.forecast_horizon = forecast_horizon
        self.macro_dim = macro_dim
        self.samples = self._prepare_samples(price_data, macro_data, news_data)

    def _prepare_samples(
        self,
        price_data: Dict[str, pd.DataFrame],
        macro_data: Dict[str, pd.DataFrame],
        news_data: Dict[str, pd.DataFrame],
    ) -> List[Dict[str, object]]:
        samples: List[Dict[str, object]] = []

        for symbol, price_df in price_data.items():
            enriched = FeatureEngineer.add_technical_indicators(price_df)
            macro_frame = macro_data.get(symbol, pd.DataFrame())
            news_frame = news_data.get(symbol, pd.DataFrame())

            embeddings = (
                news_frame['embedding'] if 'embedding' in news_frame else pd.Series(dtype='object')
            )
            sentiments = (
                news_frame['sentiment'] if 'sentiment' in news_frame else pd.Series(dtype='float')
            )

            price_features = [
                'Open', 'High', 'Low', 'Close', 'Volume',
                'returns', 'log_returns', 'rsi', 'macd', 'macd_signal',
                'sma_5', 'sma_10', 'sma_20', 'sma_50',
                'ema_12', 'ema_26', 'ema_50',
                'volatility', 'bb_width', 'volume_ratio',
            ]

            for idx in range(self.lookback, len(enriched) - self.forecast_horizon):
                window_idx = enriched.index[idx - self.lookback:idx]
                price_seq = enriched.loc[window_idx, price_features].values.astype(np.float32)
                macro_seq = self._build_macro_sequence(macro_frame, window_idx)
                text_seq = self._build_text_sequence(embeddings, window_idx)

                targets = (
                    enriched.iloc[idx:idx + self.forecast_horizon]['returns'].values.astype(np.float32)
                )
                vol = enriched.iloc[idx - 1]['volatility']
                threshold = enriched['volatility'].quantile(0.95)
                anomaly = 1 if vol > threshold else 0

                sentiment_val = 0.0
                if not sentiments.empty:
                    sentiment_val = float(sentiments.reindex(window_idx).ffill().iloc[-1])

                samples.append(
                    {
                        'price_data': price_seq,
                        'macro_data': macro_seq.astype(np.float32),
                        'text_data': text_seq.astype(np.float32),
                        'price_targets': targets,
                        'anomaly_labels': anomaly,
                        'symbol': symbol,
                        'sentiment_score': sentiment_val,
                    }
                )

        return samples

    def _build_macro_sequence(self, macro_frame: pd.DataFrame, index_window: pd.Index) -> np.ndarray:
        if macro_frame.empty:
            return np.zeros((self.lookback, self.macro_dim), dtype=np.float32)

        aligned = macro_frame.reindex(index_window, method='ffill').fillna(method='ffill').fillna(0)
        values = aligned.values
        if values.shape[1] >= self.macro_dim:
            return values[:, : self.macro_dim]
        padding = np.zeros((values.shape[0], self.macro_dim - values.shape[1]))
        return np.hstack([values, padding])

    def _build_text_sequence(self, embeddings: pd.Series, index_window: pd.Index) -> np.ndarray:
        default_vec = np.zeros(768, dtype=np.float32)
        if embeddings.empty:
            return np.tile(default_vec, (self.lookback, 1))

        seq = []
        running = default_vec
        for ts in index_window:
            if ts in embeddings.index and embeddings.loc[ts] is not None:
                running = embeddings.loc[ts]
            seq.append(running)
        return np.stack(seq, axis=0)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        return {
            'price_data': torch.tensor(sample['price_data']),
            'macro_data': torch.tensor(sample['macro_data']),
            'text_data': torch.tensor(sample['text_data']),
            'price_targets': torch.tensor(sample['price_targets']),
            'anomaly_labels': torch.tensor(sample['anomaly_labels'], dtype=torch.long),
        }


# ---------------------------------------------------------------------------
# Data loader factory
# ---------------------------------------------------------------------------


def create_dataloaders(
    symbols: List[str],
    start_date: str = '2020-01-01',
    end_date: str = '2024-01-01',
    batch_size: int = 32,
    train_split: float = 0.8,
    fred_api_key: Optional[str] = None,
) -> Tuple[DataLoader, DataLoader]:
    collector = FinancialDataCollector(fred_api_key=fred_api_key)

    print('Collecting prices...')
    price_data = collector.collect_price_data(symbols, start_date, end_date)

    print('Collecting macro indicators...')
    macro_data = collector.collect_macro_features(price_data, start_date, end_date)

    print('Collecting news sentiment...')
    news_data = collector.collect_news_embeddings(list(price_data.keys()), start_date, end_date)

    dataset = MultimodalFinancialDataset(price_data, macro_data, news_data)

    train_size = int(len(dataset) * train_split)
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    print(f'Dataset size: {len(dataset)} samples')
    print(f'Train: {len(train_dataset)} | Val: {len(val_dataset)}')

    return train_loader, val_loader
