# Deliverable 1 – Real Multimodal Inputs

## Implementation Overview

- **Macro features**
  - Normalized country keys so two-letter markets (e.g., `US`, `IN`) map to their respective FRED / World Bank bundles without losing uppercase codes.
  - Enabled FRED downloads without an API key by streaming the public CSV endpoints and persisting the results into `data/cache/macro/*.parquet` files; the JSON path with API credentials is still honored when present.
  - Kept World Bank ingestion unchanged but ensured series are cached and resampled to daily frequency so every market receives aligned macro tensors.
- **News sentiment**
  - Introduced an HTTP session with browser headers to avoid Yahoo Finance RSS rate limits and to standardize feed downloads.
  - Removed the legacy duplicate `fetch_symbol_sentiment` implementation so the class always routes through the bundler that caches both daily sentiment aggregates and per-article metadata.
  - Replaced the RSS timestamp parsing logic with `email.utils.parsedate_to_datetime`, making timezone handling robust before computing FinBERT embeddings.
- **Regression safety net**
  - Added `tests/test_macro_fetcher.py` to verify the FRED CSV fallback and cache isolation.
  - Added `tests/test_news_sentiment.py` to confirm RFC822 and ISO timestamp handling without pulling the large FinBERT weights during unit tests.

## Live Data Verification

- **US Macro Pull**  
  `.\.venv\Scripts\python.exe -c "from datetime import date, timedelta; from src.data.macro_fetcher import MacroDataFetcher; f=MacroDataFetcher(); end=date.today(); start=end-timedelta(days=30); frame=f.get_country_series('US', start.isoformat(), end.isoformat())['macro_features']; print(frame.tail())"`

  ```text
              us_treasury_10y
  date
  2025-10-18             4.02
  2025-10-19             4.02
  2025-10-20             4.00
  2025-10-21             3.98
  2025-10-22             3.97
  ```
- **India Macro Pull**  
  `.\.venv\Scripts\python.exe -c "from src.data.macro_fetcher import MacroDataFetcher; f=MacroDataFetcher(); frame=f.get_country_series('India', '2020-01-01', '2024-01-01')['macro_features']; print(frame.columns); print(frame.tail())"`

  ```text
  Index(['ind_gdp_growth', 'ind_cpi_inflation', 'ind_unemployment_rate'], dtype='object')
              ind_gdp_growth  ind_cpi_inflation  ind_unemployment_rate
  date
  2023-12-27        7.609365           6.699034                  4.822
  2023-12-28        7.609365           6.699034                  4.822
  2023-12-29        7.609365           6.699034                  4.822
  2023-12-30        7.609365           6.699034                  4.822
  2023-12-31        9.190755           5.649143                  4.172
  ```

- **News + FinBERT Sentiment (AAPL)**  
  `.\.venv\Scripts\python.exe -c "from datetime import datetime, timedelta; from src.data.news_sentiment import NewsSentimentFetcher; f=NewsSentimentFetcher(device='cpu'); end=datetime.utcnow(); start=end-timedelta(days=3); frame=f.fetch_symbol_sentiment('AAPL', start, end, max_items=6, force_refresh=True); print('rows', len(frame)); print(frame.head())"`

  ```text
  rows 1
              sentiment                                          embedding
  date
  2025-10-24  -0.016101  [0.15653623640537262, 0.36583617329597473, ...]
  ```

  Cached output can be inspected in `data/cache/news/AAPL_20251021_20251024.npz`, which now stores both normalized sentiment scores and the pooled FinBERT embeddings.

## Automated Tests

- `.\.venv\Scripts\python.exe -m unittest tests.test_macro_fetcher tests.test_news_sentiment`

  ```text
  ...
  ----------------------------------------------------------------------
  Ran 3 tests in 0.016s

  OK
  ```

## Next Steps

- Re-run `scripts/run_full_stack.py` or the FastAPI/Streamlit components whenever you are ready for an end-to-end demo; the live data contracts are now exercised and cached successfully.
- Commit the new `tests/` directory and the deliverable report once you finish reviewing the changes.
