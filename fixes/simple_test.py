# Simple test of currency and news functions
from currency_support import detect_market_currency, format_currency

# Test currency
print("=== Currency Detection Test ===")
test_symbols = ["AAPL", "RELIANCE.NS", "PETR4.SA", "BTC-USD"]
for symbol in test_symbols:
    currency = detect_market_currency(symbol)
    formatted = format_currency(150000, currency)
    print(f"{symbol} -> {currency} -> {formatted}")

# Test sentiment analysis directly
print("\n=== Sentiment Analysis Test ===")
try:
    from textblob import TextBlob
    test_texts = [
        "This is great news for investors",
        "Market crashes dramatically", 
        "Steady performance expected"
    ]
    for text in test_texts:
        sentiment = TextBlob(text).sentiment
        print(f"'{text}' -> polarity: {sentiment.polarity:.2f}")
except Exception as e:
    print(f"TextBlob error: {e}")

print("\n=== All Tests Complete ===")