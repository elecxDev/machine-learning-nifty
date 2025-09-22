from main_app import create_sample_news

# Test news generation
news = create_sample_news('AAPL', 'Apple Inc.')
print(f"Generated {len(news)} news items")
for i, item in enumerate(news[:2]):
    print(f"\nNews {i+1}:")
    print(f"Title: {item['title']}")
    print(f"Sentiment: {item['sentiment']}")