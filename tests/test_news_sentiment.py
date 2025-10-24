import unittest
from datetime import datetime

from src.data.news_sentiment import NewsSentimentFetcher


class NewsSentimentParsingTests(unittest.TestCase):
    def test_parse_rfc_timestamp(self) -> None:
        timestamp = "Fri, 24 Oct 2025 10:07:01 +0000"
        parsed = NewsSentimentFetcher._parse_published(timestamp)
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed, datetime(2025, 10, 24, 10, 7, 1))

    def test_parse_iso_timestamp(self) -> None:
        timestamp = "2025-10-24T08:30:00"
        parsed = NewsSentimentFetcher._parse_published(timestamp)
        self.assertEqual(parsed, datetime(2025, 10, 24, 8, 30, 0))


if __name__ == "__main__":
    unittest.main()
