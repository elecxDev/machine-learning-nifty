import shutil
import tempfile
import unittest
from unittest.mock import MagicMock, patch
from src.data import macro_fetcher

MacroDataFetcher = macro_fetcher.MacroDataFetcher

class MacroFetcherTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp_dir = tempfile.mkdtemp()
        self.original_cache_root = macro_fetcher.CACHE_ROOT
        macro_fetcher.CACHE_ROOT = self.tmp_dir

    def tearDown(self) -> None:
        macro_fetcher.CACHE_ROOT = self.original_cache_root
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_load_fred_series_without_api_key(self) -> None:
        fetcher = MacroDataFetcher()
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_response.text = "DATE,TEST\n2024-01-01,1.5\n2024-01-02,2.0\n"
        fetcher.session.get = MagicMock(return_value=mock_response)

        with patch("src.data.macro_fetcher.pd.DataFrame.to_parquet") as mocked_to_parquet:
            mocked_to_parquet.return_value = None
            df = fetcher._load_fred_series("TEST", "2024-01-01", "2024-01-02")

        self.assertFalse(df.empty)
        self.assertListEqual(list(df.columns), ["TEST"])
        self.assertListEqual(df.index.astype(str).tolist(), ["2024-01-01", "2024-01-02"])
        fetcher.session.get.assert_called()


if __name__ == "__main__":
    unittest.main()
