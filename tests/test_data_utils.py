"""
Unit tests for data utilities (yfinance_util.py, polygon_util.py).
"""

import unittest
import pandas as pd
from datetime import datetime
import sys
import os

# Add utils to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))

from yfinance_util import YFinanceDataProvider
from polygon_util import PolygonDataProvider

class TestDataUtils(unittest.TestCase):
    """Test cases for data utility modules."""
    
    def setUp(self):
        """Set up test environment."""
        self.yf_provider = YFinanceDataProvider()
        self.polygon_provider = PolygonDataProvider()
    
    def test_yfinance_get_historical_data(self):
        """Test getting historical data from Yahoo Finance."""
        data = self.yf_provider.get_historical_data("AAPL", "1mo")
        self.assertIsInstance(data, pd.DataFrame)
        self.assertFalse(data.empty)
        self.assertIn("Close", data.columns)
    
    def test_polygon_get_historical_data(self):
        """Test getting historical data from Polygon."""
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - pd.Timedelta(days=30)).strftime("%Y-%m-%d")
        data = self.polygon_provider.get_historical_data("AAPL", start_date, end_date)
        self.assertIsInstance(data, pd.DataFrame)
        self.assertFalse(data.empty)
        self.assertIn("Close", data.columns)

if __name__ == '__main__':
    unittest.main()

