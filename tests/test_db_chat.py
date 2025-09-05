"""
Unit tests for database and chat interface.
"""

import unittest
import pandas as pd
import os
import sys

# Add utils to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))

from database import DatabaseManager, Stock
from chat_interface import FinancialChatInterface

class TestDbChat(unittest.TestCase):
    """Test cases for database and chat interface modules."""

    def setUp(self):
        """Set up test environment."""
        self.db_manager = DatabaseManager(db_url="sqlite:///:memory:")
        self.chat_interface = FinancialChatInterface(db_path="/tmp/test.db")

    def test_database_creation(self):
        """Test database and table creation."""
        self.assertIsNotNone(self.db_manager.engine)
        self.assertIsNotNone(self.chat_interface.engine)

    def test_synthetic_data_generation(self):
        """Test synthetic data generation."""
        self.db_manager.generate_synthetic_data(num_stocks=1, num_days=10)
        stocks = self.db_manager.get_session().query(Stock).filter(Stock.name.like("%SYNTHETIC%")).all()
        self.assertEqual(len(stocks), 1)

    def test_chat_interface_initialization(self):
        """Test chat interface initialization."""
        self.assertIsNotNone(self.chat_interface.sql_database)
        self.assertIsNotNone(self.chat_interface.sql_chain)

if __name__ == '__main__':
    unittest.main()

