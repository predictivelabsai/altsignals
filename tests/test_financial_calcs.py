"""
Unit tests for financial calculations (options_pricing.py, backtesting.py).
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime
import sys
import os

# Add utils to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))

from options_pricing import BlackScholesCalculator
from backtesting import BacktestEngine, buy_and_hold_strategy

class TestFinancialCalcs(unittest.TestCase):
    """Test cases for financial calculation modules."""
    
    def setUp(self):
        """Set up test environment."""
        self.bs_calculator = BlackScholesCalculator()
        self.backtest_engine = BacktestEngine(initial_capital=100000)
    
    def test_black_scholes_call_price(self):
        """Test Black-Scholes call option pricing."""
        price = self.bs_calculator.call_price(100, 100, 0.25, 0.05, 0.2)
        self.assertAlmostEqual(price, 4.61, places=2)
    
    def test_black_scholes_put_price(self):
        """Test Black-Scholes put option pricing."""
        price = self.bs_calculator.put_price(100, 100, 0.25, 0.05, 0.2)  
        self.assertAlmostEqual(price, 3.37, places=2)
    
    def test_backtest_engine_run(self):
        """Test backtest engine execution."""
        # Create sample data
        dates = pd.to_datetime(pd.date_range(start='2023-01-01', end='2023-01-31'))
        price_data = pd.DataFrame({
            'Open': np.random.uniform(98, 102, len(dates)),
            'High': np.random.uniform(100, 105, len(dates)),
            'Low': np.random.uniform(95, 100, len(dates)),
            'Close': np.random.uniform(99, 104, len(dates)),
            'Volume': np.random.randint(100000, 500000, len(dates))
        }, index=dates)
        
        self.backtest_engine.add_data("TEST", price_data)
        self.backtest_engine.set_strategy(buy_and_hold_strategy)
        
        results = self.backtest_engine.run_backtest('2023-01-01', '2023-01-31')
        
        self.assertIsNotNone(results)
        self.assertIn('total_return', results)
        self.assertIn('sharpe_ratio', results)

if __name__ == '__main__':
    unittest.main()

