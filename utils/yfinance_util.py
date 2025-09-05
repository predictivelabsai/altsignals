"""
Yahoo Finance utility module for AltSignals platform.
Provides functions to fetch market data, fundamentals, and historical prices.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class YFinanceDataProvider:
    """Yahoo Finance data provider for market data and fundamentals."""
    
    def __init__(self):
        self.cache = {}
    
    def get_stock_info(self, symbol: str) -> Dict:
        """Get basic stock information and fundamentals."""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            # Extract key metrics
            stock_info = {
                'symbol': symbol,
                'name': info.get('longName', symbol),
                'sector': info.get('sector', 'Unknown'),
                'industry': info.get('industry', 'Unknown'),
                'market_cap': info.get('marketCap', 0),
                'current_price': info.get('currentPrice', 0),
                'pe_ratio': info.get('trailingPE', 0),
                'forward_pe': info.get('forwardPE', 0),
                'price_to_book': info.get('priceToBook', 0),
                'debt_to_equity': info.get('debtToEquity', 0),
                'roe': info.get('returnOnEquity', 0),
                'dividend_yield': info.get('dividendYield', 0),
                'beta': info.get('beta', 1.0),
                'volume': info.get('volume', 0),
                'avg_volume': info.get('averageVolume', 0),
                '52_week_high': info.get('fiftyTwoWeekHigh', 0),
                '52_week_low': info.get('fiftyTwoWeekLow', 0),
            }
            
            logger.info(f"Retrieved info for {symbol}")
            return stock_info
            
        except Exception as e:
            logger.error(f"Error fetching info for {symbol}: {e}")
            return {}
    
    def get_historical_data(self, symbol: str, period: str = "1y", 
                          interval: str = "1d") -> pd.DataFrame:
        """Get historical price data."""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period, interval=interval)
            
            if data.empty:
                logger.warning(f"No data found for {symbol}")
                return pd.DataFrame()
            
            # Clean and prepare data
            data.reset_index(inplace=True)
            data['Symbol'] = symbol
            
            logger.info(f"Retrieved {len(data)} records for {symbol}")
            return data
            
        except Exception as e:
            logger.error(f"Error fetching historical data for {symbol}: {e}")
            return pd.DataFrame()
    
    def get_multiple_stocks_data(self, symbols: List[str], 
                               period: str = "1y") -> Dict[str, pd.DataFrame]:
        """Get historical data for multiple stocks."""
        data = {}
        for symbol in symbols:
            data[symbol] = self.get_historical_data(symbol, period)
        return data
    
    def calculate_returns(self, data: pd.DataFrame, 
                         period: int = 1) -> pd.DataFrame:
        """Calculate returns for given period."""
        if data.empty or 'Close' not in data.columns:
            return data
        
        data = data.copy()
        data['Returns'] = data['Close'].pct_change(period)
        data['Log_Returns'] = np.log(data['Close'] / data['Close'].shift(period))
        
        return data
    
    def calculate_volatility(self, data: pd.DataFrame, 
                           window: int = 30) -> pd.DataFrame:
        """Calculate rolling volatility."""
        if data.empty or 'Returns' not in data.columns:
            return data
        
        data = data.copy()
        data['Volatility'] = data['Returns'].rolling(window=window).std() * np.sqrt(252)
        
        return data
    
    def get_financial_ratios(self, symbol: str) -> Dict:
        """Get key financial ratios."""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            ratios = {
                'pe_ratio': info.get('trailingPE', 0),
                'forward_pe': info.get('forwardPE', 0),
                'peg_ratio': info.get('pegRatio', 0),
                'price_to_sales': info.get('priceToSalesTrailing12Months', 0),
                'price_to_book': info.get('priceToBook', 0),
                'enterprise_to_revenue': info.get('enterpriseToRevenue', 0),
                'enterprise_to_ebitda': info.get('enterpriseToEbitda', 0),
                'profit_margin': info.get('profitMargins', 0),
                'operating_margin': info.get('operatingMargins', 0),
                'roe': info.get('returnOnEquity', 0),
                'roa': info.get('returnOnAssets', 0),
                'debt_to_equity': info.get('debtToEquity', 0),
                'current_ratio': info.get('currentRatio', 0),
                'quick_ratio': info.get('quickRatio', 0),
            }
            
            return ratios
            
        except Exception as e:
            logger.error(f"Error fetching ratios for {symbol}: {e}")
            return {}
    
    def get_options_data(self, symbol: str) -> Dict:
        """Get options data for a symbol."""
        try:
            ticker = yf.Ticker(symbol)
            options_dates = ticker.options
            
            if not options_dates:
                return {}
            
            # Get options for the nearest expiration
            nearest_expiry = options_dates[0]
            options_chain = ticker.option_chain(nearest_expiry)
            
            return {
                'expiry_dates': list(options_dates),
                'nearest_expiry': nearest_expiry,
                'calls': options_chain.calls.to_dict('records'),
                'puts': options_chain.puts.to_dict('records')
            }
            
        except Exception as e:
            logger.error(f"Error fetching options for {symbol}: {e}")
            return {}
    
    def screen_stocks(self, criteria: Dict) -> List[str]:
        """Screen stocks based on criteria (simplified implementation)."""
        # This is a basic implementation - in production, you'd use more sophisticated screening
        popular_stocks = [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX',
            'AMD', 'INTC', 'ORCL', 'CRM', 'ADBE', 'PYPL', 'UBER', 'SPOT'
        ]
        
        filtered_stocks = []
        
        for symbol in popular_stocks:
            try:
                info = self.get_stock_info(symbol)
                
                # Apply basic filters
                if criteria.get('min_market_cap') and info.get('market_cap', 0) < criteria['min_market_cap']:
                    continue
                if criteria.get('max_pe') and info.get('pe_ratio', 0) > criteria['max_pe']:
                    continue
                if criteria.get('min_volume') and info.get('volume', 0) < criteria['min_volume']:
                    continue
                
                filtered_stocks.append(symbol)
                
            except Exception as e:
                logger.warning(f"Error screening {symbol}: {e}")
                continue
        
        return filtered_stocks


def get_market_data(symbol: str, period: str = "1y") -> pd.DataFrame:
    """Convenience function to get market data."""
    provider = YFinanceDataProvider()
    return provider.get_historical_data(symbol, period)


def get_stock_fundamentals(symbol: str) -> Dict:
    """Convenience function to get stock fundamentals."""
    provider = YFinanceDataProvider()
    return provider.get_stock_info(symbol)


if __name__ == "__main__":
    # Test the module
    provider = YFinanceDataProvider()
    
    # Test with AAPL
    print("Testing with AAPL...")
    info = provider.get_stock_info("AAPL")
    print(f"Stock info: {info}")
    
    data = provider.get_historical_data("AAPL", "3mo")
    print(f"Historical data shape: {data.shape}")
    
    if not data.empty:
        data_with_returns = provider.calculate_returns(data)
        print(f"Returns calculated: {'Returns' in data_with_returns.columns}")

