"""
Polygon.io API utility module for AltSignals platform.
Provides functions to fetch real-time and historical market data.
"""

import os
import requests
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime, timedelta
from polygon import RESTClient
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PolygonDataProvider:
    """Polygon.io data provider for market data."""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv('POLYGON_API_KEY')
        if not self.api_key:
            raise ValueError("Polygon API key not found. Set POLYGON_API_KEY environment variable.")
        
        self.client = RESTClient(self.api_key)
        self.base_url = "https://api.polygon.io"
    
    def get_stock_details(self, symbol: str) -> Dict:
        """Get detailed information about a stock."""
        try:
            # Get ticker details
            ticker_details = self.client.get_ticker_details(symbol)
            
            stock_info = {
                'symbol': symbol,
                'name': getattr(ticker_details, 'name', symbol),
                'description': getattr(ticker_details, 'description', ''),
                'market': getattr(ticker_details, 'market', ''),
                'locale': getattr(ticker_details, 'locale', ''),
                'primary_exchange': getattr(ticker_details, 'primary_exchange', ''),
                'type': getattr(ticker_details, 'type', ''),
                'currency_name': getattr(ticker_details, 'currency_name', 'USD'),
                'market_cap': getattr(ticker_details, 'market_cap', 0),
                'share_class_shares_outstanding': getattr(ticker_details, 'share_class_shares_outstanding', 0),
                'weighted_shares_outstanding': getattr(ticker_details, 'weighted_shares_outstanding', 0),
                'sic_code': getattr(ticker_details, 'sic_code', ''),
                'sic_description': getattr(ticker_details, 'sic_description', ''),
                'homepage_url': getattr(ticker_details, 'homepage_url', ''),
            }
            
            logger.info(f"Retrieved details for {symbol}")
            return stock_info
            
        except Exception as e:
            logger.error(f"Error fetching details for {symbol}: {e}")
            return {}
    
    def get_historical_data(self, symbol: str, start_date: str, 
                          end_date: str, timespan: str = "day", 
                          multiplier: int = 1) -> pd.DataFrame:
        """Get historical price data."""
        try:
            # Get aggregates (OHLCV data)
            aggs = self.client.get_aggs(
                ticker=symbol,
                multiplier=multiplier,
                timespan=timespan,
                from_=start_date,
                to=end_date,
                adjusted=True,
                sort="asc",
                limit=50000
            )
            
            if not aggs:
                logger.warning(f"No data found for {symbol}")
                return pd.DataFrame()
            
            # Convert to DataFrame
            data = []
            for agg in aggs:
                data.append({
                    'Date': pd.to_datetime(agg.timestamp, unit='ms'),
                    'Open': agg.open,
                    'High': agg.high,
                    'Low': agg.low,
                    'Close': agg.close,
                    'Volume': agg.volume,
                    'VWAP': getattr(agg, 'vwap', None),
                    'Transactions': getattr(agg, 'transactions', None),
                    'Symbol': symbol
                })
            
            df = pd.DataFrame(data)
            df.set_index('Date', inplace=True)
            
            logger.info(f"Retrieved {len(df)} records for {symbol}")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching historical data for {symbol}: {e}")
            return pd.DataFrame()
    
    def get_real_time_quote(self, symbol: str) -> Dict:
        """Get real-time quote for a symbol."""
        try:
            quote = self.client.get_last_quote(symbol)
            
            if not quote:
                return {}
            
            quote_data = {
                'symbol': symbol,
                'bid': quote.bid,
                'bid_size': quote.bid_size,
                'ask': quote.ask,
                'ask_size': quote.ask_size,
                'exchange': quote.exchange,
                'timestamp': pd.to_datetime(quote.timestamp, unit='ns'),
                'spread': quote.ask - quote.bid,
                'mid_price': (quote.bid + quote.ask) / 2
            }
            
            return quote_data
            
        except Exception as e:
            logger.error(f"Error fetching quote for {symbol}: {e}")
            return {}
    
    def get_real_time_trade(self, symbol: str) -> Dict:
        """Get last trade for a symbol."""
        try:
            trade = self.client.get_last_trade(symbol)
            
            if not trade:
                return {}
            
            trade_data = {
                'symbol': symbol,
                'price': trade.price,
                'size': trade.size,
                'exchange': trade.exchange,
                'timestamp': pd.to_datetime(trade.timestamp, unit='ns'),
                'conditions': getattr(trade, 'conditions', []),
            }
            
            return trade_data
            
        except Exception as e:
            logger.error(f"Error fetching trade for {symbol}: {e}")
            return {}
    
    def get_market_status(self) -> Dict:
        """Get current market status."""
        try:
            status = self.client.get_market_status()
            
            market_info = {
                'market': status.market,
                'server_time': status.serverTime,
                'exchanges': {}
            }
            
            # Add exchange information
            if hasattr(status, 'exchanges'):
                for exchange_name, exchange_info in status.exchanges.items():
                    market_info['exchanges'][exchange_name] = {
                        'name': exchange_info.name,
                        'status': exchange_info.status,
                        'open': exchange_info.open,
                        'close': exchange_info.close
                    }
            
            return market_info
            
        except Exception as e:
            logger.error(f"Error fetching market status: {e}")
            return {}
    
    def get_news(self, symbol: str = None, limit: int = 10) -> List[Dict]:
        """Get news articles."""
        try:
            news_items = self.client.list_ticker_news(
                ticker=symbol,
                limit=limit,
                sort="published_utc",
                order="desc"
            )
            
            news_data = []
            for item in news_items:
                news_data.append({
                    'id': item.id,
                    'title': item.title,
                    'author': item.author,
                    'published_utc': item.published_utc,
                    'article_url': item.article_url,
                    'description': getattr(item, 'description', ''),
                    'keywords': getattr(item, 'keywords', []),
                    'tickers': getattr(item, 'tickers', []),
                })
            
            return news_data
            
        except Exception as e:
            logger.error(f"Error fetching news: {e}")
            return []
    
    def get_financials(self, symbol: str) -> Dict:
        """Get financial data for a symbol."""
        try:
            # Get financials
            financials = self.client.get_ticker_details(symbol)
            
            financial_data = {
                'symbol': symbol,
                'market_cap': getattr(financials, 'market_cap', 0),
                'shares_outstanding': getattr(financials, 'share_class_shares_outstanding', 0),
                'weighted_shares_outstanding': getattr(financials, 'weighted_shares_outstanding', 0),
            }
            
            return financial_data
            
        except Exception as e:
            logger.error(f"Error fetching financials for {symbol}: {e}")
            return {}
    
    def get_technical_indicators(self, symbol: str, indicator: str, 
                               timestamp: str = None, **kwargs) -> Dict:
        """Get technical indicators (requires premium subscription)."""
        try:
            # This is a placeholder - actual implementation depends on Polygon's technical indicators API
            # which requires a premium subscription
            
            indicators_data = {
                'symbol': symbol,
                'indicator': indicator,
                'timestamp': timestamp,
                'values': {},
                'note': 'Technical indicators require premium Polygon subscription'
            }
            
            return indicators_data
            
        except Exception as e:
            logger.error(f"Error fetching technical indicators for {symbol}: {e}")
            return {}
    
    def search_tickers(self, query: str, market: str = "stocks", 
                      limit: int = 10) -> List[Dict]:
        """Search for tickers."""
        try:
            url = f"{self.base_url}/v3/reference/tickers"
            params = {
                'search': query,
                'market': market,
                'active': 'true',
                'limit': limit,
                'apikey': self.api_key
            }
            
            response = requests.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            results = data.get('results', [])
            
            tickers = []
            for result in results:
                tickers.append({
                    'ticker': result.get('ticker'),
                    'name': result.get('name'),
                    'market': result.get('market'),
                    'locale': result.get('locale'),
                    'primary_exchange': result.get('primary_exchange'),
                    'type': result.get('type'),
                    'currency_name': result.get('currency_name'),
                })
            
            return tickers
            
        except Exception as e:
            logger.error(f"Error searching tickers: {e}")
            return []


def get_polygon_data(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Convenience function to get Polygon data."""
    provider = PolygonDataProvider()
    return provider.get_historical_data(symbol, start_date, end_date)


def get_real_time_data(symbol: str) -> Dict:
    """Convenience function to get real-time data."""
    provider = PolygonDataProvider()
    quote = provider.get_real_time_quote(symbol)
    trade = provider.get_real_time_trade(symbol)
    
    return {
        'quote': quote,
        'trade': trade
    }


if __name__ == "__main__":
    # Test the module
    try:
        provider = PolygonDataProvider()
        
        # Test with AAPL
        print("Testing with AAPL...")
        details = provider.get_stock_details("AAPL")
        print(f"Stock details: {details}")
        
        # Test historical data
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
        
        data = provider.get_historical_data("AAPL", start_date, end_date)
        print(f"Historical data shape: {data.shape}")
        
        # Test real-time quote
        quote = provider.get_real_time_quote("AAPL")
        print(f"Real-time quote: {quote}")
        
    except Exception as e:
        print(f"Test failed: {e}")
        print("Make sure POLYGON_API_KEY is set in your environment.")

