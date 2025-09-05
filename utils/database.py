"""
Database models and utilities for AltSignals platform.
Uses SQLite for local storage with SQLAlchemy ORM.
"""

import os
import sqlite3
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean, Text, ForeignKey
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from sqlalchemy.sql import func
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
import logging
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database setup
Base = declarative_base()


class Stock(Base):
    """Stock information table."""
    __tablename__ = 'stocks'
    
    id = Column(Integer, primary_key=True)
    symbol = Column(String(10), unique=True, nullable=False, index=True)
    name = Column(String(255), nullable=False)
    sector = Column(String(100))
    industry = Column(String(100))
    market_cap = Column(Float)
    current_price = Column(Float)
    pe_ratio = Column(Float)
    forward_pe = Column(Float)
    price_to_book = Column(Float)
    debt_to_equity = Column(Float)
    roe = Column(Float)
    dividend_yield = Column(Float)
    beta = Column(Float)
    volume = Column(Integer)
    avg_volume = Column(Integer)
    week_52_high = Column(Float)
    week_52_low = Column(Float)
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    # Relationships
    prices = relationship("StockPrice", back_populates="stock", cascade="all, delete-orphan")
    signals = relationship("Signal", back_populates="stock", cascade="all, delete-orphan")
    news = relationship("NewsItem", back_populates="stock", cascade="all, delete-orphan")


class StockPrice(Base):
    """Historical stock price data."""
    __tablename__ = 'stock_prices'
    
    id = Column(Integer, primary_key=True)
    stock_id = Column(Integer, ForeignKey('stocks.id'), nullable=False)
    date = Column(DateTime, nullable=False, index=True)
    open_price = Column(Float, nullable=False)
    high_price = Column(Float, nullable=False)
    low_price = Column(Float, nullable=False)
    close_price = Column(Float, nullable=False)
    volume = Column(Integer, nullable=False)
    adjusted_close = Column(Float)
    returns = Column(Float)
    log_returns = Column(Float)
    volatility = Column(Float)
    created_at = Column(DateTime, default=func.now())
    
    # Relationships
    stock = relationship("Stock", back_populates="prices")


class Signal(Base):
    """Trading signals and predictions."""
    __tablename__ = 'signals'
    
    id = Column(Integer, primary_key=True)
    stock_id = Column(Integer, ForeignKey('stocks.id'), nullable=False)
    signal_type = Column(String(50), nullable=False)  # 'buy', 'sell', 'hold'
    signal_source = Column(String(100), nullable=False)  # 'news_sentiment', 'technical', etc.
    strength = Column(Float, nullable=False)  # Signal strength 0-1
    confidence = Column(Float, nullable=False)  # Confidence level 0-1
    target_price = Column(Float)
    stop_loss = Column(Float)
    time_horizon = Column(String(20))  # 'short', 'medium', 'long'
    description = Column(Text)
    signal_metadata = Column(Text)  # JSON string for additional data
    created_at = Column(DateTime, default=func.now())
    expires_at = Column(DateTime)
    is_active = Column(Boolean, default=True)
    
    # Relationships
    stock = relationship("Stock", back_populates="signals")


class NewsItem(Base):
    """News articles and sentiment data."""
    __tablename__ = 'news'
    
    id = Column(Integer, primary_key=True)
    stock_id = Column(Integer, ForeignKey('stocks.id'), nullable=False)
    title = Column(String(500), nullable=False)
    content = Column(Text)
    author = Column(String(255))
    source = Column(String(255))
    url = Column(String(1000))
    published_at = Column(DateTime, nullable=False)
    sentiment_score = Column(Float)  # -1 to 1
    sentiment_label = Column(String(20))  # 'positive', 'negative', 'neutral'
    relevance_score = Column(Float)  # 0 to 1
    keywords = Column(Text)  # JSON array of keywords
    created_at = Column(DateTime, default=func.now())
    
    # Relationships
    stock = relationship("Stock", back_populates="news")


class BacktestResult(Base):
    """Backtest results storage."""
    __tablename__ = 'backtest_results'
    
    id = Column(Integer, primary_key=True)
    strategy_name = Column(String(255), nullable=False)
    symbols = Column(Text, nullable=False)  # JSON array of symbols
    start_date = Column(DateTime, nullable=False)
    end_date = Column(DateTime, nullable=False)
    initial_capital = Column(Float, nullable=False)
    final_value = Column(Float, nullable=False)
    total_return = Column(Float, nullable=False)
    annualized_return = Column(Float, nullable=False)
    volatility = Column(Float, nullable=False)
    sharpe_ratio = Column(Float, nullable=False)
    max_drawdown = Column(Float, nullable=False)
    win_rate = Column(Float, nullable=False)
    total_trades = Column(Integer, nullable=False)
    parameters = Column(Text)  # JSON string of strategy parameters
    equity_curve = Column(Text)  # JSON string of equity curve data
    trades = Column(Text)  # JSON string of trade history
    created_at = Column(DateTime, default=func.now())


class OptionsData(Base):
    """Options data and Greeks."""
    __tablename__ = 'options'
    
    id = Column(Integer, primary_key=True)
    stock_id = Column(Integer, ForeignKey('stocks.id'), nullable=False)
    option_type = Column(String(10), nullable=False)  # 'call' or 'put'
    strike_price = Column(Float, nullable=False)
    expiry_date = Column(DateTime, nullable=False)
    current_price = Column(Float, nullable=False)
    implied_volatility = Column(Float)
    delta = Column(Float)
    gamma = Column(Float)
    theta = Column(Float)
    vega = Column(Float)
    rho = Column(Float)
    open_interest = Column(Integer)
    volume = Column(Integer)
    bid = Column(Float)
    ask = Column(Float)
    last_price = Column(Float)
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())


class DatabaseManager:
    """Database manager for AltSignals platform."""

    def __init__(self, db_url: str = "sqlite:///db/altsignals.db"):
        self.db_url = db_url
        self.engine = create_engine(
            self.db_url,
            connect_args={"check_same_thread": False} if "sqlite" in self.db_url else {}
        )
        self.Session = sessionmaker(bind=self.engine)
        Base.metadata.create_all(self.engine)
        logger.info(f"Database initialized: {self.db_url}")

    def get_session(self):
        return self.Session()
    
    def generate_synthetic_data(self, num_stocks=3, num_days=365 * 2):
        """Generate synthetic data for testing."""
        session = self.get_session()
        try:
            # Add stocks
            stock_symbols = ["AAPL", "MSFT", "GOOGL"]
            for i in range(num_stocks):
                symbol = stock_symbols[i]
                stock = Stock(
                    symbol=symbol,
                    name=f"{symbol} Inc. (SYNTHETIC)",
                    current_price=np.random.uniform(100, 500),
                    market_cap=np.random.uniform(1e12, 3e12),
                    pe_ratio=np.random.uniform(15, 40),
                    sector="Technology",
                    industry="Consumer Electronics",
                )
                session.add(stock)
            session.commit()

            # Add stock prices
            for symbol in stock_symbols:
                start_date = datetime.now() - timedelta(days=num_days)
                for i in range(num_days):
                    date = start_date + timedelta(days=i)
                    stock = session.query(Stock).filter(Stock.symbol == symbol).first()
                    price = StockPrice(
                            stock_id=stock.id,
                        date=date,
                        open_price=np.random.uniform(100, 500),
                        high_price=np.random.uniform(100, 500),
                        low_price=np.random.uniform(100, 500),
                        close_price=np.random.uniform(100, 500),
                        volume=np.random.randint(1e6, 1e8),
                    )
                    session.add(price)

            session.commit()
            logger.info("Synthetic data generated.")
        except Exception as e:
            logger.error(f"Error generating synthetic data: {e}")
            session.rollback()
        finally:
            session.close()

    def add_stock(self, stock_data: Dict) -> Stock:

        """Add or update stock information."""
        session = self.get_session()
        try:
            # Check if stock exists
            stock = session.query(Stock).filter(Stock.symbol == stock_data['symbol']).first()
            
            if stock:
                # Update existing stock
                for key, value in stock_data.items():
                    if hasattr(stock, key):
                        setattr(stock, key, value)
                stock.updated_at = datetime.now()
            else:
                # Create new stock
                stock = Stock(**stock_data)
                session.add(stock)
            
            session.commit()
            session.refresh(stock)
            return stock
            
        except Exception as e:
            session.rollback()
            logger.error(f"Error adding stock: {e}")
            raise
        finally:
            session.close()
    
    def add_stock_prices(self, symbol: str, price_data: pd.DataFrame) -> int:
        """Add stock price data."""
        session = self.get_session()
        try:
            # Get stock
            stock = session.query(Stock).filter(Stock.symbol == symbol).first()
            if not stock:
                raise ValueError(f"Stock {symbol} not found")
            
            # Prepare price records
            price_records = []
            for _, row in price_data.iterrows():
                price_record = StockPrice(
                    stock_id=stock.id,
                    date=row.name if hasattr(row.name, 'date') else row.name,
                    open_price=row.get('Open', 0),
                    high_price=row.get('High', 0),
                    low_price=row.get('Low', 0),
                    close_price=row.get('Close', 0),
                    volume=row.get('Volume', 0),
                    adjusted_close=row.get('Adj Close', row.get('Close', 0)),
                    returns=row.get('Returns'),
                    log_returns=row.get('Log_Returns'),
                    volatility=row.get('Volatility')
                )
                price_records.append(price_record)
            
            # Bulk insert
            session.bulk_save_objects(price_records)
            session.commit()
            
            logger.info(f"Added {len(price_records)} price records for {symbol}")
            return len(price_records)
            
        except Exception as e:
            session.rollback()
            logger.error(f"Error adding price data: {e}")
            raise
        finally:
            session.close()
    
    def add_signal(self, signal_data: Dict) -> Signal:
        """Add trading signal."""
        session = self.get_session()
        try:
            # Get stock
            stock = session.query(Stock).filter(Stock.symbol == signal_data['symbol']).first()
            if not stock:
                raise ValueError(f"Stock {signal_data['symbol']} not found")
            
            signal_data['stock_id'] = stock.id
            signal_data.pop('symbol', None)  # Remove symbol as we have stock_id
            
            signal = Signal(**signal_data)
            session.add(signal)
            session.commit()
            session.refresh(signal)
            
            return signal
            
        except Exception as e:
            session.rollback()
            logger.error(f"Error adding signal: {e}")
            raise
        finally:
            session.close()
    
    def add_news_item(self, news_data: Dict) -> NewsItem:
        """Add news item."""
        session = self.get_session()
        try:
            # Get stock
            stock = session.query(Stock).filter(Stock.symbol == news_data['symbol']).first()
            if not stock:
                raise ValueError(f"Stock {news_data['symbol']} not found")
            
            news_data['stock_id'] = stock.id
            news_data.pop('symbol', None)
            
            news_item = NewsItem(**news_data)
            session.add(news_item)
            session.commit()
            session.refresh(news_item)
            
            return news_item
            
        except Exception as e:
            session.rollback()
            logger.error(f"Error adding news item: {e}")
            raise
        finally:
            session.close()
    
    def save_backtest_result(self, result_data: Dict) -> BacktestResult:
        """Save backtest result."""
        session = self.get_session()
        try:
            # Convert complex data to JSON strings
            if 'equity_curve' in result_data and isinstance(result_data['equity_curve'], pd.DataFrame):
                result_data['equity_curve'] = result_data['equity_curve'].to_json()
            
            if 'trades' in result_data:
                result_data['trades'] = json.dumps(result_data['trades'])
            
            if 'parameters' in result_data:
                result_data['parameters'] = json.dumps(result_data['parameters'])
            
            if 'symbols' in result_data:
                result_data['symbols'] = json.dumps(result_data['symbols'])
            
            backtest = BacktestResult(**result_data)
            session.add(backtest)
            session.commit()
            session.refresh(backtest)
            
            return backtest
            
        except Exception as e:
            session.rollback()
            logger.error(f"Error saving backtest result: {e}")
            raise
        finally:
            session.close()
    
    def get_stock_data(self, symbol: str) -> Optional[Dict]:
        """Get stock data with prices."""
        session = self.get_session()
        try:
            stock = session.query(Stock).filter(Stock.symbol == symbol).first()
            if not stock:
                return None
            
            # Get recent prices
            recent_prices = session.query(StockPrice)\
                .filter(StockPrice.stock_id == stock.id)\
                .order_by(StockPrice.date.desc())\
                .limit(252).all()  # Last year of data
            
            return {
                'stock_info': {
                    'symbol': stock.symbol,
                    'name': stock.name,
                    'sector': stock.sector,
                    'industry': stock.industry,
                    'market_cap': stock.market_cap,
                    'current_price': stock.current_price,
                    'pe_ratio': stock.pe_ratio,
                    'beta': stock.beta
                },
                'prices': [
                    {
                        'date': price.date,
                        'open': price.open_price,
                        'high': price.high_price,
                        'low': price.low_price,
                        'close': price.close_price,
                        'volume': price.volume
                    }
                    for price in recent_prices
                ]
            }
            
        except Exception as e:
            logger.error(f"Error getting stock data: {e}")
            return None
        finally:
            session.close()
    
    def get_signals(self, symbol: str = None, limit: int = 100) -> List[Dict]:
        """Get trading signals."""
        session = self.get_session()
        try:
            query = session.query(Signal).join(Stock)
            
            if symbol:
                query = query.filter(Stock.symbol == symbol)
            
            signals = query.filter(Signal.is_active == True)\
                .order_by(Signal.created_at.desc())\
                .limit(limit).all()
            
            return [
                {
                    'id': signal.id,
                    'symbol': signal.stock.symbol,
                    'signal_type': signal.signal_type,
                    'signal_source': signal.signal_source,
                    'strength': signal.strength,
                    'confidence': signal.confidence,
                    'target_price': signal.target_price,
                    'description': signal.description,
                    'created_at': signal.created_at
                }
                for signal in signals
            ]
            
        except Exception as e:
            logger.error(f"Error getting signals: {e}")
            return []
        finally:
            session.close()
    
    def search_stocks(self, query: str, limit: int = 20) -> List[Dict]:
        """Search stocks by symbol or name."""
        session = self.get_session()
        try:
            stocks = session.query(Stock)\
                .filter(
                    (Stock.symbol.ilike(f"%{query}%")) |
                    (Stock.name.ilike(f"%{query}%"))
                )\
                .limit(limit).all()
            
            return [
                {
                    'symbol': stock.symbol,
                    'name': stock.name,
                    'sector': stock.sector,
                    'current_price': stock.current_price,
                    'market_cap': stock.market_cap
                }
                for stock in stocks
            ]
            
        except Exception as e:
            logger.error(f"Error searching stocks: {e}")
            return []
        finally:
            session.close()


def generate_synthetic_data(db_manager: DatabaseManager, symbols: List[str]):
    """Generate synthetic data for testing purposes."""
    logger.info("Generating synthetic data (MARKED AS SYNTHETIC)")
    
    # Popular stocks for testing
    synthetic_stocks = [
        {'symbol': 'AAPL', 'name': 'Apple Inc. (SYNTHETIC)', 'sector': 'Technology', 'industry': 'Consumer Electronics'},
        {'symbol': 'MSFT', 'name': 'Microsoft Corporation (SYNTHETIC)', 'sector': 'Technology', 'industry': 'Software'},
        {'symbol': 'GOOGL', 'name': 'Alphabet Inc. (SYNTHETIC)', 'sector': 'Technology', 'industry': 'Internet Content'},
        {'symbol': 'AMZN', 'name': 'Amazon.com Inc. (SYNTHETIC)', 'sector': 'Consumer Discretionary', 'industry': 'E-commerce'},
        {'symbol': 'TSLA', 'name': 'Tesla Inc. (SYNTHETIC)', 'sector': 'Consumer Discretionary', 'industry': 'Electric Vehicles'},
    ]
    
    for stock_data in synthetic_stocks:
        if stock_data['symbol'] in symbols:
            # Add synthetic fundamental data
            stock_data.update({
                'market_cap': np.random.randint(100_000_000_000, 3_000_000_000_000),
                'current_price': np.random.uniform(50, 300),
                'pe_ratio': np.random.uniform(15, 40),
                'forward_pe': np.random.uniform(12, 35),
                'price_to_book': np.random.uniform(1, 10),
                'debt_to_equity': np.random.uniform(0, 100),
                'roe': np.random.uniform(0.05, 0.25),
                'dividend_yield': np.random.uniform(0, 0.05),
                'beta': np.random.uniform(0.5, 2.0),
                'volume': np.random.randint(10_000_000, 100_000_000),
                'avg_volume': np.random.randint(20_000_000, 80_000_000),
                'week_52_high': np.random.uniform(200, 400),
                'week_52_low': np.random.uniform(100, 200),
            })
            
            # Add to database
            stock = db_manager.add_stock(stock_data)
            
            # Generate synthetic price data
            dates = pd.date_range(start='2023-01-01', end='2024-12-31', freq='D')
            prices = []
            
            base_price = stock_data['current_price']
            for i, date in enumerate(dates):
                # Simple random walk for synthetic prices
                if i == 0:
                    price = base_price
                else:
                    price = prices[-1]['Close'] * (1 + np.random.normal(0, 0.02))
                
                daily_volatility = 0.02
                high = price * (1 + abs(np.random.normal(0, daily_volatility)))
                low = price * (1 - abs(np.random.normal(0, daily_volatility)))
                open_price = price * (1 + np.random.normal(0, daily_volatility/2))
                volume = np.random.randint(10_000_000, 50_000_000)
                
                prices.append({
                    'Date': date,
                    'Open': open_price,
                    'High': max(high, price, open_price),
                    'Low': min(low, price, open_price),
                    'Close': price,
                    'Volume': volume
                })
            
            price_df = pd.DataFrame(prices)
            price_df.set_index('Date', inplace=True)
            
            # Add to database
            db_manager.add_stock_prices(stock_data['symbol'], price_df)
            
            # Generate synthetic signals
            for _ in range(np.random.randint(5, 15)):
                signal_data = {
                    'symbol': stock_data['symbol'],
                    'signal_type': np.random.choice(['buy', 'sell', 'hold']),
                    'signal_source': np.random.choice(['news_sentiment', 'technical', 'fundamental', 'social_media']),
                    'strength': np.random.uniform(0.3, 1.0),
                    'confidence': np.random.uniform(0.5, 0.95),
                    'target_price': stock_data['current_price'] * np.random.uniform(0.9, 1.1),
                    'time_horizon': np.random.choice(['short', 'medium', 'long']),
                    'description': f"SYNTHETIC signal for {stock_data['symbol']} - generated for testing",
                    'created_at': datetime.now() - timedelta(days=np.random.randint(0, 30))
                }
                
                db_manager.add_signal(signal_data)
    
    logger.info("Synthetic data generation completed")


if __name__ == "__main__":
    # Test the database
    db_manager = DatabaseManager()
    
    # Generate some synthetic data
    test_symbols = ['AAPL', 'MSFT', 'GOOGL']
    generate_synthetic_data(db_manager, test_symbols)
    
    # Test queries
    print("Testing database queries...")
    
    # Search stocks
    stocks = db_manager.search_stocks("Apple")
    print(f"Found {len(stocks)} stocks matching 'Apple'")
    
    # Get stock data
    stock_data = db_manager.get_stock_data("AAPL")
    if stock_data:
        print(f"AAPL data: {len(stock_data['prices'])} price records")
    
    # Get signals
    signals = db_manager.get_signals("AAPL", limit=5)
    print(f"Found {len(signals)} signals for AAPL")
    
    print("Database testing completed!")

