"""
Backtesting utilities for trading strategies.
Based on the legacy altcap backtesting framework.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Callable
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OrderType(Enum):
    """Order types for trading."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderSide(Enum):
    """Order sides."""
    BUY = "buy"
    SELL = "sell"


@dataclass
class Order:
    """Trading order."""
    symbol: str
    side: OrderSide
    quantity: int
    order_type: OrderType
    price: Optional[float] = None
    stop_price: Optional[float] = None
    timestamp: Optional[datetime] = None
    filled: bool = False
    fill_price: Optional[float] = None


@dataclass
class Position:
    """Trading position."""
    symbol: str
    quantity: int
    avg_price: float
    current_price: float
    unrealized_pnl: float
    realized_pnl: float
    
    @property
    def market_value(self) -> float:
        return self.quantity * self.current_price
    
    @property
    def total_pnl(self) -> float:
        return self.unrealized_pnl + self.realized_pnl


class Portfolio:
    """Portfolio management for backtesting."""
    
    def __init__(self, initial_capital: float = 100000):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions: Dict[str, Position] = {}
        self.orders: List[Order] = []
        self.trades: List[Dict] = []
        self.equity_curve: List[Dict] = []
        
    def get_portfolio_value(self) -> float:
        """Calculate total portfolio value."""
        positions_value = sum(pos.market_value for pos in self.positions.values())
        return self.cash + positions_value
    
    def get_position(self, symbol: str) -> Optional[Position]:
        """Get position for a symbol."""
        return self.positions.get(symbol)
    
    def update_position_prices(self, prices: Dict[str, float]):
        """Update current prices for all positions."""
        for symbol, position in self.positions.items():
            if symbol in prices:
                old_price = position.current_price
                position.current_price = prices[symbol]
                position.unrealized_pnl = (position.current_price - position.avg_price) * position.quantity
    
    def place_order(self, order: Order) -> bool:
        """Place a trading order."""
        order.timestamp = datetime.now()
        self.orders.append(order)
        return True
    
    def execute_order(self, order: Order, current_price: float) -> bool:
        """Execute a trading order."""
        if order.filled:
            return False
        
        # Simple market order execution
        if order.order_type == OrderType.MARKET:
            return self._fill_order(order, current_price)
        
        # Limit order logic
        elif order.order_type == OrderType.LIMIT:
            if order.side == OrderSide.BUY and current_price <= order.price:
                return self._fill_order(order, order.price)
            elif order.side == OrderSide.SELL and current_price >= order.price:
                return self._fill_order(order, order.price)
        
        return False
    
    def _fill_order(self, order: Order, fill_price: float) -> bool:
        """Fill an order."""
        cost = order.quantity * fill_price
        
        if order.side == OrderSide.BUY:
            if self.cash < cost:
                logger.warning(f"Insufficient cash for order: {order}")
                return False
            
            self.cash -= cost
            
            if order.symbol in self.positions:
                # Update existing position
                pos = self.positions[order.symbol]
                total_quantity = pos.quantity + order.quantity
                total_cost = pos.quantity * pos.avg_price + cost
                pos.avg_price = total_cost / total_quantity
                pos.quantity = total_quantity
            else:
                # Create new position
                self.positions[order.symbol] = Position(
                    symbol=order.symbol,
                    quantity=order.quantity,
                    avg_price=fill_price,
                    current_price=fill_price,
                    unrealized_pnl=0.0,
                    realized_pnl=0.0
                )
        
        else:  # SELL
            if order.symbol not in self.positions:
                logger.warning(f"No position to sell: {order}")
                return False
            
            pos = self.positions[order.symbol]
            if pos.quantity < order.quantity:
                logger.warning(f"Insufficient shares to sell: {order}")
                return False
            
            self.cash += cost
            
            # Calculate realized PnL
            realized_pnl = (fill_price - pos.avg_price) * order.quantity
            pos.realized_pnl += realized_pnl
            pos.quantity -= order.quantity
            
            # Remove position if fully closed
            if pos.quantity == 0:
                del self.positions[order.symbol]
        
        # Record the trade
        order.filled = True
        order.fill_price = fill_price
        
        self.trades.append({
            'timestamp': order.timestamp,
            'symbol': order.symbol,
            'side': order.side.value,
            'quantity': order.quantity,
            'price': fill_price,
            'value': cost
        })
        
        return True


class BacktestEngine:
    """Main backtesting engine."""
    
    def __init__(self, initial_capital: float = 100000):
        self.portfolio = Portfolio(initial_capital)
        self.data: Dict[str, pd.DataFrame] = {}
        self.strategy_func: Optional[Callable] = None
        self.results: Dict = {}
        
    def add_data(self, symbol: str, data: pd.DataFrame):
        """Add price data for a symbol."""
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in data.columns for col in required_columns):
            raise ValueError(f"Data must contain columns: {required_columns}")
        
        self.data[symbol] = data.copy()
        logger.info(f"Added data for {symbol}: {len(data)} rows")
    
    def set_strategy(self, strategy_func: Callable):
        """Set the trading strategy function."""
        self.strategy_func = strategy_func
    
    def run_backtest(self, start_date: Optional[str] = None, 
                    end_date: Optional[str] = None) -> Dict:
        """Run the backtest."""
        if not self.strategy_func:
            raise ValueError("Strategy function not set")
        
        if not self.data:
            raise ValueError("No data loaded")
        
        # Get date range
        all_dates = set()
        for df in self.data.values():
            all_dates.update(df.index)
        
        all_dates = sorted(list(all_dates))
        
        if start_date:
            all_dates = [d for d in all_dates if d >= pd.to_datetime(start_date)]
        if end_date:
            all_dates = [d for d in all_dates if d <= pd.to_datetime(end_date)]
        
        logger.info(f"Running backtest from {all_dates[0]} to {all_dates[-1]}")
        
        # Run backtest day by day
        for current_date in all_dates:
            # Get current prices
            current_prices = {}
            current_data = {}
            
            for symbol, df in self.data.items():
                if current_date in df.index:
                    row = df.loc[current_date]
                    current_prices[symbol] = row['Close']
                    current_data[symbol] = row
            
            if not current_prices:
                continue
            
            # Update portfolio with current prices
            self.portfolio.update_position_prices(current_prices)
            
            # Execute pending orders
            for order in self.portfolio.orders:
                if not order.filled and order.symbol in current_prices:
                    self.portfolio.execute_order(order, current_prices[order.symbol])
            
            # Run strategy
            signals = self.strategy_func(current_data, self.portfolio, current_date)
            
            # Process signals
            if signals:
                for signal in signals:
                    order = Order(
                        symbol=signal['symbol'],
                        side=OrderSide(signal['side']),
                        quantity=signal['quantity'],
                        order_type=OrderType(signal.get('order_type', 'market')),
                        price=signal.get('price'),
                        timestamp=current_date
                    )
                    
                    if order.order_type == OrderType.MARKET:
                        self.portfolio.execute_order(order, current_prices[signal['symbol']])
                    else:
                        self.portfolio.place_order(order)
            
            # Record equity curve
            portfolio_value = self.portfolio.get_portfolio_value()
            self.portfolio.equity_curve.append({
                'date': current_date,
                'portfolio_value': portfolio_value,
                'cash': self.portfolio.cash,
                'positions_value': portfolio_value - self.portfolio.cash
            })
        
        # Calculate results
        self.results = self._calculate_results()
        return self.results
    
    def _calculate_results(self) -> Dict:
        """Calculate backtest results and metrics."""
        if not self.portfolio.equity_curve:
            return {}
        
        equity_df = pd.DataFrame(self.portfolio.equity_curve)
        equity_df.set_index('date', inplace=True)
        
        # Calculate returns
        equity_df['returns'] = equity_df['portfolio_value'].pct_change()
        equity_df['cumulative_returns'] = (1 + equity_df['returns']).cumprod() - 1
        
        # Performance metrics
        total_return = (equity_df['portfolio_value'].iloc[-1] / self.portfolio.initial_capital) - 1
        
        # Annualized return
        days = (equity_df.index[-1] - equity_df.index[0]).days
        annualized_return = (1 + total_return) ** (365.25 / days) - 1
        
        # Volatility
        volatility = equity_df['returns'].std() * np.sqrt(252)
        
        # Sharpe ratio (assuming 0% risk-free rate)
        sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
        
        # Maximum drawdown
        rolling_max = equity_df['portfolio_value'].expanding().max()
        drawdown = (equity_df['portfolio_value'] - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        # Win rate
        winning_trades = sum(1 for trade in self.portfolio.trades 
                           if self._calculate_trade_pnl(trade) > 0)
        total_trades = len(self.portfolio.trades)
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        results = {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'total_trades': total_trades,
            'final_portfolio_value': equity_df['portfolio_value'].iloc[-1],
            'equity_curve': equity_df,
            'trades': self.portfolio.trades
        }
        
        return results
    
    def _calculate_trade_pnl(self, trade: Dict) -> float:
        """Calculate PnL for a trade (simplified)."""
        # This is a simplified calculation
        # In practice, you'd need to match buy/sell orders
        return 0.0
    
    def plot_results(self):
        """Plot backtest results."""
        if not self.results:
            logger.warning("No results to plot. Run backtest first.")
            return
        
        equity_df = self.results['equity_curve']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Equity curve
        axes[0, 0].plot(equity_df.index, equity_df['portfolio_value'])
        axes[0, 0].set_title('Portfolio Value Over Time')
        axes[0, 0].set_ylabel('Portfolio Value ($)')
        
        # Returns distribution
        axes[0, 1].hist(equity_df['returns'].dropna(), bins=50, alpha=0.7)
        axes[0, 1].set_title('Returns Distribution')
        axes[0, 1].set_xlabel('Daily Returns')
        
        # Drawdown
        rolling_max = equity_df['portfolio_value'].expanding().max()
        drawdown = (equity_df['portfolio_value'] - rolling_max) / rolling_max
        axes[1, 0].fill_between(equity_df.index, drawdown, 0, alpha=0.7, color='red')
        axes[1, 0].set_title('Drawdown')
        axes[1, 0].set_ylabel('Drawdown (%)')
        
        # Monthly returns heatmap
        monthly_returns = equity_df['returns'].resample('M').apply(lambda x: (1 + x).prod() - 1)
        monthly_returns.index = monthly_returns.index.strftime('%Y-%m')
        
        if len(monthly_returns) > 1:
            # Reshape for heatmap (simplified)
            axes[1, 1].bar(range(len(monthly_returns)), monthly_returns.values)
            axes[1, 1].set_title('Monthly Returns')
            axes[1, 1].set_ylabel('Monthly Return (%)')
        
        plt.tight_layout()
        plt.show()
        
        # Print summary statistics
        print("\n=== Backtest Results ===")
        print(f"Total Return: {self.results['total_return']:.2%}")
        print(f"Annualized Return: {self.results['annualized_return']:.2%}")
        print(f"Volatility: {self.results['volatility']:.2%}")
        print(f"Sharpe Ratio: {self.results['sharpe_ratio']:.2f}")
        print(f"Maximum Drawdown: {self.results['max_drawdown']:.2%}")
        print(f"Win Rate: {self.results['win_rate']:.2%}")
        print(f"Total Trades: {self.results['total_trades']}")
        print(f"Final Portfolio Value: ${self.results['final_portfolio_value']:,.2f}")


# Example strategies
def buy_and_hold_strategy(data: Dict, portfolio: Portfolio, current_date: datetime) -> List[Dict]:
    """Simple buy and hold strategy."""
    signals = []
    
    # Buy on first day if we don't have positions
    if not portfolio.positions and portfolio.cash > 0:
        for symbol in data.keys():
            signals.append({
                'symbol': symbol,
                'side': 'buy',
                'quantity': int(portfolio.cash * 0.8 / len(data) / data[symbol]['Close']),
                'order_type': 'market'
            })
    
    return signals


def moving_average_crossover_strategy(data: Dict, portfolio: Portfolio, 
                                    current_date: datetime, 
                                    short_window: int = 20, 
                                    long_window: int = 50) -> List[Dict]:
    """Moving average crossover strategy."""
    signals = []
    
    # This is a simplified version - in practice you'd need historical data
    # for calculating moving averages
    
    return signals


if __name__ == "__main__":
    # Test the backtesting engine
    from utils.yfinance_util import YFinanceDataProvider
    
    print("Testing backtesting engine...")
    
    # Get some test data
    provider = YFinanceDataProvider()
    data = provider.get_historical_data("AAPL", "1y")
    
    if not data.empty:
        # Create backtest engine
        engine = BacktestEngine(initial_capital=100000)
        engine.add_data("AAPL", data)
        engine.set_strategy(buy_and_hold_strategy)
        
        # Run backtest
        results = engine.run_backtest()
        
        print(f"Backtest completed!")
        print(f"Total return: {results['total_return']:.2%}")
        print(f"Sharpe ratio: {results['sharpe_ratio']:.2f}")
    else:
        print("No data available for testing")

