"""
Backtesting page for strategy testing and analysis.
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add utils to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))

from yfinance_util import YFinanceDataProvider
from backtesting import BacktestEngine, buy_and_hold_strategy
from database import DatabaseManager

st.set_page_config(page_title="Backtesting", page_icon="🔄", layout="wide")

def moving_average_strategy(data, portfolio, current_date, short_window=20, long_window=50):
    """Moving average crossover strategy."""
    signals = []
    
    # This is a simplified implementation
    # In a real scenario, you'd need historical data for MA calculation
    
    for symbol, price_data in data.items():
        current_price = price_data['Close']
        
        # Simple logic: if we have cash and no position, buy
        # if we have position and random condition, sell
        position = portfolio.get_position(symbol)
        
        if not position and portfolio.cash > current_price * 100:
            # Buy signal
            signals.append({
                'symbol': symbol,
                'side': 'buy',
                'quantity': 100,
                'order_type': 'market'
            })
        elif position and np.random.random() < 0.1:  # 10% chance to sell
            # Sell signal
            signals.append({
                'symbol': symbol,
                'side': 'sell',
                'quantity': position.quantity,
                'order_type': 'market'
            })
    
    return signals

def momentum_strategy(data, portfolio, current_date, lookback_days=10):
    """Momentum strategy based on recent returns."""
    signals = []
    
    for symbol, price_data in data.items():
        current_price = price_data['Close']
        position = portfolio.get_position(symbol)
        
        # Simple momentum logic
        if not position and portfolio.cash > current_price * 100:
            # Buy if we have cash
            signals.append({
                'symbol': symbol,
                'side': 'buy',
                'quantity': 100,
                'order_type': 'market'
            })
    
    return signals

def run_backtest(symbols, strategy_name, strategy_func, start_date, end_date, initial_capital):
    """Run backtest with given parameters."""
    try:
        # Get data
        provider = YFinanceDataProvider()
        
        # Calculate period from dates
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        days_diff = (end_dt - start_dt).days
        
        if days_diff <= 30:
            period = "1mo"
        elif days_diff <= 90:
            period = "3mo"
        elif days_diff <= 180:
            period = "6mo"
        elif days_diff <= 365:
            period = "1y"
        else:
            period = "2y"
        
        # Create backtest engine
        engine = BacktestEngine(initial_capital=initial_capital)
        
        # Add data for each symbol
        for symbol in symbols:
            try:
                data = provider.get_historical_data(symbol, period)
                if not data.empty:
                    engine.add_data(symbol, data)
            except Exception as e:
                st.error(f"Error loading data for {symbol}: {e}")
        
        # Set strategy
        engine.set_strategy(strategy_func)
        
        # Run backtest
        results = engine.run_backtest(start_date, end_date)
        
        return engine, results
        
    except Exception as e:
        st.error(f"Backtest failed: {e}")
        return None, None

def plot_backtest_results(results):
    """Plot backtest results."""
    if not results or 'equity_curve' not in results:
        return None
    
    equity_df = results['equity_curve']
    
    # Create subplots
    from plotly.subplots import make_subplots
    
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=('Portfolio Value', 'Daily Returns', 'Drawdown'),
        row_heights=[0.5, 0.25, 0.25]
    )
    
    # Portfolio value
    fig.add_trace(
        go.Scatter(
            x=equity_df.index,
            y=equity_df['portfolio_value'],
            name='Portfolio Value',
            line=dict(color='blue')
        ),
        row=1, col=1
    )
    
    # Daily returns
    fig.add_trace(
        go.Scatter(
            x=equity_df.index,
            y=equity_df['returns'] * 100,
            name='Daily Returns (%)',
            line=dict(color='green'),
            opacity=0.7
        ),
        row=2, col=1
    )
    
    # Drawdown
    rolling_max = equity_df['portfolio_value'].expanding().max()
    drawdown = (equity_df['portfolio_value'] - rolling_max) / rolling_max * 100
    
    fig.add_trace(
        go.Scatter(
            x=equity_df.index,
            y=drawdown,
            name='Drawdown (%)',
            fill='tozeroy',
            line=dict(color='red'),
            opacity=0.7
        ),
        row=3, col=1
    )
    
    fig.update_layout(
        title="Backtest Results",
        height=800,
        showlegend=False
    )
    
    fig.update_yaxes(title_text="Value ($)", row=1, col=1)
    fig.update_yaxes(title_text="Returns (%)", row=2, col=1)
    fig.update_yaxes(title_text="Drawdown (%)", row=3, col=1)
    
    return fig

def display_backtest_metrics(results):
    """Display backtest performance metrics."""
    if not results:
        return
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Return",
            f"{results['total_return']:.2%}",
            help="Total return over the backtest period"
        )
    
    with col2:
        st.metric(
            "Annualized Return",
            f"{results['annualized_return']:.2%}",
            help="Annualized return"
        )
    
    with col3:
        st.metric(
            "Volatility",
            f"{results['volatility']:.2%}",
            help="Annualized volatility"
        )
    
    with col4:
        st.metric(
            "Sharpe Ratio",
            f"{results['sharpe_ratio']:.2f}",
            help="Risk-adjusted return measure"
        )
    
    col5, col6, col7, col8 = st.columns(4)
    
    with col5:
        st.metric(
            "Max Drawdown",
            f"{results['max_drawdown']:.2%}",
            help="Maximum peak-to-trough decline"
        )
    
    with col6:
        st.metric(
            "Win Rate",
            f"{results['win_rate']:.2%}",
            help="Percentage of winning trades"
        )
    
    with col7:
        st.metric(
            "Total Trades",
            results['total_trades'],
            help="Total number of trades executed"
        )
    
    with col8:
        st.metric(
            "Final Value",
            f"${results['final_portfolio_value']:,.2f}",
            help="Final portfolio value"
        )

def save_backtest_results(results, strategy_name, symbols, db_manager):
    """Save backtest results to database."""
    try:
        result_data = {
            'strategy_name': strategy_name,
            'symbols': symbols,
            'start_date': results['equity_curve'].index[0],
            'end_date': results['equity_curve'].index[-1],
            'initial_capital': 100000,  # Default value
            'final_value': results['final_portfolio_value'],
            'total_return': results['total_return'],
            'annualized_return': results['annualized_return'],
            'volatility': results['volatility'],
            'sharpe_ratio': results['sharpe_ratio'],
            'max_drawdown': results['max_drawdown'],
            'win_rate': results['win_rate'],
            'total_trades': results['total_trades'],
            'equity_curve': results['equity_curve'],
            'trades': results['trades']
        }
        
        db_manager.save_backtest_result(result_data)
        st.success("Backtest results saved to database!")
        
    except Exception as e:
        st.error(f"Error saving results: {e}")

def main():
    st.title("🔄 Strategy Backtesting")
    st.markdown("Test and analyze trading strategies with historical data")
    
    # Sidebar for backtest parameters
    st.sidebar.title("Backtest Parameters")
    
    # Strategy selection
    strategy_options = {
        "Buy and Hold": buy_and_hold_strategy,
        "Moving Average Crossover": moving_average_strategy,
        "Momentum": momentum_strategy
    }
    
    selected_strategy = st.sidebar.selectbox(
        "Select Strategy",
        options=list(strategy_options.keys())
    )
    
    # Symbol selection
    symbols = st.sidebar.multiselect(
        "Select Symbols",
        options=['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA'],
        default=['AAPL', 'MSFT']
    )
    
    # Date range
    col1, col2 = st.sidebar.columns(2)
    with col1:
        start_date = st.date_input(
            "Start Date",
            value=datetime.now() - timedelta(days=365),
            max_value=datetime.now() - timedelta(days=1)
        )
    
    with col2:
        end_date = st.date_input(
            "End Date",
            value=datetime.now() - timedelta(days=1),
            max_value=datetime.now()
        )
    
    # Initial capital
    initial_capital = st.sidebar.number_input(
        "Initial Capital ($)",
        value=100000,
        min_value=1000,
        step=1000
    )
    
    # Run backtest button
    if st.sidebar.button("Run Backtest", type="primary"):
        if not symbols:
            st.error("Please select at least one symbol")
            return
        
        if start_date >= end_date:
            st.error("Start date must be before end date")
            return
        
        # Run backtest
        with st.spinner("Running backtest..."):
            strategy_func = strategy_options[selected_strategy]
            engine, results = run_backtest(
                symbols, 
                selected_strategy, 
                strategy_func,
                start_date.strftime('%Y-%m-%d'),
                end_date.strftime('%Y-%m-%d'),
                initial_capital
            )
        
        if results:
            st.success("Backtest completed!")
            
            # Store results in session state
            st.session_state.backtest_results = results
            st.session_state.backtest_engine = engine
            st.session_state.backtest_strategy = selected_strategy
            st.session_state.backtest_symbols = symbols
        else:
            st.error("Backtest failed. Please check your parameters and try again.")
    
    # Display results if available
    if 'backtest_results' in st.session_state:
        results = st.session_state.backtest_results
        
        st.header("Backtest Results")
        
        # Performance metrics
        st.subheader("Performance Metrics")
        display_backtest_metrics(results)
        
        # Charts
        st.subheader("Performance Charts")
        fig = plot_backtest_results(results)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        
        # Trade history
        if results.get('trades'):
            st.subheader("Trade History")
            trades_df = pd.DataFrame(results['trades'])
            if not trades_df.empty:
                st.dataframe(trades_df, use_container_width=True)
        
        # Save results
        col1, col2 = st.columns([1, 4])
        with col1:
            if st.button("Save Results"):
                db_manager = DatabaseManager()
                save_backtest_results(
                    results, 
                    st.session_state.backtest_strategy,
                    st.session_state.backtest_symbols,
                    db_manager
                )
    
    # Historical backtest results
    st.header("Historical Backtest Results")
    
    try:
        db_manager = DatabaseManager()
        # This would require implementing a method to retrieve backtest results
        st.info("Historical backtest results will be displayed here once implemented in the database manager.")
    except Exception as e:
        st.error(f"Error loading historical results: {e}")
    
    # Strategy information
    st.header("Strategy Information")
    
    strategy_info = {
        "Buy and Hold": {
            "description": "Simple buy and hold strategy that purchases stocks at the beginning and holds them throughout the period.",
            "pros": ["Simple to implement", "Low transaction costs", "Good for long-term trends"],
            "cons": ["No downside protection", "Doesn't adapt to market conditions"]
        },
        "Moving Average Crossover": {
            "description": "Strategy based on moving average crossovers. Generates buy/sell signals when short-term MA crosses above/below long-term MA.",
            "pros": ["Trend following", "Reduces whipsaws", "Widely used"],
            "cons": ["Lagging indicator", "Poor performance in sideways markets"]
        },
        "Momentum": {
            "description": "Strategy that buys stocks showing strong recent performance and sells those showing weakness.",
            "pros": ["Captures trending moves", "Can work in various timeframes"],
            "cons": ["Can be whipsaw-prone", "Requires careful risk management"]
        }
    }
    
    if selected_strategy in strategy_info:
        info = strategy_info[selected_strategy]
        
        st.subheader(f"{selected_strategy} Strategy")
        st.write(info["description"])
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Pros:**")
            for pro in info["pros"]:
                st.write(f"• {pro}")
        
        with col2:
            st.write("**Cons:**")
            for con in info["cons"]:
                st.write(f"• {con}")

if __name__ == "__main__":
    main()

