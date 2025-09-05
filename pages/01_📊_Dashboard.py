"""
Dashboard page with advanced visualizations and analytics.
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add utils to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))

from yfinance_util import YFinanceDataProvider
from database import DatabaseManager

st.set_page_config(page_title="Dashboard", page_icon="📊", layout="wide")

def create_candlestick_chart(data, symbol):
    """Create candlestick chart with volume."""
    if symbol not in data or data[symbol].empty:
        st.warning(f"No data available for {symbol}")
        return
    
    df = data[symbol]
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=(f'{symbol} Price', 'Volume'),
        row_width=[0.2, 0.7]
    )
    
    # Candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name=symbol
        ),
        row=1, col=1
    )
    
    # Volume chart
    colors = ['red' if close < open else 'green' 
              for close, open in zip(df['Close'], df['Open'])]
    
    fig.add_trace(
        go.Bar(
            x=df.index,
            y=df['Volume'],
            name='Volume',
            marker_color=colors,
            opacity=0.7
        ),
        row=2, col=1
    )
    
    fig.update_layout(
        title=f"{symbol} - Candlestick Chart with Volume",
        yaxis_title="Price ($)",
        yaxis2_title="Volume",
        xaxis_rangeslider_visible=False,
        height=600
    )
    
    return fig

def create_technical_indicators(data, symbol):
    """Create technical indicators chart."""
    if symbol not in data or data[symbol].empty:
        return None
    
    df = data[symbol].copy()
    
    # Calculate moving averages
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA50'] = df['Close'].rolling(window=50).mean()
    
    # Calculate Bollinger Bands
    df['BB_middle'] = df['Close'].rolling(window=20).mean()
    df['BB_std'] = df['Close'].rolling(window=20).std()
    df['BB_upper'] = df['BB_middle'] + (df['BB_std'] * 2)
    df['BB_lower'] = df['BB_middle'] - (df['BB_std'] * 2)
    
    # Calculate RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=(f'{symbol} with Technical Indicators', 'RSI'),
        row_heights=[0.7, 0.3]
    )
    
    # Price and Bollinger Bands
    fig.add_trace(
        go.Scatter(x=df.index, y=df['Close'], name='Close', line=dict(color='blue')),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=df.index, y=df['MA20'], name='MA20', line=dict(color='orange')),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=df.index, y=df['MA50'], name='MA50', line=dict(color='red')),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=df.index, y=df['BB_upper'], name='BB Upper', 
                  line=dict(color='gray', dash='dash'), opacity=0.5),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=df.index, y=df['BB_lower'], name='BB Lower', 
                  line=dict(color='gray', dash='dash'), opacity=0.5),
        row=1, col=1
    )
    
    # RSI
    fig.add_trace(
        go.Scatter(x=df.index, y=df['RSI'], name='RSI', line=dict(color='purple')),
        row=2, col=1
    )
    
    # RSI reference lines
    fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.5, row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.5, row=2, col=1)
    
    fig.update_layout(height=700, title=f"{symbol} Technical Analysis")
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="RSI", range=[0, 100], row=2, col=1)
    
    return fig

def create_performance_metrics(data, symbols):
    """Create performance metrics comparison."""
    if not data:
        return None
    
    metrics_data = []
    
    for symbol in symbols:
        if symbol in data and not data[symbol].empty:
            df = data[symbol]
            
            # Calculate metrics
            current_price = df['Close'].iloc[-1]
            start_price = df['Close'].iloc[0]
            total_return = (current_price - start_price) / start_price * 100
            
            # Volatility (annualized)
            returns = df['Close'].pct_change().dropna()
            volatility = returns.std() * np.sqrt(252) * 100
            
            # Sharpe ratio (simplified, assuming 0% risk-free rate)
            avg_return = returns.mean() * 252
            sharpe_ratio = avg_return / (volatility / 100) if volatility > 0 else 0
            
            # Max drawdown
            rolling_max = df['Close'].expanding().max()
            drawdown = (df['Close'] - rolling_max) / rolling_max * 100
            max_drawdown = drawdown.min()
            
            metrics_data.append({
                'Symbol': symbol,
                'Total Return (%)': total_return,
                'Volatility (%)': volatility,
                'Sharpe Ratio': sharpe_ratio,
                'Max Drawdown (%)': max_drawdown,
                'Current Price': current_price
            })
    
    if not metrics_data:
        return None
    
    metrics_df = pd.DataFrame(metrics_data)
    
    # Create performance comparison chart
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=metrics_df['Symbol'],
        y=metrics_df['Total Return (%)'],
        name='Total Return (%)',
        marker_color='lightblue'
    ))
    
    fig.update_layout(
        title="Performance Comparison",
        xaxis_title="Symbol",
        yaxis_title="Total Return (%)",
        height=400
    )
    
    return fig, metrics_df

def create_risk_return_scatter(data, symbols):
    """Create risk-return scatter plot."""
    if not data or len(symbols) < 2:
        return None
    
    risk_return_data = []
    
    for symbol in symbols:
        if symbol in data and not data[symbol].empty:
            df = data[symbol]
            returns = df['Close'].pct_change().dropna()
            
            # Annualized return and volatility
            avg_return = returns.mean() * 252 * 100
            volatility = returns.std() * np.sqrt(252) * 100
            
            risk_return_data.append({
                'Symbol': symbol,
                'Return (%)': avg_return,
                'Risk (%)': volatility
            })
    
    if len(risk_return_data) < 2:
        return None
    
    risk_return_df = pd.DataFrame(risk_return_data)
    
    fig = px.scatter(
        risk_return_df,
        x='Risk (%)',
        y='Return (%)',
        text='Symbol',
        title='Risk-Return Analysis',
        labels={'Risk (%)': 'Volatility (%)', 'Return (%)': 'Annualized Return (%)'}
    )
    
    fig.update_traces(textposition="top center", marker_size=12)
    fig.update_layout(height=500)
    
    return fig

def main():
    st.title("📊 Advanced Dashboard")
    st.markdown("Comprehensive financial analysis with technical indicators and performance metrics")
    
    # Get symbols from session state or use defaults
    if 'selected_symbols' in st.session_state:
        symbols = st.session_state.selected_symbols
    else:
        symbols = ['AAPL', 'MSFT', 'GOOGL']
    
    if not symbols:
        st.warning("No symbols selected. Please go to the Home page and select symbols.")
        return
    
    # Load data
    @st.cache_data(ttl=300)  # Cache for 5 minutes
    def load_market_data(symbols_list, period='6mo'):
        provider = YFinanceDataProvider()
        data = {}
        for symbol in symbols_list:
            try:
                df = provider.get_historical_data(symbol, period)
                if not df.empty:
                    data[symbol] = df
            except Exception as e:
                st.error(f"Error loading data for {symbol}: {e}")
        return data
    
    with st.spinner("Loading market data..."):
        market_data = load_market_data(symbols)
    
    if not market_data:
        st.error("No market data available.")
        return
    
    # Symbol selection for detailed analysis
    selected_symbol = st.selectbox(
        "Select symbol for detailed analysis:",
        options=list(market_data.keys()),
        index=0
    )
    
    # Tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Charts", "🔧 Technical", "📊 Performance", "⚖️ Risk Analysis"])
    
    with tab1:
        st.subheader(f"Candlestick Chart - {selected_symbol}")
        candlestick_fig = create_candlestick_chart(market_data, selected_symbol)
        if candlestick_fig:
            st.plotly_chart(candlestick_fig, use_container_width=True)
    
    with tab2:
        st.subheader(f"Technical Indicators - {selected_symbol}")
        technical_fig = create_technical_indicators(market_data, selected_symbol)
        if technical_fig:
            st.plotly_chart(technical_fig, use_container_width=True)
        
        # Technical analysis summary
        if selected_symbol in market_data:
            df = market_data[selected_symbol]
            if len(df) >= 50:
                current_price = df['Close'].iloc[-1]
                ma20 = df['Close'].rolling(20).mean().iloc[-1]
                ma50 = df['Close'].rolling(50).mean().iloc[-1]
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    trend = "Bullish" if current_price > ma20 > ma50 else "Bearish" if current_price < ma20 < ma50 else "Neutral"
                    st.metric("Trend", trend)
                
                with col2:
                    st.metric("Price vs MA20", f"{((current_price - ma20) / ma20 * 100):.2f}%")
                
                with col3:
                    st.metric("Price vs MA50", f"{((current_price - ma50) / ma50 * 100):.2f}%")
    
    with tab3:
        st.subheader("Performance Comparison")
        performance_result = create_performance_metrics(market_data, symbols)
        
        if performance_result:
            performance_fig, metrics_df = performance_result
            st.plotly_chart(performance_fig, use_container_width=True)
            
            st.subheader("Performance Metrics Table")
            st.dataframe(metrics_df, use_container_width=True)
    
    with tab4:
        st.subheader("Risk-Return Analysis")
        risk_return_fig = create_risk_return_scatter(market_data, symbols)
        
        if risk_return_fig:
            st.plotly_chart(risk_return_fig, use_container_width=True)
            
            st.markdown("""
            **Risk-Return Analysis Interpretation:**
            - **Top-left quadrant**: High return, low risk (ideal)
            - **Top-right quadrant**: High return, high risk
            - **Bottom-left quadrant**: Low return, low risk
            - **Bottom-right quadrant**: Low return, high risk (avoid)
            """)

if __name__ == "__main__":
    main()

