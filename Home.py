"""
AltSignals - Alternative Signals Trading Platform
Main Streamlit application home page.
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
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

from yfinance_util import YFinanceDataProvider
from polygon_util import PolygonDataProvider
from database import DatabaseManager
from sentiment_util import SentimentAnalyzer
from options_pricing import BlackScholesCalculator

# Page configuration
st.set_page_config(
    page_title="AltSignals - Alternative Signals Platform",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .signal-positive {
        color: #00ff00;
        font-weight: bold;
    }
    .signal-negative {
        color: #ff0000;
        font-weight: bold;
    }
    .signal-neutral {
        color: #ffa500;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables."""
    if 'data_provider' not in st.session_state:
        st.session_state.data_provider = 'yfinance'
    if 'selected_symbols' not in st.session_state:
        st.session_state.selected_symbols = ['AAPL', 'MSFT', 'GOOGL']
    if 'db_manager' not in st.session_state:
        st.session_state.db_manager = DatabaseManager()

def create_sidebar():
    """Create sidebar with controls."""
    st.sidebar.title("🔧 Settings")
    
    # Data source selection
    st.sidebar.subheader("Data Source")
    data_provider = st.sidebar.selectbox(
        "Select Market Data Provider",
        options=['yfinance', 'polygon'],
        index=0 if st.session_state.data_provider == 'yfinance' else 1,
        help="Choose your preferred market data source"
    )
    st.session_state.data_provider = data_provider
    
    # Symbol selection
    st.sidebar.subheader("Symbols")
    
    # Popular symbols
    popular_symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX']
    
    selected_symbols = st.sidebar.multiselect(
        "Select Symbols to Analyze",
        options=popular_symbols,
        default=st.session_state.selected_symbols,
        help="Choose stocks to analyze"
    )
    
    # Custom symbol input
    custom_symbol = st.sidebar.text_input(
        "Add Custom Symbol",
        placeholder="e.g., AAPL",
        help="Enter a stock symbol to add to analysis"
    )
    
    if custom_symbol and st.sidebar.button("Add Symbol"):
        if custom_symbol.upper() not in selected_symbols:
            selected_symbols.append(custom_symbol.upper())
    
    st.session_state.selected_symbols = selected_symbols
    
    # Analysis settings
    st.sidebar.subheader("Analysis Settings")
    
    time_period = st.sidebar.selectbox(
        "Time Period",
        options=['1mo', '3mo', '6mo', '1y', '2y'],
        index=2,
        help="Historical data period"
    )
    
    return {
        'data_provider': data_provider,
        'symbols': selected_symbols,
        'time_period': time_period
    }

def get_market_data(symbols, provider, period):
    """Get market data based on selected provider."""
    data = {}
    
    if provider == 'yfinance':
        yf_provider = YFinanceDataProvider()
        for symbol in symbols:
            try:
                df = yf_provider.get_historical_data(symbol, period)
                if not df.empty:
                    data[symbol] = df
            except Exception as e:
                st.error(f"Error fetching data for {symbol}: {e}")
    
    elif provider == 'polygon':
        try:
            polygon_provider = PolygonDataProvider()
            end_date = datetime.now().strftime('%Y-%m-%d')
            
            # Convert period to days
            period_days = {
                '1mo': 30, '3mo': 90, '6mo': 180, 
                '1y': 365, '2y': 730
            }
            days = period_days.get(period, 365)
            start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
            
            for symbol in symbols:
                try:
                    df = polygon_provider.get_historical_data(symbol, start_date, end_date)
                    if not df.empty:
                        data[symbol] = df
                except Exception as e:
                    st.error(f"Error fetching Polygon data for {symbol}: {e}")
        except Exception as e:
            st.error(f"Polygon API error: {e}")
    
    return data

def create_price_chart(data, symbols):
    """Create interactive price chart."""
    if not data:
        st.warning("No data available for selected symbols")
        return
    
    fig = go.Figure()
    
    for symbol in symbols:
        if symbol in data:
            df = data[symbol]
            fig.add_trace(go.Scatter(
                x=df.index,
                y=df['Close'],
                mode='lines',
                name=symbol,
                hovertemplate=f'<b>{symbol}</b><br>' +
                             'Date: %{x}<br>' +
                             'Price: $%{y:.2f}<extra></extra>'
            ))
    
    fig.update_layout(
        title="Stock Price Comparison",
        xaxis_title="Date",
        yaxis_title="Price ($)",
        hovermode='x unified',
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)

def create_volume_chart(data, symbols):
    """Create volume chart."""
    if not data:
        return
    
    fig = go.Figure()
    
    for symbol in symbols:
        if symbol in data:
            df = data[symbol]
            fig.add_trace(go.Bar(
                x=df.index,
                y=df['Volume'],
                name=symbol,
                opacity=0.7
            ))
    
    fig.update_layout(
        title="Trading Volume",
        xaxis_title="Date",
        yaxis_title="Volume",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)

def create_returns_heatmap(data, symbols):
    """Create returns correlation heatmap."""
    if len(symbols) < 2 or not data:
        return
    
    returns_data = {}
    for symbol in symbols:
        if symbol in data:
            df = data[symbol]
            returns_data[symbol] = df['Close'].pct_change().dropna()
    
    if len(returns_data) < 2:
        return
    
    returns_df = pd.DataFrame(returns_data)
    correlation_matrix = returns_df.corr()
    
    fig = px.imshow(
        correlation_matrix,
        text_auto=True,
        aspect="auto",
        title="Returns Correlation Matrix",
        color_continuous_scale="RdBu_r"
    )
    
    st.plotly_chart(fig, use_container_width=True)

def display_stock_metrics(data, symbols):
    """Display key stock metrics."""
    if not data:
        return
    
    st.subheader("📊 Key Metrics")
    
    cols = st.columns(len(symbols))
    
    for i, symbol in enumerate(symbols):
        if symbol in data:
            df = data[symbol]
            current_price = df['Close'].iloc[-1]
            prev_price = df['Close'].iloc[-2] if len(df) > 1 else current_price
            change = current_price - prev_price
            change_pct = (change / prev_price) * 100
            
            with cols[i]:
                st.metric(
                    label=symbol,
                    value=f"${current_price:.2f}",
                    delta=f"{change_pct:.2f}%"
                )

def display_sentiment_analysis(symbols):
    """Display sentiment analysis results."""
    st.subheader("📰 Sentiment Analysis")
    
    try:
        sentiment_analyzer = SentimentAnalyzer()
        
        for symbol in symbols[:3]:  # Limit to 3 symbols to avoid API costs
            with st.expander(f"Sentiment for {symbol}"):
                with st.spinner(f"Analyzing sentiment for {symbol}..."):
                    try:
                        sentiment_summary = sentiment_analyzer.get_sentiment_summary(symbol, days_back=7)
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            sentiment_score = sentiment_summary.get('avg_sentiment', 0)
                            st.metric(
                                "Sentiment Score",
                                f"{sentiment_score:.3f}",
                                help="Range: -1 (very negative) to +1 (very positive)"
                            )
                        
                        with col2:
                            sentiment_trend = sentiment_summary.get('sentiment_trend', 'neutral')
                            color_class = f"signal-{sentiment_trend}"
                            st.markdown(f"**Trend:** <span class='{color_class}'>{sentiment_trend.upper()}</span>", 
                                      unsafe_allow_html=True)
                        
                        with col3:
                            article_count = sentiment_summary.get('analyzed_count', 0)
                            st.metric("Articles Analyzed", article_count)
                        
                        # Key themes
                        key_themes = sentiment_summary.get('key_themes', [])
                        if key_themes:
                            st.write("**Key Themes:**")
                            st.write(", ".join(key_themes[:5]))
                        
                        # Recent articles
                        recent_articles = sentiment_summary.get('recent_articles', [])
                        if recent_articles:
                            st.write("**Recent Articles:**")
                            for article in recent_articles[:3]:
                                sentiment_label = article.get('sentiment_label', 'neutral')
                                color_class = f"signal-{sentiment_label}"
                                st.markdown(f"• <span class='{color_class}'>[{sentiment_label.upper()}]</span> {article.get('title', 'No title')}", 
                                          unsafe_allow_html=True)
                    
                    except Exception as e:
                        st.error(f"Error analyzing sentiment for {symbol}: {e}")
    
    except Exception as e:
        st.error(f"Sentiment analysis unavailable: {e}")

def display_options_analysis(symbols):
    """Display options analysis."""
    st.subheader("⚙️ Options Analysis")
    
    calculator = BlackScholesCalculator()
    
    # Options parameters
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        strike_price = st.number_input("Strike Price", value=100.0, min_value=1.0)
    with col2:
        time_to_expiry = st.number_input("Days to Expiry", value=30, min_value=1) / 365
    with col3:
        volatility = st.slider("Volatility", 0.1, 1.0, 0.2, 0.01)
    with col4:
        risk_free_rate = st.slider("Risk-free Rate", 0.0, 0.1, 0.05, 0.001)
    
    # Calculate options prices for selected symbols
    for symbol in symbols[:2]:  # Limit to 2 symbols
        if symbol in st.session_state.get('market_data', {}):
            df = st.session_state.market_data[symbol]
            current_price = df['Close'].iloc[-1]
            
            # Calculate call and put options
            call_greeks = calculator.calculate_all_greeks(
                current_price, strike_price, time_to_expiry, risk_free_rate, volatility, 'call'
            )
            put_greeks = calculator.calculate_all_greeks(
                current_price, strike_price, time_to_expiry, risk_free_rate, volatility, 'put'
            )
            
            st.write(f"**{symbol} Options (Current Price: ${current_price:.2f})**")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Call Option**")
                st.write(f"Price: ${call_greeks['price']:.2f}")
                st.write(f"Delta: {call_greeks['delta']:.4f}")
                st.write(f"Gamma: {call_greeks['gamma']:.4f}")
                st.write(f"Theta: ${call_greeks['theta']:.2f}")
                st.write(f"Vega: ${call_greeks['vega']:.2f}")
            
            with col2:
                st.write("**Put Option**")
                st.write(f"Price: ${put_greeks['price']:.2f}")
                st.write(f"Delta: {put_greeks['delta']:.4f}")
                st.write(f"Gamma: {put_greeks['gamma']:.4f}")
                st.write(f"Theta: ${put_greeks['theta']:.2f}")
                st.write(f"Vega: ${put_greeks['vega']:.2f}")

def main():
    """Main application function."""
    # Initialize session state
    initialize_session_state()
    
    # Header
    st.markdown('<h1 class="main-header">📈 AltSignals Platform</h1>', unsafe_allow_html=True)
    st.markdown("**Alternative Signals Trading Platform** - Advanced financial analysis with real-time data, sentiment analysis, and options pricing")
    
    # Sidebar
    settings = create_sidebar()
    
    if not settings['symbols']:
        st.warning("Please select at least one symbol to analyze.")
        return
    
    # Get market data
    with st.spinner("Loading market data..."):
        market_data = get_market_data(
            settings['symbols'], 
            settings['data_provider'], 
            settings['time_period']
        )
        st.session_state.market_data = market_data
    
    if not market_data:
        st.error("No market data available. Please check your symbols and data provider settings.")
        return
    
    # Display metrics
    display_stock_metrics(market_data, settings['symbols'])
    
    # Charts
    col1, col2 = st.columns([2, 1])
    
    with col1:
        create_price_chart(market_data, settings['symbols'])
    
    with col2:
        create_volume_chart(market_data, settings['symbols'])
    
    # Correlation heatmap
    if len(settings['symbols']) > 1:
        create_returns_heatmap(market_data, settings['symbols'])
    
    # Sentiment Analysis
    display_sentiment_analysis(settings['symbols'])
    
    # Options Analysis
    display_options_analysis(settings['symbols'])
    
    # Footer
    st.markdown("---")
    st.markdown("**AltSignals Platform** | Data sources: Yahoo Finance, Polygon.io | Sentiment: OpenAI GPT-4")
    st.markdown("⚠️ *Some data may be marked as SYNTHETIC for testing purposes*")

if __name__ == "__main__":
    main()

