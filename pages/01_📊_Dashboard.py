"""
Dashboard Page for AltSignals platform.
Displays market data, charts, and fundamental analysis.
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add utils to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))

from yfinance_util import YFinanceDataProvider
from polygon_util import PolygonDataProvider

# Page configuration
st.set_page_config(
    page_title="Dashboard",
    page_icon="📊",
    layout="wide"
)

def main():
    st.title("📊 Advanced Analytics Dashboard")
    st.markdown("Comprehensive financial analysis and market insights")
    
    # Sidebar for symbol selection and settings
    with st.sidebar:
        st.header("🔧 Dashboard Settings")
        
        # Symbol selection
        symbol = st.selectbox(
            "Select Symbol for Analysis",
            options=["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "NFLX"],
            index=0,
            help="Choose a stock symbol for detailed analysis"
        )
        
        # Custom symbol input
        custom_symbol = st.text_input("Or enter custom symbol:", placeholder="e.g., AAPL")
        if custom_symbol:
            symbol = custom_symbol.upper()
        
        # Data provider selection
        data_provider = st.selectbox(
            "Data Provider",
            options=["Yahoo Finance", "Polygon"],
            help="Choose your preferred data source"
        )
        
        # Time period selection
        period_options = {
            "1 Month": "1mo",
            "3 Months": "3mo", 
            "6 Months": "6mo",
            "1 Year": "1y",
            "2 Years": "2y"
        }
        
        period_label = st.selectbox(
            "Time Period",
            list(period_options.keys()),
            index=3,
            help="Select time period for analysis"
        )
        period = period_options[period_label]
        
        # Refresh data button
        refresh_data = st.button("🔄 Refresh Data", type="primary")
    
    # Initialize data provider
    if data_provider == "Yahoo Finance":
        provider = YFinanceDataProvider()
    else:
        provider = PolygonDataProvider()
    
    # Main dashboard content
    try:
        # Get stock data
        with st.spinner(f"Loading data for {symbol}..."):
            stock_data = provider.get_stock_data(symbol, period=period)
            company_info = provider.get_company_info(symbol)
        
        if stock_data.empty:
            st.error(f"No data available for {symbol}")
            return
        
        # Display company information
        display_company_header(symbol, company_info, stock_data)
        
        # Create tabs for different analyses
        tab1, tab2, tab3, tab4 = st.tabs(["📈 Charts", "📊 Performance", "💰 Fundamentals", "🎯 Risk Analysis"])
        
        with tab1:
            display_charts_analysis(symbol, stock_data, provider)
        
        with tab2:
            display_performance_analysis(symbol, stock_data, company_info)
        
        with tab3:
            display_fundamental_analysis(symbol, company_info, provider)
        
        with tab4:
            display_risk_analysis(symbol, stock_data)
            
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")

def display_company_header(symbol, company_info, stock_data):
    """Display company information header."""
    current_price = stock_data['Close'].iloc[-1]
    prev_close = stock_data['Close'].iloc[-2] if len(stock_data) > 1 else current_price
    price_change = current_price - prev_close
    price_change_pct = (price_change / prev_close) * 100
    
    # Company name and basic info
    company_name = company_info.get('longName', symbol) if company_info else symbol
    sector = company_info.get('sector', 'N/A') if company_info else 'N/A'
    
    col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
    
    with col1:
        st.subheader(f"{company_name} ({symbol})")
        st.write(f"**Sector:** {sector}")
    
    with col2:
        st.metric(
            label="Current Price",
            value=f"${current_price:.2f}",
            delta=f"{price_change:+.2f} ({price_change_pct:+.2f}%)"
        )
    
    with col3:
        volume = stock_data['Volume'].iloc[-1]
        avg_volume = stock_data['Volume'].tail(20).mean()
        volume_ratio = volume / avg_volume if avg_volume > 0 else 1
        
        st.metric(
            label="Volume",
            value=f"{volume:,.0f}",
            delta=f"{volume_ratio:.1f}x avg" if volume_ratio != 1 else None
        )
    
    with col4:
        market_cap = company_info.get('marketCap', 0) if company_info else 0
        if market_cap > 0:
            if market_cap >= 1e12:
                market_cap_str = f"${market_cap/1e12:.2f}T"
            elif market_cap >= 1e9:
                market_cap_str = f"${market_cap/1e9:.2f}B"
            else:
                market_cap_str = f"${market_cap/1e6:.2f}M"
        else:
            market_cap_str = "N/A"
        
        st.metric(
            label="Market Cap",
            value=market_cap_str
        )

def display_charts_analysis(symbol, stock_data, provider):
    """Display charts and price analysis."""
    st.subheader("📈 Price Charts and Analysis")
    
    # Candlestick chart
    fig = go.Figure()
    
    fig.add_trace(go.Candlestick(
        x=stock_data.index,
        open=stock_data['Open'],
        high=stock_data['High'],
        low=stock_data['Low'],
        close=stock_data['Close'],
        name=symbol
    ))
    
    # Add volume subplot
    fig_volume = go.Figure()
    fig_volume.add_trace(go.Bar(
        x=stock_data.index,
        y=stock_data['Volume'],
        name='Volume',
        marker_color='rgba(0,100,80,0.7)'
    ))
    
    fig.update_layout(
        title=f"{symbol} Price Chart",
        yaxis_title="Price ($)",
        xaxis_title="Date",
        height=500,
        xaxis_rangeslider_visible=False
    )
    
    fig_volume.update_layout(
        title=f"{symbol} Volume",
        yaxis_title="Volume",
        xaxis_title="Date",
        height=300
    )
    
    st.plotly_chart(fig, use_container_width=True)
    st.plotly_chart(fig_volume, use_container_width=True)
    
    # Price statistics
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Price Statistics")
        
        high_52w = stock_data['High'].max()
        low_52w = stock_data['Low'].min()
        current_price = stock_data['Close'].iloc[-1]
        
        st.write(f"**52-Week High:** ${high_52w:.2f}")
        st.write(f"**52-Week Low:** ${low_52w:.2f}")
        st.write(f"**Current Price:** ${current_price:.2f}")
        
        # Price position
        price_position = (current_price - low_52w) / (high_52w - low_52w) * 100
        st.write(f"**52-Week Position:** {price_position:.1f}%")
        
        # Average prices
        avg_20 = stock_data['Close'].tail(20).mean()
        avg_50 = stock_data['Close'].tail(50).mean()
        
        st.write(f"**20-Day Average:** ${avg_20:.2f}")
        st.write(f"**50-Day Average:** ${avg_50:.2f}")
    
    with col2:
        st.subheader("📈 Returns Analysis")
        
        # Calculate returns
        returns_1d = (stock_data['Close'].iloc[-1] / stock_data['Close'].iloc[-2] - 1) * 100
        returns_1w = (stock_data['Close'].iloc[-1] / stock_data['Close'].iloc[-5] - 1) * 100 if len(stock_data) >= 5 else 0
        returns_1m = (stock_data['Close'].iloc[-1] / stock_data['Close'].iloc[-21] - 1) * 100 if len(stock_data) >= 21 else 0
        returns_3m = (stock_data['Close'].iloc[-1] / stock_data['Close'].iloc[-63] - 1) * 100 if len(stock_data) >= 63 else 0
        
        st.write(f"**1-Day Return:** {returns_1d:+.2f}%")
        st.write(f"**1-Week Return:** {returns_1w:+.2f}%")
        st.write(f"**1-Month Return:** {returns_1m:+.2f}%")
        st.write(f"**3-Month Return:** {returns_3m:+.2f}%")
        
        # Volatility
        volatility = stock_data['Close'].pct_change().std() * np.sqrt(252) * 100
        st.write(f"**Annualized Volatility:** {volatility:.1f}%")

def display_performance_analysis(symbol, stock_data, company_info):
    """Display performance analysis."""
    st.subheader("📊 Performance Analysis")
    
    # Calculate daily returns
    stock_data['Returns'] = stock_data['Close'].pct_change()
    
    # Performance metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("📈 Return Metrics")
        
        # Cumulative returns
        cumulative_returns = (1 + stock_data['Returns']).cumprod() - 1
        total_return = cumulative_returns.iloc[-1] * 100
        
        # Annualized return
        days = len(stock_data)
        annualized_return = ((1 + cumulative_returns.iloc[-1]) ** (252/days) - 1) * 100
        
        st.metric("Total Return", f"{total_return:+.2f}%")
        st.metric("Annualized Return", f"{annualized_return:+.2f}%")
        
        # Best and worst days
        best_day = stock_data['Returns'].max() * 100
        worst_day = stock_data['Returns'].min() * 100
        
        st.metric("Best Day", f"{best_day:+.2f}%")
        st.metric("Worst Day", f"{worst_day:+.2f}%")
    
    with col2:
        st.subheader("📊 Risk Metrics")
        
        # Volatility metrics
        daily_vol = stock_data['Returns'].std() * 100
        annual_vol = daily_vol * np.sqrt(252)
        
        st.metric("Daily Volatility", f"{daily_vol:.2f}%")
        st.metric("Annual Volatility", f"{annual_vol:.2f}%")
        
        # Sharpe ratio (assuming 2% risk-free rate)
        risk_free_rate = 0.02
        sharpe_ratio = (annualized_return/100 - risk_free_rate) / (annual_vol/100)
        
        st.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")
        
        # Maximum drawdown
        cumulative = (1 + stock_data['Returns']).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdown = (cumulative - rolling_max) / rolling_max
        max_drawdown = drawdown.min() * 100
        
        st.metric("Max Drawdown", f"{max_drawdown:.2f}%")
    
    with col3:
        st.subheader("📅 Time Analysis")
        
        # Positive/negative days
        positive_days = (stock_data['Returns'] > 0).sum()
        total_days = len(stock_data['Returns'].dropna())
        win_rate = (positive_days / total_days) * 100
        
        st.metric("Positive Days", f"{positive_days}/{total_days}")
        st.metric("Win Rate", f"{win_rate:.1f}%")
        
        # Average gains/losses
        avg_gain = stock_data['Returns'][stock_data['Returns'] > 0].mean() * 100
        avg_loss = stock_data['Returns'][stock_data['Returns'] < 0].mean() * 100
        
        st.metric("Avg Gain", f"{avg_gain:.2f}%")
        st.metric("Avg Loss", f"{avg_loss:.2f}%")
    
    # Returns distribution
    st.subheader("📊 Returns Distribution")
    
    fig = px.histogram(
        x=stock_data['Returns'].dropna() * 100,
        nbins=50,
        title="Daily Returns Distribution",
        labels={'x': 'Daily Returns (%)', 'y': 'Frequency'}
    )
    
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

def display_fundamental_analysis(symbol, company_info, provider):
    """Display fundamental analysis."""
    st.subheader("💰 Fundamental Analysis")
    
    if not company_info:
        st.info("Fundamental data not available")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Valuation Metrics")
        
        # Key valuation metrics
        pe_ratio = company_info.get('trailingPE', 'N/A')
        forward_pe = company_info.get('forwardPE', 'N/A')
        pb_ratio = company_info.get('priceToBook', 'N/A')
        ps_ratio = company_info.get('priceToSalesTrailing12Months', 'N/A')
        
        st.write(f"**P/E Ratio (TTM):** {pe_ratio}")
        st.write(f"**Forward P/E:** {forward_pe}")
        st.write(f"**P/B Ratio:** {pb_ratio}")
        st.write(f"**P/S Ratio:** {ps_ratio}")
        
        # Market metrics
        market_cap = company_info.get('marketCap', 0)
        enterprise_value = company_info.get('enterpriseValue', 'N/A')
        
        if market_cap > 0:
            if market_cap >= 1e12:
                market_cap_str = f"${market_cap/1e12:.2f}T"
            elif market_cap >= 1e9:
                market_cap_str = f"${market_cap/1e9:.2f}B"
            else:
                market_cap_str = f"${market_cap/1e6:.2f}M"
        else:
            market_cap_str = "N/A"
        
        st.write(f"**Market Cap:** {market_cap_str}")
        st.write(f"**Enterprise Value:** {enterprise_value}")
    
    with col2:
        st.subheader("💼 Financial Health")
        
        # Profitability metrics
        profit_margin = company_info.get('profitMargins', 'N/A')
        operating_margin = company_info.get('operatingMargins', 'N/A')
        roe = company_info.get('returnOnEquity', 'N/A')
        roa = company_info.get('returnOnAssets', 'N/A')
        
        if profit_margin != 'N/A':
            profit_margin = f"{profit_margin * 100:.2f}%"
        if operating_margin != 'N/A':
            operating_margin = f"{operating_margin * 100:.2f}%"
        if roe != 'N/A':
            roe = f"{roe * 100:.2f}%"
        if roa != 'N/A':
            roa = f"{roa * 100:.2f}%"
        
        st.write(f"**Profit Margin:** {profit_margin}")
        st.write(f"**Operating Margin:** {operating_margin}")
        st.write(f"**ROE:** {roe}")
        st.write(f"**ROA:** {roa}")
        
        # Growth metrics
        revenue_growth = company_info.get('revenueGrowth', 'N/A')
        earnings_growth = company_info.get('earningsGrowth', 'N/A')
        
        if revenue_growth != 'N/A':
            revenue_growth = f"{revenue_growth * 100:.2f}%"
        if earnings_growth != 'N/A':
            earnings_growth = f"{earnings_growth * 100:.2f}%"
        
        st.write(f"**Revenue Growth:** {revenue_growth}")
        st.write(f"**Earnings Growth:** {earnings_growth}")
    
    # Company description
    if 'longBusinessSummary' in company_info:
        st.subheader("🏢 Company Overview")
        st.write(company_info['longBusinessSummary'])

def display_risk_analysis(symbol, stock_data):
    """Display risk analysis."""
    st.subheader("🎯 Risk Analysis")
    
    # Calculate returns for risk metrics
    stock_data['Returns'] = stock_data['Close'].pct_change()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Risk Metrics")
        
        # Value at Risk (VaR)
        returns = stock_data['Returns'].dropna()
        var_95 = np.percentile(returns, 5) * 100
        var_99 = np.percentile(returns, 1) * 100
        
        st.write(f"**VaR (95%):** {var_95:.2f}%")
        st.write(f"**VaR (99%):** {var_99:.2f}%")
        
        # Conditional VaR (Expected Shortfall)
        cvar_95 = returns[returns <= np.percentile(returns, 5)].mean() * 100
        cvar_99 = returns[returns <= np.percentile(returns, 1)].mean() * 100
        
        st.write(f"**CVaR (95%):** {cvar_95:.2f}%")
        st.write(f"**CVaR (99%):** {cvar_99:.2f}%")
        
        # Skewness and Kurtosis
        skewness = returns.skew()
        kurtosis = returns.kurtosis()
        
        st.write(f"**Skewness:** {skewness:.3f}")
        st.write(f"**Kurtosis:** {kurtosis:.3f}")
    
    with col2:
        st.subheader("📈 Drawdown Analysis")
        
        # Calculate drawdown
        cumulative = (1 + returns).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdown = (cumulative - rolling_max) / rolling_max
        
        # Drawdown statistics
        max_drawdown = drawdown.min() * 100
        current_drawdown = drawdown.iloc[-1] * 100
        
        # Drawdown duration
        drawdown_periods = (drawdown < 0).astype(int)
        drawdown_groups = (drawdown_periods != drawdown_periods.shift()).cumsum()
        drawdown_durations = drawdown_periods.groupby(drawdown_groups).sum()
        max_drawdown_duration = drawdown_durations.max()
        
        st.write(f"**Max Drawdown:** {max_drawdown:.2f}%")
        st.write(f"**Current Drawdown:** {current_drawdown:.2f}%")
        st.write(f"**Max DD Duration:** {max_drawdown_duration} days")
        
        # Recovery analysis
        if current_drawdown < -1:
            st.warning("⚠️ Currently in drawdown")
        else:
            st.success("✅ Near all-time highs")
    
    # Drawdown chart
    st.subheader("📉 Drawdown Chart")
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=drawdown.index,
        y=drawdown * 100,
        mode='lines',
        name='Drawdown',
        fill='tonexty',
        fillcolor='rgba(255,0,0,0.3)',
        line=dict(color='red')
    ))
    
    fig.update_layout(
        title="Drawdown Over Time",
        xaxis_title="Date",
        yaxis_title="Drawdown (%)",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()

