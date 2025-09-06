"""
Google Trends Analysis Page for AltSignals platform.
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

from google_trends_util import GoogleTrendsAnalyzer

# Page configuration
st.set_page_config(
    page_title="Google Trends Analysis",
    page_icon="📈",
    layout="wide"
)

# Initialize session state
if 'trends_data' not in st.session_state:
    st.session_state.trends_data = {}

def main():
    st.title("📈 Google Trends Analysis")
    st.markdown("Analyze search interest and trends for stocks and investment topics")
    
    # Sidebar for settings
    with st.sidebar:
        st.header("🔧 Settings")
        
        # Analysis type
        analysis_type = st.selectbox(
            "Analysis Type",
            ["Stock Interest", "Keyword Comparison", "Trending Searches"],
            help="Choose the type of analysis to perform"
        )
        
        # Timeframe selection
        timeframe_options = {
            "Past Month": "today 1-m",
            "Past 3 Months": "today 3-m", 
            "Past Year": "today 12-m",
            "Past 5 Years": "today 5-y"
        }
        
        timeframe_label = st.selectbox(
            "Timeframe",
            list(timeframe_options.keys()),
            index=2,
            help="Select the time period for analysis"
        )
        timeframe = timeframe_options[timeframe_label]
        
        # Geography selection
        geo_options = {
            "United States": "US",
            "Global": "",
            "United Kingdom": "GB",
            "Canada": "CA",
            "Germany": "DE",
            "Japan": "JP"
        }
        
        geo_label = st.selectbox(
            "Geography",
            list(geo_options.keys()),
            help="Select geographic region for analysis"
        )
        geo = geo_options[geo_label]
        
        # Analysis-specific inputs
        if analysis_type == "Stock Interest":
            symbols = st.multiselect(
                "Select Symbols",
                options=["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "NFLX"],
                default=["AAPL", "MSFT"],
                help="Choose stocks for trends analysis"
            )
            
            custom_symbol = st.text_input("Add Custom Symbol", placeholder="e.g., AAPL")
            if custom_symbol and custom_symbol not in symbols:
                symbols.append(custom_symbol.upper())
        
        elif analysis_type == "Keyword Comparison":
            keywords_input = st.text_area(
                "Keywords (one per line)",
                value="artificial intelligence\nmachine learning\ncryptocurrency\nelectric vehicles",
                help="Enter keywords to compare, one per line"
            )
            keywords = [k.strip() for k in keywords_input.split('\n') if k.strip()]
        
        # Analysis button
        analyze_button = st.button("🔍 Analyze Trends", type="primary")
    
    # Main content area
    if analysis_type == "Stock Interest" and not symbols:
        st.info("Please select at least one symbol to analyze.")
        return
    elif analysis_type == "Keyword Comparison" and not keywords:
        st.info("Please enter at least one keyword to analyze.")
        return
    
    # Initialize trends analyzer
    trends_analyzer = GoogleTrendsAnalyzer()
    
    # Perform analysis
    if analyze_button or analysis_type not in st.session_state.trends_data:
        with st.spinner("Analyzing Google Trends data..."):
            try:
                if analysis_type == "Stock Interest":
                    results = analyze_stock_interest(trends_analyzer, symbols, timeframe, geo)
                elif analysis_type == "Keyword Comparison":
                    results = analyze_keyword_comparison(trends_analyzer, keywords, timeframe, geo)
                elif analysis_type == "Trending Searches":
                    results = analyze_trending_searches(trends_analyzer, geo)
                
                st.session_state.trends_data[analysis_type] = results
                
            except Exception as e:
                st.error(f"Error during analysis: {str(e)}")
                return
    
    # Display results
    if analysis_type in st.session_state.trends_data:
        if analysis_type == "Stock Interest":
            display_stock_interest_results(st.session_state.trends_data[analysis_type])
        elif analysis_type == "Keyword Comparison":
            display_keyword_comparison_results(st.session_state.trends_data[analysis_type])
        elif analysis_type == "Trending Searches":
            display_trending_searches_results(st.session_state.trends_data[analysis_type])
    else:
        st.info("Click 'Analyze Trends' to start the analysis.")

def analyze_stock_interest(analyzer, symbols, timeframe, geo):
    """Analyze stock interest using Google Trends."""
    results = {}
    
    for symbol in symbols:
        # Get company name mapping
        company_names = {
            "AAPL": "Apple",
            "MSFT": "Microsoft",
            "GOOGL": "Google", 
            "AMZN": "Amazon",
            "TSLA": "Tesla",
            "NVDA": "Nvidia",
            "META": "Meta",
            "NFLX": "Netflix"
        }
        
        company_name = company_names.get(symbol, None)
        stock_analysis = analyzer.analyze_stock_interest(symbol, company_name)
        results[symbol] = stock_analysis
    
    return results

def analyze_keyword_comparison(analyzer, keywords, timeframe, geo):
    """Analyze keyword comparison using Google Trends."""
    comparison = analyzer.compare_keywords(keywords, timeframe, geo)
    
    # Get related queries for the first keyword
    related_queries = {}
    if keywords:
        related_queries = analyzer.get_related_queries(keywords[0], timeframe, geo)
    
    return {
        'comparison': comparison,
        'related_queries': related_queries,
        'keywords': keywords,
        'timeframe': timeframe,
        'geo': geo
    }

def analyze_trending_searches(analyzer, geo):
    """Analyze trending searches."""
    trending = analyzer.get_trending_searches(geo)
    return {
        'trending': trending,
        'geo': geo,
        'timestamp': datetime.now()
    }

def display_stock_interest_results(results):
    """Display stock interest analysis results."""
    st.subheader("📊 Stock Interest Analysis")
    
    # Overview metrics
    symbols = list(results.keys())
    cols = st.columns(len(symbols))
    
    for i, (symbol, data) in enumerate(results.items()):
        with cols[i]:
            # Get latest interest score
            latest_score = 0
            trend_direction = "stable"
            
            if 'analysis' in data and 'past_month' in data['analysis']:
                month_data = data['analysis']['past_month']
                if 'interest_score' in month_data:
                    latest_score = month_data['interest_score'].get('score', 0)
            
            if 'overall_trends' in data:
                trend_direction = data['overall_trends'].get('interest_trajectory', 'stable')
            
            # Color based on trend
            if trend_direction == 'increasing':
                delta_color = "normal"
                delta = "📈 Increasing"
            elif trend_direction == 'decreasing':
                delta_color = "inverse"
                delta = "📉 Decreasing"
            else:
                delta_color = "off"
                delta = "➡️ Stable"
            
            st.metric(
                label=f"🔍 {symbol}",
                value=f"{latest_score}",
                delta=delta,
                delta_color=delta_color
            )
    
    # Detailed analysis tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Interest Trends", "⏱️ Time Analysis", "🔍 Keyword Analysis", "📋 Raw Data"])
    
    with tab1:
        display_interest_trends(results)
    
    with tab2:
        display_time_analysis(results)
    
    with tab3:
        display_keyword_analysis(results)
    
    with tab4:
        display_raw_trends_data(results)

def display_interest_trends(results):
    """Display interest trends over time."""
    st.subheader("📈 Search Interest Over Time")
    
    # Create time series chart
    fig = go.Figure()
    
    for symbol, data in results.items():
        if 'analysis' in data and 'past_year' in data['analysis']:
            year_data = data['analysis']['past_year']
            if 'main_keywords' in year_data and 'data' in year_data['main_keywords']:
                df = year_data['main_keywords']['data']
                
                if not df.empty and symbol in df.columns:
                    fig.add_trace(go.Scatter(
                        x=df.index,
                        y=df[symbol],
                        mode='lines+markers',
                        name=symbol,
                        line=dict(width=2)
                    ))
    
    fig.update_layout(
        title="Search Interest Trends (Past Year)",
        xaxis_title="Date",
        yaxis_title="Search Interest (0-100)",
        height=400,
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Interest level comparison
    st.subheader("📊 Current Interest Levels")
    
    symbols = []
    scores = []
    levels = []
    
    for symbol, data in results.items():
        if 'analysis' in data and 'past_month' in data['analysis']:
            month_data = data['analysis']['past_month']
            if 'interest_score' in month_data:
                symbols.append(symbol)
                scores.append(month_data['interest_score'].get('score', 0))
                levels.append(month_data['interest_score'].get('level', 'unknown'))
    
    if symbols:
        # Color mapping for interest levels
        color_map = {
            'very_high': '#FF0000',
            'high': '#FF6600', 
            'moderate': '#FFCC00',
            'low': '#66CC00',
            'very_low': '#00CC00',
            'unknown': '#CCCCCC'
        }
        
        colors = [color_map.get(level, '#CCCCCC') for level in levels]
        
        fig = go.Figure(data=[
            go.Bar(
                x=symbols,
                y=scores,
                marker_color=colors,
                text=[f"{level}" for level in levels],
                textposition='auto'
            )
        ])
        
        fig.update_layout(
            title="Current Interest Scores",
            xaxis_title="Symbols",
            yaxis_title="Interest Score",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)

def display_time_analysis(results):
    """Display time-based analysis."""
    st.subheader("⏱️ Interest Analysis Across Time Periods")
    
    # Create comparison across timeframes
    timeframes = ['past_month', 'past_3_months', 'past_year']
    timeframe_labels = ['1 Month', '3 Months', '1 Year']
    
    for symbol, data in results.items():
        st.write(f"**{symbol} - {data.get('company_name', symbol)}**")
        
        if 'analysis' in data:
            cols = st.columns(len(timeframes))
            
            for i, (timeframe, label) in enumerate(zip(timeframes, timeframe_labels)):
                with cols[i]:
                    if timeframe in data['analysis']:
                        period_data = data['analysis'][timeframe]
                        if 'interest_score' in period_data:
                            score_data = period_data['interest_score']
                            
                            st.metric(
                                label=label,
                                value=f"{score_data.get('score', 0)}",
                                delta=score_data.get('level', 'unknown').replace('_', ' ').title()
                            )
                        else:
                            st.metric(label=label, value="N/A")
                    else:
                        st.metric(label=label, value="N/A")
        
        st.divider()

def display_keyword_analysis(results):
    """Display keyword-specific analysis."""
    st.subheader("🔍 Keyword Performance Analysis")
    
    for symbol, data in results.items():
        with st.expander(f"📊 {symbol} Keyword Analysis"):
            
            if 'analysis' in data and 'past_3_months' in data['analysis']:
                analysis_data = data['analysis']['past_3_months']
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Main Keywords Performance**")
                    if 'main_keywords' in analysis_data and 'summary' in analysis_data['main_keywords']:
                        summary = analysis_data['main_keywords']['summary']
                        
                        for keyword, stats in summary.items():
                            st.write(f"*{keyword}*")
                            st.write(f"- Average: {stats.get('mean', 0)}")
                            st.write(f"- Peak: {stats.get('max', 0)}")
                            st.write(f"- Trend: {stats.get('trend_direction', 'unknown')}")
                            st.write("")
                
                with col2:
                    st.write("**Stock-Related Keywords**")
                    if 'stock_keywords' in analysis_data and 'summary' in analysis_data['stock_keywords']:
                        summary = analysis_data['stock_keywords']['summary']
                        
                        for keyword, stats in summary.items():
                            st.write(f"*{keyword}*")
                            st.write(f"- Average: {stats.get('mean', 0)}")
                            st.write(f"- Peak: {stats.get('max', 0)}")
                            st.write(f"- Trend: {stats.get('trend_direction', 'unknown')}")
                            st.write("")

def display_keyword_comparison_results(results):
    """Display keyword comparison results."""
    st.subheader("🔍 Keyword Comparison Analysis")
    
    comparison = results.get('comparison', {})
    
    if 'error' in comparison:
        st.error(f"Analysis error: {comparison['error']}")
        return
    
    # Overview metrics
    if 'summary' in comparison:
        st.subheader("📊 Keyword Performance Summary")
        
        keywords = list(comparison['summary'].keys())
        cols = st.columns(len(keywords))
        
        for i, (keyword, stats) in enumerate(comparison['summary'].items()):
            with cols[i]:
                trend_direction = stats.get('trend_direction', 'stable')
                
                if trend_direction == 'increasing':
                    delta = "📈 Rising"
                    delta_color = "normal"
                elif trend_direction == 'decreasing':
                    delta = "📉 Falling"
                    delta_color = "inverse"
                else:
                    delta = "➡️ Stable"
                    delta_color = "off"
                
                st.metric(
                    label=keyword[:20] + "..." if len(keyword) > 20 else keyword,
                    value=f"{stats.get('mean', 0):.1f}",
                    delta=delta,
                    delta_color=delta_color
                )
    
    # Trends visualization
    if 'data' in comparison and not comparison['data'].empty:
        st.subheader("📈 Interest Trends Comparison")
        
        df = comparison['data']
        
        fig = go.Figure()
        
        for column in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index,
                y=df[column],
                mode='lines+markers',
                name=column,
                line=dict(width=2)
            ))
        
        fig.update_layout(
            title="Keyword Interest Over Time",
            xaxis_title="Date",
            yaxis_title="Search Interest (0-100)",
            height=500,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Correlation analysis
        if len(df.columns) > 1:
            st.subheader("🔗 Keyword Correlations")
            
            corr_matrix = df.corr()
            
            fig = px.imshow(
                corr_matrix,
                title="Keyword Correlation Matrix",
                color_continuous_scale="RdBu",
                aspect="auto"
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    # Related queries
    related_queries = results.get('related_queries', {})
    if related_queries:
        st.subheader("🔍 Related Queries")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if 'top' in related_queries and not related_queries['top'].empty:
                st.write("**Top Related Queries**")
                st.dataframe(related_queries['top'], use_container_width=True)
        
        with col2:
            if 'rising' in related_queries and not related_queries['rising'].empty:
                st.write("**Rising Related Queries**")
                st.dataframe(related_queries['rising'], use_container_width=True)

def display_trending_searches_results(results):
    """Display trending searches results."""
    st.subheader("🔥 Trending Searches")
    
    trending = results.get('trending')
    geo = results.get('geo', 'US')
    
    if trending is not None and not trending.empty:
        st.write(f"**Current trending searches in {geo}:**")
        
        # Display as a nice list
        for i, term in enumerate(trending.iloc[:, 0].head(20), 1):
            st.write(f"{i}. {term}")
    else:
        st.info("No trending searches data available.")

def display_raw_trends_data(results):
    """Display raw trends data."""
    st.subheader("📋 Raw Trends Data")
    
    for symbol, data in results.items():
        with st.expander(f"📊 {symbol} Raw Data"):
            st.json(data)
    
    # Export option
    st.subheader("💾 Export Data")
    
    if st.button("📥 Download Trends Data as JSON"):
        import json
        json_data = json.dumps(results, indent=2, default=str)
        st.download_button(
            label="Download JSON",
            data=json_data,
            file_name=f"google_trends_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )

if __name__ == "__main__":
    main()

