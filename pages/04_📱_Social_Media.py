"""
Social Media Sentiment Analysis Page for AltSignals platform.
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

from social_media_util import SocialMediaSentimentManager

# Page configuration
st.set_page_config(
    page_title="Social Media Sentiment",
    page_icon="📱",
    layout="wide"
)

# Initialize session state
if 'social_sentiment_data' not in st.session_state:
    st.session_state.social_sentiment_data = {}

def main():
    st.title("📱 Social Media Sentiment Analysis")
    st.markdown("Analyze social media sentiment from Twitter and Reddit for investment insights")
    
    # Sidebar for settings
    with st.sidebar:
        st.header("🔧 Settings")
        
        # Data source selection
        st.subheader("Data Sources")
        use_twitter = st.checkbox("Twitter", value=True, help="Analyze Twitter sentiment")
        use_reddit = st.checkbox("Reddit", value=True, help="Analyze Reddit sentiment")
        
        # Symbol selection
        st.subheader("Analysis Settings")
        symbols = st.multiselect(
            "Select Symbols to Analyze",
            options=["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "NFLX"],
            default=["AAPL", "MSFT", "GOOGL"],
            help="Choose stocks for sentiment analysis"
        )
        
        # Custom symbol
        custom_symbol = st.text_input("Add Custom Symbol", placeholder="e.g., AAPL")
        if custom_symbol and custom_symbol not in symbols:
            symbols.append(custom_symbol.upper())
        
        # Analysis button
        analyze_button = st.button("🔍 Analyze Sentiment", type="primary")
    
    # Main content area
    if not symbols:
        st.info("Please select at least one symbol to analyze.")
        return
    
    # Initialize sentiment manager
    sentiment_manager = SocialMediaSentimentManager()
    
    # Analyze sentiment when button is clicked or data is not available
    if analyze_button or not st.session_state.social_sentiment_data:
        with st.spinner("Analyzing social media sentiment..."):
            st.session_state.social_sentiment_data = {}
            
            for symbol in symbols:
                try:
                    # Get company name mapping (simplified)
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
                    
                    company_name = company_names.get(symbol, symbol)
                    results = sentiment_manager.get_comprehensive_sentiment(symbol, company_name)
                    st.session_state.social_sentiment_data[symbol] = results
                    
                except Exception as e:
                    st.error(f"Error analyzing {symbol}: {str(e)}")
    
    # Display results
    if st.session_state.social_sentiment_data:
        display_sentiment_dashboard(st.session_state.social_sentiment_data, use_twitter, use_reddit)
    else:
        st.info("Click 'Analyze Sentiment' to start the analysis.")

def display_sentiment_dashboard(sentiment_data, use_twitter, use_reddit):
    """Display the sentiment analysis dashboard."""
    
    # Overview metrics
    st.subheader("📊 Sentiment Overview")
    
    # Create metrics columns
    cols = st.columns(len(sentiment_data))
    
    for i, (symbol, data) in enumerate(sentiment_data.items()):
        with cols[i]:
            overall = data.get('overall', {})
            sentiment_score = overall.get('sentiment_score', 0)
            sentiment_label = overall.get('sentiment_label', 'neutral')
            confidence = overall.get('confidence', 0)
            
            # Color based on sentiment
            if sentiment_label == 'positive':
                color = "🟢"
            elif sentiment_label == 'negative':
                color = "🔴"
            else:
                color = "🟡"
            
            st.metric(
                label=f"{color} {symbol}",
                value=f"{sentiment_label.title()}",
                delta=f"Score: {sentiment_score} (Conf: {confidence})"
            )
    
    # Detailed analysis tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Sentiment Trends", "🐦 Twitter Analysis", "🔴 Reddit Analysis", "📋 Detailed Data"])
    
    with tab1:
        display_sentiment_trends(sentiment_data)
    
    with tab2:
        if use_twitter:
            display_twitter_analysis(sentiment_data)
        else:
            st.info("Twitter analysis is disabled. Enable it in the sidebar.")
    
    with tab3:
        if use_reddit:
            display_reddit_analysis(sentiment_data)
        else:
            st.info("Reddit analysis is disabled. Enable it in the sidebar.")
    
    with tab4:
        display_detailed_data(sentiment_data)

def display_sentiment_trends(sentiment_data):
    """Display sentiment trends comparison."""
    st.subheader("📈 Sentiment Score Comparison")
    
    # Create comparison chart
    symbols = list(sentiment_data.keys())
    twitter_scores = []
    reddit_scores = []
    overall_scores = []
    
    for symbol in symbols:
        data = sentiment_data[symbol]
        twitter_scores.append(data.get('twitter', {}).get('sentiment_score', 0))
        reddit_scores.append(data.get('reddit', {}).get('sentiment_score', 0))
        overall_scores.append(data.get('overall', {}).get('sentiment_score', 0))
    
    # Create grouped bar chart
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='Twitter',
        x=symbols,
        y=twitter_scores,
        marker_color='#1DA1F2'
    ))
    
    fig.add_trace(go.Bar(
        name='Reddit',
        x=symbols,
        y=reddit_scores,
        marker_color='#FF4500'
    ))
    
    fig.add_trace(go.Bar(
        name='Overall',
        x=symbols,
        y=overall_scores,
        marker_color='#9146FF'
    ))
    
    fig.update_layout(
        title="Sentiment Scores by Platform",
        xaxis_title="Symbols",
        yaxis_title="Sentiment Score",
        barmode='group',
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Sentiment distribution
    st.subheader("📊 Sentiment Distribution")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Twitter sentiment distribution
        twitter_dist = {'Positive': 0, 'Negative': 0, 'Neutral': 0}
        for data in sentiment_data.values():
            twitter_data = data.get('twitter', {})
            if 'sentiment_distribution' in twitter_data:
                dist = twitter_data['sentiment_distribution']
                twitter_dist['Positive'] += dist.get('positive', 0)
                twitter_dist['Negative'] += dist.get('negative', 0)
                twitter_dist['Neutral'] += dist.get('neutral', 0)
        
        if sum(twitter_dist.values()) > 0:
            fig_twitter = px.pie(
                values=list(twitter_dist.values()),
                names=list(twitter_dist.keys()),
                title="Twitter Sentiment Distribution",
                color_discrete_map={
                    'Positive': '#00C851',
                    'Negative': '#FF4444',
                    'Neutral': '#FFBB33'
                }
            )
            st.plotly_chart(fig_twitter, use_container_width=True)
        else:
            st.info("No Twitter sentiment data available")
    
    with col2:
        # Reddit sentiment distribution
        reddit_dist = {'Positive': 0, 'Negative': 0, 'Neutral': 0}
        for data in sentiment_data.values():
            reddit_data = data.get('reddit', {})
            if 'sentiment_distribution' in reddit_data:
                dist = reddit_data['sentiment_distribution']
                reddit_dist['Positive'] += dist.get('positive', 0)
                reddit_dist['Negative'] += dist.get('negative', 0)
                reddit_dist['Neutral'] += dist.get('neutral', 0)
        
        if sum(reddit_dist.values()) > 0:
            fig_reddit = px.pie(
                values=list(reddit_dist.values()),
                names=list(reddit_dist.keys()),
                title="Reddit Sentiment Distribution",
                color_discrete_map={
                    'Positive': '#00C851',
                    'Negative': '#FF4444',
                    'Neutral': '#FFBB33'
                }
            )
            st.plotly_chart(fig_reddit, use_container_width=True)
        else:
            st.info("No Reddit sentiment data available")

def display_twitter_analysis(sentiment_data):
    """Display Twitter-specific analysis."""
    st.subheader("🐦 Twitter Sentiment Analysis")
    
    # Twitter metrics
    cols = st.columns(4)
    
    total_tweets = sum(data.get('twitter', {}).get('total_tweets', 0) for data in sentiment_data.values())
    total_engagement = sum(data.get('twitter', {}).get('total_engagement', 0) for data in sentiment_data.values())
    avg_sentiment = np.mean([data.get('twitter', {}).get('sentiment_score', 0) for data in sentiment_data.values()])
    
    with cols[0]:
        st.metric("Total Tweets", f"{total_tweets:,}")
    with cols[1]:
        st.metric("Total Engagement", f"{total_engagement:,}")
    with cols[2]:
        st.metric("Average Sentiment", f"{avg_sentiment:.3f}")
    with cols[3]:
        sentiment_label = "Positive" if avg_sentiment > 0.1 else "Negative" if avg_sentiment < -0.1 else "Neutral"
        st.metric("Overall Mood", sentiment_label)
    
    # Twitter engagement vs sentiment
    st.subheader("📊 Engagement vs Sentiment")
    
    symbols = []
    sentiments = []
    engagements = []
    tweet_counts = []
    
    for symbol, data in sentiment_data.items():
        twitter_data = data.get('twitter', {})
        if twitter_data and 'sentiment_score' in twitter_data:
            symbols.append(symbol)
            sentiments.append(twitter_data.get('sentiment_score', 0))
            engagements.append(twitter_data.get('total_engagement', 0))
            tweet_counts.append(twitter_data.get('total_tweets', 0))
    
    if symbols:
        fig = px.scatter(
            x=sentiments,
            y=engagements,
            size=tweet_counts,
            text=symbols,
            title="Twitter Engagement vs Sentiment",
            labels={
                'x': 'Sentiment Score',
                'y': 'Total Engagement',
                'size': 'Tweet Count'
            }
        )
        fig.update_traces(textposition="top center")
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No Twitter data available for visualization")

def display_reddit_analysis(sentiment_data):
    """Display Reddit-specific analysis."""
    st.subheader("🔴 Reddit Sentiment Analysis")
    
    # Reddit metrics
    cols = st.columns(4)
    
    total_posts = sum(data.get('reddit', {}).get('total_posts', 0) for data in sentiment_data.values())
    total_score = sum(data.get('reddit', {}).get('total_score', 0) for data in sentiment_data.values())
    total_comments = sum(data.get('reddit', {}).get('total_comments', 0) for data in sentiment_data.values())
    avg_sentiment = np.mean([data.get('reddit', {}).get('sentiment_score', 0) for data in sentiment_data.values()])
    
    with cols[0]:
        st.metric("Total Posts", f"{total_posts:,}")
    with cols[1]:
        st.metric("Total Score", f"{total_score:,}")
    with cols[2]:
        st.metric("Total Comments", f"{total_comments:,}")
    with cols[3]:
        sentiment_label = "Positive" if avg_sentiment > 0.1 else "Negative" if avg_sentiment < -0.1 else "Neutral"
        st.metric("Overall Mood", sentiment_label)
    
    # Subreddit breakdown
    st.subheader("📊 Subreddit Activity")
    
    # Aggregate subreddit data
    subreddit_data = {}
    for symbol, data in sentiment_data.items():
        reddit_data = data.get('reddit', {})
        if 'subreddit_breakdown' in reddit_data:
            for subreddit, count in reddit_data['subreddit_breakdown'].items():
                if subreddit not in subreddit_data:
                    subreddit_data[subreddit] = 0
                subreddit_data[subreddit] += count
    
    if subreddit_data:
        fig = px.bar(
            x=list(subreddit_data.keys()),
            y=list(subreddit_data.values()),
            title="Posts by Subreddit",
            labels={'x': 'Subreddit', 'y': 'Number of Posts'}
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No subreddit data available")

def display_detailed_data(sentiment_data):
    """Display detailed sentiment data."""
    st.subheader("📋 Detailed Sentiment Data")
    
    for symbol, data in sentiment_data.items():
        with st.expander(f"📊 {symbol} - {data.get('company_name', symbol)} Details"):
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.write("**Overall Sentiment**")
                overall = data.get('overall', {})
                st.json(overall)
            
            with col2:
                st.write("**Twitter Data**")
                twitter = data.get('twitter', {})
                if 'error' not in twitter:
                    st.json(twitter)
                else:
                    st.error(f"Twitter Error: {twitter['error']}")
            
            with col3:
                st.write("**Reddit Data**")
                reddit = data.get('reddit', {})
                if 'error' not in reddit:
                    st.json(reddit)
                else:
                    st.error(f"Reddit Error: {reddit['error']}")
    
    # Export data option
    st.subheader("💾 Export Data")
    
    if st.button("📥 Download Sentiment Data as JSON"):
        import json
        json_data = json.dumps(sentiment_data, indent=2, default=str)
        st.download_button(
            label="Download JSON",
            data=json_data,
            file_name=f"social_sentiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )
    
    # Footer with data source information
    st.markdown("---")
    st.markdown("**📱 Social Media Sentiment Analysis** | 💭 **SYNTHETIC DATA**: Generated for demonstration purposes")
    st.markdown("⚠️ *This page uses SYNTHETIC social media data. In production, this would connect to real Twitter/Reddit APIs.*")

if __name__ == "__main__":
    main()

