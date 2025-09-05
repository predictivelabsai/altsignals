"""
Sentiment analysis utility using OpenAI LLM and PostgreSQL news database.
Analyzes news articles for sentiment and relevance to stock movements.
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime, timedelta
from sqlalchemy import create_engine, text
from openai import OpenAI
from dotenv import load_dotenv
import json
import time

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SentimentAnalyzer:
    """Sentiment analyzer using OpenAI LLM for financial news."""
    
    def __init__(self, openai_api_key: Optional[str] = None, news_db_url: Optional[str] = None):
        self.openai_api_key = openai_api_key or os.getenv('OPENAI_API_KEY')
        self.news_db_url = news_db_url or os.getenv('NEWS_DB_URL')
        
        if not self.openai_api_key:
            raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable.")
        
        if not self.news_db_url:
            raise ValueError("News database URL not found. Set NEWS_DB_URL environment variable.")
        
        # Initialize OpenAI client
        self.client = OpenAI(api_key=self.openai_api_key)
        
        # Initialize database connection
        self.engine = create_engine(self.news_db_url)
        
        logger.info("SentimentAnalyzer initialized")
    
    def get_news_data(self, ticker: str = None, limit: int = 100, 
                     days_back: int = 30) -> pd.DataFrame:
        """Get news data from PostgreSQL database."""
        try:
            # Build query
            query = """
            SELECT ticker, yf_ticker, content_en, event, reason, industry, 
                   title_en, link, published_date, publisher
            FROM news
            WHERE published_date >= %s
            """
            
            params = [datetime.now() - timedelta(days=days_back)]
            
            if ticker:
                query += " AND (ticker = %s OR yf_ticker = %s)"
                params.extend([ticker, ticker])
            
            query += " ORDER BY published_date DESC LIMIT %s"
            params.append(limit)
            
            # Execute query
            df = pd.read_sql_query(query, self.engine, params=params)
            
            logger.info(f"Retrieved {len(df)} news articles")
            return df
            
        except Exception as e:
            logger.error(f"Error retrieving news data: {e}")
            return pd.DataFrame()
    
    def analyze_sentiment_openai(self, text: str, ticker: str = None) -> Dict:
        """Analyze sentiment using OpenAI LLM."""
        try:
            # Create prompt for financial sentiment analysis
            prompt = f"""
            Analyze the sentiment of the following financial news text and provide a detailed assessment:

            Text: "{text}"
            {f"Stock Ticker: {ticker}" if ticker else ""}

            Please provide your analysis in the following JSON format:
            {{
                "sentiment_score": <float between -1.0 (very negative) and 1.0 (very positive)>,
                "sentiment_label": "<positive/negative/neutral>",
                "confidence": <float between 0.0 and 1.0>,
                "relevance_score": <float between 0.0 and 1.0 indicating relevance to stock price movement>,
                "key_themes": ["<theme1>", "<theme2>", ...],
                "potential_impact": "<brief description of potential market impact>",
                "reasoning": "<explanation of the sentiment analysis>"
            }}

            Focus on:
            1. Overall sentiment towards the company/stock
            2. Potential impact on stock price
            3. Market relevance and significance
            4. Key financial themes (earnings, revenue, growth, risks, etc.)
            """
            
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a financial analyst expert in sentiment analysis of news articles and their impact on stock prices. Provide accurate, objective analysis in valid JSON format."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=1000
            )
            
            # Parse response
            response_text = response.choices[0].message.content.strip()
            
            # Extract JSON from response
            try:
                # Find JSON in the response
                start_idx = response_text.find('{')
                end_idx = response_text.rfind('}') + 1
                json_str = response_text[start_idx:end_idx]
                
                result = json.loads(json_str)
                
                # Validate and clean result
                result['sentiment_score'] = max(-1.0, min(1.0, float(result.get('sentiment_score', 0))))
                result['confidence'] = max(0.0, min(1.0, float(result.get('confidence', 0.5))))
                result['relevance_score'] = max(0.0, min(1.0, float(result.get('relevance_score', 0.5))))
                
                # Ensure sentiment_label is valid
                if result.get('sentiment_label') not in ['positive', 'negative', 'neutral']:
                    if result['sentiment_score'] > 0.1:
                        result['sentiment_label'] = 'positive'
                    elif result['sentiment_score'] < -0.1:
                        result['sentiment_label'] = 'negative'
                    else:
                        result['sentiment_label'] = 'neutral'
                
                return result
                
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse JSON response: {e}")
                # Fallback to simple analysis
                return self._fallback_sentiment_analysis(response_text)
            
        except Exception as e:
            logger.error(f"Error in OpenAI sentiment analysis: {e}")
            return {
                'sentiment_score': 0.0,
                'sentiment_label': 'neutral',
                'confidence': 0.0,
                'relevance_score': 0.0,
                'key_themes': [],
                'potential_impact': 'Unable to analyze',
                'reasoning': f'Analysis failed: {str(e)}'
            }
    
    def _fallback_sentiment_analysis(self, text: str) -> Dict:
        """Fallback sentiment analysis using simple keyword matching."""
        positive_words = ['growth', 'profit', 'revenue', 'beat', 'exceed', 'strong', 
                         'positive', 'gain', 'rise', 'increase', 'bullish', 'upgrade']
        negative_words = ['loss', 'decline', 'fall', 'weak', 'negative', 'drop', 
                         'decrease', 'bearish', 'downgrade', 'risk', 'concern']
        
        text_lower = text.lower()
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        if positive_count > negative_count:
            sentiment_score = min(0.8, positive_count * 0.2)
            sentiment_label = 'positive'
        elif negative_count > positive_count:
            sentiment_score = max(-0.8, -negative_count * 0.2)
            sentiment_label = 'negative'
        else:
            sentiment_score = 0.0
            sentiment_label = 'neutral'
        
        return {
            'sentiment_score': sentiment_score,
            'sentiment_label': sentiment_label,
            'confidence': 0.3,  # Low confidence for fallback
            'relevance_score': 0.5,
            'key_themes': [],
            'potential_impact': 'Basic keyword-based analysis',
            'reasoning': 'Fallback analysis using keyword matching'
        }
    
    def analyze_news_batch(self, news_df: pd.DataFrame, 
                          batch_size: int = 10, delay: float = 1.0) -> pd.DataFrame:
        """Analyze sentiment for a batch of news articles."""
        if news_df.empty:
            return news_df
        
        results = []
        
        for i in range(0, len(news_df), batch_size):
            batch = news_df.iloc[i:i+batch_size]
            
            for _, row in batch.iterrows():
                # Combine title and content for analysis
                text_to_analyze = f"{row.get('title_en', '')} {row.get('content_en', '')}"
                
                if not text_to_analyze.strip():
                    continue
                
                # Analyze sentiment
                sentiment_result = self.analyze_sentiment_openai(
                    text_to_analyze, 
                    row.get('ticker') or row.get('yf_ticker')
                )
                
                # Combine with original data
                result_row = row.to_dict()
                result_row.update(sentiment_result)
                results.append(result_row)
                
                # Rate limiting
                time.sleep(delay)
            
            logger.info(f"Processed batch {i//batch_size + 1}/{(len(news_df)-1)//batch_size + 1}")
        
        return pd.DataFrame(results)
    
    def get_sentiment_summary(self, ticker: str, days_back: int = 7) -> Dict:
        """Get sentiment summary for a specific ticker."""
        try:
            # Get recent news
            news_df = self.get_news_data(ticker=ticker, limit=50, days_back=days_back)
            
            if news_df.empty:
                return {
                    'ticker': ticker,
                    'article_count': 0,
                    'avg_sentiment': 0.0,
                    'sentiment_trend': 'neutral',
                    'confidence': 0.0,
                    'key_themes': [],
                    'recent_articles': []
                }
            
            # Analyze sentiment for recent articles (limit to avoid API costs)
            analyzed_df = self.analyze_news_batch(news_df.head(10), batch_size=5, delay=2.0)
            
            if analyzed_df.empty:
                return {
                    'ticker': ticker,
                    'article_count': len(news_df),
                    'avg_sentiment': 0.0,
                    'sentiment_trend': 'neutral',
                    'confidence': 0.0,
                    'key_themes': [],
                    'recent_articles': []
                }
            
            # Calculate summary statistics
            avg_sentiment = analyzed_df['sentiment_score'].mean()
            avg_confidence = analyzed_df['confidence'].mean()
            
            # Determine trend
            if avg_sentiment > 0.2:
                sentiment_trend = 'positive'
            elif avg_sentiment < -0.2:
                sentiment_trend = 'negative'
            else:
                sentiment_trend = 'neutral'
            
            # Extract key themes
            all_themes = []
            for themes in analyzed_df['key_themes']:
                if isinstance(themes, list):
                    all_themes.extend(themes)
            
            # Count theme frequency
            theme_counts = {}
            for theme in all_themes:
                theme_counts[theme] = theme_counts.get(theme, 0) + 1
            
            top_themes = sorted(theme_counts.items(), key=lambda x: x[1], reverse=True)[:5]
            key_themes = [theme for theme, count in top_themes]
            
            # Recent articles summary
            recent_articles = []
            for _, row in analyzed_df.head(5).iterrows():
                recent_articles.append({
                    'title': row.get('title_en', ''),
                    'published_date': row.get('published_date'),
                    'sentiment_score': row.get('sentiment_score', 0),
                    'sentiment_label': row.get('sentiment_label', 'neutral'),
                    'relevance_score': row.get('relevance_score', 0),
                    'link': row.get('link', '')
                })
            
            return {
                'ticker': ticker,
                'article_count': len(news_df),
                'analyzed_count': len(analyzed_df),
                'avg_sentiment': float(avg_sentiment),
                'sentiment_trend': sentiment_trend,
                'confidence': float(avg_confidence),
                'key_themes': key_themes,
                'recent_articles': recent_articles,
                'analysis_date': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting sentiment summary for {ticker}: {e}")
            return {
                'ticker': ticker,
                'error': str(e),
                'article_count': 0,
                'avg_sentiment': 0.0,
                'sentiment_trend': 'neutral',
                'confidence': 0.0
            }
    
    def get_market_sentiment_overview(self, tickers: List[str], days_back: int = 7) -> Dict:
        """Get market sentiment overview for multiple tickers."""
        results = {}
        
        for ticker in tickers:
            try:
                results[ticker] = self.get_sentiment_summary(ticker, days_back)
                time.sleep(1.0)  # Rate limiting
            except Exception as e:
                logger.error(f"Error analyzing {ticker}: {e}")
                results[ticker] = {'error': str(e)}
        
        # Calculate overall market sentiment
        valid_results = [r for r in results.values() if 'avg_sentiment' in r]
        
        if valid_results:
            overall_sentiment = sum(r['avg_sentiment'] for r in valid_results) / len(valid_results)
            overall_confidence = sum(r['confidence'] for r in valid_results) / len(valid_results)
        else:
            overall_sentiment = 0.0
            overall_confidence = 0.0
        
        return {
            'individual_results': results,
            'overall_sentiment': overall_sentiment,
            'overall_confidence': overall_confidence,
            'analysis_date': datetime.now().isoformat(),
            'tickers_analyzed': len(valid_results),
            'total_tickers': len(tickers)
        }


def analyze_ticker_sentiment(ticker: str, days_back: int = 7) -> Dict:
    """Convenience function to analyze sentiment for a single ticker."""
    analyzer = SentimentAnalyzer()
    return analyzer.get_sentiment_summary(ticker, days_back)


def get_recent_news(ticker: str = None, limit: int = 20) -> pd.DataFrame:
    """Convenience function to get recent news."""
    analyzer = SentimentAnalyzer()
    return analyzer.get_news_data(ticker=ticker, limit=limit, days_back=7)


if __name__ == "__main__":
    # Test the sentiment analyzer
    try:
        print("Testing Sentiment Analyzer...")
        
        analyzer = SentimentAnalyzer()
        
        # Test getting news data
        print("Getting recent news...")
        news_df = analyzer.get_news_data(limit=5)
        print(f"Retrieved {len(news_df)} news articles")
        
        if not news_df.empty:
            print("\nSample news articles:")
            for _, row in news_df.head(2).iterrows():
                print(f"- {row.get('title_en', 'No title')}")
                print(f"  Ticker: {row.get('ticker', 'N/A')}")
                print(f"  Published: {row.get('published_date', 'N/A')}")
        
        # Test sentiment analysis on sample text
        print("\nTesting sentiment analysis...")
        sample_text = "Apple Inc. reported strong quarterly earnings with revenue beating expectations by 15%. The company's iPhone sales showed robust growth in international markets."
        
        sentiment_result = analyzer.analyze_sentiment_openai(sample_text, "AAPL")
        print(f"Sample analysis result:")
        print(f"- Sentiment Score: {sentiment_result['sentiment_score']}")
        print(f"- Sentiment Label: {sentiment_result['sentiment_label']}")
        print(f"- Confidence: {sentiment_result['confidence']}")
        print(f"- Relevance: {sentiment_result['relevance_score']}")
        
        # Test ticker sentiment summary (if we have data)
        if not news_df.empty and 'ticker' in news_df.columns:
            sample_ticker = news_df['ticker'].iloc[0]
            if sample_ticker:
                print(f"\nTesting sentiment summary for {sample_ticker}...")
                summary = analyzer.get_sentiment_summary(sample_ticker, days_back=7)
                print(f"- Articles analyzed: {summary.get('analyzed_count', 0)}")
                print(f"- Average sentiment: {summary.get('avg_sentiment', 0):.3f}")
                print(f"- Sentiment trend: {summary.get('sentiment_trend', 'neutral')}")
        
        print("\nSentiment analyzer testing completed!")
        
    except Exception as e:
        print(f"Test failed: {e}")
        print("Make sure OPENAI_API_KEY and NEWS_DB_URL are set in your environment.")

