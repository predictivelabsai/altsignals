"""
Social Media Sentiment Analysis Utility for AltSignals platform.
Integrates with Twitter and Reddit APIs for sentiment analysis.
"""

import os
import tweepy
import praw
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import logging
import json
import time
from textblob import TextBlob
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TwitterSentimentAnalyzer:
    """Twitter sentiment analysis using Tweepy."""
    
    def __init__(self):
        """Initialize Twitter API client."""
        self.api_key = os.getenv('TWITTER_API_KEY')
        self.api_secret = os.getenv('TWITTER_API_SECRET')
        self.access_token = os.getenv('TWITTER_ACCESS_TOKEN')
        self.access_token_secret = os.getenv('TWITTER_ACCESS_TOKEN_SECRET')
        self.bearer_token = os.getenv('TWITTER_BEARER_TOKEN')
        
        if not all([self.api_key, self.api_secret, self.access_token, self.access_token_secret]):
            logger.warning("Twitter API credentials not found. Using synthetic data.")
            self.client = None
        else:
            try:
                self.client = tweepy.Client(
                    bearer_token=self.bearer_token,
                    consumer_key=self.api_key,
                    consumer_secret=self.api_secret,
                    access_token=self.access_token,
                    access_token_secret=self.access_token_secret,
                    wait_on_rate_limit=True
                )
                logger.info("Twitter API client initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize Twitter client: {e}")
                self.client = None
    
    def search_tweets(self, query: str, max_results: int = 100) -> List[Dict]:
        """Search for tweets about a specific topic."""
        if not self.client:
            return self._generate_synthetic_tweets(query, max_results)
        
        try:
            tweets = tweepy.Paginator(
                self.client.search_recent_tweets,
                query=query,
                tweet_fields=['created_at', 'author_id', 'public_metrics', 'context_annotations'],
                max_results=min(max_results, 100)
            ).flatten(limit=max_results)
            
            tweet_data = []
            for tweet in tweets:
                tweet_data.append({
                    'id': tweet.id,
                    'text': tweet.text,
                    'created_at': tweet.created_at,
                    'author_id': tweet.author_id,
                    'retweet_count': tweet.public_metrics['retweet_count'],
                    'like_count': tweet.public_metrics['like_count'],
                    'reply_count': tweet.public_metrics['reply_count'],
                    'quote_count': tweet.public_metrics['quote_count']
                })
            
            logger.info(f"Retrieved {len(tweet_data)} tweets for query: {query}")
            return tweet_data
            
        except Exception as e:
            logger.error(f"Error searching tweets: {e}")
            return self._generate_synthetic_tweets(query, max_results)
    
    def _generate_synthetic_tweets(self, query: str, count: int) -> List[Dict]:
        """Generate synthetic tweet data for testing."""
        synthetic_tweets = []
        sentiments = ['positive', 'negative', 'neutral']
        
        for i in range(count):
            sentiment = np.random.choice(sentiments)
            if sentiment == 'positive':
                text = f"Great news about {query}! Really bullish on this stock. #investing"
            elif sentiment == 'negative':
                text = f"Concerned about {query} recent performance. Might be time to sell. #stocks"
            else:
                text = f"Watching {query} closely. Mixed signals in the market. #trading"
            
            synthetic_tweets.append({
                'id': f"synthetic_{i}",
                'text': text,
                'created_at': datetime.now() - timedelta(hours=np.random.randint(1, 24)),
                'author_id': f"user_{i}",
                'retweet_count': np.random.randint(0, 100),
                'like_count': np.random.randint(0, 500),
                'reply_count': np.random.randint(0, 50),
                'quote_count': np.random.randint(0, 20),
                'synthetic': True
            })
        
        logger.info(f"Generated {len(synthetic_tweets)} synthetic tweets for {query}")
        return synthetic_tweets
    
    def analyze_sentiment(self, tweets: List[Dict]) -> Dict[str, Any]:
        """Analyze sentiment of tweets."""
        if not tweets:
            return {'sentiment_score': 0, 'sentiment_label': 'neutral', 'total_tweets': 0}
        
        sentiments = []
        total_engagement = 0
        
        for tweet in tweets:
            # Clean text
            text = self._clean_text(tweet['text'])
            
            # Analyze sentiment using TextBlob
            blob = TextBlob(text)
            sentiment_score = blob.sentiment.polarity
            sentiments.append(sentiment_score)
            
            # Weight by engagement
            engagement = (tweet.get('like_count', 0) + 
                         tweet.get('retweet_count', 0) + 
                         tweet.get('reply_count', 0))
            total_engagement += engagement
        
        avg_sentiment = np.mean(sentiments)
        sentiment_label = self._get_sentiment_label(avg_sentiment)
        
        return {
            'sentiment_score': round(avg_sentiment, 3),
            'sentiment_label': sentiment_label,
            'total_tweets': len(tweets),
            'total_engagement': total_engagement,
            'sentiment_distribution': {
                'positive': len([s for s in sentiments if s > 0.1]),
                'negative': len([s for s in sentiments if s < -0.1]),
                'neutral': len([s for s in sentiments if -0.1 <= s <= 0.1])
            }
        }
    
    def _clean_text(self, text: str) -> str:
        """Clean tweet text for sentiment analysis."""
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        # Remove user mentions and hashtags
        text = re.sub(r'@\w+|#\w+', '', text)
        # Remove extra whitespace
        text = ' '.join(text.split())
        return text
    
    def _get_sentiment_label(self, score: float) -> str:
        """Convert sentiment score to label."""
        if score > 0.1:
            return 'positive'
        elif score < -0.1:
            return 'negative'
        else:
            return 'neutral'


class RedditSentimentAnalyzer:
    """Reddit sentiment analysis using PRAW."""
    
    def __init__(self):
        """Initialize Reddit API client."""
        self.client_id = os.getenv('REDDIT_CLIENT_ID')
        self.client_secret = os.getenv('REDDIT_CLIENT_SECRET')
        self.user_agent = os.getenv('REDDIT_USER_AGENT', 'AltSignals:v1.0')
        
        if not all([self.client_id, self.client_secret]):
            logger.warning("Reddit API credentials not found. Using synthetic data.")
            self.reddit = None
        else:
            try:
                self.reddit = praw.Reddit(
                    client_id=self.client_id,
                    client_secret=self.client_secret,
                    user_agent=self.user_agent
                )
                logger.info("Reddit API client initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize Reddit client: {e}")
                self.reddit = None
    
    def search_posts(self, query: str, subreddits: List[str] = None, limit: int = 100) -> List[Dict]:
        """Search for Reddit posts about a specific topic."""
        if not self.reddit:
            return self._generate_synthetic_posts(query, limit)
        
        if subreddits is None:
            subreddits = ['investing', 'stocks', 'SecurityAnalysis', 'ValueInvesting', 'wallstreetbets']
        
        posts = []
        try:
            for subreddit_name in subreddits:
                subreddit = self.reddit.subreddit(subreddit_name)
                
                # Search recent posts
                for post in subreddit.search(query, sort='new', time_filter='week', limit=limit//len(subreddits)):
                    posts.append({
                        'id': post.id,
                        'title': post.title,
                        'text': post.selftext,
                        'score': post.score,
                        'upvote_ratio': post.upvote_ratio,
                        'num_comments': post.num_comments,
                        'created_utc': datetime.fromtimestamp(post.created_utc),
                        'subreddit': subreddit_name,
                        'url': post.url
                    })
            
            logger.info(f"Retrieved {len(posts)} Reddit posts for query: {query}")
            return posts
            
        except Exception as e:
            logger.error(f"Error searching Reddit posts: {e}")
            return self._generate_synthetic_posts(query, limit)
    
    def _generate_synthetic_posts(self, query: str, count: int) -> List[Dict]:
        """Generate synthetic Reddit post data for testing."""
        synthetic_posts = []
        subreddits = ['investing', 'stocks', 'SecurityAnalysis', 'ValueInvesting', 'wallstreetbets']
        
        for i in range(count):
            sentiment = np.random.choice(['positive', 'negative', 'neutral'])
            subreddit = np.random.choice(subreddits)
            
            if sentiment == 'positive':
                title = f"DD: Why {query} is undervalued and ready to moon 🚀"
                text = f"After extensive research, I believe {query} is significantly undervalued..."
            elif sentiment == 'negative':
                title = f"Warning: {query} showing major red flags"
                text = f"I've been analyzing {query} and there are some concerning trends..."
            else:
                title = f"Thoughts on {query}? Mixed signals here"
                text = f"What does everyone think about {query}? I'm seeing mixed indicators..."
            
            synthetic_posts.append({
                'id': f"synthetic_{i}",
                'title': title,
                'text': text,
                'score': np.random.randint(-50, 500),
                'upvote_ratio': np.random.uniform(0.5, 0.95),
                'num_comments': np.random.randint(0, 200),
                'created_utc': datetime.now() - timedelta(hours=np.random.randint(1, 168)),
                'subreddit': subreddit,
                'url': f"https://reddit.com/r/{subreddit}/synthetic_{i}",
                'synthetic': True
            })
        
        logger.info(f"Generated {len(synthetic_posts)} synthetic Reddit posts for {query}")
        return synthetic_posts
    
    def analyze_sentiment(self, posts: List[Dict]) -> Dict[str, Any]:
        """Analyze sentiment of Reddit posts."""
        if not posts:
            return {'sentiment_score': 0, 'sentiment_label': 'neutral', 'total_posts': 0}
        
        sentiments = []
        total_score = 0
        total_comments = 0
        
        for post in posts:
            # Combine title and text for analysis
            text = f"{post['title']} {post.get('text', '')}"
            text = self._clean_text(text)
            
            # Analyze sentiment
            blob = TextBlob(text)
            sentiment_score = blob.sentiment.polarity
            sentiments.append(sentiment_score)
            
            total_score += post.get('score', 0)
            total_comments += post.get('num_comments', 0)
        
        avg_sentiment = np.mean(sentiments)
        sentiment_label = self._get_sentiment_label(avg_sentiment)
        
        return {
            'sentiment_score': round(avg_sentiment, 3),
            'sentiment_label': sentiment_label,
            'total_posts': len(posts),
            'total_score': total_score,
            'total_comments': total_comments,
            'sentiment_distribution': {
                'positive': len([s for s in sentiments if s > 0.1]),
                'negative': len([s for s in sentiments if s < -0.1]),
                'neutral': len([s for s in sentiments if -0.1 <= s <= 0.1])
            },
            'subreddit_breakdown': self._get_subreddit_breakdown(posts)
        }
    
    def _clean_text(self, text: str) -> str:
        """Clean Reddit text for sentiment analysis."""
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        # Remove Reddit formatting
        text = re.sub(r'\*\*|__|\*|_|~~|`', '', text)
        # Remove extra whitespace
        text = ' '.join(text.split())
        return text
    
    def _get_sentiment_label(self, score: float) -> str:
        """Convert sentiment score to label."""
        if score > 0.1:
            return 'positive'
        elif score < -0.1:
            return 'negative'
        else:
            return 'neutral'
    
    def _get_subreddit_breakdown(self, posts: List[Dict]) -> Dict[str, int]:
        """Get breakdown of posts by subreddit."""
        breakdown = {}
        for post in posts:
            subreddit = post.get('subreddit', 'unknown')
            breakdown[subreddit] = breakdown.get(subreddit, 0) + 1
        return breakdown


class SocialMediaSentimentManager:
    """Main class for managing social media sentiment analysis."""
    
    def __init__(self):
        """Initialize sentiment analyzers."""
        self.twitter_analyzer = TwitterSentimentAnalyzer()
        self.reddit_analyzer = RedditSentimentAnalyzer()
    
    def get_comprehensive_sentiment(self, symbol: str, company_name: str = None) -> Dict[str, Any]:
        """Get comprehensive sentiment analysis from multiple sources."""
        results = {
            'symbol': symbol,
            'company_name': company_name or symbol,
            'timestamp': datetime.now(),
            'twitter': {},
            'reddit': {},
            'overall': {}
        }
        
        # Twitter analysis
        try:
            twitter_query = f"${symbol} OR {company_name or symbol}"
            tweets = self.twitter_analyzer.search_tweets(twitter_query, max_results=100)
            results['twitter'] = self.twitter_analyzer.analyze_sentiment(tweets)
            results['twitter']['source'] = 'Twitter'
        except Exception as e:
            logger.error(f"Twitter analysis failed: {e}")
            results['twitter'] = {'error': str(e)}
        
        # Reddit analysis
        try:
            reddit_query = f"{symbol} {company_name or ''}"
            posts = self.reddit_analyzer.search_posts(reddit_query, limit=50)
            results['reddit'] = self.reddit_analyzer.analyze_sentiment(posts)
            results['reddit']['source'] = 'Reddit'
        except Exception as e:
            logger.error(f"Reddit analysis failed: {e}")
            results['reddit'] = {'error': str(e)}
        
        # Calculate overall sentiment
        results['overall'] = self._calculate_overall_sentiment(results)
        
        return results
    
    def _calculate_overall_sentiment(self, results: Dict) -> Dict[str, Any]:
        """Calculate overall sentiment from multiple sources."""
        sentiments = []
        weights = []
        
        # Twitter sentiment (weight by engagement)
        if 'sentiment_score' in results['twitter']:
            twitter_weight = results['twitter'].get('total_engagement', 100)
            sentiments.append(results['twitter']['sentiment_score'])
            weights.append(twitter_weight)
        
        # Reddit sentiment (weight by score and comments)
        if 'sentiment_score' in results['reddit']:
            reddit_weight = (results['reddit'].get('total_score', 50) + 
                           results['reddit'].get('total_comments', 50))
            sentiments.append(results['reddit']['sentiment_score'])
            weights.append(reddit_weight)
        
        if not sentiments:
            return {'sentiment_score': 0, 'sentiment_label': 'neutral', 'confidence': 0}
        
        # Weighted average
        if weights:
            overall_sentiment = np.average(sentiments, weights=weights)
        else:
            overall_sentiment = np.mean(sentiments)
        
        # Calculate confidence based on data availability
        confidence = min(1.0, (sum(weights) / 1000) if weights else 0.1)
        
        return {
            'sentiment_score': round(overall_sentiment, 3),
            'sentiment_label': self._get_sentiment_label(overall_sentiment),
            'confidence': round(confidence, 2),
            'sources_analyzed': len(sentiments)
        }
    
    def _get_sentiment_label(self, score: float) -> str:
        """Convert sentiment score to label."""
        if score > 0.1:
            return 'positive'
        elif score < -0.1:
            return 'negative'
        else:
            return 'neutral'


# Test the module
if __name__ == "__main__":
    manager = SocialMediaSentimentManager()
    
    # Test with AAPL
    print("Testing social media sentiment analysis...")
    results = manager.get_comprehensive_sentiment("AAPL", "Apple")
    
    print(f"\nResults for AAPL:")
    print(f"Overall Sentiment: {results['overall']['sentiment_label']} ({results['overall']['sentiment_score']})")
    print(f"Confidence: {results['overall']['confidence']}")
    print(f"Twitter: {results['twitter'].get('sentiment_label', 'N/A')} ({results['twitter'].get('total_tweets', 0)} tweets)")
    print(f"Reddit: {results['reddit'].get('sentiment_label', 'N/A')} ({results['reddit'].get('total_posts', 0)} posts)")

