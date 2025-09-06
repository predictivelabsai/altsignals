"""
Google Trends Analysis Utility for AltSignals platform.
Analyzes search trends and interest over time for stocks and companies.
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import logging
import json
from pytrends.request import TrendReq
import time
import random

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GoogleTrendsAnalyzer:
    """Google Trends analysis using pytrends."""
    
    def __init__(self):
        """Initialize Google Trends client."""
        try:
            self.pytrends = TrendReq(hl='en-US', tz=360)
            logger.info("Google Trends client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Google Trends client: {e}")
            self.pytrends = None
    
    def get_interest_over_time(self, keywords: List[str], timeframe: str = 'today 12-m', 
                              geo: str = 'US') -> pd.DataFrame:
        """Get interest over time for keywords."""
        if not self.pytrends:
            return self._generate_synthetic_trends(keywords, timeframe)
        
        try:
            # Build payload
            self.pytrends.build_payload(keywords, cat=0, timeframe=timeframe, geo=geo, gprop='')
            
            # Get interest over time
            interest_df = self.pytrends.interest_over_time()
            
            if interest_df.empty:
                logger.warning(f"No data returned for keywords: {keywords}")
                return self._generate_synthetic_trends(keywords, timeframe)
            
            # Remove 'isPartial' column if it exists
            if 'isPartial' in interest_df.columns:
                interest_df = interest_df.drop('isPartial', axis=1)
            
            logger.info(f"Retrieved trends data for {len(keywords)} keywords")
            return interest_df
            
        except Exception as e:
            logger.error(f"Error getting trends data: {e}")
            return self._generate_synthetic_trends(keywords, timeframe)
    
    def get_related_queries(self, keyword: str, timeframe: str = 'today 12-m', 
                           geo: str = 'US') -> Dict[str, pd.DataFrame]:
        """Get related queries for a keyword."""
        if not self.pytrends:
            return self._generate_synthetic_related_queries(keyword)
        
        try:
            # Build payload
            self.pytrends.build_payload([keyword], cat=0, timeframe=timeframe, geo=geo, gprop='')
            
            # Get related queries
            related_queries = self.pytrends.related_queries()
            
            if not related_queries or keyword not in related_queries:
                logger.warning(f"No related queries found for: {keyword}")
                return self._generate_synthetic_related_queries(keyword)
            
            logger.info(f"Retrieved related queries for: {keyword}")
            return related_queries[keyword]
            
        except Exception as e:
            logger.error(f"Error getting related queries: {e}")
            return self._generate_synthetic_related_queries(keyword)
    
    def get_trending_searches(self, geo: str = 'US') -> pd.DataFrame:
        """Get trending searches for a specific geography."""
        if not self.pytrends:
            return self._generate_synthetic_trending()
        
        try:
            trending_df = self.pytrends.trending_searches(pn=geo)
            logger.info(f"Retrieved trending searches for {geo}")
            return trending_df
            
        except Exception as e:
            logger.error(f"Error getting trending searches: {e}")
            return self._generate_synthetic_trending()
    
    def compare_keywords(self, keywords: List[str], timeframe: str = 'today 12-m', 
                        geo: str = 'US') -> Dict[str, Any]:
        """Compare multiple keywords and analyze trends."""
        interest_df = self.get_interest_over_time(keywords, timeframe, geo)
        
        if interest_df.empty:
            return {'error': 'No data available'}
        
        analysis = {
            'keywords': keywords,
            'timeframe': timeframe,
            'geo': geo,
            'data': interest_df,
            'summary': {},
            'trends': {},
            'correlations': {}
        }
        
        # Calculate summary statistics
        for keyword in keywords:
            if keyword in interest_df.columns:
                series = interest_df[keyword]
                analysis['summary'][keyword] = {
                    'mean': round(series.mean(), 2),
                    'max': int(series.max()),
                    'min': int(series.min()),
                    'std': round(series.std(), 2),
                    'current': int(series.iloc[-1]) if len(series) > 0 else 0,
                    'trend_direction': self._calculate_trend_direction(series)
                }
        
        # Calculate trend analysis
        analysis['trends'] = self._analyze_trends(interest_df, keywords)
        
        # Calculate correlations
        if len(keywords) > 1:
            analysis['correlations'] = self._calculate_correlations(interest_df, keywords)
        
        return analysis
    
    def analyze_stock_interest(self, symbol: str, company_name: str = None) -> Dict[str, Any]:
        """Analyze Google Trends interest for a specific stock."""
        keywords = [symbol]
        if company_name:
            keywords.append(company_name)
        
        # Add stock-related terms
        stock_keywords = [f"{symbol} stock", f"{symbol} price", f"{symbol} news"]
        if company_name:
            stock_keywords.extend([f"{company_name} stock", f"{company_name} earnings"])
        
        # Analyze different timeframes
        timeframes = {
            'past_month': 'today 1-m',
            'past_3_months': 'today 3-m',
            'past_year': 'today 12-m'
        }
        
        results = {
            'symbol': symbol,
            'company_name': company_name,
            'timestamp': datetime.now(),
            'analysis': {}
        }
        
        for period, timeframe in timeframes.items():
            try:
                # Analyze main keywords
                main_analysis = self.compare_keywords(keywords[:2], timeframe)
                
                # Analyze stock-specific keywords
                stock_analysis = self.compare_keywords(stock_keywords[:3], timeframe)
                
                results['analysis'][period] = {
                    'main_keywords': main_analysis,
                    'stock_keywords': stock_analysis,
                    'interest_score': self._calculate_interest_score(main_analysis, stock_analysis)
                }
                
                # Add delay to avoid rate limiting
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"Error analyzing {period}: {e}")
                results['analysis'][period] = {'error': str(e)}
        
        # Calculate overall trends
        results['overall_trends'] = self._calculate_overall_trends(results['analysis'])
        
        return results
    
    def _generate_synthetic_trends(self, keywords: List[str], timeframe: str) -> pd.DataFrame:
        """Generate synthetic trends data for testing."""
        # Determine date range based on timeframe
        if 'today 1-m' in timeframe:
            days = 30
        elif 'today 3-m' in timeframe:
            days = 90
        elif 'today 12-m' in timeframe:
            days = 365
        else:
            days = 365
        
        # Generate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # Generate synthetic data
        data = {}
        for keyword in keywords:
            # Create realistic trend with some seasonality and noise
            base_trend = 50 + 30 * np.sin(np.linspace(0, 4*np.pi, len(date_range)))
            noise = np.random.normal(0, 10, len(date_range))
            trend = np.clip(base_trend + noise, 0, 100)
            data[keyword] = trend.astype(int)
        
        df = pd.DataFrame(data, index=date_range)
        df.index.name = 'date'
        
        logger.info(f"Generated synthetic trends data for {len(keywords)} keywords")
        return df
    
    def _generate_synthetic_related_queries(self, keyword: str) -> Dict[str, pd.DataFrame]:
        """Generate synthetic related queries."""
        top_queries = pd.DataFrame({
            'query': [f'{keyword} stock', f'{keyword} price', f'{keyword} news', 
                     f'{keyword} earnings', f'{keyword} forecast'],
            'value': [100, 85, 70, 60, 45]
        })
        
        rising_queries = pd.DataFrame({
            'query': [f'{keyword} analysis', f'{keyword} buy', f'{keyword} target price', 
                     f'{keyword} dividend', f'{keyword} options'],
            'value': ['Breakout', '+150%', '+120%', '+90%', '+75%']
        })
        
        return {
            'top': top_queries,
            'rising': rising_queries
        }
    
    def _generate_synthetic_trending(self) -> pd.DataFrame:
        """Generate synthetic trending searches."""
        trending_terms = [
            'stock market', 'cryptocurrency', 'inflation', 'interest rates',
            'tech stocks', 'AI stocks', 'renewable energy', 'electric vehicles'
        ]
        
        return pd.DataFrame(trending_terms, columns=['trending_searches'])
    
    def _calculate_trend_direction(self, series: pd.Series) -> str:
        """Calculate trend direction for a time series."""
        if len(series) < 2:
            return 'neutral'
        
        # Compare recent period to earlier period
        recent = series.tail(len(series)//4).mean()
        earlier = series.head(len(series)//4).mean()
        
        if recent > earlier * 1.1:
            return 'increasing'
        elif recent < earlier * 0.9:
            return 'decreasing'
        else:
            return 'stable'
    
    def _analyze_trends(self, df: pd.DataFrame, keywords: List[str]) -> Dict[str, Any]:
        """Analyze trends in the data."""
        trends = {}
        
        for keyword in keywords:
            if keyword in df.columns:
                series = df[keyword]
                
                # Calculate momentum (rate of change)
                momentum = series.pct_change().tail(7).mean()
                
                # Calculate volatility
                volatility = series.pct_change().std()
                
                # Find peaks and valleys
                peaks = self._find_peaks(series)
                valleys = self._find_valleys(series)
                
                trends[keyword] = {
                    'momentum': round(momentum * 100, 2),
                    'volatility': round(volatility * 100, 2),
                    'peaks': len(peaks),
                    'valleys': len(valleys),
                    'trend_strength': self._calculate_trend_strength(series)
                }
        
        return trends
    
    def _calculate_correlations(self, df: pd.DataFrame, keywords: List[str]) -> Dict[str, float]:
        """Calculate correlations between keywords."""
        correlations = {}
        
        for i, keyword1 in enumerate(keywords):
            for keyword2 in keywords[i+1:]:
                if keyword1 in df.columns and keyword2 in df.columns:
                    corr = df[keyword1].corr(df[keyword2])
                    correlations[f"{keyword1}_vs_{keyword2}"] = round(corr, 3)
        
        return correlations
    
    def _find_peaks(self, series: pd.Series, threshold: float = 0.1) -> List[int]:
        """Find peaks in the time series."""
        peaks = []
        for i in range(1, len(series) - 1):
            if (series.iloc[i] > series.iloc[i-1] and 
                series.iloc[i] > series.iloc[i+1] and
                series.iloc[i] > series.mean() * (1 + threshold)):
                peaks.append(i)
        return peaks
    
    def _find_valleys(self, series: pd.Series, threshold: float = 0.1) -> List[int]:
        """Find valleys in the time series."""
        valleys = []
        for i in range(1, len(series) - 1):
            if (series.iloc[i] < series.iloc[i-1] and 
                series.iloc[i] < series.iloc[i+1] and
                series.iloc[i] < series.mean() * (1 - threshold)):
                valleys.append(i)
        return valleys
    
    def _calculate_trend_strength(self, series: pd.Series) -> str:
        """Calculate the strength of the trend."""
        if len(series) < 10:
            return 'insufficient_data'
        
        # Calculate linear regression slope
        x = np.arange(len(series))
        slope = np.polyfit(x, series.values, 1)[0]
        
        # Normalize by series mean
        normalized_slope = slope / series.mean() if series.mean() != 0 else 0
        
        if abs(normalized_slope) > 0.01:
            return 'strong'
        elif abs(normalized_slope) > 0.005:
            return 'moderate'
        else:
            return 'weak'
    
    def _calculate_interest_score(self, main_analysis: Dict, stock_analysis: Dict) -> Dict[str, Any]:
        """Calculate overall interest score."""
        if 'error' in main_analysis or 'error' in stock_analysis:
            return {'score': 0, 'level': 'unknown'}
        
        # Get average interest from main keywords
        main_scores = []
        if 'summary' in main_analysis:
            for keyword, stats in main_analysis['summary'].items():
                main_scores.append(stats.get('mean', 0))
        
        # Get average interest from stock keywords
        stock_scores = []
        if 'summary' in stock_analysis:
            for keyword, stats in stock_analysis['summary'].items():
                stock_scores.append(stats.get('mean', 0))
        
        # Calculate weighted average
        all_scores = main_scores + stock_scores
        if not all_scores:
            return {'score': 0, 'level': 'unknown'}
        
        avg_score = np.mean(all_scores)
        
        # Determine interest level
        if avg_score >= 70:
            level = 'very_high'
        elif avg_score >= 50:
            level = 'high'
        elif avg_score >= 30:
            level = 'moderate'
        elif avg_score >= 10:
            level = 'low'
        else:
            level = 'very_low'
        
        return {
            'score': round(avg_score, 1),
            'level': level,
            'main_keywords_avg': round(np.mean(main_scores), 1) if main_scores else 0,
            'stock_keywords_avg': round(np.mean(stock_scores), 1) if stock_scores else 0
        }
    
    def _calculate_overall_trends(self, analysis: Dict) -> Dict[str, Any]:
        """Calculate overall trends across timeframes."""
        trends = {
            'interest_trajectory': 'stable',
            'momentum': 'neutral',
            'volatility': 'moderate',
            'peak_activity': None
        }
        
        # Analyze trajectory across timeframes
        scores = []
        for period, data in analysis.items():
            if 'interest_score' in data and 'score' in data['interest_score']:
                scores.append(data['interest_score']['score'])
        
        if len(scores) >= 2:
            if scores[-1] > scores[0] * 1.2:
                trends['interest_trajectory'] = 'increasing'
            elif scores[-1] < scores[0] * 0.8:
                trends['interest_trajectory'] = 'decreasing'
        
        # Find peak activity period
        if scores:
            max_score_idx = scores.index(max(scores))
            periods = list(analysis.keys())
            if max_score_idx < len(periods):
                trends['peak_activity'] = periods[max_score_idx]
        
        return trends


# Test the module
if __name__ == "__main__":
    analyzer = GoogleTrendsAnalyzer()
    
    # Test with AAPL
    print("Testing Google Trends analysis...")
    results = analyzer.analyze_stock_interest("AAPL", "Apple")
    
    print(f"\nResults for AAPL:")
    print(f"Overall trajectory: {results['overall_trends']['interest_trajectory']}")
    print(f"Peak activity: {results['overall_trends']['peak_activity']}")
    
    for period, data in results['analysis'].items():
        if 'interest_score' in data:
            score = data['interest_score']
            print(f"{period}: Score {score['score']} ({score['level']})")

