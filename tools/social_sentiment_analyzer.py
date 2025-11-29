#!/usr/bin/env python3
"""
Social Sentiment Analyzer

This tool monitors and analyzes social media sentiment across multiple platforms
to identify trading opportunities based on retail investor sentiment and momentum.
"""

import asyncio
import logging
import re
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json
import numpy as np
import pandas as pd
from collections import defaultdict, Counter
import aiohttp
import time

# CrewAI imports
from crewai_tools import BaseTool
from pydantic import BaseModel, Field

# Local imports
from ..core.data_manager import UnifiedDataManager
from ..core.config_manager import ConfigManager


class SentimentSource(Enum):
    """Social media sentiment sources"""
    REDDIT_WSB = "reddit_wsb"
    REDDIT_STOCKS = "reddit_stocks"
    REDDIT_INVESTING = "reddit_investing"
    TWITTER = "twitter"
    STOCKTWITS = "stocktwits"
    DISCORD = "discord"
    TELEGRAM = "telegram"
    YOUTUBE = "youtube"


class SentimentSignal(Enum):
    """Sentiment signal types"""
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"
    EXTREME_BULLISH = "extreme_bullish"
    EXTREME_BEARISH = "extreme_bearish"


@dataclass
class SentimentData:
    """Social sentiment data structure"""
    symbol: str
    source: SentimentSource
    timestamp: datetime
    sentiment_score: float  # -1 to 1
    confidence: float  # 0 to 1
    mention_count: int
    engagement_score: float
    trending_score: float
    posts: List[Dict[str, Any]] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    influencer_mentions: List[str] = field(default_factory=list)


@dataclass
class SentimentAlert:
    """Sentiment-based trading alert"""
    symbol: str
    signal: SentimentSignal
    strength: float
    timestamp: datetime
    sources: List[SentimentSource]
    description: str
    supporting_data: Dict[str, Any]
    confidence: float
    expected_duration: str


class SocialSentimentAnalyzerTool(BaseTool):
    """Social sentiment analyzer for trading signals"""
    
    name: str = "social_sentiment_analyzer"
    description: str = "Monitors and analyzes social media sentiment across multiple platforms to identify trading opportunities based on retail sentiment"
    
    def __init__(self, config_manager: ConfigManager, data_manager: UnifiedDataManager):
        super().__init__()
        self.config_manager = config_manager
        self.data_manager = data_manager
        self.logger = logging.getLogger(__name__)
        
        # Load configuration
        self.config = self._load_config()
        
        # Sentiment storage
        self.sentiment_history: Dict[str, List[SentimentData]] = defaultdict(list)
        self.alerts: List[SentimentAlert] = []
        
        # API configurations
        self.api_configs = self._load_api_configs()
        
        # Sentiment keywords and patterns
        self.bullish_keywords = [
            'moon', 'rocket', 'diamond hands', 'hodl', 'buy the dip', 'bullish',
            'calls', 'long', 'pump', 'squeeze', 'breakout', 'rally', 'surge',
            'bull run', 'to the moon', 'lambo', 'tendies', 'yolo', 'ape strong'
        ]
        
        self.bearish_keywords = [
            'crash', 'dump', 'puts', 'short', 'bearish', 'sell', 'drop',
            'fall', 'decline', 'red', 'blood', 'panic', 'bubble', 'overvalued',
            'correction', 'recession', 'bear market', 'dead cat bounce'
        ]
        
        # Emoji sentiment mapping
        self.emoji_sentiment = {
            '🚀': 0.8, '🌙': 0.7, '💎': 0.6, '🦍': 0.5, '💪': 0.4,
            '📈': 0.6, '🔥': 0.5, '💰': 0.4, '🎯': 0.3, '✅': 0.2,
            '📉': -0.6, '💀': -0.8, '🩸': -0.7, '😭': -0.5, '😱': -0.6,
            '🔴': -0.4, '⚠️': -0.3, '💸': -0.5, '🤡': -0.4, '🐻': -0.6
        }
        
        # Rate limiting
        self.rate_limits = {
            'reddit': {'calls': 0, 'reset_time': 0, 'limit': 60},
            'twitter': {'calls': 0, 'reset_time': 0, 'limit': 300},
            'stocktwits': {'calls': 0, 'reset_time': 0, 'limit': 200}
        }
    
    def _load_config(self) -> Dict[str, Any]:
        """Load sentiment analyzer configuration"""
        try:
            config = self.config_manager.get_config('sentiment_analyzer')
            return config
        except Exception as e:
            self.logger.warning(f"Could not load sentiment analyzer config: {e}")
            return {
                'update_interval': 300,  # 5 minutes
                'symbols': ['SPY', 'QQQ', 'AAPL', 'TSLA', 'NVDA', 'MSFT', 'GOOGL', 'AMZN'],
                'sentiment_threshold': 0.3,
                'mention_threshold': 10,
                'sources': ['reddit_wsb', 'reddit_stocks', 'stocktwits'],
                'max_history_days': 7
            }
    
    def _load_api_configs(self) -> Dict[str, Dict[str, str]]:
        """Load API configurations for social media platforms"""
        try:
            return {
                'reddit': {
                    'client_id': self.config_manager.get_config('reddit_api').get('client_id', ''),
                    'client_secret': self.config_manager.get_config('reddit_api').get('client_secret', ''),
                    'user_agent': 'TradingBot/1.0'
                },
                'twitter': {
                    'bearer_token': self.config_manager.get_config('twitter_api').get('bearer_token', ''),
                    'api_key': self.config_manager.get_config('twitter_api').get('api_key', ''),
                    'api_secret': self.config_manager.get_config('twitter_api').get('api_secret', '')
                },
                'stocktwits': {
                    'access_token': self.config_manager.get_config('stocktwits_api').get('access_token', '')
                }
            }
        except Exception as e:
            self.logger.warning(f"Could not load API configs: {e}")
            return {}
    
    def _run(self, action: str, parameters: Dict[str, Any] = None) -> str:
        """Synchronous sentiment analysis execution"""
        return asyncio.run(self._arun(action, parameters))
    
    async def _arun(self, action: str, parameters: Dict[str, Any] = None) -> str:
        """Asynchronous sentiment analysis execution"""
        try:
            parameters = parameters or {}
            
            if action == 'analyze_sentiment':
                return await self._analyze_sentiment(parameters)
            elif action == 'get_sentiment_data':
                return await self._get_sentiment_data(parameters)
            elif action == 'get_alerts':
                return await self._get_alerts(parameters)
            elif action == 'monitor_symbol':
                return await self._monitor_symbol(parameters)
            elif action == 'get_trending':
                return await self._get_trending_symbols()
            elif action == 'analyze_influencers':
                return await self._analyze_influencers(parameters)
            else:
                return json.dumps({
                    'error': f'Unknown action: {action}',
                    'available_actions': [
                        'analyze_sentiment', 'get_sentiment_data', 'get_alerts',
                        'monitor_symbol', 'get_trending', 'analyze_influencers'
                    ]
                })
                
        except Exception as e:
            self.logger.error(f"Sentiment analysis failed: {e}")
            return json.dumps({'error': str(e)})
    
    async def _analyze_sentiment(self, parameters: Dict[str, Any]) -> str:
        """Analyze sentiment for specified symbols"""
        symbols = parameters.get('symbols', self.config['symbols'])
        sources = parameters.get('sources', self.config['sources'])
        
        if isinstance(symbols, str):
            symbols = [symbols]
        
        results = {}
        
        for symbol in symbols:
            symbol_results = {}
            
            for source_name in sources:
                try:
                    source = SentimentSource(source_name)
                    sentiment_data = await self._fetch_sentiment_data(symbol, source)
                    
                    if sentiment_data:
                        symbol_results[source_name] = {
                            'sentiment_score': sentiment_data.sentiment_score,
                            'confidence': sentiment_data.confidence,
                            'mention_count': sentiment_data.mention_count,
                            'engagement_score': sentiment_data.engagement_score,
                            'trending_score': sentiment_data.trending_score,
                            'timestamp': sentiment_data.timestamp.isoformat()
                        }
                        
                        # Store in history
                        self.sentiment_history[symbol].append(sentiment_data)
                        
                        # Check for alerts
                        await self._check_sentiment_alerts(symbol, sentiment_data)
                
                except Exception as e:
                    self.logger.error(f"Failed to analyze {source_name} for {symbol}: {e}")
                    symbol_results[source_name] = {'error': str(e)}
            
            results[symbol] = symbol_results
        
        # Clean old history
        self._clean_old_history()
        
        return json.dumps(results, indent=2)
    
    async def _fetch_sentiment_data(self, symbol: str, source: SentimentSource) -> Optional[SentimentData]:
        """Fetch sentiment data from a specific source"""
        if source == SentimentSource.REDDIT_WSB:
            return await self._fetch_reddit_sentiment(symbol, 'wallstreetbets')
        elif source == SentimentSource.REDDIT_STOCKS:
            return await self._fetch_reddit_sentiment(symbol, 'stocks')
        elif source == SentimentSource.REDDIT_INVESTING:
            return await self._fetch_reddit_sentiment(symbol, 'investing')
        elif source == SentimentSource.TWITTER:
            return await self._fetch_twitter_sentiment(symbol)
        elif source == SentimentSource.STOCKTWITS:
            return await self._fetch_stocktwits_sentiment(symbol)
        else:
            self.logger.warning(f"Unsupported sentiment source: {source}")
            return None
    
    async def _fetch_reddit_sentiment(self, symbol: str, subreddit: str) -> Optional[SentimentData]:
        """Fetch sentiment data from Reddit"""
        if not self._check_rate_limit('reddit'):
            return None
        
        try:
            # Simulate Reddit API call (replace with actual Reddit API)
            posts = await self._simulate_reddit_data(symbol, subreddit)
            
            if not posts:
                return None
            
            # Analyze sentiment
            sentiment_scores = []
            mention_count = len(posts)
            total_engagement = 0
            keywords_found = []
            
            for post in posts:
                text = f"{post.get('title', '')} {post.get('body', '')}"
                score = self._calculate_text_sentiment(text)
                sentiment_scores.append(score)
                
                total_engagement += post.get('score', 0) + post.get('num_comments', 0)
                keywords_found.extend(self._extract_keywords(text))
            
            avg_sentiment = np.mean(sentiment_scores) if sentiment_scores else 0
            confidence = min(1.0, mention_count / 50)  # Higher confidence with more mentions
            engagement_score = total_engagement / max(mention_count, 1)
            
            # Calculate trending score based on recent activity
            trending_score = self._calculate_trending_score(posts)
            
            return SentimentData(
                symbol=symbol,
                source=SentimentSource.REDDIT_WSB if subreddit == 'wallstreetbets' else SentimentSource.REDDIT_STOCKS,
                timestamp=datetime.now(),
                sentiment_score=avg_sentiment,
                confidence=confidence,
                mention_count=mention_count,
                engagement_score=engagement_score,
                trending_score=trending_score,
                posts=posts[:5],  # Store top 5 posts
                keywords=list(set(keywords_found))
            )
        
        except Exception as e:
            self.logger.error(f"Reddit sentiment fetch failed: {e}")
            return None
    
    async def _fetch_twitter_sentiment(self, symbol: str) -> Optional[SentimentData]:
        """Fetch sentiment data from Twitter"""
        if not self._check_rate_limit('twitter'):
            return None
        
        try:
            # Simulate Twitter API call (replace with actual Twitter API v2)
            tweets = await self._simulate_twitter_data(symbol)
            
            if not tweets:
                return None
            
            # Analyze sentiment
            sentiment_scores = []
            mention_count = len(tweets)
            total_engagement = 0
            keywords_found = []
            influencer_mentions = []
            
            for tweet in tweets:
                text = tweet.get('text', '')
                score = self._calculate_text_sentiment(text)
                sentiment_scores.append(score)
                
                total_engagement += tweet.get('retweet_count', 0) + tweet.get('like_count', 0)
                keywords_found.extend(self._extract_keywords(text))
                
                # Check for influencer accounts
                if tweet.get('user', {}).get('followers_count', 0) > 10000:
                    influencer_mentions.append(tweet.get('user', {}).get('username', ''))
            
            avg_sentiment = np.mean(sentiment_scores) if sentiment_scores else 0
            confidence = min(1.0, mention_count / 100)
            engagement_score = total_engagement / max(mention_count, 1)
            trending_score = self._calculate_trending_score(tweets)
            
            return SentimentData(
                symbol=symbol,
                source=SentimentSource.TWITTER,
                timestamp=datetime.now(),
                sentiment_score=avg_sentiment,
                confidence=confidence,
                mention_count=mention_count,
                engagement_score=engagement_score,
                trending_score=trending_score,
                posts=tweets[:5],
                keywords=list(set(keywords_found)),
                influencer_mentions=list(set(influencer_mentions))
            )
        
        except Exception as e:
            self.logger.error(f"Twitter sentiment fetch failed: {e}")
            return None
    
    async def _fetch_stocktwits_sentiment(self, symbol: str) -> Optional[SentimentData]:
        """Fetch sentiment data from StockTwits"""
        if not self._check_rate_limit('stocktwits'):
            return None
        
        try:
            # Simulate StockTwits API call
            messages = await self._simulate_stocktwits_data(symbol)
            
            if not messages:
                return None
            
            # StockTwits provides sentiment labels
            bullish_count = 0
            bearish_count = 0
            total_engagement = 0
            
            for message in messages:
                sentiment = message.get('entities', {}).get('sentiment', {})
                if sentiment.get('basic') == 'Bullish':
                    bullish_count += 1
                elif sentiment.get('basic') == 'Bearish':
                    bearish_count += 1
                
                total_engagement += message.get('likes', {}).get('total', 0)
            
            total_messages = len(messages)
            if total_messages > 0:
                sentiment_score = (bullish_count - bearish_count) / total_messages
            else:
                sentiment_score = 0
            
            confidence = min(1.0, total_messages / 30)
            engagement_score = total_engagement / max(total_messages, 1)
            trending_score = self._calculate_trending_score(messages)
            
            return SentimentData(
                symbol=symbol,
                source=SentimentSource.STOCKTWITS,
                timestamp=datetime.now(),
                sentiment_score=sentiment_score,
                confidence=confidence,
                mention_count=total_messages,
                engagement_score=engagement_score,
                trending_score=trending_score,
                posts=messages[:5]
            )
        
        except Exception as e:
            self.logger.error(f"StockTwits sentiment fetch failed: {e}")
            return None
    
    def _calculate_text_sentiment(self, text: str) -> float:
        """Calculate sentiment score from text"""
        text_lower = text.lower()
        
        # Count bullish and bearish keywords
        bullish_score = sum(1 for keyword in self.bullish_keywords if keyword in text_lower)
        bearish_score = sum(1 for keyword in self.bearish_keywords if keyword in text_lower)
        
        # Count emojis
        emoji_score = sum(self.emoji_sentiment.get(char, 0) for char in text)
        
        # Calculate final score
        keyword_score = (bullish_score - bearish_score) / max(len(text.split()), 1)
        total_score = (keyword_score + emoji_score) / 2
        
        # Normalize to -1 to 1 range
        return max(-1, min(1, total_score))
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract relevant keywords from text"""
        text_lower = text.lower()
        found_keywords = []
        
        for keyword in self.bullish_keywords + self.bearish_keywords:
            if keyword in text_lower:
                found_keywords.append(keyword)
        
        return found_keywords
    
    def _calculate_trending_score(self, posts: List[Dict[str, Any]]) -> float:
        """Calculate trending score based on post timing and engagement"""
        if not posts:
            return 0
        
        now = datetime.now()
        recent_posts = 0
        total_engagement = 0
        
        for post in posts:
            # Simulate post age (in real implementation, parse actual timestamps)
            post_age_hours = np.random.uniform(0, 24)
            
            if post_age_hours < 6:  # Recent posts
                recent_posts += 1
            
            engagement = post.get('score', 0) + post.get('num_comments', 0)
            engagement += post.get('retweet_count', 0) + post.get('like_count', 0)
            total_engagement += engagement
        
        # Trending score based on recent activity and engagement
        recency_score = recent_posts / len(posts)
        engagement_score = min(1.0, total_engagement / (len(posts) * 100))
        
        return (recency_score + engagement_score) / 2
    
    def _check_rate_limit(self, service: str) -> bool:
        """Check if we can make API calls within rate limits"""
        current_time = time.time()
        rate_info = self.rate_limits.get(service, {})
        
        # Reset counter if hour has passed
        if current_time - rate_info.get('reset_time', 0) > 3600:
            rate_info['calls'] = 0
            rate_info['reset_time'] = current_time
        
        # Check if under limit
        if rate_info.get('calls', 0) < rate_info.get('limit', 100):
            rate_info['calls'] = rate_info.get('calls', 0) + 1
            return True
        
        return False
    
    async def _check_sentiment_alerts(self, symbol: str, sentiment_data: SentimentData):
        """Check if sentiment data triggers any alerts"""
        # Get recent sentiment history
        recent_data = [
            data for data in self.sentiment_history[symbol]
            if (datetime.now() - data.timestamp).seconds < 3600  # Last hour
        ]
        
        if len(recent_data) < 2:
            return
        
        # Check for sentiment shifts
        current_sentiment = sentiment_data.sentiment_score
        avg_recent_sentiment = np.mean([data.sentiment_score for data in recent_data[:-1]])
        
        sentiment_change = abs(current_sentiment - avg_recent_sentiment)
        
        # Generate alerts based on conditions
        if sentiment_change > self.config['sentiment_threshold']:
            if current_sentiment > 0.5:
                signal = SentimentSignal.EXTREME_BULLISH
            elif current_sentiment > 0.2:
                signal = SentimentSignal.BULLISH
            elif current_sentiment < -0.5:
                signal = SentimentSignal.EXTREME_BEARISH
            elif current_sentiment < -0.2:
                signal = SentimentSignal.BEARISH
            else:
                signal = SentimentSignal.NEUTRAL
            
            if signal != SentimentSignal.NEUTRAL:
                alert = SentimentAlert(
                    symbol=symbol,
                    signal=signal,
                    strength=sentiment_change,
                    timestamp=datetime.now(),
                    sources=[sentiment_data.source],
                    description=f"Sentiment shift detected: {signal.value} signal",
                    supporting_data={
                        'current_sentiment': current_sentiment,
                        'previous_sentiment': avg_recent_sentiment,
                        'mention_count': sentiment_data.mention_count,
                        'confidence': sentiment_data.confidence
                    },
                    confidence=sentiment_data.confidence,
                    expected_duration="1-6 hours"
                )
                
                self.alerts.append(alert)
                self.logger.info(f"Sentiment alert generated for {symbol}: {signal.value}")
    
    async def _get_sentiment_data(self, parameters: Dict[str, Any]) -> str:
        """Get historical sentiment data"""
        symbol = parameters.get('symbol')
        hours_back = parameters.get('hours_back', 24)
        
        if not symbol:
            return json.dumps({'error': 'Symbol parameter required'})
        
        cutoff_time = datetime.now() - timedelta(hours=hours_back)
        
        relevant_data = [
            data for data in self.sentiment_history.get(symbol, [])
            if data.timestamp > cutoff_time
        ]
        
        if not relevant_data:
            return json.dumps({'message': f'No sentiment data found for {symbol}'})
        
        # Convert to JSON-serializable format
        data_points = []
        for data in relevant_data:
            data_points.append({
                'timestamp': data.timestamp.isoformat(),
                'source': data.source.value,
                'sentiment_score': data.sentiment_score,
                'confidence': data.confidence,
                'mention_count': data.mention_count,
                'engagement_score': data.engagement_score,
                'trending_score': data.trending_score,
                'keywords': data.keywords
            })
        
        # Calculate summary statistics
        sentiment_scores = [data.sentiment_score for data in relevant_data]
        summary = {
            'avg_sentiment': np.mean(sentiment_scores),
            'sentiment_volatility': np.std(sentiment_scores),
            'total_mentions': sum(data.mention_count for data in relevant_data),
            'avg_confidence': np.mean([data.confidence for data in relevant_data]),
            'data_points': len(relevant_data)
        }
        
        return json.dumps({
            'symbol': symbol,
            'time_range_hours': hours_back,
            'summary': summary,
            'data_points': data_points
        }, indent=2)
    
    async def _get_alerts(self, parameters: Dict[str, Any]) -> str:
        """Get sentiment alerts"""
        limit = parameters.get('limit', 20)
        symbol_filter = parameters.get('symbol')
        
        filtered_alerts = self.alerts
        
        if symbol_filter:
            filtered_alerts = [
                alert for alert in filtered_alerts
                if alert.symbol == symbol_filter.upper()
            ]
        
        # Sort by timestamp (newest first) and limit
        filtered_alerts.sort(key=lambda x: x.timestamp, reverse=True)
        filtered_alerts = filtered_alerts[:limit]
        
        # Convert to JSON-serializable format
        alerts_data = []
        for alert in filtered_alerts:
            alerts_data.append({
                'symbol': alert.symbol,
                'signal': alert.signal.value,
                'strength': alert.strength,
                'timestamp': alert.timestamp.isoformat(),
                'sources': [source.value for source in alert.sources],
                'description': alert.description,
                'confidence': alert.confidence,
                'expected_duration': alert.expected_duration,
                'supporting_data': alert.supporting_data
            })
        
        return json.dumps({
            'alerts': alerts_data,
            'total_alerts': len(self.alerts)
        }, indent=2)
    
    async def _monitor_symbol(self, parameters: Dict[str, Any]) -> str:
        """Monitor a specific symbol for sentiment changes"""
        symbol = parameters.get('symbol')
        if not symbol:
            return json.dumps({'error': 'Symbol parameter required'})
        
        # Add to monitoring list if not already present
        if symbol not in self.config['symbols']:
            self.config['symbols'].append(symbol)
        
        # Perform immediate analysis
        result = await self._analyze_sentiment({'symbols': [symbol]})
        
        return json.dumps({
            'message': f'Now monitoring {symbol} for sentiment changes',
            'immediate_analysis': json.loads(result)
        })
    
    async def _get_trending_symbols(self) -> str:
        """Get trending symbols based on social media activity"""
        # Simulate trending analysis (in real implementation, aggregate across platforms)
        trending_data = await self._simulate_trending_data()
        
        return json.dumps({
            'trending_symbols': trending_data,
            'timestamp': datetime.now().isoformat(),
            'source': 'aggregated_social_media'
        }, indent=2)
    
    async def _analyze_influencers(self, parameters: Dict[str, Any]) -> str:
        """Analyze influencer sentiment and mentions"""
        symbol = parameters.get('symbol')
        if not symbol:
            return json.dumps({'error': 'Symbol parameter required'})
        
        # Get recent sentiment data with influencer mentions
        recent_data = [
            data for data in self.sentiment_history.get(symbol, [])
            if (datetime.now() - data.timestamp).seconds < 86400  # Last 24 hours
            and data.influencer_mentions
        ]
        
        if not recent_data:
            return json.dumps({'message': f'No influencer mentions found for {symbol}'})
        
        # Aggregate influencer data
        influencer_impact = {}
        for data in recent_data:
            for influencer in data.influencer_mentions:
                if influencer not in influencer_impact:
                    influencer_impact[influencer] = {
                        'mentions': 0,
                        'avg_sentiment': 0,
                        'total_engagement': 0
                    }
                
                influencer_impact[influencer]['mentions'] += 1
                influencer_impact[influencer]['avg_sentiment'] += data.sentiment_score
                influencer_impact[influencer]['total_engagement'] += data.engagement_score
        
        # Calculate averages
        for influencer in influencer_impact:
            mentions = influencer_impact[influencer]['mentions']
            influencer_impact[influencer]['avg_sentiment'] /= mentions
            influencer_impact[influencer]['total_engagement'] /= mentions
        
        return json.dumps({
            'symbol': symbol,
            'influencer_analysis': influencer_impact,
            'analysis_period': '24 hours'
        }, indent=2)
    
    def _clean_old_history(self):
        """Clean old sentiment history data"""
        cutoff_time = datetime.now() - timedelta(days=self.config['max_history_days'])
        
        for symbol in self.sentiment_history:
            self.sentiment_history[symbol] = [
                data for data in self.sentiment_history[symbol]
                if data.timestamp > cutoff_time
            ]
    
    # Simulation methods (replace with actual API calls in production)
    
    async def _simulate_reddit_data(self, symbol: str, subreddit: str) -> List[Dict[str, Any]]:
        """Simulate Reddit API data"""
        posts = []
        for i in range(np.random.randint(5, 25)):
            posts.append({
                'title': f'Discussion about {symbol} - Post {i}',
                'body': f'Some analysis of {symbol} with bullish sentiment',
                'score': np.random.randint(1, 1000),
                'num_comments': np.random.randint(0, 200),
                'created_utc': time.time() - np.random.randint(0, 86400)
            })
        return posts
    
    async def _simulate_twitter_data(self, symbol: str) -> List[Dict[str, Any]]:
        """Simulate Twitter API data"""
        tweets = []
        for i in range(np.random.randint(10, 50)):
            tweets.append({
                'text': f'Tweet about {symbol} with some sentiment',
                'retweet_count': np.random.randint(0, 100),
                'like_count': np.random.randint(0, 500),
                'user': {
                    'username': f'user_{i}',
                    'followers_count': np.random.randint(100, 100000)
                },
                'created_at': datetime.now() - timedelta(hours=np.random.randint(0, 24))
            })
        return tweets
    
    async def _simulate_stocktwits_data(self, symbol: str) -> List[Dict[str, Any]]:
        """Simulate StockTwits API data"""
        messages = []
        for i in range(np.random.randint(5, 30)):
            sentiment = np.random.choice(['Bullish', 'Bearish', None])
            messages.append({
                'body': f'StockTwits message about {symbol}',
                'entities': {
                    'sentiment': {'basic': sentiment} if sentiment else {}
                },
                'likes': {'total': np.random.randint(0, 50)},
                'created_at': datetime.now() - timedelta(hours=np.random.randint(0, 24))
            })
        return messages
    
    async def _simulate_trending_data(self) -> List[Dict[str, Any]]:
        """Simulate trending symbols data"""
        symbols = ['AAPL', 'TSLA', 'NVDA', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NFLX']
        trending = []
        
        for symbol in np.random.choice(symbols, 5, replace=False):
            trending.append({
                'symbol': symbol,
                'mention_count': np.random.randint(50, 500),
                'sentiment_score': np.random.uniform(-0.5, 0.8),
                'trending_score': np.random.uniform(0.3, 1.0),
                'change_24h': np.random.uniform(-0.3, 0.5)
            })
        
        return sorted(trending, key=lambda x: x['trending_score'], reverse=True)


# Example usage
if __name__ == "__main__":
    from pathlib import Path
    from ..core.config_manager import ConfigManager
    from ..core.data_manager import UnifiedDataManager
    
    config_manager = ConfigManager(Path("../config"))
    data_manager = UnifiedDataManager(config_manager)
    
    analyzer = SocialSentimentAnalyzerTool(config_manager, data_manager)
    
    # Test sentiment analysis
    result = analyzer._run('analyze_sentiment', {'symbols': ['AAPL', 'TSLA']})
    print("Sentiment Analysis:", result)
    
    # Get alerts
    result = analyzer._run('get_alerts')
    print("Alerts:", result)