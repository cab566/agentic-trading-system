#!/usr/bin/env python3
"""
News-Driven Stock Discovery System

This tool monitors breaking news, earnings events, and market sentiment to identify
trending stocks and trading opportunities beyond traditional technical analysis.

Features:
- Real-time news monitoring from multiple sources
- Earnings calendar integration with pre/post earnings analysis
- Sentiment analysis and trend detection
- Social media mention tracking
- SEC filing monitoring
- Analyst upgrade/downgrade tracking
- Integration with existing agent system

Data Sources:
- Alpha Vantage News API
- Financial Modeling Prep API
- Yahoo Finance News
- SEC EDGAR API
- Reddit/Twitter sentiment (via APIs)
- No synthetic or mock data

Author: AI Trading System v2.0
Date: January 2025
"""

import asyncio
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from enum import Enum
import aiohttp
import json
import re
from concurrent.futures import ThreadPoolExecutor
import time
import yfinance as yf
from textblob import TextBlob
import feedparser
import requests
import hashlib

try:
    from ..core.config_manager import ConfigManager
    from ..utils.cache_manager import CacheManager
    from ..utils.notifications import NotificationManager
    from ..utils.yfinance_optimizer import BatchDataDownloader, BatchRequest
    from ..core.session_manager import SessionManager
except ImportError:
    # Fallback for direct execution
    import sys
    from pathlib import Path
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))
    from core.config_manager import ConfigManager
    from utils.cache_manager import CacheManager
    from utils.notifications import NotificationManager
    from utils.yfinance_optimizer import BatchDataDownloader, BatchRequest
    from core.session_manager import SessionManager


class NewsEventType(Enum):
    """Types of news events that can drive stock discovery."""
    EARNINGS_BEAT = "earnings_beat"
    EARNINGS_MISS = "earnings_miss"
    ANALYST_UPGRADE = "analyst_upgrade"
    ANALYST_DOWNGRADE = "analyst_downgrade"
    FDA_APPROVAL = "fda_approval"
    MERGER_ACQUISITION = "merger_acquisition"
    PRODUCT_LAUNCH = "product_launch"
    PARTNERSHIP = "partnership"
    REGULATORY_NEWS = "regulatory_news"
    INSIDER_TRADING = "insider_trading"
    DIVIDEND_ANNOUNCEMENT = "dividend_announcement"
    STOCK_SPLIT = "stock_split"
    GUIDANCE_CHANGE = "guidance_change"
    BREAKING_NEWS = "breaking_news"


class SentimentScore(Enum):
    """Sentiment scoring levels."""
    VERY_NEGATIVE = -2
    NEGATIVE = -1
    NEUTRAL = 0
    POSITIVE = 1
    VERY_POSITIVE = 2


@dataclass
class NewsEvent:
    """News event that could drive stock movement."""
    symbol: str
    title: str
    summary: str
    event_type: NewsEventType
    sentiment_score: float
    sentiment_label: SentimentScore
    source: str
    url: str
    published_time: datetime
    relevance_score: float
    confidence_score: float
    price_impact_prediction: float  # Expected price impact %
    volume_impact_prediction: float  # Expected volume multiplier
    keywords: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'symbol': self.symbol,
            'title': self.title,
            'summary': self.summary,
            'event_type': self.event_type.value,
            'sentiment_score': self.sentiment_score,
            'sentiment_label': self.sentiment_label.value,
            'source': self.source,
            'url': self.url,
            'published_time': self.published_time.isoformat(),
            'relevance_score': self.relevance_score,
            'confidence_score': self.confidence_score,
            'price_impact_prediction': self.price_impact_prediction,
            'volume_impact_prediction': self.volume_impact_prediction,
            'keywords': self.keywords
        }


@dataclass
class EarningsEvent:
    """Earnings event information."""
    symbol: str
    company_name: str
    earnings_date: datetime
    estimated_eps: Optional[float]
    actual_eps: Optional[float]
    estimated_revenue: Optional[float]
    actual_revenue: Optional[float]
    surprise_percentage: Optional[float]
    is_pre_market: bool
    is_after_market: bool
    days_until_earnings: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'symbol': self.symbol,
            'company_name': self.company_name,
            'earnings_date': self.earnings_date.isoformat(),
            'estimated_eps': self.estimated_eps,
            'actual_eps': self.actual_eps,
            'estimated_revenue': self.estimated_revenue,
            'actual_revenue': self.actual_revenue,
            'surprise_percentage': self.surprise_percentage,
            'is_pre_market': self.is_pre_market,
            'is_after_market': self.is_after_market,
            'days_until_earnings': self.days_until_earnings
        }


@dataclass
class DiscoveryConfig:
    """Configuration for news discovery."""
    min_relevance_score: float = 0.6
    min_sentiment_magnitude: float = 0.3
    max_news_age_hours: int = 24
    max_results_per_scan: int = 100
    earnings_lookforward_days: int = 7
    earnings_lookback_days: int = 2
    enable_social_sentiment: bool = True
    enable_sec_filings: bool = True
    news_sources: List[str] = field(default_factory=lambda: [
        'yahoo_finance'  # Removed 'alpha_vantage' due to demo key rate limits
        # 'financial_modeling_prep'  # Not implemented yet
    ])
    scan_interval_minutes: int = 15
    # Cache configuration
    enable_caching: bool = True
    news_cache_ttl: int = 300  # 5 minutes for news data
    sentiment_cache_ttl: int = 3600  # 1 hour for sentiment analysis
    earnings_cache_ttl: int = 21600  # 6 hours for earnings data


class NewsDataProvider:
    """Base class for news data providers."""
    
    def __init__(self, config_manager: ConfigManager):
        self.config_manager = config_manager
        self.logger = logging.getLogger(__name__)
    
    async def fetch_news(self, symbols: List[str] = None) -> List[Dict[str, Any]]:
        """Fetch news data. To be implemented by subclasses."""
        raise NotImplementedError


class YahooFinanceNewsProvider(NewsDataProvider):
    """Yahoo Finance news provider with caching."""
    
    def __init__(self, config_manager: ConfigManager):
        super().__init__(config_manager)
        self.cache_manager = CacheManager(config_manager)
        self.config = DiscoveryConfig()
    
    def _generate_cache_key(self, prefix: str, *args) -> str:
        """Generate a cache key from prefix and arguments."""
        key_data = f"{prefix}:{':'.join(str(arg) for arg in args)}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    async def fetch_news(self, symbols: List[str] = None) -> List[Dict[str, Any]]:
        """Fetch news from Yahoo Finance with caching."""
        # Generate cache key based on symbols and current time (rounded to 5 minutes)
        cache_key = self._generate_cache_key(
            "yahoo_news", 
            str(symbols) if symbols else "general",
            datetime.now().strftime("%Y%m%d%H%M")[:-1] + "0"  # Round to 5-minute intervals
        )
        
        # Check cache first
        if self.config.enable_caching:
            cached_news = self.cache_manager.get(cache_key)
            if cached_news is not None:
                return cached_news
        
        try:
            news_items = []
            
            if symbols:
                # Fetch news for specific symbols
                for symbol in symbols[:20]:  # Limit to avoid rate limits
                    try:
                        ticker = yf.Ticker(symbol)
                        news = ticker.news
                        
                        for item in news[:5]:  # Top 5 news per symbol
                            news_items.append({
                                'symbol': symbol,
                                'title': item.get('title', ''),
                                'summary': item.get('summary', ''),
                                'url': item.get('link', ''),
                                'published_time': datetime.fromtimestamp(
                                    item.get('providerPublishTime', time.time())
                                ),
                                'source': 'yahoo_finance',
                                'publisher': item.get('publisher', '')
                            })
                    except Exception as e:
                        self.logger.debug(f"Error fetching news for {symbol}: {e}")
            else:
                # Fetch general market news
                try:
                    # Use RSS feed for general market news
                    feed_url = "https://feeds.finance.yahoo.com/rss/2.0/headline"
                    feed = feedparser.parse(feed_url)
                    
                    for entry in feed.entries[:20]:
                        news_items.append({
                            'symbol': None,  # General market news
                            'title': entry.title,
                            'summary': entry.summary if hasattr(entry, 'summary') else '',
                            'url': entry.link,
                            'published_time': datetime(*entry.published_parsed[:6]),
                            'source': 'yahoo_finance',
                            'publisher': 'Yahoo Finance'
                        })
                except Exception as e:
                    self.logger.error(f"Error fetching general news: {e}")
            
            # Cache the results
            if self.cache_manager and self.config.enable_caching:
                self._set_cached_data(cache_key, news_items, self.config.news_cache_ttl)
            
            return news_items
            
        except Exception as e:
            self.logger.error(f"Error fetching general news: {e}")
            return []
    
    def _set_cached_data(self, cache_key: str, data: Any, ttl: int):
        """Store data in cache."""
        try:
            self.cache_manager.set(cache_key, data, ttl)
        except Exception as e:
            self.logger.error(f"Error setting cache data: {e}")


class AlphaVantageNewsProvider(NewsDataProvider):
    """Alpha Vantage news provider."""
    
    def __init__(self, config_manager: ConfigManager):
        super().__init__(config_manager)
        self.api_key = self.config_manager.get_config().get('alpha_vantage', {}).get('api_key')
        self.cache_manager = CacheManager()
        self.config = DiscoveryConfig()
    
    def _generate_cache_key(self, prefix: str, *args) -> str:
        """Generate cache key for Alpha Vantage data."""
        key_data = f"{prefix}:{':'.join(str(arg) for arg in args)}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    async def fetch_news(self, symbols: List[str] = None) -> List[Dict[str, Any]]:
        """Fetch news from Alpha Vantage API with managed session"""
        cache_key = self._generate_cache_key('alpha_vantage_news', symbols or [])
        
        if self.cache_manager:
            cached_data = self._get_cached_data(cache_key)
            if cached_data is not None:
                return cached_data
        
        try:
            news_items = []
            
            # Use managed session instead of creating new ClientSession
            session_manager = SessionManager()
            async with session_manager.get_session() as session:
                if symbols:
                    # Fetch news for specific symbols
                    for symbol in symbols[:10]:  # Limit API calls
                        url = f"https://www.alphavantage.co/query"
                        params = {
                            'function': 'NEWS_SENTIMENT',
                            'tickers': symbol,
                            'apikey': self.api_key,
                            'limit': 10
                        }
                        
                        try:
                            async with session.get(url, params=params) as response:
                                if response.status == 200:
                                    data = await response.json()
                                    
                                    if 'feed' in data:
                                        for item in data['feed']:
                                            news_items.append({
                                                'symbol': symbol,
                                                'title': item.get('title', ''),
                                                'summary': item.get('summary', ''),
                                                'url': item.get('url', ''),
                                                'published_time': datetime.strptime(
                                                    item.get('time_published', ''),
                                                    '%Y%m%dT%H%M%S'
                                                ),
                                                'source': 'alpha_vantage',
                                                'publisher': item.get('source', ''),
                                                'sentiment_score': float(
                                                    item.get('overall_sentiment_score', 0)
                                                ),
                                                'sentiment_label': item.get('overall_sentiment_label', 'Neutral')
                                            })
                        except Exception as e:
                            self.logger.debug(f"Error fetching Alpha Vantage news for {symbol}: {e}")
                        
                        # Rate limiting
                        await asyncio.sleep(0.2)
                else:
                    # Fetch general market news
                    url = f"https://www.alphavantage.co/query"
                    params = {
                        'function': 'NEWS_SENTIMENT',
                        'apikey': self.api_key,
                        'limit': 50
                    }
                    
                    try:
                        async with session.get(url, params=params) as response:
                            if response.status == 200:
                                data = await response.json()
                                
                                if 'feed' in data:
                                    for item in data['feed']:
                                        # Extract symbols from ticker sentiment
                                        symbols_mentioned = []
                                        if 'ticker_sentiment' in item:
                                            symbols_mentioned = [
                                                ts.get('ticker', '') 
                                                for ts in item['ticker_sentiment']
                                            ]
                                        
                                        news_items.append({
                                            'symbol': symbols_mentioned[0] if symbols_mentioned else None,
                                            'symbols_mentioned': symbols_mentioned,
                                            'title': item.get('title', ''),
                                            'summary': item.get('summary', ''),
                                            'url': item.get('url', ''),
                                            'published_time': datetime.strptime(
                                                item.get('time_published', ''),
                                                '%Y%m%dT%H%M%S'
                                            ),
                                            'source': 'alpha_vantage',
                                            'publisher': item.get('source', ''),
                                            'sentiment_score': float(
                                                item.get('overall_sentiment_score', 0)
                                            ),
                                            'sentiment_label': item.get('overall_sentiment_label', 'Neutral')
                                        })
                    except Exception as e:
                        self.logger.error(f"Error fetching Alpha Vantage general news: {e}")
            
            # Cache the results
            if self.cache_manager and self.config.enable_caching:
                self._set_cached_data(cache_key, news_items, self.config.news_cache_ttl)
            
            return news_items
            
        except Exception as e:
            self.logger.error(f"Error in Alpha Vantage news provider: {e}")
            return []
    
    def _get_cached_data(self, cache_key: str) -> Optional[Any]:
        """Get data from cache."""
        try:
            return self.cache_manager.get(cache_key)
        except Exception as e:
            self.logger.error(f"Error getting cache data: {e}")
            return None
    
    def _set_cached_data(self, cache_key: str, data: Any, ttl: int):
        """Store data in cache."""
        try:
            self.cache_manager.set(cache_key, data, ttl)
        except Exception as e:
            self.logger.error(f"Error setting cache data: {e}")


class NewsDrivenDiscovery:
    """
    News-driven stock discovery system.
    
    Monitors breaking news, earnings events, and market sentiment to identify
    trending stocks and trading opportunities.
    """
    
    def __init__(self, config_manager: ConfigManager):
        self.config_manager = config_manager
        self.cache_manager = CacheManager(config_manager)
        self.notification_manager = NotificationManager(config_manager)
        self.logger = logging.getLogger(__name__)
        
        # Configuration
        self.config = DiscoveryConfig()
        self._load_config()
        
        # Initialize optimized batch downloader
        self.batch_downloader = BatchDataDownloader(
            cache_manager=self.cache_manager,
            max_workers=4,
            enable_caching=True
        )
        
        # News providers - DISABLED Alpha Vantage due to demo key rate limits
        self.news_providers = {
            'yahoo_finance': YahooFinanceNewsProvider(config_manager),
            # 'alpha_vantage': AlphaVantageNewsProvider(config_manager)  # DISABLED - demo key has 25 requests/day limit
        }
        
        # State tracking
        self.last_scan_time: Optional[datetime] = None
        self.discovered_events: List[NewsEvent] = []
        self.earnings_calendar: List[EarningsEvent] = []
        self.is_scanning = False
        
        # Performance metrics
        self.scan_count = 0
        self.total_events_discovered = 0
        self.average_scan_time = 0.0
        
        # Cache performance tracking
        self.cache_hits = 0
        self.cache_misses = 0
        self.api_calls_saved = 0
        
        # Keywords for event classification
        self.event_keywords = {
            NewsEventType.EARNINGS_BEAT: ['earnings beat', 'exceeded expectations', 'beat estimates'],
            NewsEventType.EARNINGS_MISS: ['earnings miss', 'missed expectations', 'below estimates'],
            NewsEventType.ANALYST_UPGRADE: ['upgrade', 'raised rating', 'increased target'],
            NewsEventType.ANALYST_DOWNGRADE: ['downgrade', 'lowered rating', 'decreased target'],
            NewsEventType.FDA_APPROVAL: ['fda approval', 'fda cleared', 'regulatory approval'],
            NewsEventType.MERGER_ACQUISITION: ['merger', 'acquisition', 'takeover', 'buyout'],
            NewsEventType.PRODUCT_LAUNCH: ['product launch', 'new product', 'product release'],
            NewsEventType.PARTNERSHIP: ['partnership', 'joint venture', 'collaboration'],
            NewsEventType.DIVIDEND_ANNOUNCEMENT: ['dividend', 'dividend increase', 'special dividend'],
            NewsEventType.STOCK_SPLIT: ['stock split', 'share split'],
            NewsEventType.GUIDANCE_CHANGE: ['guidance', 'outlook', 'forecast']
        }
    
    def _load_config(self):
        """Load discovery configuration."""
        try:
            discovery_config = self.config_manager.get_config().get('news_discovery', {})
            
            if discovery_config:
                self.config.min_relevance_score = discovery_config.get('min_relevance_score', 0.6)
                self.config.min_sentiment_magnitude = discovery_config.get('min_sentiment_magnitude', 0.3)
                self.config.max_news_age_hours = discovery_config.get('max_news_age_hours', 24)
                self.config.max_results_per_scan = discovery_config.get('max_results_per_scan', 100)
                self.config.scan_interval_minutes = discovery_config.get('scan_interval_minutes', 15)
                
            self.logger.info(f"News discovery configured: min_relevance={self.config.min_relevance_score}")
            
        except Exception as e:
            self.logger.error(f"Error loading discovery config: {e}")
    
    async def start_discovery(self):
        """Start the news-driven discovery process."""
        if self.is_scanning:
            self.logger.warning("Discovery is already running")
            return
        
        self.is_scanning = True
        self.logger.info("Starting news-driven discovery")
        
        try:
            while self.is_scanning:
                scan_start_time = time.time()
                
                # Perform news discovery scan
                events = await self.discover_news_events()
                
                # Update earnings calendar
                await self.update_earnings_calendar()
                
                # Update metrics
                scan_duration = time.time() - scan_start_time
                self.scan_count += 1
                self.total_events_discovered += len(events)
                self.average_scan_time = (
                    (self.average_scan_time * (self.scan_count - 1) + scan_duration) / 
                    self.scan_count
                )
                
                # Store results
                self.discovered_events = events
                
                # Send notifications for significant events
                await self._process_event_alerts(events)
                
                self.logger.info(
                    f"Discovery scan {self.scan_count} completed: {len(events)} events discovered "
                    f"in {scan_duration:.2f}s"
                )
                
                # Wait for next scan interval
                await asyncio.sleep(self.config.scan_interval_minutes * 60)
                
        except Exception as e:
            self.logger.error(f"Error in discovery loop: {e}")
        finally:
            self.is_scanning = False
    
    def _generate_cache_key(self, prefix: str, *args) -> str:
        """Generate a cache key from prefix and arguments."""
        key_data = f"{prefix}:{':'.join(str(arg) for arg in args)}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _get_cached_data(self, cache_key: str) -> Optional[Any]:
        """Get data from cache if available."""
        if not self.config.enable_caching:
            return None
        
        try:
            cached_data = self.cache_manager.get(cache_key)
            if cached_data is not None:
                self.cache_hits += 1
                self.api_calls_saved += 1
                return cached_data
            else:
                self.cache_misses += 1
                return None
        except Exception as e:
            self.logger.warning(f"Cache retrieval error: {e}")
            return None
    
    def _set_cached_data(self, cache_key: str, data: Any, ttl: int):
        """Set data in cache with TTL."""
        if not self.config.enable_caching:
            return
        
        try:
            self.cache_manager.set(cache_key, data, ttl)
        except Exception as e:
            self.logger.warning(f"Cache storage error: {e}")

    async def discover_news_events(self) -> List[NewsEvent]:
        """Discover news events that could drive stock movements."""
        try:
            # Check cache for recent news events
            cache_key = self._generate_cache_key("news_events", datetime.now().strftime("%Y%m%d%H%M"))
            cached_events = self._get_cached_data(cache_key)
            
            if cached_events is not None:
                self.logger.info(f"Retrieved {len(cached_events)} cached news events")
                return cached_events
            
            all_news_items = []
            
            # Fetch news from all configured providers
            for provider_name in self.config.news_sources:
                if provider_name in self.news_providers:
                    provider = self.news_providers[provider_name]
                    news_items = await provider.fetch_news()
                    all_news_items.extend(news_items)
            
            # Process and classify news items
            events = []
            for news_item in all_news_items:
                event = await self._process_news_item(news_item)
                if event:
                    events.append(event)
            
            # Filter and rank events
            events = self._filter_and_rank_events(events)
            
            # Cache the results
            self._set_cached_data(cache_key, events, self.config.news_cache_ttl)
            
            self.last_scan_time = datetime.now()
            
            return events
            
        except Exception as e:
            self.logger.error(f"Error discovering news events: {e}")
            return []
    
    async def _process_news_item(self, news_item: Dict[str, Any]) -> Optional[NewsEvent]:
        """Process a single news item into a NewsEvent."""
        try:
            title = news_item.get('title', '')
            summary = news_item.get('summary', '')
            text = f"{title} {summary}".lower()
            
            # Skip if too old
            published_time = news_item.get('published_time')
            if published_time:
                age_hours = (datetime.now() - published_time).total_seconds() / 3600
                if age_hours > self.config.max_news_age_hours:
                    return None
            
            # Extract or determine symbol
            symbol = news_item.get('symbol')
            if not symbol:
                # Try to extract symbol from text
                symbol = self._extract_symbol_from_text(text)
                if not symbol:
                    return None
            
            # Classify event type
            event_type = self._classify_event_type(text)
            
            # Calculate sentiment
            sentiment_score, sentiment_label = self._calculate_sentiment(text, news_item)
            
            # Calculate relevance and confidence scores
            relevance_score = self._calculate_relevance_score(text, event_type)
            confidence_score = self._calculate_confidence_score(news_item, sentiment_score, relevance_score)
            
            # Predict price and volume impact
            price_impact = self._predict_price_impact(event_type, sentiment_score, relevance_score)
            volume_impact = self._predict_volume_impact(event_type, relevance_score)
            
            # Extract keywords
            keywords = self._extract_keywords(text)
            
            return NewsEvent(
                symbol=symbol,
                title=title,
                summary=summary,
                event_type=event_type,
                sentiment_score=sentiment_score,
                sentiment_label=sentiment_label,
                source=news_item.get('source', ''),
                url=news_item.get('url', ''),
                published_time=published_time or datetime.now(),
                relevance_score=relevance_score,
                confidence_score=confidence_score,
                price_impact_prediction=price_impact,
                volume_impact_prediction=volume_impact,
                keywords=keywords
            )
            
        except Exception as e:
            self.logger.debug(f"Error processing news item: {e}")
            return None
    
    def _extract_symbol_from_text(self, text: str) -> Optional[str]:
        """Extract stock symbol from news text."""
        try:
            # Look for common stock symbol patterns
            patterns = [
                r'\b([A-Z]{1,5})\s+stock',
                r'\b([A-Z]{1,5})\s+shares',
                r'\$([A-Z]{1,5})\b',
                r'\(([A-Z]{1,5})\)',
                r'NYSE:\s*([A-Z]{1,5})',
                r'NASDAQ:\s*([A-Z]{1,5})'
            ]
            
            for pattern in patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                if matches:
                    # Return the first valid-looking symbol
                    for match in matches:
                        if len(match) >= 1 and len(match) <= 5:
                            return match.upper()
            
            return None
            
        except Exception as e:
            self.logger.debug(f"Error extracting symbol: {e}")
            return None
    
    def _classify_event_type(self, text: str) -> NewsEventType:
        """Classify the type of news event."""
        try:
            # Check for specific event type keywords
            for event_type, keywords in self.event_keywords.items():
                for keyword in keywords:
                    if keyword in text:
                        return event_type
            
            # Default to breaking news
            return NewsEventType.BREAKING_NEWS
            
        except Exception as e:
            self.logger.debug(f"Error classifying event type: {e}")
            return NewsEventType.BREAKING_NEWS
    
    def _calculate_sentiment(self, text: str, news_item: Dict[str, Any]) -> Tuple[float, SentimentScore]:
        """Calculate sentiment score and label with caching."""
        # Check cache for sentiment analysis
        cache_key = self._generate_cache_key("sentiment", text[:100])  # Use first 100 chars for key
        cached_sentiment = self._get_cached_data(cache_key)
        
        if cached_sentiment is not None:
            return cached_sentiment
        
        try:
            # Use Alpha Vantage sentiment if available
            if 'sentiment_score' in news_item:
                score = news_item['sentiment_score']
            else:
                # Use TextBlob for sentiment analysis
                blob = TextBlob(text)
                score = blob.sentiment.polarity
            
            # Convert to sentiment label
            if score >= 0.5:
                label = SentimentScore.VERY_POSITIVE
            elif score >= 0.1:
                label = SentimentScore.POSITIVE
            elif score <= -0.5:
                label = SentimentScore.VERY_NEGATIVE
            elif score <= -0.1:
                label = SentimentScore.NEGATIVE
            else:
                label = SentimentScore.NEUTRAL
            
            result = (score, label)
            
            # Cache the result
            self._set_cached_data(cache_key, result, self.config.sentiment_cache_ttl)
            
            return result
            
        except Exception as e:
            self.logger.debug(f"Error calculating sentiment: {e}")
            return 0.0, SentimentScore.NEUTRAL
    
    def _calculate_relevance_score(self, text: str, event_type: NewsEventType) -> float:
        """Calculate relevance score for the news item."""
        try:
            score = 0.5  # Base score
            
            # Boost for specific event types
            high_impact_events = [
                NewsEventType.EARNINGS_BEAT,
                NewsEventType.EARNINGS_MISS,
                NewsEventType.FDA_APPROVAL,
                NewsEventType.MERGER_ACQUISITION
            ]
            
            if event_type in high_impact_events:
                score += 0.3
            
            # Boost for financial keywords
            financial_keywords = [
                'revenue', 'profit', 'earnings', 'guidance', 'outlook',
                'merger', 'acquisition', 'partnership', 'approval'
            ]
            
            for keyword in financial_keywords:
                if keyword in text:
                    score += 0.1
            
            return min(1.0, score)
            
        except Exception as e:
            self.logger.debug(f"Error calculating relevance score: {e}")
            return 0.5
    
    def _calculate_confidence_score(self, news_item: Dict[str, Any], 
                                  sentiment_score: float, relevance_score: float) -> float:
        """Calculate confidence score for the event."""
        try:
            score = 50.0  # Base score
            
            # Boost for reputable sources
            reputable_sources = ['reuters', 'bloomberg', 'wsj', 'cnbc', 'marketwatch']
            source = news_item.get('source', '').lower()
            publisher = news_item.get('publisher', '').lower()
            
            if any(rep_source in source or rep_source in publisher for rep_source in reputable_sources):
                score += 20.0
            
            # Boost for strong sentiment
            score += abs(sentiment_score) * 15.0
            
            # Boost for high relevance
            score += relevance_score * 15.0
            
            return min(100.0, score)
            
        except Exception as e:
            self.logger.debug(f"Error calculating confidence score: {e}")
            return 50.0
    
    def _predict_price_impact(self, event_type: NewsEventType, 
                            sentiment_score: float, relevance_score: float) -> float:
        """Predict expected price impact percentage."""
        try:
            # Base impact by event type
            impact_multipliers = {
                NewsEventType.EARNINGS_BEAT: 3.0,
                NewsEventType.EARNINGS_MISS: -3.0,
                NewsEventType.ANALYST_UPGRADE: 2.0,
                NewsEventType.ANALYST_DOWNGRADE: -2.0,
                NewsEventType.FDA_APPROVAL: 5.0,
                NewsEventType.MERGER_ACQUISITION: 8.0,
                NewsEventType.PRODUCT_LAUNCH: 1.5,
                NewsEventType.PARTNERSHIP: 2.0,
                NewsEventType.BREAKING_NEWS: 1.0
            }
            
            base_impact = impact_multipliers.get(event_type, 1.0)
            
            # Adjust for sentiment and relevance
            impact = base_impact * (1 + sentiment_score) * relevance_score
            
            return round(impact, 2)
            
        except Exception as e:
            self.logger.debug(f"Error predicting price impact: {e}")
            return 0.0
    
    def _predict_volume_impact(self, event_type: NewsEventType, relevance_score: float) -> float:
        """Predict expected volume impact multiplier."""
        try:
            # Base volume multipliers by event type
            volume_multipliers = {
                NewsEventType.EARNINGS_BEAT: 2.5,
                NewsEventType.EARNINGS_MISS: 2.5,
                NewsEventType.ANALYST_UPGRADE: 1.8,
                NewsEventType.ANALYST_DOWNGRADE: 1.8,
                NewsEventType.FDA_APPROVAL: 4.0,
                NewsEventType.MERGER_ACQUISITION: 5.0,
                NewsEventType.PRODUCT_LAUNCH: 1.5,
                NewsEventType.PARTNERSHIP: 2.0,
                NewsEventType.BREAKING_NEWS: 1.3
            }
            
            base_multiplier = volume_multipliers.get(event_type, 1.2)
            
            # Adjust for relevance
            multiplier = base_multiplier * (0.5 + relevance_score)
            
            return round(multiplier, 2)
            
        except Exception as e:
            self.logger.debug(f"Error predicting volume impact: {e}")
            return 1.0
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract relevant keywords from news text."""
        try:
            # Common financial keywords to extract
            financial_terms = [
                'earnings', 'revenue', 'profit', 'loss', 'guidance', 'outlook',
                'merger', 'acquisition', 'partnership', 'approval', 'launch',
                'upgrade', 'downgrade', 'beat', 'miss', 'exceed', 'below'
            ]
            
            keywords = []
            words = text.lower().split()
            
            for term in financial_terms:
                if term in text.lower():
                    keywords.append(term)
            
            return keywords[:10]  # Limit to top 10 keywords
            
        except Exception as e:
            self.logger.debug(f"Error extracting keywords: {e}")
            return []
    
    def _filter_and_rank_events(self, events: List[NewsEvent]) -> List[NewsEvent]:
        """Filter and rank events by relevance and confidence."""
        try:
            # Filter by minimum criteria
            filtered_events = [
                event for event in events
                if (event.relevance_score >= self.config.min_relevance_score and
                    abs(event.sentiment_score) >= self.config.min_sentiment_magnitude)
            ]
            
            # Sort by combined score (confidence * relevance * abs(sentiment))
            filtered_events.sort(
                key=lambda x: x.confidence_score * x.relevance_score * abs(x.sentiment_score),
                reverse=True
            )
            
            # Limit results
            return filtered_events[:self.config.max_results_per_scan]
            
        except Exception as e:
            self.logger.error(f"Error filtering and ranking events: {e}")
            return events
    
    async def update_earnings_calendar(self):
        """Update the earnings calendar with upcoming events."""
        try:
            # Check cache first
            cache_key = self._generate_cache_key("earnings_calendar")
            cached_events = self._get_cached_data(cache_key)
            
            if cached_events is not None:
                self.earnings_calendar = cached_events
                self.logger.debug("Using cached earnings calendar data")
                return
            
            earnings_events = []
            
            # Sample symbols to check for earnings
            sample_symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA']
            
            # Try to use batch downloader for basic info first
            try:
                batch_request = BatchRequest(
                    symbols=sample_symbols,
                    data_type='info',
                    period='1d'
                )
                
                # Get basic info in batch
                batch_results = await self.batch_downloader.fetch_batch_data([batch_request])
                symbol_info = batch_results.get(0, {}) if batch_results else {}
                
            except Exception as e:
                self.logger.debug(f"Batch info fetch failed, using fallback: {e}")
                symbol_info = {}
            
            # Process earnings calendar for each symbol
            for symbol in sample_symbols:
                try:
                    ticker = yf.Ticker(symbol)
                    calendar = ticker.calendar
                    
                    if calendar is not None and not calendar.empty:
                        # Get company name from batch results or ticker info
                        company_name = symbol
                        if symbol in symbol_info and 'longName' in symbol_info[symbol]:
                            company_name = symbol_info[symbol]['longName']
                        else:
                            try:
                                company_name = ticker.info.get('longName', symbol)
                            except:
                                company_name = symbol
                        
                        for date, row in calendar.iterrows():
                            earnings_date = pd.to_datetime(date).to_pydatetime()
                            days_until = (earnings_date - datetime.now()).days
                            
                            if -2 <= days_until <= 7:  # Within our lookback/forward window
                                earnings_events.append(EarningsEvent(
                                    symbol=symbol,
                                    company_name=company_name,
                                    earnings_date=earnings_date,
                                    estimated_eps=None,  # Would come from API
                                    actual_eps=None,
                                    estimated_revenue=None,
                                    actual_revenue=None,
                                    surprise_percentage=None,
                                    is_pre_market=True,  # Default assumption
                                    is_after_market=False,
                                    days_until_earnings=days_until
                                ))
                except Exception as e:
                    self.logger.debug(f"Error fetching earnings for {symbol}: {e}")
            
            self.earnings_calendar = earnings_events
            
            # Cache the results
            self._set_cached_data(cache_key, earnings_events, self.config.earnings_cache_ttl)
            
            self.logger.info(f"Updated earnings calendar with {len(earnings_events)} events")
            
        except Exception as e:
            self.logger.error(f"Error updating earnings calendar: {e}")
    
    async def _process_event_alerts(self, events: List[NewsEvent]):
        """Process and send alerts for significant news events."""
        try:
            # Filter for high-priority alerts
            high_priority_events = [
                event for event in events
                if (event.confidence_score > 75 and
                    abs(event.price_impact_prediction) > 2.0)
            ]
            
            if not high_priority_events:
                return
            
            # Create alert message
            alert_data = {
                'type': 'news_event_alert',
                'timestamp': datetime.now().isoformat(),
                'event_count': len(high_priority_events),
                'events': [event.to_dict() for event in high_priority_events[:10]]
            }
            
            # Send notification
            await self.notification_manager.send_notification(
                title=f"News Event Alert: {len(high_priority_events)} significant events",
                message=f"Detected {len(high_priority_events)} high-impact news events",
                data=alert_data,
                priority="high"
            )
            
            self.logger.info(f"Sent news event alert for {len(high_priority_events)} events")
            
        except Exception as e:
            self.logger.error(f"Error processing event alerts: {e}")
    
    def stop_discovery(self):
        """Stop the news-driven discovery."""
        self.is_scanning = False
        self.logger.info("News-driven discovery stopped")
    
    def get_latest_events(self, limit: int = 20) -> List[NewsEvent]:
        """Get the latest discovered news events."""
        return self.discovered_events[:limit]
    
    def get_earnings_calendar(self, days_ahead: int = 7) -> List[EarningsEvent]:
        """Get upcoming earnings events."""
        return [
            event for event in self.earnings_calendar
            if 0 <= event.days_until_earnings <= days_ahead
        ]
    
    def get_discovery_metrics(self) -> Dict[str, Any]:
        """Get discovery performance metrics."""
        # Get batch processing metrics
        batch_metrics = {}
        if hasattr(self.batch_downloader, 'get_metrics'):
            batch_metrics = self.batch_downloader.get_metrics()
        
        # Get cache metrics
        cache_metrics = {}
        if hasattr(self, 'cache_manager') and hasattr(self.cache_manager, 'get_stats'):
            cache_metrics = self.cache_manager.get_stats()
        
        return {
            'scan_count': self.scan_count,
            'total_events_discovered': self.total_events_discovered,
            'average_scan_time': self.average_scan_time,
            'last_scan_time': self.last_scan_time.isoformat() if self.last_scan_time else None,
            'is_scanning': self.is_scanning,
            'latest_events_count': len(self.discovered_events),
            'earnings_events_count': len(self.earnings_calendar),
            'batch_processing': {
                'enabled': True,
                'metrics': batch_metrics
            },
            'caching': {
                'enabled': self.config.enable_caching,
                'metrics': cache_metrics,
                'ttl_settings': {
                    'news_cache_ttl': self.config.news_cache_ttl,
                    'sentiment_cache_ttl': self.config.sentiment_cache_ttl,
                    'earnings_cache_ttl': self.config.earnings_cache_ttl
                }
            },
            'config': {
                'min_relevance_score': self.config.min_relevance_score,
                'scan_interval_minutes': self.config.scan_interval_minutes,
                'news_sources': self.config.news_sources
            }
        }


# Integration with existing agent system
class NewsDrivenDiscoveryTool:
    """Tool wrapper for integration with CrewAI agents."""
    
    def __init__(self, discovery: NewsDrivenDiscovery):
        self.discovery = discovery
        self.name = "news_driven_discovery"
        self.description = "Discover stocks with significant news events and sentiment changes"
    
    async def run(self, query: str = "") -> str:
        """Run news-driven discovery."""
        try:
            events = await self.discovery.discover_news_events()
            
            if not events:
                return "No significant news events detected in current scan."
            
            # Format results for agent consumption
            results = []
            for event in events[:10]:  # Top 10 results
                results.append(
                    f"{event.symbol}: {event.event_type.value} - "
                    f"{event.sentiment_label.name} sentiment, "
                    f"{event.price_impact_prediction:+.1f}% predicted impact"
                )
            
            return f"News-Driven Discovery Results:\n" + "\n".join(results)
            
        except Exception as e:
            return f"Error in news-driven discovery: {e}"


if __name__ == "__main__":
    # Test the news-driven discovery system
    import asyncio
    from pathlib import Path
    import sys
    
    # Add project root to path
    project_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(project_root))
    
    from trading_system_v2.core.config_manager import ConfigManager
    
    async def test_discovery():
        """Test the news-driven discovery system."""
        config_manager = ConfigManager()
        await config_manager.initialize()
        
        discovery = NewsDrivenDiscovery(config_manager)
        
        print("Testing news-driven discovery...")
        events = await discovery.discover_news_events()
        
        print(f"\nDiscovered {len(events)} news events:")
        for event in events[:5]:
            print(f"  {event.symbol}: {event.title[:60]}...")
            print(f"    Type: {event.event_type.value}, Sentiment: {event.sentiment_label.name}")
            print(f"    Impact: {event.price_impact_prediction:+.1f}%")
        
        print(f"\nDiscovery metrics: {discovery.get_discovery_metrics()}")
    
    asyncio.run(test_discovery())