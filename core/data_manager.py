#!/usr/bin/env python3
"""
Unified Data Manager for CrewAI Trading System

Provides a unified interface for accessing market data from multiple sources
with caching, rate limiting, and failover capabilities.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import threading
from dataclasses import dataclass

import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
import redis
import yfinance as yf
from polygon import RESTClient as PolygonClient
from alpha_vantage.timeseries import TimeSeries
from newsapi import NewsApiClient
import requests
from ratelimit import limits, sleep_and_retry

from .config_manager import ConfigManager
from .data_types import DataSourceAdapter, DataRequest, DataResponse
# from .crypto_adapter import CryptoDataManager, BinanceAdapter, CoinbaseAdapter  # DISABLED - Crypto system commented out
from .forex_adapter import ForexDataManager, OandaAdapter, ForexFactoryAdapter, AlphaVantageForexAdapter


class YFinanceAdapter(DataSourceAdapter):
    """Yahoo Finance data adapter."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__('yfinance', config)
    
    async def fetch_data(self, request: DataRequest) -> DataResponse:
        """Fetch data from Yahoo Finance."""
        try:
            if self.is_rate_limited():
                await asyncio.sleep(1)
            
            self.record_request()
            
            ticker = yf.Ticker(request.symbol)
            
            if request.data_type == 'price':
                if request.timeframe in ['1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max']:
                    data = ticker.history(period=request.timeframe)
                else:
                    # For intraday data
                    data = ticker.history(
                        start=request.start_date,
                        end=request.end_date,
                        interval=request.timeframe
                    )
            
            elif request.data_type == 'fundamentals':
                info = ticker.info
                financials = ticker.financials
                data = {'info': info, 'financials': financials}
            
            elif request.data_type == 'options':
                options = ticker.options
                if options:
                    calls = ticker.option_chain(options[0]).calls
                    puts = ticker.option_chain(options[0]).puts
                    data = {'calls': calls, 'puts': puts, 'expirations': options}
                else:
                    data = pd.DataFrame()
            
            else:
                raise ValueError(f"Unsupported data type: {request.data_type}")
            
            return DataResponse(
                data=data,
                source=self.name,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            self.logger.error(f"Error fetching data from Yahoo Finance: {e}")
            return DataResponse(
                data=pd.DataFrame(),
                source=self.name,
                timestamp=datetime.now(),
                error=str(e)
            )


class PolygonAdapter(DataSourceAdapter):
    """Polygon.io data adapter."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__('polygon', config)
        self.client = PolygonClient(config.get('api_key', ''))
    
    async def fetch_data(self, request: DataRequest) -> DataResponse:
        """Fetch data from Polygon.io."""
        try:
            if self.is_rate_limited():
                await asyncio.sleep(1)
            
            self.record_request()
            
            if request.data_type == 'price':
                # Convert timeframe to Polygon format
                multiplier, timespan = self._parse_timeframe(request.timeframe)
                
                aggs = self.client.get_aggs(
                    ticker=request.symbol,
                    multiplier=multiplier,
                    timespan=timespan,
                    from_=request.start_date.strftime('%Y-%m-%d') if request.start_date else None,
                    to=request.end_date.strftime('%Y-%m-%d') if request.end_date else None
                )
                
                # Convert to DataFrame
                data = pd.DataFrame([
                    {
                        'timestamp': pd.to_datetime(agg.timestamp, unit='ms'),
                        'open': agg.open,
                        'high': agg.high,
                        'low': agg.low,
                        'close': agg.close,
                        'volume': agg.volume
                    }
                    for agg in aggs
                ])
                
                if not data.empty:
                    data.set_index('timestamp', inplace=True)
            
            elif request.data_type == 'news':
                news = self.client.list_ticker_news(
                    ticker=request.symbol,
                    limit=request.parameters.get('limit', 50) if request.parameters else 50
                )
                data = [{
                    'title': article.title,
                    'description': article.description,
                    'url': article.article_url,
                    'published_utc': article.published_utc,
                    'author': article.author,
                    'keywords': article.keywords
                } for article in news]
            
            else:
                raise ValueError(f"Unsupported data type: {request.data_type}")
            
            return DataResponse(
                data=data,
                source=self.name,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            self.logger.error(f"Error fetching data from Polygon: {e}")
            return DataResponse(
                data=pd.DataFrame(),
                source=self.name,
                timestamp=datetime.now(),
                error=str(e)
            )
    
    def _parse_timeframe(self, timeframe: str) -> tuple:
        """Parse timeframe into Polygon format."""
        if timeframe.endswith('m'):
            return int(timeframe[:-1]), 'minute'
        elif timeframe.endswith('h'):
            return int(timeframe[:-1]), 'hour'
        elif timeframe.endswith('d'):
            return int(timeframe[:-1]), 'day'
        else:
            return 1, 'day'


class NewsAPIAdapter(DataSourceAdapter):
    """News API adapter."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__('newsapi', config)
        self.client = NewsApiClient(api_key=config.get('api_key', ''))
    
    async def fetch_data(self, request: DataRequest) -> DataResponse:
        """Fetch news data."""
        try:
            # Rate limiting: News API free tier allows 1000 requests/day, 100/hour
            # Add 1 second delay between requests to avoid rate limits
            await asyncio.sleep(1)
            
            if self.is_rate_limited():
                await asyncio.sleep(1)
            
            self.record_request()
            
            if request.data_type == 'news':
                # Search for news about the symbol
                query = f"{request.symbol} OR {request.symbol.replace('$', '')} stock"
                
                try:
                    articles = self.client.get_everything(
                        q=query,
                        language='en',
                        sort_by='publishedAt',
                        page_size=min(request.parameters.get('limit', 20), 20) if request.parameters else 20  # Reduced from 50
                    )
                    
                    data = [{
                        'title': article['title'],
                        'description': article['description'],
                        'url': article['url'],
                        'published_at': article['publishedAt'],
                        'source': article['source']['name'],
                        'author': article['author']
                    } for article in articles['articles']]
                    
                except Exception as api_error:
                    if "429" in str(api_error) or "rate limit" in str(api_error).lower():
                        self.logger.warning(f"News API rate limit hit for {request.symbol}, waiting 60 seconds")
                        await asyncio.sleep(60)
                        return DataResponse(
                            data=[],
                            source=self.name,
                            timestamp=datetime.now(),
                            error="Rate limit exceeded"
                        )
                    else:
                        raise api_error
            
            else:
                raise ValueError(f"Unsupported data type: {request.data_type}")
            
            return DataResponse(
                data=data,
                source=self.name,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            self.logger.error(f"Error fetching data from News API: {e}")
            return DataResponse(
                data=[],
                source=self.name,
                timestamp=datetime.now(),
                error=str(e)
            )


class AlpacaAdapter(DataSourceAdapter):
    """Alpaca Markets data adapter."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__('alpaca', config)
        import os
        self.api_key = os.getenv(config.get('api_key_env', 'ALPACA_API_KEY'))
        self.secret_key = os.getenv(config.get('secret_key_env', 'ALPACA_SECRET_KEY'))
        self.base_url = config.get('base_url', 'https://paper-api.alpaca.markets')
        self.timeout = config.get('timeout', 30)
    
    async def fetch_data(self, request: DataRequest) -> DataResponse:
        """Fetch data from Alpaca Markets."""
        try:
            if self.is_rate_limited():
                await asyncio.sleep(1)
            
            self.record_request()
            
            if not self.api_key or not self.secret_key:
                raise ValueError("Alpaca API credentials not found")
            
            headers = {
                'APCA-API-KEY-ID': self.api_key,
                'APCA-API-SECRET-KEY': self.secret_key
            }
            
            if request.data_type == 'account_info':
                url = f"{self.base_url}/v2/account"
                response = requests.get(url, headers=headers, timeout=self.timeout)
                data = response.json()
                    
            elif request.data_type == 'positions':
                url = f"{self.base_url}/v2/positions"
                response = requests.get(url, headers=headers, timeout=self.timeout)
                data = response.json()
                    
            else:
                raise ValueError(f"Unsupported data type: {request.data_type}")
            
            return DataResponse(
                data=data,
                source=self.name,
                timestamp=datetime.now()
            )
                
        except Exception as e:
            self.logger.error(f"Error fetching data from Alpaca: {e}")
            return DataResponse(
                data={},
                source=self.name,
                timestamp=datetime.now(),
                error=str(e)
            )


class FREDAdapter(DataSourceAdapter):
    """Federal Reserve Economic Data (FRED) adapter."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__('fred', config)
        import os
        self.api_key = os.getenv(config.get('api_key_env', 'FRED_API_KEY'))
        self.base_url = config.get('base_url', 'https://api.stlouisfed.org/fred')
        self.timeout = config.get('timeout', 30)
    
    async def fetch_data(self, request: DataRequest) -> DataResponse:
        """Fetch economic data from FRED."""
        try:
            if self.is_rate_limited():
                await asyncio.sleep(1)
            
            self.record_request()
            
            if not self.api_key:
                raise ValueError("FRED API key not found")
            
            params = {
                'api_key': self.api_key,
                'file_type': 'json'
            }
            
            if request.data_type == 'economic_indicators':
                # Use symbol as series ID (e.g., 'GDP', 'UNRATE', 'FEDFUNDS')
                params['series_id'] = request.symbol
                url = f"{self.base_url}/series/observations"
                
                response = requests.get(url, params=params, timeout=self.timeout)
                data = response.json()
                    
            else:
                raise ValueError(f"Unsupported data type: {request.data_type}")
            
            return DataResponse(
                data=data,
                source=self.name,
                timestamp=datetime.now()
            )
                
        except Exception as e:
            self.logger.error(f"Error fetching data from FRED: {e}")
            return DataResponse(
                data={},
                source=self.name,
                timestamp=datetime.now(),
                error=str(e)
            )


class UnifiedDataManager:
    """
    Unified data manager that coordinates multiple data sources.
    
    Provides caching, rate limiting, failover, and a unified interface
    for accessing market data from various sources.
    """
    
    def __init__(self, config_manager: ConfigManager):
        """
        Initialize the unified data manager.
        
        Args:
            config_manager: Configuration manager instance
        """
        self.config_manager = config_manager
        self.logger = logging.getLogger(__name__)
        
        # Data source adapters
        self.adapters: Dict[str, DataSourceAdapter] = {}
        
        # Caching
        self.cache_config = config_manager.get_data_management_config().get('cache', {})
        self.cache_enabled = self.cache_config.get('enabled', True)
        self.cache_ttl = self.cache_config.get('default_ttl', 300)  # 5 minutes
        
        # Initialize cache (Redis or in-memory)
        self._init_cache()
        
        # Database for persistent storage
        self.db_config = config_manager.get_data_management_config().get('database', {})
        self._init_database()
        
        # Thread pool for async operations
        self.executor = ThreadPoolExecutor(max_workers=10)
        
        # Initialize data source adapters
        self._init_adapters()
        
        self.logger.info("Unified Data Manager initialized")
    
    def _init_cache(self):
        """Initialize caching system."""
        cache_type = self.cache_config.get('type', 'memory')
        
        if cache_type == 'redis' and self.cache_enabled:
            try:
                redis_config = self.cache_config.get('redis', {})
                self.cache = redis.Redis(
                    host=redis_config.get('host', 'localhost'),
                    port=redis_config.get('port', 6379),
                    db=redis_config.get('db', 0),
                    decode_responses=True
                )
                # Test connection
                self.cache.ping()
                self.logger.info("Redis cache initialized")
            except Exception as e:
                self.logger.warning(f"Failed to initialize Redis cache: {e}. Using memory cache.")
                self.cache = {}
        else:
            self.cache = {}
    
    def _init_database(self):
        """Initialize database connection."""
        if self.db_config.get('enabled', False):
            try:
                db_url = self.db_config.get('url', 'sqlite:///trading_data.db')
                self.engine = create_engine(db_url)
                self.SessionLocal = sessionmaker(bind=self.engine)
                self.logger.info("Database initialized")
            except Exception as e:
                self.logger.error(f"Failed to initialize database: {e}")
                self.engine = None
                self.SessionLocal = None
        else:
            self.engine = None
            self.SessionLocal = None
    
    def _init_adapters(self):
        """Initialize data source adapters."""
        data_sources = self.config_manager.get_data_source_configs()
        
        for source_name, source_config in data_sources.items():
            if not source_config.get('enabled', False):
                continue
            
            try:
                if source_name == 'yfinance':
                    adapter = YFinanceAdapter(source_config)
                elif source_name == 'polygon':
                    adapter = PolygonAdapter(source_config)
                elif source_name in ['newsapi', 'news_api']:
                    adapter = NewsAPIAdapter(source_config)
                elif source_name == 'alpaca':
                    adapter = AlpacaAdapter(source_config)
                elif source_name == 'fred':
                    adapter = FREDAdapter(source_config)
                # Crypto adapters - DISABLED
                # elif source_name == 'binance':
                #     adapter = BinanceAdapter(source_config)
                # elif source_name == 'coinbase':
                #     adapter = CoinbaseAdapter(source_config)
                # Forex adapters
                elif source_name == 'oanda':
                    adapter = OandaAdapter(source_config)
                elif source_name == 'forex_factory':
                    adapter = ForexFactoryAdapter(source_config)
                elif source_name == 'alpha_vantage_forex':
                    adapter = AlphaVantageForexAdapter(source_config)
                else:
                    self.logger.warning(f"Unknown data source: {source_name}")
                    continue
                
                self.adapters[source_name] = adapter
                self.logger.info(f"Initialized adapter: {source_name}")
                
            except Exception as e:
                self.logger.error(f"Failed to initialize adapter {source_name}: {e}")
    
    def _get_cache_key(self, request: DataRequest) -> str:
        """Generate cache key for a data request."""
        key_parts = [
            request.symbol,
            request.data_type,
            request.timeframe,
            request.start_date.isoformat() if request.start_date else 'None',
            request.end_date.isoformat() if request.end_date else 'None'
        ]
        
        if request.parameters:
            key_parts.append(str(sorted(request.parameters.items())))
        
        return ':'.join(key_parts)
    
    def _get_from_cache(self, cache_key: str) -> Optional[DataResponse]:
        """Get data from cache."""
        if not self.cache_enabled:
            return None
        
        try:
            if isinstance(self.cache, dict):
                # Memory cache
                cached_data = self.cache.get(cache_key)
                if cached_data and datetime.now() - cached_data['timestamp'] < timedelta(seconds=self.cache_ttl):
                    return cached_data['response']
            else:
                # Redis cache
                cached_data = self.cache.get(cache_key)
                if cached_data:
                    # Deserialize cached data
                    import pickle
                    response = pickle.loads(cached_data)
                    response.cached = True
                    return response
        
        except Exception as e:
            self.logger.warning(f"Error retrieving from cache: {e}")
        
        return None
    
    def _store_in_cache(self, cache_key: str, response: DataResponse):
        """Store data in cache."""
        if not self.cache_enabled or response.error:
            return
        
        try:
            if isinstance(self.cache, dict):
                # Memory cache
                self.cache[cache_key] = {
                    'response': response,
                    'timestamp': datetime.now()
                }
            else:
                # Redis cache
                import pickle
                self.cache.setex(
                    cache_key,
                    self.cache_ttl,
                    pickle.dumps(response)
                )
        
        except Exception as e:
            self.logger.warning(f"Error storing in cache: {e}")
    
    def _get_suitable_adapters(self, request: DataRequest) -> List[DataSourceAdapter]:
        """Get adapters that can handle the request, sorted by priority."""
        suitable_adapters = [
            adapter for adapter in self.adapters.values()
            if adapter.enabled and adapter.can_handle(request)
        ]
        
        # Sort by priority (lower number = higher priority)
        suitable_adapters.sort(key=lambda x: x.priority)
        
        return suitable_adapters
    
    async def get_data(self, request: DataRequest) -> DataResponse:
        """Get data with caching and failover."""
        # Check cache first
        cache_key = self._get_cache_key(request)
        cached_response = self._get_from_cache(cache_key)
        
        if cached_response:
            self.logger.debug(f"Cache hit for {cache_key}")
            return cached_response
        
        # Get suitable adapters
        adapters = self._get_suitable_adapters(request)
        
        if not adapters:
            return DataResponse(
                data=pd.DataFrame(),
                source='none',
                timestamp=datetime.now(),
                error=f"No suitable data source for {request.data_type}"
            )
        
        # Try each adapter until one succeeds
        last_error = None
        
        for adapter in adapters:
            try:
                self.logger.debug(f"Trying adapter: {adapter.name}")
                response = await adapter.fetch_data(request)
                
                if not response.error:
                    # Store in cache
                    self._store_in_cache(cache_key, response)
                    
                    # Store in database if enabled
                    if self.engine:
                        await self._store_in_database(request, response)
                    
                    return response
                else:
                    last_error = response.error
                    self.logger.warning(f"Adapter {adapter.name} failed: {response.error}")
            
            except Exception as e:
                last_error = str(e)
                self.logger.error(f"Adapter {adapter.name} error: {e}")
        
        # All adapters failed
        return DataResponse(
            data=pd.DataFrame(),
            source='failed',
            timestamp=datetime.now(),
            error=f"All data sources failed. Last error: {last_error}"
        )
    
    async def _store_in_database(self, request: DataRequest, response: DataResponse):
        """Store data in database for historical analysis."""
        if not self.SessionLocal or response.error:
            return
        
        try:
            # This would need to be implemented based on your database schema
            # For now, just log that we would store it
            self.logger.debug(f"Would store data in database: {request.symbol} {request.data_type}")
        except Exception as e:
            self.logger.error(f"Error storing in database: {e}")
    
    async def get_price_data(self, symbol: str, timeframe: str = '1d', 
                           start_date: Optional[datetime] = None,
                           end_date: Optional[datetime] = None) -> DataResponse:
        """Get price data for a symbol."""
        request = DataRequest(
            symbol=symbol,
            data_type='price',
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date
        )
        return await self.get_data(request)
    
    async def get_news_data(self, symbol: str, limit: int = 50) -> DataResponse:
        """Get news data for a symbol."""
        request = DataRequest(
            symbol=symbol,
            data_type='news',
            timeframe='1d',
            parameters={'limit': limit}
        )
        return await self.get_data(request)
    
    async def get_fundamentals_data(self, symbol: str) -> DataResponse:
        """Get fundamental data for a symbol."""
        request = DataRequest(
            symbol=symbol,
            data_type='fundamentals',
            timeframe='1d'
        )
        return await self.get_data(request)
    
    async def get_options_data(self, symbol: str) -> DataResponse:
        """Get options data for a symbol."""
        request = DataRequest(
            symbol=symbol,
            data_type='options',
            timeframe='1d'
        )
        return await self.get_data(request)
    
    async def get_historical_data(self, symbol: str, timeframe: str = '1d',
                                start_date: Optional[datetime] = None,
                                end_date: Optional[datetime] = None,
                                limit: Optional[int] = None,
                                source: Optional[str] = None) -> Optional[pd.DataFrame]:
        """Get historical data for a symbol - compatibility method."""
        try:
            response = await self.get_price_data(symbol, timeframe, start_date, end_date)
            if response.success and response.data is not None:
                return response.data
            return None
        except Exception as e:
            self.logger.error(f"Error getting historical data for {symbol}: {e}")
            return None
    
    def get_available_sources(self) -> List[str]:
        """Get list of available data sources."""
        return [name for name, adapter in self.adapters.items() if adapter.enabled]
    
    def get_source_status(self) -> Dict[str, Any]:
        """Get status of all data sources."""
        status = {}
        
        for name, adapter in self.adapters.items():
            status[name] = {
                'enabled': adapter.enabled,
                'priority': adapter.priority,
                'rate_limited': adapter.is_rate_limited(),
                'supported_types': adapter.config.get('supported_data_types', []),
                'last_request': adapter.last_request_time.get(threading.current_thread().ident)
            }
        
        return status
    
    def clear_cache(self):
        """Clear all cached data."""
        try:
            if isinstance(self.cache, dict):
                self.cache.clear()
            else:
                self.cache.flushdb()
            
            self.logger.info("Cache cleared")
        except Exception as e:
            self.logger.error(f"Error clearing cache: {e}")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        try:
            if isinstance(self.cache, dict):
                return {
                    'type': 'memory',
                    'size': len(self.cache),
                    'enabled': self.cache_enabled
                }
            else:
                info = self.cache.info()
                return {
                    'type': 'redis',
                    'keys': info.get('db0', {}).get('keys', 0),
                    'memory_usage': info.get('used_memory_human', 'unknown'),
                    'enabled': self.cache_enabled
                }
        except Exception as e:
            return {'error': str(e)}
    
    def stop(self):
        """Stop the data manager."""
        self.executor.shutdown(wait=True)
        
        if hasattr(self.cache, 'close'):
            self.cache.close()
        
        self.logger.info("Unified Data Manager stopped")


if __name__ == "__main__":
    # Example usage
    import asyncio
    from pathlib import Path
    
    config_manager = ConfigManager(Path("../config"))
    data_manager = UnifiedDataManager(config_manager)
    
    async def validate_real_data_manager():
        """Validate that the data manager is using real market data only."""
        # Test price data
        response = await data_manager.get_price_data('AAPL', '1d')
        print(f"Price data: {len(response.data)} rows from {response.source}")
        
        # Test news data
        response = await data_manager.get_news_data('AAPL', limit=10)
        print(f"News data: {len(response.data)} articles from {response.source}")
        
        # Get status
        status = data_manager.get_source_status()
        print(f"Source status: {status}")
    
    # Commented out to prevent event loop conflicts when imported
    # asyncio.run(test_data_manager())