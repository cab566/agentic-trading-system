#!/usr/bin/env python3
"""
Unified Data Manager

Provides centralized data access and management for the trading system.
Handles multiple data sources, caching, and data validation.

Author: AI Trading System v2.0
Date: January 2025
"""

import asyncio
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
import yfinance as yf
from dataclasses import dataclass
from enum import Enum
import json
import sqlite3
from pathlib import Path

try:
    from ..utils.yfinance_optimizer import BatchDataDownloader, BatchRequest, fetch_multiple_symbols_async
    from ..utils.cache_manager import CacheManager
except ImportError:
    # Fallback for direct execution
    import sys
    project_root = Path(__file__).parent.parent
    sys.path.append(str(project_root))
    from utils.yfinance_optimizer import BatchDataDownloader, BatchRequest, fetch_multiple_symbols_async
    from utils.cache_manager import CacheManager

# Setup logging
logger = logging.getLogger(__name__)

class DataSource(Enum):
    """Available data sources"""
    YAHOO_FINANCE = "yahoo_finance"
    ALPACA = "alpaca"
    POLYGON = "polygon"
    ALPHA_VANTAGE = "alpha_vantage"
    CACHE = "cache"

@dataclass
class DataRequest:
    """Data request specification"""
    symbols: List[str]
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    interval: str = "1d"  # 1m, 5m, 15m, 30m, 1h, 1d
    data_type: str = "ohlcv"  # ohlcv, news, fundamentals
    source: DataSource = DataSource.YAHOO_FINANCE

class UnifiedDataManager:
    """
    Unified data manager for centralized data access
    """
    
    def __init__(self, config_manager=None):
        """Initialize the unified data manager"""
        self.config_manager = config_manager
        self.logger = logging.getLogger(__name__)
        self.cache_dir = Path("data/cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize optimized batch downloader
        self.cache_manager = CacheManager()
        self.batch_downloader = BatchDataDownloader(
            cache_manager=self.cache_manager,
            max_workers=4,
            enable_caching=True
        )
        
        # Data source configurations
        self.data_sources = {
            DataSource.YAHOO_FINANCE: self._yahoo_finance_handler,
            DataSource.CACHE: self._cache_handler
        }
        
        # Cache settings
        self.cache_enabled = True
        self.cache_ttl_minutes = 15  # Cache time-to-live
        
        self.logger.info("Unified Data Manager initialized with optimized batch processing")
    
    async def get_market_data(self, request: DataRequest) -> pd.DataFrame:
        """
        Get market data based on request specification
        
        Args:
            request: DataRequest object specifying what data to fetch
            
        Returns:
            DataFrame with requested market data
        """
        try:
            # Check cache first if enabled
            if self.cache_enabled:
                cached_data = await self._get_cached_data(request)
                if cached_data is not None:
                    self.logger.debug(f"Returning cached data for {request.symbols}")
                    return cached_data
            
            # Fetch from primary source
            handler = self.data_sources.get(request.source, self._yahoo_finance_handler)
            data = await handler(request)
            
            # Cache the result if enabled
            if self.cache_enabled and data is not None and not data.empty:
                await self._cache_data(request, data)
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error fetching market data: {e}")
            return pd.DataFrame()
    
    async def get_real_time_quote(self, symbol: str) -> Dict[str, Any]:
        """
        Get real-time quote for a symbol
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Dictionary with current quote data
        """
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            # Get current price from fast_info if available
            try:
                fast_info = ticker.fast_info
                current_price = fast_info.get('lastPrice', info.get('currentPrice', 0))
            except:
                current_price = info.get('currentPrice', 0)
            
            quote = {
                'symbol': symbol,
                'price': current_price,
                'change': info.get('regularMarketChange', 0),
                'change_percent': info.get('regularMarketChangePercent', 0),
                'volume': info.get('regularMarketVolume', 0),
                'market_cap': info.get('marketCap', 0),
                'timestamp': datetime.now()
            }
            
            return quote
            
        except Exception as e:
            self.logger.error(f"Error fetching real-time quote for {symbol}: {e}")
            return {}
    
    async def get_historical_data(self, symbol: str, period: str = "1y", 
                                interval: str = "1d") -> pd.DataFrame:
        """
        Get historical data for a symbol
        
        Args:
            symbol: Stock symbol
            period: Time period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            interval: Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
            
        Returns:
            DataFrame with historical OHLCV data
        """
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period, interval=interval)
            
            if not data.empty:
                # Standardize column names
                data.columns = [col.lower().replace(' ', '_') for col in data.columns]
                data['symbol'] = symbol
                
            return data
            
        except Exception as e:
            self.logger.error(f"Error fetching historical data for {symbol}: {e}")
            return pd.DataFrame()
    
    async def get_multiple_quotes(self, symbols: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Get real-time quotes for multiple symbols using optimized batch processing
        
        Args:
            symbols: List of stock symbols
            
        Returns:
            Dictionary mapping symbols to quote data
        """
        try:
            # Use the optimized batch downloader for real-time quotes
            batch_request = BatchRequest(
                symbols=symbols,
                data_type="quote",
                period="1d",
                interval="1d"
            )
            
            results = await self.batch_downloader.fetch_batch_data([batch_request])
            
            # Convert results to expected format
            quotes = {}
            for symbol in symbols:
                if symbol in results and not results[symbol].empty:
                    data = results[symbol].iloc[-1]  # Get latest data point
                    quotes[symbol] = {
                        'symbol': symbol,
                        'price': float(data.get('close', 0)),
                        'change': float(data.get('close', 0) - data.get('open', 0)),
                        'change_percent': float((data.get('close', 0) - data.get('open', 0)) / data.get('open', 1) * 100),
                        'volume': int(data.get('volume', 0)),
                        'high': float(data.get('high', 0)),
                        'low': float(data.get('low', 0)),
                        'open': float(data.get('open', 0)),
                        'timestamp': datetime.now().isoformat()
                    }
                else:
                    quotes[symbol] = {}
                    
            return quotes
            
        except Exception as e:
            self.logger.error(f"Error in optimized batch quote fetch: {e}")
            # Fallback to original method
            return await self._get_multiple_quotes_fallback(symbols)
    
    async def _get_multiple_quotes_fallback(self, symbols: List[str]) -> Dict[str, Dict[str, Any]]:
        """Fallback method for getting multiple quotes"""
        quotes = {}
        
        # Process in batches to avoid rate limits
        batch_size = 10
        for i in range(0, len(symbols), batch_size):
            batch = symbols[i:i + batch_size]
            
            # Create tasks for concurrent execution
            tasks = [self.get_real_time_quote(symbol) for symbol in batch]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            for symbol, result in zip(batch, results):
                if isinstance(result, Exception):
                    self.logger.error(f"Error fetching quote for {symbol}: {result}")
                    quotes[symbol] = {}
                else:
                    quotes[symbol] = result
        
        return quotes
    
    async def get_market_overview(self) -> Dict[str, Any]:
        """
        Get market overview data
        
        Returns:
            Dictionary with market overview information
        """
        try:
            # Major indices
            indices = ['^GSPC', '^DJI', '^IXIC', '^RUT']  # S&P 500, Dow, Nasdaq, Russell 2000
            index_data = {}
            
            for index in indices:
                quote = await self.get_real_time_quote(index)
                if quote:
                    index_data[index] = quote
            
            # Market sectors (using sector ETFs)
            sectors = {
                'Technology': 'XLK',
                'Healthcare': 'XLV',
                'Financials': 'XLF',
                'Energy': 'XLE',
                'Consumer Discretionary': 'XLY',
                'Industrials': 'XLI',
                'Consumer Staples': 'XLP',
                'Utilities': 'XLU',
                'Materials': 'XLB',
                'Real Estate': 'XLRE',
                'Communication Services': 'XLC'
            }
            
            sector_data = {}
            for sector_name, etf_symbol in sectors.items():
                quote = await self.get_real_time_quote(etf_symbol)
                if quote:
                    sector_data[sector_name] = quote
            
            overview = {
                'indices': index_data,
                'sectors': sector_data,
                'timestamp': datetime.now(),
                'market_status': self._get_market_status()
            }
            
            return overview
            
        except Exception as e:
            self.logger.error(f"Error fetching market overview: {e}")
            return {}
    
    def _get_market_status(self) -> str:
        """
        Determine current market status
        
        Returns:
            Market status string
        """
        now = datetime.now()
        
        # Simple market hours check (US Eastern Time approximation)
        # This is a simplified version - in production, use proper timezone handling
        hour = now.hour
        weekday = now.weekday()
        
        # Weekend
        if weekday >= 5:  # Saturday = 5, Sunday = 6
            return "closed"
        
        # Weekday market hours (9:30 AM - 4:00 PM ET, approximated as local time)
        if 9 <= hour < 16:
            return "open"
        elif 4 <= hour < 9:
            return "pre_market"
        elif 16 <= hour < 20:
            return "after_hours"
        else:
            return "closed"
    
    async def _yahoo_finance_handler(self, request: DataRequest) -> pd.DataFrame:
        """
        Handle Yahoo Finance data requests using optimized batch processing
        
        Args:
            request: Data request object
            
        Returns:
            DataFrame with requested data
        """
        try:
            # Use the optimized batch downloader for historical data
            batch_request = BatchRequest(
                symbols=request.symbols,
                data_type=request.data_type,
                period="1y" if not request.start_date else None,
                interval=request.interval,
                start_date=request.start_date,
                end_date=request.end_date
            )
            
            results = await self.batch_downloader.fetch_batch_data([batch_request])
            
            # Combine results from all symbols
            combined_data = []
            for symbol in request.symbols:
                if symbol in results and not results[symbol].empty:
                    symbol_data = results[symbol].copy()
                    symbol_data['symbol'] = symbol
                    combined_data.append(symbol_data)
            
            if combined_data:
                return pd.concat(combined_data, ignore_index=True)
            else:
                return pd.DataFrame()
                
        except Exception as e:
            self.logger.error(f"Error in optimized Yahoo Finance handler: {e}")
            # Fallback to original method
            return await self._yahoo_finance_handler_fallback(request)
    
    async def _yahoo_finance_handler_fallback(self, request: DataRequest) -> pd.DataFrame:
        """Fallback Yahoo Finance handler using original yfinance approach"""
        try:
            all_data = []
            
            for symbol in request.symbols:
                ticker = yf.Ticker(symbol)
                
                if request.start_date and request.end_date:
                    data = ticker.history(
                        start=request.start_date,
                        end=request.end_date,
                        interval=request.interval
                    )
                else:
                    data = ticker.history(period="1y", interval=request.interval)
                
                if not data.empty:
                    # Standardize column names
                    data.columns = [col.lower().replace(' ', '_') for col in data.columns]
                    data['symbol'] = symbol
                    data.reset_index(inplace=True)
                    all_data.append(data)
            
            if all_data:
                return pd.concat(all_data, ignore_index=True)
            else:
                return pd.DataFrame()
                
        except Exception as e:
            self.logger.error(f"Error in Yahoo Finance fallback handler: {e}")
            return pd.DataFrame()
    
    async def _get_cached_data(self, request: DataRequest) -> Optional[pd.DataFrame]:
        """Get data from cache if available and fresh"""
        try:
            cache_key = self._generate_cache_key(request)
            cache_file = self.cache_dir / f"{cache_key}.json"
            
            if cache_file.exists():
                # Check if cache is still fresh
                file_age = datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
                if file_age.total_seconds() < (self.cache_ttl_minutes * 60):
                    
                    with open(cache_file, 'r') as f:
                        cache_data = json.load(f)
                    
                    # Convert back to DataFrame
                    df = pd.DataFrame(cache_data['data'])
                    if 'index' in cache_data:
                        df.index = pd.to_datetime(cache_data['index'])
                    
                    return df
            
            return None
            
        except Exception as e:
            self.logger.error(f"Cache retrieval error: {e}")
            return None
    
    async def _cache_data(self, request: DataRequest, data: pd.DataFrame):
        """Cache data for future use"""
        try:
            cache_key = self._generate_cache_key(request)
            cache_file = self.cache_dir / f"{cache_key}.json"
            
            # Prepare data for JSON serialization
            cache_data = {
                'data': data.to_dict('records'),
                'index': data.index.strftime('%Y-%m-%d %H:%M:%S').tolist() if hasattr(data.index, 'strftime') else None,
                'timestamp': datetime.now().isoformat(),
                'request': {
                    'symbols': request.symbols,
                    'interval': request.interval,
                    'data_type': request.data_type
                }
            }
            
            with open(cache_file, 'w') as f:
                json.dump(cache_data, f, default=str)
                
        except Exception as e:
            self.logger.error(f"Cache storage error: {e}")
    
    def _generate_cache_key(self, request: DataRequest) -> str:
        """Generate a unique cache key for the request"""
        symbols_str = "_".join(sorted(request.symbols))
        return f"{symbols_str}_{request.interval}_{request.data_type}"
    
    async def _cache_handler(self, request: DataRequest) -> pd.DataFrame:
        """Handle cache-only requests"""
        return await self._get_cached_data(request) or pd.DataFrame()
    
    def get_data_source_status(self) -> Dict[str, str]:
        """
        Get status of all data sources
        
        Returns:
            Dictionary mapping data sources to their status
        """
        status = {}
        
        # Test Yahoo Finance
        try:
            test_ticker = yf.Ticker("AAPL")
            test_data = test_ticker.history(period="1d")
            status[DataSource.YAHOO_FINANCE.value] = "available" if not test_data.empty else "unavailable"
        except:
            status[DataSource.YAHOO_FINANCE.value] = "unavailable"
        
        # Cache is always available if directory exists
        status[DataSource.CACHE.value] = "available" if self.cache_dir.exists() else "unavailable"
        
        return status
    
    def clear_cache(self):
        """Clear all cached data"""
        try:
            for cache_file in self.cache_dir.glob("*.json"):
                cache_file.unlink()
            self.logger.info("Cache cleared successfully")
        except Exception as e:
            self.logger.error(f"Error clearing cache: {e}")
    
    def get_cache_info(self) -> Dict[str, Any]:
        """
        Get information about cached data
        
        Returns:
            Dictionary with cache statistics
        """
        try:
            cache_files = list(self.cache_dir.glob("*.json"))
            total_size = sum(f.stat().st_size for f in cache_files)
            
            return {
                'cache_enabled': self.cache_enabled,
                'cache_directory': str(self.cache_dir),
                'cached_files': len(cache_files),
                'total_size_bytes': total_size,
                'total_size_mb': round(total_size / (1024 * 1024), 2),
                'cache_ttl_minutes': self.cache_ttl_minutes
            }
            
        except Exception as e:
            self.logger.error(f"Error getting cache info: {e}")
            return {}

# Convenience functions for backward compatibility
async def get_market_data(symbols: List[str], period: str = "1y", 
                         interval: str = "1d") -> pd.DataFrame:
    """
    Convenience function to get market data
    
    Args:
        symbols: List of stock symbols
        period: Time period
        interval: Data interval
        
    Returns:
        DataFrame with market data
    """
    manager = UnifiedDataManager()
    request = DataRequest(
        symbols=symbols,
        interval=interval,
        data_type="ohlcv"
    )
    return await manager.get_market_data(request)

async def get_real_time_quotes(symbols: List[str]) -> Dict[str, Dict[str, Any]]:
    """
    Convenience function to get real-time quotes
    
    Args:
        symbols: List of stock symbols
        
    Returns:
        Dictionary mapping symbols to quote data
    """
    manager = UnifiedDataManager()
    return await manager.get_multiple_quotes(symbols)

if __name__ == "__main__":
    # Test the unified data manager
    async def test_manager():
        manager = UnifiedDataManager()
        
        # Test single quote
        quote = await manager.get_real_time_quote("AAPL")
        print(f"AAPL Quote: {quote}")
        
        # Test historical data
        data = await manager.get_historical_data("AAPL", period="5d")
        print(f"AAPL Historical Data Shape: {data.shape}")
        
        # Test market overview
        overview = await manager.get_market_overview()
        print(f"Market Overview Keys: {list(overview.keys())}")
        
        # Test data source status
        status = manager.get_data_source_status()
        print(f"Data Source Status: {status}")
    
    asyncio.run(test_manager())