#!/usr/bin/env python3
"""
YFinance Optimizer - Enhanced Data Fetching with Connection Pooling and Batch Processing

This module provides optimized data fetching capabilities for yfinance API calls:
- Connection pooling and session reuse
- Intelligent batch processing for multiple symbols
- Rate limiting and retry logic
- Cache integration for optimal performance
- Performance monitoring and metrics

Features:
- YFinanceSessionManager: Manages HTTP sessions and connection pooling
- BatchDataDownloader: Efficient multi-symbol data fetching
- RateLimiter: Intelligent rate limiting with exponential backoff
- Performance metrics and monitoring
- Integration with existing cache system

Author: AI Trading System v2.0
Date: January 2025
"""

import asyncio
import logging
import time
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from collections import defaultdict, deque
import requests
import yfinance as yf
import pandas as pd
import numpy as np
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

try:
    from .cache_manager import CacheManager
except ImportError:
    # Fallback for direct execution
    import sys
    from pathlib import Path
    project_root = Path(__file__).parent.parent
    sys.path.append(str(project_root))
    from utils.cache_manager import CacheManager


@dataclass
class BatchRequest:
    """Batch data request specification"""
    symbols: List[str]
    period: str = "1y"
    interval: str = "1d"
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    data_type: str = "history"  # history, info, news, options
    priority: int = 1  # 1=high, 2=medium, 3=low


@dataclass
class BatchResult:
    """Batch processing result"""
    request: BatchRequest
    data: Dict[str, Any]
    success: bool
    error: Optional[str] = None
    fetch_time: float = 0.0
    cache_hit: bool = False


class RateLimiter:
    """Intelligent rate limiter with exponential backoff"""
    
    def __init__(self, max_requests_per_second: float = 2.0, max_requests_per_minute: int = 100):
        self.max_rps = max_requests_per_second
        self.max_rpm = max_requests_per_minute
        self.request_times = deque()
        self.minute_requests = deque()
        self.lock = threading.Lock()
        self.consecutive_errors = 0
        self.backoff_until = 0
        
    def wait_if_needed(self):
        """Wait if rate limit would be exceeded"""
        with self.lock:
            now = time.time()
            
            # Check if we're in backoff period
            if now < self.backoff_until:
                wait_time = self.backoff_until - now
                time.sleep(wait_time)
                return
            
            # Clean old requests
            while self.request_times and now - self.request_times[0] > 1.0:
                self.request_times.popleft()
            
            while self.minute_requests and now - self.minute_requests[0] > 60.0:
                self.minute_requests.popleft()
            
            # Check rate limits
            if len(self.request_times) >= self.max_rps:
                wait_time = 1.0 - (now - self.request_times[0])
                if wait_time > 0:
                    time.sleep(wait_time)
            
            if len(self.minute_requests) >= self.max_rpm:
                wait_time = 60.0 - (now - self.minute_requests[0])
                if wait_time > 0:
                    time.sleep(wait_time)
            
            # Record this request
            now = time.time()
            self.request_times.append(now)
            self.minute_requests.append(now)
    
    def record_error(self):
        """Record an error for backoff calculation"""
        with self.lock:
            self.consecutive_errors += 1
            # Exponential backoff: 2^errors seconds, max 60 seconds
            backoff_time = min(2 ** self.consecutive_errors, 60)
            self.backoff_until = time.time() + backoff_time
    
    def record_success(self):
        """Record a successful request"""
        with self.lock:
            self.consecutive_errors = 0
            self.backoff_until = 0


class YFinanceSessionManager:
    """Manages HTTP sessions and connection pooling for yfinance"""
    
    def __init__(self, max_connections: int = 10, max_retries: int = 3):
        self.max_connections = max_connections
        self.max_retries = max_retries
        self.session = None
        self.lock = threading.Lock()
        self.logger = logging.getLogger(__name__)
        self._setup_session()
        
    def _setup_session(self):
        """Setup HTTP session with connection pooling"""
        self.session = requests.Session()
        
        # Configure retry strategy
        retry_strategy = Retry(
            total=self.max_retries,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS"]
        )
        
        # Configure HTTP adapter with connection pooling
        adapter = HTTPAdapter(
            pool_connections=self.max_connections,
            pool_maxsize=self.max_connections,
            max_retries=retry_strategy
        )
        
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        # Set headers
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        
        self.logger.info(f"YFinance session manager initialized with {self.max_connections} connections")
    
    def get_session(self) -> requests.Session:
        """Get the configured session"""
        with self.lock:
            if self.session is None:
                self._setup_session()
            return self.session
    
    def close(self):
        """Close the session"""
        with self.lock:
            if self.session:
                self.session.close()
                self.session = None


class BatchDataDownloader:
    """Efficient batch data downloader for yfinance"""
    
    def __init__(self, cache_manager: Optional[CacheManager] = None, 
                 max_workers: int = 4, enable_caching: bool = True):
        self.cache_manager = cache_manager or CacheManager()
        self.max_workers = max_workers
        self.enable_caching = enable_caching
        self.session_manager = YFinanceSessionManager()
        self.rate_limiter = RateLimiter()
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.logger = logging.getLogger(__name__)
        
        # Performance metrics
        self.metrics = {
            'total_requests': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'api_calls': 0,
            'errors': 0,
            'total_fetch_time': 0.0,
            'average_fetch_time': 0.0
        }
        
        self.logger.info(f"Batch data downloader initialized with {max_workers} workers")
    
    def _generate_cache_key(self, request: BatchRequest) -> str:
        """Generate cache key for batch request"""
        key_data = f"{'-'.join(sorted(request.symbols))}_{request.period}_{request.interval}_{request.data_type}"
        if request.start_date:
            key_data += f"_{request.start_date.strftime('%Y%m%d')}"
        if request.end_date:
            key_data += f"_{request.end_date.strftime('%Y%m%d')}"
        
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _get_cached_data(self, request: BatchRequest) -> Optional[Dict[str, Any]]:
        """Get cached data if available"""
        if not self.enable_caching:
            return None
            
        try:
            cache_key = self._generate_cache_key(request)
            cached_data = self.cache_manager.get(cache_key)
            
            if cached_data:
                self.metrics['cache_hits'] += 1
                return cached_data
            else:
                self.metrics['cache_misses'] += 1
                return None
                
        except Exception as e:
            self.logger.error(f"Cache retrieval error: {e}")
            return None
    
    def _set_cached_data(self, request: BatchRequest, data: Dict[str, Any], ttl: int = 300):
        """Cache the fetched data"""
        if not self.enable_caching:
            return
            
        try:
            cache_key = self._generate_cache_key(request)
            self.cache_manager.set(cache_key, data, ttl)
        except Exception as e:
            self.logger.error(f"Cache storage error: {e}")
    
    def _fetch_single_batch(self, request: BatchRequest) -> BatchResult:
        """Fetch data for a single batch request"""
        start_time = time.time()
        
        try:
            # Check cache first
            cached_data = self._get_cached_data(request)
            if cached_data:
                return BatchResult(
                    request=request,
                    data=cached_data,
                    success=True,
                    fetch_time=time.time() - start_time,
                    cache_hit=True
                )
            
            # Apply rate limiting
            self.rate_limiter.wait_if_needed()
            
            # Fetch data based on request type
            data = {}
            
            if request.data_type == "history":
                data = self._fetch_history_batch(request)
            elif request.data_type == "info":
                data = self._fetch_info_batch(request)
            elif request.data_type == "news":
                data = self._fetch_news_batch(request)
            elif request.data_type == "options":
                data = self._fetch_options_batch(request)
            else:
                raise ValueError(f"Unsupported data type: {request.data_type}")
            
            # Cache the result
            self._set_cached_data(request, data)
            
            # Record success
            self.rate_limiter.record_success()
            self.metrics['api_calls'] += 1
            
            fetch_time = time.time() - start_time
            self.metrics['total_fetch_time'] += fetch_time
            
            return BatchResult(
                request=request,
                data=data,
                success=True,
                fetch_time=fetch_time
            )
            
        except Exception as e:
            self.logger.error(f"Batch fetch error: {e}")
            self.rate_limiter.record_error()
            self.metrics['errors'] += 1
            
            return BatchResult(
                request=request,
                data={},
                success=False,
                error=str(e),
                fetch_time=time.time() - start_time
            )
    
    def _fetch_history_batch(self, request: BatchRequest) -> Dict[str, pd.DataFrame]:
        """Fetch historical data for multiple symbols"""
        data = {}
        
        if len(request.symbols) == 1:
            # Single symbol
            symbol = request.symbols[0]
            ticker = yf.Ticker(symbol, session=self.session_manager.get_session())
            
            if request.start_date and request.end_date:
                hist_data = ticker.history(
                    start=request.start_date,
                    end=request.end_date,
                    interval=request.interval
                )
            else:
                hist_data = ticker.history(
                    period=request.period,
                    interval=request.interval
                )
            
            data[symbol] = hist_data
            
        else:
            # Multiple symbols - use yf.download for efficiency
            symbols_str = " ".join(request.symbols)
            
            if request.start_date and request.end_date:
                hist_data = yf.download(
                    symbols_str,
                    start=request.start_date,
                    end=request.end_date,
                    interval=request.interval,
                    group_by='ticker',
                    progress=False,
                    session=self.session_manager.get_session()
                )
            else:
                hist_data = yf.download(
                    symbols_str,
                    period=request.period,
                    interval=request.interval,
                    group_by='ticker',
                    progress=False,
                    session=self.session_manager.get_session()
                )
            
            # Parse multi-symbol data
            if len(request.symbols) > 1 and not hist_data.empty:
                for symbol in request.symbols:
                    try:
                        if symbol in hist_data.columns.levels[0]:
                            data[symbol] = hist_data[symbol]
                        else:
                            data[symbol] = pd.DataFrame()
                    except (AttributeError, KeyError):
                        data[symbol] = pd.DataFrame()
            else:
                # Single symbol result
                data[request.symbols[0]] = hist_data
        
        return data
    
    def _fetch_info_batch(self, request: BatchRequest) -> Dict[str, Dict]:
        """Fetch company info for multiple symbols"""
        data = {}
        
        for symbol in request.symbols:
            try:
                ticker = yf.Ticker(symbol, session=self.session_manager.get_session())
                data[symbol] = ticker.info
            except Exception as e:
                self.logger.error(f"Error fetching info for {symbol}: {e}")
                data[symbol] = {}
        
        return data
    
    def _fetch_news_batch(self, request: BatchRequest) -> Dict[str, List]:
        """Fetch news for multiple symbols"""
        data = {}
        
        for symbol in request.symbols:
            try:
                ticker = yf.Ticker(symbol, session=self.session_manager.get_session())
                data[symbol] = ticker.news
            except Exception as e:
                self.logger.error(f"Error fetching news for {symbol}: {e}")
                data[symbol] = []
        
        return data
    
    def _fetch_options_batch(self, request: BatchRequest) -> Dict[str, Dict]:
        """Fetch options data for multiple symbols"""
        data = {}
        
        for symbol in request.symbols:
            try:
                ticker = yf.Ticker(symbol, session=self.session_manager.get_session())
                options_dates = ticker.options
                
                if options_dates:
                    option_chain = ticker.option_chain(options_dates[0])
                    data[symbol] = {
                        'calls': option_chain.calls,
                        'puts': option_chain.puts,
                        'expirations': options_dates
                    }
                else:
                    data[symbol] = {'calls': pd.DataFrame(), 'puts': pd.DataFrame(), 'expirations': []}
                    
            except Exception as e:
                self.logger.error(f"Error fetching options for {symbol}: {e}")
                data[symbol] = {'calls': pd.DataFrame(), 'puts': pd.DataFrame(), 'expirations': []}
        
        return data
    
    async def fetch_batch_async(self, requests: List[BatchRequest]) -> List[BatchResult]:
        """Fetch multiple batch requests asynchronously"""
        self.metrics['total_requests'] += len(requests)
        
        # Sort requests by priority
        sorted_requests = sorted(requests, key=lambda x: x.priority)
        
        # Submit tasks to thread pool
        loop = asyncio.get_event_loop()
        tasks = []
        
        for request in sorted_requests:
            task = loop.run_in_executor(self.executor, self._fetch_single_batch, request)
            tasks.append(task)
        
        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        batch_results = []
        for result in results:
            if isinstance(result, Exception):
                self.logger.error(f"Batch processing error: {result}")
                self.metrics['errors'] += 1
            else:
                batch_results.append(result)
        
        # Update metrics
        if self.metrics['total_requests'] > 0:
            self.metrics['average_fetch_time'] = (
                self.metrics['total_fetch_time'] / self.metrics['total_requests']
            )
        
        return batch_results
    
    def fetch_batch_sync(self, requests: List[BatchRequest]) -> List[BatchResult]:
        """Fetch multiple batch requests synchronously"""
        self.metrics['total_requests'] += len(requests)
        
        # Sort requests by priority
        sorted_requests = sorted(requests, key=lambda x: x.priority)
        
        # Submit tasks to thread pool
        futures = []
        for request in sorted_requests:
            future = self.executor.submit(self._fetch_single_batch, request)
            futures.append(future)
        
        # Collect results
        results = []
        for future in as_completed(futures):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                self.logger.error(f"Batch processing error: {e}")
                self.metrics['errors'] += 1
        
        # Update metrics
        if self.metrics['total_requests'] > 0:
            self.metrics['average_fetch_time'] = (
                self.metrics['total_fetch_time'] / self.metrics['total_requests']
            )
        
        return results
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        cache_hit_rate = 0.0
        if self.metrics['total_requests'] > 0:
            cache_hit_rate = self.metrics['cache_hits'] / self.metrics['total_requests']
        
        return {
            'total_requests': self.metrics['total_requests'],
            'cache_hits': self.metrics['cache_hits'],
            'cache_misses': self.metrics['cache_misses'],
            'cache_hit_rate': cache_hit_rate,
            'api_calls': self.metrics['api_calls'],
            'api_calls_saved': self.metrics['cache_hits'],
            'errors': self.metrics['errors'],
            'error_rate': self.metrics['errors'] / max(self.metrics['total_requests'], 1),
            'total_fetch_time': self.metrics['total_fetch_time'],
            'average_fetch_time': self.metrics['average_fetch_time']
        }
    
    def reset_metrics(self):
        """Reset performance metrics"""
        self.metrics = {
            'total_requests': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'api_calls': 0,
            'errors': 0,
            'total_fetch_time': 0.0,
            'average_fetch_time': 0.0
        }
    
    def close(self):
        """Close the downloader and cleanup resources"""
        self.executor.shutdown(wait=True)
        self.session_manager.close()


# Convenience functions for easy integration
async def fetch_multiple_symbols_async(symbols: List[str], period: str = "1y", 
                                     interval: str = "1d", data_type: str = "history",
                                     cache_manager: Optional[CacheManager] = None) -> Dict[str, Any]:
    """Convenience function for fetching multiple symbols asynchronously"""
    downloader = BatchDataDownloader(cache_manager=cache_manager)
    
    try:
        request = BatchRequest(
            symbols=symbols,
            period=period,
            interval=interval,
            data_type=data_type
        )
        
        results = await downloader.fetch_batch_async([request])
        
        if results and results[0].success:
            return results[0].data
        else:
            return {}
            
    finally:
        downloader.close()


def fetch_multiple_symbols_sync(symbols: List[str], period: str = "1y", 
                               interval: str = "1d", data_type: str = "history",
                               cache_manager: Optional[CacheManager] = None) -> Dict[str, Any]:
    """Convenience function for fetching multiple symbols synchronously"""
    downloader = BatchDataDownloader(cache_manager=cache_manager)
    
    try:
        request = BatchRequest(
            symbols=symbols,
            period=period,
            interval=interval,
            data_type=data_type
        )
        
        results = downloader.fetch_batch_sync([request])
        
        if results and results[0].success:
            return results[0].data
        else:
            return {}
            
    finally:
        downloader.close()


if __name__ == "__main__":
    # Test the batch downloader
    import asyncio
    
    async def test_batch_downloader():
        """Test the batch data downloader"""
        symbols = ["AAPL", "GOOGL", "MSFT", "TSLA", "NVDA"]
        
        # Test historical data
        print("Testing batch historical data fetching...")
        data = await fetch_multiple_symbols_async(symbols, period="1mo", interval="1d")
        
        for symbol, df in data.items():
            if not df.empty:
                print(f"{symbol}: {len(df)} rows, latest close: ${df['Close'].iloc[-1]:.2f}")
            else:
                print(f"{symbol}: No data")
        
        # Test with batch downloader directly
        print("\nTesting BatchDataDownloader directly...")
        downloader = BatchDataDownloader()
        
        requests = [
            BatchRequest(symbols=["AAPL", "GOOGL"], period="5d", data_type="history"),
            BatchRequest(symbols=["MSFT", "TSLA"], period="5d", data_type="info"),
        ]
        
        results = await downloader.fetch_batch_async(requests)
        
        for result in results:
            print(f"Request: {result.request.symbols}, Success: {result.success}, "
                  f"Cache hit: {result.cache_hit}, Time: {result.fetch_time:.2f}s")
        
        # Print performance metrics
        metrics = downloader.get_performance_metrics()
        print(f"\nPerformance Metrics:")
        for key, value in metrics.items():
            print(f"  {key}: {value}")
        
        downloader.close()
    
    asyncio.run(test_batch_downloader())