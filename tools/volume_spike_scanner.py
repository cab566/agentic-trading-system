#!/usr/bin/env python3
"""
Volume Spike Scanner - Real-Time Market Discovery Tool

This tool monitors unusual trading volume across a broad universe of stocks
to identify potential trading opportunities beyond the "big 7 tickers".

Features:
- Real-time volume spike detection across S&P 500, NASDAQ 100, and Russell 2000
- Multiple volume spike criteria (1.5x, 2x, 3x average volume)
- Price movement correlation analysis
- Sector and market cap filtering
- Real-time alerts and notifications
- Integration with existing agent system
- Multi-level caching for performance optimization

Data Sources:
- Yahoo Finance for real-time data
- Alpha Vantage for enhanced volume metrics
- Polygon.io for institutional-grade data (if available)
- No synthetic or mock data

Author: AI Trading System v2.0
Date: January 2025
"""

import asyncio
import logging
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import aiohttp
import json
from concurrent.futures import ThreadPoolExecutor
import time
import hashlib

try:
    from ..core.config_manager import ConfigManager
    from ..utils.cache_manager import CacheManager
    from ..utils.notifications import NotificationManager
    from ..utils.yfinance_optimizer import BatchDataDownloader, BatchRequest
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


class VolumeSpikeSeverity(Enum):
    """Volume spike severity levels."""
    MODERATE = "moderate"  # 1.5x - 2x average volume
    HIGH = "high"         # 2x - 3x average volume
    EXTREME = "extreme"   # 3x+ average volume


class MarketCapCategory(Enum):
    """Market capitalization categories."""
    MEGA_CAP = "mega_cap"      # $200B+
    LARGE_CAP = "large_cap"    # $10B - $200B
    MID_CAP = "mid_cap"        # $2B - $10B
    SMALL_CAP = "small_cap"    # $300M - $2B
    MICRO_CAP = "micro_cap"    # <$300M


@dataclass
class VolumeSpike:
    """Volume spike detection result."""
    symbol: str
    current_volume: int
    average_volume: float
    volume_ratio: float
    severity: VolumeSpikeSeverity
    price_change_pct: float
    current_price: float
    market_cap: Optional[float]
    sector: Optional[str]
    timestamp: datetime
    confidence_score: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'symbol': self.symbol,
            'current_volume': self.current_volume,
            'average_volume': self.average_volume,
            'volume_ratio': self.volume_ratio,
            'severity': self.severity.value,
            'price_change_pct': self.price_change_pct,
            'current_price': self.current_price,
            'market_cap': self.market_cap,
            'sector': self.sector,
            'timestamp': self.timestamp.isoformat(),
            'confidence_score': self.confidence_score
        }


@dataclass
class ScannerConfig:
    """Configuration for volume spike scanner."""
    min_volume_ratio: float = 1.5
    min_price_change: float = 2.0  # Minimum price change percentage
    max_price: float = 1000.0      # Maximum stock price to consider
    min_market_cap: float = 100_000_000  # $100M minimum market cap
    excluded_sectors: List[str] = field(default_factory=lambda: ['Utilities'])
    scan_interval_seconds: int = 300  # 5 minutes
    lookback_days: int = 20        # Days for average volume calculation
    max_results_per_scan: int = 50
    enable_real_time: bool = True
    # Cache configuration
    cache_ttl_historical: int = 3600  # 1 hour for historical data
    cache_ttl_ticker_info: int = 86400  # 24 hours for ticker info
    cache_ttl_current_data: int = 60  # 1 minute for current data
    enable_caching: bool = True


class VolumeSpikeScanner:
    """
    Advanced volume spike scanner with multi-level caching for performance optimization.
    
    Caching Strategy:
    - Historical data: 1 hour TTL (intraday updates not critical for averages)
    - Ticker info: 24 hour TTL (market cap, sector rarely change)
    - Current data: 1 minute TTL (balance between freshness and performance)
    - Volume calculations: 5 minute TTL (derived metrics)
    """
    
    def __init__(self, config_manager: ConfigManager):
        self.config_manager = config_manager
        self.logger = logging.getLogger(__name__)
        
        # Initialize cache manager
        self.cache_manager = CacheManager()
        
        # Initialize notification manager
        self.notification_manager = NotificationManager(config_manager)
        
        # Initialize batch data downloader for optimized yfinance operations
        self.batch_downloader = BatchDataDownloader(
            cache_manager=self.cache_manager,
            max_workers=4,
            enable_caching=True
        )
        
        # Load configuration
        self.config = self._load_config()
        
        # Initialize stock universe
        self.stock_universe = self._initialize_stock_universe()
        
        # Scanner state
        self.is_scanning = False
        self.last_scan_time = None
        self.scan_count = 0
        self.total_spikes_found = 0
        
        # Performance metrics
        self.cache_hits = 0
        self.cache_misses = 0
        self.api_calls_saved = 0
        
        self.logger.info(f"Volume Spike Scanner initialized with {len(self.stock_universe)} symbols")
        self.logger.info(f"Caching enabled: {self.config.enable_caching}")
        self.logger.info("Batch data downloader initialized for optimized yfinance operations")
        
    def _load_config(self):
        """Load scanner configuration."""
        try:
            config_data = self.config_manager.get_config('volume_spike_scanner')
            return ScannerConfig(**config_data)
        except Exception as e:
            self.logger.warning(f"Failed to load config, using defaults: {e}")
            return ScannerConfig()
    
    def _initialize_stock_universe(self) -> List[str]:
        """Initialize the universe of stocks to scan."""
        try:
            # Try to get from config first
            universe = self.config_manager.get_config('stock_universe', [])
            if universe:
                return universe
            
            # Default universe - major indices
            sp500_symbols = [
                'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'BRK-B',
                'UNH', 'JNJ', 'V', 'PG', 'JPM', 'HD', 'CVX', 'MA', 'PFE', 'ABBV',
                'BAC', 'KO', 'AVGO', 'PEP', 'TMO', 'COST', 'WMT', 'DIS', 'ABT',
                'CRM', 'ACN', 'VZ', 'ADBE', 'NFLX', 'CMCSA', 'NKE', 'DHR', 'TXN',
                'NEE', 'RTX', 'QCOM', 'PM', 'UPS', 'T', 'SPGI', 'HON', 'LOW',
                'IBM', 'AMGN', 'SBUX', 'CAT', 'GS', 'INTU', 'AMD', 'BKNG'
            ]
            
            return sp500_symbols
            
        except Exception as e:
            self.logger.error(f"Error initializing stock universe: {e}")
            return ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']  # Fallback
    
    def _generate_cache_key(self, symbol: str, data_type: str, **kwargs) -> str:
        """Generate cache key for data storage."""
        key_parts = [symbol, data_type]
        for k, v in sorted(kwargs.items()):
            key_parts.append(f"{k}:{v}")
        key_string = "|".join(key_parts)
        return hashlib.md5(key_string.encode()).hexdigest()

    async def _get_cached_data(self, cache_key: str, ttl: int) -> Optional[Any]:
        """Get data from cache if available and not expired."""
        if not self.config.enable_caching:
            return None
        
        try:
            cached_data = self.cache_manager.get(cache_key)
            if cached_data is not None:
                self.cache_hits += 1
                return cached_data
            else:
                self.cache_misses += 1
                return None
        except Exception as e:
            self.logger.debug(f"Cache retrieval error: {e}")
            return None

    async def _set_cached_data(self, cache_key: str, data: Any, ttl: int):
        """Store data in cache with TTL."""
        if not self.config.enable_caching:
            return
        
        try:
            self.cache_manager.set(cache_key, data, ttl)
        except Exception as e:
            self.logger.debug(f"Cache storage error: {e}")

    async def start_scanning(self):
        """Start continuous volume spike scanning."""
        if self.is_scanning:
            self.logger.warning("Scanner is already running")
            return
        
        self.is_scanning = True
        self.logger.info("Starting volume spike scanning...")
        
        try:
            while self.is_scanning:
                scan_start = time.time()
                
                # Perform scan
                spikes = await self.scan_for_volume_spikes()
                
                # Process alerts
                if spikes:
                    await self._process_spike_alerts(spikes)
                    self.total_spikes_found += len(spikes)
                
                self.scan_count += 1
                scan_duration = time.time() - scan_start
                
                self.logger.info(
                    f"Scan #{self.scan_count} completed: {len(spikes)} spikes found "
                    f"in {scan_duration:.2f}s (Cache hits: {self.cache_hits}, "
                    f"misses: {self.cache_misses})"
                )
                
                # Wait for next scan
                if self.is_scanning:
                    await asyncio.sleep(self.config.scan_interval_seconds)
                    
        except Exception as e:
            self.logger.error(f"Error in scanning loop: {e}")
        finally:
            self.is_scanning = False
    
    async def scan_for_volume_spikes(self) -> List[VolumeSpike]:
        """
        Scan for volume spikes across the stock universe with optimized batch processing.
        
        Returns:
            List of detected volume spikes
        """
        try:
            self.logger.info(f"Scanning {len(self.stock_universe)} symbols for volume spikes")
            
            # Use batch processing for optimized data fetching
            try:
                # Create batch requests for historical data (for volume averages)
                historical_request = BatchRequest(
                    symbols=self.stock_universe,
                    period=f"{self.config.lookback_days}d",
                    interval="1d",
                    data_type="history",
                    priority=1
                )
                
                # Create batch request for current info (for real-time data)
                info_request = BatchRequest(
                    symbols=self.stock_universe,
                    data_type="info",
                    priority=2
                )
                
                # Fetch data in batches
                batch_results = await self.batch_downloader.fetch_batch_async([
                    historical_request, info_request
                ])
                
                # Process batch results
                historical_data = {}
                info_data = {}
                
                for result in batch_results:
                    if result.success:
                        if result.request.data_type == "history":
                            historical_data = result.data
                        elif result.request.data_type == "info":
                            info_data = result.data
                
                # Analyze symbols using batch data
                volume_spikes = []
                for symbol in self.stock_universe:
                    try:
                        spike = await self._analyze_symbol_volume_optimized(
                            symbol, historical_data.get(symbol), info_data.get(symbol)
                        )
                        if spike:
                            volume_spikes.append(spike)
                    except Exception as e:
                        self.logger.debug(f"Error analyzing {symbol}: {e}")
                        
            except Exception as e:
                self.logger.warning(f"Batch processing failed, falling back to individual analysis: {e}")
                # Fallback to original method
                return await self._scan_for_volume_spikes_fallback()
            
            # Sort by volume ratio (highest first)
            volume_spikes.sort(key=lambda x: x.volume_ratio, reverse=True)
            
            # Limit results
            volume_spikes = volume_spikes[:self.config.max_results_per_scan]
            
            self.last_scan_time = datetime.now()
            
            return volume_spikes
            
        except Exception as e:
            self.logger.error(f"Error scanning for volume spikes: {e}")
            return []
    
    async def _scan_for_volume_spikes_fallback(self) -> List[VolumeSpike]:
        """Fallback method using individual symbol analysis."""
        # Fetch data for all symbols concurrently
        tasks = []
        for symbol in self.stock_universe:
            task = asyncio.create_task(self._analyze_symbol_volume(symbol))
            tasks.append(task)
        
        # Wait for all analyses to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions and None results
        volume_spikes = []
        for result in results:
            if isinstance(result, VolumeSpike):
                volume_spikes.append(result)
            elif isinstance(result, Exception):
                self.logger.debug(f"Symbol analysis failed: {result}")
        
        return volume_spikes
    
    async def _analyze_symbol_volume_optimized(self, symbol: str, 
                                         historical_data: Optional[pd.DataFrame] = None,
                                         info_data: Optional[Dict] = None) -> Optional[VolumeSpike]:
        """
        Analyze a single symbol for volume spikes using batch-downloaded data.
        
        Args:
            symbol: Stock symbol to analyze
            historical_data: Pre-fetched historical data from batch request
            info_data: Pre-fetched info data from batch request
            
        Returns:
            VolumeSpike if detected, None otherwise
        """
        try:
            # Use provided data or fallback to individual fetch
            if historical_data is None or info_data is None:
                return await self._analyze_symbol_volume(symbol)
            
            if historical_data.empty or len(historical_data) < 5:
                return None
            
            # Calculate current volume and price metrics
            current_volume = historical_data['Volume'].iloc[-1]  # Latest day volume
            current_price = historical_data['Close'].iloc[-1]
            
            # Calculate price change from previous day
            if len(historical_data) >= 2:
                price_change_pct = ((current_price - historical_data['Close'].iloc[-2]) / 
                                  historical_data['Close'].iloc[-2] * 100)
            else:
                price_change_pct = 0.0
            
            # Calculate average volume (excluding current day)
            if len(historical_data) > 1:
                avg_volume = historical_data['Volume'].iloc[:-1].mean()
            else:
                return None
            
            # Calculate volume ratio
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 0
            
            # Check if it meets minimum criteria
            if (volume_ratio < self.config.min_volume_ratio or 
                abs(price_change_pct) < self.config.min_price_change or
                current_price > self.config.max_price):
                return None
            
            # Extract market cap and sector from info data
            market_cap = info_data.get('marketCap')
            sector = info_data.get('sector')
            
            # Filter by market cap and sector
            if (market_cap and market_cap < self.config.min_market_cap):
                return None
            
            if sector in self.config.excluded_sectors:
                return None
            
            # Determine severity
            if volume_ratio >= 3.0:
                severity = VolumeSpikeSeverity.EXTREME
            elif volume_ratio >= 2.0:
                severity = VolumeSpikeSeverity.HIGH
            else:
                severity = VolumeSpikeSeverity.MODERATE
            
            # Calculate confidence score
            confidence_score = min(100.0, (volume_ratio - 1.0) * 30 + abs(price_change_pct) * 2)
            
            return VolumeSpike(
                symbol=symbol,
                current_volume=int(current_volume),
                average_volume=avg_volume,
                volume_ratio=volume_ratio,
                severity=severity,
                price_change_pct=price_change_pct,
                current_price=current_price,
                market_cap=market_cap,
                sector=sector,
                timestamp=datetime.now(),
                confidence_score=confidence_score
            )
            
        except Exception as e:
            self.logger.error(f"Error analyzing volume for {symbol}: {str(e)}")
            return None

    async def _analyze_symbol_volume(self, symbol: str) -> Optional[VolumeSpike]:
        """
        Analyze a single symbol for volume spikes with comprehensive caching.
        
        Args:
            symbol: Stock symbol to analyze
            
        Returns:
            VolumeSpike if detected, None otherwise
        """
        try:
            # Generate cache keys
            historical_key = self._generate_cache_key(
                symbol, "historical", 
                lookback_days=self.config.lookback_days
            )
            current_key = self._generate_cache_key(symbol, "current")
            ticker_info_key = self._generate_cache_key(symbol, "ticker_info")
            
            # Try to get cached data
            historical_data = await self._get_cached_data(
                historical_key, self.config.cache_ttl_historical
            )
            current_data = await self._get_cached_data(
                current_key, self.config.cache_ttl_current_data
            )
            ticker_info = await self._get_cached_data(
                ticker_info_key, self.config.cache_ttl_ticker_info
            )
            
            # Fetch missing data
            ticker = yf.Ticker(symbol)
            
            if current_data is None:
                # Get current day data
                current_data = ticker.history(period="1d", interval="1m")
                if current_data.empty:
                    return None
                await self._set_cached_data(
                    current_key, current_data, self.config.cache_ttl_current_data
                )
            else:
                self.api_calls_saved += 1
            
            if historical_data is None:
                # Get historical data for average volume calculation
                historical_data = ticker.history(period=f"{self.config.lookback_days}d")
                if len(historical_data) < 5:  # Need minimum data
                    return None
                await self._set_cached_data(
                    historical_key, historical_data, self.config.cache_ttl_historical
                )
            else:
                self.api_calls_saved += 1
            
            if ticker_info is None:
                # Get additional stock info
                info = ticker.info
                ticker_info = {
                    'marketCap': info.get('marketCap'),
                    'sector': info.get('sector')
                }
                await self._set_cached_data(
                    ticker_info_key, ticker_info, self.config.cache_ttl_ticker_info
                )
            else:
                self.api_calls_saved += 1
            
            # Calculate current volume and price metrics
            current_volume = current_data['Volume'].sum()  # Total volume today
            current_price = current_data['Close'].iloc[-1]
            price_change_pct = ((current_price - historical_data['Close'].iloc[-2]) / 
                              historical_data['Close'].iloc[-2] * 100)
            
            # Calculate average volume (excluding today)
            avg_volume = historical_data['Volume'][:-1].mean()
            
            if avg_volume == 0 or current_volume == 0:
                return None
            
            # Calculate volume ratio
            volume_ratio = current_volume / avg_volume
            
            # Check if it meets spike criteria
            if volume_ratio < self.config.min_volume_ratio:
                return None
            
            # Additional filters
            if current_price > self.config.max_price:
                return None
            
            if abs(price_change_pct) < self.config.min_price_change:
                return None
            
            # Market cap and sector filters
            market_cap = ticker_info.get('marketCap')
            sector = ticker_info.get('sector')
            
            # Market cap filter
            if market_cap and market_cap < self.config.min_market_cap:
                return None
            
            # Sector filter
            if sector in self.config.excluded_sectors:
                return None
            
            # Determine spike severity
            if volume_ratio >= 3.0:
                severity = VolumeSpikeSeverity.EXTREME
            elif volume_ratio >= 2.0:
                severity = VolumeSpikeSeverity.HIGH
            else:
                severity = VolumeSpikeSeverity.MODERATE
            
            # Calculate confidence score
            confidence_score = min(100.0, (
                (volume_ratio - 1.0) * 20 +  # Volume component
                abs(price_change_pct) * 2 +   # Price movement component
                (50 if market_cap and market_cap > 1_000_000_000 else 30)  # Size component
            ))
            
            return VolumeSpike(
                symbol=symbol,
                current_volume=int(current_volume),
                average_volume=avg_volume,
                volume_ratio=volume_ratio,
                severity=severity,
                price_change_pct=price_change_pct,
                current_price=current_price,
                market_cap=market_cap,
                sector=sector,
                timestamp=datetime.now(),
                confidence_score=confidence_score
            )
            
        except Exception as e:
            self.logger.debug(f"Error analyzing {symbol}: {e}")
            return None
    
    async def _process_spike_alerts(self, spikes: List[VolumeSpike]):
        """Process and send alerts for significant volume spikes."""
        try:
            # Filter for high-priority alerts
            high_priority_spikes = [
                spike for spike in spikes 
                if spike.severity in [VolumeSpikeSeverity.HIGH, VolumeSpikeSeverity.EXTREME]
                and spike.confidence_score > 70
            ]
            
            if not high_priority_spikes:
                return
            
            # Create alert message
            alert_data = {
                'type': 'volume_spike_alert',
                'timestamp': datetime.now().isoformat(),
                'spike_count': len(high_priority_spikes),
                'spikes': [spike.to_dict() for spike in high_priority_spikes[:10]]  # Top 10
            }
            
            # Send notification
            await self.notification_manager.send_notification(
                title=f"Volume Spike Alert: {len(high_priority_spikes)} stocks detected",
                message=f"Detected {len(high_priority_spikes)} significant volume spikes",
                data=alert_data,
                priority="high"
            )
            
            self.logger.info(f"Sent volume spike alert for {len(high_priority_spikes)} stocks")
            
        except Exception as e:
            self.logger.error(f"Error processing spike alerts: {e}")
    
    def stop_scanning(self):
        """Stop the volume spike scanning."""
        self.is_scanning = False
        self.logger.info("Volume spike scanner stopped")
    
    def get_cache_statistics(self) -> Dict[str, Any]:
        """Get cache performance statistics."""
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = (self.cache_hits / total_requests * 100) if total_requests > 0 else 0
        
        return {
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'hit_rate_percent': round(hit_rate, 2),
            'api_calls_saved': self.api_calls_saved,
            'cache_size': len(self.cache_manager.cache) if hasattr(self.cache_manager, 'cache') else 0
        }

    def clear_cache(self):
        """Clear all cached data."""
        try:
            self.cache_manager.clear()
            self.cache_hits = 0
            self.cache_misses = 0
            self.api_calls_saved = 0
            self.logger.info("Cache cleared successfully")
        except Exception as e:
            self.logger.error(f"Error clearing cache: {e}")
    
    def get_scanner_metrics(self) -> Dict[str, Any]:
        """Get scanner performance metrics."""
        metrics = {
            'scan_count': self.scan_count,
            'total_spikes_found': self.total_spikes_found,
            'last_scan_time': self.last_scan_time.isoformat() if self.last_scan_time else None,
            'is_scanning': self.is_scanning,
            'universe_size': len(self.stock_universe),
            'cache_stats': self.get_cache_statistics(),
            'config': {
                'min_volume_ratio': self.config.min_volume_ratio,
                'scan_interval_seconds': self.config.scan_interval_seconds,
                'lookback_days': self.config.lookback_days,
                'caching_enabled': self.config.enable_caching
            }
        }
        
        # Add batch processing metrics if available
        if hasattr(self, 'batch_downloader') and self.batch_downloader:
            batch_metrics = self.batch_downloader.get_performance_metrics()
            metrics['batch_processing'] = batch_metrics
        
        # Add cache manager metrics if available
        if hasattr(self, 'cache_manager') and self.cache_manager:
            cache_metrics = self.cache_manager.get_metrics()
            metrics['cache_manager'] = cache_metrics
            metrics['cache_ttl_settings'] = {
                'historical_data': self.config.cache_ttl_historical,
                'ticker_info': self.config.cache_ttl_ticker_info,
                'current_data': self.config.cache_ttl_current_data
            }
        
        return metrics
    
    async def get_symbol_analysis(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get detailed volume analysis for a specific symbol."""
        try:
            spike = await self._analyze_symbol_volume(symbol)
            if spike:
                return spike.to_dict()
            return None
            
        except Exception as e:
            self.logger.error(f"Error analyzing symbol {symbol}: {e}")
            return None


# Integration with existing agent system
class VolumeSpikeDiscoveryTool:
    """Tool wrapper for integration with CrewAI agents."""
    
    def __init__(self, scanner: VolumeSpikeScanner):
        self.scanner = scanner
        self.name = "volume_spike_discovery"
        self.description = "Discover stocks with unusual volume spikes for trading opportunities"
    
    async def run(self, query: str = "") -> str:
        """Run volume spike discovery."""
        try:
            spikes = await self.scanner.scan_for_volume_spikes()
            
            if not spikes:
                return "No significant volume spikes detected in current scan."
            
            # Format results for agent consumption
            results = []
            for spike in spikes[:10]:  # Top 10 results
                results.append(
                    f"{spike.symbol}: {spike.volume_ratio:.1f}x volume, "
                    f"{spike.price_change_pct:+.1f}% price change, "
                    f"{spike.severity.value} severity"
                )
            
            return f"Volume Spike Discovery Results:\n" + "\n".join(results)
            
        except Exception as e:
            return f"Error in volume spike discovery: {e}"


if __name__ == "__main__":
    # Test the volume spike scanner
    import asyncio
    from pathlib import Path
    import sys
    
    # Add project root to path
    project_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(project_root))
    
    from trading_system_v2.core.config_manager import ConfigManager
    
    async def test_scanner():
        """Test the volume spike scanner."""
        config_manager = ConfigManager()
        await config_manager.initialize()
        
        scanner = VolumeSpikeScanner(config_manager)
        
        print("Testing volume spike scanner...")
        spikes = await scanner.scan_for_volume_spikes()
        
        print(f"\nDetected {len(spikes)} volume spikes:")
        for spike in spikes[:5]:
            print(f"  {spike.symbol}: {spike.volume_ratio:.1f}x volume, "
                  f"{spike.price_change_pct:+.1f}% price change")
        
        print(f"\nScanner metrics: {scanner.get_scanner_metrics()}")
    
    asyncio.run(test_scanner())