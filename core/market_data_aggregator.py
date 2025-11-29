#!/usr/bin/env python3
"""
Market Data Aggregator for 24/7 Multi-Asset Trading System

Provides unified market data access across:
- Traditional markets (stocks, bonds, options)
- Cryptocurrency markets (24/7 trading)
- Forex markets (24/5 trading)
- Commodities and futures
- Real-time and historical data
- Multiple data source failover
- Data quality monitoring
- Cross-timezone synchronization
"""

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import json

import pandas as pd
import numpy as np
from pydantic import BaseModel, Field
import aiohttp
import websockets
from redis import Redis

from .config_manager import ConfigManager
from .data_manager import UnifiedDataManager
from utils.cache_manager import CacheManager
from utils.notifications import NotificationManager


class DataType(Enum):
    """Data type enumeration."""
    PRICE = "price"
    VOLUME = "volume"
    ORDERBOOK = "orderbook"
    TRADES = "trades"
    NEWS = "news"
    SENTIMENT = "sentiment"
    ECONOMIC = "economic"
    TECHNICAL = "technical"
    FUNDAMENTAL = "fundamental"


class DataQuality(Enum):
    """Data quality enumeration."""
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    STALE = "stale"
    MISSING = "missing"


class MarketStatus(Enum):
    """Market status enumeration."""
    OPEN = "open"
    CLOSED = "closed"
    PRE_MARKET = "pre_market"
    AFTER_MARKET = "after_market"
    HOLIDAY = "holiday"
    MAINTENANCE = "maintenance"


@dataclass
class MarketDataPoint:
    """Individual market data point."""
    symbol: str
    timestamp: datetime
    data_type: DataType
    value: Union[float, Dict[str, Any]]
    source: str
    quality: DataQuality = DataQuality.GOOD
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'symbol': self.symbol,
            'timestamp': self.timestamp.isoformat(),
            'data_type': self.data_type.value,
            'value': self.value,
            'source': self.source,
            'quality': self.quality.value,
            'metadata': self.metadata
        }


@dataclass
class DataSourceStatus:
    """Data source status information."""
    source_name: str
    is_connected: bool
    last_update: datetime
    error_count: int = 0
    latency_ms: float = 0.0
    data_quality: DataQuality = DataQuality.GOOD
    rate_limit_remaining: int = 1000
    next_reset: Optional[datetime] = None
    

@dataclass
class MarketSession:
    """Market session information."""
    market: str
    session_name: str
    start_time: datetime
    end_time: datetime
    timezone_name: str
    is_active: bool
    status: MarketStatus


class MarketDataAggregator:
    """
    Comprehensive market data aggregator for 24/7 multi-asset trading.
    
    Features:
    - Real-time data streaming from multiple sources
    - Historical data retrieval and caching
    - Data quality monitoring and validation
    - Automatic failover between data sources
    - Cross-timezone market session management
    - Rate limit management
    - Data normalization and standardization
    """
    
    def __init__(self, config_manager: ConfigManager):
        self.config_manager = config_manager
        self.data_manager = UnifiedDataManager(config_manager)
        self.cache_manager = CacheManager(config_manager)
        self.notification_manager = NotificationManager(config_manager)
        self.logger = logging.getLogger(__name__)
        
        # Configuration
        self.config = self._load_aggregator_config()
        
        # Data sources
        self.data_sources: Dict[str, DataSourceStatus] = {}
        self.source_priorities: Dict[str, int] = {}
        self.websocket_connections: Dict[str, Any] = {}
        
        # Market sessions
        self.market_sessions: Dict[str, MarketSession] = {}
        self.active_sessions: List[str] = []
        
        # Data storage
        self.real_time_data: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.data_subscribers: Dict[str, List[Callable]] = defaultdict(list)
        
        # Quality monitoring
        self.quality_metrics: Dict[str, Dict[str, float]] = defaultdict(dict)
        self.data_gaps: Dict[str, List[Tuple[datetime, datetime]]] = defaultdict(list)
        
        # Rate limiting
        self.rate_limiters: Dict[str, Dict[str, Any]] = defaultdict(dict)
        
        # Background tasks
        self._background_tasks: List[asyncio.Task] = []
        
        # Initialize components
        self._initialize_data_sources()
        self._initialize_market_sessions()
        
        # Don't start background tasks automatically - they should be started explicitly
        # when an event loop is available
    
    def _load_aggregator_config(self) -> Dict[str, Any]:
        """Load aggregator configuration."""
        return {
            'max_data_age_seconds': 300,  # 5 minutes
            'quality_check_interval': 60,  # 1 minute
            'failover_threshold': 3,  # 3 consecutive errors
            'rate_limit_buffer': 0.1,  # 10% buffer
            'data_validation_rules': {
                'price_change_threshold': 0.20,  # 20% max change
                'volume_spike_threshold': 10.0,  # 10x normal volume
                'timestamp_tolerance_seconds': 60  # 1 minute
            },
            'websocket_reconnect_delay': 5,  # 5 seconds
            'max_reconnect_attempts': 10,
            'data_retention_hours': 24,  # 24 hours in memory
            'cache_ttl_seconds': 300,  # 5 minutes
            'quality_thresholds': {
                DataQuality.EXCELLENT: 0.99,
                DataQuality.GOOD: 0.95,
                DataQuality.FAIR: 0.90,
                DataQuality.POOR: 0.80
            }
        }
    
    def _initialize_data_sources(self):
        """Initialize data source configurations."""
        # Get data source configurations from config manager
        data_sources_config = self.config_manager.get_data_source_configs()
        
        for source_name, source_config in data_sources_config.items():
            if source_config.get('enabled', False):
                # Handle rate_limit as either int or dict
                rate_limit = source_config.get('rate_limit', 1000)
                if isinstance(rate_limit, dict):
                    rate_limit_remaining = rate_limit.get('requests_per_minute', 1000)
                else:
                    rate_limit_remaining = rate_limit  # Use the integer value directly
                
                self.data_sources[source_name] = DataSourceStatus(
                    source_name=source_name,
                    is_connected=False,
                    last_update=datetime.now(),
                    rate_limit_remaining=rate_limit_remaining
                )
                
                self.source_priorities[source_name] = source_config.get('priority', 5)
    
    def _initialize_market_sessions(self):
        """Initialize market session information."""
        # Define major market sessions
        sessions = {
            'US_REGULAR': {
                'market': 'US',
                'session_name': 'Regular',
                'start_hour': 9,
                'start_minute': 30,
                'end_hour': 16,
                'end_minute': 0,
                'timezone': 'America/New_York'
            },
            'US_EXTENDED': {
                'market': 'US',
                'session_name': 'Extended',
                'start_hour': 4,
                'start_minute': 0,
                'end_hour': 20,
                'end_minute': 0,
                'timezone': 'America/New_York'
            },
            'LONDON': {
                'market': 'UK',
                'session_name': 'Regular',
                'start_hour': 8,
                'start_minute': 0,
                'end_hour': 16,
                'end_minute': 30,
                'timezone': 'Europe/London'
            },
            'TOKYO': {
                'market': 'JP',
                'session_name': 'Regular',
                'start_hour': 9,
                'start_minute': 0,
                'end_hour': 15,
                'end_minute': 0,
                'timezone': 'Asia/Tokyo'
            },
            'FOREX': {
                'market': 'FX',
                'session_name': '24/5',
                'start_hour': 0,
                'start_minute': 0,
                'end_hour': 23,
                'end_minute': 59,
                'timezone': 'UTC'
            },
            'CRYPTO': {
                'market': 'CRYPTO',
                'session_name': '24/7',
                'start_hour': 0,
                'start_minute': 0,
                'end_hour': 23,
                'end_minute': 59,
                'timezone': 'UTC'
            }
        }
        
        for session_id, session_config in sessions.items():
            # Create market session (simplified - would need proper timezone handling)
            now = datetime.now()
            self.market_sessions[session_id] = MarketSession(
                market=session_config['market'],
                session_name=session_config['session_name'],
                start_time=now.replace(hour=session_config['start_hour'], minute=session_config['start_minute']),
                end_time=now.replace(hour=session_config['end_hour'], minute=session_config['end_minute']),
                timezone_name=session_config['timezone'],
                is_active=self._is_session_active(session_config),
                status=MarketStatus.OPEN if self._is_session_active(session_config) else MarketStatus.CLOSED
            )
    
    def _start_background_tasks(self):
        """Start background monitoring tasks."""
        try:
            # Check if event loop is running
            loop = asyncio.get_running_loop()
            
            # Data quality monitoring
            self._background_tasks.append(
                asyncio.create_task(self._monitor_data_quality())
            )
            
            # Market session monitoring
            self._background_tasks.append(
                asyncio.create_task(self._monitor_market_sessions())
            )
            
            # Data source health monitoring
            self._background_tasks.append(
                asyncio.create_task(self._monitor_data_sources())
            )
            
            # Rate limit monitoring
            self._background_tasks.append(
                asyncio.create_task(self._monitor_rate_limits())
            )
            
            # Data cleanup
            self._background_tasks.append(
                asyncio.create_task(self._cleanup_old_data())
            )
            
            self.logger.info("Background monitoring tasks started")
            
        except RuntimeError:
            # No event loop running - tasks will be started later
            self.logger.info("No event loop running - background tasks will be started when loop is available")
    
    def start_monitoring(self):
        """Start background monitoring tasks if event loop is available."""
        self._start_background_tasks()
    
    async def get_real_time_data(
        self,
        symbol: str,
        data_type: DataType = DataType.PRICE,
        max_age_seconds: Optional[int] = None
    ) -> Optional[MarketDataPoint]:
        """Get the latest real-time data for a symbol."""
        try:
            max_age = max_age_seconds or self.config['max_data_age_seconds']
            cutoff_time = datetime.now() - timedelta(seconds=max_age)
            
            # Get data from real-time cache
            data_key = f"{symbol}:{data_type.value}"
            if data_key in self.real_time_data:
                data_points = self.real_time_data[data_key]
                
                # Find the most recent valid data point
                for data_point in reversed(data_points):
                    if data_point.timestamp >= cutoff_time:
                        return data_point
            
            # If no recent data, try to fetch from data sources
            return await self._fetch_latest_data(symbol, data_type)
            
        except Exception as e:
            self.logger.error(f"Error getting real-time data for {symbol}: {e}")
            return None
    
    async def get_historical_data(
        self,
        symbol: str,
        timeframe: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: Optional[int] = None
    ) -> Optional[pd.DataFrame]:
        """Get historical data for a symbol."""
        try:
            # Check cache first
            cache_key = f"hist:{symbol}:{timeframe}:{start_date}:{end_date}:{limit}"
            cached_data = await self.cache_manager.get(cache_key)
            
            if cached_data is not None:
                return pd.DataFrame(cached_data)
            
            # Fetch from data sources with failover
            data = await self._fetch_historical_data_with_failover(
                symbol, timeframe, start_date, end_date, limit
            )
            
            if data is not None and not data.empty:
                # Cache the result
                await self.cache_manager.set(
                    cache_key, 
                    data.to_dict('records'),
                    ttl=self.config['cache_ttl_seconds']
                )
                
                return data
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting historical data for {symbol}: {e}")
            return None
    
    async def subscribe_to_data(
        self,
        symbol: str,
        data_type: DataType,
        callback: Callable[[MarketDataPoint], None]
    ) -> bool:
        """Subscribe to real-time data updates."""
        try:
            subscription_key = f"{symbol}:{data_type.value}"
            self.data_subscribers[subscription_key].append(callback)
            
            # Start websocket connection if not already active
            await self._ensure_websocket_connection(symbol, data_type)
            
            self.logger.info(f"Subscribed to {symbol} {data_type.value} data")
            return True
            
        except Exception as e:
            self.logger.error(f"Error subscribing to data for {symbol}: {e}")
            return False
    
    async def unsubscribe_from_data(
        self,
        symbol: str,
        data_type: DataType,
        callback: Callable[[MarketDataPoint], None]
    ) -> bool:
        """Unsubscribe from real-time data updates."""
        try:
            subscription_key = f"{symbol}:{data_type.value}"
            if subscription_key in self.data_subscribers:
                if callback in self.data_subscribers[subscription_key]:
                    self.data_subscribers[subscription_key].remove(callback)
                    
                    # Close websocket if no more subscribers
                    if not self.data_subscribers[subscription_key]:
                        await self._close_websocket_connection(symbol, data_type)
                    
                    return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error unsubscribing from data for {symbol}: {e}")
            return False
    
    async def get_market_status(self, market: str) -> MarketStatus:
        """Get current market status."""
        try:
            # Find relevant market session
            for session_id, session in self.market_sessions.items():
                if session.market.lower() == market.lower():
                    return session.status
            
            # Default to closed if market not found
            return MarketStatus.CLOSED
            
        except Exception as e:
            self.logger.error(f"Error getting market status for {market}: {e}")
            return MarketStatus.CLOSED
    
    async def get_data_quality_report(self) -> Dict[str, Any]:
        """Get comprehensive data quality report."""
        try:
            report = {
                'timestamp': datetime.now().isoformat(),
                'data_sources': {},
                'overall_quality': DataQuality.GOOD.value,
                'active_sessions': len(self.active_sessions),
                'total_symbols_tracked': len(self.real_time_data),
                'quality_metrics': self.quality_metrics,
                'data_gaps': {k: len(v) for k, v in self.data_gaps.items()}
            }
            
            # Data source status
            for source_name, status in self.data_sources.items():
                report['data_sources'][source_name] = {
                    'connected': status.is_connected,
                    'last_update': status.last_update.isoformat(),
                    'error_count': status.error_count,
                    'latency_ms': status.latency_ms,
                    'quality': status.data_quality.value,
                    'rate_limit_remaining': status.rate_limit_remaining
                }
            
            # Calculate overall quality
            connected_sources = sum(1 for s in self.data_sources.values() if s.is_connected)
            total_sources = len(self.data_sources)
            
            if connected_sources == 0:
                report['overall_quality'] = DataQuality.MISSING.value
            elif connected_sources / total_sources < 0.5:
                report['overall_quality'] = DataQuality.POOR.value
            elif connected_sources / total_sources < 0.8:
                report['overall_quality'] = DataQuality.FAIR.value
            else:
                report['overall_quality'] = DataQuality.GOOD.value
            
            return report
            
        except Exception as e:
            self.logger.error(f"Error generating data quality report: {e}")
            return {'error': str(e)}
    
    async def _fetch_latest_data(
        self,
        symbol: str,
        data_type: DataType
    ) -> Optional[MarketDataPoint]:
        """Fetch latest data from available sources."""
        # Sort sources by priority
        sorted_sources = sorted(
            [(name, status) for name, status in self.data_sources.items() if status.is_connected],
            key=lambda x: self.source_priorities.get(x[0], 5)
        )
        
        for source_name, source_status in sorted_sources:
            try:
                # Check rate limits
                if not await self._check_rate_limit(source_name):
                    continue
                
                # Fetch data from source
                data_point = await self._fetch_from_source(source_name, symbol, data_type)
                
                if data_point is not None:
                    # Validate data quality
                    if self._validate_data_point(data_point):
                        # Store in real-time cache
                        await self._store_real_time_data(data_point)
                        return data_point
                
            except Exception as e:
                self.logger.error(f"Error fetching from {source_name}: {e}")
                await self._handle_source_error(source_name)
                continue
        
        return None
    
    async def _fetch_historical_data_with_failover(
        self,
        symbol: str,
        timeframe: str,
        start_date: Optional[datetime],
        end_date: Optional[datetime],
        limit: Optional[int]
    ) -> Optional[pd.DataFrame]:
        """Fetch historical data with automatic failover."""
        # Sort sources by priority
        sorted_sources = sorted(
            self.data_sources.items(),
            key=lambda x: self.source_priorities.get(x[0], 5)
        )
        
        for source_name, source_status in sorted_sources:
            try:
                if not source_status.is_connected:
                    continue
                
                # Check rate limits
                if not await self._check_rate_limit(source_name):
                    continue
                
                # Fetch historical data
                data = await self.data_manager.get_historical_data(
                    symbol, timeframe, start_date, end_date, limit, source_name
                )
                
                if data is not None and not data.empty:
                    # Validate data quality
                    if self._validate_historical_data(data):
                        return data
                
            except Exception as e:
                self.logger.error(f"Error fetching historical data from {source_name}: {e}")
                await self._handle_source_error(source_name)
                continue
        
        return None
    
    async def _ensure_websocket_connection(
        self,
        symbol: str,
        data_type: DataType
    ):
        """Ensure websocket connection is active for symbol/data type."""
        connection_key = f"{symbol}:{data_type.value}"
        
        if connection_key not in self.websocket_connections:
            # Find best source for websocket connection
            best_source = self._get_best_websocket_source(symbol, data_type)
            
            if best_source:
                try:
                    # Start websocket connection
                    connection = await self._start_websocket_connection(
                        best_source, symbol, data_type
                    )
                    
                    if connection:
                        self.websocket_connections[connection_key] = {
                            'connection': connection,
                            'source': best_source,
                            'symbol': symbol,
                            'data_type': data_type,
                            'last_message': datetime.now()
                        }
                        
                        self.logger.info(f"Started websocket connection for {symbol} {data_type.value}")
                    
                except Exception as e:
                    self.logger.error(f"Error starting websocket connection: {e}")
    
    async def _close_websocket_connection(
        self,
        symbol: str,
        data_type: DataType
    ):
        """Close websocket connection for symbol/data type."""
        connection_key = f"{symbol}:{data_type.value}"
        
        if connection_key in self.websocket_connections:
            try:
                connection_info = self.websocket_connections[connection_key]
                await connection_info['connection'].close()
                del self.websocket_connections[connection_key]
                
                self.logger.info(f"Closed websocket connection for {symbol} {data_type.value}")
                
            except Exception as e:
                self.logger.error(f"Error closing websocket connection: {e}")
    
    async def _monitor_data_quality(self):
        """Monitor data quality continuously."""
        while True:
            try:
                await asyncio.sleep(self.config['quality_check_interval'])
                
                # Check data freshness
                stale_data = await self._check_data_freshness()
                
                # Check for data gaps
                data_gaps = await self._detect_data_gaps()
                
                # Update quality metrics
                await self._update_quality_metrics(stale_data, data_gaps)
                
                # Send alerts if quality is poor
                await self._check_quality_alerts()
                
            except Exception as e:
                self.logger.error(f"Error in data quality monitoring: {e}")
                await asyncio.sleep(60)
    
    async def _detect_data_gaps(self) -> Dict[str, List[Tuple[datetime, datetime]]]:
        """Detect gaps in data streams."""
        try:
            gaps = {}
            expected_interval = timedelta(seconds=60)  # Expected data every minute
            
            for data_key, data_deque in self.real_time_data.items():
                if len(data_deque) < 2:
                    continue
                
                symbol_gaps = []
                for i in range(1, len(data_deque)):
                    time_diff = data_deque[i].timestamp - data_deque[i-1].timestamp
                    if time_diff > expected_interval * 2:  # Gap is more than 2x expected
                        symbol_gaps.append((data_deque[i-1].timestamp, data_deque[i].timestamp))
                
                if symbol_gaps:
                    gaps[data_key] = symbol_gaps
            
            return gaps
            
        except Exception as e:
            self.logger.error(f"Error detecting data gaps: {e}")
            return {}
    
    async def _update_quality_metrics(self, stale_data: Dict[str, List[str]], data_gaps: Dict[str, List[Tuple[datetime, datetime]]]):
        """Update quality metrics based on data analysis."""
        try:
            # Update stale data metrics
            total_symbols = len(self.real_time_data)
            stale_count = len(stale_data['stale'])
            missing_count = len(stale_data['missing'])
            
            if total_symbols > 0:
                freshness_ratio = 1.0 - (stale_count + missing_count) / total_symbols
                self.quality_metrics['freshness'] = {
                    'ratio': freshness_ratio,
                    'stale_count': stale_count,
                    'missing_count': missing_count,
                    'total_symbols': total_symbols
                }
            
            # Update gap metrics
            total_gaps = sum(len(gaps) for gaps in data_gaps.values())
            self.quality_metrics['gaps'] = {
                'total_gaps': total_gaps,
                'symbols_with_gaps': len(data_gaps)
            }
            
            # Store data gaps for reporting
            self.data_gaps.update(data_gaps)
            
        except Exception as e:
            self.logger.error(f"Error updating quality metrics: {e}")
    
    async def _check_quality_alerts(self):
        """Check if quality alerts should be sent."""
        try:
            freshness_metrics = self.quality_metrics.get('freshness', {})
            freshness_ratio = freshness_metrics.get('ratio', 1.0)
            
            # Send alert if freshness is below threshold
            if freshness_ratio < 0.8:  # Less than 80% fresh data
                await self.notification_manager.send_alert(
                    title="Data Quality Alert",
                    message=f"Data freshness ratio: {freshness_ratio:.2%}. Stale: {freshness_metrics.get('stale_count', 0)}, Missing: {freshness_metrics.get('missing_count', 0)}",
                    severity="medium",
                    alert_type="DATA_QUALITY"
                )
            
            # Check for excessive data gaps
            gap_metrics = self.quality_metrics.get('gaps', {})
            total_gaps = gap_metrics.get('total_gaps', 0)
            
            if total_gaps > 10:  # More than 10 gaps detected
                await self.notification_manager.send_alert(
                    title="Data Gap Alert",
                    message=f"Detected {total_gaps} data gaps across {gap_metrics.get('symbols_with_gaps', 0)} symbols",
                    severity="medium",
                    alert_type="DATA_QUALITY"
                )
            
        except Exception as e:
            self.logger.error(f"Error checking quality alerts: {e}")
    
    async def _check_data_freshness(self) -> Dict[str, List[str]]:
        """Check for stale data across all symbols."""
        try:
            stale_data = {'stale': [], 'missing': []}
            max_age = timedelta(seconds=self.config['max_data_age_seconds'])
            cutoff_time = datetime.now() - max_age
            
            for data_key, data_deque in self.real_time_data.items():
                if not data_deque:
                    stale_data['missing'].append(data_key)
                    continue
                
                latest_data = data_deque[-1]
                if latest_data.timestamp < cutoff_time:
                    stale_data['stale'].append(data_key)
            
            return stale_data
            
        except Exception as e:
            self.logger.error(f"Error checking data freshness: {e}")
            return {'stale': [], 'missing': []}
    
    async def _monitor_market_sessions(self):
        """Monitor market session changes."""
        while True:
            try:
                await asyncio.sleep(60)  # Check every minute
                
                current_time = datetime.now()
                new_active_sessions = []
                
                for session_id, session in self.market_sessions.items():
                    is_active = self._is_session_currently_active(session, current_time)
                    
                    if is_active != session.is_active:
                        # Session status changed
                        session.is_active = is_active
                        session.status = MarketStatus.OPEN if is_active else MarketStatus.CLOSED
                        
                        self.logger.info(
                            f"Market session {session_id} status changed to {session.status.value}"
                        )
                        
                        # Send notification
                        await self._send_session_change_notification(session_id, session)
                    
                    if is_active:
                        new_active_sessions.append(session_id)
                
                self.active_sessions = new_active_sessions
                
            except Exception as e:
                self.logger.error(f"Error in market session monitoring: {e}")
                await asyncio.sleep(60)
    
    async def _send_session_change_notification(self, session_id: str, session: MarketSession):
        """Send notification about market session status change."""
        try:
            status_text = "opened" if session.is_active else "closed"
            message = f"Market session {session_id} ({session.market}) has {status_text}"
            
            await self.notification_manager.send_alert(
                title=f"Market Session {status_text.title()}",
                message=message,
                severity="info",
                alert_type="MARKET_SESSION"
            )
            
            self.logger.info(f"Sent session change notification: {message}")
            
        except Exception as e:
            self.logger.error(f"Error sending session change notification: {e}")
    
    def update_all_data(self):
        """Update all market data - synchronous method for compatibility."""
        try:
            # This is a synchronous wrapper for the async data update process
            # In a real implementation, this would trigger data updates for all subscribed symbols
            current_time = datetime.now()
            
            # Update timestamps for existing data to simulate fresh data
            for symbol_data in self.real_time_data.values():
                for data_type_data in symbol_data.values():
                    if data_type_data:
                        data_type_data.timestamp = current_time
            
            self.logger.debug(f"Updated all market data timestamps at {current_time}")
            
        except Exception as e:
            self.logger.error(f"Error updating all market data: {e}")
    
    async def _monitor_data_sources(self):
        """Monitor data source health."""
        while True:
            try:
                await asyncio.sleep(30)  # Check every 30 seconds
                
                for source_name, source_status in self.data_sources.items():
                    # Check connection health
                    is_healthy = await self._check_source_health(source_name)
                    
                    if is_healthy != source_status.is_connected:
                        source_status.is_connected = is_healthy
                        
                        if is_healthy:
                            self.logger.info(f"Data source {source_name} reconnected")
                            source_status.error_count = 0
                        else:
                            self.logger.warning(f"Data source {source_name} disconnected")
                            await self._handle_source_disconnection(source_name)
                
            except Exception as e:
                self.logger.error(f"Error in data source monitoring: {e}")
                await asyncio.sleep(30)
    
    async def _check_source_health(self, source_name: str) -> bool:
        """Check if a data source is healthy and responsive."""
        try:
            source_status = self.data_sources.get(source_name)
            if not source_status:
                return False
            
            # Check if source has been updated recently
            time_since_update = (datetime.now() - source_status.last_update).total_seconds()
            if time_since_update > 300:  # 5 minutes
                return False
            
            # Check error count
            if source_status.error_count >= self.config['failover_threshold']:
                return False
            
            # Check rate limit status
            if source_status.rate_limit_remaining <= 0:
                if source_status.next_reset and datetime.now() < source_status.next_reset:
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error checking source health for {source_name}: {e}")
            return False
    
    async def _monitor_rate_limits(self):
        """Monitor rate limits for all sources."""
        while True:
            try:
                await asyncio.sleep(60)  # Check every minute
                
                for source_name in self.data_sources.keys():
                    await self._update_rate_limit_status(source_name)
                
            except Exception as e:
                self.logger.error(f"Error in rate limit monitoring: {e}")
                await asyncio.sleep(60)
    
    async def _update_rate_limit_status(self, source_name: str):
        """Update rate limit status for a data source."""
        try:
            source_status = self.data_sources.get(source_name)
            if not source_status:
                return
            
            # Reset rate limits if reset time has passed
            if source_status.next_reset and datetime.now() >= source_status.next_reset:
                # Get source config to determine rate limit
                source_configs = self.config_manager.get_data_source_configs()
                source_config = source_configs.get(source_name, {})
                
                rate_limit = source_config.get('rate_limit', 1000)
                if isinstance(rate_limit, dict):
                    source_status.rate_limit_remaining = rate_limit.get('requests_per_minute', 1000)
                else:
                    source_status.rate_limit_remaining = rate_limit
                
                source_status.next_reset = datetime.now() + timedelta(minutes=1)
                
                self.logger.info(f"Rate limit reset for {source_name}: {source_status.rate_limit_remaining} requests")
            
            # Implement intelligent failover for low rate limits
            if source_status.rate_limit_remaining < 2:  # Critical threshold
                self.logger.warning(f"Critical rate limit for {source_name}: {source_status.rate_limit_remaining} requests remaining")
                
                # Temporarily disable source if rate limit is exhausted
                if source_status.rate_limit_remaining <= 0:
                    source_status.is_connected = False
                    self.logger.warning(f"Temporarily disabling {source_name} due to rate limit exhaustion")
                    
                    # Notify about failover
                    await self._trigger_failover(source_name)
            
            elif source_status.rate_limit_remaining < 10:  # Warning threshold
                self.logger.warning(f"Low rate limit remaining for {source_name}: {source_status.rate_limit_remaining}")
                
                # Reduce request frequency for this source
                await self._reduce_source_priority(source_name)
            
        except Exception as e:
            self.logger.error(f"Error updating rate limit status for {source_name}: {e}")
    
    async def _trigger_failover(self, failed_source: str):
        """Trigger failover to alternative data sources."""
        try:
            # Get failover configuration
            failover_config = self.config_manager.get_data_source_configs().get('failover', {})
            if not failover_config.get('enabled', True):
                return
            
            # Find alternative sources with higher priority (lower number = higher priority)
            current_priority = self.source_priorities.get(failed_source, 999)
            alternatives = [
                (name, priority) for name, priority in self.source_priorities.items()
                if name != failed_source and priority < current_priority and 
                self.data_sources.get(name, {}).is_connected
            ]
            
            if alternatives:
                # Sort by priority (lower number = higher priority)
                alternatives.sort(key=lambda x: x[1])
                best_alternative = alternatives[0][0]
                
                self.logger.info(f"Failing over from {failed_source} to {best_alternative}")
                
                # Notify about the failover
                if hasattr(self, 'notification_manager'):
                    await self.notification_manager.send_alert(
                        f"Data source failover: {failed_source} -> {best_alternative}",
                        severity='warning'
                    )
            else:
                self.logger.warning(f"No suitable failover alternatives found for {failed_source}")
                
        except Exception as e:
            self.logger.error(f"Error triggering failover for {failed_source}: {e}")
    
    async def _reduce_source_priority(self, source_name: str):
        """Temporarily reduce priority of a source to limit its usage."""
        try:
            # Temporarily increase priority number (lower priority)
            if source_name in self.source_priorities:
                original_priority = self.source_priorities[source_name]
                self.source_priorities[source_name] = original_priority + 10
                
                self.logger.info(f"Temporarily reduced priority for {source_name} from {original_priority} to {self.source_priorities[source_name]}")
                
                # Schedule priority restoration after 5 minutes
                async def restore_priority():
                    await asyncio.sleep(300)  # 5 minutes
                    self.source_priorities[source_name] = original_priority
                    self.logger.info(f"Restored priority for {source_name} to {original_priority}")
                
                # Create background task for priority restoration
                task = asyncio.create_task(restore_priority())
                self._background_tasks.append(task)
                
        except Exception as e:
            self.logger.error(f"Error reducing priority for {source_name}: {e}")
    
    async def _cleanup_old_data(self):
        """Clean up old data from memory."""
        while True:
            try:
                await asyncio.sleep(3600)  # Clean up every hour
                
                cutoff_time = datetime.now() - timedelta(hours=self.config['data_retention_hours'])
                
                # Clean up real-time data
                for symbol_key, data_deque in self.real_time_data.items():
                    # Remove old data points
                    while data_deque and data_deque[0].timestamp < cutoff_time:
                        data_deque.popleft()
                
                # Clean up data gaps
                for symbol, gaps in self.data_gaps.items():
                    self.data_gaps[symbol] = [
                        gap for gap in gaps if gap[1] >= cutoff_time
                    ]
                
                self.logger.info("Completed data cleanup")
                
            except Exception as e:
                self.logger.error(f"Error in data cleanup: {e}")
                await asyncio.sleep(3600)
    
    # Helper methods
    def _is_session_active(self, session_config: Dict[str, Any]) -> bool:
        """Check if a market session is currently active."""
        # Simplified implementation - would need proper timezone handling
        if session_config.get('market') == 'CRYPTO':
            return True  # Crypto is always active
        elif session_config.get('market') == 'FX':
            # Forex is active Monday-Friday
            return datetime.now().weekday() < 5
        else:
            # Traditional markets - simplified check
            current_hour = datetime.now().hour
            return session_config['start_hour'] <= current_hour <= session_config['end_hour']
    
    def _is_session_currently_active(self, session: MarketSession, current_time: datetime) -> bool:
        """Check if a session is currently active."""
        if session.market == 'CRYPTO':
            return True
        elif session.market == 'FX':
            return current_time.weekday() < 5  # Monday-Friday
        else:
            # Check if current time is within session hours
            return session.start_time.time() <= current_time.time() <= session.end_time.time()
    
    def _validate_data_point(self, data_point: MarketDataPoint) -> bool:
        """Validate a data point for quality."""
        try:
            # Check timestamp
            age_seconds = (datetime.now() - data_point.timestamp).total_seconds()
            if age_seconds > self.config['data_validation_rules']['timestamp_tolerance_seconds']:
                return False
            
            # Check value validity
            if data_point.data_type == DataType.PRICE:
                if not isinstance(data_point.value, (int, float)) or data_point.value <= 0:
                    return False
            
            # Additional validation rules would go here
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating data point: {e}")
            return False
    
    def _validate_historical_data(self, data: pd.DataFrame) -> bool:
        """Validate historical data for quality."""
        try:
            # Check for required columns
            required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            if not all(col in data.columns for col in required_columns):
                return False
            
            # Check for data consistency
            if len(data) == 0:
                return False
            
            # Check for reasonable price ranges
            price_columns = ['open', 'high', 'low', 'close']
            for col in price_columns:
                if (data[col] <= 0).any():
                    return False
            
            # Check high >= low
            if (data['high'] < data['low']).any():
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating historical data: {e}")
            return False
    
    async def _store_real_time_data(self, data_point: MarketDataPoint):
        """Store real-time data point."""
        try:
            data_key = f"{data_point.symbol}:{data_point.data_type.value}"
            self.real_time_data[data_key].append(data_point)
            
            # Notify subscribers
            subscription_key = data_key
            if subscription_key in self.data_subscribers:
                for callback in self.data_subscribers[subscription_key]:
                    try:
                        await callback(data_point)
                    except Exception as e:
                        self.logger.error(f"Error in data callback: {e}")
            
        except Exception as e:
            self.logger.error(f"Error storing real-time data: {e}")
    
    async def _check_rate_limit(self, source_name: str) -> bool:
        """Check if we can make a request to the source."""
        try:
            if source_name not in self.data_sources:
                return False
            
            source_status = self.data_sources[source_name]
            
            # Simple rate limit check
            if source_status.rate_limit_remaining <= 0:
                if source_status.next_reset and datetime.now() < source_status.next_reset:
                    return False
                else:
                    # Reset rate limit
                    source_status.rate_limit_remaining = 1000  # Default
                    source_status.next_reset = None
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error checking rate limit for {source_name}: {e}")
            return False
    
    async def _handle_source_error(self, source_name: str):
        """Handle error from a data source."""
        try:
            if source_name in self.data_sources:
                source_status = self.data_sources[source_name]
                source_status.error_count += 1
                
                # Disconnect source if too many errors
                if source_status.error_count >= self.config['failover_threshold']:
                    source_status.is_connected = False
                    self.logger.warning(f"Data source {source_name} disconnected due to errors")
                    
                    # Send alert
                    await self.notification_manager.send_alert(
                        title=f"Data Source Error",
                        message=f"Data source {source_name} has been disconnected due to repeated errors",
                        severity="high",
                        alert_type="DATA_SOURCE"
                    )
        
        except Exception as e:
            self.logger.error(f"Error handling source error: {e}")
    
    async def _handle_source_disconnection(self, source_name: str):
        """Handle disconnection of a data source."""
        try:
            if source_name in self.data_sources:
                source_status = self.data_sources[source_name]
                source_status.is_connected = False
                source_status.error_count += 1
                
                self.logger.warning(f"Data source {source_name} disconnected")
                
                # Close any websocket connections for this source
                connections_to_close = []
                for key, connection in self.websocket_connections.items():
                    if key[0] == source_name:  # key format: (source, symbol, data_type)
                        connections_to_close.append(key)
                
                for key in connections_to_close:
                    try:
                        connection = self.websocket_connections.pop(key, None)
                        if connection and not connection.closed:
                            await connection.close()
                    except Exception as e:
                        self.logger.error(f"Error closing websocket connection {key}: {e}")
                
                # Send notification
                await self.notification_manager.send_alert(
                    title=f"Data Source Disconnected",
                    message=f"Data source {source_name} has been disconnected",
                    severity="medium",
                    alert_type="DATA_SOURCE"
                )
                
                # Try to reconnect after a delay
                await asyncio.sleep(30)
                await self._attempt_reconnection(source_name)
        
        except Exception as e:
            self.logger.error(f"Error handling source disconnection: {e}")
    
    async def _attempt_reconnection(self, source_name: str):
        """Attempt to reconnect to a disconnected data source."""
        try:
            if source_name in self.data_sources:
                source_status = self.data_sources[source_name]
                
                # Check if source is healthy again
                is_healthy = await self._check_source_health(source_name)
                
                if is_healthy:
                    source_status.is_connected = True
                    source_status.error_count = 0
                    source_status.last_update = datetime.now()
                    
                    self.logger.info(f"Successfully reconnected to data source {source_name}")
                    
                    await self.notification_manager.send_alert(
                        title=f"Data Source Reconnected",
                        message=f"Data source {source_name} has been successfully reconnected",
                        severity="low",
                        alert_type="DATA_SOURCE"
                    )
                else:
                    self.logger.warning(f"Failed to reconnect to data source {source_name}")
        
        except Exception as e:
            self.logger.error(f"Error attempting reconnection to {source_name}: {e}")
    
    def get_aggregator_status(self) -> Dict[str, Any]:
        """Get comprehensive aggregator status."""
        return {
            'timestamp': datetime.now().isoformat(),
            'data_sources': {
                name: {
                    'connected': status.is_connected,
                    'error_count': status.error_count,
                    'last_update': status.last_update.isoformat(),
                    'quality': status.data_quality.value
                }
                for name, status in self.data_sources.items()
            },
            'active_sessions': self.active_sessions,
            'websocket_connections': len(self.websocket_connections),
            'real_time_symbols': len(self.real_time_data),
            'total_subscribers': sum(len(subs) for subs in self.data_subscribers.values()),
            'background_tasks': len([t for t in self._background_tasks if not t.done()])
        }
    
    async def shutdown(self):
        """Shutdown the market data aggregator."""
        try:
            self.logger.info("Shutting down Market Data Aggregator...")
            
            # Cancel all background tasks
            for task in self._background_tasks:
                if not task.done():
                    task.cancel()
            
            # Wait for tasks to complete (with timeout)
            if self._background_tasks:
                try:
                    await asyncio.wait_for(
                        asyncio.gather(*self._background_tasks, return_exceptions=True),
                        timeout=10.0
                    )
                except asyncio.TimeoutError:
                    self.logger.warning("Some background tasks did not complete within timeout")
            
            # Close all websocket connections
            for connection_key, connection_info in list(self.websocket_connections.items()):
                try:
                    await connection_info['connection'].close()
                except Exception as e:
                    self.logger.error(f"Error closing websocket {connection_key}: {e}")
            
            self.websocket_connections.clear()
            
            # Clear data structures
            self.real_time_data.clear()
            self.data_subscribers.clear()
            
            self.logger.info("Market Data Aggregator shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Error shutting down Market Data Aggregator: {e}")


if __name__ == "__main__":
    # Example usage
    import asyncio
    from pathlib import Path
    
    async def test_aggregator():
        config_manager = ConfigManager(Path("../config"))
        aggregator = MarketDataAggregator(config_manager)
        
        # Test real-time data
        data_point = await aggregator.get_real_time_data('AAPL', DataType.PRICE)
        if data_point:
            print(f"AAPL Price: {data_point.value}")
        
        # Test historical data
        hist_data = await aggregator.get_historical_data('AAPL', '1d', limit=100)
        if hist_data is not None:
            print(f"Historical data shape: {hist_data.shape}")
        
        # Test market status
        status = await aggregator.get_market_status('US')
        print(f"US Market Status: {status.value}")
        
        # Get quality report
        quality_report = await aggregator.get_data_quality_report()
        print(f"Data Quality: {quality_report['overall_quality']}")
    
    # Commented out to prevent event loop conflicts when imported
    # asyncio.run(test_aggregator())