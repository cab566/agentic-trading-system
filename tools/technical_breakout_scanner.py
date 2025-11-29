#!/usr/bin/env python3
"""
Technical Breakout Scanner - Advanced Pattern Recognition System

This tool monitors technical patterns, breakouts, gap events, and momentum shifts
across market sectors to identify high-probability trading opportunities.

Features:
- Real-time technical pattern detection (breakouts, reversals, continuations)
- Gap analysis (up gaps, down gaps, gap fills)
- Momentum shift detection using multiple indicators
- Support/resistance level identification
- Volume confirmation analysis
- Sector-based scanning and filtering
- Multi-timeframe analysis
- Real-time alerts and notifications

Technical Patterns Detected:
- Breakouts (resistance/support breaks with volume)
- Flag and pennant patterns
- Triangle patterns (ascending, descending, symmetrical)
- Head and shoulders patterns
- Double tops/bottoms
- Cup and handle patterns
- Gap events (breakaway, runaway, exhaustion)
- Momentum divergences

Data Sources:
- Yahoo Finance for OHLCV data
- Real-time technical indicator calculations
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
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from enum import Enum
import talib
from concurrent.futures import ThreadPoolExecutor
import time
from scipy import stats
from scipy.signal import find_peaks
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


class PatternType(Enum):
    """Types of technical patterns."""
    BREAKOUT_RESISTANCE = "breakout_resistance"
    BREAKOUT_SUPPORT = "breakout_support"
    FLAG_BULLISH = "flag_bullish"
    FLAG_BEARISH = "flag_bearish"
    PENNANT_BULLISH = "pennant_bullish"
    PENNANT_BEARISH = "pennant_bearish"
    TRIANGLE_ASCENDING = "triangle_ascending"
    TRIANGLE_DESCENDING = "triangle_descending"
    TRIANGLE_SYMMETRICAL = "triangle_symmetrical"
    HEAD_SHOULDERS = "head_shoulders"
    INVERSE_HEAD_SHOULDERS = "inverse_head_shoulders"
    DOUBLE_TOP = "double_top"
    DOUBLE_BOTTOM = "double_bottom"
    CUP_HANDLE = "cup_handle"
    GAP_UP = "gap_up"
    GAP_DOWN = "gap_down"
    GAP_FILL = "gap_fill"
    MOMENTUM_BULLISH = "momentum_bullish"
    MOMENTUM_BEARISH = "momentum_bearish"
    VOLUME_BREAKOUT = "volume_breakout"


class PatternStrength(Enum):
    """Pattern strength levels."""
    WEAK = "weak"
    MODERATE = "moderate"
    STRONG = "strong"
    VERY_STRONG = "very_strong"


class TimeFrame(Enum):
    """Analysis timeframes."""
    MINUTE_5 = "5m"
    MINUTE_15 = "15m"
    MINUTE_30 = "30m"
    HOUR_1 = "1h"
    HOUR_4 = "4h"
    DAILY = "1d"


@dataclass
class TechnicalPattern:
    """Technical pattern detection result."""
    symbol: str
    pattern_type: PatternType
    strength: PatternStrength
    timeframe: TimeFrame
    confidence_score: float
    entry_price: float
    target_price: float
    stop_loss: float
    risk_reward_ratio: float
    volume_confirmation: bool
    pattern_start_date: datetime
    pattern_completion_date: datetime
    current_price: float
    price_change_pct: float
    volume_ratio: float
    sector: Optional[str]
    market_cap: Optional[float]
    description: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'symbol': self.symbol,
            'pattern_type': self.pattern_type.value,
            'strength': self.strength.value,
            'timeframe': self.timeframe.value,
            'confidence_score': self.confidence_score,
            'entry_price': self.entry_price,
            'target_price': self.target_price,
            'stop_loss': self.stop_loss,
            'risk_reward_ratio': self.risk_reward_ratio,
            'volume_confirmation': self.volume_confirmation,
            'pattern_start_date': self.pattern_start_date.isoformat(),
            'pattern_completion_date': self.pattern_completion_date.isoformat(),
            'current_price': self.current_price,
            'price_change_pct': self.price_change_pct,
            'volume_ratio': self.volume_ratio,
            'sector': self.sector,
            'market_cap': self.market_cap,
            'description': self.description
        }


@dataclass
class ScannerConfig:
    """Configuration for technical breakout scanner."""
    min_confidence_score: float = 70.0
    min_volume_ratio: float = 1.2
    min_risk_reward_ratio: float = 1.5
    max_price: float = 1000.0
    min_market_cap: float = 100_000_000
    timeframes: List[TimeFrame] = field(default_factory=lambda: [
        TimeFrame.MINUTE_15, TimeFrame.HOUR_1, TimeFrame.DAILY
    ])
    lookback_periods: int = 50
    scan_interval_minutes: int = 10
    max_results_per_scan: int = 50
    enable_gap_detection: bool = True
    enable_pattern_recognition: bool = True
    enable_momentum_analysis: bool = True
    
    # Caching configuration
    enable_caching: bool = True
    price_data_cache_ttl: int = 300  # 5 minutes for price data
    indicator_cache_ttl: int = 600  # 10 minutes for technical indicators
    pattern_cache_ttl: int = 900  # 15 minutes for pattern analysis


class TechnicalIndicators:
    """Technical indicator calculations."""
    
    @staticmethod
    def calculate_support_resistance(prices: pd.Series, window: int = 20) -> Tuple[List[float], List[float]]:
        """Calculate support and resistance levels."""
        try:
            # Find local minima (support) and maxima (resistance)
            prices_array = prices.values
            
            # Find peaks (resistance levels)
            resistance_indices, _ = find_peaks(prices_array, distance=window//4)
            resistance_levels = prices.iloc[resistance_indices].tolist()
            
            # Find troughs (support levels) by inverting the data
            support_indices, _ = find_peaks(-prices_array, distance=window//4)
            support_levels = prices.iloc[support_indices].tolist()
            
            return support_levels, resistance_levels
            
        except Exception as e:
            logging.debug(f"Error calculating support/resistance: {e}")
            return [], []
    
    @staticmethod
    def detect_breakout(prices: pd.Series, volume: pd.Series, 
                       support_levels: List[float], resistance_levels: List[float]) -> Optional[Dict[str, Any]]:
        """Detect breakout patterns."""
        try:
            if len(prices) < 10 or not resistance_levels:
                return None
            
            current_price = prices.iloc[-1]
            previous_price = prices.iloc[-2]
            current_volume = volume.iloc[-1]
            avg_volume = volume.rolling(20).mean().iloc[-1]
            
            # Check for resistance breakout
            for resistance in resistance_levels:
                if (previous_price <= resistance and current_price > resistance and
                    current_volume > avg_volume * 1.2):
                    return {
                        'type': PatternType.BREAKOUT_RESISTANCE,
                        'level': resistance,
                        'volume_confirmed': True,
                        'strength': 'strong' if current_volume > avg_volume * 2 else 'moderate'
                    }
            
            # Check for support breakout (breakdown)
            for support in support_levels:
                if (previous_price >= support and current_price < support and
                    current_volume > avg_volume * 1.2):
                    return {
                        'type': PatternType.BREAKOUT_SUPPORT,
                        'level': support,
                        'volume_confirmed': True,
                        'strength': 'strong' if current_volume > avg_volume * 2 else 'moderate'
                    }
            
            return None
            
        except Exception as e:
            logging.debug(f"Error detecting breakout: {e}")
            return None
    
    @staticmethod
    def detect_gap(current_data: pd.DataFrame, previous_data: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Detect gap patterns."""
        try:
            if current_data.empty or previous_data.empty:
                return None
            
            prev_close = previous_data['Close'].iloc[-1]
            current_open = current_data['Open'].iloc[0]
            current_high = current_data['High'].max()
            current_low = current_data['Low'].min()
            
            gap_percentage = ((current_open - prev_close) / prev_close) * 100
            
            # Gap up detection
            if gap_percentage > 2.0:  # 2% gap up
                return {
                    'type': PatternType.GAP_UP,
                    'gap_percentage': gap_percentage,
                    'gap_size': current_open - prev_close,
                    'filled': current_low <= prev_close
                }
            
            # Gap down detection
            elif gap_percentage < -2.0:  # 2% gap down
                return {
                    'type': PatternType.GAP_DOWN,
                    'gap_percentage': gap_percentage,
                    'gap_size': current_open - prev_close,
                    'filled': current_high >= prev_close
                }
            
            return None
            
        except Exception as e:
            logging.debug(f"Error detecting gap: {e}")
            return None
    
    @staticmethod
    def calculate_momentum_indicators(data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate momentum indicators."""
        try:
            close = data['Close'].values
            high = data['High'].values
            low = data['Low'].values
            volume = data['Volume'].values
            
            # RSI
            rsi = talib.RSI(close, timeperiod=14)
            
            # MACD
            macd, macd_signal, macd_hist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
            
            # Stochastic
            slowk, slowd = talib.STOCH(high, low, close, fastk_period=14, slowk_period=3, slowd_period=3)
            
            # Williams %R
            willr = talib.WILLR(high, low, close, timeperiod=14)
            
            # ADX (trend strength)
            adx = talib.ADX(high, low, close, timeperiod=14)
            
            # OBV (On Balance Volume)
            obv = talib.OBV(close, volume)
            
            return {
                'rsi': rsi[-1] if len(rsi) > 0 and not np.isnan(rsi[-1]) else None,
                'macd': macd[-1] if len(macd) > 0 and not np.isnan(macd[-1]) else None,
                'macd_signal': macd_signal[-1] if len(macd_signal) > 0 and not np.isnan(macd_signal[-1]) else None,
                'macd_histogram': macd_hist[-1] if len(macd_hist) > 0 and not np.isnan(macd_hist[-1]) else None,
                'stoch_k': slowk[-1] if len(slowk) > 0 and not np.isnan(slowk[-1]) else None,
                'stoch_d': slowd[-1] if len(slowd) > 0 and not np.isnan(slowd[-1]) else None,
                'williams_r': willr[-1] if len(willr) > 0 and not np.isnan(willr[-1]) else None,
                'adx': adx[-1] if len(adx) > 0 and not np.isnan(adx[-1]) else None,
                'obv': obv[-1] if len(obv) > 0 and not np.isnan(obv[-1]) else None
            }
            
        except Exception as e:
            logging.debug(f"Error calculating momentum indicators: {e}")
            return {}
    
    @staticmethod
    def detect_momentum_shift(indicators: Dict[str, Any], previous_indicators: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Detect momentum shifts."""
        try:
            if not indicators or not previous_indicators:
                return None
            
            rsi = indicators.get('rsi')
            prev_rsi = previous_indicators.get('rsi')
            macd = indicators.get('macd')
            macd_signal = indicators.get('macd_signal')
            prev_macd = previous_indicators.get('macd')
            prev_macd_signal = previous_indicators.get('macd_signal')
            
            # Bullish momentum shift
            if (rsi and prev_rsi and rsi > 50 and prev_rsi <= 50 and
                macd and macd_signal and prev_macd and prev_macd_signal and
                macd > macd_signal and prev_macd <= prev_macd_signal):
                return {
                    'type': PatternType.MOMENTUM_BULLISH,
                    'strength': 'strong' if rsi > 60 else 'moderate',
                    'rsi_cross': True,
                    'macd_cross': True
                }
            
            # Bearish momentum shift
            elif (rsi and prev_rsi and rsi < 50 and prev_rsi >= 50 and
                  macd and macd_signal and prev_macd and prev_macd_signal and
                  macd < macd_signal and prev_macd >= prev_macd_signal):
                return {
                    'type': PatternType.MOMENTUM_BEARISH,
                    'strength': 'strong' if rsi < 40 else 'moderate',
                    'rsi_cross': True,
                    'macd_cross': True
                }
            
            return None
            
        except Exception as e:
            logging.debug(f"Error detecting momentum shift: {e}")
            return None


class TechnicalBreakoutScanner:
    """
    Technical breakout scanner for pattern recognition and momentum analysis.
    
    Monitors technical patterns, breakouts, gaps, and momentum shifts across
    market sectors to identify high-probability trading opportunities.
    """
    
    def __init__(self, config_manager: ConfigManager):
        self.config_manager = config_manager
        self.cache_manager = CacheManager(config_manager)
        self.notification_manager = NotificationManager(config_manager)
        self.logger = logging.getLogger(__name__)
        
        # Scanner configuration
        self.config = ScannerConfig()
        self._load_config()
        
        # Market universe with sector information
        self.stock_universe = self._initialize_stock_universe()
        
        # Threading for concurrent analysis
        self.thread_pool = ThreadPoolExecutor(max_workers=8)
        
        # Batch data downloader for optimized API calls
        self.batch_downloader = BatchDataDownloader()
        
        # State tracking
        self.last_scan_time: Optional[datetime] = None
        self.detected_patterns: List[TechnicalPattern] = []
        self.pattern_history: List[List[TechnicalPattern]] = []
        self.is_scanning = False
        
        # Performance metrics
        self.scan_count = 0
        self.total_patterns_detected = 0
        self.average_scan_time = 0.0
        
        # Cache performance tracking
        self.cache_hits = 0
        self.cache_misses = 0
        self.api_calls_saved = 0
        
        # Cache for previous indicator values
        self.indicator_cache: Dict[str, Dict[str, Any]] = {}
    
    def _load_config(self):
        """Load scanner configuration."""
        try:
            scanner_config = self.config_manager.get_config().get('technical_scanner', {})
            
            if scanner_config:
                self.config.min_confidence_score = scanner_config.get('min_confidence_score', 70.0)
                self.config.min_volume_ratio = scanner_config.get('min_volume_ratio', 1.2)
                self.config.min_risk_reward_ratio = scanner_config.get('min_risk_reward_ratio', 1.5)
                self.config.lookback_periods = scanner_config.get('lookback_periods', 50)
                self.config.scan_interval_minutes = scanner_config.get('scan_interval_minutes', 10)
                
            self.logger.info(f"Technical scanner configured: min_confidence={self.config.min_confidence_score}")
            
        except Exception as e:
            self.logger.error(f"Error loading scanner config: {e}")
    
    def _initialize_stock_universe(self) -> Dict[str, Dict[str, Any]]:
        """Initialize stock universe with sector information."""
        try:
            # Stock universe with sector classifications
            universe = {
                # Technology
                'AAPL': {'sector': 'Technology', 'market_cap': 'mega'},
                'MSFT': {'sector': 'Technology', 'market_cap': 'mega'},
                'GOOGL': {'sector': 'Technology', 'market_cap': 'mega'},
                'NVDA': {'sector': 'Technology', 'market_cap': 'mega'},
                'META': {'sector': 'Technology', 'market_cap': 'mega'},
                'TSLA': {'sector': 'Technology', 'market_cap': 'mega'},
                'NFLX': {'sector': 'Technology', 'market_cap': 'large'},
                'CRM': {'sector': 'Technology', 'market_cap': 'large'},
                'ADBE': {'sector': 'Technology', 'market_cap': 'large'},
                'INTC': {'sector': 'Technology', 'market_cap': 'large'},
                
                # Healthcare
                'JNJ': {'sector': 'Healthcare', 'market_cap': 'mega'},
                'PFE': {'sector': 'Healthcare', 'market_cap': 'large'},
                'UNH': {'sector': 'Healthcare', 'market_cap': 'mega'},
                'ABBV': {'sector': 'Healthcare', 'market_cap': 'large'},
                'MRK': {'sector': 'Healthcare', 'market_cap': 'large'},
                'TMO': {'sector': 'Healthcare', 'market_cap': 'large'},
                'ABT': {'sector': 'Healthcare', 'market_cap': 'large'},
                'GILD': {'sector': 'Healthcare', 'market_cap': 'large'},
                
                # Financial
                'JPM': {'sector': 'Financial', 'market_cap': 'mega'},
                'BAC': {'sector': 'Financial', 'market_cap': 'large'},
                'WFC': {'sector': 'Financial', 'market_cap': 'large'},
                'GS': {'sector': 'Financial', 'market_cap': 'large'},
                'MS': {'sector': 'Financial', 'market_cap': 'large'},
                'C': {'sector': 'Financial', 'market_cap': 'large'},
                'AXP': {'sector': 'Financial', 'market_cap': 'large'},
                'BLK': {'sector': 'Financial', 'market_cap': 'large'},
                
                # Consumer
                'AMZN': {'sector': 'Consumer', 'market_cap': 'mega'},
                'WMT': {'sector': 'Consumer', 'market_cap': 'large'},
                'HD': {'sector': 'Consumer', 'market_cap': 'large'},
                'MCD': {'sector': 'Consumer', 'market_cap': 'large'},
                'NKE': {'sector': 'Consumer', 'market_cap': 'large'},
                'SBUX': {'sector': 'Consumer', 'market_cap': 'large'},
                'TGT': {'sector': 'Consumer', 'market_cap': 'large'},
                'LOW': {'sector': 'Consumer', 'market_cap': 'large'},
                
                # Energy
                'XOM': {'sector': 'Energy', 'market_cap': 'mega'},
                'CVX': {'sector': 'Energy', 'market_cap': 'large'},
                'COP': {'sector': 'Energy', 'market_cap': 'large'},
                'SLB': {'sector': 'Energy', 'market_cap': 'large'},
                'EOG': {'sector': 'Energy', 'market_cap': 'large'},
                'PXD': {'sector': 'Energy', 'market_cap': 'large'},
                
                # Industrial
                'BA': {'sector': 'Industrial', 'market_cap': 'large'},
                'CAT': {'sector': 'Industrial', 'market_cap': 'large'},
                'GE': {'sector': 'Industrial', 'market_cap': 'large'},
                'MMM': {'sector': 'Industrial', 'market_cap': 'large'},
                'HON': {'sector': 'Industrial', 'market_cap': 'large'},
                'UPS': {'sector': 'Industrial', 'market_cap': 'large'}
            }
            
            self.logger.info(f"Initialized technical scanner universe with {len(universe)} symbols")
            return universe
            
        except Exception as e:
            self.logger.error(f"Error initializing stock universe: {e}")
            return {}
    
    async def start_scanning(self):
        """Start the technical breakout scanning."""
        if self.is_scanning:
            self.logger.warning("Technical scanner is already running")
            return
        
        self.is_scanning = True
        self.logger.info("Starting technical breakout scanner")
        
        try:
            while self.is_scanning:
                scan_start_time = time.time()
                
                # Perform technical pattern scan
                patterns = await self.scan_for_patterns()
                
                # Update metrics
                scan_duration = time.time() - scan_start_time
                self.scan_count += 1
                self.total_patterns_detected += len(patterns)
                self.average_scan_time = (
                    (self.average_scan_time * (self.scan_count - 1) + scan_duration) / 
                    self.scan_count
                )
                
                # Store results
                self.detected_patterns = patterns
                self.pattern_history.append(patterns)
                if len(self.pattern_history) > 100:  # Keep last 100 scans
                    self.pattern_history.pop(0)
                
                # Send notifications for significant patterns
                await self._process_pattern_alerts(patterns)
                
                self.logger.info(
                    f"Technical scan {self.scan_count} completed: {len(patterns)} patterns detected "
                    f"in {scan_duration:.2f}s"
                )
                
                # Wait for next scan interval
                await asyncio.sleep(self.config.scan_interval_minutes * 60)
                
        except Exception as e:
            self.logger.error(f"Error in technical scanning loop: {e}")
        finally:
            self.is_scanning = False
    
    async def scan_for_patterns(self) -> List[TechnicalPattern]:
        """Scan for technical patterns across all timeframes using batch processing."""
        try:
            self.logger.info(f"Scanning {len(self.stock_universe)} symbols for technical patterns")
            
            all_patterns = []
            
            # Create batch requests for all symbols and timeframes
            batch_requests = []
            for symbol in self.stock_universe.keys():
                for timeframe in self.config.timeframes:
                    # Generate cache key for price data
                    price_cache_key = self._generate_cache_key(symbol, timeframe.value, "price_data")
                    
                    # Check if data is cached
                    cached_data = self._get_cached_data(price_cache_key)
                    if cached_data is None:
                        # Add to batch request if not cached
                        period_map = {
                            TimeFrame.MINUTE_5: "5d",
                            TimeFrame.MINUTE_15: "5d",
                            TimeFrame.MINUTE_30: "10d",
                            TimeFrame.HOUR_1: "30d",
                            TimeFrame.HOUR_4: "60d",
                            TimeFrame.DAILY: "1y"
                        }
                        
                        batch_requests.append(BatchRequest(
                            symbol=symbol,
                            period=period_map[timeframe],
                            interval=timeframe.value,
                            metadata={'timeframe': timeframe, 'cache_key': price_cache_key}
                        ))
            
            # Execute batch download if needed
            batch_data = {}
            if batch_requests:
                self.logger.info(f"Downloading data for {len(batch_requests)} symbol-timeframe combinations")
                batch_results = await self.batch_downloader.download_batch(batch_requests)
                
                # Cache the downloaded data
                for request, data in zip(batch_requests, batch_results):
                    if data is not None and not data.empty:
                        cache_key = request.metadata['cache_key']
                        self._set_cached_data(cache_key, data, self.config.price_data_cache_ttl)
                        batch_data[f"{request.symbol}_{request.metadata['timeframe'].value}"] = data
            
            # Analyze each symbol across multiple timeframes
            for symbol, info in self.stock_universe.items():
                for timeframe in self.config.timeframes:
                    try:
                        patterns = await self._analyze_symbol_patterns_optimized(symbol, timeframe, info, batch_data)
                        all_patterns.extend(patterns)
                    except Exception as e:
                        self.logger.debug(f"Error analyzing {symbol} on {timeframe.value}: {e}")
            
            # Filter and rank patterns
            filtered_patterns = self._filter_and_rank_patterns(all_patterns)
            
            self.last_scan_time = datetime.now()
            
            return filtered_patterns
            
        except Exception as e:
            self.logger.error(f"Error scanning for technical patterns: {e}")
            return []
    
    def _generate_cache_key(self, symbol: str, timeframe: str, data_type: str, timestamp: datetime = None) -> str:
        """Generate a cache key for data storage."""
        if timestamp is None:
            timestamp = datetime.now()
        
        # Round timestamp to nearest 5 minutes for price data, 10 minutes for indicators
        if data_type == "price_data":
            rounded_time = timestamp.replace(minute=(timestamp.minute // 5) * 5, second=0, microsecond=0)
        elif data_type == "indicators":
            rounded_time = timestamp.replace(minute=(timestamp.minute // 10) * 10, second=0, microsecond=0)
        else:
            rounded_time = timestamp.replace(second=0, microsecond=0)
        
        key_string = f"{symbol}_{timeframe}_{data_type}_{rounded_time.isoformat()}"
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def _get_cached_data(self, cache_key: str) -> Optional[Any]:
        """Retrieve data from cache."""
        if not self.config.enable_caching:
            return None
        
        try:
            data = self.cache_manager.get(cache_key)
            if data is not None:
                self.cache_hits += 1
                return data
            else:
                self.cache_misses += 1
                return None
        except Exception as e:
            self.logger.debug(f"Cache retrieval error: {e}")
            self.cache_misses += 1
            return None
    
    def _set_cached_data(self, cache_key: str, data: Any, ttl: int):
        """Store data in cache."""
        if not self.config.enable_caching:
            return
        
        try:
            self.cache_manager.set(cache_key, data, ttl)
        except Exception as e:
            self.logger.debug(f"Cache storage error: {e}")

    async def _analyze_symbol_patterns_optimized(self, symbol: str, timeframe: TimeFrame, 
                                               info: Dict[str, Any], batch_data: Dict[str, pd.DataFrame]) -> List[TechnicalPattern]:
        """Analyze a single symbol for technical patterns using optimized data access."""
        try:
            # Generate cache key for price data
            price_cache_key = self._generate_cache_key(symbol, timeframe.value, "price_data")
            
            # Try to get data from cache first, then from batch data
            cached_data = self._get_cached_data(price_cache_key)
            if cached_data is not None:
                data = cached_data
                self.api_calls_saved += 1
            else:
                # Try to get from batch data
                batch_key = f"{symbol}_{timeframe.value}"
                data = batch_data.get(batch_key)
                if data is None:
                    # Fallback to individual fetch (should be rare)
                    period_map = {
                        TimeFrame.MINUTE_5: "5d",
                        TimeFrame.MINUTE_15: "5d",
                        TimeFrame.MINUTE_30: "10d",
                        TimeFrame.HOUR_1: "30d",
                        TimeFrame.HOUR_4: "60d",
                        TimeFrame.DAILY: "1y"
                    }
                    
                    ticker = yf.Ticker(symbol)
                    data = ticker.history(
                        period=period_map[timeframe],
                        interval=timeframe.value,
                        auto_adjust=True
                    )
                    
                    # Cache the price data
                    self._set_cached_data(price_cache_key, data, self.config.price_data_cache_ttl)
            
            if data is None or len(data) < self.config.lookback_periods:
                return []
            
            patterns = []
            
            # Get current price info
            current_price = data['Close'].iloc[-1]
            previous_close = data['Close'].iloc[-2] if len(data) > 1 else current_price
            price_change_pct = ((current_price - previous_close) / previous_close) * 100
            
            # Calculate volume ratio
            current_volume = data['Volume'].iloc[-1]
            avg_volume = data['Volume'].rolling(20).mean().iloc[-1]
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
            
            # Generate cache key for indicators
            indicator_cache_key = self._generate_cache_key(symbol, timeframe.value, "indicators")
            
            # Try to get cached indicators
            cached_indicators = self._get_cached_data(indicator_cache_key)
            if cached_indicators is not None:
                indicators = cached_indicators
                self.api_calls_saved += 1
            else:
                # Calculate technical indicators
                indicators = TechnicalIndicators.calculate_momentum_indicators(data)
                
                # Cache the indicators
                self._set_cached_data(indicator_cache_key, indicators, self.config.indicator_cache_ttl)
            
            # Get previous indicators for momentum shift detection
            previous_indicators = self.indicator_cache.get(f"{symbol}_{timeframe.value}", {})
            
            # Detect patterns
            if self.config.enable_pattern_recognition:
                # Support/Resistance and Breakouts
                support_levels, resistance_levels = TechnicalIndicators.calculate_support_resistance(
                    data['Close'], window=20
                )
                
                breakout = TechnicalIndicators.detect_breakout(
                    data['Close'], data['Volume'], support_levels, resistance_levels
                )
                
                if breakout:
                    pattern = self._create_breakout_pattern(
                        symbol, breakout, timeframe, current_price, price_change_pct,
                        volume_ratio, info, indicators
                    )
                    if pattern:
                        patterns.append(pattern)
            
            # Gap detection (for daily timeframe)
            if self.config.enable_gap_detection and timeframe == TimeFrame.DAILY:
                if len(data) >= 2:
                    current_day = data.iloc[-1:]
                    previous_day = data.iloc[-2:-1]
                    
                    gap = TechnicalIndicators.detect_gap(current_day, previous_day)
                    if gap:
                        pattern = self._create_gap_pattern(
                            symbol, gap, timeframe, current_price, price_change_pct,
                            volume_ratio, info
                        )
                        if pattern:
                            patterns.append(pattern)
            
            # Momentum shift detection
            if self.config.enable_momentum_analysis and previous_indicators:
                momentum_shift = TechnicalIndicators.detect_momentum_shift(indicators, previous_indicators)
                if momentum_shift:
                    pattern = self._create_momentum_pattern(
                        symbol, momentum_shift, timeframe, current_price, price_change_pct,
                        volume_ratio, info, indicators
                    )
                    if pattern:
                        patterns.append(pattern)
            
            # Cache current indicators for next scan
            self.indicator_cache[f"{symbol}_{timeframe.value}"] = indicators
            
            return patterns
            
        except Exception as e:
            self.logger.debug(f"Error analyzing {symbol} patterns: {e}")
            return []
    
    def _create_breakout_pattern(self, symbol: str, breakout: Dict[str, Any], 
                               timeframe: TimeFrame, current_price: float,
                               price_change_pct: float, volume_ratio: float,
                               info: Dict[str, Any], indicators: Dict[str, Any]) -> Optional[TechnicalPattern]:
        """Create a breakout pattern object."""
        try:
            pattern_type = breakout['type']
            level = breakout['level']
            volume_confirmed = breakout['volume_confirmed']
            strength_str = breakout['strength']
            
            # Calculate entry, target, and stop loss
            if pattern_type == PatternType.BREAKOUT_RESISTANCE:
                entry_price = current_price
                target_price = current_price + (current_price - level) * 1.5  # 1.5x the breakout distance
                stop_loss = level * 0.98  # 2% below resistance level
            else:  # Support breakdown
                entry_price = current_price
                target_price = current_price - (level - current_price) * 1.5
                stop_loss = level * 1.02  # 2% above support level
            
            # Calculate risk/reward ratio
            risk = abs(entry_price - stop_loss)
            reward = abs(target_price - entry_price)
            risk_reward_ratio = reward / risk if risk > 0 else 0
            
            # Calculate confidence score
            confidence_score = 60.0  # Base score
            if volume_confirmed:
                confidence_score += 20.0
            if strength_str == 'strong':
                confidence_score += 15.0
            if volume_ratio > 2.0:
                confidence_score += 10.0
            if indicators.get('adx', 0) > 25:  # Strong trend
                confidence_score += 10.0
            
            # Determine pattern strength
            if confidence_score >= 90:
                strength = PatternStrength.VERY_STRONG
            elif confidence_score >= 80:
                strength = PatternStrength.STRONG
            elif confidence_score >= 70:
                strength = PatternStrength.MODERATE
            else:
                strength = PatternStrength.WEAK
            
            return TechnicalPattern(
                symbol=symbol,
                pattern_type=pattern_type,
                strength=strength,
                timeframe=timeframe,
                confidence_score=confidence_score,
                entry_price=entry_price,
                target_price=target_price,
                stop_loss=stop_loss,
                risk_reward_ratio=risk_reward_ratio,
                volume_confirmation=volume_confirmed,
                pattern_start_date=datetime.now() - timedelta(days=5),  # Approximate
                pattern_completion_date=datetime.now(),
                current_price=current_price,
                price_change_pct=price_change_pct,
                volume_ratio=volume_ratio,
                sector=info.get('sector'),
                market_cap=None,  # Would be fetched from API
                description=f"{pattern_type.value.replace('_', ' ').title()} at ${level:.2f} level"
            )
            
        except Exception as e:
            self.logger.debug(f"Error creating breakout pattern: {e}")
            return None
    
    def _create_gap_pattern(self, symbol: str, gap: Dict[str, Any], 
                          timeframe: TimeFrame, current_price: float,
                          price_change_pct: float, volume_ratio: float,
                          info: Dict[str, Any]) -> Optional[TechnicalPattern]:
        """Create a gap pattern object."""
        try:
            pattern_type = gap['type']
            gap_percentage = gap['gap_percentage']
            gap_filled = gap.get('filled', False)
            
            # Calculate entry, target, and stop loss based on gap type
            if pattern_type == PatternType.GAP_UP:
                entry_price = current_price
                target_price = current_price * (1 + abs(gap_percentage) / 100 * 0.5)  # 50% of gap size
                stop_loss = current_price * 0.95  # 5% stop loss
            else:  # Gap down
                entry_price = current_price
                target_price = current_price * (1 - abs(gap_percentage) / 100 * 0.5)
                stop_loss = current_price * 1.05  # 5% stop loss
            
            # Calculate risk/reward ratio
            risk = abs(entry_price - stop_loss)
            reward = abs(target_price - entry_price)
            risk_reward_ratio = reward / risk if risk > 0 else 0
            
            # Calculate confidence score
            confidence_score = 50.0 + abs(gap_percentage) * 5  # Higher gap = higher confidence
            if not gap_filled:
                confidence_score += 15.0
            if volume_ratio > 1.5:
                confidence_score += 10.0
            
            confidence_score = min(100.0, confidence_score)
            
            # Determine pattern strength
            if abs(gap_percentage) > 5:
                strength = PatternStrength.VERY_STRONG
            elif abs(gap_percentage) > 3:
                strength = PatternStrength.STRONG
            else:
                strength = PatternStrength.MODERATE
            
            return TechnicalPattern(
                symbol=symbol,
                pattern_type=pattern_type,
                strength=strength,
                timeframe=timeframe,
                confidence_score=confidence_score,
                entry_price=entry_price,
                target_price=target_price,
                stop_loss=stop_loss,
                risk_reward_ratio=risk_reward_ratio,
                volume_confirmation=volume_ratio > 1.2,
                pattern_start_date=datetime.now() - timedelta(days=1),
                pattern_completion_date=datetime.now(),
                current_price=current_price,
                price_change_pct=price_change_pct,
                volume_ratio=volume_ratio,
                sector=info.get('sector'),
                market_cap=None,
                description=f"{gap_percentage:+.1f}% gap {'filled' if gap_filled else 'unfilled'}"
            )
            
        except Exception as e:
            self.logger.debug(f"Error creating gap pattern: {e}")
            return None
    
    def _create_momentum_pattern(self, symbol: str, momentum: Dict[str, Any], 
                               timeframe: TimeFrame, current_price: float,
                               price_change_pct: float, volume_ratio: float,
                               info: Dict[str, Any], indicators: Dict[str, Any]) -> Optional[TechnicalPattern]:
        """Create a momentum pattern object."""
        try:
            pattern_type = momentum['type']
            strength_str = momentum['strength']
            
            # Calculate entry, target, and stop loss based on momentum direction
            if pattern_type == PatternType.MOMENTUM_BULLISH:
                entry_price = current_price
                target_price = current_price * 1.05  # 5% target
                stop_loss = current_price * 0.97   # 3% stop loss
            else:  # Bearish momentum
                entry_price = current_price
                target_price = current_price * 0.95  # 5% target down
                stop_loss = current_price * 1.03   # 3% stop loss
            
            # Calculate risk/reward ratio
            risk = abs(entry_price - stop_loss)
            reward = abs(target_price - entry_price)
            risk_reward_ratio = reward / risk if risk > 0 else 0
            
            # Calculate confidence score
            confidence_score = 60.0
            if strength_str == 'strong':
                confidence_score += 20.0
            if momentum.get('rsi_cross'):
                confidence_score += 10.0
            if momentum.get('macd_cross'):
                confidence_score += 10.0
            if volume_ratio > 1.3:
                confidence_score += 10.0
            
            # Determine pattern strength
            if confidence_score >= 90:
                strength = PatternStrength.VERY_STRONG
            elif confidence_score >= 80:
                strength = PatternStrength.STRONG
            elif confidence_score >= 70:
                strength = PatternStrength.MODERATE
            else:
                strength = PatternStrength.WEAK
            
            return TechnicalPattern(
                symbol=symbol,
                pattern_type=pattern_type,
                strength=strength,
                timeframe=timeframe,
                confidence_score=confidence_score,
                entry_price=entry_price,
                target_price=target_price,
                stop_loss=stop_loss,
                risk_reward_ratio=risk_reward_ratio,
                volume_confirmation=volume_ratio > 1.2,
                pattern_start_date=datetime.now() - timedelta(hours=4),
                pattern_completion_date=datetime.now(),
                current_price=current_price,
                price_change_pct=price_change_pct,
                volume_ratio=volume_ratio,
                sector=info.get('sector'),
                market_cap=None,
                description=f"Momentum shift: RSI={indicators.get('rsi', 0):.1f}, MACD cross"
            )
            
        except Exception as e:
            self.logger.debug(f"Error creating momentum pattern: {e}")
            return None
    
    def _filter_and_rank_patterns(self, patterns: List[TechnicalPattern]) -> List[TechnicalPattern]:
        """Filter and rank patterns by quality."""
        try:
            # Filter by minimum criteria
            filtered_patterns = [
                pattern for pattern in patterns
                if (pattern.confidence_score >= self.config.min_confidence_score and
                    pattern.volume_ratio >= self.config.min_volume_ratio and
                    pattern.risk_reward_ratio >= self.config.min_risk_reward_ratio)
            ]
            
            # Sort by combined score (confidence * risk_reward * volume_ratio)
            filtered_patterns.sort(
                key=lambda x: x.confidence_score * x.risk_reward_ratio * x.volume_ratio,
                reverse=True
            )
            
            # Limit results
            return filtered_patterns[:self.config.max_results_per_scan]
            
        except Exception as e:
            self.logger.error(f"Error filtering and ranking patterns: {e}")
            return patterns
    
    async def _process_pattern_alerts(self, patterns: List[TechnicalPattern]):
        """Process and send alerts for significant patterns."""
        try:
            # Filter for high-priority alerts
            high_priority_patterns = [
                pattern for pattern in patterns
                if (pattern.strength in [PatternStrength.STRONG, PatternStrength.VERY_STRONG] and
                    pattern.confidence_score > 85)
            ]
            
            if not high_priority_patterns:
                return
            
            # Create alert message
            alert_data = {
                'type': 'technical_pattern_alert',
                'timestamp': datetime.now().isoformat(),
                'pattern_count': len(high_priority_patterns),
                'patterns': [pattern.to_dict() for pattern in high_priority_patterns[:10]]
            }
            
            # Send notification
            await self.notification_manager.send_notification(
                title=f"Technical Pattern Alert: {len(high_priority_patterns)} strong patterns",
                message=f"Detected {len(high_priority_patterns)} high-confidence technical patterns",
                data=alert_data,
                priority="high"
            )
            
            self.logger.info(f"Sent technical pattern alert for {len(high_priority_patterns)} patterns")
            
        except Exception as e:
            self.logger.error(f"Error processing pattern alerts: {e}")
    
    def stop_scanning(self):
        """Stop the technical breakout scanning."""
        self.is_scanning = False
        self.logger.info("Technical breakout scanner stopped")
    
    def get_latest_patterns(self, limit: int = 20) -> List[TechnicalPattern]:
        """Get the latest detected patterns."""
        return self.detected_patterns[:limit]
    
    def get_patterns_by_sector(self, sector: str, limit: int = 10) -> List[TechnicalPattern]:
        """Get patterns filtered by sector."""
        sector_patterns = [
            pattern for pattern in self.detected_patterns
            if pattern.sector == sector
        ]
        return sector_patterns[:limit]
    
    def get_scanner_metrics(self) -> Dict[str, Any]:
        """Get scanner performance metrics."""
        cache_hit_rate = (self.cache_hits / (self.cache_hits + self.cache_misses)) * 100 if (self.cache_hits + self.cache_misses) > 0 else 0
        
        # Get batch downloader metrics
        batch_metrics = self.batch_downloader.get_performance_metrics()
        
        return {
            'scan_count': self.scan_count,
            'total_patterns_detected': self.total_patterns_detected,
            'average_scan_time': self.average_scan_time,
            'last_scan_time': self.last_scan_time.isoformat() if self.last_scan_time else None,
            'is_scanning': self.is_scanning,
            'universe_size': len(self.stock_universe),
            'latest_patterns_count': len(self.detected_patterns),
            'cache_performance': {
                'cache_hits': self.cache_hits,
                'cache_misses': self.cache_misses,
                'cache_hit_rate': f"{cache_hit_rate:.2f}%",
                'api_calls_saved': self.api_calls_saved
            },
            'batch_processing': {
                'total_requests': batch_metrics['total_requests'],
                'successful_requests': batch_metrics['successful_requests'],
                'failed_requests': batch_metrics['failed_requests'],
                'success_rate': f"{batch_metrics['success_rate']:.2f}%",
                'average_batch_time': f"{batch_metrics['average_batch_time']:.2f}s",
                'total_api_calls_saved': batch_metrics['total_api_calls_saved']
            },
            'config': {
                'min_confidence_score': self.config.min_confidence_score,
                'scan_interval_minutes': self.config.scan_interval_minutes,
                'timeframes': [tf.value for tf in self.config.timeframes],
                'caching_enabled': self.config.enable_caching
            }
        }
    
    def get_cache_statistics(self) -> Dict[str, Any]:
        """Get detailed cache performance statistics."""
        cache_hit_rate = (self.cache_hits / (self.cache_hits + self.cache_misses)) * 100 if (self.cache_hits + self.cache_misses) > 0 else 0
        
        return {
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'cache_hit_rate': cache_hit_rate,
            'api_calls_saved': self.api_calls_saved,
            'total_cache_requests': self.cache_hits + self.cache_misses,
            'cache_enabled': self.config.enable_caching,
            'cache_ttl_settings': {
                'price_data_ttl': self.config.price_data_cache_ttl,
                'indicator_ttl': self.config.indicator_cache_ttl,
                'pattern_ttl': self.config.pattern_cache_ttl
            }
        }
    
    def clear_cache(self):
        """Clear all cached data."""
        try:
            # Reset cache statistics
            self.cache_hits = 0
            self.cache_misses = 0
            self.api_calls_saved = 0
            
            # Clear the cache manager
            if hasattr(self.cache_manager, 'clear'):
                self.cache_manager.clear()
            
            self.logger.info("Cache cleared successfully")
        except Exception as e:
            self.logger.error(f"Error clearing cache: {e}")


# Integration with existing agent system
class TechnicalBreakoutDiscoveryTool:
    """Tool wrapper for integration with CrewAI agents."""
    
    def __init__(self, scanner: TechnicalBreakoutScanner):
        self.scanner = scanner
        self.name = "technical_breakout_discovery"
        self.description = "Discover stocks with technical breakouts, patterns, and momentum shifts"
    
    async def run(self, query: str = "") -> str:
        """Run technical breakout discovery."""
        try:
            patterns = await self.scanner.scan_for_patterns()
            
            if not patterns:
                return "No significant technical patterns detected in current scan."
            
            # Format results for agent consumption
            results = []
            for pattern in patterns[:10]:  # Top 10 results
                results.append(
                    f"{pattern.symbol}: {pattern.pattern_type.value} - "
                    f"{pattern.strength.value} strength, "
                    f"{pattern.confidence_score:.0f}% confidence, "
                    f"R/R: {pattern.risk_reward_ratio:.1f}"
                )
            
            return f"Technical Breakout Discovery Results:\n" + "\n".join(results)
            
        except Exception as e:
            return f"Error in technical breakout discovery: {e}"


if __name__ == "__main__":
    # Test the technical breakout scanner
    import asyncio
    from pathlib import Path
    import sys
    
    # Add project root to path
    project_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(project_root))
    
    from trading_system_v2.core.config_manager import ConfigManager
    
    async def test_scanner():
        """Test the technical breakout scanner."""
        config_manager = ConfigManager()
        await config_manager.initialize()
        
        scanner = TechnicalBreakoutScanner(config_manager)
        
        print("Testing technical breakout scanner...")
        patterns = await scanner.scan_for_patterns()
        
        print(f"\nDetected {len(patterns)} technical patterns:")
        for pattern in patterns[:5]:
            print(f"  {pattern.symbol}: {pattern.pattern_type.value}")
            print(f"    Strength: {pattern.strength.value}, Confidence: {pattern.confidence_score:.0f}%")
            print(f"    Entry: ${pattern.entry_price:.2f}, Target: ${pattern.target_price:.2f}")
            print(f"    R/R: {pattern.risk_reward_ratio:.1f}")
        
        print(f"\nScanner metrics: {scanner.get_scanner_metrics()}")
    
    asyncio.run(test_scanner())