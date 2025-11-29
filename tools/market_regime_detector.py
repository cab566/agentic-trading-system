#!/usr/bin/env python3
"""
Market Regime Detection System

This tool identifies different market regimes (bull, bear, sideways, high volatility, etc.)
to help trading agents adapt their strategies to current market conditions.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import pickle
from pathlib import Path

# Machine Learning imports
try:
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.mixture import GaussianMixture
    HAS_ML_LIBS = True
except ImportError:
    HAS_ML_LIBS = False

# Technical analysis
try:
    import talib
    HAS_TALIB = True
except ImportError:
    HAS_TALIB = False

# CrewAI imports
from crewai_tools import BaseTool
from pydantic import BaseModel, Field

# Local imports
from ..core.data_manager import UnifiedDataManager
from ..core.config_manager import ConfigManager


class MarketRegime(Enum):
    """Market regime types"""
    BULL_MARKET = "bull_market"
    BEAR_MARKET = "bear_market"
    SIDEWAYS_MARKET = "sideways_market"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGE_BOUND = "range_bound"
    CRISIS = "crisis"
    RECOVERY = "recovery"
    EXPANSION = "expansion"
    CONTRACTION = "contraction"


class RegimeConfidence(Enum):
    """Confidence levels for regime detection"""
    VERY_HIGH = "very_high"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    VERY_LOW = "very_low"


@dataclass
class RegimeFeatures:
    """Market regime features"""
    trend_strength: float
    volatility_level: float
    momentum: float
    volume_trend: float
    correlation_breakdown: float
    vix_level: float
    yield_curve_slope: float
    sector_rotation: float
    breadth_indicators: Dict[str, float]
    technical_indicators: Dict[str, float]
    macro_indicators: Dict[str, float]


@dataclass
class RegimeDetection:
    """Market regime detection result"""
    regime: MarketRegime
    confidence: RegimeConfidence
    probability: float
    timestamp: datetime
    duration_days: int
    features: RegimeFeatures
    supporting_evidence: List[str]
    risk_factors: List[str]
    strategy_recommendations: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RegimeTransition:
    """Market regime transition"""
    from_regime: MarketRegime
    to_regime: MarketRegime
    transition_date: datetime
    transition_probability: float
    leading_indicators: List[str]
    confirmation_signals: List[str]


class RegimeDetectorConfig(BaseModel):
    """Configuration for market regime detection"""
    lookback_period_days: int = Field(default=252, description="Lookback period for analysis")
    volatility_window: int = Field(default=20, description="Volatility calculation window")
    trend_window: int = Field(default=50, description="Trend calculation window")
    regime_min_duration: int = Field(default=10, description="Minimum regime duration in days")
    confidence_threshold: float = Field(default=0.7, description="Minimum confidence for regime detection")
    update_frequency_hours: int = Field(default=24, description="Update frequency in hours")
    benchmark_symbols: List[str] = Field(default=['SPY', 'QQQ', 'IWM'], description="Benchmark symbols")
    sector_etfs: List[str] = Field(default=['XLF', 'XLK', 'XLE', 'XLV', 'XLI'], description="Sector ETFs")
    volatility_symbols: List[str] = Field(default=['VIX', 'VXX'], description="Volatility symbols")


class MarketRegimeDetectorTool(BaseTool):
    """Market regime detection and analysis system"""
    
    name: str = "market_regime_detector"
    description: str = "Detects and analyzes market regimes to help adapt trading strategies"
    
    def __init__(self, config_manager: ConfigManager, data_manager: UnifiedDataManager):
        super().__init__()
        self.config_manager = config_manager
        self.data_manager = data_manager
        self.logger = logging.getLogger(__name__)
        
        # Load configuration
        self.config = self._load_config()
        
        # Regime detection models
        self.regime_models = {}
        self.scaler = StandardScaler()
        
        # Current regime state
        self.current_regime: Optional[RegimeDetection] = None
        self.regime_history: List[RegimeDetection] = []
        self.transition_history: List[RegimeTransition] = []
        
        # Cache
        self.cache_path = Path("regime_cache")
        self.cache_path.mkdir(exist_ok=True)
        
        # Feature calculators
        self.feature_calculators = self._initialize_feature_calculators()
        
        # Regime definitions
        self.regime_definitions = self._initialize_regime_definitions()
        
        # Load cached data
        self._load_cached_regimes()
    
    def _load_config(self) -> RegimeDetectorConfig:
        """Load regime detector configuration"""
        try:
            config_dict = self.config_manager.get_config('regime_detector')
            return RegimeDetectorConfig(**config_dict)
        except Exception as e:
            self.logger.warning(f"Could not load regime detector config: {e}")
            return RegimeDetectorConfig()
    
    def _initialize_feature_calculators(self) -> Dict[str, callable]:
        """Initialize feature calculation functions"""
        calculators = {}
        
        # Trend features
        calculators['trend_strength'] = self._calculate_trend_strength
        calculators['momentum'] = self._calculate_momentum
        calculators['price_position'] = self._calculate_price_position
        
        # Volatility features
        calculators['realized_volatility'] = self._calculate_realized_volatility
        calculators['volatility_regime'] = self._calculate_volatility_regime
        calculators['volatility_clustering'] = self._calculate_volatility_clustering
        
        # Volume features
        calculators['volume_trend'] = self._calculate_volume_trend
        calculators['volume_price_trend'] = self._calculate_volume_price_trend
        
        # Breadth features
        calculators['advance_decline'] = self._calculate_advance_decline
        calculators['new_highs_lows'] = self._calculate_new_highs_lows
        calculators['sector_rotation'] = self._calculate_sector_rotation
        
        # Correlation features
        calculators['correlation_breakdown'] = self._calculate_correlation_breakdown
        calculators['dispersion'] = self._calculate_dispersion
        
        return calculators
    
    def _initialize_regime_definitions(self) -> Dict[MarketRegime, Dict[str, Any]]:
        """Initialize regime definitions and characteristics"""
        definitions = {}
        
        definitions[MarketRegime.BULL_MARKET] = {
            'trend_strength': (0.6, 1.0),
            'volatility_level': (0.0, 0.4),
            'momentum': (0.5, 1.0),
            'volume_trend': (0.3, 1.0),
            'duration_typical': 300,
            'strategies': ['momentum', 'growth', 'breakout'],
            'risk_factors': ['overvaluation', 'complacency']
        }
        
        definitions[MarketRegime.BEAR_MARKET] = {
            'trend_strength': (-1.0, -0.6),
            'volatility_level': (0.4, 1.0),
            'momentum': (-1.0, -0.5),
            'volume_trend': (0.3, 1.0),
            'duration_typical': 200,
            'strategies': ['defensive', 'short_selling', 'quality'],
            'risk_factors': ['capitulation', 'liquidity_crisis']
        }
        
        definitions[MarketRegime.SIDEWAYS_MARKET] = {
            'trend_strength': (-0.3, 0.3),
            'volatility_level': (0.2, 0.6),
            'momentum': (-0.3, 0.3),
            'volume_trend': (-0.2, 0.5),
            'duration_typical': 150,
            'strategies': ['mean_reversion', 'range_trading', 'pairs_trading'],
            'risk_factors': ['false_breakouts', 'whipsaws']
        }
        
        definitions[MarketRegime.HIGH_VOLATILITY] = {
            'volatility_level': (0.7, 1.0),
            'correlation_breakdown': (0.6, 1.0),
            'duration_typical': 30,
            'strategies': ['volatility_trading', 'options_strategies'],
            'risk_factors': ['extreme_moves', 'gap_risk']
        }
        
        definitions[MarketRegime.LOW_VOLATILITY] = {
            'volatility_level': (0.0, 0.3),
            'correlation_breakdown': (0.0, 0.3),
            'duration_typical': 100,
            'strategies': ['carry_trades', 'low_vol_strategies'],
            'risk_factors': ['complacency', 'sudden_shocks']
        }
        
        definitions[MarketRegime.CRISIS] = {
            'volatility_level': (0.8, 1.0),
            'correlation_breakdown': (0.8, 1.0),
            'trend_strength': (-1.0, -0.8),
            'duration_typical': 60,
            'strategies': ['cash', 'defensive', 'contrarian'],
            'risk_factors': ['liquidity_crisis', 'systemic_risk']
        }
        
        return definitions
    
    def _run(self, action: str, parameters: Dict[str, Any] = None) -> str:
        """Synchronous regime detection execution"""
        return asyncio.run(self._arun(action, parameters))
    
    async def _arun(self, action: str, parameters: Dict[str, Any] = None) -> str:
        """Asynchronous regime detection execution"""
        try:
            parameters = parameters or {}
            
            if action == 'detect_current_regime':
                return await self._detect_current_regime(parameters)
            elif action == 'analyze_regime_history':
                return await self._analyze_regime_history(parameters)
            elif action == 'predict_regime_transition':
                return await self._predict_regime_transition(parameters)
            elif action == 'get_regime_features':
                return await self._get_regime_features(parameters)
            elif action == 'get_strategy_recommendations':
                return await self._get_strategy_recommendations(parameters)
            elif action == 'monitor_regime_changes':
                return await self._monitor_regime_changes(parameters)
            elif action == 'calibrate_models':
                return await self._calibrate_models(parameters)
            elif action == 'get_regime_dashboard':
                return await self._get_regime_dashboard(parameters)
            else:
                return json.dumps({
                    'error': f'Unknown action: {action}',
                    'available_actions': [
                        'detect_current_regime', 'analyze_regime_history', 'predict_regime_transition',
                        'get_regime_features', 'get_strategy_recommendations', 'monitor_regime_changes',
                        'calibrate_models', 'get_regime_dashboard'
                    ]
                })
                
        except Exception as e:
            self.logger.error(f"Regime detection failed: {e}")
            return json.dumps({'error': str(e)})
    
    async def _detect_current_regime(self, parameters: Dict[str, Any]) -> str:
        """Detect current market regime"""
        symbols = parameters.get('symbols', self.config.benchmark_symbols)
        force_update = parameters.get('force_update', False)
        
        # Check if we need to update
        if (not force_update and self.current_regime and 
            (datetime.now() - self.current_regime.timestamp).total_seconds() < 
            self.config.update_frequency_hours * 3600):
            
            return json.dumps({
                'current_regime': {
                    'regime': self.current_regime.regime.value,
                    'confidence': self.current_regime.confidence.value,
                    'probability': self.current_regime.probability,
                    'timestamp': self.current_regime.timestamp.isoformat(),
                    'duration_days': self.current_regime.duration_days,
                    'strategy_recommendations': self.current_regime.strategy_recommendations,
                    'risk_factors': self.current_regime.risk_factors
                },
                'cached': True
            }, indent=2)
        
        # Get market data
        market_data = await self._get_market_data(symbols)
        if not market_data:
            return json.dumps({'error': 'Failed to get market data'})
        
        # Calculate features
        features = await self._calculate_regime_features(market_data)
        if not features:
            return json.dumps({'error': 'Failed to calculate regime features'})
        
        # Detect regime
        regime_detection = await self._classify_regime(features)
        
        # Update current regime
        if regime_detection:
            # Check for regime transition
            if (self.current_regime and 
                self.current_regime.regime != regime_detection.regime):
                
                transition = RegimeTransition(
                    from_regime=self.current_regime.regime,
                    to_regime=regime_detection.regime,
                    transition_date=datetime.now(),
                    transition_probability=regime_detection.probability,
                    leading_indicators=self._identify_leading_indicators(features),
                    confirmation_signals=self._identify_confirmation_signals(features)
                )
                
                self.transition_history.append(transition)
                self.logger.info(f"Regime transition detected: {transition.from_regime.value} -> {transition.to_regime.value}")
            
            self.current_regime = regime_detection
            self.regime_history.append(regime_detection)
            
            # Save to cache
            await self._save_regime_cache()
        
        return json.dumps({
            'current_regime': {
                'regime': regime_detection.regime.value,
                'confidence': regime_detection.confidence.value,
                'probability': regime_detection.probability,
                'timestamp': regime_detection.timestamp.isoformat(),
                'duration_days': regime_detection.duration_days,
                'supporting_evidence': regime_detection.supporting_evidence,
                'risk_factors': regime_detection.risk_factors,
                'strategy_recommendations': regime_detection.strategy_recommendations,
                'features': {
                    'trend_strength': regime_detection.features.trend_strength,
                    'volatility_level': regime_detection.features.volatility_level,
                    'momentum': regime_detection.features.momentum,
                    'volume_trend': regime_detection.features.volume_trend
                }
            },
            'symbols_analyzed': symbols,
            'detection_timestamp': datetime.now().isoformat()
        }, indent=2)
    
    async def _analyze_regime_history(self, parameters: Dict[str, Any]) -> str:
        """Analyze historical regime patterns"""
        lookback_days = parameters.get('lookback_days', 365)
        include_transitions = parameters.get('include_transitions', True)
        
        # Filter regime history
        cutoff_date = datetime.now() - timedelta(days=lookback_days)
        recent_regimes = [r for r in self.regime_history if r.timestamp >= cutoff_date]
        
        if not recent_regimes:
            return json.dumps({'error': 'No regime history available'})
        
        # Analyze regime distribution
        regime_counts = {}
        total_duration = 0
        
        for regime in recent_regimes:
            regime_type = regime.regime.value
            if regime_type not in regime_counts:
                regime_counts[regime_type] = {'count': 0, 'total_duration': 0, 'avg_confidence': 0}
            
            regime_counts[regime_type]['count'] += 1
            regime_counts[regime_type]['total_duration'] += regime.duration_days
            regime_counts[regime_type]['avg_confidence'] += regime.probability
            total_duration += regime.duration_days
        
        # Calculate averages
        for regime_type in regime_counts:
            count = regime_counts[regime_type]['count']
            regime_counts[regime_type]['percentage'] = (regime_counts[regime_type]['total_duration'] / total_duration) * 100
            regime_counts[regime_type]['avg_duration'] = regime_counts[regime_type]['total_duration'] / count
            regime_counts[regime_type]['avg_confidence'] = regime_counts[regime_type]['avg_confidence'] / count
        
        # Analyze transitions
        transition_analysis = {}
        if include_transitions:
            recent_transitions = [t for t in self.transition_history if t.transition_date >= cutoff_date]
            
            for transition in recent_transitions:
                key = f"{transition.from_regime.value}_to_{transition.to_regime.value}"
                if key not in transition_analysis:
                    transition_analysis[key] = {'count': 0, 'avg_probability': 0}
                
                transition_analysis[key]['count'] += 1
                transition_analysis[key]['avg_probability'] += transition.transition_probability
            
            # Calculate averages
            for key in transition_analysis:
                count = transition_analysis[key]['count']
                transition_analysis[key]['avg_probability'] /= count
        
        return json.dumps({
            'analysis_period': {
                'start_date': cutoff_date.isoformat(),
                'end_date': datetime.now().isoformat(),
                'total_regimes': len(recent_regimes),
                'total_transitions': len([t for t in self.transition_history if t.transition_date >= cutoff_date])
            },
            'regime_distribution': regime_counts,
            'transition_analysis': transition_analysis,
            'current_regime': self.current_regime.regime.value if self.current_regime else None,
            'regime_stability': self._calculate_regime_stability(recent_regimes)
        }, indent=2)
    
    async def _get_strategy_recommendations(self, parameters: Dict[str, Any]) -> str:
        """Get strategy recommendations based on current regime"""
        if not self.current_regime:
            await self._detect_current_regime({})
        
        if not self.current_regime:
            return json.dumps({'error': 'No current regime detected'})
        
        regime = self.current_regime.regime
        confidence = self.current_regime.confidence
        
        # Get base recommendations from regime definition
        regime_def = self.regime_definitions.get(regime, {})
        base_strategies = regime_def.get('strategies', [])
        risk_factors = regime_def.get('risk_factors', [])
        
        # Adjust recommendations based on confidence
        if confidence in [RegimeConfidence.VERY_HIGH, RegimeConfidence.HIGH]:
            strategy_weight = 1.0
            risk_adjustment = "Standard risk management"
        elif confidence == RegimeConfidence.MEDIUM:
            strategy_weight = 0.7
            risk_adjustment = "Increased position sizing caution"
        else:
            strategy_weight = 0.5
            risk_adjustment = "Reduced position sizes, increased diversification"
        
        # Generate specific recommendations
        recommendations = {
            'primary_strategies': base_strategies,
            'strategy_weight': strategy_weight,
            'risk_adjustment': risk_adjustment,
            'position_sizing': self._get_position_sizing_recommendations(regime, confidence),
            'sector_allocation': self._get_sector_recommendations(regime),
            'hedging_strategies': self._get_hedging_recommendations(regime),
            'timing_considerations': self._get_timing_recommendations(regime),
            'risk_factors': risk_factors,
            'monitoring_indicators': self._get_monitoring_indicators(regime)
        }
        
        return json.dumps({
            'current_regime': regime.value,
            'confidence': confidence.value,
            'recommendations': recommendations,
            'regime_duration': self.current_regime.duration_days,
            'last_updated': self.current_regime.timestamp.isoformat()
        }, indent=2)
    
    # Feature calculation methods
    
    async def _get_market_data(self, symbols: List[str]) -> Optional[Dict[str, pd.DataFrame]]:
        """Get market data for regime analysis"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=self.config.lookback_period_days + 50)
        
        market_data = {}
        
        for symbol in symbols:
            try:
                data = await self.data_manager.get_price_data(symbol, '1d', start_date, end_date)
                if not data.data.empty:
                    market_data[symbol] = data.data
            except Exception as e:
                self.logger.warning(f"Failed to get data for {symbol}: {e}")
        
        # Get VIX data if available
        try:
            vix_data = await self.data_manager.get_price_data('^VIX', '1d', start_date, end_date)
            if not vix_data.data.empty:
                market_data['VIX'] = vix_data.data
        except:
            pass
        
        return market_data if market_data else None
    
    async def _calculate_regime_features(self, market_data: Dict[str, pd.DataFrame]) -> Optional[RegimeFeatures]:
        """Calculate features for regime detection"""
        try:
            # Use SPY as primary benchmark
            primary_data = market_data.get('SPY') or market_data.get(list(market_data.keys())[0])
            
            if primary_data is None or len(primary_data) < 50:
                return None
            
            # Calculate individual features
            features = {}
            
            for feature_name, calculator in self.feature_calculators.items():
                try:
                    features[feature_name] = calculator(primary_data, market_data)
                except Exception as e:
                    self.logger.warning(f"Failed to calculate {feature_name}: {e}")
                    features[feature_name] = 0.0
            
            # Calculate technical indicators
            technical_indicators = {}
            if HAS_TALIB and len(primary_data) >= 50:
                try:
                    technical_indicators['rsi'] = talib.RSI(primary_data['Close'].values)[-1]
                    technical_indicators['macd'] = talib.MACD(primary_data['Close'].values)[0][-1]
                    technical_indicators['bb_position'] = self._calculate_bb_position(primary_data)
                except:
                    pass
            
            # Calculate macro indicators (simplified)
            macro_indicators = {
                'yield_curve_slope': self._calculate_yield_curve_slope(),
                'dollar_strength': self._calculate_dollar_strength(market_data),
                'commodity_trend': self._calculate_commodity_trend(market_data)
            }
            
            # Calculate breadth indicators
            breadth_indicators = {
                'advance_decline': features.get('advance_decline', 0.0),
                'new_highs_lows': features.get('new_highs_lows', 0.0),
                'sector_rotation': features.get('sector_rotation', 0.0)
            }
            
            regime_features = RegimeFeatures(
                trend_strength=features.get('trend_strength', 0.0),
                volatility_level=features.get('realized_volatility', 0.0),
                momentum=features.get('momentum', 0.0),
                volume_trend=features.get('volume_trend', 0.0),
                correlation_breakdown=features.get('correlation_breakdown', 0.0),
                vix_level=self._get_vix_level(market_data),
                yield_curve_slope=macro_indicators['yield_curve_slope'],
                sector_rotation=features.get('sector_rotation', 0.0),
                breadth_indicators=breadth_indicators,
                technical_indicators=technical_indicators,
                macro_indicators=macro_indicators
            )
            
            return regime_features
            
        except Exception as e:
            self.logger.error(f"Failed to calculate regime features: {e}")
            return None
    
    def _calculate_trend_strength(self, data: pd.DataFrame, market_data: Dict[str, pd.DataFrame]) -> float:
        """Calculate trend strength"""
        try:
            # Use multiple timeframes
            short_ma = data['Close'].rolling(20).mean()
            long_ma = data['Close'].rolling(50).mean()
            
            # Trend direction
            trend_direction = (short_ma.iloc[-1] - long_ma.iloc[-1]) / long_ma.iloc[-1]
            
            # Trend consistency
            price_above_ma = (data['Close'] > short_ma).rolling(20).mean().iloc[-1]
            
            # Combine measures
            trend_strength = trend_direction * price_above_ma
            
            # Normalize to [-1, 1]
            return np.clip(trend_strength * 10, -1, 1)
            
        except:
            return 0.0
    
    def _calculate_momentum(self, data: pd.DataFrame, market_data: Dict[str, pd.DataFrame]) -> float:
        """Calculate momentum"""
        try:
            # Price momentum
            returns_1m = data['Close'].pct_change(20).iloc[-1]
            returns_3m = data['Close'].pct_change(60).iloc[-1]
            
            # Volume momentum
            volume_ma = data['Volume'].rolling(20).mean()
            volume_momentum = (data['Volume'].iloc[-1] / volume_ma.iloc[-1]) - 1
            
            # Combine
            momentum = (returns_1m + returns_3m) * (1 + volume_momentum * 0.1)
            
            # Normalize
            return np.clip(momentum * 5, -1, 1)
            
        except:
            return 0.0
    
    def _calculate_realized_volatility(self, data: pd.DataFrame, market_data: Dict[str, pd.DataFrame]) -> float:
        """Calculate realized volatility"""
        try:
            returns = data['Close'].pct_change().dropna()
            volatility = returns.rolling(self.config.volatility_window).std().iloc[-1] * np.sqrt(252)
            
            # Normalize to [0, 1] based on historical percentiles
            historical_vol = returns.rolling(252).std() * np.sqrt(252)
            percentile = (volatility > historical_vol).mean()
            
            return min(1.0, percentile)
            
        except:
            return 0.5
    
    def _calculate_volume_trend(self, data: pd.DataFrame, market_data: Dict[str, pd.DataFrame]) -> float:
        """Calculate volume trend"""
        try:
            volume_ma_short = data['Volume'].rolling(10).mean()
            volume_ma_long = data['Volume'].rolling(50).mean()
            
            volume_trend = (volume_ma_short.iloc[-1] / volume_ma_long.iloc[-1]) - 1
            
            return np.clip(volume_trend * 2, -1, 1)
            
        except:
            return 0.0
    
    def _calculate_correlation_breakdown(self, data: pd.DataFrame, market_data: Dict[str, pd.DataFrame]) -> float:
        """Calculate correlation breakdown indicator"""
        try:
            if len(market_data) < 3:
                return 0.0
            
            # Calculate correlations between major indices
            returns_data = {}
            for symbol, df in market_data.items():
                if symbol in ['SPY', 'QQQ', 'IWM'] and len(df) > 20:
                    returns_data[symbol] = df['Close'].pct_change().dropna()
            
            if len(returns_data) < 2:
                return 0.0
            
            # Create correlation matrix
            returns_df = pd.DataFrame(returns_data).dropna()
            
            if len(returns_df) < 20:
                return 0.0
            
            # Rolling correlation
            rolling_corr = returns_df.rolling(20).corr()
            
            # Average correlation (excluding diagonal)
            recent_corr = rolling_corr.iloc[-len(returns_data):, :].values
            avg_corr = np.nanmean(recent_corr[~np.eye(len(returns_data), dtype=bool)])
            
            # Historical correlation
            historical_corr = returns_df.corr().values
            historical_avg = np.nanmean(historical_corr[~np.eye(len(returns_data), dtype=bool)])
            
            # Breakdown indicator (higher when correlations break down)
            breakdown = max(0, historical_avg - avg_corr)
            
            return min(1.0, breakdown * 5)
            
        except:
            return 0.0
    
    def _calculate_advance_decline(self, data: pd.DataFrame, market_data: Dict[str, pd.DataFrame]) -> float:
        """Calculate advance/decline indicator (simplified)"""
        try:
            # Use available market data as proxy
            advancing = 0
            declining = 0
            
            for symbol, df in market_data.items():
                if len(df) > 1:
                    if df['Close'].iloc[-1] > df['Close'].iloc[-2]:
                        advancing += 1
                    else:
                        declining += 1
            
            total = advancing + declining
            if total == 0:
                return 0.0
            
            return (advancing - declining) / total
            
        except:
            return 0.0
    
    def _calculate_new_highs_lows(self, data: pd.DataFrame, market_data: Dict[str, pd.DataFrame]) -> float:
        """Calculate new highs/lows indicator"""
        try:
            # Check if current price is near 52-week high/low
            high_52w = data['High'].rolling(252).max().iloc[-1]
            low_52w = data['Low'].rolling(252).min().iloc[-1]
            current_price = data['Close'].iloc[-1]
            
            # Position relative to 52-week range
            position = (current_price - low_52w) / (high_52w - low_52w)
            
            # Convert to new highs/lows indicator
            if position > 0.95:
                return 1.0  # Near new highs
            elif position < 0.05:
                return -1.0  # Near new lows
            else:
                return (position - 0.5) * 2  # Normalized position
                
        except:
            return 0.0
    
    def _calculate_sector_rotation(self, data: pd.DataFrame, market_data: Dict[str, pd.DataFrame]) -> float:
        """Calculate sector rotation indicator"""
        try:
            # Simplified sector rotation using available ETFs
            sector_performance = {}
            
            for symbol in ['XLF', 'XLK', 'XLE', 'XLV', 'XLI']:  # Financial, Tech, Energy, Health, Industrial
                if symbol in market_data:
                    df = market_data[symbol]
                    if len(df) > 20:
                        returns = df['Close'].pct_change(20).iloc[-1]
                        sector_performance[symbol] = returns
            
            if len(sector_performance) < 2:
                return 0.0
            
            # Calculate dispersion of sector performance
            performances = list(sector_performance.values())
            dispersion = np.std(performances)
            
            # Higher dispersion indicates more rotation
            return min(1.0, dispersion * 10)
            
        except:
            return 0.0
    
    def _get_vix_level(self, market_data: Dict[str, pd.DataFrame]) -> float:
        """Get VIX level"""
        try:
            if 'VIX' in market_data:
                vix_data = market_data['VIX']
                current_vix = vix_data['Close'].iloc[-1]
                
                # Normalize VIX (typical range 10-80)
                normalized_vix = min(1.0, max(0.0, (current_vix - 10) / 70))
                return normalized_vix
            
            return 0.5  # Default middle value
            
        except:
            return 0.5
    
    # Simplified implementations for other indicators
    
    def _calculate_yield_curve_slope(self) -> float:
        """Calculate yield curve slope (simplified)"""
        # In a real implementation, you would fetch treasury yields
        return 0.0
    
    def _calculate_dollar_strength(self, market_data: Dict[str, pd.DataFrame]) -> float:
        """Calculate dollar strength (simplified)"""
        return 0.0
    
    def _calculate_commodity_trend(self, market_data: Dict[str, pd.DataFrame]) -> float:
        """Calculate commodity trend (simplified)"""
        return 0.0
    
    def _calculate_bb_position(self, data: pd.DataFrame) -> float:
        """Calculate Bollinger Band position"""
        try:
            if HAS_TALIB:
                upper, middle, lower = talib.BBANDS(data['Close'].values)
                current_price = data['Close'].iloc[-1]
                bb_position = (current_price - lower[-1]) / (upper[-1] - lower[-1])
                return bb_position
            return 0.5
        except:
            return 0.5
    
    async def _classify_regime(self, features: RegimeFeatures) -> Optional[RegimeDetection]:
        """Classify market regime based on features"""
        try:
            # Create feature vector
            feature_vector = np.array([
                features.trend_strength,
                features.volatility_level,
                features.momentum,
                features.volume_trend,
                features.correlation_breakdown,
                features.vix_level
            ])
            
            # Rule-based classification
            regime_scores = {}
            
            for regime, definition in self.regime_definitions.items():
                score = self._calculate_regime_score(features, definition)
                regime_scores[regime] = score
            
            # Find best matching regime
            best_regime = max(regime_scores, key=regime_scores.get)
            best_score = regime_scores[best_regime]
            
            # Determine confidence
            if best_score > 0.8:
                confidence = RegimeConfidence.VERY_HIGH
            elif best_score > 0.7:
                confidence = RegimeConfidence.HIGH
            elif best_score > 0.6:
                confidence = RegimeConfidence.MEDIUM
            elif best_score > 0.5:
                confidence = RegimeConfidence.LOW
            else:
                confidence = RegimeConfidence.VERY_LOW
            
            # Calculate duration (simplified)
            duration_days = self._estimate_regime_duration(best_regime)
            
            # Generate supporting evidence and recommendations
            supporting_evidence = self._generate_supporting_evidence(features, best_regime)
            risk_factors = self.regime_definitions[best_regime].get('risk_factors', [])
            strategy_recommendations = self.regime_definitions[best_regime].get('strategies', [])
            
            regime_detection = RegimeDetection(
                regime=best_regime,
                confidence=confidence,
                probability=best_score,
                timestamp=datetime.now(),
                duration_days=duration_days,
                features=features,
                supporting_evidence=supporting_evidence,
                risk_factors=risk_factors,
                strategy_recommendations=strategy_recommendations,
                metadata={'regime_scores': {r.value: s for r, s in regime_scores.items()}}
            )
            
            return regime_detection
            
        except Exception as e:
            self.logger.error(f"Failed to classify regime: {e}")
            return None
    
    def _calculate_regime_score(self, features: RegimeFeatures, definition: Dict[str, Any]) -> float:
        """Calculate how well features match a regime definition"""
        score = 0.0
        total_weight = 0.0
        
        # Check each feature constraint
        feature_checks = [
            ('trend_strength', features.trend_strength),
            ('volatility_level', features.volatility_level),
            ('momentum', features.momentum),
            ('volume_trend', features.volume_trend),
            ('correlation_breakdown', features.correlation_breakdown)
        ]
        
        for feature_name, feature_value in feature_checks:
            if feature_name in definition:
                min_val, max_val = definition[feature_name]
                
                if min_val <= feature_value <= max_val:
                    # Feature matches - calculate how well
                    range_size = max_val - min_val
                    if range_size > 0:
                        # Closer to center of range = higher score
                        center = (min_val + max_val) / 2
                        distance_from_center = abs(feature_value - center)
                        feature_score = 1.0 - (distance_from_center / (range_size / 2))
                    else:
                        feature_score = 1.0
                else:
                    # Feature doesn't match
                    feature_score = 0.0
                
                score += feature_score
                total_weight += 1.0
        
        return score / total_weight if total_weight > 0 else 0.0
    
    def _estimate_regime_duration(self, regime: MarketRegime) -> int:
        """Estimate regime duration based on historical patterns"""
        if regime in self.regime_definitions:
            return self.regime_definitions[regime].get('duration_typical', 100)
        return 100
    
    def _generate_supporting_evidence(self, features: RegimeFeatures, regime: MarketRegime) -> List[str]:
        """Generate supporting evidence for regime classification"""
        evidence = []
        
        if regime == MarketRegime.BULL_MARKET:
            if features.trend_strength > 0.5:
                evidence.append("Strong upward trend momentum")
            if features.volatility_level < 0.4:
                evidence.append("Low volatility environment")
            if features.volume_trend > 0.3:
                evidence.append("Increasing volume supporting moves")
        
        elif regime == MarketRegime.BEAR_MARKET:
            if features.trend_strength < -0.5:
                evidence.append("Strong downward trend")
            if features.volatility_level > 0.6:
                evidence.append("Elevated volatility levels")
            if features.correlation_breakdown > 0.5:
                evidence.append("Correlation breakdown indicating stress")
        
        elif regime == MarketRegime.HIGH_VOLATILITY:
            if features.volatility_level > 0.7:
                evidence.append("Realized volatility significantly elevated")
            if features.vix_level > 0.6:
                evidence.append("VIX indicating fear in markets")
        
        # Add more evidence generation logic for other regimes
        
        return evidence
    
    def _get_position_sizing_recommendations(self, regime: MarketRegime, confidence: RegimeConfidence) -> Dict[str, str]:
        """Get position sizing recommendations"""
        if regime in [MarketRegime.BULL_MARKET, MarketRegime.TRENDING_UP]:
            base_size = "Standard to increased position sizes"
        elif regime in [MarketRegime.BEAR_MARKET, MarketRegime.CRISIS]:
            base_size = "Reduced position sizes, defensive positioning"
        elif regime == MarketRegime.HIGH_VOLATILITY:
            base_size = "Significantly reduced position sizes"
        else:
            base_size = "Standard position sizes with increased diversification"
        
        confidence_adjustment = {
            RegimeConfidence.VERY_HIGH: "Full conviction sizing",
            RegimeConfidence.HIGH: "Standard sizing",
            RegimeConfidence.MEDIUM: "Reduced sizing",
            RegimeConfidence.LOW: "Minimal sizing",
            RegimeConfidence.VERY_LOW: "Defensive sizing only"
        }
        
        return {
            'base_recommendation': base_size,
            'confidence_adjustment': confidence_adjustment[confidence]
        }
    
    def _get_sector_recommendations(self, regime: MarketRegime) -> List[str]:
        """Get sector allocation recommendations"""
        if regime == MarketRegime.BULL_MARKET:
            return ['Technology', 'Growth sectors', 'Cyclicals']
        elif regime == MarketRegime.BEAR_MARKET:
            return ['Utilities', 'Consumer staples', 'Healthcare']
        elif regime == MarketRegime.HIGH_VOLATILITY:
            return ['Low volatility sectors', 'Defensive sectors']
        else:
            return ['Balanced allocation', 'Quality sectors']
    
    def _get_hedging_recommendations(self, regime: MarketRegime) -> List[str]:
        """Get hedging strategy recommendations"""
        if regime in [MarketRegime.BEAR_MARKET, MarketRegime.CRISIS]:
            return ['Put options', 'VIX calls', 'Inverse ETFs']
        elif regime == MarketRegime.HIGH_VOLATILITY:
            return ['Volatility hedges', 'Correlation hedges']
        else:
            return ['Minimal hedging', 'Tail risk hedges only']
    
    def _get_timing_recommendations(self, regime: MarketRegime) -> List[str]:
        """Get market timing recommendations"""
        if regime == MarketRegime.BULL_MARKET:
            return ['Buy dips', 'Momentum following']
        elif regime == MarketRegime.BEAR_MARKET:
            return ['Sell rallies', 'Wait for capitulation']
        elif regime == MarketRegime.SIDEWAYS_MARKET:
            return ['Range trading', 'Mean reversion']
        else:
            return ['Dollar cost averaging', 'Gradual positioning']
    
    def _get_monitoring_indicators(self, regime: MarketRegime) -> List[str]:
        """Get key indicators to monitor for regime changes"""
        base_indicators = ['VIX', 'Yield curve', 'Dollar strength', 'Sector rotation']
        
        if regime == MarketRegime.BULL_MARKET:
            return base_indicators + ['Valuation metrics', 'Sentiment indicators']
        elif regime == MarketRegime.BEAR_MARKET:
            return base_indicators + ['Capitulation indicators', 'Policy responses']
        else:
            return base_indicators
    
    def _calculate_regime_stability(self, regimes: List[RegimeDetection]) -> float:
        """Calculate regime stability score"""
        if len(regimes) < 2:
            return 1.0
        
        # Count regime changes
        changes = 0
        for i in range(1, len(regimes)):
            if regimes[i].regime != regimes[i-1].regime:
                changes += 1
        
        # Stability = 1 - (changes / possible_changes)
        stability = 1.0 - (changes / (len(regimes) - 1))
        return stability
    
    def _identify_leading_indicators(self, features: RegimeFeatures) -> List[str]:
        """Identify leading indicators for regime transition"""
        indicators = []
        
        if features.volatility_level > 0.7:
            indicators.append("Volatility spike")
        if features.correlation_breakdown > 0.6:
            indicators.append("Correlation breakdown")
        if abs(features.momentum) > 0.8:
            indicators.append("Extreme momentum")
        
        return indicators
    
    def _identify_confirmation_signals(self, features: RegimeFeatures) -> List[str]:
        """Identify confirmation signals for regime transition"""
        signals = []
        
        if features.volume_trend > 0.5:
            signals.append("Volume confirmation")
        if features.trend_strength > 0.6:
            signals.append("Trend strength confirmation")
        
        return signals
    
    async def _save_regime_cache(self):
        """Save regime data to cache"""
        try:
            cache_data = {
                'current_regime': self.current_regime,
                'regime_history': self.regime_history[-100:],  # Keep last 100
                'transition_history': self.transition_history[-50:]  # Keep last 50
            }
            
            cache_file = self.cache_path / "regime_cache.pkl"
            with open(cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
                
        except Exception as e:
            self.logger.error(f"Failed to save regime cache: {e}")
    
    def _load_cached_regimes(self):
        """Load cached regime data"""
        try:
            cache_file = self.cache_path / "regime_cache.pkl"
            if cache_file.exists():
                with open(cache_file, 'rb') as f:
                    cache_data = pickle.load(f)
                
                self.current_regime = cache_data.get('current_regime')
                self.regime_history = cache_data.get('regime_history', [])
                self.transition_history = cache_data.get('transition_history', [])
                
        except Exception as e:
            self.logger.error(f"Failed to load regime cache: {e}")


# Example usage
if __name__ == "__main__":
    from pathlib import Path
    from ..core.config_manager import ConfigManager
    from ..core.data_manager import UnifiedDataManager
    
    config_manager = ConfigManager(Path("../config"))
    data_manager = UnifiedDataManager(config_manager)
    
    regime_detector = MarketRegimeDetectorTool(config_manager, data_manager)
    
    # Test regime detection
    result = regime_detector._run('detect_current_regime')
    print("Current Regime:", result)