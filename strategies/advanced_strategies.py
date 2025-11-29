#!/usr/bin/env python3
"""
Advanced Trading Strategies for Multi-Asset Trading System

This module contains sophisticated trading strategies that leverage:
- Real market data from multiple sources
- Technical analysis indicators
- Risk management integration
- Multi-asset class support
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

import pandas as pd
import numpy as np

from core.config_manager import ConfigManager
from core.data_manager import UnifiedDataManager
from core.risk_manager_24_7 import RiskManager24_7, AssetClass
from tools.technical_analysis_tool import TechnicalAnalysisTool
from tools.market_data_tool import MarketDataTool


class SignalStrength(Enum):
    """Signal strength enumeration."""
    WEAK = "weak"
    MODERATE = "moderate"
    STRONG = "strong"
    VERY_STRONG = "very_strong"


@dataclass
class TradingSignal:
    """Trading signal data structure."""
    symbol: str
    action: str  # 'buy', 'sell', 'hold'
    strength: SignalStrength
    confidence: float
    price_target: Optional[float] = None
    stop_loss: Optional[float] = None
    position_size: Optional[float] = None
    reasoning: str = ""
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class MomentumStrategy:
    """Momentum-based trading strategy using real market data."""
    
    def __init__(self, config_manager: ConfigManager):
        self.config_manager = config_manager
        self.data_manager = UnifiedDataManager(config_manager)
        self.technical_analysis = TechnicalAnalysisTool(config_manager)
        self.market_data = MarketDataTool(config_manager)
        self.logger = logging.getLogger(__name__)
        
        # Strategy parameters from config
        self.lookback_period = 20
        self.momentum_threshold = 0.02  # 2%
        self.volume_threshold = 1.5  # 1.5x average volume
        
    async def generate_signals(self, symbols: List[str]) -> List[TradingSignal]:
        """Generate momentum-based trading signals."""
        signals = []
        
        for symbol in symbols:
            try:
                # Get real market data
                price_data = await self.data_manager.get_price_data(
                    symbol, 
                    timeframe='1d',
                    limit=self.lookback_period + 10
                )
                
                if price_data is None or len(price_data) < self.lookback_period:
                    continue
                
                # Calculate momentum indicators
                momentum_score = self._calculate_momentum(price_data)
                volume_ratio = self._calculate_volume_ratio(price_data)
                
                # Generate signal based on momentum and volume
                signal = self._evaluate_momentum_signal(
                    symbol, price_data, momentum_score, volume_ratio
                )
                
                if signal:
                    signals.append(signal)
                    
            except Exception as e:
                self.logger.error(f"Error generating momentum signal for {symbol}: {e}")
                
        return signals
    
    def _calculate_momentum(self, price_data: pd.DataFrame) -> float:
        """Calculate momentum score using real price data."""
        if len(price_data) < self.lookback_period:
            return 0.0
            
        current_price = price_data['close'].iloc[-1]
        past_price = price_data['close'].iloc[-self.lookback_period]
        
        return (current_price - past_price) / past_price
    
    def _calculate_volume_ratio(self, price_data: pd.DataFrame) -> float:
        """Calculate volume ratio using real volume data."""
        if 'volume' not in price_data.columns or len(price_data) < 10:
            return 1.0
            
        recent_volume = price_data['volume'].iloc[-5:].mean()
        avg_volume = price_data['volume'].iloc[-20:-5].mean()
        
        if avg_volume == 0:
            return 1.0
            
        return recent_volume / avg_volume
    
    def _evaluate_momentum_signal(self, symbol: str, price_data: pd.DataFrame, 
                                momentum_score: float, volume_ratio: float) -> Optional[TradingSignal]:
        """Evaluate and create trading signal based on momentum analysis."""
        current_price = price_data['close'].iloc[-1]
        
        # Strong upward momentum with volume confirmation
        if momentum_score > self.momentum_threshold and volume_ratio > self.volume_threshold:
            return TradingSignal(
                symbol=symbol,
                action='buy',
                strength=SignalStrength.STRONG,
                confidence=min(0.95, 0.5 + abs(momentum_score) * 10),
                price_target=current_price * (1 + momentum_score * 0.5),
                stop_loss=current_price * 0.95,
                reasoning=f"Strong momentum ({momentum_score:.3f}) with volume confirmation ({volume_ratio:.2f}x)"
            )
        
        # Strong downward momentum with volume confirmation
        elif momentum_score < -self.momentum_threshold and volume_ratio > self.volume_threshold:
            return TradingSignal(
                symbol=symbol,
                action='sell',
                strength=SignalStrength.STRONG,
                confidence=min(0.95, 0.5 + abs(momentum_score) * 10),
                price_target=current_price * (1 + momentum_score * 0.5),
                stop_loss=current_price * 1.05,
                reasoning=f"Strong negative momentum ({momentum_score:.3f}) with volume confirmation ({volume_ratio:.2f}x)"
            )
        
        return None


class MeanReversionStrategy:
    """Mean reversion strategy using real market data and statistical analysis."""
    
    def __init__(self, config_manager: ConfigManager):
        self.config_manager = config_manager
        self.data_manager = UnifiedDataManager(config_manager)
        self.technical_analysis = TechnicalAnalysisTool(config_manager)
        self.logger = logging.getLogger(__name__)
        
        # Strategy parameters
        self.lookback_period = 50
        self.std_threshold = 2.0  # Standard deviations from mean
        self.min_reversion_probability = 0.6
        
    async def generate_signals(self, symbols: List[str]) -> List[TradingSignal]:
        """Generate mean reversion trading signals."""
        signals = []
        
        for symbol in symbols:
            try:
                # Get real market data
                price_data = await self.data_manager.get_price_data(
                    symbol, 
                    timeframe='1d',
                    limit=self.lookback_period + 10
                )
                
                if price_data is None or len(price_data) < self.lookback_period:
                    continue
                
                # Calculate mean reversion indicators
                z_score = self._calculate_z_score(price_data)
                reversion_probability = self._calculate_reversion_probability(price_data)
                
                # Generate signal based on mean reversion analysis
                signal = self._evaluate_reversion_signal(
                    symbol, price_data, z_score, reversion_probability
                )
                
                if signal:
                    signals.append(signal)
                    
            except Exception as e:
                self.logger.error(f"Error generating mean reversion signal for {symbol}: {e}")
                
        return signals
    
    def _calculate_z_score(self, price_data: pd.DataFrame) -> float:
        """Calculate Z-score for current price relative to historical mean."""
        prices = price_data['close'].iloc[-self.lookback_period:]
        current_price = prices.iloc[-1]
        
        mean_price = prices.mean()
        std_price = prices.std()
        
        if std_price == 0:
            return 0.0
            
        return (current_price - mean_price) / std_price
    
    def _calculate_reversion_probability(self, price_data: pd.DataFrame) -> float:
        """Calculate probability of mean reversion based on historical patterns."""
        prices = price_data['close'].iloc[-self.lookback_period:]
        
        # Count instances where extreme moves reverted
        reversion_count = 0
        total_extreme_moves = 0
        
        for i in range(5, len(prices) - 5):
            current_z = (prices.iloc[i] - prices.iloc[:i].mean()) / prices.iloc[:i].std()
            
            if abs(current_z) > self.std_threshold:
                total_extreme_moves += 1
                
                # Check if price reverted in next 5 days
                future_prices = prices.iloc[i+1:i+6]
                if len(future_prices) == 5:
                    if current_z > 0 and future_prices.iloc[-1] < prices.iloc[i]:
                        reversion_count += 1
                    elif current_z < 0 and future_prices.iloc[-1] > prices.iloc[i]:
                        reversion_count += 1
        
        if total_extreme_moves == 0:
            return 0.5
            
        return reversion_count / total_extreme_moves
    
    def _evaluate_reversion_signal(self, symbol: str, price_data: pd.DataFrame, 
                                 z_score: float, reversion_probability: float) -> Optional[TradingSignal]:
        """Evaluate and create trading signal based on mean reversion analysis."""
        current_price = price_data['close'].iloc[-1]
        mean_price = price_data['close'].iloc[-self.lookback_period:].mean()
        
        # Oversold condition with high reversion probability
        if z_score < -self.std_threshold and reversion_probability > self.min_reversion_probability:
            return TradingSignal(
                symbol=symbol,
                action='buy',
                strength=SignalStrength.MODERATE,
                confidence=reversion_probability,
                price_target=mean_price,
                stop_loss=current_price * 0.95,
                reasoning=f"Oversold (Z-score: {z_score:.2f}) with {reversion_probability:.1%} reversion probability"
            )
        
        # Overbought condition with high reversion probability
        elif z_score > self.std_threshold and reversion_probability > self.min_reversion_probability:
            return TradingSignal(
                symbol=symbol,
                action='sell',
                strength=SignalStrength.MODERATE,
                confidence=reversion_probability,
                price_target=mean_price,
                stop_loss=current_price * 1.05,
                reasoning=f"Overbought (Z-score: {z_score:.2f}) with {reversion_probability:.1%} reversion probability"
            )
        
        return None


class AdvancedStrategyManager:
    """Manager for coordinating multiple advanced trading strategies."""
    
    def __init__(self, config_manager: ConfigManager, risk_manager: RiskManager24_7):
        self.config_manager = config_manager
        self.risk_manager = risk_manager
        self.logger = logging.getLogger(__name__)
        
        # Initialize strategies
        self.momentum_strategy = MomentumStrategy(config_manager)
        self.mean_reversion_strategy = MeanReversionStrategy(config_manager)
        
        # Strategy weights
        self.strategy_weights = {
            'momentum': 0.6,
            'mean_reversion': 0.4
        }
    
    async def generate_combined_signals(self, symbols: List[str]) -> List[TradingSignal]:
        """Generate combined signals from multiple strategies."""
        all_signals = []
        
        try:
            # Get signals from each strategy
            momentum_signals = await self.momentum_strategy.generate_signals(symbols)
            reversion_signals = await self.mean_reversion_strategy.generate_signals(symbols)
            
            # Combine and weight signals
            combined_signals = self._combine_signals(momentum_signals, reversion_signals)
            
            # Apply risk management filters
            filtered_signals = await self._apply_risk_filters(combined_signals)
            
            all_signals.extend(filtered_signals)
            
        except Exception as e:
            self.logger.error(f"Error generating combined signals: {e}")
        
        return all_signals
    
    def _combine_signals(self, momentum_signals: List[TradingSignal], 
                        reversion_signals: List[TradingSignal]) -> List[TradingSignal]:
        """Combine signals from different strategies."""
        combined = {}
        
        # Process momentum signals
        for signal in momentum_signals:
            combined[signal.symbol] = {
                'momentum': signal,
                'reversion': None
            }
        
        # Process reversion signals
        for signal in reversion_signals:
            if signal.symbol in combined:
                combined[signal.symbol]['reversion'] = signal
            else:
                combined[signal.symbol] = {
                    'momentum': None,
                    'reversion': signal
                }
        
        # Create combined signals
        final_signals = []
        for symbol, signals in combined.items():
            combined_signal = self._create_combined_signal(symbol, signals)
            if combined_signal:
                final_signals.append(combined_signal)
        
        return final_signals
    
    def _create_combined_signal(self, symbol: str, signals: Dict) -> Optional[TradingSignal]:
        """Create a combined signal from multiple strategy signals."""
        momentum_signal = signals['momentum']
        reversion_signal = signals['reversion']
        
        # If both strategies agree on direction
        if (momentum_signal and reversion_signal and 
            momentum_signal.action == reversion_signal.action):
            
            # Combine confidence scores
            combined_confidence = (
                momentum_signal.confidence * self.strategy_weights['momentum'] +
                reversion_signal.confidence * self.strategy_weights['mean_reversion']
            )
            
            return TradingSignal(
                symbol=symbol,
                action=momentum_signal.action,
                strength=SignalStrength.STRONG,
                confidence=combined_confidence,
                price_target=momentum_signal.price_target,
                stop_loss=momentum_signal.stop_loss,
                reasoning=f"Combined: {momentum_signal.reasoning} + {reversion_signal.reasoning}"
            )
        
        # If only one strategy has a signal, use it with reduced confidence
        elif momentum_signal and not reversion_signal:
            momentum_signal.confidence *= self.strategy_weights['momentum']
            return momentum_signal
        elif reversion_signal and not momentum_signal:
            reversion_signal.confidence *= self.strategy_weights['mean_reversion']
            return reversion_signal
        
        # If strategies disagree, no signal
        return None
    
    async def _apply_risk_filters(self, signals: List[TradingSignal]) -> List[TradingSignal]:
        """Apply risk management filters to trading signals."""
        filtered_signals = []
        
        for signal in signals:
            try:
                # Check if trading is halted
                should_halt, reason = await self.risk_manager.should_halt_trading()
                if should_halt:
                    self.logger.warning(f"Trading halted: {reason}")
                    continue
                
                # Get position size limit
                asset_class = AssetClass.EQUITY  # Default, should be determined from symbol
                max_position_size = await self.risk_manager.get_position_size_limit(
                    signal.symbol, asset_class
                )
                
                # Apply position size limit
                if signal.position_size and signal.position_size > max_position_size:
                    signal.position_size = max_position_size
                
                # Check risk score
                risk_score = await self.risk_manager.get_position_risk_score(signal.symbol)
                if risk_score > 0.8:  # High risk threshold
                    signal.confidence *= 0.5  # Reduce confidence for high-risk positions
                
                filtered_signals.append(signal)
                
            except Exception as e:
                self.logger.error(f"Error applying risk filters to {signal.symbol}: {e}")
        
        return filtered_signals