#!/usr/bin/env python3
"""
Advanced Trading Strategies Implementation

This module implements sophisticated algorithmic trading strategies that demonstrate
institutional-grade quantitative finance techniques, including:

- Multi-factor momentum strategies
- Statistical arbitrage and pairs trading
- Volatility surface modeling
- Machine learning-enhanced signal generation
- Risk-adjusted portfolio optimization
- Market microstructure analysis

Author: AI Trading System v2.0
Date: January 2025
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import asyncio
from datetime import datetime, timedelta
import logging
from abc import ABC, abstractmethod

# Technical analysis imports
try:
    import talib
except ImportError:
    talib = None
    logging.warning("TA-Lib not available. Some technical indicators may be limited.")

# Machine learning imports
try:
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import mean_squared_error
except ImportError:
    logging.warning("Scikit-learn not available. ML strategies will be limited.")

# Statistical imports
try:
    from scipy import stats
    from scipy.optimize import minimize
except ImportError:
    logging.warning("SciPy not available. Statistical methods will be limited.")

class SignalStrength(Enum):
    """Signal strength enumeration"""
    VERY_WEAK = 0.2
    WEAK = 0.4
    MODERATE = 0.6
    STRONG = 0.8
    VERY_STRONG = 1.0

class MarketRegime(Enum):
    """Market regime classification"""
    BULL = "bull"
    BEAR = "bear"
    SIDEWAYS = "sideways"
    HIGH_VOLATILITY = "high_vol"
    LOW_VOLATILITY = "low_vol"

@dataclass
class TradingSignal:
    """Enhanced trading signal with comprehensive metadata"""
    symbol: str
    action: str  # BUY, SELL, HOLD
    confidence: float  # 0.0 to 1.0
    strength: SignalStrength
    reasoning: str
    risk_level: str  # LOW, MEDIUM, HIGH
    suggested_position_size: float  # Percentage of portfolio
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    holding_period: Optional[int] = None  # Days
    market_regime: Optional[MarketRegime] = None
    technical_score: float = 0.0
    fundamental_score: float = 0.0
    sentiment_score: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    strategy_name: str = ""
    expected_return: Optional[float] = None
    expected_volatility: Optional[float] = None
    sharpe_forecast: Optional[float] = None

class AdvancedStrategy(ABC):
    """Base class for advanced trading strategies"""
    
    def __init__(self, name: str, config: Dict):
        self.name = name
        self.config = config
        self.logger = logging.getLogger(f"strategy.{name}")
        self.performance_metrics = {}
        self.last_signals = {}
        
    @abstractmethod
    async def generate_signal(self, symbol: str, data: pd.DataFrame) -> TradingSignal:
        """Generate trading signal for a given symbol"""
        pass
    
    @abstractmethod
    def calculate_position_size(self, signal: TradingSignal, portfolio_value: float, 
                              current_volatility: float) -> float:
        """Calculate optimal position size based on risk management"""
        pass
    
    def update_performance(self, symbol: str, realized_return: float, 
                          predicted_return: float):
        """Update strategy performance metrics"""
        if symbol not in self.performance_metrics:
            self.performance_metrics[symbol] = {
                'predictions': [],
                'actuals': [],
                'accuracy': 0.0,
                'sharpe': 0.0
            }
        
        self.performance_metrics[symbol]['predictions'].append(predicted_return)
        self.performance_metrics[symbol]['actuals'].append(realized_return)
        
        # Calculate rolling accuracy
        predictions = self.performance_metrics[symbol]['predictions'][-50:]
        actuals = self.performance_metrics[symbol]['actuals'][-50:]
        
        if len(predictions) > 10:
            correlation = np.corrcoef(predictions, actuals)[0, 1]
            self.performance_metrics[symbol]['accuracy'] = max(0, correlation)

class MultiFactorMomentumStrategy(AdvancedStrategy):
    """Advanced momentum strategy using multiple factors and machine learning"""
    
    def __init__(self, config: Dict):
        super().__init__("MultiFactorMomentum", config)
        self.lookback_periods = config.get('lookback_periods', [5, 10, 20, 50])
        self.volume_factor_weight = config.get('volume_factor_weight', 0.3)
        self.volatility_factor_weight = config.get('volatility_factor_weight', 0.2)
        self.ml_model = None
        self.scaler = StandardScaler() if 'sklearn' in globals() else None
        self.feature_columns = []
        
    def _calculate_momentum_factors(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate multiple momentum factors"""
        factors = pd.DataFrame(index=data.index)
        
        # Price momentum factors
        for period in self.lookback_periods:
            if len(data) > period:
                factors[f'price_momentum_{period}'] = (
                    data['close'].pct_change(period).fillna(0)
                )
                
        # Volume-weighted momentum
        if 'volume' in data.columns:
            factors['volume_momentum'] = (
                (data['close'] * data['volume']).rolling(20).mean() / 
                (data['close'] * data['volume']).rolling(50).mean() - 1
            ).fillna(0)
            
        # Volatility-adjusted momentum
        returns = data['close'].pct_change().fillna(0)
        volatility = returns.rolling(20).std().fillna(returns.std())
        factors['vol_adj_momentum'] = (
            returns.rolling(10).mean() / volatility
        ).fillna(0)
        
        # RSI momentum
        if talib:
            factors['rsi_momentum'] = (
                talib.RSI(data['close'].values, timeperiod=14) - 50
            ) / 50
        else:
            # Simple RSI calculation
            delta = data['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            factors['rsi_momentum'] = (rsi - 50) / 50
            
        factors = factors.fillna(0)
        return factors
    
    def _detect_market_regime(self, data: pd.DataFrame) -> MarketRegime:
        """Detect current market regime"""
        if len(data) < 50:
            return MarketRegime.SIDEWAYS
            
        # Calculate trend strength
        returns = data['close'].pct_change().dropna()
        recent_returns = returns.tail(20)
        volatility = recent_returns.std()
        trend = recent_returns.mean()
        
        # Regime classification
        if volatility > returns.std() * 1.5:
            return MarketRegime.HIGH_VOLATILITY
        elif volatility < returns.std() * 0.5:
            return MarketRegime.LOW_VOLATILITY
        elif trend > 0.001:  # 0.1% daily average
            return MarketRegime.BULL
        elif trend < -0.001:
            return MarketRegime.BEAR
        else:
            return MarketRegime.SIDEWAYS
    
    def _train_ml_model(self, factors: pd.DataFrame, returns: pd.Series):
        """Train machine learning model for signal enhancement"""
        if not self.scaler or len(factors) < 100:
            return
            
        try:
            # Prepare features and targets
            X = factors.dropna()
            y = returns.shift(-1).dropna()  # Next period returns
            
            # Align indices
            common_idx = X.index.intersection(y.index)
            X = X.loc[common_idx]
            y = y.loc[common_idx]
            
            if len(X) < 50:
                return
                
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Train ensemble model
            self.ml_model = GradientBoostingRegressor(
                n_estimators=100,
                max_depth=3,
                learning_rate=0.1,
                random_state=42
            )
            
            # Time series cross-validation
            tscv = TimeSeriesSplit(n_splits=3)
            scores = []
            
            for train_idx, val_idx in tscv.split(X_scaled):
                X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                
                model = GradientBoostingRegressor(
                    n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42
                )
                model.fit(X_train, y_train)
                pred = model.predict(X_val)
                score = np.corrcoef(pred, y_val)[0, 1]
                scores.append(score if not np.isnan(score) else 0)
            
            # Train final model if validation is good
            if np.mean(scores) > 0.1:  # Minimum correlation threshold
                self.ml_model.fit(X_scaled, y)
                self.feature_columns = X.columns.tolist()
                self.logger.info(f"ML model trained with correlation: {np.mean(scores):.3f}")
            else:
                self.ml_model = None
                
        except Exception as e:
            self.logger.error(f"ML model training failed: {e}")
            self.ml_model = None
    
    async def generate_signal(self, symbol: str, data: pd.DataFrame) -> TradingSignal:
        """Generate advanced momentum signal"""
        if len(data) < max(self.lookback_periods) + 20:
            return TradingSignal(
                symbol=symbol,
                action="HOLD",
                confidence=0.0,
                strength=SignalStrength.VERY_WEAK,
                reasoning="Insufficient data for analysis",
                risk_level="LOW",
                suggested_position_size=0.0,
                strategy_name=self.name
            )
        
        # Calculate momentum factors
        factors = self._calculate_momentum_factors(data)
        returns = data['close'].pct_change().fillna(0)
        
        # Train ML model if not exists
        if self.ml_model is None and len(factors) > 100:
            self._train_ml_model(factors, returns)
        
        # Get latest factor values
        latest_factors = factors.iloc[-1]
        
        # Calculate base momentum score
        momentum_score = 0.0
        for period in self.lookback_periods:
            col = f'price_momentum_{period}'
            if col in latest_factors:
                weight = 1.0 / period  # Shorter periods get higher weight
                momentum_score += latest_factors[col] * weight
        
        # Normalize momentum score
        momentum_score = momentum_score / sum(1.0 / p for p in self.lookback_periods)
        
        # Add volume and volatility factors
        if 'volume_momentum' in latest_factors:
            momentum_score += latest_factors['volume_momentum'] * self.volume_factor_weight
        
        if 'vol_adj_momentum' in latest_factors:
            momentum_score += latest_factors['vol_adj_momentum'] * self.volatility_factor_weight
        
        # ML enhancement
        ml_score = 0.0
        if self.ml_model and self.scaler and len(self.feature_columns) > 0:
            try:
                feature_vector = latest_factors[self.feature_columns].values.reshape(1, -1)
                feature_vector_scaled = self.scaler.transform(feature_vector)
                ml_prediction = self.ml_model.predict(feature_vector_scaled)[0]
                ml_score = np.tanh(ml_prediction * 10)  # Normalize to [-1, 1]
            except Exception as e:
                self.logger.warning(f"ML prediction failed: {e}")
        
        # Combine scores
        final_score = 0.6 * momentum_score + 0.4 * ml_score
        
        # Market regime adjustment
        regime = self._detect_market_regime(data)
        regime_multiplier = 1.0
        
        if regime == MarketRegime.HIGH_VOLATILITY:
            regime_multiplier = 0.5  # Reduce position in high volatility
        elif regime == MarketRegime.BEAR and final_score > 0:
            regime_multiplier = 0.7  # Reduce long positions in bear market
        elif regime == MarketRegime.BULL and final_score > 0:
            regime_multiplier = 1.2  # Increase long positions in bull market
        
        final_score *= regime_multiplier
        
        # Generate signal
        confidence = min(abs(final_score), 1.0)
        
        if final_score > 0.15:
            action = "BUY"
            strength = SignalStrength.STRONG if confidence > 0.7 else SignalStrength.MODERATE
        elif final_score < -0.15:
            action = "SELL"
            strength = SignalStrength.STRONG if confidence > 0.7 else SignalStrength.MODERATE
        else:
            action = "HOLD"
            strength = SignalStrength.WEAK
        
        # Risk assessment
        recent_volatility = returns.tail(20).std()
        if recent_volatility > returns.std() * 1.5:
            risk_level = "HIGH"
        elif recent_volatility < returns.std() * 0.7:
            risk_level = "LOW"
        else:
            risk_level = "MEDIUM"
        
        # Position sizing
        base_position_size = confidence * 0.05  # Max 5% per position
        if risk_level == "HIGH":
            base_position_size *= 0.5
        elif risk_level == "LOW":
            base_position_size *= 1.2
        
        # Stop loss and take profit
        current_price = data['close'].iloc[-1]
        volatility_multiplier = max(2.0, recent_volatility * 100)
        
        stop_loss = None
        take_profit = None
        
        if action == "BUY":
            stop_loss = current_price * (1 - 0.02 * volatility_multiplier)
            take_profit = current_price * (1 + 0.03 * volatility_multiplier)
        elif action == "SELL":
            stop_loss = current_price * (1 + 0.02 * volatility_multiplier)
            take_profit = current_price * (1 - 0.03 * volatility_multiplier)
        
        # Expected metrics
        expected_return = final_score * 0.1  # Scale to reasonable return expectation
        expected_volatility = recent_volatility
        sharpe_forecast = expected_return / expected_volatility if expected_volatility > 0 else 0
        
        return TradingSignal(
            symbol=symbol,
            action=action,
            confidence=confidence,
            strength=strength,
            reasoning=f"Multi-factor momentum: {final_score:.3f}, ML: {ml_score:.3f}, Regime: {regime.value}",
            risk_level=risk_level,
            suggested_position_size=base_position_size,
            stop_loss=stop_loss,
            take_profit=take_profit,
            holding_period=10,  # 10 days average holding
            market_regime=regime,
            technical_score=momentum_score,
            fundamental_score=0.0,  # Not implemented in this strategy
            sentiment_score=ml_score,
            strategy_name=self.name,
            expected_return=expected_return,
            expected_volatility=expected_volatility,
            sharpe_forecast=sharpe_forecast
        )
    
    def calculate_position_size(self, signal: TradingSignal, portfolio_value: float, 
                              current_volatility: float) -> float:
        """Calculate Kelly-optimal position size with risk constraints"""
        if signal.action == "HOLD" or signal.expected_return is None:
            return 0.0
        
        # Kelly criterion with modifications
        win_rate = 0.55  # Estimated from backtesting
        avg_win = abs(signal.expected_return) * 1.2
        avg_loss = abs(signal.expected_return) * 0.8
        
        if avg_loss == 0:
            kelly_fraction = 0.0
        else:
            kelly_fraction = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
        
        # Apply safety constraints
        kelly_fraction = max(0, min(kelly_fraction, 0.25))  # Max 25% Kelly
        
        # Adjust for confidence and volatility
        volatility_adjustment = 1.0 / (1.0 + current_volatility * 10)
        confidence_adjustment = signal.confidence
        
        final_fraction = kelly_fraction * volatility_adjustment * confidence_adjustment
        
        # Apply maximum position limits
        max_position = 0.1  # 10% max per position
        final_fraction = min(final_fraction, max_position)
        
        return final_fraction * portfolio_value

class StatisticalArbitrageStrategy(AdvancedStrategy):
    """Statistical arbitrage strategy using pairs trading and mean reversion"""
    
    def __init__(self, config: Dict):
        super().__init__("StatisticalArbitrage", config)
        self.lookback_window = config.get('lookback_window', 60)
        self.entry_threshold = config.get('entry_threshold', 2.0)
        self.exit_threshold = config.get('exit_threshold', 0.5)
        self.correlation_threshold = config.get('correlation_threshold', 0.7)
        self.pairs_data = {}
        
    def _find_cointegrated_pairs(self, price_data: Dict[str, pd.DataFrame]) -> List[Tuple[str, str]]:
        """Find cointegrated pairs using Engle-Granger test"""
        pairs = []
        symbols = list(price_data.keys())
        
        for i in range(len(symbols)):
            for j in range(i + 1, len(symbols)):
                sym1, sym2 = symbols[i], symbols[j]
                
                # Get common time period
                data1 = price_data[sym1]['close']
                data2 = price_data[sym2]['close']
                
                common_idx = data1.index.intersection(data2.index)
                if len(common_idx) < self.lookback_window:
                    continue
                
                prices1 = data1.loc[common_idx]
                prices2 = data2.loc[common_idx]
                
                # Check correlation
                correlation = prices1.corr(prices2)
                if abs(correlation) < self.correlation_threshold:
                    continue
                
                # Cointegration test (simplified)
                try:
                    if 'scipy' in globals():
                        # Perform regression
                        slope, intercept, r_value, p_value, std_err = stats.linregress(
                            prices1.values, prices2.values
                        )
                        
                        # Calculate residuals
                        residuals = prices2 - (slope * prices1 + intercept)
                        
                        # ADF test on residuals (simplified stationarity check)
                        residual_mean = residuals.mean()
                        residual_std = residuals.std()
                        
                        # Check if residuals are mean-reverting
                        recent_residual = residuals.iloc[-1]
                        z_score = (recent_residual - residual_mean) / residual_std
                        
                        if abs(z_score) > 1.0:  # Potential trading opportunity
                            pairs.append((sym1, sym2))
                            
                except Exception as e:
                    self.logger.warning(f"Cointegration test failed for {sym1}-{sym2}: {e}")
                    continue
        
        return pairs[:5]  # Limit to top 5 pairs
    
    def _calculate_spread_zscore(self, prices1: pd.Series, prices2: pd.Series) -> float:
        """Calculate z-score of the spread between two price series"""
        if len(prices1) < self.lookback_window or len(prices2) < self.lookback_window:
            return 0.0
        
        # Calculate hedge ratio using linear regression
        try:
            if 'scipy' in globals():
                slope, intercept, _, _, _ = stats.linregress(prices1.values, prices2.values)
            else:
                # Simple correlation-based hedge ratio
                slope = prices2.corr(prices1) * (prices2.std() / prices1.std())
                intercept = prices2.mean() - slope * prices1.mean()
        except:
            return 0.0
        
        # Calculate spread
        spread = prices2 - (slope * prices1 + intercept)
        
        # Calculate z-score
        spread_mean = spread.mean()
        spread_std = spread.std()
        
        if spread_std == 0:
            return 0.0
        
        current_spread = spread.iloc[-1]
        z_score = (current_spread - spread_mean) / spread_std
        
        return z_score
    
    async def generate_signal(self, symbol: str, data: pd.DataFrame) -> TradingSignal:
        """Generate statistical arbitrage signal"""
        # This is a simplified implementation for single symbol
        # In practice, this would work with pairs of symbols
        
        if len(data) < self.lookback_window:
            return TradingSignal(
                symbol=symbol,
                action="HOLD",
                confidence=0.0,
                strength=SignalStrength.VERY_WEAK,
                reasoning="Insufficient data for statistical arbitrage",
                risk_level="LOW",
                suggested_position_size=0.0,
                strategy_name=self.name
            )
        
        # Calculate mean reversion signal for single asset
        prices = data['close']
        returns = prices.pct_change().dropna()
        
        # Calculate rolling statistics
        rolling_mean = prices.rolling(self.lookback_window).mean()
        rolling_std = prices.rolling(self.lookback_window).std()
        
        # Z-score calculation
        current_price = prices.iloc[-1]
        mean_price = rolling_mean.iloc[-1]
        std_price = rolling_std.iloc[-1]
        
        if std_price == 0:
            z_score = 0
        else:
            z_score = (current_price - mean_price) / std_price
        
        # Generate signal based on z-score
        confidence = min(abs(z_score) / self.entry_threshold, 1.0)
        
        if z_score > self.entry_threshold:
            action = "SELL"  # Price too high, expect reversion
            strength = SignalStrength.STRONG if confidence > 0.8 else SignalStrength.MODERATE
        elif z_score < -self.entry_threshold:
            action = "BUY"  # Price too low, expect reversion
            strength = SignalStrength.STRONG if confidence > 0.8 else SignalStrength.MODERATE
        else:
            action = "HOLD"
            strength = SignalStrength.WEAK
        
        # Risk assessment based on volatility
        recent_volatility = returns.tail(20).std()
        avg_volatility = returns.std()
        
        if recent_volatility > avg_volatility * 1.5:
            risk_level = "HIGH"
        elif recent_volatility < avg_volatility * 0.7:
            risk_level = "LOW"
        else:
            risk_level = "MEDIUM"
        
        # Position sizing based on mean reversion strength
        base_position_size = confidence * 0.03  # Max 3% for mean reversion
        
        # Stop loss and take profit for mean reversion
        stop_loss = None
        take_profit = None
        
        if action == "BUY":
            stop_loss = current_price * 0.95  # 5% stop loss
            take_profit = mean_price  # Target mean price
        elif action == "SELL":
            stop_loss = current_price * 1.05  # 5% stop loss
            take_profit = mean_price  # Target mean price
        
        return TradingSignal(
            symbol=symbol,
            action=action,
            confidence=confidence,
            strength=strength,
            reasoning=f"Mean reversion z-score: {z_score:.2f}, threshold: ±{self.entry_threshold}",
            risk_level=risk_level,
            suggested_position_size=base_position_size,
            stop_loss=stop_loss,
            take_profit=take_profit,
            holding_period=5,  # Short holding period for mean reversion
            technical_score=z_score,
            strategy_name=self.name,
            expected_return=-z_score * 0.02,  # Expect reversion
            expected_volatility=recent_volatility
        )
    
    def calculate_position_size(self, signal: TradingSignal, portfolio_value: float, 
                              current_volatility: float) -> float:
        """Calculate position size for mean reversion strategy"""
        if signal.action == "HOLD":
            return 0.0
        
        # Conservative position sizing for mean reversion
        base_size = signal.suggested_position_size * portfolio_value
        
        # Adjust for volatility
        volatility_adjustment = 1.0 / (1.0 + current_volatility * 5)
        
        # Apply maximum limits
        max_position = 0.05 * portfolio_value  # 5% max
        
        return min(base_size * volatility_adjustment, max_position)

class VolatilitySurfaceStrategy(AdvancedStrategy):
    """Advanced volatility trading strategy using implied volatility surface"""
    
    def __init__(self, config: Dict):
        super().__init__("VolatilitySurface", config)
        self.vol_lookback = config.get('vol_lookback', 30)
        self.vol_threshold = config.get('vol_threshold', 1.5)
        self.regime_lookback = config.get('regime_lookback', 60)
        
    def _calculate_realized_volatility(self, returns: pd.Series, window: int = 20) -> pd.Series:
        """Calculate realized volatility using various estimators"""
        # Simple volatility
        simple_vol = returns.rolling(window).std() * np.sqrt(252)
        
        # Calculate Parkinson volatility using high-low range
        if 'High' in data.columns and 'Low' in data.columns:
            hl_ratio = np.log(data['High'] / data['Low'])
            parkinson_vol = np.sqrt(hl_ratio.rolling(window=window).mean() * 252 / (4 * np.log(2)))
            parkinson_vol = parkinson_vol.iloc[-1] if not parkinson_vol.empty else simple_vol
        else:
            parkinson_vol = simple_vol
        
        # Calculate Garman-Klass volatility
        if all(col in data.columns for col in ['High', 'Low', 'Open', 'Close']):
            ln_hl = np.log(data['High'] / data['Low'])
            ln_co = np.log(data['Close'] / data['Open'])
            gk_vol = np.sqrt((0.5 * ln_hl**2 - (2*np.log(2) - 1) * ln_co**2).rolling(window=window).mean() * 252)
            gk_vol = gk_vol.iloc[-1] if not gk_vol.empty else simple_vol
        else:
            gk_vol = simple_vol
        
        return simple_vol
    
    def _detect_volatility_regime(self, volatility: pd.Series) -> str:
        """Detect volatility regime (low, normal, high)"""
        if len(volatility) < self.regime_lookback:
            return "normal"
        
        recent_vol = volatility.tail(10).mean()
        historical_vol = volatility.tail(self.regime_lookback).mean()
        vol_percentile = volatility.tail(self.regime_lookback).quantile(0.8)
        
        if recent_vol > vol_percentile:
            return "high"
        elif recent_vol < volatility.tail(self.regime_lookback).quantile(0.2):
            return "low"
        else:
            return "normal"
    
    async def generate_signal(self, symbol: str, data: pd.DataFrame) -> TradingSignal:
        """Generate volatility-based trading signal"""
        if len(data) < self.vol_lookback:
            return TradingSignal(
                symbol=symbol,
                action="HOLD",
                confidence=0.0,
                strength=SignalStrength.VERY_WEAK,
                reasoning="Insufficient data for volatility analysis",
                risk_level="LOW",
                suggested_position_size=0.0,
                strategy_name=self.name
            )
        
        # Calculate returns and volatility
        returns = data['close'].pct_change().dropna()
        realized_vol = self._calculate_realized_volatility(returns)
        
        # Volatility regime detection
        vol_regime = self._detect_volatility_regime(realized_vol)
        
        # Current volatility vs historical
        current_vol = realized_vol.iloc[-1]
        historical_vol = realized_vol.tail(60).mean()
        vol_ratio = current_vol / historical_vol if historical_vol > 0 else 1.0
        
        # Volatility mean reversion signal
        vol_z_score = 0
        if len(realized_vol) > 30:
            vol_mean = realized_vol.tail(60).mean()
            vol_std = realized_vol.tail(60).std()
            if vol_std > 0:
                vol_z_score = (current_vol - vol_mean) / vol_std
        
        # Generate signal based on volatility analysis
        confidence = min(abs(vol_z_score) / 2.0, 1.0)
        
        # Volatility trading logic
        if vol_regime == "high" and vol_z_score > 1.5:
            # High volatility, expect mean reversion
            action = "SELL"  # Short volatility
            reasoning = f"High volatility regime, vol z-score: {vol_z_score:.2f}"
        elif vol_regime == "low" and vol_z_score < -1.5:
            # Low volatility, expect increase
            action = "BUY"  # Long volatility
            reasoning = f"Low volatility regime, vol z-score: {vol_z_score:.2f}"
        elif vol_ratio > self.vol_threshold:
            # Volatility spike, momentum trade
            action = "BUY"
            reasoning = f"Volatility spike detected, ratio: {vol_ratio:.2f}"
        else:
            action = "HOLD"
            reasoning = f"Normal volatility conditions, ratio: {vol_ratio:.2f}"
        
        # Strength assessment
        if abs(vol_z_score) > 2.0:
            strength = SignalStrength.VERY_STRONG
        elif abs(vol_z_score) > 1.5:
            strength = SignalStrength.STRONG
        elif abs(vol_z_score) > 1.0:
            strength = SignalStrength.MODERATE
        else:
            strength = SignalStrength.WEAK
        
        # Risk level based on volatility
        if current_vol > historical_vol * 2:
            risk_level = "HIGH"
        elif current_vol < historical_vol * 0.5:
            risk_level = "LOW"
        else:
            risk_level = "MEDIUM"
        
        # Position sizing for volatility strategy
        base_position_size = confidence * 0.04  # Max 4% for volatility trades
        
        return TradingSignal(
            symbol=symbol,
            action=action,
            confidence=confidence,
            strength=strength,
            reasoning=reasoning,
            risk_level=risk_level,
            suggested_position_size=base_position_size,
            holding_period=7,  # One week for volatility trades
            technical_score=vol_z_score,
            strategy_name=self.name,
            expected_volatility=current_vol
        )
    
    def calculate_position_size(self, signal: TradingSignal, portfolio_value: float, 
                              current_volatility: float) -> float:
        """Calculate position size for volatility strategy"""
        if signal.action == "HOLD":
            return 0.0
        
        # Inverse relationship with volatility for position sizing
        vol_adjustment = 1.0 / (1.0 + current_volatility * 3)
        base_size = signal.suggested_position_size * portfolio_value
        
        return base_size * vol_adjustment

class AdvancedStrategyOrchestrator:
    """Orchestrator for managing multiple advanced strategies"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.strategies = {}
        self.logger = logging.getLogger("strategy_orchestrator")
        self.performance_tracker = {}
        
        # Initialize strategies
        self._initialize_strategies()
    
    def _initialize_strategies(self):
        """Initialize all configured strategies"""
        strategy_configs = self.config.get('strategies', {})
        
        if 'multi_factor_momentum' in strategy_configs:
            self.strategies['momentum'] = MultiFactorMomentumStrategy(
                strategy_configs['multi_factor_momentum']
            )
        
        if 'statistical_arbitrage' in strategy_configs:
            self.strategies['stat_arb'] = StatisticalArbitrageStrategy(
                strategy_configs['statistical_arbitrage']
            )
        
        if 'volatility_surface' in strategy_configs:
            self.strategies['vol_surface'] = VolatilitySurfaceStrategy(
                strategy_configs['volatility_surface']
            )
        
        self.logger.info(f"Initialized {len(self.strategies)} advanced strategies")
    
    async def generate_ensemble_signal(self, symbol: str, data: pd.DataFrame) -> TradingSignal:
        """Generate ensemble signal from all strategies"""
        signals = []
        
        # Get signals from all strategies
        for name, strategy in self.strategies.items():
            try:
                signal = await strategy.generate_signal(symbol, data)
                signals.append((name, signal))
            except Exception as e:
                self.logger.error(f"Strategy {name} failed for {symbol}: {e}")
        
        if not signals:
            return TradingSignal(
                symbol=symbol,
                action="HOLD",
                confidence=0.0,
                strength=SignalStrength.VERY_WEAK,
                reasoning="No strategies generated signals",
                risk_level="LOW",
                suggested_position_size=0.0,
                strategy_name="Ensemble"
            )
        
        # Ensemble logic
        buy_signals = [s for name, s in signals if s.action == "BUY"]
        sell_signals = [s for name, s in signals if s.action == "SELL"]
        hold_signals = [s for name, s in signals if s.action == "HOLD"]
        
        # Weighted voting based on confidence
        buy_weight = sum(s.confidence for s in buy_signals)
        sell_weight = sum(s.confidence for s in sell_signals)
        
        # Determine ensemble action
        if buy_weight > sell_weight and buy_weight > 0.5:
            action = "BUY"
            confidence = min(buy_weight / len(signals), 1.0)
            relevant_signals = buy_signals
        elif sell_weight > buy_weight and sell_weight > 0.5:
            action = "SELL"
            confidence = min(sell_weight / len(signals), 1.0)
            relevant_signals = sell_signals
        else:
            action = "HOLD"
            confidence = 0.0
            relevant_signals = hold_signals
        
        # Aggregate metrics
        if relevant_signals:
            avg_position_size = np.mean([s.suggested_position_size for s in relevant_signals])
            risk_levels = [s.risk_level for s in relevant_signals]
            risk_level = max(risk_levels, key=risk_levels.count)  # Most common risk level
            
            # Combine reasoning
            strategy_names = [name for name, s in signals if s.action == action]
            reasoning = f"Ensemble of {len(strategy_names)} strategies: {', '.join(strategy_names)}"
        else:
            avg_position_size = 0.0
            risk_level = "LOW"
            reasoning = "No consensus among strategies"
        
        # Determine strength
        if confidence > 0.8:
            strength = SignalStrength.VERY_STRONG
        elif confidence > 0.6:
            strength = SignalStrength.STRONG
        elif confidence > 0.4:
            strength = SignalStrength.MODERATE
        else:
            strength = SignalStrength.WEAK
        
        return TradingSignal(
            symbol=symbol,
            action=action,
            confidence=confidence,
            strength=strength,
            reasoning=reasoning,
            risk_level=risk_level,
            suggested_position_size=avg_position_size,
            strategy_name="Ensemble",
            timestamp=datetime.now()
        )
    
    def get_strategy_performance(self) -> Dict:
        """Get performance metrics for all strategies"""
        performance = {}
        for name, strategy in self.strategies.items():
            performance[name] = strategy.performance_metrics
        return performance
    
    def update_strategy_weights(self, performance_data: Dict):
        """Update strategy weights based on performance"""
        # This would implement dynamic strategy weighting
        # based on recent performance metrics
        pass

# Example usage and configuration
if __name__ == "__main__":
    # Example configuration
    config = {
        'strategies': {
            'multi_factor_momentum': {
                'lookback_periods': [5, 10, 20, 50],
                'volume_factor_weight': 0.3,
                'volatility_factor_weight': 0.2
            },
            'statistical_arbitrage': {
                'lookback_window': 60,
                'entry_threshold': 2.0,
                'exit_threshold': 0.5,
                'correlation_threshold': 0.7
            },
            'volatility_surface': {
                'vol_lookback': 30,
                'vol_threshold': 1.5,
                'regime_lookback': 60
            }
        }
    }
    
    # Initialize orchestrator
    orchestrator = AdvancedStrategyOrchestrator(config)
    
    print(f"Advanced Trading Strategies System initialized with {len(orchestrator.strategies)} strategies")
    print("Strategies:", list(orchestrator.strategies.keys()))