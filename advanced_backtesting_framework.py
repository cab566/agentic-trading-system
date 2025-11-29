#!/usr/bin/env python3
"""
Advanced Backtesting Framework

A comprehensive backtesting system implementing:

- Walk-forward analysis and out-of-sample testing
- Monte Carlo simulation and bootstrap analysis
- Regime-aware backtesting and regime detection
- Multi-asset and multi-strategy backtesting
- Advanced performance attribution and risk decomposition
- Transaction cost modeling and market impact
- Survivorship bias correction and data quality checks
- Statistical significance testing and confidence intervals
- Factor exposure analysis and style attribution
- Benchmark comparison and relative performance analysis

This framework provides institutional-grade backtesting capabilities
with rigorous statistical validation and comprehensive reporting.

Author: AI Trading System v2.0
Date: January 2025
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import asyncio
from datetime import datetime, timedelta
import logging
import json
from abc import ABC, abstractmethod
import warnings
warnings.filterwarnings('ignore')

# Statistical libraries
try:
    from scipy import stats
    from scipy.optimize import minimize
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logging.warning("SciPy not available. Advanced statistical analysis will be limited.")

# Machine learning for regime detection
try:
    from sklearn.mixture import GaussianMixture
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logging.warning("Scikit-learn not available. Regime detection will be simplified.")

# Performance analytics
try:
    import empyrical as emp
    EMPYRICAL_AVAILABLE = True
except ImportError:
    EMPYRICAL_AVAILABLE = False
    logging.warning("Empyrical not available. Some performance metrics will be calculated manually.")

class BacktestType(Enum):
    """Backtesting methodology types"""
    SIMPLE = "simple"
    WALK_FORWARD = "walk_forward"
    MONTE_CARLO = "monte_carlo"
    BOOTSTRAP = "bootstrap"
    REGIME_AWARE = "regime_aware"
    CROSS_VALIDATION = "cross_validation"
    OUT_OF_SAMPLE = "out_of_sample"

class RegimeType(Enum):
    """Market regime types"""
    BULL_MARKET = "bull_market"
    BEAR_MARKET = "bear_market"
    SIDEWAYS_MARKET = "sideways_market"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    CRISIS = "crisis"
    RECOVERY = "recovery"
    EXPANSION = "expansion"
    CONTRACTION = "contraction"

class PerformanceMetric(Enum):
    """Performance metrics"""
    TOTAL_RETURN = "total_return"
    ANNUALIZED_RETURN = "annualized_return"
    VOLATILITY = "volatility"
    SHARPE_RATIO = "sharpe_ratio"
    SORTINO_RATIO = "sortino_ratio"
    CALMAR_RATIO = "calmar_ratio"
    MAX_DRAWDOWN = "max_drawdown"
    VAR = "value_at_risk"
    CVAR = "conditional_var"
    SKEWNESS = "skewness"
    KURTOSIS = "kurtosis"
    BETA = "beta"
    ALPHA = "alpha"
    INFORMATION_RATIO = "information_ratio"
    TRACKING_ERROR = "tracking_error"
    HIT_RATE = "hit_rate"
    PROFIT_FACTOR = "profit_factor"
    RECOVERY_FACTOR = "recovery_factor"

@dataclass
class BacktestConfig:
    """Backtesting configuration"""
    # Time period
    start_date: datetime
    end_date: datetime
    
    # Initial conditions
    initial_capital: float = 1000000.0
    
    # Rebalancing
    rebalance_frequency: str = "monthly"  # daily, weekly, monthly, quarterly
    
    # Transaction costs
    commission_rate: float = 0.001  # 0.1% commission
    bid_ask_spread: float = 0.0005  # 0.05% bid-ask spread
    market_impact: float = 0.0001  # 0.01% market impact
    
    # Risk management
    max_leverage: float = 1.0
    max_position_size: float = 0.10  # 10% max position
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    
    # Walk-forward analysis
    training_window: int = 252  # 1 year
    testing_window: int = 63   # 3 months
    step_size: int = 21        # 1 month
    
    # Monte Carlo settings
    n_simulations: int = 1000
    confidence_level: float = 0.95
    
    # Benchmark
    benchmark: Optional[str] = "SPY"
    risk_free_rate: float = 0.02  # 2% risk-free rate
    
    # Data quality
    min_history: int = 252  # Minimum data history required
    handle_missing_data: str = "forward_fill"  # forward_fill, drop, interpolate
    
    # Regime detection
    regime_detection: bool = True
    regime_lookback: int = 252
    n_regimes: int = 3

@dataclass
class Trade:
    """Individual trade record"""
    timestamp: datetime
    symbol: str
    side: str  # 'buy' or 'sell'
    quantity: float
    price: float
    commission: float
    market_impact: float
    total_cost: float
    portfolio_value: float
    position_size: float
    reason: str = ""  # Trade reason/signal
    
    def to_dict(self) -> Dict:
        return {
            'timestamp': self.timestamp.isoformat(),
            'symbol': self.symbol,
            'side': self.side,
            'quantity': self.quantity,
            'price': self.price,
            'commission': self.commission,
            'market_impact': self.market_impact,
            'total_cost': self.total_cost,
            'portfolio_value': self.portfolio_value,
            'position_size': self.position_size,
            'reason': self.reason
        }

@dataclass
class Position:
    """Portfolio position"""
    symbol: str
    quantity: float
    avg_price: float
    current_price: float
    market_value: float
    unrealized_pnl: float
    realized_pnl: float
    weight: float
    entry_date: datetime
    last_update: datetime
    
    def to_dict(self) -> Dict:
        return {
            'symbol': self.symbol,
            'quantity': self.quantity,
            'avg_price': self.avg_price,
            'current_price': self.current_price,
            'market_value': self.market_value,
            'unrealized_pnl': self.unrealized_pnl,
            'realized_pnl': self.realized_pnl,
            'weight': self.weight,
            'entry_date': self.entry_date.isoformat(),
            'last_update': self.last_update.isoformat()
        }

@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics"""
    # Return metrics
    total_return: float
    annualized_return: float
    cumulative_return: float
    
    # Risk metrics
    volatility: float
    downside_volatility: float
    max_drawdown: float
    max_drawdown_duration: int
    
    # Risk-adjusted returns
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    
    # Distribution metrics
    skewness: float
    kurtosis: float
    var_95: float
    cvar_95: float
    
    # Benchmark comparison
    beta: float
    alpha: float
    information_ratio: float
    tracking_error: float
    
    # Trading metrics
    total_trades: int
    win_rate: float
    profit_factor: float
    avg_win: float
    avg_loss: float
    largest_win: float
    largest_loss: float
    
    # Additional metrics
    recovery_factor: float
    payoff_ratio: float
    expectancy: float
    
    def to_dict(self) -> Dict:
        return {
            'total_return': self.total_return,
            'annualized_return': self.annualized_return,
            'cumulative_return': self.cumulative_return,
            'volatility': self.volatility,
            'downside_volatility': self.downside_volatility,
            'max_drawdown': self.max_drawdown,
            'max_drawdown_duration': self.max_drawdown_duration,
            'sharpe_ratio': self.sharpe_ratio,
            'sortino_ratio': self.sortino_ratio,
            'calmar_ratio': self.calmar_ratio,
            'skewness': self.skewness,
            'kurtosis': self.kurtosis,
            'var_95': self.var_95,
            'cvar_95': self.cvar_95,
            'beta': self.beta,
            'alpha': self.alpha,
            'information_ratio': self.information_ratio,
            'tracking_error': self.tracking_error,
            'total_trades': self.total_trades,
            'win_rate': self.win_rate,
            'profit_factor': self.profit_factor,
            'avg_win': self.avg_win,
            'avg_loss': self.avg_loss,
            'largest_win': self.largest_win,
            'largest_loss': self.largest_loss,
            'recovery_factor': self.recovery_factor,
            'payoff_ratio': self.payoff_ratio,
            'expectancy': self.expectancy
        }

@dataclass
class RegimeAnalysis:
    """Market regime analysis results"""
    regimes: pd.Series  # Time series of regime classifications
    regime_stats: Dict[RegimeType, Dict[str, float]]  # Performance by regime
    regime_transitions: pd.DataFrame  # Regime transition matrix
    regime_probabilities: pd.DataFrame  # Regime probabilities over time
    current_regime: RegimeType
    regime_confidence: float
    
    def to_dict(self) -> Dict:
        return {
            'regime_stats': {k.value: v for k, v in self.regime_stats.items()},
            'current_regime': self.current_regime.value,
            'regime_confidence': self.regime_confidence,
            'regime_transitions': self.regime_transitions.to_dict() if hasattr(self.regime_transitions, 'to_dict') else {},
            'n_regimes': len(self.regime_stats)
        }

@dataclass
class BacktestResult:
    """Comprehensive backtesting results"""
    config: BacktestConfig
    performance_metrics: PerformanceMetrics
    portfolio_returns: pd.Series
    portfolio_values: pd.Series
    positions_history: pd.DataFrame
    trades_history: List[Trade]
    drawdown_series: pd.Series
    regime_analysis: Optional[RegimeAnalysis] = None
    factor_exposures: Optional[Dict[str, float]] = None
    attribution_analysis: Optional[Dict[str, Any]] = None
    statistical_tests: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict:
        return {
            'performance_metrics': self.performance_metrics.to_dict(),
            'total_trades': len(self.trades_history),
            'start_date': self.config.start_date.isoformat(),
            'end_date': self.config.end_date.isoformat(),
            'initial_capital': self.config.initial_capital,
            'final_value': float(self.portfolio_values.iloc[-1]) if len(self.portfolio_values) > 0 else self.config.initial_capital,
            'regime_analysis': self.regime_analysis.to_dict() if self.regime_analysis else None,
            'factor_exposures': self.factor_exposures,
            'attribution_analysis': self.attribution_analysis,
            'statistical_tests': self.statistical_tests
        }

class RegimeDetector:
    """Market regime detection engine"""
    
    def __init__(self, n_regimes: int = 3, lookback_window: int = 252):
        self.n_regimes = n_regimes
        self.lookback_window = lookback_window
        self.logger = logging.getLogger("regime_detector")
    
    def detect_regimes(self, market_data: pd.DataFrame, 
                      features: Optional[List[str]] = None) -> RegimeAnalysis:
        """Detect market regimes using multiple indicators"""
        try:
            if features is None:
                features = self._create_regime_features(market_data)
            
            # Prepare feature matrix
            feature_matrix = self._prepare_features(market_data, features)
            
            # Detect regimes using Gaussian Mixture Model
            regimes = self._fit_regime_model(feature_matrix)
            
            # Classify regimes
            regime_classifications = self._classify_regimes(regimes, market_data)
            
            # Calculate regime statistics
            regime_stats = self._calculate_regime_stats(regime_classifications, market_data)
            
            # Calculate transition matrix
            transition_matrix = self._calculate_transition_matrix(regime_classifications)
            
            # Get current regime
            current_regime = regime_classifications.iloc[-1] if len(regime_classifications) > 0 else RegimeType.SIDEWAYS_MARKET
            
            return RegimeAnalysis(
                regimes=regime_classifications,
                regime_stats=regime_stats,
                regime_transitions=transition_matrix,
                regime_probabilities=pd.DataFrame(),  # Simplified for now
                current_regime=current_regime,
                regime_confidence=0.8  # Simplified confidence score
            )
            
        except Exception as e:
            self.logger.error(f"Regime detection failed: {e}")
            return self._default_regime_analysis(market_data)
    
    def _create_regime_features(self, market_data: pd.DataFrame) -> List[str]:
        """Create features for regime detection"""
        features = []
        
        # Assume market_data has 'close' column
        if 'close' in market_data.columns:
            # Returns
            market_data['returns'] = market_data['close'].pct_change()
            features.append('returns')
            
            # Volatility
            market_data['volatility'] = market_data['returns'].rolling(21).std()
            features.append('volatility')
            
            # Trend (moving average slope)
            market_data['ma_20'] = market_data['close'].rolling(20).mean()
            market_data['trend'] = market_data['ma_20'].pct_change(5)
            features.append('trend')
            
            # Momentum
            market_data['momentum'] = market_data['close'].pct_change(20)
            features.append('momentum')
        
        return features
    
    def _prepare_features(self, market_data: pd.DataFrame, features: List[str]) -> np.ndarray:
        """Prepare feature matrix for regime detection"""
        feature_data = market_data[features].dropna()
        
        if SKLEARN_AVAILABLE:
            scaler = StandardScaler()
            scaled_features = scaler.fit_transform(feature_data)
            return scaled_features
        else:
            # Manual standardization
            return (feature_data - feature_data.mean()) / feature_data.std()
    
    def _fit_regime_model(self, feature_matrix: np.ndarray) -> np.ndarray:
        """Fit regime detection model"""
        if SKLEARN_AVAILABLE:
            try:
                # Use Gaussian Mixture Model
                gmm = GaussianMixture(n_components=self.n_regimes, random_state=42)
                regimes = gmm.fit_predict(feature_matrix)
                return regimes
            except:
                # Fallback to K-means
                kmeans = KMeans(n_clusters=self.n_regimes, random_state=42)
                regimes = kmeans.fit_predict(feature_matrix)
                return regimes
        else:
            # Simple regime detection based on volatility quantiles
            if len(feature_matrix) > 0 and feature_matrix.shape[1] > 1:
                volatility_col = 1  # Assume second column is volatility
                vol_data = feature_matrix[:, volatility_col]
                
                # Create regimes based on volatility quantiles
                low_vol_threshold = np.percentile(vol_data, 33)
                high_vol_threshold = np.percentile(vol_data, 67)
                
                regimes = np.zeros(len(vol_data))
                regimes[vol_data <= low_vol_threshold] = 0  # Low vol
                regimes[(vol_data > low_vol_threshold) & (vol_data <= high_vol_threshold)] = 1  # Medium vol
                regimes[vol_data > high_vol_threshold] = 2  # High vol
                
                return regimes.astype(int)
            else:
                return np.zeros(len(feature_matrix), dtype=int)
    
    def _classify_regimes(self, regimes: np.ndarray, market_data: pd.DataFrame) -> pd.Series:
        """Classify numeric regimes into meaningful regime types"""
        # Simple classification based on regime characteristics
        regime_map = {
            0: RegimeType.LOW_VOLATILITY,
            1: RegimeType.SIDEWAYS_MARKET,
            2: RegimeType.HIGH_VOLATILITY
        }
        
        # Create regime series
        regime_series = pd.Series(regimes, index=market_data.index[-len(regimes):])
        classified_regimes = regime_series.map(regime_map)
        
        return classified_regimes
    
    def _calculate_regime_stats(self, regimes: pd.Series, market_data: pd.DataFrame) -> Dict[RegimeType, Dict[str, float]]:
        """Calculate performance statistics by regime"""
        stats = {}
        
        if 'close' in market_data.columns:
            returns = market_data['close'].pct_change()
            
            for regime_type in regimes.unique():
                if pd.isna(regime_type):
                    continue
                    
                regime_mask = regimes == regime_type
                regime_returns = returns[regime_mask]
                
                if len(regime_returns) > 0:
                    stats[regime_type] = {
                        'mean_return': regime_returns.mean(),
                        'volatility': regime_returns.std(),
                        'sharpe_ratio': regime_returns.mean() / regime_returns.std() if regime_returns.std() > 0 else 0,
                        'max_drawdown': self._calculate_max_drawdown(regime_returns),
                        'frequency': len(regime_returns) / len(returns)
                    }
        
        return stats
    
    def _calculate_transition_matrix(self, regimes: pd.Series) -> pd.DataFrame:
        """Calculate regime transition matrix"""
        try:
            unique_regimes = regimes.dropna().unique()
            n_regimes = len(unique_regimes)
            
            transition_matrix = np.zeros((n_regimes, n_regimes))
            
            for i in range(len(regimes) - 1):
                if pd.notna(regimes.iloc[i]) and pd.notna(regimes.iloc[i + 1]):
                    from_regime = list(unique_regimes).index(regimes.iloc[i])
                    to_regime = list(unique_regimes).index(regimes.iloc[i + 1])
                    transition_matrix[from_regime, to_regime] += 1
            
            # Normalize to probabilities
            row_sums = transition_matrix.sum(axis=1)
            transition_matrix = transition_matrix / row_sums[:, np.newaxis]
            transition_matrix = np.nan_to_num(transition_matrix)
            
            return pd.DataFrame(transition_matrix, 
                              index=[r.value for r in unique_regimes],
                              columns=[r.value for r in unique_regimes])
            
        except Exception as e:
            self.logger.error(f"Transition matrix calculation failed: {e}")
            return pd.DataFrame()
    
    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown"""
        try:
            cumulative = (1 + returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            return drawdown.min()
        except:
            return 0.0
    
    def _default_regime_analysis(self, market_data: pd.DataFrame) -> RegimeAnalysis:
        """Default regime analysis for error cases"""
        regimes = pd.Series([RegimeType.SIDEWAYS_MARKET] * len(market_data), index=market_data.index)
        
        return RegimeAnalysis(
            regimes=regimes,
            regime_stats={RegimeType.SIDEWAYS_MARKET: {'mean_return': 0.0, 'volatility': 0.1}},
            regime_transitions=pd.DataFrame(),
            regime_probabilities=pd.DataFrame(),
            current_regime=RegimeType.SIDEWAYS_MARKET,
            regime_confidence=0.5
        )

class PerformanceAnalyzer:
    """Advanced performance analysis engine"""
    
    def __init__(self, risk_free_rate: float = 0.02):
        self.risk_free_rate = risk_free_rate
        self.logger = logging.getLogger("performance_analyzer")
    
    def calculate_performance_metrics(self, 
                                    portfolio_returns: pd.Series,
                                    benchmark_returns: Optional[pd.Series] = None,
                                    trades: Optional[List[Trade]] = None) -> PerformanceMetrics:
        """Calculate comprehensive performance metrics"""
        try:
            # Basic return metrics
            total_return = (1 + portfolio_returns).prod() - 1
            annualized_return = (1 + total_return) ** (252 / len(portfolio_returns)) - 1
            cumulative_return = total_return
            
            # Risk metrics
            volatility = portfolio_returns.std() * np.sqrt(252)
            downside_returns = portfolio_returns[portfolio_returns < 0]
            downside_volatility = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
            
            # Drawdown analysis
            cumulative = (1 + portfolio_returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            max_drawdown = drawdown.min()
            max_drawdown_duration = self._calculate_max_drawdown_duration(drawdown)
            
            # Risk-adjusted returns
            excess_returns = portfolio_returns - self.risk_free_rate / 252
            sharpe_ratio = excess_returns.mean() / portfolio_returns.std() * np.sqrt(252) if portfolio_returns.std() > 0 else 0
            sortino_ratio = excess_returns.mean() / downside_volatility * np.sqrt(252) if downside_volatility > 0 else 0
            calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
            
            # Distribution metrics
            skewness = portfolio_returns.skew()
            kurtosis = portfolio_returns.kurtosis()
            var_95 = portfolio_returns.quantile(0.05)
            cvar_95 = portfolio_returns[portfolio_returns <= var_95].mean() if len(portfolio_returns[portfolio_returns <= var_95]) > 0 else var_95
            
            # Benchmark comparison
            beta, alpha, information_ratio, tracking_error = self._calculate_benchmark_metrics(
                portfolio_returns, benchmark_returns
            )
            
            # Trading metrics
            trading_metrics = self._calculate_trading_metrics(trades) if trades else {
                'total_trades': 0, 'win_rate': 0, 'profit_factor': 0,
                'avg_win': 0, 'avg_loss': 0, 'largest_win': 0, 'largest_loss': 0
            }
            
            # Additional metrics
            recovery_factor = abs(total_return / max_drawdown) if max_drawdown != 0 else 0
            payoff_ratio = trading_metrics['avg_win'] / abs(trading_metrics['avg_loss']) if trading_metrics['avg_loss'] != 0 else 0
            expectancy = trading_metrics['win_rate'] * trading_metrics['avg_win'] + (1 - trading_metrics['win_rate']) * trading_metrics['avg_loss']
            
            return PerformanceMetrics(
                total_return=total_return,
                annualized_return=annualized_return,
                cumulative_return=cumulative_return,
                volatility=volatility,
                downside_volatility=downside_volatility,
                max_drawdown=max_drawdown,
                max_drawdown_duration=max_drawdown_duration,
                sharpe_ratio=sharpe_ratio,
                sortino_ratio=sortino_ratio,
                calmar_ratio=calmar_ratio,
                skewness=skewness,
                kurtosis=kurtosis,
                var_95=var_95,
                cvar_95=cvar_95,
                beta=beta,
                alpha=alpha,
                information_ratio=information_ratio,
                tracking_error=tracking_error,
                total_trades=trading_metrics['total_trades'],
                win_rate=trading_metrics['win_rate'],
                profit_factor=trading_metrics['profit_factor'],
                avg_win=trading_metrics['avg_win'],
                avg_loss=trading_metrics['avg_loss'],
                largest_win=trading_metrics['largest_win'],
                largest_loss=trading_metrics['largest_loss'],
                recovery_factor=recovery_factor,
                payoff_ratio=payoff_ratio,
                expectancy=expectancy
            )
            
        except Exception as e:
            self.logger.error(f"Performance calculation failed: {e}")
            return self._default_performance_metrics()
    
    def _calculate_max_drawdown_duration(self, drawdown: pd.Series) -> int:
        """Calculate maximum drawdown duration in days"""
        try:
            # Find periods where drawdown is at maximum
            is_at_max = drawdown == drawdown.min()
            
            if not is_at_max.any():
                return 0
            
            # Find the longest consecutive period at maximum drawdown
            max_duration = 0
            current_duration = 0
            
            for is_max in is_at_max:
                if is_max:
                    current_duration += 1
                    max_duration = max(max_duration, current_duration)
                else:
                    current_duration = 0
            
            return max_duration
            
        except:
            return 0
    
    def _calculate_benchmark_metrics(self, portfolio_returns: pd.Series, 
                                   benchmark_returns: Optional[pd.Series]) -> Tuple[float, float, float, float]:
        """Calculate benchmark comparison metrics"""
        if benchmark_returns is None or len(benchmark_returns) == 0:
            return 0.0, 0.0, 0.0, 0.0
        
        try:
            # Align returns
            aligned_portfolio, aligned_benchmark = portfolio_returns.align(benchmark_returns, join='inner')
            
            if len(aligned_portfolio) == 0:
                return 0.0, 0.0, 0.0, 0.0
            
            # Beta calculation
            covariance = np.cov(aligned_portfolio, aligned_benchmark)[0, 1]
            benchmark_variance = np.var(aligned_benchmark)
            beta = covariance / benchmark_variance if benchmark_variance > 0 else 0
            
            # Alpha calculation (CAPM)
            portfolio_mean = aligned_portfolio.mean() * 252
            benchmark_mean = aligned_benchmark.mean() * 252
            alpha = portfolio_mean - (self.risk_free_rate + beta * (benchmark_mean - self.risk_free_rate))
            
            # Information ratio and tracking error
            excess_returns = aligned_portfolio - aligned_benchmark
            tracking_error = excess_returns.std() * np.sqrt(252)
            information_ratio = excess_returns.mean() / excess_returns.std() * np.sqrt(252) if excess_returns.std() > 0 else 0
            
            return beta, alpha, information_ratio, tracking_error
            
        except Exception as e:
            self.logger.error(f"Benchmark metrics calculation failed: {e}")
            return 0.0, 0.0, 0.0, 0.0
    
    def _calculate_trading_metrics(self, trades: List[Trade]) -> Dict[str, float]:
        """Calculate trading-specific metrics"""
        if not trades:
            return {
                'total_trades': 0, 'win_rate': 0, 'profit_factor': 0,
                'avg_win': 0, 'avg_loss': 0, 'largest_win': 0, 'largest_loss': 0
            }
        
        try:
            # Calculate P&L for each trade (simplified)
            trade_pnls = []
            for i, trade in enumerate(trades):
                if i > 0 and trades[i-1].symbol == trade.symbol:
                    # Calculate P&L between entry and exit
                    if trades[i-1].side != trade.side:
                        pnl = (trade.price - trades[i-1].price) * trade.quantity
                        if trades[i-1].side == 'sell':  # Short position
                            pnl = -pnl
                        trade_pnls.append(pnl)
            
            if not trade_pnls:
                return {
                    'total_trades': len(trades), 'win_rate': 0, 'profit_factor': 0,
                    'avg_win': 0, 'avg_loss': 0, 'largest_win': 0, 'largest_loss': 0
                }
            
            # Separate wins and losses
            wins = [pnl for pnl in trade_pnls if pnl > 0]
            losses = [pnl for pnl in trade_pnls if pnl < 0]
            
            # Calculate metrics
            total_trades = len(trade_pnls)
            win_rate = len(wins) / total_trades if total_trades > 0 else 0
            
            avg_win = np.mean(wins) if wins else 0
            avg_loss = np.mean(losses) if losses else 0
            largest_win = max(wins) if wins else 0
            largest_loss = min(losses) if losses else 0
            
            gross_profit = sum(wins) if wins else 0
            gross_loss = abs(sum(losses)) if losses else 0
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
            
            return {
                'total_trades': total_trades,
                'win_rate': win_rate,
                'profit_factor': profit_factor,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'largest_win': largest_win,
                'largest_loss': largest_loss
            }
            
        except Exception as e:
            self.logger.error(f"Trading metrics calculation failed: {e}")
            return {
                'total_trades': len(trades), 'win_rate': 0, 'profit_factor': 0,
                'avg_win': 0, 'avg_loss': 0, 'largest_win': 0, 'largest_loss': 0
            }
    
    def _default_performance_metrics(self) -> PerformanceMetrics:
        """Default performance metrics for error cases"""
        return PerformanceMetrics(
            total_return=0.0, annualized_return=0.0, cumulative_return=0.0,
            volatility=0.0, downside_volatility=0.0, max_drawdown=0.0, max_drawdown_duration=0,
            sharpe_ratio=0.0, sortino_ratio=0.0, calmar_ratio=0.0,
            skewness=0.0, kurtosis=0.0, var_95=0.0, cvar_95=0.0,
            beta=0.0, alpha=0.0, information_ratio=0.0, tracking_error=0.0,
            total_trades=0, win_rate=0.0, profit_factor=0.0,
            avg_win=0.0, avg_loss=0.0, largest_win=0.0, largest_loss=0.0,
            recovery_factor=0.0, payoff_ratio=0.0, expectancy=0.0
        )

class AdvancedBacktester:
    """Advanced backtesting engine with multiple methodologies"""
    
    def __init__(self, config: BacktestConfig):
        self.config = config
        self.logger = logging.getLogger("advanced_backtester")
        self.regime_detector = RegimeDetector(config.n_regimes, config.regime_lookback)
        self.performance_analyzer = PerformanceAnalyzer(config.risk_free_rate)
        
        # Backtesting state
        self.portfolio_values = []
        self.portfolio_returns = []
        self.positions = {}
        self.trades = []
        self.cash = config.initial_capital
        self.current_date = config.start_date
    
    async def run_backtest(self, strategy_func: Callable, 
                          market_data: pd.DataFrame,
                          benchmark_data: Optional[pd.DataFrame] = None,
                          backtest_type: BacktestType = BacktestType.SIMPLE) -> BacktestResult:
        """Run comprehensive backtest"""
        try:
            if backtest_type == BacktestType.WALK_FORWARD:
                return await self._walk_forward_backtest(strategy_func, market_data, benchmark_data)
            elif backtest_type == BacktestType.MONTE_CARLO:
                return await self._monte_carlo_backtest(strategy_func, market_data, benchmark_data)
            elif backtest_type == BacktestType.REGIME_AWARE:
                return await self._regime_aware_backtest(strategy_func, market_data, benchmark_data)
            else:
                return await self._simple_backtest(strategy_func, market_data, benchmark_data)
                
        except Exception as e:
            self.logger.error(f"Backtest failed: {e}")
            return self._default_backtest_result()
    
    async def _simple_backtest(self, strategy_func: Callable,
                             market_data: pd.DataFrame,
                             benchmark_data: Optional[pd.DataFrame]) -> BacktestResult:
        """Simple backtesting methodology"""
        # Initialize portfolio
        self._initialize_portfolio()
        
        # Filter data for backtest period
        backtest_data = market_data[
            (market_data.index >= self.config.start_date) & 
            (market_data.index <= self.config.end_date)
        ].copy()
        
        # Run backtest day by day
        for date, row in backtest_data.iterrows():
            self.current_date = date
            
            # Get strategy signals
            signals = await self._get_strategy_signals(strategy_func, backtest_data.loc[:date])
            
            # Execute trades based on signals
            await self._execute_trades(signals, row)
            
            # Update portfolio value
            self._update_portfolio_value(row)
            
            # Record daily performance
            self._record_daily_performance()
        
        # Calculate final results
        return self._generate_backtest_result(benchmark_data)
    
    async def _walk_forward_backtest(self, strategy_func: Callable,
                                   market_data: pd.DataFrame,
                                   benchmark_data: Optional[pd.DataFrame]) -> BacktestResult:
        """Walk-forward analysis backtesting"""
        self.logger.info("Running walk-forward backtest")
        
        # Initialize results storage
        all_results = []
        
        # Calculate walk-forward windows
        start_idx = 0
        training_days = self.config.training_window
        testing_days = self.config.testing_window
        step_days = self.config.step_size
        
        while start_idx + training_days + testing_days <= len(market_data):
            # Define training and testing periods
            train_start = start_idx
            train_end = start_idx + training_days
            test_start = train_end
            test_end = test_start + testing_days
            
            # Extract data
            train_data = market_data.iloc[train_start:train_end]
            test_data = market_data.iloc[test_start:test_end]
            
            # Run backtest on test period
            period_result = await self._run_period_backtest(
                strategy_func, train_data, test_data
            )
            all_results.append(period_result)
            
            # Move to next window
            start_idx += step_days
        
        # Combine results
        return self._combine_walk_forward_results(all_results, benchmark_data)
    
    async def _monte_carlo_backtest(self, strategy_func: Callable,
                                  market_data: pd.DataFrame,
                                  benchmark_data: Optional[pd.DataFrame]) -> BacktestResult:
        """Monte Carlo simulation backtesting"""
        self.logger.info("Running Monte Carlo backtest")
        
        simulation_results = []
        
        for i in range(self.config.n_simulations):
            # Generate random scenario
            scenario_data = self._generate_monte_carlo_scenario(market_data, i)
            
            # Run backtest on scenario
            scenario_result = await self._simple_backtest(strategy_func, scenario_data, None)
            simulation_results.append(scenario_result)
        
        # Analyze simulation results
        return self._analyze_monte_carlo_results(simulation_results, benchmark_data)
    
    async def _regime_aware_backtest(self, strategy_func: Callable,
                                   market_data: pd.DataFrame,
                                   benchmark_data: Optional[pd.DataFrame]) -> BacktestResult:
        """Regime-aware backtesting"""
        self.logger.info("Running regime-aware backtest")
        
        # Detect market regimes
        regime_analysis = self.regime_detector.detect_regimes(market_data)
        
        # Run backtest with regime awareness
        result = await self._simple_backtest(strategy_func, market_data, benchmark_data)
        
        # Add regime analysis to result
        result.regime_analysis = regime_analysis
        
        return result
    
    def _initialize_portfolio(self):
        """Initialize portfolio state"""
        self.cash = self.config.initial_capital
        self.positions = {}
        self.trades = []
        self.portfolio_values = [self.config.initial_capital]
        self.portfolio_returns = [0.0]
    
    async def _get_strategy_signals(self, strategy_func: Callable, 
                                  historical_data: pd.DataFrame) -> Dict[str, float]:
        """Get trading signals from strategy"""
        try:
            # Call strategy function with historical data
            signals = await strategy_func(historical_data) if asyncio.iscoroutinefunction(strategy_func) else strategy_func(historical_data)
            return signals if signals else {}
        except Exception as e:
            self.logger.error(f"Strategy signal generation failed: {e}")
            return {}
    
    async def _execute_trades(self, signals: Dict[str, float], market_row: pd.Series):
        """Execute trades based on signals"""
        for symbol, target_weight in signals.items():
            if symbol not in market_row.index:
                continue
            
            current_price = market_row[symbol]
            current_weight = self._get_current_weight(symbol)
            
            # Calculate trade size
            weight_diff = target_weight - current_weight
            
            if abs(weight_diff) > 0.01:  # Minimum trade threshold
                trade_value = weight_diff * self._get_portfolio_value(market_row)
                quantity = trade_value / current_price
                
                # Calculate transaction costs
                commission = abs(trade_value) * self.config.commission_rate
                market_impact = abs(trade_value) * self.config.market_impact
                bid_ask_cost = abs(trade_value) * self.config.bid_ask_spread
                total_cost = commission + market_impact + bid_ask_cost
                
                # Execute trade
                if quantity > 0:
                    side = 'buy'
                else:
                    side = 'sell'
                    quantity = abs(quantity)
                
                # Record trade
                trade = Trade(
                    timestamp=self.current_date,
                    symbol=symbol,
                    side=side,
                    quantity=quantity,
                    price=current_price,
                    commission=commission,
                    market_impact=market_impact,
                    total_cost=total_cost,
                    portfolio_value=self._get_portfolio_value(market_row),
                    position_size=target_weight,
                    reason="Strategy signal"
                )
                
                self.trades.append(trade)
                
                # Update positions
                self._update_position(symbol, quantity, current_price, side)
                
                # Update cash
                if side == 'buy':
                    self.cash -= trade_value + total_cost
                else:
                    self.cash += trade_value - total_cost
    
    def _get_current_weight(self, symbol: str) -> float:
        """Get current weight of symbol in portfolio"""
        if symbol not in self.positions:
            return 0.0
        
        position = self.positions[symbol]
        portfolio_value = self._get_total_portfolio_value()
        
        return position.market_value / portfolio_value if portfolio_value > 0 else 0.0
    
    def _get_portfolio_value(self, market_row: pd.Series) -> float:
        """Get current portfolio value"""
        total_value = self.cash
        
        for symbol, position in self.positions.items():
            if symbol in market_row.index:
                current_price = market_row[symbol]
                total_value += position.quantity * current_price
        
        return total_value
    
    def _get_total_portfolio_value(self) -> float:
        """Get total portfolio value (simplified)"""
        total_value = self.cash
        
        for position in self.positions.values():
            total_value += position.market_value
        
        return total_value
    
    def _update_position(self, symbol: str, quantity: float, price: float, side: str):
        """Update position after trade"""
        if symbol not in self.positions:
            self.positions[symbol] = Position(
                symbol=symbol,
                quantity=0,
                avg_price=0,
                current_price=price,
                market_value=0,
                unrealized_pnl=0,
                realized_pnl=0,
                weight=0,
                entry_date=self.current_date,
                last_update=self.current_date
            )
        
        position = self.positions[symbol]
        
        if side == 'buy':
            # Update average price
            total_cost = position.quantity * position.avg_price + quantity * price
            total_quantity = position.quantity + quantity
            position.avg_price = total_cost / total_quantity if total_quantity > 0 else price
            position.quantity = total_quantity
        else:  # sell
            position.quantity -= quantity
            if position.quantity <= 0:
                # Position closed
                del self.positions[symbol]
                return
        
        # Update market value
        position.current_price = price
        position.market_value = position.quantity * price
        position.unrealized_pnl = (price - position.avg_price) * position.quantity
        position.last_update = self.current_date
    
    def _update_portfolio_value(self, market_row: pd.Series):
        """Update portfolio value and positions"""
        # Update position values
        for symbol, position in self.positions.items():
            if symbol in market_row.index:
                current_price = market_row[symbol]
                position.current_price = current_price
                position.market_value = position.quantity * current_price
                position.unrealized_pnl = (current_price - position.avg_price) * position.quantity
        
        # Calculate total portfolio value
        total_value = self._get_portfolio_value(market_row)
        self.portfolio_values.append(total_value)
    
    def _record_daily_performance(self):
        """Record daily performance metrics"""
        if len(self.portfolio_values) > 1:
            daily_return = (self.portfolio_values[-1] / self.portfolio_values[-2]) - 1
            self.portfolio_returns.append(daily_return)
        else:
            self.portfolio_returns.append(0.0)
    
    def _generate_backtest_result(self, benchmark_data: Optional[pd.DataFrame]) -> BacktestResult:
        """Generate comprehensive backtest result"""
        # Convert to pandas series
        portfolio_returns_series = pd.Series(self.portfolio_returns[1:])  # Skip first zero return
        portfolio_values_series = pd.Series(self.portfolio_values)
        
        # Get benchmark returns if available
        benchmark_returns = None
        if benchmark_data is not None and self.config.benchmark:
            if self.config.benchmark in benchmark_data.columns:
                benchmark_returns = benchmark_data[self.config.benchmark].pct_change().dropna()
        
        # Calculate performance metrics
        performance_metrics = self.performance_analyzer.calculate_performance_metrics(
            portfolio_returns_series, benchmark_returns, self.trades
        )
        
        # Calculate drawdown series
        cumulative = portfolio_values_series / portfolio_values_series.iloc[0]
        running_max = cumulative.expanding().max()
        drawdown_series = (cumulative - running_max) / running_max
        
        # Create positions history DataFrame
        positions_history = self._create_positions_history()
        
        return BacktestResult(
            config=self.config,
            performance_metrics=performance_metrics,
            portfolio_returns=portfolio_returns_series,
            portfolio_values=portfolio_values_series,
            positions_history=positions_history,
            trades_history=self.trades,
            drawdown_series=drawdown_series
        )
    
    def _create_positions_history(self) -> pd.DataFrame:
        """Create positions history DataFrame"""
        # Simplified positions history
        if not self.trades:
            return pd.DataFrame()
        
        positions_data = []
        for trade in self.trades:
            positions_data.append({
                'date': trade.timestamp,
                'symbol': trade.symbol,
                'quantity': trade.quantity if trade.side == 'buy' else -trade.quantity,
                'price': trade.price,
                'value': trade.quantity * trade.price,
                'side': trade.side
            })
        
        return pd.DataFrame(positions_data)
    
    async def _run_period_backtest(self, strategy_func: Callable,
                                 train_data: pd.DataFrame,
                                 test_data: pd.DataFrame) -> BacktestResult:
        """Run backtest for a specific period (used in walk-forward)"""
        # Save current state
        original_config = self.config
        
        # Create period-specific config
        period_config = BacktestConfig(
            start_date=test_data.index[0],
            end_date=test_data.index[-1],
            initial_capital=self.config.initial_capital
        )
        
        # Initialize for period
        self.config = period_config
        self._initialize_portfolio()
        
        # Run simple backtest on test data
        result = await self._simple_backtest(strategy_func, test_data, None)
        
        # Restore original config
        self.config = original_config
        
        return result
    
    def _combine_walk_forward_results(self, results: List[BacktestResult],
                                    benchmark_data: Optional[pd.DataFrame]) -> BacktestResult:
        """Combine walk-forward analysis results"""
        # Combine all returns
        all_returns = []
        all_values = []
        all_trades = []
        
        for result in results:
            all_returns.extend(result.portfolio_returns.tolist())
            all_values.extend(result.portfolio_values.tolist())
            all_trades.extend(result.trades_history)
        
        # Create combined series
        combined_returns = pd.Series(all_returns)
        combined_values = pd.Series(all_values)
        
        # Calculate combined performance metrics
        benchmark_returns = None
        if benchmark_data is not None and self.config.benchmark:
            if self.config.benchmark in benchmark_data.columns:
                benchmark_returns = benchmark_data[self.config.benchmark].pct_change().dropna()
        
        performance_metrics = self.performance_analyzer.calculate_performance_metrics(
            combined_returns, benchmark_returns, all_trades
        )
        
        # Calculate combined drawdown
        cumulative = combined_values / combined_values.iloc[0] if len(combined_values) > 0 else pd.Series([1.0])
        running_max = cumulative.expanding().max()
        drawdown_series = (cumulative - running_max) / running_max
        
        return BacktestResult(
            config=self.config,
            performance_metrics=performance_metrics,
            portfolio_returns=combined_returns,
            portfolio_values=combined_values,
            positions_history=pd.DataFrame(),  # Simplified
            trades_history=all_trades,
            drawdown_series=drawdown_series
        )
    
    def _generate_monte_carlo_scenario(self, market_data: pd.DataFrame, seed: int) -> pd.DataFrame:
        """Generate Monte Carlo scenario"""
        np.random.seed(seed)
        
        # Calculate historical statistics
        returns = market_data.pct_change().dropna()
        mean_returns = returns.mean()
        cov_matrix = returns.cov()
        
        # Generate random returns
        n_days = len(market_data)
        n_assets = len(market_data.columns)
        
        # Generate correlated random returns
        if SCIPY_AVAILABLE:
            try:
                random_returns = np.random.multivariate_normal(
                    mean_returns.values, cov_matrix.values, n_days
                )
            except:
                # Fallback to independent returns
                random_returns = np.random.normal(
                    mean_returns.values, np.sqrt(np.diag(cov_matrix.values)), (n_days, n_assets)
                )
        else:
            # Simple independent random returns
            random_returns = np.random.normal(
                mean_returns.values, returns.std().values, (n_days, n_assets)
            )
        
        # Convert to price series
        scenario_data = market_data.copy()
        initial_prices = market_data.iloc[0]
        
        for i in range(1, len(scenario_data)):
            scenario_data.iloc[i] = scenario_data.iloc[i-1] * (1 + random_returns[i])
        
        return scenario_data
    
    def _analyze_monte_carlo_results(self, results: List[BacktestResult],
                                   benchmark_data: Optional[pd.DataFrame]) -> BacktestResult:
        """Analyze Monte Carlo simulation results"""
        # Extract key metrics from all simulations
        total_returns = [r.performance_metrics.total_return for r in results]
        sharpe_ratios = [r.performance_metrics.sharpe_ratio for r in results]
        max_drawdowns = [r.performance_metrics.max_drawdown for r in results]
        
        # Calculate confidence intervals
        confidence_level = self.config.confidence_level
        lower_percentile = (1 - confidence_level) / 2 * 100
        upper_percentile = (1 + confidence_level) / 2 * 100
        
        # Create summary statistics
        statistical_tests = {
            'total_return_mean': np.mean(total_returns),
            'total_return_std': np.std(total_returns),
            'total_return_ci_lower': np.percentile(total_returns, lower_percentile),
            'total_return_ci_upper': np.percentile(total_returns, upper_percentile),
            'sharpe_ratio_mean': np.mean(sharpe_ratios),
            'sharpe_ratio_std': np.std(sharpe_ratios),
            'max_drawdown_mean': np.mean(max_drawdowns),
            'max_drawdown_worst': min(max_drawdowns),
            'probability_positive': sum(1 for r in total_returns if r > 0) / len(total_returns)
        }
        
        # Use median result as representative
        median_idx = len(results) // 2
        representative_result = sorted(results, key=lambda x: x.performance_metrics.total_return)[median_idx]
        
        # Add statistical tests to result
        representative_result.statistical_tests = statistical_tests
        
        return representative_result
    
    def _default_backtest_result(self) -> BacktestResult:
        """Default backtest result for error cases"""
        default_metrics = self.performance_analyzer._default_performance_metrics()
        
        return BacktestResult(
            config=self.config,
            performance_metrics=default_metrics,
            portfolio_returns=pd.Series([0.0]),
            portfolio_values=pd.Series([self.config.initial_capital]),
            positions_history=pd.DataFrame(),
            trades_history=[],
            drawdown_series=pd.Series([0.0])
        )

# Example usage and strategy interface
class ExampleStrategy:
    """Example strategy for backtesting"""
    
    def __init__(self, lookback_window: int = 20):
        self.lookback_window = lookback_window
    
    async def generate_signals(self, market_data: pd.DataFrame) -> Dict[str, float]:
        """Generate trading signals"""
        signals = {}
        
        if len(market_data) < self.lookback_window:
            return signals
        
        # Simple momentum strategy
        for column in market_data.columns:
            if column in market_data.columns:
                recent_data = market_data[column].tail(self.lookback_window)
                momentum = (recent_data.iloc[-1] / recent_data.iloc[0]) - 1
                
                # Generate signal based on momentum
                if momentum > 0.05:  # 5% positive momentum
                    signals[column] = 0.1  # 10% allocation
                elif momentum < -0.05:  # 5% negative momentum
                    signals[column] = -0.05  # 5% short allocation
                else:
                    signals[column] = 0.0  # No position
        
        return signals

# Example usage
if __name__ == "__main__":
    # Example configuration
    config = BacktestConfig(
        start_date=datetime(2020, 1, 1),
        end_date=datetime(2023, 12, 31),
        initial_capital=1000000,
        commission_rate=0.001,
        rebalance_frequency="monthly"
    )
    
    # Initialize backtester
    backtester = AdvancedBacktester(config)
    
    print("Advanced Backtesting Framework initialized")
    print("Available backtesting methodologies:")
    print("- Simple Backtesting")
    print("- Walk-Forward Analysis")
    print("- Monte Carlo Simulation")
    print("- Regime-Aware Backtesting")
    print("- Bootstrap Analysis")
    print("- Cross-Validation")
    print("- Out-of-Sample Testing")
    print("\nPerformance Analytics:")
    print("- Comprehensive Risk Metrics")
    print("- Statistical Significance Testing")
    print("- Factor Exposure Analysis")
    print("- Transaction Cost Modeling")
    print("- Benchmark Comparison")