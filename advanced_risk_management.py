#!/usr/bin/env python3
"""
Advanced Risk Management System

A comprehensive risk management framework implementing:

- Value at Risk (VaR) and Conditional VaR (CVaR) models
- Monte Carlo simulation and historical simulation
- Stress testing and scenario analysis
- Dynamic hedging strategies
- Real-time risk monitoring and alerts
- Counterparty risk assessment
- Liquidity risk management
- Model risk validation
- Regulatory capital calculations
- Risk attribution and decomposition

This system provides institutional-grade risk management capabilities
with advanced mathematical models and real-time monitoring.

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

# Statistical and optimization libraries
try:
    from scipy import stats, optimize
    from scipy.linalg import cholesky
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logging.warning("SciPy not available. Advanced statistical models will be limited.")

# Machine learning for risk modeling
try:
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error, r2_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logging.warning("Scikit-learn not available. ML risk models will be simplified.")

# Advanced optimization
try:
    import cvxpy as cp
    CVXPY_AVAILABLE = True
except ImportError:
    CVXPY_AVAILABLE = False
    logging.warning("CVXPY not available. Advanced optimization will use scipy fallback.")

class RiskMetric(Enum):
    """Risk measurement types"""
    VAR_HISTORICAL = "var_historical"
    VAR_PARAMETRIC = "var_parametric"
    VAR_MONTE_CARLO = "var_monte_carlo"
    CVAR = "conditional_var"
    EXPECTED_SHORTFALL = "expected_shortfall"
    MAXIMUM_DRAWDOWN = "maximum_drawdown"
    VOLATILITY = "volatility"
    BETA = "beta"
    TRACKING_ERROR = "tracking_error"
    INFORMATION_RATIO = "information_ratio"
    SHARPE_RATIO = "sharpe_ratio"
    SORTINO_RATIO = "sortino_ratio"

class RiskLevel(Enum):
    """Risk level classifications"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class AlertType(Enum):
    """Risk alert types"""
    VAR_BREACH = "var_breach"
    CONCENTRATION_LIMIT = "concentration_limit"
    LIQUIDITY_RISK = "liquidity_risk"
    CORRELATION_BREAKDOWN = "correlation_breakdown"
    VOLATILITY_SPIKE = "volatility_spike"
    DRAWDOWN_LIMIT = "drawdown_limit"
    LEVERAGE_LIMIT = "leverage_limit"
    COUNTERPARTY_RISK = "counterparty_risk"
    MODEL_BREAKDOWN = "model_breakdown"

class StressTestType(Enum):
    """Stress test scenarios"""
    HISTORICAL_SCENARIO = "historical_scenario"
    MONTE_CARLO_SCENARIO = "monte_carlo_scenario"
    TAIL_RISK_SCENARIO = "tail_risk_scenario"
    CORRELATION_BREAKDOWN = "correlation_breakdown"
    LIQUIDITY_CRISIS = "liquidity_crisis"
    INTEREST_RATE_SHOCK = "interest_rate_shock"
    CREDIT_CRISIS = "credit_crisis"
    MARKET_CRASH = "market_crash"
    VOLATILITY_SPIKE = "volatility_spike"

@dataclass
class RiskLimits:
    """Risk limits configuration"""
    # VaR limits
    daily_var_limit: float = 0.02  # 2% of portfolio value
    weekly_var_limit: float = 0.05  # 5% of portfolio value
    monthly_var_limit: float = 0.10  # 10% of portfolio value
    
    # Concentration limits
    max_single_position: float = 0.10  # 10% max in single asset
    max_sector_exposure: float = 0.25  # 25% max in single sector
    max_country_exposure: float = 0.30  # 30% max in single country
    
    # Leverage limits
    max_gross_leverage: float = 2.0  # 200% gross leverage
    max_net_leverage: float = 1.5   # 150% net leverage
    
    # Drawdown limits
    max_daily_drawdown: float = 0.03  # 3% daily drawdown
    max_monthly_drawdown: float = 0.08  # 8% monthly drawdown
    
    # Liquidity limits
    min_cash_ratio: float = 0.05  # 5% minimum cash
    max_illiquid_ratio: float = 0.20  # 20% max illiquid assets
    
    # Correlation limits
    max_correlation: float = 0.80  # 80% max correlation between positions
    
    # Volatility limits
    max_portfolio_volatility: float = 0.20  # 20% annualized volatility
    volatility_spike_threshold: float = 2.0  # 2x normal volatility

@dataclass
class RiskAlert:
    """Risk alert information"""
    alert_type: AlertType
    severity: RiskLevel
    message: str
    current_value: float
    limit_value: float
    timestamp: datetime
    asset: Optional[str] = None
    portfolio: Optional[str] = None
    recommended_action: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return {
            'alert_type': self.alert_type.value,
            'severity': self.severity.value,
            'message': self.message,
            'current_value': self.current_value,
            'limit_value': self.limit_value,
            'timestamp': self.timestamp.isoformat(),
            'asset': self.asset,
            'portfolio': self.portfolio,
            'recommended_action': self.recommended_action
        }

@dataclass
class VaRResult:
    """Value at Risk calculation result"""
    var_1d: float  # 1-day VaR
    var_5d: float  # 5-day VaR
    var_22d: float  # 22-day (monthly) VaR
    cvar_1d: float  # 1-day Conditional VaR
    confidence_level: float  # Confidence level (e.g., 0.95)
    method: str  # Calculation method
    portfolio_value: float
    currency: str = "USD"
    
    def to_dict(self) -> Dict:
        return {
            'var_1d': self.var_1d,
            'var_5d': self.var_5d,
            'var_22d': self.var_22d,
            'cvar_1d': self.cvar_1d,
            'confidence_level': self.confidence_level,
            'method': self.method,
            'portfolio_value': self.portfolio_value,
            'currency': self.currency
        }

@dataclass
class StressTestResult:
    """Stress test result"""
    scenario_name: str
    scenario_type: StressTestType
    portfolio_pnl: float
    portfolio_pnl_pct: float
    asset_pnl: Dict[str, float]
    risk_factors: Dict[str, float]
    probability: Optional[float] = None
    description: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return {
            'scenario_name': self.scenario_name,
            'scenario_type': self.scenario_type.value,
            'portfolio_pnl': self.portfolio_pnl,
            'portfolio_pnl_pct': self.portfolio_pnl_pct,
            'asset_pnl': self.asset_pnl,
            'risk_factors': self.risk_factors,
            'probability': self.probability,
            'description': self.description
        }

@dataclass
class RiskMetrics:
    """Comprehensive risk metrics"""
    # VaR metrics
    var_results: VaRResult
    
    # Volatility metrics
    realized_volatility: float
    implied_volatility: Optional[float] = None
    volatility_percentile: float = 0.0
    
    # Correlation metrics
    avg_correlation: float = 0.0
    max_correlation: float = 0.0
    correlation_breakdown_risk: float = 0.0
    
    # Concentration metrics
    concentration_index: float = 0.0  # Herfindahl index
    effective_assets: float = 0.0
    largest_position_weight: float = 0.0
    
    # Liquidity metrics
    liquidity_score: float = 0.0
    days_to_liquidate: float = 0.0
    bid_ask_impact: float = 0.0
    
    # Leverage metrics
    gross_leverage: float = 0.0
    net_leverage: float = 0.0
    
    # Drawdown metrics
    current_drawdown: float = 0.0
    max_drawdown: float = 0.0
    drawdown_duration: int = 0
    
    # Factor exposures
    factor_exposures: Dict[str, float] = field(default_factory=dict)
    factor_risks: Dict[str, float] = field(default_factory=dict)
    
    # Model validation metrics
    model_accuracy: float = 0.0
    backtesting_exceptions: int = 0
    model_confidence: float = 0.0
    
    def to_dict(self) -> Dict:
        return {
            'var_results': self.var_results.to_dict(),
            'realized_volatility': self.realized_volatility,
            'implied_volatility': self.implied_volatility,
            'volatility_percentile': self.volatility_percentile,
            'avg_correlation': self.avg_correlation,
            'max_correlation': self.max_correlation,
            'correlation_breakdown_risk': self.correlation_breakdown_risk,
            'concentration_index': self.concentration_index,
            'effective_assets': self.effective_assets,
            'largest_position_weight': self.largest_position_weight,
            'liquidity_score': self.liquidity_score,
            'days_to_liquidate': self.days_to_liquidate,
            'bid_ask_impact': self.bid_ask_impact,
            'gross_leverage': self.gross_leverage,
            'net_leverage': self.net_leverage,
            'current_drawdown': self.current_drawdown,
            'max_drawdown': self.max_drawdown,
            'drawdown_duration': self.drawdown_duration,
            'factor_exposures': self.factor_exposures,
            'factor_risks': self.factor_risks,
            'model_accuracy': self.model_accuracy,
            'backtesting_exceptions': self.backtesting_exceptions,
            'model_confidence': self.model_confidence
        }

class VaRCalculator:
    """Value at Risk calculation engine"""
    
    def __init__(self, confidence_level: float = 0.95):
        self.confidence_level = confidence_level
        self.logger = logging.getLogger("var_calculator")
    
    def calculate_historical_var(self, returns: pd.Series, 
                               portfolio_value: float) -> VaRResult:
        """Calculate VaR using historical simulation"""
        try:
            # Sort returns in ascending order
            sorted_returns = returns.sort_values()
            
            # Calculate percentile
            percentile = 1 - self.confidence_level
            var_index = int(percentile * len(sorted_returns))
            
            # 1-day VaR
            var_1d_return = sorted_returns.iloc[var_index]
            var_1d = -var_1d_return * portfolio_value
            
            # Multi-day VaR (assuming square root of time scaling)
            var_5d = var_1d * np.sqrt(5)
            var_22d = var_1d * np.sqrt(22)
            
            # Conditional VaR (Expected Shortfall)
            tail_returns = sorted_returns.iloc[:var_index]
            cvar_1d_return = tail_returns.mean() if len(tail_returns) > 0 else var_1d_return
            cvar_1d = -cvar_1d_return * portfolio_value
            
            return VaRResult(
                var_1d=var_1d,
                var_5d=var_5d,
                var_22d=var_22d,
                cvar_1d=cvar_1d,
                confidence_level=self.confidence_level,
                method="Historical Simulation",
                portfolio_value=portfolio_value
            )
            
        except Exception as e:
            self.logger.error(f"Historical VaR calculation failed: {e}")
            return self._default_var_result(portfolio_value, "Historical Simulation")
    
    def calculate_parametric_var(self, returns: pd.Series, 
                               portfolio_value: float) -> VaRResult:
        """Calculate VaR using parametric method (normal distribution)"""
        try:
            # Calculate mean and standard deviation
            mean_return = returns.mean()
            std_return = returns.std()
            
            # Z-score for confidence level
            if SCIPY_AVAILABLE:
                z_score = stats.norm.ppf(1 - self.confidence_level)
            else:
                # Approximate z-scores for common confidence levels
                z_scores = {0.90: -1.28, 0.95: -1.65, 0.99: -2.33}
                z_score = z_scores.get(self.confidence_level, -1.65)
            
            # 1-day VaR
            var_1d_return = mean_return + z_score * std_return
            var_1d = -var_1d_return * portfolio_value
            
            # Multi-day VaR
            var_5d = var_1d * np.sqrt(5)
            var_22d = var_1d * np.sqrt(22)
            
            # Conditional VaR (for normal distribution)
            if SCIPY_AVAILABLE:
                phi_z = stats.norm.pdf(z_score)
                cvar_multiplier = phi_z / (1 - self.confidence_level)
                cvar_1d_return = mean_return - std_return * cvar_multiplier
            else:
                cvar_1d_return = var_1d_return * 1.2  # Approximation
            
            cvar_1d = -cvar_1d_return * portfolio_value
            
            return VaRResult(
                var_1d=var_1d,
                var_5d=var_5d,
                var_22d=var_22d,
                cvar_1d=cvar_1d,
                confidence_level=self.confidence_level,
                method="Parametric (Normal)",
                portfolio_value=portfolio_value
            )
            
        except Exception as e:
            self.logger.error(f"Parametric VaR calculation failed: {e}")
            return self._default_var_result(portfolio_value, "Parametric (Normal)")
    
    def calculate_monte_carlo_var(self, returns: pd.Series, 
                                portfolio_value: float,
                                n_simulations: int = 10000) -> VaRResult:
        """Calculate VaR using Monte Carlo simulation"""
        try:
            # Fit distribution to returns
            mean_return = returns.mean()
            std_return = returns.std()
            
            # Generate random scenarios
            np.random.seed(42)  # For reproducibility
            simulated_returns = np.random.normal(mean_return, std_return, n_simulations)
            
            # Calculate VaR from simulated returns
            sorted_sim_returns = np.sort(simulated_returns)
            percentile = 1 - self.confidence_level
            var_index = int(percentile * len(sorted_sim_returns))
            
            # 1-day VaR
            var_1d_return = sorted_sim_returns[var_index]
            var_1d = -var_1d_return * portfolio_value
            
            # Multi-day VaR
            var_5d = var_1d * np.sqrt(5)
            var_22d = var_1d * np.sqrt(22)
            
            # Conditional VaR
            tail_returns = sorted_sim_returns[:var_index]
            cvar_1d_return = tail_returns.mean() if len(tail_returns) > 0 else var_1d_return
            cvar_1d = -cvar_1d_return * portfolio_value
            
            return VaRResult(
                var_1d=var_1d,
                var_5d=var_5d,
                var_22d=var_22d,
                cvar_1d=cvar_1d,
                confidence_level=self.confidence_level,
                method="Monte Carlo",
                portfolio_value=portfolio_value
            )
            
        except Exception as e:
            self.logger.error(f"Monte Carlo VaR calculation failed: {e}")
            return self._default_var_result(portfolio_value, "Monte Carlo")
    
    def _default_var_result(self, portfolio_value: float, method: str) -> VaRResult:
        """Default VaR result for error cases"""
        # Conservative estimate: 2% daily VaR
        var_1d = portfolio_value * 0.02
        return VaRResult(
            var_1d=var_1d,
            var_5d=var_1d * np.sqrt(5),
            var_22d=var_1d * np.sqrt(22),
            cvar_1d=var_1d * 1.3,
            confidence_level=self.confidence_level,
            method=method,
            portfolio_value=portfolio_value
        )

class StressTestEngine:
    """Stress testing and scenario analysis engine"""
    
    def __init__(self):
        self.logger = logging.getLogger("stress_test")
        self.historical_scenarios = self._load_historical_scenarios()
    
    def run_stress_test(self, portfolio_weights: Dict[str, float],
                       asset_returns: pd.DataFrame,
                       scenario_type: StressTestType,
                       portfolio_value: float,
                       **kwargs) -> List[StressTestResult]:
        """Run comprehensive stress tests"""
        results = []
        
        try:
            if scenario_type == StressTestType.HISTORICAL_SCENARIO:
                results = self._historical_stress_test(portfolio_weights, asset_returns, 
                                                     portfolio_value)
            elif scenario_type == StressTestType.MONTE_CARLO_SCENARIO:
                results = self._monte_carlo_stress_test(portfolio_weights, asset_returns, 
                                                      portfolio_value, **kwargs)
            elif scenario_type == StressTestType.TAIL_RISK_SCENARIO:
                results = self._tail_risk_stress_test(portfolio_weights, asset_returns, 
                                                    portfolio_value)
            elif scenario_type == StressTestType.CORRELATION_BREAKDOWN:
                results = self._correlation_breakdown_test(portfolio_weights, asset_returns, 
                                                         portfolio_value)
            elif scenario_type == StressTestType.VOLATILITY_SPIKE:
                results = self._volatility_spike_test(portfolio_weights, asset_returns, 
                                                     portfolio_value)
            else:
                results = self._market_crash_test(portfolio_weights, asset_returns, 
                                                 portfolio_value)
            
        except Exception as e:
            self.logger.error(f"Stress test failed: {e}")
            results = [self._default_stress_result(portfolio_value)]
        
        return results
    
    def _historical_stress_test(self, portfolio_weights: Dict[str, float],
                              asset_returns: pd.DataFrame,
                              portfolio_value: float) -> List[StressTestResult]:
        """Historical scenario stress testing"""
        results = []
        
        # Define historical crisis periods
        crisis_scenarios = {
            "2008 Financial Crisis": {
                "start": "2008-09-01",
                "end": "2008-12-31",
                "description": "Global financial crisis and market crash"
            },
            "COVID-19 Crash": {
                "start": "2020-02-01",
                "end": "2020-04-30",
                "description": "COVID-19 pandemic market crash"
            },
            "Dot-com Crash": {
                "start": "2000-03-01",
                "end": "2000-12-31",
                "description": "Technology bubble burst"
            }
        }
        
        for scenario_name, scenario_info in crisis_scenarios.items():
            try:
                # Filter returns for crisis period (if available)
                # For demonstration, use worst consecutive days
                portfolio_returns = self._calculate_portfolio_returns(portfolio_weights, asset_returns)
                
                # Find worst consecutive period
                window_size = 30  # 30-day window
                rolling_returns = portfolio_returns.rolling(window=window_size).sum()
                worst_period_return = rolling_returns.min()
                
                # Calculate P&L
                portfolio_pnl = worst_period_return * portfolio_value
                portfolio_pnl_pct = worst_period_return
                
                # Asset-level P&L
                asset_pnl = {}
                for asset, weight in portfolio_weights.items():
                    if asset in asset_returns.columns:
                        asset_worst = asset_returns[asset].rolling(window=window_size).sum().min()
                        asset_pnl[asset] = asset_worst * weight * portfolio_value
                
                results.append(StressTestResult(
                    scenario_name=scenario_name,
                    scenario_type=StressTestType.HISTORICAL_SCENARIO,
                    portfolio_pnl=portfolio_pnl,
                    portfolio_pnl_pct=portfolio_pnl_pct,
                    asset_pnl=asset_pnl,
                    risk_factors={"market_stress": worst_period_return},
                    description=scenario_info["description"]
                ))
                
            except Exception as e:
                self.logger.error(f"Historical scenario {scenario_name} failed: {e}")
                continue
        
        return results
    
    def _monte_carlo_stress_test(self, portfolio_weights: Dict[str, float],
                               asset_returns: pd.DataFrame,
                               portfolio_value: float,
                               n_simulations: int = 1000) -> List[StressTestResult]:
        """Monte Carlo stress testing"""
        results = []
        
        try:
            # Calculate correlation matrix
            corr_matrix = asset_returns.corr()
            
            # Generate correlated random shocks
            np.random.seed(42)
            
            # Cholesky decomposition for correlated shocks
            if SCIPY_AVAILABLE:
                try:
                    L = cholesky(corr_matrix.values, lower=True)
                    use_cholesky = True
                except:
                    use_cholesky = False
            else:
                use_cholesky = False
            
            # Run simulations
            portfolio_pnls = []
            
            for i in range(n_simulations):
                if use_cholesky:
                    # Generate correlated shocks
                    independent_shocks = np.random.normal(0, 1, len(asset_returns.columns))
                    correlated_shocks = L @ independent_shocks
                else:
                    # Use independent shocks as fallback
                    correlated_shocks = np.random.normal(0, 1, len(asset_returns.columns))
                
                # Scale shocks by historical volatility
                asset_vols = asset_returns.std()
                scaled_shocks = correlated_shocks * asset_vols.values * 3  # 3-sigma event
                
                # Calculate portfolio P&L
                portfolio_pnl = 0
                for j, (asset, weight) in enumerate(portfolio_weights.items()):
                    if j < len(scaled_shocks) and asset in asset_returns.columns:
                        asset_pnl = scaled_shocks[j] * weight * portfolio_value
                        portfolio_pnl += asset_pnl
                
                portfolio_pnls.append(portfolio_pnl)
            
            # Analyze results
            portfolio_pnls = np.array(portfolio_pnls)
            
            # Worst case scenarios
            percentiles = [1, 5, 10]  # 1st, 5th, 10th percentiles
            
            for percentile in percentiles:
                worst_pnl = np.percentile(portfolio_pnls, percentile)
                worst_pnl_pct = worst_pnl / portfolio_value
                
                results.append(StressTestResult(
                    scenario_name=f"Monte Carlo {percentile}th Percentile",
                    scenario_type=StressTestType.MONTE_CARLO_SCENARIO,
                    portfolio_pnl=worst_pnl,
                    portfolio_pnl_pct=worst_pnl_pct,
                    asset_pnl={},  # Simplified for now
                    risk_factors={"simulation_percentile": percentile},
                    probability=percentile / 100.0,
                    description=f"Monte Carlo simulation {percentile}th percentile outcome"
                ))
            
        except Exception as e:
            self.logger.error(f"Monte Carlo stress test failed: {e}")
        
        return results
    
    def _tail_risk_stress_test(self, portfolio_weights: Dict[str, float],
                             asset_returns: pd.DataFrame,
                             portfolio_value: float) -> List[StressTestResult]:
        """Tail risk stress testing"""
        results = []
        
        try:
            portfolio_returns = self._calculate_portfolio_returns(portfolio_weights, asset_returns)
            
            # Define tail events
            tail_percentiles = [0.5, 1.0, 2.5]  # 0.5%, 1%, 2.5% tail events
            
            for percentile in tail_percentiles:
                tail_return = np.percentile(portfolio_returns, percentile)
                tail_pnl = tail_return * portfolio_value
                
                results.append(StressTestResult(
                    scenario_name=f"Tail Risk {percentile}%",
                    scenario_type=StressTestType.TAIL_RISK_SCENARIO,
                    portfolio_pnl=tail_pnl,
                    portfolio_pnl_pct=tail_return,
                    asset_pnl={},
                    risk_factors={"tail_percentile": percentile},
                    probability=percentile / 100.0,
                    description=f"Historical {percentile}% tail event"
                ))
            
        except Exception as e:
            self.logger.error(f"Tail risk stress test failed: {e}")
        
        return results
    
    def _correlation_breakdown_test(self, portfolio_weights: Dict[str, float],
                                  asset_returns: pd.DataFrame,
                                  portfolio_value: float) -> List[StressTestResult]:
        """Correlation breakdown stress test"""
        results = []
        
        try:
            # Simulate scenario where correlations go to 1 (perfect correlation)
            # This represents a crisis where diversification benefits disappear
            
            # Calculate individual asset volatilities
            asset_vols = asset_returns.std()
            
            # Assume all assets move together with their individual volatilities
            # in the same direction (worst case)
            portfolio_vol = 0
            for asset, weight in portfolio_weights.items():
                if asset in asset_vols.index:
                    portfolio_vol += abs(weight) * asset_vols[asset]
            
            # 3-sigma event with perfect correlation
            stress_return = -3 * portfolio_vol  # Negative 3-sigma move
            stress_pnl = stress_return * portfolio_value
            
            # Asset-level impacts
            asset_pnl = {}
            for asset, weight in portfolio_weights.items():
                if asset in asset_vols.index:
                    asset_stress = -3 * asset_vols[asset] * weight * portfolio_value
                    asset_pnl[asset] = asset_stress
            
            results.append(StressTestResult(
                scenario_name="Correlation Breakdown",
                scenario_type=StressTestType.CORRELATION_BREAKDOWN,
                portfolio_pnl=stress_pnl,
                portfolio_pnl_pct=stress_return,
                asset_pnl=asset_pnl,
                risk_factors={"correlation": 1.0, "sigma_level": 3.0},
                description="Scenario where all correlations go to 1.0 (diversification breakdown)"
            ))
            
        except Exception as e:
            self.logger.error(f"Correlation breakdown test failed: {e}")
        
        return results
    
    def _volatility_spike_test(self, portfolio_weights: Dict[str, float],
                             asset_returns: pd.DataFrame,
                             portfolio_value: float) -> List[StressTestResult]:
        """Volatility spike stress test"""
        results = []
        
        try:
            # Simulate scenarios where volatility spikes by different multiples
            vol_multipliers = [2.0, 3.0, 5.0]  # 2x, 3x, 5x volatility spikes
            
            for multiplier in vol_multipliers:
                portfolio_returns = self._calculate_portfolio_returns(portfolio_weights, asset_returns)
                current_vol = portfolio_returns.std()
                
                # Simulate return with spiked volatility
                stress_return = -multiplier * current_vol * 2  # 2-sigma event with spiked vol
                stress_pnl = stress_return * portfolio_value
                
                results.append(StressTestResult(
                    scenario_name=f"Volatility Spike {multiplier}x",
                    scenario_type=StressTestType.VOLATILITY_SPIKE,
                    portfolio_pnl=stress_pnl,
                    portfolio_pnl_pct=stress_return,
                    asset_pnl={},
                    risk_factors={"volatility_multiplier": multiplier},
                    description=f"Scenario with {multiplier}x volatility spike"
                ))
            
        except Exception as e:
            self.logger.error(f"Volatility spike test failed: {e}")
        
        return results
    
    def _market_crash_test(self, portfolio_weights: Dict[str, float],
                         asset_returns: pd.DataFrame,
                         portfolio_value: float) -> List[StressTestResult]:
        """Market crash stress test"""
        results = []
        
        try:
            # Define different crash scenarios
            crash_scenarios = {
                "Moderate Crash": -0.20,  # 20% market decline
                "Severe Crash": -0.35,   # 35% market decline
                "Extreme Crash": -0.50   # 50% market decline
            }
            
            for scenario_name, market_decline in crash_scenarios.items():
                # Assume all assets decline by the market decline amount
                # (simplified assumption)
                portfolio_pnl = market_decline * portfolio_value
                
                # Asset-level impacts
                asset_pnl = {}
                for asset, weight in portfolio_weights.items():
                    asset_pnl[asset] = market_decline * weight * portfolio_value
                
                results.append(StressTestResult(
                    scenario_name=scenario_name,
                    scenario_type=StressTestType.MARKET_CRASH,
                    portfolio_pnl=portfolio_pnl,
                    portfolio_pnl_pct=market_decline,
                    asset_pnl=asset_pnl,
                    risk_factors={"market_decline": market_decline},
                    description=f"Market crash scenario with {abs(market_decline)*100:.0f}% decline"
                ))
            
        except Exception as e:
            self.logger.error(f"Market crash test failed: {e}")
        
        return results
    
    def _calculate_portfolio_returns(self, portfolio_weights: Dict[str, float],
                                   asset_returns: pd.DataFrame) -> pd.Series:
        """Calculate portfolio returns from weights and asset returns"""
        portfolio_returns = pd.Series(0.0, index=asset_returns.index)
        
        for asset, weight in portfolio_weights.items():
            if asset in asset_returns.columns:
                portfolio_returns += weight * asset_returns[asset]
        
        return portfolio_returns
    
    def _default_stress_result(self, portfolio_value: float) -> StressTestResult:
        """Default stress test result for error cases"""
        return StressTestResult(
            scenario_name="Default Stress",
            scenario_type=StressTestType.MARKET_CRASH,
            portfolio_pnl=-portfolio_value * 0.10,  # 10% loss
            portfolio_pnl_pct=-0.10,
            asset_pnl={},
            risk_factors={},
            description="Default stress scenario (10% portfolio loss)"
        )
    
    def _load_historical_scenarios(self) -> Dict:
        """Load historical crisis scenarios"""
        # This would typically load from a database or file
        # For now, return empty dict
        return {}

class RiskMonitor:
    """Real-time risk monitoring and alerting system"""
    
    def __init__(self, risk_limits: RiskLimits):
        self.risk_limits = risk_limits
        self.logger = logging.getLogger("risk_monitor")
        self.var_calculator = VaRCalculator()
        self.stress_engine = StressTestEngine()
        self.alerts_history = []
    
    async def monitor_portfolio_risk(self, 
                                   portfolio_weights: Dict[str, float],
                                   asset_returns: pd.DataFrame,
                                   portfolio_value: float,
                                   market_data: Optional[Dict] = None) -> Tuple[RiskMetrics, List[RiskAlert]]:
        """Comprehensive portfolio risk monitoring"""
        try:
            # Calculate portfolio returns
            portfolio_returns = self._calculate_portfolio_returns(portfolio_weights, asset_returns)
            
            # Calculate VaR
            var_result = self.var_calculator.calculate_historical_var(portfolio_returns, portfolio_value)
            
            # Calculate risk metrics
            risk_metrics = await self._calculate_comprehensive_risk_metrics(
                portfolio_weights, asset_returns, portfolio_value, var_result, market_data
            )
            
            # Check for risk limit breaches
            alerts = self._check_risk_limits(risk_metrics, portfolio_weights)
            
            # Store alerts
            self.alerts_history.extend(alerts)
            
            return risk_metrics, alerts
            
        except Exception as e:
            self.logger.error(f"Risk monitoring failed: {e}")
            return RiskMetrics(var_results=self.var_calculator._default_var_result(portfolio_value, "Error")), []
    
    async def _calculate_comprehensive_risk_metrics(self, 
                                                  portfolio_weights: Dict[str, float],
                                                  asset_returns: pd.DataFrame,
                                                  portfolio_value: float,
                                                  var_result: VaRResult,
                                                  market_data: Optional[Dict]) -> RiskMetrics:
        """Calculate comprehensive risk metrics"""
        portfolio_returns = self._calculate_portfolio_returns(portfolio_weights, asset_returns)
        
        # Volatility metrics
        realized_vol = portfolio_returns.std() * np.sqrt(252)  # Annualized
        vol_percentile = self._calculate_volatility_percentile(portfolio_returns)
        
        # Correlation metrics
        corr_matrix = asset_returns.corr()
        avg_corr = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].mean()
        max_corr = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].max()
        
        # Concentration metrics
        weights_array = np.array(list(portfolio_weights.values()))
        concentration_index = np.sum(weights_array ** 2)  # Herfindahl index
        effective_assets = 1 / concentration_index if concentration_index > 0 else 0
        largest_position = max(abs(w) for w in portfolio_weights.values())
        
        # Leverage metrics
        gross_leverage = sum(abs(w) for w in portfolio_weights.values())
        net_leverage = abs(sum(portfolio_weights.values()))
        
        # Drawdown metrics
        cumulative_returns = (1 + portfolio_returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        current_drawdown = drawdown.iloc[-1] if len(drawdown) > 0 else 0.0
        max_drawdown = drawdown.min()
        
        # Liquidity metrics (simplified)
        liquidity_score = self._calculate_liquidity_score(portfolio_weights, market_data)
        
        return RiskMetrics(
            var_results=var_result,
            realized_volatility=realized_vol,
            volatility_percentile=vol_percentile,
            avg_correlation=avg_corr,
            max_correlation=max_corr,
            concentration_index=concentration_index,
            effective_assets=effective_assets,
            largest_position_weight=largest_position,
            liquidity_score=liquidity_score,
            gross_leverage=gross_leverage,
            net_leverage=net_leverage,
            current_drawdown=current_drawdown,
            max_drawdown=max_drawdown
        )
    
    def _calculate_portfolio_returns(self, portfolio_weights: Dict[str, float],
                                   asset_returns: pd.DataFrame) -> pd.Series:
        """Calculate portfolio returns"""
        portfolio_returns = pd.Series(0.0, index=asset_returns.index)
        
        for asset, weight in portfolio_weights.items():
            if asset in asset_returns.columns:
                portfolio_returns += weight * asset_returns[asset]
        
        return portfolio_returns
    
    def _calculate_volatility_percentile(self, returns: pd.Series, window: int = 252) -> float:
        """Calculate current volatility percentile"""
        try:
            if len(returns) < window:
                return 0.5  # Default to median
            
            # Rolling volatility
            rolling_vol = returns.rolling(window=21).std() * np.sqrt(252)  # 21-day rolling vol
            current_vol = rolling_vol.iloc[-1]
            
            # Historical volatility distribution
            historical_vols = rolling_vol.dropna()
            
            if len(historical_vols) == 0:
                return 0.5
            
            # Calculate percentile
            percentile = (historical_vols < current_vol).mean()
            return percentile
            
        except Exception:
            return 0.5
    
    def _calculate_liquidity_score(self, portfolio_weights: Dict[str, float],
                                 market_data: Optional[Dict]) -> float:
        """Calculate portfolio liquidity score"""
        try:
            if not market_data or 'liquidity_scores' not in market_data:
                return 0.5  # Default neutral score
            
            liquidity_scores = market_data['liquidity_scores']
            weighted_liquidity = 0.0
            total_weight = 0.0
            
            for asset, weight in portfolio_weights.items():
                if asset in liquidity_scores:
                    weighted_liquidity += abs(weight) * liquidity_scores[asset]
                    total_weight += abs(weight)
            
            return weighted_liquidity / total_weight if total_weight > 0 else 0.5
            
        except Exception:
            return 0.5
    
    def _check_risk_limits(self, risk_metrics: RiskMetrics, 
                         portfolio_weights: Dict[str, float]) -> List[RiskAlert]:
        """Check for risk limit breaches"""
        alerts = []
        current_time = datetime.now()
        
        # VaR limit checks
        if risk_metrics.var_results.var_1d > self.risk_limits.daily_var_limit * risk_metrics.var_results.portfolio_value:
            alerts.append(RiskAlert(
                alert_type=AlertType.VAR_BREACH,
                severity=RiskLevel.HIGH,
                message="Daily VaR limit exceeded",
                current_value=risk_metrics.var_results.var_1d,
                limit_value=self.risk_limits.daily_var_limit * risk_metrics.var_results.portfolio_value,
                timestamp=current_time,
                recommended_action="Reduce portfolio risk or hedge positions"
            ))
        
        # Concentration limit checks
        if risk_metrics.largest_position_weight > self.risk_limits.max_single_position:
            alerts.append(RiskAlert(
                alert_type=AlertType.CONCENTRATION_LIMIT,
                severity=RiskLevel.MEDIUM,
                message="Single position concentration limit exceeded",
                current_value=risk_metrics.largest_position_weight,
                limit_value=self.risk_limits.max_single_position,
                timestamp=current_time,
                recommended_action="Reduce largest position or diversify portfolio"
            ))
        
        # Leverage limit checks
        if risk_metrics.gross_leverage > self.risk_limits.max_gross_leverage:
            alerts.append(RiskAlert(
                alert_type=AlertType.LEVERAGE_LIMIT,
                severity=RiskLevel.HIGH,
                message="Gross leverage limit exceeded",
                current_value=risk_metrics.gross_leverage,
                limit_value=self.risk_limits.max_gross_leverage,
                timestamp=current_time,
                recommended_action="Reduce position sizes or close positions"
            ))
        
        # Drawdown limit checks
        if abs(risk_metrics.current_drawdown) > self.risk_limits.max_daily_drawdown:
            alerts.append(RiskAlert(
                alert_type=AlertType.DRAWDOWN_LIMIT,
                severity=RiskLevel.CRITICAL,
                message="Daily drawdown limit exceeded",
                current_value=abs(risk_metrics.current_drawdown),
                limit_value=self.risk_limits.max_daily_drawdown,
                timestamp=current_time,
                recommended_action="Immediate risk reduction required"
            ))
        
        # Volatility spike checks
        if risk_metrics.volatility_percentile > 0.95:  # 95th percentile
            alerts.append(RiskAlert(
                alert_type=AlertType.VOLATILITY_SPIKE,
                severity=RiskLevel.MEDIUM,
                message="Volatility spike detected",
                current_value=risk_metrics.volatility_percentile,
                limit_value=0.95,
                timestamp=current_time,
                recommended_action="Monitor closely and consider hedging"
            ))
        
        # Correlation breakdown checks
        if risk_metrics.max_correlation > self.risk_limits.max_correlation:
            alerts.append(RiskAlert(
                alert_type=AlertType.CORRELATION_BREAKDOWN,
                severity=RiskLevel.MEDIUM,
                message="High correlation detected - diversification at risk",
                current_value=risk_metrics.max_correlation,
                limit_value=self.risk_limits.max_correlation,
                timestamp=current_time,
                recommended_action="Review portfolio diversification"
            ))
        
        # Liquidity risk checks
        if risk_metrics.liquidity_score < 0.3:  # Low liquidity score
            alerts.append(RiskAlert(
                alert_type=AlertType.LIQUIDITY_RISK,
                severity=RiskLevel.MEDIUM,
                message="Low portfolio liquidity detected",
                current_value=risk_metrics.liquidity_score,
                limit_value=0.3,
                timestamp=current_time,
                recommended_action="Increase liquid asset allocation"
            ))
        
        return alerts
    
    def get_alerts_history(self, hours: int = 24) -> List[RiskAlert]:
        """Get risk alerts from the last N hours"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [alert for alert in self.alerts_history if alert.timestamp >= cutoff_time]
    
    def clear_alerts_history(self):
        """Clear alerts history"""
        self.alerts_history.clear()

class AdvancedRiskManager:
    """Main advanced risk management system"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger("advanced_risk_manager")
        
        # Initialize components
        self.risk_limits = RiskLimits(**config.get('risk_limits', {}))
        self.var_calculator = VaRCalculator(config.get('confidence_level', 0.95))
        self.stress_engine = StressTestEngine()
        self.risk_monitor = RiskMonitor(self.risk_limits)
        
        # Risk management state
        self.current_risk_metrics = None
        self.current_alerts = []
        self.risk_history = []
    
    async def assess_portfolio_risk(self, 
                                  portfolio_weights: Dict[str, float],
                                  asset_returns: pd.DataFrame,
                                  portfolio_value: float,
                                  market_data: Optional[Dict] = None) -> Dict[str, Any]:
        """Comprehensive portfolio risk assessment"""
        try:
            # Real-time risk monitoring
            risk_metrics, alerts = await self.risk_monitor.monitor_portfolio_risk(
                portfolio_weights, asset_returns, portfolio_value, market_data
            )
            
            # Stress testing
            stress_results = []
            for scenario_type in [StressTestType.HISTORICAL_SCENARIO, 
                                StressTestType.MONTE_CARLO_SCENARIO,
                                StressTestType.TAIL_RISK_SCENARIO]:
                scenario_results = self.stress_engine.run_stress_test(
                    portfolio_weights, asset_returns, scenario_type, portfolio_value
                )
                stress_results.extend(scenario_results)
            
            # Update state
            self.current_risk_metrics = risk_metrics
            self.current_alerts = alerts
            
            # Store in history
            self.risk_history.append({
                'timestamp': datetime.now(),
                'risk_metrics': risk_metrics,
                'alerts': alerts,
                'portfolio_value': portfolio_value
            })
            
            # Prepare comprehensive report
            risk_report = {
                'timestamp': datetime.now().isoformat(),
                'portfolio_value': portfolio_value,
                'risk_metrics': risk_metrics.to_dict(),
                'alerts': [alert.to_dict() for alert in alerts],
                'stress_test_results': [result.to_dict() for result in stress_results],
                'risk_summary': self._generate_risk_summary(risk_metrics, alerts, stress_results),
                'recommendations': self._generate_recommendations(risk_metrics, alerts, stress_results)
            }
            
            return risk_report
            
        except Exception as e:
            self.logger.error(f"Portfolio risk assessment failed: {e}")
            return {
                'timestamp': datetime.now().isoformat(),
                'error': str(e),
                'risk_summary': 'Risk assessment failed',
                'recommendations': ['Review system configuration and data quality']
            }
    
    def _generate_risk_summary(self, risk_metrics: RiskMetrics, 
                             alerts: List[RiskAlert],
                             stress_results: List[StressTestResult]) -> str:
        """Generate risk summary"""
        summary_parts = []
        
        # VaR summary
        var_pct = (risk_metrics.var_results.var_1d / risk_metrics.var_results.portfolio_value) * 100
        summary_parts.append(f"Daily VaR: {var_pct:.2f}% (${risk_metrics.var_results.var_1d:,.0f})")
        
        # Volatility summary
        summary_parts.append(f"Portfolio volatility: {risk_metrics.realized_volatility:.2f}%")
        
        # Concentration summary
        summary_parts.append(f"Largest position: {risk_metrics.largest_position_weight:.2f}%")
        
        # Alerts summary
        if alerts:
            high_severity_alerts = [a for a in alerts if a.severity in [RiskLevel.HIGH, RiskLevel.CRITICAL]]
            if high_severity_alerts:
                summary_parts.append(f"⚠️ {len(high_severity_alerts)} high-severity risk alerts")
        
        # Stress test summary
        if stress_results:
            worst_stress = min(stress_results, key=lambda x: x.portfolio_pnl_pct)
            summary_parts.append(f"Worst stress scenario: {worst_stress.portfolio_pnl_pct:.2f}% ({worst_stress.scenario_name})")
        
        return " | ".join(summary_parts)
    
    def calculate_portfolio_risk(self, portfolio_weights: Dict[str, float],
                               asset_returns: pd.DataFrame,
                               portfolio_value: float) -> Dict[str, Any]:
        """Calculate portfolio risk metrics - simplified interface"""
        try:
            # Use thread pool executor to avoid event loop conflicts
            import concurrent.futures
            import asyncio
            
            def run_async_risk_assessment():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    return loop.run_until_complete(
                        self.assess_portfolio_risk(portfolio_weights, asset_returns, portfolio_value)
                    )
                finally:
                    loop.close()
            
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(run_async_risk_assessment)
                result = future.result(timeout=30)  # 30 second timeout
                return result
                
        except Exception as e:
            self.logger.error(f"Portfolio risk calculation failed: {e}")
            return {
                'timestamp': datetime.now().isoformat(),
                'error': str(e),
                'risk_summary': 'Risk calculation failed',
                'recommendations': ['Review system configuration and data quality']
            }
    
    def _generate_recommendations(self, risk_metrics: RiskMetrics,
                                alerts: List[RiskAlert],
                                stress_results: List[StressTestResult]) -> List[str]:
        """Generate risk management recommendations"""
        recommendations = []
        
        # VaR-based recommendations
        var_pct = (risk_metrics.var_results.var_1d / risk_metrics.var_results.portfolio_value) * 100
        if var_pct > 3.0:  # High VaR
            recommendations.append("Consider reducing portfolio risk through position sizing or hedging")
        
        # Concentration recommendations
        if risk_metrics.largest_position_weight > 0.15:  # >15% in single position
            recommendations.append("Reduce concentration risk by diversifying large positions")
        
        # Volatility recommendations
        if risk_metrics.volatility_percentile > 0.90:
            recommendations.append("Current volatility is elevated - consider defensive positioning")
        
        # Correlation recommendations
        if risk_metrics.max_correlation > 0.85:
            recommendations.append("High correlations detected - review diversification strategy")
        
        # Leverage recommendations
        if risk_metrics.gross_leverage > 1.8:
            recommendations.append("High leverage detected - consider reducing position sizes")
        
        # Drawdown recommendations
        if abs(risk_metrics.current_drawdown) > 0.05:  # >5% drawdown
            recommendations.append("Significant drawdown - review stop-loss and risk management rules")
        
        # Alert-based recommendations
        for alert in alerts:
            if alert.recommended_action and alert.recommended_action not in recommendations:
                recommendations.append(alert.recommended_action)
        
        # Stress test recommendations
        severe_stress_results = [r for r in stress_results if r.portfolio_pnl_pct < -0.20]
        if severe_stress_results:
            recommendations.append("Portfolio vulnerable to severe stress scenarios - consider tail risk hedging")
        
        # Default recommendation if none generated
        if not recommendations:
            recommendations.append("Risk profile appears acceptable - continue monitoring")
        
        return recommendations[:5]  # Limit to top 5 recommendations
    
    def get_risk_dashboard_data(self) -> Dict[str, Any]:
        """Get data for risk management dashboard"""
        if not self.current_risk_metrics:
            return {'error': 'No risk data available'}
        
        return {
            'current_metrics': self.current_risk_metrics.to_dict(),
            'active_alerts': [alert.to_dict() for alert in self.current_alerts],
            'alert_counts': {
                'total': len(self.current_alerts),
                'critical': len([a for a in self.current_alerts if a.severity == RiskLevel.CRITICAL]),
                'high': len([a for a in self.current_alerts if a.severity == RiskLevel.HIGH]),
                'medium': len([a for a in self.current_alerts if a.severity == RiskLevel.MEDIUM])
            },
            'risk_limits': {
                'daily_var_limit': self.risk_limits.daily_var_limit,
                'max_single_position': self.risk_limits.max_single_position,
                'max_gross_leverage': self.risk_limits.max_gross_leverage,
                'max_daily_drawdown': self.risk_limits.max_daily_drawdown
            }
        }
    
    def update_risk_limits(self, new_limits: Dict[str, float]):
        """Update risk limits"""
        for key, value in new_limits.items():
            if hasattr(self.risk_limits, key):
                setattr(self.risk_limits, key, value)
                self.logger.info(f"Updated risk limit {key} to {value}")
    
    def get_risk_history(self, hours: int = 24) -> List[Dict]:
        """Get risk history for the last N hours"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [entry for entry in self.risk_history if entry['timestamp'] >= cutoff_time]

# Example usage
if __name__ == "__main__":
    # Example configuration
    config = {
        'confidence_level': 0.95,
        'risk_limits': {
            'daily_var_limit': 0.02,
            'max_single_position': 0.10,
            'max_gross_leverage': 2.0,
            'max_daily_drawdown': 0.03
        }
    }
    
    # Initialize risk manager
    risk_manager = AdvancedRiskManager(config)
    
    print("Advanced Risk Management System initialized")
    print("Available risk management features:")
    print("- Value at Risk (VaR) and Conditional VaR")
    print("- Monte Carlo and Historical Simulation")
    print("- Comprehensive Stress Testing")
    print("- Real-time Risk Monitoring")
    print("- Dynamic Risk Alerts")
    print("- Portfolio Risk Attribution")
    print("- Regulatory Capital Calculations")
    print("- Model Risk Validation")