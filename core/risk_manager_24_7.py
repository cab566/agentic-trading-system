#!/usr/bin/env python3
"""
24/7 Risk Management System for Multi-Asset Trading

Provides comprehensive risk management across:
- Traditional markets (stocks, bonds, options)
- Cryptocurrency markets (24/7 trading)
- Forex markets (24/5 trading)
- Cross-asset correlation monitoring
- Real-time position monitoring
- Dynamic risk limits adjustment
- Market regime-based risk controls
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque

import pandas as pd
import numpy as np
from scipy import stats
from pydantic import BaseModel, Field

from .config_manager import ConfigManager
from .data_manager import UnifiedDataManager
from utils.notifications import NotificationManager


class RiskLevel(Enum):
    """Risk level enumeration."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AssetClass(Enum):
    """Asset class enumeration."""
    EQUITY = "equity"
    BOND = "bond"
    CRYPTO = "crypto"
    FOREX = "forex"
    COMMODITY = "commodity"
    OPTION = "option"
    FUTURE = "future"


class MarketRegime(Enum):
    """Market regime enumeration."""
    NORMAL = "normal"
    HIGH_VOLATILITY = "high_volatility"
    CRISIS = "crisis"
    TRENDING = "trending"
    RANGING = "ranging"


@dataclass
class Position:
    """Position information."""
    symbol: str
    asset_class: AssetClass
    quantity: float
    entry_price: float
    current_price: float
    market_value: float
    unrealized_pnl: float
    realized_pnl: float = 0.0
    entry_time: datetime = field(default_factory=datetime.now)
    last_update: datetime = field(default_factory=datetime.now)
    
    @property
    def pnl_percent(self) -> float:
        """Calculate PnL percentage."""
        if self.entry_price == 0:
            return 0.0
        return (self.current_price - self.entry_price) / self.entry_price
    
    @property
    def holding_period(self) -> timedelta:
        """Calculate holding period."""
        return datetime.now() - self.entry_time


@dataclass
class RiskMetrics:
    """Risk metrics for portfolio or position."""
    var_1d: float  # 1-day Value at Risk
    var_5d: float  # 5-day Value at Risk
    cvar_1d: float  # 1-day Conditional VaR
    cvar_5d: float  # 5-day Conditional VaR
    max_drawdown: float
    volatility: float
    beta: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    correlation_risk: float
    concentration_risk: float
    liquidity_risk: float
    currency_risk: float
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class RiskLimit:
    """Risk limit definition."""
    name: str
    limit_type: str  # 'absolute', 'percentage', 'ratio'
    value: float
    current_value: float
    utilization: float
    asset_class: Optional[AssetClass] = None
    symbol: Optional[str] = None
    is_breached: bool = False
    breach_time: Optional[datetime] = None
    warning_threshold: float = 0.8  # 80% of limit
    
    @property
    def is_warning(self) -> bool:
        """Check if limit is in warning zone."""
        return self.utilization >= self.warning_threshold


@dataclass
class RiskAlert:
    """Risk alert information."""
    alert_id: str
    alert_type: str
    severity: RiskLevel
    message: str
    symbol: Optional[str] = None
    asset_class: Optional[AssetClass] = None
    current_value: float = 0.0
    limit_value: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    acknowledged: bool = False
    resolved: bool = False


class RiskManager24_7:
    """
    Comprehensive 24/7 risk management system for multi-asset trading.
    
    Features:
    - Real-time position monitoring
    - Dynamic risk limit adjustment
    - Cross-asset correlation monitoring
    - Market regime detection
    - Automated risk controls
    - 24/7 monitoring and alerting
    """
    
    def __init__(self, config_manager: ConfigManager):
        self.config_manager = config_manager
        self.data_manager = UnifiedDataManager(config_manager)
        self.notification_manager = NotificationManager(config_manager)
        self.logger = logging.getLogger(__name__)
        
        # Risk configuration
        self.risk_config = self._load_risk_config()
        
        # Portfolio state
        self.positions: Dict[str, Position] = {}
        self.portfolio_value = 0.0
        self.cash_balance = 0.0
        self.total_equity = 0.0
        
        # Risk metrics and limits
        self.risk_metrics: Optional[RiskMetrics] = None
        self.risk_limits: Dict[str, RiskLimit] = {}
        self.active_alerts: Dict[str, RiskAlert] = {}
        
        # Market data
        self.price_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=252))  # 1 year
        self.correlation_matrix: Optional[pd.DataFrame] = None
        self.volatility_estimates: Dict[str, float] = {}
        
        # Market regime
        self.current_regime = MarketRegime.NORMAL
        self.regime_probability = 0.5
        self.regime_history: deque = deque(maxlen=100)
        
        # Risk controls
        self.emergency_stop = False
        self.trading_halted = False
        self.last_risk_check = datetime.now()
        
        # Initialize risk limits
        self._initialize_risk_limits()
        
        # Initialize monitoring tasks list (start monitoring separately)
        self._monitoring_tasks = []
    
    def _load_risk_config(self) -> Dict[str, Any]:
        """Load risk management configuration."""
        return {
            'max_portfolio_var': 0.02,  # 2% daily VaR
            'max_position_size': 0.10,  # 10% of portfolio
            'max_asset_class_allocation': {
                AssetClass.EQUITY: 0.60,
                AssetClass.BOND: 0.40,
                AssetClass.CRYPTO: 0.15,
                AssetClass.FOREX: 0.10,
                AssetClass.COMMODITY: 0.20,
                AssetClass.OPTION: 0.05
            },
            'max_correlation_exposure': 0.70,
            'max_drawdown': 0.15,  # 15%
            'min_liquidity_ratio': 0.05,  # 5% cash
            'max_leverage': {
                AssetClass.EQUITY: 2.0,
                AssetClass.CRYPTO: 3.0,
                AssetClass.FOREX: 10.0
            },
            'stop_loss_levels': {
                AssetClass.EQUITY: 0.08,  # 8%
                AssetClass.CRYPTO: 0.15,  # 15%
                AssetClass.FOREX: 0.05   # 5%
            },
            'regime_adjustments': {
                MarketRegime.CRISIS: {
                    'var_multiplier': 2.0,
                    'position_size_multiplier': 0.5,
                    'correlation_threshold': 0.5
                },
                MarketRegime.HIGH_VOLATILITY: {
                    'var_multiplier': 1.5,
                    'position_size_multiplier': 0.7,
                    'correlation_threshold': 0.6
                }
            }
        }
    
    def _initialize_risk_limits(self):
        """Initialize risk limits based on configuration."""
        config = self.risk_config
        
        # Portfolio-level limits
        self.risk_limits['portfolio_var'] = RiskLimit(
            name='Portfolio VaR',
            limit_type='percentage',
            value=config['max_portfolio_var'],
            current_value=0.0,
            utilization=0.0
        )
        
        self.risk_limits['max_drawdown'] = RiskLimit(
            name='Maximum Drawdown',
            limit_type='percentage',
            value=config['max_drawdown'],
            current_value=0.0,
            utilization=0.0
        )
        
        # Asset class limits
        for asset_class, limit in config['max_asset_class_allocation'].items():
            self.risk_limits[f'{asset_class.value}_allocation'] = RiskLimit(
                name=f'{asset_class.value.title()} Allocation',
                limit_type='percentage',
                value=limit,
                current_value=0.0,
                utilization=0.0,
                asset_class=asset_class
            )
        
        # Liquidity limit
        self.risk_limits['liquidity_ratio'] = RiskLimit(
            name='Liquidity Ratio',
            limit_type='percentage',
            value=config['min_liquidity_ratio'],
            current_value=0.0,
            utilization=0.0
        )
    
    def start_monitoring(self):
        """Start background monitoring tasks."""
        # Only start if we have an event loop
        try:
            loop = asyncio.get_running_loop()
            # Real-time risk monitoring
            self._monitoring_tasks.append(
                loop.create_task(self._continuous_risk_monitoring())
            )
            
            # Market regime detection
            self._monitoring_tasks.append(
                loop.create_task(self._market_regime_monitoring())
            )
            
            # Correlation monitoring
            self._monitoring_tasks.append(
                loop.create_task(self._correlation_monitoring())
            )
            
            # Alert management
            self._monitoring_tasks.append(
                loop.create_task(self._alert_management())
            )
        except RuntimeError:
            # No event loop running, monitoring will be started later
            self.logger.info("No event loop running, monitoring tasks will be started when needed")
    
    async def update_position(self, symbol: str, quantity: float, price: float, 
                            asset_class: AssetClass) -> bool:
        """Update position information."""
        try:
            current_time = datetime.now()
            
            if symbol in self.positions:
                position = self.positions[symbol]
                position.quantity = quantity
                position.current_price = price
                position.market_value = quantity * price
                position.unrealized_pnl = (price - position.entry_price) * quantity
                position.last_update = current_time
            else:
                # New position
                self.positions[symbol] = Position(
                    symbol=symbol,
                    asset_class=asset_class,
                    quantity=quantity,
                    entry_price=price,
                    current_price=price,
                    market_value=quantity * price,
                    unrealized_pnl=0.0,
                    entry_time=current_time,
                    last_update=current_time
                )
            
            # Update price history
            self.price_history[symbol].append((current_time, price))
            
            # Trigger risk check
            await self._check_position_risk(symbol)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error updating position {symbol}: {e}")
            return False
    
    async def remove_position(self, symbol: str, exit_price: float) -> bool:
        """Remove position and calculate realized PnL."""
        try:
            if symbol not in self.positions:
                return False
            
            position = self.positions[symbol]
            realized_pnl = (exit_price - position.entry_price) * position.quantity
            
            # Log the trade
            self.logger.info(
                f"Position closed: {symbol}, PnL: {realized_pnl:.2f}, "
                f"Return: {position.pnl_percent:.2%}"
            )
            
            # Remove from positions
            del self.positions[symbol]
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error removing position {symbol}: {e}")
            return False
    
    async def calculate_portfolio_risk(self) -> RiskMetrics:
        """Calculate comprehensive portfolio risk metrics."""
        try:
            if not self.positions:
                return RiskMetrics(
                    var_1d=0.0, var_5d=0.0, cvar_1d=0.0, cvar_5d=0.0,
                    max_drawdown=0.0, volatility=0.0, beta=1.0,
                    sharpe_ratio=0.0, sortino_ratio=0.0, calmar_ratio=0.0,
                    correlation_risk=0.0, concentration_risk=0.0,
                    liquidity_risk=0.0, currency_risk=0.0
                )
            
            # Get portfolio returns
            portfolio_returns = await self._calculate_portfolio_returns()
            
            if len(portfolio_returns) < 30:  # Need at least 30 days
                return self._default_risk_metrics()
            
            # Calculate VaR and CVaR
            var_1d = np.percentile(portfolio_returns, 5)  # 95% confidence
            var_5d = var_1d * np.sqrt(5)  # Scale to 5 days
            
            cvar_1d = portfolio_returns[portfolio_returns <= var_1d].mean()
            cvar_5d = cvar_1d * np.sqrt(5)
            
            # Calculate volatility
            volatility = np.std(portfolio_returns) * np.sqrt(252)  # Annualized
            
            # Calculate maximum drawdown
            cumulative_returns = (1 + pd.Series(portfolio_returns)).cumprod()
            rolling_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - rolling_max) / rolling_max
            max_drawdown = drawdown.min()
            
            # Calculate Sharpe ratio
            mean_return = np.mean(portfolio_returns) * 252  # Annualized
            risk_free_rate = 0.04  # 4%
            sharpe_ratio = (mean_return - risk_free_rate) / volatility if volatility > 0 else 0
            
            # Calculate Sortino ratio
            downside_returns = portfolio_returns[portfolio_returns < 0]
            downside_volatility = np.std(downside_returns) * np.sqrt(252) if len(downside_returns) > 0 else volatility
            sortino_ratio = (mean_return - risk_free_rate) / downside_volatility if downside_volatility > 0 else 0
            
            # Calculate Calmar ratio
            calmar_ratio = mean_return / abs(max_drawdown) if max_drawdown != 0 else 0
            
            # Calculate beta (vs market proxy)
            beta = await self._calculate_portfolio_beta(portfolio_returns)
            
            # Calculate risk factors
            correlation_risk = await self._calculate_correlation_risk()
            concentration_risk = self._calculate_concentration_risk()
            liquidity_risk = self._calculate_liquidity_risk()
            currency_risk = self._calculate_currency_risk()
            
            risk_metrics = RiskMetrics(
                var_1d=var_1d,
                var_5d=var_5d,
                cvar_1d=cvar_1d,
                cvar_5d=cvar_5d,
                max_drawdown=max_drawdown,
                volatility=volatility,
                beta=beta,
                sharpe_ratio=sharpe_ratio,
                sortino_ratio=sortino_ratio,
                calmar_ratio=calmar_ratio,
                correlation_risk=correlation_risk,
                concentration_risk=concentration_risk,
                liquidity_risk=liquidity_risk,
                currency_risk=currency_risk
            )
            
            self.risk_metrics = risk_metrics
            return risk_metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating portfolio risk: {e}")
            return self._default_risk_metrics()
    
    async def check_risk_limits(self) -> List[RiskAlert]:
        """Check all risk limits and generate alerts."""
        alerts = []
        
        try:
            # Update current values
            await self._update_risk_limit_values()
            
            # Check each limit
            for limit_name, limit in self.risk_limits.items():
                if limit.is_breached:
                    alert = RiskAlert(
                        alert_id=f"{limit_name}_{datetime.now().timestamp()}",
                        alert_type="LIMIT_BREACH",
                        severity=RiskLevel.CRITICAL,
                        message=f"{limit.name} breached: {limit.current_value:.2%} > {limit.value:.2%}",
                        current_value=limit.current_value,
                        limit_value=limit.value
                    )
                    alerts.append(alert)
                    
                elif limit.is_warning:
                    alert = RiskAlert(
                        alert_id=f"{limit_name}_warning_{datetime.now().timestamp()}",
                        alert_type="LIMIT_WARNING",
                        severity=RiskLevel.HIGH,
                        message=f"{limit.name} approaching limit: {limit.current_value:.2%} (>{limit.warning_threshold:.0%} of limit)",
                        current_value=limit.current_value,
                        limit_value=limit.value
                    )
                    alerts.append(alert)
            
            # Store alerts
            for alert in alerts:
                self.active_alerts[alert.alert_id] = alert
            
            return alerts
            
        except Exception as e:
            self.logger.error(f"Error checking risk limits: {e}")
            return []
    
    async def get_position_risk_score(self, symbol: str) -> float:
        """Calculate risk score for a specific position (0-100)."""
        try:
            if symbol not in self.positions:
                return 0.0
            
            position = self.positions[symbol]
            risk_score = 0.0
            
            # Size risk (0-30 points)
            position_weight = abs(position.market_value) / self.total_equity if self.total_equity > 0 else 0
            max_weight = self.risk_config['max_position_size']
            size_risk = min(30, (position_weight / max_weight) * 30)
            risk_score += size_risk
            
            # Volatility risk (0-25 points)
            if symbol in self.volatility_estimates:
                volatility = self.volatility_estimates[symbol]
                vol_risk = min(25, volatility * 100)  # Scale volatility to 0-25
                risk_score += vol_risk
            
            # PnL risk (0-20 points)
            pnl_percent = abs(position.pnl_percent)
            pnl_risk = min(20, pnl_percent * 100)  # Scale to 0-20
            risk_score += pnl_risk
            
            # Holding period risk (0-15 points)
            holding_days = position.holding_period.days
            holding_risk = min(15, holding_days / 30 * 15)  # Max risk at 30+ days
            risk_score += holding_risk
            
            # Liquidity risk (0-10 points)
            liquidity_score = self._get_asset_liquidity_score(symbol)
            liquidity_risk = (1 - liquidity_score) * 10
            risk_score += liquidity_risk
            
            return min(100, risk_score)
            
        except Exception as e:
            self.logger.error(f"Error calculating position risk for {symbol}: {e}")
            return 50.0  # Default medium risk
    
    async def should_halt_trading(self) -> Tuple[bool, str]:
        """Determine if trading should be halted."""
        try:
            # Check emergency stop
            if self.emergency_stop:
                return True, "Emergency stop activated"
            
            # Check critical risk limits
            critical_alerts = [
                alert for alert in self.active_alerts.values()
                if alert.severity == RiskLevel.CRITICAL and not alert.acknowledged
            ]
            
            if critical_alerts:
                return True, f"Critical risk alerts: {len(critical_alerts)}"
            
            # Check portfolio VaR
            if self.risk_metrics and abs(self.risk_metrics.var_1d) > self.risk_config['max_portfolio_var']:
                return True, f"Portfolio VaR exceeded: {self.risk_metrics.var_1d:.2%}"
            
            # Check maximum drawdown
            if self.risk_metrics and abs(self.risk_metrics.max_drawdown) > self.risk_config['max_drawdown']:
                return True, f"Maximum drawdown exceeded: {self.risk_metrics.max_drawdown:.2%}"
            
            # Check market regime
            if self.current_regime == MarketRegime.CRISIS and self.regime_probability > 0.8:
                return True, "Crisis market regime detected"
            
            return False, "All systems normal"
            
        except Exception as e:
            self.logger.error(f"Error checking trading halt conditions: {e}")
            return True, f"Error in risk check: {str(e)}"
    
    async def get_position_size_limit(self, symbol: str, asset_class: AssetClass) -> float:
        """Get maximum allowed position size for a symbol."""
        try:
            # Base position limit
            base_limit = self.risk_config['max_position_size']
            
            # Asset class limit
            asset_class_limit = self.risk_config['max_asset_class_allocation'].get(asset_class, 0.1)
            
            # Current asset class allocation
            current_allocation = self._get_current_asset_class_allocation(asset_class)
            available_allocation = max(0, asset_class_limit - current_allocation)
            
            # Market regime adjustment
            regime_multiplier = 1.0
            if self.current_regime in self.risk_config['regime_adjustments']:
                regime_multiplier = self.risk_config['regime_adjustments'][self.current_regime]['position_size_multiplier']
            
            # Volatility adjustment
            vol_multiplier = 1.0
            if symbol in self.volatility_estimates:
                volatility = self.volatility_estimates[symbol]
                # Reduce position size for high volatility assets
                vol_multiplier = max(0.1, 1.0 - (volatility - 0.2) * 2) if volatility > 0.2 else 1.0
            
            # Calculate final limit
            final_limit = min(
                base_limit,
                available_allocation,
                base_limit * regime_multiplier * vol_multiplier
            )
            
            return max(0.001, final_limit)  # Minimum 0.1%
            
        except Exception as e:
            self.logger.error(f"Error calculating position size limit for {symbol}: {e}")
            return 0.01  # Default 1%
    
    def _get_current_asset_class_allocation(self, asset_class: AssetClass) -> float:
        """Get current allocation percentage for an asset class.
        
        Args:
            asset_class: The asset class to check
            
        Returns:
            Current allocation as a percentage (0.0 to 1.0)
        """
        try:
            total_portfolio_value = sum(pos.market_value for pos in self.positions.values())
            if total_portfolio_value <= 0:
                return 0.0
            
            asset_class_value = sum(
                pos.market_value for pos in self.positions.values() 
                if pos.asset_class == asset_class
            )
            
            return asset_class_value / total_portfolio_value
            
        except Exception as e:
            self.logger.error(f"Error calculating asset class allocation for {asset_class}: {e}")
            return 0.0
    
    async def check_order_risk(self, order_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Check if an order passes risk validation.
        
        Args:
            order_dict: Dictionary containing order details
            
        Returns:
            Dict with 'approved' boolean and 'reason' string
        """
        try:
            symbol = order_dict.get('symbol')
            quantity = order_dict.get('quantity', 0)
            side = order_dict.get('side')
            price = order_dict.get('price', 0)
            
            if not symbol or not quantity:
                return {'approved': False, 'reason': 'Missing symbol or quantity'}
            
            # Check if trading is halted
            halt_trading, halt_reason = await self.should_halt_trading()
            if halt_trading:
                return {'approved': False, 'reason': f'Trading halted: {halt_reason}'}
            
            # Determine asset class (simplified)
            asset_class = AssetClass.EQUITY  # Default
            if 'BTC' in symbol or 'ETH' in symbol or 'CRYPTO' in symbol.upper():
                asset_class = AssetClass.CRYPTO
            elif 'USD' in symbol or 'EUR' in symbol or 'GBP' in symbol:
                asset_class = AssetClass.FOREX
            
            # Check position size limits
            max_position_size = await self.get_position_size_limit(symbol, asset_class)
            current_position = self.positions.get(symbol, Position(
                symbol=symbol, asset_class=asset_class, quantity=0, 
                entry_price=0, current_price=0, market_value=0, unrealized_pnl=0
            ))
            
            # Calculate new position size
            new_quantity = current_position.quantity
            if side in ['buy', 'BUY']:
                new_quantity += quantity
            elif side in ['sell', 'SELL']:
                new_quantity -= quantity
            
            # Check if new position exceeds limits
            position_value = abs(new_quantity * price) if price else abs(new_quantity * 100)  # Fallback price
            portfolio_value = sum(pos.market_value for pos in self.positions.values()) or 100000  # Fallback
            position_percentage = position_value / portfolio_value
            
            if position_percentage > max_position_size:
                return {
                    'approved': False, 
                    'reason': f'Position size {position_percentage:.2%} exceeds limit {max_position_size:.2%}'
                }
            
            # Check risk score
            risk_score = await self.get_position_risk_score(symbol)
            if risk_score > 0.8:  # High risk threshold
                return {
                    'approved': False,
                    'reason': f'High risk score {risk_score:.2f} for {symbol}'
                }
            
            return {'approved': True, 'reason': 'Order passed risk checks'}
            
        except Exception as e:
            self.logger.error(f"Error in order risk check: {e}")
            return {'approved': False, 'reason': f'Risk check error: {str(e)}'}
    
    async def _continuous_risk_monitoring(self):
        """Continuous risk monitoring task."""
        while True:
            try:
                # Update portfolio metrics
                await self.calculate_portfolio_risk()
                
                # Check risk limits
                alerts = await self.check_risk_limits()
                
                # Send notifications for new alerts
                for alert in alerts:
                    if not alert.acknowledged:
                        await self._send_risk_alert(alert)
                
                # Update last check time
                self.last_risk_check = datetime.now()
                
                # Sleep based on market activity
                sleep_duration = self._get_monitoring_interval()
                await asyncio.sleep(sleep_duration)
                
            except Exception as e:
                self.logger.error(f"Error in continuous risk monitoring: {e}")
                await asyncio.sleep(60)  # Wait 1 minute on error
    
    async def _market_regime_monitoring(self):
        """Monitor market regime changes."""
        while True:
            try:
                # Detect current market regime
                new_regime, probability = await self._detect_market_regime()
                
                # Check for regime change
                if new_regime != self.current_regime:
                    self.logger.info(f"Market regime change: {self.current_regime.value} -> {new_regime.value}")
                    
                    # Update regime
                    self.current_regime = new_regime
                    self.regime_probability = probability
                    
                    # Adjust risk limits
                    await self._adjust_risk_limits_for_regime(new_regime)
                    
                    # Send notification
                    await self._send_regime_change_alert(new_regime, probability)
                
                # Store regime history
                self.regime_history.append((datetime.now(), new_regime, probability))
                
                await asyncio.sleep(300)  # Check every 5 minutes
                
            except Exception as e:
                self.logger.error(f"Error in market regime monitoring: {e}")
                await asyncio.sleep(300)
    
    async def _correlation_monitoring(self):
        """Monitor cross-asset correlations."""
        while True:
            try:
                # Update correlation matrix
                await self._update_correlation_matrix()
                
                # Check for correlation spikes
                if self.correlation_matrix is not None:
                    avg_correlation = self._calculate_average_correlation()
                    
                    if avg_correlation > self.risk_config['max_correlation_exposure']:
                        alert = RiskAlert(
                            alert_id=f"correlation_spike_{datetime.now().timestamp()}",
                            alert_type="HIGH_CORRELATION",
                            severity=RiskLevel.HIGH,
                            message=f"High cross-asset correlation detected: {avg_correlation:.2%}",
                            current_value=avg_correlation,
                            limit_value=self.risk_config['max_correlation_exposure']
                        )
                        
                        self.active_alerts[alert.alert_id] = alert
                        await self._send_risk_alert(alert)
                
                await asyncio.sleep(600)  # Check every 10 minutes
                
            except Exception as e:
                self.logger.error(f"Error in correlation monitoring: {e}")
                await asyncio.sleep(600)
    
    async def _update_correlation_matrix(self):
        """Update the correlation matrix for all positions."""
        try:
            if len(self.positions) < 2:
                self.correlation_matrix = None
                return
            
            symbols = list(self.positions.keys())
            returns_data = {}
            
            # Get price history for each symbol
            for symbol in symbols:
                if symbol in self.price_history and len(self.price_history[symbol]) > 1:
                    prices = [price for _, price in self.price_history[symbol][-30:]]  # Last 30 data points
                    if len(prices) > 1:
                        returns = np.diff(np.log(prices))
                        returns_data[symbol] = returns
            
            if len(returns_data) >= 2:
                # Create DataFrame and calculate correlation
                min_length = min(len(returns) for returns in returns_data.values())
                if min_length > 0:
                    aligned_returns = {symbol: returns[-min_length:] for symbol, returns in returns_data.items()}
                    df = pd.DataFrame(aligned_returns)
                    self.correlation_matrix = df.corr()
                else:
                    self.correlation_matrix = None
            else:
                self.correlation_matrix = None
                
        except Exception as e:
            self.logger.error(f"Error updating correlation matrix: {e}")
            self.correlation_matrix = None
    
    async def _update_risk_limit_values(self):
        """Update current values for all risk limits."""
        try:
            current_time = datetime.now()
            
            for limit in self.risk_limits.values():
                if limit.limit_type == 'portfolio_var':
                    if self.risk_metrics:
                        limit.current_value = abs(self.risk_metrics.var_1d)
                        limit.utilization = limit.current_value / limit.value if limit.value > 0 else 0
                        limit.is_breached = limit.current_value > limit.value
                        
                elif limit.limit_type == 'max_drawdown':
                    if self.risk_metrics:
                        limit.current_value = abs(self.risk_metrics.max_drawdown)
                        limit.utilization = limit.current_value / limit.value if limit.value > 0 else 0
                        limit.is_breached = limit.current_value > limit.value
                        
                elif limit.limit_type == 'leverage':
                    total_exposure = sum(abs(pos.market_value) for pos in self.positions.values())
                    limit.current_value = total_exposure / self.portfolio_value if self.portfolio_value > 0 else 0
                    limit.utilization = limit.current_value / limit.value if limit.value > 0 else 0
                    limit.is_breached = limit.current_value > limit.value
                    
                elif limit.limit_type == 'concentration':
                    if self.positions:
                        max_position_weight = max(abs(pos.market_value) / self.portfolio_value 
                                                for pos in self.positions.values()) if self.portfolio_value > 0 else 0
                        limit.current_value = max_position_weight
                        limit.utilization = limit.current_value / limit.value if limit.value > 0 else 0
                        limit.is_breached = limit.current_value > limit.value
                
                # Update breach time if newly breached
                if limit.is_breached and limit.breach_time is None:
                    limit.breach_time = current_time
                elif not limit.is_breached:
                    limit.breach_time = None
                    
        except Exception as e:
            self.logger.error(f"Error updating risk limit values: {e}")
    
    async def _alert_management(self):
        """Manage and cleanup alerts."""
        while True:
            try:
                current_time = datetime.now()
                
                # Auto-resolve old alerts
                alerts_to_remove = []
                for alert_id, alert in self.active_alerts.items():
                    # Remove alerts older than 24 hours
                    if (current_time - alert.timestamp).total_seconds() > 86400:
                        alerts_to_remove.append(alert_id)
                
                for alert_id in alerts_to_remove:
                    del self.active_alerts[alert_id]
                
                await asyncio.sleep(3600)  # Check every hour
                
            except Exception as e:
                self.logger.error(f"Error in alert management: {e}")
                await asyncio.sleep(3600)
    
    # Helper methods
    async def _calculate_portfolio_returns(self) -> np.ndarray:
        """Calculate historical portfolio returns."""
        # Simplified implementation - would need actual portfolio history
        returns = []
        
        for symbol, position in self.positions.items():
            if symbol in self.price_history and len(self.price_history[symbol]) > 1:
                prices = [price for _, price in self.price_history[symbol]]
                symbol_returns = np.diff(prices) / prices[:-1]
                
                # Weight by position size
                weight = abs(position.market_value) / self.total_equity if self.total_equity > 0 else 0
                weighted_returns = symbol_returns * weight
                
                if len(returns) == 0:
                    returns = weighted_returns
                else:
                    # Align lengths and add
                    min_len = min(len(returns), len(weighted_returns))
                    returns = returns[-min_len:] + weighted_returns[-min_len:]
        
        return np.array(returns) if len(returns) > 0 else np.array([0.0])
    
    def _default_risk_metrics(self) -> RiskMetrics:
        """Return default risk metrics when calculation fails."""
        return RiskMetrics(
            var_1d=0.0, var_5d=0.0, cvar_1d=0.0, cvar_5d=0.0,
            max_drawdown=0.0, volatility=0.0, beta=1.0,
            sharpe_ratio=0.0, sortino_ratio=0.0, calmar_ratio=0.0,
            correlation_risk=0.0, concentration_risk=0.0,
            liquidity_risk=0.0, currency_risk=0.0
        )
    
    async def _calculate_portfolio_beta(self, portfolio_returns: np.ndarray) -> float:
        """Calculate portfolio beta vs market."""
        try:
            # Use SPY as market proxy
            market_data = await self.data_manager.get_historical_data('SPY', '1d', limit=len(portfolio_returns))
            if market_data is not None and not market_data.empty:
                # Handle different column name cases
                close_col = 'close' if 'close' in market_data.columns else 'Close'
                if close_col in market_data.columns:
                    market_returns = market_data[close_col].pct_change().dropna().values
                else:
                    # Fallback to first numeric column
                    numeric_cols = market_data.select_dtypes(include=[np.number]).columns
                    if len(numeric_cols) > 0:
                        market_returns = market_data[numeric_cols[0]].pct_change().dropna().values
                    else:
                        return 0.0
                
                # Align lengths
                min_len = min(len(portfolio_returns), len(market_returns))
                if min_len > 10:
                    port_ret = portfolio_returns[-min_len:]
                    mkt_ret = market_returns[-min_len:]
                    
                    # Calculate beta
                    covariance = np.cov(port_ret, mkt_ret)[0, 1]
                    market_variance = np.var(mkt_ret)
                    
                    return covariance / market_variance if market_variance > 0 else 1.0
            
            return 1.0  # Default beta
            
        except Exception as e:
            self.logger.error(f"Error calculating portfolio beta: {e}")
            return 1.0
    
    async def _calculate_correlation_risk(self) -> float:
        """Calculate correlation risk score."""
        if self.correlation_matrix is None or len(self.positions) < 2:
            return 0.0
        
        # Calculate weighted average correlation
        total_correlation = 0.0
        total_weight = 0.0
        
        symbols = list(self.positions.keys())
        for i, symbol1 in enumerate(symbols):
            for j, symbol2 in enumerate(symbols[i+1:], i+1):
                if symbol1 in self.correlation_matrix.index and symbol2 in self.correlation_matrix.columns:
                    correlation = abs(self.correlation_matrix.loc[symbol1, symbol2])
                    
                    # Weight by position sizes
                    weight1 = abs(self.positions[symbol1].market_value) / self.total_equity
                    weight2 = abs(self.positions[symbol2].market_value) / self.total_equity
                    weight = weight1 * weight2
                    
                    total_correlation += correlation * weight
                    total_weight += weight
        
        return total_correlation / total_weight if total_weight > 0 else 0.0
    
    def _calculate_concentration_risk(self) -> float:
        """Calculate concentration risk (Herfindahl index)."""
        if not self.positions or self.total_equity == 0:
            return 0.0
        
        weights = [abs(pos.market_value) / self.total_equity for pos in self.positions.values()]
        return sum(w**2 for w in weights)
    
    def _calculate_liquidity_risk(self) -> float:
        """Calculate liquidity risk score."""
        if not self.positions:
            return 0.0
        
        total_illiquid_weight = 0.0
        
        for position in self.positions.values():
            liquidity_score = self._get_asset_liquidity_score(position.symbol)
            weight = abs(position.market_value) / self.total_equity if self.total_equity > 0 else 0
            
            # Higher weight for less liquid assets
            illiquid_weight = (1 - liquidity_score) * weight
            total_illiquid_weight += illiquid_weight
        
        return total_illiquid_weight
    
    def _calculate_currency_risk(self) -> float:
        """Calculate currency exposure risk."""
        currency_exposure = defaultdict(float)
        
        for position in self.positions.values():
            # Simplified - assume USD base, get currency from symbol
            currency = self._get_position_currency(position.symbol)
            weight = abs(position.market_value) / self.total_equity if self.total_equity > 0 else 0
            currency_exposure[currency] += weight
        
        # Calculate concentration in non-USD currencies
        non_usd_exposure = sum(weight for currency, weight in currency_exposure.items() if currency != 'USD')
        
        return non_usd_exposure
    
    def _get_asset_liquidity_score(self, symbol: str) -> float:
        """Get liquidity score for an asset (0-1)."""
        # Simplified liquidity scoring
        if any(crypto in symbol.upper() for crypto in ['BTC', 'ETH']):
            return 0.9  # Major cryptos are liquid
        elif 'USD' in symbol or '/' in symbol:
            return 0.95  # Forex is very liquid
        elif symbol in ['SPY', 'QQQ', 'IWM', 'AAPL', 'MSFT', 'GOOGL']:
            return 0.95  # Major ETFs and stocks
        else:
            return 0.7  # Default for other assets
    
    def _get_position_currency(self, symbol: str) -> str:
        """Get the currency of a position."""
        if '/' in symbol:
            return symbol.split('/')[1]  # Forex pair
        elif any(crypto in symbol.upper() for crypto in ['BTC', 'ETH', 'ADA']):
            return 'USD'  # Crypto priced in USD
        else:
            return 'USD'  # Default
    
    async def _detect_market_regime(self) -> Tuple[MarketRegime, float]:
        """Detect current market regime."""
        try:
            # Get market data for regime detection
            market_data = await self.data_manager.get_historical_data('SPY', '1d', limit=60)
            if market_data is None or market_data.empty:
                return MarketRegime.NORMAL, 0.5
            
            # Handle different column name cases
            close_col = 'close' if 'close' in market_data.columns else 'Close'
            if close_col in market_data.columns:
                returns = market_data[close_col].pct_change().dropna()
            else:
                # Fallback to first numeric column
                numeric_cols = market_data.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    returns = market_data[numeric_cols[0]].pct_change().dropna()
                else:
                    return MarketRegime.NORMAL, 0.5
            
            # Calculate regime indicators
            volatility = returns.std() * np.sqrt(252)  # Annualized
            recent_volatility = returns.tail(20).std() * np.sqrt(252)
            
            # Correlation with other assets (simplified)
            avg_correlation = 0.5  # Placeholder
            
            # Regime classification
            if recent_volatility > 0.35 or avg_correlation > 0.8:
                return MarketRegime.CRISIS, 0.8
            elif recent_volatility > 0.25:
                return MarketRegime.HIGH_VOLATILITY, 0.7
            elif abs(returns.tail(20).mean()) > 0.002:  # Strong trend
                return MarketRegime.TRENDING, 0.6
            else:
                return MarketRegime.NORMAL, 0.5
                
        except Exception as e:
            self.logger.error(f"Error detecting market regime: {e}")
            return MarketRegime.NORMAL, 0.5
    
    def _get_monitoring_interval(self) -> int:
        """Get monitoring interval based on market activity."""
        # More frequent monitoring during market hours and high volatility
        current_hour = datetime.now().hour
        
        # Market hours (9:30 AM - 4:00 PM ET)
        if 9 <= current_hour <= 16:
            return 30  # 30 seconds during market hours
        elif self.current_regime in [MarketRegime.CRISIS, MarketRegime.HIGH_VOLATILITY]:
            return 60  # 1 minute during high volatility
        else:
            return 300  # 5 minutes otherwise
    
    async def _send_risk_alert(self, alert: RiskAlert):
        """Send risk alert notification."""
        try:
            await self.notification_manager.send_alert(
                title=f"Risk Alert: {alert.alert_type}",
                message=alert.message,
                severity=alert.severity.value,
                alert_type="RISK"
            )
        except Exception as e:
            self.logger.error(f"Error sending risk alert: {e}")
    
    async def _send_regime_change_alert(self, regime: MarketRegime, probability: float):
        """Send market regime change alert."""
        try:
            await self.notification_manager.send_alert(
                title="Market Regime Change",
                message=f"Market regime changed to {regime.value} (confidence: {probability:.1%})",
                severity="medium",
                alert_type="REGIME"
            )
        except Exception as e:
            self.logger.error(f"Error sending regime change alert: {e}")
    
    # Additional helper methods would be implemented here...
    
    def get_risk_summary(self) -> Dict[str, Any]:
        """Get comprehensive risk summary."""
        return {
            'portfolio_value': self.portfolio_value,
            'total_positions': len(self.positions),
            'risk_metrics': self.risk_metrics.__dict__ if self.risk_metrics else None,
            'active_alerts': len(self.active_alerts),
            'critical_alerts': len([a for a in self.active_alerts.values() if a.severity == RiskLevel.CRITICAL]),
            'market_regime': self.current_regime.value,
            'regime_probability': self.regime_probability,
            'emergency_stop': self.emergency_stop,
            'trading_halted': self.trading_halted,
            'last_risk_check': self.last_risk_check.isoformat()
        }


if __name__ == "__main__":
    # Example usage
    import asyncio
    from pathlib import Path
    
    async def test_risk_manager():
        config_manager = ConfigManager(Path("../config"))
        risk_manager = RiskManager24_7(config_manager)
        
        # Test position update
        await risk_manager.update_position('AAPL', 100, 150.0, AssetClass.EQUITY)
        await risk_manager.update_position('BTC-USD', 0.5, 45000.0, AssetClass.CRYPTO)
        
        # Calculate risk
        risk_metrics = await risk_manager.calculate_portfolio_risk()
        print(f"Portfolio VaR: {risk_metrics.var_1d:.2%}")
        
        # Check limits
        alerts = await risk_manager.check_risk_limits()
        print(f"Active alerts: {len(alerts)}")
        
        # Get risk summary
        summary = risk_manager.get_risk_summary()
        print(f"Risk summary: {summary}")
    
    # Commented out to prevent event loop conflicts when imported
    # asyncio.run(test_risk_manager())