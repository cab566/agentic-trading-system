#!/usr/bin/env python3
"""
Backtesting Engine for Multi-Asset Trading System

Provides comprehensive backtesting capabilities for:
- Traditional assets (stocks, bonds, options)
- Cryptocurrency markets
- Forex markets
- Multi-strategy portfolios
- Risk-adjusted performance metrics
- Transaction cost modeling
- Market impact simulation
- Walk-forward analysis
- Monte Carlo simulation
- Stress testing
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import uuid
import json
import warnings
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from .config_manager import ConfigManager
from .market_data_aggregator import MarketDataAggregator, DataType
from .execution_engine import ExecutionEngine, Order, Fill, OrderSide, OrderType, OrderStatus
from .risk_manager_24_7 import RiskManager24_7
from utils.cache_manager import CacheManager
from utils.performance_tracker import PerformanceTracker

warnings.filterwarnings('ignore')


class BacktestMode(Enum):
    """Backtesting mode enumeration."""
    HISTORICAL = "historical"
    WALK_FORWARD = "walk_forward"
    MONTE_CARLO = "monte_carlo"
    STRESS_TEST = "stress_test"
    PAPER_TRADING = "paper_trading"


class RebalanceFrequency(Enum):
    """Portfolio rebalancing frequency."""
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    ANNUALLY = "annually"
    NEVER = "never"


@dataclass
class BacktestConfig:
    """Backtesting configuration."""
    start_date: datetime
    end_date: datetime
    initial_capital: float
    mode: BacktestMode = BacktestMode.HISTORICAL
    benchmark: str = "SPY"
    commission_model: str = "fixed"  # fixed, percentage, tiered
    commission_rate: float = 0.001  # 0.1%
    slippage_model: str = "linear"  # linear, sqrt, fixed
    slippage_rate: float = 0.0005  # 0.05%
    market_impact_model: str = "sqrt"  # sqrt, linear, none
    market_impact_rate: float = 0.001  # 0.1%
    rebalance_frequency: RebalanceFrequency = RebalanceFrequency.DAILY
    max_position_size: float = 0.1  # 10% of portfolio
    risk_free_rate: float = 0.02  # 2% annual
    enable_shorting: bool = False
    enable_margin: bool = False
    margin_rate: float = 0.05  # 5% annual
    margin_requirement: float = 0.5  # 50%
    currency: str = "USD"
    timezone: str = "UTC"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'start_date': self.start_date.isoformat(),
            'end_date': self.end_date.isoformat(),
            'initial_capital': self.initial_capital,
            'mode': self.mode.value,
            'benchmark': self.benchmark,
            'commission_model': self.commission_model,
            'commission_rate': self.commission_rate,
            'slippage_model': self.slippage_model,
            'slippage_rate': self.slippage_rate,
            'market_impact_model': self.market_impact_model,
            'market_impact_rate': self.market_impact_rate,
            'rebalance_frequency': self.rebalance_frequency.value,
            'max_position_size': self.max_position_size,
            'risk_free_rate': self.risk_free_rate,
            'enable_shorting': self.enable_shorting,
            'enable_margin': self.enable_margin,
            'margin_rate': self.margin_rate,
            'margin_requirement': self.margin_requirement,
            'currency': self.currency,
            'timezone': self.timezone
        }


@dataclass
class Position:
    """Portfolio position."""
    symbol: str
    quantity: float
    average_price: float
    market_value: float
    unrealized_pnl: float
    realized_pnl: float = 0.0
    entry_date: datetime = field(default_factory=datetime.now)
    last_update: datetime = field(default_factory=datetime.now)
    
    @property
    def market_price(self) -> float:
        """Get current market price."""
        if self.quantity == 0:
            return 0.0
        return self.market_value / abs(self.quantity)
    
    @property
    def total_pnl(self) -> float:
        """Get total P&L."""
        return self.realized_pnl + self.unrealized_pnl
    
    @property
    def return_pct(self) -> float:
        """Get return percentage."""
        cost_basis = abs(self.quantity) * self.average_price
        if cost_basis == 0:
            return 0.0
        return (self.total_pnl / cost_basis) * 100


@dataclass
class PortfolioSnapshot:
    """Portfolio snapshot at a point in time."""
    timestamp: datetime
    total_value: float
    cash: float
    positions: Dict[str, Position]
    daily_return: float = 0.0
    cumulative_return: float = 0.0
    drawdown: float = 0.0
    leverage: float = 0.0
    
    @property
    def invested_value(self) -> float:
        """Get total invested value."""
        return sum(abs(pos.market_value) for pos in self.positions.values())
    
    @property
    def long_value(self) -> float:
        """Get long positions value."""
        return sum(pos.market_value for pos in self.positions.values() if pos.quantity > 0)
    
    @property
    def short_value(self) -> float:
        """Get short positions value."""
        return sum(abs(pos.market_value) for pos in self.positions.values() if pos.quantity < 0)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'total_value': self.total_value,
            'cash': self.cash,
            'invested_value': self.invested_value,
            'long_value': self.long_value,
            'short_value': self.short_value,
            'daily_return': self.daily_return,
            'cumulative_return': self.cumulative_return,
            'drawdown': self.drawdown,
            'leverage': self.leverage,
            'positions': {symbol: {
                'quantity': pos.quantity,
                'market_value': pos.market_value,
                'unrealized_pnl': pos.unrealized_pnl,
                'return_pct': pos.return_pct
            } for symbol, pos in self.positions.items()}
        }


@dataclass
class BacktestResults:
    """Comprehensive backtesting results."""
    config: BacktestConfig
    start_date: datetime
    end_date: datetime
    initial_capital: float
    final_capital: float
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown: float
    max_drawdown_duration: int
    win_rate: float
    profit_factor: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    average_trade: float
    average_win: float
    average_loss: float
    largest_win: float
    largest_loss: float
    total_commission: float
    total_slippage: float
    portfolio_history: List[PortfolioSnapshot] = field(default_factory=list)
    trade_history: List[Dict[str, Any]] = field(default_factory=list)
    benchmark_returns: Optional[pd.Series] = None
    alpha: Optional[float] = None
    beta: Optional[float] = None
    information_ratio: Optional[float] = None
    tracking_error: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'config': self.config.to_dict(),
            'start_date': self.start_date.isoformat(),
            'end_date': self.end_date.isoformat(),
            'initial_capital': self.initial_capital,
            'final_capital': self.final_capital,
            'total_return': self.total_return,
            'annualized_return': self.annualized_return,
            'volatility': self.volatility,
            'sharpe_ratio': self.sharpe_ratio,
            'sortino_ratio': self.sortino_ratio,
            'calmar_ratio': self.calmar_ratio,
            'max_drawdown': self.max_drawdown,
            'max_drawdown_duration': self.max_drawdown_duration,
            'win_rate': self.win_rate,
            'profit_factor': self.profit_factor,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'average_trade': self.average_trade,
            'average_win': self.average_win,
            'average_loss': self.average_loss,
            'largest_win': self.largest_win,
            'largest_loss': self.largest_loss,
            'total_commission': self.total_commission,
            'total_slippage': self.total_slippage,
            'alpha': self.alpha,
            'beta': self.beta,
            'information_ratio': self.information_ratio,
            'tracking_error': self.tracking_error
        }


class BacktestingEngine:
    """
    Comprehensive backtesting engine for multi-asset trading strategies.
    
    Features:
    - Historical backtesting with realistic market conditions
    - Walk-forward analysis for strategy validation
    - Monte Carlo simulation for robustness testing
    - Stress testing under extreme market conditions
    - Multi-asset class support (stocks, crypto, forex)
    - Advanced performance metrics and risk analysis
    - Transaction cost and market impact modeling
    - Portfolio optimization and rebalancing
    - Comprehensive reporting and visualization
    """
    
    def __init__(
        self,
        config_manager: ConfigManager,
        market_data_aggregator: MarketDataAggregator
    ):
        self.config_manager = config_manager
        self.market_data_aggregator = market_data_aggregator
        self.cache_manager = CacheManager(config_manager)
        self.performance_tracker = PerformanceTracker(config_manager)
        self.logger = logging.getLogger(__name__)
        
        # Backtesting state
        self.current_config: Optional[BacktestConfig] = None
        self.current_date: Optional[datetime] = None
        self.portfolio_value: float = 0.0
        self.cash: float = 0.0
        self.positions: Dict[str, Position] = {}
        self.portfolio_history: List[PortfolioSnapshot] = []
        self.trade_history: List[Dict[str, Any]] = []
        
        # Market data cache
        self.price_data: Dict[str, pd.DataFrame] = {}
        self.benchmark_data: Optional[pd.DataFrame] = None
        
        # Strategy callbacks
        self.strategy_callbacks: List[Callable] = []
        
        # Performance tracking
        self.daily_returns: List[float] = []
        self.benchmark_returns: List[float] = []
        self.drawdown_series: List[float] = []
        self.peak_value: float = 0.0
        
        # Transaction costs
        self.total_commission: float = 0.0
        self.total_slippage: float = 0.0
        self.total_market_impact: float = 0.0
    
    async def run_backtest(
        self,
        config: BacktestConfig,
        strategy_func: Callable,
        symbols: List[str],
        **kwargs
    ) -> BacktestResults:
        """
        Run comprehensive backtest.
        
        Args:
            config: Backtesting configuration
            strategy_func: Strategy function to execute
            symbols: List of symbols to trade
            **kwargs: Additional parameters for strategy
        
        Returns:
            BacktestResults: Comprehensive results
        """
        try:
            self.logger.info(f"Starting backtest from {config.start_date} to {config.end_date}")
            
            # Initialize backtest
            await self._initialize_backtest(config, symbols)
            
            # Load market data
            await self._load_market_data(symbols, config.start_date, config.end_date)
            
            # Load benchmark data
            await self._load_benchmark_data(config.benchmark, config.start_date, config.end_date)
            
            # Run backtest based on mode
            if config.mode == BacktestMode.HISTORICAL:
                results = await self._run_historical_backtest(strategy_func, **kwargs)
            elif config.mode == BacktestMode.WALK_FORWARD:
                results = await self._run_walk_forward_backtest(strategy_func, **kwargs)
            elif config.mode == BacktestMode.MONTE_CARLO:
                results = await self._run_monte_carlo_backtest(strategy_func, **kwargs)
            elif config.mode == BacktestMode.STRESS_TEST:
                results = await self._run_stress_test_backtest(strategy_func, **kwargs)
            else:
                raise ValueError(f"Unsupported backtest mode: {config.mode}")
            
            # Calculate comprehensive metrics
            results = await self._calculate_comprehensive_metrics(results)
            
            self.logger.info(f"Backtest completed. Total return: {results.total_return:.2%}")
            
            return results
        
        except Exception as e:
            self.logger.error(f"Error running backtest: {e}")
            raise
    
    async def _initialize_backtest(self, config: BacktestConfig, symbols: List[str]):
        """Initialize backtest state."""
        self.current_config = config
        self.current_date = config.start_date
        self.portfolio_value = config.initial_capital
        self.cash = config.initial_capital
        self.positions = {}
        self.portfolio_history = []
        self.trade_history = []
        self.daily_returns = []
        self.benchmark_returns = []
        self.drawdown_series = []
        self.peak_value = config.initial_capital
        self.total_commission = 0.0
        self.total_slippage = 0.0
        self.total_market_impact = 0.0
        
        # Initialize positions for all symbols
        for symbol in symbols:
            self.positions[symbol] = Position(
                symbol=symbol,
                quantity=0.0,
                average_price=0.0,
                market_value=0.0,
                unrealized_pnl=0.0
            )
    
    async def _load_market_data(self, symbols: List[str], start_date: datetime, end_date: datetime):
        """Load historical market data for all symbols."""
        self.logger.info(f"Loading market data for {len(symbols)} symbols")
        
        # Add buffer for technical indicators
        buffer_start = start_date - timedelta(days=252)  # 1 year buffer
        
        for symbol in symbols:
            try:
                # Get historical data
                data = await self.market_data_aggregator.get_historical_data(
                    symbol=symbol,
                    data_type=DataType.OHLCV,
                    start_date=buffer_start,
                    end_date=end_date,
                    interval='1d'
                )
                
                if data is not None and not data.empty:
                    self.price_data[symbol] = data
                    self.logger.debug(f"Loaded {len(data)} data points for {symbol}")
                else:
                    self.logger.warning(f"No data available for {symbol}")
            
            except Exception as e:
                self.logger.error(f"Error loading data for {symbol}: {e}")
    
    async def _load_benchmark_data(self, benchmark: str, start_date: datetime, end_date: datetime):
        """Load benchmark data for performance comparison."""
        try:
            buffer_start = start_date - timedelta(days=252)
            
            data = await self.market_data_aggregator.get_historical_data(
                symbol=benchmark,
                data_type=DataType.OHLCV,
                start_date=buffer_start,
                end_date=end_date,
                interval='1d'
            )
            
            if data is not None and not data.empty:
                self.benchmark_data = data
                self.logger.info(f"Loaded benchmark data for {benchmark}")
            else:
                self.logger.warning(f"No benchmark data available for {benchmark}")
        
        except Exception as e:
            self.logger.error(f"Error loading benchmark data: {e}")
    
    async def _run_historical_backtest(self, strategy_func: Callable, **kwargs) -> BacktestResults:
        """Run historical backtest."""
        config = self.current_config
        
        # Get trading dates
        trading_dates = pd.date_range(
            start=config.start_date,
            end=config.end_date,
            freq='D'
        )
        
        for date in trading_dates:
            self.current_date = date
            
            # Update portfolio with current market prices
            await self._update_portfolio_values(date)
            
            # Execute strategy
            try:
                signals = await strategy_func(
                    date=date,
                    portfolio=self._get_portfolio_state(),
                    market_data=self._get_market_data_slice(date),
                    **kwargs
                )
                
                # Process trading signals
                if signals:
                    await self._process_trading_signals(signals, date)
            
            except Exception as e:
                self.logger.error(f"Error executing strategy on {date}: {e}")
            
            # Rebalance portfolio if needed
            if self._should_rebalance(date):
                await self._rebalance_portfolio(date)
            
            # Record portfolio snapshot
            await self._record_portfolio_snapshot(date)
            
            # Update performance metrics
            self._update_performance_metrics(date)
        
        # Create results
        return await self._create_backtest_results()
    
    async def _run_walk_forward_backtest(self, strategy_func: Callable, **kwargs) -> BacktestResults:
        """Run walk-forward analysis."""
        config = self.current_config
        
        # Define walk-forward parameters
        training_period = kwargs.get('training_period_days', 252)  # 1 year
        testing_period = kwargs.get('testing_period_days', 63)    # 3 months
        step_size = kwargs.get('step_size_days', 21)              # 1 month
        
        results_list = []
        
        current_start = config.start_date
        while current_start + timedelta(days=training_period + testing_period) <= config.end_date:
            # Define periods
            training_end = current_start + timedelta(days=training_period)
            testing_start = training_end + timedelta(days=1)
            testing_end = testing_start + timedelta(days=testing_period)
            
            self.logger.info(f"Walk-forward: Training {current_start} to {training_end}, Testing {testing_start} to {testing_end}")
            
            # Create sub-config for this period
            sub_config = BacktestConfig(
                start_date=testing_start,
                end_date=testing_end,
                initial_capital=self.portfolio_value,
                mode=BacktestMode.HISTORICAL,
                **{k: v for k, v in config.to_dict().items() if k not in ['start_date', 'end_date', 'initial_capital', 'mode']}
            )
            
            # Run backtest for this period
            period_results = await self._run_historical_backtest(strategy_func, **kwargs)
            results_list.append(period_results)
            
            # Move to next period
            current_start += timedelta(days=step_size)
        
        # Combine results
        return await self._combine_walk_forward_results(results_list)
    
    async def _run_monte_carlo_backtest(self, strategy_func: Callable, **kwargs) -> BacktestResults:
        """Run Monte Carlo simulation."""
        num_simulations = kwargs.get('num_simulations', 1000)
        confidence_level = kwargs.get('confidence_level', 0.95)
        
        simulation_results = []
        
        for i in range(num_simulations):
            self.logger.info(f"Monte Carlo simulation {i+1}/{num_simulations}")
            
            # Generate random market scenarios
            await self._generate_random_market_scenario()
            
            # Run backtest with random scenario
            result = await self._run_historical_backtest(strategy_func, **kwargs)
            simulation_results.append(result)
            
            # Reset state for next simulation
            await self._initialize_backtest(self.current_config, list(self.positions.keys()))
        
        # Analyze Monte Carlo results
        return await self._analyze_monte_carlo_results(simulation_results, confidence_level)
    
    async def _run_stress_test_backtest(self, strategy_func: Callable, **kwargs) -> BacktestResults:
        """Run stress test scenarios."""
        stress_scenarios = kwargs.get('stress_scenarios', [
            {'name': 'Market Crash', 'shock': -0.3, 'duration': 30},
            {'name': 'High Volatility', 'volatility_multiplier': 3, 'duration': 60},
            {'name': 'Interest Rate Shock', 'rate_change': 0.05, 'duration': 90}
        ])
        
        stress_results = []
        
        for scenario in stress_scenarios:
            self.logger.info(f"Running stress test: {scenario['name']}")
            
            # Apply stress scenario to market data
            await self._apply_stress_scenario(scenario)
            
            # Run backtest with stressed data
            result = await self._run_historical_backtest(strategy_func, **kwargs)
            result.scenario_name = scenario['name']
            stress_results.append(result)
            
            # Reset market data
            await self._reset_market_data()
        
        # Combine stress test results
        return await self._combine_stress_test_results(stress_results)
    
    async def _update_portfolio_values(self, date: datetime):
        """Update portfolio values with current market prices."""
        total_market_value = 0.0
        
        for symbol, position in self.positions.items():
            if position.quantity != 0:
                # Get current price
                current_price = await self._get_price(symbol, date)
                
                if current_price is not None:
                    # Update position
                    position.market_value = position.quantity * current_price
                    position.unrealized_pnl = position.market_value - (position.quantity * position.average_price)
                    position.last_update = date
                    
                    total_market_value += position.market_value
        
        # Update portfolio value
        self.portfolio_value = self.cash + total_market_value
    
    async def _get_price(self, symbol: str, date: datetime) -> Optional[float]:
        """Get price for symbol at specific date."""
        try:
            if symbol not in self.price_data:
                return None
            
            data = self.price_data[symbol]
            
            # Find closest date
            available_dates = data.index
            closest_date = min(available_dates, key=lambda x: abs((x - date).total_seconds()))
            
            # Use close price
            return data.loc[closest_date, 'close']
        
        except Exception as e:
            self.logger.error(f"Error getting price for {symbol} on {date}: {e}")
            return None
    
    def _get_portfolio_state(self) -> Dict[str, Any]:
        """Get current portfolio state."""
        return {
            'total_value': self.portfolio_value,
            'cash': self.cash,
            'positions': {symbol: {
                'quantity': pos.quantity,
                'market_value': pos.market_value,
                'unrealized_pnl': pos.unrealized_pnl,
                'average_price': pos.average_price
            } for symbol, pos in self.positions.items() if pos.quantity != 0}
        }
    
    def _get_market_data_slice(self, date: datetime) -> Dict[str, pd.DataFrame]:
        """Get market data up to specific date."""
        market_data = {}
        
        for symbol, data in self.price_data.items():
            # Filter data up to current date
            filtered_data = data[data.index <= date]
            if not filtered_data.empty:
                market_data[symbol] = filtered_data
        
        return market_data
    
    async def _process_trading_signals(self, signals: List[Dict[str, Any]], date: datetime):
        """Process trading signals from strategy."""
        for signal in signals:
            try:
                symbol = signal['symbol']
                action = signal['action']  # 'buy', 'sell', 'hold'
                quantity = signal.get('quantity', 0)
                price = signal.get('price')  # If None, use market price
                
                if action in ['buy', 'sell'] and quantity > 0:
                    await self._execute_trade(symbol, action, quantity, price, date)
            
            except Exception as e:
                self.logger.error(f"Error processing signal {signal}: {e}")
    
    async def _execute_trade(
        self,
        symbol: str,
        action: str,
        quantity: float,
        price: Optional[float],
        date: datetime
    ):
        """Execute a trade."""
        try:
            # Get execution price
            if price is None:
                price = await self._get_price(symbol, date)
            
            if price is None:
                self.logger.warning(f"No price available for {symbol} on {date}")
                return
            
            # Apply slippage and market impact
            execution_price = await self._apply_transaction_costs(symbol, action, quantity, price)
            
            # Calculate trade value
            trade_value = quantity * execution_price
            
            # Check if we have enough cash (for buys)
            if action == 'buy' and trade_value > self.cash:
                # Adjust quantity to available cash
                quantity = self.cash / execution_price
                trade_value = quantity * execution_price
                
                if quantity < 0.01:  # Minimum trade size
                    return
            
            # Execute trade
            position = self.positions[symbol]
            
            if action == 'buy':
                # Update position
                if position.quantity >= 0:
                    # Adding to long position or opening new long
                    new_quantity = position.quantity + quantity
                    new_average_price = ((position.quantity * position.average_price) + trade_value) / new_quantity
                    position.quantity = new_quantity
                    position.average_price = new_average_price
                else:
                    # Covering short position
                    if quantity >= abs(position.quantity):
                        # Full cover and potentially go long
                        remaining_quantity = quantity - abs(position.quantity)
                        position.realized_pnl += abs(position.quantity) * (position.average_price - execution_price)
                        
                        if remaining_quantity > 0:
                            position.quantity = remaining_quantity
                            position.average_price = execution_price
                        else:
                            position.quantity = 0
                            position.average_price = 0
                    else:
                        # Partial cover
                        position.realized_pnl += quantity * (position.average_price - execution_price)
                        position.quantity += quantity  # quantity is positive, position.quantity is negative
                
                # Update cash
                self.cash -= trade_value
            
            elif action == 'sell':
                # Update position
                if position.quantity > 0:
                    # Selling long position
                    if quantity >= position.quantity:
                        # Full sale and potentially go short
                        remaining_quantity = quantity - position.quantity
                        position.realized_pnl += position.quantity * (execution_price - position.average_price)
                        
                        if remaining_quantity > 0 and self.current_config.enable_shorting:
                            position.quantity = -remaining_quantity
                            position.average_price = execution_price
                        else:
                            position.quantity = 0
                            position.average_price = 0
                    else:
                        # Partial sale
                        position.realized_pnl += quantity * (execution_price - position.average_price)
                        position.quantity -= quantity
                else:
                    # Adding to short position (if shorting enabled)
                    if self.current_config.enable_shorting:
                        new_quantity = position.quantity - quantity
                        if position.quantity == 0:
                            position.average_price = execution_price
                        else:
                            new_average_price = ((abs(position.quantity) * position.average_price) + trade_value) / abs(new_quantity)
                            position.average_price = new_average_price
                        position.quantity = new_quantity
                
                # Update cash
                self.cash += trade_value
            
            # Record trade
            trade_record = {
                'date': date,
                'symbol': symbol,
                'action': action,
                'quantity': quantity,
                'price': execution_price,
                'value': trade_value,
                'commission': self._calculate_commission(trade_value),
                'slippage': abs(execution_price - price) * quantity if price else 0
            }
            
            self.trade_history.append(trade_record)
            
            self.logger.debug(f"Executed {action} {quantity} {symbol} at {execution_price} on {date}")
        
        except Exception as e:
            self.logger.error(f"Error executing trade: {e}")
    
    async def _apply_transaction_costs(
        self,
        symbol: str,
        action: str,
        quantity: float,
        price: float
    ) -> float:
        """Apply slippage and market impact to execution price."""
        config = self.current_config
        execution_price = price
        
        # Apply slippage
        if config.slippage_model == 'fixed':
            slippage = config.slippage_rate
        elif config.slippage_model == 'linear':
            slippage = config.slippage_rate * (quantity / 1000)  # Scale with quantity
        elif config.slippage_model == 'sqrt':
            slippage = config.slippage_rate * np.sqrt(quantity / 1000)
        else:
            slippage = 0
        
        # Apply market impact
        if config.market_impact_model == 'sqrt':
            market_impact = config.market_impact_rate * np.sqrt(quantity * price / 1000000)  # Scale with trade value
        elif config.market_impact_model == 'linear':
            market_impact = config.market_impact_rate * (quantity * price / 1000000)
        else:
            market_impact = 0
        
        # Adjust price based on action
        if action == 'buy':
            execution_price = price * (1 + slippage + market_impact)
        else:  # sell
            execution_price = price * (1 - slippage - market_impact)
        
        # Track costs
        self.total_slippage += abs(execution_price - price) * quantity
        self.total_market_impact += market_impact * price * quantity
        
        return execution_price
    
    def _calculate_commission(self, trade_value: float) -> float:
        """Calculate commission for trade."""
        config = self.current_config
        
        if config.commission_model == 'fixed':
            commission = config.commission_rate
        elif config.commission_model == 'percentage':
            commission = trade_value * config.commission_rate
        elif config.commission_model == 'tiered':
            # Simplified tiered structure
            if trade_value < 10000:
                commission = trade_value * 0.001
            elif trade_value < 100000:
                commission = trade_value * 0.0005
            else:
                commission = trade_value * 0.0001
        else:
            commission = 0
        
        self.total_commission += commission
        return commission
    
    def _should_rebalance(self, date: datetime) -> bool:
        """Check if portfolio should be rebalanced."""
        config = self.current_config
        
        if config.rebalance_frequency == RebalanceFrequency.NEVER:
            return False
        elif config.rebalance_frequency == RebalanceFrequency.DAILY:
            return True
        elif config.rebalance_frequency == RebalanceFrequency.WEEKLY:
            return date.weekday() == 0  # Monday
        elif config.rebalance_frequency == RebalanceFrequency.MONTHLY:
            return date.day == 1
        elif config.rebalance_frequency == RebalanceFrequency.QUARTERLY:
            return date.month in [1, 4, 7, 10] and date.day == 1
        elif config.rebalance_frequency == RebalanceFrequency.ANNUALLY:
            return date.month == 1 and date.day == 1
        
        return False
    
    async def _rebalance_portfolio(self, date: datetime):
        """Rebalance portfolio based on target allocations."""
        # This would implement portfolio rebalancing logic
        # For now, just log the rebalancing event
        self.logger.debug(f"Portfolio rebalancing on {date}")
    
    async def _record_portfolio_snapshot(self, date: datetime):
        """Record portfolio snapshot."""
        # Calculate daily return
        daily_return = 0.0
        if self.portfolio_history:
            previous_value = self.portfolio_history[-1].total_value
            daily_return = (self.portfolio_value - previous_value) / previous_value
        
        # Calculate cumulative return
        cumulative_return = (self.portfolio_value - self.current_config.initial_capital) / self.current_config.initial_capital
        
        # Calculate drawdown
        if self.portfolio_value > self.peak_value:
            self.peak_value = self.portfolio_value
        
        drawdown = (self.peak_value - self.portfolio_value) / self.peak_value
        
        # Calculate leverage
        total_position_value = sum(abs(pos.market_value) for pos in self.positions.values())
        leverage = total_position_value / self.portfolio_value if self.portfolio_value > 0 else 0
        
        # Create snapshot
        snapshot = PortfolioSnapshot(
            timestamp=date,
            total_value=self.portfolio_value,
            cash=self.cash,
            positions=self.positions.copy(),
            daily_return=daily_return,
            cumulative_return=cumulative_return,
            drawdown=drawdown,
            leverage=leverage
        )
        
        self.portfolio_history.append(snapshot)
    
    def _update_performance_metrics(self, date: datetime):
        """Update performance tracking metrics."""
        if self.portfolio_history:
            daily_return = self.portfolio_history[-1].daily_return
            self.daily_returns.append(daily_return)
            
            drawdown = self.portfolio_history[-1].drawdown
            self.drawdown_series.append(drawdown)
            
            # Update benchmark returns if available
            if self.benchmark_data is not None:
                benchmark_return = self._calculate_benchmark_return(date)
                if benchmark_return is not None:
                    self.benchmark_returns.append(benchmark_return)
    
    def _calculate_benchmark_return(self, date: datetime) -> Optional[float]:
        """Calculate benchmark return for date."""
        try:
            if self.benchmark_data is None or len(self.benchmark_returns) == 0:
                return None
            
            # Get benchmark price for current and previous date
            current_price = self.benchmark_data.loc[self.benchmark_data.index <= date, 'close'].iloc[-1]
            
            if len(self.benchmark_returns) == 0:
                return 0.0
            
            # Find previous trading day
            previous_dates = self.benchmark_data.index[self.benchmark_data.index < date]
            if len(previous_dates) == 0:
                return 0.0
            
            previous_price = self.benchmark_data.loc[previous_dates[-1], 'close']
            
            return (current_price - previous_price) / previous_price
        
        except Exception as e:
            self.logger.error(f"Error calculating benchmark return: {e}")
            return None
    
    async def _create_backtest_results(self) -> BacktestResults:
        """Create comprehensive backtest results."""
        config = self.current_config
        
        # Basic metrics
        total_return = (self.portfolio_value - config.initial_capital) / config.initial_capital
        
        # Time-based metrics
        days = (config.end_date - config.start_date).days
        years = days / 365.25
        annualized_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
        
        # Risk metrics
        returns_series = pd.Series(self.daily_returns)
        volatility = returns_series.std() * np.sqrt(252)  # Annualized
        
        # Sharpe ratio
        excess_returns = returns_series - (config.risk_free_rate / 252)
        sharpe_ratio = excess_returns.mean() / excess_returns.std() * np.sqrt(252) if excess_returns.std() > 0 else 0
        
        # Sortino ratio
        downside_returns = returns_series[returns_series < 0]
        sortino_ratio = excess_returns.mean() / downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 and downside_returns.std() > 0 else 0
        
        # Drawdown metrics
        max_drawdown = max(self.drawdown_series) if self.drawdown_series else 0
        
        # Calmar ratio
        calmar_ratio = annualized_return / max_drawdown if max_drawdown > 0 else 0
        
        # Trade analysis
        winning_trades = [t for t in self.trade_history if self._is_winning_trade(t)]
        losing_trades = [t for t in self.trade_history if self._is_losing_trade(t)]
        
        total_trades = len(self.trade_history)
        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0
        
        # P&L analysis
        total_realized_pnl = sum(pos.realized_pnl for pos in self.positions.values())
        winning_pnl = sum(self._get_trade_pnl(t) for t in winning_trades)
        losing_pnl = sum(self._get_trade_pnl(t) for t in losing_trades)
        
        profit_factor = abs(winning_pnl / losing_pnl) if losing_pnl != 0 else float('inf')
        
        average_trade = total_realized_pnl / total_trades if total_trades > 0 else 0
        average_win = winning_pnl / len(winning_trades) if winning_trades else 0
        average_loss = losing_pnl / len(losing_trades) if losing_trades else 0
        
        largest_win = max([self._get_trade_pnl(t) for t in winning_trades], default=0)
        largest_loss = min([self._get_trade_pnl(t) for t in losing_trades], default=0)
        
        # Benchmark comparison
        alpha, beta, information_ratio, tracking_error = None, None, None, None
        benchmark_returns_series = None
        
        if self.benchmark_returns and len(self.benchmark_returns) == len(self.daily_returns):
            benchmark_returns_series = pd.Series(self.benchmark_returns)
            
            # Calculate beta
            covariance = np.cov(returns_series, benchmark_returns_series)[0, 1]
            benchmark_variance = benchmark_returns_series.var()
            beta = covariance / benchmark_variance if benchmark_variance > 0 else 0
            
            # Calculate alpha
            alpha = returns_series.mean() - (config.risk_free_rate / 252 + beta * (benchmark_returns_series.mean() - config.risk_free_rate / 252))
            alpha *= 252  # Annualize
            
            # Information ratio and tracking error
            excess_returns_vs_benchmark = returns_series - benchmark_returns_series
            tracking_error = excess_returns_vs_benchmark.std() * np.sqrt(252)
            information_ratio = excess_returns_vs_benchmark.mean() / excess_returns_vs_benchmark.std() * np.sqrt(252) if excess_returns_vs_benchmark.std() > 0 else 0
        
        # Create results object
        results = BacktestResults(
            config=config,
            start_date=config.start_date,
            end_date=config.end_date,
            initial_capital=config.initial_capital,
            final_capital=self.portfolio_value,
            total_return=total_return,
            annualized_return=annualized_return,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            calmar_ratio=calmar_ratio,
            max_drawdown=max_drawdown,
            max_drawdown_duration=self._calculate_max_drawdown_duration(),
            win_rate=win_rate,
            profit_factor=profit_factor,
            total_trades=total_trades,
            winning_trades=len(winning_trades),
            losing_trades=len(losing_trades),
            average_trade=average_trade,
            average_win=average_win,
            average_loss=average_loss,
            largest_win=largest_win,
            largest_loss=largest_loss,
            total_commission=self.total_commission,
            total_slippage=self.total_slippage,
            portfolio_history=self.portfolio_history,
            trade_history=self.trade_history,
            benchmark_returns=benchmark_returns_series,
            alpha=alpha,
            beta=beta,
            information_ratio=information_ratio,
            tracking_error=tracking_error
        )
        
        return results
    
    def _is_winning_trade(self, trade: Dict[str, Any]) -> bool:
        """Check if trade is winning."""
        # Simplified - would need more sophisticated logic for multi-leg trades
        return self._get_trade_pnl(trade) > 0
    
    def _is_losing_trade(self, trade: Dict[str, Any]) -> bool:
        """Check if trade is losing."""
        return self._get_trade_pnl(trade) < 0
    
    def _get_trade_pnl(self, trade: Dict[str, Any]) -> float:
        """Get P&L for trade."""
        # Simplified - would need more sophisticated logic
        symbol = trade['symbol']
        if symbol in self.positions:
            return self.positions[symbol].realized_pnl
        return 0
    
    def _calculate_max_drawdown_duration(self) -> int:
        """Calculate maximum drawdown duration in days."""
        if not self.drawdown_series:
            return 0
        
        max_duration = 0
        current_duration = 0
        
        for drawdown in self.drawdown_series:
            if drawdown > 0:
                current_duration += 1
                max_duration = max(max_duration, current_duration)
            else:
                current_duration = 0
        
        return max_duration
    
    async def _calculate_comprehensive_metrics(self, results: BacktestResults) -> BacktestResults:
        """Calculate additional comprehensive metrics."""
        # Add any additional metric calculations here
        return results
    
    # Visualization methods
    def create_performance_report(self, results: BacktestResults, save_path: Optional[str] = None) -> Dict[str, Any]:
        """Create comprehensive performance report with visualizations."""
        try:
            # Create subplots
            fig = make_subplots(
                rows=3, cols=2,
                subplot_titles=[
                    'Portfolio Value Over Time',
                    'Daily Returns Distribution',
                    'Drawdown Over Time',
                    'Rolling Sharpe Ratio',
                    'Position Allocation',
                    'Trade Analysis'
                ],
                specs=[
                    [{"secondary_y": True}, {"type": "histogram"}],
                    [{"secondary_y": False}, {"secondary_y": False}],
                    [{"type": "pie"}, {"type": "bar"}]
                ]
            )
            
            # Portfolio value over time
            dates = [snapshot.timestamp for snapshot in results.portfolio_history]
            values = [snapshot.total_value for snapshot in results.portfolio_history]
            
            fig.add_trace(
                go.Scatter(x=dates, y=values, name='Portfolio Value', line=dict(color='blue')),
                row=1, col=1
            )
            
            # Benchmark comparison if available
            if results.benchmark_returns is not None:
                benchmark_values = [results.config.initial_capital * (1 + results.benchmark_returns.cumsum()).iloc[i] 
                                  for i in range(len(results.benchmark_returns))]
                fig.add_trace(
                    go.Scatter(x=dates[:len(benchmark_values)], y=benchmark_values, 
                             name='Benchmark', line=dict(color='red', dash='dash')),
                    row=1, col=1
                )
            
            # Daily returns distribution
            daily_returns = [snapshot.daily_return for snapshot in results.portfolio_history[1:]]
            fig.add_trace(
                go.Histogram(x=daily_returns, name='Daily Returns', nbinsx=50),
                row=1, col=2
            )
            
            # Drawdown over time
            drawdowns = [snapshot.drawdown for snapshot in results.portfolio_history]
            fig.add_trace(
                go.Scatter(x=dates, y=drawdowns, name='Drawdown', fill='tonexty', 
                         line=dict(color='red')),
                row=2, col=1
            )
            
            # Rolling Sharpe ratio (simplified)
            if len(daily_returns) > 30:
                rolling_sharpe = pd.Series(daily_returns).rolling(30).apply(
                    lambda x: x.mean() / x.std() * np.sqrt(252) if x.std() > 0 else 0
                )
                fig.add_trace(
                    go.Scatter(x=dates[31:], y=rolling_sharpe[30:], name='30-Day Rolling Sharpe'),
                    row=2, col=2
                )
            
            # Position allocation (latest snapshot)
            if results.portfolio_history:
                latest_positions = results.portfolio_history[-1].positions
                symbols = list(latest_positions.keys())
                values = [abs(pos.market_value) for pos in latest_positions.values()]
                
                if values:
                    fig.add_trace(
                        go.Pie(labels=symbols, values=values, name='Position Allocation'),
                        row=3, col=1
                    )
            
            # Trade analysis
            if results.trade_history:
                trade_pnls = [self._get_trade_pnl(trade) for trade in results.trade_history]
                trade_dates = [trade['date'] for trade in results.trade_history]
                
                colors = ['green' if pnl > 0 else 'red' for pnl in trade_pnls]
                fig.add_trace(
                    go.Bar(x=trade_dates, y=trade_pnls, name='Trade P&L', 
                          marker_color=colors),
                    row=3, col=2
                )
            
            # Update layout
            fig.update_layout(
                title=f"Backtesting Results: {results.start_date.date()} to {results.end_date.date()}",
                height=1200,
                showlegend=True
            )
            
            # Save if path provided
            if save_path:
                fig.write_html(save_path)
            
            # Create summary statistics table
            summary_stats = {
                'Total Return': f"{results.total_return:.2%}",
                'Annualized Return': f"{results.annualized_return:.2%}",
                'Volatility': f"{results.volatility:.2%}",
                'Sharpe Ratio': f"{results.sharpe_ratio:.2f}",
                'Sortino Ratio': f"{results.sortino_ratio:.2f}",
                'Calmar Ratio': f"{results.calmar_ratio:.2f}",
                'Max Drawdown': f"{results.max_drawdown:.2%}",
                'Win Rate': f"{results.win_rate:.2%}",
                'Profit Factor': f"{results.profit_factor:.2f}",
                'Total Trades': results.total_trades,
                'Total Commission': f"${results.total_commission:,.2f}",
                'Total Slippage': f"${results.total_slippage:,.2f}"
            }
            
            if results.alpha is not None:
                summary_stats['Alpha'] = f"{results.alpha:.2%}"
                summary_stats['Beta'] = f"{results.beta:.2f}"
                summary_stats['Information Ratio'] = f"{results.information_ratio:.2f}"
                summary_stats['Tracking Error'] = f"{results.tracking_error:.2%}"
            
            return {
                'figure': fig,
                'summary_stats': summary_stats,
                'results': results.to_dict()
            }
        
        except Exception as e:
            self.logger.error(f"Error creating performance report: {e}")
            return {'error': str(e)}
    
    def export_results(self, results: BacktestResults, export_path: str, format: str = 'json'):
        """Export backtest results to file."""
        try:
            export_data = results.to_dict()
            
            if format.lower() == 'json':
                with open(export_path, 'w') as f:
                    json.dump(export_data, f, indent=2, default=str)
            
            elif format.lower() == 'csv':
                # Export portfolio history to CSV
                portfolio_df = pd.DataFrame([snapshot.to_dict() for snapshot in results.portfolio_history])
                portfolio_df.to_csv(export_path.replace('.csv', '_portfolio.csv'), index=False)
                
                # Export trade history to CSV
                if results.trade_history:
                    trades_df = pd.DataFrame(results.trade_history)
                    trades_df.to_csv(export_path.replace('.csv', '_trades.csv'), index=False)
            
            elif format.lower() == 'excel':
                with pd.ExcelWriter(export_path) as writer:
                    # Summary sheet
                    summary_df = pd.DataFrame([export_data]).T
                    summary_df.columns = ['Value']
                    summary_df.to_excel(writer, sheet_name='Summary')
                    
                    # Portfolio history
                    portfolio_df = pd.DataFrame([snapshot.to_dict() for snapshot in results.portfolio_history])
                    portfolio_df.to_excel(writer, sheet_name='Portfolio History', index=False)
                    
                    # Trade history
                    if results.trade_history:
                        trades_df = pd.DataFrame(results.trade_history)
                        trades_df.to_excel(writer, sheet_name='Trade History', index=False)
            
            self.logger.info(f"Results exported to {export_path}")
        
        except Exception as e:
            self.logger.error(f"Error exporting results: {e}")


if __name__ == "__main__":
    # Example usage
    import asyncio
    from pathlib import Path
    
    async def example_strategy(date, portfolio, market_data, **kwargs):
        """Example momentum strategy."""
        signals = []
        
        for symbol, data in market_data.items():
            if len(data) < 20:
                continue
            
            # Simple momentum signal
            recent_data = data.tail(20)
            sma_short = recent_data['close'].rolling(5).mean().iloc[-1]
            sma_long = recent_data['close'].rolling(20).mean().iloc[-1]
            current_price = recent_data['close'].iloc[-1]
            
            current_position = portfolio['positions'].get(symbol, {}).get('quantity', 0)
            
            # Buy signal
            if sma_short > sma_long and current_position <= 0:
                target_value = portfolio['total_value'] * 0.1  # 10% allocation
                quantity = target_value / current_price
                signals.append({
                    'symbol': symbol,
                    'action': 'buy',
                    'quantity': quantity,
                    'price': current_price
                })
            
            # Sell signal
            elif sma_short < sma_long and current_position > 0:
                signals.append({
                    'symbol': symbol,
                    'action': 'sell',
                    'quantity': current_position,
                    'price': current_price
                })
        
        return signals
    
    async def test_backtesting_engine():
        config_manager = ConfigManager(Path("../config"))
        market_data_aggregator = MarketDataAggregator(config_manager)
        
        engine = BacktestingEngine(config_manager, market_data_aggregator)
        
        # Create backtest configuration
        config = BacktestConfig(
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 12, 31),
            initial_capital=100000,
            commission_rate=0.001,
            slippage_rate=0.0005
        )
        
        # Run backtest
        symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA']
        results = await engine.run_backtest(config, example_strategy, symbols)
        
        print(f"Backtest Results:")
        print(f"Total Return: {results.total_return:.2%}")
        print(f"Sharpe Ratio: {results.sharpe_ratio:.2f}")
        print(f"Max Drawdown: {results.max_drawdown:.2%}")
        
        # Create performance report
        report = engine.create_performance_report(results)
        if 'figure' in report:
            report['figure'].show()
    
    # Commented out to prevent event loop conflicts when imported
    # asyncio.run(test_backtesting_engine())