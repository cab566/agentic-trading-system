#!/usr/bin/env python3
"""
Portfolio Management Tool for CrewAI Trading System

Provides agents with portfolio optimization, rebalancing,
performance tracking, and risk management capabilities.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum

import pandas as pd
import numpy as np
from scipy.optimize import minimize
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from core.data_manager import UnifiedDataManager, DataRequest


class OptimizationMethod(Enum):
    """Portfolio optimization methods."""
    MEAN_VARIANCE = "mean_variance"
    RISK_PARITY = "risk_parity"
    MINIMUM_VARIANCE = "minimum_variance"
    MAXIMUM_SHARPE = "maximum_sharpe"
    BLACK_LITTERMAN = "black_litterman"
    EQUAL_WEIGHT = "equal_weight"
    MARKET_CAP_WEIGHT = "market_cap_weight"


class RebalanceFrequency(Enum):
    """Portfolio rebalancing frequencies."""
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    SEMI_ANNUAL = "semi_annual"
    ANNUAL = "annual"
    THRESHOLD = "threshold"


@dataclass
class Position:
    """Portfolio position data structure."""
    symbol: str
    quantity: float
    current_price: float
    market_value: float
    cost_basis: float
    unrealized_pnl: float
    realized_pnl: float
    weight: float
    target_weight: float = 0.0
    last_updated: datetime = field(default_factory=datetime.now)
    
    @property
    def total_return(self) -> float:
        """Calculate total return percentage."""
        if self.cost_basis == 0:
            return 0.0
        return ((self.market_value - self.cost_basis) / self.cost_basis) * 100


@dataclass
class Portfolio:
    """Portfolio data structure."""
    portfolio_id: str
    name: str
    positions: Dict[str, Position]
    cash: float
    total_value: float
    benchmark: str = "SPY"
    created_at: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)
    
    @property
    def invested_value(self) -> float:
        """Calculate total invested value."""
        return sum(pos.market_value for pos in self.positions.values())
    
    @property
    def total_unrealized_pnl(self) -> float:
        """Calculate total unrealized P&L."""
        return sum(pos.unrealized_pnl for pos in self.positions.values())
    
    @property
    def total_realized_pnl(self) -> float:
        """Calculate total realized P&L."""
        return sum(pos.realized_pnl for pos in self.positions.values())
    
    @property
    def position_count(self) -> int:
        """Get number of positions."""
        return len([pos for pos in self.positions.values() if pos.quantity != 0])


class PortfolioManagementInput(BaseModel):
    """Input schema for portfolio management requests."""
    action: str = Field(
        ...,
        description="Action to perform: 'optimize', 'rebalance', 'analyze', 'performance', 'risk', 'allocate', 'backtest'"
    )
    portfolio_id: Optional[str] = Field(
        default="default",
        description="Portfolio identifier"
    )
    symbols: Optional[List[str]] = Field(
        default=None,
        description="List of symbols to include in portfolio"
    )
    target_weights: Optional[Dict[str, float]] = Field(
        default=None,
        description="Target allocation weights for symbols"
    )
    optimization_method: Optional[str] = Field(
        default="mean_variance",
        description="Optimization method: 'mean_variance', 'risk_parity', 'minimum_variance', 'maximum_sharpe'"
    )
    risk_tolerance: Optional[float] = Field(
        default=0.5,
        description="Risk tolerance level (0.0 to 1.0)"
    )
    expected_return: Optional[float] = Field(
        default=None,
        description="Target expected return (annualized)"
    )
    max_weight: Optional[float] = Field(
        default=0.3,
        description="Maximum weight for any single position"
    )
    min_weight: Optional[float] = Field(
        default=0.01,
        description="Minimum weight for any position"
    )
    rebalance_threshold: Optional[float] = Field(
        default=0.05,
        description="Rebalancing threshold (weight deviation)"
    )
    lookback_days: Optional[int] = Field(
        default=252,
        description="Lookback period for historical analysis (days)"
    )
    benchmark: Optional[str] = Field(
        default="SPY",
        description="Benchmark symbol for comparison"
    )
    cash_target: Optional[float] = Field(
        default=0.05,
        description="Target cash allocation (0.0 to 1.0)"
    )
    include_dividends: Optional[bool] = Field(
        default=True,
        description="Include dividends in performance calculations"
    )
    transaction_costs: Optional[float] = Field(
        default=0.001,
        description="Transaction cost as percentage of trade value"
    )
    start_date: Optional[str] = Field(
        default=None,
        description="Start date for analysis (ISO format)"
    )
    end_date: Optional[str] = Field(
        default=None,
        description="End date for analysis (ISO format)"
    )


class PortfolioManagementTool(BaseTool):
    """
    Portfolio Management Tool for CrewAI agents.
    
    Provides comprehensive portfolio management including:
    - Portfolio optimization using various methods
    - Rebalancing recommendations and execution
    - Performance analysis and attribution
    - Risk assessment and monitoring
    - Asset allocation strategies
    - Backtesting and scenario analysis
    """
    
    name: str = "portfolio_management_tool"
    description: str = (
        "Manage and optimize investment portfolios. Provides portfolio optimization, "
        "rebalancing, performance analysis, risk assessment, and asset allocation "
        "strategies for systematic portfolio management."
    )
    args_schema: type[PortfolioManagementInput] = PortfolioManagementInput
    
    # Pydantic v2 field declarations
    data_manager: UnifiedDataManager = Field(exclude=True)
    logger: Any = Field(default=None, exclude=True)
    portfolios: Dict[str, Portfolio] = Field(default_factory=dict, exclude=True)
    price_cache: Dict[str, Dict[str, float]] = Field(default_factory=dict, exclude=True)
    returns_cache: Dict[str, pd.DataFrame] = Field(default_factory=dict, exclude=True)
    risk_free_rate: float = Field(default=0.05, exclude=True)
    
    def __init__(self, data_manager: UnifiedDataManager, **kwargs):
        """
        Initialize the portfolio management tool.
        
        Args:
            data_manager: Unified data manager instance
        """
        super().__init__(
            data_manager=data_manager,
            logger=logging.getLogger(__name__),
            portfolios={},
            price_cache={},
            returns_cache={},
            risk_free_rate=0.05,
            **kwargs
        )
    
    def _run(self, **kwargs) -> str:
        """Synchronous wrapper for async execution."""
        try:
            # Create event loop if none exists
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            # Run async method
            if loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, self._arun(**kwargs))
                    return future.result()
            else:
                return loop.run_until_complete(self._arun(**kwargs))
                
        except Exception as e:
            self.logger.error(f"Error in portfolio management tool: {e}")
            return f"Error processing portfolio management request: {str(e)}"
    
    async def _arun(self, **kwargs) -> str:
        """Asynchronous execution of portfolio management."""
        try:
            # Parse input
            input_data = PortfolioManagementInput(**kwargs)
            
            # Route to appropriate handler
            if input_data.action == "optimize":
                return await self._optimize_portfolio(input_data)
            elif input_data.action == "rebalance":
                return await self._rebalance_portfolio(input_data)
            elif input_data.action == "analyze":
                return await self._analyze_portfolio(input_data)
            elif input_data.action == "performance":
                return await self._analyze_performance(input_data)
            elif input_data.action == "risk":
                return await self._analyze_risk(input_data)
            elif input_data.action == "allocate":
                return await self._allocate_assets(input_data)
            elif input_data.action == "backtest":
                return await self._backtest_strategy(input_data)
            else:
                return f"Error: Unknown action '{input_data.action}'"
                
        except Exception as e:
            self.logger.error(f"Error in async portfolio management: {e}")
            return f"Error processing portfolio management request: {str(e)}"
    
    async def _optimize_portfolio(self, input_data: PortfolioManagementInput) -> str:
        """Optimize portfolio allocation using specified method."""
        try:
            if not input_data.symbols:
                return "Error: No symbols provided for optimization"
            
            # Get historical returns
            returns_data = await self._get_returns_data(input_data.symbols, input_data.lookback_days)
            
            if returns_data.empty:
                return "Error: Unable to fetch historical data for optimization"
            
            # Calculate optimization inputs
            mean_returns = returns_data.mean() * 252  # Annualized
            cov_matrix = returns_data.cov() * 252  # Annualized
            
            # Perform optimization based on method
            method = OptimizationMethod(input_data.optimization_method)
            
            if method == OptimizationMethod.MEAN_VARIANCE:
                weights = self._optimize_mean_variance(mean_returns, cov_matrix, input_data)
            elif method == OptimizationMethod.RISK_PARITY:
                weights = self._optimize_risk_parity(cov_matrix, input_data)
            elif method == OptimizationMethod.MINIMUM_VARIANCE:
                weights = self._optimize_minimum_variance(cov_matrix, input_data)
            elif method == OptimizationMethod.MAXIMUM_SHARPE:
                weights = self._optimize_maximum_sharpe(mean_returns, cov_matrix, input_data)
            elif method == OptimizationMethod.EQUAL_WEIGHT:
                weights = self._optimize_equal_weight(input_data.symbols)
            else:
                return f"Error: Optimization method '{method.value}' not implemented"
            
            # Calculate portfolio metrics
            portfolio_return = np.sum(weights * mean_returns)
            portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_vol
            
            # Generate optimization report
            result = f"Portfolio Optimization Report\n"
            result += f"Method: {method.value.replace('_', ' ').title()}\n"
            result += f"Symbols: {', '.join(input_data.symbols)}\n"
            result += f"Lookback Period: {input_data.lookback_days} days\n\n"
            
            result += "Optimized Allocation:\n"
            for symbol, weight in weights.items():
                result += f"  {symbol}: {weight:.1%}\n"
            
            result += f"\nPortfolio Metrics:\n"
            result += f"  Expected Return: {portfolio_return:.2%} (annualized)\n"
            result += f"  Volatility: {portfolio_vol:.2%} (annualized)\n"
            result += f"  Sharpe Ratio: {sharpe_ratio:.3f}\n"
            
            # Risk metrics
            var_95 = np.percentile(returns_data.dot(weights), 5)
            result += f"  Value at Risk (95%): {var_95:.2%} (daily)\n"
            
            # Diversification metrics
            concentration = np.sum(weights ** 2)
            result += f"  Concentration (HHI): {concentration:.3f}\n"
            result += f"  Effective Number of Assets: {1/concentration:.1f}\n"
            
            # Store optimized portfolio
            portfolio = self._create_portfolio(input_data.portfolio_id, weights, input_data.symbols)
            self.portfolios[input_data.portfolio_id] = portfolio
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error optimizing portfolio: {e}")
            return f"Error optimizing portfolio: {str(e)}"
    
    async def _rebalance_portfolio(self, input_data: PortfolioManagementInput) -> str:
        """Generate rebalancing recommendations for portfolio."""
        try:
            portfolio = self.portfolios.get(input_data.portfolio_id)
            if not portfolio:
                return f"Error: Portfolio '{input_data.portfolio_id}' not found"
            
            # Update current prices and positions
            await self._update_portfolio_prices(portfolio)
            
            # Calculate current weights
            total_value = portfolio.total_value
            current_weights = {}
            for symbol, position in portfolio.positions.items():
                current_weights[symbol] = position.market_value / total_value
            
            # Get target weights (from optimization or manual input)
            target_weights = input_data.target_weights or {}
            if not target_weights:
                # Use stored target weights from positions
                target_weights = {symbol: pos.target_weight for symbol, pos in portfolio.positions.items()}
            
            # Calculate weight deviations
            deviations = {}
            rebalance_needed = False
            
            for symbol in set(list(current_weights.keys()) + list(target_weights.keys())):
                current_weight = current_weights.get(symbol, 0.0)
                target_weight = target_weights.get(symbol, 0.0)
                deviation = current_weight - target_weight
                deviations[symbol] = deviation
                
                if abs(deviation) > input_data.rebalance_threshold:
                    rebalance_needed = True
            
            # Generate rebalancing report
            result = f"Portfolio Rebalancing Analysis\n"
            result += f"Portfolio: {portfolio.name} ({input_data.portfolio_id})\n"
            result += f"Total Value: ${total_value:,.2f}\n"
            result += f"Rebalance Threshold: {input_data.rebalance_threshold:.1%}\n\n"
            
            if not rebalance_needed:
                result += "✅ No rebalancing needed - all positions within threshold\n\n"
            else:
                result += "⚠️ Rebalancing recommended\n\n"
            
            result += "Position Analysis:\n"
            result += f"{'Symbol':<8} {'Current':<8} {'Target':<8} {'Deviation':<10} {'Action':<12} {'Trade Value':<12}\n"
            result += "-" * 70 + "\n"
            
            total_trades = 0
            for symbol in sorted(set(list(current_weights.keys()) + list(target_weights.keys()))):
                current_weight = current_weights.get(symbol, 0.0)
                target_weight = target_weights.get(symbol, 0.0)
                deviation = deviations[symbol]
                
                # Calculate trade requirements
                target_value = target_weight * total_value
                current_value = current_weight * total_value
                trade_value = target_value - current_value
                
                action = "HOLD"
                if abs(deviation) > input_data.rebalance_threshold:
                    action = "BUY" if trade_value > 0 else "SELL"
                    total_trades += abs(trade_value)
                
                result += f"{symbol:<8} {current_weight:<8.1%} {target_weight:<8.1%} {deviation:<10.1%} {action:<12} ${trade_value:<12,.0f}\n"
            
            # Transaction cost analysis
            transaction_costs = total_trades * input_data.transaction_costs
            result += f"\nTransaction Cost Analysis:\n"
            result += f"  Total Trade Value: ${total_trades:,.2f}\n"
            result += f"  Estimated Costs: ${transaction_costs:,.2f} ({input_data.transaction_costs:.1%})\n"
            result += f"  Cost as % of Portfolio: {transaction_costs/total_value:.3%}\n"
            
            # Rebalancing recommendations
            result += f"\nRecommendations:\n"
            if rebalance_needed:
                if transaction_costs / total_value < 0.001:  # Less than 0.1%
                    result += "• Proceed with rebalancing - transaction costs are minimal\n"
                elif transaction_costs / total_value < 0.005:  # Less than 0.5%
                    result += "• Consider rebalancing - moderate transaction costs\n"
                else:
                    result += "• Evaluate carefully - high transaction costs may outweigh benefits\n"
                
                # Priority trades
                high_priority = [(s, abs(d)) for s, d in deviations.items() if abs(d) > input_data.rebalance_threshold * 2]
                if high_priority:
                    high_priority.sort(key=lambda x: x[1], reverse=True)
                    result += f"• Priority rebalancing for: {', '.join([s for s, _ in high_priority[:3]])}\n"
            else:
                result += "• No action required - portfolio is well-balanced\n"
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error rebalancing portfolio: {e}")
            return f"Error rebalancing portfolio: {str(e)}"
    
    async def _analyze_performance(self, input_data: PortfolioManagementInput) -> str:
        """Analyze portfolio performance metrics."""
        try:
            portfolio = self.portfolios.get(input_data.portfolio_id)
            if not portfolio:
                return f"Error: Portfolio '{input_data.portfolio_id}' not found"
            
            # Get performance data
            symbols = list(portfolio.positions.keys())
            if input_data.benchmark not in symbols:
                symbols.append(input_data.benchmark)
            
            returns_data = await self._get_returns_data(symbols, input_data.lookback_days)
            
            if returns_data.empty:
                return "Error: Unable to fetch performance data"
            
            # Calculate portfolio returns
            weights = pd.Series({symbol: pos.weight for symbol, pos in portfolio.positions.items()})
            portfolio_returns = returns_data[list(portfolio.positions.keys())].dot(weights)
            benchmark_returns = returns_data[input_data.benchmark]
            
            # Performance metrics
            portfolio_total_return = (1 + portfolio_returns).prod() - 1
            benchmark_total_return = (1 + benchmark_returns).prod() - 1
            
            portfolio_annual_return = (1 + portfolio_returns.mean()) ** 252 - 1
            benchmark_annual_return = (1 + benchmark_returns.mean()) ** 252 - 1
            
            portfolio_volatility = portfolio_returns.std() * np.sqrt(252)
            benchmark_volatility = benchmark_returns.std() * np.sqrt(252)
            
            portfolio_sharpe = (portfolio_annual_return - self.risk_free_rate) / portfolio_volatility
            benchmark_sharpe = (benchmark_annual_return - self.risk_free_rate) / benchmark_volatility
            
            # Risk metrics
            portfolio_var_95 = np.percentile(portfolio_returns, 5)
            portfolio_cvar_95 = portfolio_returns[portfolio_returns <= portfolio_var_95].mean()
            
            max_drawdown = self._calculate_max_drawdown(portfolio_returns)
            
            # Tracking metrics
            excess_returns = portfolio_returns - benchmark_returns
            tracking_error = excess_returns.std() * np.sqrt(252)
            information_ratio = excess_returns.mean() * np.sqrt(252) / tracking_error if tracking_error > 0 else 0
            
            # Beta calculation
            covariance = np.cov(portfolio_returns, benchmark_returns)[0, 1]
            benchmark_variance = np.var(benchmark_returns)
            beta = covariance / benchmark_variance if benchmark_variance > 0 else 1.0
            
            # Generate performance report
            result = f"Portfolio Performance Analysis\n"
            result += f"Portfolio: {portfolio.name} ({input_data.portfolio_id})\n"
            result += f"Benchmark: {input_data.benchmark}\n"
            result += f"Analysis Period: {input_data.lookback_days} days\n\n"
            
            result += "Return Metrics:\n"
            result += f"  Portfolio Total Return: {portfolio_total_return:.2%}\n"
            result += f"  Benchmark Total Return: {benchmark_total_return:.2%}\n"
            result += f"  Excess Return: {portfolio_total_return - benchmark_total_return:.2%}\n\n"
            
            result += f"  Portfolio Annual Return: {portfolio_annual_return:.2%}\n"
            result += f"  Benchmark Annual Return: {benchmark_annual_return:.2%}\n"
            result += f"  Alpha: {portfolio_annual_return - benchmark_annual_return:.2%}\n\n"
            
            result += "Risk Metrics:\n"
            result += f"  Portfolio Volatility: {portfolio_volatility:.2%}\n"
            result += f"  Benchmark Volatility: {benchmark_volatility:.2%}\n"
            result += f"  Beta: {beta:.3f}\n\n"
            
            result += f"  Value at Risk (95%): {portfolio_var_95:.2%}\n"
            result += f"  Conditional VaR (95%): {portfolio_cvar_95:.2%}\n"
            result += f"  Maximum Drawdown: {max_drawdown:.2%}\n\n"
            
            result += "Risk-Adjusted Returns:\n"
            result += f"  Portfolio Sharpe Ratio: {portfolio_sharpe:.3f}\n"
            result += f"  Benchmark Sharpe Ratio: {benchmark_sharpe:.3f}\n"
            result += f"  Information Ratio: {information_ratio:.3f}\n"
            result += f"  Tracking Error: {tracking_error:.2%}\n\n"
            
            # Position contribution analysis
            result += "Position Contributions:\n"
            for symbol, position in portfolio.positions.items():
                if symbol in returns_data.columns:
                    pos_return = returns_data[symbol].mean() * 252
                    contribution = position.weight * pos_return
                    result += f"  {symbol}: {contribution:.2%} (weight: {position.weight:.1%}, return: {pos_return:.2%})\n"
            
            # Performance summary
            result += f"\nPerformance Summary:\n"
            if portfolio_total_return > benchmark_total_return:
                result += f"✅ Portfolio outperformed benchmark by {portfolio_total_return - benchmark_total_return:.2%}\n"
            else:
                result += f"❌ Portfolio underperformed benchmark by {benchmark_total_return - portfolio_total_return:.2%}\n"
            
            if portfolio_sharpe > benchmark_sharpe:
                result += f"✅ Better risk-adjusted returns (Sharpe: {portfolio_sharpe:.3f} vs {benchmark_sharpe:.3f})\n"
            else:
                result += f"❌ Lower risk-adjusted returns (Sharpe: {portfolio_sharpe:.3f} vs {benchmark_sharpe:.3f})\n"
            
            if max_drawdown > -0.20:  # Less than 20% drawdown
                result += f"✅ Reasonable drawdown control ({max_drawdown:.1%})\n"
            else:
                result += f"⚠️ High maximum drawdown ({max_drawdown:.1%})\n"
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error analyzing performance: {e}")
            return f"Error analyzing performance: {str(e)}"
    
    def _optimize_mean_variance(self, mean_returns: pd.Series, cov_matrix: pd.DataFrame, 
                               input_data: PortfolioManagementInput) -> pd.Series:
        """Optimize portfolio using mean-variance optimization."""
        n_assets = len(mean_returns)
        
        # Objective function (minimize negative Sharpe ratio)
        def objective(weights):
            portfolio_return = np.sum(weights * mean_returns)
            portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            if portfolio_vol == 0:
                return -np.inf
            return -(portfolio_return - self.risk_free_rate) / portfolio_vol
        
        # Constraints
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]  # Weights sum to 1
        
        # Bounds
        bounds = [(input_data.min_weight, input_data.max_weight) for _ in range(n_assets)]
        
        # Initial guess (equal weights)
        x0 = np.array([1.0 / n_assets] * n_assets)
        
        # Optimize
        result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)
        
        if result.success:
            return pd.Series(result.x, index=mean_returns.index)
        else:
            # No fallback - raise error if optimization fails
            raise RuntimeError(f"Portfolio optimization failed: {result.message}")
    
    def _optimize_risk_parity(self, cov_matrix: pd.DataFrame, 
                             input_data: PortfolioManagementInput) -> pd.Series:
        """Optimize portfolio using risk parity approach."""
        n_assets = len(cov_matrix)
        
        # Objective function (minimize sum of squared risk contribution differences)
        def objective(weights):
            portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            if portfolio_vol == 0:
                return np.inf
            
            # Risk contributions
            marginal_contrib = np.dot(cov_matrix, weights) / portfolio_vol
            risk_contrib = weights * marginal_contrib
            target_contrib = portfolio_vol / n_assets
            
            return np.sum((risk_contrib - target_contrib) ** 2)
        
        # Constraints
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
        
        # Bounds
        bounds = [(input_data.min_weight, input_data.max_weight) for _ in range(n_assets)]
        
        # Initial guess
        x0 = np.array([1.0 / n_assets] * n_assets)
        
        # Optimize
        result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)
        
        if result.success:
            return pd.Series(result.x, index=cov_matrix.index)
        else:
            return pd.Series([1.0 / n_assets] * n_assets, index=cov_matrix.index)
    
    def _optimize_minimum_variance(self, cov_matrix: pd.DataFrame, 
                                  input_data: PortfolioManagementInput) -> pd.Series:
        """Optimize portfolio for minimum variance."""
        n_assets = len(cov_matrix)
        
        # Objective function (minimize portfolio variance)
        def objective(weights):
            return np.dot(weights.T, np.dot(cov_matrix, weights))
        
        # Constraints
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
        
        # Bounds
        bounds = [(input_data.min_weight, input_data.max_weight) for _ in range(n_assets)]
        
        # Initial guess
        x0 = np.array([1.0 / n_assets] * n_assets)
        
        # Optimize
        result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)
        
        if result.success:
            return pd.Series(result.x, index=cov_matrix.index)
        else:
            return pd.Series([1.0 / n_assets] * n_assets, index=cov_matrix.index)
    
    def _optimize_maximum_sharpe(self, mean_returns: pd.Series, cov_matrix: pd.DataFrame, 
                                input_data: PortfolioManagementInput) -> pd.Series:
        """Optimize portfolio for maximum Sharpe ratio."""
        return self._optimize_mean_variance(mean_returns, cov_matrix, input_data)
    
    def _optimize_equal_weight(self, symbols: List[str]) -> pd.Series:
        """Create equal-weighted portfolio."""
        weight = 1.0 / len(symbols)
        return pd.Series([weight] * len(symbols), index=symbols)
    
    async def _get_returns_data(self, symbols: List[str], lookback_days: int) -> pd.DataFrame:
        """Get historical returns data for symbols."""
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=lookback_days + 30)  # Extra buffer
            
            all_data = {}
            
            for symbol in symbols:
                # Check cache first
                cache_key = f"{symbol}_{lookback_days}"
                if cache_key in self.returns_cache:
                    cached_data = self.returns_cache[cache_key]
                    if len(cached_data) >= lookback_days * 0.8:  # At least 80% of requested data
                        all_data[symbol] = cached_data
                        continue
                
                # Fetch data
                request = DataRequest(
                    symbol=symbol,
                    data_type="price",
                    start_date=start_date,
                    end_date=end_date
                )
                
                response = await self.data_manager.get_data(request)
                
                if not response.error and response.data is not None:
                    # Calculate returns
                    prices = response.data['close'] if 'close' in response.data.columns else response.data.iloc[:, 0]
                    returns = prices.pct_change().dropna()
                    
                    # Take last N days
                    returns = returns.tail(lookback_days)
                    
                    all_data[symbol] = returns
                    
                    # Cache results
                    self.returns_cache[cache_key] = returns
            
            if not all_data:
                return pd.DataFrame()
            
            # Combine all returns data
            returns_df = pd.DataFrame(all_data)
            returns_df = returns_df.dropna()
            
            return returns_df
            
        except Exception as e:
            self.logger.error(f"Error getting returns data: {e}")
            return pd.DataFrame()
    
    def _create_portfolio(self, portfolio_id: str, weights: pd.Series, symbols: List[str]) -> Portfolio:
        """Create portfolio object from optimization results."""
        positions = {}
        
        for symbol in symbols:
            weight = weights.get(symbol, 0.0)
            position = Position(
                symbol=symbol,
                quantity=0.0,  # Will be set when trades are executed
                current_price=0.0,  # Will be updated
                market_value=0.0,  # Will be calculated
                cost_basis=0.0,
                unrealized_pnl=0.0,
                realized_pnl=0.0,
                weight=weight,
                target_weight=weight
            )
            positions[symbol] = position
        
        portfolio = Portfolio(
            portfolio_id=portfolio_id,
            name=f"Optimized Portfolio {portfolio_id}",
            positions=positions,
            cash=0.0,
            total_value=0.0
        )
        
        return portfolio
    
    async def _update_portfolio_prices(self, portfolio: Portfolio):
        """Update current prices for all positions in portfolio."""
        try:
            for symbol, position in portfolio.positions.items():
                # Get current price
                request = DataRequest(
                    symbol=symbol,
                    data_type="price",
                    limit=1
                )
                
                response = await self.data_manager.get_data(request)
                
                if not response.error and response.data is not None:
                    current_price = float(response.data['close'].iloc[-1])
                    position.current_price = current_price
                    position.market_value = position.quantity * current_price
                    position.unrealized_pnl = position.market_value - position.cost_basis
            
            # Update portfolio totals
            portfolio.total_value = portfolio.cash + sum(pos.market_value for pos in portfolio.positions.values())
            
            # Update weights
            if portfolio.total_value > 0:
                for position in portfolio.positions.values():
                    position.weight = position.market_value / portfolio.total_value
            
            portfolio.last_updated = datetime.now()
            
        except Exception as e:
            self.logger.error(f"Error updating portfolio prices: {e}")
    
    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown from returns series."""
        try:
            cumulative = (1 + returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            return drawdown.min()
        except Exception as e:
            self.logger.error(f"Error calculating max drawdown: {e}")
            return 0.0


if __name__ == "__main__":
    # Example usage
    import asyncio
    from pathlib import Path
    from core.config_manager import ConfigManager
    from core.data_manager import UnifiedDataManager
    
    async def test_portfolio_management_tool():
        config_manager = ConfigManager(Path("../config"))
        data_manager = UnifiedDataManager(config_manager)
        
        tool = PortfolioManagementTool(data_manager)
        
        # Test portfolio optimization
        result = tool._run(
            action="optimize",
            symbols=["AAPL", "MSFT", "GOOGL", "AMZN"],
            optimization_method="mean_variance",
            lookback_days=252
        )
        
        print("Portfolio Optimization Result:")
        print(result)
    
    # Run test
    # asyncio.run(test_portfolio_management_tool())