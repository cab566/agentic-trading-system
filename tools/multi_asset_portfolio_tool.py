#!/usr/bin/env python3
"""
Multi-Asset Portfolio Optimization Tool for CrewAI Trading System

Provides comprehensive portfolio optimization across multiple asset classes:
- Traditional equities and bonds
- Cryptocurrencies
- Forex pairs
- Commodities
- Cross-asset correlation analysis
- Risk parity and factor-based optimization
- 24/7 portfolio monitoring
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum

import pandas as pd
import numpy as np
from scipy.optimize import minimize
from scipy import stats
from langchain_core.tools import BaseTool
from pydantic import Field

from core.data_manager import UnifiedDataManager
from core.config_manager import ConfigManager


class AssetClass(Enum):
    """Asset class enumeration."""
    EQUITY = "equity"
    BOND = "bond"
    CRYPTO = "crypto"
    FOREX = "forex"
    COMMODITY = "commodity"
    REIT = "reit"
    ALTERNATIVE = "alternative"


@dataclass
class AssetInfo:
    """Asset information."""
    symbol: str
    name: str
    asset_class: AssetClass
    currency: str
    market_cap: Optional[float] = None
    sector: Optional[str] = None
    country: Optional[str] = None
    trading_hours: str = "24/7"  # "24/7", "24/5", "market_hours"
    liquidity_score: float = 1.0  # 0 to 1
    

@dataclass
class PortfolioOptimizationResult:
    """Portfolio optimization result."""
    assets: List[AssetInfo]
    optimal_weights: Dict[str, float]
    expected_return: float
    expected_volatility: float
    sharpe_ratio: float
    max_drawdown: float
    
    # Risk metrics
    var_95: float  # Value at Risk (95%)
    cvar_95: float  # Conditional Value at Risk (95%)
    beta: float
    
    # Asset class allocation
    asset_class_weights: Dict[AssetClass, float]
    
    # Diversification metrics
    effective_assets: float
    concentration_ratio: float
    correlation_risk: float
    
    # Performance attribution
    return_attribution: Dict[str, float]
    risk_attribution: Dict[str, float]
    
    # Rebalancing recommendations
    rebalancing_frequency: str
    next_rebalance_date: datetime
    rebalancing_threshold: float
    
    # Market regime analysis
    current_regime: str
    regime_probability: float
    
    optimization_method: str
    constraints: Dict[str, Any]
    timestamp: datetime


class MultiAssetPortfolioTool(BaseTool):
    """
    Multi-asset portfolio optimization tool for trading agents.
    
    This tool provides comprehensive portfolio optimization across multiple
    asset classes with advanced risk management and 24/7 monitoring capabilities.
    """
    
    name: str = "multi_asset_portfolio_tool"
    description: str = (
        "Optimize portfolios across multiple asset classes including equities, bonds, "
        "crypto, forex, and commodities. Provides risk-adjusted allocations, "
        "diversification analysis, and rebalancing recommendations with 24/7 monitoring."
    )
    
    # Pydantic field declarations
    config_manager: ConfigManager = Field(exclude=True)
    data_manager: UnifiedDataManager = Field(exclude=True)
    logger: Any = Field(default=None, exclude=True)
    default_constraints: Dict[str, Any] = Field(default_factory=dict, exclude=True)
    risk_free_rate: float = Field(default=0.04, exclude=True)
    confidence_level: float = Field(default=0.95, exclude=True)
    lookback_period: int = Field(default=252, exclude=True)
    asset_universe: Dict[str, AssetInfo] = Field(default_factory=dict, exclude=True)
    
    def __init__(self, config_manager: ConfigManager, **kwargs):
        # Initialize components before calling super().__init__
        data_manager = UnifiedDataManager(config_manager)
        logger = logging.getLogger(__name__)
        
        # Initialize default constraints
        default_constraints = {
            'max_weight': 0.25,
            'min_weight': 0.01,
            'max_asset_class_weight': {
                AssetClass.EQUITY: 0.60,
                AssetClass.BOND: 0.40,
                AssetClass.CRYPTO: 0.15,
                AssetClass.FOREX: 0.10,
                AssetClass.COMMODITY: 0.20,
                AssetClass.REIT: 0.15,
                AssetClass.ALTERNATIVE: 0.10
            },
            'min_asset_class_weight': {
                AssetClass.EQUITY: 0.20,
                AssetClass.BOND: 0.10,
                AssetClass.CRYPTO: 0.00,
                AssetClass.FOREX: 0.00,
                AssetClass.COMMODITY: 0.00,
                AssetClass.REIT: 0.00,
                AssetClass.ALTERNATIVE: 0.00
            }
        }
        
        # Initialize asset universe
        asset_universe = self._initialize_asset_universe()
        
        # Call super().__init__ with all required fields
        super().__init__(
            config_manager=config_manager,
            data_manager=data_manager,
            logger=logger,
            default_constraints=default_constraints,
            asset_universe=asset_universe,
            **kwargs
        )

    
    async def _run(
        self,
        assets: str,
        optimization_method: str = "mean_variance",
        risk_target: Optional[float] = None,
        return_target: Optional[float] = None,
        constraints: Optional[str] = None
    ) -> str:
        """
        Run multi-asset portfolio optimization.
        
        Args:
            assets: Comma-separated list of asset symbols
            optimization_method: Optimization method ('mean_variance', 'risk_parity', 
                               'black_litterman', 'factor_based', 'robust')
            risk_target: Target portfolio volatility (optional)
            return_target: Target portfolio return (optional)
            constraints: Additional constraints as JSON string (optional)
        
        Returns:
            Formatted optimization report as string
        """
        try:
            # Parse assets
            asset_symbols = [s.strip().upper() for s in assets.split(',')]
            
            # Get asset information
            asset_infos = await self._get_asset_information(asset_symbols)
            if not asset_infos:
                return "Error: Could not retrieve asset information"
            
            # Get historical data
            price_data = await self._get_price_data(asset_symbols)
            if price_data.empty:
                return "Error: Could not retrieve price data"
            
            # Calculate returns and risk metrics
            returns_data = self._calculate_returns(price_data)
            risk_metrics = self._calculate_risk_metrics(returns_data)
            
            # Perform optimization
            optimization_result = await self._optimize_portfolio(
                asset_infos,
                returns_data,
                risk_metrics,
                optimization_method,
                risk_target,
                return_target,
                constraints
            )
            
            # Analyze market regime
            market_regime = self._analyze_market_regime(returns_data)
            
            # Generate rebalancing recommendations
            rebalancing_rec = self._generate_rebalancing_recommendations(
                optimization_result, market_regime
            )
            
            # Create final result
            result = PortfolioOptimizationResult(
                assets=asset_infos,
                optimal_weights=optimization_result['weights'],
                expected_return=optimization_result['expected_return'],
                expected_volatility=optimization_result['expected_volatility'],
                sharpe_ratio=optimization_result['sharpe_ratio'],
                max_drawdown=optimization_result['max_drawdown'],
                
                var_95=optimization_result['var_95'],
                cvar_95=optimization_result['cvar_95'],
                beta=optimization_result['beta'],
                
                asset_class_weights=optimization_result['asset_class_weights'],
                
                effective_assets=optimization_result['effective_assets'],
                concentration_ratio=optimization_result['concentration_ratio'],
                correlation_risk=optimization_result['correlation_risk'],
                
                return_attribution=optimization_result['return_attribution'],
                risk_attribution=optimization_result['risk_attribution'],
                
                rebalancing_frequency=rebalancing_rec['frequency'],
                next_rebalance_date=rebalancing_rec['next_date'],
                rebalancing_threshold=rebalancing_rec['threshold'],
                
                current_regime=market_regime['regime'],
                regime_probability=market_regime['probability'],
                
                optimization_method=optimization_method,
                constraints=self.default_constraints,
                timestamp=datetime.now()
            )
            
            return self._format_optimization_report(result)
            
        except Exception as e:
            self.logger.error(f"Error in portfolio optimization: {e}")
            return f"Error optimizing portfolio: {str(e)}"
    
    async def _get_asset_information(self, symbols: List[str]) -> List[AssetInfo]:
        """Get asset information for the given symbols."""
        asset_infos = []
        
        for symbol in symbols:
            try:
                # Determine asset class based on symbol
                asset_class = self._classify_asset(symbol)
                
                # Get basic info
                info = AssetInfo(
                    symbol=symbol,
                    name=self._get_asset_name(symbol),
                    asset_class=asset_class,
                    currency=self._get_asset_currency(symbol),
                    trading_hours=self._get_trading_hours(asset_class),
                    liquidity_score=self._get_liquidity_score(symbol)
                )
                
                asset_infos.append(info)
                
            except Exception as e:
                self.logger.error(f"Error getting info for {symbol}: {e}")
                continue
        
        return asset_infos
    
    async def _get_price_data(self, symbols: List[str]) -> pd.DataFrame:
        """Get historical price data for all assets."""
        price_data = pd.DataFrame()
        
        for symbol in symbols:
            try:
                # Get data from appropriate source based on asset type
                data = await self._get_asset_price_data(symbol)
                if data is not None and not data.empty:
                    price_data[symbol] = data['close']
                    
            except Exception as e:
                self.logger.error(f"Error getting price data for {symbol}: {e}")
                continue
        
        # Forward fill missing data and align dates
        price_data = price_data.fillna(method='ffill').dropna()
        
        return price_data
    
    def _calculate_returns(self, price_data: pd.DataFrame) -> pd.DataFrame:
        """Calculate returns from price data."""
        # Calculate daily returns
        returns = price_data.pct_change().dropna()
        
        # Handle crypto and forex (24/7 assets) differently
        # For now, use simple daily returns for all assets
        return returns
    
    def _calculate_risk_metrics(self, returns_data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate comprehensive risk metrics."""
        # Basic statistics
        mean_returns = returns_data.mean() * 252  # Annualized
        volatilities = returns_data.std() * np.sqrt(252)  # Annualized
        
        # Correlation matrix
        correlation_matrix = returns_data.corr()
        
        # Covariance matrix (annualized)
        covariance_matrix = returns_data.cov() * 252
        
        # Skewness and kurtosis
        skewness = returns_data.skew()
        kurtosis = returns_data.kurtosis()
        
        # Maximum drawdown for each asset
        max_drawdowns = {}
        for col in returns_data.columns:
            cumulative = (1 + returns_data[col]).cumprod()
            rolling_max = cumulative.expanding().max()
            drawdown = (cumulative - rolling_max) / rolling_max
            max_drawdowns[col] = drawdown.min()
        
        return {
            'mean_returns': mean_returns,
            'volatilities': volatilities,
            'correlation_matrix': correlation_matrix,
            'covariance_matrix': covariance_matrix,
            'skewness': skewness,
            'kurtosis': kurtosis,
            'max_drawdowns': max_drawdowns
        }
    
    async def _optimize_portfolio(
        self,
        assets: List[AssetInfo],
        returns_data: pd.DataFrame,
        risk_metrics: Dict[str, Any],
        method: str,
        risk_target: Optional[float],
        return_target: Optional[float],
        custom_constraints: Optional[str]
    ) -> Dict[str, Any]:
        """Perform portfolio optimization."""
        
        n_assets = len(assets)
        symbols = [asset.symbol for asset in assets]
        
        # Expected returns and covariance matrix
        expected_returns = risk_metrics['mean_returns'][symbols].values
        cov_matrix = risk_metrics['covariance_matrix'].loc[symbols, symbols].values
        
        # Optimization based on method
        if method == "mean_variance":
            weights = self._optimize_mean_variance(
                expected_returns, cov_matrix, risk_target, return_target
            )
        elif method == "risk_parity":
            weights = self._optimize_risk_parity(cov_matrix)
        elif method == "black_litterman":
            weights = self._optimize_black_litterman(
                expected_returns, cov_matrix, assets
            )
        elif method == "factor_based":
            weights = self._optimize_factor_based(
                returns_data[symbols], assets
            )
        elif method == "robust":
            weights = self._optimize_robust(
                returns_data[symbols], expected_returns, cov_matrix
            )
        else:
            raise ValueError(f"Unknown optimization method: {method}")
        
        # Calculate portfolio metrics
        portfolio_return = np.sum(weights * expected_returns)
        portfolio_volatility = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))
        sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_volatility
        
        # Calculate VaR and CVaR
        portfolio_returns = np.dot(returns_data[symbols].values, weights)
        var_95 = np.percentile(portfolio_returns, 5)
        cvar_95 = portfolio_returns[portfolio_returns <= var_95].mean()
        
        # Calculate maximum drawdown
        cumulative_returns = (1 + pd.Series(portfolio_returns)).cumprod()
        rolling_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        # Calculate beta (vs market proxy)
        market_returns = self._get_market_proxy_returns(returns_data)
        if market_returns is not None:
            beta = np.cov(portfolio_returns, market_returns)[0, 1] / np.var(market_returns)
        else:
            beta = 1.0
        
        # Asset class weights
        asset_class_weights = self._calculate_asset_class_weights(assets, weights)
        
        # Diversification metrics
        effective_assets = 1 / np.sum(weights ** 2)
        concentration_ratio = np.sum(np.sort(weights)[-5:])  # Top 5 concentration
        correlation_risk = self._calculate_correlation_risk(weights, risk_metrics['correlation_matrix'].loc[symbols, symbols])
        
        # Attribution analysis
        return_attribution = {symbol: weight * ret for symbol, weight, ret in zip(symbols, weights, expected_returns)}
        risk_attribution = self._calculate_risk_attribution(weights, cov_matrix, symbols)
        
        return {
            'weights': dict(zip(symbols, weights)),
            'expected_return': portfolio_return,
            'expected_volatility': portfolio_volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'var_95': var_95,
            'cvar_95': cvar_95,
            'beta': beta,
            'asset_class_weights': asset_class_weights,
            'effective_assets': effective_assets,
            'concentration_ratio': concentration_ratio,
            'correlation_risk': correlation_risk,
            'return_attribution': return_attribution,
            'risk_attribution': risk_attribution
        }
    
    def _optimize_mean_variance(
        self,
        expected_returns: np.ndarray,
        cov_matrix: np.ndarray,
        risk_target: Optional[float],
        return_target: Optional[float]
    ) -> np.ndarray:
        """Mean-variance optimization."""
        n_assets = len(expected_returns)
        
        # Objective function
        def objective(weights):
            portfolio_return = np.sum(weights * expected_returns)
            portfolio_variance = np.dot(weights, np.dot(cov_matrix, weights))
            
            if return_target is not None:
                # Minimize variance subject to return target
                return portfolio_variance
            elif risk_target is not None:
                # Maximize return subject to risk target
                return -portfolio_return
            else:
                # Maximize Sharpe ratio
                portfolio_volatility = np.sqrt(portfolio_variance)
                return -(portfolio_return - self.risk_free_rate) / portfolio_volatility
        
        # Constraints
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]  # Weights sum to 1
        
        if return_target is not None:
            constraints.append({
                'type': 'eq',
                'fun': lambda x: np.sum(x * expected_returns) - return_target
            })
        
        if risk_target is not None:
            constraints.append({
                'type': 'eq',
                'fun': lambda x: np.sqrt(np.dot(x, np.dot(cov_matrix, x))) - risk_target
            })
        
        # Bounds
        bounds = [(self.default_constraints['min_weight'], 
                  self.default_constraints['max_weight']) for _ in range(n_assets)]
        
        # Initial guess
        x0 = np.array([1/n_assets] * n_assets)
        
        # Optimize
        result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)
        
        if result.success:
            return result.x
        else:
            self.logger.warning("Optimization failed, using equal weights")
            return np.array([1/n_assets] * n_assets)
    
    def _optimize_risk_parity(self, cov_matrix: np.ndarray) -> np.ndarray:
        """Risk parity optimization."""
        n_assets = cov_matrix.shape[0]
        
        def objective(weights):
            # Risk contribution of each asset
            portfolio_vol = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))
            marginal_contrib = np.dot(cov_matrix, weights) / portfolio_vol
            contrib = weights * marginal_contrib
            
            # Minimize sum of squared deviations from equal risk contribution
            target_contrib = portfolio_vol / n_assets
            return np.sum((contrib - target_contrib) ** 2)
        
        # Constraints and bounds
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
        bounds = [(self.default_constraints['min_weight'], 
                  self.default_constraints['max_weight']) for _ in range(n_assets)]
        
        x0 = np.array([1/n_assets] * n_assets)
        
        result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)
        
        if result.success:
            return result.x
        else:
            return np.array([1/n_assets] * n_assets)
    
    def _optimize_black_litterman(
        self,
        expected_returns: np.ndarray,
        cov_matrix: np.ndarray,
        assets: List[AssetInfo]
    ) -> np.ndarray:
        """Black-Litterman optimization (simplified)."""
        # For now, use mean-variance as placeholder
        # Full Black-Litterman would require market cap weights and views
        return self._optimize_mean_variance(expected_returns, cov_matrix, None, None)
    
    def _optimize_factor_based(
        self,
        returns_data: pd.DataFrame,
        assets: List[AssetInfo]
    ) -> np.ndarray:
        """Factor-based optimization."""
        # Simplified factor-based approach
        # Equal weight within each asset class, then optimize across asset classes
        
        asset_classes = {}
        for i, asset in enumerate(assets):
            if asset.asset_class not in asset_classes:
                asset_classes[asset.asset_class] = []
            asset_classes[asset.asset_class].append(i)
        
        n_assets = len(assets)
        weights = np.zeros(n_assets)
        
        # Equal weight within asset classes
        for asset_class, indices in asset_classes.items():
            class_weight = 1.0 / len(asset_classes)  # Equal weight across asset classes
            asset_weight = class_weight / len(indices)  # Equal weight within class
            
            for idx in indices:
                weights[idx] = asset_weight
        
        return weights
    
    def _optimize_robust(
        self,
        returns_data: pd.DataFrame,
        expected_returns: np.ndarray,
        cov_matrix: np.ndarray
    ) -> np.ndarray:
        """Robust optimization using shrinkage estimators."""
        # Use shrinkage estimator for covariance matrix
        from sklearn.covariance import LedoitWolf
        
        lw = LedoitWolf()
        shrunk_cov = lw.fit(returns_data.values).covariance_ * 252  # Annualize
        
        # Use shrunk covariance in mean-variance optimization
        return self._optimize_mean_variance(expected_returns, shrunk_cov, None, None)
    
    def _analyze_market_regime(self, returns_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze current market regime."""
        # Simple regime detection based on volatility and correlation
        recent_returns = returns_data.tail(30)  # Last 30 days
        
        avg_volatility = recent_returns.std().mean()
        avg_correlation = recent_returns.corr().values[np.triu_indices_from(recent_returns.corr().values, k=1)].mean()
        
        # Simple regime classification
        if avg_volatility > 0.02 and avg_correlation > 0.7:
            regime = "CRISIS"
            probability = 0.8
        elif avg_volatility > 0.015:
            regime = "HIGH_VOLATILITY"
            probability = 0.7
        elif avg_correlation > 0.6:
            regime = "HIGH_CORRELATION"
            probability = 0.6
        else:
            regime = "NORMAL"
            probability = 0.5
        
        return {
            'regime': regime,
            'probability': probability,
            'avg_volatility': avg_volatility,
            'avg_correlation': avg_correlation
        }
    
    def _generate_rebalancing_recommendations(
        self,
        optimization_result: Dict[str, Any],
        market_regime: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate rebalancing recommendations."""
        
        # Adjust frequency based on market regime
        if market_regime['regime'] == 'CRISIS':
            frequency = 'WEEKLY'
            threshold = 0.02  # 2% threshold
            days_to_next = 7
        elif market_regime['regime'] == 'HIGH_VOLATILITY':
            frequency = 'BI_WEEKLY'
            threshold = 0.03  # 3% threshold
            days_to_next = 14
        else:
            frequency = 'MONTHLY'
            threshold = 0.05  # 5% threshold
            days_to_next = 30
        
        next_rebalance_date = datetime.now() + timedelta(days=days_to_next)
        
        return {
            'frequency': frequency,
            'threshold': threshold,
            'next_date': next_rebalance_date
        }
    
    def _format_optimization_report(self, result: PortfolioOptimizationResult) -> str:
        """Format the optimization result into a readable report."""
        report = f"""
🎯 MULTI-ASSET PORTFOLIO OPTIMIZATION REPORT
{'=' * 60}

📊 PORTFOLIO OVERVIEW
Optimization Method: {result.optimization_method.replace('_', ' ').title()}
Number of Assets: {len(result.assets)}
Optimization Time: {result.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}

💰 EXPECTED PERFORMANCE
Expected Annual Return: {result.expected_return:.2%}
Expected Annual Volatility: {result.expected_volatility:.2%}
Sharpe Ratio: {result.sharpe_ratio:.2f}
Maximum Drawdown: {result.max_drawdown:.2%}

⚠️ RISK METRICS
Value at Risk (95%): {result.var_95:.2%}
Conditional VaR (95%): {result.cvar_95:.2%}
Beta: {result.beta:.2f}

🏗️ ASSET ALLOCATION
"""
        
        # Sort assets by weight for display
        sorted_weights = sorted(result.optimal_weights.items(), key=lambda x: x[1], reverse=True)
        
        for symbol, weight in sorted_weights:
            asset_info = next((a for a in result.assets if a.symbol == symbol), None)
            asset_class = asset_info.asset_class.value.title() if asset_info else "Unknown"
            report += f"{symbol:>8}: {weight:>6.1%} ({asset_class})\n"
        
        report += f"""
🎨 ASSET CLASS ALLOCATION
"""
        
        for asset_class, weight in result.asset_class_weights.items():
            if weight > 0:
                report += f"{asset_class.value.title():>12}: {weight:>6.1%}\n"
        
        report += f"""
📈 DIVERSIFICATION METRICS
Effective Number of Assets: {result.effective_assets:.1f}
Concentration Ratio (Top 5): {result.concentration_ratio:.1%}
Correlation Risk Score: {result.correlation_risk:.2f}

🔄 REBALANCING RECOMMENDATIONS
Frequency: {result.rebalancing_frequency}
Next Rebalance: {result.next_rebalance_date.strftime('%Y-%m-%d')}
Rebalancing Threshold: {result.rebalancing_threshold:.1%}

🌍 MARKET REGIME ANALYSIS
Current Regime: {result.current_regime}
Regime Probability: {result.regime_probability:.1%}

📊 RETURN ATTRIBUTION (Annual)
"""
        
        sorted_attribution = sorted(result.return_attribution.items(), key=lambda x: x[1], reverse=True)
        for symbol, contribution in sorted_attribution:
            report += f"{symbol:>8}: {contribution:>+6.2%}\n"
        
        report += f"""
⚖️ RISK ATTRIBUTION
"""
        
        sorted_risk_attribution = sorted(result.risk_attribution.items(), key=lambda x: x[1], reverse=True)
        for symbol, contribution in sorted_risk_attribution[:10]:  # Top 10
            report += f"{symbol:>8}: {contribution:>6.1%}\n"
        
        report += f"""
🕐 TRADING CONSIDERATIONS
24/7 Assets: {len([a for a in result.assets if a.trading_hours == '24/7'])}
24/5 Assets: {len([a for a in result.assets if a.trading_hours == '24/5'])}
Market Hours Only: {len([a for a in result.assets if a.trading_hours == 'market_hours'])}

⚠️ IMPORTANT DISCLAIMERS
- Past performance does not guarantee future results
- All investments carry risk of loss
- Consider transaction costs and taxes in implementation
- Monitor portfolio regularly and rebalance as recommended
- Diversification does not guarantee against loss
- Consider your risk tolerance and investment objectives

📋 IMPLEMENTATION NOTES
- Use limit orders to minimize market impact
- Consider fractional shares for precise allocation
- Monitor correlation changes during market stress
- Adjust allocations based on changing market conditions
- Review and update optimization parameters regularly
"""
        
        return report
    
    # Helper methods
    def _initialize_asset_universe(self) -> Dict[str, AssetInfo]:
        """Initialize the asset universe."""
        # This would be populated from a database or configuration
        return {}
    
    def _classify_asset(self, symbol: str) -> AssetClass:
        """Classify asset based on symbol."""
        if any(crypto in symbol.upper() for crypto in ['BTC', 'ETH', 'ADA', 'DOT', 'LINK']):
            return AssetClass.CRYPTO
        elif '/' in symbol or any(fx in symbol for fx in ['EUR', 'GBP', 'JPY', 'CHF', 'AUD', 'CAD', 'NZD']):
            return AssetClass.FOREX
        elif any(bond in symbol.upper() for bond in ['TLT', 'IEF', 'SHY', 'LQD', 'HYG']):
            return AssetClass.BOND
        elif any(reit in symbol.upper() for reit in ['VNQ', 'REIT']):
            return AssetClass.REIT
        elif any(commodity in symbol.upper() for commodity in ['GLD', 'SLV', 'USO', 'DBA']):
            return AssetClass.COMMODITY
        else:
            return AssetClass.EQUITY
    
    def _get_asset_name(self, symbol: str) -> str:
        """Get asset name."""
        # Placeholder - would use actual asset database
        return symbol
    
    def _get_asset_currency(self, symbol: str) -> str:
        """Get asset base currency."""
        if '/' in symbol:
            return symbol.split('/')[1]
        else:
            return 'USD'  # Default
    
    def _get_trading_hours(self, asset_class: AssetClass) -> str:
        """Get trading hours for asset class."""
        if asset_class == AssetClass.CRYPTO:
            return "24/7"
        elif asset_class == AssetClass.FOREX:
            return "24/5"
        else:
            return "market_hours"
    
    def _get_liquidity_score(self, symbol: str) -> float:
        """Get liquidity score for asset."""
        # Placeholder - would use actual liquidity metrics
        return 0.8
    
    async def _get_asset_price_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """Get price data for a specific asset."""
        try:
            # Use the unified data manager to get data
            data = await self.data_manager.get_historical_data(
                symbol, '1d', limit=self.lookback_period
            )
            return data
        except Exception as e:
            self.logger.error(f"Error getting price data for {symbol}: {e}")
            return None
    
    def _get_market_proxy_returns(self, returns_data: pd.DataFrame) -> Optional[np.ndarray]:
        """Get market proxy returns for beta calculation."""
        # Use SPY as market proxy if available
        if 'SPY' in returns_data.columns:
            return returns_data['SPY'].values
        else:
            # Use equal-weighted portfolio as proxy
            return returns_data.mean(axis=1).values
    
    def _calculate_asset_class_weights(self, assets: List[AssetInfo], weights: np.ndarray) -> Dict[AssetClass, float]:
        """Calculate asset class weights."""
        class_weights = {}
        
        for asset, weight in zip(assets, weights):
            if asset.asset_class not in class_weights:
                class_weights[asset.asset_class] = 0
            class_weights[asset.asset_class] += weight
        
        return class_weights
    
    def _calculate_correlation_risk(self, weights: np.ndarray, correlation_matrix: pd.DataFrame) -> float:
        """Calculate correlation risk score."""
        # Weighted average correlation
        weighted_corr = 0
        total_weight = 0
        
        for i in range(len(weights)):
            for j in range(i+1, len(weights)):
                corr = correlation_matrix.iloc[i, j]
                weight_product = weights[i] * weights[j]
                weighted_corr += abs(corr) * weight_product
                total_weight += weight_product
        
        return weighted_corr / total_weight if total_weight > 0 else 0
    
    def _calculate_risk_attribution(self, weights: np.ndarray, cov_matrix: np.ndarray, symbols: List[str]) -> Dict[str, float]:
        """Calculate risk attribution for each asset."""
        portfolio_variance = np.dot(weights, np.dot(cov_matrix, weights))
        marginal_contrib = np.dot(cov_matrix, weights)
        risk_contrib = weights * marginal_contrib / portfolio_variance
        
        return dict(zip(symbols, risk_contrib))


if __name__ == "__main__":
    # Test the tool
    import asyncio
    from pathlib import Path
    
    async def test_portfolio_optimization():
        config_manager = ConfigManager(Path("../config"))
        tool = MultiAssetPortfolioTool(config_manager)
        
        result = await tool._run(
            'SPY,TLT,GLD,BTC-USD,EUR/USD',
            'mean_variance',
            None,
            None,
            None
        )
        print(result)
    
    # Commented out to prevent event loop conflicts when imported
    # asyncio.run(test_portfolio_optimization())