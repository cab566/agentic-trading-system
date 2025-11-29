#!/usr/bin/env python3
"""
Risk Analysis Tool for CrewAI Trading System

Provides agents with comprehensive risk assessment capabilities
including portfolio risk, position sizing, and risk metrics.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union, Tuple

import pandas as pd
import numpy as np
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
from scipy import stats
from sklearn.covariance import LedoitWolf

from core.data_manager import UnifiedDataManager, DataRequest


class RiskAnalysisInput(BaseModel):
    """Input schema for risk analysis requests."""
    analysis_type: str = Field(
        default="portfolio",
        description="Type of analysis: 'portfolio', 'position', 'var', 'correlation', 'drawdown', 'stress_test'"
    )
    symbols: List[str] = Field(
        ...,
        description="List of symbols to analyze (e.g., ['AAPL', 'MSFT', 'GOOGL'])"
    )
    weights: Optional[List[float]] = Field(
        default=None,
        description="Portfolio weights (must sum to 1.0). If not provided, equal weights assumed."
    )
    position_size: Optional[float] = Field(
        default=None,
        description="Position size in dollars for position risk analysis"
    )
    portfolio_value: Optional[float] = Field(
        default=100000.0,
        description="Total portfolio value in dollars"
    )
    risk_free_rate: float = Field(
        default=0.02,
        description="Risk-free rate for Sharpe ratio calculations (annual)"
    )
    confidence_level: float = Field(
        default=0.95,
        description="Confidence level for VaR calculations (0.90, 0.95, 0.99)"
    )
    time_horizon: int = Field(
        default=252,
        description="Time horizon in days for risk projections"
    )
    lookback_period: str = Field(
        default="1y",
        description="Lookback period for historical data: '3mo', '6mo', '1y', '2y'"
    )
    stress_scenarios: Optional[List[str]] = Field(
        default=None,
        description="Stress test scenarios: ['market_crash', 'interest_rate_shock', 'sector_rotation']"
    )


class RiskAnalysisTool(BaseTool):
    """
    Risk Analysis Tool for CrewAI agents.
    
    Provides comprehensive risk analysis including:
    - Portfolio risk metrics (volatility, VaR, CVaR)
    - Position sizing recommendations
    - Correlation analysis
    - Drawdown analysis
    - Stress testing
    - Risk-adjusted returns (Sharpe, Sortino, Calmar ratios)
    - Monte Carlo simulations
    """
    
    name: str = "risk_analysis_tool"
    description: str = (
        "Perform comprehensive risk analysis including portfolio risk metrics, "
        "VaR calculations, correlation analysis, and stress testing. Use this tool "
        "to assess portfolio risk, optimize position sizes, and evaluate risk-adjusted returns."
    )
    args_schema: type[RiskAnalysisInput] = RiskAnalysisInput
    
    # Pydantic v2 field declarations
    data_manager: UnifiedDataManager = Field(exclude=True)
    logger: logging.Logger = Field(default_factory=lambda: logging.getLogger(__name__), exclude=True)
    risk_cache: Dict[str, Any] = Field(default_factory=dict, exclude=True)
    cache_ttl: int = Field(default=600, exclude=True)
    
    def __init__(self, data_manager: UnifiedDataManager, **kwargs):
        """
        Initialize the risk analysis tool.
        
        Args:
            data_manager: Unified data manager instance
        """
        # Initialize with data_manager as a field
        super().__init__(data_manager=data_manager, **kwargs)
        self.logger = logging.getLogger(__name__)
        
        # Risk analysis cache
        self.risk_cache = {}
        self.cache_ttl = 600  # 10 minutes
    
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
            self.logger.error(f"Error in risk analysis tool: {e}")
            return f"Error performing risk analysis: {str(e)}"
    
    async def _arun(self, **kwargs) -> str:
        """Asynchronous execution of risk analysis."""
        try:
            # Parse input
            input_data = RiskAnalysisInput(**kwargs)
            
            # Validate inputs
            if not input_data.symbols:
                return "Error: No symbols provided for analysis"
            
            # Clean symbols
            symbols = [symbol.upper().replace('$', '') for symbol in input_data.symbols]
            
            # Validate weights
            if input_data.weights:
                if len(input_data.weights) != len(symbols):
                    return "Error: Number of weights must match number of symbols"
                if abs(sum(input_data.weights) - 1.0) > 0.01:
                    return "Error: Weights must sum to 1.0"
            else:
                # Equal weights
                input_data.weights = [1.0 / len(symbols)] * len(symbols)
            
            # Get market data for all symbols
            market_data = await self._get_portfolio_data(symbols, input_data.lookback_period)
            if isinstance(market_data, str):  # Error message
                return market_data
            
            # Perform risk analysis
            analysis_result = await self._perform_risk_analysis(market_data, input_data, symbols)
            
            return analysis_result
            
        except Exception as e:
            self.logger.error(f"Error in async risk analysis: {e}")
            return f"Error processing risk analysis request: {str(e)}"
    
    async def _get_portfolio_data(self, symbols: List[str], lookback_period: str) -> Union[Dict[str, pd.DataFrame], str]:
        """Get market data for all symbols in the portfolio."""
        try:
            # Calculate date range
            end_date = datetime.now()
            
            if lookback_period == '3mo':
                start_date = end_date - timedelta(days=90)
            elif lookback_period == '6mo':
                start_date = end_date - timedelta(days=180)
            elif lookback_period == '1y':
                start_date = end_date - timedelta(days=365)
            elif lookback_period == '2y':
                start_date = end_date - timedelta(days=730)
            else:
                start_date = end_date - timedelta(days=365)  # Default to 1 year
            
            portfolio_data = {}
            
            for symbol in symbols:
                # Create data request
                request = DataRequest(
                    symbol=symbol,
                    data_type="ohlc",
                    timeframe="1d",
                    start_date=start_date,
                    end_date=end_date
                )
                
                # Get data
                response = await self.data_manager.get_data(request)
                
                if response.error:
                    self.logger.warning(f"Error fetching data for {symbol}: {response.error}")
                    continue
                
                if response.data is None or response.data.empty:
                    self.logger.warning(f"No data available for {symbol}")
                    continue
                
                portfolio_data[symbol] = response.data
            
            if not portfolio_data:
                return "Error: No data available for any of the provided symbols"
            
            return portfolio_data
            
        except Exception as e:
            self.logger.error(f"Error getting portfolio data: {e}")
            return f"Error retrieving portfolio data: {str(e)}"
    
    async def _perform_risk_analysis(self, portfolio_data: Dict[str, pd.DataFrame], 
                                   input_data: RiskAnalysisInput, symbols: List[str]) -> str:
        """Perform the requested risk analysis."""
        try:
            result = f"Risk Analysis Report\n"
            result += f"Analysis Type: {input_data.analysis_type.title()}\n"
            result += f"Symbols: {', '.join(symbols)}\n"
            result += f"Portfolio Value: ${input_data.portfolio_value:,.2f}\n"
            result += f"Lookback Period: {input_data.lookback_period}\n"
            result += f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            
            # Create returns matrix
            returns_data = self._create_returns_matrix(portfolio_data)
            if returns_data.empty:
                return "Error: Unable to calculate returns from the provided data"
            
            # Portfolio weights
            weights = np.array(input_data.weights)
            
            # Perform analysis based on type
            if input_data.analysis_type in ['portfolio', 'comprehensive']:
                portfolio_result = self._analyze_portfolio_risk(returns_data, weights, input_data)
                result += "=== PORTFOLIO RISK ANALYSIS ===\n" + portfolio_result + "\n"
            
            if input_data.analysis_type in ['var', 'comprehensive']:
                var_result = self._calculate_var_metrics(returns_data, weights, input_data)
                result += "=== VALUE AT RISK ANALYSIS ===\n" + var_result + "\n"
            
            if input_data.analysis_type in ['correlation', 'comprehensive']:
                correlation_result = self._analyze_correlations(returns_data)
                result += "=== CORRELATION ANALYSIS ===\n" + correlation_result + "\n"
            
            if input_data.analysis_type in ['drawdown', 'comprehensive']:
                drawdown_result = self._analyze_drawdowns(returns_data, weights)
                result += "=== DRAWDOWN ANALYSIS ===\n" + drawdown_result + "\n"
            
            if input_data.analysis_type == 'position' and input_data.position_size:
                position_result = self._analyze_position_risk(returns_data, input_data)
                result += "=== POSITION RISK ANALYSIS ===\n" + position_result + "\n"
            
            if input_data.analysis_type in ['stress_test', 'comprehensive'] and input_data.stress_scenarios:
                stress_result = self._perform_stress_tests(returns_data, weights, input_data)
                result += "=== STRESS TEST ANALYSIS ===\n" + stress_result + "\n"
            
            # Risk-adjusted performance metrics
            performance_result = self._calculate_risk_adjusted_metrics(returns_data, weights, input_data)
            result += "=== RISK-ADJUSTED PERFORMANCE ===\n" + performance_result + "\n"
            
            # Risk recommendations
            recommendations = self._generate_risk_recommendations(returns_data, weights, input_data)
            result += "=== RISK MANAGEMENT RECOMMENDATIONS ===\n" + recommendations
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error performing risk analysis: {e}")
            return f"Error in risk analysis: {str(e)}"
    
    def _create_returns_matrix(self, portfolio_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Create a matrix of returns for all symbols."""
        try:
            returns_dict = {}
            
            for symbol, data in portfolio_data.items():
                if 'Close' in data.columns and len(data) > 1:
                    returns = data['Close'].pct_change().dropna()
                    returns_dict[symbol] = returns
            
            if not returns_dict:
                return pd.DataFrame()
            
            # Align all returns to common dates
            returns_df = pd.DataFrame(returns_dict)
            returns_df = returns_df.dropna()
            
            return returns_df
            
        except Exception as e:
            self.logger.error(f"Error creating returns matrix: {e}")
            return pd.DataFrame()
    
    def _analyze_portfolio_risk(self, returns_data: pd.DataFrame, weights: np.ndarray, 
                              input_data: RiskAnalysisInput) -> str:
        """Analyze portfolio-level risk metrics."""
        try:
            result = ""
            
            # Portfolio returns
            portfolio_returns = (returns_data * weights).sum(axis=1)
            
            # Basic statistics
            daily_return = portfolio_returns.mean()
            daily_volatility = portfolio_returns.std()
            annualized_return = daily_return * 252
            annualized_volatility = daily_volatility * np.sqrt(252)
            
            result += f"Daily Average Return: {daily_return * 100:.3f}%\n"
            result += f"Daily Volatility: {daily_volatility * 100:.3f}%\n"
            result += f"Annualized Return: {annualized_return * 100:.2f}%\n"
            result += f"Annualized Volatility: {annualized_volatility * 100:.2f}%\n\n"
            
            # Risk metrics
            if len(portfolio_returns) > 0:
                # Skewness and Kurtosis
                skewness = stats.skew(portfolio_returns)
                kurtosis = stats.kurtosis(portfolio_returns)
                
                result += f"Skewness: {skewness:.3f}\n"
                result += f"Excess Kurtosis: {kurtosis:.3f}\n\n"
                
                # Downside metrics
                negative_returns = portfolio_returns[portfolio_returns < 0]
                if len(negative_returns) > 0:
                    downside_deviation = negative_returns.std()
                    result += f"Downside Deviation: {downside_deviation * 100:.3f}%\n"
                    result += f"Annualized Downside Dev: {downside_deviation * np.sqrt(252) * 100:.2f}%\n"
                
                # Maximum and minimum returns
                max_daily_return = portfolio_returns.max()
                min_daily_return = portfolio_returns.min()
                result += f"Best Daily Return: {max_daily_return * 100:.2f}%\n"
                result += f"Worst Daily Return: {min_daily_return * 100:.2f}%\n\n"
            
            # Individual asset contributions to risk
            if len(returns_data.columns) > 1:
                covariance_matrix = returns_data.cov().values
                portfolio_variance = np.dot(weights, np.dot(covariance_matrix, weights))
                
                # Marginal contribution to risk
                marginal_contrib = np.dot(covariance_matrix, weights) / np.sqrt(portfolio_variance)
                risk_contrib = weights * marginal_contrib
                
                result += "Risk Contribution by Asset:\n"
                for i, symbol in enumerate(returns_data.columns):
                    result += f"{symbol}: {risk_contrib[i] * 100:.2f}%\n"
                result += "\n"
            
            # Concentration risk
            herfindahl_index = np.sum(weights ** 2)
            effective_assets = 1 / herfindahl_index
            result += f"Portfolio Concentration (Herfindahl): {herfindahl_index:.3f}\n"
            result += f"Effective Number of Assets: {effective_assets:.2f}\n"
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error analyzing portfolio risk: {e}")
            return f"Error in portfolio risk analysis: {str(e)}\n"
    
    def _calculate_var_metrics(self, returns_data: pd.DataFrame, weights: np.ndarray, 
                             input_data: RiskAnalysisInput) -> str:
        """Calculate Value at Risk and related metrics."""
        try:
            result = ""
            
            # Portfolio returns
            portfolio_returns = (returns_data * weights).sum(axis=1)
            
            if len(portfolio_returns) == 0:
                return "No portfolio returns data available\n"
            
            # Historical VaR
            confidence_levels = [0.90, 0.95, 0.99]
            
            result += "Historical Value at Risk:\n"
            for conf_level in confidence_levels:
                var_percentile = (1 - conf_level) * 100
                historical_var = np.percentile(portfolio_returns, var_percentile)
                var_dollar = historical_var * input_data.portfolio_value
                
                result += f"VaR ({conf_level*100:.0f}%): {historical_var * 100:.2f}% (${var_dollar:,.2f})\n"
                
                # Conditional VaR (Expected Shortfall)
                cvar = portfolio_returns[portfolio_returns <= historical_var].mean()
                cvar_dollar = cvar * input_data.portfolio_value
                result += f"CVaR ({conf_level*100:.0f}%): {cvar * 100:.2f}% (${cvar_dollar:,.2f})\n\n"
            
            # Parametric VaR (assuming normal distribution)
            mean_return = portfolio_returns.mean()
            std_return = portfolio_returns.std()
            
            result += "Parametric Value at Risk (Normal Distribution):\n"
            for conf_level in confidence_levels:
                z_score = stats.norm.ppf(1 - conf_level)
                parametric_var = mean_return + z_score * std_return
                var_dollar = parametric_var * input_data.portfolio_value
                
                result += f"Parametric VaR ({conf_level*100:.0f}%): {parametric_var * 100:.2f}% (${var_dollar:,.2f})\n"
            
            result += "\n"
            
            # Monte Carlo VaR (simplified)
            if len(portfolio_returns) >= 30:
                mc_var_result = self._monte_carlo_var(portfolio_returns, input_data)
                result += "Monte Carlo Value at Risk:\n" + mc_var_result + "\n"
            
            # VaR backtesting
            backtest_result = self._backtest_var(portfolio_returns, input_data.confidence_level)
            result += "VaR Backtesting:\n" + backtest_result
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error calculating VaR metrics: {e}")
            return f"Error in VaR calculation: {str(e)}\n"
    
    def _analyze_correlations(self, returns_data: pd.DataFrame) -> str:
        """Analyze correlations between assets."""
        try:
            result = ""
            
            if len(returns_data.columns) < 2:
                return "Need at least 2 assets for correlation analysis\n"
            
            # Correlation matrix
            correlation_matrix = returns_data.corr()
            
            result += "Correlation Matrix:\n"
            result += correlation_matrix.round(3).to_string() + "\n\n"
            
            # Average correlation
            # Get upper triangle of correlation matrix (excluding diagonal)
            mask = np.triu(np.ones_like(correlation_matrix, dtype=bool), k=1)
            correlations = correlation_matrix.values[mask]
            avg_correlation = np.mean(correlations)
            
            result += f"Average Pairwise Correlation: {avg_correlation:.3f}\n"
            result += f"Minimum Correlation: {np.min(correlations):.3f}\n"
            result += f"Maximum Correlation: {np.max(correlations):.3f}\n\n"
            
            # Highly correlated pairs
            high_corr_threshold = 0.7
            low_corr_threshold = -0.3
            
            high_corr_pairs = []
            low_corr_pairs = []
            
            for i in range(len(correlation_matrix.columns)):
                for j in range(i+1, len(correlation_matrix.columns)):
                    corr_value = correlation_matrix.iloc[i, j]
                    pair = f"{correlation_matrix.columns[i]} - {correlation_matrix.columns[j]}"
                    
                    if corr_value > high_corr_threshold:
                        high_corr_pairs.append((pair, corr_value))
                    elif corr_value < low_corr_threshold:
                        low_corr_pairs.append((pair, corr_value))
            
            if high_corr_pairs:
                result += f"Highly Correlated Pairs (>{high_corr_threshold}):\n"
                for pair, corr in high_corr_pairs:
                    result += f"{pair}: {corr:.3f}\n"
                result += "\n"
            
            if low_corr_pairs:
                result += f"Negatively Correlated Pairs (<{low_corr_threshold}):\n"
                for pair, corr in low_corr_pairs:
                    result += f"{pair}: {corr:.3f}\n"
                result += "\n"
            
            # Diversification ratio
            weights_equal = np.ones(len(returns_data.columns)) / len(returns_data.columns)
            individual_volatilities = returns_data.std().values
            weighted_avg_vol = np.sum(weights_equal * individual_volatilities)
            
            portfolio_vol = np.sqrt(np.dot(weights_equal, np.dot(returns_data.cov().values, weights_equal)))
            diversification_ratio = weighted_avg_vol / portfolio_vol
            
            result += f"Diversification Ratio: {diversification_ratio:.3f}\n"
            result += f"(Higher values indicate better diversification)\n"
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error analyzing correlations: {e}")
            return f"Error in correlation analysis: {str(e)}\n"
    
    def _analyze_drawdowns(self, returns_data: pd.DataFrame, weights: np.ndarray) -> str:
        """Analyze drawdown characteristics."""
        try:
            result = ""
            
            # Portfolio returns
            portfolio_returns = (returns_data * weights).sum(axis=1)
            
            if len(portfolio_returns) == 0:
                return "No portfolio returns data available\n"
            
            # Calculate cumulative returns and drawdowns
            cumulative_returns = (1 + portfolio_returns).cumprod()
            rolling_max = cumulative_returns.expanding().max()
            drawdowns = (cumulative_returns - rolling_max) / rolling_max
            
            # Maximum drawdown
            max_drawdown = drawdowns.min()
            max_dd_date = drawdowns.idxmin()
            
            result += f"Maximum Drawdown: {max_drawdown * 100:.2f}%\n"
            result += f"Max Drawdown Date: {max_dd_date.strftime('%Y-%m-%d') if hasattr(max_dd_date, 'strftime') else str(max_dd_date)}\n\n"
            
            # Drawdown statistics
            negative_drawdowns = drawdowns[drawdowns < 0]
            if len(negative_drawdowns) > 0:
                avg_drawdown = negative_drawdowns.mean()
                result += f"Average Drawdown: {avg_drawdown * 100:.2f}%\n"
                
                # Drawdown frequency
                drawdown_periods = (drawdowns < -0.05).sum()  # Periods with >5% drawdown
                total_periods = len(drawdowns)
                drawdown_frequency = drawdown_periods / total_periods
                
                result += f"Periods with >5% Drawdown: {drawdown_periods} ({drawdown_frequency * 100:.1f}%)\n"
            
            # Recovery analysis
            recovery_periods = self._calculate_recovery_periods(drawdowns)
            if recovery_periods:
                avg_recovery = np.mean(recovery_periods)
                max_recovery = np.max(recovery_periods)
                result += f"\nDrawdown Recovery Analysis:\n"
                result += f"Average Recovery Period: {avg_recovery:.1f} days\n"
                result += f"Maximum Recovery Period: {max_recovery:.0f} days\n"
            
            # Calmar ratio (annualized return / max drawdown)
            if max_drawdown < 0:
                annualized_return = portfolio_returns.mean() * 252
                calmar_ratio = annualized_return / abs(max_drawdown)
                result += f"\nCalmar Ratio: {calmar_ratio:.3f}\n"
            
            # Underwater curve analysis
            underwater_periods = (drawdowns < 0).sum()
            total_periods = len(drawdowns)
            underwater_percentage = underwater_periods / total_periods
            
            result += f"Time Underwater: {underwater_periods} periods ({underwater_percentage * 100:.1f}%)\n"
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error analyzing drawdowns: {e}")
            return f"Error in drawdown analysis: {str(e)}\n"
    
    def _analyze_position_risk(self, returns_data: pd.DataFrame, input_data: RiskAnalysisInput) -> str:
        """Analyze risk for a specific position size."""
        try:
            result = ""
            
            if not input_data.position_size:
                return "Position size not provided\n"
            
            # Assume single asset analysis for position risk
            if len(returns_data.columns) == 0:
                return "No asset data available for position analysis\n"
            
            # Use first asset or create equal-weighted portfolio
            if len(returns_data.columns) == 1:
                asset_returns = returns_data.iloc[:, 0]
                asset_name = returns_data.columns[0]
            else:
                # Equal-weighted portfolio
                weights = np.ones(len(returns_data.columns)) / len(returns_data.columns)
                asset_returns = (returns_data * weights).sum(axis=1)
                asset_name = "Portfolio"
            
            position_value = input_data.position_size
            
            result += f"Position Risk Analysis for {asset_name}\n"
            result += f"Position Size: ${position_value:,.2f}\n\n"
            
            # Daily risk metrics
            daily_volatility = asset_returns.std()
            daily_var_95 = np.percentile(asset_returns, 5)
            daily_var_99 = np.percentile(asset_returns, 1)
            
            result += f"Daily Volatility: {daily_volatility * 100:.2f}%\n"
            result += f"Daily VaR (95%): {daily_var_95 * 100:.2f}% (${daily_var_95 * position_value:,.2f})\n"
            result += f"Daily VaR (99%): {daily_var_99 * 100:.2f}% (${daily_var_99 * position_value:,.2f})\n\n"
            
            # Position sizing recommendations
            portfolio_percentage = (position_value / input_data.portfolio_value) * 100
            result += f"Position as % of Portfolio: {portfolio_percentage:.2f}%\n"
            
            # Risk-based position sizing
            target_risk_levels = [0.01, 0.02, 0.05]  # 1%, 2%, 5% portfolio risk
            
            result += "\nRisk-Based Position Sizing Recommendations:\n"
            for target_risk in target_risk_levels:
                # Position size that would result in target portfolio risk
                recommended_size = (target_risk * input_data.portfolio_value) / abs(daily_var_95)
                result += f"For {target_risk*100:.0f}% portfolio risk: ${recommended_size:,.2f}\n"
            
            # Kelly criterion (simplified)
            if len(asset_returns) > 0:
                win_rate = (asset_returns > 0).mean()
                avg_win = asset_returns[asset_returns > 0].mean() if (asset_returns > 0).any() else 0
                avg_loss = asset_returns[asset_returns < 0].mean() if (asset_returns < 0).any() else 0
                
                if avg_loss != 0:
                    kelly_fraction = win_rate - ((1 - win_rate) * avg_win / abs(avg_loss))
                    kelly_position = kelly_fraction * input_data.portfolio_value
                    
                    result += f"\nKelly Criterion Analysis:\n"
                    result += f"Win Rate: {win_rate * 100:.1f}%\n"
                    result += f"Average Win: {avg_win * 100:.2f}%\n"
                    result += f"Average Loss: {avg_loss * 100:.2f}%\n"
                    result += f"Kelly Fraction: {kelly_fraction * 100:.2f}%\n"
                    result += f"Kelly Position Size: ${max(0, kelly_position):,.2f}\n"
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error analyzing position risk: {e}")
            return f"Error in position risk analysis: {str(e)}\n"
    
    def _perform_stress_tests(self, returns_data: pd.DataFrame, weights: np.ndarray, 
                            input_data: RiskAnalysisInput) -> str:
        """Perform stress tests on the portfolio."""
        try:
            result = ""
            
            if not input_data.stress_scenarios:
                return "No stress scenarios specified\n"
            
            # Portfolio returns
            portfolio_returns = (returns_data * weights).sum(axis=1)
            current_portfolio_value = input_data.portfolio_value
            
            result += "Stress Test Scenarios:\n\n"
            
            for scenario in input_data.stress_scenarios:
                scenario_lower = scenario.lower()
                
                if scenario_lower == 'market_crash':
                    # Simulate a market crash (e.g., -20% market drop)
                    stress_return = -0.20
                    scenario_name = "Market Crash (-20%)"
                    
                elif scenario_lower == 'interest_rate_shock':
                    # Simulate interest rate shock (varies by asset type)
                    stress_return = -0.10  # Simplified assumption
                    scenario_name = "Interest Rate Shock (-10%)"
                    
                elif scenario_lower == 'sector_rotation':
                    # Simulate sector rotation (some assets up, others down)
                    stress_return = -0.05  # Conservative estimate
                    scenario_name = "Sector Rotation (-5%)"
                    
                else:
                    # Custom scenario - assume moderate stress
                    stress_return = -0.15
                    scenario_name = f"Custom Scenario: {scenario}"
                
                # Calculate portfolio impact
                portfolio_impact = stress_return * current_portfolio_value
                new_portfolio_value = current_portfolio_value + portfolio_impact
                
                result += f"{scenario_name}:\n"
                result += f"Portfolio Impact: ${portfolio_impact:,.2f}\n"
                result += f"New Portfolio Value: ${new_portfolio_value:,.2f}\n"
                result += f"Percentage Loss: {(portfolio_impact / current_portfolio_value) * 100:.2f}%\n\n"
            
            # Historical stress test (worst periods)
            if len(portfolio_returns) >= 20:
                worst_periods = portfolio_returns.nsmallest(5)
                result += "Historical Worst Periods:\n"
                for i, (date, return_val) in enumerate(worst_periods.items(), 1):
                    impact = return_val * current_portfolio_value
                    date_str = date.strftime('%Y-%m-%d') if hasattr(date, 'strftime') else str(date)
                    result += f"{i}. {date_str}: {return_val * 100:.2f}% (${impact:,.2f})\n"
                result += "\n"
            
            # Tail risk analysis
            tail_threshold = 0.05  # Bottom 5%
            tail_returns = portfolio_returns.quantile(tail_threshold)
            tail_impact = tail_returns * current_portfolio_value
            
            result += f"Tail Risk Analysis (Bottom {tail_threshold*100:.0f}%):\n"
            result += f"Tail Return Threshold: {tail_returns * 100:.2f}%\n"
            result += f"Potential Loss: ${tail_impact:,.2f}\n"
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error performing stress tests: {e}")
            return f"Error in stress testing: {str(e)}\n"
    
    def _calculate_risk_adjusted_metrics(self, returns_data: pd.DataFrame, weights: np.ndarray, 
                                       input_data: RiskAnalysisInput) -> str:
        """Calculate risk-adjusted performance metrics."""
        try:
            result = ""
            
            # Portfolio returns
            portfolio_returns = (returns_data * weights).sum(axis=1)
            
            if len(portfolio_returns) == 0:
                return "No portfolio returns data available\n"
            
            # Basic metrics
            mean_return = portfolio_returns.mean()
            volatility = portfolio_returns.std()
            annualized_return = mean_return * 252
            annualized_volatility = volatility * np.sqrt(252)
            
            # Risk-free rate (daily)
            daily_rf_rate = input_data.risk_free_rate / 252
            
            # Sharpe Ratio
            excess_returns = portfolio_returns - daily_rf_rate
            if volatility > 0:
                sharpe_ratio = excess_returns.mean() / volatility
                annualized_sharpe = sharpe_ratio * np.sqrt(252)
                result += f"Sharpe Ratio (Annualized): {annualized_sharpe:.3f}\n"
            
            # Sortino Ratio
            negative_returns = portfolio_returns[portfolio_returns < daily_rf_rate]
            if len(negative_returns) > 0:
                downside_deviation = negative_returns.std()
                if downside_deviation > 0:
                    sortino_ratio = excess_returns.mean() / downside_deviation
                    annualized_sortino = sortino_ratio * np.sqrt(252)
                    result += f"Sortino Ratio (Annualized): {annualized_sortino:.3f}\n"
            
            # Information Ratio (vs equal-weighted benchmark)
            if len(returns_data.columns) > 1:
                benchmark_weights = np.ones(len(returns_data.columns)) / len(returns_data.columns)
                benchmark_returns = (returns_data * benchmark_weights).sum(axis=1)
                active_returns = portfolio_returns - benchmark_returns
                
                if active_returns.std() > 0:
                    information_ratio = active_returns.mean() / active_returns.std()
                    result += f"Information Ratio: {information_ratio:.3f}\n"
            
            # Maximum Drawdown and Calmar Ratio
            cumulative_returns = (1 + portfolio_returns).cumprod()
            rolling_max = cumulative_returns.expanding().max()
            drawdowns = (cumulative_returns - rolling_max) / rolling_max
            max_drawdown = drawdowns.min()
            
            if max_drawdown < 0:
                calmar_ratio = annualized_return / abs(max_drawdown)
                result += f"Calmar Ratio: {calmar_ratio:.3f}\n"
            
            # Treynor Ratio (simplified - using portfolio beta vs market)
            # Note: This is a simplified calculation
            if len(portfolio_returns) > 1:
                # Assume market return is the average of all assets
                market_proxy = returns_data.mean(axis=1)
                covariance = np.cov(portfolio_returns, market_proxy)[0, 1]
                market_variance = np.var(market_proxy)
                
                if market_variance > 0:
                    beta = covariance / market_variance
                    if beta != 0:
                        treynor_ratio = (annualized_return - input_data.risk_free_rate) / beta
                        result += f"Treynor Ratio: {treynor_ratio:.3f}\n"
                        result += f"Portfolio Beta: {beta:.3f}\n"
            
            # Jensen's Alpha (simplified)
            if 'beta' in locals() and beta is not None:
                market_return = returns_data.mean(axis=1).mean() * 252
                expected_return = input_data.risk_free_rate + beta * (market_return - input_data.risk_free_rate)
                alpha = annualized_return - expected_return
                result += f"Jensen's Alpha: {alpha * 100:.2f}%\n"
            
            # Risk-Return Efficiency
            if volatility > 0:
                return_to_risk = annualized_return / annualized_volatility
                result += f"Return-to-Risk Ratio: {return_to_risk:.3f}\n"
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error calculating risk-adjusted metrics: {e}")
            return f"Error calculating risk-adjusted metrics: {str(e)}\n"
    
    def _generate_risk_recommendations(self, returns_data: pd.DataFrame, weights: np.ndarray, 
                                     input_data: RiskAnalysisInput) -> str:
        """Generate risk management recommendations."""
        try:
            result = ""
            recommendations = []
            
            # Portfolio returns
            portfolio_returns = (returns_data * weights).sum(axis=1)
            
            # Volatility analysis
            annualized_volatility = portfolio_returns.std() * np.sqrt(252)
            if annualized_volatility > 0.25:  # >25% annual volatility
                recommendations.append("⚠️  High portfolio volatility detected. Consider reducing position sizes or adding defensive assets.")
            elif annualized_volatility < 0.10:  # <10% annual volatility
                recommendations.append("ℹ️  Low portfolio volatility. May consider slightly increasing risk for higher returns.")
            
            # Concentration risk
            max_weight = np.max(weights)
            if max_weight > 0.4:  # >40% in single asset
                recommendations.append("⚠️  High concentration risk. Consider diversifying across more assets.")
            
            # Correlation analysis
            if len(returns_data.columns) > 1:
                correlation_matrix = returns_data.corr()
                avg_correlation = correlation_matrix.values[np.triu_indices_from(correlation_matrix.values, k=1)].mean()
                
                if avg_correlation > 0.7:
                    recommendations.append("⚠️  High average correlation between assets. Portfolio may not be well diversified.")
                elif avg_correlation < 0.1:
                    recommendations.append("✅ Good diversification with low average correlation between assets.")
            
            # Drawdown analysis
            cumulative_returns = (1 + portfolio_returns).cumprod()
            rolling_max = cumulative_returns.expanding().max()
            drawdowns = (cumulative_returns - rolling_max) / rolling_max
            max_drawdown = abs(drawdowns.min())
            
            if max_drawdown > 0.20:  # >20% max drawdown
                recommendations.append("⚠️  Large maximum drawdown detected. Consider implementing stop-loss strategies.")
            
            # Sharpe ratio analysis
            daily_rf_rate = input_data.risk_free_rate / 252
            excess_returns = portfolio_returns - daily_rf_rate
            if portfolio_returns.std() > 0:
                sharpe_ratio = excess_returns.mean() / portfolio_returns.std() * np.sqrt(252)
                
                if sharpe_ratio < 0.5:
                    recommendations.append("⚠️  Low Sharpe ratio. Risk-adjusted returns could be improved.")
                elif sharpe_ratio > 1.5:
                    recommendations.append("✅ Excellent Sharpe ratio indicating good risk-adjusted returns.")
            
            # VaR analysis
            var_95 = np.percentile(portfolio_returns, 5)
            var_dollar = abs(var_95 * input_data.portfolio_value)
            
            if var_dollar > input_data.portfolio_value * 0.05:  # VaR > 5% of portfolio
                recommendations.append(f"⚠️  High daily VaR (${var_dollar:,.0f}). Consider hedging strategies.")
            
            # Position sizing recommendations
            if len(recommendations) == 0:
                recommendations.append("✅ Portfolio risk metrics appear to be within acceptable ranges.")
            
            # General recommendations
            recommendations.extend([
                "💡 Regularly rebalance portfolio to maintain target weights.",
                "💡 Monitor correlations as they can change during market stress.",
                "💡 Consider implementing dynamic hedging during high volatility periods.",
                "💡 Review and update risk limits based on changing market conditions."
            ])
            
            for i, rec in enumerate(recommendations, 1):
                result += f"{i}. {rec}\n"
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error generating recommendations: {e}")
            return f"Error generating recommendations: {str(e)}\n"
    
    def _monte_carlo_var(self, portfolio_returns: pd.Series, input_data: RiskAnalysisInput, 
                        num_simulations: int = 10000) -> str:
        """Perform Monte Carlo simulation for VaR calculation."""
        try:
            result = ""
            
            # Fit distribution to historical returns
            mean_return = portfolio_returns.mean()
            std_return = portfolio_returns.std()
            
            # Generate random returns
            np.random.seed(42)  # For reproducibility
            simulated_returns = np.random.normal(mean_return, std_return, num_simulations)
            
            # Calculate VaR from simulations
            confidence_levels = [0.90, 0.95, 0.99]
            
            for conf_level in confidence_levels:
                var_percentile = (1 - conf_level) * 100
                mc_var = np.percentile(simulated_returns, var_percentile)
                var_dollar = mc_var * input_data.portfolio_value
                
                result += f"MC VaR ({conf_level*100:.0f}%): {mc_var * 100:.2f}% (${var_dollar:,.2f})\n"
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in Monte Carlo VaR: {e}")
            return f"Error in Monte Carlo simulation: {str(e)}\n"
    
    def _backtest_var(self, portfolio_returns: pd.Series, confidence_level: float) -> str:
        """Backtest VaR model accuracy."""
        try:
            result = ""
            
            if len(portfolio_returns) < 100:  # Need sufficient data
                return "Insufficient data for VaR backtesting\n"
            
            # Rolling VaR calculation
            window_size = 60  # 60-day rolling window
            var_violations = 0
            total_predictions = 0
            
            for i in range(window_size, len(portfolio_returns)):
                # Calculate VaR using historical data up to this point
                historical_data = portfolio_returns.iloc[i-window_size:i]
                var_threshold = np.percentile(historical_data, (1-confidence_level)*100)
                
                # Check if actual return violated VaR
                actual_return = portfolio_returns.iloc[i]
                if actual_return <= var_threshold:
                    var_violations += 1
                
                total_predictions += 1
            
            # Calculate violation rate
            violation_rate = var_violations / total_predictions
            expected_violation_rate = 1 - confidence_level
            
            result += f"VaR Backtesting Results ({confidence_level*100:.0f}% confidence):\n"
            result += f"Actual Violation Rate: {violation_rate * 100:.2f}%\n"
            result += f"Expected Violation Rate: {expected_violation_rate * 100:.2f}%\n"
            result += f"Total Violations: {var_violations} out of {total_predictions}\n"
            
            # Model assessment
            if abs(violation_rate - expected_violation_rate) < 0.02:  # Within 2%
                assessment = "✅ VaR model appears to be well-calibrated"
            elif violation_rate > expected_violation_rate + 0.02:
                assessment = "⚠️  VaR model may be underestimating risk"
            else:
                assessment = "⚠️  VaR model may be overestimating risk"
            
            result += f"Assessment: {assessment}\n"
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in VaR backtesting: {e}")
            return f"Error in VaR backtesting: {str(e)}\n"
    
    def _calculate_recovery_periods(self, drawdowns: pd.Series) -> List[int]:
        """Calculate recovery periods from drawdowns."""
        try:
            recovery_periods = []
            in_drawdown = False
            drawdown_start = None
            
            for i, dd in enumerate(drawdowns):
                if dd < 0 and not in_drawdown:
                    # Start of drawdown
                    in_drawdown = True
                    drawdown_start = i
                elif dd >= 0 and in_drawdown:
                    # End of drawdown (recovery)
                    in_drawdown = False
                    if drawdown_start is not None:
                        recovery_period = i - drawdown_start
                        recovery_periods.append(recovery_period)
            
            return recovery_periods
            
        except Exception:
            return []


if __name__ == "__main__":
    # Example usage
    import asyncio
    from pathlib import Path
    from core.config_manager import ConfigManager
    from core.data_manager import UnifiedDataManager
    
    async def test_risk_analysis_tool():
        config_manager = ConfigManager(Path("../config"))
        data_manager = UnifiedDataManager(config_manager)
        
        tool = RiskAnalysisTool(data_manager)
        
        # Test portfolio risk analysis
        result = tool._run(
            analysis_type="comprehensive",
            symbols=["AAPL", "MSFT", "GOOGL"],
            weights=[0.4, 0.3, 0.3],
            portfolio_value=100000.0,
            risk_free_rate=0.02,
            confidence_level=0.95,
            lookback_period="1y",
            stress_scenarios=["market_crash", "interest_rate_shock"]
        )
        
        print("Risk Analysis Result:")
        print(result)
    
    # Run test
    # asyncio.run(test_risk_analysis_tool())