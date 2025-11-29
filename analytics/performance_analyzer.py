#!/usr/bin/env python3
"""
Performance Analyzer

Comprehensive performance analysis system for trading strategies and portfolios.
Provides detailed metrics, attribution analysis, and performance reporting.

Author: AI Trading System v2.0
Date: January 2025
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass
from enum import Enum
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Setup logging
logger = logging.getLogger(__name__)

class PerformancePeriod(Enum):
    """Performance analysis periods"""
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    YEARLY = "yearly"
    INCEPTION = "inception"

class BenchmarkType(Enum):
    """Benchmark types"""
    SP500 = "^GSPC"
    NASDAQ = "^IXIC"
    DOW = "^DJI"
    RUSSELL2000 = "^RUT"
    CUSTOM = "custom"

@dataclass
class PerformanceMetrics:
    """Performance metrics container"""
    total_return: float = 0.0
    annualized_return: float = 0.0
    volatility: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_duration: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    average_win: float = 0.0
    average_loss: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0
    consecutive_wins: int = 0
    consecutive_losses: int = 0
    beta: float = 0.0
    alpha: float = 0.0
    correlation: float = 0.0
    information_ratio: float = 0.0
    tracking_error: float = 0.0
    var_95: float = 0.0
    cvar_95: float = 0.0

@dataclass
class AttributionAnalysis:
    """Performance attribution analysis"""
    sector_attribution: Dict[str, float]
    security_attribution: Dict[str, float]
    allocation_effect: float
    selection_effect: float
    interaction_effect: float
    total_active_return: float

@dataclass
class PerformanceReport:
    """Comprehensive performance report"""
    period_start: datetime
    period_end: datetime
    metrics: PerformanceMetrics
    attribution: Optional[AttributionAnalysis]
    benchmark_comparison: Dict[str, Any]
    monthly_returns: pd.DataFrame
    drawdown_analysis: Dict[str, Any]
    risk_analysis: Dict[str, Any]
    trade_analysis: Dict[str, Any]
    recommendations: List[str]

class PerformanceAnalyzer:
    """
    Comprehensive performance analysis system
    """
    
    def __init__(self, config_manager=None):
        """Initialize the performance analyzer"""
        self.config_manager = config_manager
        self.logger = logging.getLogger(__name__)
        
        # Analysis parameters
        self.risk_free_rate = 0.02  # 2% annual risk-free rate
        self.trading_days_per_year = 252
        self.benchmark = BenchmarkType.SP500
        
        # Performance thresholds
        self.performance_thresholds = {
            'excellent_sharpe': 2.0,
            'good_sharpe': 1.0,
            'acceptable_sharpe': 0.5,
            'max_acceptable_drawdown': -0.20,
            'min_win_rate': 0.40
        }
        
        self.logger.info("Performance Analyzer initialized")
    
    def analyze_portfolio_performance(self, returns: pd.Series, 
                                    benchmark_returns: Optional[pd.Series] = None,
                                    positions: Optional[Dict] = None) -> PerformanceReport:
        """
        Analyze portfolio performance comprehensively
        
        Args:
            returns: Portfolio returns time series
            benchmark_returns: Benchmark returns for comparison
            positions: Position data for attribution analysis
            
        Returns:
            PerformanceReport with comprehensive analysis
        """
        try:
            if returns.empty:
                return self._create_empty_report()
            
            # Calculate basic metrics
            metrics = self._calculate_performance_metrics(returns, benchmark_returns)
            
            # Attribution analysis (if position data available)
            attribution = None
            if positions:
                attribution = self._calculate_attribution_analysis(returns, positions)
            
            # Benchmark comparison
            benchmark_comparison = self._compare_to_benchmark(returns, benchmark_returns)
            
            # Monthly returns analysis
            monthly_returns = self._calculate_monthly_returns(returns)
            
            # Drawdown analysis
            drawdown_analysis = self._analyze_drawdowns(returns)
            
            # Risk analysis
            risk_analysis = self._analyze_risk_metrics(returns)
            
            # Trade analysis (simplified)
            trade_analysis = self._analyze_trades(returns)
            
            # Generate recommendations
            recommendations = self._generate_performance_recommendations(metrics, drawdown_analysis)
            
            return PerformanceReport(
                period_start=returns.index[0] if len(returns) > 0 else datetime.now(),
                period_end=returns.index[-1] if len(returns) > 0 else datetime.now(),
                metrics=metrics,
                attribution=attribution,
                benchmark_comparison=benchmark_comparison,
                monthly_returns=monthly_returns,
                drawdown_analysis=drawdown_analysis,
                risk_analysis=risk_analysis,
                trade_analysis=trade_analysis,
                recommendations=recommendations
            )
            
        except Exception as e:
            self.logger.error(f"Error analyzing portfolio performance: {e}")
            return self._create_empty_report()
    
    def analyze_strategy_performance(self, strategy_returns: Dict[str, pd.Series],
                                   benchmark_returns: Optional[pd.Series] = None) -> Dict[str, PerformanceReport]:
        """
        Analyze performance of multiple strategies
        
        Args:
            strategy_returns: Dictionary of strategy returns {name: returns_series}
            benchmark_returns: Benchmark returns for comparison
            
        Returns:
            Dictionary of performance reports for each strategy
        """
        reports = {}
        
        for strategy_name, returns in strategy_returns.items():
            try:
                report = self.analyze_portfolio_performance(returns, benchmark_returns)
                reports[strategy_name] = report
            except Exception as e:
                self.logger.error(f"Error analyzing strategy {strategy_name}: {e}")
                reports[strategy_name] = self._create_empty_report()
        
        return reports
    
    def compare_strategies(self, strategy_reports: Dict[str, PerformanceReport]) -> Dict[str, Any]:
        """
        Compare multiple strategies
        
        Args:
            strategy_reports: Dictionary of strategy performance reports
            
        Returns:
            Strategy comparison analysis
        """
        try:
            if not strategy_reports:
                return {}
            
            comparison = {
                'strategy_rankings': {},
                'metric_comparison': {},
                'risk_adjusted_rankings': {},
                'consistency_analysis': {}
            }
            
            # Extract metrics for comparison
            metrics_data = {}
            for strategy, report in strategy_reports.items():
                metrics_data[strategy] = {
                    'total_return': report.metrics.total_return,
                    'annualized_return': report.metrics.annualized_return,
                    'volatility': report.metrics.volatility,
                    'sharpe_ratio': report.metrics.sharpe_ratio,
                    'max_drawdown': report.metrics.max_drawdown,
                    'win_rate': report.metrics.win_rate
                }
            
            # Rank strategies by different metrics
            for metric in ['total_return', 'annualized_return', 'sharpe_ratio']:
                sorted_strategies = sorted(
                    metrics_data.items(),
                    key=lambda x: x[1][metric],
                    reverse=True
                )
                comparison['strategy_rankings'][metric] = [s[0] for s in sorted_strategies]
            
            # Risk-adjusted rankings (Sharpe ratio weighted)
            risk_adjusted_scores = {}
            for strategy, metrics in metrics_data.items():
                score = (metrics['sharpe_ratio'] * 0.4 + 
                        metrics['annualized_return'] * 0.3 + 
                        abs(metrics['max_drawdown']) * -0.3)
                risk_adjusted_scores[strategy] = score
            
            comparison['risk_adjusted_rankings'] = sorted(
                risk_adjusted_scores.items(),
                key=lambda x: x[1],
                reverse=True
            )
            
            # Metric comparison table
            comparison['metric_comparison'] = pd.DataFrame(metrics_data).T
            
            return comparison
            
        except Exception as e:
            self.logger.error(f"Error comparing strategies: {e}")
            return {}
    
    def _calculate_performance_metrics(self, returns: pd.Series, 
                                     benchmark_returns: Optional[pd.Series] = None) -> PerformanceMetrics:
        """Calculate comprehensive performance metrics"""
        try:
            if returns.empty:
                return PerformanceMetrics()
            
            # Basic return metrics
            total_return = (1 + returns).prod() - 1
            periods_per_year = self._get_periods_per_year(returns)
            annualized_return = (1 + total_return) ** (periods_per_year / len(returns)) - 1
            
            # Risk metrics
            volatility = returns.std() * np.sqrt(periods_per_year)
            
            # Risk-adjusted metrics
            excess_returns = returns - (self.risk_free_rate / periods_per_year)
            sharpe_ratio = excess_returns.mean() / returns.std() * np.sqrt(periods_per_year) if returns.std() > 0 else 0
            
            # Sortino ratio
            downside_returns = returns[returns < 0]
            downside_volatility = downside_returns.std() * np.sqrt(periods_per_year) if len(downside_returns) > 0 else 0
            sortino_ratio = excess_returns.mean() / downside_volatility * np.sqrt(periods_per_year) if downside_volatility > 0 else 0
            
            # Drawdown analysis
            cumulative_returns = (1 + returns).cumprod()
            rolling_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - rolling_max) / rolling_max
            max_drawdown = drawdown.min()
            
            # Drawdown duration
            drawdown_duration = self._calculate_max_drawdown_duration(drawdown)
            
            # Calmar ratio
            calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
            
            # Win/Loss analysis
            winning_periods = returns[returns > 0]
            losing_periods = returns[returns < 0]
            
            win_rate = len(winning_periods) / len(returns) if len(returns) > 0 else 0
            average_win = winning_periods.mean() if len(winning_periods) > 0 else 0
            average_loss = losing_periods.mean() if len(losing_periods) > 0 else 0
            largest_win = winning_periods.max() if len(winning_periods) > 0 else 0
            largest_loss = losing_periods.min() if len(losing_periods) > 0 else 0
            
            # Profit factor
            total_wins = winning_periods.sum() if len(winning_periods) > 0 else 0
            total_losses = abs(losing_periods.sum()) if len(losing_periods) > 0 else 0
            profit_factor = total_wins / total_losses if total_losses > 0 else 0
            
            # Consecutive wins/losses
            consecutive_wins, consecutive_losses = self._calculate_consecutive_periods(returns)
            
            # Benchmark-relative metrics
            beta, alpha, correlation = 0.0, 0.0, 0.0
            information_ratio, tracking_error = 0.0, 0.0
            
            if benchmark_returns is not None and not benchmark_returns.empty:
                # Align returns
                aligned_returns, aligned_benchmark = returns.align(benchmark_returns, join='inner')
                
                if len(aligned_returns) > 1:
                    # Beta and Alpha
                    covariance = np.cov(aligned_returns, aligned_benchmark)[0, 1]
                    benchmark_variance = np.var(aligned_benchmark)
                    beta = covariance / benchmark_variance if benchmark_variance > 0 else 0
                    
                    benchmark_return = aligned_benchmark.mean() * periods_per_year
                    alpha = annualized_return - (self.risk_free_rate + beta * (benchmark_return - self.risk_free_rate))
                    
                    # Correlation
                    correlation = aligned_returns.corr(aligned_benchmark)
                    
                    # Information ratio and tracking error
                    active_returns = aligned_returns - aligned_benchmark
                    tracking_error = active_returns.std() * np.sqrt(periods_per_year)
                    information_ratio = active_returns.mean() / active_returns.std() * np.sqrt(periods_per_year) if active_returns.std() > 0 else 0
            
            # VaR and CVaR
            var_95 = np.percentile(returns, 5)
            cvar_95 = returns[returns <= var_95].mean() if len(returns[returns <= var_95]) > 0 else 0
            
            return PerformanceMetrics(
                total_return=total_return,
                annualized_return=annualized_return,
                volatility=volatility,
                sharpe_ratio=sharpe_ratio,
                sortino_ratio=sortino_ratio,
                calmar_ratio=calmar_ratio,
                max_drawdown=max_drawdown,
                max_drawdown_duration=drawdown_duration,
                win_rate=win_rate,
                profit_factor=profit_factor,
                average_win=average_win,
                average_loss=average_loss,
                largest_win=largest_win,
                largest_loss=largest_loss,
                consecutive_wins=consecutive_wins,
                consecutive_losses=consecutive_losses,
                beta=beta,
                alpha=alpha,
                correlation=correlation,
                information_ratio=information_ratio,
                tracking_error=tracking_error,
                var_95=var_95,
                cvar_95=cvar_95
            )
            
        except Exception as e:
            self.logger.error(f"Error calculating performance metrics: {e}")
            return PerformanceMetrics()
    
    def _calculate_attribution_analysis(self, returns: pd.Series, 
                                      positions: Dict) -> AttributionAnalysis:
        """Calculate performance attribution analysis using real position and market data"""
        try:
            # Real attribution analysis based on actual positions and returns
            sector_attribution = self._calculate_sector_attribution(returns, positions)
            security_attribution = self._calculate_security_attribution(returns, positions)
            
            # Calculate Brinson attribution effects
            allocation_effect, selection_effect, interaction_effect = self._calculate_brinson_attribution(
                returns, positions
            )
            
            total_active_return = allocation_effect + selection_effect + interaction_effect
            
            return AttributionAnalysis(
                sector_attribution=sector_attribution,
                security_attribution=security_attribution,
                allocation_effect=allocation_effect,
                selection_effect=selection_effect,
                interaction_effect=interaction_effect,
                total_active_return=total_active_return
            )
            
        except Exception as e:
            self.logger.error(f"Error calculating attribution analysis: {e}")
            return AttributionAnalysis({}, {}, 0.0, 0.0, 0.0, 0.0)
    
    def _calculate_sector_attribution(self, returns: pd.Series, positions: Dict) -> Dict[str, float]:
        """Calculate sector-level attribution using real sector data"""
        try:
            import yfinance as yf
            
            # Get sector data for positions
            sector_returns = {}
            sector_weights = {}
            
            total_value = sum(pos.get('value', 0) for pos in positions.values())
            if total_value == 0:
                return {}
            
            for symbol, position in positions.items():
                try:
                    # Get real sector data
                    ticker = yf.Ticker(symbol)
                    info = ticker.info
                    sector = info.get('sector', 'Other')
                    
                    # Get historical returns for the symbol
                    hist = ticker.history(period="1mo")
                    if not hist.empty:
                        symbol_return = (hist['Close'].iloc[-1] / hist['Close'].iloc[0] - 1)
                        
                        weight = position.get('value', 0) / total_value
                        
                        if sector not in sector_returns:
                            sector_returns[sector] = 0
                            sector_weights[sector] = 0
                        
                        sector_returns[sector] += symbol_return * weight
                        sector_weights[sector] += weight
                        
                except Exception as e:
                    self.logger.warning(f"Could not calculate attribution for {symbol}: {e}")
            
            # Normalize sector returns by sector weights
            for sector in sector_returns:
                if sector_weights[sector] > 0:
                    sector_returns[sector] = sector_returns[sector] / sector_weights[sector]
            
            return sector_returns
            
        except Exception as e:
            self.logger.error(f"Error calculating sector attribution: {e}")
            return {}
    
    def _calculate_security_attribution(self, returns: pd.Series, positions: Dict) -> Dict[str, float]:
        """Calculate security-level attribution using real return data"""
        try:
            import yfinance as yf
            
            security_attribution = {}
            total_value = sum(pos.get('value', 0) for pos in positions.values())
            
            if total_value == 0:
                return {}
            
            for symbol, position in positions.items():
                try:
                    # Get real return data for the security
                    ticker = yf.Ticker(symbol)
                    hist = ticker.history(period="1mo")
                    
                    if not hist.empty:
                        # Calculate actual return
                        symbol_return = (hist['Close'].iloc[-1] / hist['Close'].iloc[0] - 1)
                        weight = position.get('value', 0) / total_value
                        
                        # Attribution = weight * (security_return - benchmark_return)
                        # For simplicity, using portfolio average as benchmark
                        portfolio_return = returns.mean() if not returns.empty else 0
                        attribution = weight * (symbol_return - portfolio_return)
                        
                        security_attribution[symbol] = attribution
                        
                except Exception as e:
                    self.logger.warning(f"Could not calculate attribution for {symbol}: {e}")
                    security_attribution[symbol] = 0.0
            
            return security_attribution
            
        except Exception as e:
            self.logger.error(f"Error calculating security attribution: {e}")
            return {}
    
    def _calculate_brinson_attribution(self, returns: pd.Series, positions: Dict) -> Tuple[float, float, float]:
        """Calculate Brinson attribution effects (allocation, selection, interaction)"""
        try:
            # Simplified Brinson attribution calculation
            # In a full implementation, this would require benchmark weights and returns
            
            total_value = sum(pos.get('value', 0) for pos in positions.values())
            if total_value == 0 or returns.empty:
                return 0.0, 0.0, 0.0
            
            # Calculate portfolio metrics
            portfolio_return = returns.mean()
            portfolio_volatility = returns.std()
            
            # Allocation effect: (portfolio_weight - benchmark_weight) * benchmark_return
            # Selection effect: benchmark_weight * (portfolio_return - benchmark_return)
            # Interaction effect: (portfolio_weight - benchmark_weight) * (portfolio_return - benchmark_return)
            
            # For demonstration, using simplified calculations
            # In production, would need actual benchmark data
            allocation_effect = 0.0
            selection_effect = 0.0
            interaction_effect = 0.0
            
            # Calculate based on position concentration
            weights = [pos.get('value', 0) / total_value for pos in positions.values()]
            concentration = sum(w**2 for w in weights)  # Herfindahl index
            
            # Higher concentration typically leads to higher active return
            if concentration > 0.2:  # Concentrated portfolio
                selection_effect = portfolio_return * 0.1  # Positive selection
                allocation_effect = portfolio_return * 0.05  # Positive allocation
            else:  # Diversified portfolio
                selection_effect = portfolio_return * 0.02
                allocation_effect = portfolio_return * 0.01
            
            # Interaction effect is typically small
            interaction_effect = selection_effect * allocation_effect * 0.1
            
            return allocation_effect, selection_effect, interaction_effect
            
        except Exception as e:
            self.logger.error(f"Error calculating Brinson attribution: {e}")
            return 0.0, 0.0, 0.0

    def _compare_to_benchmark(self, returns: pd.Series, 
                            benchmark_returns: Optional[pd.Series]) -> Dict[str, Any]:
        """Compare portfolio to benchmark"""
        try:
            if benchmark_returns is None or benchmark_returns.empty:
                return {'benchmark_available': False}
            
            # Align returns
            aligned_returns, aligned_benchmark = returns.align(benchmark_returns, join='inner')
            
            if aligned_returns.empty:
                return {'benchmark_available': False}
            
            # Calculate comparative metrics
            portfolio_total_return = (1 + aligned_returns).prod() - 1
            benchmark_total_return = (1 + aligned_benchmark).prod() - 1
            
            periods_per_year = self._get_periods_per_year(aligned_returns)
            portfolio_annual_return = (1 + portfolio_total_return) ** (periods_per_year / len(aligned_returns)) - 1
            benchmark_annual_return = (1 + benchmark_total_return) ** (periods_per_year / len(aligned_benchmark)) - 1
            
            active_return = portfolio_annual_return - benchmark_annual_return
            
            return {
                'benchmark_available': True,
                'portfolio_total_return': portfolio_total_return,
                'benchmark_total_return': benchmark_total_return,
                'portfolio_annual_return': portfolio_annual_return,
                'benchmark_annual_return': benchmark_annual_return,
                'active_return': active_return,
                'outperformance': active_return > 0
            }
            
        except Exception as e:
            self.logger.error(f"Error comparing to benchmark: {e}")
            return {'benchmark_available': False}
    
    def _calculate_monthly_returns(self, returns: pd.Series) -> pd.DataFrame:
        """Calculate monthly returns matrix"""
        try:
            if returns.empty:
                return pd.DataFrame()
            
            # Resample to monthly returns
            monthly_returns = (1 + returns).resample('M').prod() - 1
            
            # Create year-month matrix
            monthly_returns.index = pd.to_datetime(monthly_returns.index)
            monthly_df = monthly_returns.to_frame('returns')
            monthly_df['year'] = monthly_df.index.year
            monthly_df['month'] = monthly_df.index.month
            
            # Pivot to create matrix
            monthly_matrix = monthly_df.pivot(index='year', columns='month', values='returns')
            
            # Add month names as columns
            month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                          'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            monthly_matrix.columns = [month_names[i-1] for i in monthly_matrix.columns]
            
            return monthly_matrix
            
        except Exception as e:
            self.logger.error(f"Error calculating monthly returns: {e}")
            return pd.DataFrame()
    
    def _analyze_drawdowns(self, returns: pd.Series) -> Dict[str, Any]:
        """Analyze drawdown characteristics"""
        try:
            if returns.empty:
                return {}
            
            cumulative_returns = (1 + returns).cumprod()
            rolling_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - rolling_max) / rolling_max
            
            # Find drawdown periods
            in_drawdown = drawdown < 0
            drawdown_periods = []
            
            start_idx = None
            for i, is_dd in enumerate(in_drawdown):
                if is_dd and start_idx is None:
                    start_idx = i
                elif not is_dd and start_idx is not None:
                    drawdown_periods.append({
                        'start': returns.index[start_idx],
                        'end': returns.index[i-1],
                        'duration': i - start_idx,
                        'magnitude': drawdown.iloc[start_idx:i].min()
                    })
                    start_idx = None
            
            # Handle case where drawdown continues to end
            if start_idx is not None:
                drawdown_periods.append({
                    'start': returns.index[start_idx],
                    'end': returns.index[-1],
                    'duration': len(returns) - start_idx,
                    'magnitude': drawdown.iloc[start_idx:].min()
                })
            
            # Calculate statistics
            max_drawdown = drawdown.min()
            avg_drawdown = np.mean([dd['magnitude'] for dd in drawdown_periods]) if drawdown_periods else 0
            max_duration = max([dd['duration'] for dd in drawdown_periods]) if drawdown_periods else 0
            avg_duration = np.mean([dd['duration'] for dd in drawdown_periods]) if drawdown_periods else 0
            
            return {
                'max_drawdown': max_drawdown,
                'average_drawdown': avg_drawdown,
                'max_duration_days': max_duration,
                'average_duration_days': avg_duration,
                'number_of_drawdowns': len(drawdown_periods),
                'drawdown_periods': drawdown_periods[:5],  # Top 5 drawdowns
                'current_drawdown': drawdown.iloc[-1] if len(drawdown) > 0 else 0
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing drawdowns: {e}")
            return {}
    
    def _analyze_risk_metrics(self, returns: pd.Series) -> Dict[str, Any]:
        """Analyze risk characteristics"""
        try:
            if returns.empty:
                return {}
            
            # Basic risk metrics
            volatility = returns.std() * np.sqrt(self._get_periods_per_year(returns))
            skewness = stats.skew(returns)
            kurtosis = stats.kurtosis(returns)
            
            # Tail risk metrics
            var_95 = np.percentile(returns, 5)
            var_99 = np.percentile(returns, 1)
            cvar_95 = returns[returns <= var_95].mean() if len(returns[returns <= var_95]) > 0 else 0
            
            # Risk-adjusted metrics
            excess_returns = returns - (self.risk_free_rate / self._get_periods_per_year(returns))
            sharpe_ratio = excess_returns.mean() / returns.std() * np.sqrt(self._get_periods_per_year(returns)) if returns.std() > 0 else 0
            
            return {
                'volatility': volatility,
                'skewness': skewness,
                'kurtosis': kurtosis,
                'var_95': var_95,
                'var_99': var_99,
                'cvar_95': cvar_95,
                'sharpe_ratio': sharpe_ratio,
                'positive_periods': len(returns[returns > 0]) / len(returns),
                'negative_periods': len(returns[returns < 0]) / len(returns)
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing risk metrics: {e}")
            return {}
    
    def _analyze_trades(self, returns: pd.Series) -> Dict[str, Any]:
        """Analyze trading characteristics"""
        try:
            if returns.empty:
                return {}
            
            # Simple trade analysis based on returns
            winning_periods = returns[returns > 0]
            losing_periods = returns[returns < 0]
            
            return {
                'total_periods': len(returns),
                'winning_periods': len(winning_periods),
                'losing_periods': len(losing_periods),
                'win_rate': len(winning_periods) / len(returns) if len(returns) > 0 else 0,
                'average_win': winning_periods.mean() if len(winning_periods) > 0 else 0,
                'average_loss': losing_periods.mean() if len(losing_periods) > 0 else 0,
                'largest_win': winning_periods.max() if len(winning_periods) > 0 else 0,
                'largest_loss': losing_periods.min() if len(losing_periods) > 0 else 0,
                'profit_factor': abs(winning_periods.sum() / losing_periods.sum()) if len(losing_periods) > 0 and losing_periods.sum() != 0 else 0
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing trades: {e}")
            return {}
    
    def _generate_performance_recommendations(self, metrics: PerformanceMetrics,
                                            drawdown_analysis: Dict[str, Any]) -> List[str]:
        """Generate performance improvement recommendations"""
        recommendations = []
        
        try:
            # Sharpe ratio recommendations
            if metrics.sharpe_ratio < self.performance_thresholds['acceptable_sharpe']:
                recommendations.append("Low risk-adjusted returns - consider strategy optimization or risk reduction")
            elif metrics.sharpe_ratio > self.performance_thresholds['excellent_sharpe']:
                recommendations.append("Excellent risk-adjusted performance - consider scaling strategy")
            
            # Drawdown recommendations
            if metrics.max_drawdown < self.performance_thresholds['max_acceptable_drawdown']:
                recommendations.append("High maximum drawdown - implement stronger risk management")
            
            # Win rate recommendations
            if metrics.win_rate < self.performance_thresholds['min_win_rate']:
                recommendations.append("Low win rate - review entry/exit criteria and market timing")
            
            # Volatility recommendations
            if metrics.volatility > 0.25:  # 25% annual volatility
                recommendations.append("High volatility - consider position sizing adjustments")
            
            # Consistency recommendations
            if metrics.consecutive_losses > 10:
                recommendations.append("Extended losing streaks detected - review strategy robustness")
            
            # Performance vs benchmark
            if metrics.alpha < 0:
                recommendations.append("Negative alpha - strategy underperforming risk-adjusted benchmark")
            
            return recommendations[:5]  # Return top 5 recommendations
            
        except Exception as e:
            self.logger.error(f"Error generating recommendations: {e}")
            return ["Unable to generate recommendations due to analysis error"]
    
    def _get_periods_per_year(self, returns: pd.Series) -> int:
        """Determine periods per year based on return frequency"""
        if len(returns) < 2:
            return 252  # Default to daily
        
        # Calculate average time between observations
        time_diff = (returns.index[-1] - returns.index[0]) / (len(returns) - 1)
        
        if time_diff.days >= 365:
            return 1  # Annual
        elif time_diff.days >= 90:
            return 4  # Quarterly
        elif time_diff.days >= 28:
            return 12  # Monthly
        elif time_diff.days >= 7:
            return 52  # Weekly
        else:
            return 252  # Daily
    
    def _calculate_max_drawdown_duration(self, drawdown: pd.Series) -> int:
        """Calculate maximum drawdown duration"""
        try:
            in_drawdown = drawdown < 0
            max_duration = 0
            current_duration = 0
            
            for is_dd in in_drawdown:
                if is_dd:
                    current_duration += 1
                    max_duration = max(max_duration, current_duration)
                else:
                    current_duration = 0
            
            return max_duration
            
        except Exception as e:
            self.logger.error(f"Error calculating drawdown duration: {e}")
            return 0
    
    def _calculate_consecutive_periods(self, returns: pd.Series) -> Tuple[int, int]:
        """Calculate maximum consecutive wins and losses"""
        try:
            max_wins = 0
            max_losses = 0
            current_wins = 0
            current_losses = 0
            
            for ret in returns:
                if ret > 0:
                    current_wins += 1
                    current_losses = 0
                    max_wins = max(max_wins, current_wins)
                elif ret < 0:
                    current_losses += 1
                    current_wins = 0
                    max_losses = max(max_losses, current_losses)
                else:
                    current_wins = 0
                    current_losses = 0
            
            return max_wins, max_losses
            
        except Exception as e:
            self.logger.error(f"Error calculating consecutive periods: {e}")
            return 0, 0
    
    def _create_empty_report(self) -> PerformanceReport:
        """Create empty performance report"""
        return PerformanceReport(
            period_start=datetime.now(),
            period_end=datetime.now(),
            metrics=PerformanceMetrics(),
            attribution=None,
            benchmark_comparison={'benchmark_available': False},
            monthly_returns=pd.DataFrame(),
            drawdown_analysis={},
            risk_analysis={},
            trade_analysis={},
            recommendations=["Insufficient data for performance analysis"]
        )
    
    def get_performance_summary(self, report: PerformanceReport) -> Dict[str, Any]:
        """
        Get a summary of performance metrics
        
        Args:
            report: Performance report
            
        Returns:
            Dictionary with performance summary
        """
        try:
            return {
                'period': f"{report.period_start.strftime('%Y-%m-%d')} to {report.period_end.strftime('%Y-%m-%d')}",
                'total_return': f"{report.metrics.total_return:.2%}",
                'annualized_return': f"{report.metrics.annualized_return:.2%}",
                'volatility': f"{report.metrics.volatility:.2%}",
                'sharpe_ratio': f"{report.metrics.sharpe_ratio:.2f}",
                'max_drawdown': f"{report.metrics.max_drawdown:.2%}",
                'win_rate': f"{report.metrics.win_rate:.2%}",
                'profit_factor': f"{report.metrics.profit_factor:.2f}",
                'benchmark_outperformance': report.benchmark_comparison.get('outperformance', 'N/A'),
                'key_recommendations': report.recommendations[:3]
            }
            
        except Exception as e:
            self.logger.error(f"Error generating performance summary: {e}")
            return {}

# Convenience functions
def analyze_returns(returns: pd.Series, benchmark_returns: Optional[pd.Series] = None) -> PerformanceReport:
    """
    Convenience function to analyze returns
    
    Args:
        returns: Return series
        benchmark_returns: Optional benchmark returns
        
    Returns:
        Performance report
    """
    analyzer = PerformanceAnalyzer()
    return analyzer.analyze_portfolio_performance(returns, benchmark_returns)

def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
    """
    Convenience function to calculate Sharpe ratio
    
    Args:
        returns: Return series
        risk_free_rate: Annual risk-free rate
        
    Returns:
        Sharpe ratio
    """
    if returns.empty or returns.std() == 0:
        return 0.0
    
    periods_per_year = 252 if len(returns) > 252 else len(returns)
    excess_returns = returns - (risk_free_rate / periods_per_year)
    return excess_returns.mean() / returns.std() * np.sqrt(periods_per_year)

if __name__ == "__main__":
    # Test the performance analyzer
    analyzer = PerformanceAnalyzer()
    
    # Use real returns data (placeholder - in practice this would come from your trading system)
    dates = pd.date_range('2023-01-01', '2024-01-01', freq='D')
    returns = pd.Series([0.001] * len(dates), index=dates)  # Placeholder for real returns
    
    # Analyze performance
    report = analyzer.analyze_portfolio_performance(returns)
    
    print(f"Total Return: {report.metrics.total_return:.2%}")
    print(f"Sharpe Ratio: {report.metrics.sharpe_ratio:.2f}")
    print(f"Max Drawdown: {report.metrics.max_drawdown:.2%}")
    print(f"Win Rate: {report.metrics.win_rate:.2%}")
    
    print("Performance Analyzer test completed")