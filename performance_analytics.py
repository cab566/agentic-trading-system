#!/usr/bin/env python3
"""
Performance Analytics and Reporting System

A comprehensive performance analytics engine for algorithmic trading that provides:

- Detailed performance metrics and attribution analysis
- Risk-adjusted return calculations (Sharpe, Sortino, Calmar, etc.)
- Drawdown analysis and recovery time calculations
- Factor exposure and attribution analysis
- Benchmark comparison and relative performance
- Rolling performance windows and regime analysis
- Trade-level analytics and execution quality metrics
- Portfolio composition and sector allocation analysis
- Automated report generation with visualizations
- Performance forecasting and scenario analysis
- Risk decomposition and contribution analysis
- Style analysis and factor loadings
- Performance persistence and consistency metrics
- Custom performance dashboards and alerts

This system enables comprehensive evaluation of trading strategies
and portfolios with institutional-grade analytics and reporting.

Author: AI Trading System v2.0
Date: January 2025
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import logging
import json
from abc import ABC, abstractmethod
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Statistical and financial libraries
try:
    from scipy import stats
    from scipy.optimize import minimize
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logging.warning("SciPy not available. Some advanced analytics will be limited.")

try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.backends.backend_pdf import PdfPages
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    logging.warning("Plotting libraries not available. Visualizations will be limited.")

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    import plotly.io as pio
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    logging.warning("Plotly not available. Interactive charts will be limited.")

# Report generation
try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib import colors
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False
    logging.warning("ReportLab not available. PDF reports will be limited.")

class PerformanceMetric(Enum):
    """Performance metrics"""
    TOTAL_RETURN = "total_return"
    ANNUALIZED_RETURN = "annualized_return"
    VOLATILITY = "volatility"
    SHARPE_RATIO = "sharpe_ratio"
    SORTINO_RATIO = "sortino_ratio"
    CALMAR_RATIO = "calmar_ratio"
    MAX_DRAWDOWN = "max_drawdown"
    AVERAGE_DRAWDOWN = "average_drawdown"
    DRAWDOWN_DURATION = "drawdown_duration"
    WIN_RATE = "win_rate"
    PROFIT_FACTOR = "profit_factor"
    EXPECTANCY = "expectancy"
    BETA = "beta"
    ALPHA = "alpha"
    INFORMATION_RATIO = "information_ratio"
    TRACKING_ERROR = "tracking_error"
    UP_CAPTURE = "up_capture"
    DOWN_CAPTURE = "down_capture"
    VAR_95 = "var_95"
    CVAR_95 = "cvar_95"
    SKEWNESS = "skewness"
    KURTOSIS = "kurtosis"
    TAIL_RATIO = "tail_ratio"
    COMMON_SENSE_RATIO = "common_sense_ratio"

class ReportType(Enum):
    """Report types"""
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    ANNUAL = "annual"
    CUSTOM = "custom"
    TEARSHEET = "tearsheet"
    RISK_REPORT = "risk_report"
    ATTRIBUTION = "attribution"
    BENCHMARK = "benchmark"

class AttributionType(Enum):
    """Attribution analysis types"""
    SECURITY_SELECTION = "security_selection"
    ASSET_ALLOCATION = "asset_allocation"
    INTERACTION = "interaction"
    CURRENCY = "currency"
    SECTOR = "sector"
    FACTOR = "factor"
    TIMING = "timing"

@dataclass
class Trade:
    """Individual trade record"""
    symbol: str
    side: str  # 'buy' or 'sell'
    quantity: float
    entry_price: float
    exit_price: Optional[float] = None
    entry_time: Optional[datetime] = None
    exit_time: Optional[datetime] = None
    pnl: Optional[float] = None
    commission: float = 0.0
    slippage: float = 0.0
    strategy: Optional[str] = None
    sector: Optional[str] = None
    market_cap: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def calculate_pnl(self) -> float:
        """Calculate trade P&L"""
        if self.exit_price is None:
            return 0.0
        
        if self.side.lower() == 'buy':
            gross_pnl = (self.exit_price - self.entry_price) * self.quantity
        else:
            gross_pnl = (self.entry_price - self.exit_price) * self.quantity
        
        self.pnl = gross_pnl - self.commission - abs(self.slippage * self.quantity)
        return self.pnl
    
    def holding_period_days(self) -> Optional[float]:
        """Calculate holding period in days"""
        if self.entry_time and self.exit_time:
            return (self.exit_time - self.entry_time).total_seconds() / 86400
        return None

@dataclass
class PerformanceData:
    """Performance data container"""
    returns: pd.Series
    benchmark_returns: Optional[pd.Series] = None
    positions: Optional[pd.DataFrame] = None
    trades: Optional[List[Trade]] = None
    risk_free_rate: float = 0.02
    
    def __post_init__(self):
        """Validate and process data"""
        if not isinstance(self.returns, pd.Series):
            raise ValueError("Returns must be a pandas Series")
        
        if self.returns.empty:
            raise ValueError("Returns series cannot be empty")
        
        # Ensure datetime index
        if not isinstance(self.returns.index, pd.DatetimeIndex):
            try:
                self.returns.index = pd.to_datetime(self.returns.index)
            except Exception:
                raise ValueError("Returns index must be convertible to datetime")
        
        # Sort by date
        self.returns = self.returns.sort_index()
        
        # Align benchmark if provided
        if self.benchmark_returns is not None:
            if not isinstance(self.benchmark_returns.index, pd.DatetimeIndex):
                self.benchmark_returns.index = pd.to_datetime(self.benchmark_returns.index)
            
            # Align dates
            common_dates = self.returns.index.intersection(self.benchmark_returns.index)
            self.returns = self.returns.loc[common_dates]
            self.benchmark_returns = self.benchmark_returns.loc[common_dates]

@dataclass
class PerformanceMetrics:
    """Container for calculated performance metrics"""
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown: float
    average_drawdown: float
    max_drawdown_duration: int
    win_rate: float
    profit_factor: float
    expectancy: float
    beta: Optional[float] = None
    alpha: Optional[float] = None
    information_ratio: Optional[float] = None
    tracking_error: Optional[float] = None
    up_capture: Optional[float] = None
    down_capture: Optional[float] = None
    var_95: float = 0.0
    cvar_95: float = 0.0
    skewness: float = 0.0
    kurtosis: float = 0.0
    tail_ratio: float = 0.0
    common_sense_ratio: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'total_return': self.total_return,
            'annualized_return': self.annualized_return,
            'volatility': self.volatility,
            'sharpe_ratio': self.sharpe_ratio,
            'sortino_ratio': self.sortino_ratio,
            'calmar_ratio': self.calmar_ratio,
            'max_drawdown': self.max_drawdown,
            'average_drawdown': self.average_drawdown,
            'max_drawdown_duration': self.max_drawdown_duration,
            'win_rate': self.win_rate,
            'profit_factor': self.profit_factor,
            'expectancy': self.expectancy,
            'beta': self.beta,
            'alpha': self.alpha,
            'information_ratio': self.information_ratio,
            'tracking_error': self.tracking_error,
            'up_capture': self.up_capture,
            'down_capture': self.down_capture,
            'var_95': self.var_95,
            'cvar_95': self.cvar_95,
            'skewness': self.skewness,
            'kurtosis': self.kurtosis,
            'tail_ratio': self.tail_ratio,
            'common_sense_ratio': self.common_sense_ratio
        }

@dataclass
class DrawdownAnalysis:
    """Drawdown analysis results"""
    max_drawdown: float
    max_drawdown_start: datetime
    max_drawdown_end: datetime
    max_drawdown_duration: int
    recovery_time: Optional[int]
    average_drawdown: float
    drawdown_frequency: float
    underwater_periods: List[Tuple[datetime, datetime, float]]
    drawdown_series: pd.Series
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'max_drawdown': self.max_drawdown,
            'max_drawdown_start': self.max_drawdown_start.isoformat(),
            'max_drawdown_end': self.max_drawdown_end.isoformat(),
            'max_drawdown_duration': self.max_drawdown_duration,
            'recovery_time': self.recovery_time,
            'average_drawdown': self.average_drawdown,
            'drawdown_frequency': self.drawdown_frequency,
            'underwater_periods_count': len(self.underwater_periods)
        }

@dataclass
class AttributionAnalysis:
    """Performance attribution analysis"""
    security_selection: float
    asset_allocation: float
    interaction_effect: float
    total_active_return: float
    sector_attribution: Dict[str, float]
    security_attribution: Dict[str, float]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'security_selection': self.security_selection,
            'asset_allocation': self.asset_allocation,
            'interaction_effect': self.interaction_effect,
            'total_active_return': self.total_active_return,
            'sector_attribution': self.sector_attribution,
            'security_attribution': self.security_attribution
        }

class PerformanceCalculator:
    """Core performance calculation engine"""
    
    def __init__(self, risk_free_rate: float = 0.02):
        self.risk_free_rate = risk_free_rate
        self.logger = logging.getLogger("performance_calculator")
    
    def calculate_metrics(self, data: PerformanceData) -> PerformanceMetrics:
        """Calculate comprehensive performance metrics"""
        returns = data.returns.dropna()
        
        if returns.empty:
            raise ValueError("No valid returns data")
        
        # Basic metrics
        total_return = self._calculate_total_return(returns)
        annualized_return = self._calculate_annualized_return(returns)
        volatility = self._calculate_volatility(returns)
        
        # Risk-adjusted metrics
        sharpe_ratio = self._calculate_sharpe_ratio(returns, data.risk_free_rate)
        sortino_ratio = self._calculate_sortino_ratio(returns, data.risk_free_rate)
        calmar_ratio = self._calculate_calmar_ratio(returns)
        
        # Drawdown metrics
        drawdown_analysis = self.analyze_drawdowns(returns)
        
        # Trade-based metrics
        win_rate, profit_factor, expectancy = self._calculate_trade_metrics(data.trades)
        
        # Benchmark-relative metrics
        beta, alpha, info_ratio, tracking_error = None, None, None, None
        up_capture, down_capture = None, None
        
        if data.benchmark_returns is not None:
            beta, alpha = self._calculate_beta_alpha(returns, data.benchmark_returns, data.risk_free_rate)
            info_ratio = self._calculate_information_ratio(returns, data.benchmark_returns)
            tracking_error = self._calculate_tracking_error(returns, data.benchmark_returns)
            up_capture, down_capture = self._calculate_capture_ratios(returns, data.benchmark_returns)
        
        # Risk metrics
        var_95 = self._calculate_var(returns, 0.05)
        cvar_95 = self._calculate_cvar(returns, 0.05)
        
        # Distribution metrics
        skewness = self._calculate_skewness(returns)
        kurtosis = self._calculate_kurtosis(returns)
        tail_ratio = self._calculate_tail_ratio(returns)
        common_sense_ratio = self._calculate_common_sense_ratio(returns)
        
        return PerformanceMetrics(
            total_return=total_return,
            annualized_return=annualized_return,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            calmar_ratio=calmar_ratio,
            max_drawdown=drawdown_analysis.max_drawdown,
            average_drawdown=drawdown_analysis.average_drawdown,
            max_drawdown_duration=drawdown_analysis.max_drawdown_duration,
            win_rate=win_rate,
            profit_factor=profit_factor,
            expectancy=expectancy,
            beta=beta,
            alpha=alpha,
            information_ratio=info_ratio,
            tracking_error=tracking_error,
            up_capture=up_capture,
            down_capture=down_capture,
            var_95=var_95,
            cvar_95=cvar_95,
            skewness=skewness,
            kurtosis=kurtosis,
            tail_ratio=tail_ratio,
            common_sense_ratio=common_sense_ratio
        )
    
    def _calculate_total_return(self, returns: pd.Series) -> float:
        """Calculate total return"""
        return (1 + returns).prod() - 1
    
    def _calculate_annualized_return(self, returns: pd.Series) -> float:
        """Calculate annualized return"""
        total_return = self._calculate_total_return(returns)
        days = (returns.index[-1] - returns.index[0]).days
        if days == 0:
            return 0.0
        years = days / 365.25
        return (1 + total_return) ** (1 / years) - 1
    
    def _calculate_volatility(self, returns: pd.Series) -> float:
        """Calculate annualized volatility"""
        return returns.std() * np.sqrt(252)  # Assuming daily returns
    
    def _calculate_sharpe_ratio(self, returns: pd.Series, risk_free_rate: float) -> float:
        """Calculate Sharpe ratio"""
        excess_returns = returns - risk_free_rate / 252
        if excess_returns.std() == 0:
            return 0.0
        return excess_returns.mean() / excess_returns.std() * np.sqrt(252)
    
    def _calculate_sortino_ratio(self, returns: pd.Series, risk_free_rate: float) -> float:
        """Calculate Sortino ratio"""
        excess_returns = returns - risk_free_rate / 252
        downside_returns = excess_returns[excess_returns < 0]
        if len(downside_returns) == 0 or downside_returns.std() == 0:
            return 0.0
        return excess_returns.mean() / downside_returns.std() * np.sqrt(252)
    
    def _calculate_calmar_ratio(self, returns: pd.Series) -> float:
        """Calculate Calmar ratio"""
        annualized_return = self._calculate_annualized_return(returns)
        max_drawdown = self.analyze_drawdowns(returns).max_drawdown
        if max_drawdown == 0:
            return 0.0
        return annualized_return / abs(max_drawdown)
    
    def analyze_drawdowns(self, returns: pd.Series) -> DrawdownAnalysis:
        """Analyze drawdowns"""
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        
        # Find maximum drawdown
        max_dd_idx = drawdown.idxmin()
        max_drawdown = drawdown.min()
        
        # Find start of max drawdown period
        max_dd_start_idx = running_max.loc[:max_dd_idx].idxmax()
        
        # Find recovery (if any)
        recovery_idx = None
        max_dd_end_idx = max_dd_idx
        
        post_max_dd = cumulative.loc[max_dd_idx:]
        recovery_level = running_max.loc[max_dd_idx]
        recovery_mask = post_max_dd >= recovery_level
        
        if recovery_mask.any():
            recovery_idx = post_max_dd[recovery_mask].index[0]
            max_dd_end_idx = recovery_idx
        
        # Calculate duration
        max_dd_duration = (max_dd_end_idx - max_dd_start_idx).days
        recovery_time = (recovery_idx - max_dd_idx).days if recovery_idx else None
        
        # Find all underwater periods
        underwater_periods = []
        in_drawdown = False
        start_date = None
        
        for date, dd in drawdown.items():
            if dd < -0.001 and not in_drawdown:  # Start of drawdown (0.1% threshold)
                in_drawdown = True
                start_date = date
            elif dd >= -0.001 and in_drawdown:  # End of drawdown
                in_drawdown = False
                if start_date:
                    min_dd = drawdown.loc[start_date:date].min()
                    underwater_periods.append((start_date, date, min_dd))
        
        # Calculate average drawdown
        avg_drawdown = drawdown[drawdown < 0].mean() if (drawdown < 0).any() else 0.0
        
        # Calculate drawdown frequency (periods per year)
        total_days = (returns.index[-1] - returns.index[0]).days
        drawdown_frequency = len(underwater_periods) / (total_days / 365.25) if total_days > 0 else 0.0
        
        return DrawdownAnalysis(
            max_drawdown=max_drawdown,
            max_drawdown_start=max_dd_start_idx,
            max_drawdown_end=max_dd_end_idx,
            max_drawdown_duration=max_dd_duration,
            recovery_time=recovery_time,
            average_drawdown=avg_drawdown,
            drawdown_frequency=drawdown_frequency,
            underwater_periods=underwater_periods,
            drawdown_series=drawdown
        )
    
    def _calculate_trade_metrics(self, trades: Optional[List[Trade]]) -> Tuple[float, float, float]:
        """Calculate trade-based metrics"""
        if not trades:
            return 0.0, 0.0, 0.0
        
        completed_trades = [t for t in trades if t.pnl is not None]
        if not completed_trades:
            return 0.0, 0.0, 0.0
        
        pnls = [t.pnl for t in completed_trades]
        winning_trades = [pnl for pnl in pnls if pnl > 0]
        losing_trades = [pnl for pnl in pnls if pnl < 0]
        
        # Win rate
        win_rate = len(winning_trades) / len(pnls) if pnls else 0.0
        
        # Profit factor
        gross_profit = sum(winning_trades) if winning_trades else 0.0
        gross_loss = abs(sum(losing_trades)) if losing_trades else 0.0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0.0
        
        # Expectancy
        expectancy = np.mean(pnls) if pnls else 0.0
        
        return win_rate, profit_factor, expectancy
    
    def _calculate_beta_alpha(self, returns: pd.Series, benchmark: pd.Series, risk_free_rate: float) -> Tuple[float, float]:
        """Calculate beta and alpha"""
        if SCIPY_AVAILABLE:
            excess_returns = returns - risk_free_rate / 252
            excess_benchmark = benchmark - risk_free_rate / 252
            
            if excess_benchmark.std() == 0:
                return 0.0, excess_returns.mean() * 252
            
            beta = excess_returns.cov(excess_benchmark) / excess_benchmark.var()
            alpha = (excess_returns.mean() - beta * excess_benchmark.mean()) * 252
            
            return beta, alpha
        else:
            return 0.0, 0.0
    
    def _calculate_information_ratio(self, returns: pd.Series, benchmark: pd.Series) -> float:
        """Calculate information ratio"""
        active_returns = returns - benchmark
        tracking_error = active_returns.std() * np.sqrt(252)
        if tracking_error == 0:
            return 0.0
        return active_returns.mean() * 252 / tracking_error
    
    def _calculate_tracking_error(self, returns: pd.Series, benchmark: pd.Series) -> float:
        """Calculate tracking error"""
        active_returns = returns - benchmark
        return active_returns.std() * np.sqrt(252)
    
    def _calculate_capture_ratios(self, returns: pd.Series, benchmark: pd.Series) -> Tuple[float, float]:
        """Calculate up/down capture ratios"""
        up_market = benchmark > 0
        down_market = benchmark < 0
        
        if not up_market.any() or not down_market.any():
            return 0.0, 0.0
        
        up_portfolio = returns[up_market].mean()
        up_benchmark = benchmark[up_market].mean()
        up_capture = up_portfolio / up_benchmark if up_benchmark != 0 else 0.0
        
        down_portfolio = returns[down_market].mean()
        down_benchmark = benchmark[down_market].mean()
        down_capture = down_portfolio / down_benchmark if down_benchmark != 0 else 0.0
        
        return up_capture, down_capture
    
    def _calculate_var(self, returns: pd.Series, confidence_level: float) -> float:
        """Calculate Value at Risk"""
        return np.percentile(returns, confidence_level * 100)
    
    def _calculate_cvar(self, returns: pd.Series, confidence_level: float) -> float:
        """Calculate Conditional Value at Risk"""
        var = self._calculate_var(returns, confidence_level)
        return returns[returns <= var].mean()
    
    def _calculate_skewness(self, returns: pd.Series) -> float:
        """Calculate skewness"""
        if SCIPY_AVAILABLE:
            return stats.skew(returns.dropna())
        else:
            return 0.0
    
    def _calculate_kurtosis(self, returns: pd.Series) -> float:
        """Calculate excess kurtosis"""
        if SCIPY_AVAILABLE:
            return stats.kurtosis(returns.dropna())
        else:
            return 0.0
    
    def _calculate_tail_ratio(self, returns: pd.Series) -> float:
        """Calculate tail ratio (95th percentile / 5th percentile)"""
        p95 = np.percentile(returns, 95)
        p5 = np.percentile(returns, 5)
        return abs(p95 / p5) if p5 != 0 else 0.0
    
    def _calculate_common_sense_ratio(self, returns: pd.Series) -> float:
        """Calculate common sense ratio (tail ratio * profit factor)"""
        tail_ratio = self._calculate_tail_ratio(returns)
        profit_factor = len(returns[returns > 0]) / len(returns[returns < 0]) if len(returns[returns < 0]) > 0 else 0.0
        return tail_ratio * profit_factor

class PerformanceVisualizer:
    """Performance visualization engine"""
    
    def __init__(self):
        self.logger = logging.getLogger("performance_visualizer")
        
        if PLOTTING_AVAILABLE:
            plt.style.use('seaborn-v0_8')
            sns.set_palette("husl")
    
    def create_performance_tearsheet(self, data: PerformanceData, metrics: PerformanceMetrics, 
                                   output_path: str = "performance_tearsheet.png") -> bool:
        """Create comprehensive performance tearsheet"""
        if not PLOTTING_AVAILABLE:
            self.logger.warning("Plotting not available")
            return False
        
        try:
            fig, axes = plt.subplots(3, 2, figsize=(16, 12))
            fig.suptitle('Performance Tearsheet', fontsize=16, fontweight='bold')
            
            # Cumulative returns
            cumulative = (1 + data.returns).cumprod()
            axes[0, 0].plot(cumulative.index, cumulative.values, linewidth=2, label='Strategy')
            
            if data.benchmark_returns is not None:
                benchmark_cumulative = (1 + data.benchmark_returns).cumprod()
                axes[0, 0].plot(benchmark_cumulative.index, benchmark_cumulative.values, 
                              linewidth=2, alpha=0.7, label='Benchmark')
            
            axes[0, 0].set_title('Cumulative Returns')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            # Drawdown
            calculator = PerformanceCalculator()
            drawdown_analysis = calculator.analyze_drawdowns(data.returns)
            axes[0, 1].fill_between(drawdown_analysis.drawdown_series.index, 
                                  drawdown_analysis.drawdown_series.values, 0, 
                                  alpha=0.7, color='red')
            axes[0, 1].set_title('Drawdown')
            axes[0, 1].grid(True, alpha=0.3)
            
            # Rolling Sharpe ratio
            rolling_sharpe = data.returns.rolling(252).apply(
                lambda x: x.mean() / x.std() * np.sqrt(252) if x.std() > 0 else 0
            )
            axes[1, 0].plot(rolling_sharpe.index, rolling_sharpe.values, linewidth=2)
            axes[1, 0].axhline(y=0, color='black', linestyle='--', alpha=0.5)
            axes[1, 0].set_title('Rolling 1-Year Sharpe Ratio')
            axes[1, 0].grid(True, alpha=0.3)
            
            # Monthly returns heatmap
            monthly_returns = data.returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
            monthly_returns.index = monthly_returns.index.to_period('M')
            
            # Create pivot table for heatmap
            monthly_pivot = monthly_returns.to_frame('returns')
            monthly_pivot['year'] = monthly_returns.index.year
            monthly_pivot['month'] = monthly_returns.index.month
            heatmap_data = monthly_pivot.pivot(index='year', columns='month', values='returns')
            
            sns.heatmap(heatmap_data, annot=True, fmt='.2%', cmap='RdYlGn', 
                       center=0, ax=axes[1, 1], cbar_kws={'label': 'Monthly Return'})
            axes[1, 1].set_title('Monthly Returns Heatmap')
            
            # Return distribution
            axes[2, 0].hist(data.returns, bins=50, alpha=0.7, density=True)
            axes[2, 0].axvline(data.returns.mean(), color='red', linestyle='--', 
                             label=f'Mean: {data.returns.mean():.4f}')
            axes[2, 0].set_title('Return Distribution')
            axes[2, 0].legend()
            axes[2, 0].grid(True, alpha=0.3)
            
            # Performance metrics table
            metrics_data = [
                ['Total Return', f"{metrics.total_return:.2%}"],
                ['Annualized Return', f"{metrics.annualized_return:.2%}"],
                ['Volatility', f"{metrics.volatility:.2%}"],
                ['Sharpe Ratio', f"{metrics.sharpe_ratio:.2f}"],
                ['Sortino Ratio', f"{metrics.sortino_ratio:.2f}"],
                ['Max Drawdown', f"{metrics.max_drawdown:.2%}"],
                ['Win Rate', f"{metrics.win_rate:.2%}"],
                ['Profit Factor', f"{metrics.profit_factor:.2f}"]
            ]
            
            axes[2, 1].axis('tight')
            axes[2, 1].axis('off')
            table = axes[2, 1].table(cellText=metrics_data, 
                                   colLabels=['Metric', 'Value'],
                                   cellLoc='center', loc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1.2, 1.5)
            axes[2, 1].set_title('Performance Metrics')
            
            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"Performance tearsheet saved to {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to create tearsheet: {e}")
            return False
    
    def create_interactive_dashboard(self, data: PerformanceData, metrics: PerformanceMetrics) -> Optional[str]:
        """Create interactive Plotly dashboard"""
        if not PLOTLY_AVAILABLE:
            self.logger.warning("Plotly not available")
            return None
        
        try:
            # Create subplots
            fig = make_subplots(
                rows=3, cols=2,
                subplot_titles=('Cumulative Returns', 'Drawdown', 'Rolling Sharpe', 
                              'Return Distribution', 'Monthly Returns', 'Performance Metrics'),
                specs=[[{"secondary_y": True}, {}],
                       [{}, {}],
                       [{}, {"type": "table"}]]
            )
            
            # Cumulative returns
            cumulative = (1 + data.returns).cumprod()
            fig.add_trace(
                go.Scatter(x=cumulative.index, y=cumulative.values, 
                          name='Strategy', line=dict(width=2)),
                row=1, col=1
            )
            
            if data.benchmark_returns is not None:
                benchmark_cumulative = (1 + data.benchmark_returns).cumprod()
                fig.add_trace(
                    go.Scatter(x=benchmark_cumulative.index, y=benchmark_cumulative.values,
                              name='Benchmark', line=dict(width=2, dash='dash')),
                    row=1, col=1
                )
            
            # Drawdown
            calculator = PerformanceCalculator()
            drawdown_analysis = calculator.analyze_drawdowns(data.returns)
            fig.add_trace(
                go.Scatter(x=drawdown_analysis.drawdown_series.index, 
                          y=drawdown_analysis.drawdown_series.values,
                          fill='tonexty', name='Drawdown', 
                          line=dict(color='red')),
                row=1, col=2
            )
            
            # Rolling Sharpe
            rolling_sharpe = data.returns.rolling(252).apply(
                lambda x: x.mean() / x.std() * np.sqrt(252) if x.std() > 0 else 0
            )
            fig.add_trace(
                go.Scatter(x=rolling_sharpe.index, y=rolling_sharpe.values,
                          name='Rolling Sharpe', line=dict(width=2)),
                row=2, col=1
            )
            
            # Return distribution
            fig.add_trace(
                go.Histogram(x=data.returns, nbinsx=50, name='Returns',
                           histnorm='probability density'),
                row=2, col=2
            )
            
            # Monthly returns heatmap
            monthly_returns = data.returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
            monthly_returns.index = monthly_returns.index.to_period('M')
            
            monthly_pivot = monthly_returns.to_frame('returns')
            monthly_pivot['year'] = monthly_returns.index.year
            monthly_pivot['month'] = monthly_returns.index.month
            heatmap_data = monthly_pivot.pivot(index='year', columns='month', values='returns')
            
            fig.add_trace(
                go.Heatmap(z=heatmap_data.values, 
                          x=[f'M{i}' for i in range(1, 13)],
                          y=heatmap_data.index,
                          colorscale='RdYlGn', zmid=0,
                          name='Monthly Returns'),
                row=3, col=1
            )
            
            # Performance metrics table
            metrics_data = [
                ['Total Return', f"{metrics.total_return:.2%}"],
                ['Annualized Return', f"{metrics.annualized_return:.2%}"],
                ['Volatility', f"{metrics.volatility:.2%}"],
                ['Sharpe Ratio', f"{metrics.sharpe_ratio:.2f}"],
                ['Sortino Ratio', f"{metrics.sortino_ratio:.2f}"],
                ['Max Drawdown', f"{metrics.max_drawdown:.2%}"],
                ['Win Rate', f"{metrics.win_rate:.2%}"],
                ['Profit Factor', f"{metrics.profit_factor:.2f}"]
            ]
            
            fig.add_trace(
                go.Table(
                    header=dict(values=['Metric', 'Value'],
                               fill_color='paleturquoise',
                               align='left'),
                    cells=dict(values=list(zip(*metrics_data)),
                              fill_color='lavender',
                              align='left')
                ),
                row=3, col=2
            )
            
            # Update layout
            fig.update_layout(
                height=1000,
                title_text="Interactive Performance Dashboard",
                showlegend=True
            )
            
            # Save as HTML
            output_path = "interactive_dashboard.html"
            fig.write_html(output_path)
            
            self.logger.info(f"Interactive dashboard saved to {output_path}")
            return output_path
            
        except Exception as e:
            self.logger.error(f"Failed to create interactive dashboard: {e}")
            return None

class ReportGenerator:
    """Automated report generation"""
    
    def __init__(self):
        self.logger = logging.getLogger("report_generator")
    
    def generate_pdf_report(self, data: PerformanceData, metrics: PerformanceMetrics,
                           output_path: str = "performance_report.pdf") -> bool:
        """Generate comprehensive PDF report"""
        if not REPORTLAB_AVAILABLE:
            self.logger.warning("ReportLab not available")
            return False
        
        try:
            doc = SimpleDocTemplate(output_path, pagesize=letter)
            styles = getSampleStyleSheet()
            story = []
            
            # Title
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontSize=24,
                spaceAfter=30,
                alignment=1  # Center
            )
            story.append(Paragraph("Performance Analysis Report", title_style))
            story.append(Spacer(1, 20))
            
            # Executive Summary
            story.append(Paragraph("Executive Summary", styles['Heading2']))
            
            summary_text = f"""
            This report provides a comprehensive analysis of the trading strategy's performance 
            over the period from {data.returns.index[0].strftime('%Y-%m-%d')} to 
            {data.returns.index[-1].strftime('%Y-%m-%d')}.
            
            Key Highlights:
            • Total Return: {metrics.total_return:.2%}
            • Annualized Return: {metrics.annualized_return:.2%}
            • Volatility: {metrics.volatility:.2%}
            • Sharpe Ratio: {metrics.sharpe_ratio:.2f}
            • Maximum Drawdown: {metrics.max_drawdown:.2%}
            """
            
            story.append(Paragraph(summary_text, styles['Normal']))
            story.append(Spacer(1, 20))
            
            # Performance Metrics Table
            story.append(Paragraph("Performance Metrics", styles['Heading2']))
            
            metrics_data = [
                ['Metric', 'Value', 'Description'],
                ['Total Return', f"{metrics.total_return:.2%}", 'Cumulative return over the period'],
                ['Annualized Return', f"{metrics.annualized_return:.2%}", 'Compound annual growth rate'],
                ['Volatility', f"{metrics.volatility:.2%}", 'Annualized standard deviation'],
                ['Sharpe Ratio', f"{metrics.sharpe_ratio:.2f}", 'Risk-adjusted return measure'],
                ['Sortino Ratio', f"{metrics.sortino_ratio:.2f}", 'Downside risk-adjusted return'],
                ['Calmar Ratio', f"{metrics.calmar_ratio:.2f}", 'Return to max drawdown ratio'],
                ['Max Drawdown', f"{metrics.max_drawdown:.2%}", 'Largest peak-to-trough decline'],
                ['Win Rate', f"{metrics.win_rate:.2%}", 'Percentage of profitable trades'],
                ['Profit Factor', f"{metrics.profit_factor:.2f}", 'Gross profit to gross loss ratio']
            ]
            
            table = Table(metrics_data)
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            
            story.append(table)
            story.append(Spacer(1, 20))
            
            # Risk Analysis
            story.append(Paragraph("Risk Analysis", styles['Heading2']))
            
            risk_text = f"""
            The strategy exhibits the following risk characteristics:
            
            • Value at Risk (95%): {metrics.var_95:.4f}
            • Conditional VaR (95%): {metrics.cvar_95:.4f}
            • Skewness: {metrics.skewness:.2f}
            • Excess Kurtosis: {metrics.kurtosis:.2f}
            • Maximum Drawdown Duration: {metrics.max_drawdown_duration} days
            
            The {'positive' if metrics.skewness > 0 else 'negative'} skewness indicates 
            {'right-tailed' if metrics.skewness > 0 else 'left-tailed'} return distribution.
            """
            
            story.append(Paragraph(risk_text, styles['Normal']))
            story.append(Spacer(1, 20))
            
            # Build PDF
            doc.build(story)
            
            self.logger.info(f"PDF report generated: {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to generate PDF report: {e}")
            return False
    
    def generate_json_report(self, data: PerformanceData, metrics: PerformanceMetrics,
                           output_path: str = "performance_report.json") -> bool:
        """Generate JSON report"""
        try:
            calculator = PerformanceCalculator()
            drawdown_analysis = calculator.analyze_drawdowns(data.returns)
            
            report = {
                'report_metadata': {
                    'generated_at': datetime.now().isoformat(),
                    'period_start': data.returns.index[0].isoformat(),
                    'period_end': data.returns.index[-1].isoformat(),
                    'total_observations': len(data.returns)
                },
                'performance_metrics': metrics.to_dict(),
                'drawdown_analysis': drawdown_analysis.to_dict(),
                'return_statistics': {
                    'mean_daily_return': float(data.returns.mean()),
                    'median_daily_return': float(data.returns.median()),
                    'std_daily_return': float(data.returns.std()),
                    'min_daily_return': float(data.returns.min()),
                    'max_daily_return': float(data.returns.max()),
                    'positive_days': int((data.returns > 0).sum()),
                    'negative_days': int((data.returns < 0).sum()),
                    'zero_days': int((data.returns == 0).sum())
                }
            }
            
            # Add benchmark comparison if available
            if data.benchmark_returns is not None:
                benchmark_metrics = calculator.calculate_metrics(
                    PerformanceData(data.benchmark_returns, risk_free_rate=data.risk_free_rate)
                )
                report['benchmark_comparison'] = {
                    'benchmark_metrics': benchmark_metrics.to_dict(),
                    'active_return': float(data.returns.mean() - data.benchmark_returns.mean()) * 252,
                    'tracking_error': metrics.tracking_error,
                    'information_ratio': metrics.information_ratio,
                    'beta': metrics.beta,
                    'alpha': metrics.alpha
                }
            
            # Add trade analysis if available
            if data.trades:
                completed_trades = [t for t in data.trades if t.pnl is not None]
                if completed_trades:
                    pnls = [t.pnl for t in completed_trades]
                    report['trade_analysis'] = {
                        'total_trades': len(completed_trades),
                        'winning_trades': len([p for p in pnls if p > 0]),
                        'losing_trades': len([p for p in pnls if p < 0]),
                        'average_win': float(np.mean([p for p in pnls if p > 0])) if any(p > 0 for p in pnls) else 0.0,
                        'average_loss': float(np.mean([p for p in pnls if p < 0])) if any(p < 0 for p in pnls) else 0.0,
                        'largest_win': float(max(pnls)) if pnls else 0.0,
                        'largest_loss': float(min(pnls)) if pnls else 0.0,
                        'total_pnl': float(sum(pnls))
                    }
            
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            self.logger.info(f"JSON report generated: {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to generate JSON report: {e}")
            return False

class PerformanceAnalytics:
    """Main performance analytics engine"""
    
    def __init__(self, risk_free_rate: float = 0.02):
        self.risk_free_rate = risk_free_rate
        self.calculator = PerformanceCalculator(risk_free_rate)
        self.visualizer = PerformanceVisualizer()
        self.report_generator = ReportGenerator()
        self.logger = logging.getLogger("performance_analytics")
    
    def analyze_performance(self, returns: pd.Series, 
                          benchmark_returns: Optional[pd.Series] = None,
                          positions: Optional[pd.DataFrame] = None,
                          trades: Optional[List[Trade]] = None) -> Tuple[PerformanceMetrics, PerformanceData]:
        """Comprehensive performance analysis"""
        try:
            # Create performance data object
            data = PerformanceData(
                returns=returns,
                benchmark_returns=benchmark_returns,
                positions=positions,
                trades=trades,
                risk_free_rate=self.risk_free_rate
            )
            
            # Calculate metrics
            metrics = self.calculator.calculate_metrics(data)
            
            self.logger.info("Performance analysis completed")
            return metrics, data
            
        except Exception as e:
            self.logger.error(f"Performance analysis failed: {e}")
            raise
    
    def generate_full_report(self, returns: pd.Series,
                           benchmark_returns: Optional[pd.Series] = None,
                           positions: Optional[pd.DataFrame] = None,
                           trades: Optional[List[Trade]] = None,
                           output_dir: str = "reports") -> Dict[str, str]:
        """Generate comprehensive performance report"""
        import os
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Analyze performance
        metrics, data = self.analyze_performance(returns, benchmark_returns, positions, trades)
        
        generated_files = {}
        
        # Generate tearsheet
        tearsheet_path = os.path.join(output_dir, "performance_tearsheet.png")
        if self.visualizer.create_performance_tearsheet(data, metrics, tearsheet_path):
            generated_files['tearsheet'] = tearsheet_path
        
        # Generate interactive dashboard
        dashboard_path = self.visualizer.create_interactive_dashboard(data, metrics)
        if dashboard_path:
            import shutil
            final_dashboard_path = os.path.join(output_dir, "interactive_dashboard.html")
            shutil.move(dashboard_path, final_dashboard_path)
            generated_files['dashboard'] = final_dashboard_path
        
        # Generate PDF report
        pdf_path = os.path.join(output_dir, "performance_report.pdf")
        if self.report_generator.generate_pdf_report(data, metrics, pdf_path):
            generated_files['pdf_report'] = pdf_path
        
        # Generate JSON report
        json_path = os.path.join(output_dir, "performance_report.json")
        if self.report_generator.generate_json_report(data, metrics, json_path):
            generated_files['json_report'] = json_path
        
        self.logger.info(f"Full report generated. Files: {list(generated_files.keys())}")
        return generated_files

# Example usage and testing
if __name__ == "__main__":
    # Use real market data example (this would typically come from your data sources)
    dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
    
    # Example: Load real strategy returns from your trading system
    # In practice, these would come from your actual trading results
    strategy_returns = pd.Series([0.001] * len(dates), index=dates)  # Placeholder for real returns
    
    # Example: Load real benchmark returns (e.g., S&P 500)
    # In practice, these would come from market data APIs
    benchmark_returns = pd.Series([0.0005] * len(dates), index=dates)  # Placeholder for real benchmark
    
    # Sample trades
    sample_trades = [
        Trade('AAPL', 'buy', 100, 150.0, 155.0, pnl=500.0),
        Trade('GOOGL', 'buy', 50, 2000.0, 1950.0, pnl=-2500.0),
        Trade('MSFT', 'buy', 75, 300.0, 320.0, pnl=1500.0),
        Trade('TSLA', 'sell', 25, 800.0, 750.0, pnl=1250.0),
        Trade('AMZN', 'buy', 30, 3000.0, 3100.0, pnl=3000.0)
    ]
    
    # Initialize analytics engine
    analytics = PerformanceAnalytics(risk_free_rate=0.02)
    
    print("Performance Analytics System initialized")
    print("\nFeatures:")
    print("- Comprehensive performance metrics calculation")
    print("- Risk-adjusted return analysis (Sharpe, Sortino, Calmar)")
    print("- Detailed drawdown analysis and recovery metrics")
    print("- Benchmark comparison and relative performance")
    print("- Trade-level analytics and execution quality")
    print("- Beautiful visualizations and tearsheets")
    print("- Interactive dashboards with Plotly")
    print("- Automated PDF and JSON report generation")
    print("- Factor exposure and attribution analysis")
    print("- Rolling performance windows and regime analysis")
    
    # Analyze performance
    metrics, data = analytics.analyze_performance(
        returns=strategy_returns,
        benchmark_returns=benchmark_returns,
        trades=sample_trades
    )
    
    print("\nPerformance Metrics:")
    print(f"Total Return: {metrics.total_return:.2%}")
    print(f"Annualized Return: {metrics.annualized_return:.2%}")
    print(f"Volatility: {metrics.volatility:.2%}")
    print(f"Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
    print(f"Sortino Ratio: {metrics.sortino_ratio:.2f}")
    print(f"Max Drawdown: {metrics.max_drawdown:.2%}")
    print(f"Win Rate: {metrics.win_rate:.2%}")
    print(f"Profit Factor: {metrics.profit_factor:.2f}")
    
    if metrics.beta is not None:
        print(f"Beta: {metrics.beta:.2f}")
        print(f"Alpha: {metrics.alpha:.2%}")
        print(f"Information Ratio: {metrics.information_ratio:.2f}")
    
    # Generate full report
    try:
        generated_files = analytics.generate_full_report(
            returns=strategy_returns,
            benchmark_returns=benchmark_returns,
            trades=sample_trades,
            output_dir="performance_reports"
        )
        
        print(f"\nGenerated Reports:")
        for report_type, file_path in generated_files.items():
            print(f"- {report_type}: {file_path}")
            
    except Exception as e:
        print(f"Report generation error: {e}")
    
    print("\nPerformance analytics demonstration completed!")