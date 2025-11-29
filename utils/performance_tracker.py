#!/usr/bin/env python3
"""
Performance Tracker Module

Tracks and analyzes trading system performance metrics including:
- Portfolio returns and risk metrics
- Trade execution statistics
- System performance monitoring
- Real-time performance updates
"""

import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
import pandas as pd
import numpy as np
from collections import defaultdict, deque

from core.config_manager import ConfigManager


@dataclass
class PerformanceMetrics:
    """Comprehensive performance tracking metrics."""
    total_return: float = 0.0
    daily_return: float = 0.0
    annualized_return: float = 0.0
    volatility: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    calmar_ratio: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    var_95: float = 0.0
    var_99: float = 0.0
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    active_positions: int = 0
    portfolio_value: float = 0.0
    cash_balance: float = 0.0
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class TradeMetrics:
    """Individual trade performance metrics."""
    trade_id: str
    symbol: str
    entry_time: datetime
    exit_time: Optional[datetime] = None
    entry_price: float = 0.0
    exit_price: float = 0.0
    quantity: float = 0.0
    pnl: float = 0.0
    pnl_percent: float = 0.0
    duration_minutes: float = 0.0
    trade_type: str = ""  # 'BUY', 'SELL'
    strategy: str = ""
    is_winning: bool = False


class PerformanceTracker:
    """Tracks and analyzes trading system performance."""
    
    def __init__(self, config_manager: ConfigManager):
        """Initialize performance tracker.
        
        Args:
            config_manager: Configuration manager instance
        """
        self.config_manager = config_manager
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Performance data storage
        self.daily_returns = deque(maxlen=252)  # One year of trading days
        self.trade_history: List[TradeMetrics] = []
        self.portfolio_values = deque(maxlen=1000)  # Last 1000 portfolio snapshots
        
        # Current metrics
        self.current_metrics = PerformanceMetrics()
        self.metrics_by_strategy: Dict[str, PerformanceMetrics] = defaultdict(PerformanceMetrics)
        self.metrics_by_asset: Dict[str, PerformanceMetrics] = defaultdict(PerformanceMetrics)
        
        # Configuration
        portfolio_config = config_manager.get_portfolio_config()
        self.initial_capital = portfolio_config.get('initial_capital', 100000.0)
        self.risk_free_rate = portfolio_config.get('risk_free_rate', 0.02)  # 2% annual
        
        self.logger.info("PerformanceTracker initialized")
    
    def record_trade(self, trade_data: Dict[str, Any]) -> None:
        """Record a completed trade.
        
        Args:
            trade_data: Dictionary containing trade information
        """
        try:
            trade = TradeMetrics(
                trade_id=trade_data.get('trade_id', ''),
                symbol=trade_data.get('symbol', ''),
                entry_time=trade_data.get('entry_time', datetime.now(timezone.utc)),
                exit_time=trade_data.get('exit_time'),
                entry_price=trade_data.get('entry_price', 0.0),
                exit_price=trade_data.get('exit_price', 0.0),
                quantity=trade_data.get('quantity', 0.0),
                pnl=trade_data.get('pnl', 0.0),
                pnl_percent=trade_data.get('pnl_percent', 0.0),
                duration_minutes=trade_data.get('duration_minutes', 0.0),
                trade_type=trade_data.get('trade_type', ''),
                strategy=trade_data.get('strategy', ''),
                is_winning=trade_data.get('pnl', 0.0) > 0
            )
            
            self.trade_history.append(trade)
            self._update_metrics()
            
            self.logger.debug(f"Recorded trade: {trade.symbol} PnL: {trade.pnl:.2f}")
            
        except Exception as e:
            self.logger.error(f"Error recording trade: {e}")
    
    def update_portfolio_value(self, portfolio_value: float, cash_balance: float = 0.0) -> None:
        """Update current portfolio value.
        
        Args:
            portfolio_value: Current total portfolio value
            cash_balance: Current cash balance
        """
        try:
            timestamp = datetime.now(timezone.utc)
            
            # Store portfolio snapshot
            self.portfolio_values.append({
                'timestamp': timestamp,
                'value': portfolio_value,
                'cash': cash_balance
            })
            
            # Calculate daily return if we have previous data
            if len(self.portfolio_values) > 1:
                prev_value = self.portfolio_values[-2]['value']
                daily_return = (portfolio_value - prev_value) / prev_value if prev_value > 0 else 0.0
                self.daily_returns.append(daily_return)
            
            # Update current metrics
            self.current_metrics.portfolio_value = portfolio_value
            self.current_metrics.cash_balance = cash_balance
            self.current_metrics.last_updated = timestamp
            
            self._update_metrics()
            
        except Exception as e:
            self.logger.error(f"Error updating portfolio value: {e}")
    
    def _update_metrics(self) -> None:
        """Update all performance metrics."""
        try:
            if not self.portfolio_values:
                return
            
            # Calculate returns
            current_value = self.portfolio_values[-1]['value']
            self.current_metrics.total_return = (current_value - self.initial_capital) / self.initial_capital
            
            if self.daily_returns:
                self.current_metrics.daily_return = self.daily_returns[-1]
                self.current_metrics.annualized_return = np.mean(self.daily_returns) * 252
                self.current_metrics.volatility = np.std(self.daily_returns) * np.sqrt(252)
            
            # Calculate Sharpe ratio
            if self.current_metrics.volatility > 0:
                excess_return = self.current_metrics.annualized_return - self.risk_free_rate
                self.current_metrics.sharpe_ratio = excess_return / self.current_metrics.volatility
            
            # Calculate drawdown
            if len(self.portfolio_values) > 1:
                values = [pv['value'] for pv in self.portfolio_values]
                peak = np.maximum.accumulate(values)
                drawdown = (np.array(values) - peak) / peak
                self.current_metrics.max_drawdown = abs(np.min(drawdown))
            
            # Calculate trade statistics
            if self.trade_history:
                winning_trades = [t for t in self.trade_history if t.is_winning]
                losing_trades = [t for t in self.trade_history if not t.is_winning]
                
                self.current_metrics.total_trades = len(self.trade_history)
                self.current_metrics.winning_trades = len(winning_trades)
                self.current_metrics.losing_trades = len(losing_trades)
                
                if self.current_metrics.total_trades > 0:
                    self.current_metrics.win_rate = self.current_metrics.winning_trades / self.current_metrics.total_trades
                
                # Calculate profit factor
                gross_profit = sum(t.pnl for t in winning_trades)
                gross_loss = abs(sum(t.pnl for t in losing_trades))
                
                if gross_loss > 0:
                    self.current_metrics.profit_factor = gross_profit / gross_loss
            
            # Calculate VaR (Value at Risk)
            if len(self.daily_returns) >= 20:  # Need sufficient data
                returns_array = np.array(self.daily_returns)
                self.current_metrics.var_95 = np.percentile(returns_array, 5) * current_value
                self.current_metrics.var_99 = np.percentile(returns_array, 1) * current_value
            
        except Exception as e:
            self.logger.error(f"Error updating metrics: {e}")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary.
        
        Returns:
            Dictionary containing performance metrics and statistics
        """
        try:
            summary = {
                'current_metrics': self.current_metrics,
                'total_trades': len(self.trade_history),
                'portfolio_snapshots': len(self.portfolio_values),
                'tracking_period_days': len(self.daily_returns),
                'last_updated': self.current_metrics.last_updated.isoformat(),
                'initial_capital': self.initial_capital
            }
            
            # Add recent performance
            if len(self.daily_returns) >= 7:
                recent_returns = list(self.daily_returns)[-7:]
                summary['weekly_return'] = sum(recent_returns)
                summary['weekly_volatility'] = np.std(recent_returns) * np.sqrt(7)
            
            if len(self.daily_returns) >= 30:
                recent_returns = list(self.daily_returns)[-30:]
                summary['monthly_return'] = sum(recent_returns)
                summary['monthly_volatility'] = np.std(recent_returns) * np.sqrt(30)
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error generating performance summary: {e}")
            return {'error': str(e)}
    
    def get_trade_statistics(self) -> Dict[str, Any]:
        """Get detailed trade statistics.
        
        Returns:
            Dictionary containing trade analysis
        """
        try:
            if not self.trade_history:
                return {'message': 'No trades recorded yet'}
            
            winning_trades = [t for t in self.trade_history if t.is_winning]
            losing_trades = [t for t in self.trade_history if not t.is_winning]
            
            stats = {
                'total_trades': len(self.trade_history),
                'winning_trades': len(winning_trades),
                'losing_trades': len(losing_trades),
                'win_rate': len(winning_trades) / len(self.trade_history) if self.trade_history else 0,
                'average_win': np.mean([t.pnl for t in winning_trades]) if winning_trades else 0,
                'average_loss': np.mean([t.pnl for t in losing_trades]) if losing_trades else 0,
                'largest_win': max([t.pnl for t in winning_trades]) if winning_trades else 0,
                'largest_loss': min([t.pnl for t in losing_trades]) if losing_trades else 0,
                'average_trade_duration': np.mean([t.duration_minutes for t in self.trade_history]),
                'total_pnl': sum([t.pnl for t in self.trade_history])
            }
            
            # Strategy breakdown
            strategy_stats = defaultdict(lambda: {'trades': 0, 'pnl': 0.0, 'wins': 0})
            for trade in self.trade_history:
                strategy_stats[trade.strategy]['trades'] += 1
                strategy_stats[trade.strategy]['pnl'] += trade.pnl
                if trade.is_winning:
                    strategy_stats[trade.strategy]['wins'] += 1
            
            stats['by_strategy'] = dict(strategy_stats)
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Error generating trade statistics: {e}")
            return {'error': str(e)}
    
    def reset_metrics(self) -> None:
        """Reset all performance metrics and history."""
        try:
            self.daily_returns.clear()
            self.trade_history.clear()
            self.portfolio_values.clear()
            self.current_metrics = PerformanceMetrics()
            self.metrics_by_strategy.clear()
            self.metrics_by_asset.clear()
            
            self.logger.info("Performance metrics reset")
            
        except Exception as e:
            self.logger.error(f"Error resetting metrics: {e}")
    
    def export_data(self, filepath: str) -> bool:
        """Export performance data to file.
        
        Args:
            filepath: Path to save the data
            
        Returns:
            True if successful, False otherwise
        """
        try:
            data = {
                'performance_summary': self.get_performance_summary(),
                'trade_statistics': self.get_trade_statistics(),
                'trade_history': [{
                    'trade_id': t.trade_id,
                    'symbol': t.symbol,
                    'entry_time': t.entry_time.isoformat(),
                    'exit_time': t.exit_time.isoformat() if t.exit_time else None,
                    'entry_price': t.entry_price,
                    'exit_price': t.exit_price,
                    'quantity': t.quantity,
                    'pnl': t.pnl,
                    'pnl_percent': t.pnl_percent,
                    'duration_minutes': t.duration_minutes,
                    'trade_type': t.trade_type,
                    'strategy': t.strategy,
                    'is_winning': t.is_winning
                } for t in self.trade_history],
                'portfolio_history': [{
                    'timestamp': pv['timestamp'].isoformat(),
                    'value': pv['value'],
                    'cash': pv['cash']
                } for pv in self.portfolio_values]
            }
            
            import json
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            
            self.logger.info(f"Performance data exported to {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error exporting data: {e}")
            return False