#!/usr/bin/env python3
"""
24/7 Trading Orchestrator for Multi-Asset Trading System

Minimal working implementation to enable core system functionality.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import json

from .config_manager import ConfigManager
from .data_manager import UnifiedDataManager


@dataclass
class Position:
    """Trading position data structure"""
    symbol: str
    quantity: float
    entry_price: float
    current_price: float
    market_value: float
    unrealized_pnl: float
    side: str  # 'long' or 'short'
    timestamp: datetime


@dataclass
class Trade:
    """Trade execution data structure"""
    id: str
    symbol: str
    side: str
    quantity: float
    price: float
    timestamp: datetime
    status: str
    commission: float = 0.0


class TradingOrchestrator24_7:
    """
    Minimal 24/7 Trading Orchestrator
    
    Provides basic functionality to support API endpoints while maintaining
    the interface expected by the application.
    """
    
    def __init__(self, config_manager: ConfigManager, data_manager: UnifiedDataManager):
        """Initialize the trading orchestrator"""
        self.config_manager = config_manager
        self.data_manager = data_manager
        self.logger = logging.getLogger(__name__)
        
        # System state
        self.is_running = False
        self.start_time: Optional[datetime] = None
        
        # Trading data
        self.positions: List[Position] = []
        self.trades: List[Trade] = []
        self.portfolio_value = 100000.0  # Starting portfolio value
        self.cash_balance = 100000.0
        
        # Performance metrics
        self.total_return = 0.0
        self.daily_return = 0.0
        self.max_drawdown = 0.0
        self.win_rate = 0.0
        
        self.logger.info("TradingOrchestrator24_7 initialized")
    
    async def start(self):
        """Start the trading orchestrator"""
        try:
            self.is_running = True
            self.start_time = datetime.now()
            self.logger.info("Trading orchestrator started successfully")
            
            # Initialize with some sample data for demonstration
            await self._initialize_sample_data()
            
        except Exception as e:
            self.logger.error(f"Error starting trading orchestrator: {e}")
            raise
    
    async def stop(self):
        """Stop the trading orchestrator"""
        try:
            self.is_running = False
            self.logger.info("Trading orchestrator stopped")
        except Exception as e:
            self.logger.error(f"Error stopping trading orchestrator: {e}")
    
    async def _initialize_sample_data(self):
        """Initialize with sample data for testing"""
        # Add some sample positions
        sample_positions = [
            Position(
                symbol="AAPL",
                quantity=10,
                entry_price=150.0,
                current_price=155.0,
                market_value=1550.0,
                unrealized_pnl=50.0,
                side="long",
                timestamp=datetime.now()
            ),
            Position(
                symbol="TSLA",
                quantity=5,
                entry_price=200.0,
                current_price=195.0,
                market_value=975.0,
                unrealized_pnl=-25.0,
                side="long",
                timestamp=datetime.now()
            )
        ]
        
        self.positions = sample_positions
        
        # Add some sample trades
        sample_trades = [
            Trade(
                id="trade_001",
                symbol="AAPL",
                side="buy",
                quantity=10,
                price=150.0,
                timestamp=datetime.now() - timedelta(hours=2),
                status="filled"
            ),
            Trade(
                id="trade_002",
                symbol="TSLA",
                side="buy",
                quantity=5,
                price=200.0,
                timestamp=datetime.now() - timedelta(hours=1),
                status="filled"
            )
        ]
        
        self.trades = sample_trades
        
        # Update portfolio metrics
        self._update_portfolio_metrics()
    
    def _update_portfolio_metrics(self):
        """Update portfolio performance metrics"""
        total_market_value = sum(pos.market_value for pos in self.positions)
        total_unrealized_pnl = sum(pos.unrealized_pnl for pos in self.positions)
        
        self.portfolio_value = self.cash_balance + total_market_value
        self.total_return = (total_unrealized_pnl / (self.portfolio_value - total_unrealized_pnl)) * 100
        self.daily_return = self.total_return  # Simplified for demo
        
        # Calculate win rate
        if self.trades:
            profitable_trades = sum(1 for trade in self.trades if trade.side == "sell" and trade.price > 0)
            self.win_rate = (profitable_trades / len(self.trades)) * 100 if self.trades else 0
    
    def get_positions(self) -> List[Dict[str, Any]]:
        """Get current positions"""
        return [
            {
                "symbol": pos.symbol,
                "quantity": pos.quantity,
                "entry_price": pos.entry_price,
                "current_price": pos.current_price,
                "market_value": pos.market_value,
                "unrealized_pnl": pos.unrealized_pnl,
                "side": pos.side,
                "timestamp": pos.timestamp.isoformat()
            }
            for pos in self.positions
        ]
    
    async def get_positions_async(self) -> Dict[str, Any]:
        """Get current positions (async version for API compatibility)"""
        positions_data = self.get_positions()
        total_value = sum(pos["market_value"] for pos in positions_data)
        
        return {
            "positions": positions_data,
            "total_value": total_value,
            "timestamp": datetime.now().isoformat()
        }
    
    def get_recent_trades(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent trades"""
        recent_trades = sorted(self.trades, key=lambda x: x.timestamp, reverse=True)[:limit]
        return [
            {
                "id": trade.id,
                "symbol": trade.symbol,
                "side": trade.side,
                "quantity": trade.quantity,
                "price": trade.price,
                "timestamp": trade.timestamp.isoformat(),
                "status": trade.status,
                "commission": trade.commission
            }
            for trade in recent_trades
        ]
    
    async def get_recent_trades_async(self, limit: int = 50) -> Dict[str, Any]:
        """Get recent trades (async version for API compatibility)"""
        trades_data = self.get_recent_trades(limit)
        
        return {
            "trades": trades_data,
            "total_count": len(self.trades),
            "timestamp": datetime.now().isoformat()
        }
    
    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get portfolio summary"""
        self._update_portfolio_metrics()
        
        total_positions_value = sum(pos.market_value for pos in self.positions)
        day_change = sum(pos.unrealized_pnl for pos in self.positions)
        day_change_percent = (day_change / (self.portfolio_value - day_change)) * 100 if self.portfolio_value > day_change else 0
        
        return {
            "total_value": self.portfolio_value,
            "cash_balance": self.cash_balance,
            "positions_value": total_positions_value,
            "day_change": day_change,
            "day_change_percent": day_change_percent,
            "timestamp": datetime.now().isoformat()
        }
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        self._update_portfolio_metrics()
        
        return {
            "total_return": self.total_return,
            "daily_return": self.daily_return,
            "sharpe_ratio": 1.2,  # Mock value for demo
            "max_drawdown": self.max_drawdown,
            "win_rate": self.win_rate,
            "timestamp": datetime.now().isoformat()
        }
    
    async def get_performance_metrics_async(self) -> Dict[str, Any]:
        """Get performance metrics (async version for API compatibility)"""
        return self.get_performance_metrics()
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get system status"""
        uptime = datetime.now() - self.start_time if self.start_time else timedelta(0)
        
        return {
            "trading_active": self.is_running,
            "uptime_seconds": uptime.total_seconds(),
            "positions_count": len(self.positions),
            "trades_count": len(self.trades),
            "last_update": datetime.now().isoformat()
        }
    
    async def get_portfolio_overview(self) -> Dict[str, Any]:
        """Get portfolio overview (async version for API compatibility)"""
        return self.get_portfolio_summary()
    
    def get_status(self) -> Dict[str, Any]:
        """Get orchestrator status (for compatibility with app.py)"""
        return {
            "status": "active" if self.is_running else "inactive",
            "active_sessions": {
                "main_trading": {
                    "status": "active" if self.is_running else "inactive",
                    "positions": len(self.positions),
                    "last_activity": datetime.now().isoformat()
                }
            },
            "timestamp": datetime.now().isoformat()
        }