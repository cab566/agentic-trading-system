#!/usr/bin/env python3
"""
Alpaca MCP Server Integration Tool for Trading System

This tool integrates the Alpaca MCP (Model Context Protocol) Server for:
- Enhanced trade execution with natural language interface
- Real-time market data streaming
- Portfolio management through conversational AI
- Risk management with natural language rules
- Advanced order types and execution strategies
- Paper trading and live trading modes
- Compliance and regulatory reporting
- Multi-account management

Key Features:
- Natural language trading commands
- Real-time market data integration
- Advanced order management
- Portfolio analytics and reporting
- Risk monitoring and alerts
- Compliance checking
- Multi-timeframe analysis
- Automated strategy execution

MCP Server Components:
- Trading Command Parser
- Order Management System
- Market Data Streaming
- Portfolio Analytics
- Risk Management Engine
- Compliance Monitor
- Natural Language Interface
- Execution Analytics
"""

import asyncio
import logging
import json
import yaml
import websockets
import aiohttp
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
from dataclasses import dataclass, asdict
from pathlib import Path
from abc import ABC, abstractmethod
import re
from enum import Enum

# CrewAI imports
from crewai_tools import BaseTool
from pydantic import BaseModel, Field

# Alpaca imports (would be actual Alpaca API)
try:
    # Mock Alpaca imports - in reality these would be:
    # from alpaca_trade_api import REST, Stream
    # from alpaca_trade_api.entity import Order, Position, Account
    # from alpaca_trade_api.common import URL
    
    # Mock classes for demonstration
    class REST:
        def __init__(self, *args, **kwargs):
            pass
        
        def get_account(self):
            return {"equity": 100000, "buying_power": 50000}
        
        def list_positions(self):
            return []
        
        def submit_order(self, *args, **kwargs):
            return {"id": "mock_order_123", "status": "filled"}
        
        def get_orders(self, *args, **kwargs):
            return []
    
    class Stream:
        def __init__(self, *args, **kwargs):
            pass
        
        async def subscribe_trades(self, *args, **kwargs):
            pass
        
        async def subscribe_quotes(self, *args, **kwargs):
            pass
    
    ALPACA_AVAILABLE = True
    
except ImportError:
    ALPACA_AVAILABLE = False
    REST = None
    Stream = None

# Local imports
from ..core.data_manager import UnifiedDataManager
from ..utils.performance_metrics import PerformanceAnalyzer


class AlpacaMCPInput(BaseModel):
    """Input model for Alpaca MCP integration"""
    
    command_type: str = Field(
        description="Type of MCP command: 'natural_language', 'order_management', 'portfolio_analysis', 'risk_monitoring', 'market_data', 'strategy_execution'"
    )
    
    natural_language_command: Optional[str] = Field(
        description="Natural language trading command (e.g., 'Buy 100 shares of AAPL when price drops below $150')",
        default=None
    )
    
    symbols: List[str] = Field(
        description="List of trading symbols",
        default=["SPY", "QQQ", "IWM"]
    )
    
    account_type: str = Field(
        description="Account type: 'paper' or 'live'",
        default="paper"
    )
    
    order_parameters: Dict[str, Any] = Field(
        description="Order parameters for direct order management",
        default={}
    )
    
    portfolio_filters: Dict[str, Any] = Field(
        description="Filters for portfolio analysis",
        default={}
    )
    
    risk_parameters: Dict[str, Any] = Field(
        description="Risk monitoring parameters",
        default={
            "max_position_size": 0.1,
            "max_daily_loss": 0.02,
            "max_drawdown": 0.05,
            "var_limit": 0.03
        }
    )
    
    streaming_config: Dict[str, Any] = Field(
        description="Market data streaming configuration",
        default={
            "data_types": ["trades", "quotes", "bars"],
            "timeframe": "1Min",
            "extended_hours": False
        }
    )
    
    strategy_config: Dict[str, Any] = Field(
        description="Strategy execution configuration",
        default={}
    )
    
    compliance_rules: List[str] = Field(
        description="Compliance rules to enforce",
        default=["pattern_day_trader", "position_limits", "sector_limits"]
    )


class OrderType(Enum):
    """Order types supported by Alpaca"""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    TRAILING_STOP = "trailing_stop"


class OrderSide(Enum):
    """Order sides"""
    BUY = "buy"
    SELL = "sell"


class TimeInForce(Enum):
    """Time in force options"""
    DAY = "day"
    GTC = "gtc"
    IOC = "ioc"
    FOK = "fok"


@dataclass
class TradingCommand:
    """Parsed trading command"""
    action: str  # buy, sell, hold, close
    symbol: str
    quantity: Optional[int] = None
    price: Optional[float] = None
    order_type: OrderType = OrderType.MARKET
    time_in_force: TimeInForce = TimeInForce.DAY
    conditions: List[str] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    confidence: float = 0.8


@dataclass
class MarketDataUpdate:
    """Market data update"""
    symbol: str
    timestamp: datetime
    price: float
    volume: int
    bid: Optional[float] = None
    ask: Optional[float] = None
    data_type: str = "trade"


@dataclass
class PortfolioSnapshot:
    """Portfolio snapshot"""
    timestamp: datetime
    total_equity: float
    buying_power: float
    day_trade_count: int
    positions: List[Dict[str, Any]]
    orders: List[Dict[str, Any]]
    performance_metrics: Dict[str, float]


@dataclass
class RiskAlert:
    """Risk monitoring alert"""
    alert_type: str
    severity: str  # low, medium, high, critical
    message: str
    symbol: Optional[str] = None
    current_value: Optional[float] = None
    threshold: Optional[float] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class ExecutionReport:
    """Trade execution report"""
    order_id: str
    symbol: str
    side: str
    quantity: int
    filled_quantity: int
    average_price: float
    status: str
    timestamp: datetime
    execution_quality: Dict[str, float]


class NaturalLanguageParser:
    """Parse natural language trading commands"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Command patterns
        self.patterns = {
            'buy': [
                r'buy (\d+) shares? of ([A-Z]{1,5})',
                r'purchase (\d+) ([A-Z]{1,5})',
                r'go long (\d+) ([A-Z]{1,5})',
                r'buy ([A-Z]{1,5}) (\d+) shares?'
            ],
            'sell': [
                r'sell (\d+) shares? of ([A-Z]{1,5})',
                r'short (\d+) ([A-Z]{1,5})',
                r'go short (\d+) ([A-Z]{1,5})',
                r'sell ([A-Z]{1,5}) (\d+) shares?'
            ],
            'close': [
                r'close position in ([A-Z]{1,5})',
                r'close ([A-Z]{1,5}) position',
                r'exit ([A-Z]{1,5})'
            ],
            'conditions': [
                r'when price (?:drops|falls) below \$?(\d+\.?\d*)',
                r'when price (?:rises|goes) above \$?(\d+\.?\d*)',
                r'if ([A-Z]{1,5}) (?:drops|falls) below \$?(\d+\.?\d*)',
                r'if ([A-Z]{1,5}) (?:rises|goes) above \$?(\d+\.?\d*)'
            ],
            'stop_loss': [
                r'with stop loss at \$?(\d+\.?\d*)',
                r'stop loss \$?(\d+\.?\d*)',
                r'sl \$?(\d+\.?\d*)'
            ],
            'take_profit': [
                r'with take profit at \$?(\d+\.?\d*)',
                r'take profit \$?(\d+\.?\d*)',
                r'tp \$?(\d+\.?\d*)'
            ]
        }
    
    def parse_command(self, command: str) -> TradingCommand:
        """Parse natural language command into structured format"""
        
        command = command.lower().strip()
        
        # Initialize command
        trading_command = TradingCommand(
            action="hold",
            symbol="",
            conditions=[]
        )
        
        # Parse action and basic parameters
        for action, patterns in self.patterns.items():
            if action in ['buy', 'sell', 'close']:
                for pattern in patterns:
                    match = re.search(pattern, command)
                    if match:
                        trading_command.action = action
                        
                        if action in ['buy', 'sell']:
                            if len(match.groups()) == 2:
                                if match.group(1).isdigit():
                                    trading_command.quantity = int(match.group(1))
                                    trading_command.symbol = match.group(2).upper()
                                else:
                                    trading_command.symbol = match.group(1).upper()
                                    trading_command.quantity = int(match.group(2))
                        elif action == 'close':
                            trading_command.symbol = match.group(1).upper()
                        
                        break
                
                if trading_command.action != "hold":
                    break
        
        # Parse conditions
        for pattern in self.patterns['conditions']:
            match = re.search(pattern, command)
            if match:
                if len(match.groups()) == 1:
                    price = float(match.group(1))
                    if 'below' in match.group(0):
                        trading_command.conditions.append(f"price_below_{price}")
                    elif 'above' in match.group(0):
                        trading_command.conditions.append(f"price_above_{price}")
                elif len(match.groups()) == 2:
                    symbol = match.group(1).upper()
                    price = float(match.group(2))
                    if trading_command.symbol == "":
                        trading_command.symbol = symbol
                    if 'below' in match.group(0):
                        trading_command.conditions.append(f"price_below_{price}")
                    elif 'above' in match.group(0):
                        trading_command.conditions.append(f"price_above_{price}")
        
        # Parse stop loss
        for pattern in self.patterns['stop_loss']:
            match = re.search(pattern, command)
            if match:
                trading_command.stop_loss = float(match.group(1))
                break
        
        # Parse take profit
        for pattern in self.patterns['take_profit']:
            match = re.search(pattern, command)
            if match:
                trading_command.take_profit = float(match.group(1))
                break
        
        # Determine order type based on conditions
        if trading_command.conditions:
            trading_command.order_type = OrderType.LIMIT
        else:
            trading_command.order_type = OrderType.MARKET
        
        # Calculate confidence based on parsing success
        confidence_factors = []
        if trading_command.action != "hold":
            confidence_factors.append(0.3)
        if trading_command.symbol:
            confidence_factors.append(0.3)
        if trading_command.quantity:
            confidence_factors.append(0.2)
        if trading_command.conditions:
            confidence_factors.append(0.2)
        
        trading_command.confidence = sum(confidence_factors)
        
        return trading_command


class AlpacaOrderManager:
    """Manage orders through Alpaca API"""
    
    def __init__(self, api_client: REST, account_type: str = "paper"):
        self.api_client = api_client
        self.account_type = account_type
        self.logger = logging.getLogger(__name__)
        
        # Order tracking
        self.active_orders: Dict[str, Dict[str, Any]] = {}
        self.order_history: List[Dict[str, Any]] = []
        
        # Execution tracking
        self.execution_reports: List[ExecutionReport] = []
    
    async def submit_order(self, command: TradingCommand) -> Dict[str, Any]:
        """Submit order based on trading command"""
        
        try:
            # Validate command
            if not self._validate_command(command):
                return {
                    "success": False,
                    "error": "Invalid trading command",
                    "command": asdict(command)
                }
            
            # Prepare order parameters
            order_params = self._prepare_order_params(command)
            
            # Submit order to Alpaca
            order_response = self.api_client.submit_order(**order_params)
            
            # Track order
            order_id = order_response.get("id", "unknown")
            self.active_orders[order_id] = {
                "command": command,
                "order_params": order_params,
                "response": order_response,
                "timestamp": datetime.now()
            }
            
            # Create execution report
            execution_report = ExecutionReport(
                order_id=order_id,
                symbol=command.symbol,
                side=command.action,
                quantity=command.quantity or 0,
                filled_quantity=0,  # Will be updated when filled
                average_price=0.0,
                status=order_response.get("status", "unknown"),
                timestamp=datetime.now(),
                execution_quality=self._calculate_execution_quality(order_response)
            )
            
            self.execution_reports.append(execution_report)
            
            return {
                "success": True,
                "order_id": order_id,
                "status": order_response.get("status"),
                "message": f"Order submitted successfully for {command.symbol}",
                "execution_report": asdict(execution_report)
            }
            
        except Exception as e:
            self.logger.error(f"Order submission failed: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "command": asdict(command)
            }
    
    def _validate_command(self, command: TradingCommand) -> bool:
        """Validate trading command"""
        
        if not command.symbol:
            return False
        
        if command.action in ['buy', 'sell'] and not command.quantity:
            return False
        
        if command.confidence < 0.5:
            return False
        
        return True
    
    def _prepare_order_params(self, command: TradingCommand) -> Dict[str, Any]:
        """Prepare order parameters for Alpaca API"""
        
        params = {
            "symbol": command.symbol,
            "qty": command.quantity,
            "side": command.action,
            "type": command.order_type.value,
            "time_in_force": command.time_in_force.value
        }
        
        # Add price for limit orders
        if command.order_type in [OrderType.LIMIT, OrderType.STOP_LIMIT]:
            if command.price:
                params["limit_price"] = command.price
        
        # Add stop price for stop orders
        if command.order_type in [OrderType.STOP, OrderType.STOP_LIMIT]:
            if command.stop_loss:
                params["stop_price"] = command.stop_loss
        
        # Add bracket order parameters
        if command.stop_loss or command.take_profit:
            params["order_class"] = "bracket"
            if command.stop_loss:
                params["stop_loss"] = {"stop_price": command.stop_loss}
            if command.take_profit:
                params["take_profit"] = {"limit_price": command.take_profit}
        
        return params
    
    def _calculate_execution_quality(self, order_response: Dict[str, Any]) -> Dict[str, float]:
        """Calculate execution quality metrics"""
        
        return {
            "fill_rate": 1.0,  # Mock - would calculate actual fill rate
            "price_improvement": 0.0,  # Mock - would calculate price improvement
            "market_impact": 0.001,  # Mock - would estimate market impact
            "timing_score": 0.8  # Mock - would score execution timing
        }
    
    async def get_order_status(self, order_id: str) -> Dict[str, Any]:
        """Get order status"""
        
        try:
            orders = self.api_client.get_orders(status="all")
            
            for order in orders:
                if order.get("id") == order_id:
                    return {
                        "success": True,
                        "order_id": order_id,
                        "status": order.get("status"),
                        "filled_qty": order.get("filled_qty", 0),
                        "filled_avg_price": order.get("filled_avg_price", 0.0)
                    }
            
            return {
                "success": False,
                "error": f"Order {order_id} not found"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    async def cancel_order(self, order_id: str) -> Dict[str, Any]:
        """Cancel order"""
        
        try:
            # Mock cancellation - would use actual Alpaca API
            if order_id in self.active_orders:
                del self.active_orders[order_id]
            
            return {
                "success": True,
                "message": f"Order {order_id} cancelled successfully"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }


class AlpacaPortfolioManager:
    """Manage portfolio through Alpaca API"""
    
    def __init__(self, api_client: REST, data_manager: UnifiedDataManager):
        self.api_client = api_client
        self.data_manager = data_manager
        self.logger = logging.getLogger(__name__)
        
        # Portfolio tracking
        self.portfolio_history: List[PortfolioSnapshot] = []
        self.performance_analyzer = PerformanceAnalyzer()
    
    async def get_portfolio_snapshot(self) -> PortfolioSnapshot:
        """Get current portfolio snapshot"""
        
        try:
            # Get account information
            account = self.api_client.get_account()
            
            # Get positions
            positions = self.api_client.list_positions()
            
            # Get orders
            orders = self.api_client.get_orders(status="open")
            
            # Calculate performance metrics
            performance_metrics = await self._calculate_performance_metrics()
            
            snapshot = PortfolioSnapshot(
                timestamp=datetime.now(),
                total_equity=float(account.get("equity", 0)),
                buying_power=float(account.get("buying_power", 0)),
                day_trade_count=int(account.get("daytrade_count", 0)),
                positions=[self._format_position(pos) for pos in positions],
                orders=[self._format_order(order) for order in orders],
                performance_metrics=performance_metrics
            )
            
            self.portfolio_history.append(snapshot)
            
            return snapshot
            
        except Exception as e:
            self.logger.error(f"Failed to get portfolio snapshot: {str(e)}")
            raise
    
    def _format_position(self, position: Dict[str, Any]) -> Dict[str, Any]:
        """Format position data"""
        
        return {
            "symbol": position.get("symbol", ""),
            "quantity": float(position.get("qty", 0)),
            "market_value": float(position.get("market_value", 0)),
            "cost_basis": float(position.get("cost_basis", 0)),
            "unrealized_pl": float(position.get("unrealized_pl", 0)),
            "unrealized_plpc": float(position.get("unrealized_plpc", 0)),
            "current_price": float(position.get("current_price", 0)),
            "side": position.get("side", "long")
        }
    
    def _format_order(self, order: Dict[str, Any]) -> Dict[str, Any]:
        """Format order data"""
        
        return {
            "id": order.get("id", ""),
            "symbol": order.get("symbol", ""),
            "qty": float(order.get("qty", 0)),
            "side": order.get("side", ""),
            "order_type": order.get("order_type", ""),
            "status": order.get("status", ""),
            "limit_price": float(order.get("limit_price", 0)) if order.get("limit_price") else None,
            "stop_price": float(order.get("stop_price", 0)) if order.get("stop_price") else None,
            "created_at": order.get("created_at", "")
        }
    
    async def _calculate_performance_metrics(self) -> Dict[str, float]:
        """Calculate portfolio performance metrics"""
        
        if len(self.portfolio_history) < 2:
            return {
                "daily_return": 0.0,
                "total_return": 0.0,
                "volatility": 0.0,
                "sharpe_ratio": 0.0,
                "max_drawdown": 0.0,
                "win_rate": 0.0
            }
        
        # Calculate returns
        equity_values = [snapshot.total_equity for snapshot in self.portfolio_history[-30:]]  # Last 30 snapshots
        returns = np.diff(equity_values) / equity_values[:-1]
        
        daily_return = returns[-1] if len(returns) > 0 else 0.0
        total_return = (equity_values[-1] - equity_values[0]) / equity_values[0] if len(equity_values) > 1 else 0.0
        volatility = np.std(returns) * np.sqrt(252) if len(returns) > 1 else 0.0
        
        # Sharpe ratio (assuming 2% risk-free rate)
        excess_returns = returns - (0.02 / 252)
        sharpe_ratio = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252) if len(returns) > 1 and np.std(excess_returns) > 0 else 0.0
        
        # Max drawdown
        peak = np.maximum.accumulate(equity_values)
        drawdown = (peak - equity_values) / peak
        max_drawdown = np.max(drawdown) if len(drawdown) > 0 else 0.0
        
        # Win rate (simplified)
        positive_returns = np.sum(returns > 0)
        win_rate = positive_returns / len(returns) if len(returns) > 0 else 0.0
        
        return {
            "daily_return": float(daily_return),
            "total_return": float(total_return),
            "volatility": float(volatility),
            "sharpe_ratio": float(sharpe_ratio),
            "max_drawdown": float(max_drawdown),
            "win_rate": float(win_rate)
        }


class AlpacaRiskMonitor:
    """Monitor portfolio risk through Alpaca integration"""
    
    def __init__(self, portfolio_manager: AlpacaPortfolioManager, risk_parameters: Dict[str, Any]):
        self.portfolio_manager = portfolio_manager
        self.risk_parameters = risk_parameters
        self.logger = logging.getLogger(__name__)
        
        # Risk tracking
        self.risk_alerts: List[RiskAlert] = []
        self.risk_metrics_history: List[Dict[str, float]] = []
    
    async def monitor_risk(self) -> List[RiskAlert]:
        """Monitor portfolio risk and generate alerts"""
        
        alerts = []
        
        try:
            # Get current portfolio snapshot
            snapshot = await self.portfolio_manager.get_portfolio_snapshot()
            
            # Check position size limits
            alerts.extend(self._check_position_limits(snapshot))
            
            # Check daily loss limits
            alerts.extend(self._check_daily_loss_limits(snapshot))
            
            # Check drawdown limits
            alerts.extend(self._check_drawdown_limits(snapshot))
            
            # Check concentration risk
            alerts.extend(self._check_concentration_risk(snapshot))
            
            # Check volatility limits
            alerts.extend(self._check_volatility_limits(snapshot))
            
            # Store alerts
            self.risk_alerts.extend(alerts)
            
            return alerts
            
        except Exception as e:
            self.logger.error(f"Risk monitoring failed: {str(e)}")
            return [RiskAlert(
                alert_type="system_error",
                severity="high",
                message=f"Risk monitoring system error: {str(e)}"
            )]
    
    def _check_position_limits(self, snapshot: PortfolioSnapshot) -> List[RiskAlert]:
        """Check individual position size limits"""
        
        alerts = []
        max_position_size = self.risk_parameters.get("max_position_size", 0.1)
        
        for position in snapshot.positions:
            position_weight = abs(position["market_value"]) / snapshot.total_equity
            
            if position_weight > max_position_size:
                alerts.append(RiskAlert(
                    alert_type="position_limit_breach",
                    severity="medium",
                    message=f"Position {position['symbol']} exceeds size limit",
                    symbol=position["symbol"],
                    current_value=position_weight,
                    threshold=max_position_size
                ))
        
        return alerts
    
    def _check_daily_loss_limits(self, snapshot: PortfolioSnapshot) -> List[RiskAlert]:
        """Check daily loss limits"""
        
        alerts = []
        max_daily_loss = self.risk_parameters.get("max_daily_loss", 0.02)
        
        daily_return = snapshot.performance_metrics.get("daily_return", 0.0)
        
        if daily_return < -max_daily_loss:
            alerts.append(RiskAlert(
                alert_type="daily_loss_limit",
                severity="high",
                message=f"Daily loss limit breached: {daily_return:.2%}",
                current_value=abs(daily_return),
                threshold=max_daily_loss
            ))
        
        return alerts
    
    def _check_drawdown_limits(self, snapshot: PortfolioSnapshot) -> List[RiskAlert]:
        """Check maximum drawdown limits"""
        
        alerts = []
        max_drawdown = self.risk_parameters.get("max_drawdown", 0.05)
        
        current_drawdown = snapshot.performance_metrics.get("max_drawdown", 0.0)
        
        if current_drawdown > max_drawdown:
            alerts.append(RiskAlert(
                alert_type="drawdown_limit",
                severity="critical",
                message=f"Maximum drawdown limit breached: {current_drawdown:.2%}",
                current_value=current_drawdown,
                threshold=max_drawdown
            ))
        
        return alerts
    
    def _check_concentration_risk(self, snapshot: PortfolioSnapshot) -> List[RiskAlert]:
        """Check portfolio concentration risk"""
        
        alerts = []
        max_concentration = self.risk_parameters.get("max_concentration", 0.3)
        
        # Calculate sector concentration (simplified)
        total_equity = snapshot.total_equity
        position_weights = [abs(pos["market_value"]) / total_equity for pos in snapshot.positions]
        
        if position_weights:
            max_weight = max(position_weights)
            if max_weight > max_concentration:
                alerts.append(RiskAlert(
                    alert_type="concentration_risk",
                    severity="medium",
                    message=f"Portfolio concentration risk: {max_weight:.2%}",
                    current_value=max_weight,
                    threshold=max_concentration
                ))
        
        return alerts
    
    def _check_volatility_limits(self, snapshot: PortfolioSnapshot) -> List[RiskAlert]:
        """Check portfolio volatility limits"""
        
        alerts = []
        max_volatility = self.risk_parameters.get("max_volatility", 0.2)
        
        current_volatility = snapshot.performance_metrics.get("volatility", 0.0)
        
        if current_volatility > max_volatility:
            alerts.append(RiskAlert(
                alert_type="volatility_limit",
                severity="medium",
                message=f"Portfolio volatility exceeds limit: {current_volatility:.2%}",
                current_value=current_volatility,
                threshold=max_volatility
            ))
        
        return alerts


class AlpacaMCPIntegrationTool(BaseTool):
    """
    Alpaca MCP Server Integration Tool for Trading System
    
    Integrates Alpaca's Model Context Protocol server for:
    - Natural language trading interface
    - Enhanced order management
    - Real-time portfolio monitoring
    - Advanced risk management
    - Compliance checking
    - Multi-account support
    """
    
    name: str = "alpaca_mcp_integration"
    description: str = "Integrate Alpaca MCP Server for natural language trading and enhanced execution"
    
    def __init__(self, data_manager: UnifiedDataManager):
        super().__init__()
        self.data_manager = data_manager
        self.logger = logging.getLogger(__name__)
        
        # Configuration
        self.config = self._load_config()
        
        # Components
        self.nl_parser = NaturalLanguageParser()
        self.api_client = None
        self.order_manager = None
        self.portfolio_manager = None
        self.risk_monitor = None
        
        # Initialize if Alpaca is available
        if ALPACA_AVAILABLE:
            self._initialize_alpaca_client()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load Alpaca MCP configuration"""
        
        config_path = Path(__file__).parent.parent / "config" / "alpaca_mcp_config.yaml"
        if config_path.exists():
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        
        # Default configuration
        return {
            "alpaca_config": {
                "base_url": "https://paper-api.alpaca.markets",
                "api_key": "your_api_key",
                "secret_key": "your_secret_key",
                "paper_trading": True
            },
            "risk_parameters": {
                "max_position_size": 0.1,
                "max_daily_loss": 0.02,
                "max_drawdown": 0.05,
                "max_concentration": 0.3,
                "max_volatility": 0.2,
                "var_limit": 0.03
            },
            "compliance_rules": [
                "pattern_day_trader",
                "position_limits",
                "sector_limits",
                "concentration_limits"
            ],
            "execution_parameters": {
                "default_time_in_force": "day",
                "enable_fractional_shares": True,
                "enable_extended_hours": False,
                "default_order_type": "market"
            }
        }
    
    def _initialize_alpaca_client(self):
        """Initialize Alpaca API client"""
        
        try:
            alpaca_config = self.config.get("alpaca_config", {})
            
            # Initialize REST client
            self.api_client = REST(
                key_id=alpaca_config.get("api_key", "mock_key"),
                secret_key=alpaca_config.get("secret_key", "mock_secret"),
                base_url=alpaca_config.get("base_url", "https://paper-api.alpaca.markets")
            )
            
            # Initialize components
            account_type = "paper" if alpaca_config.get("paper_trading", True) else "live"
            self.order_manager = AlpacaOrderManager(self.api_client, account_type)
            self.portfolio_manager = AlpacaPortfolioManager(self.api_client, self.data_manager)
            self.risk_monitor = AlpacaRiskMonitor(
                self.portfolio_manager, 
                self.config.get("risk_parameters", {})
            )
            
            self.logger.info("Alpaca MCP client initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Alpaca client: {str(e)}")
    
    def _run(self, command_type: str, natural_language_command: str = None,
             symbols: List[str] = None, account_type: str = "paper",
             order_parameters: Dict[str, Any] = None, portfolio_filters: Dict[str, Any] = None,
             risk_parameters: Dict[str, Any] = None, streaming_config: Dict[str, Any] = None,
             strategy_config: Dict[str, Any] = None, compliance_rules: List[str] = None) -> str:
        """
        Execute Alpaca MCP integration
        
        Args:
            command_type: Type of MCP command
            natural_language_command: Natural language trading command
            symbols: List of trading symbols
            account_type: Account type (paper/live)
            order_parameters: Order parameters
            portfolio_filters: Portfolio analysis filters
            risk_parameters: Risk monitoring parameters
            streaming_config: Market data streaming config
            strategy_config: Strategy execution config
            compliance_rules: Compliance rules
        
        Returns:
            JSON string with integration results
        """
        
        if not ALPACA_AVAILABLE:
            return json.dumps({
                "success": False,
                "error": "Alpaca API not available. Please install alpaca-trade-api.",
                "timestamp": datetime.now().isoformat()
            })
        
        try:
            # Set defaults
            if symbols is None:
                symbols = ["SPY", "QQQ", "IWM"]
            if order_parameters is None:
                order_parameters = {}
            if portfolio_filters is None:
                portfolio_filters = {}
            if risk_parameters is None:
                risk_parameters = self.config.get("risk_parameters", {})
            if streaming_config is None:
                streaming_config = {"data_types": ["trades"], "timeframe": "1Min"}
            if strategy_config is None:
                strategy_config = {}
            if compliance_rules is None:
                compliance_rules = self.config.get("compliance_rules", [])
            
            # Execute the appropriate command
            if command_type == "natural_language":
                result = asyncio.run(self._process_natural_language_command(
                    natural_language_command, symbols, account_type
                ))
            
            elif command_type == "order_management":
                result = asyncio.run(self._manage_orders(
                    order_parameters, symbols, account_type
                ))
            
            elif command_type == "portfolio_analysis":
                result = asyncio.run(self._analyze_portfolio(
                    portfolio_filters, account_type
                ))
            
            elif command_type == "risk_monitoring":
                result = asyncio.run(self._monitor_risk(
                    risk_parameters, account_type
                ))
            
            elif command_type == "market_data":
                result = asyncio.run(self._stream_market_data(
                    streaming_config, symbols
                ))
            
            elif command_type == "strategy_execution":
                result = asyncio.run(self._execute_strategy(
                    strategy_config, symbols, account_type
                ))
            
            else:
                result = {
                    "success": False,
                    "error": f"Unknown command type: {command_type}"
                }
            
            return json.dumps(result, indent=2, default=str)
            
        except Exception as e:
            self.logger.error(f"Alpaca MCP integration failed: {str(e)}")
            return json.dumps({
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            })
    
    async def _process_natural_language_command(self, command: str, symbols: List[str], 
                                                account_type: str) -> Dict[str, Any]:
        """Process natural language trading command"""
        
        try:
            if not command:
                return {
                    "success": False,
                    "error": "No natural language command provided"
                }
            
            # Parse the command
            trading_command = self.nl_parser.parse_command(command)
            
            # Validate the parsed command
            if trading_command.confidence < 0.5:
                return {
                    "success": False,
                    "error": "Could not parse command with sufficient confidence",
                    "parsed_command": asdict(trading_command),
                    "confidence": trading_command.confidence
                }
            
            # Execute the command if it's actionable
            execution_result = None
            if trading_command.action in ['buy', 'sell', 'close']:
                execution_result = await self.order_manager.submit_order(trading_command)
            
            return {
                "success": True,
                "command_type": "natural_language",
                "original_command": command,
                "parsed_command": asdict(trading_command),
                "confidence": trading_command.confidence,
                "execution_result": execution_result,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Natural language command processing failed: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def _manage_orders(self, order_parameters: Dict[str, Any], symbols: List[str],
                             account_type: str) -> Dict[str, Any]:
        """Manage orders through Alpaca API"""
        
        try:
            # Get current orders
            orders = self.api_client.get_orders(status="open")
            
            # Format orders
            formatted_orders = [self.portfolio_manager._format_order(order) for order in orders]
            
            # Execute any new orders specified in parameters
            execution_results = []
            if "new_orders" in order_parameters:
                for order_spec in order_parameters["new_orders"]:
                    # Create trading command from order spec
                    command = TradingCommand(
                        action=order_spec.get("side", "buy"),
                        symbol=order_spec.get("symbol", ""),
                        quantity=order_spec.get("qty", 0),
                        price=order_spec.get("limit_price"),
                        order_type=OrderType(order_spec.get("type", "market")),
                        time_in_force=TimeInForce(order_spec.get("time_in_force", "day"))
                    )
                    
                    result = await self.order_manager.submit_order(command)
                    execution_results.append(result)
            
            return {
                "success": True,
                "command_type": "order_management",
                "current_orders": formatted_orders,
                "execution_results": execution_results,
                "order_count": len(formatted_orders),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Order management failed: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def _analyze_portfolio(self, portfolio_filters: Dict[str, Any],
                                 account_type: str) -> Dict[str, Any]:
        """Analyze portfolio through Alpaca API"""
        
        try:
            # Get portfolio snapshot
            snapshot = await self.portfolio_manager.get_portfolio_snapshot()
            
            # Apply filters if specified
            filtered_positions = snapshot.positions
            if portfolio_filters:
                if "symbols" in portfolio_filters:
                    filtered_positions = [
                        pos for pos in snapshot.positions 
                        if pos["symbol"] in portfolio_filters["symbols"]
                    ]
                
                if "min_value" in portfolio_filters:
                    min_value = portfolio_filters["min_value"]
                    filtered_positions = [
                        pos for pos in filtered_positions 
                        if abs(pos["market_value"]) >= min_value
                    ]
            
            # Calculate additional analytics
            portfolio_analytics = self._calculate_portfolio_analytics(snapshot)
            
            return {
                "success": True,
                "command_type": "portfolio_analysis",
                "snapshot": asdict(snapshot),
                "filtered_positions": filtered_positions,
                "portfolio_analytics": portfolio_analytics,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Portfolio analysis failed: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def _monitor_risk(self, risk_parameters: Dict[str, Any],
                            account_type: str) -> Dict[str, Any]:
        """Monitor portfolio risk"""
        
        try:
            # Update risk monitor parameters
            self.risk_monitor.risk_parameters.update(risk_parameters)
            
            # Run risk monitoring
            risk_alerts = await self.risk_monitor.monitor_risk()
            
            # Get current portfolio for risk metrics
            snapshot = await self.portfolio_manager.get_portfolio_snapshot()
            
            # Calculate risk metrics
            risk_metrics = self._calculate_risk_metrics(snapshot)
            
            return {
                "success": True,
                "command_type": "risk_monitoring",
                "risk_alerts": [asdict(alert) for alert in risk_alerts],
                "risk_metrics": risk_metrics,
                "alert_count": len(risk_alerts),
                "critical_alerts": len([a for a in risk_alerts if a.severity == "critical"]),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Risk monitoring failed: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def _stream_market_data(self, streaming_config: Dict[str, Any],
                                  symbols: List[str]) -> Dict[str, Any]:
        """Stream market data through Alpaca"""
        
        try:
            # Mock streaming implementation
            # In reality, would set up WebSocket connections
            
            market_data_updates = []
            for symbol in symbols:
                update = MarketDataUpdate(
                    symbol=symbol,
                    timestamp=datetime.now(),
                    price=100.0 + np.random.randn(),  # Mock price
                    volume=1000 + int(np.random.randn() * 100),
                    bid=99.9 + np.random.randn(),
                    ask=100.1 + np.random.randn(),
                    data_type="trade"
                )
                market_data_updates.append(update)
            
            return {
                "success": True,
                "command_type": "market_data",
                "streaming_config": streaming_config,
                "symbols": symbols,
                "data_updates": [asdict(update) for update in market_data_updates],
                "update_count": len(market_data_updates),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Market data streaming failed: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def _execute_strategy(self, strategy_config: Dict[str, Any], symbols: List[str],
                                account_type: str) -> Dict[str, Any]:
        """Execute trading strategy through Alpaca"""
        
        try:
            # Mock strategy execution
            # In reality, would implement actual strategy logic
            
            strategy_name = strategy_config.get("name", "default_strategy")
            execution_results = []
            
            for symbol in symbols:
                # Mock strategy decision
                action = np.random.choice(["buy", "sell", "hold"], p=[0.3, 0.3, 0.4])
                
                if action != "hold":
                    command = TradingCommand(
                        action=action,
                        symbol=symbol,
                        quantity=100,
                        order_type=OrderType.MARKET,
                        confidence=0.8
                    )
                    
                    result = await self.order_manager.submit_order(command)
                    execution_results.append(result)
            
            return {
                "success": True,
                "command_type": "strategy_execution",
                "strategy_name": strategy_name,
                "strategy_config": strategy_config,
                "symbols": symbols,
                "execution_results": execution_results,
                "orders_submitted": len(execution_results),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Strategy execution failed: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def _calculate_portfolio_analytics(self, snapshot: PortfolioSnapshot) -> Dict[str, Any]:
        """Calculate additional portfolio analytics"""
        
        if not snapshot.positions:
            return {
                "position_count": 0,
                "total_market_value": 0.0,
                "largest_position": None,
                "sector_allocation": {},
                "risk_metrics": {}
            }
        
        # Basic analytics
        position_count = len(snapshot.positions)
        total_market_value = sum(abs(pos["market_value"]) for pos in snapshot.positions)
        
        # Largest position
        largest_position = max(snapshot.positions, key=lambda x: abs(x["market_value"]))
        
        # Mock sector allocation
        sector_allocation = {
            "Technology": 0.4,
            "Healthcare": 0.2,
            "Financial": 0.2,
            "Consumer": 0.1,
            "Other": 0.1
        }
        
        return {
            "position_count": position_count,
            "total_market_value": total_market_value,
            "largest_position": largest_position,
            "sector_allocation": sector_allocation,
            "risk_metrics": snapshot.performance_metrics
        }
    
    def _calculate_risk_metrics(self, snapshot: PortfolioSnapshot) -> Dict[str, float]:
        """Calculate comprehensive risk metrics"""
        
        return {
            "portfolio_beta": 1.0,  # Mock
            "value_at_risk_95": 0.02,  # Mock
            "expected_shortfall": 0.03,  # Mock
            "correlation_risk": 0.6,  # Mock
            "concentration_risk": 0.3,  # Mock
            "liquidity_risk": 0.1,  # Mock
            "leverage_ratio": 1.0,  # Mock
            "risk_adjusted_return": snapshot.performance_metrics.get("sharpe_ratio", 0.0)
        }


# Test the tool
if __name__ == "__main__":
    async def test_alpaca_mcp_integration():
        # Mock data manager
        class MockDataManager:
            async def get_market_data(self, symbol, start_date, end_date, interval):
                # Return mock data
                dates = pd.date_range(start_date, end_date, freq='D')
                data = pd.DataFrame({
                    'close': np.random.randn(len(dates)).cumsum() + 100,
                    'volume': np.random.randint(1000, 10000, len(dates))
                }, index=dates)
                return data
        
        # Test the tool
        tool = AlpacaMCPIntegrationTool(MockDataManager())
        
        # Test natural language command
        result = tool._run(
            command_type="natural_language",
            natural_language_command="Buy 100 shares of AAPL when price drops below $150 with stop loss at $140",
            symbols=["AAPL"],
            account_type="paper"
        )
        
        print("Alpaca MCP Integration Test Result:")
        print(result)
    
    asyncio.run(test_alpaca_mcp_integration())