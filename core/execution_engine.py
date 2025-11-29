#!/usr/bin/env python3
"""
Execution Engine for 24/7 Multi-Asset Trading System

Provides comprehensive order management and execution across:
- Traditional markets (stocks, bonds, options)
- Cryptocurrency markets (24/7 trading)
- Forex markets (24/5 trading)
- Smart order routing
- Risk-aware execution
- Multi-venue execution
- Order lifecycle management
- Execution analytics
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

import pandas as pd
import numpy as np
from pydantic import BaseModel, Field

from .config_manager import ConfigManager
from .market_data_aggregator import MarketDataAggregator, DataType
from .risk_manager_24_7 import RiskManager24_7, RiskLevel
from .trade_storage import TradeStorage
from utils.notifications import NotificationManager
from utils.cache_manager import CacheManager


class OrderType(Enum):
    """Order type enumeration."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    TRAILING_STOP = "trailing_stop"
    ICEBERG = "iceberg"
    TWAP = "twap"
    VWAP = "vwap"
    BRACKET = "bracket"
    OCO = "oco"  # One-Cancels-Other


class OrderSide(Enum):
    """Order side enumeration."""
    BUY = "buy"
    SELL = "sell"
    BUY_TO_COVER = "buy_to_cover"
    SELL_SHORT = "sell_short"


class OrderStatus(Enum):
    """Order status enumeration."""
    PENDING = "pending"
    SUBMITTED = "submitted"
    ACCEPTED = "accepted"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"
    REPLACED = "replaced"


class TimeInForce(Enum):
    """Time in force enumeration."""
    DAY = "day"
    GTC = "gtc"  # Good Till Cancelled
    IOC = "ioc"  # Immediate or Cancel
    FOK = "fok"  # Fill or Kill
    GTD = "gtd"  # Good Till Date
    ATO = "ato"  # At The Open
    ATC = "atc"  # At The Close


class VenueType(Enum):
    """Venue type enumeration."""
    STOCK_EXCHANGE = "stock_exchange"
    CRYPTO_EXCHANGE = "crypto_exchange"
    FOREX_BROKER = "forex_broker"
    ECN = "ecn"
    DARK_POOL = "dark_pool"
    MARKET_MAKER = "market_maker"


class ExecutionQuality(Enum):
    """Execution quality enumeration."""
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"


@dataclass
class Order:
    """Order representation."""
    order_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: TimeInForce = TimeInForce.DAY
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    average_fill_price: Optional[float] = None
    venue: Optional[str] = None
    venue_order_id: Optional[str] = None
    parent_order_id: Optional[str] = None
    child_order_ids: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def remaining_quantity(self) -> float:
        """Get remaining quantity to fill."""
        return max(0, self.quantity - self.filled_quantity)
    
    @property
    def is_complete(self) -> bool:
        """Check if order is complete."""
        return self.status in [OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED, OrderStatus.EXPIRED]
    
    @property
    def fill_percentage(self) -> float:
        """Get fill percentage."""
        if self.quantity == 0:
            return 0.0
        return (self.filled_quantity / self.quantity) * 100
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'order_id': self.order_id,
            'symbol': self.symbol,
            'side': self.side.value,
            'order_type': self.order_type.value,
            'quantity': self.quantity,
            'price': self.price,
            'stop_price': self.stop_price,
            'time_in_force': self.time_in_force.value,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'status': self.status.value,
            'filled_quantity': self.filled_quantity,
            'average_fill_price': self.average_fill_price,
            'venue': self.venue,
            'venue_order_id': self.venue_order_id,
            'parent_order_id': self.parent_order_id,
            'child_order_ids': self.child_order_ids,
            'metadata': self.metadata
        }


@dataclass
class Fill:
    """Trade fill representation."""
    fill_id: str
    order_id: str
    symbol: str
    side: OrderSide
    quantity: float
    price: float
    timestamp: datetime
    venue: str
    venue_fill_id: Optional[str] = None
    commission: float = 0.0
    fees: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def gross_amount(self) -> float:
        """Get gross trade amount."""
        return self.quantity * self.price
    
    @property
    def net_amount(self) -> float:
        """Get net trade amount after fees."""
        total_fees = self.commission + sum(self.fees.values())
        return self.gross_amount - total_fees
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'fill_id': self.fill_id,
            'order_id': self.order_id,
            'symbol': self.symbol,
            'side': self.side.value,
            'quantity': self.quantity,
            'price': self.price,
            'timestamp': self.timestamp.isoformat(),
            'venue': self.venue,
            'venue_fill_id': self.venue_fill_id,
            'commission': self.commission,
            'fees': self.fees,
            'gross_amount': self.gross_amount,
            'net_amount': self.net_amount,
            'metadata': self.metadata
        }


@dataclass
class VenueInfo:
    """Venue information."""
    venue_id: str
    venue_name: str
    venue_type: VenueType
    supported_assets: List[str]
    supported_order_types: List[OrderType]
    min_order_size: Dict[str, float]
    max_order_size: Dict[str, float]
    commission_structure: Dict[str, Any]
    is_active: bool = True
    latency_ms: float = 0.0
    success_rate: float = 1.0
    last_update: datetime = field(default_factory=datetime.now)
    
    def supports_symbol(self, symbol: str) -> bool:
        """Check if venue supports a symbol."""
        # Use exact matching for precise venue routing
        return symbol in self.supported_assets
    
    def supports_order_type(self, order_type: OrderType) -> bool:
        """Check if venue supports an order type."""
        return order_type in self.supported_order_types


@dataclass
class ExecutionMetrics:
    """Execution performance metrics."""
    total_orders: int = 0
    filled_orders: int = 0
    cancelled_orders: int = 0
    rejected_orders: int = 0
    average_fill_time_seconds: float = 0.0
    average_slippage_bps: float = 0.0
    total_commission: float = 0.0
    total_volume: float = 0.0
    success_rate: float = 0.0
    venue_performance: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    def update_metrics(self, order: Order, fill: Optional[Fill] = None):
        """Update metrics with new order/fill data."""
        self.total_orders += 1
        
        if order.status == OrderStatus.FILLED:
            self.filled_orders += 1
        elif order.status == OrderStatus.CANCELLED:
            self.cancelled_orders += 1
        elif order.status == OrderStatus.REJECTED:
            self.rejected_orders += 1
        
        if fill:
            self.total_commission += fill.commission
            self.total_volume += fill.gross_amount
        
        # Update success rate
        if self.total_orders > 0:
            self.success_rate = self.filled_orders / self.total_orders


class ExecutionEngine:
    """
    Comprehensive execution engine for 24/7 multi-asset trading.
    
    Features:
    - Smart order routing across multiple venues
    - Risk-aware order management
    - Advanced order types (TWAP, VWAP, Iceberg, etc.)
    - Real-time execution monitoring
    - Execution analytics and reporting
    - Multi-asset class support
    - 24/7 operation capability
    """
    
    def __init__(
        self,
        config_manager: ConfigManager,
        market_data_aggregator: MarketDataAggregator,
        risk_manager: RiskManager24_7,
        trade_storage: Optional[TradeStorage] = None
    ):
        self.config_manager = config_manager
        self.market_data_aggregator = market_data_aggregator
        self.risk_manager = risk_manager
        self.trade_storage = trade_storage or TradeStorage(config_manager)
        self.notification_manager = NotificationManager(config_manager)
        self.cache_manager = CacheManager(config_manager)
        self.logger = logging.getLogger(__name__)
        
        # Configuration
        self.config = self._load_execution_config()
        
        # Order management
        self.orders: Dict[str, Order] = {}
        self.fills: Dict[str, Fill] = {}
        self.order_history: deque = deque(maxlen=10000)
        self.fill_history: deque = deque(maxlen=10000)
        
        # Venue management
        self.venues: Dict[str, VenueInfo] = {}
        self.venue_connections: Dict[str, Any] = {}
        
        # Execution tracking
        self.execution_metrics = ExecutionMetrics()
        self.active_strategies: Dict[str, List[str]] = defaultdict(list)  # strategy_id -> order_ids
        
        # Smart routing
        self.routing_rules: Dict[str, Dict[str, Any]] = {}
        self.venue_rankings: Dict[str, List[str]] = {}  # symbol -> ranked venue list
        
        # Background tasks
        self._background_tasks: List[asyncio.Task] = []
        
        # Event handlers
        self.order_event_handlers: List[Callable] = []
        self.fill_event_handlers: List[Callable] = []
        
        # Initialize components
        self._initialize_venues()
        self._initialize_routing_rules()
        
        # Don't start background tasks automatically - they should be started explicitly
        # when an event loop is available
    
    def _load_execution_config(self) -> Dict[str, Any]:
        """Load execution engine configuration."""
        return {
            'max_order_age_hours': 24,
            'order_timeout_seconds': 300,  # 5 minutes
            'max_slippage_bps': 50,  # 0.5%
            'max_order_size_usd': 1000000,  # $1M
            'min_order_size_usd': 1,  # $1
            'venue_timeout_seconds': 30,
            'retry_attempts': 3,
            'retry_delay_seconds': 1,
            'smart_routing_enabled': True,
            'execution_quality_threshold': 0.95,
            'commission_caps': {
                'stock': 0.005,  # 0.5%
                'crypto': 0.001,  # 0.1%
                'forex': 0.0001  # 0.01%
            },
            'order_size_limits': {
                'crypto': {'min': 10, 'max': 100000},
                'forex': {'min': 1000, 'max': 10000000},
                'stock': {'min': 1, 'max': 1000000}
            },
            'venue_preferences': {
                'crypto': ['binance', 'coinbase', 'kraken'],
                'forex': ['oanda', 'fxcm', 'alpaca'],
                'stock': ['alpaca', 'interactive_brokers', 'td_ameritrade']
            }
        }
    
    def _initialize_venues(self):
        """Initialize venue configurations."""
        # Enhanced venue configurations with proper symbol mapping
        venues_config = {
            'alpaca': {
                'name': 'Alpaca Markets',
                'type': VenueType.STOCK_EXCHANGE,
                'assets': [
                    # US Stocks
                    'AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX',
                    'SPY', 'QQQ', 'IWM', 'VTI', 'VOO',
                    # Crypto (Alpaca supports some crypto)
                    'BTCUSD', 'ETHUSD', 'LTCUSD', 'BCHUSD'
                ],
                'order_types': [OrderType.MARKET, OrderType.LIMIT, OrderType.STOP, OrderType.STOP_LIMIT],
                'min_order_size': {'USD': 1},
                'max_order_size': {'USD': 1000000},
                'commission': {'stock': 0, 'crypto': 0.0025}
            },
            'binance': {
                'name': 'Binance',
                'type': VenueType.CRYPTO_EXCHANGE,
                'assets': [
                    # Major crypto pairs
                    'BTCUSDT', 'ETHUSDT', 'ADAUSDT', 'DOTUSDT', 'LINKUSDT',
                    'BNBUSDT', 'XRPUSDT', 'SOLUSDT', 'MATICUSDT', 'AVAXUSDT',
                    'ATOMUSDT', 'ALGOUSDT', 'VETUSDT', 'FILUSDT', 'TRXUSDT'
                ],
                'order_types': [OrderType.MARKET, OrderType.LIMIT, OrderType.STOP_LIMIT, OrderType.ICEBERG],
                'min_order_size': {'USD': 10},
                'max_order_size': {'USD': 100000},
                'commission': {'maker': 0.001, 'taker': 0.001}
            },
            'oanda': {
                'name': 'OANDA',
                'type': VenueType.FOREX_BROKER,
                'assets': [
                    # Major forex pairs
                    'EUR_USD', 'GBP_USD', 'USD_JPY', 'AUD_USD', 'USD_CHF', 'USD_CAD',
                    'NZD_USD', 'EUR_GBP', 'EUR_JPY', 'GBP_JPY', 'AUD_JPY', 'CHF_JPY',
                    'EUR_AUD', 'GBP_AUD', 'AUD_CAD', 'EUR_CAD', 'GBP_CAD'
                ],
                'order_types': [OrderType.MARKET, OrderType.LIMIT, OrderType.STOP, OrderType.TRAILING_STOP],
                'min_order_size': {'USD': 1000},
                'max_order_size': {'USD': 10000000},
                'commission': {'spread': 0.0001}
            }
        }
        
        for venue_id, config in venues_config.items():
            self.venues[venue_id] = VenueInfo(
                venue_id=venue_id,
                venue_name=config['name'],
                venue_type=config['type'],
                supported_assets=config['assets'],
                supported_order_types=config['order_types'],
                min_order_size=config['min_order_size'],
                max_order_size=config['max_order_size'],
                commission_structure=config['commission']
            )
    
    def _initialize_routing_rules(self):
        """Initialize smart routing rules."""
        self.routing_rules = {
            'default': {
                'criteria': ['latency', 'success_rate', 'commission'],
                'weights': {'latency': 0.3, 'success_rate': 0.4, 'commission': 0.3},
                'min_success_rate': 0.95,
                'max_latency_ms': 100
            },
            'large_order': {
                'criteria': ['liquidity', 'market_impact', 'commission'],
                'weights': {'liquidity': 0.5, 'market_impact': 0.3, 'commission': 0.2},
                'min_liquidity_usd': 100000,
                'max_market_impact_bps': 10
            },
            'urgent': {
                'criteria': ['latency', 'success_rate'],
                'weights': {'latency': 0.6, 'success_rate': 0.4},
                'max_latency_ms': 50,
                'min_success_rate': 0.98
            }
        }
    
    def _start_background_tasks(self):
        """Start background monitoring tasks."""
        try:
            # Check if event loop is running
            loop = asyncio.get_running_loop()
            
            # Order monitoring
            self._background_tasks.append(
                asyncio.create_task(self._monitor_orders())
            )
            
            # Venue health monitoring
            self._background_tasks.append(
                asyncio.create_task(self._monitor_venues())
            )
            
            # Execution analytics
            self._background_tasks.append(
                asyncio.create_task(self._update_execution_analytics())
            )
            
            # Order cleanup
            self._background_tasks.append(
                asyncio.create_task(self._cleanup_old_orders())
            )
            
            self.logger.info("Background execution tasks started")
            
        except RuntimeError:
            # No event loop running - tasks will be started later
            self.logger.info("No event loop running - background tasks will be started when loop is available")
    
    def start_monitoring(self):
        """Start background monitoring tasks if event loop is available."""
        self._start_background_tasks()
    
    async def submit_order(
        self,
        symbol: str,
        side: OrderSide,
        order_type: OrderType,
        quantity: float,
        price: Optional[float] = None,
        stop_price: Optional[float] = None,
        time_in_force: TimeInForce = TimeInForce.DAY,
        strategy_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """Submit a new order."""
        try:
            # Generate order ID
            order_id = str(uuid.uuid4())
            
            # Create order
            order = Order(
                order_id=order_id,
                symbol=symbol,
                side=side,
                order_type=order_type,
                quantity=quantity,
                price=price,
                stop_price=stop_price,
                time_in_force=time_in_force,
                metadata=metadata or {}
            )
            
            # Pre-trade risk check
            risk_check = await self.risk_manager.check_order_risk(order.to_dict())
            if not risk_check['approved']:
                self.logger.warning(f"Order {order_id} rejected by risk manager: {risk_check['reason']}")
                order.status = OrderStatus.REJECTED
                order.metadata['rejection_reason'] = risk_check['reason']
                self.orders[order_id] = order
                await self._notify_order_event(order, 'rejected')
                return None
            
            # Validate order
            validation_result = await self._validate_order(order)
            if not validation_result['valid']:
                self.logger.warning(f"Order {order_id} validation failed: {validation_result['reason']}")
                order.status = OrderStatus.REJECTED
                order.metadata['rejection_reason'] = validation_result['reason']
                self.orders[order_id] = order
                await self._notify_order_event(order, 'rejected')
                return None
            
            # Store order
            self.orders[order_id] = order
            
            # Associate with strategy
            if strategy_id:
                self.active_strategies[strategy_id].append(order_id)
                order.metadata['strategy_id'] = strategy_id
            
            # Route and execute order
            execution_result = await self._route_and_execute_order(order)
            
            if execution_result['success']:
                order.status = OrderStatus.SUBMITTED
                order.venue = execution_result['venue']
                order.venue_order_id = execution_result['venue_order_id']
                order.updated_at = datetime.now()
                
                self.logger.info(f"Order {order_id} submitted to {order.venue}")
                await self._notify_order_event(order, 'submitted')
                
                return order_id
            else:
                order.status = OrderStatus.REJECTED
                order.metadata['rejection_reason'] = execution_result['reason']
                order.updated_at = datetime.now()
                
                self.logger.error(f"Order {order_id} execution failed: {execution_result['reason']}")
                await self._notify_order_event(order, 'rejected')
                
                return None
        
        except Exception as e:
            self.logger.error(f"Error submitting order: {e}")
            return None
    
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an existing order."""
        try:
            if order_id not in self.orders:
                self.logger.warning(f"Order {order_id} not found")
                return False
            
            order = self.orders[order_id]
            
            if order.is_complete:
                self.logger.warning(f"Order {order_id} is already complete")
                return False
            
            # Cancel at venue
            if order.venue and order.venue_order_id:
                cancel_result = await self._cancel_order_at_venue(
                    order.venue, order.venue_order_id
                )
                
                if cancel_result['success']:
                    order.status = OrderStatus.CANCELLED
                    order.updated_at = datetime.now()
                    
                    self.logger.info(f"Order {order_id} cancelled")
                    await self._notify_order_event(order, 'cancelled')
                    
                    return True
                else:
                    self.logger.error(f"Failed to cancel order {order_id}: {cancel_result['reason']}")
                    return False
            else:
                # Order not yet submitted to venue
                order.status = OrderStatus.CANCELLED
                order.updated_at = datetime.now()
                
                self.logger.info(f"Order {order_id} cancelled before submission")
                await self._notify_order_event(order, 'cancelled')
                
                return True
        
        except Exception as e:
            self.logger.error(f"Error cancelling order {order_id}: {e}")
            return False
    
    async def modify_order(
        self,
        order_id: str,
        new_quantity: Optional[float] = None,
        new_price: Optional[float] = None
    ) -> bool:
        """Modify an existing order."""
        try:
            if order_id not in self.orders:
                self.logger.warning(f"Order {order_id} not found")
                return False
            
            order = self.orders[order_id]
            
            if order.is_complete:
                self.logger.warning(f"Order {order_id} is already complete")
                return False
            
            # Modify at venue
            if order.venue and order.venue_order_id:
                modify_result = await self._modify_order_at_venue(
                    order.venue, order.venue_order_id, new_quantity, new_price
                )
                
                if modify_result['success']:
                    # Update order details
                    if new_quantity is not None:
                        order.quantity = new_quantity
                    if new_price is not None:
                        order.price = new_price
                    
                    order.updated_at = datetime.now()
                    order.venue_order_id = modify_result.get('new_venue_order_id', order.venue_order_id)
                    
                    self.logger.info(f"Order {order_id} modified")
                    await self._notify_order_event(order, 'modified')
                    
                    return True
                else:
                    self.logger.error(f"Failed to modify order {order_id}: {modify_result['reason']}")
                    return False
            else:
                # Order not yet submitted, modify locally
                if new_quantity is not None:
                    order.quantity = new_quantity
                if new_price is not None:
                    order.price = new_price
                
                order.updated_at = datetime.now()
                
                self.logger.info(f"Order {order_id} modified before submission")
                await self._notify_order_event(order, 'modified')
                
                return True
        
        except Exception as e:
            self.logger.error(f"Error modifying order {order_id}: {e}")
            return False
    
    async def get_order_status(self, order_id: str) -> Optional[Dict[str, Any]]:
        """Get current order status."""
        try:
            if order_id not in self.orders:
                return None
            
            order = self.orders[order_id]
            
            # Get latest status from venue if order is active
            if not order.is_complete and order.venue and order.venue_order_id:
                venue_status = await self._get_order_status_from_venue(
                    order.venue, order.venue_order_id
                )
                
                if venue_status:
                    # Update order with latest status
                    await self._update_order_from_venue_status(order, venue_status)
            
            return {
                'order': order.to_dict(),
                'fills': [fill.to_dict() for fill in self.fills.values() if fill.order_id == order_id]
            }
        
        except Exception as e:
            self.logger.error(f"Error getting order status for {order_id}: {e}")
            return None
    
    async def get_orders(
        self,
        symbol: Optional[str] = None,
        status: Optional[OrderStatus] = None,
        strategy_id: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Get orders with optional filtering."""
        try:
            orders = list(self.orders.values())
            
            # Apply filters
            if symbol:
                orders = [o for o in orders if o.symbol == symbol]
            
            if status:
                orders = [o for o in orders if o.status == status]
            
            if strategy_id:
                orders = [o for o in orders if o.metadata.get('strategy_id') == strategy_id]
            
            # Sort by creation time (newest first)
            orders.sort(key=lambda x: x.created_at, reverse=True)
            
            # Apply limit
            if limit:
                orders = orders[:limit]
            
            return [order.to_dict() for order in orders]
        
        except Exception as e:
            self.logger.error(f"Error getting orders: {e}")
            return []
    
    async def get_fills(
        self,
        symbol: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Get fills with optional filtering."""
        try:
            fills = list(self.fills.values())
            
            # Apply filters
            if symbol:
                fills = [f for f in fills if f.symbol == symbol]
            
            if start_date:
                fills = [f for f in fills if f.timestamp >= start_date]
            
            if end_date:
                fills = [f for f in fills if f.timestamp <= end_date]
            
            # Sort by timestamp (newest first)
            fills.sort(key=lambda x: x.timestamp, reverse=True)
            
            # Apply limit
            if limit:
                fills = fills[:limit]
            
            return [fill.to_dict() for fill in fills]
        
        except Exception as e:
            self.logger.error(f"Error getting fills: {e}")
            return []
    
    async def get_execution_report(self, period_hours: int = 24) -> Dict[str, Any]:
        """Get execution performance report."""
        try:
            cutoff_time = datetime.now() - timedelta(hours=period_hours)
            
            # Filter recent orders and fills
            recent_orders = [o for o in self.orders.values() if o.created_at >= cutoff_time]
            recent_fills = [f for f in self.fills.values() if f.timestamp >= cutoff_time]
            
            # Calculate metrics
            total_orders = len(recent_orders)
            filled_orders = len([o for o in recent_orders if o.status == OrderStatus.FILLED])
            cancelled_orders = len([o for o in recent_orders if o.status == OrderStatus.CANCELLED])
            rejected_orders = len([o for o in recent_orders if o.status == OrderStatus.REJECTED])
            
            total_volume = sum(f.gross_amount for f in recent_fills)
            total_commission = sum(f.commission for f in recent_fills)
            
            # Calculate average fill time
            fill_times = []
            for order in recent_orders:
                if order.status == OrderStatus.FILLED:
                    order_fills = [f for f in recent_fills if f.order_id == order.order_id]
                    if order_fills:
                        first_fill = min(order_fills, key=lambda x: x.timestamp)
                        fill_time = (first_fill.timestamp - order.created_at).total_seconds()
                        fill_times.append(fill_time)
            
            avg_fill_time = np.mean(fill_times) if fill_times else 0
            
            # Venue performance
            venue_stats = defaultdict(lambda: {'orders': 0, 'fills': 0, 'volume': 0, 'commission': 0})
            for order in recent_orders:
                if order.venue:
                    venue_stats[order.venue]['orders'] += 1
            
            for fill in recent_fills:
                venue_stats[fill.venue]['fills'] += 1
                venue_stats[fill.venue]['volume'] += fill.gross_amount
                venue_stats[fill.venue]['commission'] += fill.commission
            
            return {
                'period_hours': period_hours,
                'timestamp': datetime.now().isoformat(),
                'summary': {
                    'total_orders': total_orders,
                    'filled_orders': filled_orders,
                    'cancelled_orders': cancelled_orders,
                    'rejected_orders': rejected_orders,
                    'fill_rate': filled_orders / total_orders if total_orders > 0 else 0,
                    'total_volume': total_volume,
                    'total_commission': total_commission,
                    'average_fill_time_seconds': avg_fill_time
                },
                'venue_performance': dict(venue_stats),
                'execution_quality': self._calculate_execution_quality(recent_orders, recent_fills)
            }
        
        except Exception as e:
            self.logger.error(f"Error generating execution report: {e}")
            return {'error': str(e)}
    
    async def _route_and_execute_order(self, order: Order) -> Dict[str, Any]:
        """Route order to best venue and execute."""
        try:
            # Get best venue for order
            best_venue = await self._select_best_venue(order)
            
            if not best_venue:
                return {
                    'success': False,
                    'reason': 'No suitable venue found'
                }
            
            # Execute order at venue
            execution_result = await self._execute_order_at_venue(best_venue, order)
            
            return execution_result
        
        except Exception as e:
            self.logger.error(f"Error routing and executing order: {e}")
            return {
                'success': False,
                'reason': f'Execution error: {str(e)}'
            }
    
    async def _select_best_venue(self, order: Order) -> Optional[str]:
        """Select the best venue for an order using smart routing."""
        try:
            # Get eligible venues
            eligible_venues = []
            
            for venue_id, venue_info in self.venues.items():
                if not venue_info.is_active:
                    continue
                
                if not venue_info.supports_symbol(order.symbol):
                    continue
                
                if not venue_info.supports_order_type(order.order_type):
                    continue
                
                # Check order size limits
                min_size = venue_info.min_order_size.get('USD', 0)
                max_size = venue_info.max_order_size.get('USD', float('inf'))
                
                order_value = order.quantity * (order.price or 100)  # Estimate
                if not (min_size <= order_value <= max_size):
                    continue
                
                eligible_venues.append(venue_id)
            
            if not eligible_venues:
                return None
            
            # If only one venue, use it
            if len(eligible_venues) == 1:
                return eligible_venues[0]
            
            # Smart routing logic
            if self.config['smart_routing_enabled']:
                return await self._smart_route_order(order, eligible_venues)
            else:
                # Use first eligible venue
                return eligible_venues[0]
        
        except Exception as e:
            self.logger.error(f"Error selecting best venue: {e}")
            return None
    
    async def _smart_route_order(self, order: Order, eligible_venues: List[str]) -> str:
        """Apply smart routing logic to select best venue."""
        try:
            # Determine routing rule based on order characteristics
            order_value = order.quantity * (order.price or 100)
            
            if order_value > 100000:  # Large order
                routing_rule = self.routing_rules['large_order']
            elif order.metadata.get('urgent', False):
                routing_rule = self.routing_rules['urgent']
            else:
                routing_rule = self.routing_rules['default']
            
            # Score venues
            venue_scores = {}
            
            for venue_id in eligible_venues:
                venue_info = self.venues[venue_id]
                score = 0
                
                # Latency score
                if 'latency' in routing_rule['criteria']:
                    latency_score = max(0, 1 - (venue_info.latency_ms / 1000))
                    score += latency_score * routing_rule['weights'].get('latency', 0)
                
                # Success rate score
                if 'success_rate' in routing_rule['criteria']:
                    score += venue_info.success_rate * routing_rule['weights'].get('success_rate', 0)
                
                # Commission score (lower is better)
                if 'commission' in routing_rule['criteria']:
                    # Simplified commission calculation
                    commission_rate = 0.001  # Default
                    commission_score = max(0, 1 - commission_rate)
                    score += commission_score * routing_rule['weights'].get('commission', 0)
                
                venue_scores[venue_id] = score
            
            # Select venue with highest score
            best_venue = max(venue_scores.keys(), key=lambda x: venue_scores[x])
            
            self.logger.info(f"Smart routing selected {best_venue} for order {order.order_id}")
            return best_venue
        
        except Exception as e:
            self.logger.error(f"Error in smart routing: {e}")
            return eligible_venues[0]  # Fallback to first venue
    
    async def _execute_order_at_venue(self, venue_id: str, order: Order) -> Dict[str, Any]:
        """Execute order at specific venue."""
        try:
            # This would integrate with actual venue APIs
            # For now, simulate execution
            
            venue_order_id = f"{venue_id}_{uuid.uuid4().hex[:8]}"
            
            # Simulate venue response
            if venue_id == 'alpaca':
                return await self._execute_alpaca_order(order, venue_order_id)
            elif venue_id == 'binance':
                return await self._execute_binance_order(order, venue_order_id)
            elif venue_id == 'oanda':
                return await self._execute_oanda_order(order, venue_order_id)
            else:
                # Generic execution
                return {
                    'success': True,
                    'venue': venue_id,
                    'venue_order_id': venue_order_id
                }
        
        except Exception as e:
            self.logger.error(f"Error executing order at {venue_id}: {e}")
            return {
                'success': False,
                'reason': f'Venue execution error: {str(e)}'
            }
    
    async def _execute_alpaca_order(self, order: Order, venue_order_id: str) -> Dict[str, Any]:
        """Execute order on Alpaca."""
        try:
            from .alpaca_client import AlpacaClient, AlpacaOrder, AlpacaOrderSide, AlpacaOrderType, AlpacaTimeInForce
            from .trading_mode_validator import validate_before_trading
            import os
            
            # Validate trading mode configuration before executing
            validation_result = validate_before_trading()
            if not validation_result['valid']:
                self.logger.error(f"Trading mode validation failed: {validation_result['errors']}")
                return {
                    'success': False,
                    'reason': f'Trading mode validation failed: {", ".join(validation_result["errors"])}'
                }
            
            # Initialize Alpaca client based on TRADING_MODE environment variable
            trading_mode = os.getenv('TRADING_MODE', 'paper').lower()
            paper_trading = trading_mode == 'paper'
            alpaca_client = AlpacaClient(paper_trading=paper_trading)
            
            # Log trading mode for transparency
            self.logger.info(f"Executing Alpaca order in {'PAPER' if paper_trading else 'LIVE'} trading mode")
            
            # Additional safety check for live trading with large orders
            if not paper_trading and order.quantity * (order.price or 0) > 1000:
                self.logger.warning(f"Large live trade detected: {order.symbol} {order.quantity} @ {order.price}")
                # In a production system, you might want to require additional confirmation here
            
            # Convert order side
            alpaca_side = AlpacaOrderSide.BUY if order.side == OrderSide.BUY else AlpacaOrderSide.SELL
            
            # Convert order type
            if order.order_type == OrderType.MARKET:
                alpaca_type = AlpacaOrderType.MARKET
            elif order.order_type == OrderType.LIMIT:
                alpaca_type = AlpacaOrderType.LIMIT
            elif order.order_type == OrderType.STOP:
                alpaca_type = AlpacaOrderType.STOP
            elif order.order_type == OrderType.STOP_LIMIT:
                alpaca_type = AlpacaOrderType.STOP_LIMIT
            else:
                alpaca_type = AlpacaOrderType.MARKET
            
            # Convert time in force
            if order.time_in_force == TimeInForce.GTC:
                alpaca_tif = AlpacaTimeInForce.GTC
            elif order.time_in_force == TimeInForce.IOC:
                alpaca_tif = AlpacaTimeInForce.IOC
            elif order.time_in_force == TimeInForce.FOK:
                alpaca_tif = AlpacaTimeInForce.FOK
            else:
                alpaca_tif = AlpacaTimeInForce.DAY
            
            # Create Alpaca order
            alpaca_order = AlpacaOrder(
                symbol=order.symbol,
                qty=order.quantity,
                side=alpaca_side,
                type=alpaca_type,
                time_in_force=alpaca_tif,
                limit_price=order.price,
                stop_price=order.stop_price,
                client_order_id=order.order_id
            )
            
            # Submit order to Alpaca
            result = await alpaca_client.submit_order(alpaca_order)
            
            return {
                'success': True,
                'venue': 'alpaca',
                'venue_order_id': result['id'],
                'status': result['status'],
                'filled_qty': float(result.get('filled_qty', 0)),
                'filled_avg_price': float(result.get('filled_avg_price', 0)) if result.get('filled_avg_price') else None
            }
            
        except Exception as e:
            self.logger.error(f"Error executing Alpaca order: {e}")
            return {
                'success': False,
                'venue': 'alpaca',
                'venue_order_id': venue_order_id,
                'error': str(e)
            }
    
    async def _execute_binance_order(self, order: Order, venue_order_id: str) -> Dict[str, Any]:
        """Execute order on Binance."""
        # Placeholder for Binance API integration
        return {
            'success': True,
            'venue': 'binance',
            'venue_order_id': venue_order_id
        }
    
    async def _execute_oanda_order(self, order: Order, venue_order_id: str) -> Dict[str, Any]:
        """Execute order on OANDA."""
        # Placeholder for OANDA API integration
        return {
            'success': True,
            'venue': 'oanda',
            'venue_order_id': venue_order_id
        }
    
    async def _validate_order(self, order: Order) -> Dict[str, Any]:
        """Validate order before submission."""
        try:
            # Basic validation
            if order.quantity <= 0:
                return {'valid': False, 'reason': 'Invalid quantity'}
            
            if order.order_type in [OrderType.LIMIT, OrderType.STOP_LIMIT] and order.price is None:
                return {'valid': False, 'reason': 'Price required for limit orders'}
            
            if order.order_type in [OrderType.STOP, OrderType.STOP_LIMIT] and order.stop_price is None:
                return {'valid': False, 'reason': 'Stop price required for stop orders'}
            
            # Order size validation
            order_value = order.quantity * (order.price or 100)  # Estimate
            
            if order_value < self.config['min_order_size_usd']:
                return {'valid': False, 'reason': 'Order size too small'}
            
            if order_value > self.config['max_order_size_usd']:
                return {'valid': False, 'reason': 'Order size too large'}
            
            # Market hours validation (for traditional assets)
            if not await self._is_market_open(order.symbol):
                if order.order_type == OrderType.MARKET:
                    return {'valid': False, 'reason': 'Market closed for market orders'}
            
            return {'valid': True}
        
        except Exception as e:
            self.logger.error(f"Error validating order: {e}")
            return {'valid': False, 'reason': f'Validation error: {str(e)}'}
    
    async def _is_market_open(self, symbol: str) -> bool:
        """Check if market is open for symbol."""
        try:
            # Determine asset class
            if any(crypto in symbol.upper() for crypto in ['BTC', 'ETH', 'ADA', 'DOT']):
                return True  # Crypto markets are always open
            elif '/' in symbol and len(symbol.split('/')) == 2:
                # Forex pair
                return datetime.now().weekday() < 5  # Monday-Friday
            else:
                # Traditional stock - simplified check
                current_hour = datetime.now().hour
                return 9 <= current_hour <= 16  # Market hours
        
        except Exception as e:
            self.logger.error(f"Error checking market status: {e}")
            return False
    
    async def _monitor_orders(self):
        """Monitor active orders for updates."""
        while True:
            try:
                await asyncio.sleep(10)  # Check every 10 seconds
                
                active_orders = [
                    order for order in self.orders.values()
                    if not order.is_complete and order.venue and order.venue_order_id
                ]
                
                for order in active_orders:
                    try:
                        # Get status from venue
                        venue_status = await self._get_order_status_from_venue(
                            order.venue, order.venue_order_id
                        )
                        
                        if venue_status:
                            await self._update_order_from_venue_status(order, venue_status)
                    
                    except Exception as e:
                        self.logger.error(f"Error monitoring order {order.order_id}: {e}")
            
            except Exception as e:
                self.logger.error(f"Error in order monitoring: {e}")
                await asyncio.sleep(30)
    
    async def _monitor_venues(self):
        """Monitor venue health and performance."""
        while True:
            try:
                await asyncio.sleep(60)  # Check every minute
                
                for venue_id, venue_info in self.venues.items():
                    # Check venue connectivity
                    is_healthy = await self._check_venue_health(venue_id)
                    
                    if is_healthy != venue_info.is_active:
                        venue_info.is_active = is_healthy
                        venue_info.last_update = datetime.now()
                        
                        status = "connected" if is_healthy else "disconnected"
                        self.logger.info(f"Venue {venue_id} {status}")
                        
                        # Send notification for venue status change
                        await self.notification_manager.send_alert(
                            title=f"Venue Status Change",
                            message=f"Venue {venue_id} is now {status}",
                            severity="medium" if is_healthy else "high",
                            alert_type="VENUE_STATUS"
                        )
            
            except Exception as e:
                self.logger.error(f"Error in venue monitoring: {e}")
                await asyncio.sleep(60)
    
    async def _check_venue_health(self, venue_id: str) -> bool:
        """Check if a venue is healthy and responsive."""
        try:
            venue_info = self.venues.get(venue_id)
            if not venue_info:
                return False
            
            # Basic connectivity check based on venue type
            if venue_id == 'alpaca':
                # Check Alpaca API status
                try:
                    # Simple API call to check connectivity
                    # This would be replaced with actual Alpaca API call
                    return False  # Venue health check not implemented yet
                except Exception:
                    return False
                    
            elif venue_id == 'binance':
                # Check Binance API status
                try:
                    # Simple API call to check connectivity
                    # This would be replaced with actual Binance API call
                    return False  # Venue health check not implemented yet
                except Exception:
                    return False
                    
            elif venue_id == 'oanda':
                # Check OANDA API status
                try:
                    # Simple API call to check connectivity
                    # This would be replaced with actual OANDA API call
                    return False  # Venue health check not implemented yet
                except Exception:
                    return False
            
            # Default to healthy if venue is recognized
            return True
            
        except Exception as e:
            self.logger.error(f"Error checking venue health for {venue_id}: {e}")
            return False
    
    async def _update_execution_analytics(self):
        """Update execution analytics and metrics."""
        while True:
            try:
                await asyncio.sleep(300)  # Update every 5 minutes
                
                # Update execution metrics
                recent_orders = [
                    order for order in self.orders.values()
                    if order.created_at >= datetime.now() - timedelta(hours=24)
                ]
                
                recent_fills = [
                    fill for fill in self.fills.values()
                    if fill.timestamp >= datetime.now() - timedelta(hours=24)
                ]
                
                # Calculate and update metrics
                self.execution_metrics = ExecutionMetrics()
                
                for order in recent_orders:
                    self.execution_metrics.update_metrics(order)
                
                for fill in recent_fills:
                    self.execution_metrics.total_commission += fill.commission
                    self.execution_metrics.total_volume += fill.gross_amount
                
                # Update venue performance metrics
                await self._update_venue_performance_metrics()
                
            except Exception as e:
                self.logger.error(f"Error updating execution analytics: {e}")
                await asyncio.sleep(300)
    
    async def _update_venue_performance_metrics(self):
        """Update performance metrics for each trading venue."""
        try:
            # Get recent orders and fills for analysis
            recent_orders = [
                order for order in self.orders.values()
                if order.created_at >= datetime.now() - timedelta(hours=24)
            ]
            
            recent_fills = [
                fill for fill in self.fills.values()
                if fill.timestamp >= datetime.now() - timedelta(hours=24)
            ]
            
            # Initialize venue metrics
            venue_metrics = {}
            
            for venue_name in self.venues.keys():
                venue_orders = [order for order in recent_orders if order.venue == venue_name]
                venue_fills = [fill for fill in recent_fills if fill.venue == venue_name]
                
                if not venue_orders:
                    continue
                
                # Calculate metrics
                total_orders = len(venue_orders)
                filled_orders = len([order for order in venue_orders if order.status == OrderStatus.FILLED])
                cancelled_orders = len([order for order in venue_orders if order.status == OrderStatus.CANCELLED])
                rejected_orders = len([order for order in venue_orders if order.status == OrderStatus.REJECTED])
                
                fill_rate = filled_orders / total_orders if total_orders > 0 else 0.0
                
                # Calculate average fill time
                fill_times = []
                for order in venue_orders:
                    if order.status == OrderStatus.FILLED and order.filled_at:
                        fill_time = (order.filled_at - order.created_at).total_seconds()
                        fill_times.append(fill_time)
                
                avg_fill_time = sum(fill_times) / len(fill_times) if fill_times else 0.0
                
                # Calculate slippage
                slippages = []
                for fill in venue_fills:
                    if fill.expected_price and fill.price:
                        slippage_bps = abs(fill.price - fill.expected_price) / fill.expected_price * 10000
                        slippages.append(slippage_bps)
                
                avg_slippage = sum(slippages) / len(slippages) if slippages else 0.0
                
                # Calculate total commission and volume
                total_commission = sum(fill.commission for fill in venue_fills)
                total_volume = sum(fill.gross_amount for fill in venue_fills)
                
                venue_metrics[venue_name] = {
                    'total_orders': total_orders,
                    'filled_orders': filled_orders,
                    'cancelled_orders': cancelled_orders,
                    'rejected_orders': rejected_orders,
                    'fill_rate': fill_rate,
                    'avg_fill_time_seconds': avg_fill_time,
                    'avg_slippage_bps': avg_slippage,
                    'total_commission': total_commission,
                    'total_volume': total_volume,
                    'last_updated': datetime.now()
                }
            
            # Update execution metrics with venue performance
            self.execution_metrics.venue_performance = venue_metrics
            
            # Log performance summary
            for venue_name, metrics in venue_metrics.items():
                self.logger.info(
                    f"Venue {venue_name} performance: "
                    f"Fill rate: {metrics['fill_rate']:.2%}, "
                    f"Avg fill time: {metrics['avg_fill_time_seconds']:.2f}s, "
                    f"Avg slippage: {metrics['avg_slippage_bps']:.2f}bps"
                )
        
        except Exception as e:
            self.logger.error(f"Error updating venue performance metrics: {e}")
    
    async def _cleanup_old_orders(self):
        """Clean up old completed orders."""
        while True:
            try:
                await asyncio.sleep(3600)  # Clean up every hour
                
                cutoff_time = datetime.now() - timedelta(hours=self.config['max_order_age_hours'])
                
                # Move old orders to history
                old_orders = [
                    order_id for order_id, order in self.orders.items()
                    if order.is_complete and order.updated_at < cutoff_time
                ]
                
                for order_id in old_orders:
                    order = self.orders.pop(order_id)
                    self.order_history.append(order.to_dict())
                
                # Move old fills to history
                old_fills = [
                    fill_id for fill_id, fill in self.fills.items()
                    if fill.timestamp < cutoff_time
                ]
                
                for fill_id in old_fills:
                    fill = self.fills.pop(fill_id)
                    self.fill_history.append(fill.to_dict())
                
                if old_orders or old_fills:
                    self.logger.info(f"Cleaned up {len(old_orders)} orders and {len(old_fills)} fills")
            
            except Exception as e:
                self.logger.error(f"Error in order cleanup: {e}")
                await asyncio.sleep(3600)
    
    # Event handling
    def add_order_event_handler(self, handler: Callable):
        """Add order event handler."""
        self.order_event_handlers.append(handler)
    
    def add_fill_event_handler(self, handler: Callable):
        """Add fill event handler."""
        self.fill_event_handlers.append(handler)
    
    async def _notify_order_event(self, order: Order, event_type: str):
        """Notify order event handlers."""
        for handler in self.order_event_handlers:
            try:
                await handler(order, event_type)
            except Exception as e:
                self.logger.error(f"Error in order event handler: {e}")
    
    async def _notify_fill_event(self, fill: Fill):
        """Notify fill event handlers and save trade to storage."""
        # Save trade to storage system
        try:
            trade_data = {
                'trade_id': fill.fill_id,
                'timestamp': fill.timestamp,
                'symbol': fill.symbol,
                'side': fill.side.value.upper(),
                'quantity': fill.quantity,
                'price': fill.price,
                'value': fill.gross_amount,
                'commission': fill.commission,
                'fees': sum(fill.fees.values()) if fill.fees else 0.0,
                'venue': fill.venue,
                'order_id': fill.order_id,
                'fill_id': fill.fill_id,
                'metadata': json.dumps(fill.metadata) if fill.metadata else None
            }
            
            success = self.trade_storage.save_trade(trade_data)
            if success:
                self.logger.info(f"Trade saved to storage: {fill.symbol} {fill.side.value} {fill.quantity}")
            else:
                self.logger.warning(f"Failed to save trade to storage: {fill.fill_id}")
                
        except Exception as e:
            self.logger.error(f"Error saving trade to storage: {e}")
        
        # Notify other fill event handlers
        for handler in self.fill_event_handlers:
            try:
                await handler(fill)
            except Exception as e:
                self.logger.error(f"Error in fill event handler: {e}")
    
    def get_positions(self) -> Dict[str, float]:
        """Get current positions based on filled orders."""
        positions = {}
        
        for fill in self.fills.values():
            symbol = fill.symbol
            if symbol not in positions:
                positions[symbol] = 0.0
            
            # Add to position for buys, subtract for sells
            if fill.side in [OrderSide.BUY, OrderSide.BUY_TO_COVER]:
                positions[symbol] += fill.quantity
            else:  # SELL or SELL_SHORT
                positions[symbol] -= fill.quantity
        
        # Remove zero positions
        return {symbol: qty for symbol, qty in positions.items() if abs(qty) > 1e-8}
    
    def get_engine_status(self) -> Dict[str, Any]:
        """Get comprehensive engine status."""
        return {
            'timestamp': datetime.now().isoformat(),
            'active_orders': len([o for o in self.orders.values() if not o.is_complete]),
            'total_orders': len(self.orders),
            'total_fills': len(self.fills),
            'active_venues': len([v for v in self.venues.values() if v.is_active]),
            'total_venues': len(self.venues),
            'execution_metrics': {
                'success_rate': self.execution_metrics.success_rate,
                'total_volume': self.execution_metrics.total_volume,
                'total_commission': self.execution_metrics.total_commission
            },
            'background_tasks': len([t for t in self._background_tasks if not t.done()])
        }


if __name__ == "__main__":
    # Example usage
    import asyncio
    from pathlib import Path
    
    async def test_execution_engine():
        config_manager = ConfigManager(Path("../config"))
        market_data_aggregator = MarketDataAggregator(config_manager)
        risk_manager = RiskManager24_7(config_manager)
        
        engine = ExecutionEngine(config_manager, market_data_aggregator, risk_manager)
        
        # Test order submission
        order_id = await engine.submit_order(
            symbol='AAPL',
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=100,
            price=150.00,
            strategy_id='test_strategy'
        )
        
        if order_id:
            print(f"Order submitted: {order_id}")
            
            # Check order status
            status = await engine.get_order_status(order_id)
            print(f"Order status: {status}")
        
        # Get execution report
        report = await engine.get_execution_report()
        print(f"Execution report: {report}")
    
    # Commented out to prevent event loop conflicts when imported
    # asyncio.run(test_execution_engine())