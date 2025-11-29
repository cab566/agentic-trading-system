#!/usr/bin/env python3
"""
Order Management Tool for CrewAI Trading System

Provides agents with comprehensive order management capabilities
including order placement, modification, cancellation, and execution tracking.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union, Tuple
from enum import Enum
from dataclasses import dataclass, field

import pandas as pd
import numpy as np
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from core.data_manager import UnifiedDataManager, DataRequest


class OrderType(Enum):
    """Order types supported by the system."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    TRAILING_STOP = "trailing_stop"
    BRACKET = "bracket"
    OCO = "oco"  # One-Cancels-Other


class OrderSide(Enum):
    """Order sides."""
    BUY = "buy"
    SELL = "sell"


class OrderStatus(Enum):
    """Order status values."""
    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


class TimeInForce(Enum):
    """Time in force options."""
    DAY = "day"
    GTC = "gtc"  # Good Till Cancelled
    IOC = "ioc"  # Immediate or Cancel
    FOK = "fok"  # Fill or Kill
    GTD = "gtd"  # Good Till Date


@dataclass
class Order:
    """Order data structure."""
    order_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: TimeInForce = TimeInForce.DAY
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    avg_fill_price: Optional[float] = None
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None
    client_order_id: Optional[str] = None
    parent_order_id: Optional[str] = None
    child_orders: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Fill:
    """Trade fill data structure."""
    fill_id: str
    order_id: str
    symbol: str
    side: OrderSide
    quantity: float
    price: float
    timestamp: datetime
    commission: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class OrderManagementInput(BaseModel):
    """Input schema for order management requests."""
    action: str = Field(
        ...,
        description="Action to perform: 'place', 'modify', 'cancel', 'status', 'list', 'fills', 'positions'"
    )
    symbol: Optional[str] = Field(
        default=None,
        description="Trading symbol (e.g., 'AAPL', 'MSFT')"
    )
    side: Optional[str] = Field(
        default=None,
        description="Order side: 'buy' or 'sell'"
    )
    order_type: Optional[str] = Field(
        default="market",
        description="Order type: 'market', 'limit', 'stop', 'stop_limit', 'trailing_stop', 'bracket', 'oco'"
    )
    quantity: Optional[float] = Field(
        default=None,
        description="Order quantity (number of shares)"
    )
    price: Optional[float] = Field(
        default=None,
        description="Limit price for limit orders"
    )
    stop_price: Optional[float] = Field(
        default=None,
        description="Stop price for stop orders"
    )
    time_in_force: Optional[str] = Field(
        default="day",
        description="Time in force: 'day', 'gtc', 'ioc', 'fok', 'gtd'"
    )
    order_id: Optional[str] = Field(
        default=None,
        description="Order ID for modify/cancel/status operations"
    )
    client_order_id: Optional[str] = Field(
        default=None,
        description="Client-specified order ID"
    )
    expires_at: Optional[str] = Field(
        default=None,
        description="Expiration datetime for GTD orders (ISO format)"
    )
    # Bracket order parameters
    take_profit_price: Optional[float] = Field(
        default=None,
        description="Take profit price for bracket orders"
    )
    stop_loss_price: Optional[float] = Field(
        default=None,
        description="Stop loss price for bracket orders"
    )
    # Trailing stop parameters
    trail_amount: Optional[float] = Field(
        default=None,
        description="Trail amount for trailing stop orders"
    )
    trail_percent: Optional[float] = Field(
        default=None,
        description="Trail percentage for trailing stop orders"
    )
    # Risk management
    max_position_size: Optional[float] = Field(
        default=None,
        description="Maximum position size limit"
    )
    max_order_value: Optional[float] = Field(
        default=None,
        description="Maximum order value limit"
    )
    # Filters
    status_filter: Optional[List[str]] = Field(
        default=None,
        description="Filter orders by status for list operations"
    )
    date_from: Optional[str] = Field(
        default=None,
        description="Start date for historical queries (ISO format)"
    )
    date_to: Optional[str] = Field(
        default=None,
        description="End date for historical queries (ISO format)"
    )


class OrderManagementTool(BaseTool):
    """
    Order Management Tool for CrewAI agents.
    
    Provides comprehensive order management including:
    - Order placement (market, limit, stop, bracket orders)
    - Order modification and cancellation
    - Order status tracking
    - Fill and execution tracking
    - Position management
    - Risk management and validation
    - Order history and reporting
    """
    
    name: str = "order_management_tool"
    description: str = (
        "Manage trading orders including placement, modification, cancellation, "
        "and tracking. Supports various order types (market, limit, stop, bracket) "
        "and provides comprehensive order and position management capabilities."
    )
    args_schema: type[OrderManagementInput] = OrderManagementInput
    data_manager: UnifiedDataManager = Field(default=None, exclude=True)
    logger: Any = Field(default=None, exclude=True)
    orders: Dict = Field(default_factory=dict, exclude=True)
    fills: Dict = Field(default_factory=dict, exclude=True)
    positions: Dict = Field(default_factory=dict, exclude=True)
    order_counter: int = Field(default=0, exclude=True)
    fill_counter: int = Field(default=0, exclude=True)
    max_position_size: float = Field(default=10000, exclude=True)
    max_order_value: float = Field(default=100000, exclude=True)
    max_daily_trades: int = Field(default=100, exclude=True)
    daily_trade_count: int = Field(default=0, exclude=True)
    last_reset_date: Any = Field(default=None, exclude=True)
    market_open_time: str = Field(default="09:30", exclude=True)
    market_close_time: str = Field(default="16:00", exclude=True)
    
    def __init__(self, data_manager: UnifiedDataManager, **kwargs):
        """
        Initialize the order management tool.
        
        Args:
            data_manager: Unified data manager instance
        """
        super().__init__(**kwargs)
        self.data_manager = data_manager
        self.logger = logging.getLogger(__name__)
        
        # Order storage (in production, this would be a database)
        self.orders: Dict[str, Order] = {}
        self.fills: Dict[str, Fill] = {}
        self.positions: Dict[str, float] = {}  # symbol -> quantity
        self.order_counter = 0
        self.fill_counter = 0
        
        # Risk limits (configurable)
        self.max_position_size = 10000  # shares
        self.max_order_value = 100000  # dollars
        self.max_daily_trades = 100
        self.daily_trade_count = 0
        self.last_reset_date = datetime.now().date()
        
        # Market hours (simplified)
        self.market_open_time = "09:30"
        self.market_close_time = "16:00"
    
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
            self.logger.error(f"Error in order management tool: {e}")
            return f"Error processing order management request: {str(e)}"
    
    async def _arun(self, **kwargs) -> str:
        """Asynchronous execution of order management."""
        try:
            # Parse input
            input_data = OrderManagementInput(**kwargs)
            
            # Reset daily counters if needed
            self._reset_daily_counters()
            
            # Route to appropriate handler
            if input_data.action == "place":
                return await self._place_order(input_data)
            elif input_data.action == "modify":
                return await self._modify_order(input_data)
            elif input_data.action == "cancel":
                return await self._cancel_order(input_data)
            elif input_data.action == "status":
                return await self._get_order_status(input_data)
            elif input_data.action == "list":
                return await self._list_orders(input_data)
            elif input_data.action == "fills":
                return await self._get_fills(input_data)
            elif input_data.action == "positions":
                return await self._get_positions(input_data)
            else:
                return f"Error: Unknown action '{input_data.action}'"
                
        except Exception as e:
            self.logger.error(f"Error in async order management: {e}")
            return f"Error processing order management request: {str(e)}"
    
    async def _place_order(self, input_data: OrderManagementInput) -> str:
        """Place a new order."""
        try:
            # Validate required fields
            if not input_data.symbol or not input_data.side or not input_data.quantity:
                return "Error: symbol, side, and quantity are required for placing orders"
            
            # Clean symbol
            symbol = input_data.symbol.upper().replace('$', '')
            
            # Validate order parameters
            validation_result = await self._validate_order(input_data, symbol)
            if validation_result:
                return validation_result
            
            # Generate order ID
            self.order_counter += 1
            order_id = f"ORD_{datetime.now().strftime('%Y%m%d')}_{self.order_counter:06d}"
            
            # Parse order parameters
            side = OrderSide(input_data.side.lower())
            order_type = OrderType(input_data.order_type.lower())
            time_in_force = TimeInForce(input_data.time_in_force.lower())
            
            # Handle expiration
            expires_at = None
            if input_data.expires_at:
                expires_at = datetime.fromisoformat(input_data.expires_at.replace('Z', '+00:00'))
            elif time_in_force == TimeInForce.DAY:
                # Set expiration to market close
                today = datetime.now().date()
                expires_at = datetime.combine(today, datetime.strptime(self.market_close_time, '%H:%M').time())
            
            # Create order
            order = Order(
                order_id=order_id,
                symbol=symbol,
                side=side,
                order_type=order_type,
                quantity=input_data.quantity,
                price=input_data.price,
                stop_price=input_data.stop_price,
                time_in_force=time_in_force,
                expires_at=expires_at,
                client_order_id=input_data.client_order_id,
                metadata={
                    'take_profit_price': input_data.take_profit_price,
                    'stop_loss_price': input_data.stop_loss_price,
                    'trail_amount': input_data.trail_amount,
                    'trail_percent': input_data.trail_percent
                }
            )
            
            # Handle special order types
            if order_type == OrderType.BRACKET:
                result = await self._create_bracket_order(order, input_data)
                if "Error" in result:
                    return result
            elif order_type == OrderType.OCO:
                result = await self._create_oco_order(order, input_data)
                if "Error" in result:
                    return result
            
            # Store order
            self.orders[order_id] = order
            
            # Simulate order submission (in production, this would call broker API)
            await self._submit_order(order)
            
            # Update daily trade count
            self.daily_trade_count += 1
            
            result = f"Order Placed Successfully\n"
            result += f"Order ID: {order_id}\n"
            result += f"Symbol: {symbol}\n"
            result += f"Side: {side.value.upper()}\n"
            result += f"Type: {order_type.value.upper()}\n"
            result += f"Quantity: {input_data.quantity:,.0f}\n"
            
            if order.price:
                result += f"Price: ${order.price:.2f}\n"
            if order.stop_price:
                result += f"Stop Price: ${order.stop_price:.2f}\n"
            
            result += f"Time in Force: {time_in_force.value.upper()}\n"
            result += f"Status: {order.status.value.upper()}\n"
            
            if expires_at:
                result += f"Expires: {expires_at.strftime('%Y-%m-%d %H:%M:%S')}\n"
            
            # Add estimated order value
            estimated_value = await self._estimate_order_value(order)
            if estimated_value:
                result += f"Estimated Value: ${estimated_value:,.2f}\n"
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error placing order: {e}")
            return f"Error placing order: {str(e)}"
    
    async def _get_order_status(self, input_data: OrderManagementInput) -> str:
        """Get status of a specific order."""
        try:
            if not input_data.order_id:
                return "Error: order_id is required for status queries"
            
            order = self.orders.get(input_data.order_id)
            if not order:
                return f"Error: Order {input_data.order_id} not found"
            
            result = f"Order Status Report\n"
            result += f"Order ID: {order.order_id}\n"
            result += f"Client Order ID: {order.client_order_id or 'N/A'}\n"
            result += f"Symbol: {order.symbol}\n"
            result += f"Side: {order.side.value.upper()}\n"
            result += f"Type: {order.order_type.value.upper()}\n"
            result += f"Quantity: {order.quantity:,.0f}\n"
            result += f"Filled Quantity: {order.filled_quantity:,.0f}\n"
            result += f"Remaining: {order.quantity - order.filled_quantity:,.0f}\n"
            
            if order.price:
                result += f"Price: ${order.price:.2f}\n"
            if order.stop_price:
                result += f"Stop Price: ${order.stop_price:.2f}\n"
            if order.avg_fill_price:
                result += f"Average Fill Price: ${order.avg_fill_price:.2f}\n"
            
            result += f"Time in Force: {order.time_in_force.value.upper()}\n"
            result += f"Status: {order.status.value.upper()}\n"
            result += f"Created: {order.created_at.strftime('%Y-%m-%d %H:%M:%S')}\n"
            result += f"Updated: {order.updated_at.strftime('%Y-%m-%d %H:%M:%S')}\n"
            
            if order.expires_at:
                result += f"Expires: {order.expires_at.strftime('%Y-%m-%d %H:%M:%S')}\n"
            
            # Show fills for this order
            order_fills = [fill for fill in self.fills.values() if fill.order_id == order.order_id]
            if order_fills:
                result += f"\nFills ({len(order_fills)}):\n"
                for fill in order_fills:
                    result += f"  {fill.timestamp.strftime('%H:%M:%S')}: {fill.quantity:,.0f} @ ${fill.price:.2f}\n"
            
            # Show child orders for bracket/OCO orders
            if order.child_orders:
                result += f"\nChild Orders ({len(order.child_orders)}):\n"
                for child_id in order.child_orders:
                    child_order = self.orders.get(child_id)
                    if child_order:
                        result += f"  {child_id}: {child_order.order_type.value.upper()} - {child_order.status.value.upper()}\n"
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error getting order status: {e}")
            return f"Error getting order status: {str(e)}"
    
    async def _validate_order(self, input_data: OrderManagementInput, symbol: str) -> Optional[str]:
        """Validate order parameters."""
        try:
            # Check market hours (simplified)
            current_time = datetime.now().time()
            market_open = datetime.strptime(self.market_open_time, '%H:%M').time()
            market_close = datetime.strptime(self.market_close_time, '%H:%M').time()
            
            if not (market_open <= current_time <= market_close):
                if input_data.order_type == "market":
                    return "Error: Market orders can only be placed during market hours"
            
            # Check daily trade limit
            if self.daily_trade_count >= self.max_daily_trades:
                return f"Error: Daily trade limit ({self.max_daily_trades}) exceeded"
            
            # Validate quantity
            if input_data.quantity <= 0:
                return "Error: Quantity must be positive"
            
            if input_data.quantity > self.max_position_size:
                return f"Error: Quantity exceeds maximum position size ({self.max_position_size})"
            
            # Validate prices
            if input_data.order_type in ["limit", "stop_limit"] and not input_data.price:
                return "Error: Limit price required for limit orders"
            
            if input_data.order_type in ["stop", "stop_limit", "trailing_stop"] and not input_data.stop_price:
                return "Error: Stop price required for stop orders"
            
            if input_data.price and input_data.price <= 0:
                return "Error: Price must be positive"
            
            if input_data.stop_price and input_data.stop_price <= 0:
                return "Error: Stop price must be positive"
            
            # Validate order value
            if input_data.price:
                order_value = input_data.quantity * input_data.price
                if order_value > self.max_order_value:
                    return f"Error: Order value (${order_value:,.2f}) exceeds maximum (${self.max_order_value:,.2f})"
            
            # Check position limits
            current_position = self.positions.get(symbol, 0)
            if input_data.side == "buy":
                new_position = current_position + input_data.quantity
            else:
                new_position = current_position - input_data.quantity
            
            if abs(new_position) > self.max_position_size:
                return f"Error: Order would result in position size ({abs(new_position):,.0f}) exceeding limit ({self.max_position_size:,.0f})"
            
            return None  # No validation errors
            
        except Exception as e:
            self.logger.error(f"Error validating order: {e}")
            return f"Error validating order: {str(e)}"
    
    async def _submit_order(self, order: Order) -> None:
        """Simulate order submission to broker."""
        try:
            # In production, this would call the broker's API
            # For simulation, we'll just update the status
            
            order.status = OrderStatus.SUBMITTED
            order.updated_at = datetime.now()
            
            # Simulate immediate fill for market orders (simplified)
            if order.order_type == OrderType.MARKET:
                await self._simulate_fill(order)
            
            self.logger.info(f"Order {order.order_id} submitted successfully")
            
        except Exception as e:
            order.status = OrderStatus.REJECTED
            order.updated_at = datetime.now()
            self.logger.error(f"Error submitting order {order.order_id}: {e}")
    
    async def _simulate_fill(self, order: Order) -> None:
        """Simulate order fill (for testing purposes)."""
        try:
            # Get current price
            current_price = await self._get_current_price(order.symbol)
            if not current_price:
                return
            
            # Determine fill price based on order type
            if order.order_type == OrderType.MARKET:
                fill_price = current_price
            elif order.order_type == OrderType.LIMIT:
                # For simulation, assume limit orders fill at limit price
                fill_price = order.price
            else:
                return  # Don't simulate fills for other order types
            
            # Create fill
            self.fill_counter += 1
            fill_id = f"FILL_{datetime.now().strftime('%Y%m%d')}_{self.fill_counter:06d}"
            
            fill = Fill(
                fill_id=fill_id,
                order_id=order.order_id,
                symbol=order.symbol,
                side=order.side,
                quantity=order.quantity,
                price=fill_price,
                timestamp=datetime.now(),
                commission=max(1.0, order.quantity * 0.005)  # Simplified commission
            )
            
            # Store fill
            self.fills[fill_id] = fill
            
            # Update order
            order.filled_quantity = order.quantity
            order.avg_fill_price = fill_price
            order.status = OrderStatus.FILLED
            order.updated_at = datetime.now()
            
            # Update position
            current_position = self.positions.get(order.symbol, 0)
            if order.side == OrderSide.BUY:
                self.positions[order.symbol] = current_position + order.quantity
            else:
                self.positions[order.symbol] = current_position - order.quantity
            
            self.logger.info(f"Order {order.order_id} filled: {order.quantity} @ ${fill_price:.2f}")
            
        except Exception as e:
            self.logger.error(f"Error simulating fill for order {order.order_id}: {e}")
    
    async def _get_current_price(self, symbol: str) -> Optional[float]:
        """Get current market price for a symbol."""
        try:
            # Create data request
            request = DataRequest(
                symbol=symbol,
                data_type="quote",
                timeframe="1m"
            )
            
            # Get current quote
            response = await self.data_manager.get_data(request)
            
            if response.error or response.data is None or response.data.empty:
                # No fallback - raise error if price data unavailable
                raise RuntimeError(f"Unable to get current price for {symbol}: {response.error}")
            
            return float(response.data['Close'].iloc[-1])
            
            # Return current price from quote
            if 'price' in response.data.columns:
                return float(response.data['price'].iloc[-1])
            elif 'Close' in response.data.columns:
                return float(response.data['Close'].iloc[-1])
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting current price for {symbol}: {e}")
            return None
    
    async def _estimate_order_value(self, order: Order) -> Optional[float]:
        """Estimate the value of an order."""
        try:
            if order.price:
                return order.quantity * order.price
            
            # For market orders, use current price
            current_price = await self._get_current_price(order.symbol)
            if current_price:
                return order.quantity * current_price
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error estimating order value: {e}")
            return None
    
    async def _create_bracket_order(self, parent_order: Order, input_data: OrderManagementInput) -> str:
        """Create bracket order with take profit and stop loss."""
        try:
            child_orders = []
            
            # Create take profit order
            if input_data.take_profit_price:
                tp_side = OrderSide.SELL if parent_order.side == OrderSide.BUY else OrderSide.BUY
                tp_order_id = f"{parent_order.order_id}_TP"
                
                tp_order = Order(
                    order_id=tp_order_id,
                    symbol=parent_order.symbol,
                    side=tp_side,
                    order_type=OrderType.LIMIT,
                    quantity=parent_order.quantity,
                    price=input_data.take_profit_price,
                    time_in_force=parent_order.time_in_force,
                    parent_order_id=parent_order.order_id
                )
                
                self.orders[tp_order_id] = tp_order
                child_orders.append(tp_order_id)
            
            # Create stop loss order
            if input_data.stop_loss_price:
                sl_side = OrderSide.SELL if parent_order.side == OrderSide.BUY else OrderSide.BUY
                sl_order_id = f"{parent_order.order_id}_SL"
                
                sl_order = Order(
                    order_id=sl_order_id,
                    symbol=parent_order.symbol,
                    side=sl_side,
                    order_type=OrderType.STOP,
                    quantity=parent_order.quantity,
                    stop_price=input_data.stop_loss_price,
                    time_in_force=parent_order.time_in_force,
                    parent_order_id=parent_order.order_id
                )
                
                self.orders[sl_order_id] = sl_order
                child_orders.append(sl_order_id)
            
            parent_order.child_orders = child_orders
            
            return "Bracket order created successfully"
            
        except Exception as e:
            self.logger.error(f"Error creating bracket order: {e}")
            return f"Error creating bracket order: {str(e)}"
    
    async def _list_orders(self, input_data: OrderManagementInput) -> str:
        """List orders with optional filtering."""
        try:
            orders_list = list(self.orders.values())
            
            # Apply status filter if provided
            if input_data.status_filter:
                status_filter = [status.lower() for status in input_data.status_filter]
                orders_list = [order for order in orders_list if order.status.value in status_filter]
            
            # Apply symbol filter if provided
            if input_data.symbol:
                orders_list = [order for order in orders_list if order.symbol == input_data.symbol]
            
            if not orders_list:
                return "No orders found matching the criteria"
            
            result = f"Found {len(orders_list)} orders:\n\n"
            for order in orders_list:
                result += f"Order ID: {order.order_id}\n"
                result += f"Symbol: {order.symbol}\n"
                result += f"Side: {order.side.value.upper()}\n"
                result += f"Type: {order.order_type.value.upper()}\n"
                result += f"Quantity: {order.quantity}\n"
                result += f"Status: {order.status.value.upper()}\n"
                if order.price:
                    result += f"Price: ${order.price:.2f}\n"
                result += f"Created: {order.created_at.strftime('%Y-%m-%d %H:%M:%S')}\n"
                result += "-" * 40 + "\n"
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error listing orders: {e}")
            return f"Error listing orders: {str(e)}"
    
    async def _get_fills(self, input_data: OrderManagementInput) -> str:
        """Get fill history with optional filtering."""
        try:
            fills_list = list(self.fills.values())
            
            # Apply symbol filter if provided
            if input_data.symbol:
                fills_list = [fill for fill in fills_list if fill.symbol == input_data.symbol]
            
            # Apply order ID filter if provided
            if input_data.order_id:
                fills_list = [fill for fill in fills_list if fill.order_id == input_data.order_id]
            
            if not fills_list:
                return "No fills found matching the criteria"
            
            result = f"Found {len(fills_list)} fills:\n\n"
            for fill in fills_list:
                result += f"Fill ID: {fill.fill_id}\n"
                result += f"Order ID: {fill.order_id}\n"
                result += f"Symbol: {fill.symbol}\n"
                result += f"Side: {fill.side.value.upper()}\n"
                result += f"Quantity: {fill.quantity}\n"
                result += f"Price: ${fill.price:.2f}\n"
                result += f"Value: ${fill.quantity * fill.price:.2f}\n"
                result += f"Commission: ${fill.commission:.2f}\n"
                result += f"Timestamp: {fill.timestamp.strftime('%Y-%m-%d %H:%M:%S')}\n"
                result += "-" * 40 + "\n"
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error getting fills: {e}")
            return f"Error getting fills: {str(e)}"
    
    async def _get_positions(self, input_data: OrderManagementInput) -> str:
        """Get current positions."""
        try:
            if not self.positions:
                return "No open positions"
            
            result = "Current Positions:\n\n"
            total_value = 0.0
            
            for symbol, position in self.positions.items():
                 if position == 0:
                     continue
                     
                 # Apply symbol filter if provided
                 if input_data.symbol and symbol != input_data.symbol:
                     continue
                 
                 # Get current price for market value calculation
                 current_price = await self._get_current_price(symbol)
                 if current_price:
                     market_value = position * current_price
                     total_value += market_value
                 else:
                     market_value = 0.0
                 
                 result += f"Symbol: {symbol}\n"
                 result += f"Quantity: {position:,.0f}\n"
                 if current_price:
                     result += f"Current Price: ${current_price:.2f}\n"
                     result += f"Market Value: ${market_value:.2f}\n"
                 result += "-" * 40 + "\n"
            
            if total_value > 0:
                result += f"\nTotal Portfolio Value: ${total_value:.2f}\n"
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error getting positions: {e}")
            return f"Error getting positions: {str(e)}"
    
    async def _modify_order(self, input_data: OrderManagementInput) -> str:
        """Modify an existing order."""
        try:
            if not input_data.order_id:
                return "Error: order_id is required for order modification"
            
            order = self.orders.get(input_data.order_id)
            if not order:
                return f"Error: Order {input_data.order_id} not found"
            
            if order.status not in [OrderStatus.PENDING, OrderStatus.SUBMITTED]:
                return f"Error: Cannot modify order in {order.status.value} status"
            
            # Update modifiable fields
            modified_fields = []
            
            if input_data.quantity and input_data.quantity != order.quantity:
                order.quantity = input_data.quantity
                modified_fields.append(f"quantity to {input_data.quantity}")
            
            if input_data.price and input_data.price != order.price:
                order.price = input_data.price
                modified_fields.append(f"price to ${input_data.price:.2f}")
            
            if input_data.stop_price and input_data.stop_price != order.stop_price:
                order.stop_price = input_data.stop_price
                modified_fields.append(f"stop price to ${input_data.stop_price:.2f}")
            
            if not modified_fields:
                return "No modifications specified"
            
            order.updated_at = datetime.now()
            
            result = f"Order {input_data.order_id} modified successfully\n"
            result += f"Modified: {', '.join(modified_fields)}\n"
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error modifying order: {e}")
            return f"Error modifying order: {str(e)}"
    
    async def _cancel_order(self, input_data: OrderManagementInput) -> str:
        """Cancel an existing order."""
        try:
            if not input_data.order_id:
                return "Error: order_id is required for order cancellation"
            
            order = self.orders.get(input_data.order_id)
            if not order:
                return f"Error: Order {input_data.order_id} not found"
            
            if order.status in [OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED]:
                return f"Error: Cannot cancel order in {order.status.value} status"
            
            order.status = OrderStatus.CANCELLED
            order.updated_at = datetime.now()
            
            # Cancel child orders for bracket orders
            cancelled_children = []
            for child_id in order.child_orders:
                child_order = self.orders.get(child_id)
                if child_order and child_order.status not in [OrderStatus.FILLED, OrderStatus.CANCELLED]:
                    child_order.status = OrderStatus.CANCELLED
                    child_order.updated_at = datetime.now()
                    cancelled_children.append(child_id)
            
            result = f"Order {input_data.order_id} cancelled successfully\n"
            if cancelled_children:
                result += f"Also cancelled child orders: {', '.join(cancelled_children)}\n"
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error cancelling order: {e}")
            return f"Error cancelling order: {str(e)}"
    
    async def _create_oco_order(self, parent_order: Order, input_data: OrderManagementInput) -> str:
        """Create One-Cancels-Other order."""
        try:
            # OCO orders require two child orders with different trigger conditions
            # This is a simplified implementation
            return "OCO orders not fully implemented in this version"
            
        except Exception as e:
            self.logger.error(f"Error creating OCO order: {e}")
            return f"Error creating OCO order: {str(e)}"
    
    def _reset_daily_counters(self) -> None:
        """Reset daily counters if it's a new day."""
        try:
            current_date = datetime.now().date()
            if current_date > self.last_reset_date:
                self.daily_trade_count = 0
                self.last_reset_date = current_date
                
        except Exception as e:
            self.logger.error(f"Error resetting daily counters: {e}")