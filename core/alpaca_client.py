#!/usr/bin/env python3
"""
Alpaca Trading API Client

Provides real trading functionality with Alpaca Markets API.
Supports both paper and live trading modes.
"""

import os
import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum

import requests
import pandas as pd
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


class AlpacaOrderSide(Enum):
    """Alpaca order side enumeration."""
    BUY = "buy"
    SELL = "sell"


class AlpacaOrderType(Enum):
    """Alpaca order type enumeration."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    TRAILING_STOP = "trailing_stop"


class AlpacaTimeInForce(Enum):
    """Alpaca time in force enumeration."""
    DAY = "day"
    GTC = "gtc"
    IOC = "ioc"
    FOK = "fok"


@dataclass
class AlpacaOrder:
    """Alpaca order data structure."""
    symbol: str
    qty: Union[int, float]
    side: AlpacaOrderSide
    type: AlpacaOrderType
    time_in_force: AlpacaTimeInForce = AlpacaTimeInForce.DAY
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    trail_price: Optional[float] = None
    trail_percent: Optional[float] = None
    extended_hours: bool = False
    client_order_id: Optional[str] = None


class AlpacaClient:
    """Alpaca Trading API Client."""
    
    def __init__(self, paper_trading: bool = True):
        """
        Initialize Alpaca client.
        
        Args:
            paper_trading: Whether to use paper trading (default: True)
        """
        self.logger = logging.getLogger(__name__)
        
        # Get API credentials from environment
        self.api_key = os.getenv('ALPACA_API_KEY')
        self.secret_key = os.getenv('ALPACA_SECRET_KEY')
        
        if not self.api_key or not self.secret_key:
            raise ValueError("Alpaca API credentials not found in environment variables")
        
        # Set base URL based on trading mode
        if paper_trading:
            self.base_url = "https://paper-api.alpaca.markets"
            self.data_url = "https://data.alpaca.markets"
        else:
            self.base_url = "https://api.alpaca.markets"
            self.data_url = "https://data.alpaca.markets"
        
        self.paper_trading = paper_trading
        
        # Setup HTTP session with retry strategy
        self.session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        # Set default headers
        self.session.headers.update({
            'APCA-API-KEY-ID': self.api_key,
            'APCA-API-SECRET-KEY': self.secret_key,
            'Content-Type': 'application/json'
        })
        
        self.logger.info(f"Alpaca client initialized - Paper Trading: {paper_trading}")
    
    async def get_account(self) -> Dict[str, Any]:
        """Get account information."""
        try:
            response = await asyncio.to_thread(
                self.session.get, f"{self.base_url}/v2/account"
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            self.logger.error(f"Error getting account info: {e}")
            raise
    
    async def get_positions(self) -> List[Dict[str, Any]]:
        """Get all positions."""
        try:
            response = await asyncio.to_thread(
                self.session.get, f"{self.base_url}/v2/positions"
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            self.logger.error(f"Error getting positions: {e}")
            raise
    
    async def get_position(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get position for a specific symbol."""
        try:
            response = await asyncio.to_thread(
                self.session.get, f"{self.base_url}/v2/positions/{symbol}"
            )
            if response.status_code == 404:
                return None
            response.raise_for_status()
            return response.json()
        except Exception as e:
            self.logger.error(f"Error getting position for {symbol}: {e}")
            raise
    
    async def submit_order(self, order: AlpacaOrder) -> Dict[str, Any]:
        """Submit an order to Alpaca."""
        try:
            order_data = {
                'symbol': order.symbol,
                'qty': str(order.qty),
                'side': order.side.value,
                'type': order.type.value,
                'time_in_force': order.time_in_force.value,
                'extended_hours': order.extended_hours
            }
            
            # Add optional parameters
            if order.limit_price is not None:
                order_data['limit_price'] = str(order.limit_price)
            if order.stop_price is not None:
                order_data['stop_price'] = str(order.stop_price)
            if order.trail_price is not None:
                order_data['trail_price'] = str(order.trail_price)
            if order.trail_percent is not None:
                order_data['trail_percent'] = str(order.trail_percent)
            if order.client_order_id is not None:
                order_data['client_order_id'] = order.client_order_id
            
            response = await asyncio.to_thread(
                self.session.post, f"{self.base_url}/v2/orders", json=order_data
            )
            response.raise_for_status()
            
            result = response.json()
            self.logger.info(f"Order submitted successfully: {result['id']} - {order.symbol} {order.side.value} {order.qty}")
            return result
            
        except Exception as e:
            self.logger.error(f"Error submitting order: {e}")
            raise
    
    async def get_orders(self, status: str = "open", limit: int = 50) -> List[Dict[str, Any]]:
        """Get orders."""
        try:
            params = {'status': status, 'limit': limit}
            response = await asyncio.to_thread(
                self.session.get, f"{self.base_url}/v2/orders", params=params
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            self.logger.error(f"Error getting orders: {e}")
            raise
    
    async def get_order(self, order_id: str) -> Dict[str, Any]:
        """Get a specific order."""
        try:
            response = await asyncio.to_thread(
                self.session.get, f"{self.base_url}/v2/orders/{order_id}"
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            self.logger.error(f"Error getting order {order_id}: {e}")
            raise
    
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an order."""
        try:
            response = await asyncio.to_thread(
                self.session.delete, f"{self.base_url}/v2/orders/{order_id}"
            )
            response.raise_for_status()
            self.logger.info(f"Order {order_id} cancelled successfully")
            return True
        except Exception as e:
            self.logger.error(f"Error cancelling order {order_id}: {e}")
            return False
    
    async def cancel_all_orders(self) -> bool:
        """Cancel all open orders."""
        try:
            response = await asyncio.to_thread(
                self.session.delete, f"{self.base_url}/v2/orders"
            )
            response.raise_for_status()
            self.logger.info("All orders cancelled successfully")
            return True
        except Exception as e:
            self.logger.error(f"Error cancelling all orders: {e}")
            return False
    
    async def get_bars(self, symbol: str, timeframe: str = "1Day", 
                      start: Optional[datetime] = None, 
                      end: Optional[datetime] = None,
                      limit: int = 1000) -> pd.DataFrame:
        """Get historical bars data."""
        try:
            params = {
                'symbols': symbol,
                'timeframe': timeframe,
                'limit': limit
            }
            
            if start:
                params['start'] = start.isoformat()
            if end:
                params['end'] = end.isoformat()
            
            response = await asyncio.to_thread(
                self.session.get, f"{self.data_url}/v2/stocks/bars", params=params
            )
            response.raise_for_status()
            
            data = response.json()
            if symbol in data.get('bars', {}):
                bars = data['bars'][symbol]
                df = pd.DataFrame(bars)
                if not df.empty:
                    df['timestamp'] = pd.to_datetime(df['t'])
                    df = df.set_index('timestamp')
                    df = df.rename(columns={'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close', 'v': 'volume'})
                    return df[['open', 'high', 'low', 'close', 'volume']]
            
            return pd.DataFrame()
            
        except Exception as e:
            self.logger.error(f"Error getting bars for {symbol}: {e}")
            return pd.DataFrame()
    
    async def get_latest_quote(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get latest quote for a symbol."""
        try:
            params = {'symbols': symbol}
            response = await asyncio.to_thread(
                self.session.get, f"{self.data_url}/v2/stocks/quotes/latest", params=params
            )
            response.raise_for_status()
            
            data = response.json()
            if symbol in data.get('quotes', {}):
                return data['quotes'][symbol]
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting latest quote for {symbol}: {e}")
            return None
    
    async def is_market_open(self) -> bool:
        """Check if market is currently open."""
        try:
            response = await asyncio.to_thread(
                self.session.get, f"{self.base_url}/v2/clock"
            )
            response.raise_for_status()
            
            clock_data = response.json()
            return clock_data.get('is_open', False)
            
        except Exception as e:
            self.logger.error(f"Error checking market status: {e}")
            return False
    
    async def get_market_calendar(self, start: Optional[datetime] = None, 
                                 end: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """Get market calendar."""
        try:
            params = {}
            if start:
                params['start'] = start.strftime('%Y-%m-%d')
            if end:
                params['end'] = end.strftime('%Y-%m-%d')
            
            response = await asyncio.to_thread(
                self.session.get, f"{self.base_url}/v2/calendar", params=params
            )
            response.raise_for_status()
            return response.json()
            
        except Exception as e:
            self.logger.error(f"Error getting market calendar: {e}")
            return []
    
    async def test_connection(self) -> bool:
        """Test connection to Alpaca API."""
        try:
            account = await self.get_account()
            self.logger.info(f"Connection test successful - Account: {account.get('account_number', 'Unknown')}")
            return True
        except Exception as e:
            self.logger.error(f"Connection test failed: {e}")
            return False


if __name__ == "__main__":
    # Test the client
    async def test_client():
        client = AlpacaClient(paper_trading=True)
        
        # Test connection
        if await client.test_connection():
            print("✅ Connection successful")
            
            # Get account info
            account = await client.get_account()
            print(f"Account Balance: ${float(account['cash']):.2f}")
            
            # Check market status
            is_open = await client.is_market_open()
            print(f"Market Open: {is_open}")
            
            # Get positions
            positions = await client.get_positions()
            print(f"Positions: {len(positions)}")
        else:
            print("❌ Connection failed")
    
    # Commented out to prevent event loop conflicts when imported
    # asyncio.run(test_client())