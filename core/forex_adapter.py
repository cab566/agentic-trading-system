#!/usr/bin/env python3
"""
Forex Data Adapter for CrewAI Trading System

Provides unified access to forex market data with 24/5 trading support,
real-time currency rates, and comprehensive forex analytics.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union
import os

import pandas as pd
import numpy as np
import requests
import json
from dataclasses import dataclass

from .data_types import DataSourceAdapter, DataRequest, DataResponse


class OandaAdapter(DataSourceAdapter):
    """OANDA forex broker adapter for live forex data."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__('oanda', config)
        self.api_key = os.getenv(config.get('api_key_env', 'OANDA_API_KEY'))
        self.account_id = os.getenv(config.get('account_id_env', 'OANDA_ACCOUNT_ID'))
        self.practice = config.get('practice', True)
        
        if self.practice:
            self.base_url = 'https://api-fxpractice.oanda.com'
        else:
            self.base_url = 'https://api-fxtrade.oanda.com'
        
        self.headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }
    
    async def fetch_data(self, request: DataRequest) -> DataResponse:
        """Fetch forex data from OANDA."""
        try:
            if self.is_rate_limited():
                await asyncio.sleep(0.1)
            
            self.record_request()
            
            if request.data_type == 'price':
                data = await self._fetch_candles(request)
            elif request.data_type == 'ticker':
                data = await self._fetch_pricing(request)
            elif request.data_type == 'spread':
                data = await self._fetch_spread(request)
            elif request.data_type == 'orderbook':
                data = await self._fetch_orderbook(request)
            else:
                raise ValueError(f"Unsupported data type: {request.data_type}")
            
            return DataResponse(
                data=data,
                source=self.name,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            self.logger.error(f"Error fetching data from OANDA: {e}")
            return DataResponse(
                data=pd.DataFrame(),
                source=self.name,
                timestamp=datetime.now(),
                error=str(e)
            )
    
    async def _fetch_candles(self, request: DataRequest) -> pd.DataFrame:
        """Fetch OHLC candle data."""
        instrument = self._format_instrument(request.symbol)
        granularity = self._convert_timeframe(request.timeframe)
        
        params = {
            'granularity': granularity,
            'count': request.parameters.get('count', 500) if request.parameters else 500
        }
        
        if request.start_date:
            params['from'] = request.start_date.isoformat() + 'Z'
        if request.end_date:
            params['to'] = request.end_date.isoformat() + 'Z'
        
        response = requests.get(
            f"{self.base_url}/v3/instruments/{instrument}/candles",
            headers=self.headers,
            params=params
        )
        response.raise_for_status()
        
        data = response.json()
        candles = data.get('candles', [])
        
        if not candles:
            return pd.DataFrame()
        
        rows = []
        for candle in candles:
            if candle['complete']:
                mid = candle['mid']
                rows.append({
                    'timestamp': pd.to_datetime(candle['time']),
                    'open': float(mid['o']),
                    'high': float(mid['h']),
                    'low': float(mid['l']),
                    'close': float(mid['c']),
                    'volume': int(candle['volume'])
                })
        
        df = pd.DataFrame(rows)
        if not df.empty:
            df.set_index('timestamp', inplace=True)
        
        return df
    
    async def _fetch_pricing(self, request: DataRequest) -> Dict[str, Any]:
        """Fetch current pricing information."""
        instrument = self._format_instrument(request.symbol)
        
        response = requests.get(
            f"{self.base_url}/v3/accounts/{self.account_id}/pricing",
            headers=self.headers,
            params={'instruments': instrument}
        )
        response.raise_for_status()
        
        data = response.json()
        prices = data.get('prices', [])
        
        return prices[0] if prices else {}
    
    async def _fetch_spread(self, request: DataRequest) -> Dict[str, Any]:
        """Fetch bid-ask spread information."""
        pricing = await self._fetch_pricing(request)
        
        if pricing:
            bid = float(pricing.get('bids', [{}])[0].get('price', 0))
            ask = float(pricing.get('asks', [{}])[0].get('price', 0))
            spread = ask - bid
            spread_pips = spread * (10000 if 'JPY' not in request.symbol else 100)
            
            return {
                'bid': bid,
                'ask': ask,
                'spread': spread,
                'spread_pips': spread_pips,
                'timestamp': pricing.get('time')
            }
        
        return {}
    
    async def _fetch_orderbook(self, request: DataRequest) -> Dict[str, Any]:
        """Fetch order book data."""
        instrument = self._format_instrument(request.symbol)
        
        response = requests.get(
            f"{self.base_url}/v3/instruments/{instrument}/orderBook",
            headers=self.headers
        )
        response.raise_for_status()
        
        return response.json()
    
    def _format_instrument(self, symbol: str) -> str:
        """Format symbol for OANDA (e.g., EURUSD -> EUR_USD)."""
        if '_' in symbol:
            return symbol.upper()
        
        # Common forex pairs
        if len(symbol) == 6:
            return f"{symbol[:3]}_{symbol[3:]}".upper()
        
        return symbol.upper()
    
    def _convert_timeframe(self, timeframe: str) -> str:
        """Convert timeframe to OANDA granularity."""
        mapping = {
            '5s': 'S5', '10s': 'S10', '15s': 'S15', '30s': 'S30',
            '1m': 'M1', '2m': 'M2', '3m': 'M3', '4m': 'M4', '5m': 'M5',
            '10m': 'M10', '15m': 'M15', '30m': 'M30',
            '1h': 'H1', '2h': 'H2', '3h': 'H3', '4h': 'H4', '6h': 'H6',
            '8h': 'H8', '12h': 'H12', '1d': 'D', '1w': 'W', '1M': 'M'
        }
        return mapping.get(timeframe, 'H1')


class ForexFactoryAdapter(DataSourceAdapter):
    """Forex Factory news and economic calendar adapter."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__('forex_factory', config)
        self.base_url = 'https://nfs.faireconomy.media/ff_calendar_thisweek.json'
    
    async def fetch_data(self, request: DataRequest) -> DataResponse:
        """Fetch forex news and economic events."""
        try:
            if self.is_rate_limited():
                await asyncio.sleep(1.0)  # Be respectful to free service
            
            self.record_request()
            
            if request.data_type == 'news':
                data = await self._fetch_economic_calendar()
            else:
                raise ValueError(f"Unsupported data type: {request.data_type}")
            
            return DataResponse(
                data=data,
                source=self.name,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            self.logger.error(f"Error fetching data from Forex Factory: {e}")
            return DataResponse(
                data=[],
                source=self.name,
                timestamp=datetime.now(),
                error=str(e)
            )
    
    async def _fetch_economic_calendar(self) -> List[Dict[str, Any]]:
        """Fetch economic calendar events."""
        response = requests.get(self.base_url)
        response.raise_for_status()
        
        events = response.json()
        
        # Filter and format events
        formatted_events = []
        for event in events:
            if event.get('impact') in ['High', 'Medium']:  # Focus on important events
                formatted_events.append({
                    'title': event.get('title', ''),
                    'country': event.get('country', ''),
                    'date': event.get('date', ''),
                    'time': event.get('time', ''),
                    'impact': event.get('impact', ''),
                    'forecast': event.get('forecast', ''),
                    'previous': event.get('previous', ''),
                    'actual': event.get('actual', '')
                })
        
        return formatted_events


class AlphaVantageForexAdapter(DataSourceAdapter):
    """Alpha Vantage forex data adapter (free tier available)."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__('alphavantage_forex', config)
        self.api_key = os.getenv(config.get('api_key_env', 'ALPHA_VANTAGE_API_KEY'))
        self.base_url = 'https://www.alphavantage.co/query'
    
    async def fetch_data(self, request: DataRequest) -> DataResponse:
        """Fetch forex data from Alpha Vantage."""
        try:
            if self.is_rate_limited():
                await asyncio.sleep(12)  # Free tier: 5 calls per minute
            
            self.record_request()
            
            if request.data_type == 'price':
                data = await self._fetch_fx_daily(request)
            elif request.data_type == 'intraday':
                data = await self._fetch_fx_intraday(request)
            elif request.data_type == 'exchange_rate':
                data = await self._fetch_exchange_rate(request)
            else:
                raise ValueError(f"Unsupported data type: {request.data_type}")
            
            return DataResponse(
                data=data,
                source=self.name,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            self.logger.error(f"Error fetching data from Alpha Vantage: {e}")
            return DataResponse(
                data=pd.DataFrame(),
                source=self.name,
                timestamp=datetime.now(),
                error=str(e)
            )
    
    async def _fetch_fx_daily(self, request: DataRequest) -> pd.DataFrame:
        """Fetch daily forex data."""
        from_symbol, to_symbol = self._parse_symbol(request.symbol)
        
        params = {
            'function': 'FX_DAILY',
            'from_symbol': from_symbol,
            'to_symbol': to_symbol,
            'apikey': self.api_key
        }
        
        response = requests.get(self.base_url, params=params)
        response.raise_for_status()
        
        data = response.json()
        time_series = data.get('Time Series (Daily)', {})
        
        if not time_series:
            return pd.DataFrame()
        
        df = pd.DataFrame.from_dict(time_series, orient='index')
        df.index = pd.to_datetime(df.index)
        df.columns = ['open', 'high', 'low', 'close']
        df = df.astype(float)
        df = df.sort_index()
        
        return df
    
    async def _fetch_fx_intraday(self, request: DataRequest) -> pd.DataFrame:
        """Fetch intraday forex data."""
        from_symbol, to_symbol = self._parse_symbol(request.symbol)
        interval = self._convert_timeframe(request.timeframe)
        
        params = {
            'function': 'FX_INTRADAY',
            'from_symbol': from_symbol,
            'to_symbol': to_symbol,
            'interval': interval,
            'apikey': self.api_key
        }
        
        response = requests.get(self.base_url, params=params)
        response.raise_for_status()
        
        data = response.json()
        time_series_key = f'Time Series ({interval})'
        time_series = data.get(time_series_key, {})
        
        if not time_series:
            return pd.DataFrame()
        
        df = pd.DataFrame.from_dict(time_series, orient='index')
        df.index = pd.to_datetime(df.index)
        df.columns = ['open', 'high', 'low', 'close']
        df = df.astype(float)
        df = df.sort_index()
        
        return df
    
    async def _fetch_exchange_rate(self, request: DataRequest) -> Dict[str, Any]:
        """Fetch real-time exchange rate."""
        from_symbol, to_symbol = self._parse_symbol(request.symbol)
        
        params = {
            'function': 'CURRENCY_EXCHANGE_RATE',
            'from_currency': from_symbol,
            'to_currency': to_symbol,
            'apikey': self.api_key
        }
        
        response = requests.get(self.base_url, params=params)
        response.raise_for_status()
        
        data = response.json()
        return data.get('Realtime Currency Exchange Rate', {})
    
    def _parse_symbol(self, symbol: str) -> tuple:
        """Parse forex symbol into from/to currencies."""
        if len(symbol) == 6:
            return symbol[:3].upper(), symbol[3:].upper()
        elif '/' in symbol:
            parts = symbol.split('/')
            return parts[0].upper(), parts[1].upper()
        else:
            raise ValueError(f"Invalid forex symbol format: {symbol}")
    
    def _convert_timeframe(self, timeframe: str) -> str:
        """Convert timeframe to Alpha Vantage format."""
        mapping = {
            '1m': '1min', '5m': '5min', '15m': '15min', '30m': '30min', '1h': '60min'
        }
        return mapping.get(timeframe, '60min')


class ForexDataManager:
    """Unified forex data manager."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.adapters = {}
        self.logger = logging.getLogger(__name__)
        
        # Initialize adapters
        if config.get('oanda', {}).get('enabled', False):
            self.adapters['oanda'] = OandaAdapter(config['oanda'])
        
        if config.get('forex_factory', {}).get('enabled', False):
            self.adapters['forex_factory'] = ForexFactoryAdapter(config['forex_factory'])
        
        if config.get('alphavantage', {}).get('enabled', False):
            self.adapters['alphavantage'] = AlphaVantageForexAdapter(config['alphavantage'])
    
    async def initialize(self):
        """Initialize the forex data manager and its adapters."""
        self.logger.info("Initializing ForexDataManager...")
        
        # Initialize each adapter if needed
        for adapter_name, adapter in self.adapters.items():
            if hasattr(adapter, 'initialize'):
                try:
                    await adapter.initialize()
                    self.logger.info(f"Initialized {adapter_name} adapter")
                except Exception as e:
                    self.logger.error(f"Failed to initialize {adapter_name} adapter: {e}")
        
        self.logger.info("ForexDataManager initialization complete")
    
    async def is_market_open(self) -> bool:
        """Check if forex market is currently open (alias for is_forex_market_open)."""
        return self.is_forex_market_open()
    
    async def get_forex_data(self, symbol: str, data_type: str = 'price', 
                           timeframe: str = '1h', **kwargs) -> DataResponse:
        """Get forex data from the best available source."""
        request = DataRequest(
            symbol=symbol,
            data_type=data_type,
            timeframe=timeframe,
            **kwargs
        )
        
        # Try adapters in priority order
        for adapter_name in ['oanda', 'alphavantage', 'forex_factory']:
            if adapter_name in self.adapters:
                try:
                    response = await self.adapters[adapter_name].fetch_data(request)
                    if not response.error:
                        return response
                except Exception as e:
                    self.logger.warning(f"Adapter {adapter_name} failed: {e}")
        
        # All adapters failed
        return DataResponse(
            data=pd.DataFrame(),
            source='failed',
            timestamp=datetime.now(),
            error="All forex adapters failed"
        )
    
    def get_major_pairs(self) -> List[str]:
        """Get list of major forex pairs."""
        return [
            'EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'AUDUSD', 'USDCAD', 'NZDUSD',
            'EURJPY', 'GBPJPY', 'EURGBP', 'AUDJPY', 'EURAUD', 'EURCHF', 'AUDNZD',
            'NZDJPY', 'GBPAUD', 'GBPCAD', 'EURNZD', 'AUDCAD', 'GBPCHF', 'AUDCHF'
        ]
    
    def is_forex_market_open(self) -> bool:
        """Check if forex market is currently open (24/5)."""
        now = datetime.now()
        weekday = now.weekday()  # 0=Monday, 6=Sunday
        
        # Forex market is closed on weekends
        if weekday == 5:  # Saturday
            return False
        elif weekday == 6:  # Sunday
            # Opens Sunday 5 PM EST (22:00 UTC)
            return now.hour >= 22
        else:
            # Monday-Friday: always open except Friday after 5 PM EST
            if weekday == 4 and now.hour >= 17:  # Friday after 5 PM EST
                return False
            return True
    
    def get_trading_sessions(self) -> Dict[str, Dict[str, int]]:
        """Get forex trading session times (UTC)."""
        return {
            'sydney': {'open': 22, 'close': 7},    # 10 PM - 7 AM UTC
            'tokyo': {'open': 0, 'close': 9},      # 12 AM - 9 AM UTC
            'london': {'open': 8, 'close': 17},    # 8 AM - 5 PM UTC
            'new_york': {'open': 13, 'close': 22}  # 1 PM - 10 PM UTC
        }
    
    def get_active_sessions(self) -> List[str]:
        """Get currently active trading sessions."""
        now = datetime.now()
        current_hour = now.hour
        sessions = self.get_trading_sessions()
        active = []
        
        for session, times in sessions.items():
            if times['open'] <= current_hour < times['close']:
                active.append(session)
        
        return active