#!/usr/bin/env python3
"""
Enhanced Data Manager

Provides robust data fetching capabilities.
Note: Crypto functionality and fallback mechanisms are currently disabled.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union, Tuple
from dataclasses import dataclass
import json

import pandas as pd
import numpy as np
# import aiohttp  # Commented out - using session manager instead
# import yfinance as yf  # Commented out - crypto functionality disabled

from .data_manager import UnifiedDataManager, DataRequest, DataResponse
from .config_manager import ConfigManager
from .session_manager import SessionManager


@dataclass
class DataSourceConfig:
    """Configuration for a data source"""
    name: str
    priority: int
    enabled: bool
    rate_limit: int
    timeout: int
    retry_attempts: int
    base_url: Optional[str] = None
    api_key_env: Optional[str] = None
    supported_assets: List[str] = None


class EnhancedDataManager:
    """Enhanced data manager - crypto functionality currently disabled"""
    
    def __init__(self, config_manager: ConfigManager):
        self.config_manager = config_manager
        self.logger = logging.getLogger(__name__)
        self.unified_manager = UnifiedDataManager(config_manager)
        
        # Load data source configurations
        self.data_sources = self._load_data_source_configs()
        
        # Initialize health tracking for sources
        self._initialize_health_tracking()

    def _load_data_source_configs(self) -> Dict[str, DataSourceConfig]:
        """Load configuration for all data sources"""
        
        # Default configurations - crypto sources commented out
        default_configs = {
            # 'binance': DataSourceConfig(
            #     name='binance',
            #     priority=1,
            #     enabled=True,
            #     rate_limit=1200,  # requests per minute
            #     timeout=30,
            #     retry_attempts=3,
            #     base_url='https://api.binance.com',
            #     supported_assets=['crypto']
            # ),
            # 'yfinance': DataSourceConfig(
            #     name='yfinance',
            #     priority=2,
            #     enabled=True,
            #     rate_limit=2000,
            #     timeout=30,
            #     retry_attempts=3,
            #     supported_assets=['stocks', 'crypto', 'forex']
            # )
        }
        
        return default_configs

    def _initialize_health_tracking(self):
        """Initialize health tracking for all data sources"""
        
        self.source_health = {}
        for source_name in self.data_sources:
            self.source_health[source_name] = {
                'is_healthy': True,
                'last_success': None,
                'last_failure': None,
                'consecutive_failures': 0,
                'total_requests': 0,
                'successful_requests': 0
            }

    # CRYPTO FUNCTIONALITY COMMENTED OUT
    # async def get_crypto_data_with_fallback(self, symbol: str, timeframe: str = '1d', 
    #                                       start_date: Optional[datetime] = None,
    #                                       end_date: Optional[datetime] = None) -> Optional[pd.DataFrame]:
    #     """
    #     Get cryptocurrency data with intelligent fallback between sources
    #     
    #     Priority order:
    #     1. Binance (native crypto format: BTCUSDT)
    #     2. Yahoo Finance (converted format: BTC-USD)
    #     """
    #     
    #     # Define fallback chain for crypto data
    #     fallback_chain = [
    #         ('binance', symbol),  # Use symbol as-is for Binance
    #         ('yfinance', self._convert_to_yahoo_format(symbol))  # Convert for Yahoo
    #     ]
    #     
    #     for source_name, converted_symbol in fallback_chain:
    #         if not self._is_source_available(source_name):
    #             continue
    #             
    #         try:
    #             self.logger.info(f"Attempting to fetch {symbol} data from {source_name}")
    #             
    #             if source_name == 'binance':
    #                 data = await self._fetch_binance_data(converted_symbol, timeframe, start_date, end_date)
    #             elif source_name == 'yfinance':
    #                 data = await self._fetch_yahoo_data(converted_symbol, timeframe, start_date, end_date)
    #             else:
    #                 continue
    #             
    #             if data is not None and not data.empty:
    #                 self._record_success(source_name)
    #                 self.logger.info(f"Successfully fetched {symbol} data from {source_name}")
    #                 return data
    #             else:
    #                 self._record_failure(source_name, "Empty data returned")
    #                 
    #         except Exception as e:
    #             self._record_failure(source_name, str(e))
    #             self.logger.warning(f"Failed to fetch {symbol} from {source_name}: {e}")
    #             continue
    #     
    #     self.logger.error(f"All data sources failed for symbol {symbol}")
    #     return None

    # async def _fetch_binance_data(self, symbol: str, timeframe: str, 
    #                             start_date: Optional[datetime], 
    #                             end_date: Optional[datetime]) -> Optional[pd.DataFrame]:
    #     """Fetch data from Binance API with managed session"""
    #     
    #     # Convert symbol format (e.g., BTC-USD -> BTCUSDT)
    #     binance_symbol = symbol.replace('-', '').upper()
    #     
    #     # Convert timeframe to Binance interval
    #     binance_interval = self._convert_timeframe_to_binance(timeframe)
    #     
    #     url = "https://api.binance.com/api/v3/klines"
    #     params = {
    #         'symbol': symbol,
    #         'interval': binance_interval,
    #         'limit': 1000
    #     }
    #     
    #     if start_date:
    #         params['startTime'] = int(start_date.timestamp() * 1000)
    #     if end_date:
    #         params['endTime'] = int(end_date.timestamp() * 1000)
    #     
    #     # Use managed session instead of creating new ClientSession
    #     session_manager = SessionManager()
    #     async with session_manager.get_session() as session:
    #         try:
    #             async with session.get(url, params=params, timeout=30) as response:
    #                 if response.status == 200:
    #                     data = await response.json()
    #                     return self._parse_binance_klines(data)
    #                 else:
    #                     raise Exception(f"Binance API error: {response.status}")
    #         except Exception as e:
    #             raise Exception(f"Binance request failed: {e}")

    # async def _fetch_yahoo_data(self, symbol: str, timeframe: str,
    #                           start_date: Optional[datetime],
    #                           end_date: Optional[datetime]) -> Optional[pd.DataFrame]:
    #     """Fetch data from Yahoo Finance"""
    #     
    #     try:
    #         ticker = yf.Ticker(symbol)
    #         
    #         if timeframe in ['1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max']:
    #             data = ticker.history(period=timeframe)
    #         else:
    #             # For custom date ranges
    #             data = ticker.history(
    #                 start=start_date or datetime.now() - timedelta(days=30),
    #                 end=end_date or datetime.now(),
    #                 interval=timeframe
    #             )
    #         
    #         if not data.empty:
    #             # Standardize column names
    #             data.columns = [col.lower().replace(' ', '_') for col in data.columns]
    #             return data
    #         else:
    #             return None
    #             
    #     except Exception as e:
    #         raise Exception(f"Yahoo Finance request failed: {e}")

    # def _convert_to_yahoo_format(self, symbol: str) -> str:
    #     """Convert Binance format symbol to Yahoo Finance format"""
    #     
    #     # Common crypto symbol conversions
    #     conversions = {
    #         'BTCUSDT': 'BTC-USD',
    #         'ETHUSDT': 'ETH-USD',
    #         'ADAUSDT': 'ADA-USD',
    #         'SOLUSDT': 'SOL-USD',
    #         'DOTUSDT': 'DOT-USD',
    #         'LINKUSDT': 'LINK-USD',
    #         'LTCUSDT': 'LTC-USD',
    #         'BCHUSDT': 'BCH-USD',
    #         'XRPUSDT': 'XRP-USD',
    #         'BNBUSDT': 'BNB-USD'
    #     }
    #     
    #     return conversions.get(symbol, symbol)

    # def _convert_timeframe_to_binance(self, timeframe: str) -> str:
    #     """Convert timeframe to Binance interval format"""
    #     
    #     conversions = {
    #         '1m': '1m',
    #         '5m': '5m',
    #         '15m': '15m',
    #         '30m': '30m',
    #         '1h': '1h',
    #         '4h': '4h',
    #         '1d': '1d',
    #         '1w': '1w',
    #         '1M': '1M'
    #     }
    #     
    #     return conversions.get(timeframe, '1d')

    # def _parse_binance_klines(self, klines_data: List) -> pd.DataFrame:
    #     """Parse Binance klines data into DataFrame"""
    #     
    #     if not klines_data:
    #         return pd.DataFrame()
    #     
    #     columns = [
    #         'open_time', 'open', 'high', 'low', 'close', 'volume',
    #         'close_time', 'quote_asset_volume', 'number_of_trades',
    #         'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
    #     ]
    #     
    #     df = pd.DataFrame(klines_data, columns=columns)
    #     
    #     # Convert timestamps
    #     df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
    #     df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
    #     
    #     # Convert price columns to float
    #     price_columns = ['open', 'high', 'low', 'close', 'volume']
    #     for col in price_columns:
    #         df[col] = df[col].astype(float)
    #     
    #     # Set index to open_time
    #     df.set_index('open_time', inplace=True)
    #     
    #     # Keep only essential columns
    #     return df[['open', 'high', 'low', 'close', 'volume']]

    def _is_source_available(self, source_name: str) -> bool:
        """Check if a data source is available and healthy"""
        
        if source_name not in self.data_sources:
            return False
        
        source_config = self.data_sources[source_name]
        if not source_config.enabled:
            return False
        
        health = self.source_health.get(source_name, {})
        
        # Consider source unhealthy if too many consecutive failures
        if health.get('consecutive_failures', 0) >= 5:
            return False
        
        return health.get('is_healthy', True)

    def _record_success(self, source_name: str):
        """Record successful request for a data source"""
        
        if source_name not in self.source_health:
            return
        
        health = self.source_health[source_name]
        health['is_healthy'] = True
        health['last_success'] = datetime.now()
        health['consecutive_failures'] = 0
        health['total_requests'] += 1
        health['successful_requests'] += 1

    def _record_failure(self, source_name: str, error_message: str):
        """Record failed request for a data source"""
        
        if source_name not in self.source_health:
            return
        
        health = self.source_health[source_name]
        health['last_failure'] = datetime.now()
        health['consecutive_failures'] += 1
        health['total_requests'] += 1
        
        # Mark as unhealthy after 3 consecutive failures
        if health['consecutive_failures'] >= 3:
            health['is_healthy'] = False
        
        self.logger.warning(f"Data source {source_name} failure #{health['consecutive_failures']}: {error_message}")

    def get_source_health_report(self) -> Dict[str, Any]:
        """Get health report for all data sources"""
        
        report = {}
        for source_name, health in self.source_health.items():
            success_rate = 0.0
            if health['total_requests'] > 0:
                success_rate = health['successful_requests'] / health['total_requests']
            
            report[source_name] = {
                'enabled': self.data_sources.get(source_name, {}).enabled if source_name in self.data_sources else False,
                'healthy': health['is_healthy'],
                'success_rate': success_rate,
                'consecutive_failures': health['consecutive_failures'],
                'last_success': health['last_success'].isoformat() if health['last_success'] else None,
                'last_failure': health['last_failure'].isoformat() if health['last_failure'] else None,
                'total_requests': health['total_requests']
            }
        
        return report

    # CRYPTO TESTING FUNCTIONALITY COMMENTED OUT
    # async def test_all_sources(self) -> Dict[str, bool]:
    #     """Test connectivity to all enabled data sources"""
    #     
    #     results = {}
    #     
    #     # Test crypto sources with a common symbol
    #     test_symbol = 'BTCUSDT'
    #     
    #     for source_name, config in self.data_sources.items():
    #         if not config.enabled:
    #             results[source_name] = False
    #             continue
    #         
    #         try:
    #             if source_name == 'binance':
    #                 data = await self._fetch_binance_data(test_symbol, '1d', None, None)
    #                 results[source_name] = data is not None and not data.empty
    #             elif source_name == 'yfinance':
    #                 yahoo_symbol = self._convert_to_yahoo_format(test_symbol)
    #                 data = await self._fetch_yahoo_data(yahoo_symbol, '1d', None, None)
    #                 results[source_name] = data is not None and not data.empty
    #             else:
    #                 results[source_name] = True  # Assume other sources are working
    #                 
    #         except Exception as e:
    #             self.logger.error(f"Test failed for {source_name}: {e}")
    #             results[source_name] = False
    #     
    #     return results


# CRYPTO CONVENIENCE FUNCTION COMMENTED OUT
# async def get_crypto_data_with_fallback(symbol: str, config_manager: ConfigManager, 
#                                       timeframe: str = '1d') -> Optional[pd.DataFrame]:
#     """Convenience function to get crypto data with fallback"""
#     
#     manager = EnhancedDataManager(config_manager)
#     return await manager.get_crypto_data_with_fallback(symbol, timeframe)


# CRYPTO TESTING MAIN FUNCTION COMMENTED OUT
# if __name__ == "__main__":
#     import asyncio
#     from pathlib import Path
#     
#     async def test_enhanced_manager():
#         config_manager = ConfigManager(Path("../config"))
#         manager = EnhancedDataManager(config_manager)
#         
#         # Test crypto data fetching
#         symbols = ['BTCUSDT', 'ETHUSDT', 'ADAUSDT']
#         
#         for symbol in symbols:
#             print(f"\nTesting {symbol}...")
#             data = await manager.get_crypto_data_with_fallback(symbol)
#             if data is not None:
#                 print(f"✅ Successfully fetched {len(data)} rows for {symbol}")
#                 print(data.head())
#             else:
#                 print(f"❌ Failed to fetch data for {symbol}")
#         
#         # Print health report
#         print("\n" + "="*50)
#         print("DATA SOURCE HEALTH REPORT")
#         print("="*50)
#         health_report = manager.get_source_health_report()
#         for source, health in health_report.items():
#             status = "✅ HEALTHY" if health['healthy'] else "❌ UNHEALTHY"
#             print(f"{source}: {status} (Success Rate: {health['success_rate']:.2%})")
#     
#     asyncio.run(test_enhanced_manager())