#!/usr/bin/env python3
"""
Market Data Tool for CrewAI Trading System

Provides agents with access to real-time and historical market data
from multiple sources with caching and error handling.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union

import pandas as pd
import numpy as np
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from core.data_manager import UnifiedDataManager, DataRequest


class MarketDataInput(BaseModel):
    """Input schema for market data requests."""
    symbol: str = Field(..., description="Stock symbol (e.g., 'AAPL', 'MSFT')")
    data_type: str = Field(
        default="price",
        description="Type of data: 'price', 'volume', 'ohlc', 'intraday'"
    )
    timeframe: str = Field(
        default="1d",
        description="Timeframe: '1m', '5m', '15m', '1h', '1d', '1w', '1mo'"
    )
    period: Optional[str] = Field(
        default="1mo",
        description="Period for historical data: '1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', 'max'"
    )
    start_date: Optional[str] = Field(
        default=None,
        description="Start date in YYYY-MM-DD format"
    )
    end_date: Optional[str] = Field(
        default=None,
        description="End date in YYYY-MM-DD format"
    )
    indicators: Optional[List[str]] = Field(
        default=None,
        description="Technical indicators to include: ['sma', 'ema', 'rsi', 'macd', 'bollinger']"
    )
    extended_hours: bool = Field(
        default=False,
        description="Include extended hours trading data"
    )


class MarketDataTool(BaseTool):
    """
    Market Data Tool for CrewAI agents.
    
    Provides comprehensive market data access including:
    - Real-time and historical price data
    - Volume and trading statistics
    - Technical indicators
    - Market hours and trading status
    - Multiple timeframes and periods
    """
    
    name: str = "market_data_tool"
    description: str = (
        "Get comprehensive market data for stocks including price, volume, "
        "and technical indicators. Supports multiple timeframes and periods. "
        "Use this tool to fetch current prices, historical data, and market statistics."
    )
    args_schema: type[MarketDataInput] = MarketDataInput
    data_manager: UnifiedDataManager = Field(default=None, exclude=True)
    logger: Any = Field(default=None, exclude=True)
    cache: Dict = Field(default_factory=dict, exclude=True)
    cache_ttl: int = Field(default=60, exclude=True)
    
    def __init__(self, data_manager: UnifiedDataManager, **kwargs):
        """
        Initialize the market data tool.
        
        Args:
            data_manager: Unified data manager instance
        """
        # Initialize with data_manager as a field
        super().__init__(data_manager=data_manager, **kwargs)
        self.logger = logging.getLogger(__name__)
        
        # Cache for frequently requested data
        self.cache = {}
        self.cache_ttl = 60  # seconds
    
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
                # If loop is already running, create a task
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, self._arun(**kwargs))
                    return future.result()
            else:
                return loop.run_until_complete(self._arun(**kwargs))
                
        except Exception as e:
            self.logger.error(f"Error in market data tool: {e}")
            return f"Error fetching market data: {str(e)}"
    
    async def _arun(self, **kwargs) -> str:
        """Asynchronous execution of market data retrieval."""
        try:
            # Parse input
            input_data = MarketDataInput(**kwargs)
            
            # Validate symbol
            if not input_data.symbol or len(input_data.symbol) < 1:
                return "Error: Invalid or missing symbol"
            
            # Clean symbol (remove $ prefix if present)
            symbol = input_data.symbol.upper().replace('$', '')
            
            # Parse dates if provided
            start_date = None
            end_date = None
            
            if input_data.start_date:
                try:
                    start_date = datetime.strptime(input_data.start_date, '%Y-%m-%d')
                except ValueError:
                    return f"Error: Invalid start_date format. Use YYYY-MM-DD"
            
            if input_data.end_date:
                try:
                    end_date = datetime.strptime(input_data.end_date, '%Y-%m-%d')
                except ValueError:
                    return f"Error: Invalid end_date format. Use YYYY-MM-DD"
            
            # If no specific dates provided, use period
            if not start_date and not end_date and input_data.period:
                end_date = datetime.now()
                if input_data.period == '1d':
                    start_date = end_date - timedelta(days=1)
                elif input_data.period == '5d':
                    start_date = end_date - timedelta(days=5)
                elif input_data.period == '1mo':
                    start_date = end_date - timedelta(days=30)
                elif input_data.period == '3mo':
                    start_date = end_date - timedelta(days=90)
                elif input_data.period == '6mo':
                    start_date = end_date - timedelta(days=180)
                elif input_data.period == '1y':
                    start_date = end_date - timedelta(days=365)
                elif input_data.period == '2y':
                    start_date = end_date - timedelta(days=730)
                elif input_data.period == '5y':
                    start_date = end_date - timedelta(days=1825)
            
            # Create data request
            request = DataRequest(
                symbol=symbol,
                data_type=input_data.data_type,
                timeframe=input_data.timeframe,
                start_date=start_date,
                end_date=end_date
            )
            
            # Get data from data manager
            response = await self.data_manager.get_data(request)
            
            if response.error:
                return f"Error fetching data for {symbol}: {response.error}"
            
            if response.data is None or (isinstance(response.data, pd.DataFrame) and response.data.empty):
                return f"No data available for {symbol} with the specified parameters"
            
            # Process and format the data
            result = self._format_market_data(response.data, input_data, symbol)
            
            # Add technical indicators if requested
            if input_data.indicators and isinstance(response.data, pd.DataFrame):
                indicators_result = self._calculate_indicators(response.data, input_data.indicators)
                if indicators_result:
                    result += "\n\nTechnical Indicators:\n" + indicators_result
            
            # Add metadata
            metadata = f"\n\nData Source: {response.source}"
            metadata += f"\nTimestamp: {response.timestamp.strftime('%Y-%m-%d %H:%M:%S')}"
            metadata += f"\nCached: {'Yes' if response.cached else 'No'}"
            
            return result + metadata
            
        except Exception as e:
            self.logger.error(f"Error in async market data retrieval: {e}")
            return f"Error processing market data request: {str(e)}"
    
    def _format_market_data(self, data: Union[pd.DataFrame, Dict, List], 
                          input_data: MarketDataInput, symbol: str) -> str:
        """Format market data for agent consumption."""
        try:
            if isinstance(data, pd.DataFrame):
                return self._format_dataframe(data, input_data, symbol)
            elif isinstance(data, dict):
                return self._format_dict(data, symbol)
            elif isinstance(data, list):
                return self._format_list(data, symbol)
            else:
                return f"Data for {symbol}: {str(data)}"
        except Exception as e:
            self.logger.error(f"Error formatting market data: {e}")
            return f"Error formatting data for {symbol}: {str(e)}"
    
    def _format_dataframe(self, df: pd.DataFrame, input_data: MarketDataInput, symbol: str) -> str:
        """Format DataFrame market data."""
        if df.empty:
            return f"No data available for {symbol}"
        
        result = f"Market Data for {symbol}:\n"
        result += f"Timeframe: {input_data.timeframe}\n"
        result += f"Data Points: {len(df)}\n\n"
        
        # Current/Latest data
        if not df.empty:
            latest = df.iloc[-1]
            result += "Latest Data:\n"
            
            if 'Close' in df.columns or 'close' in df.columns:
                close_col = 'Close' if 'Close' in df.columns else 'close'
                current_price = latest[close_col]
                result += f"Current Price: ${current_price:.2f}\n"
                
                # Price change if we have enough data
                if len(df) > 1:
                    prev_close = df.iloc[-2][close_col]
                    change = current_price - prev_close
                    change_pct = (change / prev_close) * 100
                    result += f"Change: ${change:.2f} ({change_pct:+.2f}%)\n"
            
            # Volume
            if 'Volume' in df.columns or 'volume' in df.columns:
                vol_col = 'Volume' if 'Volume' in df.columns else 'volume'
                volume = latest[vol_col]
                result += f"Volume: {volume:,.0f}\n"
            
            # OHLC data
            ohlc_cols = ['Open', 'High', 'Low', 'Close']
            ohlc_cols_lower = ['open', 'high', 'low', 'close']
            
            available_ohlc = []
            for col in ohlc_cols:
                if col in df.columns:
                    available_ohlc.append(col)
            
            if not available_ohlc:
                for col in ohlc_cols_lower:
                    if col in df.columns:
                        available_ohlc.append(col)
            
            if available_ohlc:
                result += "\nOHLC Data:\n"
                for col in available_ohlc:
                    if col in df.columns:
                        value = latest[col]
                        result += f"{col}: ${value:.2f}\n"
        
        # Summary statistics for longer periods
        if len(df) > 5:
            result += "\nSummary Statistics:\n"
            
            if 'Close' in df.columns or 'close' in df.columns:
                close_col = 'Close' if 'Close' in df.columns else 'close'
                prices = df[close_col]
                
                result += f"Period High: ${prices.max():.2f}\n"
                result += f"Period Low: ${prices.min():.2f}\n"
                result += f"Average Price: ${prices.mean():.2f}\n"
                result += f"Volatility (Std Dev): ${prices.std():.2f}\n"
                
                # Calculate returns
                returns = prices.pct_change().dropna()
                if not returns.empty:
                    total_return = ((prices.iloc[-1] / prices.iloc[0]) - 1) * 100
                    result += f"Total Return: {total_return:+.2f}%\n"
                    result += f"Daily Volatility: {returns.std() * 100:.2f}%\n"
        
        # Recent data points (last 5)
        if len(df) > 1:
            result += "\nRecent Data (Last 5 periods):\n"
            recent_df = df.tail(5)
            
            # Format recent data
            for idx, row in recent_df.iterrows():
                date_str = idx.strftime('%Y-%m-%d %H:%M') if hasattr(idx, 'strftime') else str(idx)
                
                if 'Close' in row or 'close' in row:
                    close_val = row.get('Close', row.get('close', 'N/A'))
                    vol_val = row.get('Volume', row.get('volume', 'N/A'))
                    result += f"{date_str}: ${close_val:.2f}"
                    if vol_val != 'N/A':
                        result += f" (Vol: {vol_val:,.0f})"
                    result += "\n"
        
        return result
    
    def _format_dict(self, data: Dict, symbol: str) -> str:
        """Format dictionary market data."""
        result = f"Market Data for {symbol}:\n\n"
        
        # Handle nested dictionaries (like fundamentals)
        for key, value in data.items():
            if isinstance(value, dict):
                result += f"{key}:\n"
                for sub_key, sub_value in value.items():
                    result += f"  {sub_key}: {sub_value}\n"
            else:
                result += f"{key}: {value}\n"
        
        return result
    
    def _format_list(self, data: List, symbol: str) -> str:
        """Format list market data (like news)."""
        result = f"Market Data for {symbol}:\n\n"
        
        for i, item in enumerate(data[:10]):  # Limit to first 10 items
            result += f"Item {i+1}:\n"
            if isinstance(item, dict):
                for key, value in item.items():
                    result += f"  {key}: {value}\n"
            else:
                result += f"  {item}\n"
            result += "\n"
        
        if len(data) > 10:
            result += f"... and {len(data) - 10} more items\n"
        
        return result
    
    def _calculate_indicators(self, df: pd.DataFrame, indicators: List[str]) -> str:
        """Calculate technical indicators."""
        try:
            if df.empty or 'Close' not in df.columns:
                return "Cannot calculate indicators: No price data available"
            
            result = ""
            prices = df['Close']
            
            for indicator in indicators:
                indicator = indicator.lower()
                
                if indicator == 'sma':
                    # Simple Moving Average (20 period)
                    sma = prices.rolling(window=20).mean()
                    if not sma.empty:
                        result += f"SMA(20): ${sma.iloc[-1]:.2f}\n"
                
                elif indicator == 'ema':
                    # Exponential Moving Average (20 period)
                    ema = prices.ewm(span=20).mean()
                    if not ema.empty:
                        result += f"EMA(20): ${ema.iloc[-1]:.2f}\n"
                
                elif indicator == 'rsi':
                    # Relative Strength Index
                    rsi = self._calculate_rsi(prices)
                    if rsi is not None:
                        result += f"RSI(14): {rsi:.2f}\n"
                
                elif indicator == 'macd':
                    # MACD
                    macd_line, signal_line = self._calculate_macd(prices)
                    if macd_line is not None and signal_line is not None:
                        result += f"MACD: {macd_line:.4f}\n"
                        result += f"MACD Signal: {signal_line:.4f}\n"
                
                elif indicator == 'bollinger':
                    # Bollinger Bands
                    upper, middle, lower = self._calculate_bollinger_bands(prices)
                    if upper is not None:
                        result += f"Bollinger Upper: ${upper:.2f}\n"
                        result += f"Bollinger Middle: ${middle:.2f}\n"
                        result += f"Bollinger Lower: ${lower:.2f}\n"
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error calculating indicators: {e}")
            return f"Error calculating indicators: {str(e)}"
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> Optional[float]:
        """Calculate RSI."""
        try:
            if len(prices) < period + 1:
                return None
            
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            return rsi.iloc[-1] if not rsi.empty else None
            
        except Exception:
            return None
    
    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple:
        """Calculate MACD."""
        try:
            if len(prices) < slow + signal:
                return None, None
            
            ema_fast = prices.ewm(span=fast).mean()
            ema_slow = prices.ewm(span=slow).mean()
            
            macd_line = ema_fast - ema_slow
            signal_line = macd_line.ewm(span=signal).mean()
            
            return macd_line.iloc[-1], signal_line.iloc[-1]
            
        except Exception:
            return None, None
    
    def _calculate_bollinger_bands(self, prices: pd.Series, period: int = 20, std_dev: int = 2) -> tuple:
        """Calculate Bollinger Bands."""
        try:
            if len(prices) < period:
                return None, None, None
            
            middle = prices.rolling(window=period).mean()
            std = prices.rolling(window=period).std()
            
            upper = middle + (std * std_dev)
            lower = middle - (std * std_dev)
            
            return upper.iloc[-1], middle.iloc[-1], lower.iloc[-1]
            
        except Exception:
            return None, None, None
    
    def get_market_status(self) -> str:
        """Get current market status."""
        try:
            now = datetime.now()
            
            # Simple market hours check (US Eastern Time)
            # This is a simplified version - in production, you'd want to handle
            # holidays, different exchanges, etc.
            
            weekday = now.weekday()  # 0 = Monday, 6 = Sunday
            hour = now.hour
            
            if weekday >= 5:  # Weekend
                return "Market is closed (Weekend)"
            elif hour < 9 or hour >= 16:  # Outside 9 AM - 4 PM ET
                return "Market is closed (Outside trading hours)"
            else:
                return "Market is open"
                
        except Exception as e:
            return f"Unable to determine market status: {str(e)}"
    
    def get_tool_info(self) -> Dict[str, Any]:
        """Get tool information and capabilities."""
        return {
            'name': self.name,
            'description': self.description,
            'supported_data_types': ['price', 'volume', 'ohlc', 'intraday'],
            'supported_timeframes': ['1m', '5m', '15m', '1h', '1d', '1w', '1mo'],
            'supported_periods': ['1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', 'max'],
            'supported_indicators': ['sma', 'ema', 'rsi', 'macd', 'bollinger'],
            'data_sources': self.data_manager.get_available_sources() if self.data_manager else [],
            'cache_ttl': self.cache_ttl
        }


if __name__ == "__main__":
    # Example usage
    import asyncio
    from pathlib import Path
    from core.config_manager import ConfigManager
    from core.data_manager import UnifiedDataManager
    
    async def test_market_data_tool():
        config_manager = ConfigManager(Path("../config"))
        data_manager = UnifiedDataManager(config_manager)
        
        tool = MarketDataTool(data_manager)
        
        # Test basic price data
        result = tool._run(
            symbol="AAPL",
            data_type="price",
            timeframe="1d",
            period="1mo",
            indicators=["sma", "rsi"]
        )
        
        print("Market Data Result:")
        print(result)
        
        # Test tool info
        info = tool.get_tool_info()
        print("\nTool Info:")
        print(info)
    
    # Run test
    # asyncio.run(test_market_data_tool())