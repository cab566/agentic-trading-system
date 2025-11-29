#!/usr/bin/env python3
"""
Technical Analysis Tool for CrewAI Trading System

Provides agents with comprehensive technical analysis capabilities
including indicators, patterns, and trading signals.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union, Tuple

import pandas as pd
import numpy as np
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

from core.data_manager import UnifiedDataManager, DataRequest


class TechnicalAnalysisInput(BaseModel):
    """Input schema for technical analysis requests."""
    symbol: str = Field(..., description="Stock symbol (e.g., 'AAPL', 'MSFT')")
    analysis_type: str = Field(
        default="comprehensive",
        description="Type of analysis: 'indicators', 'patterns', 'signals', 'comprehensive', 'momentum', 'trend', 'volatility'"
    )
    timeframe: str = Field(
        default="1d",
        description="Timeframe: '1m', '5m', '15m', '1h', '1d', '1w', '1mo'"
    )
    period: str = Field(
        default="3mo",
        description="Period for analysis: '1mo', '3mo', '6mo', '1y', '2y'"
    )
    indicators: Optional[List[str]] = Field(
        default=None,
        description="Specific indicators: ['sma', 'ema', 'rsi', 'macd', 'bollinger', 'stochastic', 'williams_r', 'cci', 'atr', 'obv']"
    )
    pattern_detection: bool = Field(
        default=True,
        description="Enable pattern detection (support/resistance, trends, reversals)"
    )
    signal_generation: bool = Field(
        default=True,
        description="Generate trading signals based on analysis"
    )
    confidence_threshold: float = Field(
        default=0.6,
        description="Minimum confidence threshold for signals (0.0 to 1.0)"
    )


class TechnicalAnalysisTool(BaseTool):
    """
    Technical Analysis Tool for CrewAI agents.
    
    Provides comprehensive technical analysis including:
    - Technical indicators (momentum, trend, volatility)
    - Pattern recognition (support/resistance, chart patterns)
    - Trading signal generation
    - Market structure analysis
    - Risk assessment metrics
    """
    
    name: str = "technical_analysis_tool"
    description: str = (
        "Perform comprehensive technical analysis on stocks including indicators, "
        "pattern recognition, and trading signal generation. Use this tool to analyze "
        "price trends, momentum, volatility, and generate actionable trading insights."
    )
    args_schema: type[TechnicalAnalysisInput] = TechnicalAnalysisInput
    
    # Pydantic v2 field declarations
    data_manager: UnifiedDataManager = Field(exclude=True)
    logger: logging.Logger = Field(default_factory=lambda: logging.getLogger(__name__), exclude=True)
    analysis_cache: Dict[str, Any] = Field(default_factory=dict, exclude=True)
    cache_ttl: int = Field(default=300, exclude=True)
    
    def __init__(self, data_manager: UnifiedDataManager, **kwargs):
        """
        Initialize the technical analysis tool.
        
        Args:
            data_manager: Unified data manager instance
        """
        # Initialize with data_manager as a field
        super().__init__(data_manager=data_manager, **kwargs)
        self.logger = logging.getLogger(__name__)
        
        # Analysis cache
        self.analysis_cache = {}
        self.cache_ttl = 300  # 5 minutes
    
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
            self.logger.error(f"Error in technical analysis tool: {e}")
            return f"Error performing technical analysis: {str(e)}"
    
    async def _arun(self, **kwargs) -> str:
        """Asynchronous execution of technical analysis."""
        try:
            # Parse input
            input_data = TechnicalAnalysisInput(**kwargs)
            
            # Validate symbol
            if not input_data.symbol or len(input_data.symbol) < 1:
                return "Error: Invalid or missing symbol"
            
            symbol = input_data.symbol.upper().replace('$', '')
            
            # Get market data
            market_data = await self._get_market_data(symbol, input_data)
            if isinstance(market_data, str):  # Error message
                return market_data
            
            # Perform analysis based on type
            analysis_result = await self._perform_analysis(market_data, input_data, symbol)
            
            return analysis_result
            
        except Exception as e:
            self.logger.error(f"Error in async technical analysis: {e}")
            return f"Error processing technical analysis request: {str(e)}"
    
    async def _get_market_data(self, symbol: str, input_data: TechnicalAnalysisInput) -> Union[pd.DataFrame, str]:
        """Get market data for analysis."""
        try:
            # Calculate date range
            end_date = datetime.now()
            
            if input_data.period == '1mo':
                start_date = end_date - timedelta(days=30)
            elif input_data.period == '3mo':
                start_date = end_date - timedelta(days=90)
            elif input_data.period == '6mo':
                start_date = end_date - timedelta(days=180)
            elif input_data.period == '1y':
                start_date = end_date - timedelta(days=365)
            elif input_data.period == '2y':
                start_date = end_date - timedelta(days=730)
            else:
                start_date = end_date - timedelta(days=90)  # Default to 3 months
            
            # Create data request
            request = DataRequest(
                symbol=symbol,
                data_type="ohlc",
                timeframe=input_data.timeframe,
                start_date=start_date,
                end_date=end_date
            )
            
            # Get data
            response = await self.data_manager.get_data(request)
            
            if response.error:
                return f"Error fetching data for {symbol}: {response.error}"
            
            if response.data is None or response.data.empty:
                return f"No data available for {symbol} with the specified parameters"
            
            return response.data
            
        except Exception as e:
            self.logger.error(f"Error getting market data: {e}")
            return f"Error retrieving market data: {str(e)}"
    
    async def _perform_analysis(self, df: pd.DataFrame, input_data: TechnicalAnalysisInput, symbol: str) -> str:
        """Perform the requested technical analysis."""
        try:
            result = f"Technical Analysis for {symbol}\n"
            result += f"Timeframe: {input_data.timeframe} | Period: {input_data.period}\n"
            result += f"Data Points: {len(df)}\n"
            result += f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            
            # Ensure we have required columns
            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                return f"Error: Missing required columns: {missing_cols}"
            
            # Current price info
            current_price = df['Close'].iloc[-1]
            prev_price = df['Close'].iloc[-2] if len(df) > 1 else current_price
            price_change = current_price - prev_price
            price_change_pct = (price_change / prev_price) * 100 if prev_price != 0 else 0
            
            result += f"Current Price: ${current_price:.2f}\n"
            result += f"Price Change: ${price_change:.2f} ({price_change_pct:+.2f}%)\n\n"
            
            # Perform analysis based on type
            if input_data.analysis_type in ['comprehensive', 'indicators']:
                indicators_result = self._calculate_technical_indicators(df, input_data.indicators)
                result += "=== TECHNICAL INDICATORS ===\n" + indicators_result + "\n"
            
            if input_data.analysis_type in ['comprehensive', 'momentum']:
                momentum_result = self._analyze_momentum(df)
                result += "=== MOMENTUM ANALYSIS ===\n" + momentum_result + "\n"
            
            if input_data.analysis_type in ['comprehensive', 'trend']:
                trend_result = self._analyze_trend(df)
                result += "=== TREND ANALYSIS ===\n" + trend_result + "\n"
            
            if input_data.analysis_type in ['comprehensive', 'volatility']:
                volatility_result = self._analyze_volatility(df)
                result += "=== VOLATILITY ANALYSIS ===\n" + volatility_result + "\n"
            
            if input_data.pattern_detection and input_data.analysis_type in ['comprehensive', 'patterns']:
                patterns_result = self._detect_patterns(df)
                result += "=== PATTERN ANALYSIS ===\n" + patterns_result + "\n"
            
            if input_data.signal_generation and input_data.analysis_type in ['comprehensive', 'signals']:
                signals_result = self._generate_signals(df, input_data.confidence_threshold)
                result += "=== TRADING SIGNALS ===\n" + signals_result + "\n"
            
            # Market structure analysis
            if input_data.analysis_type == 'comprehensive':
                structure_result = self._analyze_market_structure(df)
                result += "=== MARKET STRUCTURE ===\n" + structure_result + "\n"
            
            # Risk metrics
            risk_result = self._calculate_risk_metrics(df)
            result += "=== RISK METRICS ===\n" + risk_result
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error performing analysis: {e}")
            return f"Error in technical analysis: {str(e)}"
    
    def _calculate_technical_indicators(self, df: pd.DataFrame, specific_indicators: Optional[List[str]] = None) -> str:
        """Calculate technical indicators."""
        try:
            result = ""
            prices = df['Close']
            highs = df['High']
            lows = df['Low']
            volumes = df['Volume']
            
            # Default indicators if none specified
            if not specific_indicators:
                specific_indicators = ['sma', 'ema', 'rsi', 'macd', 'bollinger']
            
            for indicator in specific_indicators:
                indicator = indicator.lower()
                
                if indicator == 'sma':
                    sma_20 = prices.rolling(window=20).mean()
                    sma_50 = prices.rolling(window=50).mean()
                    if len(sma_20) > 0 and not pd.isna(sma_20.iloc[-1]):
                        result += f"SMA(20): ${sma_20.iloc[-1]:.2f}\n"
                    if len(sma_50) > 0 and not pd.isna(sma_50.iloc[-1]):
                        result += f"SMA(50): ${sma_50.iloc[-1]:.2f}\n"
                
                elif indicator == 'ema':
                    ema_12 = prices.ewm(span=12).mean()
                    ema_26 = prices.ewm(span=26).mean()
                    if len(ema_12) > 0 and not pd.isna(ema_12.iloc[-1]):
                        result += f"EMA(12): ${ema_12.iloc[-1]:.2f}\n"
                    if len(ema_26) > 0 and not pd.isna(ema_26.iloc[-1]):
                        result += f"EMA(26): ${ema_26.iloc[-1]:.2f}\n"
                
                elif indicator == 'rsi':
                    rsi = self._calculate_rsi(prices)
                    if rsi is not None:
                        rsi_signal = "Oversold" if rsi < 30 else "Overbought" if rsi > 70 else "Neutral"
                        result += f"RSI(14): {rsi:.2f} ({rsi_signal})\n"
                
                elif indicator == 'macd':
                    macd_line, signal_line, histogram = self._calculate_macd(prices)
                    if macd_line is not None:
                        macd_signal = "Bullish" if macd_line > signal_line else "Bearish"
                        result += f"MACD: {macd_line:.4f}\n"
                        result += f"MACD Signal: {signal_line:.4f} ({macd_signal})\n"
                        result += f"MACD Histogram: {histogram:.4f}\n"
                
                elif indicator == 'bollinger':
                    upper, middle, lower = self._calculate_bollinger_bands(prices)
                    if upper is not None:
                        current_price = prices.iloc[-1]
                        bb_position = (current_price - lower) / (upper - lower) * 100
                        result += f"Bollinger Upper: ${upper:.2f}\n"
                        result += f"Bollinger Middle: ${middle:.2f}\n"
                        result += f"Bollinger Lower: ${lower:.2f}\n"
                        result += f"BB Position: {bb_position:.1f}%\n"
                
                elif indicator == 'stochastic':
                    k_percent, d_percent = self._calculate_stochastic(highs, lows, prices)
                    if k_percent is not None:
                        stoch_signal = "Oversold" if k_percent < 20 else "Overbought" if k_percent > 80 else "Neutral"
                        result += f"Stochastic %K: {k_percent:.2f}\n"
                        result += f"Stochastic %D: {d_percent:.2f} ({stoch_signal})\n"
                
                elif indicator == 'williams_r':
                    williams_r = self._calculate_williams_r(highs, lows, prices)
                    if williams_r is not None:
                        wr_signal = "Oversold" if williams_r < -80 else "Overbought" if williams_r > -20 else "Neutral"
                        result += f"Williams %R: {williams_r:.2f} ({wr_signal})\n"
                
                elif indicator == 'cci':
                    cci = self._calculate_cci(highs, lows, prices)
                    if cci is not None:
                        cci_signal = "Oversold" if cci < -100 else "Overbought" if cci > 100 else "Neutral"
                        result += f"CCI(20): {cci:.2f} ({cci_signal})\n"
                
                elif indicator == 'atr':
                    atr = self._calculate_atr(highs, lows, prices)
                    if atr is not None:
                        result += f"ATR(14): ${atr:.2f}\n"
                
                elif indicator == 'obv':
                    obv = self._calculate_obv(prices, volumes)
                    if obv is not None:
                        result += f"OBV: {obv:,.0f}\n"
            
            return result if result else "No indicators calculated\n"
            
        except Exception as e:
            self.logger.error(f"Error calculating indicators: {e}")
            return f"Error calculating indicators: {str(e)}\n"
    
    def _analyze_momentum(self, df: pd.DataFrame) -> str:
        """Analyze momentum indicators."""
        try:
            result = ""
            prices = df['Close']
            
            # Price momentum
            returns_1d = prices.pct_change(1).iloc[-1] * 100
            returns_5d = prices.pct_change(5).iloc[-1] * 100 if len(prices) > 5 else None
            returns_20d = prices.pct_change(20).iloc[-1] * 100 if len(prices) > 20 else None
            
            result += f"1-Day Return: {returns_1d:+.2f}%\n"
            if returns_5d is not None:
                result += f"5-Day Return: {returns_5d:+.2f}%\n"
            if returns_20d is not None:
                result += f"20-Day Return: {returns_20d:+.2f}%\n"
            
            # Rate of Change (ROC)
            if len(prices) > 12:
                roc = ((prices.iloc[-1] / prices.iloc[-13]) - 1) * 100
                result += f"12-Period ROC: {roc:+.2f}%\n"
            
            # Momentum oscillator
            if len(prices) > 10:
                momentum = prices.iloc[-1] - prices.iloc[-11]
                result += f"10-Period Momentum: ${momentum:+.2f}\n"
            
            # Volume-weighted momentum
            if 'Volume' in df.columns and len(df) > 5:
                volume_weighted_price = (prices * df['Volume']).rolling(5).sum() / df['Volume'].rolling(5).sum()
                vwap_momentum = ((volume_weighted_price.iloc[-1] / volume_weighted_price.iloc[-6]) - 1) * 100 if len(volume_weighted_price) > 5 else None
                if vwap_momentum is not None:
                    result += f"5-Day VWAP Momentum: {vwap_momentum:+.2f}%\n"
            
            # Momentum strength assessment
            momentum_signals = []
            if returns_1d > 2:
                momentum_signals.append("Strong positive daily momentum")
            elif returns_1d < -2:
                momentum_signals.append("Strong negative daily momentum")
            
            if returns_5d is not None:
                if returns_5d > 5:
                    momentum_signals.append("Strong weekly uptrend")
                elif returns_5d < -5:
                    momentum_signals.append("Strong weekly downtrend")
            
            if momentum_signals:
                result += "\nMomentum Signals:\n"
                for signal in momentum_signals:
                    result += f"• {signal}\n"
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error analyzing momentum: {e}")
            return f"Error in momentum analysis: {str(e)}\n"
    
    def _analyze_trend(self, df: pd.DataFrame) -> str:
        """Analyze trend characteristics."""
        try:
            result = ""
            prices = df['Close']
            
            # Moving average trend
            if len(prices) >= 50:
                sma_20 = prices.rolling(20).mean().iloc[-1]
                sma_50 = prices.rolling(50).mean().iloc[-1]
                current_price = prices.iloc[-1]
                
                result += f"Price vs SMA(20): {((current_price / sma_20) - 1) * 100:+.2f}%\n"
                result += f"Price vs SMA(50): {((current_price / sma_50) - 1) * 100:+.2f}%\n"
                result += f"SMA(20) vs SMA(50): {((sma_20 / sma_50) - 1) * 100:+.2f}%\n"
                
                # Trend direction
                if current_price > sma_20 > sma_50:
                    trend_direction = "Strong Uptrend"
                elif current_price > sma_20 and sma_20 < sma_50:
                    trend_direction = "Weak Uptrend"
                elif current_price < sma_20 < sma_50:
                    trend_direction = "Strong Downtrend"
                elif current_price < sma_20 and sma_20 > sma_50:
                    trend_direction = "Weak Downtrend"
                else:
                    trend_direction = "Sideways/Consolidation"
                
                result += f"Trend Direction: {trend_direction}\n"
            
            # Linear regression trend
            if len(prices) >= 20:
                x = np.arange(len(prices[-20:]))
                y = prices[-20:].values
                slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
                
                trend_strength = abs(r_value)
                trend_direction_lr = "Upward" if slope > 0 else "Downward"
                
                result += f"\n20-Period Linear Trend:\n"
                result += f"Direction: {trend_direction_lr}\n"
                result += f"Slope: ${slope:.4f} per period\n"
                result += f"Strength (R²): {r_value**2:.3f}\n"
                result += f"Significance (p-value): {p_value:.4f}\n"
            
            # ADX (Average Directional Index) approximation
            if len(df) >= 14:
                adx = self._calculate_adx(df['High'], df['Low'], df['Close'])
                if adx is not None:
                    if adx > 25:
                        adx_signal = "Strong Trend"
                    elif adx > 20:
                        adx_signal = "Moderate Trend"
                    else:
                        adx_signal = "Weak/No Trend"
                    result += f"ADX(14): {adx:.2f} ({adx_signal})\n"
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error analyzing trend: {e}")
            return f"Error in trend analysis: {str(e)}\n"
    
    def _analyze_volatility(self, df: pd.DataFrame) -> str:
        """Analyze volatility metrics."""
        try:
            result = ""
            prices = df['Close']
            highs = df['High']
            lows = df['Low']
            
            # Historical volatility
            returns = prices.pct_change().dropna()
            if len(returns) > 1:
                daily_vol = returns.std()
                annualized_vol = daily_vol * np.sqrt(252)  # Assuming 252 trading days
                result += f"Daily Volatility: {daily_vol * 100:.2f}%\n"
                result += f"Annualized Volatility: {annualized_vol * 100:.2f}%\n"
                
                # Rolling volatility
                if len(returns) >= 20:
                    rolling_vol_20 = returns.rolling(20).std().iloc[-1]
                    result += f"20-Day Rolling Volatility: {rolling_vol_20 * 100:.2f}%\n"
            
            # True Range and ATR
            atr = self._calculate_atr(highs, lows, prices)
            if atr is not None:
                atr_pct = (atr / prices.iloc[-1]) * 100
                result += f"ATR: ${atr:.2f} ({atr_pct:.2f}% of price)\n"
            
            # High-Low volatility
            if len(df) >= 20:
                hl_range = ((highs - lows) / prices * 100).rolling(20).mean().iloc[-1]
                result += f"20-Day Avg High-Low Range: {hl_range:.2f}%\n"
            
            # Volatility regime
            if len(returns) >= 60:
                recent_vol = returns.tail(20).std()
                long_term_vol = returns.tail(60).std()
                vol_ratio = recent_vol / long_term_vol
                
                if vol_ratio > 1.5:
                    vol_regime = "High Volatility"
                elif vol_ratio > 1.2:
                    vol_regime = "Elevated Volatility"
                elif vol_ratio < 0.8:
                    vol_regime = "Low Volatility"
                else:
                    vol_regime = "Normal Volatility"
                
                result += f"Volatility Regime: {vol_regime} (Ratio: {vol_ratio:.2f})\n"
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error analyzing volatility: {e}")
            return f"Error in volatility analysis: {str(e)}\n"
    
    def _detect_patterns(self, df: pd.DataFrame) -> str:
        """Detect chart patterns and key levels."""
        try:
            result = ""
            prices = df['Close']
            highs = df['High']
            lows = df['Low']
            
            # Support and Resistance levels
            support_resistance = self._find_support_resistance(prices)
            if support_resistance:
                result += "Key Levels:\n"
                for level_type, level_value in support_resistance.items():
                    distance = ((prices.iloc[-1] / level_value) - 1) * 100
                    result += f"{level_type}: ${level_value:.2f} ({distance:+.2f}% from current)\n"
            
            # Pivot points
            if len(df) >= 3:
                pivot_points = self._calculate_pivot_points(df.iloc[-1])
                result += "\nPivot Points:\n"
                for level, value in pivot_points.items():
                    result += f"{level}: ${value:.2f}\n"
            
            # Pattern recognition
            patterns = self._recognize_patterns(df)
            if patterns:
                result += "\nDetected Patterns:\n"
                for pattern in patterns:
                    result += f"• {pattern}\n"
            
            # Breakout analysis
            breakouts = self._analyze_breakouts(df)
            if breakouts:
                result += "\nBreakout Analysis:\n"
                for breakout in breakouts:
                    result += f"• {breakout}\n"
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error detecting patterns: {e}")
            return f"Error in pattern detection: {str(e)}\n"
    
    def _generate_signals(self, df: pd.DataFrame, confidence_threshold: float) -> str:
        """Generate trading signals based on technical analysis."""
        try:
            result = ""
            signals = []
            
            prices = df['Close']
            current_price = prices.iloc[-1]
            
            # RSI signals
            rsi = self._calculate_rsi(prices)
            if rsi is not None:
                if rsi < 30:
                    signals.append(("BUY", "RSI Oversold", 0.7, f"RSI at {rsi:.1f}"))
                elif rsi > 70:
                    signals.append(("SELL", "RSI Overbought", 0.7, f"RSI at {rsi:.1f}"))
            
            # MACD signals
            macd_line, signal_line, histogram = self._calculate_macd(prices)
            if macd_line is not None and signal_line is not None:
                if len(prices) > 1:
                    prev_macd = self._calculate_macd(prices.iloc[:-1])[0]
                    prev_signal = self._calculate_macd(prices.iloc[:-1])[1]
                    
                    if prev_macd <= prev_signal and macd_line > signal_line:
                        signals.append(("BUY", "MACD Bullish Crossover", 0.8, f"MACD: {macd_line:.4f}"))
                    elif prev_macd >= prev_signal and macd_line < signal_line:
                        signals.append(("SELL", "MACD Bearish Crossover", 0.8, f"MACD: {macd_line:.4f}"))
            
            # Moving Average signals
            if len(prices) >= 50:
                sma_20 = prices.rolling(20).mean().iloc[-1]
                sma_50 = prices.rolling(50).mean().iloc[-1]
                
                if current_price > sma_20 > sma_50:
                    signals.append(("BUY", "Price Above MAs", 0.6, f"Price: ${current_price:.2f}, SMA20: ${sma_20:.2f}"))
                elif current_price < sma_20 < sma_50:
                    signals.append(("SELL", "Price Below MAs", 0.6, f"Price: ${current_price:.2f}, SMA20: ${sma_20:.2f}"))
            
            # Bollinger Band signals
            upper, middle, lower = self._calculate_bollinger_bands(prices)
            if upper is not None:
                if current_price <= lower:
                    signals.append(("BUY", "Bollinger Band Oversold", 0.7, f"Price at lower band: ${lower:.2f}"))
                elif current_price >= upper:
                    signals.append(("SELL", "Bollinger Band Overbought", 0.7, f"Price at upper band: ${upper:.2f}"))
            
            # Volume confirmation
            if 'Volume' in df.columns and len(df) >= 20:
                avg_volume = df['Volume'].rolling(20).mean().iloc[-1]
                current_volume = df['Volume'].iloc[-1]
                volume_ratio = current_volume / avg_volume
                
                # Enhance signal confidence with volume
                for i, (action, reason, confidence, details) in enumerate(signals):
                    if volume_ratio > 1.5:  # High volume
                        signals[i] = (action, reason, min(confidence + 0.1, 1.0), details + f" | High Volume: {volume_ratio:.1f}x")
                    elif volume_ratio < 0.5:  # Low volume
                        signals[i] = (action, reason, max(confidence - 0.1, 0.1), details + f" | Low Volume: {volume_ratio:.1f}x")
            
            # Filter signals by confidence threshold
            filtered_signals = [(action, reason, conf, details) for action, reason, conf, details in signals if conf >= confidence_threshold]
            
            if filtered_signals:
                result += f"Trading Signals (Confidence >= {confidence_threshold:.1f}):\n\n"
                
                for action, reason, confidence, details in filtered_signals:
                    result += f"🔸 {action} Signal\n"
                    result += f"   Reason: {reason}\n"
                    result += f"   Confidence: {confidence:.1f}\n"
                    result += f"   Details: {details}\n\n"
                
                # Overall signal summary
                buy_signals = [s for s in filtered_signals if s[0] == "BUY"]
                sell_signals = [s for s in filtered_signals if s[0] == "SELL"]
                
                if buy_signals and not sell_signals:
                    overall = "BULLISH"
                elif sell_signals and not buy_signals:
                    overall = "BEARISH"
                elif len(buy_signals) > len(sell_signals):
                    overall = "MODERATELY BULLISH"
                elif len(sell_signals) > len(buy_signals):
                    overall = "MODERATELY BEARISH"
                else:
                    overall = "NEUTRAL/MIXED"
                
                result += f"Overall Signal: {overall}\n"
                result += f"Buy Signals: {len(buy_signals)} | Sell Signals: {len(sell_signals)}\n"
            else:
                result += f"No signals meet the confidence threshold of {confidence_threshold:.1f}\n"
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error generating signals: {e}")
            return f"Error generating signals: {str(e)}\n"
    
    def _analyze_market_structure(self, df: pd.DataFrame) -> str:
        """Analyze market structure and key levels."""
        try:
            result = ""
            prices = df['Close']
            highs = df['High']
            lows = df['Low']
            
            # Higher highs and lower lows analysis
            if len(df) >= 10:
                recent_high = highs.tail(10).max()
                recent_low = lows.tail(10).min()
                prev_high = highs.iloc[:-10].tail(10).max() if len(df) >= 20 else recent_high
                prev_low = lows.iloc[:-10].tail(10).min() if len(df) >= 20 else recent_low
                
                if recent_high > prev_high and recent_low > prev_low:
                    structure = "Higher Highs & Higher Lows (Uptrend)"
                elif recent_high < prev_high and recent_low < prev_low:
                    structure = "Lower Highs & Lower Lows (Downtrend)"
                elif recent_high > prev_high and recent_low < prev_low:
                    structure = "Higher Highs & Lower Lows (Expansion)"
                elif recent_high < prev_high and recent_low > prev_low:
                    structure = "Lower Highs & Higher Lows (Contraction)"
                else:
                    structure = "Mixed/Sideways"
                
                result += f"Market Structure: {structure}\n"
                result += f"Recent High: ${recent_high:.2f}\n"
                result += f"Recent Low: ${recent_low:.2f}\n"
            
            # Price position analysis
            if len(prices) >= 20:
                price_range_20 = highs.tail(20).max() - lows.tail(20).min()
                current_position = (prices.iloc[-1] - lows.tail(20).min()) / price_range_20 * 100
                result += f"\n20-Day Range Position: {current_position:.1f}%\n"
                
                if current_position > 80:
                    position_desc = "Near 20-day high"
                elif current_position > 60:
                    position_desc = "Upper range"
                elif current_position > 40:
                    position_desc = "Middle range"
                elif current_position > 20:
                    position_desc = "Lower range"
                else:
                    position_desc = "Near 20-day low"
                
                result += f"Position Description: {position_desc}\n"
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error analyzing market structure: {e}")
            return f"Error in market structure analysis: {str(e)}\n"
    
    def _calculate_risk_metrics(self, df: pd.DataFrame) -> str:
        """Calculate risk-related metrics."""
        try:
            result = ""
            prices = df['Close']
            
            # Returns analysis
            returns = prices.pct_change().dropna()
            if len(returns) > 1:
                # Basic risk metrics
                avg_return = returns.mean()
                return_std = returns.std()
                
                result += f"Average Daily Return: {avg_return * 100:.3f}%\n"
                result += f"Return Volatility: {return_std * 100:.3f}%\n"
                
                # Sharpe ratio approximation (assuming 0% risk-free rate)
                if return_std > 0:
                    sharpe_ratio = avg_return / return_std
                    result += f"Sharpe Ratio: {sharpe_ratio:.3f}\n"
                
                # Downside deviation
                negative_returns = returns[returns < 0]
                if len(negative_returns) > 0:
                    downside_deviation = negative_returns.std()
                    result += f"Downside Deviation: {downside_deviation * 100:.3f}%\n"
                    
                    # Sortino ratio
                    if downside_deviation > 0:
                        sortino_ratio = avg_return / downside_deviation
                        result += f"Sortino Ratio: {sortino_ratio:.3f}\n"
                
                # Maximum drawdown
                cumulative_returns = (1 + returns).cumprod()
                rolling_max = cumulative_returns.expanding().max()
                drawdown = (cumulative_returns - rolling_max) / rolling_max
                max_drawdown = drawdown.min()
                
                result += f"Maximum Drawdown: {max_drawdown * 100:.2f}%\n"
                
                # Value at Risk (95% confidence)
                var_95 = np.percentile(returns, 5)
                result += f"VaR (95%): {var_95 * 100:.2f}%\n"
                
                # Expected Shortfall (Conditional VaR)
                es_95 = returns[returns <= var_95].mean()
                result += f"Expected Shortfall (95%): {es_95 * 100:.2f}%\n"
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error calculating risk metrics: {e}")
            return f"Error calculating risk metrics: {str(e)}\n"
    
    # Helper methods for technical indicators
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
            
            return rsi.iloc[-1] if not rsi.empty and not pd.isna(rsi.iloc[-1]) else None
            
        except Exception:
            return None
    
    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """Calculate MACD."""
        try:
            if len(prices) < slow + signal:
                return None, None, None
            
            ema_fast = prices.ewm(span=fast).mean()
            ema_slow = prices.ewm(span=slow).mean()
            
            macd_line = ema_fast - ema_slow
            signal_line = macd_line.ewm(span=signal).mean()
            histogram = macd_line - signal_line
            
            return macd_line.iloc[-1], signal_line.iloc[-1], histogram.iloc[-1]
            
        except Exception:
            return None, None, None
    
    def _calculate_bollinger_bands(self, prices: pd.Series, period: int = 20, std_dev: int = 2) -> Tuple[Optional[float], Optional[float], Optional[float]]:
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
    
    def _calculate_stochastic(self, highs: pd.Series, lows: pd.Series, closes: pd.Series, k_period: int = 14, d_period: int = 3) -> Tuple[Optional[float], Optional[float]]:
        """Calculate Stochastic Oscillator."""
        try:
            if len(closes) < k_period:
                return None, None
            
            lowest_low = lows.rolling(window=k_period).min()
            highest_high = highs.rolling(window=k_period).max()
            
            k_percent = 100 * ((closes - lowest_low) / (highest_high - lowest_low))
            d_percent = k_percent.rolling(window=d_period).mean()
            
            return k_percent.iloc[-1], d_percent.iloc[-1]
            
        except Exception:
            return None, None
    
    def _calculate_williams_r(self, highs: pd.Series, lows: pd.Series, closes: pd.Series, period: int = 14) -> Optional[float]:
        """Calculate Williams %R."""
        try:
            if len(closes) < period:
                return None
            
            highest_high = highs.rolling(window=period).max()
            lowest_low = lows.rolling(window=period).min()
            
            williams_r = -100 * ((highest_high - closes) / (highest_high - lowest_low))
            
            return williams_r.iloc[-1]
            
        except Exception:
            return None
    
    def _calculate_cci(self, highs: pd.Series, lows: pd.Series, closes: pd.Series, period: int = 20) -> Optional[float]:
        """Calculate Commodity Channel Index."""
        try:
            if len(closes) < period:
                return None
            
            typical_price = (highs + lows + closes) / 3
            sma_tp = typical_price.rolling(window=period).mean()
            mean_deviation = typical_price.rolling(window=period).apply(lambda x: np.mean(np.abs(x - x.mean())))
            
            cci = (typical_price - sma_tp) / (0.015 * mean_deviation)
            
            return cci.iloc[-1]
            
        except Exception:
            return None
    
    def _calculate_atr(self, highs: pd.Series, lows: pd.Series, closes: pd.Series, period: int = 14) -> Optional[float]:
        """Calculate Average True Range."""
        try:
            if len(closes) < 2:
                return None
            
            high_low = highs - lows
            high_close_prev = np.abs(highs - closes.shift(1))
            low_close_prev = np.abs(lows - closes.shift(1))
            
            true_range = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
            atr = true_range.rolling(window=period).mean()
            
            return atr.iloc[-1]
            
        except Exception:
            return None
    
    def _calculate_obv(self, prices: pd.Series, volumes: pd.Series) -> Optional[float]:
        """Calculate On-Balance Volume."""
        try:
            if len(prices) < 2:
                return None
            
            price_change = prices.diff()
            obv = volumes.copy()
            
            obv[price_change < 0] = -volumes[price_change < 0]
            obv[price_change == 0] = 0
            
            return obv.cumsum().iloc[-1]
            
        except Exception:
            return None
    
    def _calculate_adx(self, highs: pd.Series, lows: pd.Series, closes: pd.Series, period: int = 14) -> Optional[float]:
        """Calculate Average Directional Index."""
        try:
            if len(closes) < period + 1:
                return None
            
            # Calculate True Range
            tr = self._calculate_true_range_series(highs, lows, closes)
            
            # Calculate Directional Movement
            plus_dm = highs.diff()
            minus_dm = -lows.diff()
            
            plus_dm[plus_dm < 0] = 0
            minus_dm[minus_dm < 0] = 0
            
            # Smooth the values
            tr_smooth = tr.rolling(window=period).mean()
            plus_dm_smooth = plus_dm.rolling(window=period).mean()
            minus_dm_smooth = minus_dm.rolling(window=period).mean()
            
            # Calculate Directional Indicators
            plus_di = 100 * (plus_dm_smooth / tr_smooth)
            minus_di = 100 * (minus_dm_smooth / tr_smooth)
            
            # Calculate ADX
            dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
            adx = dx.rolling(window=period).mean()
            
            return adx.iloc[-1]
            
        except Exception:
            return None
    
    def _calculate_true_range_series(self, highs: pd.Series, lows: pd.Series, closes: pd.Series) -> pd.Series:
        """Calculate True Range series."""
        high_low = highs - lows
        high_close_prev = np.abs(highs - closes.shift(1))
        low_close_prev = np.abs(lows - closes.shift(1))
        
        return pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
    
    def _find_support_resistance(self, prices: pd.Series, window: int = 20) -> Dict[str, float]:
        """Find support and resistance levels."""
        try:
            levels = {}
            
            if len(prices) < window * 2:
                return levels
            
            # Recent high and low
            recent_high = prices.tail(window).max()
            recent_low = prices.tail(window).min()
            
            levels['Recent Resistance'] = recent_high
            levels['Recent Support'] = recent_low
            
            # Longer-term levels
            if len(prices) >= window * 3:
                longer_high = prices.tail(window * 3).max()
                longer_low = prices.tail(window * 3).min()
                
                if longer_high > recent_high:
                    levels['Major Resistance'] = longer_high
                if longer_low < recent_low:
                    levels['Major Support'] = longer_low
            
            return levels
            
        except Exception:
            return {}
    
    def _calculate_pivot_points(self, last_bar: pd.Series) -> Dict[str, float]:
        """Calculate pivot points."""
        try:
            high = last_bar['High']
            low = last_bar['Low']
            close = last_bar['Close']
            
            pivot = (high + low + close) / 3
            
            r1 = 2 * pivot - low
            s1 = 2 * pivot - high
            r2 = pivot + (high - low)
            s2 = pivot - (high - low)
            r3 = high + 2 * (pivot - low)
            s3 = low - 2 * (high - pivot)
            
            return {
                'Pivot': pivot,
                'R1': r1,
                'R2': r2,
                'R3': r3,
                'S1': s1,
                'S2': s2,
                'S3': s3
            }
            
        except Exception:
            return {}
    
    def _recognize_patterns(self, df: pd.DataFrame) -> List[str]:
        """Recognize basic chart patterns."""
        try:
            patterns = []
            
            if len(df) < 20:
                return patterns
            
            prices = df['Close']
            highs = df['High']
            lows = df['Low']
            
            # Double top/bottom pattern (simplified)
            recent_highs = highs.tail(10)
            recent_lows = lows.tail(10)
            
            if len(recent_highs) >= 5:
                max_high = recent_highs.max()
                high_count = (recent_highs >= max_high * 0.99).sum()
                if high_count >= 2:
                    patterns.append("Potential Double Top Pattern")
            
            if len(recent_lows) >= 5:
                min_low = recent_lows.min()
                low_count = (recent_lows <= min_low * 1.01).sum()
                if low_count >= 2:
                    patterns.append("Potential Double Bottom Pattern")
            
            # Triangle pattern (simplified)
            if len(prices) >= 20:
                recent_range = highs.tail(20).max() - lows.tail(20).min()
                very_recent_range = highs.tail(5).max() - lows.tail(5).min()
                
                if very_recent_range < recent_range * 0.5:
                    patterns.append("Potential Triangle/Consolidation Pattern")
            
            return patterns
            
        except Exception:
            return []
    
    def _analyze_breakouts(self, df: pd.DataFrame) -> List[str]:
        """Analyze potential breakouts."""
        try:
            breakouts = []
            
            if len(df) < 20:
                return breakouts
            
            prices = df['Close']
            volumes = df['Volume']
            current_price = prices.iloc[-1]
            
            # 20-day breakout
            high_20 = df['High'].tail(20).max()
            low_20 = df['Low'].tail(20).min()
            
            if current_price >= high_20 * 0.999:  # Near or at 20-day high
                avg_volume = volumes.tail(20).mean()
                current_volume = volumes.iloc[-1]
                
                if current_volume > avg_volume * 1.5:
                    breakouts.append("Upward breakout from 20-day range with high volume")
                else:
                    breakouts.append("Upward breakout from 20-day range with normal volume")
            
            elif current_price <= low_20 * 1.001:  # Near or at 20-day low
                avg_volume = volumes.tail(20).mean()
                current_volume = volumes.iloc[-1]
                
                if current_volume > avg_volume * 1.5:
                    breakouts.append("Downward breakdown from 20-day range with high volume")
                else:
                    breakouts.append("Downward breakdown from 20-day range with normal volume")
            
            # Moving average breakout
            if len(prices) >= 50:
                sma_50 = prices.rolling(50).mean().iloc[-1]
                prev_price = prices.iloc[-2]
                
                if prev_price <= sma_50 and current_price > sma_50:
                    breakouts.append("Bullish breakout above 50-day SMA")
                elif prev_price >= sma_50 and current_price < sma_50:
                    breakouts.append("Bearish breakdown below 50-day SMA")
            
            return breakouts
            
        except Exception:
            return []


if __name__ == "__main__":
    # Example usage
    import asyncio
    from pathlib import Path
    from core.config_manager import ConfigManager
    from core.data_manager import UnifiedDataManager
    
    async def test_technical_analysis_tool():
        config_manager = ConfigManager(Path("../config"))
        data_manager = UnifiedDataManager(config_manager)
        
        tool = TechnicalAnalysisTool(data_manager)
        
        # Test comprehensive analysis
        result = tool._run(
            symbol="AAPL",
            analysis_type="comprehensive",
            timeframe="1d",
            period="3mo",
            indicators=["sma", "ema", "rsi", "macd", "bollinger"],
            pattern_detection=True,
            signal_generation=True,
            confidence_threshold=0.6
        )
        
        print("Technical Analysis Result:")
        print(result)
    
    # Run test
    # asyncio.run(test_technical_analysis_tool())