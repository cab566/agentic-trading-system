#!/usr/bin/env python3
"""
Forex Analysis Tool for CrewAI Trading System

Provides comprehensive forex analysis including:
- Technical analysis with forex-specific indicators
- Economic calendar integration
- Central bank policy analysis
- Carry trade analysis
- Currency correlation analysis
- Interest rate differential analysis
"""

import os
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass

import pandas as pd
import numpy as np
import requests
from langchain_core.tools import BaseTool
from pydantic import Field

from core.forex_adapter import ForexDataManager
from core.config_manager import ConfigManager


@dataclass
class ForexAnalysisResult:
    """Forex analysis result."""
    pair: str
    base_currency: str
    quote_currency: str
    current_rate: float
    daily_change: float
    daily_change_pct: float
    
    # Technical indicators
    rsi: float
    macd_signal: str
    atr: float  # Average True Range
    pivot_points: Dict[str, float]
    trend_direction: str
    
    # Forex-specific metrics
    interest_rate_differential: float
    carry_trade_score: float
    economic_calendar_impact: str
    central_bank_sentiment: str
    
    # Currency strength
    base_currency_strength: float
    quote_currency_strength: float
    relative_strength: float
    
    # Correlations
    major_correlations: Dict[str, float]
    
    # Analysis scores
    technical_score: float  # -1 to 1
    fundamental_score: float  # -1 to 1
    carry_score: float  # -1 to 1
    overall_score: float  # -1 to 1
    
    # Trading sessions
    active_sessions: List[str]
    session_volatility: Dict[str, float]
    
    # Recommendations
    recommendation: str  # BUY, SELL, HOLD
    confidence: float  # 0 to 1
    risk_level: str  # LOW, MEDIUM, HIGH
    optimal_timeframe: str
    
    timestamp: datetime


class ForexAnalysisTool(BaseTool):
    """
    Comprehensive forex analysis tool for trading agents.
    
    This tool provides multi-dimensional analysis of currency pairs including
    technical analysis, fundamental analysis, carry trade opportunities,
    and economic calendar integration.
    """
    
    name: str = "forex_analysis_tool"
    description: str = (
        "Analyze forex currency pairs with technical indicators, economic fundamentals, "
        "carry trade analysis, and central bank policy impact. Returns comprehensive "
        "analysis with buy/sell/hold recommendations and optimal trading sessions."
    )
    
    # Pydantic field declarations
    config_manager: ConfigManager = Field(exclude=True)
    forex_manager: ForexDataManager = Field(exclude=True)
    logger: Any = Field(default=None, exclude=True)
    oanda_api_key: Optional[str] = Field(default=None, exclude=True)
    alpha_vantage_api_key: Optional[str] = Field(default=None, exclude=True)
    fxcm_api_key: Optional[str] = Field(default=None, exclude=True)
    major_pairs: List[str] = Field(default_factory=lambda: [
        'EUR/USD', 'GBP/USD', 'USD/JPY', 'USD/CHF',
        'AUD/USD', 'USD/CAD', 'NZD/USD'
    ], exclude=True)
    trading_sessions: Dict[str, Dict[str, str]] = Field(default_factory=lambda: {
        'Sydney': {'start': '21:00', 'end': '06:00', 'timezone': 'UTC'},
        'Tokyo': {'start': '00:00', 'end': '09:00', 'timezone': 'UTC'},
        'London': {'start': '08:00', 'end': '17:00', 'timezone': 'UTC'},
        'New York': {'start': '13:00', 'end': '22:00', 'timezone': 'UTC'}
    }, exclude=True)
    analysis_weights: Dict[str, float] = Field(default_factory=lambda: {
        'technical': 0.4,
        'fundamental': 0.35,
        'carry': 0.25
    }, exclude=True)
    interest_rates: Dict[str, float] = Field(default_factory=lambda: {
        'USD': 5.25, 'EUR': 4.50, 'GBP': 5.25, 'JPY': -0.10,
        'CHF': 1.75, 'AUD': 4.35, 'CAD': 5.00, 'NZD': 5.50
    }, exclude=True)
    
    def __init__(self, config_manager: ConfigManager, **kwargs):
        # Initialize components before calling super().__init__
        data_source_configs = config_manager.get_data_source_configs()
        forex_manager = ForexDataManager(data_source_configs)
        logger = logging.getLogger(__name__)
        
        # Get API keys
        data_sources = config_manager.get_data_source_configs()
        oanda_api_key = os.getenv(data_sources.get('oanda', {}).get('api_key_env', 'OANDA_API_KEY'))
        alpha_vantage_api_key = os.getenv(data_sources.get('alphavantage_forex', {}).get('api_key_env', 'ALPHA_VANTAGE_API_KEY'))
        fxcm_api_key = os.getenv('FXCM_API_KEY')
        
        # Call super().__init__ with all required fields
        super().__init__(
            config_manager=config_manager,
            forex_manager=forex_manager,
            logger=logger,
            oanda_api_key=oanda_api_key,
            alpha_vantage_api_key=alpha_vantage_api_key,
            fxcm_api_key=fxcm_api_key,
            **kwargs
        )
        
        # All attributes are now defined as Pydantic fields
    
    async def _run(self, pair: str, timeframe: str = '1H', analysis_depth: str = 'standard') -> str:
        """
        Run comprehensive forex analysis.
        
        Args:
            pair: Currency pair (e.g., 'EUR/USD', 'GBP/JPY')
            timeframe: Analysis timeframe ('1H', '4H', '1D', '1W')
            analysis_depth: Analysis depth ('quick', 'standard', 'deep')
        
        Returns:
            Formatted analysis report as string
        """
        try:
            # Normalize pair format
            pair = pair.replace('/', '').replace('-', '').upper()
            if len(pair) == 6:
                base_currency = pair[:3]
                quote_currency = pair[3:]
                formatted_pair = f"{base_currency}/{quote_currency}"
            else:
                return f"Error: Invalid currency pair format: {pair}"
            
            # Get market data
            market_data = await self._get_market_data(formatted_pair, timeframe)
            if not market_data:
                return f"Error: Could not retrieve market data for {formatted_pair}"
            
            # Perform technical analysis
            technical_analysis = await self._perform_technical_analysis(market_data, timeframe)
            
            # Get fundamental analysis
            fundamental_analysis = await self._perform_fundamental_analysis(base_currency, quote_currency)
            
            # Analyze carry trade opportunity
            carry_analysis = await self._analyze_carry_trade(base_currency, quote_currency)
            
            # Get currency strength
            currency_strength = await self._analyze_currency_strength(base_currency, quote_currency)
            
            # Get correlations
            correlations = await self._get_currency_correlations(formatted_pair)
            
            # Analyze trading sessions
            session_analysis = self._analyze_trading_sessions(formatted_pair)
            
            # Calculate overall scores
            scores = self._calculate_scores(technical_analysis, fundamental_analysis, carry_analysis)
            
            # Generate recommendation
            recommendation = self._generate_recommendation(scores, market_data, session_analysis)
            
            # Create analysis result
            result = ForexAnalysisResult(
                pair=formatted_pair,
                base_currency=base_currency,
                quote_currency=quote_currency,
                current_rate=market_data['current_rate'],
                daily_change=market_data.get('daily_change', 0),
                daily_change_pct=market_data.get('daily_change_pct', 0),
                
                rsi=technical_analysis['rsi'],
                macd_signal=technical_analysis['macd_signal'],
                atr=technical_analysis['atr'],
                pivot_points=technical_analysis['pivot_points'],
                trend_direction=technical_analysis['trend_direction'],
                
                interest_rate_differential=carry_analysis['rate_differential'],
                carry_trade_score=carry_analysis['carry_score'],
                economic_calendar_impact=fundamental_analysis['economic_impact'],
                central_bank_sentiment=fundamental_analysis['cb_sentiment'],
                
                base_currency_strength=currency_strength['base_strength'],
                quote_currency_strength=currency_strength['quote_strength'],
                relative_strength=currency_strength['relative_strength'],
                
                major_correlations=correlations,
                
                technical_score=scores['technical'],
                fundamental_score=scores['fundamental'],
                carry_score=scores['carry'],
                overall_score=scores['overall'],
                
                active_sessions=session_analysis['active_sessions'],
                session_volatility=session_analysis['session_volatility'],
                
                recommendation=recommendation['action'],
                confidence=recommendation['confidence'],
                risk_level=recommendation['risk_level'],
                optimal_timeframe=recommendation['optimal_timeframe'],
                
                timestamp=datetime.now()
            )
            
            return self._format_analysis_report(result)
            
        except Exception as e:
            self.logger.error(f"Error in forex analysis for {pair}: {e}")
            return f"Error analyzing {pair}: {str(e)}"
    
    async def _get_market_data(self, pair: str, timeframe: str) -> Optional[Dict[str, Any]]:
        """Get market data for the currency pair."""
        try:
            # Get current rate
            current_data = await self.forex_manager.get_current_rate(pair)
            
            # Get historical data
            historical_data = await self.forex_manager.get_historical_data(
                pair, timeframe, limit=100
            )
            
            if not current_data or historical_data.empty:
                return None
            
            # Calculate daily change
            if len(historical_data) >= 2:
                current_rate = historical_data['close'].iloc[-1]
                previous_rate = historical_data['close'].iloc[-2]
                daily_change = current_rate - previous_rate
                daily_change_pct = (daily_change / previous_rate) * 100
            else:
                current_rate = current_data.get('rate', 0)
                daily_change = 0
                daily_change_pct = 0
            
            return {
                'current_rate': current_rate,
                'daily_change': daily_change,
                'daily_change_pct': daily_change_pct,
                'historical_data': historical_data,
                'bid': current_data.get('bid', current_rate),
                'ask': current_data.get('ask', current_rate),
                'spread': current_data.get('spread', 0)
            }
            
        except Exception as e:
            self.logger.error(f"Error getting market data for {pair}: {e}")
            return None
    
    async def _perform_technical_analysis(self, market_data: Dict[str, Any], timeframe: str) -> Dict[str, Any]:
        """Perform technical analysis on the currency pair."""
        df = market_data['historical_data']
        
        # Calculate RSI
        rsi = self._calculate_rsi(df['close'])
        
        # Calculate MACD
        macd_line, macd_signal, macd_histogram = self._calculate_macd(df['close'])
        macd_signal_str = 'BUY' if macd_line.iloc[-1] > macd_signal.iloc[-1] else 'SELL'
        
        # Calculate ATR (Average True Range)
        atr = self._calculate_atr(df)
        
        # Calculate pivot points
        pivot_points = self._calculate_pivot_points(df)
        
        # Determine trend direction
        trend_direction = self._determine_trend(df)
        
        # Calculate support and resistance
        support_resistance = self._find_support_resistance_forex(df)
        
        return {
            'rsi': rsi.iloc[-1],
            'macd_signal': macd_signal_str,
            'atr': atr.iloc[-1],
            'pivot_points': pivot_points,
            'trend_direction': trend_direction,
            'support_resistance': support_resistance,
            'volatility': self._calculate_volatility_forex(df)
        }
    
    async def _perform_fundamental_analysis(self, base_currency: str, quote_currency: str) -> Dict[str, Any]:
        """Perform fundamental analysis for the currency pair."""
        try:
            # Get economic calendar events
            economic_events = await self._get_economic_calendar(base_currency, quote_currency)
            
            # Analyze central bank sentiment
            cb_sentiment = await self._analyze_central_bank_sentiment(base_currency, quote_currency)
            
            # Get economic indicators
            economic_indicators = await self._get_economic_indicators(base_currency, quote_currency)
            
            # Assess economic impact
            economic_impact = self._assess_economic_impact(economic_events, economic_indicators)
            
            return {
                'economic_events': economic_events,
                'cb_sentiment': cb_sentiment,
                'economic_indicators': economic_indicators,
                'economic_impact': economic_impact
            }
            
        except Exception as e:
            self.logger.error(f"Error in fundamental analysis: {e}")
            return {
                'economic_events': [],
                'cb_sentiment': 'NEUTRAL',
                'economic_indicators': {},
                'economic_impact': 'LOW'
            }
    
    async def _analyze_carry_trade(self, base_currency: str, quote_currency: str) -> Dict[str, Any]:
        """Analyze carry trade opportunity."""
        base_rate = self.interest_rates.get(base_currency, 0)
        quote_rate = self.interest_rates.get(quote_currency, 0)
        
        rate_differential = base_rate - quote_rate
        
        # Calculate carry trade score
        # Positive differential favors long position, negative favors short
        carry_score = np.tanh(rate_differential / 2)  # Normalize to -1 to 1
        
        # Adjust for risk (higher differentials can be riskier)
        risk_adjustment = 1 - min(abs(rate_differential) / 10, 0.3)
        carry_score *= risk_adjustment
        
        return {
            'base_rate': base_rate,
            'quote_rate': quote_rate,
            'rate_differential': rate_differential,
            'carry_score': carry_score,
            'carry_direction': 'LONG' if rate_differential > 0 else 'SHORT',
            'annual_carry': rate_differential
        }
    
    async def _analyze_currency_strength(self, base_currency: str, quote_currency: str) -> Dict[str, Any]:
        """Analyze individual currency strength."""
        try:
            # Get currency strength indices (placeholder implementation)
            base_strength = await self._get_currency_strength_index(base_currency)
            quote_strength = await self._get_currency_strength_index(quote_currency)
            
            relative_strength = base_strength - quote_strength
            
            return {
                'base_strength': base_strength,
                'quote_strength': quote_strength,
                'relative_strength': relative_strength
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing currency strength: {e}")
            return {
                'base_strength': 0,
                'quote_strength': 0,
                'relative_strength': 0
            }
    
    async def _get_currency_correlations(self, pair: str) -> Dict[str, float]:
        """Get correlations with other major pairs."""
        correlations = {}
        
        try:
            # Get correlation data for major pairs
            for major_pair in self.major_pairs:
                if major_pair != pair:
                    correlation = await self._calculate_pair_correlation(pair, major_pair)
                    correlations[major_pair] = correlation
                    
        except Exception as e:
            self.logger.error(f"Error getting correlations: {e}")
        
        return correlations
    
    def _analyze_trading_sessions(self, pair: str) -> Dict[str, Any]:
        """Analyze current and upcoming trading sessions."""
        current_time = datetime.utcnow().time()
        active_sessions = []
        session_volatility = {}
        
        for session_name, session_info in self.trading_sessions.items():
            start_time = datetime.strptime(session_info['start'], '%H:%M').time()
            end_time = datetime.strptime(session_info['end'], '%H:%M').time()
            
            # Check if session is active (handle overnight sessions)
            if start_time <= end_time:
                is_active = start_time <= current_time <= end_time
            else:  # Overnight session
                is_active = current_time >= start_time or current_time <= end_time
            
            if is_active:
                active_sessions.append(session_name)
            
            # Get historical volatility for this session (placeholder)
            session_volatility[session_name] = self._get_session_volatility(pair, session_name)
        
        return {
            'active_sessions': active_sessions,
            'session_volatility': session_volatility,
            'optimal_sessions': self._get_optimal_sessions(pair)
        }
    
    def _calculate_scores(self, technical: Dict[str, Any], fundamental: Dict[str, Any], carry: Dict[str, Any]) -> Dict[str, float]:
        """Calculate analysis scores."""
        # Technical score
        technical_score = 0
        
        # RSI score
        rsi = technical['rsi']
        if rsi < 30:
            rsi_score = 0.7  # Oversold - bullish
        elif rsi > 70:
            rsi_score = -0.7  # Overbought - bearish
        else:
            rsi_score = (50 - rsi) / 50  # Neutral zone
        
        technical_score += rsi_score * 0.3
        
        # MACD score
        macd_score = 0.5 if technical['macd_signal'] == 'BUY' else -0.5
        technical_score += macd_score * 0.3
        
        # Trend score
        trend_score = 0.6 if technical['trend_direction'] == 'UPTREND' else -0.6 if technical['trend_direction'] == 'DOWNTREND' else 0
        technical_score += trend_score * 0.4
        
        # Fundamental score
        fundamental_score = 0
        
        if fundamental['economic_impact'] == 'HIGH_POSITIVE':
            fundamental_score += 0.8
        elif fundamental['economic_impact'] == 'HIGH_NEGATIVE':
            fundamental_score -= 0.8
        elif fundamental['economic_impact'] == 'MODERATE_POSITIVE':
            fundamental_score += 0.4
        elif fundamental['economic_impact'] == 'MODERATE_NEGATIVE':
            fundamental_score -= 0.4
        
        if fundamental['cb_sentiment'] == 'HAWKISH':
            fundamental_score += 0.3
        elif fundamental['cb_sentiment'] == 'DOVISH':
            fundamental_score -= 0.3
        
        # Carry score
        carry_score = carry['carry_score']
        
        # Overall score
        overall_score = (
            technical_score * self.analysis_weights['technical'] +
            fundamental_score * self.analysis_weights['fundamental'] +
            carry_score * self.analysis_weights['carry']
        )
        
        return {
            'technical': technical_score,
            'fundamental': fundamental_score,
            'carry': carry_score,
            'overall': overall_score
        }
    
    def _generate_recommendation(self, scores: Dict[str, float], market_data: Dict[str, Any], session_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate trading recommendation."""
        overall_score = scores['overall']
        
        # Determine action
        if overall_score > 0.25:
            action = 'BUY'
            confidence = min(overall_score * 1.5, 0.95)
        elif overall_score < -0.25:
            action = 'SELL'
            confidence = min(abs(overall_score) * 1.5, 0.95)
        else:
            action = 'HOLD'
            confidence = 0.4
        
        # Determine risk level based on volatility and spread
        atr = market_data['historical_data']['high'].rolling(14).mean().iloc[-1] - market_data['historical_data']['low'].rolling(14).mean().iloc[-1]
        spread = market_data.get('spread', 0)
        
        if atr > 0.01 or spread > 0.0005:  # High volatility or wide spread
            risk_level = 'HIGH'
        elif atr > 0.005 or spread > 0.0002:
            risk_level = 'MEDIUM'
        else:
            risk_level = 'LOW'
        
        # Determine optimal timeframe
        if len(session_analysis['active_sessions']) >= 2:  # Session overlap
            optimal_timeframe = '15M'
        elif session_analysis['active_sessions']:
            optimal_timeframe = '1H'
        else:
            optimal_timeframe = '4H'
        
        return {
            'action': action,
            'confidence': confidence,
            'risk_level': risk_level,
            'optimal_timeframe': optimal_timeframe
        }
    
    def _format_analysis_report(self, result: ForexAnalysisResult) -> str:
        """Format the analysis result into a readable report."""
        report = f"""
🌍 FOREX ANALYSIS REPORT
{'=' * 50}

💱 CURRENCY PAIR INFORMATION
Pair: {result.pair}
Current Rate: {result.current_rate:.5f}
Daily Change: {result.daily_change:+.5f} ({result.daily_change_pct:+.2f}%)
Base Currency: {result.base_currency}
Quote Currency: {result.quote_currency}

📈 TECHNICAL ANALYSIS
RSI (14): {result.rsi:.1f} {'(Oversold)' if result.rsi < 30 else '(Overbought)' if result.rsi > 70 else '(Neutral)'}
MACD Signal: {result.macd_signal}
ATR: {result.atr:.5f}
Trend Direction: {result.trend_direction}

Pivot Points:
  Resistance 2: {result.pivot_points.get('R2', 0):.5f}
  Resistance 1: {result.pivot_points.get('R1', 0):.5f}
  Pivot: {result.pivot_points.get('PP', 0):.5f}
  Support 1: {result.pivot_points.get('S1', 0):.5f}
  Support 2: {result.pivot_points.get('S2', 0):.5f}

🏦 FUNDAMENTAL ANALYSIS
Interest Rate Differential: {result.interest_rate_differential:+.2f}%
Carry Trade Score: {result.carry_trade_score:+.2f}
Economic Calendar Impact: {result.economic_calendar_impact}
Central Bank Sentiment: {result.central_bank_sentiment}

💪 CURRENCY STRENGTH
{result.base_currency} Strength: {result.base_currency_strength:+.2f}
{result.quote_currency} Strength: {result.quote_currency_strength:+.2f}
Relative Strength: {result.relative_strength:+.2f}

🔗 MAJOR CORRELATIONS
"""
        
        for pair, correlation in list(result.major_correlations.items())[:5]:
            report += f"{pair}: {correlation:+.2f}\n"
        
        report += f"""
🕐 TRADING SESSIONS
Active Sessions: {', '.join(result.active_sessions) if result.active_sessions else 'None'}
Optimal Timeframe: {result.optimal_timeframe}

Session Volatility:
"""
        
        for session, volatility in result.session_volatility.items():
            report += f"  {session}: {volatility:.1%}\n"
        
        report += f"""
📊 ANALYSIS SCORES
Technical Score: {result.technical_score:+.2f}
Fundamental Score: {result.fundamental_score:+.2f}
Carry Score: {result.carry_score:+.2f}
Overall Score: {result.overall_score:+.2f}

🎯 RECOMMENDATION
Action: {result.recommendation}
Confidence: {result.confidence:.1%}
Risk Level: {result.risk_level}
Optimal Timeframe: {result.optimal_timeframe}

⚠️ FOREX TRADING CONSIDERATIONS
- Forex markets operate 24/5 with varying liquidity
- Major news events can cause significant volatility
- Consider economic calendar and central bank announcements
- Monitor session overlaps for optimal trading opportunities
- Use appropriate position sizing for currency volatility

Analysis Time: {result.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}
"""
        
        return report
    
    # Technical Analysis Helper Methods
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate MACD indicator."""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd_line = ema_fast - ema_slow
        macd_signal = macd_line.ewm(span=signal).mean()
        macd_histogram = macd_line - macd_signal
        return macd_line, macd_signal, macd_histogram
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range."""
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return true_range.rolling(window=period).mean()
    
    def _calculate_pivot_points(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate pivot points."""
        high = df['high'].iloc[-1]
        low = df['low'].iloc[-1]
        close = df['close'].iloc[-1]
        
        pivot = (high + low + close) / 3
        
        return {
            'PP': pivot,
            'R1': 2 * pivot - low,
            'R2': pivot + (high - low),
            'S1': 2 * pivot - high,
            'S2': pivot - (high - low)
        }
    
    def _determine_trend(self, df: pd.DataFrame, period: int = 20) -> str:
        """Determine trend direction."""
        sma = df['close'].rolling(window=period).mean()
        current_price = df['close'].iloc[-1]
        sma_current = sma.iloc[-1]
        sma_previous = sma.iloc[-5]  # 5 periods ago
        
        if current_price > sma_current and sma_current > sma_previous:
            return 'UPTREND'
        elif current_price < sma_current and sma_current < sma_previous:
            return 'DOWNTREND'
        else:
            return 'SIDEWAYS'
    
    def _find_support_resistance_forex(self, df: pd.DataFrame) -> Dict[str, List[float]]:
        """Find support and resistance levels for forex."""
        # Simplified implementation
        highs = df['high'].rolling(window=10, center=True).max()
        lows = df['low'].rolling(window=10, center=True).min()
        
        resistance_levels = df[df['high'] == highs]['high'].drop_duplicates().tail(3).tolist()
        support_levels = df[df['low'] == lows]['low'].drop_duplicates().tail(3).tolist()
        
        return {
            'support': sorted(support_levels, reverse=True),
            'resistance': sorted(resistance_levels)
        }
    
    def _calculate_volatility_forex(self, df: pd.DataFrame, period: int = 20) -> float:
        """Calculate forex-specific volatility."""
        returns = df['close'].pct_change()
        return returns.rolling(window=period).std().iloc[-1] * np.sqrt(252)  # Annualized
    
    # External API Methods (placeholders)
    async def _get_economic_calendar(self, base_currency: str, quote_currency: str) -> List[Dict[str, Any]]:
        """Get economic calendar events."""
        # Placeholder - implement with actual economic calendar API
        return [
            {
                'currency': base_currency,
                'event': 'GDP Release',
                'impact': 'HIGH',
                'forecast': '2.1%',
                'previous': '1.8%',
                'time': datetime.now() + timedelta(hours=2)
            }
        ]
    
    async def _analyze_central_bank_sentiment(self, base_currency: str, quote_currency: str) -> str:
        """Analyze central bank sentiment."""
        # Placeholder - implement with actual central bank analysis
        sentiments = ['HAWKISH', 'DOVISH', 'NEUTRAL']
        return np.random.choice(sentiments)
    
    async def _get_economic_indicators(self, base_currency: str, quote_currency: str) -> Dict[str, Any]:
        """Get economic indicators."""
        # Placeholder - implement with actual economic data API
        return {
            f'{base_currency}_gdp_growth': 2.1,
            f'{base_currency}_inflation': 3.2,
            f'{base_currency}_unemployment': 4.1,
            f'{quote_currency}_gdp_growth': 1.8,
            f'{quote_currency}_inflation': 2.8,
            f'{quote_currency}_unemployment': 5.2
        }
    
    def _assess_economic_impact(self, events: List[Dict[str, Any]], indicators: Dict[str, Any]) -> str:
        """Assess overall economic impact."""
        # Simplified assessment
        high_impact_events = [e for e in events if e.get('impact') == 'HIGH']
        
        if len(high_impact_events) >= 2:
            return 'HIGH_POSITIVE' if np.random.random() > 0.5 else 'HIGH_NEGATIVE'
        elif len(high_impact_events) == 1:
            return 'MODERATE_POSITIVE' if np.random.random() > 0.5 else 'MODERATE_NEGATIVE'
        else:
            return 'LOW'
    
    async def _get_currency_strength_index(self, currency: str) -> float:
        """Get currency strength index."""
        # Placeholder - implement with actual currency strength calculation
        return np.random.uniform(-1, 1)
    
    async def _calculate_pair_correlation(self, pair1: str, pair2: str) -> float:
        """Calculate correlation between two currency pairs."""
        # Placeholder - implement with actual correlation calculation
        return np.random.uniform(-1, 1)
    
    def _get_session_volatility(self, pair: str, session: str) -> float:
        """Get historical volatility for trading session."""
        # Placeholder - implement with actual session volatility calculation
        session_volatilities = {
            'Sydney': 0.15,
            'Tokyo': 0.25,
            'London': 0.45,
            'New York': 0.40
        }
        return session_volatilities.get(session, 0.20)
    
    def _get_optimal_sessions(self, pair: str) -> List[str]:
        """Get optimal trading sessions for the pair."""
        # Simplified logic based on currency pairs
        if 'USD' in pair and 'EUR' in pair:
            return ['London', 'New York']
        elif 'JPY' in pair:
            return ['Tokyo', 'London']
        elif 'AUD' in pair or 'NZD' in pair:
            return ['Sydney', 'Tokyo']
        else:
            return ['London', 'New York']


if __name__ == "__main__":
    # Example usage
    import asyncio
    from pathlib import Path
    
    async def test_forex_analysis():
        config_manager = ConfigManager(Path("../config"))
        tool = ForexAnalysisTool(config_manager)
        
        result = await tool._run('EUR/USD', '1H', 'standard')
        print(result)
    
    asyncio.run(test_forex_analysis())