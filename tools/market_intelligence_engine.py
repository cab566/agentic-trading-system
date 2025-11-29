#!/usr/bin/env python3
"""
Market Intelligence Engine for Advanced Trading System

This module provides comprehensive market intelligence capabilities:
- Real-time market sentiment analysis
- Economic indicator monitoring
- News impact assessment
- Market regime detection
- Cross-asset correlation analysis
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json

import pandas as pd
import numpy as np
from textblob import TextBlob

from core.config_manager import ConfigManager
from core.data_manager import UnifiedDataManager
from tools.market_data_tool import MarketDataTool
from utils.notifications import NotificationManager


class SentimentLevel(Enum):
    """Market sentiment levels."""
    VERY_BEARISH = "very_bearish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"
    BULLISH = "bullish"
    VERY_BULLISH = "very_bullish"


class MarketRegime(Enum):
    """Market regime types."""
    BULL_MARKET = "bull_market"
    BEAR_MARKET = "bear_market"
    SIDEWAYS = "sideways"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"


@dataclass
class MarketSentiment:
    """Market sentiment data structure."""
    symbol: str
    sentiment_score: float  # -1.0 to 1.0
    sentiment_level: SentimentLevel
    confidence: float
    news_count: int
    social_mentions: int = 0
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class EconomicIndicator:
    """Economic indicator data structure."""
    name: str
    value: float
    previous_value: float
    expected_value: Optional[float]
    impact_level: str  # 'low', 'medium', 'high'
    release_date: datetime
    market_impact: Optional[str] = None


@dataclass
class MarketIntelligence:
    """Comprehensive market intelligence report."""
    timestamp: datetime
    overall_sentiment: SentimentLevel
    market_regime: MarketRegime
    volatility_index: float
    fear_greed_index: float
    economic_indicators: List[EconomicIndicator]
    sector_sentiments: Dict[str, MarketSentiment]
    key_events: List[str]
    risk_factors: List[str]
    opportunities: List[str]


class MarketIntelligenceEngine:
    """Advanced market intelligence and analysis engine."""
    
    def __init__(self, config_manager: ConfigManager):
        self.config_manager = config_manager
        self.data_manager = UnifiedDataManager(config_manager)
        self.market_data_tool = MarketDataTool(config_manager)
        self.notification_manager = NotificationManager(config_manager)
        self.logger = logging.getLogger(__name__)
        
        # Intelligence parameters
        self.sentiment_lookback = 7  # days
        self.volatility_window = 20  # days
        self.correlation_window = 60  # days
        
        # Cached data
        self.sentiment_cache = {}
        self.regime_history = []
        self.last_update = None
    
    async def generate_market_intelligence(self) -> MarketIntelligence:
        """Generate comprehensive market intelligence report."""
        try:
            # Get current market data
            market_data = await self._collect_market_data()
            
            # Analyze sentiment
            overall_sentiment = await self._analyze_overall_sentiment()
            sector_sentiments = await self._analyze_sector_sentiments()
            
            # Detect market regime
            market_regime = await self._detect_market_regime(market_data)
            
            # Calculate market indicators
            volatility_index = await self._calculate_volatility_index(market_data)
            fear_greed_index = await self._calculate_fear_greed_index(market_data)
            
            # Get economic indicators
            economic_indicators = await self._get_economic_indicators()
            
            # Identify key events and risks
            key_events = await self._identify_key_events()
            risk_factors = await self._identify_risk_factors(market_data)
            opportunities = await self._identify_opportunities(market_data)
            
            return MarketIntelligence(
                timestamp=datetime.now(),
                overall_sentiment=overall_sentiment,
                market_regime=market_regime,
                volatility_index=volatility_index,
                fear_greed_index=fear_greed_index,
                economic_indicators=economic_indicators,
                sector_sentiments=sector_sentiments,
                key_events=key_events,
                risk_factors=risk_factors,
                opportunities=opportunities
            )
            
        except Exception as e:
            self.logger.error(f"Error generating market intelligence: {e}")
            raise
    
    async def _collect_market_data(self) -> Dict[str, pd.DataFrame]:
        """Collect real market data for analysis."""
        market_data = {}
        
        # Major indices
        indices = ['SPY', 'QQQ', 'IWM', 'VIX']
        for symbol in indices:
            try:
                data = await self.data_manager.get_price_data(
                    symbol, 
                    timeframe='1d',
                    limit=self.correlation_window
                )
                if data is not None:
                    market_data[symbol] = data
            except Exception as e:
                self.logger.warning(f"Could not fetch data for {symbol}: {e}")
        
        return market_data
    
    async def _analyze_overall_sentiment(self) -> SentimentLevel:
        """Analyze overall market sentiment from news and data."""
        try:
            # Get recent news
            news_data = await self.data_manager.get_news_data(
                query="market economy stocks",
                limit=100,
                days_back=self.sentiment_lookback
            )
            
            if not news_data:
                return SentimentLevel.NEUTRAL
            
            # Analyze sentiment of news headlines and content
            sentiment_scores = []
            for article in news_data:
                text = f"{article.get('title', '')} {article.get('summary', '')}"
                if text.strip():
                    blob = TextBlob(text)
                    sentiment_scores.append(blob.sentiment.polarity)
            
            if not sentiment_scores:
                return SentimentLevel.NEUTRAL
            
            avg_sentiment = np.mean(sentiment_scores)
            
            # Convert to sentiment level
            if avg_sentiment >= 0.3:
                return SentimentLevel.VERY_BULLISH
            elif avg_sentiment >= 0.1:
                return SentimentLevel.BULLISH
            elif avg_sentiment <= -0.3:
                return SentimentLevel.VERY_BEARISH
            elif avg_sentiment <= -0.1:
                return SentimentLevel.BEARISH
            else:
                return SentimentLevel.NEUTRAL
                
        except Exception as e:
            self.logger.error(f"Error analyzing overall sentiment: {e}")
            return SentimentLevel.NEUTRAL
    
    async def _analyze_sector_sentiments(self) -> Dict[str, MarketSentiment]:
        """Analyze sentiment for different market sectors."""
        sector_sentiments = {}
        
        # Define sector ETFs
        sectors = {
            'Technology': 'XLK',
            'Healthcare': 'XLV',
            'Financial': 'XLF',
            'Energy': 'XLE',
            'Consumer': 'XLY',
            'Utilities': 'XLU',
            'Industrial': 'XLI',
            'Materials': 'XLB'
        }
        
        for sector_name, etf_symbol in sectors.items():
            try:
                # Get sector-specific news
                news_data = await self.data_manager.get_news_data(
                    query=f"{sector_name} sector stocks",
                    limit=50,
                    days_back=self.sentiment_lookback
                )
                
                sentiment_scores = []
                for article in news_data:
                    text = f"{article.get('title', '')} {article.get('summary', '')}"
                    if text.strip():
                        blob = TextBlob(text)
                        sentiment_scores.append(blob.sentiment.polarity)
                
                if sentiment_scores:
                    avg_sentiment = np.mean(sentiment_scores)
                    confidence = min(1.0, len(sentiment_scores) / 20.0)  # More news = higher confidence
                    
                    # Convert to sentiment level
                    if avg_sentiment >= 0.2:
                        level = SentimentLevel.VERY_BULLISH
                    elif avg_sentiment >= 0.05:
                        level = SentimentLevel.BULLISH
                    elif avg_sentiment <= -0.2:
                        level = SentimentLevel.VERY_BEARISH
                    elif avg_sentiment <= -0.05:
                        level = SentimentLevel.BEARISH
                    else:
                        level = SentimentLevel.NEUTRAL
                    
                    sector_sentiments[sector_name] = MarketSentiment(
                        symbol=etf_symbol,
                        sentiment_score=avg_sentiment,
                        sentiment_level=level,
                        confidence=confidence,
                        news_count=len(news_data)
                    )
                    
            except Exception as e:
                self.logger.warning(f"Error analyzing sentiment for {sector_name}: {e}")
        
        return sector_sentiments
    
    async def _detect_market_regime(self, market_data: Dict[str, pd.DataFrame]) -> MarketRegime:
        """Detect current market regime based on price action and volatility."""
        try:
            if 'SPY' not in market_data:
                return MarketRegime.SIDEWAYS
            
            spy_data = market_data['SPY']
            if len(spy_data) < 50:
                return MarketRegime.SIDEWAYS
            
            # Calculate trend indicators
            current_price = spy_data['close'].iloc[-1]
            sma_20 = spy_data['close'].rolling(20).mean().iloc[-1]
            sma_50 = spy_data['close'].rolling(50).mean().iloc[-1]
            
            # Calculate volatility
            returns = spy_data['close'].pct_change().dropna()
            volatility = returns.rolling(20).std().iloc[-1] * np.sqrt(252)
            
            # Determine regime
            if current_price > sma_20 > sma_50 and volatility < 0.20:
                return MarketRegime.BULL_MARKET
            elif current_price < sma_20 < sma_50 and volatility < 0.25:
                return MarketRegime.BEAR_MARKET
            elif volatility > 0.30:
                return MarketRegime.HIGH_VOLATILITY
            elif volatility < 0.15:
                return MarketRegime.LOW_VOLATILITY
            else:
                return MarketRegime.SIDEWAYS
                
        except Exception as e:
            self.logger.error(f"Error detecting market regime: {e}")
            return MarketRegime.SIDEWAYS
    
    async def _calculate_volatility_index(self, market_data: Dict[str, pd.DataFrame]) -> float:
        """Calculate market volatility index."""
        try:
            if 'VIX' in market_data:
                return float(market_data['VIX']['close'].iloc[-1])
            elif 'SPY' in market_data:
                # Calculate implied volatility from SPY returns
                spy_returns = market_data['SPY']['close'].pct_change().dropna()
                volatility = spy_returns.rolling(20).std().iloc[-1] * np.sqrt(252) * 100
                return float(volatility)
            else:
                return 20.0  # Default moderate volatility
        except Exception as e:
            self.logger.error(f"Error calculating volatility index: {e}")
            return 20.0
    
    async def _calculate_fear_greed_index(self, market_data: Dict[str, pd.DataFrame]) -> float:
        """Calculate fear and greed index based on multiple factors."""
        try:
            factors = []
            
            # VIX factor (fear indicator)
            if 'VIX' in market_data:
                vix = market_data['VIX']['close'].iloc[-1]
                vix_factor = max(0, min(100, 100 - (vix - 10) * 2))  # Invert VIX
                factors.append(vix_factor)
            
            # Market momentum factor
            if 'SPY' in market_data:
                spy_data = market_data['SPY']
                current_price = spy_data['close'].iloc[-1]
                price_20_days_ago = spy_data['close'].iloc[-20]
                momentum = (current_price - price_20_days_ago) / price_20_days_ago
                momentum_factor = 50 + momentum * 100  # Convert to 0-100 scale
                momentum_factor = max(0, min(100, momentum_factor))
                factors.append(momentum_factor)
            
            # Breadth factor (using QQQ vs SPY)
            if 'QQQ' in market_data and 'SPY' in market_data:
                qqq_return = market_data['QQQ']['close'].pct_change(5).iloc[-1]
                spy_return = market_data['SPY']['close'].pct_change(5).iloc[-1]
                breadth_factor = 50 + (qqq_return - spy_return) * 500
                breadth_factor = max(0, min(100, breadth_factor))
                factors.append(breadth_factor)
            
            if factors:
                return float(np.mean(factors))
            else:
                return 50.0  # Neutral
                
        except Exception as e:
            self.logger.error(f"Error calculating fear/greed index: {e}")
            return 50.0
    
    async def _get_economic_indicators(self) -> List[EconomicIndicator]:
        """Get recent economic indicators from real data sources."""
        # This would integrate with real economic data APIs like FRED, Bloomberg, etc.
        # For now, return empty list as we focus on market data from Alpaca
        return []
    
    async def _identify_key_events(self) -> List[str]:
        """Identify key market events from news."""
        try:
            news_data = await self.data_manager.get_news_data(
                query="market breaking news earnings fed",
                limit=20,
                days_back=1
            )
            
            key_events = []
            for article in news_data[:10]:  # Top 10 most recent
                title = article.get('title', '')
                if any(keyword in title.lower() for keyword in 
                      ['fed', 'earnings', 'breaking', 'alert', 'urgent', 'major']):
                    key_events.append(title)
            
            return key_events
            
        except Exception as e:
            self.logger.error(f"Error identifying key events: {e}")
            return []
    
    async def _identify_risk_factors(self, market_data: Dict[str, pd.DataFrame]) -> List[str]:
        """Identify current market risk factors."""
        risk_factors = []
        
        try:
            # High volatility risk
            if 'VIX' in market_data:
                vix = market_data['VIX']['close'].iloc[-1]
                if vix > 30:
                    risk_factors.append(f"High market volatility (VIX: {vix:.1f})")
            
            # Market correlation risk
            if len(market_data) >= 2:
                correlations = []
                symbols = list(market_data.keys())
                for i in range(len(symbols)):
                    for j in range(i+1, len(symbols)):
                        if len(market_data[symbols[i]]) > 20 and len(market_data[symbols[j]]) > 20:
                            corr = market_data[symbols[i]]['close'].corr(
                                market_data[symbols[j]]['close']
                            )
                            correlations.append(abs(corr))
                
                if correlations and np.mean(correlations) > 0.8:
                    risk_factors.append("High cross-asset correlation detected")
            
            # Trend reversal risk
            if 'SPY' in market_data:
                spy_data = market_data['SPY']
                if len(spy_data) > 10:
                    recent_returns = spy_data['close'].pct_change().iloc[-10:]
                    if recent_returns.std() > 0.02:  # High recent volatility
                        risk_factors.append("Increased short-term volatility in major indices")
            
        except Exception as e:
            self.logger.error(f"Error identifying risk factors: {e}")
        
        return risk_factors
    
    async def _identify_opportunities(self, market_data: Dict[str, pd.DataFrame]) -> List[str]:
        """Identify current market opportunities."""
        opportunities = []
        
        try:
            # Low volatility opportunity
            if 'VIX' in market_data:
                vix = market_data['VIX']['close'].iloc[-1]
                if vix < 15:
                    opportunities.append(f"Low volatility environment (VIX: {vix:.1f}) - favorable for risk-taking")
            
            # Momentum opportunities
            for symbol, data in market_data.items():
                if len(data) > 20:
                    current_price = data['close'].iloc[-1]
                    sma_20 = data['close'].rolling(20).mean().iloc[-1]
                    
                    if current_price > sma_20 * 1.02:  # 2% above 20-day average
                        opportunities.append(f"{symbol} showing strong momentum above 20-day average")
            
            # Mean reversion opportunities
            for symbol, data in market_data.items():
                if len(data) > 50:
                    current_price = data['close'].iloc[-1]
                    sma_50 = data['close'].rolling(50).mean().iloc[-1]
                    std_50 = data['close'].rolling(50).std().iloc[-1]
                    
                    if current_price < sma_50 - 2 * std_50:  # 2 std devs below mean
                        opportunities.append(f"{symbol} potentially oversold - mean reversion opportunity")
            
        except Exception as e:
            self.logger.error(f"Error identifying opportunities: {e}")
        
        return opportunities
    
    async def get_symbol_intelligence(self, symbol: str) -> Dict[str, Any]:
        """Get detailed intelligence for a specific symbol."""
        try:
            # Get price data
            price_data = await self.data_manager.get_price_data(
                symbol, 
                timeframe='1d',
                limit=60
            )
            
            # Get news data
            news_data = await self.data_manager.get_news_data(
                query=symbol,
                limit=20,
                days_back=7
            )
            
            # Analyze sentiment
            sentiment_scores = []
            if news_data:
                for article in news_data:
                    text = f"{article.get('title', '')} {article.get('summary', '')}"
                    if text.strip():
                        blob = TextBlob(text)
                        sentiment_scores.append(blob.sentiment.polarity)
            
            # Calculate technical indicators
            technical_analysis = {}
            if price_data is not None and len(price_data) > 20:
                current_price = price_data['close'].iloc[-1]
                sma_20 = price_data['close'].rolling(20).mean().iloc[-1]
                sma_50 = price_data['close'].rolling(50).mean().iloc[-1] if len(price_data) > 50 else None
                
                returns = price_data['close'].pct_change().dropna()
                volatility = returns.rolling(20).std().iloc[-1] * np.sqrt(252) if len(returns) > 20 else None
                
                technical_analysis = {
                    'current_price': float(current_price),
                    'sma_20': float(sma_20),
                    'sma_50': float(sma_50) if sma_50 is not None else None,
                    'volatility': float(volatility) if volatility is not None else None,
                    'trend': 'bullish' if current_price > sma_20 else 'bearish'
                }
            
            return {
                'symbol': symbol,
                'sentiment_score': np.mean(sentiment_scores) if sentiment_scores else 0.0,
                'news_count': len(news_data) if news_data else 0,
                'technical_analysis': technical_analysis,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error getting intelligence for {symbol}: {e}")
            return {'symbol': symbol, 'error': str(e)}