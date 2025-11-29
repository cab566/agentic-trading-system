#!/usr/bin/env python3
"""
Real-Time Opportunity Scanner

This tool continuously monitors multiple data sources to identify real-time trading opportunities
including unusual volume, price movements, news events, and social sentiment spikes.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import json
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import threading
import time

# CrewAI imports
from crewai_tools import BaseTool
from pydantic import BaseModel, Field

# Local imports
from ..core.data_manager import UnifiedDataManager
from ..core.config_manager import ConfigManager
from .enhanced_web_scraper import EnhancedWebScraperTool
from .news_driven_discovery import NewsDrivenDiscovery


class OpportunityType(Enum):
    """Types of trading opportunities"""
    VOLUME_SPIKE = "volume_spike"
    PRICE_BREAKOUT = "price_breakout"
    NEWS_EVENT = "news_event"
    SENTIMENT_SHIFT = "sentiment_shift"
    INSIDER_ACTIVITY = "insider_activity"
    UNUSUAL_OPTIONS = "unusual_options"
    EARNINGS_SURPRISE = "earnings_surprise"
    ANALYST_UPGRADE = "analyst_upgrade"
    TECHNICAL_PATTERN = "technical_pattern"
    ARBITRAGE = "arbitrage"


class AlertSeverity(Enum):
    """Alert severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class TradingOpportunity:
    """Trading opportunity data structure"""
    symbol: str
    opportunity_type: OpportunityType
    severity: AlertSeverity
    timestamp: datetime
    price: float
    volume: int
    description: str
    confidence: float
    expected_move: float
    timeframe: str
    data_sources: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)


class OpportunityScanner(BaseModel):
    """Configuration for opportunity scanning"""
    name: str = Field(description="Scanner name")
    enabled: bool = Field(default=True, description="Whether scanner is enabled")
    scan_interval: int = Field(default=60, description="Scan interval in seconds")
    symbols: List[str] = Field(default=[], description="Symbols to monitor")
    thresholds: Dict[str, float] = Field(default={}, description="Alert thresholds")
    data_sources: List[str] = Field(default=[], description="Data sources to use")


class RealTimeOpportunityScannerTool(BaseTool):
    """Real-time opportunity scanner for trading signals"""
    
    name: str = "real_time_opportunity_scanner"
    description: str = "Continuously monitors multiple data sources for real-time trading opportunities including volume spikes, breakouts, news events, and sentiment shifts"
    
    def __init__(self, config_manager: ConfigManager, data_manager: UnifiedDataManager):
        super().__init__()
        self.config_manager = config_manager
        self.data_manager = data_manager
        self.logger = logging.getLogger(__name__)
        
        # Load configuration
        self.config = self._load_config()
        
        # Initialize scanners
        self.scanners = self._initialize_scanners()
        
        # Initialize tools
        self.web_scraper = EnhancedWebScraperTool(config_manager, data_manager)
        self.news_discovery = NewsDrivenDiscovery(config_manager, data_manager)
        
        # Opportunity storage
        self.opportunities: List[TradingOpportunity] = []
        self.opportunity_callbacks: List[Callable] = []
        
        # Scanning state
        self.is_scanning = False
        self.scan_tasks = []
        
        # Performance tracking
        self.scan_stats = {
            'total_scans': 0,
            'opportunities_found': 0,
            'false_positives': 0,
            'last_scan_time': None
        }
    
    def _load_config(self) -> Dict[str, Any]:
        """Load scanner configuration"""
        try:
            config = self.config_manager.get_config('opportunity_scanner')
            return config
        except Exception as e:
            self.logger.warning(f"Could not load opportunity scanner config: {e}")
            return {
                'scan_interval': 60,
                'max_opportunities': 100,
                'alert_thresholds': {
                    'volume_spike_multiplier': 3.0,
                    'price_change_threshold': 0.05,
                    'sentiment_change_threshold': 0.3
                },
                'symbols': ['SPY', 'QQQ', 'AAPL', 'TSLA', 'NVDA', 'MSFT', 'GOOGL', 'AMZN']
            }
    
    def _initialize_scanners(self) -> Dict[str, OpportunityScanner]:
        """Initialize opportunity scanners"""
        scanners = {}
        
        # Volume spike scanner
        scanners['volume_spike'] = OpportunityScanner(
            name="Volume Spike Scanner",
            scan_interval=30,
            symbols=self.config.get('symbols', []),
            thresholds={
                'volume_multiplier': 3.0,
                'min_price': 1.0,
                'min_volume': 100000
            },
            data_sources=['yfinance', 'polygon']
        )
        
        # Price breakout scanner
        scanners['price_breakout'] = OpportunityScanner(
            name="Price Breakout Scanner",
            scan_interval=60,
            symbols=self.config.get('symbols', []),
            thresholds={
                'breakout_threshold': 0.05,
                'volume_confirmation': 1.5,
                'rsi_threshold': 70
            },
            data_sources=['yfinance', 'polygon']
        )
        
        # News event scanner
        scanners['news_events'] = OpportunityScanner(
            name="News Event Scanner",
            scan_interval=120,
            symbols=self.config.get('symbols', []),
            thresholds={
                'sentiment_threshold': 0.3,
                'impact_threshold': 0.5
            },
            data_sources=['news_api', 'seeking_alpha', 'reddit']
        )
        
        # Sentiment shift scanner
        scanners['sentiment_shift'] = OpportunityScanner(
            name="Sentiment Shift Scanner",
            scan_interval=300,
            symbols=self.config.get('symbols', []),
            thresholds={
                'sentiment_change': 0.4,
                'mention_threshold': 10
            },
            data_sources=['reddit', 'twitter', 'stocktwits']
        )
        
        # Insider activity scanner
        scanners['insider_activity'] = OpportunityScanner(
            name="Insider Activity Scanner",
            scan_interval=3600,  # Hourly
            symbols=self.config.get('symbols', []),
            thresholds={
                'transaction_size': 1000000,  # $1M+
                'insider_confidence': 0.7
            },
            data_sources=['sec_edgar', 'insider_monkey']
        )
        
        return scanners
    
    def _run(self, action: str, parameters: Dict[str, Any] = None) -> str:
        """Synchronous scanner execution"""
        return asyncio.run(self._arun(action, parameters))
    
    async def _arun(self, action: str, parameters: Dict[str, Any] = None) -> str:
        """Asynchronous scanner execution"""
        try:
            parameters = parameters or {}
            
            if action == 'start_scanning':
                return await self._start_scanning()
            elif action == 'stop_scanning':
                return await self._stop_scanning()
            elif action == 'get_opportunities':
                return await self._get_opportunities(parameters)
            elif action == 'scan_once':
                return await self._scan_once(parameters)
            elif action == 'get_stats':
                return await self._get_stats()
            else:
                return json.dumps({
                    'error': f'Unknown action: {action}',
                    'available_actions': ['start_scanning', 'stop_scanning', 'get_opportunities', 'scan_once', 'get_stats']
                })
                
        except Exception as e:
            self.logger.error(f"Scanner action failed: {e}")
            return json.dumps({'error': str(e)})
    
    async def _start_scanning(self) -> str:
        """Start continuous scanning"""
        if self.is_scanning:
            return json.dumps({'message': 'Scanner already running'})
        
        self.is_scanning = True
        self.scan_tasks = []
        
        # Start scanner tasks
        for scanner_name, scanner in self.scanners.items():
            if scanner.enabled:
                task = asyncio.create_task(self._run_scanner(scanner_name, scanner))
                self.scan_tasks.append(task)
        
        self.logger.info(f"Started {len(self.scan_tasks)} opportunity scanners")
        
        return json.dumps({
            'message': 'Opportunity scanning started',
            'active_scanners': list(self.scanners.keys()),
            'scan_intervals': {name: scanner.scan_interval for name, scanner in self.scanners.items()}
        })
    
    async def _stop_scanning(self) -> str:
        """Stop continuous scanning"""
        if not self.is_scanning:
            return json.dumps({'message': 'Scanner not running'})
        
        self.is_scanning = False
        
        # Cancel all scanner tasks
        for task in self.scan_tasks:
            task.cancel()
        
        # Wait for tasks to complete
        await asyncio.gather(*self.scan_tasks, return_exceptions=True)
        
        self.logger.info("Stopped opportunity scanning")
        
        return json.dumps({
            'message': 'Opportunity scanning stopped',
            'total_opportunities_found': len(self.opportunities)
        })
    
    async def _run_scanner(self, scanner_name: str, scanner: OpportunityScanner):
        """Run individual scanner continuously"""
        while self.is_scanning:
            try:
                await self._execute_scanner(scanner_name, scanner)
                await asyncio.sleep(scanner.scan_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Scanner {scanner_name} error: {e}")
                await asyncio.sleep(scanner.scan_interval)
    
    async def _execute_scanner(self, scanner_name: str, scanner: OpportunityScanner):
        """Execute a single scanner"""
        self.scan_stats['total_scans'] += 1
        self.scan_stats['last_scan_time'] = datetime.now()
        
        if scanner_name == 'volume_spike':
            opportunities = await self._scan_volume_spikes(scanner)
        elif scanner_name == 'price_breakout':
            opportunities = await self._scan_price_breakouts(scanner)
        elif scanner_name == 'news_events':
            opportunities = await self._scan_news_events(scanner)
        elif scanner_name == 'sentiment_shift':
            opportunities = await self._scan_sentiment_shifts(scanner)
        elif scanner_name == 'insider_activity':
            opportunities = await self._scan_insider_activity(scanner)
        else:
            opportunities = []
        
        # Add new opportunities
        for opportunity in opportunities:
            self._add_opportunity(opportunity)
        
        if opportunities:
            self.logger.info(f"Scanner {scanner_name} found {len(opportunities)} opportunities")
    
    async def _scan_volume_spikes(self, scanner: OpportunityScanner) -> List[TradingOpportunity]:
        """Scan for unusual volume spikes"""
        opportunities = []
        
        try:
            for symbol in scanner.symbols:
                # Get recent volume data
                end_date = datetime.now()
                start_date = end_date - timedelta(days=5)
                
                price_data = await self.data_manager.get_price_data(
                    symbol, '1m', start_date, end_date
                )
                
                if price_data.data.empty:
                    continue
                
                df = price_data.data
                current_volume = df['Volume'].iloc[-1]
                avg_volume = df['Volume'].rolling(window=20).mean().iloc[-1]
                
                volume_multiplier = current_volume / avg_volume if avg_volume > 0 else 0
                
                # Check for volume spike
                if volume_multiplier >= scanner.thresholds['volume_multiplier']:
                    current_price = df['Close'].iloc[-1]
                    price_change = (current_price - df['Close'].iloc[-2]) / df['Close'].iloc[-2]
                    
                    severity = AlertSeverity.HIGH if volume_multiplier > 5 else AlertSeverity.MEDIUM
                    
                    opportunity = TradingOpportunity(
                        symbol=symbol,
                        opportunity_type=OpportunityType.VOLUME_SPIKE,
                        severity=severity,
                        timestamp=datetime.now(),
                        price=current_price,
                        volume=int(current_volume),
                        description=f"Volume spike: {volume_multiplier:.1f}x average volume",
                        confidence=min(0.9, volume_multiplier / 10),
                        expected_move=abs(price_change) * 2,  # Estimate continued move
                        timeframe="1-4 hours",
                        data_sources=['yfinance'],
                        metadata={
                            'volume_multiplier': volume_multiplier,
                            'avg_volume': avg_volume,
                            'price_change': price_change
                        }
                    )
                    opportunities.append(opportunity)
        
        except Exception as e:
            self.logger.error(f"Volume spike scanning failed: {e}")
        
        return opportunities
    
    async def _scan_price_breakouts(self, scanner: OpportunityScanner) -> List[TradingOpportunity]:
        """Scan for price breakouts"""
        opportunities = []
        
        try:
            for symbol in scanner.symbols:
                # Get price data for technical analysis
                end_date = datetime.now()
                start_date = end_date - timedelta(days=30)
                
                price_data = await self.data_manager.get_price_data(
                    symbol, '1h', start_date, end_date
                )
                
                if price_data.data.empty:
                    continue
                
                df = price_data.data
                
                # Calculate technical indicators
                df['SMA_20'] = df['Close'].rolling(window=20).mean()
                df['SMA_50'] = df['Close'].rolling(window=50).mean()
                df['RSI'] = self._calculate_rsi(df['Close'])
                
                # Check for breakout conditions
                current_price = df['Close'].iloc[-1]
                sma_20 = df['SMA_20'].iloc[-1]
                sma_50 = df['SMA_50'].iloc[-1]
                rsi = df['RSI'].iloc[-1]
                
                # Bullish breakout: price above both SMAs, RSI not overbought
                if (current_price > sma_20 > sma_50 and 
                    rsi < scanner.thresholds['rsi_threshold'] and
                    current_price > df['High'].rolling(window=20).max().iloc[-2]):
                    
                    price_change = (current_price - df['Close'].iloc[-2]) / df['Close'].iloc[-2]
                    
                    if abs(price_change) >= scanner.thresholds['breakout_threshold']:
                        opportunity = TradingOpportunity(
                            symbol=symbol,
                            opportunity_type=OpportunityType.PRICE_BREAKOUT,
                            severity=AlertSeverity.HIGH,
                            timestamp=datetime.now(),
                            price=current_price,
                            volume=int(df['Volume'].iloc[-1]),
                            description=f"Bullish breakout above 20-day high",
                            confidence=0.75,
                            expected_move=price_change * 1.5,
                            timeframe="1-3 days",
                            data_sources=['yfinance'],
                            metadata={
                                'rsi': rsi,
                                'sma_20': sma_20,
                                'sma_50': sma_50,
                                'breakout_level': df['High'].rolling(window=20).max().iloc[-2]
                            }
                        )
                        opportunities.append(opportunity)
        
        except Exception as e:
            self.logger.error(f"Price breakout scanning failed: {e}")
        
        return opportunities
    
    async def _scan_news_events(self, scanner: OpportunityScanner) -> List[TradingOpportunity]:
        """Scan for significant news events"""
        opportunities = []
        
        try:
            # Use news discovery tool
            for symbol in scanner.symbols[:5]:  # Limit to avoid rate limits
                news_events = await self.news_discovery.discover_opportunities(
                    symbol, hours_back=2
                )
                
                for event in news_events:
                    if (hasattr(event, 'impact_score') and 
                        event.impact_score >= scanner.thresholds['impact_threshold']):
                        
                        severity = AlertSeverity.CRITICAL if event.impact_score > 0.8 else AlertSeverity.HIGH
                        
                        opportunity = TradingOpportunity(
                            symbol=symbol,
                            opportunity_type=OpportunityType.NEWS_EVENT,
                            severity=severity,
                            timestamp=datetime.now(),
                            price=0.0,  # Will be filled by price lookup
                            volume=0,
                            description=f"News event: {event.headline[:100]}...",
                            confidence=event.confidence,
                            expected_move=event.impact_score * 0.1,  # Estimate 10% max move
                            timeframe="immediate to 1 day",
                            data_sources=['news_api'],
                            metadata={
                                'headline': event.headline,
                                'sentiment': event.sentiment,
                                'source': event.source
                            }
                        )
                        opportunities.append(opportunity)
        
        except Exception as e:
            self.logger.error(f"News event scanning failed: {e}")
        
        return opportunities
    
    async def _scan_sentiment_shifts(self, scanner: OpportunityScanner) -> List[TradingOpportunity]:
        """Scan for sentiment shifts"""
        opportunities = []
        
        try:
            for symbol in scanner.symbols[:3]:  # Limit for performance
                # Get Reddit sentiment
                reddit_data = await self.web_scraper._arun('reddit_wsb', symbol)
                reddit_result = json.loads(reddit_data)
                
                if reddit_result.get('success') and reddit_result.get('data'):
                    sentiment_data = reddit_result['data']
                    sentiment_score = sentiment_data.get('sentiment_score', 0)
                    mentions = sentiment_data.get('mentions', 0)
                    
                    # Check for significant sentiment
                    if (abs(sentiment_score) >= scanner.thresholds['sentiment_change'] and
                        mentions >= scanner.thresholds['mention_threshold']):
                        
                        severity = AlertSeverity.HIGH if abs(sentiment_score) > 0.6 else AlertSeverity.MEDIUM
                        
                        opportunity = TradingOpportunity(
                            symbol=symbol,
                            opportunity_type=OpportunityType.SENTIMENT_SHIFT,
                            severity=severity,
                            timestamp=datetime.now(),
                            price=0.0,
                            volume=0,
                            description=f"Sentiment shift: {sentiment_score:.2f} score, {mentions} mentions",
                            confidence=min(0.8, mentions / 50),
                            expected_move=abs(sentiment_score) * 0.05,
                            timeframe="1-2 days",
                            data_sources=['reddit'],
                            metadata={
                                'sentiment_score': sentiment_score,
                                'mentions': mentions,
                                'posts': sentiment_data.get('posts', [])[:3]
                            }
                        )
                        opportunities.append(opportunity)
        
        except Exception as e:
            self.logger.error(f"Sentiment shift scanning failed: {e}")
        
        return opportunities
    
    async def _scan_insider_activity(self, scanner: OpportunityScanner) -> List[TradingOpportunity]:
        """Scan for insider trading activity"""
        opportunities = []
        
        try:
            for symbol in scanner.symbols[:5]:  # Limit for performance
                # Get insider trading data
                insider_data = await self.web_scraper._arun('insider_trading', symbol)
                insider_result = json.loads(insider_data)
                
                if insider_result.get('success') and insider_result.get('data'):
                    trades = insider_result['data']
                    
                    # Look for significant recent trades
                    for trade in trades[:3]:  # Recent trades
                        if 'buy' in trade.get('transaction', '').lower():
                            # Estimate trade value (simplified)
                            shares = self._parse_number(trade.get('shares', '0'))
                            price = self._parse_number(trade.get('price', '0'))
                            trade_value = shares * price
                            
                            if trade_value >= scanner.thresholds['transaction_size']:
                                opportunity = TradingOpportunity(
                                    symbol=symbol,
                                    opportunity_type=OpportunityType.INSIDER_ACTIVITY,
                                    severity=AlertSeverity.HIGH,
                                    timestamp=datetime.now(),
                                    price=price,
                                    volume=0,
                                    description=f"Insider buy: {trade.get('insider', 'Unknown')} - ${trade_value:,.0f}",
                                    confidence=scanner.thresholds['insider_confidence'],
                                    expected_move=0.03,  # Conservative 3% estimate
                                    timeframe="1-4 weeks",
                                    data_sources=['insider_monkey'],
                                    metadata={
                                        'insider': trade.get('insider'),
                                        'position': trade.get('position'),
                                        'shares': shares,
                                        'trade_value': trade_value
                                    }
                                )
                                opportunities.append(opportunity)
        
        except Exception as e:
            self.logger.error(f"Insider activity scanning failed: {e}")
        
        return opportunities
    
    def _add_opportunity(self, opportunity: TradingOpportunity):
        """Add opportunity to storage"""
        # Check for duplicates
        for existing in self.opportunities:
            if (existing.symbol == opportunity.symbol and
                existing.opportunity_type == opportunity.opportunity_type and
                (opportunity.timestamp - existing.timestamp).seconds < 3600):  # Within 1 hour
                return  # Skip duplicate
        
        self.opportunities.append(opportunity)
        self.scan_stats['opportunities_found'] += 1
        
        # Limit storage size
        max_opportunities = self.config.get('max_opportunities', 100)
        if len(self.opportunities) > max_opportunities:
            self.opportunities = self.opportunities[-max_opportunities:]
        
        # Trigger callbacks
        for callback in self.opportunity_callbacks:
            try:
                callback(opportunity)
            except Exception as e:
                self.logger.error(f"Opportunity callback failed: {e}")
    
    async def _get_opportunities(self, parameters: Dict[str, Any]) -> str:
        """Get current opportunities"""
        limit = parameters.get('limit', 20)
        severity_filter = parameters.get('severity')
        symbol_filter = parameters.get('symbol')
        
        filtered_opportunities = self.opportunities
        
        # Apply filters
        if severity_filter:
            filtered_opportunities = [
                opp for opp in filtered_opportunities 
                if opp.severity.value == severity_filter
            ]
        
        if symbol_filter:
            filtered_opportunities = [
                opp for opp in filtered_opportunities 
                if opp.symbol == symbol_filter.upper()
            ]
        
        # Sort by timestamp (newest first) and limit
        filtered_opportunities.sort(key=lambda x: x.timestamp, reverse=True)
        filtered_opportunities = filtered_opportunities[:limit]
        
        # Convert to JSON-serializable format
        opportunities_data = []
        for opp in filtered_opportunities:
            opportunities_data.append({
                'symbol': opp.symbol,
                'type': opp.opportunity_type.value,
                'severity': opp.severity.value,
                'timestamp': opp.timestamp.isoformat(),
                'price': opp.price,
                'volume': opp.volume,
                'description': opp.description,
                'confidence': opp.confidence,
                'expected_move': opp.expected_move,
                'timeframe': opp.timeframe,
                'data_sources': opp.data_sources,
                'metadata': opp.metadata
            })
        
        return json.dumps({
            'opportunities': opportunities_data,
            'total_count': len(self.opportunities),
            'filtered_count': len(filtered_opportunities)
        }, indent=2)
    
    async def _scan_once(self, parameters: Dict[str, Any]) -> str:
        """Perform a single scan across all scanners"""
        scanner_name = parameters.get('scanner', 'all')
        
        if scanner_name == 'all':
            scanners_to_run = self.scanners.items()
        elif scanner_name in self.scanners:
            scanners_to_run = [(scanner_name, self.scanners[scanner_name])]
        else:
            return json.dumps({'error': f'Unknown scanner: {scanner_name}'})
        
        total_opportunities = 0
        
        for name, scanner in scanners_to_run:
            await self._execute_scanner(name, scanner)
            total_opportunities += len([opp for opp in self.opportunities if opp.timestamp > datetime.now() - timedelta(minutes=5)])
        
        return json.dumps({
            'message': 'Single scan completed',
            'new_opportunities': total_opportunities,
            'scanners_executed': [name for name, _ in scanners_to_run]
        })
    
    async def _get_stats(self) -> str:
        """Get scanner statistics"""
        return json.dumps({
            'scan_stats': self.scan_stats,
            'is_scanning': self.is_scanning,
            'active_scanners': len([s for s in self.scanners.values() if s.enabled]),
            'total_opportunities': len(self.opportunities),
            'opportunities_by_type': self._get_opportunity_counts_by_type(),
            'opportunities_by_severity': self._get_opportunity_counts_by_severity()
        }, indent=2)
    
    def _get_opportunity_counts_by_type(self) -> Dict[str, int]:
        """Get opportunity counts by type"""
        counts = {}
        for opp in self.opportunities:
            opp_type = opp.opportunity_type.value
            counts[opp_type] = counts.get(opp_type, 0) + 1
        return counts
    
    def _get_opportunity_counts_by_severity(self) -> Dict[str, int]:
        """Get opportunity counts by severity"""
        counts = {}
        for opp in self.opportunities:
            severity = opp.severity.value
            counts[severity] = counts.get(severity, 0) + 1
        return counts
    
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _parse_number(self, text: str) -> float:
        """Parse number from text"""
        try:
            # Remove common formatting
            cleaned = re.sub(r'[,$]', '', str(text))
            return float(cleaned)
        except:
            return 0.0
    
    def add_opportunity_callback(self, callback: Callable[[TradingOpportunity], None]):
        """Add callback for new opportunities"""
        self.opportunity_callbacks.append(callback)


# Example usage
if __name__ == "__main__":
    from pathlib import Path
    from ..core.config_manager import ConfigManager
    from ..core.data_manager import UnifiedDataManager
    
    config_manager = ConfigManager(Path("../config"))
    data_manager = UnifiedDataManager(config_manager)
    
    scanner = RealTimeOpportunityScannerTool(config_manager, data_manager)
    
    # Test single scan
    result = scanner._run('scan_once')
    print("Scan Result:", result)
    
    # Get opportunities
    result = scanner._run('get_opportunities')
    print("Opportunities:", result)