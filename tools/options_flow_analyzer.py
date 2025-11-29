#!/usr/bin/env python3
"""
Options Flow Analyzer Tool

This tool analyzes options flow data to identify unusual activity, dark pool flows,
and institutional trading patterns that may indicate upcoming price movements.

Features:
- Unusual options volume detection
- Dark pool activity monitoring
- Block trade identification
- Options flow sentiment analysis
- Gamma exposure calculations
- Put/call ratio analysis
- Institutional flow patterns
- Real-time alerts for significant activity
"""

import asyncio
import logging
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import requests
from concurrent.futures import ThreadPoolExecutor

# CrewAI imports
from crewai_tools import BaseTool
from pydantic import BaseModel, Field

# Local imports
from ..core.config_manager import ConfigManager
from ..core.data_manager import UnifiedDataManager


class FlowType(Enum):
    """Types of options flow"""
    CALL_SWEEP = "call_sweep"
    PUT_SWEEP = "put_sweep"
    CALL_BLOCK = "call_block"
    PUT_BLOCK = "put_block"
    UNUSUAL_VOLUME = "unusual_volume"
    DARK_POOL = "dark_pool"
    INSTITUTIONAL = "institutional"
    RETAIL = "retail"


class FlowSentiment(Enum):
    """Flow sentiment indicators"""
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"
    MIXED = "mixed"


class AlertLevel(Enum):
    """Alert severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class OptionsFlow:
    """Options flow data structure"""
    symbol: str
    timestamp: datetime
    flow_type: FlowType
    strike: float
    expiration: datetime
    volume: int
    open_interest: int
    premium: float
    implied_volatility: float
    delta: float
    gamma: float
    theta: float
    vega: float
    underlying_price: float
    sentiment: FlowSentiment
    confidence: float
    size_category: str  # small, medium, large, block
    exchange: str
    is_sweep: bool
    is_block: bool
    estimated_notional: float


@dataclass
class DarkPoolActivity:
    """Dark pool activity data structure"""
    symbol: str
    timestamp: datetime
    volume: int
    price: float
    side: str  # buy, sell, unknown
    size_category: str
    estimated_institutional: bool
    dark_pool_name: Optional[str]
    confidence: float


@dataclass
class FlowAlert:
    """Flow alert data structure"""
    id: str
    symbol: str
    timestamp: datetime
    alert_level: AlertLevel
    flow_type: FlowType
    description: str
    key_metrics: Dict[str, Any]
    recommended_action: str
    confidence: float
    expires_at: datetime


@dataclass
class FlowAnalysisConfig:
    """Configuration for flow analysis"""
    unusual_volume_threshold: float = 3.0  # Multiple of average volume
    block_size_threshold: int = 1000  # Minimum contracts for block trade
    sweep_threshold: int = 500  # Minimum contracts for sweep
    dark_pool_threshold: int = 10000  # Minimum shares for dark pool alert
    iv_percentile_threshold: float = 80.0  # IV percentile for unusual activity
    gamma_exposure_threshold: float = 1000000  # Dollar gamma exposure threshold
    put_call_ratio_threshold: float = 2.0  # Unusual put/call ratio
    confidence_threshold: float = 0.7  # Minimum confidence for alerts
    max_alerts_per_symbol: int = 10  # Maximum active alerts per symbol
    alert_expiry_hours: int = 24  # Hours until alert expires


class OptionsFlowAnalyzerInput(BaseModel):
    """Input model for options flow analyzer"""
    operation: str = Field(description="Operation: 'analyze_flow', 'detect_unusual_activity', 'monitor_dark_pools', 'generate_alerts', 'get_flow_summary'")
    symbols: List[str] = Field(default=["SPY", "QQQ", "AAPL"], description="Symbols to analyze")
    timeframe: str = Field(default="1h", description="Analysis timeframe")
    lookback_hours: int = Field(default=24, description="Hours of historical data to analyze")
    flow_types: List[str] = Field(default=["all"], description="Types of flow to analyze")
    min_premium: float = Field(default=10000, description="Minimum premium for analysis")
    include_dark_pools: bool = Field(default=True, description="Include dark pool analysis")


class OptionsFlowAnalyzerTool(BaseTool):
    """
    Options Flow Analyzer Tool for detecting unusual options activity and dark pool flows
    """
    
    name: str = "options_flow_analyzer"
    description: str = "Analyze options flow data to detect unusual activity, dark pool flows, and institutional trading patterns"
    
    def __init__(self):
        super().__init__()
        self.config_manager = ConfigManager()
        self.data_manager = UnifiedDataManager()
        self.logger = logging.getLogger(__name__)
        
        # Load configuration
        self.config = self._load_config()
        
        # Flow tracking
        self.options_flows: List[OptionsFlow] = []
        self.dark_pool_activities: List[DarkPoolActivity] = []
        self.flow_alerts: List[FlowAlert] = []
        
        # Analysis state
        self.flow_history: Dict[str, List[OptionsFlow]] = {}
        self.volume_baselines: Dict[str, Dict[str, float]] = {}
        self.iv_baselines: Dict[str, Dict[str, float]] = {}
        
        # Data sources (would be configured for real providers)
        self.options_data_sources = {
            "tradier": {"enabled": True, "api_key": "demo_key"},
            "polygon": {"enabled": True, "api_key": "demo_key"},
            "cboe": {"enabled": True, "api_key": "demo_key"},
            "unusual_whales": {"enabled": False, "api_key": "demo_key"}
        }
        
        self.dark_pool_sources = {
            "ats_data": {"enabled": True, "api_key": "demo_key"},
            "finra_ats": {"enabled": True, "api_key": "demo_key"},
            "iex_dark": {"enabled": True, "api_key": "demo_key"}
        }
    
    def _load_config(self) -> FlowAnalysisConfig:
        """Load flow analysis configuration"""
        try:
            config_data = self.config_manager.get_tool_config("options_flow_analyzer")
            return FlowAnalysisConfig(**config_data)
        except Exception as e:
            self.logger.warning(f"Using default config: {e}")
            return FlowAnalysisConfig()
    
    def _run(self, operation: str, symbols: List[str], timeframe: str = "1h",
             lookback_hours: int = 24, flow_types: List[str] = ["all"],
             min_premium: float = 10000, include_dark_pools: bool = True) -> str:
        """Execute options flow analysis synchronously"""
        return asyncio.run(self._arun(
            operation, symbols, timeframe, lookback_hours, 
            flow_types, min_premium, include_dark_pools
        ))
    
    async def _arun(self, operation: str, symbols: List[str], timeframe: str = "1h",
                    lookback_hours: int = 24, flow_types: List[str] = ["all"],
                    min_premium: float = 10000, include_dark_pools: bool = True) -> str:
        """Execute options flow analysis asynchronously"""
        
        try:
            self.logger.info(f"Starting options flow analysis: {operation}")
            
            if operation == "analyze_flow":
                result = await self._analyze_options_flow(
                    symbols, timeframe, lookback_hours, flow_types, min_premium
                )
            elif operation == "detect_unusual_activity":
                result = await self._detect_unusual_activity(
                    symbols, lookback_hours, min_premium
                )
            elif operation == "monitor_dark_pools":
                result = await self._monitor_dark_pools(
                    symbols, lookback_hours
                )
            elif operation == "generate_alerts":
                result = await self._generate_flow_alerts(
                    symbols, flow_types, min_premium
                )
            elif operation == "get_flow_summary":
                result = await self._get_flow_summary(
                    symbols, lookback_hours
                )
            elif operation == "calculate_gamma_exposure":
                result = await self._calculate_gamma_exposure(symbols)
            elif operation == "analyze_put_call_ratio":
                result = await self._analyze_put_call_ratio(symbols, lookback_hours)
            else:
                result = {
                    "error": f"Unknown operation: {operation}",
                    "available_operations": [
                        "analyze_flow", "detect_unusual_activity", "monitor_dark_pools",
                        "generate_alerts", "get_flow_summary", "calculate_gamma_exposure",
                        "analyze_put_call_ratio"
                    ]
                }
            
            return json.dumps(result, indent=2, default=str)
            
        except Exception as e:
            self.logger.error(f"Options flow analysis failed: {e}")
            return json.dumps({"error": str(e)})
    
    async def _analyze_options_flow(self, symbols: List[str], timeframe: str,
                                   lookback_hours: int, flow_types: List[str],
                                   min_premium: float) -> Dict[str, Any]:
        """Analyze options flow for specified symbols"""
        
        self.logger.info(f"Analyzing options flow for {symbols}")
        
        flow_analysis = {}
        
        for symbol in symbols:
            try:
                # Fetch options flow data
                flows = await self._fetch_options_flows(symbol, lookback_hours, min_premium)
                
                # Filter by flow types if specified
                if "all" not in flow_types:
                    flows = [f for f in flows if f.flow_type.value in flow_types]
                
                # Analyze flows
                analysis = self._analyze_symbol_flows(symbol, flows)
                
                # Store in history
                self.flow_history[symbol] = flows
                
                flow_analysis[symbol] = analysis
                
            except Exception as e:
                self.logger.error(f"Failed to analyze flows for {symbol}: {e}")
                flow_analysis[symbol] = {"error": str(e)}
        
        return {
            "operation": "analyze_flow",
            "symbols": symbols,
            "timeframe": timeframe,
            "lookback_hours": lookback_hours,
            "analysis": flow_analysis,
            "summary": self._generate_flow_analysis_summary(flow_analysis),
            "timestamp": datetime.now().isoformat()
        }
    
    async def _detect_unusual_activity(self, symbols: List[str], lookback_hours: int,
                                      min_premium: float) -> Dict[str, Any]:
        """Detect unusual options activity"""
        
        self.logger.info(f"Detecting unusual activity for {symbols}")
        
        unusual_activities = {}
        
        for symbol in symbols:
            try:
                # Get recent flows
                flows = await self._fetch_options_flows(symbol, lookback_hours, min_premium)
                
                # Get baseline metrics
                baseline = await self._get_volume_baseline(symbol)
                
                # Detect unusual patterns
                unusual_patterns = self._detect_unusual_patterns(flows, baseline)
                
                unusual_activities[symbol] = unusual_patterns
                
            except Exception as e:
                self.logger.error(f"Failed to detect unusual activity for {symbol}: {e}")
                unusual_activities[symbol] = {"error": str(e)}
        
        return {
            "operation": "detect_unusual_activity",
            "symbols": symbols,
            "unusual_activities": unusual_activities,
            "alerts_generated": len([a for activities in unusual_activities.values() 
                                   for a in activities.get("alerts", [])]),
            "timestamp": datetime.now().isoformat()
        }
    
    async def _monitor_dark_pools(self, symbols: List[str], lookback_hours: int) -> Dict[str, Any]:
        """Monitor dark pool activity"""
        
        self.logger.info(f"Monitoring dark pools for {symbols}")
        
        dark_pool_data = {}
        
        for symbol in symbols:
            try:
                # Fetch dark pool data
                activities = await self._fetch_dark_pool_data(symbol, lookback_hours)
                
                # Analyze dark pool patterns
                analysis = self._analyze_dark_pool_activity(symbol, activities)
                
                # Store activities
                self.dark_pool_activities.extend(activities)
                
                dark_pool_data[symbol] = analysis
                
            except Exception as e:
                self.logger.error(f"Failed to monitor dark pools for {symbol}: {e}")
                dark_pool_data[symbol] = {"error": str(e)}
        
        return {
            "operation": "monitor_dark_pools",
            "symbols": symbols,
            "dark_pool_data": dark_pool_data,
            "total_activities": sum(len(data.get("activities", [])) 
                                  for data in dark_pool_data.values()),
            "timestamp": datetime.now().isoformat()
        }
    
    async def _generate_flow_alerts(self, symbols: List[str], flow_types: List[str],
                                   min_premium: float) -> Dict[str, Any]:
        """Generate flow-based alerts"""
        
        self.logger.info(f"Generating flow alerts for {symbols}")
        
        new_alerts = []
        
        for symbol in symbols:
            try:
                # Get recent flows
                flows = await self._fetch_options_flows(symbol, 4, min_premium)  # Last 4 hours
                
                # Generate alerts based on flows
                symbol_alerts = self._create_flow_alerts(symbol, flows)
                
                new_alerts.extend(symbol_alerts)
                
            except Exception as e:
                self.logger.error(f"Failed to generate alerts for {symbol}: {e}")
        
        # Add to alert list
        self.flow_alerts.extend(new_alerts)
        
        # Clean up expired alerts
        self._cleanup_expired_alerts()
        
        return {
            "operation": "generate_alerts",
            "symbols": symbols,
            "new_alerts": len(new_alerts),
            "total_active_alerts": len(self.flow_alerts),
            "alerts": [asdict(alert) for alert in new_alerts],
            "timestamp": datetime.now().isoformat()
        }
    
    async def _get_flow_summary(self, symbols: List[str], lookback_hours: int) -> Dict[str, Any]:
        """Get comprehensive flow summary"""
        
        self.logger.info(f"Getting flow summary for {symbols}")
        
        summary = {
            "overview": {},
            "top_flows": [],
            "sentiment_analysis": {},
            "volume_analysis": {},
            "unusual_activity": {},
            "dark_pool_summary": {}
        }
        
        all_flows = []
        
        for symbol in symbols:
            try:
                flows = await self._fetch_options_flows(symbol, lookback_hours, 1000)
                all_flows.extend(flows)
                
                # Symbol-specific summary
                summary["overview"][symbol] = {
                    "total_flows": len(flows),
                    "total_premium": sum(f.premium for f in flows),
                    "avg_iv": np.mean([f.implied_volatility for f in flows]) if flows else 0,
                    "bullish_flows": len([f for f in flows if f.sentiment == FlowSentiment.BULLISH]),
                    "bearish_flows": len([f for f in flows if f.sentiment == FlowSentiment.BEARISH])
                }
                
            except Exception as e:
                self.logger.error(f"Failed to get summary for {symbol}: {e}")
        
        # Overall analysis
        if all_flows:
            summary["top_flows"] = sorted(all_flows, key=lambda x: x.premium, reverse=True)[:10]
            summary["sentiment_analysis"] = self._analyze_overall_sentiment(all_flows)
            summary["volume_analysis"] = self._analyze_volume_patterns(all_flows)
        
        return {
            "operation": "get_flow_summary",
            "symbols": symbols,
            "lookback_hours": lookback_hours,
            "summary": summary,
            "timestamp": datetime.now().isoformat()
        }
    
    async def _calculate_gamma_exposure(self, symbols: List[str]) -> Dict[str, Any]:
        """Calculate gamma exposure for symbols"""
        
        self.logger.info(f"Calculating gamma exposure for {symbols}")
        
        gamma_data = {}
        
        for symbol in symbols:
            try:
                # Get options chain data
                options_chain = await self._fetch_options_chain(symbol)
                
                # Calculate gamma exposure
                gamma_exposure = self._calculate_symbol_gamma_exposure(symbol, options_chain)
                
                gamma_data[symbol] = gamma_exposure
                
            except Exception as e:
                self.logger.error(f"Failed to calculate gamma exposure for {symbol}: {e}")
                gamma_data[symbol] = {"error": str(e)}
        
        return {
            "operation": "calculate_gamma_exposure",
            "symbols": symbols,
            "gamma_data": gamma_data,
            "timestamp": datetime.now().isoformat()
        }
    
    async def _analyze_put_call_ratio(self, symbols: List[str], lookback_hours: int) -> Dict[str, Any]:
        """Analyze put/call ratio patterns"""
        
        self.logger.info(f"Analyzing put/call ratio for {symbols}")
        
        pc_analysis = {}
        
        for symbol in symbols:
            try:
                # Get flows
                flows = await self._fetch_options_flows(symbol, lookback_hours, 1000)
                
                # Calculate put/call ratios
                pc_ratios = self._calculate_put_call_ratios(flows)
                
                pc_analysis[symbol] = pc_ratios
                
            except Exception as e:
                self.logger.error(f"Failed to analyze put/call ratio for {symbol}: {e}")
                pc_analysis[symbol] = {"error": str(e)}
        
        return {
            "operation": "analyze_put_call_ratio",
            "symbols": symbols,
            "pc_analysis": pc_analysis,
            "timestamp": datetime.now().isoformat()
        }
    
    async def _fetch_options_flows(self, symbol: str, lookback_hours: int,
                                  min_premium: float) -> List[OptionsFlow]:
        """Fetch options flow data (simulated for demo)"""
        
        # In production, this would fetch from real options flow providers
        flows = []
        
        # Generate simulated flow data
        current_time = datetime.now()
        
        for i in range(np.random.randint(10, 50)):  # Random number of flows
            flow_time = current_time - timedelta(hours=np.random.uniform(0, lookback_hours))
            
            # Simulate flow data
            flow = OptionsFlow(
                symbol=symbol,
                timestamp=flow_time,
                flow_type=np.random.choice(list(FlowType)),
                strike=np.random.uniform(80, 120),  # Relative to current price
                expiration=current_time + timedelta(days=np.random.randint(1, 60)),
                volume=np.random.randint(100, 5000),
                open_interest=np.random.randint(1000, 50000),
                premium=np.random.uniform(min_premium, min_premium * 10),
                implied_volatility=np.random.uniform(0.15, 0.8),
                delta=np.random.uniform(-1, 1),
                gamma=np.random.uniform(0, 0.1),
                theta=np.random.uniform(-0.5, 0),
                vega=np.random.uniform(0, 0.3),
                underlying_price=100.0,  # Simulated price
                sentiment=np.random.choice(list(FlowSentiment)),
                confidence=np.random.uniform(0.5, 1.0),
                size_category=np.random.choice(["small", "medium", "large", "block"]),
                exchange=np.random.choice(["CBOE", "NASDAQ", "NYSE", "PHLX"]),
                is_sweep=np.random.choice([True, False]),
                is_block=np.random.choice([True, False]),
                estimated_notional=np.random.uniform(10000, 1000000)
            )
            
            flows.append(flow)
        
        return flows
    
    async def _fetch_dark_pool_data(self, symbol: str, lookback_hours: int) -> List[DarkPoolActivity]:
        """Fetch dark pool activity data (simulated for demo)"""
        
        activities = []
        current_time = datetime.now()
        
        for i in range(np.random.randint(5, 20)):  # Random number of activities
            activity_time = current_time - timedelta(hours=np.random.uniform(0, lookback_hours))
            
            activity = DarkPoolActivity(
                symbol=symbol,
                timestamp=activity_time,
                volume=np.random.randint(10000, 500000),
                price=np.random.uniform(95, 105),
                side=np.random.choice(["buy", "sell", "unknown"]),
                size_category=np.random.choice(["medium", "large", "block"]),
                estimated_institutional=np.random.choice([True, False]),
                dark_pool_name=np.random.choice(["Goldman Sachs", "Morgan Stanley", "UBS", None]),
                confidence=np.random.uniform(0.6, 1.0)
            )
            
            activities.append(activity)
        
        return activities
    
    async def _fetch_options_chain(self, symbol: str) -> Dict[str, Any]:
        """Fetch options chain data (simulated for demo)"""
        
        # Simulate options chain
        chain = {
            "calls": [],
            "puts": [],
            "underlying_price": 100.0,
            "expiration_dates": []
        }
        
        # Generate sample options data
        strikes = np.arange(80, 121, 5)  # Strike prices from 80 to 120
        
        for strike in strikes:
            # Call option
            call = {
                "strike": strike,
                "volume": np.random.randint(0, 1000),
                "open_interest": np.random.randint(100, 10000),
                "gamma": np.random.uniform(0, 0.1),
                "delta": np.random.uniform(0, 1)
            }
            chain["calls"].append(call)
            
            # Put option
            put = {
                "strike": strike,
                "volume": np.random.randint(0, 1000),
                "open_interest": np.random.randint(100, 10000),
                "gamma": np.random.uniform(0, 0.1),
                "delta": np.random.uniform(-1, 0)
            }
            chain["puts"].append(put)
        
        return chain
    
    def _analyze_symbol_flows(self, symbol: str, flows: List[OptionsFlow]) -> Dict[str, Any]:
        """Analyze flows for a specific symbol"""
        
        if not flows:
            return {"error": "No flows to analyze"}
        
        # Basic statistics
        total_premium = sum(f.premium for f in flows)
        avg_iv = np.mean([f.implied_volatility for f in flows])
        
        # Sentiment analysis
        bullish_count = len([f for f in flows if f.sentiment == FlowSentiment.BULLISH])
        bearish_count = len([f for f in flows if f.sentiment == FlowSentiment.BEARISH])
        
        # Flow type distribution
        flow_types = {}
        for flow in flows:
            flow_type = flow.flow_type.value
            flow_types[flow_type] = flow_types.get(flow_type, 0) + 1
        
        # Size analysis
        block_trades = len([f for f in flows if f.is_block])
        sweeps = len([f for f in flows if f.is_sweep])
        
        return {
            "total_flows": len(flows),
            "total_premium": total_premium,
            "avg_implied_volatility": avg_iv,
            "sentiment_distribution": {
                "bullish": bullish_count,
                "bearish": bearish_count,
                "neutral": len(flows) - bullish_count - bearish_count
            },
            "flow_type_distribution": flow_types,
            "block_trades": block_trades,
            "sweeps": sweeps,
            "largest_trade": max(flows, key=lambda x: x.premium).premium if flows else 0
        }
    
    def _detect_unusual_patterns(self, flows: List[OptionsFlow], baseline: Dict[str, float]) -> Dict[str, Any]:
        """Detect unusual patterns in flows"""
        
        patterns = {
            "unusual_volume": [],
            "unusual_iv": [],
            "large_blocks": [],
            "sweep_activity": [],
            "alerts": []
        }
        
        for flow in flows:
            # Check for unusual volume
            if flow.volume > baseline.get("avg_volume", 0) * self.config.unusual_volume_threshold:
                patterns["unusual_volume"].append({
                    "flow": asdict(flow),
                    "volume_multiple": flow.volume / baseline.get("avg_volume", 1)
                })
            
            # Check for unusual IV
            if flow.implied_volatility > baseline.get("avg_iv", 0) * 1.5:
                patterns["unusual_iv"].append({
                    "flow": asdict(flow),
                    "iv_percentile": (flow.implied_volatility / baseline.get("avg_iv", 1)) * 100
                })
            
            # Check for large blocks
            if flow.volume >= self.config.block_size_threshold:
                patterns["large_blocks"].append(asdict(flow))
            
            # Check for sweep activity
            if flow.is_sweep and flow.volume >= self.config.sweep_threshold:
                patterns["sweep_activity"].append(asdict(flow))
        
        return patterns
    
    def _analyze_dark_pool_activity(self, symbol: str, activities: List[DarkPoolActivity]) -> Dict[str, Any]:
        """Analyze dark pool activity patterns"""
        
        if not activities:
            return {"error": "No dark pool activities to analyze"}
        
        total_volume = sum(a.volume for a in activities)
        institutional_volume = sum(a.volume for a in activities if a.estimated_institutional)
        
        # Side analysis
        buy_volume = sum(a.volume for a in activities if a.side == "buy")
        sell_volume = sum(a.volume for a in activities if a.side == "sell")
        
        # Size distribution
        size_distribution = {}
        for activity in activities:
            size_cat = activity.size_category
            size_distribution[size_cat] = size_distribution.get(size_cat, 0) + activity.volume
        
        return {
            "total_activities": len(activities),
            "total_volume": total_volume,
            "institutional_volume": institutional_volume,
            "institutional_percentage": (institutional_volume / total_volume * 100) if total_volume > 0 else 0,
            "buy_sell_ratio": buy_volume / sell_volume if sell_volume > 0 else float('inf'),
            "size_distribution": size_distribution,
            "activities": [asdict(a) for a in activities[:10]]  # Top 10 activities
        }
    
    def _create_flow_alerts(self, symbol: str, flows: List[OptionsFlow]) -> List[FlowAlert]:
        """Create alerts based on flow analysis"""
        
        alerts = []
        
        for flow in flows:
            alert_level = None
            description = ""
            
            # Determine alert level and description
            if flow.premium > 100000 and flow.is_block:
                alert_level = AlertLevel.HIGH
                description = f"Large block trade: {flow.volume} contracts, ${flow.premium:,.0f} premium"
            elif flow.is_sweep and flow.volume > self.config.sweep_threshold:
                alert_level = AlertLevel.MEDIUM
                description = f"Options sweep: {flow.volume} contracts across multiple exchanges"
            elif flow.volume > 2000:
                alert_level = AlertLevel.MEDIUM
                description = f"High volume options activity: {flow.volume} contracts"
            
            if alert_level:
                alert = FlowAlert(
                    id=f"alert_{symbol}_{int(flow.timestamp.timestamp())}",
                    symbol=symbol,
                    timestamp=flow.timestamp,
                    alert_level=alert_level,
                    flow_type=flow.flow_type,
                    description=description,
                    key_metrics={
                        "volume": flow.volume,
                        "premium": flow.premium,
                        "iv": flow.implied_volatility,
                        "strike": flow.strike,
                        "expiration": flow.expiration.isoformat()
                    },
                    recommended_action=self._get_recommended_action(flow),
                    confidence=flow.confidence,
                    expires_at=datetime.now() + timedelta(hours=self.config.alert_expiry_hours)
                )
                
                alerts.append(alert)
        
        return alerts
    
    def _get_recommended_action(self, flow: OptionsFlow) -> str:
        """Get recommended action based on flow"""
        
        if flow.sentiment == FlowSentiment.BULLISH:
            if flow.flow_type in [FlowType.CALL_SWEEP, FlowType.CALL_BLOCK]:
                return "Consider bullish position or call spreads"
            else:
                return "Monitor for potential upward movement"
        elif flow.sentiment == FlowSentiment.BEARISH:
            if flow.flow_type in [FlowType.PUT_SWEEP, FlowType.PUT_BLOCK]:
                return "Consider bearish position or put spreads"
            else:
                return "Monitor for potential downward movement"
        else:
            return "Monitor for directional clarity"
    
    async def _get_volume_baseline(self, symbol: str) -> Dict[str, float]:
        """Get volume baseline for unusual activity detection"""
        
        # In production, this would calculate from historical data
        return {
            "avg_volume": np.random.uniform(500, 2000),
            "avg_iv": np.random.uniform(0.2, 0.4),
            "avg_premium": np.random.uniform(5000, 50000)
        }
    
    def _calculate_symbol_gamma_exposure(self, symbol: str, options_chain: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate gamma exposure for a symbol"""
        
        total_call_gamma = 0
        total_put_gamma = 0
        
        underlying_price = options_chain["underlying_price"]
        
        # Calculate call gamma exposure
        for call in options_chain["calls"]:
            gamma_exposure = call["gamma"] * call["open_interest"] * 100 * underlying_price
            total_call_gamma += gamma_exposure
        
        # Calculate put gamma exposure (negative for puts)
        for put in options_chain["puts"]:
            gamma_exposure = -put["gamma"] * put["open_interest"] * 100 * underlying_price
            total_put_gamma += gamma_exposure
        
        net_gamma = total_call_gamma + total_put_gamma
        
        return {
            "total_call_gamma": total_call_gamma,
            "total_put_gamma": total_put_gamma,
            "net_gamma_exposure": net_gamma,
            "gamma_level": "high" if abs(net_gamma) > self.config.gamma_exposure_threshold else "normal"
        }
    
    def _calculate_put_call_ratios(self, flows: List[OptionsFlow]) -> Dict[str, Any]:
        """Calculate put/call ratios from flows"""
        
        call_volume = sum(f.volume for f in flows if f.flow_type in [FlowType.CALL_SWEEP, FlowType.CALL_BLOCK])
        put_volume = sum(f.volume for f in flows if f.flow_type in [FlowType.PUT_SWEEP, FlowType.PUT_BLOCK])
        
        call_premium = sum(f.premium for f in flows if f.flow_type in [FlowType.CALL_SWEEP, FlowType.CALL_BLOCK])
        put_premium = sum(f.premium for f in flows if f.flow_type in [FlowType.PUT_SWEEP, FlowType.PUT_BLOCK])
        
        volume_ratio = put_volume / call_volume if call_volume > 0 else float('inf')
        premium_ratio = put_premium / call_premium if call_premium > 0 else float('inf')
        
        return {
            "volume_put_call_ratio": volume_ratio,
            "premium_put_call_ratio": premium_ratio,
            "call_volume": call_volume,
            "put_volume": put_volume,
            "call_premium": call_premium,
            "put_premium": put_premium,
            "ratio_signal": "bearish" if volume_ratio > self.config.put_call_ratio_threshold else "bullish"
        }
    
    def _generate_flow_analysis_summary(self, flow_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary of flow analysis"""
        
        total_symbols = len(flow_analysis)
        successful_analyses = len([a for a in flow_analysis.values() if "error" not in a])
        
        if successful_analyses == 0:
            return {"error": "No successful analyses"}
        
        # Aggregate metrics
        total_flows = sum(a.get("total_flows", 0) for a in flow_analysis.values() if "error" not in a)
        total_premium = sum(a.get("total_premium", 0) for a in flow_analysis.values() if "error" not in a)
        
        return {
            "total_symbols_analyzed": total_symbols,
            "successful_analyses": successful_analyses,
            "total_flows_analyzed": total_flows,
            "total_premium_analyzed": total_premium,
            "avg_flows_per_symbol": total_flows / successful_analyses if successful_analyses > 0 else 0
        }
    
    def _analyze_overall_sentiment(self, flows: List[OptionsFlow]) -> Dict[str, Any]:
        """Analyze overall sentiment from all flows"""
        
        sentiment_counts = {}
        for flow in flows:
            sentiment = flow.sentiment.value
            sentiment_counts[sentiment] = sentiment_counts.get(sentiment, 0) + 1
        
        total_flows = len(flows)
        sentiment_percentages = {k: (v / total_flows * 100) for k, v in sentiment_counts.items()}
        
        # Determine overall sentiment
        if sentiment_percentages.get("bullish", 0) > 60:
            overall_sentiment = "bullish"
        elif sentiment_percentages.get("bearish", 0) > 60:
            overall_sentiment = "bearish"
        else:
            overall_sentiment = "mixed"
        
        return {
            "sentiment_counts": sentiment_counts,
            "sentiment_percentages": sentiment_percentages,
            "overall_sentiment": overall_sentiment
        }
    
    def _analyze_volume_patterns(self, flows: List[OptionsFlow]) -> Dict[str, Any]:
        """Analyze volume patterns across flows"""
        
        volumes = [f.volume for f in flows]
        premiums = [f.premium for f in flows]
        
        return {
            "total_volume": sum(volumes),
            "avg_volume": np.mean(volumes),
            "max_volume": max(volumes),
            "total_premium": sum(premiums),
            "avg_premium": np.mean(premiums),
            "max_premium": max(premiums),
            "volume_distribution": {
                "small": len([v for v in volumes if v < 500]),
                "medium": len([v for v in volumes if 500 <= v < 2000]),
                "large": len([v for v in volumes if v >= 2000])
            }
        }
    
    def _cleanup_expired_alerts(self):
        """Remove expired alerts"""
        current_time = datetime.now()
        self.flow_alerts = [alert for alert in self.flow_alerts if alert.expires_at > current_time]


if __name__ == "__main__":
    async def test_options_flow_analyzer():
        """Test the options flow analyzer"""
        tool = OptionsFlowAnalyzerTool()
        
        # Test flow analysis
        result = await tool._arun(
            operation="analyze_flow",
            symbols=["AAPL", "TSLA", "SPY"],
            timeframe="1h",
            lookback_hours=24,
            min_premium=10000
        )
        
        print("Options Flow Analysis Result:")
        print(result)
    
    asyncio.run(test_options_flow_analyzer())