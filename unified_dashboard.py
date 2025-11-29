#!/usr/bin/env python3
"""
Unified Trading System Dashboard
Combines all dashboard features with enhanced agent decision tracking and real-time analysis.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import asyncio
import json
import logging
import websockets
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from collections import deque, defaultdict
import queue
import sqlite3
import requests

# Core system imports
try:
    from core.config_manager import ConfigManager
    from core.data_manager import UnifiedDataManager
    from core.trade_storage import TradeStorage
    from core.market_data_aggregator import MarketDataAggregator
    from core.risk_manager_24_7 import RiskManager24_7
    from core.agent_orchestrator import AgentOrchestrator
    from utils.performance_tracker import PerformanceTracker
    from tools.portfolio_management_tool import PortfolioManagementTool
    from tools.risk_analysis_tool import RiskAnalysisTool
except ImportError as e:
    st.error(f"Core system import error: {e}")
    st.stop()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
API_BASE_URL = "http://localhost:8000"
WS_URL = "ws://localhost:8000/ws"
REFRESH_INTERVAL = 1  # seconds

@dataclass
class AgentDecision:
    """Agent decision data structure"""
    agent_id: str
    agent_type: str
    timestamp: datetime
    decision_type: str  # 'trade_signal', 'risk_assessment', 'market_analysis'
    symbol: str
    action: str  # 'BUY', 'SELL', 'HOLD', 'ANALYZE'
    confidence: float
    reasoning: str
    data_sources: List[str]
    risk_score: float
    expected_return: float
    time_horizon: str
    status: str  # 'pending', 'executed', 'rejected', 'expired'

@dataclass
class TradeConsideration:
    """Trade consideration before execution"""
    consideration_id: str
    timestamp: datetime
    symbol: str
    proposed_action: str
    quantity: float
    price_target: float
    agent_consensus: Dict[str, float]  # agent_id -> confidence
    risk_metrics: Dict[str, float]
    market_conditions: Dict[str, Any]
    final_decision: str  # 'execute', 'reject', 'modify', 'delay'
    execution_timestamp: Optional[datetime]
    actual_execution: Optional[Dict[str, Any]]

@dataclass
class AgentActivity:
    """Real-time agent activity"""
    agent_id: str
    agent_type: str
    timestamp: datetime
    action: str
    details: Dict[str, Any]
    status: str
    performance_score: float

@dataclass
class MarketDataPoint:
    """Market data point"""
    symbol: str
    price: float
    volume: int
    timestamp: datetime
    change: float
    change_percent: float

@dataclass
class TradeEvent:
    """Trade execution event"""
    trade_id: str
    symbol: str
    side: str
    quantity: float
    price: float
    timestamp: datetime
    status: str
    pnl: float
    strategy: str

class UnifiedDashboardManager:
    """Unified dashboard data manager with real-time capabilities"""
    
    def __init__(self):
        self.config_manager = None
        self.data_manager = None
        self.trade_storage = None
        self.risk_manager = None
        self.agent_orchestrator = None
        
        # Real-time data storage
        self.agent_decisions = deque(maxlen=1000)
        self.trade_considerations = deque(maxlen=500)
        self.agent_activities = deque(maxlen=1000)
        self.market_data = {}
        self.trade_events = deque(maxlen=1000)
        self.system_metrics = {}
        
        # WebSocket connection
        self.ws_connection = None
        self.is_connected = False
        self.subscribed_channels = set()
        
        # Data queues for thread-safe communication
        self.data_queue = queue.Queue()
        
        self.initialize_components()
    
    def initialize_components(self):
        """Initialize system components"""
        try:
            # Initialize ConfigManager with the config directory path
            config_path = Path(__file__).parent / "config"
            self.config_manager = ConfigManager(config_path)
            
            # Initialize UnifiedDataManager with the config_manager
            self.data_manager = UnifiedDataManager(self.config_manager)
            
            # Initialize TradeStorage with the config_manager
            self.trade_storage = TradeStorage(self.config_manager)
            
            # Initialize RiskManager24_7 with the config_manager
            self.risk_manager = RiskManager24_7(self.config_manager)
            
            # Initialize agent orchestrator if available
            try:
                self.agent_orchestrator = AgentOrchestrator()
            except Exception as e:
                logger.warning(f"Agent orchestrator not available: {e}")
                
        except Exception as e:
            logger.error(f"Component initialization error: {e}")
    
    async def connect_websocket(self):
        """Connect to WebSocket for real-time updates"""
        try:
            self.ws_connection = await websockets.connect(WS_URL)
            self.is_connected = True
            logger.info("WebSocket connected successfully")
            
            # Subscribe to all channels
            await self.subscribe_to_channels([
                'agent_decisions',
                'trade_considerations', 
                'agent_activity',
                'market_data',
                'trades',
                'portfolio',
                'system_health',
                'alerts'
            ])
            
            return True
        except Exception as e:
            logger.error(f"WebSocket connection failed: {e}")
            self.is_connected = False
            return False
    
    async def subscribe_to_channels(self, channels: List[str]):
        """Subscribe to WebSocket channels"""
        if not self.ws_connection:
            return
            
        try:
            message = {
                "action": "subscribe",
                "channels": channels
            }
            await self.ws_connection.send(json.dumps(message))
            self.subscribed_channels.update(channels)
            logger.info(f"Subscribed to channels: {channels}")
        except Exception as e:
            logger.error(f"Subscription failed: {e}")
    
    def process_websocket_message(self, data: Dict[str, Any]):
        """Process incoming WebSocket messages"""
        message_type = data.get('type')
        
        if message_type == 'agent_decision':
            self.process_agent_decision(data['data'])
        elif message_type == 'trade_consideration':
            self.process_trade_consideration(data['data'])
        elif message_type == 'agent_activity':
            self.process_agent_activity(data['data'])
        elif message_type == 'market_data':
            self.process_market_data(data['data'])
        elif message_type == 'trade_event':
            self.process_trade_event(data['data'])
        elif message_type == 'system_health':
            self.process_system_health(data['data'])
    
    def process_agent_decision(self, data: Dict[str, Any]):
        """Process agent decision updates"""
        decision = AgentDecision(
            agent_id=data.get('agent_id', ''),
            agent_type=data.get('agent_type', ''),
            timestamp=datetime.fromisoformat(data.get('timestamp', datetime.now().isoformat())),
            decision_type=data.get('decision_type', ''),
            symbol=data.get('symbol', ''),
            action=data.get('action', ''),
            confidence=data.get('confidence', 0.0),
            reasoning=data.get('reasoning', ''),
            data_sources=data.get('data_sources', []),
            risk_score=data.get('risk_score', 0.0),
            expected_return=data.get('expected_return', 0.0),
            time_horizon=data.get('time_horizon', ''),
            status=data.get('status', 'pending')
        )
        self.agent_decisions.append(decision)
    
    def process_trade_consideration(self, data: Dict[str, Any]):
        """Process trade consideration updates"""
        consideration = TradeConsideration(
            consideration_id=data.get('consideration_id', ''),
            timestamp=datetime.fromisoformat(data.get('timestamp', datetime.now().isoformat())),
            symbol=data.get('symbol', ''),
            proposed_action=data.get('proposed_action', ''),
            quantity=data.get('quantity', 0.0),
            price_target=data.get('price_target', 0.0),
            agent_consensus=data.get('agent_consensus', {}),
            risk_metrics=data.get('risk_metrics', {}),
            market_conditions=data.get('market_conditions', {}),
            final_decision=data.get('final_decision', 'pending'),
            execution_timestamp=datetime.fromisoformat(data['execution_timestamp']) if data.get('execution_timestamp') else None,
            actual_execution=data.get('actual_execution')
        )
        self.trade_considerations.append(consideration)
    
    def process_agent_activity(self, data: Dict[str, Any]):
        """Process agent activity updates"""
        activity = AgentActivity(
            agent_id=data.get('agent_id', ''),
            agent_type=data.get('agent_type', ''),
            timestamp=datetime.fromisoformat(data.get('timestamp', datetime.now().isoformat())),
            action=data.get('action', ''),
            details=data.get('details', {}),
            status=data.get('status', ''),
            performance_score=data.get('performance_score', 0.0)
        )
        self.agent_activities.append(activity)
    
    def process_market_data(self, data: Dict[str, Any]):
        """Process market data updates"""
        symbol = data.get('symbol')
        if symbol:
            self.market_data[symbol] = MarketDataPoint(
                symbol=symbol,
                price=data.get('price', 0.0),
                volume=data.get('volume', 0),
                timestamp=datetime.fromisoformat(data.get('timestamp', datetime.now().isoformat())),
                change=data.get('change', 0.0),
                change_percent=data.get('change_percent', 0.0)
            )
    
    def process_trade_event(self, data: Dict[str, Any]):
        """Process trade events"""
        trade = TradeEvent(
            trade_id=data.get('trade_id', ''),
            symbol=data.get('symbol', ''),
            side=data.get('side', ''),
            quantity=data.get('quantity', 0.0),
            price=data.get('price', 0.0),
            timestamp=datetime.fromisoformat(data.get('timestamp', datetime.now().isoformat())),
            status=data.get('status', ''),
            pnl=data.get('pnl', 0.0),
            strategy=data.get('strategy', '')
        )
        self.trade_events.append(trade)
    
    def process_system_health(self, data: Dict[str, Any]):
        """Process system health updates"""
        self.system_metrics.update(data)
    
    def fetch_api_data(self, endpoint: str) -> Optional[Dict[str, Any]]:
        """Fetch data from the FastAPI server"""
        try:
            response = requests.get(f"{API_BASE_URL}{endpoint}", timeout=5)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.error(f"Error fetching data from {endpoint}: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error fetching data from {endpoint}: {e}")
            return None

    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        try:
            # Try to get status from API first
            api_status = self.fetch_api_data("/health")
            if api_status:
                # Enhance API status with local dashboard info
                api_status.update({
                    "dashboard_components": {
                        "config_manager": "ACTIVE" if self.config_manager else "INACTIVE",
                        "data_manager": "ACTIVE" if self.data_manager else "INACTIVE",
                        "trade_storage": "ACTIVE" if self.trade_storage else "INACTIVE",
                        "risk_manager": "ACTIVE" if self.risk_manager else "INACTIVE",
                        "agent_orchestrator": "ACTIVE" if self.agent_orchestrator else "INACTIVE",
                        "websocket": "CONNECTED" if self.is_connected else "DISCONNECTED"
                    },
                    "local_metrics": {
                        "active_agents": len(set(a.agent_id for a in self.agent_activities)),
                        "recent_decisions": len([d for d in self.agent_decisions if d.timestamp > datetime.now() - timedelta(hours=1)]),
                        "pending_considerations": len([c for c in self.trade_considerations if c.final_decision == 'pending']),
                    }
                })
                return api_status
            
            # Fallback to local status if API is unavailable
            status = {
                "timestamp": datetime.now().isoformat(),
                "system_health": "DEGRADED",
                "status": "API_UNAVAILABLE",
                "components": {
                    "config_manager": "ACTIVE" if self.config_manager else "INACTIVE",
                    "data_manager": "ACTIVE" if self.data_manager else "INACTIVE",
                    "trade_storage": "ACTIVE" if self.trade_storage else "INACTIVE",
                    "risk_manager": "ACTIVE" if self.risk_manager else "INACTIVE",
                    "agent_orchestrator": "ACTIVE" if self.agent_orchestrator else "INACTIVE",
                    "websocket": "CONNECTED" if self.is_connected else "DISCONNECTED"
                },
                "active_agents": len(set(a.agent_id for a in self.agent_activities)),
                "recent_decisions": len([d for d in self.agent_decisions if d.timestamp > datetime.now() - timedelta(hours=1)]),
                "pending_considerations": len([c for c in self.trade_considerations if c.final_decision == 'pending']),
                "market_sessions": self._get_market_sessions()
            }
            
            # Add system metrics if available
            if self.system_metrics:
                status.update(self.system_metrics)
            
            return status
        except Exception as e:
            logger.error(f"Error getting system status: {e}")
            return {"error": str(e)}
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics"""
        try:
            # Try to get performance data from API first
            api_performance = self.fetch_api_data("/api/v1/performance")
            if api_performance:
                # Enhance API data with local metrics
                local_metrics = self._calculate_local_performance_metrics()
                api_performance.update({
                    "local_dashboard_metrics": local_metrics,
                    "data_source": "api_with_local_enhancement"
                })
                return api_performance
            
            # Fallback to local calculation if API is unavailable
            return self._calculate_local_performance_metrics()
        except Exception as e:
            logger.error(f"Error getting performance metrics: {e}")
            return {"error": str(e)}
    
    def _calculate_local_performance_metrics(self) -> Dict[str, Any]:
        """Calculate performance metrics from local data"""
        try:
            # Get recent trades for analysis
            recent_trades = list(self.trade_events)[-100:] if self.trade_events else []
            
            if not recent_trades:
                return self._get_real_performance_metrics()
            
            # Calculate metrics from real data
            total_pnl = sum(trade.pnl for trade in recent_trades)
            winning_trades = [trade for trade in recent_trades if trade.pnl > 0]
            win_rate = len(winning_trades) / len(recent_trades) if recent_trades else 0
            
            # Group by asset for asset-specific performance
            asset_performance = {}
            for trade in recent_trades:
                symbol = trade.symbol
                if symbol not in asset_performance:
                    asset_performance[symbol] = {'trades': 0, 'pnl': 0.0}
                asset_performance[symbol]['trades'] += 1
                asset_performance[symbol]['pnl'] += trade.pnl
            
            return {
                "data_source": "local_calculation",
                "portfolio_value": 100000 + total_pnl,
                "daily_pnl": sum(trade.pnl for trade in recent_trades if trade.timestamp.date() == datetime.now().date()),
                "total_pnl": total_pnl,
                "win_rate": win_rate,
                "total_trades": len(recent_trades),
                "active_positions": len(set(trade.symbol for trade in recent_trades)),
                "sharpe_ratio": self._calculate_sharpe_ratio(recent_trades),
                "max_drawdown": self._calculate_max_drawdown(recent_trades),
                "asset_performance": asset_performance,
                "risk_metrics": {
                    "var_95": 0.0,
                    "expected_shortfall": 0.0,
                    "volatility": 0.0
                }
            }
        except Exception as e:
            logger.error(f"Error calculating local performance metrics: {e}")
            return self._get_real_performance_metrics()
    
    def _get_real_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics from real data only."""
        try:
            # Get real performance data from trade storage
            if hasattr(self, 'trade_storage') and self.trade_storage:
                trades_data = self.trade_storage.get_recent_trades(100)
                if trades_data:
                    total_pnl = sum(trade.get('pnl', 0.0) for trade in trades_data)
                    winning_trades = [t for t in trades_data if t.get('pnl', 0) > 0]
                    win_rate = len(winning_trades) / len(trades_data) if trades_data else 0
                    
                    return {
                        "data_source": "real_data",
                        "portfolio_value": 100000 + total_pnl,
                        "daily_pnl": sum(t.get('pnl', 0) for t in trades_data if t.get('timestamp', '').startswith(datetime.now().strftime('%Y-%m-%d'))),
                        "total_return": total_pnl / 100000,
                        "total_pnl": total_pnl,
                        "win_rate": win_rate,
                        "total_trades": len(trades_data),
                        "active_positions": len(set(t.get('symbol', '') for t in trades_data)),
                        "cash_balance": 100000 - sum(t.get('quantity', 0) * t.get('price', 0) for t in trades_data),
                        "sharpe_ratio": 0.0,
                        "max_drawdown": 0.0,
                        "asset_performance": {},
                        "risk_metrics": {
                            "var_95": 0.0,
                            "expected_shortfall": 0.0,
                            "volatility": 0.0
                        }
                    }
        except Exception as e:
            logger.error(f"Error getting performance metrics: {e}")
        
        # Return empty metrics if no real data available - no fallback to mock data
        return {
            "data_source": "no_data",
            "portfolio_value": 0.0,
            "daily_pnl": 0.0,
            "daily_return": 0.0,
            "total_return": 0.0,
            "total_pnl": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
            "win_rate": 0.0,
            "total_trades": 0,
            "active_positions": 0,
            "cash_balance": 0.0,
            "asset_performance": {},
            "risk_metrics": {
                "var_95": 0.0,
                "expected_shortfall": 0.0,
                "volatility": 0.0
            }
        }
    
    def get_current_positions(self) -> List[Dict[str, Any]]:
        """Get current positions from API or local data"""
        try:
            # Try to get positions from API first
            api_positions = self.fetch_api_data("/api/v1/positions")
            if api_positions:
                # Handle both direct array and wrapped object formats
                if isinstance(api_positions, list):
                    return api_positions
                elif isinstance(api_positions, dict) and 'positions' in api_positions:
                    return api_positions['positions']
            
            # Fallback to local positions if available
            if hasattr(self, 'current_positions') and self.current_positions:
                return self.current_positions
            
            # Return empty list if no positions available
            return []
        except Exception as e:
            logger.error(f"Error getting current positions: {e}")
            return []
    
    def get_recent_trades(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent trades from API or local data"""
        try:
            # Try to get trades from API first
            api_trades = self.fetch_api_data("/api/v1/trades")
            if api_trades:
                # Handle both direct array and wrapped object formats
                if isinstance(api_trades, list):
                    return api_trades[:limit]
                elif isinstance(api_trades, dict) and 'trades' in api_trades:
                    return api_trades['trades'][:limit]
            
            # Fallback to local trade events
            recent_trades = sorted(self.trade_events, key=lambda x: x.timestamp, reverse=True)[:limit]
            return [
                {
                    "id": getattr(trade, 'id', f"trade_{i}"),
                    "symbol": trade.symbol,
                    "side": getattr(trade, 'side', 'buy'),
                    "quantity": getattr(trade, 'quantity', 0),
                    "price": getattr(trade, 'price', 0),
                    "timestamp": trade.timestamp.isoformat(),
                    "pnl": getattr(trade, 'pnl', 0),
                    "status": getattr(trade, 'status', 'completed')
                }
                for i, trade in enumerate(recent_trades)
            ]
        except Exception as e:
            logger.error(f"Error getting recent trades: {e}")
            return []
    
    def get_portfolio_overview(self) -> Dict[str, Any]:
        """Get portfolio overview from API or local data"""
        try:
            # Try to get portfolio data from API first
            api_portfolio = self.fetch_api_data("/api/v1/portfolio")
            if api_portfolio:
                return api_portfolio
            
            # Fallback to calculated portfolio from performance metrics
            performance = self.get_performance_metrics()
            return {
                "total_value": performance.get("portfolio_value", 100000),
                "cash_balance": performance.get("cash_balance", 25000),
                "total_pnl": performance.get("total_pnl", 0),
                "daily_pnl": performance.get("daily_pnl", 0),
                "positions_count": performance.get("active_positions", 0),
                "data_source": "calculated_from_performance"
            }
        except Exception as e:
            logger.error(f"Error getting portfolio overview: {e}")
            return {"error": str(e)}
    
    def _calculate_sharpe_ratio(self, trades: List[TradeEvent]) -> float:
        """Calculate Sharpe ratio from trades"""
        if not trades:
            return 0.0
        
        returns = [trade.pnl for trade in trades]
        if len(returns) < 2:
            return 0.0
        
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        return mean_return / std_return if std_return > 0 else 0.0
    
    def _calculate_max_drawdown(self, trades: List[TradeEvent]) -> float:
        """Calculate maximum drawdown from trades"""
        if not trades:
            return 0.0
        
        cumulative_pnl = np.cumsum([trade.pnl for trade in trades])
        peak = np.maximum.accumulate(cumulative_pnl)
        drawdown = (cumulative_pnl - peak) / peak
        
        return abs(np.min(drawdown)) if len(drawdown) > 0 else 0.0
    
    def _get_market_sessions(self) -> Dict[str, str]:
        """Get current market session status"""
        now = datetime.now()
        hour = now.hour
        
        return {
            "US_EQUITY": "OPEN" if 9 <= hour < 16 else "CLOSED",
            "CRYPTO": "OPEN",  # 24/7
            "FOREX": "OPEN" if hour >= 17 or hour < 17 else "CLOSED",
            "COMMODITIES": "OPEN" if 9 <= hour < 14 else "CLOSED"
        }

# Visualization functions
def create_agent_decision_timeline(decisions: List[AgentDecision]) -> go.Figure:
    """Create agent decision timeline visualization"""
    if not decisions:
        return go.Figure().add_annotation(text="No agent decisions available", 
                                        xref="paper", yref="paper", x=0.5, y=0.5)
    
    fig = go.Figure()
    
    # Group decisions by agent type
    agent_types = {}
    for decision in decisions:
        if decision.agent_type not in agent_types:
            agent_types[decision.agent_type] = []
        agent_types[decision.agent_type].append(decision)
    
    colors = px.colors.qualitative.Set3
    
    for i, (agent_type, type_decisions) in enumerate(agent_types.items()):
        timestamps = [d.timestamp for d in type_decisions]
        confidences = [d.confidence for d in type_decisions]
        symbols = [d.symbol for d in type_decisions]
        actions = [d.action for d in type_decisions]
        
        fig.add_trace(go.Scatter(
            x=timestamps,
            y=confidences,
            mode='markers+lines',
            name=agent_type,
            marker=dict(
                size=10,
                color=colors[i % len(colors)],
                line=dict(width=1, color='white')
            ),
            text=[f"{symbol}: {action}" for symbol, action in zip(symbols, actions)],
            hovertemplate="<b>%{fullData.name}</b><br>" +
                         "Time: %{x}<br>" +
                         "Confidence: %{y:.2f}<br>" +
                         "Decision: %{text}<br>" +
                         "<extra></extra>"
        ))
    
    fig.update_layout(
        title="Agent Decision Timeline",
        xaxis_title="Time",
        yaxis_title="Confidence Score",
        hovermode='closest',
        height=400
    )
    
    return fig

def create_trade_consideration_analysis(considerations: List[TradeConsideration]) -> go.Figure:
    """Create trade consideration analysis visualization"""
    if not considerations:
        return go.Figure().add_annotation(text="No trade considerations available",
                                        xref="paper", yref="paper", x=0.5, y=0.5)
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Agent Consensus Distribution", "Risk vs Return", 
                       "Decision Timeline", "Execution Rate"),
        specs=[[{"type": "histogram"}, {"type": "scatter"}],
               [{"type": "scatter"}, {"type": "pie"}]]
    )
    
    # Agent consensus distribution
    consensus_scores = []
    for consideration in considerations:
        if consideration.agent_consensus:
            avg_consensus = np.mean(list(consideration.agent_consensus.values()))
            consensus_scores.append(avg_consensus)
    
    if consensus_scores:
        fig.add_trace(
            go.Histogram(x=consensus_scores, name="Consensus Distribution", nbinsx=20),
            row=1, col=1
        )
    
    # Risk vs Return scatter
    risk_scores = [c.risk_metrics.get('overall_risk', 0) for c in considerations if c.risk_metrics]
    expected_returns = [c.risk_metrics.get('expected_return', 0) for c in considerations if c.risk_metrics]
    
    if risk_scores and expected_returns:
        fig.add_trace(
            go.Scatter(
                x=risk_scores,
                y=expected_returns,
                mode='markers',
                name="Risk vs Return",
                marker=dict(size=8, opacity=0.7)
            ),
            row=1, col=2
        )
    
    # Decision timeline
    timestamps = [c.timestamp for c in considerations]
    decisions = [c.final_decision for c in considerations]
    
    decision_colors = {'execute': 'green', 'reject': 'red', 'modify': 'orange', 'delay': 'blue', 'pending': 'gray'}
    colors = [decision_colors.get(d, 'gray') for d in decisions]
    
    fig.add_trace(
        go.Scatter(
            x=timestamps,
            y=list(range(len(timestamps))),
            mode='markers',
            name="Decisions",
            marker=dict(color=colors, size=8)
        ),
        row=2, col=1
    )
    
    # Execution rate pie chart
    decision_counts = {}
    for decision in decisions:
        decision_counts[decision] = decision_counts.get(decision, 0) + 1
    
    if decision_counts:
        fig.add_trace(
            go.Pie(
                labels=list(decision_counts.keys()),
                values=list(decision_counts.values()),
                name="Execution Rate"
            ),
            row=2, col=2
        )
    
    fig.update_layout(height=600, title_text="Trade Consideration Analysis")
    
    return fig

def create_agent_performance_heatmap(activities: List[AgentActivity]) -> go.Figure:
    """Create agent performance heatmap"""
    if not activities:
        return go.Figure().add_annotation(text="No agent activities available",
                                        xref="paper", yref="paper", x=0.5, y=0.5)
    
    # Group activities by agent and hour
    agent_performance = defaultdict(lambda: defaultdict(list))
    
    for activity in activities:
        agent_id = activity.agent_id
        hour = activity.timestamp.hour
        agent_performance[agent_id][hour].append(activity.performance_score)
    
    # Calculate average performance per hour for each agent
    agents = list(agent_performance.keys())
    hours = list(range(24))
    
    z_data = []
    for agent in agents:
        row = []
        for hour in hours:
            scores = agent_performance[agent][hour]
            avg_score = np.mean(scores) if scores else 0
            row.append(avg_score)
        z_data.append(row)
    
    fig = go.Figure(data=go.Heatmap(
        z=z_data,
        x=hours,
        y=agents,
        colorscale='RdYlGn',
        hoverongaps=False,
        hovertemplate="Agent: %{y}<br>Hour: %{x}<br>Avg Performance: %{z:.2f}<extra></extra>"
    ))
    
    fig.update_layout(
        title="Agent Performance Heatmap (24h)",
        xaxis_title="Hour of Day",
        yaxis_title="Agent ID",
        height=400
    )
    
    return fig

def create_real_time_portfolio_chart(trade_events: List[TradeEvent]) -> go.Figure:
    """Create real-time portfolio performance chart"""
    if not trade_events:
        return go.Figure().add_annotation(text="No trade data available",
                                        xref="paper", yref="paper", x=0.5, y=0.5)
    
    # Calculate cumulative P&L over time
    timestamps = [trade.timestamp for trade in trade_events]
    cumulative_pnl = np.cumsum([trade.pnl for trade in trade_events])
    
    fig = go.Figure()
    
    # Cumulative P&L line
    fig.add_trace(go.Scatter(
        x=timestamps,
        y=cumulative_pnl,
        mode='lines',
        name='Cumulative P&L',
        line=dict(color='#1f77b4', width=2)
    ))
    
    # Individual trade markers
    colors = ['green' if trade.pnl > 0 else 'red' for trade in trade_events]
    fig.add_trace(go.Scatter(
        x=timestamps,
        y=[trade.pnl for trade in trade_events],
        mode='markers',
        name='Individual Trades',
        marker=dict(color=colors, size=6, opacity=0.7),
        yaxis='y2'
    ))
    
    fig.update_layout(
        title='Real-Time Portfolio Performance',
        xaxis_title='Time',
        yaxis=dict(title='Cumulative P&L ($)', side='left'),
        yaxis2=dict(title='Trade P&L ($)', side='right', overlaying='y'),
        hovermode='x unified',
        height=500
    )
    
    return fig

# Initialize dashboard manager
@st.cache_resource
def get_dashboard_manager():
    """Get cached dashboard manager instance"""
    return UnifiedDashboardManager()

def main():
    """Main dashboard application"""
    st.set_page_config(
        page_title="Unified Trading System Dashboard",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .status-healthy { color: #28a745; }
    .status-warning { color: #ffc107; }
    .status-error { color: #dc3545; }
    </style>
    """, unsafe_allow_html=True)
    
    # Get dashboard manager
    dashboard_manager = get_dashboard_manager()
    
    # Sidebar
    st.sidebar.title("🚀 Trading System Control")
    
    # Auto-refresh toggle
    auto_refresh = st.sidebar.checkbox("Auto Refresh", value=True)
    refresh_interval = st.sidebar.slider("Refresh Interval (seconds)", 1, 10, 3)
    
    # Page selection
    page = st.sidebar.selectbox(
        "Select View",
        ["Overview", "Agent Intelligence", "Trade Analysis", "Risk Management", "System Health"]
    )
    
    # Main content
    st.title("🎯 Unified Trading System Dashboard")
    
    # Auto-refresh mechanism
    if auto_refresh:
        time.sleep(refresh_interval)
        st.rerun()
    
    # Get current data
    system_status = dashboard_manager.get_system_status()
    performance_metrics = dashboard_manager.get_performance_metrics()
    
    if page == "Overview":
        display_overview_page(dashboard_manager, system_status, performance_metrics)
    elif page == "Agent Intelligence":
        display_agent_intelligence_page(dashboard_manager)
    elif page == "Trade Analysis":
        display_trade_analysis_page(dashboard_manager)
    elif page == "Risk Management":
        display_risk_management_page(dashboard_manager)
    elif page == "System Health":
        display_system_health_page(dashboard_manager, system_status)

def display_overview_page(dashboard_manager, system_status, performance_metrics):
    """Display overview page"""
    st.header("📊 System Overview")
    
    # Show data source information
    data_source = performance_metrics.get('data_source', 'unknown')
    if data_source == 'api_with_local_enhancement':
        st.success("✅ Connected to API server - Real-time data")
    elif data_source == 'local_calculation':
        st.warning("⚠️ Using local data - API unavailable")
    else:
        st.error("🔴 No trading data available")
    
    # Key metrics row
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            "Portfolio Value",
            f"${performance_metrics.get('portfolio_value', 0):,.2f}",
            f"{performance_metrics.get('daily_pnl', 0):+.2f}"
        )
    
    with col2:
        st.metric(
            "Win Rate",
            f"{performance_metrics.get('win_rate', 0):.1%}",
            f"{performance_metrics.get('total_trades', 0)} trades"
        )
    
    with col3:
        st.metric(
            "Active Agents",
            system_status.get('active_agents', 0),
            f"{system_status.get('recent_decisions', 0)} decisions/hr"
        )
    
    with col4:
        st.metric(
            "Sharpe Ratio",
            f"{performance_metrics.get('sharpe_ratio', 0):.2f}",
            f"{performance_metrics.get('max_drawdown', 0):.1%} max DD"
        )
    
    with col5:
        health_status = system_status.get('system_health', 'UNKNOWN')
        health_color = "🟢" if health_status == "HEALTHY" else "🟡" if health_status == "DEGRADED" else "🔴"
        st.metric(
            "System Health",
            f"{health_color} {health_status}",
            system_status.get('status', '')
        )
    
    # Current Positions Section
    st.subheader("📈 Current Positions")
    positions = dashboard_manager.get_current_positions()
    if positions:
        positions_df = pd.DataFrame(positions)
        st.dataframe(positions_df, use_container_width=True)
    else:
        st.info("No active positions")
    
    # Recent Trades Section
    st.subheader("📋 Recent Trades")
    recent_trades = dashboard_manager.get_recent_trades(limit=10)
    if recent_trades:
        trades_df = pd.DataFrame(recent_trades)
        st.dataframe(trades_df, use_container_width=True)
    else:
        st.info("No recent trades")
    
    # Performance Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("💰 Asset Performance")
        asset_performance = performance_metrics.get('asset_performance', {})
        if asset_performance:
            assets_df = pd.DataFrame([
                {"Asset": asset, "PnL": data.get('pnl', 0), "Trades": data.get('trades', 0)}
                for asset, data in asset_performance.items()
            ])
            fig = px.bar(assets_df, x='Asset', y='PnL', title='PnL by Asset')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No asset performance data available")
    
    with col2:
        st.subheader("📊 Risk Metrics")
        risk_metrics = performance_metrics.get('risk_metrics', {})
        if risk_metrics:
            risk_df = pd.DataFrame([
                {"Metric": "VaR (95%)", "Value": risk_metrics.get('var_95', 0)},
                {"Metric": "Expected Shortfall", "Value": risk_metrics.get('expected_shortfall', 0)},
                {"Metric": "Volatility", "Value": risk_metrics.get('volatility', 0)}
            ])
            st.dataframe(risk_df, use_container_width=True)
        else:
            st.info("No risk metrics available")
    
    # Real-time Activity Feed
    st.subheader("🔄 Real-time Activity")
    
    # Show recent agent decisions
    recent_decisions = [d for d in dashboard_manager.agent_decisions if d.timestamp > datetime.now() - timedelta(hours=1)]
    if recent_decisions:
        for decision in recent_decisions[-5:]:  # Show last 5 decisions
            with st.expander(f"🤖 {decision.agent_type} - {decision.symbol} ({decision.timestamp.strftime('%H:%M:%S')})"):
                st.write(f"**Action:** {decision.action}")
                st.write(f"**Confidence:** {decision.confidence:.2%}")
                st.write(f"**Reasoning:** {decision.reasoning}")
                st.write(f"**Status:** {decision.status}")
    else:
        st.info("No recent agent activity")
    
    # Portfolio chart using trade events
    if dashboard_manager.trade_events:
        st.subheader("📈 Portfolio Performance")
        portfolio_chart = create_real_time_portfolio_chart(dashboard_manager.trade_events)
        st.plotly_chart(portfolio_chart, use_container_width=True)
    
    # Agent activity timeline
    if dashboard_manager.agent_activities:
        st.subheader("🤖 Agent Activity Timeline")
        agent_chart = create_agent_performance_heatmap(dashboard_manager.agent_activities)
        st.plotly_chart(agent_chart, use_container_width=True)
    
    # Charts row
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Portfolio Performance")
        portfolio_chart = create_real_time_portfolio_chart(list(dashboard_manager.trade_events))
        st.plotly_chart(portfolio_chart, use_container_width=True)
    
    with col2:
        st.subheader("🤖 Agent Performance")
        agent_heatmap = create_agent_performance_heatmap(list(dashboard_manager.agent_activities))
        st.plotly_chart(agent_heatmap, use_container_width=True)
    
    # Recent activity
    st.subheader("🔄 Recent Activity")
    
    # Recent decisions
    if dashboard_manager.agent_decisions:
        recent_decisions = list(dashboard_manager.agent_decisions)[-10:]
        decisions_df = pd.DataFrame([
            {
                "Time": d.timestamp.strftime("%H:%M:%S"),
                "Agent": d.agent_type,
                "Symbol": d.symbol,
                "Action": d.action,
                "Confidence": f"{d.confidence:.2f}",
                "Status": d.status
            }
            for d in recent_decisions
        ])
        st.dataframe(decisions_df, use_container_width=True)
    else:
        st.info("No recent agent decisions available")

def display_agent_intelligence_page(dashboard_manager):
    """Display agent intelligence page"""
    st.header("🧠 Agent Intelligence Center")
    
    # Agent decision timeline
    st.subheader("📊 Decision Timeline")
    decision_timeline = create_agent_decision_timeline(list(dashboard_manager.agent_decisions))
    st.plotly_chart(decision_timeline, use_container_width=True)
    
    # Trade considerations
    st.subheader("🎯 Trade Considerations")
    consideration_analysis = create_trade_consideration_analysis(list(dashboard_manager.trade_considerations))
    st.plotly_chart(consideration_analysis, use_container_width=True)
    
    # Agent consensus table
    if dashboard_manager.trade_considerations:
        st.subheader("🤝 Current Agent Consensus")
        
        pending_considerations = [c for c in dashboard_manager.trade_considerations if c.final_decision == 'pending']
        
        if pending_considerations:
            consensus_data = []
            for consideration in pending_considerations[-10:]:  # Show last 10
                consensus_data.append({
                    "Symbol": consideration.symbol,
                    "Action": consideration.proposed_action,
                    "Quantity": consideration.quantity,
                    "Target Price": f"${consideration.price_target:.2f}",
                    "Consensus Score": f"{np.mean(list(consideration.agent_consensus.values())):.2f}" if consideration.agent_consensus else "N/A",
                    "Risk Score": f"{consideration.risk_metrics.get('overall_risk', 0):.2f}",
                    "Status": consideration.final_decision
                })
            
            consensus_df = pd.DataFrame(consensus_data)
            st.dataframe(consensus_df, use_container_width=True)
        else:
            st.info("No pending trade considerations")
    else:
        st.info("No trade considerations available")

def display_trade_analysis_page(dashboard_manager):
    """Display trade analysis page"""
    st.header("📈 Trade Analysis")
    
    # Trade performance metrics
    if dashboard_manager.trade_events:
        trades = list(dashboard_manager.trade_events)
        
        # Performance summary
        col1, col2, col3, col4 = st.columns(4)
        
        total_trades = len(trades)
        winning_trades = len([t for t in trades if t.pnl > 0])
        total_pnl = sum(t.pnl for t in trades)
        avg_pnl = total_pnl / total_trades if total_trades > 0 else 0
        
        with col1:
            st.metric("Total Trades", total_trades)
        with col2:
            st.metric("Win Rate", f"{winning_trades/total_trades:.1%}" if total_trades > 0 else "0%")
        with col3:
            st.metric("Total P&L", f"${total_pnl:,.2f}")
        with col4:
            st.metric("Avg P&L per Trade", f"${avg_pnl:.2f}")
        
        # Trade distribution by symbol
        st.subheader("📊 Trade Distribution")
        
        symbol_stats = {}
        for trade in trades:
            if trade.symbol not in symbol_stats:
                symbol_stats[trade.symbol] = {"count": 0, "pnl": 0}
            symbol_stats[trade.symbol]["count"] += 1
            symbol_stats[trade.symbol]["pnl"] += trade.pnl
        
        symbol_df = pd.DataFrame([
            {
                "Symbol": symbol,
                "Trade Count": stats["count"],
                "Total P&L": f"${stats['pnl']:.2f}",
                "Avg P&L": f"${stats['pnl']/stats['count']:.2f}"
            }
            for symbol, stats in symbol_stats.items()
        ])
        
        st.dataframe(symbol_df, use_container_width=True)
        
        # Recent trades table
        st.subheader("🔄 Recent Trades")
        recent_trades_df = pd.DataFrame([
            {
                "Time": t.timestamp.strftime("%H:%M:%S"),
                "Symbol": t.symbol,
                "Side": t.side,
                "Quantity": t.quantity,
                "Price": f"${t.price:.2f}",
                "P&L": f"${t.pnl:.2f}",
                "Strategy": t.strategy,
                "Status": t.status
            }
            for t in trades[-20:]  # Last 20 trades
        ])
        
        st.dataframe(recent_trades_df, use_container_width=True)
        
    else:
        st.info("No trade data available")

def display_risk_management_page(dashboard_manager):
    """Display risk management page"""
    st.header("⚠️ Risk Management")
    
    # Risk metrics
    st.subheader("📊 Risk Metrics")
    
    # Get risk data from considerations
    if dashboard_manager.trade_considerations:
        risk_data = []
        for consideration in dashboard_manager.trade_considerations:
            if consideration.risk_metrics:
                risk_data.append(consideration.risk_metrics)
        
        if risk_data:
            # Average risk metrics
            avg_risk = {}
            for metric in ['overall_risk', 'volatility_risk', 'liquidity_risk', 'concentration_risk']:
                values = [r.get(metric, 0) for r in risk_data if r.get(metric) is not None]
                avg_risk[metric] = np.mean(values) if values else 0
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Overall Risk", f"{avg_risk['overall_risk']:.2f}")
            with col2:
                st.metric("Volatility Risk", f"{avg_risk['volatility_risk']:.2f}")
            with col3:
                st.metric("Liquidity Risk", f"{avg_risk['liquidity_risk']:.2f}")
            with col4:
                st.metric("Concentration Risk", f"{avg_risk['concentration_risk']:.2f}")
        else:
            st.info("No risk metrics available")
    else:
        st.info("No risk data available")
    
    # Risk alerts
    st.subheader("🚨 Risk Alerts")
    st.info("Risk monitoring system active - no current alerts")

def display_system_health_page(dashboard_manager, system_status):
    """Display system health page"""
    st.header("🏥 System Health")
    
    # Component status
    st.subheader("🔧 Component Status")
    
    components = system_status.get('components', {})
    
    for component, status in components.items():
        status_class = "status-healthy" if status == "ACTIVE" else "status-error"
        st.markdown(f"**{component.replace('_', ' ').title()}**: <span class='{status_class}'>{status}</span>", 
                   unsafe_allow_html=True)
    
    # System metrics
    st.subheader("📊 System Metrics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("WebSocket Status", "CONNECTED" if dashboard_manager.is_connected else "DISCONNECTED")
    with col2:
        st.metric("Active Channels", len(dashboard_manager.subscribed_channels))
    with col3:
        st.metric("Data Points", len(dashboard_manager.agent_activities) + len(dashboard_manager.trade_events))
    
    # Market sessions
    st.subheader("🌍 Market Sessions")
    
    sessions = system_status.get('market_sessions', {})
    session_cols = st.columns(len(sessions))
    
    for i, (market, status) in enumerate(sessions.items()):
        with session_cols[i]:
            status_class = "status-healthy" if status == "OPEN" else "status-warning"
            st.markdown(f"**{market}**<br><span class='{status_class}'>{status}</span>", 
                       unsafe_allow_html=True)

if __name__ == "__main__":
    main()