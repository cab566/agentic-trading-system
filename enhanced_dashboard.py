#!/usr/bin/env python3
"""
Enhanced Real-Time Trading Dashboard
====================================

A comprehensive real-time dashboard for monitoring:
- Live agent activity and decision making
- Real-time market data and trading performance
- System health and data source status
- Live portfolio tracking and risk metrics
- Trading alerts and notifications

Features:
- WebSocket-based real-time updates
- Multi-tab interface for different views
- Interactive charts and visualizations
- Live agent conversation logs
- Real-time P&L and position tracking
"""

import streamlit as st
import asyncio
import json
import websocket
import threading
import time
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import requests
from typing import Dict, List, Any, Optional
import logging
from dataclasses import dataclass
import numpy as np
from collections import deque
import queue
import random

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
API_BASE_URL = "http://localhost:8000"
WS_URL = "ws://localhost:8000/ws"
REFRESH_INTERVAL = 1  # seconds

@dataclass
class AgentActivity:
    """Agent activity data structure"""
    agent_id: str
    agent_type: str
    timestamp: datetime
    action: str
    details: Dict[str, Any]
    status: str
    performance_score: float

@dataclass
class MarketDataPoint:
    """Market data point structure"""
    symbol: str
    price: float
    volume: int
    timestamp: datetime
    change: float
    change_percent: float

@dataclass
class TradeEvent:
    """Trade event structure"""
    trade_id: str
    symbol: str
    side: str
    quantity: float
    price: float
    timestamp: datetime
    status: str
    pnl: float
    strategy: str

class RealTimeDataManager:
    """Manages real-time data connections and updates"""
    
    def __init__(self):
        self.ws_connection = None
        self.data_queue = queue.Queue()
        self.is_connected = False
        self.subscribed_channels = set()
        
        # Data storage
        self.agent_activities = deque(maxlen=1000)
        self.market_data = {}
        self.trade_events = deque(maxlen=500)
        self.system_metrics = {}
        self.portfolio_data = {}
        
    async def connect_websocket(self):
        """Connect to WebSocket server"""
        try:
            self.ws_connection = await websockets.connect(WS_URL)
            self.is_connected = True
            logger.info("WebSocket connected successfully")
            
            # Subscribe to all relevant channels
            await self.subscribe_to_channels([
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
    
    async def listen_for_updates(self):
        """Listen for WebSocket updates"""
        if not self.ws_connection:
            return
            
        try:
            async for message in self.ws_connection:
                data = json.loads(message)
                self.process_websocket_message(data)
        except Exception as e:
            logger.error(f"WebSocket listening error: {e}")
            self.is_connected = False
    
    def process_websocket_message(self, data: Dict[str, Any]):
        """Process incoming WebSocket messages"""
        message_type = data.get('type')
        
        if message_type == 'agent_activity':
            self.process_agent_activity(data['data'])
        elif message_type == 'market_data':
            self.process_market_data(data['data'])
        elif message_type == 'trade_update':
            self.process_trade_event(data['data'])
        elif message_type == 'portfolio_update':
            self.process_portfolio_update(data['data'])
        elif message_type == 'system_health':
            self.process_system_health(data['data'])
    
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
    
    def process_portfolio_update(self, data: Dict[str, Any]):
        """Process portfolio updates"""
        self.portfolio_data = data
    
    def process_system_health(self, data: Dict[str, Any]):
        """Process system health updates"""
        self.system_metrics = data

# Global data manager
@st.cache_resource
def get_data_manager():
    return RealTimeDataManager()

def fetch_api_data(endpoint: str) -> Optional[Dict[str, Any]]:
    """Fetch data from API endpoint"""
    try:
        response = requests.get(f"{API_BASE_URL}{endpoint}", timeout=5)
        if response.status_code == 200:
            return response.json()
        else:
            logger.warning(f"API request failed: {response.status_code}")
            return None
    except Exception as e:
        logger.error(f"API request error: {e}")
        return None

def create_agent_activity_chart(activities: List[AgentActivity]) -> go.Figure:
    """Create agent activity timeline chart"""
    if not activities:
        return go.Figure().add_annotation(text="No agent activity data", showarrow=False)
    
    # Convert to DataFrame for easier manipulation
    df_data = []
    for activity in activities[-50:]:  # Last 50 activities
        df_data.append({
            'timestamp': activity.timestamp,
            'agent_id': activity.agent_id,
            'agent_type': activity.agent_type,
            'action': activity.action,
            'status': activity.status,
            'performance_score': activity.performance_score
        })
    
    df = pd.DataFrame(df_data)
    
    # Create timeline chart
    fig = px.scatter(df, 
                    x='timestamp', 
                    y='agent_id',
                    color='agent_type',
                    size='performance_score',
                    hover_data=['action', 'status'],
                    title="Real-Time Agent Activity Timeline")
    
    fig.update_layout(
        height=400,
        xaxis_title="Time",
        yaxis_title="Agent ID",
        showlegend=True
    )
    
    return fig

def create_market_data_chart(market_data: Dict[str, MarketDataPoint]) -> go.Figure:
    """Create real-time market data chart"""
    if not market_data:
        return go.Figure().add_annotation(text="No market data available", showarrow=False)
    
    # Create subplots for price and volume
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Price Movement', 'Volume'),
        vertical_spacing=0.1,
        row_heights=[0.7, 0.3]
    )
    
    for symbol, data in market_data.items():
        # Price chart
        fig.add_trace(
            go.Scatter(
                x=[data.timestamp],
                y=[data.price],
                mode='markers+lines',
                name=f"{symbol} Price",
                line=dict(color='green' if data.change >= 0 else 'red')
            ),
            row=1, col=1
        )
        
        # Volume chart
        fig.add_trace(
            go.Bar(
                x=[data.timestamp],
                y=[data.volume],
                name=f"{symbol} Volume",
                showlegend=False
            ),
            row=2, col=1
        )
    
    fig.update_layout(
        height=500,
        title="Real-Time Market Data",
        showlegend=True
    )
    
    return fig

def create_portfolio_performance_chart(portfolio_data: Dict[str, Any]) -> go.Figure:
    """Create portfolio performance chart"""
    if not portfolio_data:
        return go.Figure().add_annotation(text="No portfolio data available", showarrow=False)
    
    # Create gauge chart for portfolio performance
    fig = go.Figure()
    
    total_value = portfolio_data.get('total_value', 0)
    day_pnl = portfolio_data.get('day_pnl', 0)
    day_pnl_percent = (day_pnl / total_value * 100) if total_value > 0 else 0
    
    fig.add_trace(go.Indicator(
        mode = "gauge+number+delta",
        value = day_pnl_percent,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Daily P&L %"},
        delta = {'reference': 0},
        gauge = {
            'axis': {'range': [-10, 10]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [-10, -2], 'color': "lightgray"},
                {'range': [-2, 2], 'color': "gray"},
                {'range': [2, 10], 'color': "lightgreen"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 5
            }
        }
    ))
    
    fig.update_layout(height=300)
    return fig

def create_system_health_indicators(system_metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Create system health indicators"""
    if not system_metrics:
        return {
            'status': 'Unknown',
            'uptime': 'N/A',
            'connections': 'N/A',
            'errors': 'N/A'
        }
    
    return {
        'status': system_metrics.get('system_status', 'Unknown'),
        'uptime': f"{system_metrics.get('uptime_hours', 0):.1f}h",
        'connections': system_metrics.get('api_connections', {}),
        'errors': system_metrics.get('errors_count', 0)
    }

def main():
    """Main dashboard application"""
    st.set_page_config(
        page_title="Enhanced Trading Dashboard",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize data manager
    data_manager = get_data_manager()
    
    # Dashboard header
    st.title("🚀 Enhanced Real-Time Trading Dashboard")
    st.markdown("---")
    
    # Sidebar for controls
    with st.sidebar:
        st.header("Dashboard Controls")
        
        # Connection status
        if data_manager.is_connected:
            st.success("🟢 WebSocket Connected")
        else:
            st.error("🔴 WebSocket Disconnected")
            if st.button("Reconnect"):
                # Attempt reconnection (simplified for demo)
                st.info("Attempting to reconnect...")
        
        # Refresh controls
        auto_refresh = st.checkbox("Auto Refresh", value=True)
        refresh_rate = st.slider("Refresh Rate (seconds)", 1, 10, REFRESH_INTERVAL)
        
        # Data source status
        st.subheader("Data Sources")
        api_data = fetch_api_data("/api/v1/health")
        if api_data:
            for service, status in api_data.get('services', {}).items():
                if status.get('healthy', False):
                    st.success(f"✅ {service}")
                else:
                    st.error(f"❌ {service}")
    
    # Main dashboard tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🤖 Agent Activity", 
        "📊 Market Data", 
        "💼 Portfolio", 
        "🔧 System Health",
        "📋 Trade Log"
    ])
    
    # Tab 1: Agent Activity
    with tab1:
        st.header("Real-Time Agent Activity")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Agent activity timeline
            if data_manager.agent_activities:
                fig = create_agent_activity_chart(list(data_manager.agent_activities))
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No agent activity data available. Agents will appear here when active.")
        
        with col2:
            # Agent status summary
            st.subheader("Active Agents")
            
            # Fetch agent status from API
            agent_data = fetch_api_data("/api/v1/agents/status")
            if agent_data:
                for agent in agent_data.get('agents', []):
                    status_color = "🟢" if agent.get('status') == 'active' else "🔴"
                    st.write(f"{status_color} **{agent.get('name', 'Unknown')}**")
                    st.write(f"   Type: {agent.get('type', 'N/A')}")
                    st.write(f"   Performance: {agent.get('performance_score', 0):.2f}")
                    st.write("---")
            else:
                st.info("Loading agent data...")
        
        # Recent agent decisions
        st.subheader("Recent Agent Decisions")
        if data_manager.agent_activities:
            recent_activities = list(data_manager.agent_activities)[-10:]
            for activity in reversed(recent_activities):
                with st.expander(f"{activity.agent_id} - {activity.action} ({activity.timestamp.strftime('%H:%M:%S')})"):
                    st.json(activity.details)
    
    # Tab 2: Market Data
    with tab2:
        st.header("Real-Time Market Data")
        
        # Market data chart
        if data_manager.market_data:
            fig = create_market_data_chart(data_manager.market_data)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No real-time market data available. Data will appear when market is active.")
        
        # Market data table
        st.subheader("Current Prices")
        if data_manager.market_data:
            market_df = pd.DataFrame([
                {
                    'Symbol': data.symbol,
                    'Price': f"${data.price:.2f}",
                    'Change': f"{data.change:+.2f}",
                    'Change %': f"{data.change_percent:+.2f}%",
                    'Volume': f"{data.volume:,}",
                    'Last Update': data.timestamp.strftime('%H:%M:%S')
                }
                for data in data_manager.market_data.values()
            ])
            st.dataframe(market_df, use_container_width=True)
    
    # Tab 3: Portfolio
    with tab3:
        st.header("💼 Portfolio & Trading Overview")
        
        # Portfolio metrics
        portfolio_data = fetch_api_data("/api/v1/portfolio/summary")
        if portfolio_data:
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                st.metric("Total Value", f"${portfolio_data.get('total_value', 0):,.2f}")
            with col2:
                day_pnl = portfolio_data.get('day_pnl', 0)
                pnl_color = "normal" if day_pnl >= 0 else "inverse"
                st.metric("Day P&L", f"${day_pnl:+,.2f}", delta_color=pnl_color)
            with col3:
                st.metric("Cash Balance", f"${portfolio_data.get('cash_balance', 0):,.2f}")
            with col4:
                st.metric("Positions", portfolio_data.get('total_positions', 0))
            with col5:
                unrealized_pnl = portfolio_data.get('unrealized_pnl', 0)
                unrealized_color = "normal" if unrealized_pnl >= 0 else "inverse"
                st.metric("Unrealized P&L", f"${unrealized_pnl:+,.2f}", delta_color=unrealized_color)
            
            # Risk metrics
            st.subheader("📊 Risk Metrics")
            risk_col1, risk_col2, risk_col3, risk_col4 = st.columns(4)
            
            with risk_col1:
                portfolio_beta = portfolio_data.get('beta', 1.0)
                st.metric("Portfolio Beta", f"{portfolio_beta:.2f}")
            with risk_col2:
                sharpe_ratio = portfolio_data.get('sharpe_ratio', 0.0)
                st.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")
            with risk_col3:
                max_drawdown = portfolio_data.get('max_drawdown', 0.0)
                st.metric("Max Drawdown", f"{max_drawdown:.2f}%")
            with risk_col4:
                var_95 = portfolio_data.get('var_95', 0.0)
                st.metric("VaR (95%)", f"${var_95:,.2f}")
            
            # Performance gauge
            fig = create_portfolio_performance_chart(portfolio_data)
            st.plotly_chart(fig, use_container_width=True)
        
        # Positions table with enhanced metrics
        st.subheader("📈 Position Details")
        positions_data = fetch_api_data("/api/v1/portfolio/positions")
        if positions_data and positions_data.get('positions'):
            positions = positions_data['positions']
            positions_df = pd.DataFrame([
                {
                    'Symbol': pos.get('symbol', ''),
                    'Side': pos.get('side', 'LONG'),
                    'Quantity': f"{pos.get('quantity', 0):,}",
                    'Avg Price': f"${pos.get('avg_price', 0):.2f}",
                    'Current Price': f"${pos.get('current_price', 0):.2f}",
                    'Market Value': f"${pos.get('market_value', 0):,.2f}",
                    'Unrealized P&L': f"${pos.get('unrealized_pnl', 0):,.2f}",
                    'P&L %': f"{pos.get('unrealized_pnl_percent', 0):.2f}%",
                    'Weight': f"{pos.get('weight', 0):.1f}%",
                    'Risk Score': pos.get('risk_score', 'N/A')
                }
                for pos in positions
            ])
            st.dataframe(positions_df, use_container_width=True)
            
            # Portfolio allocation and performance charts
            chart_col1, chart_col2 = st.columns(2)
            
            with chart_col1:
                # Portfolio allocation pie chart
                fig_pie = px.pie(
                    values=[pos.get('market_value', 0) for pos in positions],
                    names=[pos.get('symbol', '') for pos in positions],
                    title="Portfolio Allocation by Market Value"
                )
                fig_pie.update_layout(height=400)
                st.plotly_chart(fig_pie, use_container_width=True)
                
            with chart_col2:
                # P&L by position bar chart
                fig_bar = px.bar(
                    x=[pos.get('symbol', '') for pos in positions],
                    y=[pos.get('unrealized_pnl', 0) for pos in positions],
                    title="Unrealized P&L by Position",
                    color=[pos.get('unrealized_pnl', 0) for pos in positions],
                    color_continuous_scale=['red', 'yellow', 'green']
                )
                fig_bar.update_layout(height=400)
                st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.info("No open positions")
            
        # Order book section
        st.subheader("📋 Live Order Book")
        order_book_data = fetch_api_data("/api/v1/market/orderbook")
        if order_book_data:
            book_col1, book_col2 = st.columns(2)
            
            with book_col1:
                st.write("**Bids (Buy Orders)**")
                bids = order_book_data.get('bids', [])
                if bids:
                    # Create DataFrame from dictionary structure
                    bids_data = []
                    for bid in bids:
                        price = bid['price']
                        size = bid['size']
                        total = price * size
                        bids_data.append({'Price': f"${price:.2f}", 'Size': f"{size:,}", 'Total': f"${total:,.2f}"})
                    bids_df = pd.DataFrame(bids_data)
                    st.dataframe(bids_df, use_container_width=True)
                else:
                    st.info("No bid orders")
                    
            with book_col2:
                st.write("**Asks (Sell Orders)**")
                asks = order_book_data.get('asks', [])
                if asks:
                    # Create DataFrame from dictionary structure
                    asks_data = []
                    for ask in asks:
                        price = ask['price']
                        size = ask['size']
                        total = price * size
                        asks_data.append({'Price': f"${price:.2f}", 'Size': f"{size:,}", 'Total': f"${total:,.2f}"})
                    asks_df = pd.DataFrame(asks_data)
                    st.dataframe(asks_df, use_container_width=True)
                else:
                    st.info("No ask orders")
                    
            # Order book spread
            if bids and asks:
                # Extract price from dictionary structure
                best_bid = max([bid['price'] for bid in bids])
                best_ask = min([ask['price'] for ask in asks])
                spread = best_ask - best_bid
                spread_pct = (spread / best_bid) * 100
                
                spread_col1, spread_col2, spread_col3 = st.columns(3)
                with spread_col1:
                    st.metric("Best Bid", f"${best_bid:.2f}")
                with spread_col2:
                    st.metric("Best Ask", f"${best_ask:.2f}")
                with spread_col3:
                    st.metric("Spread", f"${spread:.2f} ({spread_pct:.3f}%)")
        else:
            st.info("No order book data available")
    
    # Tab 4: System Health
    with tab4:
        st.header("🏥 System Health & Performance")
        
        # System metrics
        system_data = fetch_api_data("/api/v1/system/status")
        
        # If no system data available, show error message
        if not system_data:
            st.error("❌ Unable to connect to system health API. Please check if the trading system backend is running.")
            st.info("💡 To start the backend: `docker-compose up trading-system`")
            return
        
        health_indicators = create_system_health_indicators(system_data)
        
        # System overview metrics
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            status_color = "🟢" if health_indicators['status'] == 'RUNNING' else "🔴"
            st.metric("System Status", f"{status_color} {health_indicators['status']}")
        
        with col2:
            st.metric("Uptime", health_indicators['uptime'])
        
        with col3:
            st.metric("Active Strategies", system_data.get('strategies_running', 0))
        
        with col4:
            st.metric("Error Count", health_indicators['errors'])
            
        with col5:
            cpu_usage = system_data.get('cpu_usage', 0)
            cpu_color = "normal" if cpu_usage < 80 else "inverse"
            st.metric("CPU Usage", f"{cpu_usage:.1f}%", delta_color=cpu_color)
            
        # Performance metrics
        st.subheader("⚡ Performance Metrics")
        perf_col1, perf_col2, perf_col3, perf_col4 = st.columns(4)
        
        with perf_col1:
            memory_usage = system_data.get('memory_usage', 0)
            memory_color = "normal" if memory_usage < 85 else "inverse"
            st.metric("Memory Usage", f"{memory_usage:.1f}%", delta_color=memory_color)
            
        with perf_col2:
            avg_latency = system_data.get('avg_latency', 0)
            latency_color = "normal" if avg_latency < 100 else "inverse"
            st.metric("Avg Latency", f"{avg_latency:.1f}ms", delta_color=latency_color)
            
        with perf_col3:
            throughput = system_data.get('throughput', 0)
            st.metric("Throughput", f"{throughput:,} req/s")
            
        with perf_col4:
            active_connections = system_data.get('active_connections', 0)
            st.metric("Active Connections", active_connections)
        
        # Data source health
        st.subheader("📡 Data Source Health")
        health_data = fetch_api_data("/api/v1/health")
        if health_data:
            services = health_data.get('services', {})
            health_df = pd.DataFrame([
                {
                    'Service': service,
                    'Status': '🟢 Healthy' if info.get('healthy', False) else '🔴 Unhealthy',
                    'Response Time': f"{info.get('response_time', 0):.2f}ms",
                    'Last Check': info.get('last_check', 'N/A'),
                    'Health Score': f"{info.get('health_score', 100):.1f}%",
                    'Errors (24h)': info.get('error_count', 0)
                }
                for service, info in services.items()
            ])
            st.dataframe(health_df, use_container_width=True)
            
        # Resource usage trends
        st.subheader("📊 Resource Usage Trends")
        
        # Fetch real system metrics from API
        try:
            response = requests.get(f"{API_BASE_URL}/api/v1/system/status")
            if response.status_code == 200:
                system_data = response.json().get('system_metrics', {})
            else:
                system_data = {}
        except Exception as e:
            st.error(f"Failed to fetch system metrics: {e}")
            system_data = {}
        
        chart_col1, chart_col2 = st.columns(2)
        
        with chart_col1:
            # CPU and Memory usage chart using real data
            cpu_usage = system_data.get('cpu_usage', 0)
            memory_usage = system_data.get('memory_usage', 0)
            
            fig_resources = go.Figure()
            
            fig_resources.add_trace(go.Scatter(
                x=[datetime.now()],
                y=[cpu_usage],
                mode='lines+markers',
                name='CPU Usage (%)',
                line=dict(color='red', width=2)
            ))
            
            fig_resources.add_trace(go.Scatter(
                x=[datetime.now()],
                y=[memory_usage],
                mode='lines+markers',
                name='Memory Usage (%)',
                line=dict(color='orange')
            ))
            
            fig_resources.update_layout(
                title="System Resource Usage",
                xaxis_title="Time",
                yaxis_title="Usage (%)",
                height=400
            )
            
            st.plotly_chart(fig_resources, use_container_width=True)
            
        with chart_col2:
            # Latency chart using real system data
            current_time = datetime.now()
            avg_latency = system_data.get('avg_latency', 0)
            
            fig_latency = go.Figure()
            
            fig_latency.add_trace(go.Scatter(
                x=[current_time],
                y=[avg_latency],
                mode='lines+markers',
                name='Latency (ms)',
                line=dict(color='green', width=2)
            ))
            
            fig_latency.update_layout(
                title="System Latency",
                xaxis_title="Time",
                yaxis_title="Latency (ms)",
                height=400,
                yaxis=dict(range=[0, max(100, avg_latency * 1.2)])
            )
            
            st.plotly_chart(fig_latency, use_container_width=True)
            
        # Error logs
        st.subheader("🚨 Recent System Events")
        error_logs = system_data.get('error_logs', [])
        
        if error_logs:
            error_df = pd.DataFrame([
                {
                    'Timestamp': log.get('timestamp', ''),
                    'Level': log.get('level', 'INFO'),
                    'Service': log.get('service', 'Unknown'),
                    'Message': log.get('message', ''),
                    'Count': log.get('count', 1)
                }
                for log in error_logs[-10:]  # Show last 10 events
            ])
            
            st.dataframe(error_df, use_container_width=True)
        else:
            st.success("No recent errors detected! 🎉")
    
    # Tab 5: Trade Log
    with tab5:
        st.header("Live Trade Log")
        
        # Recent trades
        if data_manager.trade_events:
            trades_df = pd.DataFrame([
                {
                    'Time': trade.timestamp.strftime('%H:%M:%S'),
                    'Symbol': trade.symbol,
                    'Side': trade.side,
                    'Quantity': trade.quantity,
                    'Price': f"${trade.price:.2f}",
                    'P&L': f"${trade.pnl:+.2f}",
                    'Strategy': trade.strategy,
                    'Status': trade.status
                }
                for trade in reversed(list(data_manager.trade_events)[-20:])
            ])
            st.dataframe(trades_df, use_container_width=True)
        else:
            st.info("No trades executed yet. Trade events will appear here in real-time.")
        
        # Trade statistics
        st.subheader("Trading Statistics")
        trade_stats = fetch_api_data("/api/v1/trades/statistics")
        if trade_stats:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Trades", trade_stats.get('total_trades', 0))
            
            with col2:
                st.metric("Successful Trades", trade_stats.get('successful_trades', 0))
            
            with col3:
                win_rate = trade_stats.get('win_rate', 0)
                st.metric("Win Rate", f"{win_rate:.1f}%")
            
            with col4:
                avg_pnl = trade_stats.get('average_pnl', 0)
                st.metric("Avg P&L", f"${avg_pnl:+.2f}")
    
    # Auto-refresh functionality
    if auto_refresh:
        time.sleep(refresh_rate)
        st.rerun()

if __name__ == "__main__":
    main()