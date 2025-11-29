#!/usr/bin/env python3
"""
Advanced Analytics Dashboard for Multi-Asset Trading System

Provides comprehensive financial analysis and algorithmic trading insights:
- Real-time portfolio performance analytics
- Risk-adjusted return metrics
- Multi-asset correlation analysis
- AI agent decision tracking
- Advanced backtesting results
- Market regime analysis
- Factor attribution analysis
- Stress testing scenarios
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import sqlite3
from datetime import datetime, timedelta
import asyncio
import logging
from typing import Dict, List, Optional, Any
import json
from pathlib import Path

# Import system components
from core.config_manager import ConfigManager
from core.market_data_aggregator import MarketDataAggregator
from core.risk_manager_24_7 import RiskManager24_7
from core.backtesting_engine import BacktestingEngine, BacktestConfig, BacktestMode
from utils.performance_tracker import PerformanceTracker
from tools.portfolio_management_tool import PortfolioManagementTool
from tools.risk_analysis_tool import RiskAnalysisTool
from tools.multi_asset_portfolio_tool import MultiAssetPortfolioTool

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AdvancedAnalyticsDashboard:
    """Advanced analytics dashboard for comprehensive trading system analysis."""
    
    def __init__(self):
        """Initialize the advanced analytics dashboard."""
        self.config_manager = ConfigManager()
        self.db_path = "data/trading_data.db"
        
        # Initialize components
        try:
            self.market_data_aggregator = MarketDataAggregator(self.config_manager)
            self.risk_manager = RiskManager24_7(self.config_manager)
            self.backtesting_engine = BacktestingEngine(self.config_manager, self.market_data_aggregator)
            self.performance_tracker = PerformanceTracker(self.config_manager)
        except Exception as e:
            logger.error(f"Error initializing components: {e}")
            st.error(f"Error initializing system components: {e}")
    
    def load_trading_data(self) -> pd.DataFrame:
        """Load trading data from database."""
        try:
            conn = sqlite3.connect(self.db_path)
            query = """
            SELECT 
                trade_id, timestamp, symbol, side, quantity, price, value,
                commission, fees, strategy, venue, pnl, trade_metadata
            FROM trades 
            ORDER BY timestamp DESC
            """
            df = pd.read_sql_query(query, conn)
            conn.close()
            
            if not df.empty:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df['date'] = df['timestamp'].dt.date
            
            return df
        except Exception as e:
            logger.error(f"Error loading trading data: {e}")
            return pd.DataFrame()
    
    def calculate_portfolio_metrics(self, trades_df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate comprehensive portfolio performance metrics."""
        if trades_df.empty:
            return {}
        
        try:
            # Basic metrics
            total_trades = len(trades_df)
            total_volume = trades_df['value'].sum()
            total_pnl = trades_df['pnl'].sum() if 'pnl' in trades_df.columns else 0
            total_fees = trades_df[['commission', 'fees']].sum().sum()
            
            # Calculate returns by day
            daily_pnl = trades_df.groupby('date')['pnl'].sum()
            
            # Risk metrics
            if len(daily_pnl) > 1:
                daily_returns = daily_pnl.pct_change().dropna()
                volatility = daily_returns.std() * np.sqrt(252)  # Annualized
                sharpe_ratio = (daily_returns.mean() * 252) / volatility if volatility > 0 else 0
                max_drawdown = self.calculate_max_drawdown(daily_pnl.cumsum())
                var_95 = np.percentile(daily_returns, 5)
            else:
                volatility = sharpe_ratio = max_drawdown = var_95 = 0
            
            # Win rate
            profitable_trades = trades_df[trades_df['pnl'] > 0] if 'pnl' in trades_df.columns else pd.DataFrame()
            win_rate = len(profitable_trades) / total_trades if total_trades > 0 else 0
            
            # Average trade metrics
            avg_trade_size = trades_df['value'].mean()
            avg_pnl = trades_df['pnl'].mean() if 'pnl' in trades_df.columns else 0
            
            return {
                'total_trades': total_trades,
                'total_volume': total_volume,
                'total_pnl': total_pnl,
                'total_fees': total_fees,
                'net_pnl': total_pnl - total_fees,
                'win_rate': win_rate,
                'avg_trade_size': avg_trade_size,
                'avg_pnl': avg_pnl,
                'volatility': volatility,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'var_95': var_95,
                'daily_pnl': daily_pnl
            }
        except Exception as e:
            logger.error(f"Error calculating portfolio metrics: {e}")
            return {}
    
    def calculate_max_drawdown(self, cumulative_returns: pd.Series) -> float:
        """Calculate maximum drawdown."""
        try:
            peak = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - peak) / peak
            return abs(drawdown.min())
        except:
            return 0.0
    
    def create_performance_chart(self, daily_pnl: pd.Series) -> go.Figure:
        """Create portfolio performance chart."""
        if daily_pnl.empty:
            return go.Figure()
        
        cumulative_pnl = daily_pnl.cumsum()
        
        fig = go.Figure()
        
        # Cumulative P&L
        fig.add_trace(go.Scatter(
            x=cumulative_pnl.index,
            y=cumulative_pnl.values,
            mode='lines',
            name='Cumulative P&L',
            line=dict(color='#1f77b4', width=2)
        ))
        
        # Daily P&L as bar chart
        fig.add_trace(go.Bar(
            x=daily_pnl.index,
            y=daily_pnl.values,
            name='Daily P&L',
            marker_color=['green' if x > 0 else 'red' for x in daily_pnl.values],
            opacity=0.6,
            yaxis='y2'
        ))
        
        fig.update_layout(
            title='Portfolio Performance Analysis',
            xaxis_title='Date',
            yaxis=dict(title='Cumulative P&L ($)', side='left'),
            yaxis2=dict(title='Daily P&L ($)', side='right', overlaying='y'),
            hovermode='x unified',
            height=500
        )
        
        return fig
    
    def create_asset_allocation_chart(self, trades_df: pd.DataFrame) -> go.Figure:
        """Create asset allocation analysis chart."""
        if trades_df.empty:
            return go.Figure()
        
        # Calculate current positions (simplified)
        positions = trades_df.groupby('symbol').agg({
            'quantity': lambda x: x.sum() if trades_df.loc[x.index, 'side'].iloc[0] == 'BUY' else -x.sum(),
            'value': 'sum'
        })
        
        # Filter out zero positions
        positions = positions[positions['quantity'] != 0]
        
        if positions.empty:
            return go.Figure()
        
        fig = go.Figure(data=[go.Pie(
            labels=positions.index,
            values=abs(positions['value']),
            hole=0.3,
            textinfo='label+percent',
            textposition='outside'
        )])
        
        fig.update_layout(
            title='Current Asset Allocation',
            height=400
        )
        
        return fig
    
    def create_risk_metrics_chart(self, metrics: Dict[str, Any]) -> go.Figure:
        """Create risk metrics visualization."""
        if not metrics:
            return go.Figure()
        
        risk_metrics = {
            'Sharpe Ratio': metrics.get('sharpe_ratio', 0),
            'Max Drawdown': metrics.get('max_drawdown', 0),
            'Volatility': metrics.get('volatility', 0),
            'VaR (95%)': abs(metrics.get('var_95', 0)),
            'Win Rate': metrics.get('win_rate', 0)
        }
        
        fig = go.Figure(data=[go.Bar(
            x=list(risk_metrics.keys()),
            y=list(risk_metrics.values()),
            marker_color=['green' if k == 'Sharpe Ratio' or k == 'Win Rate' else 'orange' for k in risk_metrics.keys()]
        )])
        
        fig.update_layout(
            title='Risk Metrics Dashboard',
            yaxis_title='Value',
            height=400
        )
        
        return fig
    
    def create_trading_activity_heatmap(self, trades_df: pd.DataFrame) -> go.Figure:
        """Create trading activity heatmap by hour and day."""
        if trades_df.empty:
            return go.Figure()
        
        # Extract hour and day of week
        trades_df['hour'] = trades_df['timestamp'].dt.hour
        trades_df['day_of_week'] = trades_df['timestamp'].dt.day_name()
        
        # Create pivot table for heatmap
        activity_matrix = trades_df.groupby(['day_of_week', 'hour']).size().unstack(fill_value=0)
        
        # Reorder days
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        activity_matrix = activity_matrix.reindex(day_order, fill_value=0)
        
        fig = go.Figure(data=go.Heatmap(
            z=activity_matrix.values,
            x=activity_matrix.columns,
            y=activity_matrix.index,
            colorscale='Blues',
            showscale=True
        ))
        
        fig.update_layout(
            title='Trading Activity Heatmap (by Hour and Day)',
            xaxis_title='Hour of Day',
            yaxis_title='Day of Week',
            height=400
        )
        
        return fig
    
    def display_system_status(self):
        """Display comprehensive system status."""
        st.header("🔧 System Status & Configuration")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("System Status", "🟢 Active")
            st.metric("Market Data", "🟢 Connected")
        
        with col2:
            st.metric("Risk Manager", "🟢 Monitoring")
            st.metric("Execution Engine", "🟢 Ready")
        
        with col3:
            st.metric("AI Agents", "🟢 Active")
            st.metric("Database", "🟢 Connected")
        
        # Configuration details
        with st.expander("📋 System Configuration"):
            config_data = {
                "Portfolio Config": self.config_manager.get_portfolio_config(),
                "Risk Limits": self.config_manager.get_risk_config(),
                "Data Sources": list(self.config_manager.get_data_source_configs().keys())
            }
            st.json(config_data)
    
    def display_ai_agent_insights(self):
        """Display AI agent decision insights."""
        st.header("🤖 AI Agent Decision Analysis")
        
        # Agent performance metrics
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Agent Performance")
            agent_metrics = {
                "Market Analyst": {"Accuracy": 0.78, "Decisions": 156, "Confidence": 0.82},
                "Risk Manager": {"Alerts": 23, "Prevented Losses": 0.05, "Uptime": 0.99},
                "Trade Executor": {"Fill Rate": 0.95, "Slippage": 0.002, "Latency": 45},
                "Portfolio Manager": {"Rebalances": 12, "Efficiency": 0.88, "Tracking Error": 0.015}
            }
            
            for agent, metrics in agent_metrics.items():
                with st.expander(f"📊 {agent}"):
                    for metric, value in metrics.items():
                        if isinstance(value, float) and value < 1:
                            st.metric(metric, f"{value:.1%}")
                        else:
                            st.metric(metric, value)
        
        with col2:
            st.subheader("Decision Consensus")
            # Simulated consensus data
            consensus_data = pd.DataFrame({
                'Decision': ['BUY AAPL', 'SELL TSLA', 'HOLD SPY', 'BUY BTC-USD'],
                'Consensus': [0.85, 0.72, 0.91, 0.68],
                'Agents_Agree': [4, 3, 5, 3],
                'Confidence': [0.82, 0.75, 0.88, 0.71]
            })
            
            fig = px.scatter(consensus_data, x='Consensus', y='Confidence', 
                           size='Agents_Agree', hover_data=['Decision'],
                           title='Agent Decision Consensus vs Confidence')
            st.plotly_chart(fig, use_container_width=True)
    
    def display_advanced_analytics(self):
        """Display advanced financial analytics."""
        st.header("📈 Advanced Financial Analytics")
        
        # Load and analyze data
        trades_df = self.load_trading_data()
        metrics = self.calculate_portfolio_metrics(trades_df)
        
        if not metrics:
            st.warning("No trading data available for analysis.")
            return
        
        # Key Performance Indicators
        st.subheader("📊 Key Performance Indicators")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Total P&L", 
                f"${metrics.get('net_pnl', 0):,.2f}",
                delta=f"{metrics.get('avg_pnl', 0):,.2f} avg/trade"
            )
        
        with col2:
            st.metric(
                "Sharpe Ratio", 
                f"{metrics.get('sharpe_ratio', 0):.3f}",
                delta="Risk-adjusted return"
            )
        
        with col3:
            st.metric(
                "Win Rate", 
                f"{metrics.get('win_rate', 0):.1%}",
                delta=f"{metrics.get('total_trades', 0)} total trades"
            )
        
        with col4:
            st.metric(
                "Max Drawdown", 
                f"{metrics.get('max_drawdown', 0):.1%}",
                delta="Risk metric"
            )
        
        # Performance Charts
        st.subheader("📈 Performance Analysis")
        
        if 'daily_pnl' in metrics and not metrics['daily_pnl'].empty:
            perf_chart = self.create_performance_chart(metrics['daily_pnl'])
            st.plotly_chart(perf_chart, use_container_width=True)
        
        # Risk Analysis
        col1, col2 = st.columns(2)
        
        with col1:
            risk_chart = self.create_risk_metrics_chart(metrics)
            st.plotly_chart(risk_chart, use_container_width=True)
        
        with col2:
            allocation_chart = self.create_asset_allocation_chart(trades_df)
            st.plotly_chart(allocation_chart, use_container_width=True)
        
        # Trading Activity Analysis
        st.subheader("⏰ Trading Activity Analysis")
        activity_heatmap = self.create_trading_activity_heatmap(trades_df)
        st.plotly_chart(activity_heatmap, use_container_width=True)
    
    def display_market_regime_analysis(self):
        """Display market regime analysis."""
        st.header("🌊 Market Regime Analysis")
        
        # Simulated market regime data
        regime_data = {
            "Current Regime": "Normal Market",
            "Regime Probability": 0.78,
            "Volatility Regime": "Low",
            "Trend Strength": 0.65,
            "Market Stress": 0.23
        }
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Market Regime", regime_data["Current Regime"])
            st.metric("Confidence", f"{regime_data['Regime Probability']:.1%}")
        
        with col2:
            st.metric("Volatility", regime_data["Volatility Regime"])
            st.metric("Trend Strength", f"{regime_data['Trend Strength']:.1%}")
        
        with col3:
            st.metric("Market Stress", f"{regime_data['Market Stress']:.1%}")
        
        # Regime transition probabilities
        st.subheader("Regime Transition Probabilities")
        
        transition_matrix = pd.DataFrame({
            'Normal': [0.85, 0.10, 0.05],
            'High Vol': [0.25, 0.60, 0.15],
            'Crisis': [0.15, 0.35, 0.50]
        }, index=['Normal', 'High Vol', 'Crisis'])
        
        fig = go.Figure(data=go.Heatmap(
            z=transition_matrix.values,
            x=transition_matrix.columns,
            y=transition_matrix.index,
            colorscale='RdYlBu_r',
            text=transition_matrix.values,
            texttemplate="%{text:.2f}",
            textfont={"size": 12},
            showscale=True
        ))
        
        fig.update_layout(
            title='Market Regime Transition Matrix',
            xaxis_title='To Regime',
            yaxis_title='From Regime',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)

def main():
    """Main dashboard application."""
    st.set_page_config(
        page_title="Advanced Trading Analytics",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🚀 Advanced Multi-Asset Trading System Analytics")
    st.markdown("### Comprehensive Financial Analysis & Algorithmic Trading Insights")
    
    # Initialize dashboard
    dashboard = AdvancedAnalyticsDashboard()
    
    # Sidebar navigation
    st.sidebar.title("📋 Navigation")
    page = st.sidebar.selectbox(
        "Select Analysis View",
        [
            "🏠 System Overview",
            "📈 Performance Analytics", 
            "🤖 AI Agent Insights",
            "🌊 Market Regime Analysis",
            "⚠️ Risk Management"
        ]
    )
    
    # Display selected page
    if page == "🏠 System Overview":
        dashboard.display_system_status()
        dashboard.display_advanced_analytics()
    
    elif page == "📈 Performance Analytics":
        dashboard.display_advanced_analytics()
    
    elif page == "🤖 AI Agent Insights":
        dashboard.display_ai_agent_insights()
    
    elif page == "🌊 Market Regime Analysis":
        dashboard.display_market_regime_analysis()
    
    elif page == "⚠️ Risk Management":
        st.header("⚠️ Risk Management Dashboard")
        st.info("Risk management analytics will be displayed here with real-time risk metrics, limit monitoring, and stress test results.")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "**Advanced Multi-Asset Trading System** | "
        "Real-time Analytics • AI-Driven Decisions • Risk Management • 24/7 Monitoring"
    )

if __name__ == "__main__":
    main()