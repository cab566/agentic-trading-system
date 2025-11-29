#!/usr/bin/env python3
"""
Unified Trading System Dashboard

A comprehensive web-based dashboard that consolidates all monitoring information
into a single, multi-tab interface for 24-hour trading system oversight.
"""

import asyncio
import json
import logging
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from pathlib import Path
import pandas as pd
from dataclasses import asdict

# Web framework imports
try:
    from flask import Flask, render_template, jsonify, request
    from flask_socketio import SocketIO, emit
except ImportError:
    print("Installing required packages...")
    import subprocess
    subprocess.check_call(["pip", "install", "flask", "flask-socketio", "plotly", "dash"])
    from flask import Flask, render_template, jsonify, request
    from flask_socketio import SocketIO, emit

import plotly.graph_objs as go
import plotly.utils
from plotly.subplots import make_subplots

# Import monitoring components
try:
    from .comprehensive_24h_monitor import ComprehensiveMonitor
    from ..core.alpaca_client import AlpacaClient
except ImportError:
    import sys
    sys.path.append('..')
    from comprehensive_24h_monitor import ComprehensiveMonitor
    from core.alpaca_client import AlpacaClient

class UnifiedDashboard:
    """Unified web dashboard for trading system monitoring"""
    
    def __init__(self, host='localhost', port=5000):
        self.host = host
        self.port = port
        self.app = Flask(__name__)
        self.app.config['SECRET_KEY'] = 'trading_dashboard_secret_key'
        self.socketio = SocketIO(self.app, cors_allowed_origins="*")
        
        self.logger = logging.getLogger(__name__)
        self.monitor = None
        self.alpaca_client = None
        
        # Dashboard data cache
        self.dashboard_data = {
            'system_status': {},
            'trading_metrics': {},
            'portfolio_data': {},
            'alerts': [],
            'performance_charts': {},
            'error_log': [],
            'trade_history': []
        }
        
        self._setup_routes()
        self._setup_socketio_events()
    
    def _setup_routes(self):
        """Setup Flask routes"""
        
        @self.app.route('/')
        def index():
            return render_template('dashboard.html')
        
        @self.app.route('/api/status')
        def get_status():
            return jsonify(self.dashboard_data['system_status'])
        
        @self.app.route('/api/metrics')
        def get_metrics():
            return jsonify(self.dashboard_data['trading_metrics'])
        
        @self.app.route('/api/portfolio')
        def get_portfolio():
            return jsonify(self.dashboard_data['portfolio_data'])
        
        @self.app.route('/api/alerts')
        def get_alerts():
            return jsonify(self.dashboard_data['alerts'])
        
        @self.app.route('/api/trades')
        def get_trades():
            return jsonify(self.dashboard_data['trade_history'])
        
        @self.app.route('/api/charts')
        def get_charts():
            return jsonify(self.dashboard_data['performance_charts'])
        
        @self.app.route('/api/errors')
        def get_errors():
            return jsonify(self.dashboard_data['error_log'])
    
    def _setup_socketio_events(self):
        """Setup SocketIO events for real-time updates"""
        
        @self.socketio.on('connect')
        def handle_connect():
            self.logger.info('Client connected to dashboard')
            emit('status', {'msg': 'Connected to trading dashboard'})
        
        @self.socketio.on('disconnect')
        def handle_disconnect():
            self.logger.info('Client disconnected from dashboard')
        
        @self.socketio.on('request_update')
        def handle_update_request():
            emit('dashboard_update', self.dashboard_data)
    
    async def initialize(self):
        """Initialize dashboard components"""
        try:
            self.logger.info("Initializing unified dashboard...")
            
            # Initialize monitoring system
            self.monitor = ComprehensiveMonitor()
            await self.monitor.initialize()
            
            # Initialize Alpaca client based on TRADING_MODE environment variable
            trading_mode = os.getenv('TRADING_MODE', 'paper').lower()
            paper_trading = trading_mode == 'paper'
            self.alpaca_client = AlpacaClient(paper_trading=paper_trading)
            print(f"Dashboard: Alpaca client initialized in {'PAPER' if paper_trading else 'LIVE'} trading mode")
            await self.alpaca_client.test_connection()
            
            # Start background data collection
            asyncio.create_task(self._update_dashboard_data())
            
            self.logger.info("Unified dashboard initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize dashboard: {e}")
            return False
    
    async def _update_dashboard_data(self):
        """Continuously update dashboard data"""
        while True:
            try:
                # Update system status
                await self._update_system_status()
                
                # Update trading metrics
                await self._update_trading_metrics()
                
                # Update portfolio data
                await self._update_portfolio_data()
                
                # Update alerts
                await self._update_alerts()
                
                # Update performance charts
                await self._update_performance_charts()
                
                # Update error log
                await self._update_error_log()
                
                # Update trade history
                await self._update_trade_history()
                
                # Emit real-time update to connected clients
                self.socketio.emit('dashboard_update', self.dashboard_data)
                
            except Exception as e:
                self.logger.error(f"Error updating dashboard data: {e}")
            
            await asyncio.sleep(30)  # Update every 30 seconds
    
    async def _update_system_status(self):
        """Update system status information"""
        try:
            if self.monitor:
                status_report = await self.monitor.get_status_report()
                
                self.dashboard_data['system_status'] = {
                    'monitoring_active': status_report.get('status') == 'active',
                    'runtime_hours': status_report.get('runtime_hours', 0),
                    'last_update': datetime.now().isoformat(),
                    'api_connections': {
                        'alpaca': 'connected',  # Will be updated by actual tests
                    },
                    'data_feeds': {
                        'active': 1,
                        'total': 1,
                        'uptime_pct': 100.0
                    },
                    'system_health': {
                        'cpu_usage': 0,  # Will be updated by system monitor
                        'memory_usage': 0,
                        'disk_usage': 0
                    }
                }
        except Exception as e:
            self.logger.error(f"Error updating system status: {e}")
    
    async def _update_trading_metrics(self):
        """Update trading performance metrics"""
        try:
            if self.monitor:
                status_report = await self.monitor.get_status_report()
                
                self.dashboard_data['trading_metrics'] = {
                    'total_trades': status_report.get('total_trades', 0),
                    'successful_trades': status_report.get('successful_trades', 0),
                    'failed_trades': status_report.get('failed_trades', 0),
                    'success_rate': (status_report.get('successful_trades', 0) / 
                                   max(status_report.get('total_trades', 1), 1)) * 100,
                    'total_volume': status_report.get('performance_stats', {}).get('total_volume', 0),
                    'total_pnl': status_report.get('performance_stats', {}).get('total_pnl', 0),
                    'win_rate': status_report.get('performance_stats', {}).get('win_rate', 0),
                    'sharpe_ratio': status_report.get('performance_stats', {}).get('sharpe_ratio', 0),
                    'max_drawdown': status_report.get('performance_stats', {}).get('max_drawdown', 0)
                }
        except Exception as e:
            self.logger.error(f"Error updating trading metrics: {e}")
    
    async def _update_portfolio_data(self):
        """Update portfolio information"""
        try:
            if self.alpaca_client:
                # Get account information
                account = await self.alpaca_client.get_account()
                positions = await self.alpaca_client.get_positions()
                
                portfolio_value = float(account.get('portfolio_value', 0))
                equity = float(account.get('equity', 0))
                buying_power = float(account.get('buying_power', 0))
                
                # Calculate daily P&L
                daily_pnl = equity - float(account.get('last_equity', equity))
                daily_pnl_pct = (daily_pnl / equity) * 100 if equity > 0 else 0
                
                # Process positions
                position_data = []
                for pos in positions:
                    position_data.append({
                        'symbol': pos.get('symbol'),
                        'qty': float(pos.get('qty', 0)),
                        'market_value': float(pos.get('market_value', 0)),
                        'unrealized_pl': float(pos.get('unrealized_pl', 0)),
                        'unrealized_plpc': float(pos.get('unrealized_plpc', 0)) * 100,
                        'current_price': float(pos.get('current_price', 0))
                    })
                
                self.dashboard_data['portfolio_data'] = {
                    'portfolio_value': portfolio_value,
                    'equity': equity,
                    'buying_power': buying_power,
                    'daily_pnl': daily_pnl,
                    'daily_pnl_pct': daily_pnl_pct,
                    'positions': position_data,
                    'positions_count': len(position_data),
                    'cash': float(account.get('cash', 0)),
                    'last_update': datetime.now().isoformat()
                }
        except Exception as e:
            self.logger.error(f"Error updating portfolio data: {e}")
    
    async def _update_alerts(self):
        """Update alerts and notifications"""
        try:
            if self.monitor and hasattr(self.monitor, 'alert_log'):
                # Get recent alerts (last 24 hours)
                recent_alerts = [
                    {
                        'timestamp': alert['timestamp'].isoformat(),
                        'type': alert['type'],
                        'severity': alert['severity'],
                        'data': alert['data']
                    }
                    for alert in self.monitor.alert_log
                    if (datetime.now() - alert['timestamp']).total_seconds() < 86400
                ]
                
                self.dashboard_data['alerts'] = recent_alerts
        except Exception as e:
            self.logger.error(f"Error updating alerts: {e}")
    
    async def _update_performance_charts(self):
        """Update performance charts data"""
        try:
            if self.monitor and hasattr(self.monitor, 'metrics_history'):
                # Prepare data for charts
                metrics_data = [asdict(m) for m in self.monitor.metrics_history]
                
                if metrics_data:
                    df = pd.DataFrame(metrics_data)
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    
                    # Create portfolio value chart
                    portfolio_chart = {
                        'x': df['timestamp'].dt.strftime('%H:%M').tolist(),
                        'y': df['portfolio_value'].tolist(),
                        'type': 'scatter',
                        'mode': 'lines',
                        'name': 'Portfolio Value'
                    }
                    
                    # Create P&L chart
                    pnl_chart = {
                        'x': df['timestamp'].dt.strftime('%H:%M').tolist(),
                        'y': df['daily_pnl'].tolist(),
                        'type': 'scatter',
                        'mode': 'lines',
                        'name': 'Daily P&L'
                    }
                    
                    # Create trades chart
                    trades_chart = {
                        'x': df['timestamp'].dt.strftime('%H:%M').tolist(),
                        'y': df['trades_executed'].tolist(),
                        'type': 'bar',
                        'name': 'Trades Executed'
                    }
                    
                    self.dashboard_data['performance_charts'] = {
                        'portfolio_value': portfolio_chart,
                        'daily_pnl': pnl_chart,
                        'trades_executed': trades_chart
                    }
        except Exception as e:
            self.logger.error(f"Error updating performance charts: {e}")
    
    async def _update_error_log(self):
        """Update error log"""
        try:
            if self.monitor and hasattr(self.monitor, 'error_log'):
                # Get recent errors (last 24 hours)
                recent_errors = [
                    {
                        'timestamp': error['timestamp'].isoformat(),
                        'component': error['component'],
                        'error': error['error']
                    }
                    for error in self.monitor.error_log
                    if (datetime.now() - error['timestamp']).total_seconds() < 86400
                ]
                
                self.dashboard_data['error_log'] = recent_errors
        except Exception as e:
            self.logger.error(f"Error updating error log: {e}")
    
    async def _update_trade_history(self):
        """Update trade history"""
        try:
            if self.monitor and hasattr(self.monitor, 'trade_log'):
                # Get recent trades
                recent_trades = [
                    {
                        'timestamp': trade['timestamp'].isoformat(),
                        'symbol': trade['symbol'],
                        'side': trade['side'],
                        'qty': trade['qty'],
                        'status': trade['status'],
                        'filled_qty': trade['filled_qty'],
                        'avg_fill_price': trade['avg_fill_price']
                    }
                    for trade in self.monitor.trade_log
                ]
                
                self.dashboard_data['trade_history'] = recent_trades
        except Exception as e:
            self.logger.error(f"Error updating trade history: {e}")
    
    def create_dashboard_template(self):
        """Create HTML template for dashboard"""
        template_dir = Path(self.app.template_folder or 'templates')
        template_dir.mkdir(exist_ok=True)
        
        dashboard_html = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Trading System Dashboard</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.0.1/socket.io.js"></script>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 0;
            background-color: #1a1a1a;
            color: #ffffff;
        }
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            text-align: center;
            box-shadow: 0 2px 10px rgba(0,0,0,0.3);
        }
        .header h1 {
            margin: 0;
            font-size: 2.5em;
            font-weight: 300;
        }
        .status-bar {
            background-color: #2d2d2d;
            padding: 10px 20px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid #444;
        }
        .status-item {
            display: flex;
            align-items: center;
            gap: 10px;
        }
        .status-indicator {
            width: 12px;
            height: 12px;
            border-radius: 50%;
            background-color: #4CAF50;
        }
        .status-indicator.warning { background-color: #FF9800; }
        .status-indicator.error { background-color: #F44336; }
        .container {
            display: flex;
            height: calc(100vh - 140px);
        }
        .sidebar {
            width: 250px;
            background-color: #2d2d2d;
            padding: 20px;
            border-right: 1px solid #444;
        }
        .main-content {
            flex: 1;
            padding: 20px;
            overflow-y: auto;
        }
        .tab-container {
            display: flex;
            border-bottom: 1px solid #444;
            margin-bottom: 20px;
        }
        .tab {
            padding: 10px 20px;
            cursor: pointer;
            border-bottom: 2px solid transparent;
            transition: all 0.3s ease;
        }
        .tab:hover {
            background-color: #3d3d3d;
        }
        .tab.active {
            border-bottom-color: #667eea;
            background-color: #3d3d3d;
        }
        .tab-content {
            display: none;
        }
        .tab-content.active {
            display: block;
        }
        .metric-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        .metric-card {
            background-color: #2d2d2d;
            padding: 20px;
            border-radius: 8px;
            border: 1px solid #444;
            box-shadow: 0 2px 10px rgba(0,0,0,0.2);
        }
        .metric-card h3 {
            margin: 0 0 10px 0;
            color: #667eea;
            font-size: 1.1em;
        }
        .metric-value {
            font-size: 2em;
            font-weight: bold;
            margin: 10px 0;
        }
        .metric-change {
            font-size: 0.9em;
            opacity: 0.8;
        }
        .positive { color: #4CAF50; }
        .negative { color: #F44336; }
        .chart-container {
            background-color: #2d2d2d;
            padding: 20px;
            border-radius: 8px;
            border: 1px solid #444;
            margin-bottom: 20px;
        }
        .table-container {
            background-color: #2d2d2d;
            border-radius: 8px;
            border: 1px solid #444;
            overflow: hidden;
        }
        table {
            width: 100%;
            border-collapse: collapse;
        }
        th, td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #444;
        }
        th {
            background-color: #3d3d3d;
            font-weight: 600;
        }
        .alert {
            padding: 10px 15px;
            margin: 10px 0;
            border-radius: 4px;
            border-left: 4px solid;
        }
        .alert.high { border-left-color: #F44336; background-color: rgba(244, 67, 54, 0.1); }
        .alert.medium { border-left-color: #FF9800; background-color: rgba(255, 152, 0, 0.1); }
        .alert.low { border-left-color: #2196F3; background-color: rgba(33, 150, 243, 0.1); }
        .log-entry {
            padding: 8px 12px;
            margin: 5px 0;
            background-color: #3d3d3d;
            border-radius: 4px;
            font-family: monospace;
            font-size: 0.9em;
        }
        .timestamp {
            color: #888;
            margin-right: 10px;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🚀 Trading System Dashboard</h1>
    </div>
    
    <div class="status-bar">
        <div class="status-item">
            <div class="status-indicator" id="system-status"></div>
            <span id="system-status-text">System Status</span>
        </div>
        <div class="status-item">
            <span id="runtime">Runtime: 0h 0m</span>
        </div>
        <div class="status-item">
            <span id="last-update">Last Update: --</span>
        </div>
    </div>
    
    <div class="container">
        <div class="sidebar">
            <h3>Quick Stats</h3>
            <div id="quick-stats">
                <div class="metric-card">
                    <h3>Total Trades</h3>
                    <div class="metric-value" id="total-trades">0</div>
                </div>
                <div class="metric-card">
                    <h3>Success Rate</h3>
                    <div class="metric-value" id="success-rate">0%</div>
                </div>
                <div class="metric-card">
                    <h3>Daily P&L</h3>
                    <div class="metric-value" id="daily-pnl">$0.00</div>
                </div>
            </div>
        </div>
        
        <div class="main-content">
            <div class="tab-container">
                <div class="tab active" onclick="showTab('overview')">Overview</div>
                <div class="tab" onclick="showTab('portfolio')">Portfolio</div>
                <div class="tab" onclick="showTab('trades')">Trades</div>
                <div class="tab" onclick="showTab('alerts')">Alerts</div>
                <div class="tab" onclick="showTab('logs')">Logs</div>
            </div>
            
            <div id="overview" class="tab-content active">
                <div class="metric-grid">
                    <div class="metric-card">
                        <h3>Portfolio Value</h3>
                        <div class="metric-value" id="portfolio-value">$0.00</div>
                        <div class="metric-change" id="portfolio-change">+$0.00 (0.00%)</div>
                    </div>
                    <div class="metric-card">
                        <h3>Active Positions</h3>
                        <div class="metric-value" id="positions-count">0</div>
                    </div>
                    <div class="metric-card">
                        <h3>Buying Power</h3>
                        <div class="metric-value" id="buying-power">$0.00</div>
                    </div>
                    <div class="metric-card">
                        <h3>System Uptime</h3>
                        <div class="metric-value" id="uptime">0h 0m</div>
                    </div>
                </div>
                
                <div class="chart-container">
                    <h3>Portfolio Performance</h3>
                    <div id="portfolio-chart" style="height: 400px;"></div>
                </div>
                
                <div class="chart-container">
                    <h3>Trading Activity</h3>
                    <div id="trades-chart" style="height: 300px;"></div>
                </div>
            </div>
            
            <div id="portfolio" class="tab-content">
                <div class="table-container">
                    <table>
                        <thead>
                            <tr>
                                <th>Symbol</th>
                                <th>Quantity</th>
                                <th>Market Value</th>
                                <th>Unrealized P&L</th>
                                <th>Unrealized P&L %</th>
                                <th>Current Price</th>
                            </tr>
                        </thead>
                        <tbody id="positions-table">
                            <tr><td colspan="6">No positions</td></tr>
                        </tbody>
                    </table>
                </div>
            </div>
            
            <div id="trades" class="tab-content">
                <div class="table-container">
                    <table>
                        <thead>
                            <tr>
                                <th>Time</th>
                                <th>Symbol</th>
                                <th>Side</th>
                                <th>Quantity</th>
                                <th>Status</th>
                                <th>Fill Price</th>
                            </tr>
                        </thead>
                        <tbody id="trades-table">
                            <tr><td colspan="6">No trades</td></tr>
                        </tbody>
                    </table>
                </div>
            </div>
            
            <div id="alerts" class="tab-content">
                <div id="alerts-container">
                    <p>No alerts</p>
                </div>
            </div>
            
            <div id="logs" class="tab-content">
                <div id="logs-container">
                    <p>No logs</p>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        const socket = io();
        
        socket.on('connect', function() {
            console.log('Connected to dashboard');
        });
        
        socket.on('dashboard_update', function(data) {
            updateDashboard(data);
        });
        
        function showTab(tabName) {
            // Hide all tab contents
            document.querySelectorAll('.tab-content').forEach(content => {
                content.classList.remove('active');
            });
            
            // Remove active class from all tabs
            document.querySelectorAll('.tab').forEach(tab => {
                tab.classList.remove('active');
            });
            
            // Show selected tab content
            document.getElementById(tabName).classList.add('active');
            
            // Add active class to clicked tab
            event.target.classList.add('active');
        }
        
        function updateDashboard(data) {
            // Update system status
            const systemStatus = data.system_status || {};
            const statusIndicator = document.getElementById('system-status');
            const statusText = document.getElementById('system-status-text');
            
            if (systemStatus.monitoring_active) {
                statusIndicator.className = 'status-indicator';
                statusText.textContent = 'System Active';
            } else {
                statusIndicator.className = 'status-indicator error';
                statusText.textContent = 'System Inactive';
            }
            
            // Update runtime
            const runtime = systemStatus.runtime_hours || 0;
            const hours = Math.floor(runtime);
            const minutes = Math.floor((runtime - hours) * 60);
            document.getElementById('runtime').textContent = `Runtime: ${hours}h ${minutes}m`;
            
            // Update last update time
            document.getElementById('last-update').textContent = `Last Update: ${new Date().toLocaleTimeString()}`;
            
            // Update trading metrics
            const metrics = data.trading_metrics || {};
            document.getElementById('total-trades').textContent = metrics.total_trades || 0;
            document.getElementById('success-rate').textContent = `${(metrics.success_rate || 0).toFixed(1)}%`;
            
            // Update portfolio data
            const portfolio = data.portfolio_data || {};
            document.getElementById('portfolio-value').textContent = `$${(portfolio.portfolio_value || 0).toLocaleString('en-US', {minimumFractionDigits: 2})}`;;
            document.getElementById('daily-pnl').textContent = `$${(portfolio.daily_pnl || 0).toFixed(2)}`;
            document.getElementById('positions-count').textContent = portfolio.positions_count || 0;
            document.getElementById('buying-power').textContent = `$${(portfolio.buying_power || 0).toLocaleString('en-US', {minimumFractionDigits: 2})}`;
            document.getElementById('uptime').textContent = `${hours}h ${minutes}m`;
            
            // Update portfolio change
            const dailyPnl = portfolio.daily_pnl || 0;
            const dailyPnlPct = portfolio.daily_pnl_pct || 0;
            const changeElement = document.getElementById('portfolio-change');
            const changeText = `${dailyPnl >= 0 ? '+' : ''}$${dailyPnl.toFixed(2)} (${dailyPnlPct.toFixed(2)}%)`;
            changeElement.textContent = changeText;
            changeElement.className = `metric-change ${dailyPnl >= 0 ? 'positive' : 'negative'}`;
            
            // Update charts
            updateCharts(data.performance_charts || {});
            
            // Update positions table
            updatePositionsTable(portfolio.positions || []);
            
            // Update trades table
            updateTradesTable(data.trade_history || []);
            
            // Update alerts
            updateAlerts(data.alerts || []);
            
            // Update logs
            updateLogs(data.error_log || []);
        }
        
        function updateCharts(charts) {
            // Update portfolio chart
            if (charts.portfolio_value) {
                const layout = {
                    title: '',
                    paper_bgcolor: 'rgba(0,0,0,0)',
                    plot_bgcolor: 'rgba(0,0,0,0)',
                    font: { color: '#ffffff' },
                    xaxis: { gridcolor: '#444' },
                    yaxis: { gridcolor: '#444' }
                };
                Plotly.newPlot('portfolio-chart', [charts.portfolio_value], layout);
            }
            
            // Update trades chart
            if (charts.trades_executed) {
                const layout = {
                    title: '',
                    paper_bgcolor: 'rgba(0,0,0,0)',
                    plot_bgcolor: 'rgba(0,0,0,0)',
                    font: { color: '#ffffff' },
                    xaxis: { gridcolor: '#444' },
                    yaxis: { gridcolor: '#444' }
                };
                Plotly.newPlot('trades-chart', [charts.trades_executed], layout);
            }
        }
        
        function updatePositionsTable(positions) {
            const tbody = document.getElementById('positions-table');
            if (positions.length === 0) {
                tbody.innerHTML = '<tr><td colspan="6">No positions</td></tr>';
                return;
            }
            
            tbody.innerHTML = positions.map(pos => `
                <tr>
                    <td>${pos.symbol}</td>
                    <td>${pos.qty}</td>
                    <td>$${pos.market_value.toFixed(2)}</td>
                    <td class="${pos.unrealized_pl >= 0 ? 'positive' : 'negative'}">$${pos.unrealized_pl.toFixed(2)}</td>
                    <td class="${pos.unrealized_plpc >= 0 ? 'positive' : 'negative'}">${pos.unrealized_plpc.toFixed(2)}%</td>
                    <td>$${pos.current_price.toFixed(2)}</td>
                </tr>
            `).join('');
        }
        
        function updateTradesTable(trades) {
            const tbody = document.getElementById('trades-table');
            if (trades.length === 0) {
                tbody.innerHTML = '<tr><td colspan="6">No trades</td></tr>';
                return;
            }
            
            tbody.innerHTML = trades.slice(-20).reverse().map(trade => `
                <tr>
                    <td>${new Date(trade.timestamp).toLocaleTimeString()}</td>
                    <td>${trade.symbol}</td>
                    <td>${trade.side}</td>
                    <td>${trade.qty}</td>
                    <td>${trade.status}</td>
                    <td>${trade.avg_fill_price ? '$' + parseFloat(trade.avg_fill_price).toFixed(2) : '--'}</td>
                </tr>
            `).join('');
        }
        
        function updateAlerts(alerts) {
            const container = document.getElementById('alerts-container');
            if (alerts.length === 0) {
                container.innerHTML = '<p>No alerts</p>';
                return;
            }
            
            container.innerHTML = alerts.slice(-10).reverse().map(alert => `
                <div class="alert ${alert.severity}">
                    <strong>${alert.type.replace('_', ' ').toUpperCase()}</strong>
                    <span class="timestamp">${new Date(alert.timestamp).toLocaleString()}</span>
                    <div>${JSON.stringify(alert.data)}</div>
                </div>
            `).join('');
        }
        
        function updateLogs(logs) {
            const container = document.getElementById('logs-container');
            if (logs.length === 0) {
                container.innerHTML = '<p>No logs</p>';
                return;
            }
            
            container.innerHTML = logs.slice(-20).reverse().map(log => `
                <div class="log-entry">
                    <span class="timestamp">${new Date(log.timestamp).toLocaleTimeString()}</span>
                    <strong>[${log.component}]</strong> ${log.error}
                </div>
            `).join('');
        }
        
        // Request initial update
        socket.emit('request_update');
        
        // Request updates every 30 seconds
        setInterval(() => {
            socket.emit('request_update');
        }, 30000);
    </script>
</body>
</html>
        '''
        
        with open(template_dir / 'dashboard.html', 'w') as f:
            f.write(dashboard_html)
    
    def run(self, debug=False):
        """Run the dashboard server"""
        self.create_dashboard_template()
        self.logger.info(f"Starting dashboard server on http://{self.host}:{self.port}")
        self.socketio.run(self.app, host=self.host, port=self.port, debug=debug)

# CLI interface
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Unified Trading Dashboard")
    parser.add_argument("--host", type=str, default="localhost", help="Dashboard host")
    parser.add_argument("--port", type=int, default=5000, help="Dashboard port")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    async def main():
        dashboard = UnifiedDashboard(host=args.host, port=args.port)
        
        if await dashboard.initialize():
            print(f"🌐 Dashboard available at http://{args.host}:{args.port}")
            dashboard.run(debug=args.debug)
        else:
            print("❌ Failed to initialize dashboard")
    
    asyncio.run(main())