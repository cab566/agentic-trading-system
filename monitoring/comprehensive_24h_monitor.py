#!/usr/bin/env python3
"""
Comprehensive 24-Hour Trading System Monitor

This module provides real-time monitoring of the trading system including:
- Trade execution tracking
- Data feed validation
- Strategy performance assessment
- Error detection and alerting
- System health monitoring
- API connectivity status
"""

import asyncio
import logging
import json
import os
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from pathlib import Path
import aiohttp
import pandas as pd
from collections import defaultdict, deque

# Import system components
try:
    from ..core.alpaca_client import AlpacaClient
    from ..core.execution_engine import ExecutionEngine
    from ..core.data_manager import UnifiedDataManager
    from ..core.orchestrator import TradingOrchestrator
except ImportError:
    # Fallback for direct execution
    import sys
    sys.path.append('..')
    from core.alpaca_client import AlpacaClient
    from core.execution_engine import ExecutionEngine
    from core.data_manager import UnifiedDataManager
    from core.orchestrator import TradingOrchestrator

@dataclass
class MonitoringMetrics:
    """Container for monitoring metrics"""
    timestamp: datetime
    trades_executed: int
    trades_successful: int
    trades_failed: int
    data_feeds_active: int
    data_feeds_total: int
    strategies_running: int
    api_connections_healthy: int
    api_connections_total: int
    system_errors: int
    portfolio_value: float
    daily_pnl: float
    positions_count: int
    
class ComprehensiveMonitor:
    """24-hour comprehensive trading system monitor"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.logger = logging.getLogger(__name__)
        self.config_path = config_path
        self.monitoring_active = False
        self.start_time = None
        
        # Metrics storage
        self.metrics_history = deque(maxlen=1440)  # 24 hours of minute-by-minute data
        self.error_log = deque(maxlen=1000)
        self.trade_log = deque(maxlen=500)
        self.alert_log = deque(maxlen=100)
        
        # Component instances
        self.alpaca_client = None
        self.execution_engine = None
        self.data_manager = None
        self.orchestrator = None
        
        # Monitoring thresholds
        self.thresholds = {
            'max_consecutive_failures': 5,
            'min_data_feed_uptime': 0.95,
            'max_api_response_time': 5.0,
            'max_daily_loss_pct': 0.05,
            'min_trades_per_hour': 1,
            'max_position_concentration': 0.20
        }
        
        # Performance tracking
        self.performance_stats = {
            'total_trades': 0,
            'successful_trades': 0,
            'failed_trades': 0,
            'total_volume': 0.0,
            'total_pnl': 0.0,
            'max_drawdown': 0.0,
            'sharpe_ratio': 0.0,
            'win_rate': 0.0
        }
        
    async def initialize(self):
        """Initialize monitoring components"""
        try:
            self.logger.info("Initializing 24-hour monitoring system...")
            
            # Initialize Alpaca client based on TRADING_MODE environment variable
            trading_mode = os.getenv('TRADING_MODE', 'paper').lower()
            paper_trading = trading_mode == 'paper'
            self.alpaca_client = AlpacaClient(paper_trading=paper_trading)
            self.logger.info(f"Alpaca client initialized in {'PAPER' if paper_trading else 'LIVE'} trading mode")
            await self.alpaca_client.test_connection()
            
            # Initialize other components
            self.data_manager = UnifiedDataManager()
            self.execution_engine = ExecutionEngine()
            
            # Test all connections
            await self._test_all_connections()
            
            self.logger.info("24-hour monitoring system initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize monitoring system: {e}")
            return False
    
    async def start_monitoring(self):
        """Start 24-hour monitoring cycle"""
        if self.monitoring_active:
            self.logger.warning("Monitoring already active")
            return
            
        self.monitoring_active = True
        self.start_time = datetime.now()
        
        self.logger.info(f"Starting 24-hour monitoring at {self.start_time}")
        
        # Start monitoring tasks
        tasks = [
            asyncio.create_task(self._monitor_trading_activity()),
            asyncio.create_task(self._monitor_data_feeds()),
            asyncio.create_task(self._monitor_system_health()),
            asyncio.create_task(self._monitor_api_connectivity()),
            asyncio.create_task(self._monitor_portfolio_performance()),
            asyncio.create_task(self._generate_alerts()),
            asyncio.create_task(self._log_metrics())
        ]
        
        try:
            await asyncio.gather(*tasks)
        except Exception as e:
            self.logger.error(f"Monitoring error: {e}")
        finally:
            self.monitoring_active = False
    
    async def _monitor_trading_activity(self):
        """Monitor trading execution and performance"""
        while self.monitoring_active:
            try:
                # Get recent trades from Alpaca
                trades = await self.alpaca_client.get_orders(status='all', limit=50)
                
                # Analyze trade execution
                recent_trades = [t for t in trades if self._is_recent_trade(t)]
                
                for trade in recent_trades:
                    self.trade_log.append({
                        'timestamp': datetime.now(),
                        'symbol': trade.get('symbol'),
                        'side': trade.get('side'),
                        'qty': trade.get('qty'),
                        'status': trade.get('status'),
                        'filled_qty': trade.get('filled_qty', 0),
                        'avg_fill_price': trade.get('filled_avg_price')
                    })
                    
                    # Update performance stats
                    if trade.get('status') == 'filled':
                        self.performance_stats['successful_trades'] += 1
                    elif trade.get('status') in ['canceled', 'rejected']:
                        self.performance_stats['failed_trades'] += 1
                
                # Check for trading anomalies
                await self._check_trading_anomalies()
                
            except Exception as e:
                self.logger.error(f"Error monitoring trading activity: {e}")
                self.error_log.append({
                    'timestamp': datetime.now(),
                    'component': 'trading_monitor',
                    'error': str(e)
                })
            
            await asyncio.sleep(60)  # Check every minute
    
    async def _monitor_data_feeds(self):
        """Monitor data feed health and quality"""
        while self.monitoring_active:
            try:
                data_feed_status = {}
                
                # Test Alpaca data feed
                try:
                    account_info = await self.alpaca_client.get_account()
                    data_feed_status['alpaca'] = {
                        'status': 'healthy',
                        'last_update': datetime.now(),
                        'response_time': 0.5  # Placeholder
                    }
                except Exception as e:
                    data_feed_status['alpaca'] = {
                        'status': 'error',
                        'error': str(e),
                        'last_update': datetime.now()
                    }
                
                # Test other data sources if available
                # Add more data source checks here
                
                # Calculate data feed uptime
                healthy_feeds = sum(1 for feed in data_feed_status.values() if feed['status'] == 'healthy')
                total_feeds = len(data_feed_status)
                uptime_ratio = healthy_feeds / total_feeds if total_feeds > 0 else 0
                
                if uptime_ratio < self.thresholds['min_data_feed_uptime']:
                    await self._trigger_alert('data_feed_degraded', {
                        'uptime_ratio': uptime_ratio,
                        'failed_feeds': [name for name, status in data_feed_status.items() if status['status'] != 'healthy']
                    })
                
            except Exception as e:
                self.logger.error(f"Error monitoring data feeds: {e}")
                self.error_log.append({
                    'timestamp': datetime.now(),
                    'component': 'data_feed_monitor',
                    'error': str(e)
                })
            
            await asyncio.sleep(30)  # Check every 30 seconds
    
    async def _monitor_system_health(self):
        """Monitor overall system health"""
        while self.monitoring_active:
            try:
                # Check system resources
                import psutil
                
                cpu_usage = psutil.cpu_percent(interval=1)
                memory_usage = psutil.virtual_memory().percent
                disk_usage = psutil.disk_usage('/').percent
                
                # Check for resource constraints
                if cpu_usage > 90:
                    await self._trigger_alert('high_cpu_usage', {'cpu_usage': cpu_usage})
                
                if memory_usage > 90:
                    await self._trigger_alert('high_memory_usage', {'memory_usage': memory_usage})
                
                if disk_usage > 90:
                    await self._trigger_alert('high_disk_usage', {'disk_usage': disk_usage})
                
                # Log system metrics
                self.logger.debug(f"System health - CPU: {cpu_usage}%, Memory: {memory_usage}%, Disk: {disk_usage}%")
                
            except Exception as e:
                self.logger.error(f"Error monitoring system health: {e}")
                self.error_log.append({
                    'timestamp': datetime.now(),
                    'component': 'system_health_monitor',
                    'error': str(e)
                })
            
            await asyncio.sleep(120)  # Check every 2 minutes
    
    async def _monitor_api_connectivity(self):
        """Monitor API connectivity and response times"""
        while self.monitoring_active:
            try:
                api_status = {}
                
                # Test Alpaca API
                start_time = time.time()
                try:
                    await self.alpaca_client.test_connection()
                    response_time = time.time() - start_time
                    api_status['alpaca'] = {
                        'status': 'connected',
                        'response_time': response_time,
                        'last_check': datetime.now()
                    }
                    
                    if response_time > self.thresholds['max_api_response_time']:
                        await self._trigger_alert('slow_api_response', {
                            'api': 'alpaca',
                            'response_time': response_time
                        })
                        
                except Exception as e:
                    api_status['alpaca'] = {
                        'status': 'disconnected',
                        'error': str(e),
                        'last_check': datetime.now()
                    }
                    
                    await self._trigger_alert('api_disconnected', {
                        'api': 'alpaca',
                        'error': str(e)
                    })
                
                # Test other APIs as needed
                
            except Exception as e:
                self.logger.error(f"Error monitoring API connectivity: {e}")
                self.error_log.append({
                    'timestamp': datetime.now(),
                    'component': 'api_connectivity_monitor',
                    'error': str(e)
                })
            
            await asyncio.sleep(60)  # Check every minute
    
    async def _monitor_portfolio_performance(self):
        """Monitor portfolio performance and risk metrics"""
        while self.monitoring_active:
            try:
                # Get current portfolio state
                account = await self.alpaca_client.get_account()
                positions = await self.alpaca_client.get_positions()
                
                portfolio_value = float(account.get('portfolio_value', 0))
                equity = float(account.get('equity', 0))
                day_trade_buying_power = float(account.get('daytrading_buying_power', 0))
                
                # Calculate daily P&L
                daily_pnl = equity - float(account.get('last_equity', equity))
                daily_pnl_pct = (daily_pnl / equity) * 100 if equity > 0 else 0
                
                # Check for excessive losses
                if abs(daily_pnl_pct) > self.thresholds['max_daily_loss_pct'] * 100:
                    await self._trigger_alert('excessive_daily_loss', {
                        'daily_pnl': daily_pnl,
                        'daily_pnl_pct': daily_pnl_pct
                    })
                
                # Check position concentration
                if positions:
                    max_position_value = max(float(pos.get('market_value', 0)) for pos in positions)
                    concentration = max_position_value / portfolio_value if portfolio_value > 0 else 0
                    
                    if concentration > self.thresholds['max_position_concentration']:
                        await self._trigger_alert('high_position_concentration', {
                            'concentration': concentration,
                            'max_position_value': max_position_value
                        })
                
                # Update performance stats
                self.performance_stats['total_pnl'] = daily_pnl
                
                self.logger.info(f"Portfolio update - Value: ${portfolio_value:,.2f}, Daily P&L: ${daily_pnl:,.2f} ({daily_pnl_pct:.2f}%)")
                
            except Exception as e:
                self.logger.error(f"Error monitoring portfolio performance: {e}")
                self.error_log.append({
                    'timestamp': datetime.now(),
                    'component': 'portfolio_monitor',
                    'error': str(e)
                })
            
            await asyncio.sleep(300)  # Check every 5 minutes
    
    async def _generate_alerts(self):
        """Generate and manage alerts"""
        while self.monitoring_active:
            try:
                # Process pending alerts
                current_time = datetime.now()
                
                # Check for lack of trading activity
                if len(self.trade_log) > 0:
                    last_trade_time = max(trade['timestamp'] for trade in self.trade_log)
                    hours_since_last_trade = (current_time - last_trade_time).total_seconds() / 3600
                    
                    if hours_since_last_trade > 2:  # No trades for 2 hours
                        await self._trigger_alert('no_trading_activity', {
                            'hours_since_last_trade': hours_since_last_trade
                        })
                
                # Check error frequency
                recent_errors = [e for e in self.error_log if (current_time - e['timestamp']).total_seconds() < 3600]
                if len(recent_errors) > 10:  # More than 10 errors in the last hour
                    await self._trigger_alert('high_error_rate', {
                        'error_count': len(recent_errors),
                        'recent_errors': recent_errors[-5:]  # Last 5 errors
                    })
                
            except Exception as e:
                self.logger.error(f"Error generating alerts: {e}")
            
            await asyncio.sleep(300)  # Check every 5 minutes
    
    async def _log_metrics(self):
        """Log comprehensive metrics"""
        while self.monitoring_active:
            try:
                # Collect current metrics
                current_metrics = MonitoringMetrics(
                    timestamp=datetime.now(),
                    trades_executed=len(self.trade_log),
                    trades_successful=self.performance_stats['successful_trades'],
                    trades_failed=self.performance_stats['failed_trades'],
                    data_feeds_active=1,  # Placeholder
                    data_feeds_total=1,   # Placeholder
                    strategies_running=1, # Placeholder
                    api_connections_healthy=1, # Placeholder
                    api_connections_total=1,   # Placeholder
                    system_errors=len(self.error_log),
                    portfolio_value=0.0,  # Will be updated by portfolio monitor
                    daily_pnl=self.performance_stats['total_pnl'],
                    positions_count=0     # Placeholder
                )
                
                self.metrics_history.append(current_metrics)
                
                # Log summary every hour
                if len(self.metrics_history) % 60 == 0:
                    await self._log_hourly_summary()
                
            except Exception as e:
                self.logger.error(f"Error logging metrics: {e}")
            
            await asyncio.sleep(60)  # Log every minute
    
    async def _trigger_alert(self, alert_type: str, data: Dict[str, Any]):
        """Trigger an alert"""
        alert = {
            'timestamp': datetime.now(),
            'type': alert_type,
            'data': data,
            'severity': self._get_alert_severity(alert_type)
        }
        
        self.alert_log.append(alert)
        
        # Log alert
        severity_emoji = {'low': '🟡', 'medium': '🟠', 'high': '🔴'}
        emoji = severity_emoji.get(alert['severity'], '⚠️')
        
        self.logger.warning(f"{emoji} ALERT [{alert_type.upper()}]: {json.dumps(data, default=str)}")
        
        # Could add email/SMS notifications here
    
    def _get_alert_severity(self, alert_type: str) -> str:
        """Determine alert severity"""
        high_severity = ['api_disconnected', 'excessive_daily_loss', 'high_error_rate']
        medium_severity = ['data_feed_degraded', 'slow_api_response', 'high_position_concentration']
        
        if alert_type in high_severity:
            return 'high'
        elif alert_type in medium_severity:
            return 'medium'
        else:
            return 'low'
    
    async def _test_all_connections(self):
        """Test all system connections"""
        connections = {}
        
        # Test Alpaca
        try:
            await self.alpaca_client.test_connection()
            connections['alpaca'] = 'connected'
        except Exception as e:
            connections['alpaca'] = f'failed: {e}'
        
        self.logger.info(f"Connection test results: {connections}")
        return connections
    
    def _is_recent_trade(self, trade: Dict[str, Any]) -> bool:
        """Check if trade is recent (within last hour)"""
        try:
            trade_time = datetime.fromisoformat(trade.get('created_at', '').replace('Z', '+00:00'))
            return (datetime.now() - trade_time.replace(tzinfo=None)).total_seconds() < 3600
        except:
            return False
    
    async def _check_trading_anomalies(self):
        """Check for trading anomalies"""
        if len(self.trade_log) < 5:
            return
        
        recent_trades = list(self.trade_log)[-10:]  # Last 10 trades
        
        # Check for consecutive failures
        consecutive_failures = 0
        for trade in reversed(recent_trades):
            if trade['status'] in ['canceled', 'rejected']:
                consecutive_failures += 1
            else:
                break
        
        if consecutive_failures >= self.thresholds['max_consecutive_failures']:
            await self._trigger_alert('consecutive_trade_failures', {
                'consecutive_failures': consecutive_failures
            })
    
    async def _log_hourly_summary(self):
        """Log hourly summary"""
        if not self.metrics_history:
            return
        
        recent_metrics = list(self.metrics_history)[-60:]  # Last hour
        
        summary = {
            'timestamp': datetime.now(),
            'trades_executed': sum(m.trades_executed for m in recent_metrics),
            'success_rate': (sum(m.trades_successful for m in recent_metrics) / 
                           max(sum(m.trades_executed for m in recent_metrics), 1)) * 100,
            'errors': sum(m.system_errors for m in recent_metrics),
            'alerts': len([a for a in self.alert_log if (datetime.now() - a['timestamp']).total_seconds() < 3600])
        }
        
        self.logger.info(f"📊 HOURLY SUMMARY: {json.dumps(summary, default=str)}")
    
    async def get_status_report(self) -> Dict[str, Any]:
        """Generate comprehensive status report"""
        if not self.monitoring_active:
            return {'status': 'inactive'}
        
        runtime = datetime.now() - self.start_time if self.start_time else timedelta(0)
        
        return {
            'status': 'active',
            'runtime_hours': runtime.total_seconds() / 3600,
            'total_trades': len(self.trade_log),
            'successful_trades': self.performance_stats['successful_trades'],
            'failed_trades': self.performance_stats['failed_trades'],
            'total_errors': len(self.error_log),
            'total_alerts': len(self.alert_log),
            'recent_alerts': [a for a in self.alert_log if (datetime.now() - a['timestamp']).total_seconds() < 3600],
            'performance_stats': self.performance_stats,
            'last_metrics': asdict(self.metrics_history[-1]) if self.metrics_history else None
        }
    
    async def stop_monitoring(self):
        """Stop monitoring and generate final report"""
        self.monitoring_active = False
        
        final_report = await self.get_status_report()
        self.logger.info(f"🏁 FINAL MONITORING REPORT: {json.dumps(final_report, default=str, indent=2)}")
        
        return final_report

# CLI interface for standalone execution
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="24-Hour Trading System Monitor")
    parser.add_argument("--duration", type=int, default=24, help="Monitoring duration in hours")
    parser.add_argument("--config", type=str, help="Configuration file path")
    parser.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'monitoring_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
            logging.StreamHandler()
        ]
    )
    
    async def main():
        monitor = ComprehensiveMonitor(config_path=args.config)
        
        if await monitor.initialize():
            print(f"🚀 Starting {args.duration}-hour monitoring session...")
            
            # Run monitoring for specified duration
            monitoring_task = asyncio.create_task(monitor.start_monitoring())
            
            try:
                await asyncio.wait_for(monitoring_task, timeout=args.duration * 3600)
            except asyncio.TimeoutError:
                print(f"⏰ {args.duration}-hour monitoring period completed")
            
            final_report = await monitor.stop_monitoring()
            print("\n📋 Final Report:")
            print(json.dumps(final_report, default=str, indent=2))
        else:
            print("❌ Failed to initialize monitoring system")
    
    asyncio.run(main())