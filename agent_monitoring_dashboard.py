#!/usr/bin/env python3
"""
Agent Monitoring Dashboard - Real-time Agent Performance Tracking

Comprehensive monitoring system for tracking agent effectiveness,
connectivity, and performance in the trading system.
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import json
from pathlib import Path
import psutil
import threading
from dataclasses import dataclass, field
from collections import defaultdict, deque
import sqlite3

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.live import Live
    from rich.layout import Layout
    from rich.text import Text
    from rich.progress import Progress, BarColumn, TextColumn
except ImportError:
    print("Rich library not available. Install with: pip install rich")
    Console = None

@dataclass
class AgentMetrics:
    """Agent performance metrics"""
    agent_id: str
    agent_type: str
    status: str = "unknown"
    last_activity: Optional[datetime] = None
    tasks_completed: int = 0
    tasks_failed: int = 0
    response_time_avg: float = 0.0
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    error_count: int = 0
    uptime_seconds: float = 0.0
    
    @property
    def success_rate(self) -> float:
        total = self.tasks_completed + self.tasks_failed
        return (self.tasks_completed / total * 100) if total > 0 else 0.0
    
    @property
    def is_healthy(self) -> bool:
        if not self.last_activity:
            return False
        time_since_activity = (datetime.now() - self.last_activity).total_seconds()
        return time_since_activity < 300  # 5 minutes

@dataclass
class SystemConnectivity:
    """System connectivity status"""
    trading_system_connected: bool = False
    market_data_connected: bool = False
    execution_engine_connected: bool = False
    risk_manager_connected: bool = False
    orchestrator_connected: bool = False
    database_connected: bool = False
    
    @property
    def overall_health(self) -> float:
        connections = [
            self.trading_system_connected,
            self.market_data_connected,
            self.execution_engine_connected,
            self.risk_manager_connected,
            self.orchestrator_connected,
            self.database_connected
        ]
        return sum(connections) / len(connections) * 100

class AgentMonitoringDashboard:
    """Real-time agent monitoring dashboard"""
    
    def __init__(self):
        self.console = Console() if Console else None
        self.logger = self._setup_logging()
        self.agents: Dict[str, AgentMetrics] = {}
        self.connectivity = SystemConnectivity()
        self.system_metrics = {
            'cpu_usage': 0.0,
            'memory_usage': 0.0,
            'disk_usage': 0.0,
            'network_io': 0.0,
            'active_processes': 0
        }
        self.alerts = deque(maxlen=100)
        self.running = False
        self.start_time = datetime.now()
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration"""
        logger = logging.getLogger("agent_monitor")
        logger.setLevel(logging.INFO)
        
        # File handler
        log_file = Path("logs/agent_monitoring.log")
        log_file.parent.mkdir(exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        return logger
    
    async def start_monitoring(self):
        """Start the monitoring dashboard"""
        self.running = True
        self.logger.info("Starting Agent Monitoring Dashboard")
        
        # Start background tasks
        tasks = [
            asyncio.create_task(self._monitor_system_health()),
            asyncio.create_task(self._monitor_connectivity()),
            asyncio.create_task(self._monitor_agents()),
            asyncio.create_task(self._update_system_metrics())
        ]
        
        if self.console:
            tasks.append(asyncio.create_task(self._display_dashboard()))
        
        try:
            await asyncio.gather(*tasks)
        except KeyboardInterrupt:
            self.logger.info("Monitoring stopped by user")
        finally:
            self.running = False
    
    async def _monitor_system_health(self):
        """Monitor overall system health"""
        while self.running:
            try:
                # Check system processes
                processes = []
                for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                    try:
                        cmdline = ' '.join(proc.info['cmdline'] or [])
                        if any(keyword in cmdline.lower() for keyword in 
                               ['trading', 'orchestrator', 'agent', 'system_orchestrator']):
                            processes.append({
                                'pid': proc.info['pid'],
                                'name': proc.info['name'],
                                'cmdline': cmdline
                            })
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        continue
                
                self.system_metrics['active_processes'] = len(processes)
                
                # Log system health
                if len(processes) > 0:
                    self.logger.info(f"Found {len(processes)} trading-related processes")
                else:
                    self.logger.warning("No trading-related processes found")
                    self._add_alert("warning", "No trading processes detected")
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                self.logger.error(f"Error monitoring system health: {e}")
                await asyncio.sleep(10)
    
    async def _monitor_connectivity(self):
        """Monitor system connectivity"""
        while self.running:
            try:
                # Check for running processes
                trading_process_found = False
                orchestrator_process_found = False
                dashboard_process_found = False
                
                for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                    try:
                        if proc.info['cmdline']:
                            cmdline = ' '.join(proc.info['cmdline'])
                            if 'continuous_trading_24h.py' in cmdline:
                                trading_process_found = True
                            elif 'system_orchestrator.py' in cmdline:
                                orchestrator_process_found = True
                            elif 'agent_monitoring_dashboard.py' in cmdline:
                                dashboard_process_found = True
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        continue
                
                self.connectivity.trading_system_connected = trading_process_found
                self.connectivity.orchestrator_connected = orchestrator_process_found
                
                # Check database connectivity by verifying database files exist
                db_files = [
                    Path("alerts.db"),
                    Path("data/trading_data.db"),
                    Path("../trading_data.duckdb"),
                    Path("../trading_system.db")
                ]
                self.connectivity.database_connected = any(db.exists() for db in db_files)
                
                # Check market data connectivity by verifying config files and recent activity
                config_path = Path("config/data_sources.yaml")
                self.connectivity.market_data_connected = config_path.exists() and trading_process_found
                
                # Check execution engine and risk manager (connected if trading system is active)
                self.connectivity.execution_engine_connected = trading_process_found and orchestrator_process_found
                self.connectivity.risk_manager_connected = trading_process_found and orchestrator_process_found
                
                # Log connectivity status
                health = self.connectivity.overall_health
                if health < 80:
                    self._add_alert("warning", f"System connectivity at {health:.1f}%")
                else:
                    # Log successful connectivity for debugging
                    self.logger.info(f"System connectivity healthy at {health:.1f}%")
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                self.logger.error(f"Error monitoring connectivity: {e}")
                await asyncio.sleep(10)
    
    async def _monitor_agents(self):
        """Monitor individual agents"""
        while self.running:
            try:
                # Simulate agent discovery and monitoring
                # In a real implementation, this would connect to the agent orchestrator
                
                # Fetch real agent data from API
                try:
                    import requests
                    response = requests.get("http://localhost:8000/api/v1/agents/status")
                    if response.status_code == 200:
                        agents_data = response.json()
                    else:
                        agents_data = []
                except Exception as e:
                    self.logger.error(f"Failed to fetch agent data: {e}")
                    agents_data = []
                
                for agent_data in agents_data:
                    agent_id = agent_data.get("id", f"agent_{len(self.agents)}")
                    if agent_id not in self.agents:
                        self.agents[agent_id] = AgentMetrics(
                            agent_id=agent_id,
                            agent_type=agent_data.get("type", "unknown")
                        )
                    
                    # Update agent metrics with real data
                    agent = self.agents[agent_id]
                    agent.status = agent_data.get("status", "unknown")
                    agent.last_activity = datetime.now()
                    agent.uptime_seconds = (datetime.now() - self.start_time).total_seconds()
                
                await asyncio.sleep(45)  # Check every 45 seconds
                
            except Exception as e:
                self.logger.error(f"Error monitoring agents: {e}")
                await asyncio.sleep(10)
    
    async def _update_system_metrics(self):
        """Update system performance metrics"""
        while self.running:
            try:
                # Get system metrics
                self.system_metrics['cpu_usage'] = psutil.cpu_percent(interval=1)
                self.system_metrics['memory_usage'] = psutil.virtual_memory().percent
                self.system_metrics['disk_usage'] = psutil.disk_usage('/').percent
                
                # Network I/O (simplified)
                net_io = psutil.net_io_counters()
                self.system_metrics['network_io'] = (net_io.bytes_sent + net_io.bytes_recv) / 1024 / 1024  # MB
                
                await asyncio.sleep(30)  # Update every 30 seconds
                
            except Exception as e:
                self.logger.error(f"Error updating system metrics: {e}")
                await asyncio.sleep(10)
    
    async def _display_dashboard(self):
        """Display the monitoring dashboard"""
        if not self.console:
            return
        
        with Live(self._generate_dashboard(), refresh_per_second=1, console=self.console) as live:
            while self.running:
                live.update(self._generate_dashboard())
                await asyncio.sleep(2)
    
    def _generate_dashboard(self) -> Layout:
        """Generate the dashboard layout"""
        layout = Layout()
        
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="main"),
            Layout(name="footer", size=5)
        )
        
        layout["main"].split_row(
            Layout(name="left"),
            Layout(name="right")
        )
        
        # Header
        uptime = datetime.now() - self.start_time
        header_text = Text(f"🤖 Agent Monitoring Dashboard - Uptime: {str(uptime).split('.')[0]}", 
                          style="bold blue")
        layout["header"].update(Panel(header_text, title="System Status"))
        
        # Left panel - Agent Status
        agent_table = Table(title="Agent Status")
        agent_table.add_column("Agent ID", style="cyan")
        agent_table.add_column("Type", style="magenta")
        agent_table.add_column("Status", style="green")
        agent_table.add_column("Success Rate", justify="right")
        agent_table.add_column("Last Activity")
        
        for agent in self.agents.values():
            status_color = "green" if agent.is_healthy else "red"
            last_activity = agent.last_activity.strftime("%H:%M:%S") if agent.last_activity else "Never"
            agent_table.add_row(
                agent.agent_id,
                agent.agent_type,
                f"[{status_color}]{agent.status}[/{status_color}]",
                f"{agent.success_rate:.1f}%",
                last_activity
            )
        
        layout["left"].update(agent_table)
        
        # Right panel - System Metrics
        metrics_table = Table(title="System Metrics")
        metrics_table.add_column("Metric", style="cyan")
        metrics_table.add_column("Value", justify="right")
        
        metrics_table.add_row("CPU Usage", f"{self.system_metrics['cpu_usage']:.1f}%")
        metrics_table.add_row("Memory Usage", f"{self.system_metrics['memory_usage']:.1f}%")
        metrics_table.add_row("Disk Usage", f"{self.system_metrics['disk_usage']:.1f}%")
        metrics_table.add_row("Active Processes", str(self.system_metrics['active_processes']))
        metrics_table.add_row("Connectivity Health", f"{self.connectivity.overall_health:.1f}%")
        
        layout["right"].update(metrics_table)
        
        # Footer - Recent Alerts
        alerts_text = "Recent Alerts: "
        if self.alerts:
            recent_alerts = list(self.alerts)[-3:]  # Last 3 alerts
            alerts_text += " | ".join([f"{alert['type']}: {alert['message']}" for alert in recent_alerts])
        else:
            alerts_text += "No recent alerts"
        
        layout["footer"].update(Panel(Text(alerts_text), title="Alerts"))
        
        return layout
    
    def _add_alert(self, alert_type: str, message: str):
        """Add an alert to the system"""
        alert = {
            'timestamp': datetime.now(),
            'type': alert_type,
            'message': message
        }
        self.alerts.append(alert)
        self.logger.warning(f"Alert [{alert_type}]: {message}")
    
    def get_status_report(self) -> Dict[str, Any]:
        """Get comprehensive status report"""
        return {
            'timestamp': datetime.now().isoformat(),
            'uptime_seconds': (datetime.now() - self.start_time).total_seconds(),
            'agents': {agent_id: {
                'type': agent.agent_type,
                'status': agent.status,
                'is_healthy': agent.is_healthy,
                'success_rate': agent.success_rate,
                'tasks_completed': agent.tasks_completed,
                'tasks_failed': agent.tasks_failed
            } for agent_id, agent in self.agents.items()},
            'connectivity': {
                'overall_health': self.connectivity.overall_health,
                'trading_system': self.connectivity.trading_system_connected,
                'orchestrator': self.connectivity.orchestrator_connected,
                'market_data': self.connectivity.market_data_connected
            },
            'system_metrics': self.system_metrics,
            'alert_count': len(self.alerts)
        }

async def main():
    """Main function to run the monitoring dashboard"""
    dashboard = AgentMonitoringDashboard()
    
    print("🚀 Starting Agent Monitoring Dashboard...")
    print("Press Ctrl+C to stop")
    
    try:
        await dashboard.start_monitoring()
    except KeyboardInterrupt:
        print("\n👋 Monitoring dashboard stopped")
    except Exception as e:
        print(f"❌ Error: {e}")
        logging.error(f"Dashboard error: {e}")

if __name__ == "__main__":
    asyncio.run(main())