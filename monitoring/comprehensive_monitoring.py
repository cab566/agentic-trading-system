#!/usr/bin/env python3
"""
Comprehensive Monitoring and Benchmarking Framework

Tracks:
- System performance metrics
- AI model performance
- Trading strategy effectiveness
- Resource utilization
- Error rates and anomalies
- Business KPIs
"""

import asyncio
import logging
import time
import json
import yaml
import psutil
import docker
import requests
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import sqlite3
from contextlib import asynccontextmanager
import aiohttp
import schedule
from threading import Thread

# Import session manager for proper aiohttp session handling
try:
    from ..core.session_manager import SessionManager
except ImportError:
    import sys
    from pathlib import Path
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))
    from core.session_manager import SessionManager

import asyncio
import logging
import time
import json
import yaml
import psutil
import docker
import requests
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import sqlite3
from contextlib import asynccontextmanager
import aiohttp
import schedule
from threading import Thread

# Rich for beautiful output
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.live import Live
from rich.layout import Layout
from rich.text import Text

# Prometheus client for metrics
try:
    from prometheus_client import Counter, Histogram, Gauge, start_http_server, CollectorRegistry
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

console = Console()

@dataclass
class SystemMetrics:
    """System performance metrics"""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    disk_usage_percent: float
    network_io: Dict[str, int]
    docker_stats: Dict[str, Any]

@dataclass
class TradingMetrics:
    """Trading performance metrics"""
    timestamp: datetime
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    total_trades: int
    active_positions: int
    portfolio_value: float
    daily_pnl: float

@dataclass
class AIModelMetrics:
    """AI model performance metrics"""
    timestamp: datetime
    model_name: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    inference_time: float
    prediction_confidence: float

@dataclass
class AlertConfig:
    """Alert configuration"""
    metric_name: str
    threshold: float
    comparison: str  # 'gt', 'lt', 'eq'
    severity: str  # 'low', 'medium', 'high', 'critical'
    cooldown_minutes: int = 15

class MetricsCollector:
    """Collects various system and business metrics"""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.docker_client = docker.from_env()
        
        # Database for storing metrics
        self.db_path = project_root / "monitoring" / "metrics.db"
        self.db_path.parent.mkdir(exist_ok=True)
        
        # Initialize database
        self._init_database()
        
        # Prometheus metrics (if available)
        if PROMETHEUS_AVAILABLE:
            self.registry = CollectorRegistry()
            self._init_prometheus_metrics()
        
        # Alert tracking
        self.last_alerts = {}
        
        # Configuration
        self.config = self._load_config()

    def _init_database(self):
        """Initialize SQLite database for metrics storage"""
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS system_metrics (
                    timestamp TEXT PRIMARY KEY,
                    cpu_percent REAL,
                    memory_percent REAL,
                    disk_usage_percent REAL,
                    network_io TEXT,
                    docker_stats TEXT
                );
                
                CREATE TABLE IF NOT EXISTS trading_metrics (
                    timestamp TEXT PRIMARY KEY,
                    total_return REAL,
                    sharpe_ratio REAL,
                    max_drawdown REAL,
                    win_rate REAL,
                    total_trades INTEGER,
                    active_positions INTEGER,
                    portfolio_value REAL,
                    daily_pnl REAL
                );
                
                CREATE TABLE IF NOT EXISTS ai_model_metrics (
                    timestamp TEXT,
                    model_name TEXT,
                    accuracy REAL,
                    precision REAL,
                    recall REAL,
                    f1_score REAL,
                    inference_time REAL,
                    prediction_confidence REAL,
                    PRIMARY KEY (timestamp, model_name)
                );
                
                CREATE TABLE IF NOT EXISTS alerts (
                    timestamp TEXT,
                    metric_name TEXT,
                    value REAL,
                    threshold REAL,
                    severity TEXT,
                    message TEXT
                );
                
                CREATE INDEX IF NOT EXISTS idx_system_timestamp ON system_metrics(timestamp);
                CREATE INDEX IF NOT EXISTS idx_trading_timestamp ON trading_metrics(timestamp);
                CREATE INDEX IF NOT EXISTS idx_ai_timestamp ON ai_model_metrics(timestamp);
                CREATE INDEX IF NOT EXISTS idx_alerts_timestamp ON alerts(timestamp);
            """)

    def _init_prometheus_metrics(self):
        """Initialize Prometheus metrics"""
        if not PROMETHEUS_AVAILABLE:
            return
        
        # System metrics
        self.cpu_gauge = Gauge('system_cpu_percent', 'CPU usage percentage', registry=self.registry)
        self.memory_gauge = Gauge('system_memory_percent', 'Memory usage percentage', registry=self.registry)
        self.disk_gauge = Gauge('system_disk_percent', 'Disk usage percentage', registry=self.registry)
        
        # Trading metrics
        self.portfolio_value_gauge = Gauge('trading_portfolio_value', 'Portfolio value', registry=self.registry)
        self.daily_pnl_gauge = Gauge('trading_daily_pnl', 'Daily P&L', registry=self.registry)
        self.total_trades_counter = Counter('trading_total_trades', 'Total number of trades', registry=self.registry)
        
        # AI model metrics
        self.model_accuracy_gauge = Gauge('ai_model_accuracy', 'Model accuracy', ['model_name'], registry=self.registry)
        self.inference_time_histogram = Histogram('ai_inference_time_seconds', 'Model inference time', ['model_name'], registry=self.registry)

    def _load_config(self) -> Dict[str, Any]:
        """Load monitoring configuration"""
        config_file = self.project_root / "config" / "monitoring_config.yaml"
        
        default_config = {
            "collection_interval": 60,  # seconds
            "retention_days": 30,
            "alerts": [
                {
                    "metric_name": "cpu_percent",
                    "threshold": 80.0,
                    "comparison": "gt",
                    "severity": "high",
                    "cooldown_minutes": 15
                },
                {
                    "metric_name": "memory_percent",
                    "threshold": 85.0,
                    "comparison": "gt",
                    "severity": "high",
                    "cooldown_minutes": 15
                },
                {
                    "metric_name": "max_drawdown",
                    "threshold": -0.1,  # 10% drawdown
                    "comparison": "lt",
                    "severity": "critical",
                    "cooldown_minutes": 5
                }
            ],
            "endpoints": {
                "trading_api": "http://localhost:8000/api/v1",
                "prometheus": "http://localhost:9090"
            }
        }
        
        if config_file.exists():
            try:
                with open(config_file, 'r') as f:
                    config = yaml.safe_load(f)
                return {**default_config, **config}
            except Exception as e:
                console.print(f"[yellow]Failed to load config, using defaults: {e}[/yellow]")
        
        return default_config

    async def collect_system_metrics(self) -> SystemMetrics:
        """Collect system performance metrics"""
        try:
            # CPU and memory
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Network I/O
            network = psutil.net_io_counters()
            network_io = {
                "bytes_sent": network.bytes_sent,
                "bytes_recv": network.bytes_recv,
                "packets_sent": network.packets_sent,
                "packets_recv": network.packets_recv
            }
            
            # Docker stats
            docker_stats = {}
            try:
                containers = self.docker_client.containers.list()
                for container in containers:
                    stats = container.stats(stream=False)
                    docker_stats[container.name] = {
                        "cpu_percent": self._calculate_cpu_percent(stats),
                        "memory_usage": stats['memory_stats'].get('usage', 0),
                        "memory_limit": stats['memory_stats'].get('limit', 0),
                        "network_rx": stats['networks'].get('eth0', {}).get('rx_bytes', 0) if 'networks' in stats else 0,
                        "network_tx": stats['networks'].get('eth0', {}).get('tx_bytes', 0) if 'networks' in stats else 0
                    }
            except Exception as e:
                console.print(f"[yellow]Docker stats collection failed: {e}[/yellow]")
            
            metrics = SystemMetrics(
                timestamp=datetime.now(),
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                disk_usage_percent=disk.percent,
                network_io=network_io,
                docker_stats=docker_stats
            )
            
            # Update Prometheus metrics
            if PROMETHEUS_AVAILABLE:
                self.cpu_gauge.set(cpu_percent)
                self.memory_gauge.set(memory.percent)
                self.disk_gauge.set(disk.percent)
            
            return metrics
            
        except Exception as e:
            console.print(f"[red]System metrics collection failed: {e}[/red]")
            raise

    def _calculate_cpu_percent(self, stats: Dict) -> float:
        """Calculate CPU percentage from Docker stats"""
        try:
            cpu_delta = stats['cpu_stats']['cpu_usage']['total_usage'] - \
                       stats['precpu_stats']['cpu_usage']['total_usage']
            system_delta = stats['cpu_stats']['system_cpu_usage'] - \
                          stats['precpu_stats']['system_cpu_usage']
            
            if system_delta > 0:
                cpu_percent = (cpu_delta / system_delta) * len(stats['cpu_stats']['cpu_usage']['percpu_usage']) * 100
                return round(cpu_percent, 2)
        except (KeyError, ZeroDivisionError):
            pass
        
        return 0.0

    async def collect_trading_metrics(self) -> Optional[TradingMetrics]:
        """Collect trading performance metrics"""
        try:
            # Get metrics from trading API using managed session
            session_manager = SessionManager()
            async with session_manager.get_session() as session:
                endpoints = [
                    "/portfolio/summary",
                    "/trading/performance",
                    "/positions/active"
                ]
                
                results = {}
                for endpoint in endpoints:
                    url = f"{self.config['endpoints']['trading_api']}{endpoint}"
                    try:
                        async with session.get(url, timeout=10) as response:
                            if response.status == 200:
                                results[endpoint] = await response.json()
                    except Exception as e:
                        console.print(f"[yellow]Failed to fetch {endpoint}: {e}[/yellow]")
                
                if not results:
                    return None
                
                # Extract metrics
                portfolio = results.get("/portfolio/summary", {})
                performance = results.get("/trading/performance", {})
                positions = results.get("/positions/active", {})
                
                metrics = TradingMetrics(
                    timestamp=datetime.now(),
                    total_return=performance.get("total_return", 0.0),
                    sharpe_ratio=performance.get("sharpe_ratio", 0.0),
                    max_drawdown=performance.get("max_drawdown", 0.0),
                    win_rate=performance.get("win_rate", 0.0),
                    total_trades=performance.get("total_trades", 0),
                    active_positions=len(positions.get("positions", [])),
                    portfolio_value=portfolio.get("total_value", 0.0),
                    daily_pnl=portfolio.get("daily_pnl", 0.0)
                )
                
                # Update Prometheus metrics
                if PROMETHEUS_AVAILABLE:
                    self.portfolio_value_gauge.set(metrics.portfolio_value)
                    self.daily_pnl_gauge.set(metrics.daily_pnl)
                
                return metrics
                
        except Exception as e:
            console.print(f"[red]Trading metrics collection failed: {e}[/red]")
            return None

    async def collect_ai_model_metrics(self) -> List[AIModelMetrics]:
        """Collect AI model performance metrics"""
        try:
            metrics_list = []
            
            # Get model metrics from AI service using managed session
            session_manager = SessionManager()
            async with session_manager.get_session() as session:
                url = f"{self.config['endpoints']['trading_api']}/ai/models/metrics"
                try:
                    async with session.get(url, timeout=10) as response:
                        if response.status == 200:
                            models_data = await response.json()
                            
                            for model_name, model_metrics in models_data.items():
                                metrics = AIModelMetrics(
                                    timestamp=datetime.now(),
                                    model_name=model_name,
                                    accuracy=model_metrics.get("accuracy", 0.0),
                                    precision=model_metrics.get("precision", 0.0),
                                    recall=model_metrics.get("recall", 0.0),
                                    f1_score=model_metrics.get("f1_score", 0.0),
                                    inference_time=model_metrics.get("inference_time", 0.0),
                                    prediction_confidence=model_metrics.get("prediction_confidence", 0.0)
                                )
                                metrics_list.append(metrics)
                                
                                # Update Prometheus metrics
                                if PROMETHEUS_AVAILABLE:
                                    self.model_accuracy_gauge.labels(model_name=model_name).set(metrics.accuracy)
                                    self.inference_time_histogram.labels(model_name=model_name).observe(metrics.inference_time)
                
                except Exception as e:
                    console.print(f"[yellow]Failed to fetch AI model metrics: {e}[/yellow]")
            
            return metrics_list
            
        except Exception as e:
            console.print(f"[red]AI model metrics collection failed: {e}[/red]")
            return []

    def store_metrics(self, system_metrics: SystemMetrics, 
                     trading_metrics: Optional[TradingMetrics],
                     ai_metrics: List[AIModelMetrics]):
        """Store metrics in database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Store system metrics
                conn.execute("""
                    INSERT OR REPLACE INTO system_metrics 
                    (timestamp, cpu_percent, memory_percent, disk_usage_percent, network_io, docker_stats)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    system_metrics.timestamp.isoformat(),
                    system_metrics.cpu_percent,
                    system_metrics.memory_percent,
                    system_metrics.disk_usage_percent,
                    json.dumps(system_metrics.network_io),
                    json.dumps(system_metrics.docker_stats)
                ))
                
                # Store trading metrics
                if trading_metrics:
                    conn.execute("""
                        INSERT OR REPLACE INTO trading_metrics 
                        (timestamp, total_return, sharpe_ratio, max_drawdown, win_rate, 
                         total_trades, active_positions, portfolio_value, daily_pnl)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        trading_metrics.timestamp.isoformat(),
                        trading_metrics.total_return,
                        trading_metrics.sharpe_ratio,
                        trading_metrics.max_drawdown,
                        trading_metrics.win_rate,
                        trading_metrics.total_trades,
                        trading_metrics.active_positions,
                        trading_metrics.portfolio_value,
                        trading_metrics.daily_pnl
                    ))
                
                # Store AI model metrics
                for ai_metric in ai_metrics:
                    conn.execute("""
                        INSERT OR REPLACE INTO ai_model_metrics 
                        (timestamp, model_name, accuracy, precision, recall, f1_score, 
                         inference_time, prediction_confidence)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        ai_metric.timestamp.isoformat(),
                        ai_metric.model_name,
                        ai_metric.accuracy,
                        ai_metric.precision,
                        ai_metric.recall,
                        ai_metric.f1_score,
                        ai_metric.inference_time,
                        ai_metric.prediction_confidence
                    ))
                
                conn.commit()
                
        except Exception as e:
            console.print(f"[red]Failed to store metrics: {e}[/red]")

    def check_alerts(self, system_metrics: SystemMetrics, 
                    trading_metrics: Optional[TradingMetrics]):
        """Check for alert conditions"""
        try:
            current_time = datetime.now()
            
            for alert_config in self.config.get("alerts", []):
                alert_key = alert_config["metric_name"]
                
                # Check cooldown
                if alert_key in self.last_alerts:
                    last_alert_time = self.last_alerts[alert_key]
                    cooldown = timedelta(minutes=alert_config.get("cooldown_minutes", 15))
                    if current_time - last_alert_time < cooldown:
                        continue
                
                # Get metric value
                metric_value = self._get_metric_value(
                    alert_config["metric_name"], 
                    system_metrics, 
                    trading_metrics
                )
                
                if metric_value is None:
                    continue
                
                # Check threshold
                threshold = alert_config["threshold"]
                comparison = alert_config["comparison"]
                
                alert_triggered = False
                if comparison == "gt" and metric_value > threshold:
                    alert_triggered = True
                elif comparison == "lt" and metric_value < threshold:
                    alert_triggered = True
                elif comparison == "eq" and abs(metric_value - threshold) < 0.001:
                    alert_triggered = True
                
                if alert_triggered:
                    self._trigger_alert(alert_config, metric_value, current_time)
                    self.last_alerts[alert_key] = current_time
                    
        except Exception as e:
            console.print(f"[red]Alert checking failed: {e}[/red]")

    def _get_metric_value(self, metric_name: str, 
                         system_metrics: SystemMetrics,
                         trading_metrics: Optional[TradingMetrics]) -> Optional[float]:
        """Get metric value by name"""
        # System metrics
        if metric_name == "cpu_percent":
            return system_metrics.cpu_percent
        elif metric_name == "memory_percent":
            return system_metrics.memory_percent
        elif metric_name == "disk_usage_percent":
            return system_metrics.disk_usage_percent
        
        # Trading metrics
        if trading_metrics:
            if metric_name == "total_return":
                return trading_metrics.total_return
            elif metric_name == "sharpe_ratio":
                return trading_metrics.sharpe_ratio
            elif metric_name == "max_drawdown":
                return trading_metrics.max_drawdown
            elif metric_name == "win_rate":
                return trading_metrics.win_rate
            elif metric_name == "portfolio_value":
                return trading_metrics.portfolio_value
            elif metric_name == "daily_pnl":
                return trading_metrics.daily_pnl
        
        return None

    def _trigger_alert(self, alert_config: Dict, value: float, timestamp: datetime):
        """Trigger an alert"""
        message = f"Alert: {alert_config['metric_name']} = {value:.2f} (threshold: {alert_config['threshold']})"
        
        # Store alert in database
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO alerts (timestamp, metric_name, value, threshold, severity, message)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    timestamp.isoformat(),
                    alert_config["metric_name"],
                    value,
                    alert_config["threshold"],
                    alert_config["severity"],
                    message
                ))
                conn.commit()
        except Exception as e:
            console.print(f"[red]Failed to store alert: {e}[/red]")
        
        # Display alert
        severity_colors = {
            "low": "yellow",
            "medium": "orange",
            "high": "red",
            "critical": "bold red"
        }
        color = severity_colors.get(alert_config["severity"], "yellow")
        console.print(f"[{color}]{message}[/{color}]")

    def get_metrics_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get metrics summary for the last N hours"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours)
            
            with sqlite3.connect(self.db_path) as conn:
                # System metrics summary
                system_query = """
                    SELECT 
                        AVG(cpu_percent) as avg_cpu,
                        MAX(cpu_percent) as max_cpu,
                        AVG(memory_percent) as avg_memory,
                        MAX(memory_percent) as max_memory,
                        AVG(disk_usage_percent) as avg_disk,
                        COUNT(*) as data_points
                    FROM system_metrics 
                    WHERE timestamp > ?
                """
                system_result = conn.execute(system_query, (cutoff_time.isoformat(),)).fetchone()
                
                # Trading metrics summary
                trading_query = """
                    SELECT 
                        AVG(total_return) as avg_return,
                        MIN(max_drawdown) as worst_drawdown,
                        AVG(win_rate) as avg_win_rate,
                        SUM(total_trades) as total_trades,
                        AVG(portfolio_value) as avg_portfolio_value,
                        SUM(daily_pnl) as total_pnl
                    FROM trading_metrics 
                    WHERE timestamp > ?
                """
                trading_result = conn.execute(trading_query, (cutoff_time.isoformat(),)).fetchone()
                
                # Recent alerts
                alerts_query = """
                    SELECT severity, COUNT(*) as count
                    FROM alerts 
                    WHERE timestamp > ?
                    GROUP BY severity
                """
                alerts_results = conn.execute(alerts_query, (cutoff_time.isoformat(),)).fetchall()
                
                return {
                    "period_hours": hours,
                    "system": {
                        "avg_cpu_percent": system_result[0] or 0,
                        "max_cpu_percent": system_result[1] or 0,
                        "avg_memory_percent": system_result[2] or 0,
                        "max_memory_percent": system_result[3] or 0,
                        "avg_disk_percent": system_result[4] or 0,
                        "data_points": system_result[5] or 0
                    },
                    "trading": {
                        "avg_return": trading_result[0] or 0,
                        "worst_drawdown": trading_result[1] or 0,
                        "avg_win_rate": trading_result[2] or 0,
                        "total_trades": trading_result[3] or 0,
                        "avg_portfolio_value": trading_result[4] or 0,
                        "total_pnl": trading_result[5] or 0
                    } if trading_result[0] is not None else None,
                    "alerts": {row[0]: row[1] for row in alerts_results}
                }
                
        except Exception as e:
            console.print(f"[red]Failed to get metrics summary: {e}[/red]")
            return {}

    def cleanup_old_data(self):
        """Clean up old metrics data"""
        try:
            cutoff_time = datetime.now() - timedelta(days=self.config.get("retention_days", 30))
            
            with sqlite3.connect(self.db_path) as conn:
                tables = ["system_metrics", "trading_metrics", "ai_model_metrics", "alerts"]
                
                for table in tables:
                    result = conn.execute(f"DELETE FROM {table} WHERE timestamp < ?", 
                                        (cutoff_time.isoformat(),))
                    if result.rowcount > 0:
                        console.print(f"[yellow]Cleaned up {result.rowcount} old records from {table}[/yellow]")
                
                conn.commit()
                
        except Exception as e:
            console.print(f"[red]Data cleanup failed: {e}[/red]")

class MonitoringDashboard:
    """Real-time monitoring dashboard"""
    
    def __init__(self, collector: MetricsCollector):
        self.collector = collector
        self.layout = Layout()
        self._setup_layout()

    def _setup_layout(self):
        """Setup dashboard layout"""
        self.layout.split_column(
            Layout(name="header", size=3),
            Layout(name="main"),
            Layout(name="footer", size=3)
        )
        
        self.layout["main"].split_row(
            Layout(name="left"),
            Layout(name="right")
        )
        
        self.layout["left"].split_column(
            Layout(name="system"),
            Layout(name="trading")
        )
        
        self.layout["right"].split_column(
            Layout(name="ai_models"),
            Layout(name="alerts")
        )

    async def update_dashboard(self):
        """Update dashboard with latest metrics"""
        try:
            # Collect latest metrics
            system_metrics = await self.collector.collect_system_metrics()
            trading_metrics = await self.collector.collect_trading_metrics()
            ai_metrics = await self.collector.collect_ai_model_metrics()
            
            # Update header
            self.layout["header"].update(Panel(
                f"[bold blue]AI Trading System Monitoring Dashboard[/bold blue]\n"
                f"Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                border_style="blue"
            ))
            
            # Update system metrics
            system_table = Table(title="System Metrics")
            system_table.add_column("Metric", style="cyan")
            system_table.add_column("Value", style="green")
            system_table.add_column("Status", style="yellow")
            
            system_table.add_row("CPU Usage", f"{system_metrics.cpu_percent:.1f}%", 
                                self._get_status(system_metrics.cpu_percent, 80, 90))
            system_table.add_row("Memory Usage", f"{system_metrics.memory_percent:.1f}%",
                                self._get_status(system_metrics.memory_percent, 80, 90))
            system_table.add_row("Disk Usage", f"{system_metrics.disk_usage_percent:.1f}%",
                                self._get_status(system_metrics.disk_usage_percent, 80, 90))
            
            self.layout["system"].update(Panel(system_table, title="System", border_style="green"))
            
            # Update trading metrics
            if trading_metrics:
                trading_table = Table(title="Trading Performance")
                trading_table.add_column("Metric", style="cyan")
                trading_table.add_column("Value", style="green")
                
                trading_table.add_row("Portfolio Value", f"${trading_metrics.portfolio_value:,.2f}")
                trading_table.add_row("Daily P&L", f"${trading_metrics.daily_pnl:,.2f}")
                trading_table.add_row("Total Return", f"{trading_metrics.total_return:.2%}")
                trading_table.add_row("Sharpe Ratio", f"{trading_metrics.sharpe_ratio:.2f}")
                trading_table.add_row("Max Drawdown", f"{trading_metrics.max_drawdown:.2%}")
                trading_table.add_row("Win Rate", f"{trading_metrics.win_rate:.1%}")
                trading_table.add_row("Active Positions", str(trading_metrics.active_positions))
                
                self.layout["trading"].update(Panel(trading_table, title="Trading", border_style="blue"))
            else:
                self.layout["trading"].update(Panel("Trading metrics unavailable", title="Trading", border_style="red"))
            
            # Update AI model metrics
            if ai_metrics:
                ai_table = Table(title="AI Model Performance")
                ai_table.add_column("Model", style="cyan")
                ai_table.add_column("Accuracy", style="green")
                ai_table.add_column("F1 Score", style="green")
                ai_table.add_column("Inference Time", style="yellow")
                
                for metric in ai_metrics:
                    ai_table.add_row(
                        metric.model_name,
                        f"{metric.accuracy:.3f}",
                        f"{metric.f1_score:.3f}",
                        f"{metric.inference_time:.3f}s"
                    )
                
                self.layout["ai_models"].update(Panel(ai_table, title="AI Models", border_style="magenta"))
            else:
                self.layout["ai_models"].update(Panel("AI model metrics unavailable", title="AI Models", border_style="red"))
            
            # Update alerts (show recent alerts)
            summary = self.collector.get_metrics_summary(hours=1)
            alerts_info = summary.get("alerts", {})
            
            alerts_text = "Recent Alerts (1h):\n"
            if alerts_info:
                for severity, count in alerts_info.items():
                    color = {"low": "yellow", "medium": "orange", "high": "red", "critical": "bold red"}.get(severity, "white")
                    alerts_text += f"[{color}]{severity.upper()}: {count}[/{color}]\n"
            else:
                alerts_text += "[green]No alerts[/green]"
            
            self.layout["alerts"].update(Panel(alerts_text, title="Alerts", border_style="yellow"))
            
            # Update footer
            self.layout["footer"].update(Panel(
                f"[dim]Monitoring Framework v1.0 | Data Points: {summary.get('system', {}).get('data_points', 0)} | "
                f"Retention: {self.collector.config.get('retention_days', 30)} days[/dim]",
                border_style="dim"
            ))
            
        except Exception as e:
            console.print(f"[red]Dashboard update failed: {e}[/red]")

    def _get_status(self, value: float, warning_threshold: float, critical_threshold: float) -> str:
        """Get status indicator based on thresholds"""
        if value >= critical_threshold:
            return "[red]CRITICAL[/red]"
        elif value >= warning_threshold:
            return "[yellow]WARNING[/yellow]"
        else:
            return "[green]OK[/green]"

class MonitoringService:
    """Main monitoring service"""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.collector = MetricsCollector(project_root)
        self.dashboard = MonitoringDashboard(self.collector)
        self.running = False

    async def start(self, dashboard_mode: bool = False):
        """Start monitoring service"""
        console.print("[green]Starting monitoring service...[/green]")
        
        # Start Prometheus metrics server if available
        if PROMETHEUS_AVAILABLE:
            try:
                start_http_server(8001, registry=self.collector.registry)
                console.print("[green]Prometheus metrics server started on port 8001[/green]")
            except Exception as e:
                console.print(f"[yellow]Failed to start Prometheus server: {e}[/yellow]")
        
        self.running = True
        
        if dashboard_mode:
            await self._run_dashboard()
        else:
            await self._run_background()

    async def _run_dashboard(self):
        """Run with live dashboard"""
        with Live(self.dashboard.layout, refresh_per_second=1, screen=True):
            while self.running:
                await self.dashboard.update_dashboard()
                await asyncio.sleep(self.collector.config.get("collection_interval", 60))

    async def _run_background(self):
        """Run in background mode"""
        while self.running:
            try:
                # Collect metrics
                system_metrics = await self.collector.collect_system_metrics()
                trading_metrics = await self.collector.collect_trading_metrics()
                ai_metrics = await self.collector.collect_ai_model_metrics()
                
                # Store metrics
                self.collector.store_metrics(system_metrics, trading_metrics, ai_metrics)
                
                # Check alerts
                self.collector.check_alerts(system_metrics, trading_metrics)
                
                # Periodic cleanup
                if datetime.now().hour == 2 and datetime.now().minute < 5:  # Daily at 2 AM
                    self.collector.cleanup_old_data()
                
                await asyncio.sleep(self.collector.config.get("collection_interval", 60))
                
            except Exception as e:
                console.print(f"[red]Monitoring cycle failed: {e}[/red]")
                await asyncio.sleep(30)  # Wait before retrying

    def stop(self):
        """Stop monitoring service"""
        console.print("[yellow]Stopping monitoring service...[/yellow]")
        self.running = False

async def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Comprehensive Monitoring System")
    parser.add_argument("--project-root", default=".", help="Project root directory")
    parser.add_argument("--dashboard", action="store_true", help="Run with live dashboard")
    parser.add_argument("--summary", action="store_true", help="Show metrics summary")
    parser.add_argument("--hours", type=int, default=24, help="Hours for summary")
    
    args = parser.parse_args()
    
    project_root = Path(args.project_root).resolve()
    service = MonitoringService(project_root)
    
    if args.summary:
        summary = service.collector.get_metrics_summary(args.hours)
        console.print(Panel(json.dumps(summary, indent=2), title=f"Metrics Summary ({args.hours}h)"))
        return
    
    try:
        await service.start(dashboard_mode=args.dashboard)
    except KeyboardInterrupt:
        service.stop()
        console.print("[green]Monitoring service stopped[/green]")

if __name__ == "__main__":
    asyncio.run(main())