#!/usr/bin/env python3
"""
Health Monitor for CrewAI Trading System

Monitors system health, performance metrics, and component status
with alerting and automatic recovery capabilities.
"""

import asyncio
import logging
import psutil
import threading
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import json
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Float, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

from .config_manager import ConfigManager


class HealthStatus(Enum):
    """Health status levels."""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


class AlertLevel(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class HealthMetric:
    """Health metric data structure."""
    name: str
    value: float
    unit: str
    timestamp: datetime
    status: HealthStatus = HealthStatus.HEALTHY
    threshold_warning: Optional[float] = None
    threshold_critical: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Alert:
    """Alert data structure."""
    id: str
    level: AlertLevel
    component: str
    message: str
    timestamp: datetime
    resolved: bool = False
    resolved_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class ComponentMonitor:
    """Base class for component-specific monitors."""
    
    def __init__(self, name: str, config: Dict[str, Any]):
        self.name = name
        self.config = config
        self.enabled = config.get('enabled', True)
        self.check_interval = config.get('check_interval', 60)  # seconds
        self.logger = logging.getLogger(f"{__name__}.{name}")
        
        # Health metrics
        self.metrics: Dict[str, HealthMetric] = {}
        self.last_check = None
        self.status = HealthStatus.UNKNOWN
        
        # Callbacks
        self.alert_callback: Optional[Callable] = None
    
    def set_alert_callback(self, callback: Callable):
        """Set callback for alerts."""
        self.alert_callback = callback
    
    def _send_alert(self, level: AlertLevel, message: str, metadata: Dict[str, Any] = None):
        """Send an alert."""
        if self.alert_callback:
            alert = Alert(
                id=f"{self.name}_{datetime.now().isoformat()}",
                level=level,
                component=self.name,
                message=message,
                timestamp=datetime.now(),
                metadata=metadata or {}
            )
            self.alert_callback(alert)
    
    async def check_health(self) -> HealthStatus:
        """Check component health. To be implemented by subclasses."""
        raise NotImplementedError
    
    def get_metrics(self) -> Dict[str, HealthMetric]:
        """Get current metrics."""
        return self.metrics.copy()
    
    def get_status(self) -> HealthStatus:
        """Get current status."""
        return self.status


class SystemMonitor(ComponentMonitor):
    """System resource monitor."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__('system', config)
        
        # Thresholds
        self.cpu_warning = config.get('cpu_warning_threshold', 80.0)
        self.cpu_critical = config.get('cpu_critical_threshold', 95.0)
        self.memory_warning = config.get('memory_warning_threshold', 80.0)
        self.memory_critical = config.get('memory_critical_threshold', 95.0)
        self.disk_warning = config.get('disk_warning_threshold', 85.0)
        self.disk_critical = config.get('disk_critical_threshold', 95.0)
    
    async def check_health(self) -> HealthStatus:
        """Check system health."""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_status = self._get_status_from_thresholds(
                cpu_percent, self.cpu_warning, self.cpu_critical
            )
            
            self.metrics['cpu_usage'] = HealthMetric(
                name='cpu_usage',
                value=cpu_percent,
                unit='%',
                timestamp=datetime.now(),
                status=cpu_status,
                threshold_warning=self.cpu_warning,
                threshold_critical=self.cpu_critical
            )
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_status = self._get_status_from_thresholds(
                memory.percent, self.memory_warning, self.memory_critical
            )
            
            self.metrics['memory_usage'] = HealthMetric(
                name='memory_usage',
                value=memory.percent,
                unit='%',
                timestamp=datetime.now(),
                status=memory_status,
                threshold_warning=self.memory_warning,
                threshold_critical=self.memory_critical,
                metadata={
                    'total': memory.total,
                    'available': memory.available,
                    'used': memory.used
                }
            )
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            disk_status = self._get_status_from_thresholds(
                disk_percent, self.disk_warning, self.disk_critical
            )
            
            self.metrics['disk_usage'] = HealthMetric(
                name='disk_usage',
                value=disk_percent,
                unit='%',
                timestamp=datetime.now(),
                status=disk_status,
                threshold_warning=self.disk_warning,
                threshold_critical=self.disk_critical,
                metadata={
                    'total': disk.total,
                    'used': disk.used,
                    'free': disk.free
                }
            )
            
            # Network I/O
            network = psutil.net_io_counters()
            self.metrics['network_bytes_sent'] = HealthMetric(
                name='network_bytes_sent',
                value=network.bytes_sent,
                unit='bytes',
                timestamp=datetime.now()
            )
            
            self.metrics['network_bytes_recv'] = HealthMetric(
                name='network_bytes_recv',
                value=network.bytes_recv,
                unit='bytes',
                timestamp=datetime.now()
            )
            
            # Overall status
            statuses = [cpu_status, memory_status, disk_status]
            if HealthStatus.CRITICAL in statuses:
                self.status = HealthStatus.CRITICAL
                self._send_alert(AlertLevel.CRITICAL, "System resources critical")
            elif HealthStatus.WARNING in statuses:
                self.status = HealthStatus.WARNING
                self._send_alert(AlertLevel.WARNING, "System resources warning")
            else:
                self.status = HealthStatus.HEALTHY
            
            self.last_check = datetime.now()
            return self.status
            
        except Exception as e:
            self.logger.error(f"Error checking system health: {e}")
            self.status = HealthStatus.UNKNOWN
            self._send_alert(AlertLevel.ERROR, f"System health check failed: {e}")
            return self.status
    
    def _get_status_from_thresholds(self, value: float, warning: float, critical: float) -> HealthStatus:
        """Get status based on thresholds."""
        if value >= critical:
            return HealthStatus.CRITICAL
        elif value >= warning:
            return HealthStatus.WARNING
        else:
            return HealthStatus.HEALTHY


class AgentMonitor(ComponentMonitor):
    """CrewAI agent monitor."""
    
    def __init__(self, agent_name: str, config: Dict[str, Any]):
        super().__init__(f'agent_{agent_name}', config)
        self.agent_name = agent_name
        
        # Agent-specific metrics
        self.task_count = 0
        self.success_count = 0
        self.error_count = 0
        self.last_task_time = None
        self.average_task_duration = 0.0
        self.task_durations = []
    
    async def check_health(self) -> HealthStatus:
        """Check agent health."""
        try:
            now = datetime.now()
            
            # Task completion rate
            total_tasks = self.task_count
            success_rate = (self.success_count / total_tasks * 100) if total_tasks > 0 else 100
            
            self.metrics['task_success_rate'] = HealthMetric(
                name='task_success_rate',
                value=success_rate,
                unit='%',
                timestamp=now,
                status=HealthStatus.HEALTHY if success_rate >= 90 else 
                       HealthStatus.WARNING if success_rate >= 70 else 
                       HealthStatus.CRITICAL
            )
            
            # Task count
            self.metrics['total_tasks'] = HealthMetric(
                name='total_tasks',
                value=total_tasks,
                unit='count',
                timestamp=now
            )
            
            # Error rate
            error_rate = (self.error_count / total_tasks * 100) if total_tasks > 0 else 0
            self.metrics['error_rate'] = HealthMetric(
                name='error_rate',
                value=error_rate,
                unit='%',
                timestamp=now,
                status=HealthStatus.HEALTHY if error_rate <= 5 else 
                       HealthStatus.WARNING if error_rate <= 15 else 
                       HealthStatus.CRITICAL
            )
            
            # Average task duration
            if self.task_durations:
                self.average_task_duration = sum(self.task_durations) / len(self.task_durations)
            
            self.metrics['avg_task_duration'] = HealthMetric(
                name='avg_task_duration',
                value=self.average_task_duration,
                unit='seconds',
                timestamp=now
            )
            
            # Last activity
            if self.last_task_time:
                time_since_last = (now - self.last_task_time).total_seconds()
                self.metrics['time_since_last_task'] = HealthMetric(
                    name='time_since_last_task',
                    value=time_since_last,
                    unit='seconds',
                    timestamp=now
                )
            
            # Overall status
            success_status = self.metrics['task_success_rate'].status
            error_status = self.metrics['error_rate'].status
            
            if success_status == HealthStatus.CRITICAL or error_status == HealthStatus.CRITICAL:
                self.status = HealthStatus.CRITICAL
            elif success_status == HealthStatus.WARNING or error_status == HealthStatus.WARNING:
                self.status = HealthStatus.WARNING
            else:
                self.status = HealthStatus.HEALTHY
            
            self.last_check = now
            return self.status
            
        except Exception as e:
            self.logger.error(f"Error checking agent health: {e}")
            self.status = HealthStatus.UNKNOWN
            return self.status
    
    def record_task_start(self):
        """Record task start."""
        self.task_count += 1
        self.last_task_time = datetime.now()
    
    def record_task_success(self, duration: float):
        """Record successful task completion."""
        self.success_count += 1
        self.task_durations.append(duration)
        
        # Keep only last 100 durations
        if len(self.task_durations) > 100:
            self.task_durations = self.task_durations[-100:]
    
    def record_task_error(self, error: str):
        """Record task error."""
        self.error_count += 1
        self._send_alert(AlertLevel.WARNING, f"Agent task failed: {error}")


class DataSourceMonitor(ComponentMonitor):
    """Data source monitor."""
    
    def __init__(self, source_name: str, config: Dict[str, Any]):
        super().__init__(f'datasource_{source_name}', config)
        self.source_name = source_name
        
        # Data source metrics
        self.request_count = 0
        self.success_count = 0
        self.error_count = 0
        self.last_request_time = None
        self.response_times = []
    
    async def check_health(self) -> HealthStatus:
        """Check data source health."""
        try:
            now = datetime.now()
            
            # Success rate
            total_requests = self.request_count
            success_rate = (self.success_count / total_requests * 100) if total_requests > 0 else 100
            
            self.metrics['request_success_rate'] = HealthMetric(
                name='request_success_rate',
                value=success_rate,
                unit='%',
                timestamp=now,
                status=HealthStatus.HEALTHY if success_rate >= 95 else 
                       HealthStatus.WARNING if success_rate >= 80 else 
                       HealthStatus.CRITICAL
            )
            
            # Request count
            self.metrics['total_requests'] = HealthMetric(
                name='total_requests',
                value=total_requests,
                unit='count',
                timestamp=now
            )
            
            # Average response time
            avg_response_time = sum(self.response_times) / len(self.response_times) if self.response_times else 0
            self.metrics['avg_response_time'] = HealthMetric(
                name='avg_response_time',
                value=avg_response_time,
                unit='seconds',
                timestamp=now,
                status=HealthStatus.HEALTHY if avg_response_time <= 2.0 else 
                       HealthStatus.WARNING if avg_response_time <= 5.0 else 
                       HealthStatus.CRITICAL
            )
            
            # Last request time
            if self.last_request_time:
                time_since_last = (now - self.last_request_time).total_seconds()
                self.metrics['time_since_last_request'] = HealthMetric(
                    name='time_since_last_request',
                    value=time_since_last,
                    unit='seconds',
                    timestamp=now
                )
            
            # Overall status
            success_status = self.metrics['request_success_rate'].status
            response_status = self.metrics['avg_response_time'].status
            
            if success_status == HealthStatus.CRITICAL or response_status == HealthStatus.CRITICAL:
                self.status = HealthStatus.CRITICAL
            elif success_status == HealthStatus.WARNING or response_status == HealthStatus.WARNING:
                self.status = HealthStatus.WARNING
            else:
                self.status = HealthStatus.HEALTHY
            
            self.last_check = now
            return self.status
            
        except Exception as e:
            self.logger.error(f"Error checking data source health: {e}")
            self.status = HealthStatus.UNKNOWN
            return self.status
    
    def record_request(self):
        """Record a request."""
        self.request_count += 1
        self.last_request_time = datetime.now()
    
    def record_success(self, response_time: float):
        """Record successful request."""
        self.success_count += 1
        self.response_times.append(response_time)
        
        # Keep only last 100 response times
        if len(self.response_times) > 100:
            self.response_times = self.response_times[-100:]
    
    def record_error(self, error: str):
        """Record request error."""
        self.error_count += 1
        self._send_alert(AlertLevel.WARNING, f"Data source error: {error}")


class HealthMonitor:
    """
    Main health monitoring system.
    
    Coordinates multiple component monitors and provides
    centralized health reporting and alerting.
    """
    
    def __init__(self, config_manager: ConfigManager):
        """
        Initialize the health monitor.
        
        Args:
            config_manager: Configuration manager instance
        """
        self.config_manager = config_manager
        self.logger = logging.getLogger(__name__)
        
        # Component monitors
        self.monitors: Dict[str, ComponentMonitor] = {}
        
        # Alerts
        self.alerts: List[Alert] = []
        self.alert_callbacks: List[Callable] = []
        
        # Monitoring control
        self.monitoring_active = False
        self.monitoring_task = None
        self.check_interval = 30  # seconds
        
        # Database for storing metrics
        self.db_config = config_manager.get_data_management_config().get('database', {})
        self._init_database()
        
        # Initialize monitors
        self._init_monitors()
        
        self.logger.info("Health Monitor initialized")
    
    def _init_database(self):
        """Initialize database for storing health metrics."""
        if self.db_config.get('enabled', False):
            try:
                db_url = self.db_config.get('url', 'sqlite:///health_metrics.db')
                self.engine = create_engine(db_url)
                self.SessionLocal = sessionmaker(bind=self.engine)
                
                # Create tables if they don't exist
                Base = declarative_base()
                
                class HealthMetricRecord(Base):
                    __tablename__ = 'health_metrics'
                    
                    id = Column(Integer, primary_key=True)
                    component = Column(String(100), nullable=False)
                    metric_name = Column(String(100), nullable=False)
                    value = Column(Float, nullable=False)
                    unit = Column(String(20))
                    status = Column(String(20))
                    timestamp = Column(DateTime, nullable=False)
                    metadata = Column(Text)
                
                class AlertRecord(Base):
                    __tablename__ = 'alerts'
                    
                    id = Column(String(200), primary_key=True)
                    level = Column(String(20), nullable=False)
                    component = Column(String(100), nullable=False)
                    message = Column(Text, nullable=False)
                    timestamp = Column(DateTime, nullable=False)
                    resolved = Column(Integer, default=0)
                    resolved_at = Column(DateTime)
                    metadata = Column(Text)
                
                Base.metadata.create_all(self.engine)
                self.logger.info("Health metrics database initialized")
                
            except Exception as e:
                self.logger.error(f"Failed to initialize health database: {e}")
                self.engine = None
                self.SessionLocal = None
        else:
            self.engine = None
            self.SessionLocal = None
    
    def _init_monitors(self):
        """Initialize component monitors."""
        # System monitor
        system_config = self.config_manager.get_config('monitoring').get('system', {})
        if system_config.get('enabled', True):
            self.monitors['system'] = SystemMonitor(system_config)
            self.monitors['system'].set_alert_callback(self._handle_alert)
        
        # Agent monitors
        agents = self.config_manager.get_agent_configs()
        for agent_name in agents.keys():
            agent_config = self.config_manager.get_config('monitoring').get('agents', {})
            if agent_config.get('enabled', True):
                monitor = AgentMonitor(agent_name, agent_config)
                monitor.set_alert_callback(self._handle_alert)
                self.monitors[f'agent_{agent_name}'] = monitor
        
        # Data source monitors
        data_sources = self.config_manager.get_enabled_data_sources()
        for source_name in data_sources:
            source_config = self.config_manager.get_config('monitoring').get('data_sources', {})
            if source_config.get('enabled', True):
                monitor = DataSourceMonitor(source_name, source_config)
                monitor.set_alert_callback(self._handle_alert)
                self.monitors[f'datasource_{source_name}'] = monitor
    
    def _handle_alert(self, alert: Alert):
        """Handle incoming alerts."""
        self.alerts.append(alert)
        
        # Store in database
        if self.SessionLocal:
            try:
                session = self.SessionLocal()
                # This would store the alert in the database
                session.close()
            except Exception as e:
                self.logger.error(f"Error storing alert in database: {e}")
        
        # Notify callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                self.logger.error(f"Error in alert callback: {e}")
        
        self.logger.warning(f"Alert: {alert.level.value} - {alert.component}: {alert.message}")
    
    def add_alert_callback(self, callback: Callable):
        """Add alert callback."""
        self.alert_callbacks.append(callback)
    
    def start_monitoring(self):
        """Start health monitoring."""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        self.logger.info("Health monitoring started")
    
    async def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.monitoring_active:
            try:
                await self._check_all_components()
                await asyncio.sleep(self.check_interval)
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(5)  # Short delay before retrying
    
    async def _check_all_components(self):
        """Check health of all components."""
        tasks = []
        
        for monitor in self.monitors.values():
            if monitor.enabled:
                tasks.append(monitor.check_health())
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    
    def stop_monitoring(self):
        """Stop health monitoring."""
        self.monitoring_active = False
        
        if self.monitoring_task:
            self.monitoring_task.cancel()
        
        self.logger.info("Health monitoring stopped")
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health status."""
        component_statuses = {}
        overall_status = HealthStatus.HEALTHY
        
        for name, monitor in self.monitors.items():
            status = monitor.get_status()
            component_statuses[name] = {
                'status': status.value,
                'last_check': monitor.last_check.isoformat() if monitor.last_check else None,
                'enabled': monitor.enabled
            }
            
            # Determine overall status
            if status == HealthStatus.CRITICAL:
                overall_status = HealthStatus.CRITICAL
            elif status == HealthStatus.WARNING and overall_status != HealthStatus.CRITICAL:
                overall_status = HealthStatus.WARNING
            elif status == HealthStatus.UNKNOWN and overall_status == HealthStatus.HEALTHY:
                overall_status = HealthStatus.UNKNOWN
        
        return {
            'overall_status': overall_status.value,
            'components': component_statuses,
            'timestamp': datetime.now().isoformat(),
            'monitoring_active': self.monitoring_active
        }
    
    def get_component_metrics(self, component_name: str) -> Dict[str, HealthMetric]:
        """Get metrics for a specific component."""
        if component_name in self.monitors:
            return self.monitors[component_name].get_metrics()
        return {}
    
    def get_all_metrics(self) -> Dict[str, Dict[str, HealthMetric]]:
        """Get all metrics from all components."""
        all_metrics = {}
        
        for name, monitor in self.monitors.items():
            all_metrics[name] = monitor.get_metrics()
        
        return all_metrics
    
    def get_recent_alerts(self, hours: int = 24) -> List[Alert]:
        """Get recent alerts."""
        cutoff = datetime.now() - timedelta(hours=hours)
        return [alert for alert in self.alerts if alert.timestamp > cutoff]
    
    def get_unresolved_alerts(self) -> List[Alert]:
        """Get unresolved alerts."""
        return [alert for alert in self.alerts if not alert.resolved]
    
    def resolve_alert(self, alert_id: str):
        """Mark an alert as resolved."""
        for alert in self.alerts:
            if alert.id == alert_id:
                alert.resolved = True
                alert.resolved_at = datetime.now()
                self.logger.info(f"Alert resolved: {alert_id}")
                break
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        summary = {
            'system': {},
            'agents': {},
            'data_sources': {},
            'alerts': {
                'total': len(self.alerts),
                'unresolved': len(self.get_unresolved_alerts()),
                'recent_24h': len(self.get_recent_alerts(24))
            }
        }
        
        # System metrics
        if 'system' in self.monitors:
            system_metrics = self.monitors['system'].get_metrics()
            summary['system'] = {
                'cpu_usage': system_metrics.get('cpu_usage', {}).value if 'cpu_usage' in system_metrics else 0,
                'memory_usage': system_metrics.get('memory_usage', {}).value if 'memory_usage' in system_metrics else 0,
                'disk_usage': system_metrics.get('disk_usage', {}).value if 'disk_usage' in system_metrics else 0
            }
        
        # Agent metrics
        for name, monitor in self.monitors.items():
            if name.startswith('agent_'):
                agent_name = name[6:]  # Remove 'agent_' prefix
                metrics = monitor.get_metrics()
                summary['agents'][agent_name] = {
                    'success_rate': metrics.get('task_success_rate', {}).value if 'task_success_rate' in metrics else 0,
                    'total_tasks': metrics.get('total_tasks', {}).value if 'total_tasks' in metrics else 0,
                    'error_rate': metrics.get('error_rate', {}).value if 'error_rate' in metrics else 0,
                    'avg_task_duration': metrics.get('avg_task_duration', {}).value if 'avg_task_duration' in metrics else 0,
                    'status': monitor.status.value
                }
        
        # Data source metrics
        for name, monitor in self.monitors.items():
            if name.startswith('datasource_'):
                source_name = name[11:]  # Remove 'datasource_' prefix
                metrics = monitor.get_metrics()
                summary['data_sources'][source_name] = {
                    'status': monitor.status.value,
                    'last_update': metrics.get('last_update', {}).timestamp.isoformat() if 'last_update' in metrics else None
                }
        
        return summary
    
    async def get_agent_status(self) -> Dict[str, Any]:
        """Get agent status and decisions."""
        agents = []
        active_count = 0
        
        # Get agent data from monitors
        for name, monitor in self.monitors.items():
            if name.startswith('agent_'):
                agent_name = name[6:]  # Remove 'agent_' prefix
                metrics = monitor.get_metrics()
                
                agent_data = {
                    'id': agent_name,
                    'name': agent_name,
                    'status': monitor.status.value,
                    'success_rate': metrics.get('task_success_rate', {}).value if 'task_success_rate' in metrics else 0,
                    'total_tasks': metrics.get('total_tasks', {}).value if 'total_tasks' in metrics else 0,
                    'error_count': metrics.get('error_rate', {}).value if 'error_rate' in metrics else 0,
                    'avg_duration': metrics.get('avg_task_duration', {}).value if 'avg_task_duration' in metrics else 0,
                    'last_activity': metrics.get('time_since_last_task', {}).timestamp.isoformat() if 'time_since_last_task' in metrics else None,
                    'decisions': []  # Would be populated with actual agent decisions
                }
                
                agents.append(agent_data)
                
                if monitor.status == HealthStatus.HEALTHY:
                    active_count += 1
        
        return {
            'agents': agents,
            'active_count': active_count,
            'timestamp': datetime.now().isoformat()
        }
    
    def export_metrics(self, format: str = 'json', hours: int = 24) -> str:
        """Export metrics in specified format."""
        cutoff = datetime.now() - timedelta(hours=hours)
        
        export_data = {
            'timestamp': datetime.now().isoformat(),
            'period_hours': hours,
            'system_health': self.get_system_health(),
            'performance_summary': self.get_performance_summary(),
            'recent_alerts': [{
                'id': alert.id,
                'level': alert.level.value,
                'component': alert.component,
                'message': alert.message,
                'timestamp': alert.timestamp.isoformat(),
                'resolved': alert.resolved
            } for alert in self.get_recent_alerts(hours)]
        }
        
        if format.lower() == 'json':
            return json.dumps(export_data, indent=2)
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def get_monitor(self, component_name: str) -> Optional[ComponentMonitor]:
        """Get a specific component monitor."""
        return self.monitors.get(component_name)
    
    def stop(self):
        """Stop the health monitor."""
        self.stop_monitoring()
        self.logger.info("Health Monitor stopped")


if __name__ == "__main__":
    # Example usage
    import asyncio
    from pathlib import Path
    
    config_manager = ConfigManager(Path("../config"))
    health_monitor = HealthMonitor(config_manager)
    
    async def test_health_monitor():
        # Start monitoring
        health_monitor.start_monitoring()
        
        # Wait a bit
        await asyncio.sleep(5)
        
        # Get system health
        health = health_monitor.get_system_health()
        print(f"System health: {health}")
        
        # Get performance summary
        summary = health_monitor.get_performance_summary()
        print(f"Performance summary: {summary}")
        
        # Stop monitoring
        health_monitor.stop_monitoring()
    
    # Commented out to prevent event loop conflicts when imported
    # asyncio.run(test_health_monitor())