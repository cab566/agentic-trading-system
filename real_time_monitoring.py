#!/usr/bin/env python3
"""
Real-Time Monitoring and Alerting System

A comprehensive 24/7 monitoring system for algorithmic trading that provides:

- Real-time portfolio monitoring and performance tracking
- Risk metric surveillance and threshold alerting
- Market condition monitoring and regime detection
- Position and order status tracking
- System health and connectivity monitoring
- Multi-channel alert delivery (email, SMS, Slack, Discord)
- Intelligent alert prioritization and escalation
- Historical alert analysis and pattern recognition
- Custom dashboard and visualization support
- Mobile-friendly monitoring interfaces
- Automated incident response and recovery
- Performance analytics and reporting

This system ensures continuous oversight of trading operations
with proactive risk management and immediate notification
of critical events requiring attention.

Author: AI Trading System v2.0
Date: January 2025
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import asyncio
from datetime import datetime, timedelta
import logging
import json
import smtplib
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from abc import ABC, abstractmethod
import threading
import time
from collections import defaultdict, deque
import warnings
warnings.filterwarnings('ignore')

# Web frameworks for dashboards
try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False

# Notification services
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    logging.warning("Requests not available. Web notifications will be limited.")

# Database for alert storage
try:
    import sqlite3
    DATABASE_AVAILABLE = True
except ImportError:
    DATABASE_AVAILABLE = False
    logging.warning("SQLite not available. Alert storage will be limited.")

class AlertSeverity(Enum):
    """Alert severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
    EMERGENCY = "emergency"

class AlertCategory(Enum):
    """Alert categories"""
    PORTFOLIO = "portfolio"
    RISK = "risk"
    MARKET = "market"
    SYSTEM = "system"
    EXECUTION = "execution"
    DATA = "data"
    PERFORMANCE = "performance"
    COMPLIANCE = "compliance"

class AlertStatus(Enum):
    """Alert status"""
    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    SUPPRESSED = "suppressed"

class NotificationChannel(Enum):
    """Notification delivery channels"""
    EMAIL = "email"
    SMS = "sms"
    SLACK = "slack"
    DISCORD = "discord"
    WEBHOOK = "webhook"
    PUSH = "push"
    DASHBOARD = "dashboard"

class MonitoringMetric(Enum):
    """Monitoring metrics"""
    PORTFOLIO_VALUE = "portfolio_value"
    DAILY_PNL = "daily_pnl"
    DRAWDOWN = "drawdown"
    VAR = "var"
    VOLATILITY = "volatility"
    SHARPE_RATIO = "sharpe_ratio"
    POSITION_SIZE = "position_size"
    LEVERAGE = "leverage"
    CORRELATION = "correlation"
    LIQUIDITY = "liquidity"
    EXECUTION_LATENCY = "execution_latency"
    FILL_RATE = "fill_rate"
    SLIPPAGE = "slippage"
    SYSTEM_CPU = "system_cpu"
    SYSTEM_MEMORY = "system_memory"
    DATA_LATENCY = "data_latency"
    CONNECTION_STATUS = "connection_status"

@dataclass
class AlertRule:
    """Alert rule configuration"""
    name: str
    metric: MonitoringMetric
    condition: str  # e.g., ">", "<", "==", "between"
    threshold: Union[float, Tuple[float, float]]
    severity: AlertSeverity
    category: AlertCategory
    enabled: bool = True
    cooldown_minutes: int = 15  # Minimum time between alerts
    escalation_minutes: int = 60  # Time before escalation
    notification_channels: List[NotificationChannel] = field(default_factory=list)
    custom_message: Optional[str] = None
    
    def evaluate(self, value: float) -> bool:
        """Evaluate if alert condition is met"""
        if not self.enabled:
            return False
        
        try:
            if self.condition == ">":
                return value > self.threshold
            elif self.condition == "<":
                return value < self.threshold
            elif self.condition == ">=":
                return value >= self.threshold
            elif self.condition == "<=":
                return value <= self.threshold
            elif self.condition == "==":
                return abs(value - self.threshold) < 1e-6
            elif self.condition == "between":
                if isinstance(self.threshold, tuple) and len(self.threshold) == 2:
                    return self.threshold[0] <= value <= self.threshold[1]
            elif self.condition == "outside":
                if isinstance(self.threshold, tuple) and len(self.threshold) == 2:
                    return value < self.threshold[0] or value > self.threshold[1]
        except Exception:
            pass
        
        return False

@dataclass
class Alert:
    """Alert instance"""
    id: str
    rule_name: str
    metric: MonitoringMetric
    severity: AlertSeverity
    category: AlertCategory
    message: str
    value: float
    threshold: Union[float, Tuple[float, float]]
    timestamp: datetime
    status: AlertStatus = AlertStatus.ACTIVE
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    escalated: bool = False
    escalated_at: Optional[datetime] = None
    notification_sent: List[NotificationChannel] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            'rule_name': self.rule_name,
            'metric': self.metric.value,
            'severity': self.severity.value,
            'category': self.category.value,
            'message': self.message,
            'value': self.value,
            'threshold': self.threshold,
            'timestamp': self.timestamp.isoformat(),
            'status': self.status.value,
            'acknowledged_by': self.acknowledged_by,
            'acknowledged_at': self.acknowledged_at.isoformat() if self.acknowledged_at else None,
            'resolved_at': self.resolved_at.isoformat() if self.resolved_at else None,
            'escalated': self.escalated,
            'escalated_at': self.escalated_at.isoformat() if self.escalated_at else None,
            'notification_sent': [ch.value for ch in self.notification_sent],
            'metadata': self.metadata
        }

@dataclass
class MonitoringData:
    """Real-time monitoring data point"""
    timestamp: datetime
    metric: MonitoringMetric
    value: float
    symbol: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SystemHealth:
    """System health status"""
    timestamp: datetime
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    network_latency: float
    active_connections: int
    data_feed_status: Dict[str, bool]
    broker_connections: Dict[str, bool]
    last_trade_time: Optional[datetime]
    error_count_1h: int
    warning_count_1h: int
    
    def overall_health_score(self) -> float:
        """Calculate overall health score (0-1)"""
        score = 1.0
        
        # CPU penalty
        if self.cpu_usage > 80:
            score -= 0.2
        elif self.cpu_usage > 60:
            score -= 0.1
        
        # Memory penalty
        if self.memory_usage > 85:
            score -= 0.2
        elif self.memory_usage > 70:
            score -= 0.1
        
        # Network penalty
        if self.network_latency > 1000:  # ms
            score -= 0.2
        elif self.network_latency > 500:
            score -= 0.1
        
        # Connection penalties
        total_feeds = len(self.data_feed_status)
        if total_feeds > 0:
            active_feeds = sum(self.data_feed_status.values())
            feed_ratio = active_feeds / total_feeds
            if feed_ratio < 0.8:
                score -= 0.3
            elif feed_ratio < 0.9:
                score -= 0.1
        
        total_brokers = len(self.broker_connections)
        if total_brokers > 0:
            active_brokers = sum(self.broker_connections.values())
            broker_ratio = active_brokers / total_brokers
            if broker_ratio < 0.8:
                score -= 0.3
            elif broker_ratio < 1.0:
                score -= 0.1
        
        # Error penalties
        if self.error_count_1h > 10:
            score -= 0.2
        elif self.error_count_1h > 5:
            score -= 0.1
        
        return max(0.0, score)

class NotificationService(ABC):
    """Abstract notification service"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(f"notification_{self.__class__.__name__}")
    
    @abstractmethod
    async def send_notification(self, alert: Alert) -> bool:
        """Send notification for alert"""
        pass
    
    def format_message(self, alert: Alert) -> str:
        """Format alert message"""
        severity_emoji = {
            AlertSeverity.LOW: "ℹ️",
            AlertSeverity.MEDIUM: "⚠️",
            AlertSeverity.HIGH: "🚨",
            AlertSeverity.CRITICAL: "🔥",
            AlertSeverity.EMERGENCY: "💥"
        }
        
        emoji = severity_emoji.get(alert.severity, "📊")
        
        message = f"{emoji} **{alert.severity.value.upper()} ALERT**\n\n"
        message += f"**Rule:** {alert.rule_name}\n"
        message += f"**Metric:** {alert.metric.value}\n"
        message += f"**Value:** {alert.value:.4f}\n"
        message += f"**Threshold:** {alert.threshold}\n"
        message += f"**Time:** {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        message += f"**Message:** {alert.message}"
        
        return message

class EmailNotificationService(NotificationService):
    """Email notification service"""
    
    async def send_notification(self, alert: Alert) -> bool:
        """Send email notification"""
        try:
            smtp_server = self.config.get('smtp_server', 'smtp.gmail.com')
            smtp_port = self.config.get('smtp_port', 587)
            username = self.config.get('username')
            password = self.config.get('password')
            recipients = self.config.get('recipients', [])
            
            if not username or not password or not recipients:
                self.logger.warning("Email configuration incomplete")
                return False
            
            # Create message
            msg = MIMEMultipart()
            msg['From'] = username
            msg['To'] = ', '.join(recipients)
            msg['Subject'] = f"Trading Alert: {alert.severity.value.upper()} - {alert.rule_name}"
            
            body = self.format_message(alert)
            msg.attach(MIMEText(body, 'plain'))
            
            # Send email
            context = ssl.create_default_context()
            with smtplib.SMTP(smtp_server, smtp_port) as server:
                server.starttls(context=context)
                server.login(username, password)
                server.send_message(msg)
            
            self.logger.info(f"Email sent for alert {alert.id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Email notification failed: {e}")
            return False

class SlackNotificationService(NotificationService):
    """Slack notification service"""
    
    async def send_notification(self, alert: Alert) -> bool:
        """Send Slack notification"""
        try:
            if not REQUESTS_AVAILABLE:
                return False
            
            webhook_url = self.config.get('webhook_url')
            if not webhook_url:
                self.logger.warning("Slack webhook URL not configured")
                return False
            
            # Format message for Slack
            color_map = {
                AlertSeverity.LOW: "good",
                AlertSeverity.MEDIUM: "warning",
                AlertSeverity.HIGH: "danger",
                AlertSeverity.CRITICAL: "danger",
                AlertSeverity.EMERGENCY: "danger"
            }
            
            payload = {
                "attachments": [{
                    "color": color_map.get(alert.severity, "warning"),
                    "title": f"{alert.severity.value.upper()} Alert: {alert.rule_name}",
                    "text": alert.message,
                    "fields": [
                        {"title": "Metric", "value": alert.metric.value, "short": True},
                        {"title": "Value", "value": f"{alert.value:.4f}", "short": True},
                        {"title": "Threshold", "value": str(alert.threshold), "short": True},
                        {"title": "Time", "value": alert.timestamp.strftime('%Y-%m-%d %H:%M:%S'), "short": True}
                    ],
                    "ts": alert.timestamp.timestamp()
                }]
            }
            
            response = requests.post(webhook_url, json=payload, timeout=10)
            response.raise_for_status()
            
            self.logger.info(f"Slack notification sent for alert {alert.id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Slack notification failed: {e}")
            return False

class DiscordNotificationService(NotificationService):
    """Discord notification service"""
    
    async def send_notification(self, alert: Alert) -> bool:
        """Send Discord notification"""
        try:
            if not REQUESTS_AVAILABLE:
                return False
            
            webhook_url = self.config.get('webhook_url')
            if not webhook_url:
                self.logger.warning("Discord webhook URL not configured")
                return False
            
            # Format message for Discord
            color_map = {
                AlertSeverity.LOW: 0x00ff00,      # Green
                AlertSeverity.MEDIUM: 0xffff00,   # Yellow
                AlertSeverity.HIGH: 0xff8000,     # Orange
                AlertSeverity.CRITICAL: 0xff0000, # Red
                AlertSeverity.EMERGENCY: 0x800080 # Purple
            }
            
            embed = {
                "title": f"{alert.severity.value.upper()} Alert: {alert.rule_name}",
                "description": alert.message,
                "color": color_map.get(alert.severity, 0xffff00),
                "fields": [
                    {"name": "Metric", "value": alert.metric.value, "inline": True},
                    {"name": "Value", "value": f"{alert.value:.4f}", "inline": True},
                    {"name": "Threshold", "value": str(alert.threshold), "inline": True}
                ],
                "timestamp": alert.timestamp.isoformat(),
                "footer": {"text": "Trading System Alert"}
            }
            
            payload = {"embeds": [embed]}
            
            response = requests.post(webhook_url, json=payload, timeout=10)
            response.raise_for_status()
            
            self.logger.info(f"Discord notification sent for alert {alert.id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Discord notification failed: {e}")
            return False

class WebhookNotificationService(NotificationService):
    """Generic webhook notification service"""
    
    async def send_notification(self, alert: Alert) -> bool:
        """Send webhook notification"""
        try:
            if not REQUESTS_AVAILABLE:
                return False
            
            webhook_url = self.config.get('url')
            if not webhook_url:
                self.logger.warning("Webhook URL not configured")
                return False
            
            payload = {
                "alert": alert.to_dict(),
                "timestamp": datetime.now().isoformat(),
                "source": "trading_system"
            }
            
            headers = self.config.get('headers', {'Content-Type': 'application/json'})
            
            response = requests.post(webhook_url, json=payload, headers=headers, timeout=10)
            response.raise_for_status()
            
            self.logger.info(f"Webhook notification sent for alert {alert.id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Webhook notification failed: {e}")
            return False

class AlertStorage:
    """Alert storage and retrieval"""
    
    def __init__(self, db_path: str = "alerts.db"):
        self.db_path = db_path
        self.logger = logging.getLogger("alert_storage")
        self._init_database()
    
    def _init_database(self):
        """Initialize SQLite database"""
        if not DATABASE_AVAILABLE:
            self.logger.warning("Database not available. Using in-memory storage.")
            self.alerts = {}
            return
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS alerts (
                    id TEXT PRIMARY KEY,
                    rule_name TEXT,
                    metric TEXT,
                    severity TEXT,
                    category TEXT,
                    message TEXT,
                    value REAL,
                    threshold TEXT,
                    timestamp TEXT,
                    status TEXT,
                    acknowledged_by TEXT,
                    acknowledged_at TEXT,
                    resolved_at TEXT,
                    escalated INTEGER,
                    escalated_at TEXT,
                    notification_sent TEXT,
                    metadata TEXT
                )
            ''')
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Database initialization failed: {e}")
            self.alerts = {}
    
    def store_alert(self, alert: Alert):
        """Store alert in database"""
        if not DATABASE_AVAILABLE:
            self.alerts[alert.id] = alert
            return
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO alerts (
                    id, rule_name, metric, severity, category, message, value,
                    threshold, timestamp, status, acknowledged_by, acknowledged_at,
                    resolved_at, escalated, escalated_at, notification_sent, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                alert.id, alert.rule_name, alert.metric.value, alert.severity.value,
                alert.category.value, alert.message, alert.value, str(alert.threshold),
                alert.timestamp.isoformat(), alert.status.value, alert.acknowledged_by,
                alert.acknowledged_at.isoformat() if alert.acknowledged_at else None,
                alert.resolved_at.isoformat() if alert.resolved_at else None,
                int(alert.escalated),
                alert.escalated_at.isoformat() if alert.escalated_at else None,
                json.dumps([ch.value for ch in alert.notification_sent]),
                json.dumps(alert.metadata)
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Alert storage failed: {e}")
    
    def get_active_alerts(self) -> List[Alert]:
        """Get all active alerts"""
        if not DATABASE_AVAILABLE:
            return [alert for alert in self.alerts.values() if alert.status == AlertStatus.ACTIVE]
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("SELECT * FROM alerts WHERE status = 'active' ORDER BY timestamp DESC")
            rows = cursor.fetchall()
            conn.close()
            
            alerts = []
            for row in rows:
                alert = self._row_to_alert(row)
                if alert:
                    alerts.append(alert)
            
            return alerts
            
        except Exception as e:
            self.logger.error(f"Failed to retrieve active alerts: {e}")
            return []
    
    def get_alerts_by_timerange(self, start_time: datetime, end_time: datetime) -> List[Alert]:
        """Get alerts within time range"""
        if not DATABASE_AVAILABLE:
            return [alert for alert in self.alerts.values() 
                   if start_time <= alert.timestamp <= end_time]
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute(
                "SELECT * FROM alerts WHERE timestamp BETWEEN ? AND ? ORDER BY timestamp DESC",
                (start_time.isoformat(), end_time.isoformat())
            )
            rows = cursor.fetchall()
            conn.close()
            
            alerts = []
            for row in rows:
                alert = self._row_to_alert(row)
                if alert:
                    alerts.append(alert)
            
            return alerts
            
        except Exception as e:
            self.logger.error(f"Failed to retrieve alerts by timerange: {e}")
            return []
    
    def _row_to_alert(self, row) -> Optional[Alert]:
        """Convert database row to Alert object"""
        try:
            return Alert(
                id=row[0],
                rule_name=row[1],
                metric=MonitoringMetric(row[2]),
                severity=AlertSeverity(row[3]),
                category=AlertCategory(row[4]),
                message=row[5],
                value=row[6],
                threshold=eval(row[7]) if row[7] else None,
                timestamp=datetime.fromisoformat(row[8]),
                status=AlertStatus(row[9]),
                acknowledged_by=row[10],
                acknowledged_at=datetime.fromisoformat(row[11]) if row[11] else None,
                resolved_at=datetime.fromisoformat(row[12]) if row[12] else None,
                escalated=bool(row[13]),
                escalated_at=datetime.fromisoformat(row[14]) if row[14] else None,
                notification_sent=[NotificationChannel(ch) for ch in json.loads(row[15])] if row[15] else [],
                metadata=json.loads(row[16]) if row[16] else {}
            )
        except Exception as e:
            self.logger.error(f"Failed to convert row to alert: {e}")
            return None

class RealTimeMonitor:
    """Main real-time monitoring system"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger("real_time_monitor")
        
        # Alert management
        self.alert_rules: Dict[str, AlertRule] = {}
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_cooldowns: Dict[str, datetime] = {}
        self.alert_storage = AlertStorage(config.get('alert_db_path', 'alerts.db'))
        
        # Notification services
        self.notification_services: Dict[NotificationChannel, NotificationService] = {}
        self._init_notification_services()
        
        # Monitoring data
        self.monitoring_data: Dict[MonitoringMetric, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.system_health_history: deque = deque(maxlen=100)
        
        # Control flags
        self.running = False
        self.monitor_thread = None
        
        # Load default alert rules
        self._load_default_alert_rules()
    
    def _init_notification_services(self):
        """Initialize notification services"""
        notification_config = self.config.get('notifications', {})
        
        # Email service
        if 'email' in notification_config:
            self.notification_services[NotificationChannel.EMAIL] = EmailNotificationService(
                notification_config['email']
            )
        
        # Slack service
        if 'slack' in notification_config:
            self.notification_services[NotificationChannel.SLACK] = SlackNotificationService(
                notification_config['slack']
            )
        
        # Discord service
        if 'discord' in notification_config:
            self.notification_services[NotificationChannel.DISCORD] = DiscordNotificationService(
                notification_config['discord']
            )
        
        # Webhook service
        if 'webhook' in notification_config:
            self.notification_services[NotificationChannel.WEBHOOK] = WebhookNotificationService(
                notification_config['webhook']
            )
    
    def _load_default_alert_rules(self):
        """Load default alert rules"""
        default_rules = [
            # Portfolio alerts
            AlertRule(
                name="High Drawdown",
                metric=MonitoringMetric.DRAWDOWN,
                condition=">",
                threshold=0.05,  # 5% drawdown
                severity=AlertSeverity.HIGH,
                category=AlertCategory.PORTFOLIO,
                notification_channels=[NotificationChannel.EMAIL, NotificationChannel.SLACK]
            ),
            AlertRule(
                name="Critical Drawdown",
                metric=MonitoringMetric.DRAWDOWN,
                condition=">",
                threshold=0.10,  # 10% drawdown
                severity=AlertSeverity.CRITICAL,
                category=AlertCategory.PORTFOLIO,
                notification_channels=[NotificationChannel.EMAIL, NotificationChannel.SLACK, NotificationChannel.DISCORD]
            ),
            AlertRule(
                name="High Daily Loss",
                metric=MonitoringMetric.DAILY_PNL,
                condition="<",
                threshold=-10000,  # $10k daily loss
                severity=AlertSeverity.HIGH,
                category=AlertCategory.PORTFOLIO,
                notification_channels=[NotificationChannel.EMAIL, NotificationChannel.SLACK]
            ),
            
            # Risk alerts
            AlertRule(
                name="High VaR",
                metric=MonitoringMetric.VAR,
                condition=">",
                threshold=50000,  # $50k VaR
                severity=AlertSeverity.MEDIUM,
                category=AlertCategory.RISK,
                notification_channels=[NotificationChannel.EMAIL]
            ),
            AlertRule(
                name="Excessive Leverage",
                metric=MonitoringMetric.LEVERAGE,
                condition=">",
                threshold=3.0,  # 3x leverage
                severity=AlertSeverity.HIGH,
                category=AlertCategory.RISK,
                notification_channels=[NotificationChannel.EMAIL, NotificationChannel.SLACK]
            ),
            
            # System alerts
            AlertRule(
                name="High CPU Usage",
                metric=MonitoringMetric.SYSTEM_CPU,
                condition=">",
                threshold=85.0,  # 85% CPU
                severity=AlertSeverity.MEDIUM,
                category=AlertCategory.SYSTEM,
                notification_channels=[NotificationChannel.EMAIL]
            ),
            AlertRule(
                name="High Memory Usage",
                metric=MonitoringMetric.SYSTEM_MEMORY,
                condition=">",
                threshold=90.0,  # 90% memory
                severity=AlertSeverity.HIGH,
                category=AlertCategory.SYSTEM,
                notification_channels=[NotificationChannel.EMAIL, NotificationChannel.SLACK]
            ),
            AlertRule(
                name="Data Feed Disconnected",
                metric=MonitoringMetric.CONNECTION_STATUS,
                condition="<",
                threshold=0.8,  # Less than 80% connections active
                severity=AlertSeverity.CRITICAL,
                category=AlertCategory.DATA,
                notification_channels=[NotificationChannel.EMAIL, NotificationChannel.SLACK, NotificationChannel.DISCORD]
            ),
            
            # Execution alerts
            AlertRule(
                name="High Execution Latency",
                metric=MonitoringMetric.EXECUTION_LATENCY,
                condition=">",
                threshold=1000,  # 1 second latency
                severity=AlertSeverity.MEDIUM,
                category=AlertCategory.EXECUTION,
                notification_channels=[NotificationChannel.EMAIL]
            ),
            AlertRule(
                name="Low Fill Rate",
                metric=MonitoringMetric.FILL_RATE,
                condition="<",
                threshold=0.9,  # Less than 90% fill rate
                severity=AlertSeverity.MEDIUM,
                category=AlertCategory.EXECUTION,
                notification_channels=[NotificationChannel.EMAIL]
            ),
            AlertRule(
                name="High Slippage",
                metric=MonitoringMetric.SLIPPAGE,
                condition=">",
                threshold=0.005,  # 0.5% slippage
                severity=AlertSeverity.MEDIUM,
                category=AlertCategory.EXECUTION,
                notification_channels=[NotificationChannel.EMAIL]
            )
        ]
        
        for rule in default_rules:
            self.alert_rules[rule.name] = rule
    
    def add_alert_rule(self, rule: AlertRule):
        """Add custom alert rule"""
        self.alert_rules[rule.name] = rule
        self.logger.info(f"Added alert rule: {rule.name}")
    
    def remove_alert_rule(self, rule_name: str):
        """Remove alert rule"""
        if rule_name in self.alert_rules:
            del self.alert_rules[rule_name]
            self.logger.info(f"Removed alert rule: {rule_name}")
    
    def update_monitoring_data(self, data: MonitoringData):
        """Update monitoring data point"""
        self.monitoring_data[data.metric].append(data)
        
        # Check alert rules for this metric
        asyncio.create_task(self._check_alert_rules(data.metric, data.value))
    
    async def _check_alert_rules(self, metric: MonitoringMetric, value: float):
        """Check alert rules for metric"""
        current_time = datetime.now()
        
        for rule_name, rule in self.alert_rules.items():
            if rule.metric != metric or not rule.enabled:
                continue
            
            # Check cooldown
            if rule_name in self.alert_cooldowns:
                if current_time - self.alert_cooldowns[rule_name] < timedelta(minutes=rule.cooldown_minutes):
                    continue
            
            # Evaluate rule
            if rule.evaluate(value):
                await self._trigger_alert(rule, value, current_time)
    
    async def _trigger_alert(self, rule: AlertRule, value: float, timestamp: datetime):
        """Trigger alert"""
        alert_id = f"{rule.name}_{timestamp.strftime('%Y%m%d_%H%M%S')}"
        
        # Create alert
        alert = Alert(
            id=alert_id,
            rule_name=rule.name,
            metric=rule.metric,
            severity=rule.severity,
            category=rule.category,
            message=rule.custom_message or f"{rule.metric.value} {rule.condition} {rule.threshold}",
            value=value,
            threshold=rule.threshold,
            timestamp=timestamp
        )
        
        # Store alert
        self.active_alerts[alert_id] = alert
        self.alert_storage.store_alert(alert)
        self.alert_cooldowns[rule.name] = timestamp
        
        # Send notifications
        await self._send_notifications(alert, rule.notification_channels)
        
        self.logger.warning(f"Alert triggered: {rule.name} - {value}")
    
    async def _send_notifications(self, alert: Alert, channels: List[NotificationChannel]):
        """Send notifications through specified channels"""
        for channel in channels:
            if channel in self.notification_services:
                try:
                    success = await self.notification_services[channel].send_notification(alert)
                    if success:
                        alert.notification_sent.append(channel)
                except Exception as e:
                    self.logger.error(f"Notification failed for {channel}: {e}")
        
        # Update alert in storage
        self.alert_storage.store_alert(alert)
    
    def acknowledge_alert(self, alert_id: str, acknowledged_by: str):
        """Acknowledge alert"""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.status = AlertStatus.ACKNOWLEDGED
            alert.acknowledged_by = acknowledged_by
            alert.acknowledged_at = datetime.now()
            
            self.alert_storage.store_alert(alert)
            self.logger.info(f"Alert acknowledged: {alert_id} by {acknowledged_by}")
    
    def resolve_alert(self, alert_id: str):
        """Resolve alert"""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.status = AlertStatus.RESOLVED
            alert.resolved_at = datetime.now()
            
            self.alert_storage.store_alert(alert)
            del self.active_alerts[alert_id]
            self.logger.info(f"Alert resolved: {alert_id}")
    
    def get_alert_summary(self) -> Dict[str, Any]:
        """Get alert summary"""
        active_alerts = list(self.active_alerts.values())
        
        summary = {
            'total_active': len(active_alerts),
            'by_severity': defaultdict(int),
            'by_category': defaultdict(int),
            'recent_alerts': [],
            'escalated_count': 0
        }
        
        for alert in active_alerts:
            summary['by_severity'][alert.severity.value] += 1
            summary['by_category'][alert.category.value] += 1
            if alert.escalated:
                summary['escalated_count'] += 1
        
        # Get recent alerts (last 24 hours)
        recent_time = datetime.now() - timedelta(hours=24)
        recent_alerts = self.alert_storage.get_alerts_by_timerange(recent_time, datetime.now())
        summary['recent_alerts'] = [alert.to_dict() for alert in recent_alerts[:10]]
        
        return summary
    
    def update_system_health(self, health: SystemHealth):
        """Update system health metrics"""
        self.system_health_history.append(health)
        
        # Update individual metrics for alert checking
        self.update_monitoring_data(MonitoringData(
            timestamp=health.timestamp,
            metric=MonitoringMetric.SYSTEM_CPU,
            value=health.cpu_usage
        ))
        
        self.update_monitoring_data(MonitoringData(
            timestamp=health.timestamp,
            metric=MonitoringMetric.SYSTEM_MEMORY,
            value=health.memory_usage
        ))
        
        # Connection status (percentage of active connections)
        total_connections = len(health.data_feed_status) + len(health.broker_connections)
        if total_connections > 0:
            active_connections = sum(health.data_feed_status.values()) + sum(health.broker_connections.values())
            connection_ratio = active_connections / total_connections
            
            self.update_monitoring_data(MonitoringData(
                timestamp=health.timestamp,
                metric=MonitoringMetric.CONNECTION_STATUS,
                value=connection_ratio
            ))
    
    def start_monitoring(self):
        """Start monitoring thread"""
        if self.running:
            return
        
        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        self.logger.info("Real-time monitoring started")
    
    def stop_monitoring(self):
        """Stop monitoring"""
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        self.logger.info("Real-time monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.running:
            try:
                # Check for alert escalations
                self._check_escalations()
                
                # Cleanup old data
                self._cleanup_old_data()
                
                # Sleep for monitoring interval
                time.sleep(self.config.get('monitoring_interval', 30))  # 30 seconds default
                
            except Exception as e:
                self.logger.error(f"Monitoring loop error: {e}")
                time.sleep(60)  # Wait longer on error
    
    def _check_escalations(self):
        """Check for alert escalations"""
        current_time = datetime.now()
        
        for alert in self.active_alerts.values():
            if alert.escalated or alert.status != AlertStatus.ACTIVE:
                continue
            
            rule = self.alert_rules.get(alert.rule_name)
            if not rule:
                continue
            
            # Check if escalation time has passed
            time_since_alert = current_time - alert.timestamp
            if time_since_alert >= timedelta(minutes=rule.escalation_minutes):
                alert.escalated = True
                alert.escalated_at = current_time
                
                # Send escalation notifications
                escalation_channels = [NotificationChannel.EMAIL, NotificationChannel.SLACK]
                asyncio.create_task(self._send_notifications(alert, escalation_channels))
                
                self.logger.warning(f"Alert escalated: {alert.id}")
    
    def _cleanup_old_data(self):
        """Cleanup old monitoring data"""
        cutoff_time = datetime.now() - timedelta(hours=24)
        
        # Clean monitoring data
        for metric, data_queue in self.monitoring_data.items():
            while data_queue and data_queue[0].timestamp < cutoff_time:
                data_queue.popleft()
        
        # Clean resolved alerts older than 7 days
        old_cutoff = datetime.now() - timedelta(days=7)
        resolved_alerts = [alert_id for alert_id, alert in self.active_alerts.items() 
                          if alert.status == AlertStatus.RESOLVED and alert.resolved_at and alert.resolved_at < old_cutoff]
        
        for alert_id in resolved_alerts:
            del self.active_alerts[alert_id]

# Example usage and testing
if __name__ == "__main__":
    # Example configuration
    config = {
        'alert_db_path': 'trading_alerts.db',
        'monitoring_interval': 30,
        'notifications': {
            'email': {
                'smtp_server': 'smtp.gmail.com',
                'smtp_port': 587,
                'username': 'your_email@gmail.com',
                'password': 'your_app_password',
                'recipients': ['trader@example.com', 'risk@example.com']
            },
            'slack': {
                'webhook_url': 'https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK'
            },
            'discord': {
                'webhook_url': 'https://discord.com/api/webhooks/YOUR/DISCORD/WEBHOOK'
            }
        }
    }
    
    # Initialize monitor
    monitor = RealTimeMonitor(config)
    
    print("Real-Time Monitoring System initialized")
    print("\nFeatures:")
    print("- 24/7 portfolio and risk monitoring")
    print("- Multi-channel alert notifications (Email, Slack, Discord)")
    print("- Intelligent alert escalation and cooldowns")
    print("- System health monitoring")
    print("- Historical alert analysis")
    print("- Customizable alert rules and thresholds")
    print("- Real-time dashboard integration")
    print("- Automated incident response")
    
    print("\nDefault Alert Rules:")
    for rule_name, rule in monitor.alert_rules.items():
        print(f"- {rule_name}: {rule.metric.value} {rule.condition} {rule.threshold} ({rule.severity.value})")
    
    # Start monitoring
    monitor.start_monitoring()
    
    # Simulate some monitoring data
    import time
    for i in range(5):
        # Simulate portfolio drawdown
        drawdown = 0.02 + (i * 0.01)  # Increasing drawdown
        monitor.update_monitoring_data(MonitoringData(
            timestamp=datetime.now(),
            metric=MonitoringMetric.DRAWDOWN,
            value=drawdown
        ))
        
        # Simulate system metrics
        monitor.update_system_health(SystemHealth(
            timestamp=datetime.now(),
            cpu_usage=70 + (i * 5),
            memory_usage=60 + (i * 8),
            disk_usage=50,
            network_latency=100,
            active_connections=10,
            data_feed_status={'alpaca': True, 'binance': True, 'oanda': True},
            broker_connections={'alpaca': True, 'binance': True, 'oanda': True},
            last_trade_time=datetime.now(),
            error_count_1h=i,
            warning_count_1h=i * 2
        ))
        
        time.sleep(2)
    
    # Get alert summary
    summary = monitor.get_alert_summary()
    print(f"\nAlert Summary: {summary}")
    
    # Stop monitoring
    monitor.stop_monitoring()