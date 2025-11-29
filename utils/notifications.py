#!/usr/bin/env python3
"""
Notification Manager for Trading System

Provides notification functionality for alerts, errors, and system events.
"""

import logging
from typing import Any, Optional, Dict, List
from datetime import datetime
from enum import Enum


class NotificationLevel(Enum):
    """Notification severity levels."""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class NotificationManager:
    """Simple notification manager."""
    
    def __init__(self, config_manager=None):
        self.config_manager = config_manager
        self.logger = logging.getLogger(__name__)
        self._notifications: List[Dict[str, Any]] = []
        self.max_notifications = 1000
        
    def send_notification(
        self,
        message: str,
        level: NotificationLevel = NotificationLevel.INFO,
        category: str = "general",
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Send a notification."""
        notification = {
            'timestamp': datetime.now(),
            'message': message,
            'level': level.value,
            'category': category,
            'metadata': metadata or {}
        }
        
        # Add to internal storage
        self._notifications.append(notification)
        
        # Keep only recent notifications
        if len(self._notifications) > self.max_notifications:
            self._notifications = self._notifications[-self.max_notifications:]
        
        # Log the notification
        log_level = getattr(logging, level.value.upper(), logging.INFO)
        self.logger.log(log_level, f"[{category}] {message}", extra=metadata or {})
    
    def send_alert(self, message: str, title: Optional[str] = None, severity: Optional[str] = None, alert_type: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Send an alert notification."""
        # Combine title and message if title is provided
        full_message = f"{title}: {message}" if title else message
        
        # Map severity to notification level
        level = NotificationLevel.WARNING
        if severity:
            severity_map = {
                'info': NotificationLevel.INFO,
                'warning': NotificationLevel.WARNING,
                'error': NotificationLevel.ERROR,
                'critical': NotificationLevel.CRITICAL,
                'medium': NotificationLevel.WARNING,
                'high': NotificationLevel.ERROR
            }
            level = severity_map.get(severity.lower(), NotificationLevel.WARNING)
        
        # Add alert_type to metadata if provided
        if metadata is None:
            metadata = {}
        if alert_type:
            metadata['alert_type'] = alert_type
            
        self.send_notification(full_message, level, "alert", metadata)
    
    def send_error(self, message: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Send an error notification."""
        self.send_notification(message, NotificationLevel.ERROR, "error", metadata)
    
    def send_trade_notification(
        self,
        message: str,
        trade_data: Optional[Dict[str, Any]] = None
    ) -> None:
        """Send a trade-related notification."""
        self.send_notification(message, NotificationLevel.INFO, "trade", trade_data)
    
    def get_recent_notifications(
        self,
        limit: int = 50,
        level: Optional[NotificationLevel] = None,
        category: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get recent notifications with optional filtering."""
        notifications = self._notifications.copy()
        
        # Filter by level if specified
        if level:
            notifications = [
                n for n in notifications if n['level'] == level.value
            ]
        
        # Filter by category if specified
        if category:
            notifications = [
                n for n in notifications if n['category'] == category
            ]
        
        # Return most recent first, limited by count
        return notifications[-limit:]
    
    def clear_notifications(self) -> None:
        """Clear all stored notifications."""
        self._notifications.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get notification statistics."""
        level_counts = {}
        category_counts = {}
        
        for notification in self._notifications:
            level = notification['level']
            category = notification['category']
            
            level_counts[level] = level_counts.get(level, 0) + 1
            category_counts[category] = category_counts.get(category, 0) + 1
        
        return {
            'total_notifications': len(self._notifications),
            'level_counts': level_counts,
            'category_counts': category_counts
        }