#!/usr/bin/env python3
"""
24/7 Trading Scheduler for CrewAI Trading System

Intelligent scheduler that manages trading across multiple asset classes:
- Cryptocurrencies: 24/7/365 trading
- Forex: 24/5 trading (Sunday 5 PM EST - Friday 5 PM EST)
- Stocks/Options: Traditional market hours
- International Markets: Global market sessions
"""

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import pytz


class AssetClass(Enum):
    """Asset class enumeration."""
    CRYPTO = "crypto"
    FOREX = "forex"
    STOCKS = "stocks"
    OPTIONS = "options"
    COMMODITIES = "commodities"
    BONDS = "bonds"


class MarketSession(Enum):
    """Market session enumeration."""
    PRE_MARKET = "pre_market"
    REGULAR = "regular"
    AFTER_HOURS = "after_hours"
    CLOSED = "closed"


class TradingMode(Enum):
    """Trading mode enumeration."""
    CONSERVATIVE = "conservative"
    NORMAL = "normal"
    AGGRESSIVE = "aggressive"
    MAINTENANCE = "maintenance"


@dataclass
class TradingSession:
    """Trading session configuration."""
    name: str
    asset_class: AssetClass
    timezone: str
    open_time: str  # HH:MM format
    close_time: str  # HH:MM format
    days: List[int]  # 0=Monday, 6=Sunday
    priority: int = 1  # Higher number = higher priority
    enabled: bool = True


@dataclass
class ScheduledTask:
    """Scheduled trading task."""
    name: str
    asset_class: AssetClass
    callback: Callable
    interval: int  # seconds
    next_run: datetime
    enabled: bool = True
    session_dependent: bool = True


class TradingScheduler24_7:
    """24/7 Trading scheduler with multi-asset support."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.running = False
        self.tasks = {}
        self.sessions = self._initialize_sessions()
        self.timezone_cache = {}
        
        # Task intervals (seconds)
        self.intervals = {
            AssetClass.CRYPTO: config.get('crypto_interval', 60),      # 1 minute
            AssetClass.FOREX: config.get('forex_interval', 300),       # 5 minutes
            AssetClass.STOCKS: config.get('stocks_interval', 60),      # 1 minute
            AssetClass.OPTIONS: config.get('options_interval', 300),   # 5 minutes
        }
    
    def _initialize_sessions(self) -> Dict[str, TradingSession]:
        """Initialize trading sessions for different markets."""
        sessions = {}
        
        # Cryptocurrency (24/7/365)
        sessions['crypto_global'] = TradingSession(
            name='Crypto Global',
            asset_class=AssetClass.CRYPTO,
            timezone='UTC',
            open_time='00:00',
            close_time='23:59',
            days=list(range(7)),  # All days
            priority=1
        )
        
        # Forex Sessions (24/5)
        sessions['forex_sydney'] = TradingSession(
            name='Forex Sydney',
            asset_class=AssetClass.FOREX,
            timezone='Australia/Sydney',
            open_time='07:00',
            close_time='16:00',
            days=[0, 1, 2, 3, 4],  # Monday-Friday
            priority=2
        )
        
        sessions['forex_tokyo'] = TradingSession(
            name='Forex Tokyo',
            asset_class=AssetClass.FOREX,
            timezone='Asia/Tokyo',
            open_time='09:00',
            close_time='18:00',
            days=[0, 1, 2, 3, 4],  # Monday-Friday
            priority=3
        )
        
        sessions['forex_london'] = TradingSession(
            name='Forex London',
            asset_class=AssetClass.FOREX,
            timezone='Europe/London',
            open_time='08:00',
            close_time='17:00',
            days=[0, 1, 2, 3, 4],  # Monday-Friday
            priority=4
        )
        
        sessions['forex_new_york'] = TradingSession(
            name='Forex New York',
            asset_class=AssetClass.FOREX,
            timezone='America/New_York',
            open_time='08:00',
            close_time='17:00',
            days=[0, 1, 2, 3, 4],  # Monday-Friday
            priority=5
        )
        
        # US Stock Market
        sessions['us_stocks_pre'] = TradingSession(
            name='US Stocks Pre-Market',
            asset_class=AssetClass.STOCKS,
            timezone='America/New_York',
            open_time='04:00',
            close_time='09:30',
            days=[0, 1, 2, 3, 4],  # Monday-Friday
            priority=2
        )
        
        sessions['us_stocks_regular'] = TradingSession(
            name='US Stocks Regular',
            asset_class=AssetClass.STOCKS,
            timezone='America/New_York',
            open_time='09:30',
            close_time='16:00',
            days=[0, 1, 2, 3, 4],  # Monday-Friday
            priority=5
        )
        
        sessions['us_stocks_after'] = TradingSession(
            name='US Stocks After Hours',
            asset_class=AssetClass.STOCKS,
            timezone='America/New_York',
            open_time='16:00',
            close_time='20:00',
            days=[0, 1, 2, 3, 4],  # Monday-Friday
            priority=3
        )
        
        # European Markets
        sessions['london_stocks'] = TradingSession(
            name='London Stock Exchange',
            asset_class=AssetClass.STOCKS,
            timezone='Europe/London',
            open_time='08:00',
            close_time='16:30',
            days=[0, 1, 2, 3, 4],  # Monday-Friday
            priority=4
        )
        
        sessions['frankfurt_stocks'] = TradingSession(
            name='Frankfurt Stock Exchange',
            asset_class=AssetClass.STOCKS,
            timezone='Europe/Berlin',
            open_time='09:00',
            close_time='17:30',
            days=[0, 1, 2, 3, 4],  # Monday-Friday
            priority=4
        )
        
        # Asian Markets
        sessions['tokyo_stocks'] = TradingSession(
            name='Tokyo Stock Exchange',
            asset_class=AssetClass.STOCKS,
            timezone='Asia/Tokyo',
            open_time='09:00',
            close_time='15:00',
            days=[0, 1, 2, 3, 4],  # Monday-Friday
            priority=4
        )
        
        sessions['hong_kong_stocks'] = TradingSession(
            name='Hong Kong Stock Exchange',
            asset_class=AssetClass.STOCKS,
            timezone='Asia/Hong_Kong',
            open_time='09:30',
            close_time='16:00',
            days=[0, 1, 2, 3, 4],  # Monday-Friday
            priority=4
        )
        
        return sessions
    
    def add_task(self, name: str, asset_class: AssetClass, callback: Callable,
                 interval: Optional[int] = None, session_dependent: bool = True):
        """Add a scheduled trading task."""
        if interval is None:
            interval = self.intervals.get(asset_class, 300)
        
        task = ScheduledTask(
            name=name,
            asset_class=asset_class,
            callback=callback,
            interval=interval,
            next_run=datetime.now() + timedelta(seconds=interval),
            session_dependent=session_dependent
        )
        
        self.tasks[name] = task
        self.logger.info(f"Added task: {name} for {asset_class.value} (interval: {interval}s)")
    
    def remove_task(self, name: str):
        """Remove a scheduled task."""
        if name in self.tasks:
            del self.tasks[name]
            self.logger.info(f"Removed task: {name}")
    
    def is_session_active(self, session: TradingSession) -> bool:
        """Check if a trading session is currently active."""
        try:
            # Get timezone
            if session.timezone not in self.timezone_cache:
                self.timezone_cache[session.timezone] = pytz.timezone(session.timezone)
            
            tz = self.timezone_cache[session.timezone]
            now = datetime.now(tz)
            
            # Check if today is a trading day
            if now.weekday() not in session.days:
                return False
            
            # Parse session times
            open_hour, open_minute = map(int, session.open_time.split(':'))
            close_hour, close_minute = map(int, session.close_time.split(':'))
            
            # Create time objects for today
            session_open = now.replace(hour=open_hour, minute=open_minute, second=0, microsecond=0)
            session_close = now.replace(hour=close_hour, minute=close_minute, second=0, microsecond=0)
            
            # Handle overnight sessions (close time < open time)
            if session_close <= session_open:
                session_close += timedelta(days=1)
            
            # Check if current time is within session
            return session_open <= now <= session_close
            
        except Exception as e:
            self.logger.error(f"Error checking session {session.name}: {e}")
            return False
    
    def get_active_sessions(self, asset_class: Optional[AssetClass] = None) -> List[TradingSession]:
        """Get currently active trading sessions."""
        active_sessions = []
        
        for session in self.sessions.values():
            if not session.enabled:
                continue
            
            if asset_class and session.asset_class != asset_class:
                continue
            
            if self.is_session_active(session):
                active_sessions.append(session)
        
        # Sort by priority (higher priority first)
        active_sessions.sort(key=lambda s: s.priority, reverse=True)
        return active_sessions
    
    def get_market_status(self, asset_class: AssetClass) -> Dict[str, Any]:
        """Get comprehensive market status for an asset class."""
        active_sessions = self.get_active_sessions(asset_class)
        
        status = {
            'asset_class': asset_class.value,
            'is_open': len(active_sessions) > 0,
            'active_sessions': [s.name for s in active_sessions],
            'primary_session': active_sessions[0].name if active_sessions else None,
            'next_open': self._get_next_session_open(asset_class),
            'trading_intensity': self._calculate_trading_intensity(active_sessions)
        }
        
        return status
    
    def is_market_open(self, asset_class: Optional[AssetClass] = None) -> bool:
        """Check if any market is open for trading."""
        if asset_class:
            # Check specific asset class
            active_sessions = self.get_active_sessions(asset_class)
            return len(active_sessions) > 0
        else:
            # Check if any market is open
            for ac in AssetClass:
                if len(self.get_active_sessions(ac)) > 0:
                    return True
            return False
    
    def _get_next_session_open(self, asset_class: AssetClass) -> Optional[datetime]:
        """Get the next session opening time for an asset class."""
        now = datetime.now()
        next_opens = []
        
        for session in self.sessions.values():
            if session.asset_class != asset_class or not session.enabled:
                continue
            
            try:
                tz = pytz.timezone(session.timezone)
                local_now = now.astimezone(tz)
                
                # Find next opening time
                for day_offset in range(8):  # Check next 7 days
                    check_date = local_now + timedelta(days=day_offset)
                    if check_date.weekday() in session.days:
                        open_hour, open_minute = map(int, session.open_time.split(':'))
                        session_open = check_date.replace(
                            hour=open_hour, minute=open_minute, second=0, microsecond=0
                        )
                        
                        if session_open > local_now:
                            next_opens.append(session_open.astimezone(pytz.UTC))
                            break
            
            except Exception as e:
                self.logger.error(f"Error calculating next open for {session.name}: {e}")
        
        return min(next_opens) if next_opens else None
    
    def _calculate_trading_intensity(self, active_sessions: List[TradingSession]) -> float:
        """Calculate trading intensity based on active sessions."""
        if not active_sessions:
            return 0.0
        
        # Weight by session priority and overlap
        total_priority = sum(session.priority for session in active_sessions)
        overlap_bonus = min(len(active_sessions) * 0.2, 1.0)  # Max 100% bonus
        
        return min(total_priority / 10.0 + overlap_bonus, 1.0)
    
    def should_run_task(self, task: ScheduledTask) -> bool:
        """Determine if a task should run based on market conditions."""
        now = datetime.now()
        
        # Check if it's time to run
        if now < task.next_run:
            return False
        
        # If task is not session-dependent, always run
        if not task.session_dependent:
            return True
        
        # Check if relevant market sessions are active
        active_sessions = self.get_active_sessions(task.asset_class)
        
        # Special handling for crypto (always active)
        if task.asset_class == AssetClass.CRYPTO:
            return True
        
        # For other assets, require active session
        return len(active_sessions) > 0
    
    async def run_task(self, task: ScheduledTask):
        """Execute a scheduled task."""
        try:
            self.logger.debug(f"Running task: {task.name}")
            
            # Get market context
            market_status = self.get_market_status(task.asset_class)
            
            # Execute callback with market context
            if asyncio.iscoroutinefunction(task.callback):
                await task.callback(market_status)
            else:
                task.callback(market_status)
            
            # Schedule next run
            task.next_run = datetime.now() + timedelta(seconds=task.interval)
            
        except Exception as e:
            self.logger.error(f"Error running task {task.name}: {e}")
            # Still schedule next run to prevent task from getting stuck
            task.next_run = datetime.now() + timedelta(seconds=task.interval)
    
    async def start(self):
        """Start the 24/7 trading scheduler."""
        self.running = True
        self.logger.info("Starting 24/7 Trading Scheduler")
        
        while self.running:
            try:
                # Check and run due tasks
                for task in list(self.tasks.values()):
                    if task.enabled and self.should_run_task(task):
                        asyncio.create_task(self.run_task(task))
                
                # Log market status periodically
                if datetime.now().minute % 15 == 0:  # Every 15 minutes
                    await self._log_market_status()
                
                # Sleep for a short interval
                await asyncio.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                self.logger.error(f"Error in scheduler main loop: {e}")
                await asyncio.sleep(30)  # Wait longer on error
    
    async def _log_market_status(self):
        """Log current market status for all asset classes."""
        try:
            status_summary = []
            
            for asset_class in AssetClass:
                status = self.get_market_status(asset_class)
                if status['is_open']:
                    intensity = status['trading_intensity']
                    primary = status['primary_session']
                    status_summary.append(f"{asset_class.value}: {primary} ({intensity:.1%})")
            
            if status_summary:
                self.logger.info(f"Active Markets: {', '.join(status_summary)}")
            else:
                self.logger.info("No active trading sessions")
                
        except Exception as e:
            self.logger.error(f"Error logging market status: {e}")
    
    def stop(self):
        """Stop the scheduler."""
        self.running = False
        self.logger.info("Stopping 24/7 Trading Scheduler")
    
    def get_scheduler_stats(self) -> Dict[str, Any]:
        """Get scheduler statistics."""
        now = datetime.now()
        
        stats = {
            'running': self.running,
            'total_tasks': len(self.tasks),
            'enabled_tasks': sum(1 for t in self.tasks.values() if t.enabled),
            'total_sessions': len(self.sessions),
            'enabled_sessions': sum(1 for s in self.sessions.values() if s.enabled),
            'active_sessions_by_asset': {},
            'next_task_runs': {}
        }
        
        # Active sessions by asset class
        for asset_class in AssetClass:
            active = self.get_active_sessions(asset_class)
            stats['active_sessions_by_asset'][asset_class.value] = len(active)
        
        # Next task run times
        for name, task in self.tasks.items():
            if task.enabled:
                time_until = (task.next_run - now).total_seconds()
                stats['next_task_runs'][name] = max(0, int(time_until))
        
        return stats
    
    def enable_asset_class(self, asset_class: AssetClass, enabled: bool = True):
        """Enable or disable all sessions for an asset class."""
        count = 0
        for session in self.sessions.values():
            if session.asset_class == asset_class:
                session.enabled = enabled
                count += 1
        
        action = "Enabled" if enabled else "Disabled"
        self.logger.info(f"{action} {count} sessions for {asset_class.value}")
    
    def enable_task(self, task_name: str, enabled: bool = True):
        """Enable or disable a specific task."""
        if task_name in self.tasks:
            self.tasks[task_name].enabled = enabled
            action = "Enabled" if enabled else "Disabled"
            self.logger.info(f"{action} task: {task_name}")
        else:
            self.logger.warning(f"Task not found: {task_name}")