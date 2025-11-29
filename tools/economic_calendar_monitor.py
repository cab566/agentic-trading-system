#!/usr/bin/env python3
"""
Economic Calendar Monitor

This tool monitors economic events, announcements, and data releases
to help trading agents anticipate market movements and volatility.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json
import requests
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import pytz

# CrewAI imports
from crewai_tools import BaseTool
from pydantic import BaseModel, Field

# Local imports
from ..core.data_manager import UnifiedDataManager
from ..core.config_manager import ConfigManager


class EventImportance(Enum):
    """Economic event importance levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class EventCategory(Enum):
    """Economic event categories"""
    EMPLOYMENT = "employment"
    INFLATION = "inflation"
    GDP = "gdp"
    MONETARY_POLICY = "monetary_policy"
    MANUFACTURING = "manufacturing"
    CONSUMER = "consumer"
    HOUSING = "housing"
    TRADE = "trade"
    EARNINGS = "earnings"
    CENTRAL_BANK = "central_bank"
    GEOPOLITICAL = "geopolitical"
    OTHER = "other"


class MarketImpact(Enum):
    """Expected market impact"""
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"
    VOLATILE = "volatile"
    UNKNOWN = "unknown"


@dataclass
class EconomicEvent:
    """Economic event data structure"""
    event_id: str
    title: str
    country: str
    category: EventCategory
    importance: EventImportance
    scheduled_time: datetime
    actual_value: Optional[float] = None
    forecast_value: Optional[float] = None
    previous_value: Optional[float] = None
    unit: Optional[str] = None
    description: str = ""
    source: str = ""
    market_impact: MarketImpact = MarketImpact.UNKNOWN
    affected_currencies: List[str] = field(default_factory=list)
    affected_sectors: List[str] = field(default_factory=list)
    volatility_expected: bool = False
    pre_event_positioning: Dict[str, str] = field(default_factory=dict)
    post_event_analysis: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EventAlert:
    """Economic event alert"""
    event: EconomicEvent
    alert_type: str  # 'upcoming', 'released', 'surprise'
    alert_time: datetime
    message: str
    trading_implications: List[str]
    recommended_actions: List[str]


@dataclass
class MarketReaction:
    """Market reaction to economic event"""
    event_id: str
    symbol: str
    price_before: float
    price_after: float
    price_change_pct: float
    volume_change_pct: float
    volatility_change: float
    reaction_time_minutes: int
    sustained_move: bool


class EconomicCalendarConfig(BaseModel):
    """Configuration for economic calendar monitoring"""
    update_frequency_minutes: int = Field(default=60, description="Update frequency in minutes")
    lookback_days: int = Field(default=7, description="Days to look back for events")
    lookahead_days: int = Field(default=14, description="Days to look ahead for events")
    min_importance: EventImportance = Field(default=EventImportance.MEDIUM, description="Minimum event importance")
    monitored_countries: List[str] = Field(default=['US', 'EU', 'GB', 'JP', 'CA'], description="Countries to monitor")
    alert_before_minutes: List[int] = Field(default=[60, 15, 5], description="Alert times before events")
    track_market_reaction: bool = Field(default=True, description="Track market reactions to events")
    reaction_window_minutes: int = Field(default=30, description="Window to measure market reaction")


class EconomicCalendarMonitorTool(BaseTool):
    """Economic calendar monitoring and analysis system"""
    
    name: str = "economic_calendar_monitor"
    description: str = "Monitors economic events and their market impact for trading decisions"
    
    def __init__(self, config_manager: ConfigManager, data_manager: UnifiedDataManager):
        super().__init__()
        self.config_manager = config_manager
        self.data_manager = data_manager
        self.logger = logging.getLogger(__name__)
        
        # Load configuration
        self.config = self._load_config()
        
        # Event storage
        self.upcoming_events: List[EconomicEvent] = []
        self.past_events: List[EconomicEvent] = []
        self.event_alerts: List[EventAlert] = []
        self.market_reactions: List[MarketReaction] = []
        
        # Data sources
        self.data_sources = self._initialize_data_sources()
        
        # Event impact models
        self.impact_models = self._initialize_impact_models()
        
        # Last update time
        self.last_update: Optional[datetime] = None
        
        # Initialize with current events
        asyncio.create_task(self._initial_load())
    
    def _load_config(self) -> EconomicCalendarConfig:
        """Load economic calendar configuration"""
        try:
            config_dict = self.config_manager.get_config('economic_calendar')
            return EconomicCalendarConfig(**config_dict)
        except Exception as e:
            self.logger.warning(f"Could not load economic calendar config: {e}")
            return EconomicCalendarConfig()
    
    def _initialize_data_sources(self) -> Dict[str, Dict[str, Any]]:
        """Initialize economic data sources"""
        sources = {}
        
        # Trading Economics API (example)
        sources['trading_economics'] = {
            'base_url': 'https://api.tradingeconomics.com',
            'endpoints': {
                'calendar': '/calendar',
                'indicators': '/indicators',
                'forecasts': '/forecasts'
            },
            'requires_auth': True
        }
        
        # Economic Calendar API (example)
        sources['economic_calendar'] = {
            'base_url': 'https://api.economiccalendar.com',
            'endpoints': {
                'events': '/events',
                'releases': '/releases'
            },
            'requires_auth': True
        }
        
        # Alpha Vantage Economic Indicators
        sources['alpha_vantage'] = {
            'base_url': 'https://www.alphavantage.co/query',
            'functions': [
                'REAL_GDP', 'INFLATION', 'UNEMPLOYMENT', 'FEDERAL_FUNDS_RATE',
                'CPI', 'NONFARM_PAYROLL', 'RETAIL_SALES'
            ],
            'requires_auth': True
        }
        
        # FRED (Federal Reserve Economic Data)
        sources['fred'] = {
            'base_url': 'https://api.stlouisfed.org/fred',
            'endpoints': {
                'series': '/series/observations',
                'releases': '/releases'
            },
            'requires_auth': True
        }
        
        return sources
    
    def _initialize_impact_models(self) -> Dict[str, Dict[str, Any]]:
        """Initialize event impact prediction models"""
        models = {}
        
        # Employment events impact
        models['employment'] = {
            'primary_indicators': ['NONFARM_PAYROLL', 'UNEMPLOYMENT_RATE'],
            'affected_currencies': ['USD'],
            'affected_sectors': ['financials', 'consumer_discretionary'],
            'volatility_multiplier': 1.5,
            'impact_duration_hours': 4
        }
        
        # Inflation events impact
        models['inflation'] = {
            'primary_indicators': ['CPI', 'PPI', 'PCE'],
            'affected_currencies': ['USD'],
            'affected_sectors': ['utilities', 'real_estate', 'financials'],
            'volatility_multiplier': 1.3,
            'impact_duration_hours': 6
        }
        
        # GDP events impact
        models['gdp'] = {
            'primary_indicators': ['GDP_QOQ', 'GDP_YOY'],
            'affected_currencies': ['USD'],
            'affected_sectors': ['broad_market'],
            'volatility_multiplier': 1.2,
            'impact_duration_hours': 8
        }
        
        # Monetary policy events impact
        models['monetary_policy'] = {
            'primary_indicators': ['INTEREST_RATE', 'FOMC_DECISION'],
            'affected_currencies': ['USD'],
            'affected_sectors': ['financials', 'real_estate', 'utilities'],
            'volatility_multiplier': 2.0,
            'impact_duration_hours': 24
        }
        
        return models
    
    def _run(self, action: str, parameters: Dict[str, Any] = None) -> str:
        """Synchronous economic calendar execution"""
        return asyncio.run(self._arun(action, parameters))
    
    async def _arun(self, action: str, parameters: Dict[str, Any] = None) -> str:
        """Asynchronous economic calendar execution"""
        try:
            parameters = parameters or {}
            
            if action == 'get_upcoming_events':
                return await self._get_upcoming_events(parameters)
            elif action == 'get_event_alerts':
                return await self._get_event_alerts(parameters)
            elif action == 'analyze_event_impact':
                return await self._analyze_event_impact(parameters)
            elif action == 'get_market_reactions':
                return await self._get_market_reactions(parameters)
            elif action == 'update_calendar':
                return await self._update_calendar(parameters)
            elif action == 'get_event_by_id':
                return await self._get_event_by_id(parameters)
            elif action == 'get_calendar_summary':
                return await self._get_calendar_summary(parameters)
            elif action == 'track_event_reaction':
                return await self._track_event_reaction(parameters)
            else:
                return json.dumps({
                    'error': f'Unknown action: {action}',
                    'available_actions': [
                        'get_upcoming_events', 'get_event_alerts', 'analyze_event_impact',
                        'get_market_reactions', 'update_calendar', 'get_event_by_id',
                        'get_calendar_summary', 'track_event_reaction'
                    ]
                })
                
        except Exception as e:
            self.logger.error(f"Economic calendar operation failed: {e}")
            return json.dumps({'error': str(e)})
    
    async def _initial_load(self):
        """Initial load of economic calendar data"""
        try:
            await self._update_calendar({})
        except Exception as e:
            self.logger.error(f"Initial calendar load failed: {e}")
    
    async def _get_upcoming_events(self, parameters: Dict[str, Any]) -> str:
        """Get upcoming economic events"""
        days_ahead = parameters.get('days_ahead', self.config.lookahead_days)
        importance_filter = parameters.get('importance', None)
        country_filter = parameters.get('countries', None)
        category_filter = parameters.get('categories', None)
        
        # Update calendar if needed
        if self._should_update_calendar():
            await self._update_calendar({})
        
        # Filter events
        now = datetime.now(pytz.UTC)
        end_time = now + timedelta(days=days_ahead)
        
        filtered_events = []
        for event in self.upcoming_events:
            # Time filter
            if not (now <= event.scheduled_time <= end_time):
                continue
            
            # Importance filter
            if importance_filter and event.importance.value != importance_filter:
                continue
            
            # Country filter
            if country_filter and event.country not in country_filter:
                continue
            
            # Category filter
            if category_filter and event.category.value not in category_filter:
                continue
            
            filtered_events.append(event)
        
        # Sort by scheduled time
        filtered_events.sort(key=lambda x: x.scheduled_time)
        
        # Format response
        events_data = []
        for event in filtered_events:
            event_data = {
                'event_id': event.event_id,
                'title': event.title,
                'country': event.country,
                'category': event.category.value,
                'importance': event.importance.value,
                'scheduled_time': event.scheduled_time.isoformat(),
                'forecast_value': event.forecast_value,
                'previous_value': event.previous_value,
                'unit': event.unit,
                'description': event.description,
                'market_impact': event.market_impact.value,
                'affected_currencies': event.affected_currencies,
                'affected_sectors': event.affected_sectors,
                'volatility_expected': event.volatility_expected,
                'time_until_event': str(event.scheduled_time - now)
            }
            events_data.append(event_data)
        
        return json.dumps({
            'upcoming_events': events_data,
            'total_events': len(events_data),
            'filter_criteria': {
                'days_ahead': days_ahead,
                'importance_filter': importance_filter,
                'country_filter': country_filter,
                'category_filter': category_filter
            },
            'last_updated': self.last_update.isoformat() if self.last_update else None
        }, indent=2)
    
    async def _get_event_alerts(self, parameters: Dict[str, Any]) -> str:
        """Get event alerts"""
        alert_type_filter = parameters.get('alert_type', None)
        hours_back = parameters.get('hours_back', 24)
        
        # Filter alerts
        cutoff_time = datetime.now(pytz.UTC) - timedelta(hours=hours_back)
        
        filtered_alerts = []
        for alert in self.event_alerts:
            if alert.alert_time >= cutoff_time:
                if not alert_type_filter or alert.alert_type == alert_type_filter:
                    filtered_alerts.append(alert)
        
        # Sort by alert time (most recent first)
        filtered_alerts.sort(key=lambda x: x.alert_time, reverse=True)
        
        # Format response
        alerts_data = []
        for alert in filtered_alerts:
            alert_data = {
                'event_id': alert.event.event_id,
                'event_title': alert.event.title,
                'alert_type': alert.alert_type,
                'alert_time': alert.alert_time.isoformat(),
                'message': alert.message,
                'trading_implications': alert.trading_implications,
                'recommended_actions': alert.recommended_actions,
                'event_importance': alert.event.importance.value,
                'scheduled_time': alert.event.scheduled_time.isoformat()
            }
            alerts_data.append(alert_data)
        
        return json.dumps({
            'alerts': alerts_data,
            'total_alerts': len(alerts_data),
            'filter_criteria': {
                'alert_type_filter': alert_type_filter,
                'hours_back': hours_back
            }
        }, indent=2)
    
    async def _analyze_event_impact(self, parameters: Dict[str, Any]) -> str:
        """Analyze potential impact of economic event"""
        event_id = parameters.get('event_id')
        if not event_id:
            return json.dumps({'error': 'event_id parameter required'})
        
        # Find event
        event = None
        for e in self.upcoming_events + self.past_events:
            if e.event_id == event_id:
                event = e
                break
        
        if not event:
            return json.dumps({'error': f'Event {event_id} not found'})
        
        # Analyze impact
        impact_analysis = await self._perform_impact_analysis(event)
        
        return json.dumps({
            'event': {
                'event_id': event.event_id,
                'title': event.title,
                'category': event.category.value,
                'importance': event.importance.value,
                'scheduled_time': event.scheduled_time.isoformat()
            },
            'impact_analysis': impact_analysis
        }, indent=2)
    
    async def _perform_impact_analysis(self, event: EconomicEvent) -> Dict[str, Any]:
        """Perform detailed impact analysis for an event"""
        category = event.category.value
        impact_model = self.impact_models.get(category, {})
        
        analysis = {
            'expected_volatility': self._calculate_expected_volatility(event),
            'affected_assets': self._identify_affected_assets(event),
            'directional_bias': self._predict_directional_bias(event),
            'time_horizon': self._estimate_impact_duration(event),
            'confidence_level': self._calculate_confidence_level(event),
            'historical_precedents': await self._find_historical_precedents(event),
            'trading_strategies': self._suggest_trading_strategies(event),
            'risk_factors': self._identify_risk_factors(event)
        }
        
        return analysis
    
    def _calculate_expected_volatility(self, event: EconomicEvent) -> Dict[str, float]:
        """Calculate expected volatility from event"""
        base_volatility = {
            EventImportance.CRITICAL: 0.8,
            EventImportance.HIGH: 0.6,
            EventImportance.MEDIUM: 0.4,
            EventImportance.LOW: 0.2
        }.get(event.importance, 0.3)
        
        # Adjust based on category
        category_multipliers = {
            EventCategory.MONETARY_POLICY: 1.5,
            EventCategory.EMPLOYMENT: 1.3,
            EventCategory.INFLATION: 1.2,
            EventCategory.GDP: 1.1,
            EventCategory.EARNINGS: 1.0
        }
        
        multiplier = category_multipliers.get(event.category, 1.0)
        expected_volatility = base_volatility * multiplier
        
        return {
            'base_volatility': base_volatility,
            'category_multiplier': multiplier,
            'expected_volatility': min(1.0, expected_volatility),
            'volatility_duration_hours': self.impact_models.get(event.category.value, {}).get('impact_duration_hours', 4)
        }
    
    def _identify_affected_assets(self, event: EconomicEvent) -> Dict[str, List[str]]:
        """Identify assets likely to be affected by event"""
        affected_assets = {
            'currencies': event.affected_currencies.copy(),
            'sectors': event.affected_sectors.copy(),
            'indices': [],
            'commodities': []
        }
        
        # Add default affected assets based on category
        if event.category == EventCategory.EMPLOYMENT:
            affected_assets['indices'].extend(['SPY', 'QQQ', 'IWM'])
            affected_assets['sectors'].extend(['XLF', 'XLY'])
        
        elif event.category == EventCategory.INFLATION:
            affected_assets['indices'].extend(['SPY', 'TLT'])
            affected_assets['sectors'].extend(['XLU', 'XLRE'])
            affected_assets['commodities'].extend(['GLD', 'TIP'])
        
        elif event.category == EventCategory.MONETARY_POLICY:
            affected_assets['indices'].extend(['SPY', 'QQQ', 'TLT'])
            affected_assets['sectors'].extend(['XLF', 'XLU', 'XLRE'])
        
        # Add country-specific assets
        if event.country == 'US':
            if 'USD' not in affected_assets['currencies']:
                affected_assets['currencies'].append('USD')
        
        return affected_assets
    
    def _predict_directional_bias(self, event: EconomicEvent) -> Dict[str, str]:
        """Predict directional bias based on event"""
        bias = {}
        
        # Simple heuristics based on event type and expected vs actual
        if event.forecast_value is not None and event.actual_value is not None:
            surprise = event.actual_value - event.forecast_value
            
            if event.category == EventCategory.EMPLOYMENT:
                if surprise > 0:  # Better than expected employment
                    bias['equities'] = 'bullish'
                    bias['bonds'] = 'bearish'
                    bias['USD'] = 'bullish'
                else:
                    bias['equities'] = 'bearish'
                    bias['bonds'] = 'bullish'
                    bias['USD'] = 'bearish'
            
            elif event.category == EventCategory.INFLATION:
                if surprise > 0:  # Higher than expected inflation
                    bias['equities'] = 'bearish'
                    bias['bonds'] = 'bearish'
                    bias['USD'] = 'bullish'
                    bias['commodities'] = 'bullish'
                else:
                    bias['equities'] = 'bullish'
                    bias['bonds'] = 'bullish'
                    bias['USD'] = 'bearish'
                    bias['commodities'] = 'bearish'
        
        else:
            # Pre-event bias based on expectations
            bias = {
                'equities': 'neutral',
                'bonds': 'neutral',
                'USD': 'neutral',
                'volatility': 'bullish' if event.importance in [EventImportance.HIGH, EventImportance.CRITICAL] else 'neutral'
            }
        
        return bias
    
    def _estimate_impact_duration(self, event: EconomicEvent) -> Dict[str, int]:
        """Estimate how long the impact will last"""
        base_duration = {
            EventImportance.CRITICAL: 48,
            EventImportance.HIGH: 24,
            EventImportance.MEDIUM: 8,
            EventImportance.LOW: 2
        }.get(event.importance, 4)
        
        category_adjustments = {
            EventCategory.MONETARY_POLICY: 2.0,
            EventCategory.GDP: 1.5,
            EventCategory.EMPLOYMENT: 1.2,
            EventCategory.INFLATION: 1.3
        }
        
        multiplier = category_adjustments.get(event.category, 1.0)
        
        return {
            'immediate_impact_hours': int(base_duration * 0.25),
            'primary_impact_hours': int(base_duration * multiplier),
            'residual_impact_hours': int(base_duration * multiplier * 2)
        }
    
    def _calculate_confidence_level(self, event: EconomicEvent) -> float:
        """Calculate confidence level in impact prediction"""
        confidence = 0.5  # Base confidence
        
        # Adjust based on event importance
        importance_boost = {
            EventImportance.CRITICAL: 0.3,
            EventImportance.HIGH: 0.2,
            EventImportance.MEDIUM: 0.1,
            EventImportance.LOW: 0.0
        }.get(event.importance, 0.0)
        
        confidence += importance_boost
        
        # Adjust based on data availability
        if event.forecast_value is not None:
            confidence += 0.1
        if event.previous_value is not None:
            confidence += 0.1
        
        return min(1.0, confidence)
    
    async def _find_historical_precedents(self, event: EconomicEvent) -> List[Dict[str, Any]]:
        """Find historical precedents for similar events"""
        # Simplified implementation - in practice, would query historical database
        precedents = []
        
        # Look for similar events in past_events
        similar_events = [
            e for e in self.past_events
            if e.category == event.category and e.importance == event.importance
        ]
        
        for similar_event in similar_events[-3:]:  # Last 3 similar events
            if similar_event.post_event_analysis:
                precedent = {
                    'date': similar_event.scheduled_time.isoformat(),
                    'title': similar_event.title,
                    'actual_vs_forecast': None,
                    'market_reaction': similar_event.post_event_analysis.get('market_reaction', 'unknown')
                }
                
                if (similar_event.actual_value is not None and 
                    similar_event.forecast_value is not None):
                    precedent['actual_vs_forecast'] = similar_event.actual_value - similar_event.forecast_value
                
                precedents.append(precedent)
        
        return precedents
    
    def _suggest_trading_strategies(self, event: EconomicEvent) -> List[Dict[str, str]]:
        """Suggest trading strategies for the event"""
        strategies = []
        
        if event.importance in [EventImportance.HIGH, EventImportance.CRITICAL]:
            # High impact events
            strategies.append({
                'strategy': 'Volatility Play',
                'description': 'Buy straddles or strangles to profit from volatility',
                'instruments': 'Options on affected indices/ETFs',
                'risk_level': 'Medium'
            })
            
            strategies.append({
                'strategy': 'Event Fade',
                'description': 'Fade initial reaction if it appears overdone',
                'instruments': 'Affected stocks/ETFs',
                'risk_level': 'High'
            })
        
        if event.category == EventCategory.EMPLOYMENT:
            strategies.append({
                'strategy': 'Sector Rotation',
                'description': 'Rotate between cyclical and defensive sectors',
                'instruments': 'Sector ETFs (XLF, XLY, XLP)',
                'risk_level': 'Medium'
            })
        
        elif event.category == EventCategory.MONETARY_POLICY:
            strategies.append({
                'strategy': 'Rate Sensitivity Play',
                'description': 'Trade rate-sensitive sectors',
                'instruments': 'Financials (XLF), REITs (XLRE), Utilities (XLU)',
                'risk_level': 'Medium'
            })
        
        return strategies
    
    def _identify_risk_factors(self, event: EconomicEvent) -> List[str]:
        """Identify risk factors for the event"""
        risks = []
        
        if event.importance in [EventImportance.HIGH, EventImportance.CRITICAL]:
            risks.append("High volatility and potential for large price gaps")
            risks.append("Increased correlation across asset classes")
        
        if event.category == EventCategory.MONETARY_POLICY:
            risks.append("Policy surprise could cause significant market dislocation")
            risks.append("Forward guidance changes could impact long-term positioning")
        
        if event.category == EventCategory.EMPLOYMENT:
            risks.append("Revision to previous data could amplify or dampen reaction")
            risks.append("Seasonal adjustments may distort headline numbers")
        
        risks.append("Market positioning ahead of event could amplify moves")
        risks.append("Liquidity may be reduced around event time")
        
        return risks
    
    async def _update_calendar(self, parameters: Dict[str, Any]) -> str:
        """Update economic calendar data"""
        force_update = parameters.get('force_update', False)
        
        if not force_update and not self._should_update_calendar():
            return json.dumps({
                'message': 'Calendar is up to date',
                'last_updated': self.last_update.isoformat() if self.last_update else None
            })
        
        try:
            # Fetch events from multiple sources
            new_events = await self._fetch_calendar_events()
            
            # Update event lists
            self._update_event_lists(new_events)
            
            # Generate alerts for upcoming events
            await self._generate_event_alerts()
            
            # Update timestamp
            self.last_update = datetime.now(pytz.UTC)
            
            return json.dumps({
                'message': 'Calendar updated successfully',
                'events_loaded': len(new_events),
                'upcoming_events': len(self.upcoming_events),
                'last_updated': self.last_update.isoformat()
            })
            
        except Exception as e:
            self.logger.error(f"Failed to update calendar: {e}")
            return json.dumps({'error': f'Failed to update calendar: {str(e)}'})
    
    def _should_update_calendar(self) -> bool:
        """Check if calendar should be updated"""
        if not self.last_update:
            return True
        
        time_since_update = datetime.now(pytz.UTC) - self.last_update
        return time_since_update.total_seconds() > (self.config.update_frequency_minutes * 60)
    
    async def _fetch_calendar_events(self) -> List[EconomicEvent]:
        """Fetch economic calendar events from data sources"""
        events = []
        
        # In a real implementation, you would fetch from actual APIs
        # For now, we'll create some sample events
        events.extend(self._create_sample_events())
        
        return events
    
    def _create_sample_events(self) -> List[EconomicEvent]:
        """Create sample economic events for demonstration"""
        now = datetime.now(pytz.UTC)
        events = []
        
        # Sample upcoming events
        sample_events = [
            {
                'title': 'Non-Farm Payrolls',
                'category': EventCategory.EMPLOYMENT,
                'importance': EventImportance.HIGH,
                'days_ahead': 2,
                'forecast': 200000,
                'previous': 180000,
                'unit': 'jobs'
            },
            {
                'title': 'Consumer Price Index (CPI)',
                'category': EventCategory.INFLATION,
                'importance': EventImportance.HIGH,
                'days_ahead': 5,
                'forecast': 0.3,
                'previous': 0.2,
                'unit': '%'
            },
            {
                'title': 'Federal Reserve Interest Rate Decision',
                'category': EventCategory.MONETARY_POLICY,
                'importance': EventImportance.CRITICAL,
                'days_ahead': 7,
                'forecast': 5.25,
                'previous': 5.00,
                'unit': '%'
            },
            {
                'title': 'GDP Quarterly Growth',
                'category': EventCategory.GDP,
                'importance': EventImportance.HIGH,
                'days_ahead': 10,
                'forecast': 2.1,
                'previous': 2.4,
                'unit': '%'
            },
            {
                'title': 'Retail Sales',
                'category': EventCategory.CONSUMER,
                'importance': EventImportance.MEDIUM,
                'days_ahead': 3,
                'forecast': 0.4,
                'previous': 0.1,
                'unit': '%'
            }
        ]
        
        for i, event_data in enumerate(sample_events):
            event = EconomicEvent(
                event_id=f"event_{i}_{int(now.timestamp())}",
                title=event_data['title'],
                country='US',
                category=event_data['category'],
                importance=event_data['importance'],
                scheduled_time=now + timedelta(days=event_data['days_ahead']),
                forecast_value=event_data.get('forecast'),
                previous_value=event_data.get('previous'),
                unit=event_data.get('unit'),
                description=f"Economic indicator: {event_data['title']}",
                source='sample_data',
                affected_currencies=['USD'],
                volatility_expected=event_data['importance'] in [EventImportance.HIGH, EventImportance.CRITICAL]
            )
            
            # Set market impact based on category
            if event.category == EventCategory.EMPLOYMENT:
                event.market_impact = MarketImpact.VOLATILE
                event.affected_sectors = ['financials', 'consumer_discretionary']
            elif event.category == EventCategory.INFLATION:
                event.market_impact = MarketImpact.VOLATILE
                event.affected_sectors = ['utilities', 'real_estate']
            elif event.category == EventCategory.MONETARY_POLICY:
                event.market_impact = MarketImpact.VOLATILE
                event.affected_sectors = ['financials', 'real_estate', 'utilities']
            
            events.append(event)
        
        return events
    
    def _update_event_lists(self, new_events: List[EconomicEvent]):
        """Update upcoming and past event lists"""
        now = datetime.now(pytz.UTC)
        
        # Clear old events and add new ones
        self.upcoming_events = [e for e in new_events if e.scheduled_time > now]
        
        # Move past events to past_events list
        past_events_to_add = [e for e in new_events if e.scheduled_time <= now]
        self.past_events.extend(past_events_to_add)
        
        # Keep only recent past events (last 30 days)
        cutoff_date = now - timedelta(days=30)
        self.past_events = [e for e in self.past_events if e.scheduled_time >= cutoff_date]
        
        # Sort lists
        self.upcoming_events.sort(key=lambda x: x.scheduled_time)
        self.past_events.sort(key=lambda x: x.scheduled_time, reverse=True)
    
    async def _generate_event_alerts(self):
        """Generate alerts for upcoming events"""
        now = datetime.now(pytz.UTC)
        
        for event in self.upcoming_events:
            time_until_event = (event.scheduled_time - now).total_seconds() / 60  # minutes
            
            # Check if we should generate alerts
            for alert_minutes in self.config.alert_before_minutes:
                if (alert_minutes - 5) <= time_until_event <= (alert_minutes + 5):
                    # Generate alert if not already generated
                    existing_alert = any(
                        a.event.event_id == event.event_id and 
                        a.alert_type == 'upcoming' and
                        abs((a.alert_time - now).total_seconds()) < 300  # Within 5 minutes
                        for a in self.event_alerts
                    )
                    
                    if not existing_alert:
                        alert = self._create_event_alert(event, 'upcoming', alert_minutes)
                        self.event_alerts.append(alert)
    
    def _create_event_alert(self, event: EconomicEvent, alert_type: str, minutes_before: int = None) -> EventAlert:
        """Create an event alert"""
        now = datetime.now(pytz.UTC)
        
        if alert_type == 'upcoming':
            message = f"Upcoming {event.importance.value} impact event: {event.title} in {minutes_before} minutes"
            trading_implications = [
                f"Expected volatility in {', '.join(event.affected_currencies + event.affected_sectors)}",
                f"Market impact: {event.market_impact.value}"
            ]
            recommended_actions = [
                "Review positions in affected assets",
                "Consider volatility hedges if needed",
                "Monitor for early market reactions"
            ]
        
        elif alert_type == 'released':
            message = f"Economic data released: {event.title}"
            trading_implications = [
                "Analyze actual vs forecast deviation",
                "Monitor immediate market reaction"
            ]
            recommended_actions = [
                "Assess impact on existing positions",
                "Look for trading opportunities in affected assets"
            ]
        
        else:  # surprise
            message = f"Surprise in economic data: {event.title}"
            trading_implications = [
                "Significant deviation from expectations",
                "Potential for extended market reaction"
            ]
            recommended_actions = [
                "Reassess market outlook",
                "Consider position adjustments"
            ]
        
        return EventAlert(
            event=event,
            alert_type=alert_type,
            alert_time=now,
            message=message,
            trading_implications=trading_implications,
            recommended_actions=recommended_actions
        )
    
    async def _get_calendar_summary(self, parameters: Dict[str, Any]) -> str:
        """Get economic calendar summary"""
        days_ahead = parameters.get('days_ahead', 7)
        
        # Update calendar if needed
        if self._should_update_calendar():
            await self._update_calendar({})
        
        now = datetime.now(pytz.UTC)
        end_time = now + timedelta(days=days_ahead)
        
        # Filter upcoming events
        upcoming = [e for e in self.upcoming_events if now <= e.scheduled_time <= end_time]
        
        # Categorize by importance and day
        summary = {
            'critical_events': [],
            'high_impact_events': [],
            'daily_breakdown': {},
            'key_themes': [],
            'risk_assessment': {}
        }
        
        for event in upcoming:
            event_date = event.scheduled_time.date().isoformat()
            
            if event_date not in summary['daily_breakdown']:
                summary['daily_breakdown'][event_date] = []
            
            event_summary = {
                'title': event.title,
                'time': event.scheduled_time.strftime('%H:%M UTC'),
                'importance': event.importance.value,
                'category': event.category.value
            }
            
            summary['daily_breakdown'][event_date].append(event_summary)
            
            if event.importance == EventImportance.CRITICAL:
                summary['critical_events'].append(event_summary)
            elif event.importance == EventImportance.HIGH:
                summary['high_impact_events'].append(event_summary)
        
        # Identify key themes
        category_counts = {}
        for event in upcoming:
            category = event.category.value
            category_counts[category] = category_counts.get(category, 0) + 1
        
        summary['key_themes'] = [
            f"{count} {category} events" 
            for category, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True)
        ]
        
        # Risk assessment
        high_impact_count = len([e for e in upcoming if e.importance in [EventImportance.HIGH, EventImportance.CRITICAL]])
        total_events = len(upcoming)
        
        if high_impact_count >= 3:
            risk_level = "High"
            risk_description = "Multiple high-impact events may cause significant volatility"
        elif high_impact_count >= 1:
            risk_level = "Medium"
            risk_description = "Some market-moving events scheduled"
        else:
            risk_level = "Low"
            risk_description = "Relatively quiet period for economic data"
        
        summary['risk_assessment'] = {
            'level': risk_level,
            'description': risk_description,
            'high_impact_events': high_impact_count,
            'total_events': total_events
        }
        
        return json.dumps({
            'summary_period': {
                'start_date': now.date().isoformat(),
                'end_date': end_time.date().isoformat(),
                'days_ahead': days_ahead
            },
            'summary': summary,
            'last_updated': self.last_update.isoformat() if self.last_update else None
        }, indent=2)


# Example usage
if __name__ == "__main__":
    from pathlib import Path
    from ..core.config_manager import ConfigManager
    from ..core.data_manager import UnifiedDataManager
    
    config_manager = ConfigManager(Path("../config"))
    data_manager = UnifiedDataManager(config_manager)
    
    calendar_monitor = EconomicCalendarMonitorTool(config_manager, data_manager)
    
    # Test calendar functionality
    result = calendar_monitor._run('get_upcoming_events')
    print("Upcoming Events:", result)