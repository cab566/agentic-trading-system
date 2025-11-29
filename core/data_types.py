#!/usr/bin/env python3
"""Base data types and classes for the trading system."""

import asyncio
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass

import pandas as pd


@dataclass
class DataRequest:
    """Data request structure."""
    symbol: str
    data_type: str  # 'price', 'news', 'fundamentals', 'options', 'technical'
    timeframe: str  # '1m', '5m', '1h', '1d', etc.
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    parameters: Optional[Dict[str, Any]] = None


@dataclass
class DataResponse:
    """Data response structure."""
    data: Union[pd.DataFrame, Dict[str, Any], List[Dict[str, Any]]]
    source: str
    timestamp: datetime
    cached: bool = False
    error: Optional[str] = None
    success: bool = True
    
    def __post_init__(self):
        """Set success based on error status."""
        if self.error is not None:
            self.success = False


class DataSourceAdapter:
    """Base class for data source adapters."""
    
    def __init__(self, name: str, config: Dict[str, Any]):
        self.name = name
        self.config = config
        self.logger = logging.getLogger(f"data_adapter.{name}")
        self.enabled = config.get('enabled', True)
        self.priority = config.get('priority', 5)
        self.last_request_time = {}
        self.rate_limit_info = {
            'requests_per_minute': config.get('rate_limit', 60),
            'requests_made': 0,
            'last_reset': datetime.now()
        }
        
    def can_handle(self, request: DataRequest) -> bool:
        """Check if this adapter can handle the request."""
        return True
        
    def is_rate_limited(self) -> bool:
        """Check if adapter is currently rate limited."""
        now = datetime.now()
        time_diff = (now - self.rate_limit_info['last_reset']).total_seconds()
        
        if time_diff >= 60:  # Reset every minute
            self.rate_limit_info['requests_made'] = 0
            self.rate_limit_info['last_reset'] = now
            
        return self.rate_limit_info['requests_made'] >= self.rate_limit_info['requests_per_minute']
        
    def record_request(self):
        """Record a request for rate limiting."""
        self.rate_limit_info['requests_made'] += 1
        
    async def fetch_data(self, request: DataRequest) -> DataResponse:
        """Fetch data from the source. Must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement fetch_data method")