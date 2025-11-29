#!/usr/bin/env python3
"""
Cache Manager for Trading System

Provides caching functionality for market data and other frequently accessed data.
"""

import logging
from typing import Any, Optional, Dict
from datetime import datetime, timedelta


class CacheManager:
    """Simple in-memory cache manager."""
    
    def __init__(self, config_manager=None):
        self.config_manager = config_manager
        self.logger = logging.getLogger(__name__)
        self._cache: Dict[str, Dict[str, Any]] = {}
        self.default_ttl = 300  # 5 minutes
        
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        if key in self._cache:
            entry = self._cache[key]
            if datetime.now() < entry['expires_at']:
                return entry['value']
            else:
                # Expired, remove from cache
                del self._cache[key]
        return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value in cache with TTL."""
        ttl = ttl or self.default_ttl
        expires_at = datetime.now() + timedelta(seconds=ttl)
        self._cache[key] = {
            'value': value,
            'expires_at': expires_at
        }
    
    def delete(self, key: str) -> bool:
        """Delete key from cache."""
        if key in self._cache:
            del self._cache[key]
            return True
        return False
    
    def clear(self) -> None:
        """Clear all cache entries."""
        self._cache.clear()
    
    def cleanup_expired(self) -> int:
        """Remove expired entries and return count of removed items."""
        now = datetime.now()
        expired_keys = [
            key for key, entry in self._cache.items()
            if now >= entry['expires_at']
        ]
        
        for key in expired_keys:
            del self._cache[key]
            
        return len(expired_keys)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            'total_entries': len(self._cache),
            'cache_keys': list(self._cache.keys())
        }