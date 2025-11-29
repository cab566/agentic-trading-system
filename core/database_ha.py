"""
High-Availability Database Manager
Provides connection pooling, circuit breakers, and read/write splitting
"""

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional, Tuple, Union
from enum import Enum
import asyncpg
import redis.sentinel
from sqlalchemy import create_engine, text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import QueuePool
import psycopg2
from psycopg2 import pool


class DatabaseRole(Enum):
    """Database role enumeration"""
    MASTER = "master"
    SLAVE = "slave"


class CircuitBreakerState(Enum):
    """Circuit breaker state enumeration"""
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    """Circuit breaker for database connections"""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 30):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = CircuitBreakerState.CLOSED
        self.logger = logging.getLogger(__name__)
    
    def call(self, func, *args, **kwargs):
        """Execute function with circuit breaker protection"""
        if self.state == CircuitBreakerState.OPEN:
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = CircuitBreakerState.HALF_OPEN
                self.logger.info("Circuit breaker transitioning to HALF_OPEN")
            else:
                raise Exception("Circuit breaker is OPEN")
        
        try:
            result = func(*args, **kwargs)
            if self.state == CircuitBreakerState.HALF_OPEN:
                self.reset()
            return result
        except Exception as e:
            self.record_failure()
            raise e
    
    def record_failure(self):
        """Record a failure and update circuit breaker state"""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = CircuitBreakerState.OPEN
            self.logger.warning(f"Circuit breaker opened after {self.failure_count} failures")
    
    def reset(self):
        """Reset circuit breaker to closed state"""
        self.failure_count = 0
        self.last_failure_time = None
        self.state = CircuitBreakerState.CLOSED
        self.logger.info("Circuit breaker reset to CLOSED")


class DatabaseHAManager:
    """High-Availability Database Manager with connection pooling and failover"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Connection pools
        self.master_pool = None
        self.slave_pool = None
        self.async_master_engine = None
        self.async_slave_engine = None
        
        # Circuit breakers
        self.master_circuit_breaker = CircuitBreaker(
            failure_threshold=config.get('circuit_breaker_failure_threshold', 5),
            recovery_timeout=config.get('circuit_breaker_recovery_timeout', 30)
        )
        self.slave_circuit_breaker = CircuitBreaker(
            failure_threshold=config.get('circuit_breaker_failure_threshold', 5),
            recovery_timeout=config.get('circuit_breaker_recovery_timeout', 30)
        )
        
        # Redis Sentinel for caching
        self.redis_sentinel = None
        self.redis_master = None
        
        # Connection settings
        self.pool_size = config.get('pool_size', 20)
        self.max_overflow = config.get('max_overflow', 30)
        self.pool_timeout = config.get('pool_timeout', 30)
        self.pool_recycle = config.get('pool_recycle', 3600)
        
        # Health check settings
        self.health_check_interval = config.get('health_check_interval', 30)
        self.last_health_check = {}
        
    async def initialize(self):
        """Initialize database connections and pools"""
        try:
            await self._setup_database_pools()
            await self._setup_redis_sentinel()
            await self._start_health_monitoring()
            self.logger.info("Database HA Manager initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize Database HA Manager: {e}")
            raise
    
    async def _setup_database_pools(self):
        """Setup database connection pools for master and slave"""
        # Master database connection
        master_url = self.config.get('master_url', 'postgresql://trading_user:trading_pass@pgbouncer:5432/trading_db')
        self.master_pool = asyncpg.create_pool(
            master_url,
            min_size=self.pool_size // 2,
            max_size=self.pool_size,
            max_queries=50000,
            max_inactive_connection_lifetime=300,
            command_timeout=60
        )
        
        # Async SQLAlchemy engine for master
        self.async_master_engine = create_async_engine(
            master_url.replace('postgresql://', 'postgresql+asyncpg://'),
            poolclass=QueuePool,
            pool_size=self.pool_size,
            max_overflow=self.max_overflow,
            pool_timeout=self.pool_timeout,
            pool_recycle=self.pool_recycle,
            echo=False
        )
        
        # Slave database connection (read replica)
        slave_url = self.config.get('slave_url', 'postgresql://trading_user:trading_pass@postgres-slave:5432/trading_db')
        self.slave_pool = asyncpg.create_pool(
            slave_url,
            min_size=self.pool_size // 4,
            max_size=self.pool_size // 2,
            max_queries=50000,
            max_inactive_connection_lifetime=300,
            command_timeout=60
        )
        
        # Async SQLAlchemy engine for slave
        self.async_slave_engine = create_async_engine(
            slave_url.replace('postgresql://', 'postgresql+asyncpg://'),
            poolclass=QueuePool,
            pool_size=self.pool_size // 2,
            max_overflow=self.max_overflow // 2,
            pool_timeout=self.pool_timeout,
            pool_recycle=self.pool_recycle,
            echo=False
        )
        
        self.logger.info("Database connection pools initialized")
    
    async def _setup_redis_sentinel(self):
        """Setup Redis Sentinel for high availability caching"""
        try:
            sentinel_hosts = self.config.get('redis_sentinel_hosts', ['redis-sentinel:26379'])
            sentinel_list = [(host.split(':')[0], int(host.split(':')[1])) for host in sentinel_hosts]
            
            self.redis_sentinel = redis.sentinel.Sentinel(
                sentinel_list,
                socket_timeout=0.1,
                socket_connect_timeout=0.1,
                socket_keepalive=True,
                socket_keepalive_options={}
            )
            
            master_name = self.config.get('redis_master_name', 'mymaster')
            self.redis_master = self.redis_sentinel.master_for(
                master_name,
                socket_timeout=0.1,
                socket_connect_timeout=0.1,
                socket_keepalive=True,
                socket_keepalive_options={}
            )
            
            self.logger.info("Redis Sentinel initialized")
        except Exception as e:
            self.logger.warning(f"Redis Sentinel setup failed: {e}")
    
    async def _start_health_monitoring(self):
        """Start background health monitoring"""
        asyncio.create_task(self._health_monitor_loop())
    
    async def _health_monitor_loop(self):
        """Background health monitoring loop"""
        while True:
            try:
                await self._check_database_health()
                await asyncio.sleep(self.health_check_interval)
            except Exception as e:
                self.logger.error(f"Health monitor error: {e}")
                await asyncio.sleep(5)
    
    async def _check_database_health(self):
        """Check health of database connections"""
        current_time = time.time()
        
        # Check master health
        try:
            async with self.master_pool.acquire() as conn:
                await conn.fetchval("SELECT 1")
            self.last_health_check['master'] = current_time
            if self.master_circuit_breaker.state != CircuitBreakerState.CLOSED:
                self.master_circuit_breaker.reset()
        except Exception as e:
            self.logger.warning(f"Master database health check failed: {e}")
            self.master_circuit_breaker.record_failure()
        
        # Check slave health
        try:
            async with self.slave_pool.acquire() as conn:
                await conn.fetchval("SELECT 1")
            self.last_health_check['slave'] = current_time
            if self.slave_circuit_breaker.state != CircuitBreakerState.CLOSED:
                self.slave_circuit_breaker.reset()
        except Exception as e:
            self.logger.warning(f"Slave database health check failed: {e}")
            self.slave_circuit_breaker.record_failure()
    
    @asynccontextmanager
    async def get_connection(self, role: DatabaseRole = DatabaseRole.MASTER):
        """Get database connection with circuit breaker protection"""
        if role == DatabaseRole.MASTER:
            pool = self.master_pool
            circuit_breaker = self.master_circuit_breaker
        else:
            pool = self.slave_pool
            circuit_breaker = self.slave_circuit_breaker
        
        # Fallback to master if slave is unavailable
        if role == DatabaseRole.SLAVE and circuit_breaker.state == CircuitBreakerState.OPEN:
            self.logger.warning("Slave unavailable, falling back to master")
            pool = self.master_pool
            circuit_breaker = self.master_circuit_breaker
        
        try:
            async with pool.acquire() as conn:
                yield conn
        except Exception as e:
            circuit_breaker.record_failure()
            raise e
    
    async def execute_query(self, query: str, *args, role: DatabaseRole = DatabaseRole.SLAVE) -> List[Any]:
        """Execute a SELECT query with automatic role selection"""
        async with self.get_connection(role) as conn:
            return await conn.fetch(query, *args)
    
    async def execute_command(self, query: str, *args) -> str:
        """Execute a write command on master database"""
        async with self.get_connection(DatabaseRole.MASTER) as conn:
            return await conn.execute(query, *args)
    
    async def execute_transaction(self, queries: List[Tuple[str, tuple]]) -> List[Any]:
        """Execute multiple queries in a transaction on master"""
        async with self.get_connection(DatabaseRole.MASTER) as conn:
            async with conn.transaction():
                results = []
                for query, args in queries:
                    result = await conn.fetch(query, *args)
                    results.append(result)
                return results
    
    async def get_session(self, role: DatabaseRole = DatabaseRole.MASTER) -> AsyncSession:
        """Get SQLAlchemy async session"""
        engine = self.async_master_engine if role == DatabaseRole.MASTER else self.async_slave_engine
        
        # Fallback to master if slave is unavailable
        if role == DatabaseRole.SLAVE and self.slave_circuit_breaker.state == CircuitBreakerState.OPEN:
            engine = self.async_master_engine
        
        async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
        return async_session()
    
    async def cache_get(self, key: str) -> Optional[str]:
        """Get value from Redis cache"""
        if not self.redis_master:
            return None
        
        try:
            return self.redis_master.get(key)
        except Exception as e:
            self.logger.warning(f"Cache get failed: {e}")
            return None
    
    async def cache_set(self, key: str, value: str, ttl: int = 300):
        """Set value in Redis cache"""
        if not self.redis_master:
            return
        
        try:
            self.redis_master.setex(key, ttl, value)
        except Exception as e:
            self.logger.warning(f"Cache set failed: {e}")
    
    async def get_connection_stats(self) -> Dict[str, Any]:
        """Get connection pool statistics"""
        stats = {
            'master_pool': {
                'size': self.master_pool.get_size() if self.master_pool else 0,
                'min_size': self.master_pool.get_min_size() if self.master_pool else 0,
                'max_size': self.master_pool.get_max_size() if self.master_pool else 0,
                'circuit_breaker_state': self.master_circuit_breaker.state.value,
                'failure_count': self.master_circuit_breaker.failure_count
            },
            'slave_pool': {
                'size': self.slave_pool.get_size() if self.slave_pool else 0,
                'min_size': self.slave_pool.get_min_size() if self.slave_pool else 0,
                'max_size': self.slave_pool.get_max_size() if self.slave_pool else 0,
                'circuit_breaker_state': self.slave_circuit_breaker.state.value,
                'failure_count': self.slave_circuit_breaker.failure_count
            },
            'last_health_check': self.last_health_check
        }
        return stats
    
    async def close(self):
        """Close all database connections"""
        try:
            if self.master_pool:
                await self.master_pool.close()
            if self.slave_pool:
                await self.slave_pool.close()
            if self.async_master_engine:
                await self.async_master_engine.dispose()
            if self.async_slave_engine:
                await self.async_slave_engine.dispose()
            
            self.logger.info("Database HA Manager closed")
        except Exception as e:
            self.logger.error(f"Error closing Database HA Manager: {e}")


# Global instance
db_ha_manager = None


async def get_db_ha_manager() -> DatabaseHAManager:
    """Get global database HA manager instance"""
    global db_ha_manager
    if db_ha_manager is None:
        config = {
            'master_url': 'postgresql://trading_user:trading_pass@pgbouncer:5432/trading_db',
            'slave_url': 'postgresql://trading_user:trading_pass@postgres-slave:5432/trading_db',
            'redis_sentinel_hosts': ['redis-sentinel:26379'],
            'redis_master_name': 'mymaster',
            'pool_size': 20,
            'max_overflow': 30,
            'pool_timeout': 30,
            'pool_recycle': 3600,
            'circuit_breaker_failure_threshold': 5,
            'circuit_breaker_recovery_timeout': 30,
            'health_check_interval': 30
        }
        db_ha_manager = DatabaseHAManager(config)
        await db_ha_manager.initialize()
    
    return db_ha_manager