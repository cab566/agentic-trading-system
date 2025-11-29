"""
Circuit Breaker and Enhanced Retry Logic
Provides fault tolerance for external API calls and service dependencies
"""

import asyncio
import logging
import time
import random
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Callable, Union
from dataclasses import dataclass
from enum import Enum
import aiohttp
import functools
from contextlib import asynccontextmanager


class CircuitBreakerState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, rejecting calls
    HALF_OPEN = "half_open"  # Testing if service recovered


class RetryStrategy(Enum):
    """Retry strategy types"""
    FIXED = "fixed"
    EXPONENTIAL = "exponential"
    LINEAR = "linear"
    JITTER = "jitter"


@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration"""
    failure_threshold: int = 5
    recovery_timeout: int = 30
    success_threshold: int = 3  # For half-open state
    timeout: float = 10.0
    expected_exception: tuple = (Exception,)


@dataclass
class RetryConfig:
    """Retry configuration"""
    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL
    backoff_multiplier: float = 2.0
    jitter: bool = True
    retryable_exceptions: tuple = (aiohttp.ClientError, asyncio.TimeoutError)


@dataclass
class CallResult:
    """Result of a circuit breaker call"""
    success: bool
    result: Any = None
    exception: Optional[Exception] = None
    duration: float = 0.0
    attempts: int = 1


class CircuitBreaker:
    """Circuit breaker implementation with retry logic"""
    
    def __init__(self, name: str, config: CircuitBreakerConfig):
        self.name = name
        self.config = config
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        self.last_success_time = None
        self.call_count = 0
        self.success_rate = 1.0
        self.logger = logging.getLogger(f"{__name__}.{name}")
        
        # Statistics
        self.total_calls = 0
        self.total_failures = 0
        self.total_successes = 0
        self.average_response_time = 0.0
        self.response_times = []
        
    def _should_attempt_call(self) -> bool:
        """Check if call should be attempted based on circuit breaker state"""
        if self.state == CircuitBreakerState.CLOSED:
            return True
        elif self.state == CircuitBreakerState.OPEN:
            if self.last_failure_time and time.time() - self.last_failure_time > self.config.recovery_timeout:
                self.state = CircuitBreakerState.HALF_OPEN
                self.success_count = 0
                self.logger.info(f"Circuit breaker {self.name} transitioning to HALF_OPEN")
                return True
            return False
        elif self.state == CircuitBreakerState.HALF_OPEN:
            return True
        
        return False
    
    def _record_success(self, duration: float):
        """Record successful call"""
        self.total_calls += 1
        self.total_successes += 1
        self.last_success_time = time.time()
        
        # Update response time statistics
        self.response_times.append(duration)
        if len(self.response_times) > 100:  # Keep last 100 response times
            self.response_times.pop(0)
        self.average_response_time = sum(self.response_times) / len(self.response_times)
        
        if self.state == CircuitBreakerState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.config.success_threshold:
                self.state = CircuitBreakerState.CLOSED
                self.failure_count = 0
                self.logger.info(f"Circuit breaker {self.name} recovered, transitioning to CLOSED")
        elif self.state == CircuitBreakerState.CLOSED:
            self.failure_count = 0
        
        # Update success rate
        self.success_rate = self.total_successes / self.total_calls if self.total_calls > 0 else 1.0
    
    def _record_failure(self, exception: Exception):
        """Record failed call"""
        self.total_calls += 1
        self.total_failures += 1
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.state == CircuitBreakerState.CLOSED:
            if self.failure_count >= self.config.failure_threshold:
                self.state = CircuitBreakerState.OPEN
                self.logger.warning(f"Circuit breaker {self.name} opened after {self.failure_count} failures")
        elif self.state == CircuitBreakerState.HALF_OPEN:
            self.state = CircuitBreakerState.OPEN
            self.logger.warning(f"Circuit breaker {self.name} failed in HALF_OPEN, returning to OPEN")
        
        # Update success rate
        self.success_rate = self.total_successes / self.total_calls if self.total_calls > 0 else 0.0
    
    async def call(self, func: Callable, *args, retry_config: Optional[RetryConfig] = None, **kwargs) -> CallResult:
        """Execute function with circuit breaker protection and retry logic"""
        if not self._should_attempt_call():
            return CallResult(
                success=False,
                exception=Exception(f"Circuit breaker {self.name} is OPEN"),
                attempts=0
            )
        
        retry_config = retry_config or RetryConfig()
        attempts = 0
        last_exception = None
        start_time = time.time()
        
        for attempt in range(retry_config.max_attempts):
            attempts += 1
            
            try:
                # Execute the function with timeout
                if asyncio.iscoroutinefunction(func):
                    result = await asyncio.wait_for(
                        func(*args, **kwargs),
                        timeout=self.config.timeout
                    )
                else:
                    result = func(*args, **kwargs)
                
                duration = time.time() - start_time
                self._record_success(duration)
                
                return CallResult(
                    success=True,
                    result=result,
                    duration=duration,
                    attempts=attempts
                )
                
            except Exception as e:
                last_exception = e
                
                # Check if exception is retryable
                if not isinstance(e, retry_config.retryable_exceptions):
                    break
                
                # Don't retry on last attempt
                if attempt == retry_config.max_attempts - 1:
                    break
                
                # Calculate delay for next attempt
                delay = self._calculate_delay(attempt, retry_config)
                self.logger.warning(f"Attempt {attempt + 1} failed for {self.name}, retrying in {delay:.2f}s: {e}")
                
                await asyncio.sleep(delay)
        
        # All attempts failed
        duration = time.time() - start_time
        self._record_failure(last_exception)
        
        return CallResult(
            success=False,
            exception=last_exception,
            duration=duration,
            attempts=attempts
        )
    
    def _calculate_delay(self, attempt: int, config: RetryConfig) -> float:
        """Calculate delay for retry attempt"""
        if config.strategy == RetryStrategy.FIXED:
            delay = config.base_delay
        elif config.strategy == RetryStrategy.LINEAR:
            delay = config.base_delay * (attempt + 1)
        elif config.strategy == RetryStrategy.EXPONENTIAL:
            delay = config.base_delay * (config.backoff_multiplier ** attempt)
        elif config.strategy == RetryStrategy.JITTER:
            base_delay = config.base_delay * (config.backoff_multiplier ** attempt)
            delay = base_delay + random.uniform(0, base_delay * 0.1)
        else:
            delay = config.base_delay
        
        # Apply jitter if enabled
        if config.jitter and config.strategy != RetryStrategy.JITTER:
            jitter_amount = delay * 0.1
            delay += random.uniform(-jitter_amount, jitter_amount)
        
        return min(delay, config.max_delay)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get circuit breaker statistics"""
        return {
            'name': self.name,
            'state': self.state.value,
            'total_calls': self.total_calls,
            'total_successes': self.total_successes,
            'total_failures': self.total_failures,
            'success_rate': self.success_rate,
            'failure_count': self.failure_count,
            'average_response_time': self.average_response_time,
            'last_failure_time': self.last_failure_time,
            'last_success_time': self.last_success_time
        }
    
    def reset(self):
        """Reset circuit breaker to closed state"""
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.logger.info(f"Circuit breaker {self.name} manually reset")


class CircuitBreakerManager:
    """Manages multiple circuit breakers"""
    
    def __init__(self):
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.logger = logging.getLogger(__name__)
    
    def create_circuit_breaker(self, name: str, config: CircuitBreakerConfig) -> CircuitBreaker:
        """Create and register a circuit breaker"""
        circuit_breaker = CircuitBreaker(name, config)
        self.circuit_breakers[name] = circuit_breaker
        self.logger.info(f"Created circuit breaker: {name}")
        return circuit_breaker
    
    def get_circuit_breaker(self, name: str) -> Optional[CircuitBreaker]:
        """Get circuit breaker by name"""
        return self.circuit_breakers.get(name)
    
    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all circuit breakers"""
        return {name: cb.get_stats() for name, cb in self.circuit_breakers.items()}
    
    def reset_all(self):
        """Reset all circuit breakers"""
        for cb in self.circuit_breakers.values():
            cb.reset()
        self.logger.info("All circuit breakers reset")


# Decorator for easy circuit breaker usage
def circuit_breaker(name: str, config: Optional[CircuitBreakerConfig] = None, retry_config: Optional[RetryConfig] = None):
    """Decorator to add circuit breaker protection to functions"""
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            cb_config = config or CircuitBreakerConfig()
            r_config = retry_config or RetryConfig()
            
            # Get or create circuit breaker
            cb = circuit_breaker_manager.get_circuit_breaker(name)
            if cb is None:
                cb = circuit_breaker_manager.create_circuit_breaker(name, cb_config)
            
            result = await cb.call(func, *args, retry_config=r_config, **kwargs)
            
            if result.success:
                return result.result
            else:
                raise result.exception
        
        return wrapper
    return decorator


# HTTP Client with circuit breaker
class ResilientHttpClient:
    """HTTP client with built-in circuit breaker and retry logic"""
    
    def __init__(self, base_url: str = "", default_timeout: float = 10.0):
        self.base_url = base_url
        self.default_timeout = default_timeout
        self.session = None
        self.circuit_breaker_manager = CircuitBreakerManager()
        self.logger = logging.getLogger(__name__)
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.default_timeout),
            connector=aiohttp.TCPConnector(
                limit=100,
                limit_per_host=30,
                keepalive_timeout=30,
                enable_cleanup_closed=True
            )
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def request(self, method: str, url: str, circuit_breaker_name: str = None, 
                     circuit_breaker_config: CircuitBreakerConfig = None,
                     retry_config: RetryConfig = None, **kwargs) -> aiohttp.ClientResponse:
        """Make HTTP request with circuit breaker protection"""
        if not self.session:
            raise RuntimeError("HTTP client not initialized. Use async context manager.")
        
        full_url = f"{self.base_url.rstrip('/')}/{url.lstrip('/')}" if self.base_url else url
        cb_name = circuit_breaker_name or f"http_{method.lower()}_{url.replace('/', '_')}"
        
        # Get or create circuit breaker
        cb = self.circuit_breaker_manager.get_circuit_breaker(cb_name)
        if cb is None:
            cb_config = circuit_breaker_config or CircuitBreakerConfig(
                failure_threshold=3,
                recovery_timeout=30,
                timeout=self.default_timeout
            )
            cb = self.circuit_breaker_manager.create_circuit_breaker(cb_name, cb_config)
        
        r_config = retry_config or RetryConfig(
            max_attempts=3,
            retryable_exceptions=(aiohttp.ClientError, asyncio.TimeoutError)
        )
        
        async def make_request():
            async with self.session.request(method, full_url, **kwargs) as response:
                response.raise_for_status()
                return response
        
        result = await cb.call(make_request, retry_config=r_config)
        
        if result.success:
            return result.result
        else:
            raise result.exception
    
    async def get(self, url: str, **kwargs) -> aiohttp.ClientResponse:
        """GET request with circuit breaker"""
        return await self.request('GET', url, **kwargs)
    
    async def post(self, url: str, **kwargs) -> aiohttp.ClientResponse:
        """POST request with circuit breaker"""
        return await self.request('POST', url, **kwargs)
    
    async def put(self, url: str, **kwargs) -> aiohttp.ClientResponse:
        """PUT request with circuit breaker"""
        return await self.request('PUT', url, **kwargs)
    
    async def delete(self, url: str, **kwargs) -> aiohttp.ClientResponse:
        """DELETE request with circuit breaker"""
        return await self.request('DELETE', url, **kwargs)
    
    def get_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all circuit breakers"""
        return self.circuit_breaker_manager.get_all_stats()


# Global circuit breaker manager
circuit_breaker_manager = CircuitBreakerManager()


# Predefined circuit breakers for common services
def create_default_circuit_breakers():
    """Create default circuit breakers for common services"""
    
    # Database circuit breaker
    circuit_breaker_manager.create_circuit_breaker(
        'database',
        CircuitBreakerConfig(
            failure_threshold=5,
            recovery_timeout=30,
            timeout=10.0
        )
    )
    
    # External API circuit breaker
    circuit_breaker_manager.create_circuit_breaker(
        'external_api',
        CircuitBreakerConfig(
            failure_threshold=3,
            recovery_timeout=60,
            timeout=15.0
        )
    )
    
    # Market data circuit breaker
    circuit_breaker_manager.create_circuit_breaker(
        'market_data',
        CircuitBreakerConfig(
            failure_threshold=3,
            recovery_timeout=30,
            timeout=5.0
        )
    )
    
    # Trading API circuit breaker
    circuit_breaker_manager.create_circuit_breaker(
        'trading_api',
        CircuitBreakerConfig(
            failure_threshold=2,
            recovery_timeout=120,
            timeout=20.0
        )
    )


# Initialize default circuit breakers
create_default_circuit_breakers()


# Example usage functions
async def example_database_call():
    """Example of using circuit breaker for database calls"""
    @circuit_breaker('database', retry_config=RetryConfig(max_attempts=3))
    async def fetch_data():
        # Simulate database call
        await asyncio.sleep(0.1)
        return "data"
    
    return await fetch_data()


async def example_api_call():
    """Example of using resilient HTTP client"""
    async with ResilientHttpClient("https://api.example.com") as client:
        response = await client.get("/data", 
                                  circuit_breaker_name="example_api",
                                  retry_config=RetryConfig(max_attempts=2))
        return await response.json()


if __name__ == "__main__":
    # Test circuit breaker functionality
    async def test_circuit_breaker():
        logging.basicConfig(level=logging.INFO)
        
        # Test basic circuit breaker
        cb = circuit_breaker_manager.get_circuit_breaker('database')
        
        async def failing_function():
            raise Exception("Simulated failure")
        
        # Test multiple failures
        for i in range(10):
            result = await cb.call(failing_function)
            print(f"Attempt {i+1}: Success={result.success}, State={cb.state.value}")
            await asyncio.sleep(1)
        
        print("Final stats:", cb.get_stats())
    
    asyncio.run(test_circuit_breaker())