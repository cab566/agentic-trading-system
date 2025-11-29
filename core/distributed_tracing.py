"""
Distributed Tracing System with OpenTelemetry
Provides comprehensive observability across the trading system
"""

import asyncio
import logging
import time
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Callable, Union
from dataclasses import dataclass, field
from contextlib import contextmanager, asynccontextmanager
from functools import wraps
import json
import threading
from collections import defaultdict, deque

# OpenTelemetry imports (would be installed via pip)
try:
    from opentelemetry import trace, baggage, context
    from opentelemetry.exporter.jaeger.thrift import JaegerExporter
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.instrumentation.aiohttp_client import AioHttpClientInstrumentor
    from opentelemetry.instrumentation.asyncpg import AsyncPGInstrumentor
    from opentelemetry.instrumentation.redis import RedisInstrumentor
    OPENTELEMETRY_AVAILABLE = True
except ImportError:
    OPENTELEMETRY_AVAILABLE = False


class SpanStatus:
    """Span status constants"""
    OK = "OK"
    ERROR = "ERROR"
    TIMEOUT = "TIMEOUT"
    CANCELLED = "CANCELLED"


@dataclass
class TraceContext:
    """Trace context information"""
    trace_id: str
    span_id: str
    parent_span_id: Optional[str] = None
    baggage: Dict[str, str] = field(default_factory=dict)
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    duration_ms: Optional[float] = None
    status: str = SpanStatus.OK
    tags: Dict[str, Any] = field(default_factory=dict)
    logs: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class Span:
    """Custom span implementation for when OpenTelemetry is not available"""
    operation_name: str
    trace_context: TraceContext
    service_name: str = "trading-system"
    component: str = "unknown"
    
    def set_tag(self, key: str, value: Any):
        """Set span tag"""
        self.trace_context.tags[key] = value
    
    def set_status(self, status: str):
        """Set span status"""
        self.trace_context.status = status
    
    def log(self, message: str, level: str = "INFO", **kwargs):
        """Add log entry to span"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'level': level,
            'message': message,
            **kwargs
        }
        self.trace_context.logs.append(log_entry)
    
    def finish(self):
        """Finish the span"""
        self.trace_context.end_time = datetime.now()
        if self.trace_context.start_time:
            duration = self.trace_context.end_time - self.trace_context.start_time
            self.trace_context.duration_ms = duration.total_seconds() * 1000


class TracingManager:
    """Manages distributed tracing across the trading system"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.service_name = config.get('service_name', 'trading-system')
        self.jaeger_endpoint = config.get('jaeger_endpoint', 'http://localhost:14268/api/traces')
        
        # Span storage for custom implementation
        self.active_spans: Dict[str, Span] = {}
        self.completed_spans: deque = deque(maxlen=10000)
        self.span_metrics: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            'count': 0,
            'total_duration': 0,
            'error_count': 0,
            'avg_duration': 0
        })
        
        # Thread-local storage for current span
        self.local = threading.local()
        
        # Initialize tracing
        self.tracer = None
        self._initialize_tracing()
    
    def _initialize_tracing(self):
        """Initialize OpenTelemetry tracing or fallback to custom implementation"""
        if OPENTELEMETRY_AVAILABLE and self.config.get('use_opentelemetry', True):
            try:
                self._setup_opentelemetry()
                self.logger.info("OpenTelemetry tracing initialized")
            except Exception as e:
                self.logger.warning(f"Failed to initialize OpenTelemetry: {e}, falling back to custom tracing")
                self._setup_custom_tracing()
        else:
            self.logger.info("Using custom tracing implementation")
            self._setup_custom_tracing()
    
    def _setup_opentelemetry(self):
        """Setup OpenTelemetry with Jaeger exporter"""
        resource = Resource.create({
            "service.name": self.service_name,
            "service.version": "1.0.0"
        })
        
        trace.set_tracer_provider(TracerProvider(resource=resource))
        
        # Configure Jaeger exporter
        jaeger_exporter = JaegerExporter(
            agent_host_name=self.config.get('jaeger_host', 'localhost'),
            agent_port=self.config.get('jaeger_port', 6831),
        )
        
        span_processor = BatchSpanProcessor(jaeger_exporter)
        trace.get_tracer_provider().add_span_processor(span_processor)
        
        self.tracer = trace.get_tracer(__name__)
        
        # Auto-instrument common libraries
        if self.config.get('auto_instrument', True):
            AioHttpClientInstrumentor().instrument()
            AsyncPGInstrumentor().instrument()
            RedisInstrumentor().instrument()
    
    def _setup_custom_tracing(self):
        """Setup custom tracing implementation"""
        self.tracer = self  # Use self as tracer for custom implementation
    
    def start_span(self, operation_name: str, parent_context: Optional[TraceContext] = None, 
                   tags: Dict[str, Any] = None, component: str = "unknown") -> Span:
        """Start a new span"""
        if OPENTELEMETRY_AVAILABLE and hasattr(self.tracer, 'start_span'):
            # Use OpenTelemetry
            otel_span = self.tracer.start_span(operation_name)
            if tags:
                for key, value in tags.items():
                    otel_span.set_attribute(key, value)
            return otel_span
        else:
            # Use custom implementation
            trace_id = parent_context.trace_id if parent_context else str(uuid.uuid4())
            span_id = str(uuid.uuid4())
            parent_span_id = parent_context.span_id if parent_context else None
            
            trace_context = TraceContext(
                trace_id=trace_id,
                span_id=span_id,
                parent_span_id=parent_span_id,
                tags=tags or {}
            )
            
            span = Span(
                operation_name=operation_name,
                trace_context=trace_context,
                service_name=self.service_name,
                component=component
            )
            
            self.active_spans[span_id] = span
            self._set_current_span(span)
            
            return span
    
    def _set_current_span(self, span: Span):
        """Set current span in thread-local storage"""
        self.local.current_span = span
    
    def get_current_span(self) -> Optional[Span]:
        """Get current active span"""
        return getattr(self.local, 'current_span', None)
    
    def finish_span(self, span: Span):
        """Finish a span and record metrics"""
        if hasattr(span, 'finish'):
            span.finish()
        
        if hasattr(span, 'trace_context'):
            # Custom span
            span_id = span.trace_context.span_id
            if span_id in self.active_spans:
                del self.active_spans[span_id]
            
            self.completed_spans.append(span)
            
            # Update metrics
            metrics = self.span_metrics[span.operation_name]
            metrics['count'] += 1
            
            if span.trace_context.duration_ms:
                metrics['total_duration'] += span.trace_context.duration_ms
                metrics['avg_duration'] = metrics['total_duration'] / metrics['count']
            
            if span.trace_context.status == SpanStatus.ERROR:
                metrics['error_count'] += 1
    
    @contextmanager
    def trace(self, operation_name: str, tags: Dict[str, Any] = None, component: str = "unknown"):
        """Context manager for tracing operations"""
        current_span = self.get_current_span()
        parent_context = current_span.trace_context if current_span else None
        
        span = self.start_span(operation_name, parent_context, tags, component)
        
        try:
            yield span
        except Exception as e:
            if hasattr(span, 'set_status'):
                span.set_status(SpanStatus.ERROR)
            if hasattr(span, 'set_tag'):
                span.set_tag('error', True)
                span.set_tag('error.message', str(e))
            raise
        finally:
            self.finish_span(span)
    
    @asynccontextmanager
    async def async_trace(self, operation_name: str, tags: Dict[str, Any] = None, component: str = "unknown"):
        """Async context manager for tracing operations"""
        current_span = self.get_current_span()
        parent_context = current_span.trace_context if current_span else None
        
        span = self.start_span(operation_name, parent_context, tags, component)
        
        try:
            yield span
        except Exception as e:
            if hasattr(span, 'set_status'):
                span.set_status(SpanStatus.ERROR)
            if hasattr(span, 'set_tag'):
                span.set_tag('error', True)
                span.set_tag('error.message', str(e))
            raise
        finally:
            self.finish_span(span)
    
    def trace_function(self, operation_name: str = None, tags: Dict[str, Any] = None, component: str = "unknown"):
        """Decorator for tracing functions"""
        def decorator(func):
            op_name = operation_name or f"{func.__module__}.{func.__name__}"
            
            if asyncio.iscoroutinefunction(func):
                @wraps(func)
                async def async_wrapper(*args, **kwargs):
                    async with self.async_trace(op_name, tags, component) as span:
                        if hasattr(span, 'set_tag'):
                            span.set_tag('function.name', func.__name__)
                            span.set_tag('function.module', func.__module__)
                        return await func(*args, **kwargs)
                return async_wrapper
            else:
                @wraps(func)
                def sync_wrapper(*args, **kwargs):
                    with self.trace(op_name, tags, component) as span:
                        if hasattr(span, 'set_tag'):
                            span.set_tag('function.name', func.__name__)
                            span.set_tag('function.module', func.__module__)
                        return func(*args, **kwargs)
                return sync_wrapper
        
        return decorator
    
    def trace_database_operation(self, operation: str, table: str = None, query: str = None):
        """Specialized tracing for database operations"""
        tags = {
            'db.operation': operation,
            'db.type': 'postgresql'
        }
        if table:
            tags['db.table'] = table
        if query:
            tags['db.query'] = query[:200]  # Truncate long queries
        
        return self.trace(f"db.{operation}", tags, "database")
    
    def trace_http_request(self, method: str, url: str, status_code: int = None):
        """Specialized tracing for HTTP requests"""
        tags = {
            'http.method': method,
            'http.url': url,
            'component': 'http'
        }
        if status_code:
            tags['http.status_code'] = status_code
        
        return self.trace(f"http.{method.lower()}", tags, "http")
    
    def trace_trading_operation(self, operation: str, symbol: str = None, quantity: float = None):
        """Specialized tracing for trading operations"""
        tags = {
            'trading.operation': operation,
            'component': 'trading'
        }
        if symbol:
            tags['trading.symbol'] = symbol
        if quantity:
            tags['trading.quantity'] = quantity
        
        return self.trace(f"trading.{operation}", tags, "trading")
    
    def get_trace_metrics(self) -> Dict[str, Any]:
        """Get tracing metrics and statistics"""
        return {
            'active_spans': len(self.active_spans),
            'completed_spans': len(self.completed_spans),
            'span_metrics': dict(self.span_metrics),
            'service_name': self.service_name
        }
    
    def get_recent_traces(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent completed traces"""
        recent_spans = list(self.completed_spans)[-limit:]
        return [
            {
                'operation_name': span.operation_name,
                'trace_id': span.trace_context.trace_id,
                'span_id': span.trace_context.span_id,
                'parent_span_id': span.trace_context.parent_span_id,
                'start_time': span.trace_context.start_time.isoformat(),
                'end_time': span.trace_context.end_time.isoformat() if span.trace_context.end_time else None,
                'duration_ms': span.trace_context.duration_ms,
                'status': span.trace_context.status,
                'tags': span.trace_context.tags,
                'component': span.component
            }
            for span in recent_spans
            if hasattr(span, 'trace_context')
        ]
    
    def export_traces(self, format: str = 'json') -> str:
        """Export traces in specified format"""
        traces = self.get_recent_traces()
        
        if format.lower() == 'json':
            return json.dumps(traces, indent=2, default=str)
        else:
            raise ValueError(f"Unsupported export format: {format}")


# Global tracing manager instance
tracing_manager = None


def get_tracing_manager() -> TracingManager:
    """Get global tracing manager instance"""
    global tracing_manager
    if tracing_manager is None:
        config = {
            'service_name': 'trading-system',
            'jaeger_endpoint': 'http://localhost:14268/api/traces',
            'jaeger_host': 'localhost',
            'jaeger_port': 6831,
            'use_opentelemetry': True,
            'auto_instrument': True
        }
        tracing_manager = TracingManager(config)
    
    return tracing_manager


# Convenience functions
def trace(operation_name: str, tags: Dict[str, Any] = None, component: str = "unknown"):
    """Convenience function for tracing"""
    return get_tracing_manager().trace(operation_name, tags, component)


def async_trace(operation_name: str, tags: Dict[str, Any] = None, component: str = "unknown"):
    """Convenience function for async tracing"""
    return get_tracing_manager().async_trace(operation_name, tags, component)


def trace_function(operation_name: str = None, tags: Dict[str, Any] = None, component: str = "unknown"):
    """Convenience decorator for tracing functions"""
    return get_tracing_manager().trace_function(operation_name, tags, component)


# Example usage
@trace_function(component="trading")
async def example_trading_function(symbol: str, quantity: float):
    """Example trading function with tracing"""
    with trace("validate_order", {"symbol": symbol, "quantity": quantity}, "validation"):
        # Validation logic
        await asyncio.sleep(0.1)
    
    async with async_trace("execute_order", {"symbol": symbol}, "execution"):
        # Order execution logic
        await asyncio.sleep(0.2)
        return {"order_id": "12345", "status": "filled"}


@trace_function(component="database")
async def example_database_query(table: str, query: str):
    """Example database query with tracing"""
    with get_tracing_manager().trace_database_operation("SELECT", table, query):
        # Database query logic
        await asyncio.sleep(0.05)
        return [{"id": 1, "data": "example"}]


if __name__ == "__main__":
    # Test tracing functionality
    async def test_tracing():
        logging.basicConfig(level=logging.INFO)
        
        # Test basic tracing
        result = await example_trading_function("AAPL", 100)
        print("Trading result:", result)
        
        # Test database tracing
        db_result = await example_database_query("trades", "SELECT * FROM trades")
        print("Database result:", db_result)
        
        # Print metrics
        manager = get_tracing_manager()
        print("Trace metrics:", manager.get_trace_metrics())
        print("Recent traces:", len(manager.get_recent_traces()))
    
    asyncio.run(test_tracing())