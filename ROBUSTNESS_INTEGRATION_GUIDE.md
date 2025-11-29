# Trading System Robustness Integration Guide

This guide provides comprehensive instructions for integrating and deploying the enhanced robustness features implemented for the trading system.

## Overview of Robustness Enhancements

The following robustness improvements have been implemented:

1. **Enhanced Database Resilience** - Master-Slave PostgreSQL with connection pooling
2. **Comprehensive Health Monitoring** - Advanced monitoring with predictive alerting
3. **Circuit Breakers & Retry Logic** - Resilient external API handling
4. **Distributed Tracing** - OpenTelemetry-based observability
5. **Auto-Scaling Infrastructure** - Dynamic resource scaling

## Prerequisites

### Required Dependencies

Add these to your `requirements.txt`:

```txt
# Database resilience
asyncpg>=0.28.0
psycopg2-binary>=2.9.0
redis>=4.5.0
redis-sentinel>=0.1.0

# Circuit breakers and retry logic
tenacity>=8.2.0
aiohttp>=3.8.0

# Distributed tracing (optional - fallback to custom implementation if not available)
opentelemetry-api>=1.20.0
opentelemetry-sdk>=1.20.0
opentelemetry-exporter-jaeger-thrift>=1.20.0
opentelemetry-instrumentation-aiohttp-client>=0.41b0
opentelemetry-instrumentation-asyncpg>=0.41b0
opentelemetry-instrumentation-redis>=0.41b0

# Auto-scaling
docker>=6.1.0
psutil>=5.9.0

# Health monitoring
numpy>=1.24.0
scikit-learn>=1.3.0
```

### Docker Services

Ensure Docker and Docker Compose are installed for the high-availability setup.

## Integration Steps

### 1. Database Resilience Integration

#### Step 1.1: Deploy High-Availability Database

```bash
# Deploy the HA database setup
docker-compose -f docker-compose.ha.yml up -d postgres-master postgres-slave pgbouncer redis-master redis-sentinel
```

#### Step 1.2: Update Database Configuration

In your main application configuration:

```python
# config/database.py
DATABASE_CONFIG = {
    'master': {
        'host': 'localhost',
        'port': 5432,  # PgBouncer port
        'database': 'trading_db',
        'user': 'trading_user',
        'password': 'your_password'
    },
    'slave': {
        'host': 'localhost',
        'port': 5433,  # Direct slave connection
        'database': 'trading_db',
        'user': 'trading_user',
        'password': 'your_password'
    },
    'redis_sentinels': [
        ('localhost', 26379),
        ('localhost', 26380),
        ('localhost', 26381)
    ],
    'redis_service_name': 'mymaster'
}
```

#### Step 1.3: Replace Database Manager

Update your existing database connections to use the new HA manager:

```python
# In your main application file
from core.database_ha import DatabaseHAManager

# Replace existing database initialization
db_manager = DatabaseHAManager(DATABASE_CONFIG)
await db_manager.initialize()

# Use read/write splitting
# For read operations
result = await db_manager.execute_read_query("SELECT * FROM trades WHERE symbol = $1", "AAPL")

# For write operations
await db_manager.execute_write_query("INSERT INTO trades (symbol, quantity, price) VALUES ($1, $2, $3)", 
                                    "AAPL", 100, 150.0)
```

### 2. Health Monitoring Integration

#### Step 2.1: Initialize Health Monitor

```python
# In your main application startup
from core.health_monitor import HealthMonitor

health_config = {
    'database': DATABASE_CONFIG,
    'alert_thresholds': {
        'cpu_percent': 80.0,
        'memory_percent': 85.0,
        'disk_percent': 90.0,
        'response_time_ms': 1000.0
    },
    'notification_channels': {
        'email': {
            'enabled': True,
            'smtp_server': 'smtp.gmail.com',
            'smtp_port': 587,
            'username': 'your_email@gmail.com',
            'password': 'your_app_password',
            'recipients': ['admin@yourcompany.com']
        },
        'slack': {
            'enabled': True,
            'webhook_url': 'https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK'
        }
    }
}

health_monitor = HealthMonitor(health_config)
await health_monitor.start_monitoring()
```

#### Step 2.2: Add Health Endpoints

```python
# Add to your FastAPI/Flask application
@app.get("/health")
async def health_check():
    status = await health_monitor.get_system_health()
    return status

@app.get("/health/detailed")
async def detailed_health():
    return await health_monitor.get_performance_summary()
```

### 3. Circuit Breakers Integration

#### Step 3.1: Configure Circuit Breakers

```python
from core.circuit_breaker import CircuitBreakerManager, ResilientHttpClient

# Initialize circuit breaker manager
cb_config = {
    'default': {
        'failure_threshold': 5,
        'recovery_timeout': 60,
        'expected_exception': Exception
    },
    'api_calls': {
        'failure_threshold': 3,
        'recovery_timeout': 30,
        'expected_exception': (ConnectionError, TimeoutError)
    }
}

circuit_breaker_manager = CircuitBreakerManager(cb_config)
```

#### Step 3.2: Use Resilient HTTP Client

```python
# Replace existing HTTP clients
http_client = ResilientHttpClient(
    circuit_breaker_manager=circuit_breaker_manager,
    default_timeout=30.0
)

# Use for external API calls
try:
    response = await http_client.get("https://api.example.com/data")
    data = response.json()
except Exception as e:
    logger.error(f"API call failed: {e}")
```

#### Step 3.3: Add Circuit Breaker Decorators

```python
from core.circuit_breaker import circuit_breaker

@circuit_breaker(name="external_api", failure_threshold=3)
async def call_external_api(symbol: str):
    # Your external API call logic
    pass

@circuit_breaker(name="database_operation", failure_threshold=5)
async def complex_database_operation():
    # Your database operation logic
    pass
```

### 4. Distributed Tracing Integration

#### Step 4.1: Initialize Tracing

```python
from core.distributed_tracing import get_tracing_manager, trace_function

# Initialize tracing (done automatically on first use)
tracing_manager = get_tracing_manager()
```

#### Step 4.2: Add Tracing to Functions

```python
# Use decorators for automatic tracing
@trace_function(component="trading")
async def execute_trade(symbol: str, quantity: float, price: float):
    # Your trading logic
    pass

@trace_function(component="data")
async def fetch_market_data(symbol: str):
    # Your data fetching logic
    pass
```

#### Step 4.3: Manual Tracing

```python
from core.distributed_tracing import trace, async_trace

# Synchronous operations
with trace("validate_order", {"symbol": "AAPL"}, "validation"):
    # Validation logic
    pass

# Asynchronous operations
async with async_trace("execute_order", {"symbol": "AAPL"}, "execution"):
    # Order execution logic
    pass
```

### 5. Auto-Scaling Integration

#### Step 5.1: Configure Auto-Scaling

```python
from core.auto_scaling import AutoScaler

scaling_config = {
    'monitoring_interval': 30,
    'scaling_rules': [
        {
            'name': 'cpu_scaling',
            'resource_type': 'cpu',
            'threshold_up': 80.0,
            'threshold_down': 30.0,
            'duration_minutes': 5,
            'cooldown_minutes': 10,
            'min_instances': 1,
            'max_instances': 5,
            'enabled': True
        }
    ],
    'services': [
        {
            'name': 'trading-system',
            'image': 'trading-system:latest',
            'port': 8000,
            'min_instances': 1,
            'max_instances': 5
        }
    ]
}

auto_scaler = AutoScaler(scaling_config)
await auto_scaler.start()
```

#### Step 5.2: Add Scaling Endpoints

```python
@app.get("/scaling/status")
async def scaling_status():
    return auto_scaler.get_scaling_status()

@app.get("/scaling/metrics")
async def scaling_metrics():
    return auto_scaler.get_resource_metrics()
```

## Deployment Guide

### Production Deployment

#### Step 1: Environment Setup

```bash
# Create production environment file
cat > .env.production << EOF
# Database
DB_MASTER_HOST=your-master-db-host
DB_SLAVE_HOST=your-slave-db-host
DB_PASSWORD=your-secure-password

# Redis
REDIS_SENTINELS=sentinel1:26379,sentinel2:26379,sentinel3:26379
REDIS_SERVICE_NAME=mymaster

# Monitoring
HEALTH_CHECK_INTERVAL=30
ALERT_EMAIL=admin@yourcompany.com
SLACK_WEBHOOK=https://hooks.slack.com/services/YOUR/WEBHOOK

# Tracing
JAEGER_ENDPOINT=http://jaeger:14268/api/traces
ENABLE_TRACING=true

# Scaling
ENABLE_AUTO_SCALING=true
MAX_INSTANCES=10
EOF
```

#### Step 2: Deploy Infrastructure

```bash
# Deploy the complete HA stack
docker-compose -f docker-compose.ha.yml up -d

# Wait for services to be ready
sleep 30

# Verify database replication
docker exec postgres-master psql -U trading_user -d trading_db -c "SELECT * FROM pg_stat_replication;"
```

#### Step 3: Deploy Application

```bash
# Build and deploy your application with robustness features
docker build -t trading-system:robust .
docker-compose -f docker-compose.ha.yml up -d trading-system
```

### Monitoring and Observability

#### Jaeger Tracing Setup

```bash
# Deploy Jaeger for distributed tracing
docker run -d --name jaeger \
  -p 16686:16686 \
  -p 14268:14268 \
  jaegertracing/all-in-one:latest
```

Access Jaeger UI at: http://localhost:16686

#### Grafana Dashboard Setup

```bash
# Deploy Grafana for metrics visualization
docker run -d --name grafana \
  -p 3000:3000 \
  grafana/grafana:latest
```

Access Grafana at: http://localhost:3000 (admin/admin)

## Configuration Reference

### Database HA Configuration

```python
DATABASE_HA_CONFIG = {
    'master': {
        'host': 'localhost',
        'port': 5432,
        'database': 'trading_db',
        'user': 'trading_user',
        'password': 'password',
        'min_connections': 5,
        'max_connections': 20
    },
    'slave': {
        'host': 'localhost',
        'port': 5433,
        'database': 'trading_db',
        'user': 'trading_user',
        'password': 'password',
        'min_connections': 3,
        'max_connections': 15
    },
    'circuit_breaker': {
        'failure_threshold': 5,
        'recovery_timeout': 60
    },
    'redis_sentinels': [
        ('localhost', 26379),
        ('localhost', 26380),
        ('localhost', 26381)
    ],
    'redis_service_name': 'mymaster',
    'health_check_interval': 30
}
```

### Health Monitoring Configuration

```python
HEALTH_CONFIG = {
    'monitoring_interval': 30,
    'alert_thresholds': {
        'cpu_percent': 80.0,
        'memory_percent': 85.0,
        'disk_percent': 90.0,
        'response_time_ms': 1000.0,
        'error_rate_percent': 5.0
    },
    'predictive_alerting': {
        'enabled': True,
        'lookback_minutes': 60,
        'prediction_minutes': 15,
        'anomaly_threshold': 2.0
    },
    'notification_channels': {
        'email': {
            'enabled': True,
            'smtp_server': 'smtp.gmail.com',
            'smtp_port': 587,
            'username': 'alerts@yourcompany.com',
            'password': 'app_password',
            'recipients': ['admin@yourcompany.com', 'ops@yourcompany.com']
        },
        'slack': {
            'enabled': True,
            'webhook_url': 'https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK',
            'channel': '#trading-alerts'
        }
    }
}
```

## Testing the Integration

### Unit Tests

```python
import pytest
from core.database_ha import DatabaseHAManager
from core.health_monitor import HealthMonitor
from core.circuit_breaker import CircuitBreakerManager

@pytest.mark.asyncio
async def test_database_ha():
    config = {...}  # Your test config
    db_manager = DatabaseHAManager(config)
    await db_manager.initialize()
    
    # Test read/write splitting
    result = await db_manager.execute_read_query("SELECT 1")
    assert result is not None

@pytest.mark.asyncio
async def test_health_monitoring():
    config = {...}  # Your test config
    health_monitor = HealthMonitor(config)
    
    status = await health_monitor.get_system_health()
    assert status['status'] in ['healthy', 'degraded', 'unhealthy']

def test_circuit_breaker():
    cb_manager = CircuitBreakerManager({})
    cb = cb_manager.get_circuit_breaker('test')
    
    # Test circuit breaker functionality
    assert cb.state == 'closed'
```

### Integration Tests

```bash
# Run integration tests
python -m pytest tests/integration/ -v

# Test database failover
python tests/test_database_failover.py

# Test auto-scaling
python tests/test_auto_scaling.py
```

## Troubleshooting

### Common Issues

1. **Database Connection Issues**
   ```bash
   # Check database connectivity
   docker exec postgres-master pg_isready -U trading_user
   docker exec postgres-slave pg_isready -U trading_user
   ```

2. **Redis Sentinel Issues**
   ```bash
   # Check sentinel status
   docker exec redis-sentinel redis-cli -p 26379 sentinel masters
   ```

3. **Circuit Breaker Not Working**
   - Check circuit breaker configuration
   - Verify exception types match expected exceptions
   - Check logs for circuit breaker state changes

4. **Tracing Not Appearing**
   - Verify Jaeger is running and accessible
   - Check OpenTelemetry configuration
   - Ensure tracing is enabled in configuration

### Performance Tuning

1. **Database Connection Pooling**
   - Adjust `min_connections` and `max_connections` based on load
   - Monitor connection pool utilization

2. **Circuit Breaker Tuning**
   - Adjust `failure_threshold` based on service reliability
   - Tune `recovery_timeout` based on service recovery patterns

3. **Auto-Scaling Thresholds**
   - Monitor resource utilization patterns
   - Adjust thresholds based on application behavior
   - Consider different scaling rules for different times of day

## Maintenance

### Regular Tasks

1. **Database Maintenance**
   ```bash
   # Check replication lag
   docker exec postgres-master psql -U trading_user -d trading_db -c "SELECT * FROM pg_stat_replication;"
   
   # Vacuum and analyze
   docker exec postgres-master psql -U trading_user -d trading_db -c "VACUUM ANALYZE;"
   ```

2. **Health Check Validation**
   ```bash
   # Test health endpoints
   curl http://localhost:8000/health
   curl http://localhost:8000/health/detailed
   ```

3. **Log Rotation**
   ```bash
   # Rotate application logs
   logrotate /etc/logrotate.d/trading-system
   ```

### Monitoring Dashboards

Create monitoring dashboards for:
- Database performance and replication lag
- Circuit breaker states and failure rates
- Auto-scaling events and resource utilization
- Distributed tracing metrics
- Health check status and alerts

## Security Considerations

1. **Database Security**
   - Use strong passwords for database users
   - Enable SSL/TLS for database connections
   - Restrict network access to database ports

2. **Redis Security**
   - Configure Redis authentication
   - Use Redis ACLs for fine-grained access control
   - Enable TLS for Redis connections

3. **API Security**
   - Implement rate limiting
   - Use API keys or OAuth for external API access
   - Monitor for suspicious activity patterns

This integration guide provides a comprehensive approach to implementing all robustness features while maintaining system performance and reliability.