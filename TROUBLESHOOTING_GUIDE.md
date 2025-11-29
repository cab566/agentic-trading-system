# Troubleshooting Guide 🔧

Comprehensive troubleshooting guide for the Trading System v2.0. This guide covers common issues, diagnostic steps, and solutions.

## 🚨 Quick Diagnostics

### System Health Check
```bash
# Check overall system status
curl http://localhost:8000/api/v1/health

# Check Docker containers
docker-compose ps

# Check system resources
docker stats --no-stream

# Check logs for errors
docker-compose logs --tail=50
```

### Emergency Commands
```bash
# Emergency stop all trading
curl -X POST http://localhost:8000/api/v1/emergency/stop

# Restart the entire system
docker-compose restart

# Force rebuild and restart
docker-compose down && docker-compose up --build -d
```

## 🐳 Docker Issues

### Container Won't Start

#### Symptoms
- Container exits immediately
- "Exited (1)" status
- Build failures

#### Diagnosis
```bash
# Check container logs
docker-compose logs [service_name]

# Check Docker daemon
docker info

# Check available resources
docker system df
docker system prune  # Clean up if needed
```

#### Solutions

**1. Port Conflicts**
```bash
# Check what's using port 8000
lsof -i :8000

# Kill process using the port
kill -9 $(lsof -t -i:8000)

# Or change port in docker-compose.yml
ports:
  - "8001:8000"  # Use different external port
```

**2. Memory Issues**
```bash
# Increase Docker memory limit (Docker Desktop)
# Settings > Resources > Memory > Increase to 8GB+

# Check current memory usage
docker stats --no-stream
```

**3. Build Cache Issues**
```bash
# Clear Docker build cache
docker builder prune -a

# Rebuild without cache
docker-compose build --no-cache
```

### Container Keeps Restarting

#### Symptoms
- Container status shows "Restarting"
- Frequent restarts in logs

#### Diagnosis
```bash
# Check restart count
docker-compose ps

# Monitor restart events
docker events --filter container=trading-system

# Check resource limits
docker inspect trading-system | jq '.[0].HostConfig.Memory'
```

#### Solutions

**1. Application Crashes**
```bash
# Check application logs
docker-compose logs trading-system --tail=100

# Common issues to look for:
# - Import errors
# - Configuration errors
# - Database connection failures
# - API key issues
```

**2. Health Check Failures**
```bash
# Check health check endpoint
curl http://localhost:8000/health

# Disable health check temporarily (docker-compose.yml)
healthcheck:
  disable: true
```

### Volume Mount Issues

#### Symptoms
- Configuration files not found
- Data not persisting
- Permission errors

#### Solutions
```bash
# Check volume mounts
docker-compose config

# Fix permissions (macOS/Linux)
sudo chown -R $(whoami):$(whoami) ./data
sudo chmod -R 755 ./data

# Recreate volumes
docker-compose down -v
docker-compose up -d
```

## 🔌 API Issues

### API Not Responding

#### Symptoms
- Connection refused errors
- Timeout errors
- 502/503 errors

#### Diagnosis
```bash
# Check if API is running
curl -I http://localhost:8000/health

# Check container status
docker-compose ps trading-system

# Check port binding
docker port trading-system
```

#### Solutions

**1. Service Not Started**
```bash
# Start the service
docker-compose up trading-system -d

# Check startup logs
docker-compose logs trading-system --follow
```

**2. Port Issues**
```bash
# Check port configuration
grep -r "8000" docker-compose*.yml

# Test different port
curl http://localhost:8001/health
```

### API Errors (4xx/5xx)

#### Common API Errors

**401 Unauthorized**
```bash
# Check API key configuration
curl http://localhost:8000/api/v1/config | jq '.api_keys'

# Verify environment variables
docker-compose exec trading-system env | grep API_KEY
```

**500 Internal Server Error**
```bash
# Check application logs
docker-compose logs trading-system --tail=50

# Common causes:
# - Database connection issues
# - Missing environment variables
# - Broker API failures
```

**429 Rate Limited**
```bash
# Check rate limiting configuration
curl http://localhost:8000/api/v1/config | jq '.rate_limits'

# Wait and retry, or adjust rate limits in config
```

## 📊 Data Feed Issues

### No Market Data

#### Symptoms
- Empty price responses
- Stale timestamps
- "No data available" errors

#### Diagnosis
```bash
# Check data sources
curl http://localhost:8000/api/v1/data/sources

# Check latest prices
curl http://localhost:8000/api/v1/data/prices/AAPL

# Check WebSocket connection
wscat -c ws://localhost:8000/ws/data
```

#### Solutions

**1. API Key Issues**
```bash
# Verify API keys in environment
docker-compose exec trading-system env | grep -E "(ALPHA_VANTAGE|POLYGON|IEX)"

# Test API key directly
curl "https://api.polygon.io/v2/aggs/ticker/AAPL/prev?apikey=YOUR_KEY"
```

**2. Rate Limiting**
```bash
# Check rate limit status
curl http://localhost:8000/api/v1/data/rate_limits

# Implement backoff strategy or upgrade API plan
```

**3. Market Hours**
```bash
# Check if markets are open
curl http://localhost:8000/api/v1/market/status

# For testing, use extended hours data or paper trading
```

### Data Quality Issues

#### Symptoms
- Inconsistent prices
- Missing data points
- Delayed updates

#### Solutions
```bash
# Check data validation settings
curl http://localhost:8000/api/v1/config | jq '.data_validation'

# Enable data quality monitoring
curl -X POST http://localhost:8000/api/v1/monitoring/data_quality/enable

# Review data source priorities
curl http://localhost:8000/api/v1/data/sources | jq '.priorities'
```

## 🤖 Trading Agent Issues

### Agents Not Making Decisions

#### Symptoms
- No trades being executed
- Agents showing "idle" status
- No strategy signals

#### Diagnosis
```bash
# Check agent status
curl http://localhost:8000/api/v1/agents/status

# Check strategy status
curl http://localhost:8000/api/v1/strategies/status

# Check recent decisions
curl http://localhost:8000/api/v1/agents/decisions?limit=10
```

#### Solutions

**1. Agents Disabled**
```bash
# Enable all agents
curl -X POST http://localhost:8000/api/v1/agents/enable_all

# Enable specific agent
curl -X POST http://localhost:8000/api/v1/agents/momentum_agent/enable
```

**2. No Market Opportunities**
```bash
# Check market conditions
curl http://localhost:8000/api/v1/market/conditions

# Lower strategy thresholds for testing
curl -X POST http://localhost:8000/api/v1/strategies/momentum/config \
  -H "Content-Type: application/json" \
  -d '{"entry_threshold": 0.01}'
```

**3. Risk Limits Exceeded**
```bash
# Check risk status
curl http://localhost:8000/api/v1/risk/status

# Adjust risk limits
curl -X POST http://localhost:8000/api/v1/risk/limits \
  -H "Content-Type: application/json" \
  -d '{"max_position_size": 0.10}'
```

### Strategy Performance Issues

#### Symptoms
- Consistent losses
- High drawdowns
- Poor Sharpe ratios

#### Analysis
```bash
# Get performance metrics
curl http://localhost:8000/api/v1/performance/summary

# Analyze recent trades
curl http://localhost:8000/api/v1/trades?limit=50 | jq '.[] | {symbol, pnl, timestamp}'

# Check strategy parameters
curl http://localhost:8000/api/v1/strategies/config
```

#### Solutions

**1. Parameter Optimization**
```bash
# Run parameter optimization
curl -X POST http://localhost:8000/api/v1/optimization/start \
  -H "Content-Type: application/json" \
  -d '{"strategy": "momentum", "lookback_days": 30}'
```

**2. Market Regime Analysis**
```bash
# Check market regime
curl http://localhost:8000/api/v1/market/regime

# Adjust strategy for current conditions
curl -X POST http://localhost:8000/api/v1/strategies/momentum/adapt_to_regime
```

## 💾 Database Issues

### Database Connection Errors

#### Symptoms
- "Connection refused" errors
- "Database not found" errors
- Slow query responses

#### Diagnosis
```bash
# Check database container
docker-compose ps postgres

# Check database logs
docker-compose logs postgres

# Test database connection
docker-compose exec postgres psql -U trading_user -d trading_db -c "SELECT 1;"
```

#### Solutions

**1. Database Not Running**
```bash
# Start database
docker-compose up postgres -d

# Check if it's healthy
docker-compose exec postgres pg_isready
```

**2. Connection Configuration**
```bash
# Check database URL
docker-compose exec trading-system env | grep DATABASE_URL

# Verify connection parameters in .env
DATABASE_HOST=postgres
DATABASE_PORT=5432
DATABASE_NAME=trading_db
DATABASE_USER=trading_user
```

**3. Database Corruption**
```bash
# Check database integrity
docker-compose exec postgres psql -U trading_user -d trading_db -c "SELECT pg_database_size('trading_db');"

# Backup and restore if needed
docker-compose exec postgres pg_dump -U trading_user trading_db > backup.sql
```

### Performance Issues

#### Symptoms
- Slow API responses
- High database CPU usage
- Query timeouts

#### Solutions
```bash
# Check database performance
docker-compose exec postgres psql -U trading_user -d trading_db -c "
SELECT query, calls, total_time, mean_time 
FROM pg_stat_statements 
ORDER BY total_time DESC LIMIT 10;"

# Optimize queries or add indexes as needed
```

## 🔐 Security Issues

### Authentication Failures

#### Symptoms
- 401 Unauthorized errors
- Token validation failures
- Access denied messages

#### Solutions
```bash
# Check JWT configuration
curl http://localhost:8000/api/v1/config | jq '.jwt'

# Regenerate API keys
curl -X POST http://localhost:8000/api/v1/auth/regenerate_keys

# Verify environment variables
docker-compose exec trading-system env | grep -E "(JWT_SECRET|API_KEY)"
```

### SSL/TLS Issues

#### Symptoms
- Certificate errors
- HTTPS connection failures
- Mixed content warnings

#### Solutions
```bash
# Check certificate status
openssl s_client -connect localhost:8443 -servername localhost

# Regenerate certificates
./scripts/generate_ssl_certs.sh

# Update certificate paths in configuration
```

## 🌐 Network Issues

### External API Connectivity

#### Symptoms
- Broker connection failures
- Data feed timeouts
- DNS resolution errors

#### Diagnosis
```bash
# Test external connectivity
docker-compose exec trading-system curl -I https://api.alpaca.markets
docker-compose exec trading-system nslookup api.polygon.io

# Check firewall/proxy settings
docker-compose exec trading-system env | grep -E "(HTTP_PROXY|HTTPS_PROXY)"
```

#### Solutions
```bash
# Configure proxy if needed
HTTP_PROXY=http://proxy.company.com:8080
HTTPS_PROXY=http://proxy.company.com:8080

# Add to docker-compose.yml environment section
```

### WebSocket Issues

#### Symptoms
- Real-time data not updating
- WebSocket connection drops
- "Connection closed" errors

#### Solutions
```bash
# Test WebSocket connection
wscat -c ws://localhost:8000/ws/data

# Check WebSocket configuration
curl http://localhost:8000/api/v1/config | jq '.websocket'

# Restart WebSocket service
curl -X POST http://localhost:8000/api/v1/websocket/restart
```

## 📈 Performance Issues

### High CPU Usage

#### Diagnosis
```bash
# Check container resource usage
docker stats --no-stream

# Check system processes
docker-compose exec trading-system top

# Profile application
curl -X POST http://localhost:8000/api/v1/debug/profile/start
```

#### Solutions
```bash
# Increase CPU limits
# In docker-compose.yml:
deploy:
  resources:
    limits:
      cpus: '2.0'

# Optimize strategy calculations
curl -X POST http://localhost:8000/api/v1/strategies/optimize_performance
```

### High Memory Usage

#### Diagnosis
```bash
# Check memory usage
docker stats --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}"

# Check for memory leaks
curl http://localhost:8000/api/v1/debug/memory_usage
```

#### Solutions
```bash
# Increase memory limits
# In docker-compose.yml:
deploy:
  resources:
    limits:
      memory: 4G

# Clear caches
curl -X POST http://localhost:8000/api/v1/cache/clear
```

## 🔍 Monitoring and Logging

### Log Analysis

#### Finding Errors
```bash
# Search for errors in logs
docker-compose logs | grep -i error

# Search for specific patterns
docker-compose logs | grep -E "(failed|exception|timeout)"

# Get logs for specific time range
docker-compose logs --since="2024-01-01T10:00:00" --until="2024-01-01T11:00:00"
```

#### Log Levels
```bash
# Increase log verbosity
curl -X POST http://localhost:8000/api/v1/config/log_level \
  -H "Content-Type: application/json" \
  -d '{"level": "DEBUG"}'

# Check current log level
curl http://localhost:8000/api/v1/config | jq '.log_level'
```

### Metrics Collection

#### System Metrics
```bash
# Get system metrics
curl http://localhost:8000/api/v1/metrics/system

# Get trading metrics
curl http://localhost:8000/api/v1/metrics/trading

# Export metrics for external monitoring
curl http://localhost:8000/metrics  # Prometheus format
```

## 🚨 Emergency Procedures

### System Failure Recovery

#### Complete System Failure
```bash
# 1. Stop all services
docker-compose down

# 2. Check system resources
df -h  # Disk space
free -h  # Memory
top  # CPU usage

# 3. Clean up if needed
docker system prune -a

# 4. Restore from backup
./scripts/restore_backup.sh latest

# 5. Restart system
docker-compose up -d

# 6. Verify functionality
curl http://localhost:8000/health
```

#### Data Corruption Recovery
```bash
# 1. Stop trading immediately
curl -X POST http://localhost:8000/api/v1/emergency/stop

# 2. Backup current state
./scripts/backup_system.sh emergency_$(date +%Y%m%d_%H%M%S)

# 3. Restore from known good backup
./scripts/restore_backup.sh 20240101_120000

# 4. Verify data integrity
curl http://localhost:8000/api/v1/data/integrity_check

# 5. Resume operations cautiously
curl -X POST http://localhost:8000/api/v1/trading/resume
```

### Trading Halt Procedures

#### Market Emergency
```bash
# 1. Immediate halt
curl -X POST http://localhost:8000/api/v1/emergency/halt_trading

# 2. Close all positions (if safe)
curl -X POST http://localhost:8000/api/v1/positions/close_all

# 3. Cancel pending orders
curl -X POST http://localhost:8000/api/v1/orders/cancel_all

# 4. Switch to paper trading
curl -X POST http://localhost:8000/api/v1/config/trading_mode \
  -H "Content-Type: application/json" \
  -d '{"mode": "paper"}'
```

## 📞 Getting Help

### Support Channels

#### System Logs
```bash
# Generate support bundle
./scripts/generate_support_bundle.sh

# This creates a zip file with:
# - System logs
# - Configuration (sanitized)
# - Performance metrics
# - Error reports
```

#### Community Resources
- **Documentation**: Check all `.md` files in the project
- **GitHub Issues**: Search existing issues for similar problems
- **Stack Overflow**: Tag questions with `trading-system-v2`

#### Professional Support
- **Critical Issues**: Use emergency contact procedures
- **Performance Optimization**: Schedule consultation
- **Custom Development**: Contact development team

### Diagnostic Information to Collect

When reporting issues, include:

```bash
# System information
uname -a
docker --version
docker-compose --version

# Container status
docker-compose ps

# Recent logs
docker-compose logs --tail=100 > system_logs.txt

# Configuration (sanitized)
curl http://localhost:8000/api/v1/config > config.json

# Performance metrics
curl http://localhost:8000/api/v1/metrics/system > metrics.json
```

---

**🔧 This troubleshooting guide covers the most common issues. For complex problems, always start with the Quick Diagnostics section and work through the relevant category. Remember to backup your system before making significant changes.**