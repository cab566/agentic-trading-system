# System Monitoring Guide 📊

Complete guide for monitoring the Trading System v2.0 health, performance, and operations.

## 🎯 Overview

This guide covers all aspects of monitoring your trading system, from basic health checks to advanced performance analytics and alerting.

## 🏥 Health Monitoring

### System Health Dashboard

Access the main health dashboard at: **http://localhost:3000** (Grafana)

**Default Credentials:**
- Username: `admin`
- Password: `admin` (change on first login)

### Core Health Metrics

#### 1. System Status Check
```bash
# Quick health check
curl http://localhost:8000/health

# Detailed system status
curl http://localhost:8000/api/v1/status | jq .
```

**Key Indicators:**
- ✅ `status: "healthy"` - All services operational
- ⚠️ `status: "degraded"` - Some services experiencing issues
- ❌ `status: "unhealthy"` - Critical services down

#### 2. Container Health
```bash
# Check all containers
docker ps

# Check container health
docker-compose ps

# View container resource usage
docker stats
```

**Expected Containers:**
- `trading-system` - Main application
- `trading-redis` - Cache and message broker
- `trading-postgres` - Database
- `trading-nginx` - Reverse proxy
- `trading-grafana` - Monitoring dashboard
- `trading-prometheus` - Metrics collection
- `trading-fluentd` - Log aggregation

#### 3. Service Dependencies
```bash
# Test database connection
docker exec trading-postgres pg_isready

# Test Redis connection
docker exec trading-redis redis-cli ping

# Test API connectivity
curl -f http://localhost:8000/health || echo "API Down"
```

## 📈 Performance Monitoring

### Key Performance Indicators (KPIs)

#### 1. Trading Performance
- **Portfolio Value**: Current total portfolio value
- **Daily P&L**: Profit/Loss for current trading day
- **Total Return**: Cumulative return since inception
- **Sharpe Ratio**: Risk-adjusted return metric
- **Maximum Drawdown**: Largest peak-to-trough decline

#### 2. System Performance
- **API Response Time**: Average response time for API calls
- **Order Execution Latency**: Time from signal to order placement
- **Data Processing Speed**: Market data ingestion rate
- **Memory Usage**: RAM consumption across services
- **CPU Utilization**: Processing load across cores

#### 3. Strategy Performance
- **Win Rate**: Percentage of profitable trades
- **Average Trade Duration**: Time positions are held
- **Strategy Allocation**: Capital allocated per strategy
- **Signal Accuracy**: Percentage of correct predictions
- **Risk-Adjusted Returns**: Performance per unit of risk

### Monitoring Commands

#### Real-time System Metrics
```bash
# System resource usage
docker stats --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.NetIO}}"

# Application logs (real-time)
docker-compose logs -f trading-system

# Database performance
docker exec trading-postgres psql -U trading -d trading_db -c "
SELECT 
    schemaname,
    tablename,
    n_tup_ins as inserts,
    n_tup_upd as updates,
    n_tup_del as deletes
FROM pg_stat_user_tables;
"

# Redis performance
docker exec trading-redis redis-cli info stats
```

#### Portfolio Monitoring
```bash
# Current portfolio status
curl -s http://localhost:8000/api/v1/portfolio | jq '{
  total_value: .total_value,
  day_pnl: .day_pnl,
  positions: .positions | length
}'

# Active strategies
curl -s http://localhost:8000/api/v1/strategies | jq '.strategies[] | {
  name: .name,
  status: .status,
  allocation: .allocation
}'

# Recent trades
curl -s http://localhost:8000/api/v1/trades?limit=10 | jq '.trades[] | {
  symbol: .symbol,
  side: .side,
  quantity: .quantity,
  price: .price,
  timestamp: .timestamp
}'
```

## 📊 Grafana Dashboards

### Pre-configured Dashboards

#### 1. System Overview Dashboard
- **URL**: http://localhost:3000/d/system-overview
- **Metrics**: CPU, Memory, Disk, Network
- **Services**: All container health status
- **Alerts**: Critical system alerts

#### 2. Trading Performance Dashboard
- **URL**: http://localhost:3000/d/trading-performance
- **Metrics**: P&L, Portfolio value, Trade count
- **Charts**: Performance over time, drawdown analysis
- **Strategy**: Individual strategy performance

#### 3. Risk Management Dashboard
- **URL**: http://localhost:3000/d/risk-management
- **Metrics**: VaR, Position sizes, Correlation
- **Alerts**: Risk limit breaches
- **Exposure**: Asset class and geographic exposure

#### 4. Market Data Dashboard
- **URL**: http://localhost:3000/d/market-data
- **Metrics**: Data feed latency, API calls
- **Coverage**: Market hours, data quality
- **Sources**: Data provider status

### Custom Dashboard Creation

#### Creating New Dashboard
1. Navigate to Grafana: http://localhost:3000
2. Click "+" → "Dashboard"
3. Add panels with relevant queries
4. Save with descriptive name

#### Example Prometheus Queries
```promql
# CPU usage by container
rate(container_cpu_usage_seconds_total[5m]) * 100

# Memory usage percentage
(container_memory_usage_bytes / container_spec_memory_limit_bytes) * 100

# API request rate
rate(http_requests_total[5m])

# Trading system uptime
up{job="trading-system"}

# Portfolio value over time
portfolio_total_value

# Active positions count
active_positions_count
```

## 🚨 Alerting & Notifications

### Alert Configuration

#### 1. Critical System Alerts
```yaml
# prometheus/alerts.yml
groups:
  - name: trading_system
    rules:
      - alert: SystemDown
        expr: up{job="trading-system"} == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Trading system is down"
          
      - alert: HighMemoryUsage
        expr: (container_memory_usage_bytes / container_spec_memory_limit_bytes) * 100 > 90
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High memory usage detected"
          
      - alert: DatabaseConnectionFailed
        expr: postgres_up == 0
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: "Database connection failed"
```

#### 2. Trading Performance Alerts
```yaml
  - name: trading_performance
    rules:
      - alert: LargeDrawdown
        expr: portfolio_drawdown > 0.10
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Portfolio drawdown exceeds 10%"
          
      - alert: StrategyUnderperforming
        expr: strategy_sharpe_ratio < 0.5
        for: 1h
        labels:
          severity: warning
        annotations:
          summary: "Strategy Sharpe ratio below threshold"
```

### Notification Channels

#### Slack Integration
```bash
# Add to docker-compose.yml environment
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK
ALERT_CHANNEL=#trading-alerts
```

#### Email Notifications
```bash
# SMTP configuration
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your-email@gmail.com
SMTP_PASSWORD=your-app-password
ALERT_EMAIL=alerts@yourcompany.com
```

## 📋 Log Management

### Log Locations

#### Container Logs
```bash
# View all logs
docker-compose logs

# Specific service logs
docker-compose logs trading-system
docker-compose logs trading-postgres
docker-compose logs trading-redis

# Follow logs in real-time
docker-compose logs -f --tail=100 trading-system
```

#### Application Logs
```bash
# Inside container
docker exec trading-system ls -la /app/logs/

# Copy logs to host
docker cp trading-system:/app/logs/ ./local-logs/
```

### Log Analysis

#### Key Log Patterns
```bash
# Error patterns
docker-compose logs trading-system | grep -i error

# Trading activity
docker-compose logs trading-system | grep -i "order\|trade\|position"

# Performance issues
docker-compose logs trading-system | grep -i "slow\|timeout\|latency"

# API calls
docker-compose logs trading-system | grep -i "api\|request\|response"
```

#### Log Aggregation with Fluentd
Logs are automatically collected and forwarded to:
- **Elasticsearch**: For searching and analysis
- **File Storage**: For backup and compliance
- **External Services**: For centralized logging

## 🔍 Troubleshooting Monitoring Issues

### Common Problems

#### 1. Grafana Not Loading
```bash
# Check Grafana container
docker ps | grep grafana

# Restart Grafana
docker-compose restart trading-grafana

# Check Grafana logs
docker-compose logs trading-grafana
```

#### 2. Missing Metrics
```bash
# Check Prometheus targets
curl http://localhost:9090/api/v1/targets

# Verify metrics endpoint
curl http://localhost:8000/metrics

# Restart Prometheus
docker-compose restart trading-prometheus
```

#### 3. Database Connection Issues
```bash
# Test database connectivity
docker exec trading-postgres pg_isready -h localhost -p 5432

# Check database logs
docker-compose logs trading-postgres

# Verify database credentials
docker exec trading-postgres psql -U trading -d trading_db -c "\l"
```

## 📱 Mobile Monitoring

### Grafana Mobile App
1. Download Grafana mobile app
2. Connect to: http://your-server-ip:3000
3. Login with your credentials
4. Access dashboards on-the-go

### API Monitoring Scripts
```python
# monitoring_script.py
import requests
import time
import json

def check_system_health():
    try:
        response = requests.get('http://localhost:8000/health', timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ System Status: {data['status']}")
            return True
        else:
            print(f"❌ API returned status code: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False

def check_portfolio():
    try:
        response = requests.get('http://localhost:8000/api/v1/portfolio', timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"💰 Portfolio Value: ${data['total_value']:,.2f}")
            print(f"📈 Day P&L: ${data['day_pnl']:,.2f}")
            return True
        else:
            print(f"❌ Portfolio check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Portfolio check error: {e}")
        return False

if __name__ == "__main__":
    while True:
        print(f"\n🔍 Monitoring Check - {time.strftime('%Y-%m-%d %H:%M:%S')}")
        check_system_health()
        check_portfolio()
        time.sleep(60)  # Check every minute
```

## 📊 Performance Benchmarks

### Expected Performance Metrics

#### System Performance
- **API Response Time**: < 100ms (95th percentile)
- **Order Execution**: < 500ms from signal to order
- **Memory Usage**: < 2GB per container
- **CPU Usage**: < 50% average load
- **Uptime**: > 99.9%

#### Trading Performance
- **Data Latency**: < 1 second for market data
- **Strategy Execution**: < 2 seconds from signal to trade
- **Portfolio Updates**: Real-time (< 100ms)
- **Risk Calculations**: < 1 second for full portfolio

### Optimization Tips

#### 1. Performance Tuning
```bash
# Increase container resources
docker-compose up --scale trading-system=2

# Optimize database
docker exec trading-postgres psql -U trading -d trading_db -c "VACUUM ANALYZE;"

# Clear Redis cache
docker exec trading-redis redis-cli FLUSHDB
```

#### 2. Monitoring Optimization
```bash
# Reduce metric collection frequency
# Edit prometheus.yml
scrape_interval: 30s  # Increase from 15s

# Limit log retention
# Edit docker-compose.yml
logging:
  options:
    max-size: "100m"
    max-file: "3"
```

## 🎯 Best Practices

### 1. Regular Monitoring Tasks
- **Daily**: Check system health, portfolio performance
- **Weekly**: Review strategy performance, system resources
- **Monthly**: Analyze long-term trends, optimize configurations
- **Quarterly**: Full system audit, capacity planning

### 2. Alert Management
- Set appropriate thresholds to avoid alert fatigue
- Use escalation policies for critical alerts
- Regularly review and update alert rules
- Test notification channels monthly

### 3. Data Retention
- Keep 1 year of detailed metrics
- Archive older data for compliance
- Regular database maintenance
- Monitor disk space usage

### 4. Security Monitoring
- Monitor failed login attempts
- Track API usage patterns
- Alert on unusual trading activity
- Regular security audits

---

**📊 This monitoring guide provides comprehensive coverage of system oversight. Regular monitoring ensures optimal performance and early detection of issues. For specific troubleshooting scenarios, refer to the troubleshooting guide.**