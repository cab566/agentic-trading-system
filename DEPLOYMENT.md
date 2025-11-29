# Trading System v2 - Production Deployment Guide

## Overview

This guide provides comprehensive instructions for deploying Trading System v2 to production environments. The system is designed for high availability, security, and scalability.

## Prerequisites

### System Requirements
- **OS**: Linux (Ubuntu 20.04+ recommended) or macOS
- **CPU**: Minimum 4 cores, 8 cores recommended
- **RAM**: Minimum 8GB, 16GB recommended
- **Storage**: Minimum 100GB SSD
- **Network**: Stable internet connection with low latency to market data providers

### Software Dependencies
- Docker 24.0+
- Docker Compose 2.0+
- PostgreSQL 15+ (for production database)
- Redis 7+ (for caching)
- Nginx (for reverse proxy)

### Required API Keys
- OpenAI API Key (for AI agents)
- Alpaca Trading API credentials
- Market data provider API keys (Alpha Vantage, Polygon.io, etc.)

## Environment Configuration

### 1. Production Environment Setup

Copy the production environment template:
```bash
cp .env.production .env
```

### 2. Configure Environment Variables

Edit `.env` and set the following critical variables:

#### AI/LLM Configuration
```bash
OPENAI_API_KEY=your-actual-openai-api-key
```

#### Database Configuration
```bash
DATABASE_URL=postgresql://username:password@host:port/database
DB_USER=trading_user
DB_PASSWORD=secure_password
DB_HOST=localhost
DB_PORT=5432
DB_NAME=trading_system_prod
```

#### Trading Configuration
```bash
# Start with paper trading for safety
TRADING_MODE=paper_trading
ALPACA_API_KEY=your-paper-trading-api-key
ALPACA_SECRET_KEY=your-paper-trading-secret-key
```

#### Security Configuration
```bash
SECRET_KEY=your-256-bit-secret-key
JWT_SECRET_KEY=your-jwt-secret-key
ALLOWED_HOSTS=your-domain.com,localhost
```

## Deployment Methods

### Method 1: Docker Compose (Recommended)

#### 1. Build and Deploy
```bash
# Build production images
docker-compose -f docker-compose.production.yml build

# Deploy the stack
docker-compose -f docker-compose.production.yml up -d
```

#### 2. Verify Deployment
```bash
# Check service status
docker-compose -f docker-compose.production.yml ps

# Check logs
docker-compose -f docker-compose.production.yml logs -f trading-system
```

#### 3. Health Check
```bash
curl http://localhost:8000/health
```

### Method 2: Manual Docker Deployment

#### 1. Build Production Image
```bash
docker build -f Dockerfile.production -t trading-system-v2:latest .
```

#### 2. Run Database
```bash
docker run -d \
  --name trading-postgres \
  -e POSTGRES_DB=trading_system \
  -e POSTGRES_USER=trading_user \
  -e POSTGRES_PASSWORD=secure_password \
  -v postgres_data:/var/lib/postgresql/data \
  postgres:15-alpine
```

#### 3. Run Redis
```bash
docker run -d \
  --name trading-redis \
  -v redis_data:/data \
  redis:7-alpine
```

#### 4. Run Application
```bash
docker run -d \
  --name trading-system-v2 \
  --env-file .env \
  -p 8000:8000 \
  --link trading-postgres:postgres \
  --link trading-redis:redis \
  trading-system-v2:latest
```

## Production Checklist

### Pre-Deployment
- [ ] All environment variables configured
- [ ] API keys validated and working
- [ ] Database connection tested
- [ ] Redis connection tested
- [ ] SSL certificates installed
- [ ] Firewall rules configured
- [ ] Backup systems disabled (as per requirements)
- [ ] Monitoring systems configured

### Post-Deployment
- [ ] Health checks passing
- [ ] All services running
- [ ] Logs being generated correctly
- [ ] Monitoring dashboards accessible
- [ ] Alert systems functional
- [ ] Performance metrics within acceptable ranges

## Security Considerations

### 1. Network Security
- Use HTTPS/TLS for all external communications
- Configure firewall to allow only necessary ports
- Use VPN for administrative access
- Implement rate limiting

### 2. Application Security
- Keep all dependencies updated
- Use strong, unique passwords
- Enable audit logging
- Regular security scans

### 3. Data Security
- Encrypt sensitive data at rest
- Use secure database connections
- Regular security backups (stored securely)
- Implement data retention policies

## Monitoring and Observability

### 1. Application Monitoring
- **Prometheus**: Metrics collection (port 9090)
- **Grafana**: Dashboards and visualization (port 3000)
- **Health Checks**: Automated health monitoring

### 2. Log Management
- **Structured Logging**: JSON format for easy parsing
- **Log Rotation**: Automatic log rotation and cleanup
- **Centralized Logging**: Fluentd for log aggregation

### 3. Alerting
- **Email Alerts**: Critical system events
- **Slack Integration**: Real-time notifications
- **PagerDuty**: Critical incident management

## Performance Optimization

### 1. Application Performance
- **Connection Pooling**: Database connection optimization
- **Caching**: Redis for frequently accessed data
- **Async Processing**: Non-blocking operations
- **Resource Limits**: Container resource constraints

### 2. Database Performance
- **Indexing**: Optimized database indexes
- **Query Optimization**: Efficient database queries
- **Connection Pooling**: PostgreSQL connection pooling

### 3. Infrastructure Performance
- **Load Balancing**: Nginx reverse proxy
- **Auto-scaling**: Container orchestration
- **CDN**: Static asset delivery optimization

## Troubleshooting

### Common Issues

#### 1. Application Won't Start
```bash
# Check logs
docker-compose logs trading-system

# Check environment variables
docker exec trading-system-v2 env | grep -E "(DATABASE|REDIS|OPENAI)"

# Verify dependencies
docker-compose ps
```

#### 2. Database Connection Issues
```bash
# Test database connection
docker exec trading-postgres psql -U trading_user -d trading_system -c "SELECT 1;"

# Check database logs
docker-compose logs postgres
```

#### 3. API Connection Issues
```bash
# Test API connectivity
curl -H "Authorization: Bearer $OPENAI_API_KEY" https://api.openai.com/v1/models

# Check network connectivity
docker exec trading-system-v2 ping api.alpaca.markets
```

### Performance Issues

#### 1. High Memory Usage
- Check for memory leaks in logs
- Monitor container resource usage
- Adjust container memory limits

#### 2. Slow Response Times
- Check database query performance
- Monitor Redis cache hit rates
- Review application metrics

## Maintenance

### 1. Regular Updates
- **Security Updates**: Monthly security patches
- **Dependency Updates**: Quarterly dependency reviews
- **System Updates**: Regular OS and container updates

### 2. Database Maintenance
- **Index Optimization**: Monthly index analysis
- **Query Performance**: Regular query optimization
- **Data Cleanup**: Automated data retention policies

### 3. Monitoring Review
- **Alert Tuning**: Monthly alert threshold review
- **Dashboard Updates**: Quarterly dashboard improvements
- **Performance Analysis**: Regular performance reviews

## Scaling

### 1. Horizontal Scaling
- **Load Balancer**: Nginx for request distribution
- **Multiple Instances**: Docker Swarm or Kubernetes
- **Database Replication**: Read replicas for scaling

### 2. Vertical Scaling
- **Resource Allocation**: Increase container resources
- **Database Optimization**: Optimize database configuration
- **Cache Optimization**: Increase Redis memory allocation

## Disaster Recovery

### 1. Backup Strategy
- **Database Backups**: Automated PostgreSQL backups
- **Configuration Backups**: Environment and config files
- **Code Backups**: Git repository with tags

### 2. Recovery Procedures
- **Service Recovery**: Automated service restart
- **Data Recovery**: Database restoration procedures
- **Full System Recovery**: Complete system rebuild process

## Support and Maintenance

### 1. Monitoring Dashboards
- **Grafana**: http://localhost:3000 (admin/admin)
- **Prometheus**: http://localhost:9090
- **Application Health**: http://localhost:8000/health

### 2. Log Locations
- **Application Logs**: `/app/logs/`
- **Container Logs**: `docker-compose logs`
- **System Logs**: `/var/log/`

### 3. Emergency Contacts
- **System Administrator**: [Contact Information]
- **Development Team**: [Contact Information]
- **Infrastructure Team**: [Contact Information]

## Compliance and Regulations

### 1. Financial Regulations
- **Data Retention**: 7-year data retention policy
- **Audit Trails**: Complete transaction logging
- **Reporting**: Automated compliance reporting

### 2. Data Protection
- **GDPR Compliance**: Data protection measures
- **Data Encryption**: End-to-end encryption
- **Access Controls**: Role-based access control

---

**Note**: This system starts in paper trading mode for safety. Only switch to live trading after thorough testing and validation in the production environment.

For additional support or questions, please refer to the project documentation or contact the development team.