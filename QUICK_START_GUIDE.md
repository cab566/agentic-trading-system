# 🚀 Quick Start Guide - Trading System v2

Get your AI-powered trading system up and running in minutes!

## 🎯 What You'll Achieve

By the end of this guide, you'll have:
- ✅ A fully operational trading system running in Docker
- ✅ Paper trading enabled with $100k virtual portfolio
- ✅ 3 AI strategies actively monitoring markets
- ✅ Real-time monitoring dashboards
- ✅ API access for system control

## 📋 Prerequisites Checklist

Before starting, ensure you have:

- [ ] **Docker Desktop** installed and running ([Download here](https://www.docker.com/products/docker-desktop))
- [ ] **Git** installed ([Download here](https://git-scm.com/downloads))
- [ ] **API Keys** (see [API Keys Section](#-get-your-api-keys) below)
- [ ] **10 minutes** of your time

## 🔑 Get Your API Keys

### Required (Free Tier Available)
1. **Alpaca** (Stock Trading - Paper Account)
   - Sign up at [alpaca.markets](https://alpaca.markets)
   - Go to "Paper Trading" → "API Keys"
   - Copy your API Key and Secret Key

2. **Polygon** (Market Data)
   - Sign up at [polygon.io](https://polygon.io)
   - Get your free API key (2 calls/minute)

### Optional (Enhanced Features)
3. **News API** (Sentiment Analysis)
   - Sign up at [newsapi.org](https://newsapi.org)
   - Free tier: 1000 requests/day

4. **Financial Modeling Prep** (Fundamental Data)
   - Sign up at [financialmodelingprep.com](https://financialmodelingprep.com)
   - Free tier: 250 requests/day

## 🚀 5-Minute Setup

### Step 1: Clone the Repository
```bash
git clone <repository-url>
cd trading_system_v2
```

### Step 2: Configure Environment
```bash
# Copy the example environment file
cp .env.example .env

# Edit with your API keys (use any text editor)
nano .env  # or vim .env, or code .env
```

**Minimal .env configuration:**
```bash
# Required for basic functionality
ALPACA_API_KEY=your_alpaca_key_here
ALPACA_SECRET_KEY=your_alpaca_secret_here
ALPACA_BASE_URL=https://paper-api.alpaca.markets
POLYGON_API_KEY=your_polygon_key_here

# System settings
TRADING_MODE=paper
LOG_LEVEL=INFO
```

### Step 3: Launch the System
```bash
# Start all services (this will take 2-3 minutes on first run)
docker-compose -f docker-compose.production.yml up --build
```

**What's happening:**
- 🐳 Building Docker containers
- 🗄️ Starting PostgreSQL database
- 🔄 Starting Redis cache
- 🤖 Initializing AI agents
- 📊 Starting monitoring services

### Step 4: Verify Everything Works
```bash
# In a new terminal, check system status
curl http://localhost:8000/api/v1/status

# Expected response:
{
  "status": "running",
  "trading_active": false,
  "portfolio_value": 100000.00,
  "active_strategies": 3,
  "system_health": "healthy"
}
```

## 🎉 You're Live!

### Access Your Dashboards

1. **Main API**: http://localhost:8000
   - System status and control

2. **Grafana Dashboard**: http://localhost:3000
   - Username: `admin`
   - Password: `admin`
   - Real-time system metrics

3. **Prometheus Metrics**: http://localhost:9090
   - Raw system metrics

### What's Running Now

Your system is now:
- 🟢 **Monitoring Markets**: 24/7 across stocks, crypto, and forex
- 🧠 **AI Analysis**: 7 specialized agents analyzing opportunities
- 📊 **Paper Trading**: Safe virtual trading with $100k portfolio
- 🛡️ **Risk Management**: Automatic position sizing and stop losses
- 📈 **Strategy Execution**: 3 active strategies (Covered Calls, Momentum, Mean Reversion)

## 🎮 Try It Out

### Check Your Portfolio
```bash
curl http://localhost:8000/api/v1/portfolio
```

### View Active Strategies
```bash
curl http://localhost:8000/api/v1/strategies
```

### Monitor System Health
```bash
curl http://localhost:8000/api/v1/health
```

### View Recent Activity
```bash
# Check logs
docker-compose logs -f trading-system

# View specific service logs
docker-compose logs -f trading-redis
docker-compose logs -f trading-postgres
```

## 🔧 Common First-Time Issues

### Issue: "Connection refused" on API calls
**Solution:**
```bash
# Check if containers are running
docker ps

# If not running, restart
docker-compose -f docker-compose.production.yml up --build
```

### Issue: "Invalid API key" errors
**Solution:**
1. Double-check your `.env` file has correct API keys
2. Ensure no extra spaces around the `=` sign
3. Restart containers: `docker-compose down && docker-compose -f docker-compose.production.yml up --build`

### Issue: High CPU usage
**Solution:**
```bash
# Check resource usage
docker stats

# Reduce log level in .env
LOG_LEVEL=WARNING
```

## 📚 Next Steps

### 1. Explore the System
- Review the main [README.md](README.md) for detailed documentation
- Check out [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md) for advanced configuration

### 2. Customize Your Setup
- Modify `config/strategies.yaml` to adjust trading strategies
- Update `config/agents.yaml` to configure AI behavior
- Edit `config/data_sources.yaml` for data source preferences

### 3. Monitor Performance
- Set up alerts in Grafana
- Review daily performance reports
- Analyze strategy effectiveness

### 4. Go Live (When Ready)
⚠️ **Important**: Only switch to live trading after thorough testing
1. Get live API keys from your brokers
2. Update `.env` with `TRADING_MODE=live`
3. Start with small position sizes
4. Monitor closely for the first few days

## 🆘 Need Help?

- **Documentation**: Check [README.md](README.md) and [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md)
- **Logs**: `docker-compose logs -f trading-system`
- **System Status**: `curl http://localhost:8000/api/v1/status`
- **Health Check**: `curl http://localhost:8000/api/v1/health`

## 🎯 Success Checklist

- [ ] System responds to API calls
- [ ] Grafana dashboard loads
- [ ] Portfolio shows $100k starting value
- [ ] 3 strategies are active
- [ ] No error messages in logs
- [ ] All Docker containers are "healthy"

**Congratulations! Your AI trading system is now operational! 🎉**

---

*Remember: This system is running in paper trading mode with virtual money. Always test thoroughly before considering live trading.*