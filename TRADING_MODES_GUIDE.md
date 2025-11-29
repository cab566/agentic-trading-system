# Trading Modes Guide 📈

Complete guide to understanding and configuring paper trading vs live trading modes in the Trading System v2.0.

## 🎯 Overview

The Trading System v2.0 supports multiple trading modes to accommodate different use cases, from safe testing to live production trading. This guide covers configuration, features, and best practices for each mode.

## 🧪 Paper Trading Mode

### What is Paper Trading?

Paper trading (also called "simulated trading") allows you to test strategies with real market data but without risking actual capital. All trades are simulated, providing a safe environment for:

- **Strategy Development**: Test new algorithms
- **System Validation**: Verify system functionality
- **Performance Analysis**: Evaluate strategy performance
- **Risk Assessment**: Understand potential drawdowns
- **Learning**: Practice without financial risk

### Current Status
Your system is currently running in **Paper Trading Mode** with:
- 💰 **Virtual Portfolio**: $100,000 starting capital
- 📊 **Current Value**: $97,052.34
- 🔄 **Active Strategies**: 3 strategies running
- 📈 **Real Market Data**: Live data feeds for accurate simulation

### Configuration

#### Environment Variables (.env)
```bash
# Trading Mode Configuration
TRADING_MODE=paper              # Options: paper, live, backtest
DEMO_MODE=true                 # Enable demo/simulation features
PAPER_TRADING_BALANCE=100000   # Starting virtual balance
PAPER_TRADING_CURRENCY=USD     # Base currency

# Risk Management (Paper Mode)
MAX_POSITION_SIZE=0.05         # 5% max position size
MAX_DAILY_LOSS=0.02           # 2% max daily loss
ENABLE_STOP_LOSSES=true       # Enable stop loss orders
```

#### Configuration File (config/trading.yaml)
```yaml
trading:
  mode: paper
  paper_trading:
    initial_balance: 100000
    currency: USD
    commission_rate: 0.001      # 0.1% commission simulation
    slippage_model: linear      # linear, sqrt, fixed
    slippage_rate: 0.0005      # 0.05% slippage simulation
    
  risk_management:
    max_position_size: 0.05
    max_portfolio_risk: 0.20
    max_correlation: 0.70
    
  execution:
    order_timeout: 30           # seconds
    retry_attempts: 3
    partial_fills: true
```

### Features in Paper Mode

#### ✅ Available Features
- **Real Market Data**: Live price feeds from all configured sources
- **Strategy Execution**: Full AI agent decision-making
- **Order Management**: Complete order lifecycle simulation
- **Portfolio Tracking**: Real-time portfolio updates
- **Risk Management**: All risk controls active
- **Performance Analytics**: Comprehensive performance metrics
- **API Access**: Full API functionality
- **WebSocket Streams**: Real-time data streams
- **Monitoring**: Complete system monitoring

#### 🚫 Simulated Elements
- **Order Execution**: No real broker orders
- **Capital**: Virtual money only
- **Commissions**: Simulated transaction costs
- **Slippage**: Modeled market impact
- **Fills**: Simulated order fills based on market data

### Paper Trading Benefits

#### 1. Risk-Free Testing
```bash
# Test aggressive strategies without risk
curl -X POST http://localhost:8000/api/v1/strategies/aggressive_momentum/enable
```

#### 2. Strategy Validation
- Test strategies across different market conditions
- Validate entry/exit logic
- Assess risk management effectiveness
- Measure performance metrics

#### 3. System Reliability
- Verify system stability under load
- Test error handling and recovery
- Validate data feed reliability
- Ensure monitoring systems work

#### 4. Performance Analysis
```bash
# Get detailed performance metrics
curl http://localhost:8000/api/v1/performance/summary | jq '{
  total_return: .total_return,
  sharpe_ratio: .sharpe_ratio,
  max_drawdown: .max_drawdown,
  win_rate: .win_rate
}'
```

## 💰 Live Trading Mode

### What is Live Trading?

Live trading mode connects to real brokers and executes actual trades with real money. This mode should only be used after thorough testing in paper mode.

⚠️ **WARNING**: Live trading involves real financial risk. Only use after extensive testing and validation.

### Configuration

#### Environment Variables (.env)
```bash
# Trading Mode Configuration
TRADING_MODE=live              # Switch to live trading
DEMO_MODE=false               # Disable demo features
LIVE_TRADING_ENABLED=true     # Explicit live trading confirmation

# Broker API Keys (REQUIRED for live trading)
ALPACA_API_KEY=your_live_api_key
ALPACA_SECRET_KEY=your_live_secret_key
ALPACA_BASE_URL=https://api.alpaca.markets  # Live endpoint

# Crypto Exchanges (if using)
BINANCE_API_KEY=your_binance_key
BINANCE_SECRET_KEY=your_binance_secret
COINBASE_API_KEY=your_coinbase_key
COINBASE_SECRET=your_coinbase_secret

# Forex (if using)
OANDA_API_KEY=your_oanda_key
OANDA_ACCOUNT_ID=your_account_id
```

#### Live Trading Configuration (config/live_trading.yaml)
```yaml
live_trading:
  enabled: true
  confirmation_required: true    # Require manual confirmation for trades
  
  brokers:
    alpaca:
      environment: live          # live or sandbox
      account_type: margin       # cash or margin
      
  risk_management:
    max_position_size: 0.02      # More conservative in live mode
    max_daily_loss: 0.01         # 1% max daily loss
    max_portfolio_risk: 0.10     # 10% max portfolio risk
    emergency_stop_loss: 0.05    # 5% emergency stop
    
  execution:
    order_confirmation: true     # Require confirmation
    max_order_value: 10000      # Maximum single order value
    cool_down_period: 300       # 5 minutes between large orders
```

### Live Trading Safety Features

#### 1. Multi-Level Confirmations
```python
# Example confirmation flow
def execute_live_trade(order):
    # Level 1: Strategy validation
    if not validate_strategy_signal(order):
        return False
        
    # Level 2: Risk management check
    if not risk_manager.validate_order(order):
        return False
        
    # Level 3: Manual confirmation (if enabled)
    if REQUIRE_CONFIRMATION:
        if not get_manual_confirmation(order):
            return False
            
    # Level 4: Final execution
    return broker.execute_order(order)
```

#### 2. Circuit Breakers
```yaml
circuit_breakers:
  daily_loss_limit: 0.02        # Stop trading if 2% daily loss
  consecutive_losses: 5         # Stop after 5 consecutive losses
  volatility_threshold: 0.30    # Pause in high volatility
  news_event_pause: true        # Pause during major news
```

#### 3. Emergency Controls
```bash
# Emergency stop all trading
curl -X POST http://localhost:8000/api/v1/emergency/stop

# Cancel all open orders
curl -X POST http://localhost:8000/api/v1/orders/cancel_all

# Close all positions
curl -X POST http://localhost:8000/api/v1/positions/close_all
```

## 🔄 Backtest Mode

### What is Backtesting?

Backtesting runs strategies against historical data to evaluate performance over past market conditions.

### Configuration
```bash
# Environment Variables
TRADING_MODE=backtest
BACKTEST_START_DATE=2023-01-01
BACKTEST_END_DATE=2023-12-31
BACKTEST_INITIAL_CAPITAL=100000
```

### Running Backtests
```bash
# Run backtest for specific strategy
python main.py backtest --strategy momentum --start 2023-01-01 --end 2023-12-31

# Run backtest for all strategies
python main.py backtest --all --start 2023-01-01 --end 2023-12-31

# Generate backtest report
python main.py backtest --strategy momentum --report --output backtest_report.html
```

## 🔧 Mode Switching

### Switching from Paper to Live Trading

#### Step 1: Validate Paper Trading Performance
```bash
# Check performance metrics
curl http://localhost:8000/api/v1/performance/summary

# Verify risk metrics
curl http://localhost:8000/api/v1/risk/metrics

# Review trade history
curl http://localhost:8000/api/v1/trades?limit=100
```

#### Step 2: Update Configuration
```bash
# Stop the system
docker-compose down

# Update .env file
sed -i 's/TRADING_MODE=paper/TRADING_MODE=live/' .env
sed -i 's/DEMO_MODE=true/DEMO_MODE=false/' .env

# Add live broker credentials
echo "ALPACA_API_KEY=your_live_key" >> .env
echo "ALPACA_SECRET_KEY=your_live_secret" >> .env
```

#### Step 3: Restart with Live Configuration
```bash
# Restart system
docker-compose -f docker-compose.production.yml up --build

# Verify live mode
curl http://localhost:8000/api/v1/config | jq '.trading_mode'
```

#### Step 4: Start with Conservative Settings
```bash
# Enable only one strategy initially
curl -X POST http://localhost:8000/api/v1/strategies/conservative_momentum/enable

# Set lower position sizes
curl -X POST http://localhost:8000/api/v1/risk/limits \
  -H "Content-Type: application/json" \
  -d '{"max_position_size": 0.01}'
```

## 📊 Mode Comparison

| Feature | Paper Trading | Live Trading | Backtest |
|---------|---------------|--------------|----------|
| **Risk** | None | Real money | None |
| **Market Data** | Real-time | Real-time | Historical |
| **Execution** | Simulated | Real brokers | Simulated |
| **Costs** | Simulated | Real commissions | None |
| **Speed** | Real-time | Real-time | Fast |
| **Purpose** | Testing/Learning | Production | Analysis |
| **Capital Required** | None | Real money | None |

## 🛡️ Risk Management by Mode

### Paper Trading Risk Management
- Focus on strategy validation
- Test extreme scenarios
- Validate risk calculations
- No financial risk

### Live Trading Risk Management
- Conservative position sizing
- Multiple confirmation levels
- Real-time monitoring
- Emergency stop mechanisms
- Regular performance reviews

### Backtest Risk Management
- Historical scenario analysis
- Stress testing
- Monte Carlo simulations
- Walk-forward analysis

## 📈 Performance Tracking

### Paper Trading Metrics
```bash
# Current paper trading performance
curl http://localhost:8000/api/v1/performance/summary | jq '{
  mode: "paper",
  total_return: .total_return,
  sharpe_ratio: .sharpe_ratio,
  max_drawdown: .max_drawdown,
  total_trades: .total_trades,
  win_rate: .win_rate
}'
```

### Live Trading Metrics
```bash
# Live trading performance (when in live mode)
curl http://localhost:8000/api/v1/performance/live | jq '{
  mode: "live",
  realized_pnl: .realized_pnl,
  unrealized_pnl: .unrealized_pnl,
  commissions_paid: .commissions_paid,
  slippage_cost: .slippage_cost
}'
```

## 🔍 Monitoring by Mode

### Paper Trading Monitoring
- Focus on strategy performance
- System stability
- Data quality
- Algorithm correctness

### Live Trading Monitoring
- Real-time P&L tracking
- Risk limit monitoring
- Execution quality
- Broker connectivity
- Regulatory compliance

## 🚨 Alerts and Notifications

### Paper Trading Alerts
```yaml
paper_trading_alerts:
  - large_drawdown: >5%
  - strategy_underperformance: sharpe < 0.5
  - system_errors: any error
  - data_quality_issues: stale data
```

### Live Trading Alerts
```yaml
live_trading_alerts:
  - any_loss: >$1000
  - daily_loss: >1%
  - position_size: >2%
  - execution_failure: any failed order
  - broker_disconnect: connection lost
  - regulatory_breach: any compliance issue
```

## 📋 Best Practices

### Paper Trading Best Practices
1. **Realistic Simulation**: Use realistic commission and slippage models
2. **Sufficient Testing**: Run for at least 3-6 months
3. **Multiple Scenarios**: Test in different market conditions
4. **Performance Validation**: Achieve consistent positive returns
5. **Risk Assessment**: Understand maximum drawdowns

### Live Trading Best Practices
1. **Start Small**: Begin with minimal position sizes
2. **Gradual Scaling**: Increase size only after proven performance
3. **Continuous Monitoring**: Watch system 24/7 initially
4. **Regular Reviews**: Daily performance and risk reviews
5. **Emergency Preparedness**: Have stop procedures ready

### Mode Transition Checklist
- [ ] Paper trading shows consistent profitability (>6 months)
- [ ] Risk metrics are within acceptable limits
- [ ] System stability is proven (>99% uptime)
- [ ] All monitoring and alerting systems tested
- [ ] Emergency procedures documented and tested
- [ ] Live broker accounts funded and tested
- [ ] Regulatory compliance verified
- [ ] Team trained on live trading procedures

## 🔧 Troubleshooting Mode Issues

### Paper Trading Issues
```bash
# Check paper trading balance
curl http://localhost:8000/api/v1/portfolio | jq '.cash'

# Verify simulation settings
curl http://localhost:8000/api/v1/config | jq '.trading_mode'

# Reset paper trading portfolio
curl -X POST http://localhost:8000/api/v1/paper/reset
```

### Live Trading Issues
```bash
# Verify broker connection
curl http://localhost:8000/api/v1/brokers/status

# Check account balance
curl http://localhost:8000/api/v1/account/balance

# Emergency stop trading
curl -X POST http://localhost:8000/api/v1/emergency/stop
```

---

**📈 This trading modes guide provides comprehensive coverage of all trading modes. Always prioritize safety and thorough testing before transitioning to live trading. For specific configuration details, refer to the configuration files and environment variable documentation.**