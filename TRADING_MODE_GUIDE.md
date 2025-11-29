# Trading Mode Configuration Guide

This guide explains how to safely switch between paper trading and live trading modes in the Advanced Trading System v2.0.

## 🚨 IMPORTANT SAFETY NOTICE

**LIVE TRADING USES REAL MONEY. ALWAYS TEST THOROUGHLY IN PAPER MODE FIRST.**

## Overview

The trading system supports two modes:
- **Paper Trading**: Simulated trading with fake money (safe for testing)
- **Live Trading**: Real trading with actual money (requires extreme caution)

## Configuration Files

### Paper Trading Configuration (`.env`)
The default configuration file for paper trading:
```bash
# Trading Mode
TRADING_MODE=paper
DEMO_MODE=true

# Alpaca Paper Trading API
ALPACA_API_KEY=your_paper_api_key
ALPACA_SECRET_KEY=your_paper_secret_key
ALPACA_BASE_URL=https://paper-api.alpaca.markets
ALPACA_DATA_URL=https://data.alpaca.markets
```

### Live Trading Configuration (`.env.live`)
Template for live trading configuration:
```bash
# Trading Mode
TRADING_MODE=live
DEMO_MODE=false

# Alpaca Live Trading API
ALPACA_API_KEY=your_live_api_key
ALPACA_SECRET_KEY=your_live_secret_key
ALPACA_BASE_URL=https://api.alpaca.markets
ALPACA_DATA_URL=https://data.alpaca.markets
```

## Switching Between Modes

### From Paper to Live Trading

⚠️ **CRITICAL STEPS - DO NOT SKIP ANY**

1. **Thorough Testing in Paper Mode**
   ```bash
   # Ensure you're in paper mode
   grep "TRADING_MODE=paper" .env
   
   # Run comprehensive tests
   python test_paper_trading.py
   ```

2. **Backup Current Configuration**
   ```bash
   cp .env .env.paper.backup
   ```

3. **Set Up Live Trading Environment**
   ```bash
   # Copy live template
   cp .env.live .env
   
   # Edit with your LIVE API credentials
   nano .env
   ```

4. **Update Live API Credentials**
   - Replace `your_live_api_key` with your actual Alpaca live API key
   - Replace `your_live_secret_key` with your actual Alpaca live secret key
   - Verify URLs point to live endpoints

5. **Validate Configuration**
   ```bash
   python -c "from core.trading_mode_validator import validate_before_trading; print(validate_before_trading())"
   ```

6. **Manual Confirmation Required**
   - The system will prompt for manual confirmation before live trading
   - Type "yes" only if you're absolutely certain

### From Live to Paper Trading

1. **Stop Any Running Trading**
   ```bash
   # Stop all trading processes
   pkill -f "python.*trading"
   ```

2. **Switch to Paper Configuration**
   ```bash
   # Restore paper trading config
   cp .env.paper.backup .env
   # OR manually edit .env to set TRADING_MODE=paper
   ```

3. **Verify Paper Mode**
   ```bash
   python test_paper_trading.py
   ```

## Safety Features

### Automatic Validation
The system automatically validates configuration before trading:

- **Environment Variables**: Checks all required variables are set
- **API Endpoints**: Validates URLs match trading mode
- **Demo Mode**: Ensures consistency with trading mode
- **Risk Limits**: Validates risk management settings

### Manual Confirmation
For live trading, the system requires manual confirmation:
```
You are about to start LIVE TRADING with real money.
Please ensure you have tested thoroughly in paper trading mode.
Are you sure you want to proceed? (yes/no):
```

### Large Trade Protection
Additional confirmation for large live trades:
- Trades over $10,000 require extra confirmation
- System logs all live trading activities

## Validation Checklist

Before switching to live trading, ensure:

- [ ] ✅ Thoroughly tested in paper mode for at least 1 week
- [ ] ✅ Strategy shows consistent profitability in paper trading
- [ ] ✅ Risk management rules are properly configured
- [ ] ✅ Position sizing is appropriate for your account
- [ ] ✅ Stop-loss and take-profit levels are set
- [ ] ✅ You understand all trading strategies being used
- [ ] ✅ Live API credentials are correct and active
- [ ] ✅ Account has sufficient funds
- [ ] ✅ You're prepared to monitor trades actively

## Environment Variables Reference

### Required for All Modes
```bash
# Trading Configuration
TRADING_MODE=paper|live
DEMO_MODE=true|false
INITIAL_CAPITAL=100000

# API Keys
ALPACA_API_KEY=your_api_key
ALPACA_SECRET_KEY=your_secret_key
ALPACA_BASE_URL=api_endpoint
ALPACA_DATA_URL=data_endpoint
```

### Risk Management
```bash
# Position Sizing
MAX_POSITION_SIZE=0.05
MAX_PORTFOLIO_RISK=0.02

# Risk Limits
DAILY_LOSS_LIMIT=1000
MAX_DRAWDOWN=0.10
```

## Testing Commands

### Validate Current Configuration
```bash
python -c "from core.trading_mode_validator import validate_before_trading; result = validate_before_trading(); print(f'Valid: {result[\"valid\"]}, Mode: {result[\"mode\"]}, Errors: {result[\"errors\"]}')"
```

### Test System Initialization
```bash
python test_validation.py
```

### Comprehensive Paper Trading Test
```bash
python test_paper_trading.py
```

## Troubleshooting

### Common Issues

1. **Validation Fails**
   - Check all environment variables are set
   - Verify API credentials are correct
   - Ensure URLs match trading mode

2. **API Connection Errors**
   - Verify internet connection
   - Check API key permissions
   - Confirm API endpoints are correct

3. **Permission Denied**
   - Ensure API keys have trading permissions
   - Check account status with broker

### Getting Help

1. Check validation output for specific errors
2. Review system logs for detailed error messages
3. Verify configuration against this guide
4. Test in paper mode first

## Best Practices

### Development Workflow
1. Always develop and test in paper mode
2. Run automated tests before switching modes
3. Keep separate configuration files for each mode
4. Monitor live trading closely, especially initially
5. Have a plan for emergency stops

### Risk Management
1. Start with small position sizes in live trading
2. Gradually increase exposure as confidence grows
3. Always use stop-losses
4. Monitor drawdowns closely
5. Have predefined exit criteria

### Monitoring
1. Check system health regularly
2. Monitor trade execution and fills
3. Review performance metrics daily
4. Watch for any unusual behavior
5. Keep detailed logs of all activities

---

## 🚨 FINAL WARNING

**LIVE TRADING INVOLVES REAL FINANCIAL RISK**

- You can lose money, potentially more than your initial investment
- Always test thoroughly in paper mode first
- Start with small amounts in live trading
- Never trade with money you cannot afford to lose
- Monitor your trades actively
- Have a clear exit strategy

**The developers of this system are not responsible for any financial losses incurred through its use.**