# Trading System v2 API Documentation 📚

Complete API reference for the Advanced Multi-Asset Trading System v2.0

## 🌐 Base Information

- **Base URL**: `http://localhost:8000`
- **API Version**: v2.0.0
- **Authentication**: Bearer Token (for protected endpoints)
- **Content-Type**: `application/json`
- **Documentation**: Available at `/docs` (Swagger UI) and `/redoc` (ReDoc)

## 🔐 Authentication

Most endpoints require authentication using Bearer tokens:

```bash
curl -H "Authorization: Bearer YOUR_TOKEN" http://localhost:8000/api/v1/endpoint
```

## 📋 Core API Endpoints

### System Health & Status

#### GET `/health`
**Description**: System health check with service status
**Authentication**: None required
**Response Model**: `HealthResponse`

```bash
curl http://localhost:8000/health
```

**Response**:
```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00Z",
  "version": "2.0.0",
  "environment": "production",
  "services": {
    "database": {
      "status": "healthy",
      "response_time": "< 10ms"
    },
    "redis": {
      "status": "healthy", 
      "response_time": "< 5ms"
    }
  }
}
```

#### GET `/api/v1/status`
**Description**: Detailed trading system status
**Authentication**: None required
**Response Model**: `SystemStatusResponse`

```bash
curl http://localhost:8000/api/v1/status
```

**Response**:
```json
{
  "trading_active": false,
  "market_hours": {
    "us_market_open": false,
    "crypto_market_open": true,
    "forex_market_open": true
  },
  "connected_exchanges": ["alpaca", "binance"],
  "active_strategies": ["momentum", "mean_reversion", "covered_calls"],
  "system_metrics": {
    "uptime": "16h 42m",
    "memory_usage": "45%",
    "cpu_usage": "12%"
  }
}
```

#### GET `/api/v1/config`
**Description**: Get system configuration (non-sensitive data only)
**Authentication**: None required

```bash
curl http://localhost:8000/api/v1/config
```

**Response**:
```json
{
  "environment": "production",
  "trading_mode": "paper",
  "demo_mode": "true",
  "version": "2.0.0"
}
```

### Portfolio Management

#### GET `/api/v1/portfolio`
**Description**: Get current portfolio overview
**Authentication**: Required

```bash
curl -H "Authorization: Bearer YOUR_TOKEN" \
     http://localhost:8000/api/v1/portfolio
```

**Response**:
```json
{
  "total_value": 97052.34,
  "cash": 50000.00,
  "positions_value": 47052.34,
  "day_pnl": 1234.56,
  "total_pnl": -2947.66,
  "positions": [
    {
      "symbol": "AAPL",
      "quantity": 100,
      "avg_price": 150.00,
      "market_value": 15500.00,
      "unrealized_pnl": 500.00,
      "updated_at": "2024-01-15T10:30:00Z"
    }
  ]
}
```

#### GET `/api/v1/positions`
**Description**: Get detailed position information
**Authentication**: Required
**Response Model**: `List[PositionResponse]`

```bash
curl -H "Authorization: Bearer YOUR_TOKEN" \
     http://localhost:8000/api/v1/positions
```

### Trading Operations

#### POST `/api/v1/orders`
**Description**: Place a new trading order
**Authentication**: Required
**Request Model**: `TradeRequest`

```bash
curl -X POST \
     -H "Authorization: Bearer YOUR_TOKEN" \
     -H "Content-Type: application/json" \
     -d '{
       "symbol": "AAPL",
       "side": "buy",
       "quantity": 100,
       "order_type": "market",
       "strategy": "momentum"
     }' \
     http://localhost:8000/api/v1/orders
```

**Request Body**:
```json
{
  "symbol": "AAPL",
  "side": "buy",
  "quantity": 100,
  "order_type": "market",
  "price": 150.00,
  "strategy": "momentum"
}
```

**Response**:
```json
{
  "order_id": "ord_123456789",
  "status": "submitted",
  "symbol": "AAPL",
  "side": "buy",
  "quantity": 100,
  "order_type": "market",
  "submitted_at": "2024-01-15T10:30:00Z"
}
```

#### GET `/api/v1/orders/{order_id}`
**Description**: Get order details by ID
**Authentication**: Required

```bash
curl -H "Authorization: Bearer YOUR_TOKEN" \
     http://localhost:8000/api/v1/orders/ord_123456789
```

#### DELETE `/api/v1/orders/{order_id}`
**Description**: Cancel an open order
**Authentication**: Required

```bash
curl -X DELETE \
     -H "Authorization: Bearer YOUR_TOKEN" \
     http://localhost:8000/api/v1/orders/ord_123456789
```

### Strategy Management

#### GET `/api/v1/strategies`
**Description**: List all available trading strategies
**Authentication**: Required

```bash
curl -H "Authorization: Bearer YOUR_TOKEN" \
     http://localhost:8000/api/v1/strategies
```

**Response**:
```json
{
  "strategies": [
    {
      "id": "momentum",
      "name": "Momentum Strategy",
      "status": "active",
      "allocation": 0.33,
      "performance": {
        "total_return": 0.0523,
        "sharpe_ratio": 1.42,
        "max_drawdown": 0.0234
      }
    }
  ]
}
```

#### POST `/api/v1/strategies/{strategy_id}/enable`
**Description**: Enable a trading strategy
**Authentication**: Required

```bash
curl -X POST \
     -H "Authorization: Bearer YOUR_TOKEN" \
     http://localhost:8000/api/v1/strategies/momentum/enable
```

#### POST `/api/v1/strategies/{strategy_id}/disable`
**Description**: Disable a trading strategy
**Authentication**: Required

```bash
curl -X POST \
     -H "Authorization: Bearer YOUR_TOKEN" \
     http://localhost:8000/api/v1/strategies/momentum/disable
```

#### GET `/api/v1/strategies/{strategy_id}/performance`
**Description**: Get detailed strategy performance metrics
**Authentication**: Required

```bash
curl -H "Authorization: Bearer YOUR_TOKEN" \
     http://localhost:8000/api/v1/strategies/momentum/performance
```

### Risk Management

#### GET `/api/v1/risk/metrics`
**Description**: Get current risk metrics
**Authentication**: Required

```bash
curl -H "Authorization: Bearer YOUR_TOKEN" \
     http://localhost:8000/api/v1/risk/metrics
```

**Response**:
```json
{
  "portfolio_var": 0.0234,
  "max_position_size": 0.05,
  "current_leverage": 1.2,
  "risk_score": 3.2,
  "correlation_risk": 0.15
}
```

#### GET `/api/v1/risk/limits`
**Description**: Get current risk limits
**Authentication**: Required

#### POST `/api/v1/risk/limits`
**Description**: Update risk limits
**Authentication**: Required

#### GET `/api/v1/risk/alerts`
**Description**: Get active risk alerts
**Authentication**: Required

### Performance Analytics

#### GET `/api/v1/performance/summary`
**Description**: Get performance summary
**Authentication**: Required

```bash
curl -H "Authorization: Bearer YOUR_TOKEN" \
     http://localhost:8000/api/v1/performance/summary
```

#### GET `/api/v1/performance/daily`
**Description**: Get daily performance data
**Authentication**: Required

#### GET `/api/v1/performance/trades`
**Description**: Get trade history and performance
**Authentication**: Required

#### GET `/api/v1/performance/attribution`
**Description**: Get performance attribution by strategy
**Authentication**: Required

## 🔌 WebSocket API

### Connection
Connect to real-time data streams via WebSocket:

```javascript
const ws = new WebSocket('ws://localhost:8000/ws');
```

### Subscription Management

#### Subscribe to Channels
```javascript
ws.send(JSON.stringify({
  "action": "subscribe",
  "channels": ["portfolio", "orders", "market_data"]
}));
```

#### Unsubscribe from Channels
```javascript
ws.send(JSON.stringify({
  "action": "unsubscribe", 
  "channels": ["market_data"]
}));
```

#### Ping/Pong
```javascript
// Send ping
ws.send(JSON.stringify({
  "action": "ping"
}));

// Receive pong
{
  "type": "pong",
  "timestamp": "2024-01-15T10:30:00Z"
}
```

### Real-time Data Streams

#### Portfolio Updates
```javascript
{
  "type": "portfolio_update",
  "data": {
    "total_value": 97052.34,
    "day_pnl": 1234.56,
    "timestamp": "2024-01-15T10:30:00Z"
  }
}
```

#### Order Updates
```javascript
{
  "type": "order_update",
  "data": {
    "order_id": "ord_123456789",
    "status": "filled",
    "fill_price": 150.25,
    "timestamp": "2024-01-15T10:30:00Z"
  }
}
```

#### Market Data
```javascript
{
  "type": "market_data",
  "data": {
    "symbol": "AAPL",
    "price": 150.25,
    "volume": 1000000,
    "timestamp": "2024-01-15T10:30:00Z"
  }
}
```

## 📊 Data Models

### HealthResponse
```python
{
  "status": str,           # "healthy", "degraded", "unhealthy"
  "timestamp": datetime,
  "version": str,
  "environment": str,
  "services": Dict[str, Any]
}
```

### SystemStatusResponse
```python
{
  "trading_active": bool,
  "market_hours": Dict[str, Any],
  "connected_exchanges": List[str],
  "active_strategies": List[str], 
  "system_metrics": Dict[str, Any]
}
```

### TradeRequest
```python
{
  "symbol": str,           # Required
  "side": str,             # "buy" or "sell"
  "quantity": float,       # Required
  "order_type": str,       # "market", "limit", "stop"
  "price": Optional[float], # Required for limit orders
  "strategy": str          # Strategy identifier
}
```

### PositionResponse
```python
{
  "symbol": str,
  "quantity": float,
  "avg_price": float,
  "market_value": float,
  "unrealized_pnl": float,
  "updated_at": datetime
}
```

## 🚨 Error Handling

### HTTP Status Codes
- `200`: Success
- `400`: Bad Request (invalid parameters)
- `401`: Unauthorized (missing/invalid token)
- `404`: Not Found (endpoint/resource not found)
- `429`: Too Many Requests (rate limited)
- `500`: Internal Server Error

### Error Response Format
```json
{
  "detail": "Error description",
  "timestamp": "2024-01-15T10:30:00Z"
}
```

## 🔧 Rate Limiting

- **Default Rate Limit**: 100 requests per minute per IP
- **Authenticated Rate Limit**: 1000 requests per minute per token
- **WebSocket Connections**: 10 concurrent connections per IP

## 📝 Examples

### Complete Trading Workflow

1. **Check System Status**
```bash
curl http://localhost:8000/api/v1/status
```

2. **Get Portfolio Overview**
```bash
curl -H "Authorization: Bearer YOUR_TOKEN" \
     http://localhost:8000/api/v1/portfolio
```

3. **Place Market Order**
```bash
curl -X POST \
     -H "Authorization: Bearer YOUR_TOKEN" \
     -H "Content-Type: application/json" \
     -d '{
       "symbol": "AAPL",
       "side": "buy", 
       "quantity": 100,
       "order_type": "market",
       "strategy": "momentum"
     }' \
     http://localhost:8000/api/v1/orders
```

4. **Monitor Order Status**
```bash
curl -H "Authorization: Bearer YOUR_TOKEN" \
     http://localhost:8000/api/v1/orders/ord_123456789
```

5. **Check Updated Portfolio**
```bash
curl -H "Authorization: Bearer YOUR_TOKEN" \
     http://localhost:8000/api/v1/portfolio
```

## 🛠️ Development & Testing

### Interactive API Documentation
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### Testing Endpoints
Use the interactive documentation or tools like Postman, curl, or HTTPie to test endpoints.

### Environment Variables
Configure API behavior using environment variables:
- `TRADING_MODE`: "paper" or "live"
- `API_RATE_LIMIT`: Requests per minute
- `LOG_LEVEL`: "DEBUG", "INFO", "WARNING", "ERROR"

---

**📚 This API documentation provides comprehensive coverage of all available endpoints. For implementation details and advanced usage, refer to the main README.md and system architecture documentation.**