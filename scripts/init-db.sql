-- Initialize trading database
-- This script runs when the PostgreSQL container starts for the first time

-- Create the trading database if it doesn't exist
CREATE DATABASE trading_db;

-- Create the trading user if it doesn't exist
DO
$do$
BEGIN
   IF NOT EXISTS (
      SELECT FROM pg_catalog.pg_roles
      WHERE  rolname = 'trading_user') THEN

      CREATE ROLE trading_user LOGIN PASSWORD 'trading_pass';
   END IF;
END
$do$;

-- Grant privileges to the trading user
GRANT ALL PRIVILEGES ON DATABASE trading_db TO trading_user;

-- Connect to the trading database
\c trading_db;

-- Create tables for the trading system
CREATE TABLE IF NOT EXISTS trades (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(10) NOT NULL,
    side VARCHAR(4) NOT NULL CHECK (side IN ('buy', 'sell')),
    quantity DECIMAL(10,2) NOT NULL,
    price DECIMAL(10,2) NOT NULL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    strategy VARCHAR(50),
    profit_loss DECIMAL(10,2),
    status VARCHAR(20) DEFAULT 'pending'
);

CREATE TABLE IF NOT EXISTS positions (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(10) NOT NULL UNIQUE,
    quantity DECIMAL(10,2) NOT NULL DEFAULT 0,
    avg_price DECIMAL(10,2) NOT NULL DEFAULT 0,
    market_value DECIMAL(10,2) NOT NULL DEFAULT 0,
    unrealized_pnl DECIMAL(10,2) NOT NULL DEFAULT 0,
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS portfolio (
    id SERIAL PRIMARY KEY,
    total_value DECIMAL(12,2) NOT NULL DEFAULT 0,
    cash DECIMAL(12,2) NOT NULL DEFAULT 0,
    buying_power DECIMAL(12,2) NOT NULL DEFAULT 0,
    day_trade_count INTEGER DEFAULT 0,
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS market_data (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(10) NOT NULL,
    price DECIMAL(10,2) NOT NULL,
    volume BIGINT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    source VARCHAR(20)
);

CREATE TABLE IF NOT EXISTS alerts (
    id SERIAL PRIMARY KEY,
    message TEXT NOT NULL,
    level VARCHAR(10) NOT NULL CHECK (level IN ('info', 'warning', 'error')),
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    acknowledged BOOLEAN DEFAULT FALSE
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol);
CREATE INDEX IF NOT EXISTS idx_trades_timestamp ON trades(timestamp);
CREATE INDEX IF NOT EXISTS idx_positions_symbol ON positions(symbol);
CREATE INDEX IF NOT EXISTS idx_market_data_symbol ON market_data(symbol);
CREATE INDEX IF NOT EXISTS idx_market_data_timestamp ON market_data(timestamp);
CREATE INDEX IF NOT EXISTS idx_alerts_timestamp ON alerts(timestamp);

-- Insert initial portfolio record
INSERT INTO portfolio (total_value, cash, buying_power) 
VALUES (10000.00, 10000.00, 10000.00)
ON CONFLICT DO NOTHING;

-- Grant permissions to the trading user
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO trading_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO trading_user;