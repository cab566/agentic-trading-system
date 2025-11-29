-- PostgreSQL Master Database Initialization
-- This script configures the master database for streaming replication

-- Create replication user
CREATE USER replicator WITH REPLICATION ENCRYPTED PASSWORD 'repl_pass';

-- Grant necessary permissions
GRANT CONNECT ON DATABASE trading_db TO replicator;
GRANT USAGE ON SCHEMA public TO replicator;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO replicator;
GRANT SELECT ON ALL SEQUENCES IN SCHEMA public TO replicator;

-- Create replication slot for slave
SELECT pg_create_physical_replication_slot('replica_slot');

-- Configure pg_hba.conf for replication (this will be handled by the container)
-- host replication replicator 0.0.0.0/0 md5

-- Initialize trading database tables
CREATE TABLE IF NOT EXISTS trades (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(10) NOT NULL,
    side VARCHAR(4) NOT NULL CHECK (side IN ('buy', 'sell')),
    quantity DECIMAL(15,8) NOT NULL,
    price DECIMAL(15,8) NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    strategy VARCHAR(50),
    profit_loss DECIMAL(15,8),
    status VARCHAR(20) DEFAULT 'pending',
    commission DECIMAL(10,4),
    slippage DECIMAL(10,4),
    metadata JSONB
);

CREATE TABLE IF NOT EXISTS positions (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(10) NOT NULL UNIQUE,
    quantity DECIMAL(15,8) NOT NULL DEFAULT 0,
    avg_price DECIMAL(15,8) NOT NULL DEFAULT 0,
    market_value DECIMAL(15,2) NOT NULL DEFAULT 0,
    unrealized_pnl DECIMAL(15,2) NOT NULL DEFAULT 0,
    last_updated TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS portfolio (
    id SERIAL PRIMARY KEY,
    total_value DECIMAL(15,2) NOT NULL DEFAULT 0,
    cash_balance DECIMAL(15,2) NOT NULL DEFAULT 100000,
    total_pnl DECIMAL(15,2) NOT NULL DEFAULT 0,
    daily_pnl DECIMAL(15,2) NOT NULL DEFAULT 0,
    last_updated TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS risk_metrics (
    id SERIAL PRIMARY KEY,
    portfolio_value DECIMAL(15,2) NOT NULL,
    var_95 DECIMAL(15,2) NOT NULL,
    cvar_95 DECIMAL(15,2) NOT NULL,
    max_drawdown DECIMAL(8,4) NOT NULL,
    sharpe_ratio DECIMAL(8,4),
    calculated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol);
CREATE INDEX IF NOT EXISTS idx_trades_timestamp ON trades(timestamp);
CREATE INDEX IF NOT EXISTS idx_trades_strategy ON trades(strategy);
CREATE INDEX IF NOT EXISTS idx_positions_symbol ON positions(symbol);
CREATE INDEX IF NOT EXISTS idx_risk_metrics_calculated_at ON risk_metrics(calculated_at);

-- Create function to update last_updated timestamp
CREATE OR REPLACE FUNCTION update_last_updated_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.last_updated = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create triggers for automatic timestamp updates
CREATE TRIGGER update_positions_last_updated 
    BEFORE UPDATE ON positions 
    FOR EACH ROW EXECUTE FUNCTION update_last_updated_column();

CREATE TRIGGER update_portfolio_last_updated 
    BEFORE UPDATE ON portfolio 
    FOR EACH ROW EXECUTE FUNCTION update_last_updated_column();

-- Insert initial portfolio record
INSERT INTO portfolio (cash_balance) VALUES (100000.00) ON CONFLICT DO NOTHING;

-- Grant permissions to trading_user
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO trading_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO trading_user;
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA public TO trading_user;