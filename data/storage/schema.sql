-- BTC Signal Trader Database Schema
-- Comprehensive schema for real-time trading and backtesting

-- ============================================================================
-- MARKET DATA TABLES
-- ============================================================================

-- Raw trades from exchanges (for CVD calculation and analysis)
CREATE TABLE IF NOT EXISTS trades (
    id BIGSERIAL PRIMARY KEY,
    timestamp TIMESTAMP NOT NULL,
    exchange VARCHAR(50) NOT NULL,  -- coinbase, kraken, binance
    symbol VARCHAR(20) NOT NULL,    -- BTC-USD, XBT-USD, etc.
    trade_id VARCHAR(100),          -- Exchange trade ID
    price DECIMAL(18, 8) NOT NULL,
    size DECIMAL(18, 8) NOT NULL,
    side VARCHAR(10) NOT NULL,      -- buy, sell
    is_buyer_maker BOOLEAN,
    cvd_at_trade DECIMAL(18, 8),    -- CVD value at this trade

    -- Indexes for fast queries
    CONSTRAINT trades_exchange_symbol_timestamp_idx
        UNIQUE (exchange, symbol, timestamp, trade_id)
);

CREATE INDEX idx_trades_timestamp ON trades(timestamp DESC);
CREATE INDEX idx_trades_exchange_symbol ON trades(exchange, symbol);
CREATE INDEX idx_trades_cvd ON trades(cvd_at_trade);


-- Orderbook snapshots (for imbalance analysis)
CREATE TABLE IF NOT EXISTS orderbook_snapshots (
    id BIGSERIAL PRIMARY KEY,
    timestamp TIMESTAMP NOT NULL,
    exchange VARCHAR(50) NOT NULL,
    symbol VARCHAR(20) NOT NULL,

    -- Best bid/ask
    best_bid DECIMAL(18, 8),
    best_ask DECIMAL(18, 8),
    spread DECIMAL(18, 8),
    spread_pct DECIMAL(10, 6),

    -- Aggregated metrics (top N levels)
    total_bid_volume DECIMAL(18, 8),
    total_ask_volume DECIMAL(18, 8),
    imbalance DECIMAL(10, 4),  -- Percentage: (bid_vol - ask_vol) / total_vol * 100

    -- Full orderbook data (JSON for flexibility)
    bids_json JSONB,  -- [{"price": 50000, "volume": 1.5}, ...]
    asks_json JSONB,

    -- Metadata
    depth_levels INTEGER,  -- Number of levels captured

    CONSTRAINT orderbook_unique_snapshot
        UNIQUE (exchange, symbol, timestamp)
);

CREATE INDEX idx_orderbook_timestamp ON orderbook_snapshots(timestamp DESC);
CREATE INDEX idx_orderbook_imbalance ON orderbook_snapshots(imbalance);
CREATE INDEX idx_orderbook_exchange_symbol ON orderbook_snapshots(exchange, symbol);


-- OHLCV candles (for technical analysis and backtesting)
CREATE TABLE IF NOT EXISTS candles (
    id BIGSERIAL PRIMARY KEY,
    timestamp TIMESTAMP NOT NULL,
    exchange VARCHAR(50) NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    interval VARCHAR(10) NOT NULL,  -- 1m, 5m, 15m, 1h, 4h, 1d

    open DECIMAL(18, 8) NOT NULL,
    high DECIMAL(18, 8) NOT NULL,
    low DECIMAL(18, 8) NOT NULL,
    close DECIMAL(18, 8) NOT NULL,
    volume DECIMAL(18, 8) NOT NULL,

    -- Additional metrics
    trade_count INTEGER,
    vwap DECIMAL(18, 8),  -- Volume-weighted average price

    CONSTRAINT candles_unique
        UNIQUE (exchange, symbol, interval, timestamp)
);

CREATE INDEX idx_candles_timestamp ON candles(timestamp DESC);
CREATE INDEX idx_candles_exchange_symbol_interval ON candles(exchange, symbol, interval);


-- CVD (Cumulative Volume Delta) time series
CREATE TABLE IF NOT EXISTS cvd_snapshots (
    id BIGSERIAL PRIMARY KEY,
    timestamp TIMESTAMP NOT NULL,
    exchange VARCHAR(50) NOT NULL,
    symbol VARCHAR(20) NOT NULL,

    cvd DECIMAL(18, 8) NOT NULL,
    cvd_change DECIMAL(18, 8),  -- Change from previous snapshot

    -- Moving averages for trend analysis
    cvd_ma_5 DECIMAL(18, 8),
    cvd_ma_20 DECIMAL(18, 8),

    CONSTRAINT cvd_unique_snapshot
        UNIQUE (exchange, symbol, timestamp)
);

CREATE INDEX idx_cvd_timestamp ON cvd_snapshots(timestamp DESC);
CREATE INDEX idx_cvd_exchange_symbol ON cvd_snapshots(exchange, symbol);


-- ============================================================================
-- TECHNICAL INDICATORS
-- ============================================================================

CREATE TABLE IF NOT EXISTS indicators (
    id BIGSERIAL PRIMARY KEY,
    timestamp TIMESTAMP NOT NULL,
    exchange VARCHAR(50) NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    interval VARCHAR(10) NOT NULL,

    -- Indicator details
    indicator_name VARCHAR(50) NOT NULL,  -- RSI, MACD, BB, etc.
    value DECIMAL(18, 8),

    -- Additional values for complex indicators
    metadata JSONB,  -- {"signal": 0.5, "histogram": 0.2} for MACD, etc.

    CONSTRAINT indicators_unique
        UNIQUE (exchange, symbol, interval, indicator_name, timestamp)
);

CREATE INDEX idx_indicators_timestamp ON indicators(timestamp DESC);
CREATE INDEX idx_indicators_name ON indicators(indicator_name);


-- ============================================================================
-- TRADING SIGNALS
-- ============================================================================

CREATE TABLE IF NOT EXISTS signals (
    id BIGSERIAL PRIMARY KEY,
    timestamp TIMESTAMP NOT NULL,

    -- Signal details
    signal_type VARCHAR(20) NOT NULL,  -- STRONG_BUY, BUY, NEUTRAL, SELL, STRONG_SELL
    confidence_level VARCHAR(20) NOT NULL,  -- VERY_HIGH, HIGH, MEDIUM, LOW, VERY_LOW
    confidence_score DECIMAL(5, 2),  -- 0-100

    -- Market data at signal time
    price DECIMAL(18, 8),
    cvd DECIMAL(18, 8),
    cvd_trend VARCHAR(20),  -- bullish, bearish, neutral
    orderbook_imbalance DECIMAL(10, 4),

    -- Signal reasoning
    reasons TEXT[],  -- Array of reason strings
    warnings TEXT[],  -- Array of warning strings

    -- Source data references
    exchange VARCHAR(50),
    symbol VARCHAR(20),

    -- Metadata
    strategy_name VARCHAR(100),  -- Name of strategy that generated signal
    metadata JSONB  -- Additional data
);

CREATE INDEX idx_signals_timestamp ON signals(timestamp DESC);
CREATE INDEX idx_signals_type ON signals(signal_type);
CREATE INDEX idx_signals_confidence ON signals(confidence_score DESC);


-- ============================================================================
-- POSITIONS & TRADES (EXECUTED OR SIMULATED)
-- ============================================================================

CREATE TABLE IF NOT EXISTS positions (
    id BIGSERIAL PRIMARY KEY,

    -- Position identification
    position_id VARCHAR(100) UNIQUE NOT NULL,
    signal_id BIGINT REFERENCES signals(id),

    -- Position details
    exchange VARCHAR(50) NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    side VARCHAR(10) NOT NULL,  -- long, short

    -- Entry
    entry_timestamp TIMESTAMP NOT NULL,
    entry_price DECIMAL(18, 8) NOT NULL,
    entry_signal_type VARCHAR(20),

    -- Exit
    exit_timestamp TIMESTAMP,
    exit_price DECIMAL(18, 8),
    exit_signal_type VARCHAR(20),
    exit_reason VARCHAR(100),  -- signal, stop_loss, take_profit, manual

    -- Size and risk
    quantity DECIMAL(18, 8) NOT NULL,
    notional_value DECIMAL(18, 2),  -- Entry price * quantity

    -- Risk management
    stop_loss DECIMAL(18, 8),
    take_profit DECIMAL(18, 8),

    -- Results
    pnl DECIMAL(18, 8),  -- Profit/Loss in BTC or USD
    pnl_pct DECIMAL(10, 4),  -- Percentage return
    fees DECIMAL(18, 8),

    -- Status
    status VARCHAR(20) NOT NULL,  -- open, closed, stopped, partial

    -- Metadata
    is_backtest BOOLEAN DEFAULT FALSE,
    backtest_id VARCHAR(100),
    notes TEXT
);

CREATE INDEX idx_positions_status ON positions(status);
CREATE INDEX idx_positions_entry_timestamp ON positions(entry_timestamp DESC);
CREATE INDEX idx_positions_pnl ON positions(pnl DESC);
CREATE INDEX idx_positions_backtest ON positions(backtest_id) WHERE is_backtest = TRUE;


-- ============================================================================
-- BACKTESTING
-- ============================================================================

CREATE TABLE IF NOT EXISTS backtest_runs (
    id BIGSERIAL PRIMARY KEY,
    backtest_id VARCHAR(100) UNIQUE NOT NULL,

    -- Configuration
    strategy_name VARCHAR(100) NOT NULL,
    strategy_config JSONB,  -- Full strategy parameters

    -- Time period
    start_date TIMESTAMP NOT NULL,
    end_date TIMESTAMP NOT NULL,

    -- Data sources
    exchange VARCHAR(50),
    symbol VARCHAR(20),

    -- Results summary
    total_trades INTEGER,
    winning_trades INTEGER,
    losing_trades INTEGER,
    win_rate DECIMAL(5, 2),

    -- Performance metrics
    total_pnl DECIMAL(18, 8),
    total_pnl_pct DECIMAL(10, 4),
    avg_pnl_per_trade DECIMAL(18, 8),
    largest_win DECIMAL(18, 8),
    largest_loss DECIMAL(18, 8),

    -- Risk metrics
    sharpe_ratio DECIMAL(10, 4),
    sortino_ratio DECIMAL(10, 4),
    max_drawdown DECIMAL(10, 4),
    max_drawdown_duration INTERVAL,

    -- Additional metrics
    profit_factor DECIMAL(10, 4),  -- Gross profit / gross loss
    avg_trade_duration INTERVAL,
    total_fees DECIMAL(18, 8),

    -- Execution
    run_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    duration_seconds INTEGER,

    -- Status
    status VARCHAR(20),  -- running, completed, failed
    error_message TEXT,

    -- Metadata
    notes TEXT
);

CREATE INDEX idx_backtest_strategy ON backtest_runs(strategy_name);
CREATE INDEX idx_backtest_dates ON backtest_runs(start_date, end_date);
CREATE INDEX idx_backtest_sharpe ON backtest_runs(sharpe_ratio DESC);


-- Detailed equity curve for backtests
CREATE TABLE IF NOT EXISTS backtest_equity_curve (
    id BIGSERIAL PRIMARY KEY,
    backtest_id VARCHAR(100) REFERENCES backtest_runs(backtest_id),
    timestamp TIMESTAMP NOT NULL,

    equity DECIMAL(18, 8) NOT NULL,
    drawdown DECIMAL(10, 4),
    position_count INTEGER,

    CONSTRAINT equity_curve_unique
        UNIQUE (backtest_id, timestamp)
);

CREATE INDEX idx_equity_backtest ON backtest_equity_curve(backtest_id);


-- ============================================================================
-- SYSTEM & MONITORING
-- ============================================================================

CREATE TABLE IF NOT EXISTS system_metrics (
    id BIGSERIAL PRIMARY KEY,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    metric_name VARCHAR(100) NOT NULL,
    metric_value DECIMAL(18, 8),
    metadata JSONB,

    -- Categories: data_collection, signal_generation, execution, etc.
    category VARCHAR(50)
);

CREATE INDEX idx_system_metrics_timestamp ON system_metrics(timestamp DESC);
CREATE INDEX idx_system_metrics_name ON system_metrics(metric_name);


-- Alerts and notifications
CREATE TABLE IF NOT EXISTS alerts (
    id BIGSERIAL PRIMARY KEY,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    alert_type VARCHAR(50) NOT NULL,  -- signal_generated, position_opened, stop_loss, etc.
    severity VARCHAR(20),  -- info, warning, error, critical

    title VARCHAR(200),
    message TEXT,

    -- Associated data
    signal_id BIGINT REFERENCES signals(id),
    position_id BIGINT REFERENCES positions(id),

    -- Notification status
    sent BOOLEAN DEFAULT FALSE,
    sent_timestamp TIMESTAMP,
    notification_channels TEXT[],  -- telegram, email, etc.

    metadata JSONB
);

CREATE INDEX idx_alerts_timestamp ON alerts(timestamp DESC);
CREATE INDEX idx_alerts_type ON alerts(alert_type);


-- ============================================================================
-- VIEWS FOR COMMON QUERIES
-- ============================================================================

-- Recent signals with performance
CREATE VIEW recent_signals_with_performance AS
SELECT
    s.*,
    p.pnl,
    p.pnl_pct,
    p.status as position_status,
    p.exit_timestamp
FROM signals s
LEFT JOIN positions p ON s.id = p.signal_id
ORDER BY s.timestamp DESC
LIMIT 100;


-- Active positions summary
CREATE VIEW active_positions_summary AS
SELECT
    exchange,
    symbol,
    side,
    COUNT(*) as position_count,
    SUM(quantity) as total_quantity,
    SUM(notional_value) as total_notional,
    AVG(entry_price) as avg_entry_price,
    SUM(pnl) as total_pnl
FROM positions
WHERE status = 'open'
GROUP BY exchange, symbol, side;


-- Daily performance summary
CREATE VIEW daily_performance AS
SELECT
    DATE(entry_timestamp) as date,
    COUNT(*) as total_trades,
    SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as winning_trades,
    SUM(CASE WHEN pnl < 0 THEN 1 ELSE 0 END) as losing_trades,
    SUM(pnl) as total_pnl,
    AVG(pnl) as avg_pnl,
    MAX(pnl) as max_win,
    MIN(pnl) as max_loss
FROM positions
WHERE status = 'closed'
GROUP BY DATE(entry_timestamp)
ORDER BY date DESC;
