-- Create BTC price candles table for OHLC data

CREATE TABLE IF NOT EXISTS btc_price_candles (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP NOT NULL,
    timeframe VARCHAR(10) NOT NULL,
    exchange VARCHAR(50) NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    open DECIMAL(18, 8) NOT NULL,
    high DECIMAL(18, 8) NOT NULL,
    low DECIMAL(18, 8) NOT NULL,
    close DECIMAL(18, 8) NOT NULL,
    volume DECIMAL(18, 8),
    num_trades INTEGER,

    -- Unique constraint to prevent duplicate candles
    CONSTRAINT uq_btc_candle UNIQUE (timestamp, timeframe, exchange, symbol)
);

-- Create indices for efficient queries
CREATE INDEX IF NOT EXISTS idx_btc_candle_timestamp ON btc_price_candles(timestamp);
CREATE INDEX IF NOT EXISTS idx_btc_candle_timeframe ON btc_price_candles(timeframe);
CREATE INDEX IF NOT EXISTS idx_btc_candle_exchange ON btc_price_candles(exchange);
CREATE INDEX IF NOT EXISTS idx_btc_candle_timestamp_timeframe ON btc_price_candles(timestamp, timeframe);
CREATE INDEX IF NOT EXISTS idx_btc_candle_exchange_timeframe ON btc_price_candles(exchange, timeframe);
CREATE INDEX IF NOT EXISTS idx_btc_candle_full ON btc_price_candles(exchange, timeframe, timestamp);
