# Database Schema Design
## BTC Signal Trader - Comprehensive Data Architecture

This schema is designed to support:
- ✅ Real-time data collection
- ✅ Trading signal generation
- ✅ Position tracking
- ✅ Comprehensive backtesting
- ✅ Performance analysis
- ✅ Risk management

---

## 📊 Schema Overview

### 1. **Market Data Layer**
Stores raw and processed market data for analysis and backtesting.

#### `trades`
- **Purpose**: Store every individual trade from exchanges
- **Use Cases**:
  - CVD calculation
  - Volume analysis
  - Tick-level backtesting
  - Order flow analysis
- **Key Fields**:
  - `timestamp`: When trade occurred
  - `exchange, symbol`: Where and what
  - `price, size, side`: Trade details
  - `cvd_at_trade`: Cumulative volume delta at this trade

#### `orderbook_snapshots`
- **Purpose**: Periodic snapshots of the orderbook state
- **Use Cases**:
  - Imbalance analysis
  - Support/resistance detection
  - Liquidity analysis
  - Whale wall detection
- **Key Fields**:
  - `imbalance`: Bid/ask volume imbalance percentage
  - `bids_json, asks_json`: Full orderbook data (flexible JSONB)
  - `spread`: Bid-ask spread metrics

#### `candles` (OHLCV)
- **Purpose**: Aggregated price data for technical analysis
- **Use Cases**:
  - Chart visualization
  - Technical indicators (RSI, MACD, etc.)
  - Backtesting with candlestick strategies
  - Pattern recognition
- **Intervals**: 1m, 5m, 15m, 1h, 4h, 1d
- **Key Fields**:
  - `open, high, low, close, volume`: Standard OHLCV
  - `vwap`: Volume-weighted average price
  - `trade_count`: Number of trades in period

#### `cvd_snapshots`
- **Purpose**: Time series of CVD values
- **Use Cases**:
  - Trend analysis
  - Divergence detection
  - CVD-based signals
- **Key Fields**:
  - `cvd`: Current cumulative volume delta
  - `cvd_change`: Delta from previous snapshot
  - `cvd_ma_5, cvd_ma_20`: Moving averages

---

### 2. **Technical Indicators Layer**

#### `indicators`
- **Purpose**: Store calculated technical indicators
- **Use Cases**:
  - Multi-indicator strategies
  - Indicator-based signals
  - Backtesting with indicators
- **Supported Indicators**:
  - RSI (Relative Strength Index)
  - MACD (Moving Average Convergence Divergence)
  - Bollinger Bands
  - Moving Averages (SMA, EMA)
  - Volume indicators
  - Custom indicators
- **Flexible Design**: `metadata` JSONB field allows complex indicators

---

### 3. **Signal Generation Layer**

#### `signals`
- **Purpose**: Store every trading signal generated
- **Use Cases**:
  - Signal history tracking
  - Signal performance analysis
  - Backtesting signal accuracy
  - Strategy optimization
- **Key Fields**:
  - `signal_type`: STRONG_BUY, BUY, NEUTRAL, SELL, STRONG_SELL
  - `confidence_score`: 0-100 confidence rating
  - `reasons`: Array of why signal was generated
  - `warnings`: Array of risk factors
  - `price, cvd, orderbook_imbalance`: Market state at signal time

**Signal Quality Metrics**:
```sql
-- Signal win rate
SELECT
    signal_type,
    COUNT(*) as total_signals,
    AVG(CASE WHEN p.pnl > 0 THEN 1 ELSE 0 END) as win_rate,
    AVG(p.pnl) as avg_pnl
FROM signals s
JOIN positions p ON s.id = p.signal_id
GROUP BY signal_type;
```

---

### 4. **Trading & Positions Layer**

#### `positions`
- **Purpose**: Track all trades (real or simulated)
- **Use Cases**:
  - Live trading tracking
  - Backtest results
  - P&L calculation
  - Risk management
  - Performance analysis
- **Lifecycle**:
  1. `entry_timestamp`: Position opened
  2. `status`: open → closed/stopped
  3. `exit_timestamp`: Position closed
  4. `pnl`: Calculated profit/loss
- **Risk Management Fields**:
  - `stop_loss`: Stop loss price
  - `take_profit`: Take profit target
  - `quantity`: Position size
- **Backtest Support**:
  - `is_backtest`: Flag for backtest positions
  - `backtest_id`: Links to backtest run

---

### 5. **Backtesting Layer**

#### `backtest_runs`
- **Purpose**: Store backtesting results and metadata
- **Use Cases**:
  - Strategy comparison
  - Parameter optimization
  - Performance tracking
  - Historical validation
- **Performance Metrics**:
  - `win_rate`: Percentage of winning trades
  - `total_pnl`: Total profit/loss
  - `sharpe_ratio`: Risk-adjusted returns
  - `max_drawdown`: Largest peak-to-trough decline
  - `profit_factor`: Gross profit / gross loss
- **Configuration**:
  - `strategy_config`: Full strategy parameters (JSONB)
  - `start_date, end_date`: Test period

#### `backtest_equity_curve`
- **Purpose**: Detailed equity progression during backtest
- **Use Cases**:
  - Visualize equity growth
  - Identify drawdown periods
  - Compare strategies
- **Key Fields**:
  - `timestamp`: Point in time
  - `equity`: Account balance at that time
  - `drawdown`: Current drawdown percentage

**Example Backtest Query**:
```sql
-- Compare strategies
SELECT
    strategy_name,
    COUNT(*) as runs,
    AVG(win_rate) as avg_win_rate,
    AVG(sharpe_ratio) as avg_sharpe,
    AVG(max_drawdown) as avg_max_dd
FROM backtest_runs
WHERE status = 'completed'
GROUP BY strategy_name
ORDER BY avg_sharpe DESC;
```

---

### 6. **System & Monitoring Layer**

#### `system_metrics`
- **Purpose**: Track system health and performance
- **Use Cases**:
  - Monitor data collection rates
  - Track signal generation latency
  - Identify bottlenecks
- **Metrics Examples**:
  - `trades_per_second`
  - `orderbook_update_latency`
  - `signal_generation_time`

#### `alerts`
- **Purpose**: Store alerts and notifications
- **Use Cases**:
  - Telegram notifications
  - Email alerts
  - System warnings
  - Critical event tracking
- **Alert Types**:
  - `signal_generated`: New trading signal
  - `position_opened`: New position entered
  - `stop_loss`: Stop loss triggered
  - `take_profit`: Take profit hit
  - `system_error`: System issues

---

## 🔍 Key Design Decisions

### 1. **Timestamp-First Design**
Every table has `timestamp` as a primary time reference for:
- Time-series analysis
- Backtesting chronology
- Performance tracking

### 2. **JSONB for Flexibility**
Using PostgreSQL JSONB for:
- `orderbook_snapshots`: Full orderbook data
- `indicators`: Complex indicator values
- `strategy_config`: Flexible strategy parameters
- `metadata`: Additional context

Benefits:
- Schema flexibility
- No data loss
- Easy to query specific fields

### 3. **Exchange & Symbol Normalization**
Separate fields for `exchange` and `symbol`:
- Support multi-exchange analysis
- Cross-exchange arbitrage
- Exchange-specific strategies

### 4. **Backtest Isolation**
`is_backtest` flag on positions:
- Separate backtest from live data
- Enable production/test environments
- Clear audit trail

### 5. **Indexed for Performance**
Strategic indexes on:
- Timestamps (most queries are time-based)
- Exchange + Symbol combinations
- Signal types and confidence
- P&L and performance metrics

---

## 📈 Common Query Patterns

### Real-Time Trading

```sql
-- Latest market state
SELECT
    (SELECT cvd FROM cvd_snapshots
     WHERE exchange = 'coinbase'
     ORDER BY timestamp DESC LIMIT 1) as cvd,
    (SELECT imbalance FROM orderbook_snapshots
     WHERE exchange = 'kraken'
     ORDER BY timestamp DESC LIMIT 1) as imbalance,
    (SELECT close FROM candles
     WHERE exchange = 'coinbase' AND interval = '1m'
     ORDER BY timestamp DESC LIMIT 1) as price;
```

### Performance Analysis

```sql
-- Last 7 days performance
SELECT
    DATE(entry_timestamp) as date,
    COUNT(*) as trades,
    SUM(pnl) as daily_pnl,
    AVG(pnl_pct) as avg_return
FROM positions
WHERE status = 'closed'
  AND entry_timestamp > NOW() - INTERVAL '7 days'
GROUP BY DATE(entry_timestamp)
ORDER BY date DESC;
```

### Signal Quality

```sql
-- Best performing signal types
SELECT
    s.signal_type,
    s.confidence_level,
    COUNT(*) as count,
    AVG(p.pnl) as avg_pnl,
    STDDEV(p.pnl) as pnl_stddev
FROM signals s
JOIN positions p ON s.id = p.signal_id
WHERE p.status = 'closed'
GROUP BY s.signal_type, s.confidence_level
HAVING COUNT(*) > 10
ORDER BY avg_pnl DESC;
```

### Backtest Comparison

```sql
-- Compare strategy performance
WITH strategy_metrics AS (
    SELECT
        backtest_id,
        strategy_name,
        win_rate,
        sharpe_ratio,
        max_drawdown,
        total_pnl
    FROM backtest_runs
    WHERE status = 'completed'
)
SELECT *
FROM strategy_metrics
ORDER BY sharpe_ratio DESC
LIMIT 10;
```

---

## 🎯 Next Steps

1. **Database Setup**: Create PostgreSQL database
2. **ORM Models**: Create SQLAlchemy models
3. **Data Writers**: Implement data collection → DB storage
4. **Backtesting Engine**: Build backtest execution framework
5. **Analytics Dashboard**: Visualize performance

---

## 📊 Storage Estimates

### Data Volume Projections (1 year)

**Trades** (high frequency):
- ~10 trades/sec = 315M trades/year
- ~50 bytes/row = 15.75 GB/year

**Orderbook Snapshots** (1/second):
- 31.5M snapshots/year
- ~500 bytes/row = 15.75 GB/year

**Candles** (multiple timeframes):
- 1m: 525,600/year
- 5m: 105,120/year
- Total ~1-2 GB/year

**Signals** (moderate frequency):
- ~100 signals/day = 36,500/year
- ~1 KB/row = 36.5 MB/year

**Total**: ~35-40 GB/year (easily manageable)

### Optimization Strategies

1. **Partitioning**: Time-based partitions (monthly)
2. **Archival**: Move old data to cold storage
3. **Aggregation**: Pre-calculate common metrics
4. **Compression**: PostgreSQL table compression
5. **Indexes**: Selective indexing on hot paths

---

## 🔐 Data Integrity

### Constraints
- Unique constraints prevent duplicates
- Foreign keys maintain referential integrity
- NOT NULL ensures required data

### Validation
- Price/volume must be positive
- Timestamps must be sequential
- P&L calculation verification

### Backup Strategy
- Daily full backups
- Continuous WAL archiving
- Point-in-time recovery capability
