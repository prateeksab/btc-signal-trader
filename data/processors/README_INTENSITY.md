# Trade Intensity Analyzer

Real-time trade flow analysis that calculates market pressure, aggression, and velocity metrics.

## Overview

The Trade Intensity Analyzer processes raw trade data to generate actionable market microstructure metrics:

- **Trade Frequency**: Trades per second across 10s, 30s, and 60s windows
- **Trade Size**: Average trade sizes to detect whale activity
- **Buy/Sell Pressure**: Ratio of buy vs sell volume
- **Aggression Score**: -1 (aggressive selling) to +1 (aggressive buying)
- **Velocity**: Is trading accelerating or decelerating?

## Setup

### 1. Create Database Table

```bash
venv/bin/python scripts/add_trade_intensity_table.py
```

### 2. Start Data Collectors

Make sure trade collectors are running:

```bash
# These should already be running from your setup
ps aux | grep "data/collectors"
```

If not running, start them:
```bash
venv/bin/python data/collectors/coinbase_trades.py &
venv/bin/python data/collectors/kraken_trades.py &
```

### 3. Start Intensity Analyzer Service

Run the collector service (analyzes every 5 seconds):

```bash
# Foreground (see live output)
venv/bin/python data/processors/collector_service.py

# Background
venv/bin/python data/processors/collector_service.py &
```

## Usage

### Test Current Intensity

Run a one-time analysis:

```bash
venv/bin/python data/processors/trade_intensity.py
```

Example output:
```
================================================================================
TRADE INTENSITY ANALYZER - TEST
================================================================================

Timestamp: 2025-11-26 21:56:25
Exchange: coinbase
Symbol: BTC-USD

--------------------------------------------------------------------------------
TRADE FREQUENCY
--------------------------------------------------------------------------------
  10s window: 2.45 TPS
  30s window: 3.12 TPS
  60s window: 2.89 TPS

--------------------------------------------------------------------------------
TRADE SIZE
--------------------------------------------------------------------------------
  30s avg: 0.0523 BTC
  60s avg: 0.0498 BTC

--------------------------------------------------------------------------------
BUY/SELL PRESSURE
--------------------------------------------------------------------------------
  Buy/Sell Ratio: 0.68 (68.0% buys)
  Aggression Score: 0.36 (AGGRESSIVE BUYING)

--------------------------------------------------------------------------------
VELOCITY
--------------------------------------------------------------------------------
  Velocity Change: +15.32% (ACCELERATING)

================================================================================
```

### Query Intensity Data

```python
from data.storage.database import db_manager
from data.storage.models import TradeIntensity
from sqlalchemy import desc

with db_manager.get_session() as session:
    # Get latest intensity
    latest = session.query(TradeIntensity).order_by(
        desc(TradeIntensity.timestamp)
    ).first()

    print(f"TPS (60s): {latest.trades_per_sec_60s}")
    print(f"Aggression: {latest.aggression_score}")
    print(f"Velocity: {latest.velocity_change}")
```

## Metrics Explained

### Trade Frequency (TPS)
- **10s window**: Very short-term spike detection
- **30s window**: Short-term flow
- **60s window**: Medium-term sustained activity

**Interpretation**:
- High TPS (>5): Active trading, high liquidity
- Low TPS (<1): Quiet market, low liquidity

### Buy/Sell Ratio
Scale: 0.0 to 1.0 (percentage of volume from buys)
- **0.5**: Balanced (50% buys, 50% sells)
- **>0.6**: Buy pressure dominates
- **<0.4**: Sell pressure dominates

### Aggression Score
Scale: -1.0 to 1.0
- **+0.3 to +1.0**: AGGRESSIVE BUYING (strong buy pressure)
- **-0.3 to -1.0**: AGGRESSIVE SELLING (strong sell pressure)
- **-0.3 to +0.3**: NEUTRAL (balanced)

**Calculation**:
```python
if buy_sell_ratio > 0.6:
    aggression = (buy_sell_ratio - 0.5) * 2  # 0 to 1
elif buy_sell_ratio < 0.4:
    aggression = (0.5 - buy_sell_ratio) * -2  # -1 to 0
else:
    aggression = 0  # neutral
```

### Velocity Change
Percentage change in trade frequency (60s window)
- **Positive**: Trading is accelerating
- **Negative**: Trading is decelerating
- **>0.1 (10%)**: Strong acceleration
- **<-0.1 (-10%)**: Strong deceleration

## Files

- `data/storage/models.py`: TradeIntensity database model
- `data/processors/trade_intensity.py`: Core analyzer logic
- `data/processors/collector_service.py`: Background service (runs every 5s)
- `scripts/add_trade_intensity_table.py`: Database migration

## Database Schema

```sql
CREATE TABLE trade_intensity (
    id INTEGER PRIMARY KEY,
    timestamp TIMESTAMP NOT NULL,
    exchange VARCHAR(50) NOT NULL,
    symbol VARCHAR(20) NOT NULL,

    -- Frequency metrics
    trades_per_sec_10s DECIMAL(10,4),
    trades_per_sec_30s DECIMAL(10,4),
    trades_per_sec_60s DECIMAL(10,4),

    -- Size metrics
    avg_trade_size_30s DECIMAL(18,8),
    avg_trade_size_60s DECIMAL(18,8),

    -- Pressure metrics
    buy_sell_ratio DECIMAL(10,4),
    aggression_score DECIMAL(10,4),

    -- Velocity
    velocity_change DECIMAL(10,4)
);

CREATE INDEX idx_trade_intensity_timestamp ON trade_intensity(timestamp);
CREATE INDEX idx_trade_intensity_exchange_symbol ON trade_intensity(exchange, symbol);
```

## Integration with Trading Signals

The trade intensity metrics can enhance your trading signals:

1. **Confirmation**: High aggression score confirms CVD signals
2. **Divergence**: Rising price + negative aggression = potential reversal
3. **Velocity**: Acceleration often precedes volatility spikes
4. **Liquidity**: Low TPS = wider spreads, higher slippage

Example:
```python
# Strong buy signal = CVD rising + aggressive buying + acceleration
if (cvd_trend == 'up' and
    aggression_score > 0.3 and
    velocity_change > 0.1):
    signal = 'STRONG_BUY'
```
