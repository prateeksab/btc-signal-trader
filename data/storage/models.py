"""
SQLAlchemy Database Models

ORM models for the BTC Signal Trader database.
"""

from datetime import datetime
from sqlalchemy import (
    Column, Integer, BigInteger, String, DECIMAL, Boolean,
    TIMESTAMP, Text, ForeignKey, Index, JSON, UniqueConstraint
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()


# ============================================================================
# MARKET DATA MODELS
# ============================================================================

class Trade(Base):
    """Individual trades from exchanges"""
    __tablename__ = 'trades'

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(TIMESTAMP, nullable=False, index=True)
    exchange = Column(String(50), nullable=False)
    symbol = Column(String(20), nullable=False)
    trade_id = Column(String(100))
    price = Column(DECIMAL(18, 8), nullable=False)
    size = Column(DECIMAL(18, 8), nullable=False)
    side = Column(String(10), nullable=False)
    is_buyer_maker = Column(Boolean)
    cvd_at_trade = Column(DECIMAL(18, 8))

    __table_args__ = (
        Index('idx_trades_timestamp', 'timestamp'),
        Index('idx_trades_exchange_symbol', 'exchange', 'symbol'),
        Index('idx_trades_cvd', 'cvd_at_trade'),
    )

    def __repr__(self):
        return f"<Trade(exchange={self.exchange}, price={self.price}, size={self.size}, side={self.side})>"


class OrderbookSnapshot(Base):
    """Orderbook state snapshots"""
    __tablename__ = 'orderbook_snapshots'

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(TIMESTAMP, nullable=False, index=True)
    exchange = Column(String(50), nullable=False)
    symbol = Column(String(20), nullable=False)

    # Best bid/ask
    best_bid = Column(DECIMAL(18, 8))
    best_ask = Column(DECIMAL(18, 8))
    spread = Column(DECIMAL(18, 8))
    spread_pct = Column(DECIMAL(10, 6))

    # Aggregated metrics
    total_bid_volume = Column(DECIMAL(18, 8))
    total_ask_volume = Column(DECIMAL(18, 8))
    imbalance = Column(DECIMAL(10, 4))

    # Full orderbook data
    bids_json = Column(JSON)
    asks_json = Column(JSON)

    depth_levels = Column(Integer)

    __table_args__ = (
        Index('idx_orderbook_timestamp', 'timestamp'),
        Index('idx_orderbook_imbalance', 'imbalance'),
        Index('idx_orderbook_exchange_symbol', 'exchange', 'symbol'),
    )

    def __repr__(self):
        return f"<OrderbookSnapshot(exchange={self.exchange}, imbalance={self.imbalance})>"


class Candle(Base):
    """OHLCV candles for technical analysis"""
    __tablename__ = 'candles'

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(TIMESTAMP, nullable=False)
    exchange = Column(String(50), nullable=False)
    symbol = Column(String(20), nullable=False)
    interval = Column(String(10), nullable=False)  # 1m, 5m, 15m, 1h, 4h, 1d

    open = Column(DECIMAL(18, 8), nullable=False)
    high = Column(DECIMAL(18, 8), nullable=False)
    low = Column(DECIMAL(18, 8), nullable=False)
    close = Column(DECIMAL(18, 8), nullable=False)
    volume = Column(DECIMAL(18, 8), nullable=False)

    trade_count = Column(Integer)
    vwap = Column(DECIMAL(18, 8))

    __table_args__ = (
        Index('idx_candles_timestamp', 'timestamp'),
        Index('idx_candles_exchange_symbol_interval', 'exchange', 'symbol', 'interval'),
    )

    def __repr__(self):
        return f"<Candle(exchange={self.exchange}, interval={self.interval}, close={self.close})>"


class CVDSnapshot(Base):
    """CVD time series"""
    __tablename__ = 'cvd_snapshots'

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(TIMESTAMP, nullable=False, index=True)
    exchange = Column(String(50), nullable=False)
    symbol = Column(String(20), nullable=False)

    cvd = Column(DECIMAL(18, 8), nullable=False)
    cvd_change = Column(DECIMAL(18, 8))

    cvd_ma_5 = Column(DECIMAL(18, 8))
    cvd_ma_20 = Column(DECIMAL(18, 8))

    __table_args__ = (
        Index('idx_cvd_timestamp', 'timestamp'),
        Index('idx_cvd_exchange_symbol', 'exchange', 'symbol'),
    )

    def __repr__(self):
        return f"<CVDSnapshot(exchange={self.exchange}, cvd={self.cvd})>"


# ============================================================================
# TECHNICAL INDICATORS
# ============================================================================

class Indicator(Base):
    """Technical indicators"""
    __tablename__ = 'indicators'

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(TIMESTAMP, nullable=False, index=True)
    exchange = Column(String(50), nullable=False)
    symbol = Column(String(20), nullable=False)
    interval = Column(String(10), nullable=False)

    indicator_name = Column(String(50), nullable=False)
    value = Column(DECIMAL(18, 8))
    extra_metadata = Column(JSON)

    __table_args__ = (
        Index('idx_indicators_timestamp', 'timestamp'),
        Index('idx_indicators_name', 'indicator_name'),
    )

    def __repr__(self):
        return f"<Indicator(name={self.indicator_name}, value={self.value})>"


# ============================================================================
# TRADING SIGNALS
# ============================================================================

class Signal(Base):
    """Trading signals"""
    __tablename__ = 'signals'

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(TIMESTAMP, nullable=False, index=True)

    # Signal details
    signal_type = Column(String(20), nullable=False)
    confidence_level = Column(String(20), nullable=False)
    confidence_score = Column(DECIMAL(5, 2))

    # Market data at signal time
    price = Column(DECIMAL(18, 8))
    cvd = Column(DECIMAL(18, 8))
    cvd_trend = Column(String(20))
    orderbook_imbalance = Column(DECIMAL(10, 4))

    # Signal reasoning
    reasons = Column(JSON)  # Store as JSON array for SQLite compatibility
    warnings = Column(JSON)  # Store as JSON array for SQLite compatibility

    # Source data
    exchange = Column(String(50))
    symbol = Column(String(20))

    # Metadata
    strategy_name = Column(String(100))
    extra_metadata = Column(JSON)

    # Relationship to positions
    positions = relationship("Position", back_populates="signal")

    __table_args__ = (
        Index('idx_signals_timestamp', 'timestamp'),
        Index('idx_signals_type', 'signal_type'),
        Index('idx_signals_confidence', 'confidence_score'),
    )

    def __repr__(self):
        return f"<Signal(type={self.signal_type}, confidence={self.confidence_score})>"


# ============================================================================
# POSITIONS & TRADES
# ============================================================================

class Position(Base):
    """Trading positions (live or backtest)"""
    __tablename__ = 'positions'

    id = Column(Integer, primary_key=True, autoincrement=True)

    # Position identification
    position_id = Column(String(100), unique=True, nullable=False)
    signal_id = Column(Integer, ForeignKey('signals.id'))

    # Position details
    exchange = Column(String(50), nullable=False)
    symbol = Column(String(20), nullable=False)
    side = Column(String(10), nullable=False)  # long, short

    # Entry
    entry_timestamp = Column(TIMESTAMP, nullable=False, index=True)
    entry_price = Column(DECIMAL(18, 8), nullable=False)
    entry_signal_type = Column(String(20))

    # Exit
    exit_timestamp = Column(TIMESTAMP)
    exit_price = Column(DECIMAL(18, 8))
    exit_signal_type = Column(String(20))
    exit_reason = Column(String(100))

    # Size and risk
    quantity = Column(DECIMAL(18, 8), nullable=False)
    notional_value = Column(DECIMAL(18, 2))

    # Risk management
    stop_loss = Column(DECIMAL(18, 8))
    take_profit = Column(DECIMAL(18, 8))

    # Results
    pnl = Column(DECIMAL(18, 8))
    pnl_pct = Column(DECIMAL(10, 4))
    fees = Column(DECIMAL(18, 8))

    # Status
    status = Column(String(20), nullable=False)  # open, closed, stopped

    # Backtest
    is_backtest = Column(Boolean, default=False)
    backtest_id = Column(String(100))

    notes = Column(Text)

    # Relationship to signal
    signal = relationship("Signal", back_populates="positions")

    __table_args__ = (
        Index('idx_positions_status', 'status'),
        Index('idx_positions_entry_timestamp', 'entry_timestamp'),
        Index('idx_positions_pnl', 'pnl'),
        Index('idx_positions_backtest', 'backtest_id'),
    )

    def __repr__(self):
        return f"<Position(side={self.side}, entry={self.entry_price}, pnl={self.pnl})>"


# ============================================================================
# BACKTESTING
# ============================================================================

class BacktestRun(Base):
    """Backtest execution results"""
    __tablename__ = 'backtest_runs'

    id = Column(Integer, primary_key=True, autoincrement=True)
    backtest_id = Column(String(100), unique=True, nullable=False)

    # Configuration
    strategy_name = Column(String(100), nullable=False)
    strategy_config = Column(JSON)

    # Time period
    start_date = Column(TIMESTAMP, nullable=False)
    end_date = Column(TIMESTAMP, nullable=False)

    # Data sources
    exchange = Column(String(50))
    symbol = Column(String(20))

    # Results summary
    total_trades = Column(Integer)
    winning_trades = Column(Integer)
    losing_trades = Column(Integer)
    win_rate = Column(DECIMAL(5, 2))

    # Performance metrics
    total_pnl = Column(DECIMAL(18, 8))
    total_pnl_pct = Column(DECIMAL(10, 4))
    avg_pnl_per_trade = Column(DECIMAL(18, 8))
    largest_win = Column(DECIMAL(18, 8))
    largest_loss = Column(DECIMAL(18, 8))

    # Risk metrics
    sharpe_ratio = Column(DECIMAL(10, 4))
    sortino_ratio = Column(DECIMAL(10, 4))
    max_drawdown = Column(DECIMAL(10, 4))
    max_drawdown_duration = Column(String(50))

    # Additional metrics
    profit_factor = Column(DECIMAL(10, 4))
    avg_trade_duration = Column(String(50))
    total_fees = Column(DECIMAL(18, 8))

    # Execution
    run_timestamp = Column(TIMESTAMP, default=datetime.utcnow)
    duration_seconds = Column(Integer)

    # Status
    status = Column(String(20))
    error_message = Column(Text)

    notes = Column(Text)

    __table_args__ = (
        Index('idx_backtest_strategy', 'strategy_name'),
        Index('idx_backtest_dates', 'start_date', 'end_date'),
        Index('idx_backtest_sharpe', 'sharpe_ratio'),
    )

    def __repr__(self):
        return f"<BacktestRun(strategy={self.strategy_name}, sharpe={self.sharpe_ratio})>"


# ============================================================================
# SYSTEM & MONITORING
# ============================================================================

class SystemMetric(Base):
    """System performance metrics"""
    __tablename__ = 'system_metrics'

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(TIMESTAMP, default=datetime.utcnow)

    metric_name = Column(String(100), nullable=False)
    metric_value = Column(DECIMAL(18, 8))
    extra_metadata = Column(JSON)

    category = Column(String(50))

    __table_args__ = (
        Index('idx_system_metrics_timestamp', 'timestamp'),
        Index('idx_system_metrics_name', 'metric_name'),
    )

    def __repr__(self):
        return f"<SystemMetric(name={self.metric_name}, value={self.metric_value})>"


class Alert(Base):
    """Alerts and notifications"""
    __tablename__ = 'alerts'

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(TIMESTAMP, default=datetime.utcnow)

    alert_type = Column(String(50), nullable=False)
    severity = Column(String(20))

    title = Column(String(200))
    message = Column(Text)

    # Associated data
    signal_id = Column(Integer, ForeignKey('signals.id'))
    position_id = Column(Integer, ForeignKey('positions.id'))

    # Notification status
    sent = Column(Boolean, default=False)
    sent_timestamp = Column(TIMESTAMP)
    notification_channels = Column(JSON)  # Store as JSON array for SQLite compatibility

    extra_metadata = Column(JSON)

    __table_args__ = (
        Index('idx_alerts_timestamp', 'timestamp'),
        Index('idx_alerts_type', 'alert_type'),
    )

    def __repr__(self):
        return f"<Alert(type={self.alert_type}, title={self.title})>"


# ============================================================================
# PREDICTION MARKETS
# ============================================================================

class PredictionMarket(Base):
    """Prediction market data (Kalshi, Polymarket, etc.)"""
    __tablename__ = 'prediction_markets'

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(TIMESTAMP, nullable=False, index=True)

    # Source
    source = Column(String(50), nullable=False)  # 'kalshi', 'polymarket', etc.

    # Market identification
    ticker = Column(String(100), nullable=False)
    event_ticker = Column(String(100))
    series_ticker = Column(String(50))

    # Market details
    title = Column(Text)
    subtitle = Column(Text)

    # Timing
    close_time = Column(TIMESTAMP)
    expiration_time = Column(TIMESTAMP, index=True)
    expected_expiration_time = Column(TIMESTAMP)

    # Strike/target price
    strike_price = Column(DECIMAL(18, 8))  # For price threshold markets
    strike_type = Column(String(20))  # 'greater', 'less', 'between'

    # Pricing (in cents, 0-100)
    yes_ask = Column(DECIMAL(10, 4))
    yes_bid = Column(DECIMAL(10, 4))
    no_ask = Column(DECIMAL(10, 4))
    no_bid = Column(DECIMAL(10, 4))
    last_price = Column(DECIMAL(10, 4))

    # Implied probability (yes_ask / 100)
    implied_probability = Column(DECIMAL(5, 2))

    # Volume & liquidity
    volume = Column(Integer)
    volume_24h = Column(Integer)
    open_interest = Column(Integer)
    liquidity = Column(Integer)

    # Market state
    status = Column(String(20))

    # Additional data
    extra_metadata = Column(JSON)

    __table_args__ = (
        Index('idx_pm_timestamp', 'timestamp'),
        Index('idx_pm_expiration', 'expiration_time'),
        Index('idx_pm_ticker', 'ticker'),
        Index('idx_pm_source', 'source'),
    )

    def __repr__(self):
        return f"<PredictionMarket(source={self.source}, ticker={self.ticker}, prob={self.implied_probability}%)>"


# ============================================================================
# TRADE INTENSITY ANALYTICS
# ============================================================================

class TradeIntensity(Base):
    """Trade intensity and flow metrics"""
    __tablename__ = 'trade_intensity'

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(TIMESTAMP, nullable=False, index=True)

    # Exchange
    exchange = Column(String(50), nullable=False)
    symbol = Column(String(20), nullable=False)

    # Trade frequency metrics (rolling windows)
    trades_per_sec_10s = Column(DECIMAL(10, 4))
    trades_per_sec_30s = Column(DECIMAL(10, 4))
    trades_per_sec_60s = Column(DECIMAL(10, 4))

    # Trade size metrics
    avg_trade_size_30s = Column(DECIMAL(18, 8))
    avg_trade_size_60s = Column(DECIMAL(18, 8))

    # Buy/sell pressure
    buy_sell_ratio = Column(DECIMAL(10, 4))  # 0-1 scale (% of volume from buys)
    aggression_score = Column(DECIMAL(10, 4))  # -1 to 1 (sell to buy pressure)

    # Velocity metrics
    velocity_change = Column(DECIMAL(10, 4))  # Rate of change in trading speed

    __table_args__ = (
        Index('idx_trade_intensity_timestamp', 'timestamp'),
        Index('idx_trade_intensity_exchange_symbol', 'exchange', 'symbol'),
    )

    def __repr__(self):
        return f"<TradeIntensity(exchange={self.exchange}, tps={self.trades_per_sec_60s}, aggression={self.aggression_score})>"


# ============================================================================
# FUTURES BASIS TRACKING
# ============================================================================

class FuturesBasis(Base):
    """Futures-spot basis tracking"""
    __tablename__ = 'futures_basis'

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(TIMESTAMP, nullable=False, index=True)

    # Source
    source = Column(String(50), nullable=False)  # 'binance_perp', 'coinbase_futures', etc.
    symbol = Column(String(20), nullable=False)  # 'BTC-USD', 'BTCUSDT', etc.

    # Prices
    spot_price = Column(DECIMAL(18, 8), nullable=False)
    futures_price = Column(DECIMAL(18, 8), nullable=False)

    # Basis metrics
    basis = Column(DECIMAL(18, 8))  # futures_price - spot_price
    basis_pct = Column(DECIMAL(10, 4))  # (basis / spot_price) * 100
    basis_annualized_pct = Column(DECIMAL(10, 4))  # Annualized basis %

    # Additional metadata
    funding_rate = Column(DECIMAL(10, 6))  # For perpetuals
    time_to_expiry_hours = Column(DECIMAL(10, 2))  # For dated futures

    __table_args__ = (
        Index('idx_futures_basis_timestamp', 'timestamp'),
        Index('idx_futures_basis_source', 'source'),
    )

    def __repr__(self):
        return f"<FuturesBasis(source={self.source}, basis={self.basis}, basis_pct={self.basis_pct}%)>"


class FundingRateMetrics(Base):
    """Binance funding rate metrics for BTC perpetual futures"""
    __tablename__ = 'funding_rate_metrics'

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(TIMESTAMP, nullable=False, index=True)

    # Core metrics
    funding_rate = Column(DECIMAL(12, 8), nullable=False)  # Current funding rate
    mark_price = Column(DECIMAL(18, 8), nullable=False)     # Mark price
    index_price = Column(DECIMAL(18, 8), nullable=False)    # Index (spot) price

    # Mark premium metrics
    mark_premium = Column(DECIMAL(10, 6))  # (mark - index) / index * 100
    mark_premium_change_1min = Column(DECIMAL(10, 6))

    # Funding rate changes
    funding_rate_change_30s = Column(DECIMAL(12, 8))
    funding_rate_change_1min = Column(DECIMAL(12, 8))
    funding_rate_change_5min = Column(DECIMAL(12, 8))

    # Velocity and acceleration
    funding_rate_velocity = Column(DECIMAL(12, 8))  # Rate of change (per minute)
    funding_rate_acceleration = Column(DECIMAL(12, 8))  # 2nd derivative

    # Metadata
    next_funding_time = Column(TIMESTAMP)  # When next settlement occurs

    __table_args__ = (
        Index('idx_funding_rate_timestamp', 'timestamp'),
    )

    def __repr__(self):
        return f"<FundingRateMetrics(rate={self.funding_rate}, mark_premium={self.mark_premium}%)>"


class SpotPriceSnapshot(Base):
    """Unified spot price tracking from all sources"""
    __tablename__ = 'spot_price_snapshots'

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(TIMESTAMP, nullable=False, index=True)

    # Source identifier
    source = Column(String(50), nullable=False, index=True)  # 'coinbase', 'kraken', 'binance', 'kalshi_implied'

    # Price data
    price = Column(DECIMAL(18, 8), nullable=False)

    # Volume/liquidity context
    volume_24h = Column(DECIMAL(24, 8))  # 24h volume if available

    # Confidence/quality indicator
    confidence = Column(DECIMAL(5, 2))  # 0-100, how confident we are in this price

    # For Kalshi implied price, track the method
    derivation_method = Column(String(100))  # 'kalshi_probability_transition', 'kalshi_weighted_average', etc.

    # Metadata
    extra_data = Column(JSON)  # Any source-specific metadata

    __table_args__ = (
        Index('idx_spot_price_timestamp_source', 'timestamp', 'source'),
    )

    def __repr__(self):
        return f"<SpotPriceSnapshot(source={self.source}, price=${self.price})>"


class KalshiOrderbookSnapshot(Base):
    """Kalshi market orderbook snapshots"""
    __tablename__ = 'kalshi_orderbook_snapshots'

    # Primary key
    id = Column(Integer, primary_key=True, autoincrement=True)

    # Timestamp
    timestamp = Column(TIMESTAMP, nullable=False, index=True)

    # Market identification
    ticker = Column(String(100), nullable=False, index=True)
    event_ticker = Column(String(100))
    strike_price = Column(DECIMAL(18, 8), index=True)

    # Current BTC spot price at time of snapshot
    current_btc_price = Column(DECIMAL(18, 8), nullable=False)
    btc_price_source = Column(String(50), nullable=False)  # 'coinbase_trade', 'kalshi_implied', etc.

    # Derived metrics
    distance_from_current = Column(DECIMAL(18, 8))  # strike_price - current_btc_price (+ = above, - = below)
    moneyness_pct = Column(DECIMAL(10, 4))  # (strike_price / current_btc_price - 1) * 100
    moneyness_category = Column(String(20))  # 'ITM', 'ATM', 'OTM'

    # Best bid/ask prices (in cents, 0-100)
    best_yes_bid = Column(DECIMAL(10, 4))
    best_yes_ask = Column(DECIMAL(10, 4))
    best_no_bid = Column(DECIMAL(10, 4))
    best_no_ask = Column(DECIMAL(10, 4))

    # Spreads (in cents)
    yes_spread = Column(DECIMAL(10, 4))
    no_spread = Column(DECIMAL(10, 4))

    # Liquidity metrics (total contracts in top N levels)
    yes_liquidity = Column(Integer)
    no_liquidity = Column(Integer)

    # Number of orderbook levels captured
    yes_levels = Column(Integer)
    no_levels = Column(Integer)

    # Full orderbook data as JSON (top 6 levels)
    # Format: [[price, size], [price, size], ...]
    yes_bids = Column(JSON)
    no_bids = Column(JSON)

    # Contract times
    contract_start_time = Column(TIMESTAMP)  # When the contract/market opened
    contract_end_time = Column(TIMESTAMP)  # When the contract expires
    contract_time_window = Column(String(20))  # Time window in "HH:MM to HH:MM" format

    # Volatility metrics
    implied_volatility = Column(DECIMAL(10, 4))  # Implied vol estimate from market prices (annualized %)
    realized_vol_30min = Column(DECIMAL(10, 4))  # Realized volatility last 30 minutes (annualized %)
    realized_vol_10min = Column(DECIMAL(10, 4))  # Realized volatility last 10 minutes (annualized %)
    realized_vol_yesterday_same_hour = Column(DECIMAL(10, 4))  # RV from same hour yesterday (annualized %)
    realized_vol_last_week_same_hour = Column(DECIMAL(10, 4))  # RV from same hour last week (annualized %)
    implied_volatility_rank = Column(DECIMAL(5, 2))  # IV Rank (0-100, where current IV stands in its range)

    # Option Greeks (for binary options)
    delta = Column(DECIMAL(10, 6))  # Rate of change of option value w.r.t. underlying price
    gamma = Column(DECIMAL(10, 6))  # Rate of change of delta w.r.t. underlying price
    vega = Column(DECIMAL(10, 6))  # Rate of change of option value w.r.t. volatility
    theta = Column(DECIMAL(10, 6))  # Rate of change of option value w.r.t. time
    vanna = Column(DECIMAL(10, 6))  # Cross-derivative: ∂Delta/∂σ = ∂²V/∂S∂σ
    volga = Column(DECIMAL(10, 6))  # Second derivative w.r.t. volatility: ∂²V/∂σ² (also called vomma)

    __table_args__ = (
        Index('idx_kalshi_ob_timestamp', 'timestamp'),
        Index('idx_kalshi_ob_ticker', 'ticker'),
        Index('idx_kalshi_ob_strike', 'strike_price'),
        Index('idx_kalshi_ob_timestamp_strike', 'timestamp', 'strike_price'),
        Index('idx_kalshi_ob_moneyness', 'moneyness_category'),
    )

    def __repr__(self):
        return f"<KalshiOrderbookSnapshot(ticker={self.ticker}, strike=${self.strike_price}, moneyness={self.moneyness_category})>"


class BtcPriceCandle(Base):
    """OHLC candlestick data for BTC price across various timeframes"""
    __tablename__ = 'btc_price_candles'

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(TIMESTAMP, nullable=False, index=True)  # Start of the candle period

    # Timeframe
    timeframe = Column(String(10), nullable=False, index=True)  # '1min', '5min', '15min', '30min', '1hour', '1day', '1week'

    # Exchange/Source
    exchange = Column(String(50), nullable=False, index=True)  # 'coinbase', 'binance', 'kalshi', etc.
    symbol = Column(String(20), nullable=False)  # 'BTC-USD', 'BTCUSDT', etc.

    # OHLC data
    open = Column(DECIMAL(18, 8), nullable=False)
    high = Column(DECIMAL(18, 8), nullable=False)
    low = Column(DECIMAL(18, 8), nullable=False)
    close = Column(DECIMAL(18, 8), nullable=False)

    # Volume
    volume = Column(DECIMAL(18, 8))  # Total volume in the period

    # Additional metrics
    num_trades = Column(Integer)  # Number of trades in this candle

    __table_args__ = (
        Index('idx_btc_candle_timestamp_timeframe', 'timestamp', 'timeframe'),
        Index('idx_btc_candle_exchange_timeframe', 'exchange', 'timeframe'),
        Index('idx_btc_candle_full', 'exchange', 'timeframe', 'timestamp'),
        # Unique constraint to prevent duplicate candles
        UniqueConstraint('timestamp', 'timeframe', 'exchange', 'symbol', name='uq_btc_candle'),
    )

    def __repr__(self):
        return f"<BtcPriceCandle(exchange={self.exchange}, timeframe={self.timeframe}, timestamp={self.timestamp}, close=${self.close})>"
