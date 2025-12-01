#!/usr/bin/env python3
"""
Database Test Script

Tests database operations by inserting and querying sample data.
"""

import sys
import os
from datetime import datetime
from decimal import Decimal

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.storage.database import db_manager
from data.storage.models import (
    Trade, OrderbookSnapshot, CVDSnapshot,
    Signal, Position, Candle
)

def test_database():
    """Test database operations"""
    print("=" * 80)
    print("Database Test - Insert and Query Operations")
    print("=" * 80)

    try:
        # Test 1: Insert a Trade
        print("\n1. Testing Trade insertion...")
        with db_manager.get_session() as session:
            trade = Trade(
                timestamp=datetime.utcnow(),
                exchange='coinbase',
                symbol='BTC-USD',
                trade_id='12345',
                price=Decimal('95234.50'),
                size=Decimal('0.15'),
                side='buy',
                is_buyer_maker=False,
                cvd_at_trade=Decimal('125.45')
            )
            session.add(trade)
            session.commit()
            trade_id = trade.id
        print(f"   ✅ Trade inserted with ID: {trade_id}")

        # Test 2: Insert an Orderbook Snapshot
        print("\n2. Testing Orderbook Snapshot insertion...")
        with db_manager.get_session() as session:
            orderbook = OrderbookSnapshot(
                timestamp=datetime.utcnow(),
                exchange='kraken',
                symbol='XBT/USD',
                best_bid=Decimal('95230.00'),
                best_ask=Decimal('95235.00'),
                spread=Decimal('5.00'),
                spread_pct=Decimal('0.0052'),
                total_bid_volume=Decimal('45.2'),
                total_ask_volume=Decimal('32.8'),
                imbalance=Decimal('15.9'),
                bids_json=[
                    [95230.00, 10.5],
                    [95225.00, 8.3]
                ],
                asks_json=[
                    [95235.00, 7.2],
                    [95240.00, 9.1]
                ],
                depth_levels=10
            )
            session.add(orderbook)
            session.commit()
            orderbook_id = orderbook.id
        print(f"   ✅ Orderbook Snapshot inserted with ID: {orderbook_id}")

        # Test 3: Insert a CVD Snapshot
        print("\n3. Testing CVD Snapshot insertion...")
        with db_manager.get_session() as session:
            cvd = CVDSnapshot(
                timestamp=datetime.utcnow(),
                exchange='coinbase',
                symbol='BTC-USD',
                cvd=Decimal('125.45'),
                cvd_change=Decimal('2.35'),
                cvd_ma_5=Decimal('122.10'),
                cvd_ma_20=Decimal('118.75')
            )
            session.add(cvd)
            session.commit()
            cvd_id = cvd.id
        print(f"   ✅ CVD Snapshot inserted with ID: {cvd_id}")

        # Test 4: Insert a Signal
        print("\n4. Testing Signal insertion...")
        with db_manager.get_session() as session:
            signal = Signal(
                timestamp=datetime.utcnow(),
                signal_type='STRONG_BUY',
                confidence_level='HIGH',
                confidence_score=Decimal('85.5'),
                price=Decimal('95234.50'),
                cvd=Decimal('125.45'),
                cvd_trend='BULLISH',
                orderbook_imbalance=Decimal('15.9'),
                reasons=[
                    'Strong CVD uptrend',
                    'High orderbook buy pressure',
                    'Price breaking resistance'
                ],
                warnings=[
                    'High volatility period'
                ],
                exchange='coinbase',
                symbol='BTC-USD',
                strategy_name='CVD_Imbalance_Strategy',
                extra_metadata={
                    'cvd_ma_cross': True,
                    'volume_surge': True
                }
            )
            session.add(signal)
            session.commit()
            signal_id = signal.id
        print(f"   ✅ Signal inserted with ID: {signal_id}")

        # Test 5: Insert a Position linked to the Signal
        print("\n5. Testing Position insertion...")
        with db_manager.get_session() as session:
            position = Position(
                position_id='POS-001',
                signal_id=signal_id,
                exchange='coinbase',
                symbol='BTC-USD',
                side='long',
                entry_timestamp=datetime.utcnow(),
                entry_price=Decimal('95234.50'),
                entry_signal_type='STRONG_BUY',
                quantity=Decimal('0.1'),
                notional_value=Decimal('9523.45'),
                stop_loss=Decimal('94000.00'),
                take_profit=Decimal('98000.00'),
                status='open',
                is_backtest=False
            )
            session.add(position)
            session.commit()
            position_id = position.id
        print(f"   ✅ Position inserted with ID: {position_id}")

        # Test 6: Insert a Candle
        print("\n6. Testing Candle insertion...")
        with db_manager.get_session() as session:
            candle = Candle(
                timestamp=datetime.utcnow(),
                exchange='coinbase',
                symbol='BTC-USD',
                interval='1m',
                open=Decimal('95200.00'),
                high=Decimal('95350.00'),
                low=Decimal('95180.00'),
                close=Decimal('95234.50'),
                volume=Decimal('12.5'),
                trade_count=145,
                vwap=Decimal('95245.25')
            )
            session.add(candle)
            session.commit()
            candle_id = candle.id
        print(f"   ✅ Candle inserted with ID: {candle_id}")

        # Test 7: Query data back
        print("\n7. Querying data back...")

        with db_manager.get_session() as session:
            # Query trades
            trades = session.query(Trade).all()
            print(f"   📊 Found {len(trades)} trades")

            # Query orderbook snapshots
            orderbooks = session.query(OrderbookSnapshot).all()
            print(f"   📊 Found {len(orderbooks)} orderbook snapshots")

            # Query CVD snapshots
            cvds = session.query(CVDSnapshot).all()
            print(f"   📊 Found {len(cvds)} CVD snapshots")

            # Query signals
            signals = session.query(Signal).all()
            print(f"   📊 Found {len(signals)} signals")
            for sig in signals:
                print(f"      - {sig.signal_type} at ${sig.price} (confidence: {sig.confidence_score})")
                print(f"        Reasons: {sig.reasons}")
                print(f"        Warnings: {sig.warnings}")

            # Query positions with signal relationship
            positions = session.query(Position).all()
            print(f"   📊 Found {len(positions)} positions")
            for pos in positions:
                print(f"      - {pos.side} position at ${pos.entry_price} ({pos.status})")
                if pos.signal:
                    print(f"        Linked to signal: {pos.signal.signal_type}")

            # Query candles
            candles = session.query(Candle).all()
            print(f"   📊 Found {len(candles)} candles")
            for c in candles:
                print(f"      - {c.interval} candle: O={c.open} H={c.high} L={c.low} C={c.close}")

        print("\n" + "=" * 80)
        print("✅ All database tests passed successfully!")
        print("=" * 80)

        return True

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_database()
    sys.exit(0 if success else 1)
