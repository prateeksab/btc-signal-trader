#!/usr/bin/env python3
"""
Database Query Examples

Comprehensive examples of how to query the trading signals database.
"""

import sys
import os
from datetime import datetime, timedelta
from decimal import Decimal

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from data.storage.database import db_manager
from data.storage.models import (
    Trade, OrderbookSnapshot, CVDSnapshot, Signal, Position
)
from sqlalchemy import func, and_, or_, desc


def example_1_basic_queries():
    """Example 1: Basic queries - Get all records"""
    print("\n" + "="*80)
    print("EXAMPLE 1: Basic Queries")
    print("="*80)

    with db_manager.get_session() as session:
        # Get all trades
        all_trades = session.query(Trade).all()
        print(f"\nTotal trades: {len(all_trades)}")

        # Get all signals
        all_signals = session.query(Signal).all()
        print(f"Total signals: {len(all_signals)}")


def example_2_filtering():
    """Example 2: Filtering with WHERE clauses"""
    print("\n" + "="*80)
    print("EXAMPLE 2: Filtering Data")
    print("="*80)

    with db_manager.get_session() as session:
        # Get only buy trades
        buy_trades = session.query(Trade).filter(Trade.side == 'buy').all()
        print(f"\nBuy trades: {len(buy_trades)}")

        # Get only STRONG_BUY or BUY signals
        buy_signals = session.query(Signal).filter(
            Signal.signal_type.in_(['STRONG_BUY', 'BUY'])
        ).all()
        print(f"Buy signals: {len(buy_signals)}")

        # Get high confidence signals (>70%)
        high_conf_signals = session.query(Signal).filter(
            Signal.confidence_score > 70
        ).all()
        print(f"High confidence signals (>70%): {len(high_conf_signals)}")


def example_3_ordering():
    """Example 3: Ordering results"""
    print("\n" + "="*80)
    print("EXAMPLE 3: Ordering Results")
    print("="*80)

    with db_manager.get_session() as session:
        # Get latest 5 trades
        latest_trades = session.query(Trade).order_by(
            Trade.timestamp.desc()
        ).limit(5).all()

        print("\nLatest 5 trades:")
        for t in latest_trades:
            print(f"  {t.timestamp} | ${t.price} | {t.side} | {t.size} BTC")

        # Get highest confidence signals
        top_signals = session.query(Signal).order_by(
            Signal.confidence_score.desc()
        ).limit(5).all()

        print(f"\nTop 5 signals by confidence:")
        for s in top_signals:
            print(f"  {s.signal_type} | {s.confidence_score}% | {s.timestamp}")


def example_4_aggregations():
    """Example 4: Aggregate functions (COUNT, SUM, AVG, etc.)"""
    print("\n" + "="*80)
    print("EXAMPLE 4: Aggregations")
    print("="*80)

    with db_manager.get_session() as session:
        # Count trades by side
        buy_count = session.query(func.count(Trade.id)).filter(
            Trade.side == 'buy'
        ).scalar()
        sell_count = session.query(func.count(Trade.id)).filter(
            Trade.side == 'sell'
        ).scalar()

        print(f"\nTrade counts:")
        print(f"  Buys:  {buy_count}")
        print(f"  Sells: {sell_count}")

        # Sum volume by side
        buy_volume = session.query(func.sum(Trade.size)).filter(
            Trade.side == 'buy'
        ).scalar() or 0
        sell_volume = session.query(func.sum(Trade.size)).filter(
            Trade.side == 'sell'
        ).scalar() or 0

        print(f"\nTotal volume:")
        print(f"  Buy volume:  {buy_volume:.4f} BTC")
        print(f"  Sell volume: {sell_volume:.4f} BTC")
        print(f"  Net (CVD):   {buy_volume - sell_volume:+.4f} BTC")

        # Average price
        avg_price = session.query(func.avg(Trade.price)).scalar()
        print(f"\nAverage trade price: ${avg_price:,.2f}")

        # Min/Max prices
        min_price = session.query(func.min(Trade.price)).scalar()
        max_price = session.query(func.max(Trade.price)).scalar()
        print(f"Price range: ${min_price:,.2f} - ${max_price:,.2f}")


def example_5_groupby():
    """Example 5: GROUP BY queries"""
    print("\n" + "="*80)
    print("EXAMPLE 5: Group By")
    print("="*80)

    with db_manager.get_session() as session:
        # Count signals by type
        signal_counts = session.query(
            Signal.signal_type,
            func.count(Signal.id)
        ).group_by(Signal.signal_type).all()

        print("\nSignals by type:")
        for signal_type, count in signal_counts:
            print(f"  {signal_type}: {count}")

        # Count signals by confidence level
        conf_counts = session.query(
            Signal.confidence_level,
            func.count(Signal.id)
        ).group_by(Signal.confidence_level).all()

        print("\nSignals by confidence level:")
        for conf_level, count in conf_counts:
            print(f"  {conf_level}: {count}")


def example_6_time_ranges():
    """Example 6: Time-based queries"""
    print("\n" + "="*80)
    print("EXAMPLE 6: Time-based Queries")
    print("="*80)

    with db_manager.get_session() as session:
        # Get trades from last 5 minutes
        five_min_ago = datetime.utcnow() - timedelta(minutes=5)
        recent_trades = session.query(Trade).filter(
            Trade.timestamp >= five_min_ago
        ).all()

        print(f"\nTrades in last 5 minutes: {len(recent_trades)}")

        # Get signals from last hour
        one_hour_ago = datetime.utcnow() - timedelta(hours=1)
        recent_signals = session.query(Signal).filter(
            Signal.timestamp >= one_hour_ago
        ).all()

        print(f"Signals in last hour: {len(recent_signals)}")

        # Get trades in a specific time range
        start_time = datetime.utcnow() - timedelta(minutes=10)
        end_time = datetime.utcnow() - timedelta(minutes=5)

        range_trades = session.query(Trade).filter(
            and_(
                Trade.timestamp >= start_time,
                Trade.timestamp <= end_time
            )
        ).all()

        print(f"Trades between 10-5 minutes ago: {len(range_trades)}")


def example_7_complex_queries():
    """Example 7: Complex multi-condition queries"""
    print("\n" + "="*80)
    print("EXAMPLE 7: Complex Queries")
    print("="*80)

    with db_manager.get_session() as session:
        # Find strong buy signals with high orderbook imbalance
        strong_buys = session.query(Signal).filter(
            and_(
                Signal.signal_type.in_(['STRONG_BUY', 'BUY']),
                Signal.orderbook_imbalance > 50
            )
        ).all()

        print(f"\nBuy signals with >50% orderbook imbalance: {len(strong_buys)}")

        # Find large trades (>0.1 BTC)
        large_trades = session.query(Trade).filter(
            Trade.size > 0.1
        ).all()

        print(f"Large trades (>0.1 BTC): {len(large_trades)}")

        # Find signals with reasons
        signals_with_reasons = session.query(Signal).filter(
            and_(
                Signal.reasons != None,
                Signal.reasons != '[]'
            )
        ).all()

        print(f"Signals with reasons: {len(signals_with_reasons)}")


def example_8_joins():
    """Example 8: Joins between tables"""
    print("\n" + "="*80)
    print("EXAMPLE 8: Joins (Relationships)")
    print("="*80)

    with db_manager.get_session() as session:
        # Get signals with their associated positions (if any)
        signals_with_positions = session.query(Signal).join(
            Position, Position.signal_id == Signal.id, isouter=True
        ).all()

        print(f"\nSignals queried: {len(signals_with_positions)}")

        # Note: Currently no positions exist, but this shows how to join
        # When positions exist, you can access them via signal.positions


def example_9_custom_analytics():
    """Example 9: Custom analytics queries"""
    print("\n" + "="*80)
    print("EXAMPLE 9: Custom Analytics")
    print("="*80)

    with db_manager.get_session() as session:
        # CVD progression
        first_trade = session.query(Trade).order_by(Trade.timestamp).first()
        last_trade = session.query(Trade).order_by(Trade.timestamp.desc()).first()

        if first_trade and last_trade:
            cvd_change = float(last_trade.cvd_at_trade - first_trade.cvd_at_trade)
            time_diff = (last_trade.timestamp - first_trade.timestamp).total_seconds()

            print(f"\nCVD Analysis:")
            print(f"  Start CVD: {first_trade.cvd_at_trade:+.4f} BTC")
            print(f"  End CVD:   {last_trade.cvd_at_trade:+.4f} BTC")
            print(f"  Change:    {cvd_change:+.4f} BTC")
            print(f"  Time span: {time_diff:.0f} seconds")
            print(f"  CVD/second: {cvd_change/time_diff:+.6f} BTC/s")

        # Signal accuracy (would need position results to calculate)
        total_signals = session.query(func.count(Signal.id)).scalar()
        buy_signals = session.query(func.count(Signal.id)).filter(
            Signal.signal_type.in_(['STRONG_BUY', 'BUY'])
        ).scalar()

        print(f"\nSignal Distribution:")
        print(f"  Total signals: {total_signals}")
        print(f"  Buy signals:   {buy_signals}")
        print(f"  Buy %:         {(buy_signals/total_signals*100):.1f}%")


def example_10_raw_sql():
    """Example 10: Raw SQL queries (advanced)"""
    print("\n" + "="*80)
    print("EXAMPLE 10: Raw SQL Queries")
    print("="*80)

    with db_manager.get_session() as session:
        from sqlalchemy import text

        # Execute raw SQL
        result = session.execute(text("""
            SELECT
                signal_type,
                COUNT(*) as count,
                AVG(confidence_score) as avg_confidence,
                AVG(orderbook_imbalance) as avg_imbalance
            FROM signals
            GROUP BY signal_type
        """))

        print("\nSignal Statistics (Raw SQL):")
        for row in result:
            print(f"  {row.signal_type:12s}: {row.count:3d} signals | "
                  f"Avg Conf: {row.avg_confidence:5.1f}% | "
                  f"Avg Imbalance: {row.avg_imbalance:+6.1f}%")


def main():
    """Run all examples"""
    print("\n" + "="*80)
    print("DATABASE QUERY EXAMPLES")
    print("="*80)

    example_1_basic_queries()
    example_2_filtering()
    example_3_ordering()
    example_4_aggregations()
    example_5_groupby()
    example_6_time_ranges()
    example_7_complex_queries()
    example_8_joins()
    example_9_custom_analytics()
    example_10_raw_sql()

    print("\n" + "="*80)
    print("All examples completed!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
