#!/usr/bin/env python3
"""
Query Kalshi Prediction Markets

Examples of how to query the prediction markets data.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.storage.database import db_manager
from data.storage.models import PredictionMarket
from sqlalchemy import func, desc

def example_1_latest_snapshot():
    """Get the most recent snapshot for each market"""
    print("\n" + "="*80)
    print("EXAMPLE 1: Latest Market Snapshot")
    print("="*80)

    with db_manager.get_session() as session:
        # Get latest snapshot for the next expiring event
        latest = session.query(PredictionMarket).order_by(
            desc(PredictionMarket.timestamp)
        ).first()

        if latest:
            print(f"\nLatest snapshot time: {latest.timestamp}")
            print(f"Event: {latest.event_ticker}")
            print(f"Expires: {latest.expected_expiration_time}")

            # Get all markets from that snapshot
            markets = session.query(PredictionMarket).filter(
                PredictionMarket.event_ticker == latest.event_ticker,
                PredictionMarket.timestamp == latest.timestamp
            ).order_by(desc(PredictionMarket.yes_ask)).limit(10).all()

            print(f"\nTop 10 markets by probability:")
            print(f"{'Strike':>10s} | {'Prob':>4s} | {'Volume':>8s} | {'Open Int':>8s}")
            print("-" * 50)

            for m in markets:
                strike = f"${m.strike_price:,.0f}" if m.strike_price else "N/A"
                prob = f"{int(m.yes_ask)}%" if m.yes_ask else "N/A"
                vol = str(m.volume) if m.volume else "0"
                oi = str(m.open_interest) if m.open_interest else "0"
                print(f"{strike:>10s} | {prob:>4s} | {vol:>8s} | {oi:>8s}")


def example_2_price_range_estimate():
    """Estimate current BTC price range from market probabilities"""
    print("\n" + "="*80)
    print("EXAMPLE 2: BTC Price Range Estimate")
    print("="*80)

    with db_manager.get_session() as session:
        # Get latest timestamp
        latest_time = session.query(func.max(PredictionMarket.timestamp)).scalar()

        if latest_time:
            # Get markets with moderate probabilities (likely near current price)
            markets = session.query(PredictionMarket).filter(
                PredictionMarket.timestamp == latest_time,
                PredictionMarket.yes_ask.between(10, 90)
            ).order_by(desc(PredictionMarket.yes_ask)).all()

            print(f"\nSnapshot time: {latest_time}")
            print(f"\nMarkets with 10-90% probability (near current BTC price):")
            print(f"{'Strike':>10s} | {'Subtitle':35s} | {'Prob':>5s}")
            print("-" * 60)

            for m in markets[:10]:
                strike = f"${m.strike_price:,.0f}" if m.strike_price else "N/A"
                subtitle = (m.subtitle[:32] + '...') if m.subtitle and len(m.subtitle) > 35 else (m.subtitle or '')
                prob = f"{int(m.yes_ask)}%" if m.yes_ask else "N/A"
                print(f"{strike:>10s} | {subtitle:35s} | {prob:>5s}")

            # Estimate price range
            if len(markets) >= 2:
                high_prob = markets[0]  # Highest probability in range
                low_prob = markets[-1]   # Lowest probability in range
                print(f"\nEstimated BTC price range: ${low_prob.strike_price:,.0f} - ${high_prob.strike_price:,.0f}")


def example_3_time_series():
    """Show how probabilities changed over time for a specific strike"""
    print("\n" + "="*80)
    print("EXAMPLE 3: Probability Time Series")
    print("="*80)

    with db_manager.get_session() as session:
        # Pick a specific strike price to track
        target_strike = 90000

        markets = session.query(PredictionMarket).filter(
            PredictionMarket.strike_price == target_strike
        ).order_by(PredictionMarket.timestamp).all()

        if markets:
            print(f"\nProbability evolution for ${target_strike:,} strike:")
            print(f"{'Timestamp':20s} | {'Event':20s} | {'Prob':>5s} | {'Volume':>8s}")
            print("-" * 70)

            for m in markets:
                ts = str(m.timestamp)[:19]  # Trim microseconds
                prob = f"{int(m.yes_ask)}%" if m.yes_ask else "N/A"
                vol = str(m.volume) if m.volume else "0"
                print(f"{ts:20s} | {m.event_ticker:20s} | {prob:>5s} | {vol:>8s}")


def example_4_volume_analysis():
    """Analyze trading volume across markets"""
    print("\n" + "="*80)
    print("EXAMPLE 4: Volume Analysis")
    print("="*80)

    with db_manager.get_session() as session:
        # Get latest timestamp
        latest_time = session.query(func.max(PredictionMarket.timestamp)).scalar()

        # Get markets with highest volume
        markets = session.query(PredictionMarket).filter(
            PredictionMarket.timestamp == latest_time
        ).order_by(desc(PredictionMarket.volume)).limit(10).all()

        print(f"\nTop 10 markets by volume (latest snapshot):")
        print(f"{'Strike':>10s} | {'Prob':>4s} | {'Volume':>10s} | {'Open Int':>10s}")
        print("-" * 50)

        total_volume = 0
        for m in markets:
            strike = f"${m.strike_price:,.0f}" if m.strike_price else "N/A"
            prob = f"{int(m.yes_ask)}%" if m.yes_ask else "N/A"
            vol = m.volume if m.volume else 0
            oi = m.open_interest if m.open_interest else 0
            total_volume += vol
            print(f"{strike:>10s} | {prob:>4s} | {vol:>10,d} | {oi:>10,d}")

        print(f"\nTotal volume (top 10): {total_volume:,}")


def example_5_raw_sql():
    """Use raw SQL for complex queries"""
    print("\n" + "="*80)
    print("EXAMPLE 5: Raw SQL Query")
    print("="*80)

    with db_manager.get_session() as session:
        from sqlalchemy import text

        # Find the price range with 50% probability
        result = session.execute(text("""
            SELECT
                strike_price,
                yes_ask as probability,
                volume,
                open_interest,
                timestamp
            FROM prediction_markets
            WHERE timestamp = (SELECT MAX(timestamp) FROM prediction_markets)
                AND yes_ask BETWEEN 40 AND 60
            ORDER BY yes_ask DESC
        """))

        print(f"\nMarkets with 40-60% probability (50/50 zone):")
        print(f"{'Strike':>10s} | {'Prob':>5s} | {'Volume':>8s} | {'OI':>8s}")
        print("-" * 50)

        for row in result:
            strike = f"${row.strike_price:,.0f}" if row.strike_price else "N/A"
            prob = f"{int(row.probability)}%" if row.probability else "N/A"
            vol = str(row.volume) if row.volume else "0"
            oi = str(row.open_interest) if row.open_interest else "0"
            print(f"{strike:>10s} | {prob:>5s} | {vol:>8s} | {oi:>8s}")


def main():
    """Run all examples"""
    print("\n" + "="*80)
    print("KALSHI PREDICTION MARKETS - QUERY EXAMPLES")
    print("="*80)

    example_1_latest_snapshot()
    example_2_price_range_estimate()
    example_3_time_series()
    example_4_volume_analysis()
    example_5_raw_sql()

    print("\n" + "="*80)
    print("All examples completed!")
    print("="*80 + "\n")


if __name__ == "__main__":
    import sys

    # Map of example names to functions
    examples = {
        '1': example_1_latest_snapshot,
        '2': example_2_price_range_estimate,
        '3': example_3_time_series,
        '4': example_4_volume_analysis,
        '5': example_5_raw_sql,
        'all': main
    }

    # Check if a specific example was requested
    if len(sys.argv) > 1:
        example_num = sys.argv[1]
        if example_num in examples:
            examples[example_num]()
        else:
            print(f"Unknown example: {example_num}")
            print(f"Available examples: {', '.join(examples.keys())}")
    else:
        # Run all examples if no argument provided
        main()
