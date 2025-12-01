#!/usr/bin/env python3
"""
Compare BTC Spot Prices Across All Sources

Shows price deltas between:
- Coinbase (exchange spot)
- Kraken (exchange spot)
- Binance Index (spot aggregate from Binance)
- Kalshi Implied (derived from prediction markets)
"""

import sys
import os
from datetime import datetime, timedelta

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.storage.database import db_manager
from data.storage.models import SpotPriceSnapshot
from sqlalchemy import desc, func


def compare_latest_prices():
    """Compare most recent prices from all sources"""
    print("\n" + "="*100)
    print("LATEST BTC SPOT PRICES - Cross-Source Comparison")
    print("="*100 + "\n")

    with db_manager.get_session() as session:
        # Get latest price from each source
        sources = ['coinbase', 'kraken', 'binance_index', 'kalshi_implied']
        prices = {}

        for source in sources:
            latest = session.query(SpotPriceSnapshot).filter(
                SpotPriceSnapshot.source == source
            ).order_by(desc(SpotPriceSnapshot.timestamp)).first()

            if latest:
                age = (datetime.utcnow() - latest.timestamp).total_seconds()
                prices[source] = {
                    'price': float(latest.price),
                    'confidence': float(latest.confidence) if latest.confidence else 0,
                    'timestamp': latest.timestamp,
                    'age_seconds': age
                }

        if not prices:
            print("No spot price data found")
            return

        # Display each source
        print(f"{'Source':<20} {'Price':>12} {'Age':>10} {'Confidence':>12} {'Method':>25}")
        print("-" * 100)

        for source in sources:
            if source in prices:
                data = prices[source]
                age_str = f"{data['age_seconds']:.0f}s" if data['age_seconds'] < 3600 else f"{data['age_seconds']/3600:.1f}h"
                conf_str = f"{data['confidence']:.0f}%" if data['confidence'] > 0 else "N/A"

                # Get method if available
                method = ""
                if source == 'kalshi_implied':
                    latest_full = session.query(SpotPriceSnapshot).filter(
                        SpotPriceSnapshot.source == source
                    ).order_by(desc(SpotPriceSnapshot.timestamp)).first()
                    if latest_full and latest_full.derivation_method:
                        method = latest_full.derivation_method

                print(f"{source:<20} ${data['price']:>11,.2f} {age_str:>10} {conf_str:>12} {method:>25}")
            else:
                print(f"{source:<20} {'N/A':>12} {'N/A':>10} {'N/A':>12} {'':>25}")

        # Calculate spreads and deltas
        if len(prices) > 1:
            print("\n" + "="*100)
            print("PRICE DELTAS & ARBITRAGE OPPORTUNITIES")
            print("="*100 + "\n")

            # Reference price (use Coinbase as baseline)
            if 'coinbase' in prices:
                ref_source = 'coinbase'
                ref_price = prices['coinbase']['price']
            else:
                ref_source = list(prices.keys())[0]
                ref_price = prices[ref_source]['price']

            print(f"Reference: {ref_source} @ ${ref_price:,.2f}\n")
            print(f"{'vs Source':<20} {'Delta $':>12} {'Delta %':>12} {'Arbitrage':>15}")
            print("-" * 100)

            for source, data in prices.items():
                if source != ref_source:
                    delta = data['price'] - ref_price
                    delta_pct = (delta / ref_price) * 100

                    # Determine arbitrage direction
                    if abs(delta) < 1:
                        arb = "None"
                    elif delta > 0:
                        arb = f"Buy {ref_source}"
                    else:
                        arb = f"Sell {ref_source}"

                    print(f"{source:<20} ${delta:>11,.2f} {delta_pct:>11.3f}% {arb:>15}")

            # Overall spread
            price_values = [data['price'] for data in prices.values()]
            spread = max(price_values) - min(price_values)
            spread_pct = (spread / min(price_values)) * 100

            print(f"\n{'Overall Spread':<20} ${spread:>11,.2f} {spread_pct:>11.3f}%")

        print("\n" + "="*100 + "\n")


def compare_time_series():
    """Compare prices over last hour"""
    print("\n" + "="*100)
    print("PRICE COMPARISON - Last Hour")
    print("="*100 + "\n")

    with db_manager.get_session() as session:
        one_hour_ago = datetime.utcnow() - timedelta(hours=1)

        sources = ['coinbase', 'kraken', 'binance_index', 'kalshi_implied']

        for source in sources:
            recent = session.query(SpotPriceSnapshot).filter(
                SpotPriceSnapshot.source == source,
                SpotPriceSnapshot.timestamp >= one_hour_ago
            ).order_by(SpotPriceSnapshot.timestamp).all()

            if recent:
                prices = [float(r.price) for r in recent]
                avg_price = sum(prices) / len(prices)
                min_price = min(prices)
                max_price = max(prices)
                volatility = max_price - min_price

                print(f"{source:>20}: {len(recent):>3} readings | Avg: ${avg_price:>10,.2f} | Range: ${min_price:>10,.2f} - ${max_price:>10,.2f} | Vol: ${volatility:>8,.2f}")

    print("\n" + "="*100 + "\n")


def check_data_quality():
    """Check data quality and freshness"""
    print("\n" + "="*100)
    print("DATA QUALITY CHECK")
    print("="*100 + "\n")

    with db_manager.get_session() as session:
        sources = ['coinbase', 'kraken', 'binance_index', 'kalshi_implied']

        print(f"{'Source':<20} {'Total Records':>15} {'Latest':>25} {'Status':>15}")
        print("-" * 100)

        for source in sources:
            count = session.query(func.count(SpotPriceSnapshot.id)).filter(
                SpotPriceSnapshot.source == source
            ).scalar()

            latest = session.query(SpotPriceSnapshot).filter(
                SpotPriceSnapshot.source == source
            ).order_by(desc(SpotPriceSnapshot.timestamp)).first()

            if latest:
                age = (datetime.utcnow() - latest.timestamp).total_seconds()

                if age < 120:
                    status = "✅ Live"
                elif age < 3600:
                    status = "⚠️  Delayed"
                else:
                    status = "❌ Stale"

                latest_str = str(latest.timestamp)[:19]
            else:
                latest_str = "N/A"
                status = "❌ No Data"

            print(f"{source:<20} {count:>15} {latest_str:>25} {status:>15}")

    print("\n" + "="*100 + "\n")


def main():
    """Run comparison queries"""
    import argparse

    parser = argparse.ArgumentParser(description='Compare BTC spot prices across sources')
    parser.add_argument('query', nargs='?', default='latest',
                       choices=['latest', 'timeseries', 'quality', 'all'],
                       help='Query type')

    args = parser.parse_args()

    if args.query == 'latest':
        compare_latest_prices()
    elif args.query == 'timeseries':
        compare_time_series()
    elif args.query == 'quality':
        check_data_quality()
    elif args.query == 'all':
        compare_latest_prices()
        compare_time_series()
        check_data_quality()


if __name__ == "__main__":
    main()
