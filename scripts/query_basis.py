#!/usr/bin/env python3
"""
Query Futures Basis Data

Examples for querying and analyzing futures-spot basis data.
"""

import sys
import os
from datetime import datetime, timedelta

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.storage.database import db_manager
from data.storage.models import FuturesBasis
from sqlalchemy import func, desc


def example_1_latest_basis():
    """Get the latest futures basis reading"""
    print("\n" + "="*80)
    print("EXAMPLE 1: Latest Futures Basis")
    print("="*80)

    with db_manager.get_session() as session:
        latest = session.query(FuturesBasis).order_by(
            desc(FuturesBasis.timestamp)
        ).first()

        if latest:
            print(f"\nTimestamp: {latest.timestamp}")
            print(f"Source: {latest.source}")
            print(f"Symbol: {latest.symbol}")
            print(f"\nSpot Price: ${float(latest.spot_price):,.2f}")
            print(f"Futures Price: ${float(latest.futures_price):,.2f}")
            print(f"\nBasis: ${float(latest.basis):+,.2f}")
            print(f"Basis %: {float(latest.basis_pct):+.4f}%")

            if float(latest.basis) > 0:
                print(f"\n💡 Futures trading at PREMIUM to spot (bullish signal)")
            else:
                print(f"\n💡 Futures trading at DISCOUNT to spot (bearish signal)")
        else:
            print("\nNo basis data found")


def example_2_basis_time_series():
    """Get basis over the last hour"""
    print("\n" + "="*80)
    print("EXAMPLE 2: Basis Time Series (Last Hour)")
    print("="*80)

    with db_manager.get_session() as session:
        one_hour_ago = datetime.utcnow() - timedelta(hours=1)

        basis_records = session.query(FuturesBasis).filter(
            FuturesBasis.timestamp >= one_hour_ago
        ).order_by(FuturesBasis.timestamp).all()

        if basis_records:
            print(f"\nFound {len(basis_records)} basis readings in the last hour\n")
            print(f"{'Timestamp':<20} {'Basis':>12} {'Basis %':>12}")
            print("-" * 80)

            for record in basis_records[-10:]:  # Show last 10
                print(f"{str(record.timestamp):<20} "
                      f"${float(record.basis):>+11,.2f} "
                      f"{float(record.basis_pct):>+11.4f}%")
        else:
            print("\nNo basis data found in last hour")


def example_3_basis_statistics():
    """Calculate basis statistics"""
    print("\n" + "="*80)
    print("EXAMPLE 3: Basis Statistics")
    print("="*80)

    with db_manager.get_session() as session:
        # Get all basis records
        records = session.query(FuturesBasis).all()

        if records:
            basis_values = [float(r.basis_pct) for r in records]

            avg_basis = sum(basis_values) / len(basis_values)
            min_basis = min(basis_values)
            max_basis = max(basis_values)

            print(f"\nTotal records: {len(records)}")
            print(f"\nBasis % Statistics:")
            print(f"  Average: {avg_basis:+.4f}%")
            print(f"  Min: {min_basis:+.4f}%")
            print(f"  Max: {max_basis:+.4f}%")
            print(f"  Range: {max_basis - min_basis:.4f}%")

            # Count premium vs discount
            premium_count = sum(1 for b in basis_values if b > 0)
            discount_count = sum(1 for b in basis_values if b < 0)

            print(f"\nMarket Sentiment:")
            print(f"  Premium (bullish): {premium_count} ({premium_count/len(records)*100:.1f}%)")
            print(f"  Discount (bearish): {discount_count} ({discount_count/len(records)*100:.1f}%)")
        else:
            print("\nNo basis data found")


def example_4_basis_by_source():
    """Compare basis across different sources"""
    print("\n" + "="*80)
    print("EXAMPLE 4: Basis by Source")
    print("="*80)

    with db_manager.get_session() as session:
        # Get latest from each source
        sources = session.query(FuturesBasis.source).distinct().all()

        for (source,) in sources:
            latest = session.query(FuturesBasis).filter(
                FuturesBasis.source == source
            ).order_by(desc(FuturesBasis.timestamp)).first()

            if latest:
                age = (datetime.utcnow() - latest.timestamp).total_seconds()
                print(f"\n{source}:")
                print(f"  Latest: {latest.timestamp} ({age:.1f}s ago)")
                print(f"  Basis: ${float(latest.basis):+,.2f} ({float(latest.basis_pct):+.4f}%)")


def main():
    """Run all examples"""
    print("\n" + "="*80)
    print("FUTURES BASIS QUERY EXAMPLES")
    print("="*80)

    example_1_latest_basis()
    example_2_basis_time_series()
    example_3_basis_statistics()
    example_4_basis_by_source()

    print("\n" + "="*80)
    print("USAGE:")
    print("  Run specific example: venv/bin/python scripts/query_basis.py 1")
    print("  Run all examples:     venv/bin/python scripts/query_basis.py")
    print("="*80 + "\n")


if __name__ == "__main__":
    import sys

    examples = {
        '1': example_1_latest_basis,
        '2': example_2_basis_time_series,
        '3': example_3_basis_statistics,
        '4': example_4_basis_by_source,
        'all': main
    }

    if len(sys.argv) > 1:
        example_num = sys.argv[1]
        if example_num in examples:
            examples[example_num]()
        else:
            print(f"Unknown example: {example_num}")
            print(f"Available: {', '.join(examples.keys())}")
    else:
        main()
