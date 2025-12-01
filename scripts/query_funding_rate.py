#!/usr/bin/env python3
"""
Query and analyze funding rate data from the database.

Examples:
1. Latest funding rate metrics
2. Time series with changes
3. Detect patterns and alerts
4. Statistics over time window
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.storage.database import db_manager
from data.storage.models import FundingRateMetrics
from sqlalchemy import desc, func
from datetime import datetime, timedelta


def example_1_latest():
    """Get the latest funding rate metrics"""
    print("\n" + "="*80)
    print("EXAMPLE 1: Latest Funding Rate Metrics")
    print("="*80 + "\n")

    with db_manager.get_session() as session:
        latest = session.query(FundingRateMetrics).order_by(
            desc(FundingRateMetrics.timestamp)
        ).first()

        if not latest:
            print("No funding rate data found")
            return

        # Convert funding rate to percentage (it's stored as decimal per 8 hours)
        rate_pct = float(latest.funding_rate) * 100

        print(f"Timestamp: {latest.timestamp}")
        print(f"\nCore Metrics:")
        print(f"  Funding Rate: {rate_pct:.4f}% per 8 hours")
        print(f"  Mark Price: ${float(latest.mark_price):,.2f}")
        print(f"  Index Price: ${float(latest.index_price):,.2f}")
        print(f"  Mark Premium: {float(latest.mark_premium):+.4f}%")

        if latest.next_funding_time:
            time_until = latest.next_funding_time - datetime.utcnow()
            hours = time_until.total_seconds() / 3600
            print(f"  Next Funding: {latest.next_funding_time} ({hours:.1f}h)")

        print(f"\nChanges:")
        if latest.funding_rate_change_30s is not None:
            print(f"  30s Δ: {float(latest.funding_rate_change_30s):.8f}")
        if latest.funding_rate_change_1min is not None:
            print(f"  1min Δ: {float(latest.funding_rate_change_1min):.8f}")
        if latest.funding_rate_change_5min is not None:
            print(f"  5min Δ: {float(latest.funding_rate_change_5min):.8f}")

        print(f"\nDerivatives:")
        if latest.funding_rate_velocity is not None:
            print(f"  Velocity: {float(latest.funding_rate_velocity):.8f} per minute")
        if latest.funding_rate_acceleration is not None:
            print(f"  Acceleration: {float(latest.funding_rate_acceleration):.8f}")

        if latest.mark_premium_change_1min is not None:
            print(f"  Mark Premium 1min Δ: {float(latest.mark_premium_change_1min):+.6f}%")

        # Interpret
        print(f"\n💡 Interpretation:")
        if float(latest.funding_rate) > 0:
            print(f"  ✅ Positive funding rate → Longs paying shorts (bullish sentiment)")
            if float(latest.funding_rate) > 0.0001:
                print(f"  ⚠️  HIGH funding rate → Potential long squeeze risk")
        else:
            print(f"  🔴 Negative funding rate → Shorts paying longs (bearish sentiment)")

        if float(latest.mark_premium) > 0.05:
            print(f"  📈 Mark premium high → Strong long demand")
        elif float(latest.mark_premium) < -0.05:
            print(f"  📉 Mark discount → Weak long demand or short pressure")


def example_2_time_series():
    """Show recent time series with changes"""
    print("\n" + "="*80)
    print("EXAMPLE 2: Recent Time Series (Last 10 readings)")
    print("="*80 + "\n")

    with db_manager.get_session() as session:
        recent = session.query(FundingRateMetrics).order_by(
            desc(FundingRateMetrics.timestamp)
        ).limit(10).all()

        if not recent:
            print("No funding rate data found")
            return

        recent.reverse()  # Show oldest to newest

        print(f"{'Timestamp':<20} {'Rate %':<10} {'Premium %':<12} {'1min Δ':<15} {'Velocity':<15}")
        print("-" * 80)

        for record in recent:
            rate_pct = float(record.funding_rate) * 100
            premium = float(record.mark_premium) if record.mark_premium else 0

            change_1min = ""
            if record.funding_rate_change_1min is not None:
                change_1min = f"{float(record.funding_rate_change_1min):+.8f}"

            velocity = ""
            if record.funding_rate_velocity is not None:
                velocity = f"{float(record.funding_rate_velocity):+.8f}"

            print(f"{str(record.timestamp)[:19]:<20} {rate_pct:>8.4f}  {premium:>+10.4f}  {change_1min:<15} {velocity:<15}")


def example_3_patterns():
    """Detect patterns and generate alerts"""
    print("\n" + "="*80)
    print("EXAMPLE 3: Pattern Detection & Alerts (Last hour)")
    print("="*80 + "\n")

    with db_manager.get_session() as session:
        one_hour_ago = datetime.utcnow() - timedelta(hours=1)

        recent = session.query(FundingRateMetrics).filter(
            FundingRateMetrics.timestamp >= one_hour_ago
        ).order_by(desc(FundingRateMetrics.timestamp)).all()

        if not recent:
            print("No funding rate data found in last hour")
            return

        alerts = []

        for record in recent:
            rate = float(record.funding_rate)
            change_1min = float(record.funding_rate_change_1min) if record.funding_rate_change_1min else None
            premium = float(record.mark_premium) if record.mark_premium else 0

            # Spike detection
            if change_1min is not None and rate != 0:
                pct_change = abs(change_1min / rate) * 100
                if pct_change > 50:
                    if change_1min > 0:
                        alerts.append((record.timestamp, "⚠️  SPIKE", f"Rate jumped {pct_change:.1f}% in 1min"))
                    else:
                        alerts.append((record.timestamp, "⚠️  DROP", f"Rate dropped {pct_change:.1f}% in 1min"))
                elif pct_change > 20:
                    if change_1min > 0:
                        alerts.append((record.timestamp, "↗️  Rising", f"Rate up {pct_change:.1f}% in 1min"))
                    else:
                        alerts.append((record.timestamp, "↘️  Falling", f"Rate down {pct_change:.1f}% in 1min"))

            # Extreme rates
            if rate > 0.0001:
                alerts.append((record.timestamp, "🔴 HIGH", f"Extreme positive rate: {rate*100:.4f}%"))
            elif rate < -0.0001:
                alerts.append((record.timestamp, "🟢 NEGATIVE", f"Extreme negative rate: {rate*100:.4f}%"))

            # Divergence
            if abs(premium) > 0.05 and rate != 0:
                ratio = abs(premium) / abs(rate * 100)
                if ratio > 100:
                    alerts.append((record.timestamp, "⚡ DIVERGENCE", f"Premium/Rate ratio: {ratio:.0f}x"))

        if alerts:
            print(f"Found {len(alerts)} alerts:\n")
            for timestamp, alert_type, message in alerts[-10:]:  # Show last 10
                print(f"{str(timestamp)[:19]} | {alert_type:<15} | {message}")
        else:
            print("No significant patterns detected in last hour ✅")


def example_4_statistics():
    """Calculate statistics over time window"""
    print("\n" + "="*80)
    print("EXAMPLE 4: Statistics (Last 1 hour)")
    print("="*80 + "\n")

    with db_manager.get_session() as session:
        one_hour_ago = datetime.utcnow() - timedelta(hours=1)

        stats = session.query(
            func.count(FundingRateMetrics.id).label('count'),
            func.avg(FundingRateMetrics.funding_rate).label('avg_rate'),
            func.min(FundingRateMetrics.funding_rate).label('min_rate'),
            func.max(FundingRateMetrics.funding_rate).label('max_rate'),
            func.avg(FundingRateMetrics.mark_premium).label('avg_premium'),
            func.min(FundingRateMetrics.mark_premium).label('min_premium'),
            func.max(FundingRateMetrics.mark_premium).label('max_premium'),
        ).filter(
            FundingRateMetrics.timestamp >= one_hour_ago
        ).first()

        if not stats or stats.count == 0:
            print("No data in last hour")
            return

        print(f"Data points: {stats.count}")
        print(f"\nFunding Rate:")
        print(f"  Average: {float(stats.avg_rate)*100:.4f}%")
        print(f"  Min: {float(stats.min_rate)*100:.4f}%")
        print(f"  Max: {float(stats.max_rate)*100:.4f}%")
        print(f"  Range: {(float(stats.max_rate) - float(stats.min_rate))*100:.4f}%")

        print(f"\nMark Premium:")
        print(f"  Average: {float(stats.avg_premium):+.4f}%")
        print(f"  Min: {float(stats.min_premium):+.4f}%")
        print(f"  Max: {float(stats.max_premium):+.4f}%")
        print(f"  Range: {float(stats.max_premium) - float(stats.min_premium):.4f}%")


def main():
    """Run examples based on command line argument"""
    import argparse

    parser = argparse.ArgumentParser(description='Query funding rate data')
    parser.add_argument('example', type=int, nargs='?', default=0,
                       help='Example number (1-4), or 0 for all')

    args = parser.parse_args()

    examples = {
        1: example_1_latest,
        2: example_2_time_series,
        3: example_3_patterns,
        4: example_4_statistics,
    }

    if args.example == 0:
        # Run all examples
        for example_func in examples.values():
            example_func()
    elif args.example in examples:
        examples[args.example]()
    else:
        print(f"Invalid example number: {args.example}")
        print("Valid options: 1-4, or 0 for all")
        sys.exit(1)


if __name__ == "__main__":
    main()
