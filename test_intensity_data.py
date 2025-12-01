#!/usr/bin/env python3
"""Quick test to verify trade intensity data is being collected"""

import sys
import os
from datetime import datetime, timedelta

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from data.storage.database import db_manager
from data.storage.models import TradeIntensity
from sqlalchemy import func

with db_manager.get_session() as session:
    # Get total intensity record count
    total_count = session.query(func.count(TradeIntensity.id)).scalar()

    # Get intensity records from last 1 minute
    one_min_ago = datetime.utcnow() - timedelta(minutes=1)
    recent_count = session.query(func.count(TradeIntensity.id)).filter(
        TradeIntensity.timestamp >= one_min_ago
    ).scalar()

    # Get most recent intensity records
    recent_records = session.query(TradeIntensity).order_by(
        TradeIntensity.timestamp.desc()
    ).limit(5).all()

    print(f"\n{'='*80}")
    print("TRADE INTENSITY DATABASE STATUS")
    print(f"{'='*80}")
    print(f"\nTotal intensity records: {total_count:,}")
    print(f"Records in last 1 minute: {recent_count}")
    print(f"\nMost recent intensity snapshots:")
    print(f"{'-'*80}")

    for record in recent_records:
        age_seconds = (datetime.utcnow() - record.timestamp).total_seconds()
        print(f"\n{record.timestamp} ({age_seconds:.1f}s ago)")
        print(f"  TPS (60s): {float(record.trades_per_sec_60s):.2f}")
        print(f"  Aggression: {float(record.aggression_score):+.2f}")
        print(f"  Velocity: {float(record.velocity_change) if record.velocity_change else 0:+.2%}")

    print(f"\n{'='*80}\n")
