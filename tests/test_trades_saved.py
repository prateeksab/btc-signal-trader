#!/usr/bin/env python3
"""Quick test to verify trades are being saved to database"""

import sys
import os
from datetime import datetime, timedelta

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from data.storage.database import db_manager
from data.storage.models import Trade
from sqlalchemy import func

with db_manager.get_session() as session:
    # Get total trade count
    total_count = session.query(func.count(Trade.id)).scalar()

    # Get trades from last 1 minute
    one_min_ago = datetime.utcnow() - timedelta(minutes=1)
    recent_count = session.query(func.count(Trade.id)).filter(
        Trade.timestamp >= one_min_ago
    ).scalar()

    # Get most recent trades
    recent_trades = session.query(Trade).order_by(
        Trade.timestamp.desc()
    ).limit(5).all()

    print(f"\n{'='*80}")
    print("TRADE DATABASE STATUS")
    print(f"{'='*80}")
    print(f"\nTotal trades: {total_count:,}")
    print(f"Trades in last 1 minute: {recent_count}")
    print(f"\nMost recent trades:")
    print(f"{'-'*80}")

    for trade in recent_trades:
        age_seconds = (datetime.utcnow() - trade.timestamp).total_seconds()
        print(f"{trade.timestamp} ({age_seconds:.1f}s ago) - {trade.exchange} - {trade.side} - ${float(trade.price):,.2f}")

    print(f"{'='*80}\n")
