#!/usr/bin/env python3
"""
Run migration to create btc_price_candles table
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from data.storage.database import db_manager
from data.storage.models import Base, BtcPriceCandle

def run_migration():
    """Create the btc_price_candles table"""
    print("Creating btc_price_candles table...")

    # Get the engine
    engine = db_manager.engine

    # Create only the BtcPriceCandle table
    BtcPriceCandle.__table__.create(engine, checkfirst=True)

    print("✓ btc_price_candles table created successfully!")
    print(f"  - Columns: {', '.join([c.name for c in BtcPriceCandle.__table__.columns])}")
    print(f"  - Indices: {len(BtcPriceCandle.__table__.indexes)}")

if __name__ == "__main__":
    run_migration()
