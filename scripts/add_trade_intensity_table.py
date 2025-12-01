#!/usr/bin/env python3
"""
Add Trade Intensity Table

Adds the trade_intensity table to existing database.
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.storage.database import db_manager
from data.storage.models import TradeIntensity

print("=" * 80)
print("Adding Trade Intensity Table")
print("=" * 80)

try:
    # Create only the trade_intensity table
    TradeIntensity.__table__.create(db_manager.engine, checkfirst=True)

    print("\n✅ trade_intensity table created successfully!")
    print("\nTable structure:")
    print("  - timestamp, exchange, symbol")
    print("  - trades_per_sec_10s, trades_per_sec_30s, trades_per_sec_60s")
    print("  - avg_trade_size_30s, avg_trade_size_60s")
    print("  - buy_sell_ratio, aggression_score")
    print("  - velocity_change")
    print("\nIndexes:")
    print("  - idx_trade_intensity_timestamp")
    print("  - idx_trade_intensity_exchange_symbol")

except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
