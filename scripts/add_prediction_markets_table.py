#!/usr/bin/env python3
"""
Add Prediction Markets Table

Adds the prediction_markets table to existing database.
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.storage.database import db_manager
from data.storage.models import PredictionMarket

print("=" * 80)
print("Adding Prediction Markets Table")
print("=" * 80)

try:
    # Create only the prediction_markets table
    from data.storage.models import Base
    PredictionMarket.__table__.create(db_manager.engine, checkfirst=True)

    print("\n✅ prediction_markets table created successfully!")
    print("\nTable structure:")
    print("  - timestamp, source, ticker")
    print("  - title, subtitle, expiration_time")
    print("  - strike_price, strike_type")
    print("  - yes_ask, yes_bid, no_ask, no_bid")
    print("  - implied_probability")
    print("  - volume, open_interest, liquidity")
    print("  - status, extra_metadata")

except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
