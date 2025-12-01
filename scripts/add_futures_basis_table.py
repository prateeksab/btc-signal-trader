#!/usr/bin/env python3
"""
Add Futures Basis Table

Adds the futures_basis table to existing database.
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.storage.database import db_manager
from data.storage.models import FuturesBasis

print("=" * 80)
print("Adding Futures Basis Table")
print("=" * 80)

try:
    # Create only the futures_basis table
    FuturesBasis.__table__.create(db_manager.engine, checkfirst=True)

    print("\n✅ futures_basis table created successfully!")
    print("\nTable structure:")
    print("  - timestamp, source, symbol")
    print("  - spot_price, futures_price")
    print("  - basis, basis_pct, basis_annualized_pct")
    print("  - funding_rate, time_to_expiry_hours")
    print("\nIndexes:")
    print("  - idx_futures_basis_timestamp")
    print("  - idx_futures_basis_source")

except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
