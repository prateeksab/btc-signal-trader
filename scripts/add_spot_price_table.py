#!/usr/bin/env python3
"""
Add spot_price_snapshots table to database
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.storage.database import db_manager
from data.storage.models import SpotPriceSnapshot

def main():
    """Create the spot_price_snapshots table"""
    print("Creating spot_price_snapshots table...")

    try:
        # Create table
        SpotPriceSnapshot.__table__.create(db_manager.engine, checkfirst=True)
        print("✅ Table created successfully!")

    except Exception as e:
        print(f"❌ Error creating table: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
