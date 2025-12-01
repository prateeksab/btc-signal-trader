#!/usr/bin/env python3
"""
Database Initialization Script

Creates database tables and optionally seeds with sample data.
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.storage.database import db_manager, init_database
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Initialize database"""
    print("="*80)
    print("BTC Signal Trader - Database Initialization")
    print("="*80)

    # Get database URL from env or use default
    database_url = os.getenv('DATABASE_URL', 'sqlite:///data/trading_signals.db')

    print(f"\nDatabase URL: {database_url}")
    print("\nThis will create all database tables.")

    response = input("\nContinue? (yes/no): ")

    if response.lower() not in ['yes', 'y']:
        print("Aborted.")
        return

    try:
        # Initialize database
        init_database(database_url)

        print("\n✅ Database initialized successfully!")
        print("\nCreated tables:")
        print("  - trades")
        print("  - orderbook_snapshots")
        print("  - candles")
        print("  - cvd_snapshots")
        print("  - indicators")
        print("  - signals")
        print("  - positions")
        print("  - backtest_runs")
        print("  - system_metrics")
        print("  - alerts")

        print("\nDatabase is ready to use!")

    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
