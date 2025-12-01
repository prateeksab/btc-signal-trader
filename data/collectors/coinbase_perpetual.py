#!/usr/bin/env python3
"""
Coinbase BTC Perpetual Futures Basis Tracker

Polls Coinbase REST API for BTC-PERP-INTX perpetual futures price
and calculates the basis against spot prices.

Uses REST API polling (no authentication required) instead of WebSocket.
"""

import asyncio
import logging
import signal
import sys
import os
from datetime import datetime, timedelta
from typing import Optional
from decimal import Decimal
from collections import deque
import requests

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from data.storage.database import db_manager
from data.storage.models import FuturesBasis, Trade
from sqlalchemy import desc


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('logs/coinbase_perpetual.log')
    ]
)
logger = logging.getLogger(__name__)


class CoinbasePerpetualBasisTracker:
    """
    REST API client for tracking Coinbase BTC perpetual futures basis.

    Tracks:
    - Coinbase perpetual futures price (BTC-PERP-INTX)
    - Coinbase spot price (from our database)
    - Basis and basis percentage
    - Basis changes over 1min and 5min windows
    """

    def __init__(
        self,
        api_url: str = "https://api.coinbase.com/api/v3/brokerage/market/products/BTC-PERP-INTX",
        update_interval: int = 60
    ):
        """
        Initialize the Coinbase perpetual basis tracker.

        Args:
            api_url: Coinbase API endpoint for perpetual ticker
            update_interval: How often to poll and save basis (seconds)
        """
        self.api_url = api_url
        self.update_interval = update_interval

        # Session for requests (no auth needed for public market endpoint)
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'btc-signal-trader/1.0',
            'Accept': 'application/json'
        })

        # Basis history for tracking changes
        # Store (timestamp, basis_pct) tuples
        self.basis_history = deque(maxlen=100)  # Keep last 100 readings (~8 minutes at 5s intervals)

        # Running state
        self.is_running = False

        logger.info("Initialized CoinbasePerpetualBasisTracker")

    def fetch_perpetual_price(self) -> Optional[float]:
        """
        Fetch current perpetual futures price from Coinbase public API.

        Returns:
            Perpetual price or None if fetch fails
        """
        try:
            response = self.session.get(self.api_url, timeout=10)
            response.raise_for_status()

            data = response.json()

            # Public market endpoint response format:
            # {"product_id": "BTC-PERP-INTX", "price": "91558.3", ...}
            price = float(data['price'])
            logger.debug(f"Fetched perpetual price: ${price:,.2f}")
            return price

        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching perpetual price: {e}")
            return None
        except (KeyError, ValueError, TypeError) as e:
            logger.error(f"Error parsing perpetual price: {e}")
            return None

    def get_latest_spot_price(self) -> Optional[float]:
        """
        Get the latest Coinbase spot price from database.

        Returns:
            Latest spot price or None if not available
        """
        try:
            with db_manager.get_session() as session:
                # Get most recent Coinbase trade
                latest_trade = session.query(Trade).filter(
                    Trade.exchange == 'coinbase',
                    Trade.symbol == 'BTC-USD'
                ).order_by(desc(Trade.timestamp)).first()

                if latest_trade:
                    # Check if trade is recent (within last 30 seconds)
                    age = (datetime.utcnow() - latest_trade.timestamp).total_seconds()
                    if age > 30:
                        logger.warning(f"Latest spot price is {age:.1f}s old")
                        return None

                    return float(latest_trade.price)

                logger.warning("No Coinbase spot trades found in database")
                return None

        except Exception as e:
            logger.error(f"Error fetching spot price: {e}")
            return None

    def calculate_basis(self, perp_price: float, spot_price: float) -> dict:
        """
        Calculate basis metrics.

        Args:
            perp_price: Perpetual futures price
            spot_price: Spot price

        Returns:
            Dict with basis metrics
        """
        basis = perp_price - spot_price
        basis_pct = (basis / spot_price) * 100 if spot_price > 0 else 0

        return {
            'basis': Decimal(str(round(basis, 8))),
            'basis_pct': Decimal(str(round(basis_pct, 4))),
        }

    def get_basis_change(self, current_basis_pct: float, minutes: int) -> Optional[float]:
        """
        Calculate basis change over the last N minutes.

        Args:
            current_basis_pct: Current basis percentage
            minutes: Time window in minutes

        Returns:
            Basis change (percentage points) or None if insufficient data
        """
        if not self.basis_history:
            return None

        # Find basis reading from N minutes ago
        cutoff_time = datetime.utcnow() - timedelta(minutes=minutes)

        # Find the oldest reading within our time window
        for timestamp, basis_pct in self.basis_history:
            if timestamp <= cutoff_time:
                # Calculate change (in percentage points, not percent change)
                change = current_basis_pct - basis_pct
                return round(change, 4)

        # Not enough history yet
        return None

    def save_basis(self, perp_price: float, spot_price: float):
        """
        Save basis metrics to database.

        Args:
            perp_price: Perpetual futures price
            spot_price: Spot price
        """
        try:
            basis_metrics = self.calculate_basis(perp_price, spot_price)
            basis_pct = float(basis_metrics['basis_pct'])

            # Store in history for tracking changes
            self.basis_history.append((datetime.utcnow(), basis_pct))

            # Calculate basis changes
            basis_change_1min = self.get_basis_change(basis_pct, 1)
            basis_change_5min = self.get_basis_change(basis_pct, 5)

            with db_manager.get_session() as session:
                basis_record = FuturesBasis(
                    timestamp=datetime.utcnow(),
                    source='coinbase_perp',
                    symbol='BTC-PERP-INTX',
                    spot_price=Decimal(str(round(spot_price, 8))),
                    futures_price=Decimal(str(round(perp_price, 8))),
                    basis=basis_metrics['basis'],
                    basis_pct=basis_metrics['basis_pct'],
                    basis_annualized_pct=None,  # Not applicable for perpetuals
                    funding_rate=None,  # Could add funding rate fetching if needed
                    time_to_expiry_hours=None  # Perpetual has no expiry
                )
                session.add(basis_record)

            # Log with change indicators
            change_str = ""
            if basis_change_1min is not None:
                change_str = f", 1m: {basis_change_1min:+.4f}pp"
            if basis_change_5min is not None:
                change_str += f", 5m: {basis_change_5min:+.4f}pp"

            logger.info(
                f"Basis saved: Perp=${perp_price:,.2f}, Spot=${spot_price:,.2f}, "
                f"Basis=${float(basis_metrics['basis']):+,.2f} ({basis_pct:+.4f}%){change_str}"
            )

        except Exception as e:
            logger.error(f"Error saving basis to database: {e}")

    async def update_loop(self):
        """Main update loop - polls API and calculates basis."""
        while self.is_running:
            try:
                # Fetch perpetual price
                perp_price = self.fetch_perpetual_price()
                if perp_price is None:
                    logger.warning("No perpetual price available")
                    await asyncio.sleep(self.update_interval)
                    continue

                # Get spot price
                spot_price = self.get_latest_spot_price()
                if spot_price is None:
                    logger.warning("No spot price available")
                    await asyncio.sleep(self.update_interval)
                    continue

                # Calculate and save basis
                self.save_basis(perp_price, spot_price)

                # Wait for next update
                await asyncio.sleep(self.update_interval)

            except Exception as e:
                logger.error(f"Error in update loop: {e}")
                await asyncio.sleep(self.update_interval)

    async def start(self):
        """Start the basis tracker."""
        self.is_running = True

        print(f"\n{'='*80}")
        print(f"Coinbase BTC Perpetual Futures Basis Tracker")
        print(f"{'='*80}")
        print(f"Perpetual: Coinbase BTC-PERP-INTX")
        print(f"Spot: Coinbase BTC-USD (from database)")
        print(f"Update interval: {self.update_interval}s")
        print(f"Method: REST API polling")
        print(f"{'='*80}\n")

        try:
            await self.update_loop()
        except KeyboardInterrupt:
            logger.info("Received keyboard interrupt")
        except Exception as e:
            logger.error(f"Unexpected error in start: {e}")
        finally:
            await self.stop()

    async def stop(self):
        """Stop the basis tracker gracefully."""
        logger.info("Stopping CoinbasePerpetualBasisTracker...")
        self.is_running = False
        self.session.close()

        print(f"\n{'='*80}")
        print(f"Coinbase Perpetual Basis Tracker Stopped")
        print(f"{'='*80}\n")


async def main():
    """Main function for testing the collector."""


    tracker = CoinbasePerpetualBasisTracker()

    # Setup signal handlers for graceful shutdown
    def signal_handler(sig, frame):
        """Handle Ctrl+C for graceful shutdown."""
        print("\n\nReceived interrupt signal. Shutting down gracefully...")
        asyncio.create_task(tracker.stop())

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        await tracker.start()
    except KeyboardInterrupt:
        await tracker.stop()
    except Exception as e:
        logger.error(f"Fatal error in main: {e}")
        await tracker.stop()
        sys.exit(1)


if __name__ == "__main__":
    """Run the collector when executed directly."""
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nShutdown complete.")
        sys.exit(0)
