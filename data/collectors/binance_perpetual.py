#!/usr/bin/env python3
"""
Binance BTC Perpetual Futures Basis Tracker

Connects to Binance perpetual futures WebSocket to track the futures-spot basis.
This is a reliable, no-auth alternative to Coinbase futures.

Calculation:
- Perpetual price from Binance BTCUSDT perpetual contract
- Spot price from our Coinbase spot trades
- Basis = perpetual_price - spot_price
- Basis % = (basis / spot_price) * 100
"""

import asyncio
import json
import logging
import signal
import sys
import os
from datetime import datetime, timedelta
from typing import Optional
from decimal import Decimal
import websockets
from websockets.exceptions import ConnectionClosed, WebSocketException

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
        logging.FileHandler('logs/binance_perpetual.log')
    ]
)
logger = logging.getLogger(__name__)


class BinancePerpetualBasisTracker:
    """
    WebSocket client for tracking BTC perpetual futures basis on Binance.

    Tracks:
    - Binance perpetual futures price (BTCUSDT-PERP)
    - Coinbase spot price (from our database)
    - Basis and basis percentage
    """

    def __init__(
        self,
        websocket_url: str = "wss://fstream.binance.com/ws/btcusdt@aggTrade",
        update_interval: int = 60
    ):
        """
        Initialize the Binance perpetual basis tracker.

        Args:
            websocket_url: Binance futures WebSocket URL
            update_interval: How often to calculate and save basis (seconds)
        """
        self.websocket_url = websocket_url
        self.update_interval = update_interval

        # Price tracking
        self.latest_perp_price = None
        self.latest_perp_time = None

        # Connection management
        self.websocket = None
        self.is_running = False
        self.reconnect_delay = 1
        self.max_reconnect_delay = 60
        self.reconnect_attempts = 0

        logger.info("Initialized BinancePerpetualBasisTracker")

    async def connect(self):
        """Establish WebSocket connection to Binance."""
        try:
            self.websocket = await websockets.connect(self.websocket_url)
            logger.info(f"Connected to Binance Futures WebSocket: {self.websocket_url}")

            # Reset reconnect parameters
            self.reconnect_delay = 1
            self.reconnect_attempts = 0

            return True
        except Exception as e:
            logger.error(f"Failed to connect to WebSocket: {e}")
            return False

    def parse_trade(self, data: dict) -> Optional[float]:
        """
        Parse aggTrade message from Binance.

        Message format:
        {
            "e": "aggTrade",
            "E": 1234567890,  # event time
            "s": "BTCUSDT",
            "p": "95000.50",  # price
            "q": "0.5",       # quantity
            ...
        }

        Args:
            data: Raw trade data from Binance

        Returns:
            Perpetual price or None if parsing fails
        """
        try:
            if data.get('e') == 'aggTrade':
                price = float(data['p'])
                self.latest_perp_price = price
                self.latest_perp_time = datetime.utcnow()
                return price
            return None
        except (KeyError, ValueError, TypeError) as e:
            logger.error(f"Error parsing trade data: {e}")
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

        # For perpetuals, annualized basis doesn't apply (use funding rate instead)
        # We'll set it to None
        basis_annualized_pct = None

        return {
            'basis': Decimal(str(round(basis, 8))),
            'basis_pct': Decimal(str(round(basis_pct, 4))),
            'basis_annualized_pct': basis_annualized_pct,
        }

    def save_basis(self, perp_price: float, spot_price: float):
        """
        Save basis metrics to database.

        Args:
            perp_price: Perpetual futures price
            spot_price: Spot price
        """
        try:
            basis_metrics = self.calculate_basis(perp_price, spot_price)

            with db_manager.get_session() as session:
                basis_record = FuturesBasis(
                    timestamp=datetime.utcnow(),
                    source='binance_perp',
                    symbol='BTCUSDT',
                    spot_price=Decimal(str(round(spot_price, 8))),
                    futures_price=Decimal(str(round(perp_price, 8))),
                    basis=basis_metrics['basis'],
                    basis_pct=basis_metrics['basis_pct'],
                    basis_annualized_pct=basis_metrics['basis_annualized_pct'],
                    funding_rate=None,  # Could fetch from funding rate endpoint if needed
                    time_to_expiry_hours=None  # Perpetual has no expiry
                )
                session.add(basis_record)

            logger.info(
                f"Basis saved: Perp=${perp_price:,.2f}, Spot=${spot_price:,.2f}, "
                f"Basis=${float(basis_metrics['basis']):+,.2f} ({float(basis_metrics['basis_pct']):+.4f}%)"
            )

        except Exception as e:
            logger.error(f"Error saving basis to database: {e}")

    async def basis_update_loop(self):
        """Periodic task to calculate and save basis."""
        while self.is_running:
            try:
                await asyncio.sleep(self.update_interval)

                # Check if we have recent perp price
                if self.latest_perp_price is None:
                    logger.warning("No perpetual price available yet")
                    continue

                # Check perp price age
                if self.latest_perp_time:
                    perp_age = (datetime.utcnow() - self.latest_perp_time).total_seconds()
                    if perp_age > 30:
                        logger.warning(f"Perpetual price is {perp_age:.1f}s old")
                        continue

                # Get spot price
                spot_price = self.get_latest_spot_price()
                if spot_price is None:
                    logger.warning("No spot price available")
                    continue

                # Calculate and save basis
                self.save_basis(self.latest_perp_price, spot_price)

            except Exception as e:
                logger.error(f"Error in basis update loop: {e}")

    async def handle_message(self, message: str):
        """
        Handle incoming WebSocket message from Binance.

        Args:
            message: Raw WebSocket message
        """
        try:
            data = json.loads(message)
            self.parse_trade(data)

        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode JSON message: {e}")
        except Exception as e:
            logger.error(f"Error handling message: {e}")

    async def listen(self):
        """Listen for incoming WebSocket messages."""
        try:
            async for message in self.websocket:
                await self.handle_message(message)
        except ConnectionClosed:
            logger.warning("WebSocket connection closed")
            raise
        except Exception as e:
            logger.error(f"Error in listen loop: {e}")
            raise

    async def reconnect(self):
        """Handle reconnection with exponential backoff."""
        self.reconnect_attempts += 1

        logger.info(f"Attempting to reconnect (attempt {self.reconnect_attempts})...")
        logger.info(f"Waiting {self.reconnect_delay} seconds before reconnecting...")

        await asyncio.sleep(self.reconnect_delay)

        # Exponential backoff
        self.reconnect_delay = min(self.reconnect_delay * 2, self.max_reconnect_delay)

        return await self.connect()

    async def start(self):
        """Start the WebSocket collector with auto-reconnect."""
        self.is_running = True

        print(f"\n{'='*80}")
        print(f"Binance BTC Perpetual Futures Basis Tracker")
        print(f"{'='*80}")
        print(f"Perpetual: Binance BTCUSDT-PERP")
        print(f"Spot: Coinbase BTC-USD (from database)")
        print(f"Update interval: {self.update_interval}s")
        print(f"{'='*80}\n")

        # Start the basis update loop in background
        asyncio.create_task(self.basis_update_loop())

        while self.is_running:
            try:
                # Connect to WebSocket
                connected = await self.connect()
                if not connected:
                    await self.reconnect()
                    continue

                # Listen for messages
                await self.listen()

            except ConnectionClosed:
                if self.is_running:
                    logger.warning("Connection lost, attempting to reconnect...")
                    await self.reconnect()
                else:
                    logger.info("Connection closed gracefully")
                    break

            except WebSocketException as e:
                if self.is_running:
                    logger.error(f"WebSocket error: {e}")
                    await self.reconnect()
                else:
                    break

            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                if self.is_running:
                    await self.reconnect()
                else:
                    break

    async def stop(self):
        """Stop the WebSocket collector gracefully."""
        logger.info("Stopping BinancePerpetualBasisTracker...")
        self.is_running = False

        if self.websocket:
            await self.websocket.close()
            logger.info("WebSocket connection closed")

        print(f"\n{'='*80}")
        print(f"Binance Perpetual Basis Tracker Stopped")
        print(f"{'='*80}\n")


async def main():
    """Main function for testing the collector."""
    tracker = BinancePerpetualBasisTracker()

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
