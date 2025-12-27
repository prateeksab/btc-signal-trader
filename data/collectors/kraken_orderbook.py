"""
Kraken XBT/USD Real-Time Order Book WebSocket Collector

This module connects to Kraken's WebSocket stream to collect real-time
order book data and calculates order book imbalance.
"""

import asyncio
import json
import logging
import signal
import sys
import os
from datetime import datetime
from typing import Optional, Dict, Any, List
from collections import OrderedDict
from decimal import Decimal
import websockets
from websockets.exceptions import ConnectionClosed, WebSocketException

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from data.storage.database import db_manager
from data.storage.models import OrderbookSnapshot

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('logs/kraken_orderbook.log')
    ]
)
logger = logging.getLogger(__name__)


class KrakenOrderBookCollector:
    """
    WebSocket client for collecting real-time XBT/USD order book from Kraken.

    Maintains order book state and calculates:
    - Order book imbalance
    - Bid/Ask spread
    - Depth metrics
    """

    def __init__(self, pair: str = "XBT/USD", depth: int = 10, websocket_url: str = "wss://ws.kraken.com/"):
        """
        Initialize the Kraken order book collector.

        Args:
            pair: Trading pair symbol (default: XBT/USD)
            depth: Order book depth to subscribe to (default: 10)
            websocket_url: Kraken WebSocket URL
        """
        self.pair = pair
        self.depth = depth
        self.websocket_url = websocket_url

        # Order book state
        self.bids: OrderedDict = OrderedDict()  # price -> volume
        self.asks: OrderedDict = OrderedDict()  # price -> volume
        self.is_snapshot_received = False

        # Metrics tracking
        self.imbalance_history = []
        self.update_count = 0

        # Connection management
        self.websocket = None
        self.is_running = False
        self.reconnect_delay = 1
        self.max_reconnect_delay = 60
        self.reconnect_attempts = 0

        logger.info(f"Initialized KrakenOrderBookCollector for {self.pair} (depth: {self.depth})")

    async def connect(self):
        """Establish WebSocket connection to Kraken."""
        try:
            self.websocket = await websockets.connect(self.websocket_url)
            logger.info(f"Connected to Kraken WebSocket: {self.websocket_url}")

            # Subscribe to order book
            subscribe_message = {
                "event": "subscribe",
                "pair": [self.pair],
                "subscription": {
                    "name": "book",
                    "depth": self.depth
                }
            }

            await self.websocket.send(json.dumps(subscribe_message))
            logger.info(f"Sent subscription request for {self.pair} order book (depth: {self.depth})")

            # Reset reconnect parameters on successful connection
            self.reconnect_delay = 1
            self.reconnect_attempts = 0

            return True
        except Exception as e:
            logger.error(f"Failed to connect to WebSocket: {e}")
            return False

    def update_order_book(self, bids: List[List], asks: List[List], is_snapshot: bool = False):
        """
        Update order book with new data.

        Args:
            bids: List of [price, volume, timestamp] for bids
            asks: List of [price, volume, timestamp] for asks
            is_snapshot: True if this is a snapshot (replaces entire book)
        """
        try:
            if is_snapshot:
                # Clear existing order book
                self.bids.clear()
                self.asks.clear()
                logger.info("Received order book snapshot")

            # Update bids
            for bid in bids:
                price = float(bid[0])
                volume = float(bid[1])

                if volume == 0:
                    # Remove price level
                    self.bids.pop(price, None)
                else:
                    # Update price level
                    self.bids[price] = volume

            # Update asks
            for ask in asks:
                price = float(ask[0])
                volume = float(ask[1])

                if volume == 0:
                    # Remove price level
                    self.asks.pop(price, None)
                else:
                    # Update price level
                    self.asks[price] = volume

            # Sort order books
            self.bids = OrderedDict(sorted(self.bids.items(), reverse=True))  # Highest bid first
            self.asks = OrderedDict(sorted(self.asks.items()))  # Lowest ask first

            if is_snapshot:
                self.is_snapshot_received = True

        except (ValueError, TypeError, IndexError) as e:
            logger.error(f"Error updating order book: {e}")

    def calculate_imbalance(self, levels: int = 5) -> Optional[float]:
        """
        Calculate order book imbalance.

        Imbalance = (bid_volume - ask_volume) / (bid_volume + ask_volume)

        Values range from -1 to +1:
        - Positive: more bid volume (buying pressure)
        - Negative: more ask volume (selling pressure)

        Args:
            levels: Number of price levels to include in calculation

        Returns:
            Imbalance ratio or None if insufficient data
        """
        if not self.is_snapshot_received:
            return None

        try:
            # Get top N levels
            bid_prices = list(self.bids.keys())[:levels]
            ask_prices = list(self.asks.keys())[:levels]

            if not bid_prices or not ask_prices:
                return None

            # Sum volumes
            bid_volume = sum(self.bids[price] for price in bid_prices)
            ask_volume = sum(self.asks[price] for price in ask_prices)

            total_volume = bid_volume + ask_volume

            if total_volume == 0:
                return 0.0

            imbalance = (bid_volume - ask_volume) / total_volume
            return imbalance

        except Exception as e:
            logger.error(f"Error calculating imbalance: {e}")
            return None

    def get_spread(self) -> Optional[Dict[str, float]]:
        """
        Calculate bid-ask spread.

        Returns:
            Dictionary with best_bid, best_ask, spread, and spread_pct
        """
        if not self.is_snapshot_received or not self.bids or not self.asks:
            return None

        try:
            best_bid = list(self.bids.keys())[0]
            best_ask = list(self.asks.keys())[0]

            spread = best_ask - best_bid
            spread_pct = (spread / best_ask) * 100

            return {
                'best_bid': best_bid,
                'best_ask': best_ask,
                'spread': spread,
                'spread_pct': spread_pct
            }
        except Exception as e:
            logger.error(f"Error calculating spread: {e}")
            return None

    def save_to_database(self):
        """Save order book snapshot to database."""
        if not self.is_snapshot_received:
            return

        try:
            spread = self.get_spread()
            imbalance = self.calculate_imbalance()

            if not spread:
                return

            best_bid = list(self.bids.keys())[0] if self.bids else None
            best_ask = list(self.asks.keys())[0] if self.asks else None

            if not best_bid or not best_ask:
                return

            # Calculate total volumes
            total_bid_volume = sum(list(self.bids.values())[:5])
            total_ask_volume = sum(list(self.asks.values())[:5])

            with db_manager.get_session() as session:
                snapshot = OrderbookSnapshot(
                    timestamp=datetime.utcnow(),
                    exchange='kraken',
                    symbol='XBT/USD',
                    best_bid=Decimal(str(best_bid)),
                    best_ask=Decimal(str(best_ask)),
                    spread=Decimal(str(spread['spread'])),
                    spread_pct=Decimal(str(spread['spread_pct'])),
                    total_bid_volume=Decimal(str(total_bid_volume)),
                    total_ask_volume=Decimal(str(total_ask_volume)),
                    imbalance=Decimal(str(imbalance * 100)) if imbalance is not None else None,
                    bids_json={'bids': [[str(p), str(v)] for p, v in list(self.bids.items())[:10]]},
                    asks_json={'asks': [[str(p), str(v)] for p, v in list(self.asks.items())[:10]]},
                    depth_levels=min(len(self.bids), len(self.asks))
                )
                session.add(snapshot)

        except Exception as e:
            logger.error(f"Error saving orderbook to database: {e}")

    def print_order_book_snapshot(self):
        """Print current order book state to console."""
        if not self.is_snapshot_received:
            return

        spread = self.get_spread()
        imbalance = self.calculate_imbalance()

        if not spread:
            return

        # Save to database
        self.save_to_database()

        print(f"\n{'='*80}")
        print(f"Order Book Snapshot - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*80}")

        # Display asks (highest to lowest, limited to 5)
        ask_prices = list(self.asks.keys())[:5]
        ask_prices.reverse()

        print(f"\n{'ASKS':^40}")
        print(f"{'Price':>20} | {'Volume':>15}")
        print(f"{'-'*40}")
        for price in ask_prices:
            print(f"\033[91m${price:>18,.2f}\033[0m | {self.asks[price]:>15.6f}")

        # Display spread
        print(f"\n{'SPREAD':^40}")
        print(f"${spread['spread']:.2f} ({spread['spread_pct']:.3f}%)")

        # Display bids (highest to lowest, limited to 5)
        bid_prices = list(self.bids.keys())[:5]

        print(f"\n{'BIDS':^40}")
        print(f"{'Price':>20} | {'Volume':>15}")
        print(f"{'-'*40}")
        for price in bid_prices:
            print(f"\033[92m${price:>18,.2f}\033[0m | {self.bids[price]:>15.6f}")

        # Display imbalance
        if imbalance is not None:
            imbalance_pct = imbalance * 100
            color = "\033[92m" if imbalance > 0 else "\033[91m"
            direction = "BUYING PRESSURE" if imbalance > 0 else "SELLING PRESSURE"
            print(f"\n{'IMBALANCE':^40}")
            print(f"{color}{imbalance_pct:+.2f}%\033[0m - {direction}")

        print(f"{'='*80}\n")

    async def handle_message(self, message: str):
        """
        Handle incoming WebSocket message from Kraken.

        Args:
            message: Raw WebSocket message
        """
        try:
            data = json.loads(message)

            # Handle subscription status messages
            if isinstance(data, dict):
                if data.get('event') == 'subscriptionStatus':
                    if data.get('status') == 'subscribed':
                        logger.info(f"Successfully subscribed to {data.get('pair')} {data.get('subscription', {}).get('name')}")
                    elif data.get('status') == 'error':
                        logger.error(f"Subscription error: {data.get('errorMessage')}")
                elif data.get('event') == 'heartbeat':
                    pass
                elif data.get('event') == 'systemStatus':
                    logger.info(f"Kraken system status: {data.get('status')}")
                return

            # Handle order book data (comes as array)
            if isinstance(data, list) and len(data) >= 4:
                channel_name = data[-2]  # Second to last element is channel name

                if 'book' in channel_name:
                    book_data = data[1]

                    # Check if this is a snapshot or update
                    is_snapshot = 'as' in book_data and 'bs' in book_data

                    if is_snapshot:
                        # Snapshot: full order book
                        asks = book_data.get('as', [])
                        bids = book_data.get('bs', [])
                        self.update_order_book(bids, asks, is_snapshot=True)
                        self.print_order_book_snapshot()
                    else:
                        # Incremental update
                        asks = book_data.get('a', [])
                        bids = book_data.get('b', [])
                        self.update_order_book(bids, asks, is_snapshot=False)

                        self.update_count += 1

                        # Print snapshot every 50 updates
                        if self.update_count % 50 == 0:
                            self.print_order_book_snapshot()

                        # Calculate and track imbalance
                        imbalance = self.calculate_imbalance()
                        if imbalance is not None:
                            self.imbalance_history.append({
                                'timestamp': datetime.now(),
                                'imbalance': imbalance
                            })

                            # Keep only last 100 imbalance values
                            if len(self.imbalance_history) > 100:
                                self.imbalance_history.pop(0)

        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode JSON message: {e}")
        except Exception as e:
            logger.error(f"Error handling message: {e}")
            logger.debug(f"Problematic message: {message}")

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
        """
        Handle reconnection with exponential backoff.

        Returns:
            True if reconnection successful, False otherwise
        """
        self.reconnect_attempts += 1

        logger.info(f"Attempting to reconnect (attempt {self.reconnect_attempts})...")
        logger.info(f"Waiting {self.reconnect_delay} seconds before reconnecting...")

        await asyncio.sleep(self.reconnect_delay)

        # Exponential backoff: double the delay up to max
        self.reconnect_delay = min(self.reconnect_delay * 2, self.max_reconnect_delay)

        # Reset order book state on reconnect
        self.is_snapshot_received = False
        self.bids.clear()
        self.asks.clear()

        return await self.connect()

    async def start(self):
        """Start the WebSocket collector with auto-reconnect."""
        self.is_running = True

        print(f"\n{'='*80}")
        print(f"XBT/USD Real-Time Order Book - Kraken")
        print(f"{'='*80}")
        print(f"Tracking order book depth: {self.depth} levels")
        print(f"Calculating order book imbalance")
        print(f"{'='*80}\n")

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
        logger.info("Stopping KrakenOrderBookCollector...")
        self.is_running = False

        if self.websocket:
            # Unsubscribe
            unsubscribe_message = {
                "event": "unsubscribe",
                "pair": [self.pair],
                "subscription": {
                    "name": "book",
                    "depth": self.depth
                }
            }
            try:
                await self.websocket.send(json.dumps(unsubscribe_message))
                await asyncio.sleep(0.5)
            except:
                pass

            await self.websocket.close()
            logger.info("WebSocket connection closed")

        # Final snapshot
        if self.is_snapshot_received:
            self.print_order_book_snapshot()


async def main():
    """Main function for testing the collector."""
    collector = KrakenOrderBookCollector()

    # Setup signal handlers for graceful shutdown
    def signal_handler(sig, frame):
        """Handle Ctrl+C for graceful shutdown."""
        print("\n\nReceived interrupt signal. Shutting down gracefully...")
        asyncio.create_task(collector.stop())

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        await collector.start()
    except KeyboardInterrupt:
        await collector.stop()
    except Exception as e:
        logger.error(f"Fatal error in main: {e}")
        await collector.stop()
        sys.exit(1)


if __name__ == "__main__":
    """Run the collector when executed directly."""
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nShutdown complete.")
        sys.exit(0)
