"""
Coinbase BTC-USD Real-Time Order Book WebSocket Collector

This module connects to Coinbase's WebSocket stream to collect real-time
order book data and calculates order book imbalance.
"""

import asyncio
import json
import logging
import signal
import sys
from datetime import datetime
from typing import Optional, Dict, Any, List
from collections import OrderedDict
import websockets
from websockets.exceptions import ConnectionClosed, WebSocketException


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('logs/coinbase_orderbook.log')
    ]
)
logger = logging.getLogger(__name__)


class CoinbaseOrderBookCollector:
    """
    WebSocket client for collecting real-time BTC-USD order book from Coinbase.

    Maintains order book state and calculates:
    - Order book imbalance
    - Bid/Ask spread
    - Depth metrics
    """

    def __init__(self, product_id: str = "BTC-USD", websocket_url: str = "wss://ws-feed.exchange.coinbase.com"):
        """
        Initialize the Coinbase order book collector.

        Args:
            product_id: Trading pair product ID (default: BTC-USD)
            websocket_url: Coinbase WebSocket URL
        """
        self.product_id = product_id
        self.websocket_url = websocket_url

        # Order book state
        self.bids: OrderedDict = OrderedDict()  # price -> size
        self.asks: OrderedDict = OrderedDict()  # price -> size
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

        logger.info(f"Initialized CoinbaseOrderBookCollector for {self.product_id}")

    async def connect(self):
        """Establish WebSocket connection to Coinbase."""
        try:
            self.websocket = await websockets.connect(self.websocket_url)
            logger.info(f"Connected to Coinbase WebSocket: {self.websocket_url}")

            # Subscribe to level2 channel (order book)
            subscribe_message = {
                "type": "subscribe",
                "product_ids": [self.product_id],
                "channels": ["level2_batch"]
            }

            await self.websocket.send(json.dumps(subscribe_message))
            logger.info(f"Sent subscription request for {self.product_id} level2")

            # Reset reconnect parameters on successful connection
            self.reconnect_delay = 1
            self.reconnect_attempts = 0

            return True
        except Exception as e:
            logger.error(f"Failed to connect to WebSocket: {e}")
            return False

    def process_snapshot(self, snapshot_data: dict):
        """
        Process initial snapshot of the order book.

        Args:
            snapshot_data: Snapshot message from Coinbase
        """
        try:
            # Clear existing order book
            self.bids.clear()
            self.asks.clear()

            # Process bids
            for bid in snapshot_data.get('bids', []):
                price = float(bid[0])
                size = float(bid[1])
                if size > 0:
                    self.bids[price] = size

            # Process asks
            for ask in snapshot_data.get('asks', []):
                price = float(ask[0])
                size = float(ask[1])
                if size > 0:
                    self.asks[price] = size

            # Sort order books
            self.bids = OrderedDict(sorted(self.bids.items(), reverse=True))  # Highest bid first
            self.asks = OrderedDict(sorted(self.asks.items()))  # Lowest ask first

            self.is_snapshot_received = True
            logger.info("Received and processed order book snapshot")

        except (ValueError, TypeError, IndexError) as e:
            logger.error(f"Error processing snapshot: {e}")

    def process_l2update(self, update_data: dict):
        """
        Process incremental l2update message.

        Args:
            update_data: l2update message from Coinbase
        """
        try:
            changes = update_data.get('changes', [])

            for change in changes:
                side = change[0]  # 'buy' or 'sell'
                price = float(change[1])
                size = float(change[2])

                if side == 'buy':
                    if size == 0:
                        # Remove price level
                        self.bids.pop(price, None)
                    else:
                        # Update price level
                        self.bids[price] = size
                elif side == 'sell':
                    if size == 0:
                        # Remove price level
                        self.asks.pop(price, None)
                    else:
                        # Update price level
                        self.asks[price] = size

            # Sort order books
            self.bids = OrderedDict(sorted(self.bids.items(), reverse=True))
            self.asks = OrderedDict(sorted(self.asks.items()))

        except (ValueError, TypeError, IndexError) as e:
            logger.error(f"Error processing l2update: {e}")

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

    def print_order_book_snapshot(self):
        """Print current order book state to console."""
        if not self.is_snapshot_received:
            return

        spread = self.get_spread()
        imbalance = self.calculate_imbalance()

        if not spread:
            return

        print(f"\n{'='*80}")
        print(f"Order Book Snapshot - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*80}")

        # Display asks (highest to lowest, limited to 5)
        ask_prices = list(self.asks.keys())[:5]
        ask_prices.reverse()

        print(f"\n{'ASKS':^40}")
        print(f"{'Price':>20} | {'Size':>15}")
        print(f"{'-'*40}")
        for price in ask_prices:
            print(f"\033[91m${price:>18,.2f}\033[0m | {self.asks[price]:>15.8f}")

        # Display spread
        print(f"\n{'SPREAD':^40}")
        print(f"${spread['spread']:.2f} ({spread['spread_pct']:.4f}%)")

        # Display bids (highest to lowest, limited to 5)
        bid_prices = list(self.bids.keys())[:5]

        print(f"\n{'BIDS':^40}")
        print(f"{'Price':>20} | {'Size':>15}")
        print(f"{'-'*40}")
        for price in bid_prices:
            print(f"\033[92m${price:>18,.2f}\033[0m | {self.bids[price]:>15.8f}")

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
        Handle incoming WebSocket message from Coinbase.

        Args:
            message: Raw WebSocket message
        """
        try:
            data = json.loads(message)
            msg_type = data.get('type')

            # Handle subscription confirmations
            if msg_type == 'subscriptions':
                channels = data.get('channels', [])
                logger.info(f"Subscription confirmed: {channels}")
                return

            # Handle heartbeats
            if msg_type == 'heartbeat':
                return

            # Handle error messages
            if msg_type == 'error':
                logger.error(f"Error from Coinbase: {data.get('message')}")
                return

            # Handle snapshot
            if msg_type == 'snapshot':
                self.process_snapshot(data)
                self.print_order_book_snapshot()
                return

            # Handle incremental updates
            if msg_type == 'l2update':
                if not self.is_snapshot_received:
                    return  # Ignore updates until we have a snapshot

                self.process_l2update(data)
                self.update_count += 1

                # Print snapshot every 100 updates
                if self.update_count % 100 == 0:
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
        print(f"BTC-USD Real-Time Order Book - Coinbase")
        print(f"{'='*80}")
        print(f"Tracking full order book with L2 data")
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
        logger.info("Stopping CoinbaseOrderBookCollector...")
        self.is_running = False

        if self.websocket:
            # Unsubscribe
            unsubscribe_message = {
                "type": "unsubscribe",
                "product_ids": [self.product_id],
                "channels": ["level2"]
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
    collector = CoinbaseOrderBookCollector()

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
