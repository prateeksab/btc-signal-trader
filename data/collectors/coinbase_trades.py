"""
Coinbase BTC-USD Real-Time Trades WebSocket Collector

This module connects to Coinbase's WebSocket stream to collect real-time
trade data and calculates Cumulative Volume Delta (CVD).
"""

import asyncio
import json
import logging
import signal
import sys
import os
from datetime import datetime
from typing import Optional, Dict, Any
from decimal import Decimal
import websockets
from websockets.exceptions import ConnectionClosed, WebSocketException

# Add project root to path for database imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from data.storage.database import db_manager
from data.storage.models import Trade


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('logs/coinbase_trades.log')
    ]
)
logger = logging.getLogger(__name__)


class CoinbaseTradesCollector:
    """
    WebSocket client for collecting real-time BTC-USD trades from Coinbase.

    Calculates Cumulative Volume Delta (CVD):
    - Buy orders (taker is buyer): add volume to CVD
    - Sell orders (taker is seller): subtract volume from CVD
    """

    def __init__(self, product_id: str = "BTC-USD", websocket_url: str = "wss://ws-feed.exchange.coinbase.com"):
        """
        Initialize the Coinbase trades collector.

        Args:
            product_id: Trading pair product ID (default: BTC-USD)
            websocket_url: Coinbase WebSocket URL
        """
        self.product_id = product_id
        self.websocket_url = websocket_url

        # CVD tracking
        self.cumulative_volume_delta = 0.0

        # Connection management
        self.websocket = None
        self.is_running = False
        self.reconnect_delay = 1  # Initial reconnect delay in seconds
        self.max_reconnect_delay = 60  # Maximum reconnect delay
        self.reconnect_attempts = 0

        logger.info(f"Initialized CoinbaseTradesCollector for {self.product_id}")

    async def connect(self):
        """Establish WebSocket connection to Coinbase."""
        try:
            self.websocket = await websockets.connect(self.websocket_url)
            logger.info(f"Connected to Coinbase WebSocket: {self.websocket_url}")

            # Subscribe to matches (trades) channel
            subscribe_message = {
                "type": "subscribe",
                "product_ids": [self.product_id],
                "channels": ["matches"]
            }

            await self.websocket.send(json.dumps(subscribe_message))
            logger.info(f"Sent subscription request for {self.product_id} matches")

            # Reset reconnect parameters on successful connection
            self.reconnect_delay = 1
            self.reconnect_attempts = 0

            return True
        except Exception as e:
            logger.error(f"Failed to connect to WebSocket: {e}")
            return False

    def parse_trade(self, trade_data: dict) -> Optional[Dict[str, Any]]:
        """
        Parse raw trade data from Coinbase WebSocket.

        Coinbase match message format:
        {
            "type": "match",
            "trade_id": 12345,
            "maker_order_id": "...",
            "taker_order_id": "...",
            "side": "buy" or "sell",  # taker side
            "size": "0.01234",
            "price": "50000.00",
            "product_id": "BTC-USD",
            "sequence": 1234567890,
            "time": "2014-11-07T08:19:27.028459Z"
        }

        Args:
            trade_data: Raw trade data from Coinbase

        Returns:
            Parsed trade dictionary or None if parsing fails
        """
        try:
            trade = {
                'trade_id': trade_data['trade_id'],
                'price': float(trade_data['price']),
                'size': float(trade_data['size']),
                'side': trade_data['side'],  # 'buy' or 'sell' (taker side)
                'timestamp': datetime.fromisoformat(trade_data['time'].replace('Z', '+00:00')),
                'sequence': trade_data['sequence']
            }
            return trade
        except (KeyError, ValueError, TypeError) as e:
            logger.error(f"Error parsing trade data: {e}")
            return None

    def update_cvd(self, size: float, side: str) -> float:
        """
        Update Cumulative Volume Delta (CVD).

        Logic:
        - Buy (taker is buyer): add volume to CVD
        - Sell (taker is seller): subtract volume from CVD

        Args:
            size: Trade size in BTC
            side: Trade side ('buy' or 'sell')

        Returns:
            Updated CVD value
        """
        if side == 'buy':
            # Buyer is aggressive - add volume
            self.cumulative_volume_delta += size
        elif side == 'sell':
            # Seller is aggressive - subtract volume
            self.cumulative_volume_delta -= size

        return self.cumulative_volume_delta

    def save_trade(self, trade: Dict[str, Any], cvd: float):
        """
        Save trade to database.

        Args:
            trade: Parsed trade dictionary
            cvd: Current CVD value
        """
        try:
            with db_manager.get_session() as session:
                db_trade = Trade(
                    timestamp=trade['timestamp'],
                    exchange='coinbase',
                    symbol=self.product_id,
                    trade_id=str(trade['trade_id']),
                    price=Decimal(str(trade['price'])),
                    size=Decimal(str(trade['size'])),
                    side=trade['side'],
                    cvd_at_trade=Decimal(str(cvd))
                )
                session.add(db_trade)
        except Exception as e:
            logger.error(f"Error saving trade to database: {e}")

    def print_trade(self, trade: Dict[str, Any], cvd: float):
        """
        Print trade information to console.

        Args:
            trade: Parsed trade dictionary
            cvd: Current CVD value
        """
        side = "BUY " if trade['side'] == 'buy' else "SELL"
        side_color = "\033[92m" if trade['side'] == 'buy' else "\033[91m"  # Green for buy, red for sell
        reset_color = "\033[0m"

        print(f"{trade['timestamp'].strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]} | "
              f"{side_color}{side}{reset_color} | "
              f"Price: ${trade['price']:,.2f} | "
              f"Size: {trade['size']:.8f} BTC | "
              f"CVD: {cvd:+,.6f}")

    async def handle_message(self, message: str):
        """
        Handle incoming WebSocket message from Coinbase.

        Coinbase sends different message types:
        - Subscription confirmations (type: "subscriptions")
        - Trade data (type: "match")
        - Heartbeats (type: "heartbeat")

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
                # Coinbase sends periodic heartbeats
                return

            # Handle error messages
            if msg_type == 'error':
                logger.error(f"Error from Coinbase: {data.get('message')}")
                return

            # Handle trade data
            if msg_type == 'match':
                trade = self.parse_trade(data)
                if not trade:
                    return

                # Update CVD
                cvd = self.update_cvd(trade['size'], trade['side'])

                # Save trade to database
                self.save_trade(trade, cvd)

                # Print trade information
                self.print_trade(trade, cvd)

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

        return await self.connect()

    async def start(self):
        """Start the WebSocket collector with auto-reconnect."""
        self.is_running = True

        print(f"\n{'='*80}")
        print(f"BTC-USD Real-Time Trades Stream - Coinbase")
        print(f"{'='*80}")
        print(f"Cumulative Volume Delta (CVD) Tracker")
        print(f"  GREEN = Buy Orders (adds to CVD)")
        print(f"  RED   = Sell Orders (subtracts from CVD)")
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
        logger.info("Stopping CoinbaseTradesCollector...")
        self.is_running = False

        if self.websocket:
            # Unsubscribe
            unsubscribe_message = {
                "type": "unsubscribe",
                "product_ids": [self.product_id],
                "channels": ["matches"]
            }
            try:
                await self.websocket.send(json.dumps(unsubscribe_message))
                await asyncio.sleep(0.5)  # Give time for unsubscribe
            except:
                pass

            await self.websocket.close()
            logger.info("WebSocket connection closed")

        print(f"\n{'='*80}")
        print(f"Final CVD: {self.cumulative_volume_delta:+,.6f}")
        print(f"{'='*80}\n")


async def main():
    """Main function for testing the collector."""
    collector = CoinbaseTradesCollector()

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
