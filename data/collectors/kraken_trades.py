"""
Kraken XBT/USD Real-Time Trades WebSocket Collector

This module connects to Kraken's WebSocket stream to collect real-time
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
        logging.FileHandler('logs/kraken_trades.log')
    ]
)
logger = logging.getLogger(__name__)


class KrakenTradesCollector:
    """
    WebSocket client for collecting real-time XBT/USD trades from Kraken.

    Calculates Cumulative Volume Delta (CVD):
    - Buy orders: add volume to CVD
    - Sell orders: subtract volume from CVD
    """

    def __init__(self, pair: str = "XBT/USD", websocket_url: str = "wss://ws.kraken.com/"):
        """
        Initialize the Kraken trades collector.

        Args:
            pair: Trading pair symbol (default: XBT/USD)
            websocket_url: Kraken WebSocket URL
        """
        self.pair = pair
        self.websocket_url = websocket_url

        # CVD tracking
        self.cumulative_volume_delta = 0.0

        # Connection management
        self.websocket = None
        self.is_running = False
        self.reconnect_delay = 1  # Initial reconnect delay in seconds
        self.max_reconnect_delay = 60  # Maximum reconnect delay
        self.reconnect_attempts = 0

        logger.info(f"Initialized KrakenTradesCollector for {self.pair}")

    async def connect(self):
        """Establish WebSocket connection to Kraken."""
        try:
            self.websocket = await websockets.connect(self.websocket_url)
            logger.info(f"Connected to Kraken WebSocket: {self.websocket_url}")

            # Subscribe to trades
            subscribe_message = {
                "event": "subscribe",
                "pair": [self.pair],
                "subscription": {"name": "trade"}
            }

            await self.websocket.send(json.dumps(subscribe_message))
            logger.info(f"Sent subscription request for {self.pair} trades")

            # Reset reconnect parameters on successful connection
            self.reconnect_delay = 1
            self.reconnect_attempts = 0

            return True
        except Exception as e:
            logger.error(f"Failed to connect to WebSocket: {e}")
            return False

    def parse_trade(self, trade_data: list) -> Optional[Dict[str, Any]]:
        """
        Parse raw trade data from Kraken WebSocket.

        Kraken trade format: [price, volume, time, side, orderType, misc]
        side: "b" for buy, "s" for sell

        Args:
            trade_data: Raw trade data array from Kraken

        Returns:
            Parsed trade dictionary or None if parsing fails
        """
        try:
            trade = {
                'price': float(trade_data[0]),
                'volume': float(trade_data[1]),
                'timestamp': datetime.fromtimestamp(float(trade_data[2])),
                'side': trade_data[3],  # 'b' for buy, 's' for sell
                'order_type': trade_data[4],  # 'm' for market, 'l' for limit
            }
            return trade
        except (IndexError, ValueError, TypeError) as e:
            logger.error(f"Error parsing trade data: {e}")
            return None

    def update_cvd(self, volume: float, side: str) -> float:
        """
        Update Cumulative Volume Delta (CVD).

        Logic:
        - Buy orders (side='b'): add volume
        - Sell orders (side='s'): subtract volume

        Args:
            volume: Trade volume
            side: Trade side ('b' for buy, 's' for sell)

        Returns:
            Updated CVD value
        """
        if side == 'b':
            # Buy - add volume
            self.cumulative_volume_delta += volume
        elif side == 's':
            # Sell - subtract volume
            self.cumulative_volume_delta -= volume

        return self.cumulative_volume_delta

    def save_trade(self, trade: Dict[str, Any], cvd: float):
        """
        Save trade to database.

        Args:
            trade: Parsed trade dictionary
            cvd: Current CVD value
        """
        try:
            # Normalize side: 'b' -> 'buy', 's' -> 'sell'
            side = 'buy' if trade['side'] == 'b' else 'sell'

            with db_manager.get_session() as session:
                db_trade = Trade(
                    timestamp=trade['timestamp'],
                    exchange='kraken',
                    symbol='XBT-USD',
                    price=Decimal(str(trade['price'])),
                    size=Decimal(str(trade['volume'])),
                    side=side,
                    is_buyer_maker=(trade['order_type'] == 'l'),  # limit orders are maker
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
        side = "BUY " if trade['side'] == 'b' else "SELL"
        side_color = "\033[92m" if trade['side'] == 'b' else "\033[91m"  # Green for buy, red for sell
        reset_color = "\033[0m"
        order_type = "MKT" if trade['order_type'] == 'm' else "LMT"

        print(f"{trade['timestamp'].strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]} | "
              f"{side_color}{side}{reset_color} | "
              f"{order_type} | "
              f"Price: ${trade['price']:,.2f} | "
              f"Vol: {trade['volume']:.6f} XBT | "
              f"CVD: {cvd:+,.4f}")

    async def handle_message(self, message: str):
        """
        Handle incoming WebSocket message from Kraken.

        Kraken sends different message types:
        - Subscription confirmations (dict with 'event' field)
        - Trade data (list)

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
                    # Kraken sends periodic heartbeats
                    pass
                elif data.get('event') == 'systemStatus':
                    logger.info(f"Kraken system status: {data.get('status')}")
                return

            # Handle trade data (comes as array)
            if isinstance(data, list) and len(data) >= 4:
                channel_name = data[-2]  # Second to last element is channel name

                if 'trade' in channel_name:
                    # Trade data is in data[1]
                    trades = data[1]

                    for trade_data in trades:
                        trade = self.parse_trade(trade_data)
                        if not trade:
                            continue

                        # Update CVD
                        cvd = self.update_cvd(trade['volume'], trade['side'])

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
        print(f"XBT/USD Real-Time Trades Stream - Kraken")
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
        logger.info("Stopping KrakenTradesCollector...")
        self.is_running = False

        if self.websocket:
            # Unsubscribe
            unsubscribe_message = {
                "event": "unsubscribe",
                "pair": [self.pair],
                "subscription": {"name": "trade"}
            }
            try:
                await self.websocket.send(json.dumps(unsubscribe_message))
                await asyncio.sleep(0.5)  # Give time for unsubscribe
            except:
                pass

            await self.websocket.close()
            logger.info("WebSocket connection closed")

        print(f"\n{'='*80}")
        print(f"Final CVD: {self.cumulative_volume_delta:+,.4f}")
        print(f"{'='*80}\n")


async def main():
    """Main function for testing the collector."""
    collector = KrakenTradesCollector()

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
