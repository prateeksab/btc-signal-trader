"""
Binance BTC/USDT Real-Time Trades WebSocket Collector

This module connects to Binance's WebSocket stream to collect real-time
trade data and calculates Cumulative Volume Delta (CVD).
"""

import asyncio
import json
import logging
import signal
import sys
from datetime import datetime
from typing import Optional
import websockets
from websockets.exceptions import ConnectionClosed, WebSocketException


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('logs/binance_trades.log')
    ]
)
logger = logging.getLogger(__name__)


class BinanceTradesCollector:
    """
    WebSocket client for collecting real-time BTC/USDT trades from Binance.

    Calculates Cumulative Volume Delta (CVD):
    - Buyer maker (seller aggressive): subtract volume from CVD
    - Not buyer maker (buyer aggressive): add volume to CVD
    """

    def __init__(self, symbol: str = "btcusdt", websocket_url: Optional[str] = None):
        """
        Initialize the Binance trades collector.

        Args:
            symbol: Trading pair symbol (default: btcusdt)
            websocket_url: Custom WebSocket URL (optional)
        """
        self.symbol = symbol.lower()
        self.websocket_url = websocket_url or f"wss://stream.binance.us:9443/ws/{self.symbol}@trade"

        # CVD tracking
        self.cumulative_volume_delta = 0.0

        # Connection management
        self.websocket = None
        self.is_running = False
        self.reconnect_delay = 1  # Initial reconnect delay in seconds
        self.max_reconnect_delay = 60  # Maximum reconnect delay
        self.reconnect_attempts = 0

        logger.info(f"Initialized BinanceTradesCollector for {self.symbol.upper()}")

    async def connect(self):
        """Establish WebSocket connection to Binance."""
        try:
            self.websocket = await websockets.connect(self.websocket_url)
            logger.info(f"Connected to Binance WebSocket: {self.websocket_url}")

            # Reset reconnect parameters on successful connection
            self.reconnect_delay = 1
            self.reconnect_attempts = 0

            return True
        except Exception as e:
            logger.error(f"Failed to connect to WebSocket: {e}")
            return False

    def parse_trade(self, trade_data: dict) -> Optional[dict]:
        """
        Parse raw trade data from Binance WebSocket.

        Args:
            trade_data: Raw trade data from WebSocket

        Returns:
            Parsed trade dictionary or None if parsing fails
        """
        try:
            # Extract relevant fields
            trade = {
                'timestamp': datetime.fromtimestamp(trade_data['T'] / 1000),
                'price': float(trade_data['p']),
                'quantity': float(trade_data['q']),
                'is_buyer_maker': trade_data['m'],  # True if buyer is maker (seller aggressive)
                'trade_id': trade_data['t']
            }
            return trade
        except (KeyError, ValueError, TypeError) as e:
            logger.error(f"Error parsing trade data: {e}")
            return None

    def update_cvd(self, quantity: float, is_buyer_maker: bool) -> float:
        """
        Update Cumulative Volume Delta (CVD).

        Logic:
        - If buyer is maker (seller is aggressive): subtract volume
        - If seller is maker (buyer is aggressive): add volume

        Args:
            quantity: Trade quantity
            is_buyer_maker: True if buyer is the maker

        Returns:
            Updated CVD value
        """
        if is_buyer_maker:
            # Seller is aggressive - subtract volume
            self.cumulative_volume_delta -= quantity
        else:
            # Buyer is aggressive - add volume
            self.cumulative_volume_delta += quantity

        return self.cumulative_volume_delta

    def print_trade(self, trade: dict, cvd: float):
        """
        Print trade information to console.

        Args:
            trade: Parsed trade dictionary
            cvd: Current CVD value
        """
        side = "SELL" if trade['is_buyer_maker'] else "BUY "
        side_color = "\033[91m" if trade['is_buyer_maker'] else "\033[92m"  # Red for sell, green for buy
        reset_color = "\033[0m"

        print(f"{trade['timestamp'].strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]} | "
              f"{side_color}{side}{reset_color} | "
              f"Price: ${trade['price']:,.2f} | "
              f"Qty: {trade['quantity']:.6f} BTC | "
              f"CVD: {cvd:+,.2f}")

    async def handle_message(self, message: str):
        """
        Handle incoming WebSocket message.

        Args:
            message: Raw WebSocket message
        """
        try:
            data = json.loads(message)

            # Parse trade data
            trade = self.parse_trade(data)
            if not trade:
                return

            # Update CVD
            cvd = self.update_cvd(trade['quantity'], trade['is_buyer_maker'])

            # Print trade information
            self.print_trade(trade, cvd)

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
        print(f"BTC/USDT Real-Time Trades Stream - Binance")
        print(f"{'='*80}")
        print(f"Cumulative Volume Delta (CVD) Tracker")
        print(f"  GREEN = Buyer Aggressive (adds to CVD)")
        print(f"  RED   = Seller Aggressive (subtracts from CVD)")
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
        logger.info("Stopping BinanceTradesCollector...")
        self.is_running = False

        if self.websocket:
            await self.websocket.close()
            logger.info("WebSocket connection closed")

        print(f"\n{'='*80}")
        print(f"Final CVD: {self.cumulative_volume_delta:+,.2f}")
        print(f"{'='*80}\n")


async def main():
    """Main function for testing the collector."""
    collector = BinanceTradesCollector()

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
