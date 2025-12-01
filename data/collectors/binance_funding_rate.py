#!/usr/bin/env python3
"""
Binance BTC Perpetual Funding Rate Tracker

Connects to Binance mark price WebSocket to track funding rate in real-time.
Updates every 3 seconds with:
- Current funding rate (positive = longs pay shorts, negative = shorts pay longs)
- Mark premium (mark price vs index price)
- Rate of change metrics (30s, 1min, 5min)
- Velocity and acceleration
- Pattern detection (spikes, drops, divergences)
"""

import asyncio
import json
import logging
import signal
import sys
import os
from datetime import datetime, timedelta
from typing import Optional, Deque
from decimal import Decimal
from collections import deque
import websockets
from websockets.exceptions import ConnectionClosed, WebSocketException

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from data.storage.database import db_manager
from data.storage.models import FundingRateMetrics


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('logs/binance_funding_rate.log')
    ]
)
logger = logging.getLogger(__name__)


class BinanceFundingRateTracker:
    """
    WebSocket client for tracking BTC perpetual funding rate on Binance.

    Tracks:
    - Funding rate (last settled rate)
    - Mark price vs Index price premium
    - Funding rate changes over multiple time windows
    - Velocity and acceleration of funding rate
    - Pattern detection for trading signals
    """

    def __init__(
        self,
        websocket_url: str = "wss://fstream.binance.com/ws/btcusdt@markPrice",
        save_interval: int = 3
    ):
        """
        Initialize the Binance funding rate tracker.

        Args:
            websocket_url: Binance futures mark price WebSocket URL
            save_interval: How often to save to database (seconds)
        """
        self.websocket_url = websocket_url
        self.save_interval = save_interval

        # Price and rate tracking
        self.latest_funding_rate = None
        self.latest_mark_price = None
        self.latest_index_price = None
        self.next_funding_time = None

        # History tracking for calculating changes
        # Store (timestamp, funding_rate, mark_premium) tuples
        self.rate_history: Deque = deque(maxlen=500)  # ~25 minutes at 3s intervals
        self.premium_history: Deque = deque(maxlen=500)

        # Velocity tracking (for acceleration calculation)
        self.velocity_history: Deque = deque(maxlen=100)

        # Connection management
        self.websocket = None
        self.is_running = False
        self.reconnect_delay = 1
        self.max_reconnect_delay = 60
        self.reconnect_attempts = 0

        # Last save time
        self.last_save_time = None

        logger.info("Initialized BinanceFundingRateTracker")

    async def connect(self):
        """Establish WebSocket connection to Binance."""
        try:
            self.websocket = await websockets.connect(self.websocket_url)
            logger.info(f"Connected to Binance Mark Price WebSocket: {self.websocket_url}")

            # Reset reconnect parameters
            self.reconnect_delay = 1
            self.reconnect_attempts = 0

            return True
        except Exception as e:
            logger.error(f"Failed to connect to WebSocket: {e}")
            return False

    def parse_mark_price_update(self, data: dict) -> Optional[dict]:
        """
        Parse markPriceUpdate message from Binance.

        Message format:
        {
            "e": "markPriceUpdate",
            "s": "BTCUSDT",
            "p": "95123.45",     # mark price
            "i": "95100.00",     # index (spot) price
            "r": "0.00010000",   # funding rate
            "T": 1234567890000   # next funding time (ms)
        }

        Args:
            data: Raw mark price data from Binance

        Returns:
            Parsed data dict or None if parsing fails
        """
        try:
            if data.get('e') == 'markPriceUpdate':
                funding_rate = float(data['r'])
                mark_price = float(data['p'])
                index_price = float(data['i'])
                next_funding_time = datetime.fromtimestamp(int(data['T']) / 1000)

                # Update latest values
                self.latest_funding_rate = funding_rate
                self.latest_mark_price = mark_price
                self.latest_index_price = index_price
                self.next_funding_time = next_funding_time

                return {
                    'funding_rate': funding_rate,
                    'mark_price': mark_price,
                    'index_price': index_price,
                    'next_funding_time': next_funding_time
                }
            return None
        except (KeyError, ValueError, TypeError) as e:
            logger.error(f"Error parsing mark price data: {e}")
            return None

    def calculate_mark_premium(self, mark_price: float, index_price: float) -> float:
        """
        Calculate mark premium percentage.

        Args:
            mark_price: Mark price
            index_price: Index (spot) price

        Returns:
            Premium as percentage
        """
        if index_price <= 0:
            return 0.0
        return ((mark_price - index_price) / index_price) * 100

    def get_rate_change(self, current_rate: float, seconds: int) -> Optional[float]:
        """
        Calculate funding rate change over the last N seconds.

        Args:
            current_rate: Current funding rate
            seconds: Time window in seconds

        Returns:
            Rate change or None if insufficient data
        """
        if not self.rate_history:
            return None

        # Find rate from N seconds ago
        cutoff_time = datetime.utcnow() - timedelta(seconds=seconds)

        # Find the oldest reading within our time window
        for timestamp, rate, _ in self.rate_history:
            if timestamp <= cutoff_time:
                return current_rate - rate

        # Not enough history yet
        return None

    def get_premium_change(self, current_premium: float, seconds: int) -> Optional[float]:
        """
        Calculate mark premium change over the last N seconds.

        Args:
            current_premium: Current mark premium
            seconds: Time window in seconds

        Returns:
            Premium change or None if insufficient data
        """
        if not self.premium_history:
            return None

        cutoff_time = datetime.utcnow() - timedelta(seconds=seconds)

        for timestamp, premium in self.premium_history:
            if timestamp <= cutoff_time:
                return current_premium - premium

        return None

    def calculate_velocity(self, current_rate: float) -> Optional[float]:
        """
        Calculate funding rate velocity (rate of change per minute).

        Args:
            current_rate: Current funding rate

        Returns:
            Velocity (rate change per minute) or None if insufficient data
        """
        # Use 1-minute change as velocity
        change_1min = self.get_rate_change(current_rate, 60)
        if change_1min is not None:
            return change_1min  # Already per-minute
        return None

    def calculate_acceleration(self, current_velocity: float) -> Optional[float]:
        """
        Calculate funding rate acceleration (2nd derivative).

        Args:
            current_velocity: Current velocity

        Returns:
            Acceleration or None if insufficient data
        """
        if not self.velocity_history or current_velocity is None:
            return None

        # Get velocity from 1 minute ago
        cutoff_time = datetime.utcnow() - timedelta(seconds=60)

        for timestamp, velocity in self.velocity_history:
            if timestamp <= cutoff_time:
                # Acceleration = change in velocity per minute
                return current_velocity - velocity

        return None

    def detect_patterns(
        self,
        rate: float,
        change_1min: Optional[float],
        velocity: Optional[float],
        premium: float
    ) -> list:
        """
        Detect significant patterns in funding rate.

        Args:
            rate: Current funding rate
            change_1min: 1-minute change
            velocity: Current velocity
            premium: Mark premium

        Returns:
            List of detected pattern strings
        """
        patterns = []

        if change_1min is not None:
            # Spike detection (>50% increase in 1 minute)
            if rate != 0 and abs(change_1min / rate) > 0.5:
                if change_1min > 0:
                    patterns.append("⚠️  SPIKE: Funding rate jumped >50% in 1min (strong long pressure)")
                else:
                    patterns.append("⚠️  DROP: Funding rate dropped >50% in 1min (long unwinding)")

            # Significant changes (>20%)
            elif rate != 0 and abs(change_1min / rate) > 0.2:
                if change_1min > 0:
                    patterns.append("↗️  Rising funding rate (increasing long pressure)")
                else:
                    patterns.append("↘️  Falling funding rate (decreasing long pressure)")

        # Extreme funding rates
        if rate > 0.0001:  # 0.01% per 8 hours = high
            patterns.append("🔴 HIGH funding rate (longs paying shorts - potential reversal)")
        elif rate < -0.0001:
            patterns.append("🟢 NEGATIVE funding rate (shorts paying longs - bullish signal)")

        # Divergence: premium rising but rate not keeping up (or vice versa)
        # This could indicate upcoming funding rate adjustment
        if abs(premium) > 0.05 and rate != 0:
            premium_to_rate_ratio = abs(premium) / abs(rate * 100)
            if premium_to_rate_ratio > 100:
                patterns.append("⚡ DIVERGENCE: Mark premium high but funding rate low (squeeze potential)")

        return patterns

    def save_metrics(
        self,
        funding_rate: float,
        mark_price: float,
        index_price: float,
        next_funding_time: datetime
    ):
        """
        Calculate all metrics and save to database.

        Args:
            funding_rate: Current funding rate
            mark_price: Mark price
            index_price: Index price
            next_funding_time: Next funding settlement time
        """
        try:
            now = datetime.utcnow()

            # Calculate mark premium
            mark_premium = self.calculate_mark_premium(mark_price, index_price)

            # Store in history
            self.rate_history.append((now, funding_rate, mark_premium))
            self.premium_history.append((now, mark_premium))

            # Calculate changes
            rate_change_30s = self.get_rate_change(funding_rate, 30)
            rate_change_1min = self.get_rate_change(funding_rate, 60)
            rate_change_5min = self.get_rate_change(funding_rate, 300)

            premium_change_1min = self.get_premium_change(mark_premium, 60)

            # Calculate velocity and acceleration
            velocity = self.calculate_velocity(funding_rate)
            if velocity is not None:
                self.velocity_history.append((now, velocity))

            acceleration = self.calculate_acceleration(velocity)

            # Detect patterns
            patterns = self.detect_patterns(funding_rate, rate_change_1min, velocity, mark_premium)

            # Save to database
            with db_manager.get_session() as session:
                metrics = FundingRateMetrics(
                    timestamp=now,
                    funding_rate=Decimal(str(funding_rate)),
                    mark_price=Decimal(str(mark_price)),
                    index_price=Decimal(str(index_price)),
                    mark_premium=Decimal(str(round(mark_premium, 6))) if mark_premium else None,
                    mark_premium_change_1min=Decimal(str(round(premium_change_1min, 6))) if premium_change_1min is not None else None,
                    funding_rate_change_30s=Decimal(str(rate_change_30s)) if rate_change_30s is not None else None,
                    funding_rate_change_1min=Decimal(str(rate_change_1min)) if rate_change_1min is not None else None,
                    funding_rate_change_5min=Decimal(str(rate_change_5min)) if rate_change_5min is not None else None,
                    funding_rate_velocity=Decimal(str(velocity)) if velocity is not None else None,
                    funding_rate_acceleration=Decimal(str(acceleration)) if acceleration is not None else None,
                    next_funding_time=next_funding_time
                )
                session.add(metrics)

            # Build log message
            log_parts = [
                f"Rate: {funding_rate:.8f}",
                f"Premium: {mark_premium:+.4f}%",
            ]

            if rate_change_1min is not None:
                log_parts.append(f"1m Δ: {rate_change_1min:+.8f}")

            if velocity is not None:
                log_parts.append(f"Vel: {velocity:+.8f}/min")

            logger.info(" | ".join(log_parts))

            # Log patterns
            for pattern in patterns:
                logger.warning(pattern)

        except Exception as e:
            logger.error(f"Error saving metrics to database: {e}")

    async def handle_message(self, message: str):
        """
        Handle incoming WebSocket message from Binance.

        Args:
            message: Raw WebSocket message
        """
        try:
            data = json.loads(message)
            parsed = self.parse_mark_price_update(data)

            if parsed:
                # Check if it's time to save (every N seconds)
                now = datetime.utcnow()
                if (self.last_save_time is None or
                    (now - self.last_save_time).total_seconds() >= self.save_interval):

                    self.save_metrics(
                        parsed['funding_rate'],
                        parsed['mark_price'],
                        parsed['index_price'],
                        parsed['next_funding_time']
                    )
                    self.last_save_time = now

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
        print(f"Binance BTC Perpetual Funding Rate Tracker")
        print(f"{'='*80}")
        print(f"Symbol: BTCUSDT Perpetual")
        print(f"Update frequency: Every ~3 seconds")
        print(f"Save interval: {self.save_interval}s")
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
        logger.info("Stopping BinanceFundingRateTracker...")
        self.is_running = False

        if self.websocket:
            await self.websocket.close()
            logger.info("WebSocket connection closed")

        print(f"\n{'='*80}")
        print(f"Binance Funding Rate Tracker Stopped")
        print(f"{'='*80}\n")


async def main():
    """Main function for testing the collector."""
    tracker = BinanceFundingRateTracker()

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
