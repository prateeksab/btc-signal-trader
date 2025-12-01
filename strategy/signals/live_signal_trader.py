"""
Live Signal Trading System

Combines real-time data from multiple sources and generates trading signals.

Data Sources:
- Coinbase Trades (CVD calculation)
- Kraken Orderbook (Imbalance calculation)
"""

import asyncio
import json
import logging
import signal as sys_signal
import sys
from datetime import datetime
from decimal import Decimal
import websockets
from websockets.exceptions import ConnectionClosed

# Import our signal generator
sys.path.append('/home/prateek/projects/btc-signal-trader')
from strategy.signals.signal_generator import SignalGenerator

# Import database
from data.storage.database import db_manager
from data.storage.models import (
    Trade, OrderbookSnapshot, CVDSnapshot, Signal
)


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class LiveSignalTrader:
    """
    Combines multiple WebSocket feeds and generates real-time signals.
    """

    def __init__(self):
        # Signal generator
        self.signal_generator = SignalGenerator()

        # Data tracking
        self.coinbase_cvd = 0.0
        self.kraken_imbalance = 0.0
        self.current_price = None

        # Kraken orderbook state
        self.kraken_bids = {}
        self.kraken_asks = {}

        # WebSocket connections
        self.coinbase_ws = None
        self.kraken_ws = None

        # Control
        self.is_running = False

        # Signal generation interval
        self.last_signal_time = datetime.now()
        self.signal_interval_seconds = 5  # Generate signal every 5 seconds

        # Database tracking
        self.trades_saved = 0
        self.orderbooks_saved = 0
        self.cvds_saved = 0
        self.signals_saved = 0
        self.last_cvd_snapshot_time = datetime.now()
        self.cvd_snapshot_interval = 10  # Save CVD snapshot every 10 seconds

        logger.info("LiveSignalTrader initialized")

    async def connect_coinbase(self):
        """Connect to Coinbase trades WebSocket"""
        try:
            self.coinbase_ws = await websockets.connect("wss://ws-feed.exchange.coinbase.com")
            subscribe_msg = {
                "type": "subscribe",
                "product_ids": ["BTC-USD"],
                "channels": ["matches"]
            }
            await self.coinbase_ws.send(json.dumps(subscribe_msg))
            logger.info("Connected to Coinbase WebSocket")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Coinbase: {e}")
            return False

    async def connect_kraken(self):
        """Connect to Kraken orderbook WebSocket"""
        try:
            self.kraken_ws = await websockets.connect("wss://ws.kraken.com/")
            subscribe_msg = {
                "event": "subscribe",
                "pair": ["XBT/USD"],
                "subscription": {"name": "book", "depth": 10}
            }
            await self.kraken_ws.send(json.dumps(subscribe_msg))
            logger.info("Connected to Kraken WebSocket")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Kraken: {e}")
            return False

    def handle_coinbase_trade(self, data: dict):
        """Process Coinbase trade and update CVD"""
        try:
            if data.get('type') != 'match':
                return

            size = float(data['size'])
            price = float(data['price'])
            side = data['side']
            trade_id = data.get('trade_id', '')
            timestamp = datetime.fromisoformat(data['time'].replace('Z', '+00:00'))

            # Update price
            self.current_price = price

            # Update CVD
            if side == 'buy':
                self.coinbase_cvd += size
            else:
                self.coinbase_cvd -= size

            # Save trade to database
            self.save_trade(timestamp, price, size, side, trade_id)

        except Exception as e:
            logger.error(f"Error handling Coinbase trade: {e}")

    def save_trade(self, timestamp, price, size, side, trade_id):
        """Save trade to database"""
        try:
            with db_manager.get_session() as session:
                trade = Trade(
                    timestamp=timestamp,
                    exchange='coinbase',
                    symbol='BTC-USD',
                    trade_id=str(trade_id),
                    price=Decimal(str(price)),
                    size=Decimal(str(size)),
                    side=side,
                    is_buyer_maker=(side == 'sell'),
                    cvd_at_trade=Decimal(str(self.coinbase_cvd))
                )
                session.add(trade)
                self.trades_saved += 1
        except Exception as e:
            logger.error(f"Error saving trade: {e}")

    def calculate_kraken_imbalance(self) -> float:
        """Calculate orderbook imbalance from Kraken data"""
        try:
            if not self.kraken_bids or not self.kraken_asks:
                return 0.0

            # Get top 5 levels
            bid_prices = sorted(self.kraken_bids.keys(), reverse=True)[:5]
            ask_prices = sorted(self.kraken_asks.keys())[:5]

            if not bid_prices or not ask_prices:
                return 0.0

            # Calculate volumes
            bid_volume = sum(self.kraken_bids[p] for p in bid_prices)
            ask_volume = sum(self.kraken_asks[p] for p in ask_prices)

            total_volume = bid_volume + ask_volume
            if total_volume == 0:
                return 0.0

            # Imbalance percentage
            imbalance = ((bid_volume - ask_volume) / total_volume) * 100
            return imbalance

        except Exception as e:
            logger.error(f"Error calculating imbalance: {e}")
            return 0.0

    def handle_kraken_orderbook(self, data: list):
        """Process Kraken orderbook update"""
        try:
            if len(data) < 4:
                return

            book_data = data[1]

            # Handle snapshot
            if 'as' in book_data and 'bs' in book_data:
                # Clear existing
                self.kraken_bids.clear()
                self.kraken_asks.clear()

                # Process asks
                for ask in book_data['as']:
                    price = float(ask[0])
                    volume = float(ask[1])
                    if volume > 0:
                        self.kraken_asks[price] = volume

                # Process bids
                for bid in book_data['bs']:
                    price = float(bid[0])
                    volume = float(bid[1])
                    if volume > 0:
                        self.kraken_bids[price] = volume

                # Update price if available
                if self.kraken_asks:
                    best_ask = min(self.kraken_asks.keys())
                    if not self.current_price:
                        self.current_price = best_ask

            # Handle updates
            else:
                # Process ask updates
                for ask in book_data.get('a', []):
                    price = float(ask[0])
                    volume = float(ask[1])
                    if volume == 0:
                        self.kraken_asks.pop(price, None)
                    else:
                        self.kraken_asks[price] = volume

                # Process bid updates
                for bid in book_data.get('b', []):
                    price = float(bid[0])
                    volume = float(bid[1])
                    if volume == 0:
                        self.kraken_bids.pop(price, None)
                    else:
                        self.kraken_bids[price] = volume

            # Update imbalance
            self.kraken_imbalance = self.calculate_kraken_imbalance()

            # Save orderbook snapshot (only on full snapshots to avoid too much data)
            if 'as' in book_data and 'bs' in book_data:
                self.save_orderbook_snapshot()

        except Exception as e:
            logger.error(f"Error handling Kraken orderbook: {e}")

    def save_orderbook_snapshot(self):
        """Save orderbook snapshot to database"""
        try:
            if not self.kraken_bids or not self.kraken_asks:
                return

            # Get best bid/ask
            best_bid = max(self.kraken_bids.keys()) if self.kraken_bids else 0
            best_ask = min(self.kraken_asks.keys()) if self.kraken_asks else 0
            spread = best_ask - best_bid if best_ask and best_bid else 0
            spread_pct = (spread / best_ask * 100) if best_ask else 0

            # Calculate total volumes
            total_bid_volume = sum(self.kraken_bids.values())
            total_ask_volume = sum(self.kraken_asks.values())

            # Prepare orderbook data for JSON storage
            bids_json = [[price, volume] for price, volume in sorted(self.kraken_bids.items(), reverse=True)[:10]]
            asks_json = [[price, volume] for price, volume in sorted(self.kraken_asks.items())[:10]]

            with db_manager.get_session() as session:
                snapshot = OrderbookSnapshot(
                    timestamp=datetime.utcnow(),
                    exchange='kraken',
                    symbol='XBT/USD',
                    best_bid=Decimal(str(best_bid)),
                    best_ask=Decimal(str(best_ask)),
                    spread=Decimal(str(spread)),
                    spread_pct=Decimal(str(spread_pct)),
                    total_bid_volume=Decimal(str(total_bid_volume)),
                    total_ask_volume=Decimal(str(total_ask_volume)),
                    imbalance=Decimal(str(self.kraken_imbalance)),
                    bids_json=bids_json,
                    asks_json=asks_json,
                    depth_levels=10
                )
                session.add(snapshot)
                self.orderbooks_saved += 1
        except Exception as e:
            logger.error(f"Error saving orderbook snapshot: {e}")

    async def listen_coinbase(self):
        """Listen to Coinbase WebSocket"""
        try:
            async for message in self.coinbase_ws:
                if not self.is_running:
                    break

                data = json.loads(message)
                self.handle_coinbase_trade(data)

        except ConnectionClosed:
            logger.warning("Coinbase connection closed")
        except Exception as e:
            logger.error(f"Error in Coinbase listener: {e}")

    async def listen_kraken(self):
        """Listen to Kraken WebSocket"""
        try:
            async for message in self.kraken_ws:
                if not self.is_running:
                    break

                data = json.loads(message)

                # Handle orderbook data
                if isinstance(data, list):
                    self.handle_kraken_orderbook(data)

        except ConnectionClosed:
            logger.warning("Kraken connection closed")
        except Exception as e:
            logger.error(f"Error in Kraken listener: {e}")

    async def generate_signals_loop(self):
        """Periodically generate trading signals"""
        while self.is_running:
            try:
                # Wait for interval
                await asyncio.sleep(self.signal_interval_seconds)

                # Save CVD snapshot periodically
                now = datetime.now()
                if (now - self.last_cvd_snapshot_time).total_seconds() >= self.cvd_snapshot_interval:
                    self.save_cvd_snapshot()
                    self.last_cvd_snapshot_time = now

                # Generate signal
                signal = self.signal_generator.generate_signal(
                    cvd=self.coinbase_cvd,
                    orderbook_imbalance=self.kraken_imbalance,
                    price=self.current_price
                )

                # Print signal
                self.signal_generator.print_signal(signal)

                # Save signal to database
                self.save_signal(signal)

            except Exception as e:
                logger.error(f"Error generating signal: {e}")

    def save_cvd_snapshot(self):
        """Save CVD snapshot to database"""
        try:
            with db_manager.get_session() as session:
                cvd_snapshot = CVDSnapshot(
                    timestamp=datetime.utcnow(),
                    exchange='coinbase',
                    symbol='BTC-USD',
                    cvd=Decimal(str(self.coinbase_cvd)),
                    cvd_change=Decimal('0'),  # Would need to track previous value
                )
                session.add(cvd_snapshot)
                self.cvds_saved += 1
        except Exception as e:
            logger.error(f"Error saving CVD snapshot: {e}")

    def save_signal(self, signal):
        """Save trading signal to database"""
        try:
            with db_manager.get_session() as session:
                db_signal = Signal(
                    timestamp=signal.timestamp,
                    signal_type=signal.signal_type.value,  # Extract enum value
                    confidence_level=signal.confidence.value,  # Extract enum value
                    confidence_score=Decimal(str(signal.confidence_score)),
                    price=Decimal(str(signal.price)) if signal.price else None,
                    cvd=Decimal(str(signal.cvd_value)) if signal.cvd_value is not None else None,
                    cvd_trend=signal.cvd_trend,
                    orderbook_imbalance=Decimal(str(signal.orderbook_imbalance)) if signal.orderbook_imbalance is not None else None,
                    reasons=signal.reasons if signal.reasons else [],
                    warnings=signal.warnings if signal.warnings else [],
                    exchange='coinbase',
                    symbol='BTC-USD',
                    strategy_name='CVD_Imbalance_Strategy',
                    extra_metadata={
                        'signal_interval': self.signal_interval_seconds,
                        'trades_saved': self.trades_saved,
                        'orderbooks_saved': self.orderbooks_saved
                    }
                )
                session.add(db_signal)
                self.signals_saved += 1
        except Exception as e:
            logger.error(f"Error saving signal: {e}")

    async def start(self):
        """Start the live signal trading system"""
        self.is_running = True

        print(f"\n{'='*80}")
        print(f"Live Trading Signal System")
        print(f"{'='*80}")
        print(f"Data Sources:")
        print(f"  - Coinbase: BTC-USD Trades (CVD)")
        print(f"  - Kraken:   XBT/USD Orderbook (Imbalance)")
        print(f"\nSignal Generation: Every {self.signal_interval_seconds} seconds")
        print(f"{'='*80}\n")

        # Connect to both sources
        coinbase_connected = await self.connect_coinbase()
        kraken_connected = await self.connect_kraken()

        if not coinbase_connected or not kraken_connected:
            logger.error("Failed to connect to data sources")
            return

        # Run all tasks concurrently
        try:
            await asyncio.gather(
                self.listen_coinbase(),
                self.listen_kraken(),
                self.generate_signals_loop()
            )
        except Exception as e:
            logger.error(f"Error in main loop: {e}")
        finally:
            await self.stop()

    async def stop(self):
        """Stop the trading system"""
        logger.info("Stopping LiveSignalTrader...")
        self.is_running = False

        if self.coinbase_ws:
            await self.coinbase_ws.close()
        if self.kraken_ws:
            await self.kraken_ws.close()

        print(f"\n{'='*80}")
        print(f"Trading System Stopped")
        print(f"{'='*80}")
        print(f"Final CVD: {self.coinbase_cvd:+,.4f}")
        print(f"Final Imbalance: {self.kraken_imbalance:+.2f}%")
        print(f"\nDatabase Statistics:")
        print(f"  📊 Trades saved:     {self.trades_saved}")
        print(f"  📊 Orderbooks saved: {self.orderbooks_saved}")
        print(f"  📊 CVDs saved:       {self.cvds_saved}")
        print(f"  📊 Signals saved:    {self.signals_saved}")
        print(f"{'='*80}\n")


async def main():
    """Main entry point"""
    trader = LiveSignalTrader()

    # Setup signal handlers
    def signal_handler(sig, frame):
        print("\n\nShutting down gracefully...")
        asyncio.create_task(trader.stop())

    sys_signal.signal(sys_signal.SIGINT, signal_handler)
    sys_signal.signal(sys_signal.SIGTERM, signal_handler)

    try:
        await trader.start()
    except KeyboardInterrupt:
        await trader.stop()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nShutdown complete.")
        sys.exit(0)
