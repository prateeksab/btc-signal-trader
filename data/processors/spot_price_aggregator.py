#!/usr/bin/env python3
"""
Spot Price Aggregator

Periodically aggregates BTC spot prices from all available sources:
- Coinbase (latest trades)
- Kraken (latest trades)
- Binance (index price from funding rate metrics)
- Kalshi (already saved by kalshi_markets.py)

Saves to spot_price_snapshots table for cross-source comparison.
"""

import sys
import os
import time
import logging
from datetime import datetime, timedelta
from decimal import Decimal

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from data.storage.database import db_manager
from data.storage.models import Trade, FundingRateMetrics, SpotPriceSnapshot
from sqlalchemy import desc

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SpotPriceAggregator:
    """Aggregates spot prices from all sources"""

    def __init__(self, interval_seconds=60):
        """
        Initialize aggregator.

        Args:
            interval_seconds: How often to collect spot prices
        """
        self.interval = interval_seconds
        logger.info(f"SpotPriceAggregator initialized (interval: {interval_seconds}s)")

    def get_coinbase_spot_price(self):
        """Get latest Coinbase spot price from trades"""
        try:
            with db_manager.get_session() as session:
                latest_trade = session.query(Trade).filter(
                    Trade.exchange == 'coinbase',
                    Trade.symbol == 'BTC-USD'
                ).order_by(desc(Trade.timestamp)).first()

                if latest_trade:
                    # Check if trade is recent (within last 60 seconds)
                    age = (datetime.utcnow() - latest_trade.timestamp).total_seconds()
                    if age > 60:
                        logger.warning(f"Coinbase price is {age:.0f}s old")
                        return None

                    return {
                        'price': float(latest_trade.price),
                        'confidence': 95.0,  # High confidence - direct exchange data
                        'extra': {
                            'age_seconds': age,
                            'trade_id': latest_trade.trade_id
                        }
                    }

                logger.warning("No Coinbase trades found")
                return None

        except Exception as e:
            logger.error(f"Error fetching Coinbase price: {e}")
            return None

    def get_kraken_spot_price(self):
        """Get latest Kraken spot price from trades"""
        try:
            with db_manager.get_session() as session:
                latest_trade = session.query(Trade).filter(
                    Trade.exchange == 'kraken',
                    Trade.symbol == 'XBT-USD'
                ).order_by(desc(Trade.timestamp)).first()

                if latest_trade:
                    age = (datetime.utcnow() - latest_trade.timestamp).total_seconds()
                    if age > 60:
                        logger.warning(f"Kraken price is {age:.0f}s old")
                        return None

                    return {
                        'price': float(latest_trade.price),
                        'confidence': 95.0,
                        'extra': {
                            'age_seconds': age
                        }
                    }

                logger.warning("No Kraken trades found")
                return None

        except Exception as e:
            logger.error(f"Error fetching Kraken price: {e}")
            return None

    def get_binance_index_price(self):
        """Get latest Binance index (spot) price from funding rate metrics"""
        try:
            with db_manager.get_session() as session:
                latest_metric = session.query(FundingRateMetrics).order_by(
                    desc(FundingRateMetrics.timestamp)
                ).first()

                if latest_metric:
                    age = (datetime.utcnow() - latest_metric.timestamp).total_seconds()
                    if age > 60:
                        logger.warning(f"Binance index price is {age:.0f}s old")
                        return None

                    return {
                        'price': float(latest_metric.index_price),
                        'confidence': 90.0,  # Slightly lower - it's an index
                        'extra': {
                            'age_seconds': age,
                            'mark_price': float(latest_metric.mark_price),
                            'mark_premium': float(latest_metric.mark_premium)
                        }
                    }

                logger.warning("No Binance funding rate data found")
                return None

        except Exception as e:
            logger.error(f"Error fetching Binance index price: {e}")
            return None

    def save_spot_price(self, source, price_data, timestamp):
        """
        Save spot price to database.

        Args:
            source: Source name ('coinbase', 'kraken', 'binance_index')
            price_data: Dict with 'price', 'confidence', 'extra'
            timestamp: Timestamp for this snapshot
        """
        try:
            with db_manager.get_session() as session:
                snapshot = SpotPriceSnapshot(
                    timestamp=timestamp,
                    source=source,
                    price=Decimal(str(round(price_data['price'], 2))),
                    confidence=Decimal(str(price_data['confidence'])),
                    extra_data=price_data.get('extra')
                )
                session.add(snapshot)

            logger.debug(f"{source}: ${price_data['price']:,.2f}")

        except Exception as e:
            logger.error(f"Error saving {source} spot price: {e}")

    def collect_all_prices(self):
        """Collect prices from all sources and save to database"""
        timestamp = datetime.utcnow()

        prices = {}

        # Collect from all sources
        coinbase_data = self.get_coinbase_spot_price()
        if coinbase_data:
            prices['coinbase'] = coinbase_data
            self.save_spot_price('coinbase', coinbase_data, timestamp)

        kraken_data = self.get_kraken_spot_price()
        if kraken_data:
            prices['kraken'] = kraken_data
            self.save_spot_price('kraken', kraken_data, timestamp)

        binance_data = self.get_binance_index_price()
        if binance_data:
            prices['binance_index'] = binance_data
            self.save_spot_price('binance_index', binance_data, timestamp)

        # Log summary
        if prices:
            summary = " | ".join([
                f"{source}: ${data['price']:,.2f}"
                for source, data in prices.items()
            ])

            # Calculate spread
            if len(prices) > 1:
                price_values = [data['price'] for data in prices.values()]
                spread = max(price_values) - min(price_values)
                spread_pct = (spread / min(price_values)) * 100
                summary += f" | Spread: ${spread:.2f} ({spread_pct:.3f}%)"

            logger.info(summary)

            return prices
        else:
            logger.warning("No spot prices available from any source")
            return None

    def run(self):
        """Run continuous spot price aggregation"""
        logger.info(f"Starting spot price aggregation (every {self.interval}s)")
        logger.info("Press Ctrl+C to stop")

        try:
            while True:
                self.collect_all_prices()
                time.sleep(self.interval)

        except KeyboardInterrupt:
            logger.info("\nStopping aggregator...")


def main():
    """Main entry point"""
    aggregator = SpotPriceAggregator(interval_seconds=60)
    aggregator.run()


if __name__ == "__main__":
    main()
