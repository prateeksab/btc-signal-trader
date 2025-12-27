#!/usr/bin/env python3
"""
BTC Price Candle Aggregator

Aggregates trade data into OHLC candlesticks at various timeframes.
"""

import sys
import os
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Optional, List
import pandas as pd
from sqlalchemy import and_

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from data.storage.database import db_manager
from data.storage.models import Trade, BtcPriceCandle


class CandleAggregator:
    """
    Aggregates trade data into OHLC candlesticks.
    """

    # Timeframe to minutes mapping
    TIMEFRAME_MINUTES = {
        '1min': 1,
        '5min': 5,
        '15min': 15,
        '30min': 30,
        '1hour': 60,
        '1day': 1440,
        '1week': 10080
    }

    def __init__(self):
        pass

    def floor_timestamp(self, timestamp: datetime, minutes: int) -> datetime:
        """
        Floor timestamp to the start of a candle period.

        Args:
            timestamp: The timestamp to floor
            minutes: Candle period in minutes

        Returns:
            Floored timestamp
        """
        # Get total minutes since epoch
        epoch = datetime(1970, 1, 1, tzinfo=timezone.utc)
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)

        total_minutes = int((timestamp - epoch).total_seconds() / 60)

        # Floor to nearest period
        floored_minutes = (total_minutes // minutes) * minutes

        # Convert back to timestamp
        return epoch + timedelta(minutes=floored_minutes)

    def aggregate_trades_to_candles(
        self,
        exchange: str,
        symbol: str,
        timeframe: str,
        start_time: datetime,
        end_time: datetime
    ) -> List[BtcPriceCandle]:
        """
        Aggregate trades into OHLC candles.

        Args:
            exchange: Exchange name (e.g., 'coinbase')
            symbol: Symbol (e.g., 'BTC-USD')
            timeframe: Timeframe (e.g., '1min', '5min', '1hour')
            start_time: Start of period
            end_time: End of period

        Returns:
            List of BtcPriceCandle objects
        """
        if timeframe not in self.TIMEFRAME_MINUTES:
            raise ValueError(f"Invalid timeframe: {timeframe}")

        minutes = self.TIMEFRAME_MINUTES[timeframe]

        # Query trades
        with db_manager.get_session() as session:
            trades = session.query(Trade).filter(
                and_(
                    Trade.exchange == exchange,
                    Trade.symbol == symbol,
                    Trade.timestamp >= start_time,
                    Trade.timestamp < end_time
                )
            ).order_by(Trade.timestamp).all()

            if not trades:
                return []

            # Convert to DataFrame for easier aggregation
            df = pd.DataFrame([{
                'timestamp': t.timestamp,
                'price': float(t.price),
                'size': float(t.size)
            } for t in trades])

            # Add candle_start column (floored timestamp)
            df['candle_start'] = df['timestamp'].apply(
                lambda x: self.floor_timestamp(x.replace(tzinfo=None) if x.tzinfo else x, minutes)
            )

            # Remove timezone from candle_start for groupby
            df['candle_start'] = df['candle_start'].apply(lambda x: x.replace(tzinfo=None))

            # Group by candle periods and aggregate
            candles = []
            for candle_start, group in df.groupby('candle_start'):
                candle = BtcPriceCandle(
                    timestamp=candle_start,
                    timeframe=timeframe,
                    exchange=exchange,
                    symbol=symbol,
                    open=Decimal(str(group.iloc[0]['price'])),
                    high=Decimal(str(group['price'].max())),
                    low=Decimal(str(group['price'].min())),
                    close=Decimal(str(group.iloc[-1]['price'])),
                    volume=Decimal(str(group['size'].sum())),
                    num_trades=len(group)
                )
                candles.append(candle)

            return candles

    def save_candles(self, candles: List[BtcPriceCandle]) -> int:
        """
        Save candles to database.

        Args:
            candles: List of candles to save

        Returns:
            Number of candles saved
        """
        if not candles:
            return 0

        saved_count = 0
        with db_manager.get_session() as session:
            for candle in candles:
                try:
                    # Use merge to handle duplicates (upsert)
                    session.merge(candle)
                    saved_count += 1
                except Exception as e:
                    # Skip duplicates or errors
                    pass

            session.commit()

        return saved_count

    def aggregate_and_save(
        self,
        exchange: str = 'coinbase',
        symbol: str = 'BTC-USD',
        timeframes: List[str] = None,
        hours_back: int = 24
    ) -> dict:
        """
        Aggregate trades into candles for multiple timeframes and save.

        Args:
            exchange: Exchange name
            symbol: Trading symbol
            timeframes: List of timeframes to aggregate (default: all)
            hours_back: How many hours of historical data to aggregate

        Returns:
            Dictionary with counts per timeframe
        """
        if timeframes is None:
            timeframes = list(self.TIMEFRAME_MINUTES.keys())

        end_time = datetime.now(timezone.utc).replace(tzinfo=None)
        start_time = end_time - timedelta(hours=hours_back)

        results = {}
        for timeframe in timeframes:
            candles = self.aggregate_trades_to_candles(
                exchange=exchange,
                symbol=symbol,
                timeframe=timeframe,
                start_time=start_time,
                end_time=end_time
            )

            saved = self.save_candles(candles)
            results[timeframe] = saved
            print(f"✓ {timeframe}: Aggregated and saved {saved} candles")

        return results


def main():
    """Test the candle aggregator"""
    aggregator = CandleAggregator()

    print("Aggregating trades into candlesticks...")
    print("=" * 50)

    # Aggregate last 24 hours for all timeframes
    results = aggregator.aggregate_and_save(
        exchange='coinbase',
        symbol='BTC-USD',
        hours_back=72  # 3 days
    )

    print("=" * 50)
    print(f"Total candles created: {sum(results.values())}")


if __name__ == "__main__":
    main()
