#!/usr/bin/env python3
"""
Trade Intensity Analyzer

Calculates real-time trade flow metrics including:
- Trade frequency (TPS) across multiple time windows
- Average trade size
- Buy/sell pressure ratios
- Market aggression scores
- Velocity changes (acceleration/deceleration)
"""

import sys
import os
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from data.storage.database import db_manager
from data.storage.models import Trade, TradeIntensity
from sqlalchemy import and_


class TradeIntensityAnalyzer:
    """Analyzes trade flow and intensity metrics"""

    def __init__(self, exchange='coinbase', symbol='BTC-USD'):
        self.exchange = exchange
        self.symbol = symbol
        self.previous_tps_60s = None  # For velocity calculation

    def get_recent_trades(self, seconds: int) -> List[Trade]:
        """Get trades from the last N seconds"""
        with db_manager.get_session() as session:
            cutoff = datetime.utcnow() - timedelta(seconds=seconds)
            trades = session.query(Trade).filter(
                and_(
                    Trade.exchange == self.exchange,
                    Trade.symbol == self.symbol,
                    Trade.timestamp >= cutoff
                )
            ).order_by(Trade.timestamp).all()

            # Detach from session
            return [
                {
                    'timestamp': t.timestamp,
                    'price': float(t.price),
                    'size': float(t.size),
                    'side': t.side
                }
                for t in trades
            ]

    def calculate_trades_per_second(self, trades: List[Dict], window_seconds: int) -> float:
        """Calculate trades per second for a time window"""
        if not trades:
            return 0.0

        if len(trades) < 2:
            return 0.0

        # Use actual time span
        time_span = (trades[-1]['timestamp'] - trades[0]['timestamp']).total_seconds()
        if time_span <= 0:
            return 0.0

        return len(trades) / time_span

    def calculate_avg_trade_size(self, trades: List[Dict]) -> float:
        """Calculate average trade size"""
        if not trades:
            return 0.0

        total_size = sum(t['size'] for t in trades)
        return total_size / len(trades)

    def calculate_buy_sell_ratio(self, trades: List[Dict]) -> float:
        """Calculate buy/sell ratio (0-1 scale, % of volume from buys)"""
        if not trades:
            return 0.5  # Neutral

        buy_volume = sum(t['size'] for t in trades if t['side'] == 'buy')
        sell_volume = sum(t['size'] for t in trades if t['side'] == 'sell')

        total_volume = buy_volume + sell_volume
        if total_volume == 0:
            return 0.5

        return buy_volume / total_volume

    def calculate_aggression_score(self, buy_sell_ratio: float) -> float:
        """
        Calculate market aggression score (-1 to 1)

        > 0.6: Aggressive buying (0 to 1)
        < 0.4: Aggressive selling (-1 to 0)
        0.4-0.6: Neutral (0)
        """
        if buy_sell_ratio > 0.6:
            # Aggressive buying: scale from 0 to 1
            return (buy_sell_ratio - 0.5) * 2
        elif buy_sell_ratio < 0.4:
            # Aggressive selling: scale from -1 to 0
            return (0.5 - buy_sell_ratio) * -2
        else:
            # Neutral zone
            return 0.0

    def calculate_velocity_change(self, current_tps: float) -> Optional[float]:
        """
        Calculate velocity change (is trading accelerating or decelerating?)

        Returns:
            - Positive: Accelerating
            - Negative: Decelerating
            - None: First measurement
        """
        if self.previous_tps_60s is None:
            self.previous_tps_60s = current_tps
            return 0.0

        if self.previous_tps_60s == 0:
            velocity_change = 0.0
        else:
            # Calculate % change
            velocity_change = (current_tps - self.previous_tps_60s) / self.previous_tps_60s

        self.previous_tps_60s = current_tps
        return velocity_change

    def calculate_intensity(self) -> Dict:
        """
        Calculate all intensity metrics

        Returns:
            Dict with all calculated metrics
        """
        # Fetch trades for different windows
        trades_60s = self.get_recent_trades(60)
        trades_30s = [t for t in trades_60s if t['timestamp'] >= datetime.utcnow() - timedelta(seconds=30)]
        trades_10s = [t for t in trades_60s if t['timestamp'] >= datetime.utcnow() - timedelta(seconds=10)]

        # Calculate trades per second
        tps_10s = self.calculate_trades_per_second(trades_10s, 10)
        tps_30s = self.calculate_trades_per_second(trades_30s, 30)
        tps_60s = self.calculate_trades_per_second(trades_60s, 60)

        # Calculate average trade sizes
        avg_size_30s = self.calculate_avg_trade_size(trades_30s)
        avg_size_60s = self.calculate_avg_trade_size(trades_60s)

        # Calculate buy/sell metrics
        buy_sell_ratio = self.calculate_buy_sell_ratio(trades_60s)
        aggression_score = self.calculate_aggression_score(buy_sell_ratio)

        # Calculate velocity change
        velocity_change = self.calculate_velocity_change(tps_60s)

        return {
            'timestamp': datetime.utcnow(),
            'exchange': self.exchange,
            'symbol': self.symbol,
            'trades_per_sec_10s': Decimal(str(round(tps_10s, 4))),
            'trades_per_sec_30s': Decimal(str(round(tps_30s, 4))),
            'trades_per_sec_60s': Decimal(str(round(tps_60s, 4))),
            'avg_trade_size_30s': Decimal(str(round(avg_size_30s, 8))),
            'avg_trade_size_60s': Decimal(str(round(avg_size_60s, 8))),
            'buy_sell_ratio': Decimal(str(round(buy_sell_ratio, 4))),
            'aggression_score': Decimal(str(round(aggression_score, 4))),
            'velocity_change': Decimal(str(round(velocity_change, 4))) if velocity_change is not None else None,
        }

    def save_intensity(self, metrics: Dict):
        """Save intensity metrics to database"""
        with db_manager.get_session() as session:
            intensity = TradeIntensity(**metrics)
            session.add(intensity)

    def run_once(self):
        """Calculate and save intensity metrics"""
        metrics = self.calculate_intensity()
        self.save_intensity(metrics)
        return metrics


def main():
    """Test function - loads recent trades and prints current intensity"""
    print("\n" + "=" * 80)
    print("TRADE INTENSITY ANALYZER - TEST")
    print("=" * 80)

    analyzer = TradeIntensityAnalyzer()
    metrics = analyzer.calculate_intensity()

    print(f"\nTimestamp: {metrics['timestamp']}")
    print(f"Exchange: {metrics['exchange']}")
    print(f"Symbol: {metrics['symbol']}")
    print("\n" + "-" * 80)
    print("TRADE FREQUENCY")
    print("-" * 80)
    print(f"  10s window: {metrics['trades_per_sec_10s']:.2f} TPS")
    print(f"  30s window: {metrics['trades_per_sec_30s']:.2f} TPS")
    print(f"  60s window: {metrics['trades_per_sec_60s']:.2f} TPS")

    print("\n" + "-" * 80)
    print("TRADE SIZE")
    print("-" * 80)
    print(f"  30s avg: {metrics['avg_trade_size_30s']:.4f} BTC")
    print(f"  60s avg: {metrics['avg_trade_size_60s']:.4f} BTC")

    print("\n" + "-" * 80)
    print("BUY/SELL PRESSURE")
    print("-" * 80)
    print(f"  Buy/Sell Ratio: {metrics['buy_sell_ratio']:.2f} ({float(metrics['buy_sell_ratio'])*100:.1f}% buys)")
    print(f"  Aggression Score: {metrics['aggression_score']:.2f}", end="")

    if float(metrics['aggression_score']) > 0.3:
        print(" (AGGRESSIVE BUYING)")
    elif float(metrics['aggression_score']) < -0.3:
        print(" (AGGRESSIVE SELLING)")
    else:
        print(" (NEUTRAL)")

    print("\n" + "-" * 80)
    print("VELOCITY")
    print("-" * 80)
    if metrics['velocity_change'] is not None:
        vc = float(metrics['velocity_change'])
        print(f"  Velocity Change: {vc:+.2%}", end="")
        if vc > 0.1:
            print(" (ACCELERATING)")
        elif vc < -0.1:
            print(" (DECELERATING)")
        else:
            print(" (STABLE)")
    else:
        print("  Velocity Change: N/A (first measurement)")

    print("\n" + "=" * 80)

    # Save to database
    print("\nSaving to database...")
    analyzer.save_intensity(metrics)
    print("✅ Saved!")


if __name__ == "__main__":
    main()
