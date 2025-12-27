#!/usr/bin/env python3
"""
Master Collector & Signal Aggregator

Starts all data collectors and provides real-time market analysis
combining all data sources into actionable signals.
"""

import sys
import os
import time
import subprocess
import signal as sig
from datetime import datetime, timedelta
from decimal import Decimal

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from data.storage.database import db_manager
from data.storage.models import (
    Trade, TradeIntensity, FuturesBasis,
    OrderbookSnapshot, CVDSnapshot, PredictionMarket
)
from sqlalchemy import desc, func


class CollectorManager:
    """Manages all data collector processes"""

    COLLECTORS = [
        ('Binance Trades', 'venv/bin/python data/collectors/binance_trades.py'),
        ('Coinbase Trades', 'venv/bin/python data/collectors/coinbase_trades.py'),
        ('Kraken Trades', 'venv/bin/python data/collectors/kraken_trades.py'),
        ('Coinbase Orderbook', 'venv/bin/python data/collectors/coinbase_orderbook.py'),
        ('Kraken Orderbook', 'venv/bin/python data/collectors/kraken_orderbook.py'),
        ('Binance Perpetual', 'venv/bin/python data/collectors/binance_perpetual.py'),
        ('Coinbase Perpetual', 'venv/bin/python data/collectors/coinbase_perpetual.py'),
        ('Binance Funding Rate', 'venv/bin/python data/collectors/binance_funding_rate.py'),
        ('Kalshi Orderbook', 'venv/bin/python data/collectors/kalshi_orderbook.py --continuous 5'),
        ('Trade Intensity', 'venv/bin/python data/processors/collector_service.py'),
    ]

    def __init__(self):
        self.processes = []

    def start_all(self):
        """Start all collector processes"""
        print("\n" + "="*80)
        print("STARTING ALL COLLECTORS")
        print("="*80)

        # Kill any existing collectors
        subprocess.run(['pkill', '-f', 'data/collectors'], stderr=subprocess.DEVNULL)
        subprocess.run(['pkill', '-f', 'data/processors'], stderr=subprocess.DEVNULL)
        time.sleep(1)

        for name, cmd in self.COLLECTORS:
            try:
                log_file = f"logs/{name.lower().replace(' ', '_')}.log"
                with open(log_file, 'w') as f:
                    proc = subprocess.Popen(
                        cmd.split(),
                        stdout=f,
                        stderr=subprocess.STDOUT,
                        start_new_session=True
                    )
                self.processes.append(proc)
                print(f"✅ Started: {name}")
                time.sleep(0.5)
            except Exception as e:
                print(f"❌ Failed to start {name}: {e}")

        print(f"\n✅ All collectors started!")
        print("="*80 + "\n")

    def stop_all(self):
        """Stop all collector processes"""
        print("\n" + "="*80)
        print("STOPPING ALL COLLECTORS")
        print("="*80)

        subprocess.run(['pkill', '-f', 'data/collectors'])
        subprocess.run(['pkill', '-f', 'data/processors'])

        print("✅ All collectors stopped")
        print("="*80 + "\n")


class SignalAggregator:
    """Aggregates signals from all data sources"""

    def __init__(self):
        self.signals = {}
        self.confidence = 0.0
        self.recommendation = "NEUTRAL"

    def get_trade_intensity_signal(self):
        """Analyze trade intensity metrics"""
        try:
            with db_manager.get_session() as session:
                latest = session.query(TradeIntensity).order_by(
                    desc(TradeIntensity.timestamp)
                ).first()

                if not latest:
                    return None, "No data"

                age = (datetime.utcnow() - latest.timestamp).total_seconds()
                if age > 120:  # Data older than 2 minutes
                    return None, f"Stale ({age:.0f}s old)"

                aggression = float(latest.aggression_score)
                velocity = float(latest.velocity_change) if latest.velocity_change else 0
                tps = float(latest.trades_per_sec_60s)

                # Signal logic
                if aggression > 0.3 and velocity > 0.1:
                    signal = "BULLISH"
                    confidence = min(abs(aggression) + abs(velocity), 1.0)
                elif aggression < -0.3 and velocity < -0.1:
                    signal = "BEARISH"
                    confidence = min(abs(aggression) + abs(velocity), 1.0)
                elif aggression > 0.2:
                    signal = "BULLISH"
                    confidence = abs(aggression) * 0.7
                elif aggression < -0.2:
                    signal = "BEARISH"
                    confidence = abs(aggression) * 0.7
                else:
                    signal = "NEUTRAL"
                    confidence = 0.3

                return {
                    'signal': signal,
                    'confidence': confidence,
                    'aggression': aggression,
                    'velocity': velocity,
                    'tps': tps
                }, None

        except Exception as e:
            return None, str(e)

    def get_futures_basis_signal(self):
        """Analyze futures-spot basis"""
        try:
            with db_manager.get_session() as session:
                latest = session.query(FuturesBasis).order_by(
                    desc(FuturesBasis.timestamp)
                ).first()

                if not latest:
                    return None, "No data"

                age = (datetime.utcnow() - latest.timestamp).total_seconds()
                if age > 120:
                    return None, f"Stale ({age:.0f}s old)"

                basis_pct = float(latest.basis_pct)

                # Signal logic
                if basis_pct > 0.05:  # >0.05% premium
                    signal = "BULLISH"
                    confidence = min(basis_pct * 10, 0.8)
                elif basis_pct < -0.05:  # <-0.05% discount
                    signal = "BEARISH"
                    confidence = min(abs(basis_pct) * 10, 0.8)
                else:
                    signal = "NEUTRAL"
                    confidence = 0.3

                return {
                    'signal': signal,
                    'confidence': confidence,
                    'basis_pct': basis_pct,
                    'spot': float(latest.spot_price),
                    'futures': float(latest.futures_price)
                }, None

        except Exception as e:
            return None, str(e)

    def get_orderbook_signal(self):
        """Analyze orderbook imbalance"""
        try:
            with db_manager.get_session() as session:
                latest = session.query(OrderbookSnapshot).order_by(
                    desc(OrderbookSnapshot.timestamp)
                ).first()

                if not latest:
                    return None, "No data"

                age = (datetime.utcnow() - latest.timestamp).total_seconds()
                if age > 120:
                    return None, f"Stale ({age:.0f}s old)"

                imbalance = float(latest.imbalance) if latest.imbalance else 0

                # Signal logic
                if imbalance > 0.6:
                    signal = "BULLISH"
                    confidence = min((imbalance - 0.5) * 2, 0.7)
                elif imbalance < 0.4:
                    signal = "BEARISH"
                    confidence = min((0.5 - imbalance) * 2, 0.7)
                else:
                    signal = "NEUTRAL"
                    confidence = 0.3

                return {
                    'signal': signal,
                    'confidence': confidence,
                    'imbalance': imbalance,
                    'spread_pct': float(latest.spread_pct) if latest.spread_pct else 0
                }, None

        except Exception as e:
            return None, str(e)

    def get_kalshi_signal(self):
        """Analyze Kalshi prediction market"""
        try:
            with db_manager.get_session() as session:
                # Get latest snapshot
                latest_time = session.query(func.max(PredictionMarket.timestamp)).scalar()
                if not latest_time:
                    return None, "No data"

                age = (datetime.utcnow() - latest_time).total_seconds()
                if age > 120:
                    return None, f"Stale ({age:.0f}s old)"

                # Get current BTC price
                latest_trade = session.query(Trade).filter(
                    Trade.exchange == 'coinbase'
                ).order_by(desc(Trade.timestamp)).first()

                if not latest_trade:
                    return None, "No price data"

                current_price = float(latest_trade.price)

                # Get markets near current price
                markets = session.query(PredictionMarket).filter(
                    PredictionMarket.timestamp == latest_time
                ).all()

                if not markets:
                    return None, "No markets"

                # Find markets above and below current price
                above_markets = [m for m in markets if m.strike_price and float(m.strike_price) > current_price]
                below_markets = [m for m in markets if m.strike_price and float(m.strike_price) <= current_price]

                # Get probability of going up
                prob_up = 0
                if above_markets:
                    closest_above = min(above_markets, key=lambda m: float(m.strike_price))
                    prob_up = float(closest_above.implied_probability) if closest_above.implied_probability else 50

                # Signal logic
                if prob_up > 60:
                    signal = "BULLISH"
                    confidence = min((prob_up - 50) / 50, 0.6)
                elif prob_up < 40:
                    signal = "BEARISH"
                    confidence = min((50 - prob_up) / 50, 0.6)
                else:
                    signal = "NEUTRAL"
                    confidence = 0.3

                return {
                    'signal': signal,
                    'confidence': confidence,
                    'prob_up': prob_up,
                    'current_price': current_price
                }, None

        except Exception as e:
            return None, str(e)

    def aggregate_signals(self):
        """Combine all signals into overall recommendation"""
        signals = {}

        # Collect all signals
        intensity_data, intensity_err = self.get_trade_intensity_signal()
        if intensity_data:
            signals['intensity'] = intensity_data

        basis_data, basis_err = self.get_futures_basis_signal()
        if basis_data:
            signals['basis'] = basis_data

        orderbook_data, orderbook_err = self.get_orderbook_signal()
        if orderbook_data:
            signals['orderbook'] = orderbook_data

        kalshi_data, kalshi_err = self.get_kalshi_signal()
        if kalshi_data:
            signals['kalshi'] = kalshi_data

        if not signals:
            return "NEUTRAL", 0.0, {}, {
                'intensity': intensity_err,
                'basis': basis_err,
                'orderbook': orderbook_err,
                'kalshi': kalshi_err
            }

        # Weight signals
        weights = {
            'intensity': 0.35,  # Highest weight - direct market action
            'basis': 0.30,      # Strong signal from derivatives
            'orderbook': 0.20,  # Medium weight
            'kalshi': 0.15      # Prediction market
        }

        # Calculate weighted scores
        bullish_score = 0
        bearish_score = 0
        total_weight = 0

        for source, data in signals.items():
            weight = weights.get(source, 0)
            confidence = data['confidence']

            if data['signal'] == 'BULLISH':
                bullish_score += weight * confidence
            elif data['signal'] == 'BEARISH':
                bearish_score += weight * confidence

            total_weight += weight

        # Normalize scores
        if total_weight > 0:
            bullish_score /= total_weight
            bearish_score /= total_weight

        # Determine overall signal
        if bullish_score > bearish_score + 0.15:
            recommendation = "BULLISH"
            confidence = bullish_score
        elif bearish_score > bullish_score + 0.15:
            recommendation = "BEARISH"
            confidence = bearish_score
        else:
            recommendation = "NEUTRAL"
            confidence = 0.5 - abs(bullish_score - bearish_score)

        errors = {
            'intensity': intensity_err,
            'basis': basis_err,
            'orderbook': orderbook_err,
            'kalshi': kalshi_err
        }

        return recommendation, confidence, signals, errors

    def print_analysis(self):
        """Print formatted analysis to terminal"""
        recommendation, confidence, signals, errors = self.aggregate_signals()

        print("\n" + "="*80)
        print(f"MARKET ANALYSIS - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*80)

        # Overall Recommendation
        color = "\033[92m" if recommendation == "BULLISH" else "\033[91m" if recommendation == "BEARISH" else "\033[93m"
        reset = "\033[0m"

        print(f"\n🎯 RECOMMENDATION: {color}{recommendation}{reset}")
        print(f"📊 CONFIDENCE: {confidence*100:.1f}%")

        # Individual Signals
        print("\n" + "-"*80)
        print("SIGNAL BREAKDOWN")
        print("-"*80)

        if 'intensity' in signals:
            data = signals['intensity']
            print(f"\n⚡ TRADE INTENSITY:")
            print(f"   Signal: {data['signal']} ({data['confidence']*100:.0f}% conf)")
            print(f"   Aggression: {data['aggression']:+.2f} | Velocity: {data['velocity']:+.2%} | TPS: {data['tps']:.2f}")
        elif errors.get('intensity'):
            print(f"\n⚡ TRADE INTENSITY: ❌ {errors['intensity']}")

        if 'basis' in signals:
            data = signals['basis']
            print(f"\n📈 FUTURES BASIS:")
            print(f"   Signal: {data['signal']} ({data['confidence']*100:.0f}% conf)")
            print(f"   Basis: {data['basis_pct']:+.4f}% | Spot: ${data['spot']:,.2f} | Futures: ${data['futures']:,.2f}")
        elif errors.get('basis'):
            print(f"\n📈 FUTURES BASIS: ❌ {errors['basis']}")

        if 'orderbook' in signals:
            data = signals['orderbook']
            print(f"\n📚 ORDERBOOK:")
            print(f"   Signal: {data['signal']} ({data['confidence']*100:.0f}% conf)")
            print(f"   Imbalance: {data['imbalance']:.2f} | Spread: {data['spread_pct']:.4f}%")
        elif errors.get('orderbook'):
            print(f"\n📚 ORDERBOOK: ❌ {errors['orderbook']}")

        if 'kalshi' in signals:
            data = signals['kalshi']
            print(f"\n🎲 PREDICTION MARKET:")
            print(f"   Signal: {data['signal']} ({data['confidence']*100:.0f}% conf)")
            print(f"   Probability Up: {data['prob_up']:.1f}% | Current: ${data['current_price']:,.2f}")
        elif errors.get('kalshi'):
            print(f"\n🎲 PREDICTION MARKET: ❌ {errors['kalshi']}")

        print("\n" + "="*80 + "\n")


def main():
    """Main execution"""
    import argparse

    parser = argparse.ArgumentParser(description='Master Collector & Signal Aggregator')
    parser.add_argument('action', nargs='?', default='both',
                       choices=['start', 'monitor', 'both', 'stop'],
                       help='Action to perform')

    args = parser.parse_args()

    manager = CollectorManager()
    aggregator = SignalAggregator()

    if args.action == 'stop':
        manager.stop_all()
        return

    if args.action in ['start', 'both']:
        manager.start_all()
        print("⏳ Waiting 90 seconds for initial data collection...")
        time.sleep(90)

    if args.action in ['monitor', 'both']:
        print("\n" + "="*80)
        print("STARTING MARKET MONITORING (Ctrl+C to stop)")
        print("Updates every 60 seconds")
        print("="*80)

        def signal_handler(sig, frame):
            print("\n\n👋 Stopping monitoring...")
            sys.exit(0)

        sig.signal(sig.SIGINT, signal_handler)

        try:
            while True:
                aggregator.print_analysis()
                time.sleep(60)
        except KeyboardInterrupt:
            print("\n\n👋 Monitoring stopped")


if __name__ == "__main__":
    main()
