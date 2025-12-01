#!/usr/bin/env python3
"""
Kalshi Bitcoin Markets Collector

Fetches Bitcoin hourly prediction markets from Kalshi and stores them in the database.
Focuses on the next expiring hourly markets.
"""

import sys
import os
import time
import logging
import requests
from datetime import datetime, timedelta
from decimal import Decimal

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from data.storage.database import db_manager
from data.storage.models import PredictionMarket, SpotPriceSnapshot

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class KalshiCollector:
    """
    Collects Bitcoin prediction market data from Kalshi.
    """

    def __init__(self):
        self.base_url = 'https://api.elections.kalshi.com/trade-api/v2'
        self.series_ticker = 'KXBTCD'  # Hourly Bitcoin markets
        self.markets_saved = 0
        logger.info("KalshiCollector initialized")

    def fetch_markets(self):
        """Fetch open Bitcoin markets from Kalshi"""
        try:
            response = requests.get(
                f'{self.base_url}/markets',
                params={
                    'series_ticker': self.series_ticker,
                    'status': 'open',
                    'limit': 100
                },
                timeout=10
            )

            if response.status_code == 200:
                return response.json().get('markets', [])
            else:
                logger.error(f"API error: {response.status_code} - {response.text}")
                return []

        except Exception as e:
            logger.error(f"Error fetching markets: {e}")
            return []

    def get_next_expiring_markets(self, markets):
        """
        Get all markets for the next expiring event (hour).
        Returns markets grouped by event ticker.
        """
        if not markets:
            return []

        # Find the next expiring event
        now = datetime.utcnow()
        events = {}

        for market in markets:
            exp_time_str = market.get('expected_expiration_time', '')
            if not exp_time_str:
                continue

            try:
                exp_time = datetime.fromisoformat(exp_time_str.replace('Z', '+00:00'))
                exp_time = exp_time.replace(tzinfo=None)  # Remove timezone for comparison

                if exp_time > now:
                    event_ticker = market.get('event_ticker', '')
                    if event_ticker not in events:
                        events[event_ticker] = {
                            'exp_time': exp_time,
                            'markets': []
                        }
                    events[event_ticker]['markets'].append(market)
            except:
                continue

        if not events:
            return []

        # Get the event with the earliest expiration
        next_event_ticker = min(events.keys(), key=lambda k: events[k]['exp_time'])
        next_event = events[next_event_ticker]

        logger.info(f"Next expiring event: {next_event_ticker}")
        logger.info(f"  Expires in: {(next_event['exp_time'] - now).total_seconds() / 60:.1f} minutes")
        logger.info(f"  Markets: {len(next_event['markets'])}")

        return next_event['markets']

    def parse_strike_price(self, market):
        """Extract strike price from market data"""
        # Try floor_strike first
        strike = market.get('floor_strike')
        if strike:
            return Decimal(str(strike))

        # Try parsing from subtitle
        subtitle = market.get('subtitle', '')
        if '$' in subtitle:
            # Extract number from subtitle like "$96,500 or above"
            import re
            match = re.search(r'\$([0-9,]+)', subtitle)
            if match:
                price_str = match.group(1).replace(',', '')
                return Decimal(price_str)

        return None

    def save_market(self, market, snapshot_time):
        """Save a single market to database"""
        try:
            # Parse timestamp
            exp_time_str = market.get('expected_expiration_time', '')
            close_time_str = market.get('close_time', '')
            expiration_time_str = market.get('expiration_time', '')

            exp_time = None
            if exp_time_str:
                exp_time = datetime.fromisoformat(exp_time_str.replace('Z', '+00:00'))
                exp_time = exp_time.replace(tzinfo=None)

            close_time = None
            if close_time_str:
                close_time = datetime.fromisoformat(close_time_str.replace('Z', '+00:00'))
                close_time = close_time.replace(tzinfo=None)

            expiration_time = None
            if expiration_time_str:
                expiration_time = datetime.fromisoformat(expiration_time_str.replace('Z', '+00:00'))
                expiration_time = expiration_time.replace(tzinfo=None)

            # Parse strike price
            strike_price = self.parse_strike_price(market)

            # Calculate implied probability
            yes_ask = market.get('yes_ask', 0)
            implied_prob = Decimal(str(yes_ask)) if yes_ask else None

            # Create prediction market record
            with db_manager.get_session() as session:
                pm = PredictionMarket(
                    timestamp=snapshot_time,
                    source='kalshi',
                    ticker=market.get('ticker'),
                    event_ticker=market.get('event_ticker'),
                    series_ticker=self.series_ticker,
                    title=market.get('title'),
                    subtitle=market.get('subtitle'),
                    close_time=close_time,
                    expiration_time=expiration_time,
                    expected_expiration_time=exp_time,
                    strike_price=strike_price,
                    strike_type=market.get('strike_type'),
                    yes_ask=Decimal(str(yes_ask)) if yes_ask else None,
                    yes_bid=Decimal(str(market.get('yes_bid', 0))) if market.get('yes_bid') else None,
                    no_ask=Decimal(str(market.get('no_ask', 0))) if market.get('no_ask') else None,
                    no_bid=Decimal(str(market.get('no_bid', 0))) if market.get('no_bid') else None,
                    last_price=Decimal(str(market.get('last_price', 0))) if market.get('last_price') else None,
                    implied_probability=implied_prob,
                    volume=market.get('volume'),
                    volume_24h=market.get('volume_24h'),
                    open_interest=market.get('open_interest'),
                    liquidity=market.get('liquidity'),
                    status=market.get('status'),
                    extra_metadata={
                        'no_sub_title': market.get('no_sub_title'),
                        'yes_sub_title': market.get('yes_sub_title'),
                        'rules_primary': market.get('rules_primary'),
                    }
                )
                session.add(pm)
                self.markets_saved += 1

        except Exception as e:
            logger.error(f"Error saving market {market.get('ticker')}: {e}")

    def calculate_implied_btc_price(self, markets):
        """
        Calculate implied BTC spot price from Kalshi prediction markets.

        Uses multiple methods:
        1. Transition method: Find where probability crosses 50%
        2. Weighted average: Weight strikes by their probabilities

        Args:
            markets: List of market data

        Returns:
            dict with 'price', 'confidence', 'method', and 'metadata'
        """
        if not markets:
            return None

        # Sort markets by strike price
        sorted_markets = sorted(
            [m for m in markets if m.get('floor_strike')],
            key=lambda m: m.get('floor_strike', 0)
        )

        if not sorted_markets:
            return None

        # Method 1: Transition point (most reliable)
        transition_price = None
        transition_confidence = 0

        for i, m in enumerate(sorted_markets):
            yes_ask = m.get('yes_ask', 0)
            strike = m.get('floor_strike', 0)

            # Find where probability transitions from >50% to <50%
            if yes_ask <= 50:
                if i > 0:
                    prev_strike = sorted_markets[i-1].get('floor_strike', 0)
                    prev_prob = sorted_markets[i-1].get('yes_ask', 0)

                    # Interpolate between the two strikes
                    if prev_prob > 50 and yes_ask < 50:
                        # Linear interpolation to find 50% point
                        prob_diff = prev_prob - yes_ask
                        if prob_diff > 0:
                            weight = (prev_prob - 50) / prob_diff
                            transition_price = prev_strike + (strike - prev_strike) * weight

                            # Confidence based on how sharp the transition is
                            # Sharp transition (big prob change) = high confidence
                            transition_confidence = min(prob_diff / 100 * 100, 95)
                        else:
                            transition_price = (prev_strike + strike) / 2
                            transition_confidence = 60
                    else:
                        transition_price = strike
                        transition_confidence = 50
                else:
                    transition_price = strike
                    transition_confidence = 40
                break

        # Method 2: Weighted average (backup method)
        weighted_sum = 0
        weight_total = 0

        for m in sorted_markets:
            strike = m.get('floor_strike', 0)
            yes_ask = m.get('yes_ask', 0)

            # Weight = how close to 50% (markets near 50% are most informative)
            distance_from_50 = abs(yes_ask - 50)
            weight = max(0, 50 - distance_from_50)  # 0 to 50

            weighted_sum += strike * weight
            weight_total += weight

        weighted_avg_price = weighted_sum / weight_total if weight_total > 0 else None

        # Prefer transition method, fall back to weighted average
        if transition_price:
            return {
                'price': Decimal(str(round(transition_price, 2))),
                'confidence': Decimal(str(round(transition_confidence, 2))),
                'method': 'probability_transition',
                'metadata': {
                    'weighted_avg_price': float(weighted_avg_price) if weighted_avg_price else None,
                    'num_markets': len(sorted_markets)
                }
            }
        elif weighted_avg_price:
            return {
                'price': Decimal(str(round(weighted_avg_price, 2))),
                'confidence': Decimal('50.00'),  # Lower confidence for weighted average
                'method': 'weighted_average',
                'metadata': {
                    'num_markets': len(sorted_markets)
                }
            }

        return None

    def save_spot_price(self, implied_price_data, snapshot_time):
        """
        Save Kalshi implied BTC spot price to database.

        Args:
            implied_price_data: Dict from calculate_implied_btc_price()
            snapshot_time: Timestamp for this snapshot
        """
        if not implied_price_data:
            return

        try:
            with db_manager.get_session() as session:
                spot_snapshot = SpotPriceSnapshot(
                    timestamp=snapshot_time,
                    source='kalshi_implied',
                    price=implied_price_data['price'],
                    confidence=implied_price_data['confidence'],
                    derivation_method=implied_price_data['method'],
                    extra_data=implied_price_data['metadata']
                )
                session.add(spot_snapshot)

            logger.info(
                f"Kalshi implied BTC price: ${float(implied_price_data['price']):,.2f} "
                f"(confidence: {float(implied_price_data['confidence']):.0f}%, "
                f"method: {implied_price_data['method']})"
            )

        except Exception as e:
            logger.error(f"Error saving Kalshi spot price: {e}")

    def run_once(self):
        """Run one collection cycle"""
        logger.info("=" * 80)
        logger.info("Fetching Kalshi Bitcoin Markets...")
        logger.info("=" * 80)

        # Fetch all markets
        all_markets = self.fetch_markets()
        logger.info(f"Fetched {len(all_markets)} total markets")

        # Get next expiring markets
        next_markets = self.get_next_expiring_markets(all_markets)

        if next_markets:
            logger.info(f"\nSaving {len(next_markets)} markets to database...")

            # Create a single snapshot timestamp for all markets
            snapshot_time = datetime.utcnow()

            for market in next_markets:
                self.save_market(market, snapshot_time)

            logger.info(f"✅ Saved {self.markets_saved} markets to database")

            # Calculate and save implied BTC spot price
            implied_price = self.calculate_implied_btc_price(next_markets)
            if implied_price:
                self.save_spot_price(implied_price, snapshot_time)

            # Display summary
            print("\n" + "=" * 80)
            print(f"Next Hourly Bitcoin Markets on Kalshi")
            print("=" * 80)
            print(f"Event: {next_markets[0].get('event_ticker')}")
            print(f"Expires: {next_markets[0].get('expected_expiration_time')}")

            # Sort by strike price to find markets near current BTC price
            sorted_by_strike = sorted(
                [m for m in next_markets if m.get('floor_strike')],
                key=lambda m: m.get('floor_strike', 0)
            )

            # Find the transition point (where probability goes from >50% to <50%)
            # This indicates the current BTC price range
            transition_idx = None
            for i, m in enumerate(sorted_by_strike):
                yes_ask = m.get('yes_ask', 0)
                if yes_ask <= 50:
                    transition_idx = i
                    break

            if transition_idx is None:
                transition_idx = len(sorted_by_strike) // 2

            # Show 3 strikes below and 3 strikes above current price
            start_idx = max(0, transition_idx - 3)
            end_idx = min(len(sorted_by_strike), transition_idx + 3)
            display_markets = sorted_by_strike[start_idx:end_idx]

            print(f"\nMarkets near current BTC price:")
            print(f"{'Strike':>10s} | {'Prob':>4s} | {'Volume':>8s} | {'Open Int':>8s}")
            print("-" * 50)

            for m in display_markets:
                strike = m.get('floor_strike', 0)
                yes_ask = m.get('yes_ask', 0)
                volume = m.get('volume', 0)
                oi = m.get('open_interest', 0)

                # Highlight the transition zone (current price area)
                marker = " <--" if 40 <= yes_ask <= 60 else ""
                print(f"${strike:>9,.0f} | {yes_ask:3d}% | {volume:>8,d} | {oi:>8,d}{marker}")

            print("=" * 80)
        else:
            logger.warning("No upcoming markets found")

    def run_continuous(self, interval_seconds=60):
        """Run continuous collection every interval_seconds"""
        logger.info(f"Starting continuous collection (every {interval_seconds}s)")
        logger.info("Press Ctrl+C to stop")

        try:
            while True:
                self.run_once()
                logger.info(f"\nWaiting {interval_seconds} seconds...")
                time.sleep(interval_seconds)

        except KeyboardInterrupt:
            logger.info("\nStopping collector...")
            logger.info(f"Total markets saved: {self.markets_saved}")


def main():
    """Main entry point"""
    collector = KalshiCollector()

    import sys
    if len(sys.argv) > 1 and sys.argv[1] == '--continuous':
        collector.run_continuous(interval_seconds=60)
    else:
        collector.run_once()


if __name__ == "__main__":
    main()
