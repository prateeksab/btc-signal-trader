#!/usr/bin/env python3
"""
Kalshi Orderbook Collector

Collects orderbook data (spreads, depth, liquidity) for Bitcoin hourly markets
near the current BTC price. Runs at high frequency (5 seconds).
"""

import sys
import os
import time
import logging
import requests
import warnings
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from typing import List, Dict, Optional

# Suppress deprecation warnings for datetime.utcnow()
warnings.filterwarnings('ignore', category=DeprecationWarning, module='__main__')

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from data.storage.database import db_manager
from data.storage.models import Trade, KalshiOrderbookSnapshot, BtcPriceCandle
from data.processors.volatility_calculator import VolatilityCalculator
from data.collectors.kalshi_base import KalshiBaseCollector

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class KalshiOrderbookCollector(KalshiBaseCollector):
    """
    Collects orderbook data from Kalshi Bitcoin hourly markets.
    Focuses on markets near current BTC price (5 above, 5 below).
    Inherits shared API methods from KalshiBaseCollector.
    """

    def __init__(self):
        super().__init__()  # Initialize base class
        self.snapshots_collected = 0
        self.btc_price_source = None  # Track source of BTC price
        self.vol_calculator = VolatilityCalculator()  # Volatility calculator
        logger.info("KalshiOrderbookCollector initialized")

    def fetch_coinbase_btc_price(self) -> Optional[float]:
        """Fetch current BTC price directly from Coinbase API"""
        try:
            response = requests.get(
                'https://api.exchange.coinbase.com/products/BTC-USD/ticker',
                timeout=5
            )

            if response.status_code == 200:
                data = response.json()
                price = float(data.get('price', 0))
                if price > 0:
                    logger.info(f"Fetched BTC price from Coinbase: ${price:,.2f}")
                    return price
                else:
                    logger.error("Invalid price from Coinbase API")
                    return None
            else:
                logger.error(f"Coinbase API error: {response.status_code}")
                return None

        except Exception as e:
            logger.error(f"Error fetching BTC price from Coinbase: {e}")
            return None

    def get_current_btc_price(self) -> Optional[float]:
        """
        Get current BTC price, trying Coinbase API first, then database as fallback
        """
        # Try Coinbase API first
        price = self.fetch_coinbase_btc_price()
        if price:
            self.btc_price_source = 'coinbase_api'
            return price

        # Fallback to database
        logger.warning("Coinbase API failed, falling back to database")
        try:
            with db_manager.get_session() as session:
                latest_trade = session.query(Trade).filter(
                    Trade.exchange == 'coinbase'
                ).order_by(Trade.timestamp.desc()).first()

                if latest_trade:
                    age = (datetime.now(timezone.utc).replace(tzinfo=None) - latest_trade.timestamp).total_seconds()
                    logger.warning(f"Using database price (age: {age:.0f}s)")
                    self.btc_price_source = 'coinbase_db'
                    return float(latest_trade.price)

            logger.error("No recent BTC price found in database")
            self.btc_price_source = None
            return None

        except Exception as e:
            logger.error(f"Error getting BTC price from database: {e}")
            self.btc_price_source = None
            return None


    def get_next_expiring_markets_near_price(
        self,
        markets: List[Dict],
        current_price: float,
        num_above: int = 5,
        num_below: int = 5
    ) -> List[Dict]:
        """
        Get markets from the next expiring event that are near the current BTC price.

        Args:
            markets: List of all open markets
            current_price: Current BTC price
            num_above: Number of markets to get above current price
            num_below: Number of markets to get below current price

        Returns:
            List of markets near current price
        """
        if not markets:
            return []

        # Group markets by event (expiration time)
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        events = {}

        for market in markets:
            exp_time_str = market.get('expected_expiration_time', '')
            if not exp_time_str:
                continue

            try:
                exp_time = datetime.fromisoformat(exp_time_str.replace('Z', '+00:00'))
                exp_time = exp_time.replace(tzinfo=None)

                if exp_time > now:
                    event_ticker = market.get('event_ticker', '')
                    if event_ticker not in events:
                        events[event_ticker] = {
                            'exp_time': exp_time,
                            'markets': []
                        }

                    # Only add markets with valid strike prices
                    strike = market.get('floor_strike')
                    if strike:
                        events[event_ticker]['markets'].append(market)
            except:
                continue

        if not events:
            return []

        # Get the next expiring event
        next_event_ticker = min(events.keys(), key=lambda k: events[k]['exp_time'])
        next_event = events[next_event_ticker]

        logger.info(f"Next expiring event: {next_event_ticker}")
        logger.info(f"  Expires at: {next_event['exp_time']}")
        logger.info(f"  Total markets: {len(next_event['markets'])}")

        # Sort markets by strike price
        sorted_markets = sorted(
            next_event['markets'],
            key=lambda m: m.get('floor_strike', 0)
        )

        # Find markets above and below current price
        markets_below = [m for m in sorted_markets if m.get('floor_strike', 0) <= current_price]
        markets_above = [m for m in sorted_markets if m.get('floor_strike', 0) > current_price]

        # Get the closest N markets on each side
        selected_below = markets_below[-num_below:] if len(markets_below) >= num_below else markets_below
        selected_above = markets_above[:num_above] if len(markets_above) >= num_above else markets_above

        selected_markets = selected_below + selected_above

        logger.info(f"  Selected {len(selected_markets)} markets near ${current_price:,.2f}")
        logger.info(f"    Below: {len(selected_below)} markets")
        logger.info(f"    Above: {len(selected_above)} markets")

        return selected_markets


    def parse_orderbook(self, orderbook: Dict, num_levels: int = 6) -> Dict:
        """
        Parse orderbook data and calculate key metrics.

        Args:
            orderbook: Raw orderbook data from API
            num_levels: Number of price levels to keep

        Returns:
            Parsed orderbook with spreads and metrics
        """
        yes_bids = orderbook.get('yes') or []
        no_bids = orderbook.get('no') or []

        # Arrays are sorted ascending, best prices at the END
        # Get top N levels (from the end)
        top_yes_bids = yes_bids[-num_levels:] if len(yes_bids) > num_levels else yes_bids
        top_no_bids = no_bids[-num_levels:] if len(no_bids) > num_levels else no_bids

        # Reverse to show best first
        top_yes_bids = list(reversed(top_yes_bids))
        top_no_bids = list(reversed(top_no_bids))

        # Calculate best bid/ask
        best_yes_bid = yes_bids[-1][0] if yes_bids else None
        best_no_bid = no_bids[-1][0] if no_bids else None

        # Implied asks (bid for opposite side = ask for this side)
        # YES ask = 100 - NO bid
        # NO ask = 100 - YES bid
        best_yes_ask = (100 - best_no_bid) if best_no_bid else None
        best_no_ask = (100 - best_yes_bid) if best_yes_bid else None

        # Calculate spreads
        yes_spread = (best_yes_ask - best_yes_bid) if (best_yes_ask and best_yes_bid) else None
        no_spread = (best_no_ask - best_no_bid) if (best_no_ask and best_no_bid) else None

        # Calculate total liquidity at top levels
        yes_liquidity = sum(level[1] for level in top_yes_bids)
        no_liquidity = sum(level[1] for level in top_no_bids)

        return {
            'yes_bids': top_yes_bids,
            'no_bids': top_no_bids,
            'best_yes_bid': best_yes_bid,
            'best_yes_ask': best_yes_ask,
            'best_no_bid': best_no_bid,
            'best_no_ask': best_no_ask,
            'yes_spread': yes_spread,
            'no_spread': no_spread,
            'yes_liquidity': yes_liquidity,
            'no_liquidity': no_liquidity,
            'yes_levels': len(top_yes_bids),
            'no_levels': len(top_no_bids),
        }

    def calculate_moneyness(self, strike_price: float, current_price: float) -> Dict:
        """
        Calculate moneyness metrics for a strike.

        Args:
            strike_price: Strike price of the market
            current_price: Current BTC spot price

        Returns:
            Dict with distance, moneyness_pct, and moneyness_category
        """
        distance = Decimal(str(strike_price)) - Decimal(str(current_price))
        moneyness_pct = ((Decimal(str(strike_price)) / Decimal(str(current_price))) - 1) * 100

        # Categorize (for YES contracts)
        # ATM: within ±0.2% of current price
        # ITM: below current price - 0.2%
        # OTM: above current price + 0.2%
        if abs(float(moneyness_pct)) <= 0.2:
            category = 'ATM'
        elif float(moneyness_pct) < -0.2:
            category = 'ITM'
        else:
            category = 'OTM'

        return {
            'distance': distance,
            'moneyness_pct': moneyness_pct,
            'category': category
        }

    def calculate_volatility_metrics(
        self,
        market: Dict,
        orderbook_data: Dict,
        current_price: float,
        snapshot_time: datetime
    ) -> Dict:
        """
        Calculate all volatility metrics for a market.

        Returns dict with volatility metrics or None values if calculation fails.
        """
        metrics = {
            'implied_vol': None,
            'rv_30min': None,
            'rv_10min': None,
            'rv_yesterday': None,
            'rv_last_week': None,
            'iv_rank': None,
            'delta': None,
            'gamma': None,
            'vega': None,
            'theta': None,
            'vanna': None,
            'volga': None,
            'contract_start_time': None,
            'contract_end_time': None,
            'contract_time_window': None
        }

        try:
            # Get market details
            strike_price = market.get('floor_strike')
            ticker = market.get('ticker')
            exp_time_str = market.get('expected_expiration_time')
            open_time_str = market.get('open_time')

            if not exp_time_str or not strike_price:
                return metrics

            # Parse contract times
            metrics['contract_end_time'] = datetime.fromisoformat(exp_time_str.replace('Z', '+00:00')).replace(tzinfo=None)
            if open_time_str:
                metrics['contract_start_time'] = datetime.fromisoformat(open_time_str.replace('Z', '+00:00')).replace(tzinfo=None)

            # Calculate time window string (HH:MM to HH:MM)
            if metrics['contract_start_time'] and metrics['contract_end_time']:
                start_time = metrics['contract_start_time'].strftime('%H:%M')
                end_time = metrics['contract_end_time'].strftime('%H:%M')
                metrics['contract_time_window'] = f"{start_time} to {end_time}"

            # Calculate time to expiry
            time_to_expiry = (metrics['contract_end_time'] - snapshot_time).total_seconds() / 3600  # hours

            if time_to_expiry <= 0:
                return metrics

            # 1. Implied Volatility (from YES bid price)
            yes_bid = orderbook_data.get('best_yes_bid')
            if yes_bid and yes_bid > 0 and yes_bid < 100:
                metrics['implied_vol'] = self.vol_calculator.estimate_implied_volatility_from_kalshi(
                    spot_price=current_price,
                    strike_price=strike_price,
                    market_price=float(yes_bid),
                    time_to_expiry_hours=time_to_expiry,
                    ticker=ticker  # Pass ticker for previous IV lookup
                )

            # 2. Realized Volatility - Last 30 minutes
            metrics['rv_30min'] = self.vol_calculator.calculate_realized_vol_from_db(
                end_time=snapshot_time,
                window_minutes=30
            )

            # 3. Realized Volatility - Last 10 minutes
            metrics['rv_10min'] = self.vol_calculator.calculate_realized_vol_from_db(
                end_time=snapshot_time,
                window_minutes=10
            )

            # 4. Realized Volatility - Yesterday same hour
            yesterday_same_time = snapshot_time - timedelta(days=1)
            metrics['rv_yesterday'] = self.vol_calculator.calculate_realized_vol_from_db(
                end_time=yesterday_same_time,
                window_minutes=60  # 1 hour window
            )

            # 5. Realized Volatility - Last week same hour
            last_week_same_time = snapshot_time - timedelta(days=7)
            metrics['rv_last_week'] = self.vol_calculator.calculate_realized_vol_from_db(
                end_time=last_week_same_time,
                window_minutes=60  # 1 hour window
            )

            # 6. Implied Volatility Rank
            if metrics['implied_vol']:
                metrics['iv_rank'] = self.vol_calculator.calculate_implied_volatility_rank(
                    current_iv=metrics['implied_vol'],
                    ticker=ticker,
                    lookback_days=30
                )

            # 7. Option Greeks (Delta, Gamma, Vega, Theta, Vanna, Volga)
            if metrics['implied_vol']:
                greeks = self.vol_calculator.calculate_binary_option_greeks(
                    spot_price=current_price,
                    strike_price=strike_price,
                    time_to_expiry_hours=time_to_expiry,
                    implied_vol=metrics['implied_vol']
                )
                if greeks:
                    metrics['delta'] = greeks['delta']
                    metrics['gamma'] = greeks['gamma']
                    metrics['vega'] = greeks['vega']
                    metrics['theta'] = greeks['theta']
                    metrics['vanna'] = greeks['vanna']
                    metrics['volga'] = greeks['volga']

        except Exception as e:
            logger.error(f"Error calculating volatility metrics: {e}")

        return metrics

    def save_orderbook_to_db(
        self,
        market: Dict,
        orderbook_data: Dict,
        current_price: float,
        snapshot_time: datetime
    ):
        """Save orderbook snapshot to database"""
        try:
            ticker = market.get('ticker')
            event_ticker = market.get('event_ticker')
            strike_price = market.get('floor_strike')

            if not strike_price:
                logger.warning(f"No strike price for {ticker}, skipping DB save")
                return

            # Calculate moneyness
            moneyness = self.calculate_moneyness(strike_price, current_price)

            # Calculate volatility metrics
            vol_metrics = self.calculate_volatility_metrics(
                market, orderbook_data, current_price, snapshot_time
            )

            with db_manager.get_session() as session:
                snapshot = KalshiOrderbookSnapshot(
                    timestamp=snapshot_time,
                    ticker=ticker,
                    event_ticker=event_ticker,
                    strike_price=Decimal(str(strike_price)),
                    current_btc_price=Decimal(str(current_price)),
                    btc_price_source=self.btc_price_source or 'unknown',
                    distance_from_current=moneyness['distance'],
                    moneyness_pct=moneyness['moneyness_pct'],
                    moneyness_category=moneyness['category'],
                    best_yes_bid=Decimal(str(orderbook_data['best_yes_bid'])) if orderbook_data['best_yes_bid'] else None,
                    best_yes_ask=Decimal(str(orderbook_data['best_yes_ask'])) if orderbook_data['best_yes_ask'] else None,
                    best_no_bid=Decimal(str(orderbook_data['best_no_bid'])) if orderbook_data['best_no_bid'] else None,
                    best_no_ask=Decimal(str(orderbook_data['best_no_ask'])) if orderbook_data['best_no_ask'] else None,
                    yes_spread=Decimal(str(orderbook_data['yes_spread'])) if orderbook_data['yes_spread'] else None,
                    no_spread=Decimal(str(orderbook_data['no_spread'])) if orderbook_data['no_spread'] else None,
                    yes_liquidity=orderbook_data['yes_liquidity'],
                    no_liquidity=orderbook_data['no_liquidity'],
                    yes_levels=orderbook_data['yes_levels'],
                    no_levels=orderbook_data['no_levels'],
                    yes_bids=orderbook_data['yes_bids'],
                    no_bids=orderbook_data['no_bids'],
                    # Contract times
                    contract_start_time=vol_metrics['contract_start_time'],
                    contract_end_time=vol_metrics['contract_end_time'],
                    contract_time_window=vol_metrics['contract_time_window'],
                    # Volatility metrics
                    implied_volatility=Decimal(str(vol_metrics['implied_vol'])) if vol_metrics['implied_vol'] else None,
                    realized_vol_30min=Decimal(str(vol_metrics['rv_30min'])) if vol_metrics['rv_30min'] else None,
                    realized_vol_10min=Decimal(str(vol_metrics['rv_10min'])) if vol_metrics['rv_10min'] else None,
                    realized_vol_yesterday_same_hour=Decimal(str(vol_metrics['rv_yesterday'])) if vol_metrics['rv_yesterday'] else None,
                    realized_vol_last_week_same_hour=Decimal(str(vol_metrics['rv_last_week'])) if vol_metrics['rv_last_week'] else None,
                    implied_volatility_rank=Decimal(str(vol_metrics['iv_rank'])) if vol_metrics['iv_rank'] else None,
                    # Option Greeks
                    delta=Decimal(str(vol_metrics['delta'])) if vol_metrics['delta'] else None,
                    gamma=Decimal(str(vol_metrics['gamma'])) if vol_metrics['gamma'] else None,
                    vega=Decimal(str(vol_metrics['vega'])) if vol_metrics['vega'] else None,
                    theta=Decimal(str(vol_metrics['theta'])) if vol_metrics['theta'] else None,
                    vanna=Decimal(str(vol_metrics['vanna'])) if vol_metrics['vanna'] else None,
                    volga=Decimal(str(vol_metrics['volga'])) if vol_metrics['volga'] else None
                )
                session.add(snapshot)

        except Exception as e:
            logger.error(f"Error saving orderbook to DB for {ticker}: {e}")

    def display_orderbook(self, market: Dict, orderbook_data: Dict):
        """Display orderbook data in a readable format"""
        strike = market.get('floor_strike', 0)
        ticker = market.get('ticker', '')

        print("\n" + "="*80)
        print(f"Market: {ticker}")
        print(f"Strike: ${strike:,.2f}")
        print("="*80)

        # YES side
        print(f"\nYES Side:")
        print(f"  Spread: {orderbook_data['yes_spread']:.1f}¢" if orderbook_data['yes_spread'] else "  Spread: N/A")
        print(f"  Best Bid: {orderbook_data['best_yes_bid']}¢ | Best Ask: {orderbook_data['best_yes_ask']}¢")
        print(f"  Top {orderbook_data['yes_levels']} levels liquidity: {orderbook_data['yes_liquidity']:,} contracts")

        print(f"\n  {'Price (¢)':>10s} | {'Size':>10s}")
        print(f"  {'-'*10} | {'-'*10}")
        for price, size in orderbook_data['yes_bids'][:6]:
            print(f"  {price:>10d} | {size:>10,d}")

        # NO side
        print(f"\nNO Side:")
        print(f"  Spread: {orderbook_data['no_spread']:.1f}¢" if orderbook_data['no_spread'] else "  Spread: N/A")
        print(f"  Best Bid: {orderbook_data['best_no_bid']}¢ | Best Ask: {orderbook_data['best_no_ask']}¢")
        print(f"  Top {orderbook_data['no_levels']} levels liquidity: {orderbook_data['no_liquidity']:,} contracts")

        print(f"\n  {'Price (¢)':>10s} | {'Size':>10s}")
        print(f"  {'-'*10} | {'-'*10}")
        for price, size in orderbook_data['no_bids'][:6]:
            print(f"  {price:>10d} | {size:>10,d}")

        print("="*80)

    def save_btc_price_candle(self, price: float, timestamp: datetime):
        """
        Save BTC price as a 1-minute candle.
        Updates existing candle or creates new one.

        Args:
            price: Current BTC price
            timestamp: Current timestamp
        """
        try:
            # Floor timestamp to 1-minute interval
            floored_time = timestamp.replace(second=0, microsecond=0)

            with db_manager.get_session() as session:
                # Check if candle already exists for this minute
                existing_candle = session.query(BtcPriceCandle).filter(
                    BtcPriceCandle.timestamp == floored_time,
                    BtcPriceCandle.timeframe == '1min',
                    BtcPriceCandle.exchange == 'kalshi',
                    BtcPriceCandle.symbol == 'BTC-USD'
                ).first()

                if existing_candle:
                    # Update existing candle (update high, low, close)
                    existing_candle.high = max(float(existing_candle.high), price)
                    existing_candle.low = min(float(existing_candle.low), price)
                    existing_candle.close = Decimal(str(price))
                else:
                    # Create new candle
                    new_candle = BtcPriceCandle(
                        timestamp=floored_time,
                        timeframe='1min',
                        exchange='kalshi',
                        symbol='BTC-USD',
                        open=Decimal(str(price)),
                        high=Decimal(str(price)),
                        low=Decimal(str(price)),
                        close=Decimal(str(price)),
                        volume=Decimal('0'),  # Kalshi doesn't provide volume
                        num_trades=1
                    )
                    session.add(new_candle)

                session.commit()
                logger.debug(f"Saved BTC price candle: ${price:,.2f} at {floored_time}")

        except Exception as e:
            logger.error(f"Error saving BTC price candle: {e}")

    def run_once(self, current_price=None):
        """Run one collection cycle"""
        logger.info("="*80)
        logger.info("Collecting Kalshi Orderbook Data...")
        logger.info("="*80)

        # Get current BTC price if not provided
        if current_price is None:
            current_price = self.get_current_btc_price()
            if not current_price:
                logger.error("Cannot proceed without current BTC price")
                return

            logger.info(f"Current BTC Price: ${current_price:,.2f}")

            # Save BTC price as 1-minute candle (only when fetching price ourselves)
            snapshot_time = datetime.now(timezone.utc).replace(tzinfo=None)
            self.save_btc_price_candle(current_price, snapshot_time)
        else:
            logger.info(f"Current BTC Price: ${current_price:,.2f}")

        # Fetch markets
        all_markets = self.fetch_markets()
        logger.info(f"Fetched {len(all_markets)} total markets")

        # Get markets near current price
        selected_markets = self.get_next_expiring_markets_near_price(
            all_markets,
            current_price,
            num_above=5,
            num_below=5
        )

        if not selected_markets:
            logger.warning("No markets found near current price")
            return

        # Collect orderbook for each market
        snapshot_time = datetime.now(timezone.utc).replace(tzinfo=None)

        print("\n" + "="*80)
        print(f"ORDERBOOK SNAPSHOT - {snapshot_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Current BTC: ${current_price:,.2f}")
        print(f"Collecting {len(selected_markets)} markets")
        print("="*80)

        successful = 0
        for market in selected_markets:
            ticker = market.get('ticker')

            # Fetch orderbook
            orderbook = self.fetch_orderbook(ticker)
            if not orderbook:
                logger.warning(f"Failed to fetch orderbook for {ticker}")
                continue

            # Parse orderbook
            orderbook_data = self.parse_orderbook(orderbook, num_levels=6)

            # Save to database
            self.save_orderbook_to_db(market, orderbook_data, current_price, snapshot_time)

            # Display
            self.display_orderbook(market, orderbook_data)

            successful += 1
            self.snapshots_collected += 1

            # Small delay between requests
            time.sleep(0.1)

        logger.info(f"✅ Collected {successful}/{len(selected_markets)} orderbooks")
        logger.info(f"Total snapshots this session: {self.snapshots_collected}")

    def run_continuous(self, interval_seconds=5, btc_price_interval=10):
        """
        Run continuous collection at specified interval.

        Args:
            interval_seconds: Interval for orderbook collection (default 60s)
            btc_price_interval: Interval for BTC price updates (default 10s)
        """
        logger.info(f"Starting continuous collection:")
        logger.info(f"  - Orderbook collection: every {interval_seconds}s")
        logger.info(f"  - BTC price updates: every {btc_price_interval}s")
        logger.info("Press Ctrl+C to stop")

        try:
            elapsed = 0
            last_price = None

            while True:
                # Always collect BTC price
                current_price = self.get_current_btc_price()
                if current_price:
                    snapshot_time = datetime.now(timezone.utc).replace(tzinfo=None)
                    self.save_btc_price_candle(current_price, snapshot_time)
                    logger.info(f"💰 BTC Price: ${current_price:,.2f}")
                    last_price = current_price

                # Collect orderbook data at the longer interval
                if elapsed % interval_seconds == 0:
                    if last_price:
                        self.run_once(current_price=last_price)
                    else:
                        self.run_once()
                    logger.info(f"\nNext orderbook collection in {interval_seconds}s...")
                else:
                    logger.info(f"Next BTC price update in {btc_price_interval}s...")

                time.sleep(btc_price_interval)
                elapsed += btc_price_interval

        except KeyboardInterrupt:
            logger.info("\nStopping collector...")
            logger.info(f"Total orderbook snapshots collected: {self.snapshots_collected}")


def main():
    """Main entry point"""
    collector = KalshiOrderbookCollector()

    import sys
    if len(sys.argv) > 1 and sys.argv[1] == '--continuous':
        interval = int(sys.argv[2]) if len(sys.argv) > 2 else 5
        collector.run_continuous(interval_seconds=interval)
    else:
        collector.run_once()


if __name__ == "__main__":
    main()
