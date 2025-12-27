#!/usr/bin/env python3
"""
Kalshi Base Collector

Shared functionality for all Kalshi collectors.
Provides common API methods, timestamp parsing, and market filtering.
"""

import logging
import requests
from datetime import datetime
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)


class KalshiBaseCollector:
    """
    Base class for Kalshi collectors with shared API and utility methods.
    """

    def __init__(self):
        self.base_url = 'https://api.elections.kalshi.com/trade-api/v2'
        self.series_ticker = 'KXBTCD'  # Hourly Bitcoin markets

    def fetch_markets(self) -> List[Dict]:
        """
        Fetch open Bitcoin hourly markets from Kalshi API.

        Returns:
            List of market dictionaries
        """
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

    def parse_timestamp(self, time_str: str) -> Optional[datetime]:
        """
        Parse ISO timestamp from Kalshi API to datetime.

        Args:
            time_str: ISO format timestamp string

        Returns:
            datetime object with timezone removed, or None if parsing fails
        """
        if not time_str:
            return None

        try:
            dt = datetime.fromisoformat(time_str.replace('Z', '+00:00'))
            return dt.replace(tzinfo=None)  # Remove timezone for DB consistency
        except Exception as e:
            logger.error(f"Error parsing timestamp '{time_str}': {e}")
            return None

    def get_next_expiring_event(self, markets: List[Dict]) -> Optional[Dict]:
        """
        Find the next expiring event from a list of markets.

        Args:
            markets: List of market dictionaries

        Returns:
            Dict with 'event_ticker', 'exp_time', and 'markets' list,
            or None if no valid events found
        """
        if not markets:
            return None

        now = datetime.utcnow()
        events = {}

        for market in markets:
            exp_time_str = market.get('expected_expiration_time', '')
            if not exp_time_str:
                continue

            exp_time = self.parse_timestamp(exp_time_str)
            if not exp_time or exp_time <= now:
                continue

            event_ticker = market.get('event_ticker', '')
            if not event_ticker:
                continue

            if event_ticker not in events:
                events[event_ticker] = {
                    'event_ticker': event_ticker,
                    'exp_time': exp_time,
                    'markets': []
                }
            events[event_ticker]['markets'].append(market)

        if not events:
            return None

        # Return the event with earliest expiration
        next_event_ticker = min(events.keys(), key=lambda k: events[k]['exp_time'])
        return events[next_event_ticker]

    def fetch_orderbook(self, ticker: str) -> Optional[Dict]:
        """
        Fetch orderbook for a specific market ticker.

        Args:
            ticker: Market ticker symbol

        Returns:
            Orderbook dictionary or None if fetch fails
        """
        try:
            response = requests.get(
                f'{self.base_url}/markets/{ticker}/orderbook',
                timeout=5
            )

            if response.status_code == 200:
                return response.json().get('orderbook', {})
            else:
                logger.error(f"Orderbook API error for {ticker}: {response.status_code}")
                return None

        except Exception as e:
            logger.error(f"Error fetching orderbook for {ticker}: {e}")
            return None
