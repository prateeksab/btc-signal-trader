#!/usr/bin/env python3
"""
Volatility Calculator

Calculates realized and implied volatility metrics for Bitcoin markets.
"""

import sys
import os
import numpy as np
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Optional, Dict
from scipy import stats
from scipy.optimize import brentq

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from data.storage.database import db_manager
from data.storage.models import Trade, KalshiOrderbookSnapshot


class VolatilityCalculator:
    """
    Calculates various volatility metrics for Bitcoin markets.
    """

    def __init__(self):
        pass

    def calculate_realized_volatility(
        self,
        prices: list,
        window_minutes: int
    ) -> Optional[float]:
        """
        Calculate realized volatility from price data.

        Uses the standard deviation of log returns, annualized.

        Args:
            prices: List of prices (most recent last)
            window_minutes: Time window in minutes

        Returns:
            Annualized realized volatility as percentage (e.g., 65.5 for 65.5%)
        """
        if len(prices) < 2:
            return None

        try:
            # Calculate log returns
            prices_array = np.array(prices, dtype=float)
            log_returns = np.diff(np.log(prices_array))

            if len(log_returns) == 0:
                return None

            # Calculate standard deviation
            vol = np.std(log_returns, ddof=1)

            # Annualize: sqrt(periods per year)
            # Assuming 1-second data, periods per year = 365.25 * 24 * 60 * 60
            periods_per_year = 365.25 * 24 * 60 * 60 / (window_minutes * 60)
            annualized_vol = vol * np.sqrt(periods_per_year)

            # Convert to percentage
            return float(annualized_vol * 100)

        except Exception as e:
            return None

    def get_btc_prices_from_db(
        self,
        end_time: datetime,
        window_minutes: int,
        exchange: str = 'coinbase'
    ) -> list:
        """
        Fetch BTC prices from database for a time window.

        Args:
            end_time: End of the time window
            window_minutes: Length of window in minutes
            exchange: Exchange to fetch from

        Returns:
            List of prices (chronological order, oldest first)
        """
        start_time = end_time - timedelta(minutes=window_minutes)

        try:
            with db_manager.get_session() as session:
                trades = session.query(Trade).filter(
                    Trade.exchange == exchange,
                    Trade.timestamp >= start_time,
                    Trade.timestamp <= end_time
                ).order_by(Trade.timestamp).all()

                return [float(t.price) for t in trades]

        except Exception as e:
            return []

    def calculate_realized_vol_from_db(
        self,
        end_time: datetime,
        window_minutes: int,
        exchange: str = 'coinbase'
    ) -> Optional[float]:
        """
        Calculate realized volatility from database prices.

        Args:
            end_time: End time for calculation
            window_minutes: Time window in minutes
            exchange: Exchange to use

        Returns:
            Annualized realized volatility as percentage
        """
        prices = self.get_btc_prices_from_db(end_time, window_minutes, exchange)

        if len(prices) < 10:  # Need minimum data points
            return None

        return self.calculate_realized_volatility(prices, window_minutes)

    def black_scholes_call_price(
        self,
        S: float,    # Current price
        K: float,    # Strike price
        T: float,    # Time to expiration (years)
        r: float,    # Risk-free rate
        sigma: float # Volatility (decimal, e.g., 0.65 for 65%)
    ) -> float:
        """
        Calculate Black-Scholes call option price.

        Args:
            S: Current spot price
            K: Strike price
            T: Time to expiration in years
            r: Risk-free rate (annual)
            sigma: Volatility (annualized, as decimal)

        Returns:
            Call option price
        """
        if T <= 0 or sigma <= 0:
            return max(S - K, 0)  # Intrinsic value

        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        call_price = S * stats.norm.cdf(d1) - K * np.exp(-r * T) * stats.norm.cdf(d2)
        return call_price

    def estimate_implied_volatility_from_kalshi(
        self,
        spot_price: float,
        strike_price: float,
        market_price: float,  # Probability in cents (0-100)
        time_to_expiry_hours: float,
        risk_free_rate: float = 0.0,  # Default 0% for short-term options
        ticker: str = None,  # Optional ticker to fetch previous IV
        previous_iv: float = None  # Optional previous IV estimate
    ) -> Optional[float]:
        """
        Estimate implied volatility from Kalshi market price.

        Kalshi markets are binary options. We treat the YES price as
        the probability of BTC being above the strike at expiry.

        Args:
            spot_price: Current BTC price
            strike_price: Strike price of the market
            market_price: Market price in cents (0-100, where 50 = 50% probability)
            time_to_expiry_hours: Hours until expiration
            risk_free_rate: Annual risk-free rate (default 0% for short-term options)
            ticker: Optional ticker to fetch previous IV estimate
            previous_iv: Optional previous IV estimate to use as starting point

        Returns:
            Implied volatility as percentage (e.g., 65.5 for 65.5%)
        """
        if time_to_expiry_hours <= 0 or market_price <= 0 or market_price >= 100:
            return None

        try:
            # Convert to years
            T = time_to_expiry_hours / (365.25 * 24)

            # Convert market price to probability
            market_prob = market_price / 100.0

            # Get previous IV estimate if available
            if previous_iv is None and ticker:
                previous_iv = self._get_previous_iv(ticker)

            # For a binary call option (pays 1 if S > K, else 0):
            # The market probability should approximately equal N(d2) in Black-Scholes
            # where d2 = (ln(S/K) + (r - 0.5*sigma^2)*T) / (sigma*sqrt(T))

            # Define function to solve for sigma
            def objective(sigma):
                if sigma <= 0.001:
                    sigma = 0.001
                d2 = (np.log(spot_price / strike_price) + (risk_free_rate - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
                return stats.norm.cdf(d2) - market_prob

            # Determine search bounds based on previous IV
            if previous_iv and 1 < previous_iv < 500:
                # Use previous IV to narrow search bounds
                prev_sigma = previous_iv / 100.0  # Convert to decimal
                lower = max(0.01, prev_sigma * 0.5)  # 50% lower
                upper = min(5.0, prev_sigma * 2.0)   # 100% higher
            else:
                # Default wide bounds
                lower = 0.01
                upper = 5.0

            # Try to find sigma using Brent's method
            try:
                implied_vol = brentq(objective, lower, upper, maxiter=50)
                return float(implied_vol * 100)  # Convert to percentage
            except ValueError:
                # If no solution found in narrowed range, try wide range
                try:
                    implied_vol = brentq(objective, 0.01, 5.0, maxiter=100)
                    return float(implied_vol * 100)
                except ValueError:
                    return None

        except Exception as e:
            return None

    def _get_previous_iv(self, ticker: str) -> Optional[float]:
        """
        Get the most recent IV estimate for a ticker.

        Args:
            ticker: Market ticker

        Returns:
            Previous IV as percentage, or None
        """
        try:
            lookback_start = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(hours=2)

            with db_manager.get_session() as session:
                snapshot = session.query(KalshiOrderbookSnapshot).filter(
                    KalshiOrderbookSnapshot.ticker == ticker,
                    KalshiOrderbookSnapshot.timestamp >= lookback_start,
                    KalshiOrderbookSnapshot.implied_volatility.isnot(None)
                ).order_by(KalshiOrderbookSnapshot.timestamp.desc()).first()

                if snapshot:
                    return float(snapshot.implied_volatility)

        except Exception as e:
            pass

        return None

    def calculate_binary_option_greeks(
        self,
        spot_price: float,
        strike_price: float,
        time_to_expiry_hours: float,
        implied_vol: float,  # As percentage (e.g., 65.5 for 65.5%)
        risk_free_rate: float = 0.0
    ) -> Optional[Dict[str, float]]:
        """
        Calculate Greeks for a binary call option.

        For binary options (pays $1 if S > K, else $0), the Greeks are:
        - Delta: Rate of change of option value w.r.t. underlying price
        - Gamma: Rate of change of delta w.r.t. underlying price
        - Vega: Rate of change of option value w.r.t. volatility
        - Theta: Rate of change of option value w.r.t. time

        Args:
            spot_price: Current BTC price
            strike_price: Strike price
            time_to_expiry_hours: Hours until expiration
            implied_vol: Implied volatility as percentage (e.g., 65.5)
            risk_free_rate: Annual risk-free rate (default 0%)

        Returns:
            Dictionary with delta, gamma, vega, theta
        """
        if time_to_expiry_hours <= 0 or implied_vol is None or implied_vol <= 0:
            return None

        try:
            # Convert to appropriate units
            T = time_to_expiry_hours / (365.25 * 24)  # Years
            sigma = implied_vol / 100.0  # Decimal
            S = spot_price
            K = strike_price
            r = risk_free_rate

            # Calculate d1 and d2
            d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
            d2 = d1 - sigma * np.sqrt(T)

            # Standard normal PDF and CDF
            phi_d2 = stats.norm.pdf(d2)  # PDF at d2
            Phi_d2 = stats.norm.cdf(d2)  # CDF at d2

            # Binary option Greeks
            exp_rT = np.exp(-r * T)

            # Delta: ∂V/∂S = φ(d2)/(S·σ·√T) · e^(-rT)
            delta = (phi_d2 / (S * sigma * np.sqrt(T))) * exp_rT

            # Gamma: ∂²V/∂S² = -d1·φ(d2)/(S²·σ²·T) · e^(-rT)
            gamma = -(d1 * phi_d2) / (S ** 2 * sigma ** 2 * T) * exp_rT

            # Vega: ∂V/∂σ = -d1·φ(d2)/σ · e^(-rT)
            # Note: Vega is typically expressed per 1% change in volatility
            vega = -(d1 * phi_d2) / sigma * exp_rT / 100.0  # Divide by 100 for 1% change

            # Theta: ∂V/∂t = [φ(d2)/(2√T)]·[(ln(S/K)/T)-(r-0.5σ²)] + r·e^(-rT)·Φ(d2)
            # Convert to per-day theta
            sqrt_T = np.sqrt(T)
            ln_S_K = np.log(S / K)
            theta_term1 = (phi_d2 / (2 * sqrt_T)) * ((ln_S_K / T) - (r - 0.5 * sigma ** 2))
            theta_term2 = r * exp_rT * Phi_d2
            theta_yearly = theta_term1 + theta_term2
            theta = theta_yearly / 365.25  # Convert to per-day

            # Vanna: ∂²V/∂S∂σ = φ(d2)·d1/(S·σ²·√T) · e^(-rT)
            vanna = (phi_d2 * d1) / (S * sigma ** 2 * sqrt_T) * exp_rT

            # Volga (Vomma): ∂²V/∂σ² = -d1·d2·φ(d2)/σ² · e^(-rT)
            # Expressed per 1% change in volatility
            volga = -(d1 * d2 * phi_d2) / (sigma ** 2) * exp_rT / 10000.0  # Divide by 10000 for 1% change

            return {
                'delta': float(delta),
                'gamma': float(gamma),
                'vega': float(vega),
                'theta': float(theta),
                'vanna': float(vanna),
                'volga': float(volga)
            }

        except Exception as e:
            return None

    def calculate_implied_volatility_rank(
        self,
        current_iv: float,
        ticker: str,
        lookback_days: int = 30
    ) -> Optional[float]:
        """
        Calculate Implied Volatility Rank.

        IV Rank = (Current IV - Min IV) / (Max IV - Min IV) * 100

        Args:
            current_iv: Current implied volatility
            ticker: Market ticker
            lookback_days: Number of days to look back for min/max

        Returns:
            IV Rank (0-100)
        """
        if current_iv is None:
            return None

        try:
            lookback_start = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(days=lookback_days)

            with db_manager.get_session() as session:
                # Get historical IV data for this ticker
                snapshots = session.query(KalshiOrderbookSnapshot).filter(
                    KalshiOrderbookSnapshot.ticker == ticker,
                    KalshiOrderbookSnapshot.timestamp >= lookback_start,
                    KalshiOrderbookSnapshot.implied_volatility.isnot(None)
                ).all()

                if len(snapshots) < 10:  # Need sufficient history
                    return None

                ivs = [float(s.implied_volatility) for s in snapshots]
                ivs.append(current_iv)  # Include current

                min_iv = min(ivs)
                max_iv = max(ivs)

                if max_iv == min_iv:
                    return 50.0  # Middle if no range

                iv_rank = ((current_iv - min_iv) / (max_iv - min_iv)) * 100
                return float(iv_rank)

        except Exception as e:
            return None


def main():
    """Test the volatility calculator"""
    calc = VolatilityCalculator()

    # Test realized volatility
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    rv_30min = calc.calculate_realized_vol_from_db(now, 30)
    rv_10min = calc.calculate_realized_vol_from_db(now, 10)

    print(f"Realized Vol (30min): {rv_30min:.2f}%" if rv_30min else "No data")
    print(f"Realized Vol (10min): {rv_10min:.2f}%" if rv_10min else "No data")

    # Test implied volatility
    iv = calc.estimate_implied_volatility_from_kalshi(
        spot_price=90000,
        strike_price=90500,
        market_price=45,  # 45% probability
        time_to_expiry_hours=1.0
    )
    print(f"Implied Vol: {iv:.2f}%" if iv else "No solution")


if __name__ == "__main__":
    main()
