"""
Real-Time Trading Signal Generator

Combines multiple data sources to generate trading signals:
- CVD (Cumulative Volume Delta) from Coinbase/Kraken trades
- Orderbook Imbalance from Kraken orderbook
- Price action
"""

import logging
from datetime import datetime
from enum import Enum
from typing import Dict, Optional, List
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SignalType(Enum):
    """Trading signal types"""
    STRONG_BUY = "STRONG_BUY"
    BUY = "BUY"
    NEUTRAL = "NEUTRAL"
    SELL = "SELL"
    STRONG_SELL = "STRONG_SELL"


class ConfidenceLevel(Enum):
    """Signal confidence levels"""
    VERY_HIGH = "VERY_HIGH"  # 90-100%
    HIGH = "HIGH"            # 70-90%
    MEDIUM = "MEDIUM"        # 50-70%
    LOW = "LOW"              # 30-50%
    VERY_LOW = "VERY_LOW"    # 0-30%


@dataclass
class TradingSignal:
    """Trading signal with metadata"""
    signal_type: SignalType
    confidence: ConfidenceLevel
    confidence_score: float  # 0-100
    timestamp: datetime

    # Contributing factors
    cvd_value: Optional[float] = None
    cvd_trend: Optional[str] = None
    orderbook_imbalance: Optional[float] = None
    price: Optional[float] = None

    # New signal factors
    spread_pct: Optional[float] = None
    spread_signal: Optional[str] = None
    trade_intensity_tps: Optional[float] = None
    trade_intensity_aggression: Optional[float] = None
    intensity_signal: Optional[str] = None
    futures_basis_pct: Optional[float] = None
    basis_signal: Optional[str] = None
    funding_rate: Optional[float] = None
    funding_rate_signal: Optional[str] = None

    # Signal details
    reasons: List[str] = None
    warnings: List[str] = None

    def __post_init__(self):
        if self.reasons is None:
            self.reasons = []
        if self.warnings is None:
            self.warnings = []


class SignalGenerator:
    """
    Generates trading signals from multiple data sources.

    Combines:
    - CVD trend (bullish/bearish momentum)
    - Orderbook imbalance (buying/selling pressure)
    - Price action
    - Spread (market liquidity)
    - Trade intensity (market activity)
    - Futures basis (contango/backwardation)
    - Funding rate (sentiment indicator)
    """

    def __init__(self):
        # CVD tracking
        self.cvd_history: List[float] = []
        self.cvd_window = 20  # Track last 20 CVD values

        # Orderbook tracking
        self.imbalance_history: List[float] = []
        self.imbalance_window = 10

        # Price tracking
        self.price_history: List[float] = []
        self.price_window = 20

        # Spread tracking
        self.spread_history: List[float] = []
        self.spread_window = 10

        # Trade intensity tracking
        self.intensity_history: List[Dict] = []
        self.intensity_window = 10

        # Futures basis tracking
        self.basis_history: List[float] = []
        self.basis_window = 10

        # Funding rate tracking
        self.funding_rate_history: List[float] = []
        self.funding_rate_window = 10

        # Thresholds - Orderbook
        self.strong_imbalance_threshold = 60  # ±60%
        self.moderate_imbalance_threshold = 30  # ±30%

        # Thresholds - CVD
        self.strong_cvd_threshold = 10  # 10 BTC

        # Thresholds - Spread (basis points)
        self.tight_spread_threshold = 0.01  # 0.01% = tight spread (good liquidity)
        self.wide_spread_threshold = 0.05   # 0.05% = wide spread (poor liquidity)

        # Thresholds - Trade Intensity
        self.high_intensity_threshold = 5.0   # TPS
        self.low_intensity_threshold = 1.0    # TPS
        self.aggressive_buying_threshold = 0.3  # Aggression score
        self.aggressive_selling_threshold = -0.3

        # Thresholds - Futures Basis
        self.strong_contango_threshold = 0.5    # 0.5% basis
        self.strong_backwardation_threshold = -0.5

        # Thresholds - Funding Rate
        self.high_funding_threshold = 0.0001     # 0.01% per 8h
        self.negative_funding_threshold = -0.0001

        logger.info("SignalGenerator initialized with 6 signal sources")

    def update_cvd(self, cvd_value: float):
        """
        Update CVD value and maintain history.

        Args:
            cvd_value: Current cumulative volume delta
        """
        self.cvd_history.append(cvd_value)

        # Keep only recent history
        if len(self.cvd_history) > self.cvd_window:
            self.cvd_history.pop(0)

    def update_orderbook(self, imbalance: float):
        """
        Update orderbook imbalance and maintain history.

        Args:
            imbalance: Orderbook imbalance (-100 to +100)
        """
        self.imbalance_history.append(imbalance)

        # Keep only recent history
        if len(self.imbalance_history) > self.imbalance_window:
            self.imbalance_history.pop(0)

    def update_price(self, price: float):
        """
        Update price and maintain history.

        Args:
            price: Current BTC price
        """
        self.price_history.append(price)

        # Keep only recent history
        if len(self.price_history) > self.price_window:
            self.price_history.pop(0)

    def update_spread(self, spread_pct: float):
        """
        Update spread percentage and maintain history.

        Args:
            spread_pct: Spread as percentage of price
        """
        self.spread_history.append(spread_pct)

        # Keep only recent history
        if len(self.spread_history) > self.spread_window:
            self.spread_history.pop(0)

    def update_trade_intensity(self, intensity_data: Dict):
        """
        Update trade intensity metrics and maintain history.

        Args:
            intensity_data: Dict with keys: tps, aggression_score, buy_sell_ratio
        """
        self.intensity_history.append(intensity_data)

        # Keep only recent history
        if len(self.intensity_history) > self.intensity_window:
            self.intensity_history.pop(0)

    def update_futures_basis(self, basis_pct: float):
        """
        Update futures basis percentage and maintain history.

        Args:
            basis_pct: Futures-spot basis as percentage
        """
        self.basis_history.append(basis_pct)

        # Keep only recent history
        if len(self.basis_history) > self.basis_window:
            self.basis_history.pop(0)

    def update_funding_rate(self, funding_rate: float):
        """
        Update funding rate and maintain history.

        Args:
            funding_rate: Current funding rate
        """
        self.funding_rate_history.append(funding_rate)

        # Keep only recent history
        if len(self.funding_rate_history) > self.funding_rate_window:
            self.funding_rate_history.pop(0)

    def get_cvd_trend(self) -> Optional[str]:
        """
        Analyze CVD trend.

        Returns:
            'bullish', 'bearish', or 'neutral'
        """
        if len(self.cvd_history) < 5:
            return None

        # Compare recent CVD to earlier CVD
        recent_avg = sum(self.cvd_history[-5:]) / 5
        earlier_avg = sum(self.cvd_history[-10:-5]) / 5 if len(self.cvd_history) >= 10 else recent_avg

        change = recent_avg - earlier_avg

        if change > self.strong_cvd_threshold:
            return 'bullish'
        elif change < -self.strong_cvd_threshold:
            return 'bearish'
        else:
            return 'neutral'

    def get_orderbook_signal(self) -> Optional[str]:
        """
        Analyze orderbook imbalance for signal.

        Returns:
            'strong_buy', 'buy', 'neutral', 'sell', 'strong_sell'
        """
        if not self.imbalance_history:
            return None

        current_imbalance = self.imbalance_history[-1]

        if current_imbalance > self.strong_imbalance_threshold:
            return 'strong_buy'
        elif current_imbalance > self.moderate_imbalance_threshold:
            return 'buy'
        elif current_imbalance < -self.strong_imbalance_threshold:
            return 'strong_sell'
        elif current_imbalance < -self.moderate_imbalance_threshold:
            return 'sell'
        else:
            return 'neutral'

    def check_divergence(self) -> Optional[str]:
        """
        Check for divergence between CVD and orderbook.

        Returns:
            Warning message if divergence detected
        """
        if not self.cvd_history or not self.imbalance_history:
            return None

        cvd_trend = self.get_cvd_trend()
        current_imbalance = self.imbalance_history[-1]

        # Bullish CVD but bearish orderbook
        if cvd_trend == 'bullish' and current_imbalance < -self.moderate_imbalance_threshold:
            return "DIVERGENCE: Bullish CVD but bearish orderbook - potential reversal"

        # Bearish CVD but bullish orderbook
        if cvd_trend == 'bearish' and current_imbalance > self.moderate_imbalance_threshold:
            return "DIVERGENCE: Bearish CVD but bullish orderbook - potential reversal"

        return None

    def get_spread_signal(self) -> Optional[str]:
        """
        Analyze spread for liquidity conditions.

        Returns:
            'tight' (good liquidity), 'normal', 'wide' (poor liquidity)
        """
        if not self.spread_history:
            return None

        current_spread = self.spread_history[-1]

        if current_spread <= self.tight_spread_threshold:
            return 'tight'
        elif current_spread >= self.wide_spread_threshold:
            return 'wide'
        else:
            return 'normal'

    def get_intensity_signal(self) -> Optional[str]:
        """
        Analyze trade intensity for market activity.

        Returns:
            'high_buying', 'moderate_buying', 'neutral', 'moderate_selling', 'high_selling'
        """
        if not self.intensity_history:
            return None

        current = self.intensity_history[-1]
        tps = current.get('tps', 0)
        aggression = current.get('aggression_score', 0)

        # High intensity with aggressive buying
        if tps > self.high_intensity_threshold and aggression > self.aggressive_buying_threshold:
            return 'high_buying'

        # Moderate intensity with buying
        elif aggression > self.aggressive_buying_threshold:
            return 'moderate_buying'

        # High intensity with aggressive selling
        elif tps > self.high_intensity_threshold and aggression < self.aggressive_selling_threshold:
            return 'high_selling'

        # Moderate intensity with selling
        elif aggression < self.aggressive_selling_threshold:
            return 'moderate_selling'

        else:
            return 'neutral'

    def get_basis_signal(self) -> Optional[str]:
        """
        Analyze futures basis for market sentiment.

        Returns:
            'strong_contango', 'contango', 'neutral', 'backwardation', 'strong_backwardation'
        """
        if not self.basis_history:
            return None

        current_basis = self.basis_history[-1]

        if current_basis >= self.strong_contango_threshold:
            return 'strong_contango'
        elif current_basis > 0:
            return 'contango'
        elif current_basis <= self.strong_backwardation_threshold:
            return 'strong_backwardation'
        elif current_basis < 0:
            return 'backwardation'
        else:
            return 'neutral'

    def get_funding_rate_signal(self) -> Optional[str]:
        """
        Analyze funding rate for sentiment.

        Returns:
            'high_long_pressure', 'moderate_long_pressure', 'neutral',
            'moderate_short_pressure', 'high_short_pressure'
        """
        if not self.funding_rate_history:
            return None

        current_rate = self.funding_rate_history[-1]

        if current_rate >= self.high_funding_threshold:
            return 'high_long_pressure'
        elif current_rate > 0:
            return 'moderate_long_pressure'
        elif current_rate <= self.negative_funding_threshold:
            return 'high_short_pressure'
        elif current_rate < 0:
            return 'moderate_short_pressure'
        else:
            return 'neutral'

    def calculate_confidence(self, signal_type: SignalType, reasons: List[str]) -> tuple:
        """
        Calculate confidence level and score based on signal strength.

        Returns:
            (ConfidenceLevel, score)
        """
        score = 0

        # Base score from number of confirming factors
        score += len(reasons) * 20

        # Bonus for strong signals
        if signal_type in [SignalType.STRONG_BUY, SignalType.STRONG_SELL]:
            score += 20

        # Bonus for trend alignment
        cvd_trend = self.get_cvd_trend()
        orderbook_signal = self.get_orderbook_signal()

        if cvd_trend and orderbook_signal:
            if (cvd_trend == 'bullish' and 'buy' in orderbook_signal) or \
               (cvd_trend == 'bearish' and 'sell' in orderbook_signal):
                score += 20

        # Cap at 100
        score = min(score, 100)

        # Determine confidence level
        if score >= 90:
            confidence = ConfidenceLevel.VERY_HIGH
        elif score >= 70:
            confidence = ConfidenceLevel.HIGH
        elif score >= 50:
            confidence = ConfidenceLevel.MEDIUM
        elif score >= 30:
            confidence = ConfidenceLevel.LOW
        else:
            confidence = ConfidenceLevel.VERY_LOW

        return confidence, score

    def generate_signal(
        self,
        cvd: Optional[float] = None,
        orderbook_imbalance: Optional[float] = None,
        price: Optional[float] = None,
        spread_pct: Optional[float] = None,
        trade_intensity: Optional[Dict] = None,
        futures_basis_pct: Optional[float] = None,
        funding_rate: Optional[float] = None
    ) -> TradingSignal:
        """
        Generate trading signal from current market data.

        Args:
            cvd: Current CVD value
            orderbook_imbalance: Current orderbook imbalance (-100 to +100)
            price: Current BTC price
            spread_pct: Bid-ask spread as percentage
            trade_intensity: Dict with tps, aggression_score, buy_sell_ratio
            futures_basis_pct: Futures-spot basis as percentage
            funding_rate: Current funding rate

        Returns:
            TradingSignal with type, confidence, and reasoning
        """
        # Update internal state
        if cvd is not None:
            self.update_cvd(cvd)
        if orderbook_imbalance is not None:
            self.update_orderbook(orderbook_imbalance)
        if price is not None:
            self.update_price(price)
        if spread_pct is not None:
            self.update_spread(spread_pct)
        if trade_intensity is not None:
            self.update_trade_intensity(trade_intensity)
        if futures_basis_pct is not None:
            self.update_futures_basis(futures_basis_pct)
        if funding_rate is not None:
            self.update_funding_rate(funding_rate)

        # Collect signal factors
        reasons = []
        warnings = []

        # Get current values
        current_cvd = self.cvd_history[-1] if self.cvd_history else None
        current_imbalance = self.imbalance_history[-1] if self.imbalance_history else None
        current_price = self.price_history[-1] if self.price_history else None
        current_spread = self.spread_history[-1] if self.spread_history else None
        current_intensity = self.intensity_history[-1] if self.intensity_history else None
        current_basis = self.basis_history[-1] if self.basis_history else None
        current_funding = self.funding_rate_history[-1] if self.funding_rate_history else None

        # Analyze all signals
        cvd_trend = self.get_cvd_trend()
        orderbook_signal = self.get_orderbook_signal()
        spread_signal = self.get_spread_signal()
        intensity_signal = self.get_intensity_signal()
        basis_signal = self.get_basis_signal()
        funding_signal = self.get_funding_rate_signal()

        # Check for divergences
        divergence = self.check_divergence()
        if divergence:
            warnings.append(divergence)

        # Determine signal type using scoring system
        signal_type = SignalType.NEUTRAL
        buy_score = 0
        sell_score = 0

        # Score CVD (weight: 2 points per direction)
        if cvd_trend == 'bullish':
            buy_score += 2
            reasons.append(f"Bullish CVD trend")
        elif cvd_trend == 'bearish':
            sell_score += 2
            reasons.append(f"Bearish CVD trend")

        # Score Orderbook (weight: 3 points for strong, 1.5 for moderate)
        if orderbook_signal == 'strong_buy':
            buy_score += 3
            reasons.append(f"Strong orderbook buying pressure ({current_imbalance:+.1f}%)")
        elif orderbook_signal == 'buy':
            buy_score += 1.5
            reasons.append(f"Orderbook buying pressure ({current_imbalance:+.1f}%)")
        elif orderbook_signal == 'strong_sell':
            sell_score += 3
            reasons.append(f"Strong orderbook selling pressure ({current_imbalance:+.1f}%)")
        elif orderbook_signal == 'sell':
            sell_score += 1.5
            reasons.append(f"Orderbook selling pressure ({current_imbalance:+.1f}%)")

        # Score Trade Intensity (weight: 2 points for high, 1 for moderate)
        if intensity_signal == 'high_buying':
            buy_score += 2
            reasons.append(f"High intensity buying pressure")
        elif intensity_signal == 'moderate_buying':
            buy_score += 1
            reasons.append(f"Moderate buying intensity")
        elif intensity_signal == 'high_selling':
            sell_score += 2
            reasons.append(f"High intensity selling pressure")
        elif intensity_signal == 'moderate_selling':
            sell_score += 1
            reasons.append(f"Moderate selling intensity")

        # Score Futures Basis (weight: 1.5 points)
        # Contango = bullish short-term, backwardation = bearish short-term
        if basis_signal == 'strong_contango':
            buy_score += 1.5
            reasons.append(f"Strong contango ({current_basis:+.2f}%) - bullish sentiment")
        elif basis_signal == 'contango':
            buy_score += 0.75
            reasons.append(f"Contango ({current_basis:+.2f}%)")
        elif basis_signal == 'strong_backwardation':
            sell_score += 1.5
            reasons.append(f"Strong backwardation ({current_basis:+.2f}%) - bearish sentiment")
        elif basis_signal == 'backwardation':
            sell_score += 0.75
            reasons.append(f"Backwardation ({current_basis:+.2f}%)")

        # Score Funding Rate (weight: 1.5 points)
        # High funding = contrarian bearish (longs overextended)
        # Negative funding = contrarian bullish (shorts overextended)
        if funding_signal == 'high_long_pressure':
            sell_score += 1.5
            warnings.append(f"⚠️  High funding rate ({current_funding:.6f}) - potential long squeeze")
        elif funding_signal == 'moderate_long_pressure':
            sell_score += 0.75
        elif funding_signal == 'high_short_pressure':
            buy_score += 1.5
            reasons.append(f"Negative funding rate ({current_funding:.6f}) - shorts paying longs")
        elif funding_signal == 'moderate_short_pressure':
            buy_score += 0.75
            reasons.append(f"Shorts under pressure")

        # Spread warning (doesn't affect signal but adds warning)
        if spread_signal == 'wide':
            warnings.append(f"⚠️  Wide spread ({current_spread:.4f}%) - poor liquidity")
        elif spread_signal == 'tight':
            reasons.append(f"Tight spread ({current_spread:.4f}%) - good liquidity")

        # Determine signal type based on scores
        # Thresholds: Strong = 5+, Moderate = 2.5+
        net_score = buy_score - sell_score

        if buy_score >= 5 and net_score >= 3:
            signal_type = SignalType.STRONG_BUY
        elif buy_score >= 2.5 and net_score >= 1.5:
            signal_type = SignalType.BUY
        elif sell_score >= 5 and net_score <= -3:
            signal_type = SignalType.STRONG_SELL
        elif sell_score >= 2.5 and net_score <= -1.5:
            signal_type = SignalType.SELL
        else:
            signal_type = SignalType.NEUTRAL

        # Calculate confidence
        confidence, confidence_score = self.calculate_confidence(signal_type, reasons)

        # Extract intensity metrics if available
        intensity_tps = current_intensity.get('tps') if current_intensity else None
        intensity_aggression = current_intensity.get('aggression_score') if current_intensity else None

        # Create signal
        signal = TradingSignal(
            signal_type=signal_type,
            confidence=confidence,
            confidence_score=confidence_score,
            timestamp=datetime.now(),
            cvd_value=current_cvd,
            cvd_trend=cvd_trend,
            orderbook_imbalance=current_imbalance,
            price=current_price,
            spread_pct=current_spread,
            spread_signal=spread_signal,
            trade_intensity_tps=intensity_tps,
            trade_intensity_aggression=intensity_aggression,
            intensity_signal=intensity_signal,
            futures_basis_pct=current_basis,
            basis_signal=basis_signal,
            funding_rate=current_funding,
            funding_rate_signal=funding_signal,
            reasons=reasons,
            warnings=warnings
        )

        return signal

    def print_signal(self, signal: TradingSignal):
        """
        Print formatted signal to console.

        Args:
            signal: TradingSignal to display
        """
        # Color codes
        colors = {
            SignalType.STRONG_BUY: "\033[92m",  # Bright green
            SignalType.BUY: "\033[32m",         # Green
            SignalType.NEUTRAL: "\033[37m",     # White
            SignalType.SELL: "\033[91m",        # Red
            SignalType.STRONG_SELL: "\033[31m", # Dark red
        }
        reset = "\033[0m"

        color = colors.get(signal.signal_type, reset)

        print(f"\n{'='*80}")
        print(f"{color}{signal.signal_type.value}{reset} | "
              f"Confidence: {signal.confidence.value} ({signal.confidence_score:.0f}%)")
        print(f"Time: {signal.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*80}")

        # Market data
        if signal.price:
            print(f"Price: ${signal.price:,.2f}")
        if signal.cvd_value is not None:
            print(f"CVD: {signal.cvd_value:+,.4f} BTC ({signal.cvd_trend or 'unknown'} trend)")
        if signal.orderbook_imbalance is not None:
            print(f"Orderbook Imbalance: {signal.orderbook_imbalance:+.2f}%")
        if signal.spread_pct is not None:
            print(f"Spread: {signal.spread_pct:.4f}% ({signal.spread_signal or 'unknown'})")
        if signal.trade_intensity_tps is not None:
            print(f"Trade Intensity: {signal.trade_intensity_tps:.2f} TPS (aggression: {signal.trade_intensity_aggression:+.2f})")
        if signal.futures_basis_pct is not None:
            print(f"Futures Basis: {signal.futures_basis_pct:+.2f}% ({signal.basis_signal or 'unknown'})")
        if signal.funding_rate is not None:
            print(f"Funding Rate: {signal.funding_rate:.6f} ({signal.funding_rate_signal or 'unknown'})")

        # Reasons
        if signal.reasons:
            print(f"\nReasons:")
            for reason in signal.reasons:
                print(f"  ✓ {reason}")

        # Warnings
        if signal.warnings:
            print(f"\n⚠️  Warnings:")
            for warning in signal.warnings:
                print(f"  ! {warning}")

        print(f"{'='*80}\n")


def main():
    """Example usage of SignalGenerator"""
    generator = SignalGenerator()

    # Simulate market data updates
    print("Signal Generator Example\n")

    # Scenario 1: Strong buying
    print("Scenario 1: Strong Buying Signal")
    signal = generator.generate_signal(
        cvd=25.5,
        orderbook_imbalance=75.0,
        price=89850.00
    )
    generator.print_signal(signal)

    # Scenario 2: Moderate selling
    print("Scenario 2: Moderate Selling Signal")
    signal = generator.generate_signal(
        cvd=-15.2,
        orderbook_imbalance=-45.0,
        price=89820.00
    )
    generator.print_signal(signal)

    # Scenario 3: Divergence
    print("Scenario 3: Divergence Warning")
    for i in range(10):
        generator.generate_signal(
            cvd=30 + i,  # Increasing CVD (bullish)
            orderbook_imbalance=-50 - i,  # Decreasing imbalance (bearish)
            price=89800 - i * 10
        )

    signal = generator.generate_signal(
        cvd=40,
        orderbook_imbalance=-70,
        price=89700
    )
    generator.print_signal(signal)


if __name__ == "__main__":
    main()
