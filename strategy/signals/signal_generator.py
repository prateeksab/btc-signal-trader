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

        # Thresholds
        self.strong_imbalance_threshold = 60  # ±60%
        self.moderate_imbalance_threshold = 30  # ±30%
        self.strong_cvd_threshold = 10  # 10 BTC

        logger.info("SignalGenerator initialized")

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
        price: Optional[float] = None
    ) -> TradingSignal:
        """
        Generate trading signal from current market data.

        Args:
            cvd: Current CVD value
            orderbook_imbalance: Current orderbook imbalance (-100 to +100)
            price: Current BTC price

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

        # Collect signal factors
        reasons = []
        warnings = []

        # Get current values
        current_cvd = self.cvd_history[-1] if self.cvd_history else None
        current_imbalance = self.imbalance_history[-1] if self.imbalance_history else None
        current_price = self.price_history[-1] if self.price_history else None

        # Analyze CVD trend
        cvd_trend = self.get_cvd_trend()

        # Analyze orderbook
        orderbook_signal = self.get_orderbook_signal()

        # Check for divergences
        divergence = self.check_divergence()
        if divergence:
            warnings.append(divergence)

        # Determine signal type
        signal_type = SignalType.NEUTRAL

        # STRONG BUY conditions
        if (cvd_trend == 'bullish' and orderbook_signal == 'strong_buy'):
            signal_type = SignalType.STRONG_BUY
            reasons.append(f"Bullish CVD trend")
            reasons.append(f"Strong orderbook buying pressure ({current_imbalance:+.1f}%)")

        # BUY conditions
        elif (cvd_trend == 'bullish' and orderbook_signal in ['buy', 'neutral']) or \
             (cvd_trend == 'neutral' and orderbook_signal == 'strong_buy'):
            signal_type = SignalType.BUY
            if cvd_trend == 'bullish':
                reasons.append(f"Bullish CVD trend")
            if orderbook_signal in ['buy', 'strong_buy']:
                reasons.append(f"Orderbook buying pressure ({current_imbalance:+.1f}%)")

        # STRONG SELL conditions
        elif (cvd_trend == 'bearish' and orderbook_signal == 'strong_sell'):
            signal_type = SignalType.STRONG_SELL
            reasons.append(f"Bearish CVD trend")
            reasons.append(f"Strong orderbook selling pressure ({current_imbalance:+.1f}%)")

        # SELL conditions
        elif (cvd_trend == 'bearish' and orderbook_signal in ['sell', 'neutral']) or \
             (cvd_trend == 'neutral' and orderbook_signal == 'strong_sell'):
            signal_type = SignalType.SELL
            if cvd_trend == 'bearish':
                reasons.append(f"Bearish CVD trend")
            if orderbook_signal in ['sell', 'strong_sell']:
                reasons.append(f"Orderbook selling pressure ({current_imbalance:+.1f}%)")

        # Calculate confidence
        confidence, confidence_score = self.calculate_confidence(signal_type, reasons)

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
