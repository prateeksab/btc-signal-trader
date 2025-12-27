#!/usr/bin/env python3
"""
Test Integrated Signal System

Tests that all 6 signals are properly integrated:
1. CVD
2. Orderbook Imbalance
3. Spread
4. Trade Intensity
5. Futures Basis
6. Funding Rate
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from strategy.signals.signal_generator import SignalGenerator, SignalType


def test_all_signals_bullish():
    """Test with all signals showing bullish conditions"""
    print("\n" + "="*80)
    print("TEST 1: All Signals Bullish")
    print("="*80)

    generator = SignalGenerator()

    # Scenario: Strong bullish market
    signal = generator.generate_signal(
        cvd=50.0,                    # Strong bullish CVD
        orderbook_imbalance=70.0,    # Strong buy imbalance
        price=95000.00,
        spread_pct=0.005,            # Tight spread (good liquidity)
        trade_intensity={
            'tps': 6.0,              # High trading activity
            'aggression_score': 0.5, # Aggressive buying
            'buy_sell_ratio': 0.75
        },
        futures_basis_pct=0.6,       # Strong contango
        funding_rate=-0.0002         # Negative funding (shorts paying)
    )

    generator.print_signal(signal)

    assert signal.signal_type == SignalType.STRONG_BUY, f"Expected STRONG_BUY, got {signal.signal_type}"
    print("✅ Test passed: Generated STRONG_BUY signal\n")


def test_all_signals_bearish():
    """Test with all signals showing bearish conditions"""
    print("\n" + "="*80)
    print("TEST 2: All Signals Bearish")
    print("="*80)

    generator = SignalGenerator()

    # Scenario: Strong bearish market
    signal = generator.generate_signal(
        cvd=-50.0,                   # Strong bearish CVD
        orderbook_imbalance=-70.0,   # Strong sell imbalance
        price=95000.00,
        spread_pct=0.008,            # Normal spread
        trade_intensity={
            'tps': 6.0,              # High trading activity
            'aggression_score': -0.5, # Aggressive selling
            'buy_sell_ratio': 0.25
        },
        futures_basis_pct=-0.6,      # Strong backwardation
        funding_rate=0.0002          # High funding (longs paying)
    )

    generator.print_signal(signal)

    assert signal.signal_type == SignalType.STRONG_SELL, f"Expected STRONG_SELL, got {signal.signal_type}"
    print("✅ Test passed: Generated STRONG_SELL signal\n")


def test_mixed_signals():
    """Test with mixed signals"""
    print("\n" + "="*80)
    print("TEST 3: Mixed Signals (Should be NEUTRAL)")
    print("="*80)

    generator = SignalGenerator()

    # Scenario: Conflicting signals
    signal = generator.generate_signal(
        cvd=15.0,                    # Bullish CVD
        orderbook_imbalance=-40.0,   # Bearish orderbook
        price=95000.00,
        spread_pct=0.01,             # Tight spread
        trade_intensity={
            'tps': 2.0,              # Low activity
            'aggression_score': 0.1,
            'buy_sell_ratio': 0.55
        },
        futures_basis_pct=0.2,       # Slight contango
        funding_rate=0.00005         # Slight positive funding
    )

    generator.print_signal(signal)

    assert signal.signal_type == SignalType.NEUTRAL, f"Expected NEUTRAL, got {signal.signal_type}"
    print("✅ Test passed: Generated NEUTRAL signal\n")


def test_partial_data():
    """Test with only some signals available"""
    print("\n" + "="*80)
    print("TEST 4: Partial Data (CVD and Orderbook only)")
    print("="*80)

    generator = SignalGenerator()

    # Scenario: Only basic signals available
    signal = generator.generate_signal(
        cvd=25.0,
        orderbook_imbalance=65.0,
        price=95000.00,
        # No spread, intensity, basis, or funding rate
    )

    generator.print_signal(signal)

    # Should still generate a signal with available data
    assert signal.signal_type in [SignalType.BUY, SignalType.STRONG_BUY], \
        f"Expected BUY or STRONG_BUY, got {signal.signal_type}"
    print("✅ Test passed: Generated signal with partial data\n")


def test_signal_fields_populated():
    """Test that all signal fields are properly populated"""
    print("\n" + "="*80)
    print("TEST 5: Verify All Fields Populated")
    print("="*80)

    generator = SignalGenerator()

    # Build up history first (need at least 5 values for CVD trend)
    for i in range(10):
        generator.generate_signal(
            cvd=10.0 + i * 2.0,  # Increasing CVD
            orderbook_imbalance=50.0,
            price=95000.00,
            spread_pct=0.006,
            trade_intensity={
                'tps': 4.5,
                'aggression_score': 0.35,
                'buy_sell_ratio': 0.68
            },
            futures_basis_pct=0.4,
            funding_rate=-0.00008
        )

    # Now generate final signal with enough history
    signal = generator.generate_signal(
        cvd=30.0,
        orderbook_imbalance=50.0,
        price=95000.00,
        spread_pct=0.006,
        trade_intensity={
            'tps': 4.5,
            'aggression_score': 0.35,
            'buy_sell_ratio': 0.68
        },
        futures_basis_pct=0.4,
        funding_rate=-0.00008
    )

    # Check all fields are populated
    assert signal.cvd_value == 30.0, "CVD not set"
    assert signal.orderbook_imbalance == 50.0, "Orderbook imbalance not set"
    assert signal.price == 95000.00, "Price not set"
    assert signal.spread_pct == 0.006, "Spread not set"
    assert signal.trade_intensity_tps == 4.5, "TPS not set"
    assert signal.trade_intensity_aggression == 0.35, "Aggression not set"
    assert signal.futures_basis_pct == 0.4, "Basis not set"
    assert signal.funding_rate == -0.00008, "Funding rate not set"

    # Check signal classifications are set
    assert signal.cvd_trend is not None, "CVD trend not set"
    assert signal.spread_signal is not None, "Spread signal not set"
    assert signal.intensity_signal is not None, "Intensity signal not set"
    assert signal.basis_signal is not None, "Basis signal not set"
    assert signal.funding_rate_signal is not None, "Funding rate signal not set"

    print("✅ All fields properly populated:")
    print(f"  - CVD: {signal.cvd_value} ({signal.cvd_trend})")
    print(f"  - Orderbook: {signal.orderbook_imbalance}%")
    print(f"  - Spread: {signal.spread_pct}% ({signal.spread_signal})")
    print(f"  - Intensity: {signal.trade_intensity_tps} TPS ({signal.intensity_signal})")
    print(f"  - Basis: {signal.futures_basis_pct}% ({signal.basis_signal})")
    print(f"  - Funding: {signal.funding_rate} ({signal.funding_rate_signal})")
    print("\n✅ Test passed: All fields populated correctly\n")


def main():
    """Run all tests"""
    print("\n" + "="*80)
    print("INTEGRATED SIGNAL SYSTEM TESTS")
    print("="*80)
    print("\nTesting integration of all 6 signal sources:")
    print("  1. CVD (Cumulative Volume Delta)")
    print("  2. Orderbook Imbalance")
    print("  3. Spread")
    print("  4. Trade Intensity")
    print("  5. Futures Basis")
    print("  6. Funding Rate")
    print("="*80)

    try:
        test_all_signals_bullish()
        test_all_signals_bearish()
        test_mixed_signals()
        test_partial_data()
        test_signal_fields_populated()

        print("\n" + "="*80)
        print("✅ ALL TESTS PASSED!")
        print("="*80)
        print("\nSignal integration successful! All 6 signals are working correctly.")
        print("="*80 + "\n")

    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}\n")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ ERROR: {e}\n")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
