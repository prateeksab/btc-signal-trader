#!/usr/bin/env python3
"""
Verify Live Data Collection

Query and display data collected from live signal trader.
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.storage.database import db_manager
from data.storage.models import (
    Trade, OrderbookSnapshot, CVDSnapshot, Signal
)

print("=" * 80)
print("Database Verification - Live Data Collection")
print("=" * 80)

with db_manager.get_session() as session:
    # Query trades
    trades = session.query(Trade).order_by(Trade.timestamp).all()
    print(f"\n📊 Trades: {len(trades)}")
    if trades:
        print(f"   First trade: {trades[0].timestamp} | ${trades[0].price} | {trades[0].side} | {trades[0].size} BTC")
        print(f"   Last trade:  {trades[-1].timestamp} | ${trades[-1].price} | {trades[-1].side} | {trades[-1].size} BTC")
        print(f"   CVD progression: {trades[0].cvd_at_trade} → {trades[-1].cvd_at_trade}")

    # Query orderbook snapshots
    orderbooks = session.query(OrderbookSnapshot).all()
    print(f"\n📊 Orderbook Snapshots: {len(orderbooks)}")
    for ob in orderbooks:
        print(f"   Timestamp: {ob.timestamp}")
        print(f"   Best Bid: ${ob.best_bid} | Best Ask: ${ob.best_ask}")
        print(f"   Spread: ${ob.spread} ({ob.spread_pct}%)")
        print(f"   Imbalance: {ob.imbalance}%")
        print(f"   Bid Volume: {ob.total_bid_volume} | Ask Volume: {ob.total_ask_volume}")
        print(f"   Bids in JSON: {len(ob.bids_json)} levels")
        print(f"   Asks in JSON: {len(ob.asks_json)} levels")

    # Query CVD snapshots
    cvds = session.query(CVDSnapshot).order_by(CVDSnapshot.timestamp).all()
    print(f"\n📊 CVD Snapshots: {len(cvds)}")
    for cvd in cvds:
        print(f"   {cvd.timestamp} | CVD: {cvd.cvd} BTC")

    # Query signals
    signals = session.query(Signal).order_by(Signal.timestamp).all()
    print(f"\n📊 Signals Generated: {len(signals)}")
    for i, sig in enumerate(signals, 1):
        print(f"\n   Signal #{i}:")
        print(f"   Time: {sig.timestamp}")
        print(f"   Type: {sig.signal_type} ({sig.confidence_level})")
        print(f"   Confidence: {sig.confidence_score}%")
        print(f"   Price: ${sig.price}")
        print(f"   CVD: {sig.cvd} ({sig.cvd_trend})")
        print(f"   Orderbook Imbalance: {sig.orderbook_imbalance}%")
        if sig.reasons:
            print(f"   Reasons: {sig.reasons}")
        if sig.warnings:
            print(f"   Warnings: {sig.warnings}")
        print(f"   Metadata: {sig.extra_metadata}")

print("\n" + "=" * 80)
print("✅ End-to-End Test Complete!")
print("=" * 80)
