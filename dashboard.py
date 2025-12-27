#!/usr/bin/env python3
"""
BTC Signal Trader Dashboard

Real-time visualization of all trading signals and market data.
Displays time series graphs for all 6 signal sources.
"""

import sys
import os
from datetime import datetime, timedelta, timezone
from decimal import Decimal

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from data.storage.database import db_manager
from data.storage.models import (
    Trade, OrderbookSnapshot, CVDSnapshot, Signal,
    TradeIntensity, FuturesBasis, FundingRateMetrics,
    SpotPriceSnapshot, KalshiOrderbookSnapshot, BtcPriceCandle
)


# Configure Streamlit page
st.set_page_config(
    page_title="BTC Signal Trader Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)


def get_cvd_data(hours=24):
    """Get CVD time series data"""
    with db_manager.get_session() as session:
        cutoff = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(hours=hours)

        # First try CVD snapshots
        cvd_data = session.query(CVDSnapshot).filter(
            CVDSnapshot.timestamp >= cutoff
        ).order_by(CVDSnapshot.timestamp).all()

        if cvd_data:
            return pd.DataFrame([{
                'timestamp': d.timestamp,
                'cvd': float(d.cvd)
            } for d in cvd_data])

        # Fallback to trades
        trades = session.query(Trade).filter(
            Trade.timestamp >= cutoff,
            Trade.cvd_at_trade.isnot(None)
        ).order_by(Trade.timestamp).all()

        return pd.DataFrame([{
            'timestamp': t.timestamp,
            'cvd': float(t.cvd_at_trade)
        } for t in trades])


def get_orderbook_data(hours=24):
    """Get orderbook imbalance time series data"""
    with db_manager.get_session() as session:
        cutoff = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(hours=hours)

        data = session.query(OrderbookSnapshot).filter(
            OrderbookSnapshot.timestamp >= cutoff
        ).order_by(OrderbookSnapshot.timestamp).all()

        return pd.DataFrame([{
            'timestamp': d.timestamp,
            'imbalance': float(d.imbalance),
            'spread_pct': float(d.spread_pct) if d.spread_pct else None,
            'best_bid': float(d.best_bid) if d.best_bid else None,
            'best_ask': float(d.best_ask) if d.best_ask else None
        } for d in data])


def get_trade_intensity_data(hours=24):
    """Get trade intensity time series data"""
    with db_manager.get_session() as session:
        cutoff = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(hours=hours)

        data = session.query(TradeIntensity).filter(
            TradeIntensity.timestamp >= cutoff
        ).order_by(TradeIntensity.timestamp).all()

        return pd.DataFrame([{
            'timestamp': d.timestamp,
            'tps_60s': float(d.trades_per_sec_60s),
            'aggression_score': float(d.aggression_score),
            'buy_sell_ratio': float(d.buy_sell_ratio)
        } for d in data])


def get_futures_basis_data(hours=24):
    """Get futures basis time series data"""
    with db_manager.get_session() as session:
        cutoff = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(hours=hours)

        data = session.query(FuturesBasis).filter(
            FuturesBasis.timestamp >= cutoff
        ).order_by(FuturesBasis.timestamp).all()

        return pd.DataFrame([{
            'timestamp': d.timestamp,
            'basis_pct': float(d.basis_pct) if d.basis_pct else None,
            'spot_price': float(d.spot_price),
            'futures_price': float(d.futures_price)
        } for d in data])


def get_funding_rate_data(hours=24):
    """Get funding rate time series data"""
    with db_manager.get_session() as session:
        cutoff = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(hours=hours)

        data = session.query(FundingRateMetrics).filter(
            FundingRateMetrics.timestamp >= cutoff
        ).order_by(FundingRateMetrics.timestamp).all()

        return pd.DataFrame([{
            'timestamp': d.timestamp,
            'funding_rate': float(d.funding_rate),
            'mark_premium': float(d.mark_premium) if d.mark_premium else None,
            'mark_price': float(d.mark_price),
            'index_price': float(d.index_price)
        } for d in data])


def get_signals_data(hours=24):
    """Get trading signals time series data"""
    with db_manager.get_session() as session:
        cutoff = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(hours=hours)

        data = session.query(Signal).filter(
            Signal.timestamp >= cutoff
        ).order_by(Signal.timestamp).all()

        return pd.DataFrame([{
            'timestamp': d.timestamp,
            'signal_type': d.signal_type,
            'confidence_score': float(d.confidence_score),
            'price': float(d.price) if d.price else None,
            'cvd': float(d.cvd) if d.cvd else None,
            'orderbook_imbalance': float(d.orderbook_imbalance) if d.orderbook_imbalance else None
        } for d in data])


def get_price_data(hours=24):
    """Get BTC price data"""
    with db_manager.get_session() as session:
        cutoff = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(hours=hours)

        # Try BTC price candles first (1min candles from Kalshi or other sources)
        candles = session.query(BtcPriceCandle).filter(
            BtcPriceCandle.timestamp >= cutoff,
            BtcPriceCandle.timeframe == '1min'
        ).order_by(BtcPriceCandle.timestamp).all()

        if candles:
            return pd.DataFrame([{
                'timestamp': c.timestamp,
                'price': float(c.close),  # Use close price from candle
                'source': c.exchange
            } for c in candles])

        # Fallback to spot price snapshots
        data = session.query(SpotPriceSnapshot).filter(
            SpotPriceSnapshot.timestamp >= cutoff,
            SpotPriceSnapshot.source.in_(['coinbase', 'kraken'])
        ).order_by(SpotPriceSnapshot.timestamp).all()

        if data:
            return pd.DataFrame([{
                'timestamp': d.timestamp,
                'price': float(d.price),
                'source': d.source
            } for d in data])

        # Fallback to trades
        trades = session.query(Trade).filter(
            Trade.timestamp >= cutoff
        ).order_by(Trade.timestamp).all()

        return pd.DataFrame([{
            'timestamp': t.timestamp,
            'price': float(t.price),
            'source': t.exchange
        } for t in trades])


def plot_cvd(df):
    """Plot CVD time series"""
    fig = go.Figure()

    if not df.empty:
        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['cvd'],
            mode='lines',
            name='CVD',
            line=dict(color='blue', width=2),
            fill='tozeroy',
            fillcolor='rgba(0, 100, 255, 0.1)'
        ))

        # Add zero line
        fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)

    fig.update_layout(
        title='Cumulative Volume Delta (CVD)',
        xaxis_title='Time',
        yaxis_title='CVD (BTC)',
        hovermode='x unified',
        height=400
    )

    return fig


def plot_orderbook(df):
    """Plot orderbook imbalance and spread"""
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Orderbook Imbalance', 'Bid-Ask Spread'),
        vertical_spacing=0.15
    )

    if not df.empty:
        # Imbalance
        fig.add_trace(
            go.Scatter(
                x=df['timestamp'],
                y=df['imbalance'],
                mode='lines',
                name='Imbalance',
                line=dict(color='purple', width=2)
            ),
            row=1, col=1
        )

        # Add threshold lines
        fig.add_hline(y=60, line_dash="dash", line_color="green", opacity=0.3, row=1, col=1)
        fig.add_hline(y=-60, line_dash="dash", line_color="red", opacity=0.3, row=1, col=1)
        fig.add_hline(y=0, line_dash="solid", line_color="gray", opacity=0.3, row=1, col=1)

        # Spread
        if 'spread_pct' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df['timestamp'],
                    y=df['spread_pct'],
                    mode='lines',
                    name='Spread %',
                    line=dict(color='orange', width=2)
                ),
                row=2, col=1
            )

    fig.update_xaxes(title_text="Time", row=2, col=1)
    fig.update_yaxes(title_text="Imbalance (%)", row=1, col=1)
    fig.update_yaxes(title_text="Spread (%)", row=2, col=1)

    fig.update_layout(
        hovermode='x unified',
        height=600,
        showlegend=True
    )

    return fig


def plot_trade_intensity(df):
    """Plot trade intensity metrics"""
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Trades Per Second', 'Market Aggression Score'),
        vertical_spacing=0.15
    )

    if not df.empty:
        # TPS
        fig.add_trace(
            go.Scatter(
                x=df['timestamp'],
                y=df['tps_60s'],
                mode='lines',
                name='TPS (60s)',
                line=dict(color='cyan', width=2)
            ),
            row=1, col=1
        )

        # Aggression Score
        fig.add_trace(
            go.Scatter(
                x=df['timestamp'],
                y=df['aggression_score'],
                mode='lines',
                name='Aggression',
                line=dict(color='red', width=2),
                fill='tozeroy'
            ),
            row=2, col=1
        )

        # Add threshold lines
        fig.add_hline(y=0.3, line_dash="dash", line_color="green", opacity=0.3, row=2, col=1)
        fig.add_hline(y=-0.3, line_dash="dash", line_color="red", opacity=0.3, row=2, col=1)
        fig.add_hline(y=0, line_dash="solid", line_color="gray", opacity=0.3, row=2, col=1)

    fig.update_xaxes(title_text="Time", row=2, col=1)
    fig.update_yaxes(title_text="TPS", row=1, col=1)
    fig.update_yaxes(title_text="Score (-1 to 1)", row=2, col=1)

    fig.update_layout(
        hovermode='x unified',
        height=600,
        showlegend=True
    )

    return fig


def plot_futures_basis(df):
    """Plot futures basis"""
    fig = go.Figure()

    if not df.empty:
        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['basis_pct'],
            mode='lines',
            name='Basis %',
            line=dict(color='magenta', width=2),
            fill='tozeroy'
        ))

        # Add threshold lines
        fig.add_hline(y=0.5, line_dash="dash", line_color="green", opacity=0.3, annotation_text="Strong Contango")
        fig.add_hline(y=-0.5, line_dash="dash", line_color="red", opacity=0.3, annotation_text="Strong Backwardation")
        fig.add_hline(y=0, line_dash="solid", line_color="gray", opacity=0.5)

    fig.update_layout(
        title='Futures-Spot Basis',
        xaxis_title='Time',
        yaxis_title='Basis (%)',
        hovermode='x unified',
        height=400
    )

    return fig


def plot_funding_rate(df):
    """Plot funding rate"""
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Funding Rate', 'Mark Premium'),
        vertical_spacing=0.15
    )

    if not df.empty:
        # Funding Rate
        fig.add_trace(
            go.Scatter(
                x=df['timestamp'],
                y=df['funding_rate'],
                mode='lines',
                name='Funding Rate',
                line=dict(color='teal', width=2)
            ),
            row=1, col=1
        )

        # Add threshold lines
        fig.add_hline(y=0.0001, line_dash="dash", line_color="red", opacity=0.3, row=1, col=1)
        fig.add_hline(y=-0.0001, line_dash="dash", line_color="green", opacity=0.3, row=1, col=1)
        fig.add_hline(y=0, line_dash="solid", line_color="gray", opacity=0.3, row=1, col=1)

        # Mark Premium
        if 'mark_premium' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df['timestamp'],
                    y=df['mark_premium'],
                    mode='lines',
                    name='Mark Premium',
                    line=dict(color='brown', width=2)
                ),
                row=2, col=1
            )

    fig.update_xaxes(title_text="Time", row=2, col=1)
    fig.update_yaxes(title_text="Funding Rate", row=1, col=1)
    fig.update_yaxes(title_text="Premium (%)", row=2, col=1)

    fig.update_layout(
        hovermode='x unified',
        height=600,
        showlegend=True
    )

    return fig


def plot_signals_with_price(price_df, signals_df):
    """Plot BTC price with trading signals overlay"""
    fig = go.Figure()

    # Plot price
    if not price_df.empty:
        # Group by source and plot separately
        for source in price_df['source'].unique():
            source_data = price_df[price_df['source'] == source]
            fig.add_trace(go.Scatter(
                x=source_data['timestamp'],
                y=source_data['price'],
                mode='lines',
                name=f'Price ({source})',
                line=dict(width=2),
                opacity=0.7
            ))

    # Add signals as markers
    if not signals_df.empty:
        # Buy signals
        buy_signals = signals_df[signals_df['signal_type'].str.contains('BUY')]
        if not buy_signals.empty:
            fig.add_trace(go.Scatter(
                x=buy_signals['timestamp'],
                y=buy_signals['price'],
                mode='markers',
                name='BUY Signal',
                marker=dict(
                    symbol='triangle-up',
                    size=15,
                    color='green',
                    line=dict(color='darkgreen', width=2)
                )
            ))

        # Sell signals
        sell_signals = signals_df[signals_df['signal_type'].str.contains('SELL')]
        if not sell_signals.empty:
            fig.add_trace(go.Scatter(
                x=sell_signals['timestamp'],
                y=sell_signals['price'],
                mode='markers',
                name='SELL Signal',
                marker=dict(
                    symbol='triangle-down',
                    size=15,
                    color='red',
                    line=dict(color='darkred', width=2)
                )
            ))

    fig.update_layout(
        title='BTC Price with Trading Signals',
        xaxis_title='Time',
        yaxis_title='Price (USD)',
        hovermode='x unified',
        height=500
    )

    return fig


def plot_signal_distribution(signals_df):
    """Plot signal type distribution"""
    if signals_df.empty:
        return go.Figure()

    signal_counts = signals_df['signal_type'].value_counts()

    colors = {
        'STRONG_BUY': 'darkgreen',
        'BUY': 'lightgreen',
        'NEUTRAL': 'gray',
        'SELL': 'lightcoral',
        'STRONG_SELL': 'darkred'
    }

    fig = go.Figure(data=[go.Bar(
        x=signal_counts.index,
        y=signal_counts.values,
        marker_color=[colors.get(sig, 'blue') for sig in signal_counts.index]
    )])

    fig.update_layout(
        title='Signal Distribution',
        xaxis_title='Signal Type',
        yaxis_title='Count',
        height=300
    )

    return fig


def get_volatility_data(hours=24):
    """
    Get volatility data for ATM strike and 3 strikes on either side.

    Returns DataFrame with columns: timestamp, strike_price, moneyness_pct,
    implied_volatility, realized_vol_30min, realized_vol_10min, etc.
    """
    with db_manager.get_session() as session:
        cutoff = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(hours=hours)

        # Get all snapshots from the time window
        snapshots = session.query(KalshiOrderbookSnapshot).filter(
            KalshiOrderbookSnapshot.timestamp >= cutoff,
            KalshiOrderbookSnapshot.implied_volatility.isnot(None)
        ).order_by(KalshiOrderbookSnapshot.timestamp).all()

        if not snapshots:
            return pd.DataFrame()

        data = []
        for snap in snapshots:
            data.append({
                'timestamp': snap.timestamp,
                'strike_price': float(snap.strike_price),
                'current_btc_price': float(snap.current_btc_price),
                'distance_from_current': float(snap.distance_from_current),
                'moneyness_pct': float(snap.moneyness_pct),
                'moneyness_category': snap.moneyness_category,
                'implied_volatility': float(snap.implied_volatility) if snap.implied_volatility else None,
                'realized_vol_30min': float(snap.realized_vol_30min) if snap.realized_vol_30min else None,
                'realized_vol_10min': float(snap.realized_vol_10min) if snap.realized_vol_10min else None,
                'realized_vol_yesterday': float(snap.realized_vol_yesterday_same_hour) if snap.realized_vol_yesterday_same_hour else None,
                'realized_vol_last_week': float(snap.realized_vol_last_week_same_hour) if snap.realized_vol_last_week_same_hour else None,
                'iv_rank': float(snap.implied_volatility_rank) if snap.implied_volatility_rank else None,
                'yes_bid': float(snap.best_yes_bid) if snap.best_yes_bid else None,
                'ticker': snap.ticker,
                'contract_time_window': snap.contract_time_window,
                'contract_start_time': snap.contract_start_time,
                'contract_end_time': snap.contract_end_time,
                'day_of_week': snap.timestamp.strftime('%A') if snap.timestamp else None,
                'date': snap.timestamp.date() if snap.timestamp else None
            })

        return pd.DataFrame(data)


def get_btc_candle_data(timeframe='5min', hours=24, exchange='coinbase'):
    """
    Get BTC price candle data for a specific timeframe.

    Args:
        timeframe: Candle timeframe ('1min', '5min', '15min', '30min', '1hour', '1day', '1week')
        hours: Time range in hours
        exchange: Exchange to fetch from

    Returns:
        DataFrame with OHLC data
    """
    with db_manager.get_session() as session:
        cutoff = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(hours=hours)

        candles = session.query(BtcPriceCandle).filter(
            BtcPriceCandle.timeframe == timeframe,
            BtcPriceCandle.exchange == exchange,
            BtcPriceCandle.timestamp >= cutoff
        ).order_by(BtcPriceCandle.timestamp).all()

        if not candles:
            return pd.DataFrame()

        data = []
        for candle in candles:
            data.append({
                'timestamp': candle.timestamp,
                'open': float(candle.open),
                'high': float(candle.high),
                'low': float(candle.low),
                'close': float(candle.close),
                'volume': float(candle.volume) if candle.volume else 0,
                'num_trades': candle.num_trades
            })

        return pd.DataFrame(data)


def plot_btc_candlestick_chart(df, timeframe='5min'):
    """
    Create a candlestick chart from OHLC data.

    Args:
        df: DataFrame with columns: timestamp, open, high, low, close, volume
        timeframe: Candle timeframe for title

    Returns:
        Plotly figure
    """
    if df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No candle data available. Run candle aggregator to populate data.",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=14)
        )
        fig.update_layout(height=500)
        return fig

    # Create figure with secondary y-axis for volume
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.7, 0.3],
        subplot_titles=('Price', 'Volume')
    )

    # Candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=df['timestamp'],
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name='BTC Price',
            increasing_line_color='green',
            decreasing_line_color='red'
        ),
        row=1, col=1
    )

    # Volume bars
    colors = ['green' if row['close'] >= row['open'] else 'red' for _, row in df.iterrows()]
    fig.add_trace(
        go.Bar(
            x=df['timestamp'],
            y=df['volume'],
            name='Volume',
            marker_color=colors,
            showlegend=False
        ),
        row=2, col=1
    )

    fig.update_layout(
        title=f'BTC Price Candlestick Chart ({timeframe})',
        xaxis_rangeslider_visible=False,
        height=600,
        hovermode='x unified'
    )

    fig.update_xaxes(title_text="Time", row=2, col=1)
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)

    return fig


def plot_fixed_strike_iv(df):
    """
    Plot implied volatility over time for FIXED strikes.
    Tracks specific strike prices (e.g., $90k, $95k, $100k) over time.
    """
    if df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No volatility data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=20)
        )
        fig.update_layout(height=400)
        return fig

    # Get all unique strikes
    all_strikes = df['strike_price'].unique()

    # Find the most common strikes (those that appear in most snapshots)
    strike_counts = df.groupby('strike_price').size().sort_values(ascending=False)

    # Select top 5-7 most frequently occurring strikes
    top_strikes = strike_counts.head(7).index.tolist()

    # Filter data for these strikes
    fig = go.Figure()

    for strike in sorted(top_strikes):
        strike_data = df[df['strike_price'] == strike].sort_values('timestamp')

        if len(strike_data) > 0:
            fig.add_trace(go.Scatter(
                x=strike_data['timestamp'],
                y=strike_data['implied_volatility'],
                mode='lines',
                name=f"${strike:,.0f}",
                line=dict(width=2)
            ))

    fig.update_layout(
        title='Implied Volatility - Fixed Strikes Over Time',
        xaxis_title='Time',
        yaxis_title='Implied Volatility (%)',
        height=450,
        hovermode='x unified',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )

    return fig


def plot_atm_strike_iv(df):
    """
    Plot implied volatility over time for the ATM strike.
    The strike price itself varies as BTC price moves, but we track
    whatever strike is currently ATM at each point in time.
    """
    if df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No volatility data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=20)
        )
        fig.update_layout(height=400)
        return fig

    # For each timestamp, find the ATM strike
    timestamps = df['timestamp'].unique()

    atm_data = []
    for ts in sorted(timestamps):
        ts_data = df[df['timestamp'] == ts].copy()

        # Find ATM strike (closest to moneyness = 0)
        ts_data['abs_moneyness'] = ts_data['moneyness_pct'].abs()
        atm_row = ts_data.loc[ts_data['abs_moneyness'].idxmin()]

        atm_data.append({
            'timestamp': ts,
            'strike_price': atm_row['strike_price'],
            'implied_volatility': atm_row['implied_volatility'],
            'current_btc_price': atm_row['current_btc_price']
        })

    atm_df = pd.DataFrame(atm_data)

    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Add IV trace
    fig.add_trace(
        go.Scatter(
            x=atm_df['timestamp'],
            y=atm_df['implied_volatility'],
            mode='lines',
            name='ATM Implied Vol',
            line=dict(color='blue', width=3)
        ),
        secondary_y=False
    )

    # Add strike price trace (to show which strike is ATM)
    fig.add_trace(
        go.Scatter(
            x=atm_df['timestamp'],
            y=atm_df['strike_price'],
            mode='lines',
            name='ATM Strike Price',
            line=dict(color='gray', width=1, dash='dot'),
            opacity=0.5
        ),
        secondary_y=True
    )

    # Update axes
    fig.update_xaxes(title_text="Time")
    fig.update_yaxes(title_text="Implied Volatility (%)", secondary_y=False)
    fig.update_yaxes(title_text="Strike Price ($)", secondary_y=True)

    fig.update_layout(
        title='Implied Volatility - ATM Strike Over Time',
        height=450,
        hovermode='x unified',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )

    return fig


def plot_volatility_term_structure(df):
    """
    Plot volatility smile(s) - IV across strikes.
    Shows multiple smiles if data contains multiple unique timestamps.
    Shows only the latest if there are too many timestamps.
    """
    if df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No volatility data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=20)
        )
        fig.update_layout(height=300)
        return fig

    # Get unique timestamps
    unique_timestamps = sorted(df['timestamp'].unique())

    # Determine how many smiles to plot
    # If > 10 unique timestamps, show only latest (too many lines)
    # If <= 10, show all
    if len(unique_timestamps) > 10:
        timestamps_to_plot = [max(unique_timestamps)]
        title_suffix = "Latest"
    else:
        timestamps_to_plot = unique_timestamps
        if len(timestamps_to_plot) == 1:
            title_suffix = "Latest"
        else:
            title_suffix = f"{len(timestamps_to_plot)} Time Points"

    fig = go.Figure()

    # Color palette for multiple smiles
    colors = ['blue', 'red', 'green', 'purple', 'orange', 'brown', 'pink', 'gray', 'cyan', 'magenta']

    for idx, timestamp in enumerate(timestamps_to_plot):
        timestamp_data = df[df['timestamp'] == timestamp].sort_values('strike_price')

        if timestamp_data.empty:
            continue

        # Get current BTC price for this timestamp
        current_price = timestamp_data.iloc[0]['current_btc_price']
        time_window = timestamp_data.iloc[0]['contract_time_window'] if 'contract_time_window' in timestamp_data.columns else ''

        # Create label
        time_str = timestamp.strftime("%m/%d %H:%M")
        label = f"{time_str}"
        if time_window:
            label += f" ({time_window})"

        color = colors[idx % len(colors)]

        # Add IV line
        fig.add_trace(go.Scatter(
            x=timestamp_data['strike_price'],
            y=timestamp_data['implied_volatility'],
            mode='lines+markers',
            name=label,
            line=dict(color=color, width=2),
            marker=dict(size=6),
            legendgroup=f'group{idx}'
        ))

        # Add vertical line at current price (only for the latest smile)
        # Latest is the last timestamp in the sorted list
        if idx == len(timestamps_to_plot) - 1:
            fig.add_vline(
                x=current_price,
                line_dash="dash",
                line_color="red",
                annotation_text=f"Spot: ${current_price:,.0f}",
                annotation_position="top"
            )

        # Highlight ATM strike
        atm_data = timestamp_data[timestamp_data['moneyness_category'] == 'ATM']
        if not atm_data.empty:
            fig.add_trace(go.Scatter(
                x=atm_data['strike_price'],
                y=atm_data['implied_volatility'],
                mode='markers',
                name=f'ATM ({time_str})',
                marker=dict(size=12, color=color, symbol='star', line=dict(width=1, color='white')),
                showlegend=False,
                legendgroup=f'group{idx}'
            ))

    fig.update_layout(
        title=f'Volatility Smile ({title_suffix})',
        xaxis_title='Strike Price ($)',
        yaxis_title='Implied Volatility (%)',
        height=400,
        hovermode='x',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )

    return fig


def plot_realized_vs_implied_vol(df, rv_metric='realized_vol_30min'):
    """
    Compare realized volatility vs implied volatility by contract.
    Shows each 1-hour contract as a separate series with x-axis as minutes into contract (0-60).

    Args:
        df: DataFrame with volatility data
        rv_metric: Which realized vol column to use ('realized_vol_10min', 'realized_vol_30min', 'realized_vol_60min')
    """
    if df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No volatility data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=20)
        )
        fig.update_layout(height=300)
        return fig

    # Filter to ATM strikes only (most representative)
    atm_data = df[df['moneyness_category'] == 'ATM'].copy()

    if atm_data.empty:
        # If no ATM data, use all data
        atm_data = df.copy()

    # Need contract_start_time to calculate elapsed time
    if 'contract_start_time' not in atm_data.columns or atm_data['contract_start_time'].isna().all():
        # Fallback to old chart if no contract times
        fig = go.Figure()
        fig.add_annotation(
            text="Contract time data not available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=14)
        )
        fig.update_layout(height=400)
        return fig

    # Calculate minutes elapsed from contract start
    atm_data['minutes_elapsed'] = (atm_data['timestamp'] - atm_data['contract_start_time']).dt.total_seconds() / 60.0

    # Group by contract (date + time window) to show each contract separately
    atm_data['contract_id'] = atm_data['date'].astype(str) + ' ' + atm_data['contract_time_window'].astype(str)

    # Get unique contracts
    contracts = atm_data['contract_id'].dropna().unique()

    # Limit to reasonable number of contracts to avoid clutter
    if len(contracts) > 10:
        # Show only the 10 most recent contracts
        latest_contracts = sorted(contracts, reverse=True)[:10]
        atm_data = atm_data[atm_data['contract_id'].isin(latest_contracts)]
        contracts = latest_contracts

    fig = go.Figure()

    # Color palettes for IV and RV
    colors = ['blue', 'red', 'green', 'purple', 'orange', 'brown', 'pink', 'cyan', 'magenta', 'olive']

    for idx, contract_id in enumerate(sorted(contracts)):
        contract_data = atm_data[atm_data['contract_id'] == contract_id].sort_values('minutes_elapsed')

        if contract_data.empty:
            continue

        color = colors[idx % len(colors)]

        # Format contract label (e.g., "12/14 18:00-19:00")
        date_str = contract_data.iloc[0]['date'].strftime('%m/%d') if hasattr(contract_data.iloc[0]['date'], 'strftime') else str(contract_data.iloc[0]['date'])
        time_window = contract_data.iloc[0]['contract_time_window'] if 'contract_time_window' in contract_data.columns else ''
        label = f"{date_str} {time_window}"

        # Implied Vol line (solid)
        fig.add_trace(go.Scatter(
            x=contract_data['minutes_elapsed'],
            y=contract_data['implied_volatility'],
            mode='lines',
            name=f'IV: {label}',
            line=dict(color=color, width=2),
            legendgroup=f'group{idx}'
        ))

        # Realized Vol line (dashed)
        if rv_metric in contract_data.columns and contract_data[rv_metric].notna().any():
            fig.add_trace(go.Scatter(
                x=contract_data['minutes_elapsed'],
                y=contract_data[rv_metric],
                mode='lines',
                name=f'RV: {label}',
                line=dict(color=color, width=2, dash='dash'),
                legendgroup=f'group{idx}'
            ))

    # Get friendly name for RV metric
    rv_name_map = {
        'realized_vol_10min': '10min',
        'realized_vol_30min': '30min',
        'realized_vol_yesterday': 'Yesterday',
        'realized_vol_last_week': 'Last Week'
    }
    rv_name = rv_name_map.get(rv_metric, rv_metric)

    fig.update_layout(
        title=f'Implied vs Realized Vol ({rv_name}) by Contract',
        xaxis_title='Minutes into Contract',
        yaxis_title='Volatility (%)',
        height=400,
        hovermode='x unified',
        xaxis=dict(range=[0, 60]),  # Always show 0-60 minutes
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )

    return fig


def main():
    """Main dashboard application"""
    st.title("📈 BTC Signal Trader Dashboard")
    st.markdown("Real-time visualization of all 6 trading signal sources")

    # Sidebar controls
    st.sidebar.header("Settings")
    hours = st.sidebar.slider("Time Range (hours)", 1, 168, 72)
    auto_refresh = st.sidebar.checkbox("Auto Refresh", value=False)
    refresh_interval = st.sidebar.slider("Refresh Interval (seconds)", 5, 60, 10)

    if auto_refresh:
        st.sidebar.info(f"Auto-refreshing every {refresh_interval} seconds")
        import time
        time.sleep(refresh_interval)
        st.rerun()

    if st.sidebar.button("🔄 Refresh Now"):
        st.rerun()

    # Load data
    with st.spinner("Loading data..."):
        price_df = get_price_data(hours)
        cvd_df = get_cvd_data(hours)
        orderbook_df = get_orderbook_data(hours)
        intensity_df = get_trade_intensity_data(hours)
        basis_df = get_futures_basis_data(hours)
        funding_df = get_funding_rate_data(hours)
        signals_df = get_signals_data(hours)

    # Display stats
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if not signals_df.empty:
            latest_signal = signals_df.iloc[-1]
            signal_color = {
                'STRONG_BUY': '🟢',
                'BUY': '🟩',
                'NEUTRAL': '⚪',
                'SELL': '🟥',
                'STRONG_SELL': '🔴'
            }.get(latest_signal['signal_type'], '⚪')
            st.metric("Latest Signal", f"{signal_color} {latest_signal['signal_type']}")
        else:
            st.metric("Latest Signal", "No data")

    with col2:
        if not price_df.empty:
            latest_price = price_df.iloc[-1]['price']
            st.metric("BTC Price", f"${latest_price:,.2f}")
        else:
            st.metric("BTC Price", "No data")

    with col3:
        if not signals_df.empty:
            st.metric("Signals Generated", len(signals_df))
        else:
            st.metric("Signals Generated", "0")

    with col4:
        if not cvd_df.empty:
            latest_cvd = cvd_df.iloc[-1]['cvd']
            st.metric("Current CVD", f"{latest_cvd:+,.2f} BTC")
        else:
            st.metric("Current CVD", "No data")

    # Main charts
    st.header("Price & Signals")
    col1, col2 = st.columns([3, 1])
    with col1:
        st.plotly_chart(plot_signals_with_price(price_df, signals_df), width='stretch')
    with col2:
        st.plotly_chart(plot_signal_distribution(signals_df), width='stretch')

    # ALL SIGNAL SOURCES ON MAIN PAGE
    st.header("📊 All Signal Sources - Time Series")

    # Row 1: CVD and Orderbook
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📊 CVD (Cumulative Volume Delta)")
        st.plotly_chart(plot_cvd(cvd_df), width='stretch')
    with col2:
        st.subheader("📚 Orderbook Imbalance & Spread")
        st.plotly_chart(plot_orderbook(orderbook_df), width='stretch')

    # Row 2: Trade Intensity and Futures Basis
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("⚡ Trade Intensity")
        st.plotly_chart(plot_trade_intensity(intensity_df), width='stretch')
    with col2:
        st.subheader("💹 Futures Basis")
        st.plotly_chart(plot_futures_basis(basis_df), width='stretch')

    # Row 3: Funding Rate (full width)
    st.subheader("💰 Funding Rate & Mark Premium")
    st.plotly_chart(plot_funding_rate(funding_df), width='stretch')

    # BTC Price Candlestick Section
    st.header("📊 BTC Price Candlestick Chart")

    # Timeframe selector
    timeframe_options = {
        '1 Minute': '1min',
        '5 Minutes': '5min',
        '15 Minutes': '15min',
        '30 Minutes': '30min',
        '1 Hour': '1hour',
        '1 Day': '1day',
        '1 Week': '1week'
    }

    col1, col2 = st.columns([1, 3])
    with col1:
        selected_timeframe_label = st.selectbox(
            "Candle Size",
            list(timeframe_options.keys()),
            index=1,  # Default to 5 minutes
            key="btc_timeframe"
        )
        selected_timeframe = timeframe_options[selected_timeframe_label]

    # Fetch and display candlestick chart
    candle_df = get_btc_candle_data(timeframe=selected_timeframe, hours=hours, exchange='kalshi')
    st.plotly_chart(plot_btc_candlestick_chart(candle_df, timeframe=selected_timeframe_label), use_container_width=True)

    # Volatility Section
    st.header("📈 Volatility Tracking (Kalshi Hourly Markets)")

    # Get volatility data
    volatility_df = get_volatility_data(hours=hours)

    if not volatility_df.empty:
        # Filters section
        st.subheader("🔍 Filters")
        filter_col1, filter_col2, filter_col3, filter_col4 = st.columns(4)

        with filter_col1:
            # Time window filter
            time_windows = ['All'] + sorted(volatility_df['contract_time_window'].dropna().unique().tolist())
            selected_time_window = st.selectbox("Contract Time Window", time_windows, key="vol_time_window")

        with filter_col2:
            # Specific date filter
            dates = ['All'] + sorted([str(d) for d in volatility_df['date'].dropna().unique()])
            selected_date = st.selectbox("Date", dates, key="vol_date")

        with filter_col3:
            # Day of week filter
            days_of_week = ['All', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            selected_day = st.selectbox("Day of Week", days_of_week, key="vol_day")

        with filter_col4:
            # Strike price filter (ticker-based)
            tickers = ['All'] + sorted(volatility_df['ticker'].dropna().unique().tolist())
            selected_ticker = st.selectbox("Contract (Ticker)", tickers, key="vol_ticker")

        # Apply filters - Create two filtered datasets:
        # 1. Full filter (including ticker) for time series charts
        # 2. No ticker filter for volatility smile (shows IV across strikes)

        filtered_df_full = volatility_df.copy()
        filtered_df_smile = volatility_df.copy()

        # Apply time window filter to both
        if selected_time_window != 'All':
            filtered_df_full = filtered_df_full[filtered_df_full['contract_time_window'] == selected_time_window]
            filtered_df_smile = filtered_df_smile[filtered_df_smile['contract_time_window'] == selected_time_window]

        # Apply date filter to both
        if selected_date != 'All':
            filtered_df_full = filtered_df_full[filtered_df_full['date'].astype(str) == selected_date]
            filtered_df_smile = filtered_df_smile[filtered_df_smile['date'].astype(str) == selected_date]

        # Apply day of week filter to both
        if selected_day != 'All':
            filtered_df_full = filtered_df_full[filtered_df_full['day_of_week'] == selected_day]
            filtered_df_smile = filtered_df_smile[filtered_df_smile['day_of_week'] == selected_day]

        # Apply ticker filter ONLY to full filtered df (NOT to smile)
        if selected_ticker != 'All':
            filtered_df_full = filtered_df_full[filtered_df_full['ticker'] == selected_ticker]

        # Show filter results
        if len(filtered_df_full) != len(volatility_df):
            st.info(f"Showing {len(filtered_df_full)} of {len(volatility_df)} data points after filtering")

        # Use filtered_df_full for time series charts, filtered_df_smile for volatility smile
        volatility_df = filtered_df_full
        volatility_df_smile = filtered_df_smile

    if not volatility_df.empty:
        # Summary metrics - Get latest data
        latest_data = volatility_df.sort_values('timestamp').iloc[-1]
        latest_btc_price = latest_data['current_btc_price']
        latest_timestamp = latest_data['timestamp']

        # Get ATM strike info
        atm_data = volatility_df[volatility_df['moneyness_category'] == 'ATM'].sort_values('timestamp')
        if not atm_data.empty:
            latest_atm = atm_data.iloc[-1]
            latest_atm_strike = latest_atm['strike_price']
            latest_atm_iv = latest_atm['implied_volatility']
        else:
            latest_atm_strike = None
            latest_atm_iv = None

        # Top row: BTC Price and ATM Strike (most important info)
        col1, col2 = st.columns(2)
        with col1:
            st.metric("🪙 BTC Spot Price", f"${latest_btc_price:,.2f}",
                     delta=f"as of {latest_timestamp.strftime('%H:%M:%S')}")
        with col2:
            if latest_atm_strike:
                st.metric("🎯 ATM Strike", f"${latest_atm_strike:,.2f}",
                         delta=f"IV: {latest_atm_iv:.2f}%" if latest_atm_iv else "No IV")
            else:
                st.metric("🎯 ATM Strike", "N/A")

        # Second row: Volatility metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("ATM Implied Vol", f"{latest_atm_iv:.2f}%" if latest_atm_iv else "N/A")
        with col2:
            avg_iv = volatility_df['implied_volatility'].mean()
            st.metric("Avg Implied Vol", f"{avg_iv:.2f}%")
        with col3:
            latest_rv_30 = volatility_df['realized_vol_30min'].dropna().iloc[-1] if not volatility_df['realized_vol_30min'].dropna().empty else None
            st.metric("Realized Vol (30min)", f"{latest_rv_30:.2f}%" if latest_rv_30 else "N/A")
        with col4:
            iv_rv_diff = (latest_atm_iv - latest_rv_30) if (latest_atm_iv and latest_rv_30) else None
            st.metric("IV - RV Spread", f"{iv_rv_diff:+.2f}%" if iv_rv_diff else "N/A")

        # Row 1: Fixed Strike IV and ATM Strike IV
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("📌 Fixed Strike Implied Vol")
            st.plotly_chart(plot_fixed_strike_iv(volatility_df), width='stretch')
        with col2:
            st.subheader("🎯 ATM Strike Implied Vol")
            st.plotly_chart(plot_atm_strike_iv(volatility_df), width='stretch')

        # Row 2: Volatility Smile and IV vs RV
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("😊 Volatility Smile")
            st.plotly_chart(plot_volatility_term_structure(volatility_df_smile), width='stretch')
        with col2:
            st.subheader("📉 Implied vs Realized Vol")
            # Filter for RV metric selection
            rv_options = {
                'Realized Vol (10min)': 'realized_vol_10min',
                'Realized Vol (30min)': 'realized_vol_30min',
                'RV Yesterday (same hour)': 'realized_vol_yesterday',
                'RV Last Week (same hour)': 'realized_vol_last_week'
            }
            selected_rv_label = st.selectbox("Realized Vol Metric", list(rv_options.keys()), index=1, key="rv_metric")
            selected_rv_metric = rv_options[selected_rv_label]
            st.plotly_chart(plot_realized_vs_implied_vol(volatility_df, rv_metric=selected_rv_metric), width='stretch')
    else:
        st.info("No volatility data available. Start the Kalshi orderbook collector to populate volatility metrics.")

    # Data tables in expandable sections
    st.header("📋 Recent Data")

    with st.expander("📊 CVD Data"):
        if not cvd_df.empty:
            st.dataframe(cvd_df.tail(20), width='stretch')
        else:
            st.info("No CVD data available")

    with st.expander("📚 Orderbook Data"):
        if not orderbook_df.empty:
            st.dataframe(orderbook_df.tail(20), width='stretch')
        else:
            st.info("No orderbook data available")

    with st.expander("⚡ Trade Intensity Data"):
        if not intensity_df.empty:
            st.dataframe(intensity_df.tail(20), width='stretch')
        else:
            st.info("No trade intensity data available")

    with st.expander("💹 Futures Basis Data"):
        if not basis_df.empty:
            st.dataframe(basis_df.tail(20), width='stretch')
        else:
            st.info("No futures basis data available")

    with st.expander("💰 Funding Rate Data"):
        if not funding_df.empty:
            st.dataframe(funding_df.tail(20), width='stretch')
        else:
            st.info("No funding rate data available")

    with st.expander("🎯 Trading Signals History"):
        if not signals_df.empty:
            st.dataframe(signals_df.tail(20), width='stretch')
        else:
            st.info("No signals generated yet")

    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("### About")
    st.sidebar.info(
        "This dashboard visualizes all 6 trading signal sources:\n\n"
        "1. **CVD** - Cumulative Volume Delta\n"
        "2. **Orderbook** - Bid/Ask Imbalance\n"
        "3. **Spread** - Bid-Ask Spread\n"
        "4. **Trade Intensity** - Market Activity\n"
        "5. **Futures Basis** - Contango/Backwardation\n"
        "6. **Funding Rate** - Sentiment Indicator"
    )


if __name__ == "__main__":
    main()
