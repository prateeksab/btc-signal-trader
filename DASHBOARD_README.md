# BTC Signal Trader Dashboard 📈

Interactive web dashboard for visualizing all 6 trading signal sources in real-time.

## Features

### 📊 Time Series Visualizations
- **CVD (Cumulative Volume Delta)** - Track buy/sell pressure over time
- **Orderbook Imbalance** - Monitor bid/ask imbalance and spread
- **Trade Intensity** - View market activity and aggression scores
- **Futures Basis** - Analyze contango/backwardation trends
- **Funding Rate** - Track funding rate and mark premium
- **BTC Price** - Price chart with trading signals overlay

### 🎯 Trading Signals
- Visual markers for BUY/SELL signals on price chart
- Signal distribution breakdown
- Recent signals table with full details
- Confidence scores and reasoning

### ⚡ Real-Time Features
- Auto-refresh capability (configurable interval)
- Adjustable time range (1-168 hours)
- Latest metrics at a glance
- Interactive charts with zoom/pan

## Quick Start

### Option 1: Launch Script (Easiest)
```bash
./launch_dashboard.sh
```

### Option 2: Direct Launch
```bash
streamlit run dashboard.py
```

### Option 3: Custom Port
```bash
streamlit run dashboard.py --server.port 8501
```

## Dashboard Layout

### Top Metrics Bar
- Latest Trading Signal
- Current BTC Price
- Total Signals Generated
- Current CVD Value

### Main Charts
1. **Price & Signals** - BTC price with buy/sell signal markers
2. **Signal Distribution** - Pie chart of signal types

### Signal Source Tabs
- **CVD Tab** - Volume delta chart with historical data table
- **Orderbook Tab** - Imbalance and spread charts
- **Trade Intensity Tab** - TPS and aggression score charts
- **Futures Basis Tab** - Basis percentage over time
- **Funding Rate Tab** - Funding rate and mark premium
- **All Signals Tab** - Complete signals history table

## Configuration

### Sidebar Controls
- **Time Range** - Slider to adjust hours of historical data (1-168)
- **Auto Refresh** - Toggle automatic data refresh
- **Refresh Interval** - Set seconds between refreshes (5-60)
- **Refresh Now** - Manual refresh button

### Auto-Refresh
Enable auto-refresh in the sidebar to keep data updated automatically. Useful for live monitoring.

## Data Sources

The dashboard queries these database tables:
- `cvd_snapshots` - CVD time series
- `orderbook_snapshots` - Orderbook data
- `trade_intensity` - Trade intensity metrics
- `futures_basis` - Futures-spot basis
- `funding_rate_metrics` - Funding rate data
- `signals` - Generated trading signals
- `trades` - Raw trade data (fallback)
- `spot_price_snapshots` - Price data

## Requirements

Installed automatically by launch script:
- `streamlit==1.29.0`
- `plotly==5.18.0`
- `pandas==2.2.0`

Already in your venv from main requirements.

## Usage Tips

1. **Start with 24 hours** - Default time range gives good overview
2. **Enable auto-refresh** - For live monitoring during trading
3. **Check signal distribution** - Understand market sentiment
4. **Use tabs** - Deep dive into specific signal sources
5. **Hover on charts** - See exact values and timestamps

## Keyboard Shortcuts

- **R** - Rerun the app (refresh data)
- **C** - Clear cache
- **S** - Open settings

## Troubleshooting

### No data showing
- Ensure data collectors are running
- Check database has recent data
- Try increasing time range

### Dashboard won't start
```bash
# Reinstall dependencies
pip install --upgrade streamlit plotly

# Check for port conflicts
streamlit run dashboard.py --server.port 8502
```

### Slow performance
- Reduce time range (fewer hours)
- Disable auto-refresh when not needed
- Clear browser cache

## Examples

### Monitor Live Trading
```bash
# Terminal 1: Run data collectors
python3 master_collector.py

# Terminal 2: Run signal generator
python3 strategy/signals/live_signal_trader.py

# Terminal 3: Launch dashboard
./launch_dashboard.sh
```

### View Historical Analysis
```bash
# Just launch dashboard to analyze stored data
./launch_dashboard.sh
```

## Customization

Edit `dashboard.py` to:
- Add more charts
- Change colors/themes
- Modify time windows
- Add custom metrics
- Export data to CSV

## Support

- Dashboard built with [Streamlit](https://streamlit.io/)
- Charts powered by [Plotly](https://plotly.com/)
- Issues: Check console for error messages

Enjoy your trading dashboard! 🚀📊
