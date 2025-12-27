#!/bin/bash
# Launch BTC Signal Trader Dashboard

echo "🚀 Launching BTC Signal Trader Dashboard..."
echo ""
echo "Dashboard will open in your default browser"
echo "Press Ctrl+C to stop the dashboard"
echo ""

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
fi

# Install/update requirements
echo "Checking dependencies..."
pip install -q streamlit plotly 2>/dev/null || echo "Dependencies already installed"

# Launch Streamlit dashboard
streamlit run dashboard.py
