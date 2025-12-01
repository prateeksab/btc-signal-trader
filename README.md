# BTC Signal Trader

A real-time Bitcoin trading signal system that collects market data, generates trading signals, and executes trades on Binance.

## Features

- Real-time market data collection via WebSocket
- Technical indicator-based signal generation
- Backtesting framework for strategy validation
- Automated trade execution
- Risk management and position sizing
- Performance monitoring and logging

## Project Structure

```
btc-signal-trader/
├── data/
│   ├── collectors/     # Data collection modules
│   ├── processors/     # Data processing and indicators
│   └── storage/        # Database models and storage
├── strategy/
│   └── signals/        # Trading signal generation
├── backtesting/        # Backtesting framework
├── execution/          # Trade execution logic
├── monitoring/         # Performance monitoring
├── notebooks/          # Jupyter notebooks for analysis
├── tests/              # Unit tests
├── config/             # Configuration files
└── logs/               # Application logs
```

## Setup

1. Clone the repository
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Copy `.env.example` to `.env` and configure your API keys:
   ```bash
   cp .env.example .env
   ```

5. Edit `.env` with your Binance API credentials

## Usage

(To be added as components are implemented)

## Disclaimer

This software is for educational purposes only. Use at your own risk. Cryptocurrency trading carries a high level of risk and may not be suitable for all investors.

## License

MIT License
