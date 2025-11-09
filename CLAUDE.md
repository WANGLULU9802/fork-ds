# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a cryptocurrency trading bot called "fork-ds" that implements an automated trading system using DeepSeek AI for market analysis and signal generation. The bot supports Binance Futures and OKX Swap exchanges with an EMA-based trading strategy.

## Core Commands

### Environment Setup
```bash
# Install Python dependencies
pip install -r requirements.txt

# Set up environment variables in .env file:
# DEEPSEEK_API_KEY - DeepSeek API key for AI analysis
# BINANCE_API_KEY - Binance futures API key
# BINANCE_SECRET - Binance futures secret key
# OKX_API_KEY - OKX API key
# OKX_SECRET - OKX secret key
# OKX_PASSWORD - OKX account password
```

### Running the Bot
```bash
# Run the main trading bot
python ema_strategy.py

# The bot includes test mode by default (set test_mode=True in code)
# Bot will start automatically and run based on configured timeframe
```

### Development Commands
```bash
# View logs in real-time
tail -f app.log

# Check Python environment
python --version

# Install new dependencies
pip install <package_name> && pip freeze > requirements.txt
```

## Architecture Overview

### Core Components

- **`ema_strategy.py`** (421 lines) - Main trading bot implementation containing:
  - DeepSeek AI integration for market analysis
  - CCXT exchange connectivity
  - EMA trading strategy logic
  - Position management with leverage
  - Risk management (stop-loss/take-profit)
  - Scheduled execution system

- **Configuration** - Managed through `.env` file and in-code config dictionary:
  - Exchange API credentials
  - Trading parameters (leverage, position size, timeframe)
  - Proxy settings for Windows systems
  - Test mode controls

### Key Features

- **Multi-Exchange Support**: Primary Binance Futures with OKX backup versions
- **AI-Powered Analysis**: DeepSeek Chat API for real-time market signal generation
- **Timeframe Flexibility**: Supports 5m, 15m, and 1h intervals
- **Risk Management**: Configurable stop-loss, take-profit, and position sizing
- **Proxy Support**: Windows-compatible proxy configuration
- **Comprehensive Logging**: UTF-8 logging with exception tracking

### Trading Strategy

The bot implements an EMA (Exponential Moving Average) strategy with:
- Technical indicator analysis
- Multi-timeframe signal confirmation
- Confidence-based position sizing
- Automated entry/exit management
- Default leverage: 10x (configurable)

## Development Workflow

### Project Structure
```
fork-ds/
├── ema_strategy.py          # Main trading bot
├── .env                     # Environment variables & API keys
├── requirements.txt         # Python dependencies (44 packages)
├── app.log                  # Bot activity logs (created at runtime)
└── backup/                  # Development iterations (6 Python files)
    ├── deepseek_ok_带市场情绪+指标版本.py
    ├── deepseek_ok_带指标plus版本.py
    └── ... (other development versions)
```

### Safety Considerations

- **Test Mode**: Bot runs in test mode by default to prevent real trading
- **Position Validation**: All trades are validated before execution
- **Error Handling**: Comprehensive exception handling with logging
- **API Security**: API keys loaded from environment variables only

### Dependencies Management

Key dependencies include:
- `ccxt==4.5.14` - Exchange trading library
- `openai==2.6.1` - DeepSeek API client
- `pandas==2.3.3` - Data analysis
- `schedule==1.2.2` - Task scheduling
- `python-dotenv==1.2.1` - Environment management

## Important Notes

- The bot is configured for SOL/USDT trading by default
- Proxy is automatically configured for Windows systems (localhost:7890)
- All trading activities are logged to `app.log` with UTF-8 encoding
- Recent commits show active development with 5m timeframe support and bug fixes
- Backup directory contains iterative development versions showing feature evolution