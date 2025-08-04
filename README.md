# Algo-Trading System with ML & Automation

## Overview
This project is a Python-based automated trading system for NIFTY 50 stocks. It fetches stock data, applies a technical strategy, runs ML predictions, logs trades to Google Sheets, and sends Telegram alerts.

## Features
- Fetches daily stock data for RELIANCE.NS, TCS.NS, HDFCBANK.NS using Yahoo Finance
- Implements RSI < 30 and 20-DMA > 50-DMA as buy signal
- Backtests strategy over 6 months
- Trains Decision Tree model to predict next-day price direction
- Logs trades and summary to Google Sheets
- Sends Telegram alerts for trade signals
- Modular, PEP8-compliant code with logging

## Setup
1. **Clone the repository and install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Google Sheets Setup:**
   - Create a Google Sheet named `AlgoTradingLog` with two tabs: `TradeLog` and `Summary`.
   - Set up Google Service Account and share the sheet with its email.
   - Place your `service_account.json` in the project folder.
3. **Telegram Bot Setup:**
   - Create a bot via BotFather and get the token.
   - Add your chat ID.
   - Store both in `.env` file:
     ```
     BOT_TOKEN=your_bot_token
     CHAT_ID=your_chat_id
     ```

## Usage
Run the script:
```bash
python algo_trading.py
```
The script will:
- Print trade signals and summary to the console
- Log trades and summary to Google Sheets
- Send Telegram alerts

## Output
- **Console:** Trade signals and summary stats are printed.
- **Google Sheets:** Each trade and summary stats are logged.
- **Telegram:** Alerts for each trade signal.

## Modular Functions
- `fetch_data()`
- `calculate_indicators()`
- `generate_signals()`
- `run_backtest()`
- `train_ml_model()`
- `update_google_sheets()`
- `send_telegram_alert()`

## Scheduling
- Uses `schedule` to run daily at 15:30 (market close)

## License
MIT
