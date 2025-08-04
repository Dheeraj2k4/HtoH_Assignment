import yfinance as yf
import pandas as pd
import pandas_ta as ta
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np
import gspread  # or use pygsheets
import schedule
import logging
import requests
import time
from telegram import Bot
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv

# Logging setup
logging.basicConfig(
    filename='algo_trading.log',
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s'
)

# Constants
STOCKS = ['RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS']
RSI_PERIOD = 14
DMA_SHORT = 20
DMA_LONG = 50
BACKTEST_MONTHS = 6


# Load environment variables from .env file
load_dotenv()
BOT_TOKEN = os.getenv("BOT_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")

def fetch_data(ticker, period='6mo', interval='1d'):
    """
    Fetch historical stock data from Yahoo Finance.
    """
    try:
        data = yf.download(ticker, period='1y', interval=interval)
        # Flatten MultiIndex columns if present
        if isinstance(data.columns, pd.MultiIndex):
            # If only one ticker, flatten to first level (e.g., 'Close')
            data.columns = [col[0] for col in data.columns]
        if data.empty:
            print(f"⚠️ No data for {ticker}. Skipping.")
            return None
        print(f"Fetched {len(data)} rows for {ticker}")
        print("Columns:", data.columns.tolist())
        logging.info(f"Fetched data for {ticker}")
        return data
    except Exception as e:
        logging.error(f"Error fetching data for {ticker}: {e}")
        return None

def calculate_indicators(df):
    """
    Calculate RSI, DMA, MACD, and other indicators.
    """
    df['RSI'] = ta.rsi(df['Close'], length=RSI_PERIOD)
    df['DMA20'] = df['Close'].rolling(DMA_SHORT).mean()
    df['DMA50'] = df['Close'].rolling(DMA_LONG).mean()
    macd = ta.macd(df['Close'])
    if macd is not None:
        df['MACD'] = macd['MACD_12_26_9']
        df['MACD_signal'] = macd['MACDs_12_26_9']
    else:
        df['MACD'] = np.nan
        df['MACD_signal'] = np.nan
    return df

def generate_signals(df):
    """
    Generate buy/sell signals based on strategy.
    """
    # Improved trading strategy
    trades = []
    in_position = False
    entry_price = None
    entry_date = None
    entry_idx = None
    stop_loss = None
    take_profit = None
    for i in range(1, len(df)):
        macd = df['MACD'].iloc[i]
        macd_signal = df['MACD_signal'].iloc[i]
        macd_prev = df['MACD'].iloc[i-1]
        macd_signal_prev = df['MACD_signal'].iloc[i-1]
        rsi = df['RSI'].iloc[i]
        close = df['Close'].iloc[i]
        # Entry: MACD crossover, RSI > 40
        if (
            pd.notnull(macd) and pd.notnull(macd_signal) and pd.notnull(macd_prev) and pd.notnull(macd_signal_prev) and pd.notnull(rsi)
            and macd_prev < macd_signal_prev and macd > macd_signal and rsi > 40 and not in_position
        ):
            entry_price = close
            entry_date = df.index[i]
            entry_idx = i
            stop_loss = entry_price * 0.98
            take_profit = entry_price * 1.03
            in_position = True
            print(f"Entered position at {entry_date} price {entry_price}")
        # Exit: RSI < 70, after 5 days, stop-loss or take-profit
        elif in_position:
            exit = False
            reason = ""
            # RSI exit
            if rsi < 70:
                exit = True
                reason = "RSI < 70"
            # 5-day exit
            elif (i - entry_idx) >= 5:
                exit = True
                reason = "5 days held"
            # Stop-loss
            elif close <= stop_loss:
                exit = True
                reason = "Stop-loss hit"
            # Take-profit
            elif close >= take_profit:
                exit = True
                reason = "Take-profit hit"
            if exit:
                exit_price = close
                exit_date = df.index[i]
                pnl = exit_price - entry_price
                trades.append({
                    'ticker': df.get('Ticker', 'N/A'),
                    'entry_date': entry_date,
                    'entry_price': entry_price,
                    'exit_date': exit_date,
                    'exit_price': exit_price,
                    'pnl': pnl,
                    'exit_reason': reason
                })
                print(f"Exited position at {exit_date} price {exit_price} P&L: {pnl} Reason: {reason}")
                in_position = False
                entry_price = None
                entry_date = None
                entry_idx = None
                stop_loss = None
                take_profit = None
    # If still in position at end, close at last price
    if in_position:
        exit_price = df['Close'].iloc[-1]
        exit_date = df.index[-1]
        pnl = exit_price - entry_price
        trades.append({
            'ticker': df.get('Ticker', 'N/A'),
            'entry_date': entry_date,
            'entry_price': entry_price,
            'exit_date': exit_date,
            'exit_price': exit_price,
            'pnl': pnl,
            'exit_reason': 'End of data'
        })
        print(f"Force-exited position at {exit_date} price {exit_price} P&L: {pnl} Reason: End of data")
    return trades

def run_backtest(df, signals):
    """
    Backtest strategy and calculate P&L, win ratio.
    """
    pnl = 0.0
    wins = 0
    losses = 0
    for trade in signals:
        trade_pnl = trade['pnl']
        pnl += trade_pnl
        if trade_pnl > 0:
            wins += 1
        else:
            losses += 1
    win_ratio = wins / (wins + losses) if (wins + losses) > 0 else 0
    summary = {'total_pnl': pnl, 'win_ratio': win_ratio, 'trades': signals}
    logging.info(f"Backtest summary: {summary}")
    return summary

def train_ml_model(df):
    """
    Train Decision Tree or Logistic Regression to predict next-day direction.
    """
    # Feature engineering
    df['Direction'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    features = ['RSI', 'MACD', 'MACD_signal', 'Volume']
    # Ensure all required columns exist
    for col in features + ['Direction']:
        if col not in df.columns:
            df[col] = np.nan
    print("Columns available:", df.columns.tolist())
    print("Missing columns:", [col for col in features + ['Direction'] if col not in df.columns])
    df = df.dropna(subset=features + ['Direction'])
    X = df[features]
    y = df['Direction']
    print("Direction distribution:", y.value_counts().to_dict())
    if len(X) == 0 or len(y) == 0:
        logging.warning("No valid data for ML model training.")
        return None, 0.0
    if y.nunique() < 2:
        print("⚠️ Not enough class variation for ML. Skipping.")
        return None, 0.0
    model = DecisionTreeClassifier()  # or LogisticRegression()
    model.fit(X, y)
    preds = model.predict(X)
    acc = accuracy_score(y, preds)
    logging.info(f"Model accuracy: {acc:.2f}")
    return model, acc

def update_google_sheets(trade_log, summary):
    """
    Log trades and summary stats to Google Sheets.
    """
    try:
        gc = gspread.service_account(filename='htoh-468014-c8802914cedc.json')
        sh = gc.open("AlgoTradingLog")
        # Ensure TradeLog worksheet exists
        try:
            log_ws = sh.worksheet("TradeLog")
        except gspread.exceptions.WorksheetNotFound:
            log_ws = sh.add_worksheet(title="TradeLog", rows="100", cols="20")
        log_ws.clear()
        log_ws.update('A1', [["Ticker", "Entry Date", "Entry Price", "Exit Date", "Exit Price", "P&L"]])
        trades_data = []
        for trade in trade_log['trades']:
            row = [
                trade.get('ticker', ''),
                str(trade.get('entry_date', '')),
                trade.get('entry_price', ''),
                str(trade.get('exit_date', '')),
                trade.get('exit_price', ''),
                trade.get('pnl', '')
            ]
            print("Logging to Google Sheets:", row)
            trades_data.append(row)
        if trades_data:
            log_ws.append_rows(trades_data)
        # Ensure Summary worksheet exists
        try:
            summary_ws = sh.worksheet("Summary")
        except gspread.exceptions.WorksheetNotFound:
            summary_ws = sh.add_worksheet(title="Summary", rows="10", cols="5")
        summary_ws.update("A1", [["Total P&L", "Win Ratio"], [trade_log['total_pnl'], trade_log['win_ratio']]])
        logging.info("Google Sheets updated successfully.")
    except Exception as e:
        print(f"Error updating Google Sheets: {e}")
        logging.error(f"Error updating Google Sheets: {e}")

def send_telegram_alert(message):
    """
    Send alert via Telegram Bot API.
    """
    try:
        def send_telegram_message(bot_token, chat_id, message):
            url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
            data = {"chat_id": chat_id, "text": message}
            response = requests.post(url, data=data)
            if response.status_code == 200:
                logging.info(f"Sent Telegram alert: {message}")
            else:
                logging.error(f"Telegram alert failed: {response.text}")
        send_telegram_message(BOT_TOKEN, CHAT_ID, message)
    except Exception as e:
        logging.error(f"Error sending Telegram alert: {e}")

def main():
    """
    Main loop for scheduled trading logic.
    """
    for ticker in STOCKS:
        df = fetch_data(ticker)
        if df is not None:
            df = calculate_indicators(df)
            signals = generate_signals(df)
            backtest_summary = run_backtest(df, signals)
            model, acc = train_ml_model(df)
            update_google_sheets(backtest_summary, backtest_summary)
            # Console output for trade signals
            print(f"\n--- {ticker} Trade Signals ---")
            if signals:
                for trade in signals:
                    msg = (f"BUY signal for {ticker} at {trade['entry_date'].strftime('%Y-%m-%d')}, "
                           f"entry price: {trade['entry_price']}, exit: {trade['exit_date'].strftime('%Y-%m-%d')}, "
                           f"exit price: {trade['exit_price']}, P&L: {trade['pnl']:.2f}")
                    print(msg)
                    send_telegram_alert(msg)
            else:
                print("No trade signals.")
            # Console output for backtest summary
            print(f"Total P&L: {backtest_summary['total_pnl']:.2f}")
            print(f"Win Ratio: {backtest_summary['win_ratio']:.2f}")
            print(f"Trades: {len(backtest_summary['trades'])}")
            print(f"Model Accuracy: {acc:.2f}")
    logging.info("Algo trading run complete.")

if __name__ == "__main__":
    import schedule
    def run_trading():
        main()

    schedule.every().day.at("15:30").do(run_trading)
    print("[Scheduler] Trading logic will run every day at 15:30.")
    while True:
        schedule.run_pending()
        time.sleep(30)