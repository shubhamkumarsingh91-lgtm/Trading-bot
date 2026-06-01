"""
Claude x Alpaca Intraday Trading Bot
=====================================
Trades stocks automatically using RSI + Moving Average signals.
Runs on Alpaca LIVE Trading with real money.

Setup:
  pip install alpaca-py pandas numpy requests schedule

Run:
  python trading_bot.py
"""

import os
import time
import logging
import datetime
import numpy as np
import pandas as pd
import schedule

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

# ─────────────────────────────────────────────
# CONFIG — reads API keys from environment variables (secure)
# ─────────────────────────────────────────────
API_KEY    = os.environ.get("ALPACA_API_KEY", "").strip()
SECRET_KEY = os.environ.get("ALPACA_SECRET_KEY", "").strip()
PAPER      = False  # ← LIVE trading with real money

if not API_KEY or not SECRET_KEY:
    raise ValueError("Missing ALPACA_API_KEY or ALPACA_SECRET_KEY environment variables!")

# Debug: print first 4 chars of key to verify it loaded (safe to log)
print(f"[DEBUG] API_KEY loaded: {API_KEY[:4]}... (length: {len(API_KEY)})")
print(f"[DEBUG] SECRET_KEY loaded: {SECRET_KEY[:4]}... (length: {len(SECRET_KEY)})")

# ─────────────────────────────────────────────
# STRATEGY SETTINGS
# ─────────────────────────────────────────────
WATCHLIST = ["CRWD", "AVGO", "SNOW", "PANW", "GTLB"]   # stocks to watch — updated June 2 2026
MAX_POSITION_USD   = 100    # $100 per stock = $500 spread across 5 stocks
RISK_PER_TRADE_PCT = 0.03   # stop loss = 3% below entry
PROFIT_TARGET_PCT  = 0.20   # take profit = 20% — capture big earnings moves
RSI_OVERSOLD       = 35     # buy signal threshold
RSI_OVERBOUGHT     = 75     # sell signal threshold
MA_SHORT           = 9      # fast moving average periods
MA_LONG            = 21     # slow moving average periods

# TRAILING STOP — locks in profit as stock moves up
TRAILING_STOP_PCT  = 0.05   # sell if stock drops 5% from its peak after buying

# ─────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("bot.log"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# CLIENTS
# ─────────────────────────────────────────────
trading_client = TradingClient(API_KEY, SECRET_KEY, paper=PAPER)
data_client    = StockHistoricalDataClient(API_KEY, SECRET_KEY)


# ─────────────────────────────────────────────
# INDICATORS
# ─────────────────────────────────────────────
def compute_rsi(prices: pd.Series, period: int = 14) -> float:
    delta = prices.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / loss
    rsi   = 100 - (100 / (1 + rs))
    return round(float(rsi.iloc[-1]), 2)


def compute_ma(prices: pd.Series, period: int) -> float:
    return round(float(prices.rolling(period).mean().iloc[-1]), 4)


# ─────────────────────────────────────────────
# MARKET DATA
# ─────────────────────────────────────────────
def get_bars(symbol: str, lookback_days: int = 5) -> pd.DataFrame:
    end   = datetime.datetime.now(datetime.timezone.utc)
    start = end - datetime.timedelta(days=lookback_days)
    req   = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame(15, TimeFrameUnit.Minute),  # 15-min bars
        start=start,
        end=end,
        feed="iex",  # free data feed for free Alpaca accounts
    )
    bars = data_client.get_stock_bars(req).df
    if isinstance(bars.index, pd.MultiIndex):
        bars = bars.xs(symbol, level="symbol")
    return bars


# ─────────────────────────────────────────────
# ACCOUNT HELPERS
# ─────────────────────────────────────────────
def get_buying_power() -> float:
    account = trading_client.get_account()
    return float(account.cash)


def get_position(symbol: str):
    try:
        return trading_client.get_open_position(symbol)
    except Exception:
        return None


def get_current_price(symbol: str, bars: pd.DataFrame) -> float:
    return float(bars["close"].iloc[-1])


# ─────────────────────────────────────────────
# ORDER HELPERS
# ─────────────────────────────────────────────
def place_buy(symbol: str, price: float):
    qty = int(MAX_POSITION_USD // price)
    if qty < 1:
        log.warning(f"[{symbol}] Not enough buying power for even 1 share at ${price:.2f}")
        return

    req = MarketOrderRequest(
        symbol=symbol,
        qty=qty,
        side=OrderSide.BUY,
        time_in_force=TimeInForce.DAY,
    )
    order = trading_client.submit_order(req)
    log.info(f"✅ BUY {qty} shares of {symbol} @ ~${price:.2f} | Order ID: {order.id}")

    stop  = round(price * (1 - RISK_PER_TRADE_PCT), 2)
    target = round(price * (1 + PROFIT_TARGET_PCT), 2)
    log.info(f"   Stop Loss: ${stop} | Take Profit: ${target}")
    return order


def place_sell(symbol: str, qty: float):
    req = MarketOrderRequest(
        symbol=symbol,
        qty=int(qty),
        side=OrderSide.SELL,
        time_in_force=TimeInForce.DAY,
    )
    order = trading_client.submit_order(req)
    log.info(f"🔴 SELL {int(qty)} shares of {symbol} | Order ID: {order.id}")
    return order


# ─────────────────────────────────────────────
# SIGNAL ENGINE
# ─────────────────────────────────────────────
def get_signal(symbol: str) -> str:
    """
    Returns: 'BUY', 'SELL', or 'HOLD'
    Strategy: RSI + MA crossover confirmation
    """
    try:
        bars   = get_bars(symbol)
        closes = bars["close"]

        rsi      = compute_rsi(closes)
        ma_short = compute_ma(closes, MA_SHORT)
        ma_long  = compute_ma(closes, MA_LONG)
        price    = get_current_price(symbol, bars)

        log.info(f"[{symbol}] Price=${price:.2f} | RSI={rsi} | MA{MA_SHORT}={ma_short:.2f} | MA{MA_LONG}={ma_long:.2f}")

        # BUY: Momentum strategy — MA9 above MA21 + RSI not extremely overbought
        if ma_short > ma_long and rsi < 75:
            return "BUY"

        # SELL: MA9 crosses below MA21 OR RSI extremely overbought
        if ma_short < ma_long or rsi > 85:
            return "SELL"

        return "HOLD"

    except Exception as e:
        log.error(f"[{symbol}] Signal error: {e}")
        return "HOLD"


# ─────────────────────────────────────────────
# POSITION MONITOR (stop loss / trailing stop / take profit)
# ─────────────────────────────────────────────
peak_prices = {}  # tracks highest price reached per position

def monitor_positions():
    for symbol in WATCHLIST:
        pos = get_position(symbol)
        if not pos:
            peak_prices.pop(symbol, None)  # reset peak if no position
            continue

        entry_price   = float(pos.avg_entry_price)
        current_price = float(pos.current_price)
        qty           = float(pos.qty)
        pnl_pct       = (current_price - entry_price) / entry_price

        # Track highest price reached
        if symbol not in peak_prices or current_price > peak_prices[symbol]:
            peak_prices[symbol] = current_price

        peak_price    = peak_prices[symbol]
        drop_from_peak = (peak_price - current_price) / peak_price

        log.info(f"[{symbol}] Entry=${entry_price:.2f} | Now=${current_price:.2f} | Peak=${peak_price:.2f} | PnL={pnl_pct*100:.1f}%")

        # Hard stop loss — protect capital
        if pnl_pct <= -RISK_PER_TRADE_PCT:
            log.warning(f"[{symbol}] 🛑 STOP LOSS at {pnl_pct*100:.1f}%")
            place_sell(symbol, qty)
            peak_prices.pop(symbol, None)

        # Trailing stop — lock in profits (only activates after 5% gain)
        elif pnl_pct >= 0.05 and drop_from_peak >= TRAILING_STOP_PCT:
            log.info(f"[{symbol}] 🔒 TRAILING STOP — dropped {drop_from_peak*100:.1f}% from peak ${peak_price:.2f} | Locking profit at {pnl_pct*100:.1f}%")
            place_sell(symbol, qty)
            peak_prices.pop(symbol, None)

        # Max take profit safety net at 20%
        elif pnl_pct >= PROFIT_TARGET_PCT:
            log.info(f"[{symbol}] 🎯 MAX TAKE PROFIT at {pnl_pct*100:.1f}%")
            place_sell(symbol, qty)
            peak_prices.pop(symbol, None)


# ─────────────────────────────────────────────
# MARKET HOURS CHECK
# ─────────────────────────────────────────────
def is_market_open() -> bool:
    clock = trading_client.get_clock()
    return clock.is_open


def close_all_positions_eod():
    """Close all open positions 5 min before market close (3:55 PM ET)"""
    log.info("⏰ End of day — closing all open positions")
    for symbol in WATCHLIST:
        pos = get_position(symbol)
        if pos:
            place_sell(symbol, float(pos.qty))


# ─────────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────────
def run_strategy():
    if not is_market_open():
        log.info("Market is closed. Skipping.")
        return

    log.info("=" * 50)
    log.info("🤖 Running strategy scan...")
    log.info(f"💰 Buying Power: ${get_buying_power():,.2f}")

    # Check existing positions first
    monitor_positions()

    # Scan for new entries
    for symbol in WATCHLIST:
        pos = get_position(symbol)
        if pos:
            log.info(f"[{symbol}] Already in position — skipping entry scan")
            continue

        signal = get_signal(symbol)
        log.info(f"[{symbol}] Signal: {signal}")

        if signal == "BUY":
            bars  = get_bars(symbol)
            price = get_current_price(symbol, bars)
            if get_buying_power() >= price:
                place_buy(symbol, price)
            else:
                log.warning(f"[{symbol}] Insufficient buying power")

    log.info("✅ Scan complete")


# ─────────────────────────────────────────────
# SCHEDULER
# ─────────────────────────────────────────────
if __name__ == "__main__":
    log.info("🚀 Trading bot started!")
    log.info(f"   Watchlist: {WATCHLIST}")
    log.info(f"   Max per trade: ${MAX_POSITION_USD}")
    log.info(f"   Stop loss: {RISK_PER_TRADE_PCT*100}% | Take profit: {PROFIT_TARGET_PCT*100}%")
    log.info(f"   Mode: {'📄 PAPER (fake money)' if PAPER else '💵 LIVE'}")

    # Run every 15 minutes during market hours
    schedule.every(15).minutes.do(run_strategy)

    # Close all positions at end of day (3:55 PM ET)
    schedule.every().monday.at("15:55").do(close_all_positions_eod)
    schedule.every().tuesday.at("15:55").do(close_all_positions_eod)
    schedule.every().wednesday.at("15:55").do(close_all_positions_eod)
    schedule.every().thursday.at("15:55").do(close_all_positions_eod)
    schedule.every().friday.at("15:55").do(close_all_positions_eod)

    # Run once immediately on start
    run_strategy()

    while True:
        schedule.run_pending()
        time.sleep(30)
