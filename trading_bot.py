"""
╔══════════════════════════════════════════════════════════════╗
║   SHUBHAM · MULTI-LEGEND DAY TRADING BOT                    ║
║                                                              ║
║   ✅ PDT RULE ELIMINATED — June 4, 2026 (FINRA Rule 4210)  ║
║   ✅ Unlimited day trades — no $25k minimum required        ║
║                                                              ║
║   Strategies:                                                ║
║   • Mark Minervini  — SEPA Stage 2 trend template           ║
║   • Linda Raschke   — Holy Grail (ADX + EMA pullback)       ║
║   • Paul Tudor Jones — 2% risk rule, 3:1 reward/risk        ║
║   • William O'Neil  — Volume surge confirmation              ║
║   • Jesse Livermore — No chasing, pivot confirmation         ║
╚══════════════════════════════════════════════════════════════╝

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
from zoneinfo import ZoneInfo
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

# ══════════════════════════════════════════════════════════════
#  CONFIG  ← only edit this section
# ══════════════════════════════════════════════════════════════

API_KEY    = "AKY4VXSWEKQMSQESHZW7RWMER3"
SECRET_KEY = "pc8ktPALExaPbUKpFz4CPJ4wFiWSEWnFfHSKmSPBYj3"
PAPER      = False   # False = real money, True = paper/fake money

# ── WATCHLIST ─────────────────────────────────────────────────
# Tuned for $499 account — all under $250/share
# NVDA : AI leader, ~$130, strong Stage 2 uptrend
# PANW : Palo Alto cybersecurity, ~$185, consistent momentum
# AVGO : Broadcom AI chips, ~$200, sector leader
# SOFI : SoFi Technologies, ~$15, high volume, very affordable
# PLTR : Palantir AI/data, ~$40, strong institutional buying
WATCHLIST = ["NVDA", "PANW", "AVGO", "SOFI", "PLTR"]

# ── Paul Tudor Jones Risk Rules ───────────────────────────────
ACCOUNT_RISK_PCT  = 0.02   # Risk 2% of account per trade (~$10 on $499)
REWARD_RISK_RATIO = 3.0    # 3:1 reward-to-risk minimum (PTJ rule)
MAX_POSITION_USD  = 450    # Hard cap per position

# ── Stop / Target (auto-calculated from risk rules) ───────────
STOP_LOSS_PCT    = 0.02                              # 2% stop loss
TAKE_PROFIT_PCT  = STOP_LOSS_PCT * REWARD_RISK_RATIO # 6% take profit
TRAILING_STOP_PCT = 0.03                             # 3% trailing stop

# ── Minervini EMA Stack ───────────────────────────────────────
EMA_FAST = 9    # Fast EMA — short-term trend
EMA_MID  = 20   # Mid EMA  — Raschke pullback target
EMA_SLOW = 50   # Slow EMA — Stage 2 confirmation

# ── Raschke Holy Grail ────────────────────────────────────────
ADX_PERIOD = 14
ADX_MIN    = 25   # ADX must be above 25 (strong trend)

# ── O'Neil Volume Confirmation ────────────────────────────────
VOLUME_SURGE_MULT = 1.5   # Volume must be 1.5× average
VOLUME_AVG_PERIOD = 20

# ── RSI Filter ───────────────────────────────────────────────
RSI_PERIOD  = 14
RSI_MIN_BUY = 40   # Don't buy oversold — wait for momentum
RSI_MAX_BUY = 72   # Don't buy overbought
RSI_SELL    = 80   # Exit on blow-off top

# ── Timing ───────────────────────────────────────────────────
SCAN_INTERVAL_MIN = 10                          # Scan every 10 min (no PDT limit!)
EOD_TIME          = datetime.time(15, 50)       # EOD review at 3:50 PM ET
MIN_PROFIT_HOLD   = 0.02                        # Hold overnight if up 2%+

ET = ZoneInfo("America/New_York")

# ══════════════════════════════════════════════════════════════
#  LOGGING
# ══════════════════════════════════════════════════════════════

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler("bot.log"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger("bot")

# ══════════════════════════════════════════════════════════════
#  CLIENTS
# ══════════════════════════════════════════════════════════════

trading = TradingClient(API_KEY, SECRET_KEY, paper=PAPER)
market  = StockHistoricalDataClient(API_KEY, SECRET_KEY)

peak_prices = {}   # Trailing stop tracker — highest price seen per position

# ══════════════════════════════════════════════════════════════
#  INDICATORS
# ══════════════════════════════════════════════════════════════

def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def compute_rsi(close: pd.Series, period: int = 14) -> float:
    delta = close.diff()
    gain  = delta.clip(lower=0).ewm(com=period-1, min_periods=period).mean()
    loss  = (-delta.clip(upper=0)).ewm(com=period-1, min_periods=period).mean()
    rs    = gain / loss
    return round(float((100 - (100 / (1 + rs))).iloc[-1]), 2)


def compute_adx(df: pd.DataFrame, period: int = 14) -> float:
    """ADX — Raschke uses > 25 to confirm trend strength before entering."""
    high, low, close = df["high"], df["low"], df["close"]
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs()
    ], axis=1).max(axis=1)
    dm_plus  = ((high - high.shift()) > (low.shift() - low)).astype(float) * (high - high.shift()).clip(lower=0)
    dm_minus = ((low.shift() - low) > (high - high.shift())).astype(float) * (low.shift() - low).clip(lower=0)
    atr      = tr.ewm(span=period, adjust=False).mean()
    di_plus  = 100 * dm_plus.ewm(span=period, adjust=False).mean() / atr
    di_minus = 100 * dm_minus.ewm(span=period, adjust=False).mean() / atr
    dx  = (100 * (di_plus - di_minus).abs() / (di_plus + di_minus).replace(0, np.nan)).fillna(0)
    return round(float(dx.ewm(span=period, adjust=False).mean().iloc[-1]), 2)


def volume_surge(df: pd.DataFrame) -> tuple[float, float]:
    vol_avg = df["volume"].rolling(VOLUME_AVG_PERIOD).mean().iloc[-1]
    vol_now = df["volume"].iloc[-1]
    return float(vol_now), float(vol_avg)

# ══════════════════════════════════════════════════════════════
#  MARKET DATA
# ══════════════════════════════════════════════════════════════

def get_bars(symbol: str, lookback_days: int = 5) -> pd.DataFrame:
    end   = datetime.datetime.now(datetime.timezone.utc)
    start = end - datetime.timedelta(days=lookback_days)
    req   = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame(15, TimeFrameUnit.Minute),
        start=start, end=end,
        feed="iex",
    )
    bars = market.get_stock_bars(req).df
    if isinstance(bars.index, pd.MultiIndex):
        bars = bars.xs(symbol, level="symbol")
    return bars.reset_index()

# ══════════════════════════════════════════════════════════════
#  POSITION SIZING  (Paul Tudor Jones method)
# ══════════════════════════════════════════════════════════════

def calc_position_size(entry_price: float) -> int:
    """
    PTJ: Risk exactly 2% of account equity per trade.
    Shares = (Equity × 2%) / (Entry × Stop%)
    """
    try:
        equity         = float(trading.get_account().equity)
        risk_dollars   = equity * ACCOUNT_RISK_PCT
        risk_per_share = entry_price * STOP_LOSS_PCT
        shares         = int(risk_dollars / risk_per_share)
        max_shares     = int(MAX_POSITION_USD / entry_price)
        return min(shares, max_shares)
    except Exception as e:
        log.error(f"Position size error: {e}")
        return int(MAX_POSITION_USD // entry_price)

# ══════════════════════════════════════════════════════════════
#  SIGNAL ENGINE
#  Minervini Stage 2 + Raschke Holy Grail + O'Neil Volume
# ══════════════════════════════════════════════════════════════

def get_signal(symbol: str) -> dict:
    result = {
        "symbol": symbol, "signal": "HOLD",
        "price": 0, "rsi": 0, "adx": 0,
        "ema9": 0, "ema20": 0, "ema50": 0,
        "vol": 0, "vol_avg": 0, "reason": ""
    }
    try:
        df    = get_bars(symbol)
        close = df["close"]
        price = float(close.iloc[-1])

        e9  = ema(close, EMA_FAST)
        e20 = ema(close, EMA_MID)
        e50 = ema(close, EMA_SLOW)

        rsi          = compute_rsi(close, RSI_PERIOD)
        adx          = compute_adx(df, ADX_PERIOD)
        vol, vol_avg = volume_surge(df)
        cur_e9       = float(e9.iloc[-1])
        cur_e20      = float(e20.iloc[-1])
        cur_e50      = float(e50.iloc[-1])
        prev_e9      = float(e9.iloc[-2])

        result.update({
            "price": price, "rsi": rsi, "adx": adx,
            "ema9": cur_e9, "ema20": cur_e20, "ema50": cur_e50,
            "vol": vol, "vol_avg": vol_avg
        })

        # ── SELL conditions (any one fires) ──────────────────
        if cur_e9 < cur_e20:
            result["signal"] = "SELL"
            result["reason"] = f"EMA9 < EMA20 — trend broken (Minervini exit)"
            return result
        if rsi > RSI_SELL:
            result["signal"] = "SELL"
            result["reason"] = f"RSI {rsi} — blow-off top (Livermore: sell into strength)"
            return result
        if adx < 18:
            result["signal"] = "SELL"
            result["reason"] = f"ADX {adx} — trend collapsing (Raschke: exit)"
            return result

        # ── BUY conditions (all must pass) ───────────────────
        stage2         = cur_e9 > cur_e20 > cur_e50
        strong_trend   = adx >= ADX_MIN
        pullback_entry = close.iloc[-3] <= cur_e20 * 1.01 or (price > cur_e9 and prev_e9 <= cur_e20)
        vol_confirmed  = vol >= vol_avg * VOLUME_SURGE_MULT
        rsi_ok         = RSI_MIN_BUY <= rsi <= RSI_MAX_BUY

        if not stage2:
            result["reason"] = f"Not Stage 2 — EMAs: {cur_e9:.2f}/{cur_e20:.2f}/{cur_e50:.2f}"
        elif not strong_trend:
            result["reason"] = f"ADX {adx:.1f} < {ADX_MIN} — weak trend"
        elif not rsi_ok:
            result["reason"] = f"RSI {rsi} outside {RSI_MIN_BUY}–{RSI_MAX_BUY}"
        elif not pullback_entry and not vol_confirmed:
            result["reason"] = f"No pullback entry & volume only {vol/vol_avg:.1f}× (need {VOLUME_SURGE_MULT}×)"
        else:
            result["signal"] = "BUY"
            tags = []
            if pullback_entry: tags.append("Raschke pullback ✓")
            if vol_confirmed:  tags.append(f"O'Neil vol {vol/vol_avg:.1f}× ✓")
            result["reason"] = f"Minervini Stage2 ✓ | ADX {adx} ✓ | {' | '.join(tags)}"

    except Exception as e:
        log.error(f"[{symbol}] Signal error: {e}")

    return result

# ══════════════════════════════════════════════════════════════
#  ORDER EXECUTION
# ══════════════════════════════════════════════════════════════

def place_buy(symbol: str, price: float):
    qty = calc_position_size(price)
    if qty < 1:
        log.warning(f"  [{symbol}] Price ${price:.2f} too high for position size — skip")
        return
    stop   = round(price * (1 - STOP_LOSS_PCT), 2)
    target = round(price * (1 + TAKE_PROFIT_PCT), 2)
    try:
        order = trading.submit_order(MarketOrderRequest(
            symbol=symbol, qty=qty,
            side=OrderSide.BUY,
            time_in_force=TimeInForce.DAY
        ))
        log.info(f"  ✅ BUY  {qty} × {symbol} @ ~${price:.2f}")
        log.info(f"     Stop ${stop}  →  Target ${target}  (PTJ 3:1 R/R)")
        log.info(f"     Spend: ${qty*price:.2f}  |  Max loss: ${qty*price*STOP_LOSS_PCT:.2f}")
        return order
    except Exception as e:
        log.error(f"  [{symbol}] Buy failed: {e}")


def place_sell(symbol: str, qty: float, reason: str = ""):
    try:
        trading.close_position(symbol)
        log.info(f"  🔴 SELL {int(qty)} × {symbol}  [{reason}]")
        peak_prices.pop(symbol, None)
    except Exception as e:
        log.error(f"  [{symbol}] Sell failed: {e}")

# ══════════════════════════════════════════════════════════════
#  POSITION MONITOR
#  Runs every scan — checks stop loss, trailing stop, take profit
#  No PDT restrictions — can sell and re-buy same stock same day
# ══════════════════════════════════════════════════════════════

def monitor_positions():
    for sym in WATCHLIST:
        try:
            pos = trading.get_open_position(sym)
        except Exception:
            peak_prices.pop(sym, None)
            continue

        entry  = float(pos.avg_entry_price)
        price  = float(pos.current_price)
        qty    = float(pos.qty)
        pnl_pct = (price - entry) / entry

        # Track peak for trailing stop
        if sym not in peak_prices or price > peak_prices[sym]:
            peak_prices[sym] = price
        peak           = peak_prices[sym]
        drop_from_peak = (peak - price) / peak

        log.info(
            f"  [{sym}]  ${entry:.2f} → ${price:.2f}  "
            f"P&L: {'+' if pnl_pct>=0 else ''}{pnl_pct*100:.1f}%  "
            f"Peak: ${peak:.2f}  Drop: {drop_from_peak*100:.1f}%"
        )

        # PTJ: cut losses hard at 2%
        if pnl_pct <= -STOP_LOSS_PCT:
            place_sell(sym, qty, reason=f"STOP LOSS {pnl_pct*100:.1f}% (PTJ rule)")

        # Trailing stop: locks in gains after 3% rise
        elif pnl_pct >= 0.03 and drop_from_peak >= TRAILING_STOP_PCT:
            place_sell(sym, qty, reason=f"TRAILING STOP — peak ${peak:.2f}, dropped {drop_from_peak*100:.1f}%")

        # Take profit at 6%
        elif pnl_pct >= TAKE_PROFIT_PCT:
            place_sell(sym, qty, reason=f"TAKE PROFIT +{pnl_pct*100:.1f}%")

        # Trend signal exit
        else:
            sig = get_signal(sym)
            if sig["signal"] == "SELL":
                place_sell(sym, qty, reason=f"TREND: {sig['reason']}")

# ══════════════════════════════════════════════════════════════
#  EOD REVIEW — 3:50 PM ET
#  Livermore: never hold losers overnight
#  Minervini: hold winners if trend still intact
# ══════════════════════════════════════════════════════════════

def eod_review():
    log.info("⏰ EOD — swing review (Livermore rule: no losers overnight)...")
    for sym in WATCHLIST:
        try:
            pos = trading.get_open_position(sym)
        except Exception:
            continue

        entry   = float(pos.avg_entry_price)
        price   = float(pos.current_price)
        qty     = float(pos.qty)
        pnl_pct = (price - entry) / entry

        if pnl_pct >= MIN_PROFIT_HOLD:
            sig = get_signal(sym)
            if sig["ema9"] > sig["ema20"] and sig["adx"] >= 20 and sig["rsi"] < 78:
                log.info(f"  [{sym}] 🌙 HOLD OVERNIGHT +{pnl_pct*100:.1f}% — trend intact")
                continue

        reason = f"EOD +{pnl_pct*100:.1f}% profit" if pnl_pct > 0 else f"EOD {pnl_pct*100:.1f}% — cutting loss"
        place_sell(sym, qty, reason=reason)

    log.info("⏰ EOD complete")

# ══════════════════════════════════════════════════════════════
#  MAIN SCAN — runs every 10 minutes
# ══════════════════════════════════════════════════════════════

def run_strategy():
    clock = trading.get_clock()
    if not clock.is_open:
        log.info("Market closed — waiting")
        return

    now_et = datetime.datetime.now(ET)
    if now_et.time() >= EOD_TIME:
        eod_review()
        return

    account = trading.get_account()
    equity  = float(account.equity)
    cash    = float(account.cash)

    log.info("=" * 65)
    log.info(f"  📡 {now_et.strftime('%a %b %d  %H:%M ET')}  "
             f"Equity: ${equity:,.2f}  Cash: ${cash:,.2f}")
    log.info("=" * 65)

    # Step 1 — protect existing positions
    monitor_positions()

    # Step 2 — scan for new entries
    # PDT rule gone — can re-enter same stock same day if signal fires again
    for sym in WATCHLIST:
        try:
            trading.get_open_position(sym)
            log.info(f"  [{sym}] In position — skipping entry scan")
            continue
        except Exception:
            pass

        sig = get_signal(sym)
        vol_ratio = sig['vol'] / sig['vol_avg'] if sig['vol_avg'] > 0 else 0
        log.info(
            f"  [{sym}]  ${sig['price']:.2f}  "
            f"RSI={sig['rsi']}  ADX={sig['adx']}  "
            f"EMA {sig['ema9']:.1f}/{sig['ema20']:.1f}/{sig['ema50']:.1f}  "
            f"Vol {vol_ratio:.1f}×  → {sig['signal']}  ({sig['reason']})"
        )

        if sig["signal"] == "BUY":
            if cash < sig["price"]:
                log.warning(f"  [{sym}] Not enough cash (${cash:.2f})")
                continue
            place_buy(sym, sig["price"])
            account = trading.get_account()  # refresh cash after buy
            cash    = float(account.cash)

    log.info("")

# ══════════════════════════════════════════════════════════════
#  STARTUP BANNER
# ══════════════════════════════════════════════════════════════

def print_banner():
    acct = trading.get_account()
    mode = "📄 PAPER (fake money)" if PAPER else "💵 LIVE (real money)"
    log.info("╔" + "═" * 63 + "╗")
    log.info("║  SHUBHAM · MULTI-LEGEND DAY TRADING BOT" + " " * 22 + "║")
    log.info("║  ✅ PDT rule eliminated June 4, 2026 — unlimited trades  ║")
    log.info("║" + " " * 63 + "║")
    log.info(f"║  Mode      : {mode}" + " " * (50 - len(mode)) + "║")
    log.info(f"║  Watchlist : {', '.join(WATCHLIST)}" + " " * (50 - len(', '.join(WATCHLIST))) + "║")
    log.info(f"║  Scan      : every {SCAN_INTERVAL_MIN} minutes" + " " * 43 + "║")
    log.info(f"║  Stop loss : {STOP_LOSS_PCT*100:.0f}%  |  Take profit: {TAKE_PROFIT_PCT*100:.0f}%  |  Trailing: {TRAILING_STOP_PCT*100:.0f}%  ║")
    log.info(f"║  R/R       : 1:{REWARD_RISK_RATIO:.0f}  (Paul Tudor Jones minimum)" + " " * 25 + "║")
    log.info(f"║  Equity    : ${float(acct.equity):>10,.2f}" + " " * 44 + "║")
    log.info(f"║  Cash      : ${float(acct.cash):>10,.2f}" + " " * 44 + "║")
    log.info("╚" + "═" * 63 + "╝")


# ══════════════════════════════════════════════════════════════
#  RUN
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print_banner()

    schedule.every(SCAN_INTERVAL_MIN).minutes.do(run_strategy)

    for day in ["monday", "tuesday", "wednesday", "thursday", "friday"]:
        getattr(schedule.every(), day).at("15:50").do(eod_review)

    run_strategy()  # run once immediately on start

    while True:
        schedule.run_pending()
        time.sleep(30)
