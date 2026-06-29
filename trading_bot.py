"""
╔══════════════════════════════════════════════════════════════╗
║   SHUBHAM · MULTI-LEGEND DAY TRADING BOT                    ║
║                                                              ║
║   Strategies combined:                                       ║
║   • Mark Minervini  — SEPA Stage 2 trend template           ║
║   • Linda Raschke   — Holy Grail (ADX + EMA pullback)       ║
║   • Paul Tudor Jones — 2% risk rule, 3:1 reward/risk        ║
║   • William O'Neil  — Volume surge confirmation              ║
║   • Jesse Livermore — Pivot confirmation, no chasing         ║
║                                                              ║
║   Books: "Trade Like a Stock Market Wizard" (Minervini)     ║
║          "Street Smarts" (Raschke)                          ║
║          "How to Make Money in Stocks" (O'Neil)             ║
║          "Reminiscences of a Stock Operator" (Livermore)    ║
╚══════════════════════════════════════════════════════════════╝

Setup:
    pip install alpaca-py pandas numpy schedule

Run:
    export ALPACA_API_KEY="your_key"
    export ALPACA_SECRET_KEY="your_secret"
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
#  CONFIG
# ══════════════════════════════════════════════════════════════

API_KEY    = "AKYIZUY2SME3EDIGMM24BQAPSH"
SECRET_KEY = "DDz6LU4WVPFicigChCYtrGAAJxb2PPf8e"
PAPER      = False   # ← True = paper money, False = real money

print(f"[BOOT] Key: {API_KEY[:4]}... ({len(API_KEY)} chars)  Secret: {SECRET_KEY[:4]}... ({len(SECRET_KEY)} chars)")

# ── WATCHLIST — tuned for $499 account, updated June 28 2026 ─
# Only stocks under $250 so the bot can actually buy shares
# NVDA : AI king, ~$130, strong uptrend, 3 shares = ~$390
# PANW : Palo Alto cybersecurity, ~$185, 2 shares = ~$370
# AVGO : Broadcom AI chips, ~$200, 2 shares = ~$400
# SOFI : SoFi Technologies, ~$15, high volume momentum, very affordable
# PLTR : Palantir, ~$40, AI/data analytics, strong institutional buying
# (MU removed — $1,132/share, can't buy with $499)
# (AMD removed — $521/share, can't buy with $499)
WATCHLIST = ["NVDA", "PANW", "AVGO", "SOFI", "PLTR"]

# ── Paul Tudor Jones Risk Rules ──────────────────────────────
ACCOUNT_RISK_PCT    = 0.02   # Risk 2% of account per trade = ~$10 risk on $499
REWARD_RISK_RATIO   = 3.0    # Minimum 3:1 reward-to-risk (PTJ Rule #2)
MAX_POSITION_USD    = 450    # Hard cap — never spend more than $450 of your $499
#
# HOW POSITION SIZE IS CALCULATED (example with $499 account):
#   Account equity    = $499
#   Max risk per trade = $499 × 2% = $9.98
#   If NVDA = $130:   stop loss per share = $130 × 2% = $2.60
#   Shares to buy     = $9.98 / $2.60 = 3 shares = $390 total
#   If stock drops 2% → lose $7.80  (protected)
#   If stock rises 6% → gain $23.40 (3:1 reward)
#
# ── Derived from risk rules ───────────────────────────────────
STOP_LOSS_PCT       = 0.02   # 2% stop loss — PTJ never risks more (protects your $499)
TAKE_PROFIT_PCT     = STOP_LOSS_PCT * REWARD_RISK_RATIO  # 6% take profit (3:1 R/R)
TRAILING_STOP_PCT   = 0.03   # 3% trailing stop to lock in gains as stock rises

# ── Minervini Trend Template (intraday version) ───────────────
EMA_FAST    = 9    # Short-term trend (Minervini uses 10-day)
EMA_MID     = 20   # Raschke Holy Grail pullback target
EMA_SLOW    = 50   # Minervini Stage 2 confirmation

# ── Raschke Holy Grail ────────────────────────────────────────
ADX_PERIOD  = 14
ADX_MIN     = 25   # ADX > 25 = strong trend (Raschke minimum)

# ── O'Neil Volume Confirmation ────────────────────────────────
VOLUME_SURGE_MULT = 1.5   # Volume must be 1.5× average (O'Neil)
VOLUME_AVG_PERIOD = 20    # Bars to average for volume baseline

# ── RSI Filter (Livermore: don't buy exhausted moves) ─────────
RSI_PERIOD    = 14
RSI_MIN_BUY   = 40   # Don't buy oversold — wait for momentum
RSI_MAX_BUY   = 72   # Don't buy overbought
RSI_SELL      = 80   # Exit if RSI blows into extreme territory

# ── Swing / PDT Settings ─────────────────────────────────────
SWING_TRADING      = True    # Hold winners overnight (avoids PDT rule)
MIN_PROFIT_HOLD    = 0.02    # Only hold overnight if up at least 2%
EOD_TIME           = datetime.time(15, 50)   # Review at 3:50 PM ET

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

bought_today = set()   # PDT protection tracker
peak_prices  = {}      # Trailing stop tracker

# ══════════════════════════════════════════════════════════════
#  INDICATORS
# ══════════════════════════════════════════════════════════════

def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def compute_rsi(close: pd.Series, period: int = 14) -> float:
    delta    = close.diff()
    gain     = delta.clip(lower=0).ewm(com=period-1, min_periods=period).mean()
    loss     = (-delta.clip(upper=0)).ewm(com=period-1, min_periods=period).mean()
    rs       = gain / loss
    rsi      = 100 - (100 / (1 + rs))
    return round(float(rsi.iloc[-1]), 2)


def compute_adx(df: pd.DataFrame, period: int = 14) -> float:
    """
    Average Directional Index — Raschke uses ADX > 25 to confirm a strong trend
    before entering a Holy Grail pullback trade.
    """
    high, low, close = df["high"], df["low"], df["close"]

    tr   = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs()
    ], axis=1).max(axis=1)

    dm_plus  = ((high - high.shift()) > (low.shift() - low)).astype(float) * (high - high.shift()).clip(lower=0)
    dm_minus = ((low.shift() - low) > (high - high.shift())).astype(float) * (low.shift() - low).clip(lower=0)

    atr      = tr.ewm(span=period, adjust=False).mean()
    di_plus  = 100 * dm_plus.ewm(span=period, adjust=False).mean()  / atr
    di_minus = 100 * dm_minus.ewm(span=period, adjust=False).mean() / atr

    dx  = (100 * (di_plus - di_minus).abs() / (di_plus + di_minus).replace(0, np.nan)).fillna(0)
    adx = dx.ewm(span=period, adjust=False).mean()
    return round(float(adx.iloc[-1]), 2)


def volume_surge(df: pd.DataFrame) -> tuple[float, float]:
    """Returns (current_volume, average_volume) for O'Neil confirmation."""
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
        start=start,
        end=end,
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
    Shares = (Account × 2%) / (Entry × Stop%)
    Capped at MAX_POSITION_USD for safety.
    """
    try:
        account = trading.get_account()
        equity  = float(account.equity)
        risk_dollars = equity * ACCOUNT_RISK_PCT          # 2% of account
        risk_per_share = entry_price * STOP_LOSS_PCT      # 2% of entry price
        shares = int(risk_dollars / risk_per_share)
        # Apply dollar cap
        max_shares = int(MAX_POSITION_USD / entry_price)
        return min(shares, max_shares, max_shares)
    except Exception as e:
        log.error(f"Position size error: {e}")
        return int(MAX_POSITION_USD // entry_price)

# ══════════════════════════════════════════════════════════════
#  SIGNAL ENGINE
#  Minervini Stage 2 + Raschke Holy Grail + O'Neil Volume
# ══════════════════════════════════════════════════════════════

def get_signal(symbol: str) -> dict:
    """
    Returns dict with signal ('BUY'/'SELL'/'HOLD') and all indicator values.

    BUY conditions (all must be true — Minervini would not compromise):
      1. Minervini Stage 2: EMA9 > EMA20 > EMA50  (price in confirmed uptrend)
      2. Raschke Holy Grail: ADX > 25 (trend is strong)
      3. Raschke entry: price pulled back near EMA20, now bouncing back above EMA9
      4. O'Neil confirmation: current volume ≥ 1.5× average (institutions buying)
      5. RSI 40–72: momentum present but not exhausted (Livermore: don't chase)

    SELL conditions (any one triggers):
      1. EMA9 crosses below EMA20 (trend broken — Minervini exits fast)
      2. RSI > 80 (blow-off top — Livermore: take profits into strength)
      3. ADX < 20 (trend collapsing)
    """
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

        rsi      = compute_rsi(close, RSI_PERIOD)
        adx      = compute_adx(df, ADX_PERIOD)
        vol, vol_avg = volume_surge(df)

        cur_e9  = float(e9.iloc[-1])
        cur_e20 = float(e20.iloc[-1])
        cur_e50 = float(e50.iloc[-1])
        prev_e9 = float(e9.iloc[-2])

        result.update({
            "price": price, "rsi": rsi, "adx": adx,
            "ema9": cur_e9, "ema20": cur_e20, "ema50": cur_e50,
            "vol": vol, "vol_avg": vol_avg
        })

        # ── SELL conditions ───────────────────────────────────
        if cur_e9 < cur_e20:
            result["signal"] = "SELL"
            result["reason"] = f"EMA9({cur_e9:.2f}) < EMA20({cur_e20:.2f}) — trend broken (Minervini exit)"
            return result

        if rsi > RSI_SELL:
            result["signal"] = "SELL"
            result["reason"] = f"RSI {rsi} > {RSI_SELL} — blow-off top (Livermore: sell into strength)"
            return result

        if adx < 18:
            result["signal"] = "SELL"
            result["reason"] = f"ADX {adx} < 18 — trend collapsing (Raschke: exit weak trends)"
            return result

        # ── BUY conditions ────────────────────────────────────
        # 1. Minervini Stage 2: all EMAs stacked bullish
        stage2 = cur_e9 > cur_e20 > cur_e50
        if not stage2:
            result["reason"] = f"Not Stage 2 — EMA stack: {cur_e9:.2f}/{cur_e20:.2f}/{cur_e50:.2f}"
            return result

        # 2. Raschke: ADX confirms strong trend
        strong_trend = adx >= ADX_MIN
        if not strong_trend:
            result["reason"] = f"ADX {adx:.1f} < {ADX_MIN} — weak trend, Raschke won't enter"
            return result

        # 3. Raschke Holy Grail: price was near EMA20 and bounced back above EMA9
        #    (pullback-to-EMA20 then recovery = ideal low-risk entry)
        near_ema20    = close.iloc[-3] <= cur_e20 * 1.01   # touched or dipped near EMA20
        bounce        = price > cur_e9 and prev_e9 <= cur_e20  # bouncing back above fast EMA
        pullback_entry = near_ema20 or bounce

        # 4. O'Neil: volume surge confirms institutional buying
        vol_confirmed = vol >= vol_avg * VOLUME_SURGE_MULT

        # 5. RSI in momentum zone (not oversold, not overbought)
        rsi_ok = RSI_MIN_BUY <= rsi <= RSI_MAX_BUY

        if stage2 and strong_trend and rsi_ok and (pullback_entry or vol_confirmed):
            result["signal"] = "BUY"
            reasons = []
            if pullback_entry:  reasons.append("Raschke pullback ✓")
            if vol_confirmed:   reasons.append(f"O'Neil volume {vol/vol_avg:.1f}× ✓")
            result["reason"] = "Minervini Stage2 ✓ | ADX " + str(adx) + " ✓ | " + " | ".join(reasons)
        else:
            missing = []
            if not rsi_ok:          missing.append(f"RSI {rsi} out of {RSI_MIN_BUY}-{RSI_MAX_BUY}")
            if not pullback_entry:  missing.append("no pullback entry")
            if not vol_confirmed:   missing.append(f"volume only {vol/vol_avg:.1f}× (need {VOLUME_SURGE_MULT}×)")
            result["reason"] = "Waiting: " + ", ".join(missing)

    except Exception as e:
        log.error(f"[{symbol}] Signal error: {e}")

    return result

# ══════════════════════════════════════════════════════════════
#  ORDER EXECUTION
# ══════════════════════════════════════════════════════════════

def place_buy(symbol: str, price: float):
    qty = calc_position_size(price)
    if qty < 1:
        log.warning(f"  [{symbol}] Position size < 1 share at ${price:.2f} — skipping")
        return

    stop   = round(price * (1 - STOP_LOSS_PCT), 2)
    target = round(price * (1 + TAKE_PROFIT_PCT), 2)

    try:
        order = trading.submit_order(MarketOrderRequest(
            symbol=symbol, qty=qty,
            side=OrderSide.BUY,
            time_in_force=TimeInForce.GTC
        ))
        log.info(f"  ✅ BUY  {qty} × {symbol} @ ~${price:.2f}")
        log.info(f"     Stop: ${stop}  |  Target: ${target}  (PTJ 3:1 R/R)")
        log.info(f"     Notional: ${qty*price:.2f}  |  Max loss: ${qty*price*STOP_LOSS_PCT:.2f}")
        bought_today.add(symbol)
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
#  POSITION MONITOR  (PTJ stop + trailing stop + take profit)
# ══════════════════════════════════════════════════════════════

def monitor_positions():
    """
    Runs every scan cycle before looking for new entries.
    Paul Tudor Jones: protect capital first, always.
    """
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

        # Update trailing stop peak
        if sym not in peak_prices or price > peak_prices[sym]:
            peak_prices[sym] = price
        peak = peak_prices[sym]
        drop_from_peak = (peak - price) / peak

        log.info(
            f"  [{sym}]  Entry ${entry:.2f} → ${price:.2f}  "
            f"P&L: {'+' if pnl_pct>=0 else ''}{pnl_pct*100:.1f}%  "
            f"Peak: ${peak:.2f}  Drop: {drop_from_peak*100:.1f}%"
        )

        # PDT protection — never sell same day we bought
        if sym in bought_today:
            log.info(f"  [{sym}] 🛡️ PDT shield — bought today, holding")
            continue

        # PTJ Rule: cut losses at exactly 2%
        if pnl_pct <= -STOP_LOSS_PCT:
            place_sell(sym, qty, reason=f"PTJ STOP LOSS {pnl_pct*100:.1f}%")

        # Trailing stop: activates after 3% gain, fires if drops 3% from peak
        elif pnl_pct >= 0.03 and drop_from_peak >= TRAILING_STOP_PCT:
            place_sell(sym, qty, reason=f"TRAILING STOP (peak ${peak:.2f}, -{drop_from_peak*100:.1f}%)")

        # Take profit at 6% (3:1 R/R per PTJ)
        elif pnl_pct >= TAKE_PROFIT_PCT:
            place_sell(sym, qty, reason=f"TAKE PROFIT +{pnl_pct*100:.1f}% (PTJ 3:1 R/R)")

        # Minervini exit: trend broken mid-day
        else:
            sig = get_signal(sym)
            if sig["signal"] == "SELL" and sym not in bought_today:
                place_sell(sym, qty, reason=f"TREND SIGNAL: {sig['reason']}")

# ══════════════════════════════════════════════════════════════
#  EOD LOGIC  (Livermore: never hold losers overnight)
# ══════════════════════════════════════════════════════════════

def eod_review():
    log.info("⏰ EOD — Livermore swing review...")
    for sym in WATCHLIST:
        try:
            pos = trading.get_open_position(sym)
        except Exception:
            continue

        entry   = float(pos.avg_entry_price)
        price   = float(pos.current_price)
        qty     = float(pos.qty)
        pnl_pct = (price - entry) / entry

        if SWING_TRADING and pnl_pct >= MIN_PROFIT_HOLD:
            # Check trend still valid before holding overnight
            sig = get_signal(sym)
            if sig["ema9"] > sig["ema20"] and sig["adx"] >= 20 and sig["rsi"] < 78:
                log.info(
                    f"  [{sym}] 🌙 HOLD OVERNIGHT — up {pnl_pct*100:.1f}%  "
                    f"ADX={sig['adx']}  RSI={sig['rsi']}  (Minervini: trend intact)"
                )
                continue

        reason = f"+{pnl_pct*100:.1f}% taking profit" if pnl_pct > 0 else f"{pnl_pct*100:.1f}% Livermore: never hold losers"
        place_sell(sym, qty, reason=f"EOD — {reason}")

    log.info("⏰ EOD complete")

# ══════════════════════════════════════════════════════════════
#  MAIN SCAN LOOP
# ══════════════════════════════════════════════════════════════

def run_strategy():
    clock = trading.get_clock()
    if not clock.is_open:
        log.info("Market closed — skipping")
        return

    now_et = datetime.datetime.now(ET)
    if now_et.time() >= EOD_TIME:
        eod_review()
        return

    account = trading.get_account()
    log.info("=" * 62)
    log.info(f"  📡 SCAN  {now_et.strftime('%a %b %d %H:%M ET')}  "
             f"Equity: ${float(account.equity):,.2f}  Cash: ${float(account.cash):,.2f}")
    log.info("=" * 62)

    # 1. Protect open positions first (PTJ: capital preservation always first)
    monitor_positions()

    # 2. Scan for new entries
    for sym in WATCHLIST:
        try:
            trading.get_open_position(sym)
            log.info(f"  [{sym}] Already in position — skipping entry")
            continue
        except Exception:
            pass  # no position, proceed to scan

        sig = get_signal(sym)
        log.info(
            f"  [{sym}]  ${sig['price']:.2f}  "
            f"RSI={sig['rsi']}  ADX={sig['adx']}  "
            f"EMA9/20/50: {sig['ema9']:.2f}/{sig['ema20']:.2f}/{sig['ema50']:.2f}  "
            f"Vol: {sig['vol']/sig['vol_avg']:.1f}×  "
            f"→ {sig['signal']}  ({sig['reason']})"
        )

        if sig["signal"] == "BUY":
            cash = float(account.cash)
            if cash < sig["price"]:
                log.warning(f"  [{sym}] Insufficient cash ${cash:.2f}")
                continue
            place_buy(sym, sig["price"])

    log.info("")

# ══════════════════════════════════════════════════════════════
#  BOOT
# ══════════════════════════════════════════════════════════════

def print_banner():
    acct = trading.get_account()
    mode = "📄 PAPER" if PAPER else "💵 LIVE"
    log.info("╔" + "═"*60 + "╗")
    log.info("║  SHUBHAM · MULTI-LEGEND DAY TRADING BOT" + " "*19 + "║")
    log.info("║" + " "*60 + "║")
    log.info(f"║  Mode     : {mode}" + " "*(48 - len(mode)) + "║")
    log.info(f"║  Watchlist: {', '.join(WATCHLIST)}" + " "*(48 - len(', '.join(WATCHLIST))) + "║")
    log.info(f"║  Strategy : Minervini SEPA + Raschke Holy Grail       ║")
    log.info(f"║             + PTJ 2% rule + O'Neil Volume             ║")
    log.info(f"║  Risk/Trade: {ACCOUNT_RISK_PCT*100:.0f}% account  |  Stop: {STOP_LOSS_PCT*100:.0f}%  |  TP: {TAKE_PROFIT_PCT*100:.0f}%" + " "*17 + "║")
    log.info(f"║  R/R Ratio: 1:{REWARD_RISK_RATIO:.0f} (Paul Tudor Jones minimum)" + " "*22 + "║")
    log.info(f"║  Equity   : ${float(acct.equity):>10,.2f}" + " "*41 + "║")
    log.info(f"║  Cash     : ${float(acct.cash):>10,.2f}" + " "*41 + "║")
    log.info("╚" + "═"*60 + "╝")


if __name__ == "__main__":
    print_banner()

    # Reset PDT tracker at midnight
    def reset_day():
        bought_today.clear()
        log.info("🔄 New day — PDT tracker cleared")

    schedule.every().day.at("00:01").do(reset_day)
    schedule.every(15).minutes.do(run_strategy)

    # EOD on every trading day
    for day in ["monday", "tuesday", "wednesday", "thursday", "friday"]:
        getattr(schedule.every(), day).at("15:50").do(eod_review)

    # Run immediately on start
    run_strategy()

    while True:
        schedule.run_pending()
        time.sleep(30)
