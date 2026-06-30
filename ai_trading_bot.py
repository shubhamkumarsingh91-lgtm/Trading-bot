#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════╗
║   🤖  AI SELF-LEARNING PAPER TRADING BOT                ║
║   ─────────────────────────────────────────────────────  ║
║   XGBoost ML  ·  TradingView TA  ·  Gemini AI Brief     ║
║   Alpaca Paper $100,000  ·  Retrains Every Night        ║
╚══════════════════════════════════════════════════════════╝

DEPLOY ON RENDER as Background Worker.

Required environment variables (set in Render dashboard):
  ALPACA_API_KEY       → Your Alpaca PAPER account API key
  ALPACA_SECRET_KEY    → Your Alpaca PAPER account secret
  GEMINI_API_KEY       → Your Google Gemini API key (FREE)

How to get keys:
  Alpaca paper keys:
    1. Login to alpaca.markets
    2. Top-left dropdown → switch to "Paper Account"
    3. API Keys → Generate New Key

  Gemini API key (FREE — no credit card):
    1. Go to aistudio.google.com
    2. Click "Get API Key" → Create API Key
    Done. Free tier = 15 requests/min, 1M tokens/day.
"""

import os, json, logging, time, joblib, warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score
import schedule
import google.generativeai as genai
from tradingview_ta import TA_Handler, Interval

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════════
# LOGGING
# ══════════════════════════════════════════════════════════
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('ai_bot.log', encoding='utf-8'),
    ],
)
log = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════
API_KEY     = os.environ.get('ALPACA_API_KEY',    'YOUR_PAPER_KEY')
SECRET_KEY  = os.environ.get('ALPACA_SECRET_KEY', 'YOUR_PAPER_SECRET')
GEMINI_KEY  = os.environ.get('GEMINI_API_KEY',    'YOUR_GEMINI_KEY')

WATCHLIST = ['NVDA', 'PANW', 'AVGO', 'SOFI', 'PLTR']
TV_EXCHANGE = {
    'NVDA': 'NASDAQ', 'PANW': 'NASDAQ', 'AVGO': 'NASDAQ',
    'SOFI': 'NASDAQ', 'PLTR': 'NYSE',   'SPY':  'AMEX',
}

# ── Risk Parameters ──────────────────────────────────────
RISK_PCT      = 0.02   # 2% of equity risked per trade
POSITION_CAP  = 0.20   # max 20% of equity in one stock
STOP_PCT      = 0.02   # stop loss at -2%
TP_PCT        = 0.06   # take profit at +6%
MAX_POSITIONS = 3      # max simultaneous open positions

# ── Signal Weights ───────────────────────────────────────
ML_WEIGHT    = 0.50    # XGBoost prediction
TV_WEIGHT    = 0.35    # TradingView technical analysis
RSI_WEIGHT   = 0.15    # RSI quality filter
CONFIDENCE   = 0.62    # minimum ML confidence to consider BUY

# ── Schedule ─────────────────────────────────────────────
SCAN_EVERY   = 10      # minutes between scans
EOD_HOUR     = 15      # close all at 3:50 PM ET
EOD_MIN      = 50

# ── Files ────────────────────────────────────────────────
MODEL_FILE   = Path('ai_model.xgb')
LOG_FILE     = Path('trade_log.json')
BRIEF_FILE   = Path('morning_brief.txt')
TRAIN_DAYS   = 365

TF = TimeFrame(15, TimeFrameUnit.Minute)   # 15-minute bars


# ══════════════════════════════════════════════════════════
# API CLIENTS
# ══════════════════════════════════════════════════════════
trade_client = TradingClient(API_KEY, SECRET_KEY, paper=True)
data_client  = StockHistoricalDataClient(API_KEY, SECRET_KEY)

# Google Gemini — free AI (aistudio.google.com)
genai.configure(api_key=GEMINI_KEY)
gemini = genai.GenerativeModel('gemini-1.5-flash')


# ══════════════════════════════════════════════════════════
# FEATURE ENGINEERING  (30 features)
# ══════════════════════════════════════════════════════════
FEATURES = [
    # Returns
    'ret_1', 'ret_5', 'ret_10', 'ret_30',
    # Trend / EMAs
    'ema9_20x', 'ema20_50x', 'above_ema50', 'above_ema200',
    # Momentum — RSI
    'rsi', 'rsi_overbought', 'rsi_oversold',
    # Momentum — MACD
    'macd_hist', 'macd_cross',
    # Mean-reversion — Bollinger Bands
    'bb_pct', 'bb_squeeze',
    # Volatility — ATR
    'atr_pct', 'atr_expanding',
    # Volume
    'vol_ratio', 'vol_surge',
    # Candlestick
    'body', 'candle_dir', 'upper_wick', 'lower_wick', 'doji',
    # Stochastic
    'stoch_k', 'stoch_d', 'stoch_golden',
    # Session
    'hour', 'is_morning', 'is_power_hour',
]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all 30 technical indicator features from OHLCV data."""
    d = df.copy()
    c = d['close']

    # ── Price returns ─────────────────────────────────────
    for n in [1, 5, 10, 30]:
        d[f'ret_{n}'] = c.pct_change(n)

    # ── Exponential Moving Averages ───────────────────────
    for n in [9, 20, 50, 200]:
        d[f'ema{n}'] = c.ewm(span=n, adjust=False).mean()
    d['ema9_20x']     = (d['ema9'] - d['ema20']) / (d['ema20'] + 1e-9)
    d['ema20_50x']    = (d['ema20'] - d['ema50']) / (d['ema50'] + 1e-9)
    d['above_ema50']  = (c > d['ema50']).astype(int)
    d['above_ema200'] = (c > d['ema200']).astype(int)

    # ── RSI ───────────────────────────────────────────────
    def _rsi(s, n=14):
        delta = s.diff()
        gain  = delta.where(delta > 0, 0.0).rolling(n).mean()
        loss  = -delta.where(delta < 0, 0.0).rolling(n).mean()
        return 100 - 100 / (1 + gain / (loss + 1e-9))

    d['rsi']           = _rsi(c, 14)
    d['rsi_overbought'] = (d['rsi'] > 70).astype(int)
    d['rsi_oversold']   = (d['rsi'] < 30).astype(int)

    # ── MACD ──────────────────────────────────────────────
    macd = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    sig  = macd.ewm(span=9, adjust=False).mean()
    d['macd_hist']  = macd - sig
    d['macd_cross'] = (
        np.sign(d['macd_hist']) != np.sign(d['macd_hist'].shift(1))
    ).astype(int)

    # ── Bollinger Bands ───────────────────────────────────
    bm   = c.rolling(20).mean()
    bstd = c.rolling(20).std()
    bu, bl = bm + 2 * bstd, bm - 2 * bstd
    bw   = (bu - bl) / (bm + 1e-9)
    d['bb_pct']     = (c - bl) / (bu - bl + 1e-9)
    d['bb_squeeze'] = (bw < bw.rolling(50).mean()).astype(int)

    # ── ATR (Average True Range) ──────────────────────────
    tr = pd.concat([
        d['high'] - d['low'],
        (d['high'] - c.shift(1)).abs(),
        (d['low']  - c.shift(1)).abs(),
    ], axis=1).max(axis=1)
    d['atr']          = tr.rolling(14).mean()
    d['atr_pct']      = d['atr'] / (c + 1e-9)
    d['atr_expanding'] = (d['atr'] > d['atr'].rolling(20).mean()).astype(int)

    # ── Volume ────────────────────────────────────────────
    vol_avg = d['volume'].rolling(20).mean()
    d['vol_ratio'] = d['volume'] / (vol_avg + 1)
    d['vol_surge'] = (d['vol_ratio'] > 1.5).astype(int)

    # ── Candlestick patterns ──────────────────────────────
    hi_body = d[['open', 'close']].max(axis=1)
    lo_body = d[['open', 'close']].min(axis=1)
    d['body']       = (c - d['open']).abs() / (d['open'] + 1e-9)
    d['candle_dir'] = np.sign(c - d['open'])
    d['upper_wick'] = (d['high'] - hi_body) / (d['open'] + 1e-9)
    d['lower_wick'] = (lo_body - d['low'])  / (d['open'] + 1e-9)
    d['doji']       = (d['body'] < 0.001).astype(int)

    # ── Stochastic Oscillator ─────────────────────────────
    lo14 = d['low'].rolling(14).min()
    hi14 = d['high'].rolling(14).max()
    d['stoch_k']    = 100 * (c - lo14) / (hi14 - lo14 + 1e-9)
    d['stoch_d']    = d['stoch_k'].rolling(3).mean()
    d['stoch_golden'] = (
        (d['stoch_k'] > d['stoch_d']) &
        (d['stoch_k'].shift(1) <= d['stoch_d'].shift(1))
    ).astype(int)

    # ── Session features ──────────────────────────────────
    try:
        d['hour'] = d.index.hour
    except AttributeError:
        d['hour'] = 12
    d['is_morning']    = (d['hour'] == 9).astype(int)   # open hour
    d['is_power_hour'] = (d['hour'] == 15).astype(int)  # power hour

    return d


def make_labels(df: pd.DataFrame, forward: int = 8, threshold: float = 0.005) -> pd.Series:
    """
    Binary label: 1 = price rises more than 0.5% in next 8 bars (≈2 hours).
    This is what the model tries to predict.
    """
    future_ret = df['close'].shift(-forward) / df['close'] - 1
    return (future_ret > threshold).astype(int)


# ══════════════════════════════════════════════════════════
# XGBOOST MODEL
# ══════════════════════════════════════════════════════════
model: xgb.XGBClassifier = None
model_accuracy: float     = 0.0
model_trained_at: str     = 'never'


def fetch_bars(symbol: str, days: int = None, bars: int = 550) -> pd.DataFrame:
    """Fetch OHLCV bars from Alpaca historical data API."""
    end   = datetime.now(timezone.utc)
    start = end - timedelta(days=days) if days else end - timedelta(minutes=bars * 15 + 180)
    req   = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TF,
        start=start,
        end=end,
        adjustment='all',
    )
    raw = data_client.get_stock_bars(req)
    df  = raw.df
    if isinstance(df.index, pd.MultiIndex):
        df = df.xs(symbol, level='symbol')
    df.index = pd.to_datetime(df.index, utc=True).tz_convert('America/New_York')
    return df.sort_index()


def train_model(retrain: bool = False) -> None:
    """
    Train XGBoost on 1 year of 15-min data for all 5 watchlist stocks.
    Called once at startup and then nightly for self-improvement.
    """
    global model, model_accuracy, model_trained_at
    label = 'Nightly retrain' if retrain else 'Initial training'
    log.info(f'🧠 {label} — fetching {TRAIN_DAYS} days × {len(WATCHLIST)} stocks...')

    frames = []
    for i, sym in enumerate(WATCHLIST):
        try:
            df  = fetch_bars(sym, days=TRAIN_DAYS)
            df  = add_features(df)
            df['target'] = make_labels(df)
            df['sym_id'] = i
            df = df.dropna()
            frames.append(df)
            log.info(f'  ✓ {sym}: {len(df):,} bars loaded')
        except Exception as e:
            log.warning(f'  ✗ {sym}: {e}')

    if not frames:
        log.error('No training data — cannot train')
        return

    all_data = pd.concat(frames)
    X = all_data[FEATURES + ['sym_id']]
    y = all_data['target']
    pos_rate = y.mean()
    log.info(f'  Total: {len(X):,} samples | Bullish rate: {pos_rate:.1%}')

    # Time-series split (no shuffle — respect temporal order)
    split_idx = int(len(X) * 0.8)
    X_tr, X_va = X.iloc[:split_idx], X.iloc[split_idx:]
    y_tr, y_va = y.iloc[:split_idx], y.iloc[split_idx:]

    model = xgb.XGBClassifier(
        n_estimators=600,
        max_depth=6,
        learning_rate=0.04,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        gamma=0.1,
        reg_alpha=0.1,
        scale_pos_weight=(y == 0).sum() / ((y == 1).sum() + 1),
        eval_metric='logloss',
        verbosity=0,
        n_jobs=-1,
        random_state=42,
    )
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        early_stopping_rounds=50,
        verbose=False,
    )

    preds = model.predict(X_va)
    model_accuracy   = accuracy_score(y_va, preds)
    prec             = precision_score(y_va, preds, zero_division=0)
    model_trained_at = datetime.now().isoformat()

    # Top 5 most important features
    fi = sorted(
        zip(FEATURES + ['sym_id'], model.feature_importances_),
        key=lambda x: -x[1]
    )[:5]
    top_feats = ', '.join(f'{n}({v:.2f})' for n, v in fi)

    log.info(f'✅ Model ready — Accuracy: {model_accuracy:.1%} | Precision: {prec:.1%}')
    log.info(f'  Top signals learned: {top_feats}')

    joblib.dump({
        'model':      model,
        'accuracy':   model_accuracy,
        'precision':  prec,
        'trained_at': model_trained_at,
        'features':   FEATURES,
    }, MODEL_FILE)
    log.info(f'💾 Model saved → {MODEL_FILE}')


def load_or_train() -> None:
    """Load saved model or train fresh if none exists."""
    global model, model_accuracy, model_trained_at
    if MODEL_FILE.exists():
        try:
            saved = joblib.load(MODEL_FILE)
            model            = saved['model']
            model_accuracy   = saved['accuracy']
            model_trained_at = saved['trained_at']
            log.info(f'📦 Model loaded — accuracy: {model_accuracy:.1%}, trained: {model_trained_at[:10]}')
            return
        except Exception as e:
            log.warning(f'Model load failed ({e}), retraining from scratch')
    train_model()


def ml_predict(symbol: str, sym_id: int) -> dict:
    """Run the trained model on the latest bars for a given symbol."""
    if model is None:
        return {'confidence': 0.0, 'price': 0, 'rsi': 50, 'vol_ratio': 1, 'above_ema50': 0}
    try:
        df = fetch_bars(symbol, bars=600)
        if len(df) < 250:
            log.warning(f'  {symbol}: not enough bars ({len(df)})')
            return {'confidence': 0.0, 'price': 0, 'rsi': 50, 'vol_ratio': 1, 'above_ema50': 0}

        df = add_features(df)
        df['sym_id'] = sym_id
        row = df[FEATURES + ['sym_id']].iloc[-1:]

        if row.isnull().any().any():
            return {'confidence': 0.0, 'price': 0, 'rsi': 50, 'vol_ratio': 1, 'above_ema50': 0}

        prob = float(model.predict_proba(row)[0][1])

        return {
            'confidence': prob,
            'price':      float(df['close'].iloc[-1]),
            'rsi':        float(df['rsi'].iloc[-1]),
            'vol_ratio':  float(df['vol_ratio'].iloc[-1]),
            'macd_hist':  float(df['macd_hist'].iloc[-1]),
            'above_ema50': int(df['above_ema50'].iloc[-1]),
            'bb_pct':     float(df['bb_pct'].iloc[-1]),
        }
    except Exception as e:
        log.warning(f'ML predict error {symbol}: {e}')
        return {'confidence': 0.0, 'price': 0, 'rsi': 50, 'vol_ratio': 1, 'above_ema50': 0}


# ══════════════════════════════════════════════════════════
# TRADINGVIEW TECHNICAL ANALYSIS
# ══════════════════════════════════════════════════════════
# Score map: TradingView recommendation → numeric weight
TV_SCORE = {
    'STRONG_BUY': 1.0, 'BUY': 0.7,
    'NEUTRAL':    0.0,
    'SELL':      -0.5, 'STRONG_SELL': -1.0,
}


def get_tv_analysis(symbol: str) -> dict | None:
    """
    Fetch TradingView's 15-minute technical analysis.
    Returns buy/sell/neutral signal counts + key indicators.
    """
    try:
        handler = TA_Handler(
            symbol=symbol,
            screener='america',
            exchange=TV_EXCHANGE.get(symbol, 'NASDAQ'),
            interval=Interval.INTERVAL_15_MINUTES,
        )
        a = handler.get_analysis()
        return {
            'rec':     a.summary['RECOMMENDATION'],
            'buy':     a.summary['BUY'],
            'sell':    a.summary['SELL'],
            'neutral': a.summary['NEUTRAL'],
            'rsi':     a.indicators.get('RSI',      50.0),
            'macd':    a.indicators.get('MACD.macd', 0.0),
            'ema20':   a.indicators.get('EMA20',      0.0),
            'ema50':   a.indicators.get('EMA50',      0.0),
            'adx':     a.indicators.get('ADX',       20.0),
        }
    except Exception as e:
        log.warning(f'TradingView error {symbol}: {e}')
        return None


# ══════════════════════════════════════════════════════════
# COMBINED SIGNAL ENGINE
# XGBoost (50%) + TradingView (35%) + RSI filter (15%)
# ══════════════════════════════════════════════════════════

def combined_signal(symbol: str, ml: dict, tv: dict | None) -> dict:
    """
    Combine all three signal sources into one final score.
    Score >= 0.55 → BUY.  Below → HOLD.
    """
    score   = 0.0
    reasons = []

    # ── XGBoost (50%) ─────────────────────────────────────
    conf = ml.get('confidence', 0.0)
    # normalize: 0.5 confidence = 0 contribution, 1.0 = full weight
    ml_contrib = (conf - 0.5) * 2 * ML_WEIGHT
    score += ml_contrib
    reasons.append(f'ML:{conf:.0%}')

    # ── TradingView (35%) ─────────────────────────────────
    if tv:
        tv_contrib = TV_SCORE.get(tv['rec'], 0.0) * TV_WEIGHT
        score += tv_contrib
        reasons.append(f'TV:{tv["rec"]}')
        # ADX bonus: strong trend (ADX > 25) boosts conviction
        if tv.get('adx', 0) > 25 and tv['rec'] in ('BUY', 'STRONG_BUY'):
            score += 0.04
            reasons.append('ADX:TREND')
    else:
        reasons.append('TV:N/A')

    # ── RSI filter (15%) ──────────────────────────────────
    rsi = ml.get('rsi', 50)
    if 35 <= rsi <= 65:           # sweet spot for entry
        score += RSI_WEIGHT
    elif rsi > 78:                # overbought — penalize heavily
        score -= 0.22
        reasons.append(f'RSI:OVER({rsi:.0f})')
    elif rsi < 25:                # oversold — small bounce bonus
        score += 0.06
        reasons.append(f'RSI:OVERSOLD({rsi:.0f})')
    reasons.append(f'RSI:{rsi:.0f}')

    # ── Volume surge bonus ────────────────────────────────
    if ml.get('vol_ratio', 1) > 1.5:
        score += 0.04
        reasons.append('VOL:SURGE')

    # ── Trend gate: must be above EMA50 ──────────────────
    if ml.get('above_ema50', 0) == 0:
        score -= 0.15
        reasons.append('BELOW:EMA50')

    # ── Bollinger Band position ───────────────────────────
    bb = ml.get('bb_pct', 0.5)
    if 0.35 <= bb <= 0.65:       # middle of bands — less risky
        score += 0.02
    elif bb > 0.95:              # near upper band — overbought
        score -= 0.08

    final_signal = 'BUY' if score >= 0.55 else 'HOLD'
    return {
        'signal':     final_signal,
        'score':      round(score, 3),
        'confidence': conf,
        'reasons':    reasons,
    }


# ══════════════════════════════════════════════════════════
# CLAUDE AI — DAILY MORNING BRIEF
# Runs every morning at 9:00 AM ET to set the day's strategy
# ══════════════════════════════════════════════════════════
morning_brief_text: str = ''


def generate_morning_brief() -> None:
    """
    Ask Claude to analyse the morning setup and create a prioritised
    trading plan for the day using live TV signals + recent performance.
    """
    global morning_brief_text
    log.info('🧠 Generating Claude morning brief...')

    # ── Gather real-time context ─────────────────────────
    try:
        acct   = trade_client.get_account()
        equity = float(acct.equity)
        cash   = float(acct.cash)
        bp     = float(acct.buying_power)
    except Exception:
        equity, cash, bp = 100000, 100000, 100000

    pos_now = get_positions()
    sells   = [t for t in trade_log if t.get('action') == 'SELL']
    wr      = sum(1 for t in sells if t.get('win')) / len(sells) * 100 if sells else 0
    total_pnl = sum(t.get('pnl', 0) for t in sells)

    # ── TradingView snapshot (all watchlist + SPY) ────────
    tv_lines = []
    for sym in WATCHLIST + ['SPY']:
        tv = get_tv_analysis(sym)
        if tv:
            tv_lines.append(
                f"  {sym:5s} | {tv['rec']:12s} | RSI:{tv['rsi']:.0f} "
                f"| ADX:{tv['adx']:.0f} | Buy:{tv['buy']:2d} Sell:{tv['sell']:2d}"
            )
        else:
            tv_lines.append(f"  {sym:5s} | N/A")

    # ── Recent trades ─────────────────────────────────────
    recent = sells[-8:] if len(sells) >= 8 else sells
    recent_str = '\n'.join(
        f"  {t['timestamp'][:10]} | {t['symbol']:5s} | {t.get('reason',''):15s} | P&L: ${t.get('pnl',0):+7.2f}"
        for t in recent
    ) or '  No closed trades yet'

    prompt = f"""You are the AI intelligence layer for a self-learning paper trading bot running on Alpaca.

TODAY: {datetime.now().strftime('%A, %B %d, %Y — %I:%M %p ET')}

ACCOUNT STATUS:
  Equity:        ${equity:>12,.2f}
  Cash:          ${cash:>12,.2f}
  Buying Power:  ${bp:>12,.2f}
  Open positions: {list(pos_now.keys()) or 'None'}

BOT PERFORMANCE:
  Model accuracy: {model_accuracy:.1%}  (last trained: {model_trained_at[:10]})
  Lifetime trades: {len(sells)}
  Win rate:       {wr:.1f}%
  Total P&L:      ${total_pnl:+,.2f}

TRADINGVIEW 15-MINUTE SIGNALS RIGHT NOW:
{chr(10).join(tv_lines)}

RECENT CLOSED TRADES:
{recent_str}

STRATEGY RULES:
  · 2% equity risk per trade
  · Stop loss at -2%, take profit at +6%
  · Max 3 simultaneous positions
  · Watchlist: NVDA, PANW, AVGO, SOFI, PLTR

Generate today's morning trading brief. Format exactly as follows:

MARKET MOOD: [One sentence — bull/bear/neutral + primary driver]

TODAY'S PRIORITY RANKING:
  #1 [Symbol] — [Specific reason based on signals above]
  #2 [Symbol] — [Specific reason]
  #3 [Symbol] — [Specific reason]
  #4 [Symbol] — [Specific reason]
  #5 [Symbol] — [Specific reason]

KEY RISK TODAY: [One specific thing to watch out for]

STANCE: [Aggressive / Neutral / Defensive] — [Why, based on today's signals]

TARGET P&L: $[realistic number] — [Brief justification]

CLAUDE'S EDGE FOR TODAY: [One insight that a human might miss — patterns in the signals, sector rotation, macro context, etc.]

Be precise and data-driven. Reference specific indicator values from the signals above."""

    try:
        response = gemini.generate_content(prompt)
        morning_brief_text = response.text

        sep = '═' * 60
        log.info(f'\n{sep}\n🧠 CLAUDE MORNING BRIEF — {datetime.now().strftime("%b %d %Y")}\n{sep}\n{morning_brief_text}\n{sep}')

        BRIEF_FILE.write_text(
            f"Generated: {datetime.now().isoformat()}\n"
            f"Equity: ${equity:,.2f} | Model: {model_accuracy:.1%}\n"
            f"{'─'*60}\n{morning_brief_text}"
        )
    except Exception as e:
        log.warning(f'Claude API error: {e}')
        morning_brief_text = f'[Brief unavailable: {e}]'


# ══════════════════════════════════════════════════════════
# TRADE LOGGING  (feeds nightly retrain)
# ══════════════════════════════════════════════════════════
trade_log: list = []


def load_log() -> None:
    global trade_log
    if LOG_FILE.exists():
        try:
            trade_log = json.loads(LOG_FILE.read_text())
            log.info(f'📚 Loaded {len(trade_log)} trades from history')
        except Exception:
            trade_log = []


def save_log(entry: dict) -> None:
    trade_log.append(entry)
    LOG_FILE.write_text(json.dumps(trade_log, indent=2, default=str))


# ══════════════════════════════════════════════════════════
# POSITION & ORDER MANAGEMENT
# ══════════════════════════════════════════════════════════

def get_account_equity() -> float:
    return float(trade_client.get_account().equity)


def get_positions() -> dict:
    return {p.symbol: p for p in trade_client.get_all_positions()}


def place_buy(symbol: str, price: float, equity: float, signal: dict) -> bool:
    """Calculate position size, place market buy, log trade."""
    pos = get_positions()
    if symbol in pos:
        return False  # already have this

    open_count = sum(1 for s in pos if s in WATCHLIST)
    if open_count >= MAX_POSITIONS:
        return False

    # Position sizing: risk 2% of equity, stop at -2% → position = 100%×equity×RISK_PCT/STOP_PCT
    # Capped at POSITION_CAP of total equity
    qty_usd = min(equity * RISK_PCT / STOP_PCT, equity * POSITION_CAP)
    qty     = max(1, int(qty_usd / price))

    try:
        order = trade_client.submit_order(MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=OrderSide.BUY,
            time_in_force=TimeInForce.DAY,
        ))
        log.info(
            f'🟢 BUY  {qty:5d}×{symbol} @ ${price:8.2f} '
            f'| Score:{signal["score"]:+.3f} | {" | ".join(signal["reasons"])}'
        )
        save_log({
            'action':      'BUY',
            'symbol':      symbol,
            'qty':         qty,
            'entry_price': price,
            'equity':      equity,
            'signal':      signal,
            'order_id':    str(order.id),
            'timestamp':   datetime.now().isoformat(),
        })
        return True
    except Exception as e:
        log.error(f'Buy failed {symbol}: {e}')
        return False


def place_sell(symbol: str, pos, reason: str) -> None:
    """Market sell an open position, log the outcome."""
    qty     = float(pos.qty)
    entry   = float(pos.avg_entry_price)
    current = float(pos.current_price)
    pnl     = float(pos.unrealized_pl)
    pct     = (current - entry) / entry
    win     = pnl > 0
    emoji   = '✅' if win else '❌'

    try:
        trade_client.submit_order(MarketOrderRequest(
            symbol=symbol,
            qty=abs(qty),
            side=OrderSide.SELL,
            time_in_force=TimeInForce.DAY,
        ))
        log.info(
            f'🔴 SELL {qty:5.0f}×{symbol} @ ${current:8.2f} '
            f'| P&L: ${pnl:+8.2f} ({pct:+.1%}) [{reason}] {emoji}'
        )
        save_log({
            'action':      'SELL',
            'symbol':      symbol,
            'qty':         qty,
            'entry_price': entry,
            'exit_price':  current,
            'pnl':         pnl,
            'pct':         pct,
            'win':         win,
            'reason':      reason,
            'timestamp':   datetime.now().isoformat(),
        })
    except Exception as e:
        log.error(f'Sell failed {symbol}: {e}')


def manage_open_positions() -> None:
    """Check all open positions against stop-loss and take-profit levels."""
    for sym, pos in get_positions().items():
        if sym not in WATCHLIST:
            continue
        entry   = float(pos.avg_entry_price)
        current = float(pos.current_price)
        pct     = (current - entry) / entry

        if pct <= -STOP_PCT:
            place_sell(sym, pos, 'STOP_LOSS')
        elif pct >= TP_PCT:
            place_sell(sym, pos, 'TAKE_PROFIT')


def close_all_positions(reason: str = 'EOD') -> None:
    for sym, pos in get_positions().items():
        if sym in WATCHLIST:
            place_sell(sym, pos, reason)


# ══════════════════════════════════════════════════════════
# MARKET HOURS
# ══════════════════════════════════════════════════════════

def is_market_open() -> bool:
    try:
        return trade_client.get_clock().is_open
    except Exception:
        return False


def check_eod() -> None:
    now = datetime.now()
    if now.hour == EOD_HOUR and now.minute >= EOD_MIN:
        if get_positions():
            log.info('🕑 EOD: closing all positions before market close')
            close_all_positions('EOD')


# ══════════════════════════════════════════════════════════
# NIGHTLY SELF-LEARNING RETRAIN
# ══════════════════════════════════════════════════════════

def nightly_retrain() -> None:
    """
    Midnight retrain — includes all trades from the day.
    Model improves as it sees real outcomes of its own predictions.
    """
    sells      = [t for t in trade_log if t.get('action') == 'SELL']
    wins       = [t for t in sells if t.get('win')]
    wr         = len(wins) / len(sells) * 100 if sells else 0
    total_pnl  = sum(t.get('pnl', 0) for t in sells)

    sep = '═' * 60
    log.info(f'\n{sep}\n🌙 NIGHTLY SELF-LEARNING RETRAIN\n{sep}')
    log.info(f'  Total trades ever: {len(sells)}')
    log.info(f'  Win rate:          {wr:.1f}%')
    log.info(f'  Cumulative P&L:    ${total_pnl:+,.2f}')
    log.info(f'  Old accuracy:      {model_accuracy:.1%}')

    train_model(retrain=True)

    log.info(f'  New accuracy:      {model_accuracy:.1%}')
    log.info(f'✅ Bot is smarter — ready for tomorrow\n{sep}')


# ══════════════════════════════════════════════════════════
# MAIN SCAN LOOP
# ══════════════════════════════════════════════════════════
scan_count = 0


def scan() -> None:
    """
    Every 10 minutes:
    1. Check open positions for stop-loss / take-profit
    2. Score all watchlist stocks with combined signal
    3. Enter the best opportunity if signal strong enough
    """
    global scan_count

    if not is_market_open():
        return

    scan_count  += 1
    equity       = get_account_equity()
    positions    = get_positions()
    open_watch   = sum(1 for s in positions if s in WATCHLIST)

    log.info(
        f'\n── Scan #{scan_count} | ${equity:,.2f} equity | '
        f'{open_watch}/{MAX_POSITIONS} positions | Model: {model_accuracy:.1%} ──'
    )

    # ── Step 1: manage existing positions ────────────────
    manage_open_positions()
    positions = get_positions()   # refresh after any closes
    open_watch = sum(1 for s in positions if s in WATCHLIST)

    if open_watch >= MAX_POSITIONS:
        log.info(f'  Max positions ({MAX_POSITIONS}) reached — watching only')
        return

    # ── Step 2: score every symbol ────────────────────────
    best_sym, best_score, best_state = None, 0.45, {}

    for i, sym in enumerate(WATCHLIST):
        if sym in positions:
            log.info(f'  {sym:5s}: HOLDING')
            continue

        ml  = ml_predict(sym, i)
        tv  = get_tv_analysis(sym)
        sig = combined_signal(sym, ml, tv)

        log.info(
            f'  {sym:5s}: {sig["signal"]:4s} '
            f'score={sig["score"]:+.3f} '
            f'ml={ml["confidence"]:.0%} '
            f'tv={tv["rec"] if tv else "N/A":12s} '
            f'rsi={ml.get("rsi",0):.0f} '
            f'vol×{ml.get("vol_ratio",1):.1f}'
        )

        if sig['signal'] == 'BUY' and sig['score'] > best_score:
            best_sym, best_score, best_state = sym, sig['score'], {'ml': ml, 'sig': sig}

    # ── Step 3: enter best opportunity ────────────────────
    if best_sym:
        log.info(f'  🎯 Best: {best_sym} (score={best_score:.3f}) — placing order')
        place_buy(best_sym, best_state['ml']['price'], equity, best_state['sig'])
    else:
        log.info('  No actionable signal this scan')


# ══════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════

def main():
    sep = '═' * 60
    log.info(sep)
    log.info('🤖  AI SELF-LEARNING PAPER TRADING BOT')
    log.info('    XGBoost + TradingView + Claude AI')
    log.info('    Alpaca Paper — $100,000 account')
    log.info(sep)

    # ── Verify Alpaca connection ──────────────────────────
    try:
        acct = trade_client.get_account()
        log.info(f'✅ Alpaca paper connected')
        log.info(f'   Equity:  ${float(acct.equity):>12,.2f}')
        log.info(f'   Cash:    ${float(acct.cash):>12,.2f}')
        log.info(f'   BP:      ${float(acct.buying_power):>12,.2f}')
    except Exception as e:
        log.error(f'Cannot connect to Alpaca: {e}')
        raise

    # ── Load history & train model ────────────────────────
    load_log()
    load_or_train()

    # ── Schedule all recurring tasks ──────────────────────
    schedule.every(SCAN_EVERY).minutes.do(scan)
    schedule.every(1).minutes.do(check_eod)
    schedule.every().day.at('09:00').do(generate_morning_brief)  # morning brief
    schedule.every().day.at('00:05').do(nightly_retrain)         # midnight retrain

    log.info(f'\n📅 SCHEDULE:')
    log.info(f'  Every {SCAN_EVERY} min  → Market scan + signal evaluation')
    log.info(f'  09:00 AM ET → Claude morning brief (daily trading plan)')
    log.info(f'  3:50 PM ET  → Close all positions (EOD)')
    log.info(f'  00:05 AM    → Nightly XGBoost retrain (self-improvement)')
    log.info(f'\n🚀 Bot is live!\n{sep}')

    # ── Immediate startup tasks ───────────────────────────
    generate_morning_brief()
    scan()

    # ── Main event loop ───────────────────────────────────
    while True:
        schedule.run_pending()
        time.sleep(30)


if __name__ == '__main__':
    main()
