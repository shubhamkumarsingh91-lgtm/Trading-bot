[README_SETUP.md](https://github.com/user-attachments/files/28449015/README_SETUP.md)

# Trading Bot Setup Guide

## What this bot does
- Scans SNOW, GTLB, PANW, DG every 15 minutes during market hours
- Buys when RSI is oversold + moving average crossover confirms uptrend
- Sells automatically at +4% profit or -2% stop loss
- Closes ALL positions at 3:55 PM ET (end of day)
- Currently set to PAPER TRADING (fake money) — safe to test!

---

## Step 1 — Deploy to Render (free cloud hosting)

1. Create a free account at https://render.com
2. Click **"New"** → **"Web Service"**
3. Connect your GitHub (upload the bot files first to a private repo)
4. Set **Build Command**: `pip install -r requirements.txt`
5. Set **Start Command**: `python trading_bot.py`
6. Deploy!

---

## Step 2 — Run locally (simpler option)

```bash
# Install dependencies
pip install alpaca-py pandas numpy schedule requests

# Run the bot
python trading_bot.py
```

Keep your computer on during market hours (9:30 AM – 4:00 PM ET).

---

## Step 3 — Switch to live trading (when ready)

In `trading_bot.py`, change line:
```python
PAPER = True   # change to False
```

Then replace the API keys with your LIVE Alpaca keys (different from paper keys).

---

## Strategy Settings (easy to adjust)

| Setting | Default | What it means |
|---------|---------|---------------|
| MAX_POSITION_USD | $400 | Max spend per trade |
| RISK_PER_TRADE_PCT | 2% | Stop loss level |
| PROFIT_TARGET_PCT | 4% | Take profit level |
| RSI_OVERSOLD | 35 | Buy signal trigger |
| RSI_OVERBOUGHT | 65 | Sell signal trigger |

---

## Important warnings
- This is algorithmic trading — losses are possible
- Always test on paper trading before going live
- Never risk money you can't afford to lose
- This is not financial advice
