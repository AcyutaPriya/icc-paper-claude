# ICC Trading Bot (Alpaca + Financial Modeling Prep + OpenAI + GitHub Actions)

Automates the Indication → Correction → Continuation method on a stock watchlist.
**Paper trading by default.** It will refuse to run live unless you add an explicit confirmation secret.

## How it works

| Stage | What happens |
|---|---|
| Day filter | Skips holidays and early closes (Alpaca's market calendar), high-impact US macro days that you list in `blackout_dates` (FOMC, CPI, jobs report), and any symbol with earnings yesterday/today/tomorrow (FMP **free-plan** earnings calendar plus your own `manual_earnings_dates`). If the session check or earnings lookup fails, the bot does nothing (fails closed). |
| Signal (`icc.py`) | **Indication:** a 1H candle *closes* beyond a swing high/low (impulse must be >= 1.5 ATR). **Correction:** pullback of 30-85% of the leg without closing through the key zone. **Continuation:** 15m price touches the zone twice (the "2nd disrespect"), then closes beyond the post-touch high. |
| AI review (OpenAI) | Receives the setup, recent bars and Alpaca news headlines. It can only approve or veto, never change prices. API failure = no trade. |
| Execution (Alpaca) | Bracket order: market entry + stop (below correction low) + target (indication peak). Size = 0.5% equity risk, capped at 10% of equity per position. |

Guards: max 3 open positions, max 3 trades/day, no entries in the first 15 min or last 45 min, one order per symbol per day per side (duplicate-proof order IDs), minimum 1.5 R:R.

## Setup (about 15 minutes)

1. **Create accounts / keys**
   - Alpaca: sign up, switch to **Paper Trading**, generate API key + secret.
   - Financial Modeling Prep (FMP): API key from financialmodelingprep.com.
   - OpenAI: API key from platform.openai.com (add a few dollars of credit).
2. **Create a private GitHub repo** and upload everything in this folder, keeping the structure
   (`.github/workflows/icc-bot.yml` must stay at that path).
3. **Add secrets:** repo → Settings → Secrets and variables → Actions → *New repository secret*:
   `ALPACA_API_KEY`, `ALPACA_SECRET_KEY`, `FMP_API_KEY`, `OPENAI_API_KEY`.
4. *(Optional)* Variables tab: `OPENAI_MODEL` (default `gpt-4o-mini`; set to any chat model your account has).
5. **Test:** Actions tab → *ICC Bot* → *Run workflow* with **dry_run = true**. Open the run log: you should see
   `mode=PAPER dry_run=True` and either "Market closed", a skip reason, or setups. Download the `decisions-*` artifact to see every decision.
6. **Go live on paper:** the schedule (every 15 min, weekdays) is already active once the workflow is on your default branch.
   To place paper orders manually, run the workflow with dry_run = false.

## Tuning (`config.json`)

- `watchlist`: liquid symbols only. `strategy.min_touches: 1` gives more signals (looser "2nd touch" rule).
- `strategy.allow_shorts`: off by default (shorting needs a margin-enabled account).
- `filters.blackout_dates`: dates to sit out (FOMC decisions, CPI, jobs reports). The free FMP plan doesn't clearly include an economic calendar, so this list is the default. Set `filters.use_fmp_economic_calendar` to `true` only if your plan includes that endpoint.
- `filters.manual_earnings_dates`: e.g. `{"BAC": ["2027-01-14"]}`. FMP's free earnings calendar only covers a limited list of symbols, so add dates here for any stock it misses.
  I pre-filled **2026-12-09** (FOMC decision day); verify at federalreserve.gov and add CPI/jobs-report dates.
- `risk.*`: risk per trade, position cap, daily trade limit.

## Before you trust it

- **Rules are my interpretation** of Trades By Sci's ICC from public notes and the study guide, not his exact system. It is unit-tested on synthetic price paths only, **not backtested**. Paper trade for at least a month (30+ signals) and review `decisions.jsonl` before deciding anything further.
- FMP free plan: 250 calls/day. The bot makes one FMP call per run (the earnings calendar), roughly 30 a day. If that call fails, the bot fails closed. Headlines come from Alpaca's news feed and are best-effort, never blocking a trade.
- Free Alpaca data is IEX-only (thinner volume than the full tape), so 1H/15m candles can differ slightly from TradingView.
- GitHub Actions cron can run late or skip runs, so entries can lag a few minutes.
- Day trades in a margin account under $25k are subject to the pattern day trader rule (paper accounts simulate it).
- Going live later: set variable `ALPACA_PAPER=false`, use live keys, and add secret `LIVE_CONFIRM=I_UNDERSTAND_THE_RISKS`. Start tiny.
- I'm not a financial advisor; this is educational code, and automated trading can lose money quickly.

## Files

```
bot.py                         orchestration: day filter (Alpaca + FMP), Alpaca data/orders, OpenAI gate
icc.py                         pure-pandas ICC detector (no network)
test_icc.py                    offline tests (run automatically before each bot run)
config.json                    watchlist, strategy, risk, filters
requirements.txt
.github/workflows/icc-bot.yml  schedule + secrets wiring
```
