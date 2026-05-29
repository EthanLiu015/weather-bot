# Kalshi Temperature Trading Bot

Algorithmic trading bot for Kalshi daily temperature markets. Runs two parallel strategies:

- **Strategy A (Ensemble)**: ECMWF + GEFS ensemble → NGBoost + QRF → calibrated probability → limit orders
- **Strategy B (D-0)**: Intraday ASOS/METAR polling → conditional probability as uncertainty collapses

Targets: **ORD** (Chicago O'Hare), **JFK** (New York), **LAX** (Los Angeles)

---

## Architecture

```
ingestion/ → processing/ → models/ → strategies/ → trading/ → risk/
                                          ↓
                                    scheduler/
                                          ↓
                                      api/ + frontend/
```

**Data flow:**
1. GEFS (31 members) + ECMWF → QC → downscale/bias-correct → features (~40)
2. NGBoost (Normal distribution) + QRF (quantile grid) → weighted blend → isotonic calibration
3. Strategy A: slow path on each new model run (~4×/day)
4. Strategy B: fast path every 20 min on resolution day (METAR-driven conditional prob)
5. Kelly sizing → limit orders → risk gating → Kalshi REST API
6. FastAPI + WebSocket → React dashboard

---

## Setup

### 1. Install dependencies

```bash
pip install -e ".[dev]"
```

### 2. Configure environment

```bash
cp .env.example .env
# Fill in:
# KALSHI_API_KEY, KALSHI_PRIVATE_KEY_PATH, ECMWF_API_KEY, NOAA_CDO_TOKEN
mkdir -p keys
# Place your RSA private key at ./keys/kalshi_private.pem
```

### 3. Bootstrap historical data (~2 hours first run)

```bash
export NOAA_CDO_TOKEN=your_token_here
python scripts/bootstrap_history.py
```

This downloads 5 years of hourly ASOS observations for KORD/KJFK/KLAX, builds diurnal climatology per station×month, and generates 12-cluster synoptic regime labels.

### 4. Train initial models

```bash
python scripts/initial_train.py
```

Runs walk-forward backtesting, then trains final NGBoost + QRF + residual LightGBM + isotonic calibrators on all available data.

### 5. Backtest (optional, standalone)

```bash
python scripts/run_backtest.py --start 2022-01-01 --end 2024-12-31
# Output: data/backtest_results.csv, data/backtest_report.html
```

### 6. Run the bot

```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

### 7. Run the dashboard

```bash
cd frontend
npm install
npm run dev
# Open http://localhost:5173
```

---

## Kill Switch

Three ways to halt all trading and cancel resting orders:

1. **Dashboard**: Click the red "KILL SWITCH" button, type `KILL` to confirm
2. **API**: `curl -X POST http://localhost:8000/api/controls/kill`
3. **env**: Set `BOT_ACTIVE=false` in `.env` and restart

To resume: `curl -X POST http://localhost:8000/api/controls/resume` or Dashboard button.

---

## Risk Controls

| Parameter | Default | Description |
|---|---|---|
| `MAX_DAILY_LOSS_USD` | $500 | Halt trading if daily realized PnL drops below |
| `MAX_EXPOSURE_PER_TICKER_USD` | $200 | Max dollar exposure per single ticker |
| `KELLY_FRACTION` | 0.25 | Fractional Kelly multiplier |
| `MIN_EDGE_CENTS` | 4¢ | Minimum model-vs-market edge to trade |
| `MAX_CI_WIDTH` | 0.12 | Skip trade if calibration CI too wide |

---

## API Reference

| Endpoint | Method | Description |
|---|---|---|
| `/api/markets` | GET | All active markets with fair values |
| `/api/markets/{ticker}` | GET | Single market detail + price history |
| `/api/positions` | GET | Open positions |
| `/api/positions/pnl` | GET | Cumulative PnL series |
| `/api/model/fairvalues` | GET | Current fair values per ticker |
| `/api/model/status` | GET | Model status, calibration health |
| `/api/controls/kill` | POST | Trigger kill switch |
| `/api/controls/resume` | POST | Resume trading |
| `/api/controls/retrain` | POST | Trigger manual model retrain |
| `/ws/live` | WebSocket | Live state push every 10s |

---

## Running Tests

```bash
pytest tests/ -v
```

Key test coverage:
- `test_ingestion.py` — METAR parsing, QC validation
- `test_features.py` — feature matrix shape, cyclical encodings in [-1,1]
- `test_models.py` — NGBoost sigma > 0, calibration CI covers true rate
- `test_kelly.py` — zero size when no edge, max exposure cap, lock halving
- `test_risk_controls.py` — kill switch, daily loss limit
- `test_order_manager.py` — maker-only enforcement, stale cancel logic
- `test_d0_strategy.py` — near-certain probabilities at extremes
- `test_strategies.py` — shared state thread safety, blending, locking
