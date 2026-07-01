# Sources of Edge for Kalshi KXHIGH (Daily-High Temperature) Markets — Research Report

_Generated 2026-07-01. Anchored on this repo's own empirical evals (v1 + v2), then
extended with a 2024–2026 literature and practitioner review. Every recommendation is
tagged with the lever it moves: **[ACCURACY]**, **[CALIBRATION]**, or **[EXECUTION]**._

---

## 0. Executive summary (read this first)

Your premise — _"reasonably accurate forecasts, but unprofitable; is it execution?"_ —
is only half right, and your own data already answers it. Three things are true
simultaneously:

1. **Your forecast is not competitive with the book.** On 13,670 real Kalshi markets
   (Apr–Jun 2026), the market's Brier is ~0.095. Your model's is ~0.145 at 24h (v1) and
   ~0.18 across every lead 24–168h (v2 step 1, run today). **No lead beats the market;
   every lead's gated P&L and Sharpe are negative.** So "improve execution on a good
   forecast" is the wrong frame — the forecast is the weaker of the two, not the market.

2. **The book is near-perfect once the day's high is observable.** Your intraday
   obs-conditioned test (running-max from 1-min ASOS, settlement-aligned to the ASOS
   5-min average) shows market Brier ≈ **0.0001** by 8–10pm local. The market prices the
   observable running max as well as or better than you can, at every afternoon hour.
   Per-station residual refinement (today) did not change this.

3. **The microstructure is stacked against a slow taker.** The real Kalshi fee is
   `7%·p·(1−p)` per contract as a **taker**, ~4× lower as a **maker**. An independent
   practitioner running essentially your strategy lost money and traced it to (a) fat
   forecast-error tails (2σ events occur 10–12% of the time, not 5%), (b) fee drag in the
   sub-$0.15 zone, and (c) 15–60 min NWS polling that made them "exit liquidity for the
   bots." They pivoted off weather entirely.

**Conclusion.** KXHIGH is a near-efficient, fee-heavy, latency-dominated venue. The
market already embeds NBM/MOS (station-calibrated) plus live observations, which is
exactly why you can't beat it with a global model. **The realistic solo-dev edge is
microstructure and selectivity, not a better global weather model.** If you pursue
forecast alpha at all, the only two levers with real evidence are (i) multi-provider
_disagreement_ as a trade filter and (ii) fat-tailed, station-specific _calibration_ on
the thin far-tail brackets. Size expectations to "shave the negative into flat-to-small,"
not "print money."

**Highest expected-ROI improvements, in order:**

| # | Change | Lever | Effort | Expected edge |
|---|--------|-------|--------|---------------|
| 1 | Fix the fee model + trade as **maker**, enforce **price floor ≥ $0.15** | EXECUTION | Low | Removes a structural ~1–2¢/contract leak; turns marginal-negative cells flat |
| 2 | Mine your existing segment tables for any (station,bracket,lead) cell where model Brier < market Brier, trade **only** those | EXECUTION/CALIBRATION | Low | Converts a blanket-negative strategy into a selective one; may still be empty |
| 3 | Ingest **multi-model NWP disagreement** via Open-Meteo (AIFS-ENS + GraphCast + GFS + ICON) as a **trade filter**, not a new point forecast | ACCURACY | Medium | The one forecast-side lever with practitioner + literature support |
| 4 | Replace the Gaussian bracket model with **fat-tailed / empirical (DRN/quantile) calibration** per station-lead | CALIBRATION | Medium | Directly targets the 2°F-bracket mispricing the postmortem blames |
| 5 | Latency: sub-minute obs/forecast-drop reaction | EXECUTION | High | Real but contested by faster bots; your own data shows the afternoon book is already sharp |

---

## 1. What your own evals already settled (don't re-litigate)

- **No forecast edge at any lead.** `backtest/real_market_eval.py --multilead` (today):
  model Brier 0.178–0.185 vs constant market Brier ~0.095 for leads {24,48,72,96,120,168}h,
  all P&L negative (−$21 to −$39 on ~3,200 gated trades), all Sharpe negative.
- **No intraday obs edge.** `research/intraday_edge.py`: no afternoon hour beats the book;
  by 8–10pm mktB ≈ 0.0001. Robust to pooled → per-(station,offset) residuals.
- **Settlement subtlety you already found:** Kalshi settles on the **NWS Daily Climate
  Report**, which reflects the ASOS **5-minute average**, not the 1-minute peak (+1°F bias).
  Markets are 6 brackets (middle four 2°F wide, edges cumulative), launched 10am prior day.

These are strong, repo-specific priors. Any new idea below is worth pursuing **only** if it
plausibly flips a _segment_ from negative to positive; blanket "better forecast" does not,
because v1/v2 already falsified it.

---

## 2. Forecast models (0–3 day daily high) — SOTA and reality

**Bottom line: the market's NBM/MOS is the thing to beat, and AI global models don't beat
it on station 2m-temperature yet.**

| Model | 2m-temp / daily-high skill | Uncertainty | Public API | Realistic to use |
|-------|---------------------------|-------------|------------|------------------|
| **NBM** (NOAA National Blend of Models) | The operational US benchmark for station highs; MOS-style, bias-corrected per station. **This is largely what the Kalshi book is pricing.** | Percentiles available | NOMADS/AWS | Yes — and you should treat it as the _competitor_, not an input you can out-forecast |
| **ECMWF IFS-ENS** | Gold-standard physics ensemble; still best-in-class for surface 2m temp | 51-member ensemble | ECMWF open data (free, delayed) | Yes |
| **ECMWF AIFS / AIFS-ENS** | Operational since Feb 2025 (ENS since 1 Jul 2025). Beats physics models up to ~20% on some upper-air fields, **but at 1° still _behind_ operational IFS for 2m temperature** | 51-member ENS | ECMWF open data; Open-Meteo; Meteomatics | Yes — but no free lunch for surface temp |
| **GenCast** (DeepMind, Nature 2024) | Diffusion-based probabilistic; strong ensemble skill at low cost | Native probabilistic samples | Weights public; harder to run live | Medium |
| **GraphCast** (DeepMind) | Deterministic, competitive medium-range | None (single member) | Open-Meteo serves it; earth2studio | Yes (easy via Open-Meteo) |
| **Aurora** (Microsoft) | 1.3B-param foundation model, fine-tunable | Task-dependent | Weights public | Medium/High effort |
| **Pangu / FourCastNet** | Early AI NWP; superseded on surface skill | Limited | earth2studio (NVIDIA) | Medium |
| **HRRR / NAM / GFS** | HRRR is high-res convective (good for same-day); GFS the US global baseline | GEFS ensemble for GFS | NOMADS / Open-Meteo | Yes |

**Key facts (2025–2026):**
- AI models have overtaken physics on many upper-air metrics but **2m temperature over
  land, verified against stations, is still a relative weak spot** — precisely the variable
  you trade. AIFS at 1° trails IFS there.
- **NBM/MOS is station-calibrated and observation-anchored.** A raw global model (AI or
  physics) is systematically worse at a specific airport than NBM until you post-process it
  to that station — which is what NBM already did. This is the structural reason the book
  beats you.

**Implication:** Don't chase a single "better" global model. The only forecast-side value
is in **combining** models and exploiting **disagreement** (Section 3 + 10).

---

## 3. Ensembles & multi-model combination — the best-evidence forecast lever

Literature is clear that **combining** forecasts beats any single source, and that simple
methods are hard to beat:
- **BMA / EMOS / stacking / confidence-weighted averaging** consistently outperform the
  best individual member on 2m temp (EUPPBench results across 24/72/120h leads).
- For a solo dev, **inverse-error weighting** and **online/adaptive weighting** capture most
  of the gain with a fraction of the complexity of full BMA.

**But** — and this is the trap — the market already sees a blend (NBM). So an ensemble that
merely reconstructs NBM has no edge. The value is:
- **[ACCURACY] Multi-provider disagreement as a _filter_.** When AIFS-ENS, GraphCast, GFS,
  and ICON **agree tightly** but the Kalshi book is spread out (round-number anchoring),
  that is a candidate mispricing. When they **disagree**, stand down — that's where your
  0.18-Brier tail lives. Trade selectivity beats forecast precision here.
- **Cheapest path:** Open-Meteo's free API serves AIFS, GraphCast, GFS, ICON, and their
  ensembles plus a historical archive — one endpoint, no infra. This is the single most
  practical data upgrade for a solo dev.

---

## 4. Probabilistic forecasting → market probabilities (calibration)

Kalshi needs a full distribution over the daily high, and your 2°F brackets need
**sub-1°F sharpness**. The postmortem's #1 loss driver was assuming Gaussian errors when
**temperature-error tails are fat** (2σ ≈ 10–12%, not 5%).

Best-practice, in rough order of effort/payoff:
- **[CALIBRATION] Empirical / quantile residuals per (station, lead).** You already started
  this in `research/intraday_edge.py`. Extend it: fit the residual distribution
  non-parametrically (or with quantile regression / QRF, which you have) rather than a
  normal, so the tails are honest. **Highest calibration ROI for lowest effort.**
- **DRN (Rasp & Lerch 2018)** — neural EMOS: NN maps features → distribution parameters.
  The reference method for NN 2m-temp post-processing; strong on EUPPBench.
- **EMOS (Gneiting 2005)** — per-station, per-lead Gaussian regression; the baseline DRN
  extends. Trivial to implement, surprisingly strong, good sanity check.
- **Conformal prediction** — distribution-free coverage guarantees; a clean way to get
  honest bracket probabilities without assuming a family. Good fit for the fat-tail problem.
- **CRPS-optimized training** — train the distribution to minimize CRPS (proper score),
  not point MAE. You already reference inverse-CRPS blend weights in `plans/model-gaps.md`.
- **Isotonic / spline recalibration** on the final bracket probabilities (you have a
  calibrator) — cheap last-mile fix, but can't repair a mis-specified tail upstream.

**Reality check:** market Brier 0.095 is already well-calibrated. Better calibration only
pays where the book is _mis_-calibrated — the thin far-tail brackets and round-number
strikes (Section 10), not the liquid center.

---

## 5. Forecast-error modeling (residual corrections)

Error grows with lead, and varies by station, regime, season, and temperature range —
model it explicitly:
- **[CALIBRATION] Per-(station, lead-bucket, month) bias + spread.** You have lead buckets
  {D1-2, D3-4, D5-7}; add month and a coarse regime tag. `plans/model-gaps.md #7`
  (per-station-month calibration) is the right item.
- **Regime conditioning:** clear/dry radiative days vs advective/cloudy days have very
  different high-temp error (radiative days blow the upper tail). A cloud-cover / dewpoint
  feature to split the residual model is cheap and directly targets the fat tail.
- **Analog / weather-regime clustering** for the residual distribution (k-NN on synoptic
  features) is a well-supported way to get regime-dependent uncertainty without a big model.

---

## 6. Features that most improve daily-high prediction

Beyond temperature, the highest-signal additions for **daily max** specifically:
- **Cloud cover** (overnight + daytime) — dominant control on whether the high is reached.
- **Dewpoint / boundary-layer moisture** — caps daytime heating; strong high-temp predictor.
- **Boundary-layer height / mixing** and **850hPa temperature + thickness** — advection.
- **Wind (speed/direction)** — onshore vs offshore flips coastal stations (you saw the
  Seattle 91→77°F UTC/local bug; coastal regime matters).
- **Soil moisture / recent precip** — wet ground suppresses highs (evaporative cooling).
- **Snow cover / albedo** — big cold bias when present (seasonal).
- **Radiation / clear-sky index**, **CAPE** (convective capping), **pressure tendency**.
- **Urban-heat-island / airport-instrument quirks** — per-station constants (Section 10).

Practitioner note: for a gradient-boosted post-processor, ~10–15 of these features capture
almost all the gain; more mainly adds variance.

---

## 7. ML approach for tabular post-processing

For station-level tabular met data, **gradient-boosted trees win** and are what you should
keep:
- **LightGBM / XGBoost / CatBoost** — SOTA on tabular; you use LightGBM (fine). CatBoost is
  worth a bake-off for its categorical (station/regime) handling.
- **NGBoost / QRF** (you have both) — native probabilistic output; keep for the distribution.
- **DRN / small MLP** — competitive _only_ with enough data and careful regularization;
  marginal over GBMs here.
- **TabNet / Transformers / TFT / LSTM** — **not worth it** at your data scale; they lose to
  GBMs on tabular met post-processing and add engineering cost.
- **GNNs** — genuinely useful for _spatial_ post-processing across a station network (recent
  papers show gains by sharing information between stations). Higher effort; consider only
  after Sections 1–4 are exhausted.

**Verdict:** your model class is already correct. The problem is not the learner; it's that
the target (beat NBM+obs) is near-unbeatable, plus calibration tails and execution.

---

## 8. Kalshi microstructure — where your losses actually come from

This is the highest-ROI area because it's cheap and your data says the forecast side is
tapped out.

- **[EXECUTION] Fee model is wrong in your code.** `per_trade_pnl` books `FEE_RATE=0.05 ×
  mid`. Actual Kalshi **taker** fee ≈ `7%·p·(1−p)` (max $0.0175/contract at p=0.5);
  **maker** fee ≈ `1.75%·p·(1−p)` (~4× lower). Your model over-charges at the extremes and
  under-charges near 50/50 — it is mis-ranking which trades clear the gate. Fix this first;
  it changes every P&L number.
- **[EXECUTION] Be a maker, not a taker.** Resting limit orders pay ~¼ the fee and earn the
  spread instead of paying it. Over 100×100-contract round-trips near 50¢ that's ~$262/mo
  saved. For a marginal strategy, maker vs taker is the difference between negative and flat.
- **[EXECUTION] Hard price floor ≥ $0.15.** Below that, fee drag alone requires >83% win
  just to break even (postmortem). Never gate a trade into the sub-15¢ zone.
- **[EXECUTION] Liquidity/selectivity.** Thin far-tail brackets are where mispricing lives
  but also where you move the book and can't exit. Size to displayed depth; skip markets you
  can't exit.
- **[EXECUTION] Latency.** Your 15–60 min poll cadence is the postmortem's #3 killer — you're
  exit liquidity for faster bots after each NWS/obs drop. Either react in **seconds** to
  obs/forecast updates (hard, contested) or **avoid** competing on freshness and trade only
  slow, structural mispricings (round-number anchoring at market open, 10am prior day).
- **Kelly / inventory:** only relevant _after_ a positive-edge cell exists. Fractional Kelly
  (¼–½) on a genuine edge; hard per-market and per-day exposure caps. Don't tune this until
  Sections 1–2 surface a positive cell.

**Answer to "is it execution?"** Partly yes — fees, maker/taker, price floor, and latency
are real, fixable leaks. But execution fixes alone won't make money on a forecast that's
Brier-0.18 against a 0.095 book. You need selectivity (trade the few good cells) _and_ clean
execution together.

---

## 9. Backtesting methodology (protect any positive result)

Your harness is already unusually honest (temporal split, real settlements, no look-ahead,
`leakage_audit.py`). Keep and reinforce:
- **Walk-forward with rolling retrain**, not a single split — regime drift is real.
- **Score on trading metrics** (gated P&L, Sharpe, hit-rate vs breakeven-after-fees), not
  just Brier. You already do this; make breakeven fee-aware with the corrected fee model.
- **Reliability diagrams** per station/lead to see _where_ calibration fails (the tails).
- **Purged/embargoed splits** around each settlement date to kill any obs leakage.
- **Honesty rule (from handoff):** audit every data artifact against an independent source;
  a positive result must survive `leakage_audit.py` + a no-look-ahead check. You've been
  burned by silent data bugs before (NBM boustrophedon scramble, date off-by-ones).

---

## 10. Unconventional alpha — ranked by evidence

1. **[EXECUTION] Round-number / anchoring mispricing at market open (best evidence).** Retail
   anchors on round strikes and underprices forecast certainty (multiple practitioner
   accounts). Markets open 10am the prior day — the least-informed moment. A disciplined,
   fee-aware model vs the _opening_ book is the most plausible structural edge.
2. **[ACCURACY] Provider disagreement filter (Section 3).** Trade only when models agree and
   the book is wide; stand down otherwise. Evidence: ensemble literature + practitioner use.
3. **[CALIBRATION] Persistent station biases / airport instrument quirks.** Specific ASOS
   sites have known warm/cool micro-biases and siting effects. A per-station constant learned
   from history is cheap and real — but the market's NBM already corrects most of it, so the
   residual edge is small.
4. **[CALIBRATION] Fat-tail / extreme-event brackets.** The far "over/under" cumulative
   brackets are where the Gaussian assumption fails and where the book may be lazy. Your
   corrected fat-tailed calibration (Section 4) points here.
5. **[EXECUTION] Rapid forecast-update reaction.** Real but bot-contested and infra-heavy;
   your own intraday result shows the afternoon book is already sharp. Low priority for a
   solo dev.
6. **Human forecaster adjustments (NWS AFD text).** NWS forecasters sometimes override guidance;
   parsing the Area Forecast Discussion for "warmer/cooler than guidance" language is a
   speculative, low-confidence signal. Interesting, unproven.

---

## 11. Prioritized roadmap (easiest/highest-ROI → hardest/highest-payoff)

1. **Correct the fee model + maker execution + $0.15 floor.** [EXECUTION] Low effort.
   Re-run the multi-lead and intraday gates with real fees — this alone may move several
   cells from negative to flat. _Do this before any modeling._
2. **Segment mining.** [EXECUTION/CALIBRATION] Low. From your existing by-station /
   by-strike / by-volume / by-lead tables, isolate any cell with model Brier < market Brier
   AND positive fee-aware P&L over a walk-forward. If none survive, that is itself the answer.
3. **Multi-model ingestion via Open-Meteo (AIFS-ENS, GraphCast, GFS, ICON) + disagreement
   filter.** [ACCURACY] Medium. Trade selectivity, not point precision.
4. **Fat-tailed per-(station,lead) calibration** (empirical/quantile → conformal/DRN),
   CRPS-scored. [CALIBRATION] Medium. Targets the 2°F brackets and cumulative tails.
5. **Regime-conditioned residuals** (cloud/dewpoint/soil-moisture split). [CALIBRATION]
   Medium. Fixes the radiative-day upper-tail blowups.
6. **Spatial GNN post-processing across the station network.** [ACCURACY] High. Only after
   1–5 and only if a positive cell exists to amplify.
7. **Sub-minute latency reaction stack.** [EXECUTION] High. Contested by bots; pursue only
   if you've found a freshness-sensitive cell that survives fees.

**If after steps 1–4 no cell is positive:** the honest, evidence-backed conclusion (matching
your v1/v2 data _and_ the practitioner postmortem) is that KXHIGH is efficient enough that a
solo dev's expected edge is ~zero after fees, and capital is better deployed on
higher-price, base-rate-divergence markets (FOMC/CPI/jobs) where fee drag is smaller — which
is exactly where the postmortem author pivoted.

---

## Appendix — papers, repos, APIs, datasets

**Papers (2018–2026):**
- Rasp & Lerch (2018), _Neural networks for post-processing ensemble weather forecasts_ (DRN).
- Gneiting et al. (2005), EMOS / non-homogeneous Gaussian regression.
- EUPPBench (ESSD 15:2635, 2023) — 2m-temp post-processing benchmark, 24/72/120h leads.
- Permutation-invariant NN post-processing — AMS AIES 3(1), 2024 (arXiv 2309.04452).
- GNN + spatial post-processing (arXiv 2407.11050); sharpness in NN parametric PP (arXiv 2606.08587).
- AIFS (arXiv 2406.01465; update arXiv 2509.18994).
- GenCast (Nature, 2024).
- AI-NWP monsoon assessment (arXiv 2509.01879) — surface-skill caveats.

**Repos:**
- `google-deepmind/graphcast` (GraphCast + GenCast).
- `NVIDIA/earth2studio` (run GraphCast/FourCastNet/Pangu locally; PyPI `earth2studio`).
- `EUPP-benchmark` (org) — benchmark data + baselines.
- `jaychempan/Awesome-LWMs` — curated large-weather-model list.
- Microsoft Aurora (weights public).

**APIs / datasets:**
- **Open-Meteo** — free; serves AIFS, GraphCast, GFS, ICON + ensembles + historical archive.
  _Best single upgrade for a solo dev._
- **ECMWF open data** — AIFS / IFS-ENS, free (delayed).
- **Meteomatics** — AIFS-ENS + MetX, commercial.
- **NOAA NOMADS / AWS Open Data** — NBM, GFS/GEFS, HRRR, NAM.
- **IEM ASOS 1-minute** — sub-hourly obs (you already use it for running-max).
- **NWS Daily Climate Report / CLI** — the settlement source; ASOS 5-min-average high.

**Sources (web):**
- Kalshi Help — Weather Markets: https://help.kalshi.com/en/articles/13823837-weather-markets
- Kalshi fee schedule (2026): https://kalshi.com/docs/kalshi-fee-schedule.pdf
- Practitioner postmortem: https://www.northlakelabs.com/max/blog/kalshi-weather-postmortem-and-pivot/
- Lychee guide to Kalshi weather pricing: https://lycheedata.com/guides/kalshi-weather-prediction-markets-analysis
- ECMWF AIFS operational: https://www.ecmwf.int/en/about/media-centre/news/2025/ecmwfs-ai-forecasts-become-operational
- Open-Meteo ECMWF/AIFS + GraphCast: https://open-meteo.com/en/docs/ecmwf-api ; https://openmeteo.substack.com/p/exploring-graphcast
- EUPPBench: https://essd.copernicus.org/articles/15/2635/2023/
