# The tennis order-flow edge, explained from scratch

Companion to `plans/tennis-mm-next-steps.md` (which has the raw numbers and
is the living next-steps doc). This file explains *what the edge actually
is*, in plain language, and is honest about where it could be wrong.

## 1. The basics — what a Kalshi tennis market is

Each Kalshi tennis market is a yes/no contract on one outcome, e.g. "will
Player A win this match." The price is always between $0.01 and $0.99, and
it's read directly as a probability: a "yes" contract trading at $0.65
means the market thinks Player A has a 65% chance to win. Two prices exist
at any moment:

- **best bid** — the highest price someone is currently willing to *buy*
  yes at (there's a resting order waiting there)
- **best ask** — the lowest price someone is currently willing to *sell*
  yes at (equivalently, the highest price someone will buy "no" at)

The gap between them is the **spread**. If you want a fill *right now* you
have to "cross the spread" — buy at the ask or sell at the bid, both worse
than the midpoint. If you're willing to wait, you can rest an order at the
bid or ask and hope someone trades against you — but then you're exposed to
the price moving before anyone does.

Every time a trade actually happens, it happens at one of those two prices,
and it tells you something: a trade at the **bid** means someone was
willing to *sell* yes right now (bearish pressure). A trade at the **ask**
means someone was willing to *buy* yes right now (bullish pressure). This
sign — which side of the book a trade printed on — is the "trade sign."

## 2. What we actually tested (in plain language, no math)

We already knew (from the earlier market-making test) that resting orders
lose money: the moment right after your resting order gets filled, the
price tends to keep moving in the direction that hurts you. That's called
**adverse selection** — you only get filled because someone with better
information than you decided to trade.

That raised an obvious follow-up question: if the price reliably keeps
moving after a trade, could *you* be the one causing that move to work for
you instead of against you — by reacting to the trade, not resting ahead of
it?

So we watched every trade print across three days of full order-book data
(2026-07-16, 07-17, 07-18 — ~108,000 trades, spread across 204
distinct match-markets) and asked: after a trade prints with a given sign,
does the price keep drifting the same direction over the next 5 to 120
seconds, more than you'd expect from random noise? Then, assuming you
reacted instantly and paid the real cost of trading (crossing the spread +
Kalshi's per-trade fee), would buying into that direction have made money?

## 3. The edge, plainly stated

**Yes, but only in one direction.** When a trade prints at the bid (someone
sold yes — bearish pressure), the price reliably kept drifting down over
the following seconds/minutes, and buying "no" right at that moment,
net of fees, made money on average across 159 different matches (67% of
them individually profitable). That's the real part of the edge.

When a trade prints at the ask (someone bought yes — bullish pressure), the
same drift barely showed up after accounting for trading costs — buying
"yes" in response was only weakly profitable, close to a coin flip
market-by-market. **Half the theoretical strategy doesn't actually work.**

So the honest one-sentence version: *there is a real, fee-surviving,
short-horizon momentum signal in tennis order flow, but it's asymmetric —
it only reliably pays off when following selling pressure, not buying
pressure.*

## 4. Why might this exist? (intuition, not proven)

A few plausible, non-exclusive explanations:

- **Informed flow reacting faster than the rest of the market.** Someone
  watching the match (or a faster data feed) sees something — a break of
  serve, an injury, a shift in momentum — before the price fully reflects
  it, and starts trading. The price doesn't jump instantly to the new
  fair value; it drifts there over the following seconds as more
  participants react. This is a well-documented phenomenon in real
  financial markets too (order flow predicting short-term returns), not
  something unique to Kalshi or tennis.
- **Thin liquidity slows price discovery.** These are much less liquid
  markets than, say, S&P 500 futures — fewer participants means it takes
  longer for a price to fully absorb new information, which is exactly the
  kind of gap a fast reactor can exploit.
- **The bearish/bullish asymmetry might reflect a behavioral bias.** It's
  a similar shape to the well-known "favorite-longshot bias" in betting
  markets, where bettors are systematically too willing to buy the
  "exciting" outcome (here: staying long yes, the favorite continuing to
  win) and slower to react to bad news for their side. That would explain
  why downside moves (bad news) are underpriced and correct more reliably
  than upside moves.

None of these are confirmed — they're reasonable stories that fit the
asymmetry, not something we've separately verified.

## 5. Pros — why this looks like a real finding, not noise

- **Self-contained.** Doesn't need external odds data, a live score feed,
  or any "true probability" model — it only uses Kalshi's own public order
  book and trade tape, the cheapest possible data source.
- **Passed integrity checks, not just a raw correlation.** A placebo test
  (randomly shuffling trade signs within each market) collapsed the
  correlation from 0.48 to 0.07 — ruling out an indexing/alignment bug in
  the code. The result also held up when computed *per market* rather than
  pooled (avoiding one or two big movers faking a signal) and when the
  biggest 1% of price moves were excluded (not driven by tail events).
- **Reasonable sample size.** 108,346 individual trade signals, but more
  importantly ~150-200 *independent* market-clusters after accounting for
  the fact that trades within one match aren't independent of each other —
  that's the number that actually matters for trusting the statistics, and
  it's not tiny.
- **Net of real costs.** The P&L number already subtracts Kalshi's taker
  fee (`TAKER_FEE_COEF = 0.07`, four times the fee a resting maker order
  pays) and uses the actual ask/bid a taker would have to cross, not the
  friendlier midpoint.

## 6. Cons — real limitations, not just theoretical caveats

- **Only half the signal survives contact with costs.** The "yes"-side
  (following bullish trades) is weak and inconsistent market-to-market —
  don't trade it as-is. Real usable opportunity set is smaller than the
  headline numbers suggest.
- **Lottery-ticket-shaped payoff.** Most individual trades lose a small
  amount; the positive average comes from a smaller number of trades that
  catch a real sustained move. Concretely: the median trade lost money,
  the mean trade made money. That's a legitimate payoff shape for a
  trend-following strategy, but it means a real account running this would
  *feel* like it's losing most of the time even while being profitable in
  aggregate — a real psychological and risk-management problem, not just a
  math one.
- **Small calendar window.** All of this comes from three consecutive days
  (07-16 to 07-18) — the only days so far with full order-book depth
  data (the recorder that captures it only started 07-16, then died for
  nine days from an unrelated cron bug and was just restarted). There's no
  out-of-sample test on a separate, later period yet. Three days could
  share idiosyncratic conditions (which tournaments were live, which
  players, what the overall market mood was) that won't repeat.

## 7. Where this could be wrong — specific failure modes to keep in mind

- **Latency is assumed away.** The simulation reacts *instantly* to a
  trade print, using the exact book state already captured. In reality,
  by the time your order reaches Kalshi, the price you saw may already be
  gone — you're not the only one seeing that same trade print. If the
  edge exists *because* fast reactors are exploiting slow ones, you have
  no evidence yet that you'd be fast enough to be on the winning side of
  that race rather than the losing one. This hasn't been tested — it's the
  single biggest open question.
- **Size is assumed away.** Every simulated trade is 1 contract. Real
  money requires real size, and larger orders walk the book (worse average
  fill price) the same way a big market maker's resting orders get picked
  off — this analysis doesn't yet account for that cost at all.
- **Statistical fragility from clustering.** The naive statistical
  significance (treating each of the 108,346 trades as independent) gave
  an absurd t-statistic around +83 — a dead giveaway that the math was
  being fooled by autocorrelation (trades in the same match aren't
  independent events). The corrected, per-market number (t≈7.2 for the
  "no" side) is much more credible, but it's still built on roughly 150-200
  clusters, not tens of thousands of truly independent samples. Reported
  numbers should be trusted at that more modest confidence level, not the
  inflated one.
- **Fee model may not transfer exactly.** The fee formula used
  (`backtest/track_b.py`, `TAKER_FEE_COEF = 0.07`) was derived for the
  weather-forecasting work on different Kalshi series. It hasn't been
  separately confirmed that Kalshi doesn't apply different fee terms to
  the tennis series specifically.
- **Settlement contamination isn't fully ruled out.** We checked that the
  earlier momentum correlation wasn't just "a match near its end trends
  toward its obvious outcome" by filtering to mid-range prices, but the
  taker P&L test's forward-looking window (up to 120 seconds) could still
  occasionally straddle an actual match end for a market that was mid-range
  at signal time but resolves within the window — that would show up as an
  artificially clean win/loss, not a real predictive signal. Not separately
  checked yet.
- **Selection-after-the-fact risk.** We tested a few related signals
  (imbalance, a 5-trade momentum sum, raw trade sign) and reported trade
  sign because it looked cleanest — there was no separate holdout period
  to confirm that choice wasn't just the best-looking one among several by
  chance.

## 8. Bottom line

There's a real, cost-surviving, statistically defensible signal here — but
it's narrower than the headline numbers suggest (one side only), built on a
short and possibly unrepresentative window of data, and completely silent
on the two things that would matter most for actually trading it: whether
you can react fast enough to be a beneficiary of the effect rather than a
victim of it, and whether it holds up at a size worth trading.
