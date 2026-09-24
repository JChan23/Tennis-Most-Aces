# Tennis ace comparison model

Forecasts **P(player A hits more aces than player B)** from bookmaker ace ladders
and the total-games market.

Built for a live tennis prediction-market competition during the 2026 US Open,
where it was the best-performing of eight markets modelled. Market made up 
**35.6% of my total score**. Final rank 13 globally.

```bash
pip install -r requirements.txt
python ace_model.py example_input.xlsx
```

Most of the code was written by me, with some assistance from Claude when I got stuck

## Input

One excel workbook with two sheets, formatted as below.

`ace_ladders`

| match | player | threshold | odds |
|---|---|---|---|
| A. Bublik vs J. Wolf | Bublik | 15+ | 1.80 |
| A. Bublik vs J. Wolf | Bublik | 20+ | 3.00 |
| A. Bublik vs J. Wolf | Wolf | 10+ | 1.53 |

`games_lines`

| match | line | odds_over | odds_under |
|---|---|---|---|
| A. Bublik vs J. Wolf | 38.5 | 1.87 | 1.95 |
| A. Bublik vs J. Wolf | 39.5 | 2.05 | 1.80 |

Lines and odds are to be taken from a bookmaker. I used Pinnacle and Bet365.

## Method

### 1. Recover each player's ace distribution from a one-sided ladder

Bookmakers quote "N+ aces" with no under side, so you cannot de-vig by
rescaling to sum to 1. The margin has to be estimated jointly with the
distribution:

```
1/odds_i  =  c · P(X ≥ N_i),        X ~ NegBin(μ, k)
```

Taking logs makes `log c` a free **intercept**, and minimising over an
intercept is algebraically identical to fitting only the **ratios** between
rungs. Ratios are margin-free:

```
(1/odds_i) / (1/odds_j)  =  S(N_i) / S(N_j)
```

So the margin never needs to be known. We fit the *shape* of the ladder and let
the *level* float away; `c` is discarded and never enters `P(A>B)`. Verified:
fitting ratios directly gives μ = 21.50 against 21.48 from the free-`c` fit.

This matters because **margins differ between ladders** — fitted `c` ranged
0.99 to 1.20 across the sample. Forcing a common value biases whichever
player's ladder is priced differently: on the worked example it shifted one
player's mean by 0.54 aces and the final answer by 3 points.

Fitted by weighted least squares on the log scale, weights `√(p(1−p))` so
near-certain rungs (1.001, pure rounding) don't dominate. Quotes at 26.00 are a
price ceiling rather than a fair price and are dropped.

Negative binomial rather than Poisson because ace counts are heavily
over-dispersed — fitted `k` is typically 3.5–5, so the `μ²/k` term dominates.

### When can a ladder fit its own margin?

Not simply "when there are three or more prices". What matters is the **span**
— how much of the distribution the rungs actually cover:

```
span = max(1/odds) − min(1/odds)   across the usable rungs
```

A ladder of 1+, 3+, 5+ for a big server is three prices all sitting at 0.98–1.00.
They probe the same corner of the curve and carry roughly one observation's
worth of information between them. A ladder of 1+, 15+, 40+ covers almost the
whole distribution with the same three prices.

Tested exhaustively on every subset of a 9-rung ladder (reference μ = 21.48):

| rungs | span | free `c` error | imposed `c` error |
|---|---|---|---|
| 3 | < 0.6 | **1.33** | 0.50 |
| 3 | ≥ 0.6 | **0.26** | 0.56 |
| 5 | < 0.6 | **1.14** | 0.11 |
| 5 | ≥ 0.6 | **0.24** | 0.51 |

Three well-spread rungs beat five bunched ones by five to one — and note the
reversal, where a narrow span makes the *imposed* margin better, because there
isn't enough information to estimate `c` and letting it float only adds noise.

Rung count is a weak proxy for the same thing:

| rungs | free `c` median error | 90th pct | beats imposed |
|---|---|---|---|
| 3 | 0.35 | **1.95** | 63% |
| 4 | 0.29 | 1.20 | 67% |
| 5 | 0.25 | 1.13 | 79% |
| **6** | **0.19** | **0.50** | **90%** |
| 7 | 0.15 | 0.31 | 97% |

At three rungs the free fit has the better median but a worse *tail* — usually
fine, occasionally badly wrong. Six is the first count where it wins on both,
and where every subset spans widely enough anyway.

**The rule in the code:** fit `c` freely when there are ≥3 rungs *and* span
≥ 0.6; otherwise impose `c = 1.033`.

A caution about diagnostics: the bunched subsets fit to machine precision
(SSE ~1e−15) while returning a mean off by a factor of two. **A low residual
tells you nothing here** — three points through three unknowns always fits.

### 2. Read match length from the total-games market

Two half-lines give both the centre and the width: the median from
interpolating where the survival curve crosses 0.5, and the spread from the
gap between the two de-vigged probabilities, since that gap *is* the local
density (`sd = φ(0)/gap`). De-vigged with the power method, which corrects
favourite-longshot bias.

When two lines are priced almost identically the density estimate explodes, so
the code falls back to a typical spread for the format.

### 3. Compare the players, conditioning on length

Both ace counts scale with the same match length, so they are positively
correlated. Writing `A = μ_A·r + noise` with `r` the relative length:

```
Cov(A, B) = μ_A · μ_B · CV²
```

Convolving independently sets that to zero, which overstates `Var(A − B)` by
`2·CV²·μ_A·μ_B` and drags every answer toward 0.5. On the worked example it
moves the answer from 0.69 to 0.61.

The correct variance, verified against simulation:

```
Var(A − B) = φ(μ_A + μ_B) + CV²(μ_A − μ_B)²
```

| μ_A | μ_B | CV | formula | simulated |
|---|---|---|---|---|
| 12 | 12 | 0.25 | 47.0 | 46.9 |
| 16 | 12 | 0.25 | 55.9 | 55.8 |
| 22 | 8 | 0.25 | 71.0 | 70.9 |
| 22 | 8 | 0.40 | 90.2 | 87.8 |

**The second term vanishes when the players are evenly matched.** Length
uncertainty cancels entirely, because it inflates both counts by the same
factor and the difference is untouched. Numerically, for μ_A = μ_B = 12 the
answer is 0.470 whether the games spread is 0 or 15.

Sensitivity is therefore asymmetric:

| | even (12 v 12) | lopsided (22 v 8) |
|---|---|---|
| spread 0 → 15 games | 0.470 → 0.467 | 0.966 → 0.951 |
| median 36 → 28 games | — | +0.026 |

The **median** matters (≈2.5 points per 8 games of error); the **spread**
barely does. Convenient, since the median is what two half-lines pin down
reliably and the spread is what they often can't.

Scaling is applied on games rather than points. Points per game drifts only
3% across the plausible range (6.13 at 24 games, 6.33 at 54), so the two are
interchangeable in practice.

Implemented by drawing a relative length, scaling both means together,
convolving at `k = μ/(φ−1)`, and averaging.

## The dispersion constant

`PHI = 1.96` is the one empirically fitted input, and getting it wrong was the
model's biggest single error.

Given match length, ace counts are **not** Poisson. Measured across 192 ATP
players with 25+ hard-court matches — expected aces = career rate × that
match's actual service points, so length variation is removed entirely:

```
φ = mean((actual − expected)²) / mean(expected)     Poisson would give 1
```

**median 1.976, mean 1.988, IQR 1.71–2.29**

Ace rate genuinely drifts within a match: balls change every nine games,
servers go bigger at 40-0 and safer on break point, wind and fatigue shift.

The conditional dispersion follows directly — `Var(a|N) = φ·μ`, so
`k = μ/(φ − 1)`. Deriving it instead as a residual (ladder variance minus
length variance) returned a degenerate zero-dispersion answer in 35 of 45
realistic parameter combinations, producing near-certain probabilities with no
justification.

*Caveat:* φ varies by player, and opponent-adjusting the expected count pushes
the median to 2.77 rather than down, so the crude opponent adjustment adds
noise. 1.96 comes from the better-behaved specification and is likely a floor.

## Ties settle NO

`P(A>B) + P(B>A) = 1 − P(tie)`, and the observed ace tie rate is **6.7%**. Two
evenly-matched servers sit near 0.47 each, not 0.50.

Never compute the reverse direction as `1 − P(A>B)` — that equals
`P(B>A) + P(tie)` and inflates it by the whole tie mass.

## Output

```
                 match tour  games_median  exp_points  player_a  exp_aces_a  player_b  exp_aces_b  P(A>B)  P(B>A)  P(tie)
  A. Bublik vs J. Wolf  ATP          38.5         241    Bublik       21.46      Wolf       13.28   0.820   0.147   0.033
```

Also written to `ace_predictions.csv`.

## What this model does not do

It reads the bookmaker's view more carefully than the crowd does; it does not
beat the bookmaker. A coherence test comparing the ladder's implied length
dispersion against the games market suggested 15–29% edges on low ace
thresholds — but measuring φ showed those were entirely an artefact of the
Poisson assumption. Checking all 208 ladders for monotonicity violations found
zero.

The margin itself is only weakly identifiable from one-sided prices — you
cannot separate "the book thinks 0.500 and charges 5%" from "the book thinks
0.525 and charges nothing". Two independent routes agree, though: the median
fitted `c` across 28 ladders is **1.033**, and the overround on the same book's
two-way total-games market is **3.8%**. That is where `POOLED_MARGIN = 1.033`
comes from.

The competition scores against crowd consensus, not against the book. That is
where the edge came from.
