import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize, brentq
from scipy.stats import nbinom, norm


# CONSTANTS

phi = 1.96
# Natural "wobble" of players' acing ability during the match
# Measured on 192 ATP players with 25+ hard court matches (median 1.976)
# Var(aces | match length) = 1.96*E(aces | match length). Distribution isn't Poisson

PriceCap = 25.0 # Bookmaker multiples above this are a ceiling, not a fair price

PooledMargin = 1.033
# Pooled bookmaker margin, used as "c" when a ladder cannot identify its own, where "c" is the "level" of the curve
# Median fitted c across 28 ace ladders = 1.033 (3.3%)

MinSpan = 0.6
MinRungs = 3
# Minimum "span" of implied probabilities before we trust a ladder to fit its own margin
# Span = max(1/odds) - min(1/odds) across the usable rungs (i.e. how much of the distribution the prices actually cover)
# Rungs bunched together all probe the same part of the curve (e.g. rungs of low ace counts only)

Points = {'ATP': (-8.6, 6.484), 'WTA': (-3.7, 6.410)}
# Points = a + b*games
# Straight line fitted by least squares on ATP and WTA hard-court matches

DefaultSD = {'ATP': 9.0, 'WTA': 5.5}
# Default SD of number of games
# Used only when the two games lines are priced so close together that the spread can't be read from them.


# FUNCTIONS

def fit_ladder(thresholds, odds): # Both are arrays
    pts = [(n, o) for n, o in zip(thresholds, odds) if o and o < PriceCap]
    if len(pts) < 2:
        return None # Not enough rungs to price, need two at minimum

    implied = [1.0 / o for _, o in pts]
    span = max(implied) - min(implied)
    free_c = len(pts) >= MinRungs and span >= MinSpan # Boolean
    N = np.array([p[0] for p in pts], float) # Array of the lines (n+ aces)
    P = np.array([1.0 / p[1] for p in pts]) # Array of the implied probabilities
    W = np.clip(P * (1 - P), 1e-3, None) ** 0.5 # Array of standard deviations
    # W correctly identifies which rungs to ignore (down weighting of uninformative rungs) e.g. 1+ ace

    def sse(t): # Scoring function
        mu, k = max(t[0], 0.05), np.exp(np.clip(t[1], -2, 6))
        c = np.exp(np.clip(t[2], -1, 1)) if free_c else PooledMargin
        pred = np.clip(c * (1 - nbinom.cdf(N - 1, k, k / (k + mu))), 1e-9, 1 - 1e-9)
        return float((W * (np.log(pred) - np.log(P)) ** 2).sum())

    start = [0, np.log(15.), .03][:3 if free_c else 2]
    best = min((minimize(sse, [m] + start[1:], method='Nelder-Mead', options={'maxiter': 4000, 'xatol': 1e-6, 'fatol': 1e-9}) for m in (3, 6, 10, 16, 24)), key=lambda r: r.fun)
    mu, k = max(best.x[0], 0.05), float(np.exp(best.x[1]))
    c = float(np.exp(best.x[2])) if free_c else PooledMargin
    # c far from 1 means the curve is not fitting, not that the book is generous
    if mu > 60 or not 0.90 <= c <= 1.35: # bad ladder -> reject
        return None
    return mu, k


def games_median_sd(lines): # Turns two betting lines into a match-length distribution
    lines = sorted(lines)
    tour = 'ATP' if lines[0][0] >= 26 else 'WTA'
    p = []
    for _, o, u in lines: # strip margin
        pi = 1.0 / np.array([o, u])
        try: # power method
            k = brentq(lambda k: (pi ** k).sum() - 1.0, 0.2, 3.0, xtol=1e-12) # chooses k s.t. sum is 1
            p.append((pi ** k)[0])
        except ValueError: # multiplicative method
            p.append((pi / pi.sum())[0])
    if len(lines) >= 2 and p[0] > p[1]:
        med = lines[0][0] + (p[0] - 0.5) / (p[0] - p[1]) # find median via interpolation
        sd = norm.pdf(0) / (p[0] - p[1])
        if not 2.0 <= sd <= 1.6 * DefaultSD[tour]: # when lines are priced closely sd will be too large
            sd = DefaultSD[tour]
    else:
        med, sd = lines[0][0], DefaultSD[tour]
    return tour, float(med), float(sd)


def p_more(mu_a, mu_b, med, sd, draws=200, kmax=160):
    ks = np.arange(kmax + 1)
    rng = np.random.default_rng(5)
    g = np.clip(rng.normal(med, max(sd, 1e-6), draws), 0.35 * med, None) # draw possible match lengths
    total = 0.0
    for r in g / med: # scale expected ace counts with match length
        ma, mb = mu_a * r, mu_b * r
        ka, kb = ma / (phi - 1), mb / (phi - 1) # dispersions
        fa = nbinom.pmf(ks, ka, ka / (ka + ma))
        fb = nbinom.pmf(ks, kb, kb / (kb + mb))
        total += float((fa[1:] * np.cumsum(fb)[:-1]).sum()) # P(A=k) and P(B<k)
    return total / len(g)



args = [a for a in sys.argv[1:] if a.strip()]
DEFAULT_INPUT = 'example_input.xlsx'
path = Path(args[0]).expanduser() if args else Path(__file__).resolve().parent / DEFAULT_INPUT
lad = pd.read_excel(path, sheet_name='ace_ladders')
gam = pd.read_excel(path, sheet_name='games_lines')
lad['threshold'] = lad['threshold'].astype(str).str.replace('+', '', regex=False).astype(int)

rows = []
for match, g in lad.groupby('match', sort=False):
    players = list(g['player'].unique())
    gl = gam[gam['match'] == match]
    if len(players) != 2 or gl.empty:
        print(f"  skip {match}: need 2 players and a games line")
        continue

    tour, med, sd = games_median_sd(list(zip(gl['line'], gl['odds_over'], gl['odds_under'])))
    fits = {p: fit_ladder(g[g['player'] == p]['threshold'].tolist(), g[g['player'] == p]['odds'].tolist()) for p in players}
    if any(f is None for f in fits.values()):
        print(f"  skip {match}: ladder failed the sanity check")
        continue

    a, b = players
    mu_a, mu_b = fits[a][0], fits[b][0]
    a0, b0 = Points[tour]
    rows.append({'match': match, 'tour': tour, 'games_median': round(med, 1),
                 'exp_Points': round(a0 + b0 * med),
                 'player_a': a, 'exp_aces_a': round(mu_a, 2),
                 'player_b': b, 'exp_aces_b': round(mu_b, 2),
                 'P(A>B)': round(p_more(mu_a, mu_b, med, sd), 3),
                 'P(B>A)': round(p_more(mu_b, mu_a, med, sd), 3)})

df = pd.DataFrame(rows)
if df.empty:
    print("no priceable matches")
else:
    df['P(tie)'] = (1 - df['P(A>B)'] - df['P(B>A)']).round(3)
    pd.set_option('display.width', 200, 'display.max_columns', 20)
    print(df.to_string(index=False))
    df.to_csv('ace_predictions.csv', index=False)
    print(f"\n{len(df)} matches priced -> ace_predictions.csv")
    print(f"mean implied tie probability {df['P(tie)'].mean():.3f} "
          f"(historical ace tie rate 0.067)")
