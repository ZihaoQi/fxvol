"""Spot x vol revaluation grid + bucketed vega + limit monitor.

WHY A GRID, NOT JUST GREEKS
A single delta number tells you your risk for an infinitesimal move. It hides
everything that matters in the tails: gamma flipping sign, vega concentrating at
one strike, P&L cratering past a barrier, pin risk into an expiry. So desks
REVALUE the whole book across a ladder of spot shocks x vol shocks and look at
the P&L surface. Where it craters is your real risk - the thing limits exist to
contain.

WHAT THIS MODULE PRODUCES
  1. RiskGrid     : full-reval P&L for every (spot shock, vol shock) cell.
  2. bucketed vega: vega per tenor bucket (1w/1m/3m/6m/1y...), because a flat
     'total vega' hides being long short-dated vol and short long-dated vol -
     a term-structure bet, not flat risk.
  3. LimitMonitor : configurable limits per Greek/bucket with breach flags.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..core.black_scholes import gk_greeks
from ..pnl.book import Book
from ..pnl.engine import MarketState


@dataclass
class RiskGrid:
    spot_shocks: list[float]      # fractional, e.g. -0.05..+0.05
    vol_shocks: list[float]       # absolute vol, e.g. -0.05..+0.05
    pnl: np.ndarray               # shape (len(spot_shocks), len(vol_shocks))

    def worst_cell(self) -> tuple[float, float, float]:
        i, j = np.unravel_index(np.argmin(self.pnl), self.pnl.shape)
        return self.spot_shocks[i], self.vol_shocks[j], float(self.pnl[i, j])

    def report(self) -> str:
        lines = ["Spot x Vol revaluation grid (P&L)", "=" * 60,
                 "rows = spot shock, cols = vol shock (abs vol pts)"]
        header = "spot\\vol " + " ".join(f"{v*100:+6.1f}" for v in self.vol_shocks)
        lines.append(header)
        for i, s in enumerate(self.spot_shocks):
            cells = " ".join(f"{self.pnl[i, j]:+6.0f}"
                             for j in range(len(self.vol_shocks)))
            lines.append(f"{s*100:+6.1f}% {cells}")
        ss, vs, worst = self.worst_cell()
        lines += ["=" * 60,
                  f"worst cell: spot {ss*100:+.1f}%, vol {vs*100:+.1f}pts "
                  f"-> P&L {worst:+,.0f}"]
        return "\n".join(lines)


def _reprice_book(book: Book, m: MarketState, spot_mult: float,
                  vol_add: float) -> float:
    """Total book value under a spot shock (multiplicative) and vol shock (additive)."""
    S = m.spot * spot_mult
    total = 0.0
    for p in book.positions:
        v = m.surface.implied_vol(p.strike, p.expiry) + vol_add
        v = max(v, 1e-4)
        g = gk_greeks(S, p.strike, p.expiry, m.r_dom, m.r_for, v, p.is_call)
        total += g.price * p.notional
    return total


def build_risk_grid(book: Book, m: MarketState,
                    spot_range: float = 0.05, spot_steps: int = 5,
                    vol_range: float = 0.05, vol_steps: int = 5) -> RiskGrid:
    """Full-reval P&L grid relative to the base (unshocked) book value."""
    spot_shocks = list(np.linspace(-spot_range, spot_range, spot_steps))
    vol_shocks = list(np.linspace(-vol_range, vol_range, vol_steps))
    base = _reprice_book(book, m, 1.0, 0.0)
    pnl = np.zeros((spot_steps, vol_steps))
    for i, ss in enumerate(spot_shocks):
        for j, vs in enumerate(vol_shocks):
            pnl[i, j] = _reprice_book(book, m, 1.0 + ss, vs) - base
    return RiskGrid(spot_shocks, vol_shocks, pnl)


# ---- bucketed vega ---------------------------------------------------------

# Standard tenor buckets (year-fraction upper bounds), desk-style.
BUCKETS: list[tuple[str, float]] = [
    ("1w", 7 / 365), ("2w", 14 / 365), ("1m", 1 / 12), ("2m", 2 / 12),
    ("3m", 0.25), ("6m", 0.5), ("9m", 0.75), ("1y", 1.0), ("2y", 2.0),
    ("LT", 1e9),
]


def bucketed_vega(book: Book, m: MarketState) -> dict[str, float]:
    """Vega per tenor bucket (per 1.00 vol move), notionals applied."""
    out = {name: 0.0 for name, _ in BUCKETS}
    for p in book.positions:
        v = m.surface.implied_vol(p.strike, p.expiry)
        g = gk_greeks(m.spot, p.strike, p.expiry, m.r_dom, m.r_for, v, p.is_call)
        for name, ub in BUCKETS:
            if p.expiry <= ub:
                out[name] += g.vega * p.notional
                break
    return out


# ---- limit monitor ---------------------------------------------------------

@dataclass
class Limit:
    name: str
    metric: str          # 'delta','gamma','vega','vega_bucket','grid_worst'
    bound: float         # absolute limit (breach if |value| > bound)
    bucket: str = ""     # for metric == 'vega_bucket'


@dataclass
class Breach:
    limit: Limit
    value: float

    def __str__(self) -> str:
        b = f"[{self.limit.bucket}]" if self.limit.bucket else ""
        return (f"BREACH {self.limit.name}{b}: |{self.value:,.0f}| "
                f"> {self.limit.bound:,.0f}")


@dataclass
class LimitMonitor:
    limits: list[Limit] = field(default_factory=list)

    def check(self, book: Book, m: MarketState) -> list[Breach]:
        # aggregate book Greeks
        agg = dict(delta=0.0, gamma=0.0, vega=0.0)
        for p in book.positions:
            v = m.surface.implied_vol(p.strike, p.expiry)
            g = gk_greeks(m.spot, p.strike, p.expiry, m.r_dom, m.r_for, v, p.is_call)
            agg["delta"] += g.delta * p.notional
            agg["gamma"] += g.gamma * p.notional
            agg["vega"] += g.vega * p.notional
        vbuckets = bucketed_vega(book, m)

        breaches: list[Breach] = []
        for lim in self.limits:
            if lim.metric in agg:
                val = agg[lim.metric]
            elif lim.metric == "vega_bucket":
                val = vbuckets.get(lim.bucket, 0.0)
            elif lim.metric == "grid_worst":
                _, _, val = build_risk_grid(book, m).worst_cell()
            else:
                continue
            if abs(val) > lim.bound:
                breaches.append(Breach(lim, val))
        return breaches

    def report(self, book: Book, m: MarketState) -> str:
        breaches = self.check(book, m)
        if not breaches:
            return "Limit monitor: all limits OK"
        return "Limit monitor: " + str(len(breaches)) + " breach(es)\n" + \
               "\n".join("  " + str(b) for b in breaches)
