"""Cross pair risk aggregation, concentration flags, and stress scenarios.

WHY A SINGLE PAIR RISK GRID IS NOT ENOUGH

risk.grid gives a full spot times vol revaluation for one pair's book. A real
desk runs several pairs at once, and the risk that actually gets a trader
called into a meeting is rarely visible in any single pair's numbers. Two
failure modes a one-pair view cannot see by construction:

  concentration : the book looks fine in total, but 80 percent of the vega
                  sits in one pair, or one tenor bucket, so a single move in
                  one place does most of the damage. A flat total vega number
                  and a concentrated one look identical until you bucket them.
  correlated stress : pairs do not move independently. A scenario where two
                  pairs move together (say, broad USD strength) or apart
                  (a correlation break) can hurt in a way that shocking each
                  pair on its own, one at a time, never reveals.

This module aggregates a multi pair book by pair and by tenor bucket, flags
concentration above a threshold, and runs a small library of named stress
scenarios across the whole book at once.

Simplifying assumption, kept consistent with the rest of the project: every
position's price times notional is already expressed in the desk's reporting
currency (the same assumption pnl.engine and risk.grid make for a single
pair), so P&L across pairs is summed directly rather than converted through a
cross rate.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from fxvol.pricing.core.black_scholes import gk_greeks

from ..pnl.book import Book, OptionPosition
from ..pnl.engine import MarketState
from ..pnl.valuation import ShockedMarketState, reprice_multi_pair_book
from .grid import BUCKETS, bucket_for


def positions_by_pair(book: Book) -> dict[str, list[OptionPosition]]:
    out: dict[str, list[OptionPosition]] = {}
    for p in book.positions:
        out.setdefault(p.pair, []).append(p)
    return out


@dataclass
class PairGreeks:
    pair: str
    delta: float = 0.0
    gamma: float = 0.0
    vega: float = 0.0
    vega_by_bucket: dict[str, float] = field(default_factory=dict)


def aggregate_by_pair(book: Book, markets: dict[str, MarketState]
                      ) -> dict[str, PairGreeks]:
    """Delta, gamma, and bucketed vega per pair. Requires one MarketState per
    pair traded, keyed by the same pair strings used in the book."""
    out: dict[str, PairGreeks] = {}
    for pair, positions in positions_by_pair(book).items():
        m = markets[pair]
        agg = PairGreeks(pair=pair, vega_by_bucket={name: 0.0 for name, _ in BUCKETS})
        for p in positions:
            vol = m.vol(p.strike, p.expiry)
            g = gk_greeks(m.spot, p.strike, p.expiry, m.r_dom, m.r_for, vol,
                         p.is_call)
            agg.delta += g.delta * p.notional
            agg.gamma += g.gamma * p.notional
            agg.vega += g.vega * p.notional
            agg.vega_by_bucket[bucket_for(p.expiry)] += g.vega * p.notional
        out[pair] = agg
    return out


@dataclass
class ConcentrationFlag:
    category: str            # 'pair' or 'bucket'
    name: str
    share: float              # fraction of total absolute vega
    threshold: float

    def __str__(self) -> str:
        return (f"CONCENTRATION [{self.category}] {self.name}: "
                f"{self.share:.0%} of book vega > {self.threshold:.0%} threshold")


def check_concentration(book: Book, markets: dict[str, MarketState],
                        pair_threshold: float = 0.60,
                        bucket_threshold: float = 0.60
                        ) -> list[ConcentrationFlag]:
    """Flag when one pair, or one tenor bucket summed across ALL pairs, holds
    more than its threshold share of total absolute vega. Absolute vega is
    used for the denominator (not net vega) because a book that is long vega
    in one pair and short vega in another does not net out risk between the
    two; they are different underlyings and must be hedged separately."""
    by_pair = aggregate_by_pair(book, markets)
    total_abs_vega = sum(abs(g.vega) for g in by_pair.values())
    flags: list[ConcentrationFlag] = []
    if total_abs_vega < 1e-9:
        return flags

    for pair, g in by_pair.items():
        share = abs(g.vega) / total_abs_vega
        if share > pair_threshold:
            flags.append(ConcentrationFlag("pair", pair, share, pair_threshold))

    bucket_totals: dict[str, float] = {name: 0.0 for name, _ in BUCKETS}
    for g in by_pair.values():
        for name, v in g.vega_by_bucket.items():
            bucket_totals[name] += abs(v)
    for name, v in bucket_totals.items():
        share = v / total_abs_vega
        if share > bucket_threshold:
            flags.append(ConcentrationFlag("bucket", name, share, bucket_threshold))

    return flags


# ---- stress scenarios -------------------------------------------------------

@dataclass
class StressScenario:
    name: str
    spot_mult: dict[str, float] = field(default_factory=dict)   # pair -> multiplier, default 1.0
    vol_add: dict[str, float] = field(default_factory=dict)     # pair -> add, default 0.0


def apply_scenario(book: Book, markets: dict[str, MarketState],
                   scenario: StressScenario) -> float:
    """P&L of the whole book under a named scenario, relative to the current
    (unshocked) mark."""
    base_value = reprice_multi_pair_book(book, markets)
    shocked = {
        pair: ShockedMarketState(m, spot_mult=scenario.spot_mult.get(pair, 1.0),
                                 vol_add=scenario.vol_add.get(pair, 0.0))
        for pair, m in markets.items()
    }
    shocked_value = reprice_multi_pair_book(book, shocked)
    return shocked_value - base_value


def standard_scenarios(pairs: list[str]) -> list[StressScenario]:
    """A small, named stress library covering the shapes of move a vol desk
    actually reviews: a broad directional move, a broad vol regime change,
    and a correlation break where pairs move against each other rather than
    together (the case a single-pair or parallel-shock view cannot see)."""
    scenarios = [
        StressScenario("Spot down 5% (all pairs)",
                       spot_mult={p: 0.95 for p in pairs}),
        StressScenario("Spot up 5% (all pairs)",
                       spot_mult={p: 1.05 for p in pairs}),
        StressScenario("Vol up 5 vol points (all pairs)",
                       vol_add={p: 0.05 for p in pairs}),
        StressScenario("Vol down 5 vol points (all pairs)",
                       vol_add={p: -0.05 for p in pairs}),
        StressScenario("Spot down + vol up (risk-off, all pairs)",
                       spot_mult={p: 0.95 for p in pairs},
                       vol_add={p: 0.05 for p in pairs}),
    ]
    if len(pairs) >= 2:
        half = len(pairs) // 2 or 1
        up_pairs, down_pairs = pairs[:half], pairs[half:]
        scenarios.append(StressScenario(
            "Correlation break (first half up, second half down)",
            spot_mult={**{p: 1.03 for p in up_pairs},
                      **{p: 0.97 for p in down_pairs}}))
    return scenarios


@dataclass
class StressReport:
    results: dict[str, float]

    def report(self) -> str:
        width = max((len(name) for name in self.results), default=10)
        lines = ["Stress scenario P&L", "=" * (width + 16)]
        for name, pnl in sorted(self.results.items(), key=lambda kv: kv[1]):
            lines.append(f"{name:<{width}} {pnl:+14,.0f}")
        return "\n".join(lines)


def run_stress_suite(book: Book, markets: dict[str, MarketState],
                     scenarios: list[StressScenario] | None = None
                     ) -> StressReport:
    pairs = list(markets.keys())
    scenarios = scenarios if scenarios is not None else standard_scenarios(pairs)
    results = {s.name: apply_scenario(book, markets, s) for s in scenarios}
    return StressReport(results=results)
