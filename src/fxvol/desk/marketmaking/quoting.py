"""Two way vol quoting with inventory driven skew.

WHY THIS MODULE EXISTS

A pricer answers one question: what is fair value. A market maker answers a
different one: given what I already hold, how far from fair value do I need to
quote on each side to make the flow that arrives work in my favor rather than
against me.

The mechanism is inventory skew. If a desk is already long vega in a tenor
bucket, it does not want more of the same risk, so it quotes a lower bid (less
willing to buy more vol) and a lower offer (more willing to sell vol to reduce
the position). The skew moves both sides of the market in the same direction,
unlike the bid offer spread, which moves them apart. Two separate knobs, two
separate jobs:

  spread : compensation for taking any risk at all, and for the cost of
           hedging it. Widens with size and with how hard the position is to
           hedge (already large, illiquid tenor, wide surface uncertainty).
  skew   : compensation for taking MORE of the risk you already have. Zero
           when flat, and grows with the signed inventory in the relevant
           bucket.

This module keeps the two separate on purpose, because conflating them (for
example widening the whole market instead of skewing it) fails to reward flow
that flattens the book and fails to penalize flow that concentrates it.
"""
from __future__ import annotations

from dataclasses import dataclass

from fxvol.pricing.core.black_scholes import gk_greeks

from ..pnl.book import Book
from ..pnl.engine import MarketLike
from ..risk.grid import bucket_for


@dataclass(frozen=True)
class Quote:
    """A two way vol market at one strike and tenor."""
    pair: str
    strike: float
    expiry: float
    is_call: bool
    bid_vol: float
    ask_vol: float
    fair_vol: float

    @property
    def mid_vol(self) -> float:
        return 0.5 * (self.bid_vol + self.ask_vol)

    @property
    def half_spread(self) -> float:
        return 0.5 * (self.ask_vol - self.bid_vol)


@dataclass
class InventorySkewModel:
    """Maps current bucketed inventory to a vol skew and a size dependent spread.

    base_half_spread_vol : half spread quoted at zero size and zero inventory,
        in absolute vol points (0.001 = 10 bp of vol).
    size_spread_slope    : extra half spread per unit of notional traded,
        relative to `size_reference_notional`. Captures that bigger clips are
        harder to hedge cleanly and therefore cost more.
    size_reference_notional : the notional at which size_spread_slope applies
        at full strength (a normalization constant, not a hard cap).
    skew_per_vega : vol points of skew per unit of vega already held in the
        relevant tenor bucket, expressed per `vega_reference`. This is the
        single most important number in the model: too small and the desk
        never lays off risk through its own pricing; too large and it prices
        itself out of every trade once it holds any position at all.
    vega_reference : normalizes skew_per_vega so it can be quoted as a round
        number regardless of the notional scale of the book.
    """
    base_half_spread_vol: float = 0.0015
    size_spread_slope: float = 0.0020
    size_reference_notional: float = 5_000_000.0
    skew_per_vega: float = 0.00006
    vega_reference: float = 100_000.0

    def half_spread(self, notional: float) -> float:
        size_factor = abs(notional) / self.size_reference_notional
        return self.base_half_spread_vol + self.size_spread_slope * size_factor

    def skew(self, bucket_vega: float) -> float:
        """Vol points to shift BOTH sides by. Positive inventory (long vega)
        pushes the skew negative, so the desk quotes lower vol on both the bid
        and the offer, discouraging further buying of vol and encouraging
        selling it back."""
        return -self.skew_per_vega * (bucket_vega / self.vega_reference)


@dataclass
class QuoteEngine:
    """Produces two way vol quotes off a fair value surface, adjusted for the
    inventory the desk is already carrying."""
    skew_model: InventorySkewModel

    def bucket_vega(self, book: Book, market: MarketLike, pair: str,
                    expiry: float) -> float:
        """Signed vega the desk already holds in the tenor bucket that
        `expiry` falls into, for the given pair. Long book vega means the desk
        will skew AWAY from buying more of that risk."""
        target_bucket = bucket_for(expiry)
        total = 0.0
        for p in book.positions:
            if p.pair != pair:
                continue
            if bucket_for(p.expiry) != target_bucket:
                continue
            vol = market.vol(p.strike, p.expiry)
            g = gk_greeks(market.spot, p.strike, p.expiry, market.r_dom,
                         market.r_for, vol, p.is_call)
            total += g.vega * p.notional
        return total

    def quote(self, book: Book, market: MarketLike, pair: str, strike: float,
              expiry: float, is_call: bool, notional: float) -> Quote:
        fair = market.vol(strike, expiry)
        bucket_vega = self.bucket_vega(book, market, pair, expiry)
        skew = self.skew_model.skew(bucket_vega)
        half = self.skew_model.half_spread(notional)
        mid = fair + skew
        return Quote(pair=pair, strike=strike, expiry=expiry, is_call=is_call,
                    bid_vol=mid - half, ask_vol=mid + half, fair_vol=fair)
