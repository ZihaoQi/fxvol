"""Shared market-shocking and book valuation helpers.

WHY THIS MODULE EXISTS

Two different parts of the desk toolkit need to move a market forward without
refitting a whole new VolSurface: the market making simulator moves the
market after every client order (desk.marketmaking.simulator), and the risk
dashboard moves the market to test named stress scenarios
(desk.risk.dashboard). Both used to carry their own copy of a "shocked
market" wrapper and their own book revaluation loop. That kind of duplication
is exactly what silently drifts over time, for example a bug fix lands in one
copy and not the other. This module is the single implementation both depend
on now.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from fxvol.pricing.core.black_scholes import gk_greeks

from .book import Book
from .engine import MarketLike, MarketState


@dataclass
class ShockedMarketState:
    """A MarketState with a spot multiplier and an additive vol shock layered
    on top. Duck types as a MarketLike (the same spot / r_dom / r_for /
    vol(strike, expiry) interface as MarketState), so it can be passed
    anywhere a MarketLike is expected without needing to rebuild a whole
    VolSurface for every shock applied.
    """
    base: MarketState
    spot_mult: float = 1.0
    vol_add: float = 0.0

    @property
    def spot(self) -> float:
        return self.base.spot * self.spot_mult

    @property
    def r_dom(self) -> float:
        return self.base.r_dom

    @property
    def r_for(self) -> float:
        return self.base.r_for

    def vol(self, strike: float, expiry: float) -> float:
        return max(self.base.vol(strike, expiry) + self.vol_add, 1e-4)


def reprice_book(book: Book, market: MarketLike) -> float:
    """Total fair value of a book where every position is priced against the
    SAME market: the single-pair case used by the market making simulator and
    by a stress scenario run one pair at a time."""
    total = 0.0
    for p in book.positions:
        vol = market.vol(p.strike, p.expiry)
        price = gk_greeks(market.spot, p.strike, p.expiry, market.r_dom,
                          market.r_for, vol, p.is_call).price
        total += price * p.notional
    return total


def reprice_multi_pair_book(book: Book, markets: Mapping[str, MarketLike]) -> float:
    """Total fair value of a book spanning several pairs, each priced against
    its own market: the cross-pair risk dashboard case.

    Simplifying assumption, kept consistent with `engine.explain`: each
    position's price times notional is already expressed in the desk's
    reporting currency, so P&L across pairs is summed directly rather than
    converted through a cross rate.
    """
    total = 0.0
    for p in book.positions:
        m = markets[p.pair]
        vol = m.vol(p.strike, p.expiry)
        price = gk_greeks(m.spot, p.strike, p.expiry, m.r_dom, m.r_for, vol,
                          p.is_call).price
        total += price * p.notional
    return total
