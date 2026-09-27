"""Tests for the shared market-shocking and book valuation helpers, used by
both the market making simulator and the cross-pair risk dashboard.
"""
from __future__ import annotations

from fxvol.desk.pnl.book import Book, OptionPosition
from fxvol.desk.pnl.engine import MarketState
from fxvol.desk.pnl.valuation import (ShockedMarketState, reprice_book,
                                      reprice_multi_pair_book)
from fxvol.data.synthetic.g10 import build_surface


def _market(pair: str) -> MarketState:
    s = build_surface(pair)
    return MarketState(spot=s.spot, r_dom=s.r_dom, r_for=s.r_for, surface=s)


def test_shocked_market_applies_spot_and_vol_shift():
    m = _market("EURUSD")
    shocked = ShockedMarketState(m, spot_mult=1.05, vol_add=0.02)
    assert shocked.spot == m.spot * 1.05
    assert shocked.r_dom == m.r_dom
    assert shocked.r_for == m.r_for
    assert abs(shocked.vol(m.spot, 0.25) - (m.vol(m.spot, 0.25) + 0.02)) < 1e-12


def test_shocked_market_floors_vol_at_a_positive_minimum():
    m = _market("EURUSD")
    shocked = ShockedMarketState(m, vol_add=-10.0)   # absurd shock on purpose
    assert shocked.vol(m.spot, 0.25) > 0.0


def test_reprice_book_matches_hand_computed_value():
    m = _market("EURUSD")
    book = Book([OptionPosition("EURUSD", m.spot, 0.25, True, 1_000_000)])
    from fxvol.pricing.core.black_scholes import gk_greeks
    vol = m.vol(m.spot, 0.25)
    expected = gk_greeks(m.spot, m.spot, 0.25, m.r_dom, m.r_for, vol, True).price
    assert abs(reprice_book(book, m) - expected * 1_000_000) < 1e-6


def test_reprice_multi_pair_book_sums_across_pairs():
    markets = {"EURUSD": _market("EURUSD"), "USDJPY": _market("USDJPY")}
    book = Book([
        OptionPosition("EURUSD", markets["EURUSD"].spot, 0.25, True, 1_000_000),
        OptionPosition("USDJPY", markets["USDJPY"].spot, 0.25, True, 1_000_000),
    ])
    single_pair_books = [
        Book([p]) for p in book.positions
    ]
    expected = sum(reprice_book(b, markets[b.positions[0].pair])
                  for b in single_pair_books)
    assert abs(reprice_multi_pair_book(book, markets) - expected) < 1e-6
