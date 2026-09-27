"""Tests for the market making package: quoting, flow, and the simulator.

The point of these tests is not just "does it run" but "does the mechanism
do the economic thing it claims to do": skew should move the market in
inventory reducing direction, spread should widen with size, and skew should
measurably reduce the net risk a desk ends up carrying versus a desk that
quotes with no skew at all.
"""
from __future__ import annotations

from fxvol.desk.marketmaking.flow import FlowConfig, generate_orders, toxic_vol_shock
from fxvol.desk.marketmaking.quoting import InventorySkewModel, QuoteEngine
from fxvol.desk.marketmaking.simulator import MarketMakingSimulator, fill_probability
from fxvol.desk.pnl.book import Book, OptionPosition
from fxvol.desk.pnl.engine import MarketState
from fxvol.data.synthetic.g10 import build_surface


def _market() -> MarketState:
    surface = build_surface("EURUSD")
    return MarketState(spot=surface.spot, r_dom=surface.r_dom,
                       r_for=surface.r_for, surface=surface)


def test_flat_book_has_zero_skew():
    engine = QuoteEngine(InventorySkewModel())
    m = _market()
    q = engine.quote(Book([]), m, "EURUSD", strike=m.spot, expiry=0.25,
                     is_call=True, notional=1_000_000)
    assert abs(q.mid_vol - q.fair_vol) < 1e-12


def test_long_vega_skews_market_down():
    """Holding long vega in a bucket should push both the bid and the offer
    below fair value, since the desk wants less of that risk, not more."""
    engine = QuoteEngine(InventorySkewModel())
    m = _market()
    long_book = Book([OptionPosition("EURUSD", strike=m.spot, expiry=0.2,
                                     is_call=True, notional=20_000_000)])
    q = engine.quote(long_book, m, "EURUSD", strike=m.spot, expiry=0.25,
                     is_call=True, notional=1_000_000)
    assert q.bid_vol < q.fair_vol
    assert q.ask_vol < q.fair_vol


def test_spread_widens_with_size():
    model = InventorySkewModel()
    assert model.half_spread(10_000_000) > model.half_spread(100_000)


def test_toxic_shock_direction_matches_client_side():
    buy = next(o for o in generate_orders(
        FlowConfig("EURUSD", [1.08], [0.25], toxic_fraction=1.0), 5, seed=1)
        if o.direction > 0)
    sell = next(o for o in generate_orders(
        FlowConfig("EURUSD", [1.08], [0.25], toxic_fraction=1.0), 5, seed=1)
        if o.direction < 0)
    assert toxic_vol_shock(buy) > 0
    assert toxic_vol_shock(sell) < 0


def test_fill_probability_falls_with_deviation():
    """A desk that is already long vega should quote a LOWER bid (discouraging
    a client from selling it more vol, which would make the desk longer
    still), so a client sell order should fill less often against a long book
    than against a flat one.
    """
    m = _market()
    engine = QuoteEngine(InventorySkewModel())
    long_book = Book([OptionPosition("EURUSD", strike=m.spot, expiry=0.2,
                                     is_call=True, notional=50_000_000)])
    order_cfg_strike = m.spot
    from fxvol.desk.marketmaking.flow import ClientOrder
    sell_order = ClientOrder("EURUSD", order_cfg_strike, 0.25, True,
                             direction=-1, notional=1_000_000, is_toxic=False)
    p_flat = fill_probability(
        engine.quote(Book([]), m, "EURUSD", order_cfg_strike, 0.25, True,
                    1_000_000),
        sell_order)
    p_long = fill_probability(
        engine.quote(long_book, m, "EURUSD", order_cfg_strike, 0.25, True,
                    1_000_000),
        sell_order)
    assert p_long < p_flat


def test_zero_noise_zero_toxicity_is_pure_spread_capture():
    m = _market()
    cfg = FlowConfig("EURUSD", strikes=[m.spot * 0.98, m.spot, m.spot * 1.02],
                     expiries=[1 / 12, 0.25, 0.5], toxic_fraction=0.0)
    sim = MarketMakingSimulator(pair="EURUSD",
                                quote_engine=QuoteEngine(InventorySkewModel()),
                                base_market=m, flow_cfg=cfg)
    result = sim.run(n_orders=200, seed=3, toxic_shock=0.004,
                     benign_vol_noise_std=0.0)
    assert result.n_toxic == 0
    assert abs(result.inventory_pnl) < 1.0        # no vol moves -> no mark-to-market P&L
    assert result.spread_pnl > 0                  # spreads should pay on average


def test_skew_reduces_net_inventory_versus_no_skew():
    """The whole point of inventory skew is that it is self correcting: a
    desk that skews its market should end up carrying less absolute risk than
    one that quotes a flat spread with no skew at all, given the same order
    flow.
    """
    m = _market()
    cfg = FlowConfig("EURUSD", strikes=[m.spot], expiries=[0.25],
                     toxic_fraction=0.2)

    skewed = MarketMakingSimulator(
        pair="EURUSD",
        quote_engine=QuoteEngine(InventorySkewModel(skew_per_vega=0.0004)),
        base_market=m, flow_cfg=cfg)
    naive = MarketMakingSimulator(
        pair="EURUSD",
        quote_engine=QuoteEngine(InventorySkewModel(skew_per_vega=0.0)),
        base_market=m, flow_cfg=cfg)

    skewed_result = skewed.run(n_orders=400, seed=7)
    naive_result = naive.run(n_orders=400, seed=7)

    def net_notional(book: Book) -> float:
        return sum(p.notional for p in book.positions)

    assert abs(net_notional(skewed_result.final_book)) < \
        abs(net_notional(naive_result.final_book))
