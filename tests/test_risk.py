"""Risk grid, bucketed vega, and limit monitor tests."""
from fxvol.core.quotes import SmileQuote
from fxvol.pnl.book import Book, OptionPosition
from fxvol.pnl.engine import MarketState
from fxvol.risk.grid import (
    Limit, LimitMonitor, bucketed_vega, build_risk_grid,
)
from fxvol.surface.surface import VolSurface


def _market():
    q = [SmileQuote(0.25, 0.10, -0.01, 0.002, -0.018, 0.006),
         SmileQuote(0.5, 0.105, -0.011, 0.0022, -0.02, 0.0065),
         SmileQuote(1.0, 0.11, -0.012, 0.0025, -0.022, 0.007)]
    return MarketState(1.10, 0.03, 0.01,
                       VolSurface.from_quotes(q, S=1.10, r_d=0.03, r_f=0.01))


def _book():
    return Book([
        OptionPosition('EURUSD', 1.10, 0.5, True, -10_000_000, 'short 6m ATM C'),
        OptionPosition('EURUSD', 1.13, 0.5, True, 6_000_000, 'long 6m 25dC'),
        OptionPosition('EURUSD', 1.07, 0.25, False, -8_000_000, 'short 3m 25dP'),
        OptionPosition('EURUSD', 1.10, 1.0, True, 12_000_000, 'long 1y ATM C'),
    ])


def test_grid_center_is_zero():
    """The unshocked cell (0 spot, 0 vol) must be exactly zero P&L by construction."""
    grid = build_risk_grid(_book(), _market(), spot_steps=5, vol_steps=5)
    mid_i, mid_j = 2, 2  # center of a 5x5 grid spanning +/- range symmetrically
    assert abs(grid.pnl[mid_i, mid_j]) < 1e-6


def test_bucketed_vega_assigns_by_tenor():
    """Each position's vega lands in exactly one tenor bucket; 3m/6m/1y populated."""
    vb = bucketed_vega(_book(), _market())
    assert vb["3m"] != 0.0
    assert vb["6m"] != 0.0
    assert vb["1y"] != 0.0
    assert vb["1w"] == 0.0  # nothing that short-dated in the book


def test_limit_breach_detection():
    mon = LimitMonitor([
        Limit('net vega', 'vega', 300_000),
        Limit('tiny delta', 'delta', 1.0),  # absurdly tight -> must breach
    ])
    breaches = mon.check(_book(), _market())
    names = {b.limit.name for b in breaches}
    assert 'tiny delta' in names


def test_no_breach_when_limits_loose():
    mon = LimitMonitor([Limit('huge', 'vega', 1e12)])
    assert mon.check(_book(), _market()) == []
