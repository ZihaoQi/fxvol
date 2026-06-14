"""Greek P&L explain engine tests."""
from fxvol.core.quotes import SmileQuote
from fxvol.pnl.book import Book, OptionPosition
from fxvol.pnl.engine import MarketState, explain
from fxvol.surface.surface import VolSurface


def _surf(S, bump=0.0):
    q = [SmileQuote(0.25, 0.10 + bump, -0.01, 0.002, -0.018, 0.006),
         SmileQuote(0.5, 0.105 + bump, -0.011, 0.0022, -0.02, 0.0065),
         SmileQuote(1.0, 0.11 + bump, -0.012, 0.0025, -0.022, 0.007)]
    return VolSurface.from_quotes(q, S=S, r_d=0.03, r_f=0.01)


def _book():
    return Book([
        OptionPosition('EURUSD', 1.10, 0.5, True, -10_000_000, 'short 6m ATM C'),
        OptionPosition('EURUSD', 1.13, 0.5, True, 6_000_000, 'long 6m 25dC'),
        OptionPosition('EURUSD', 1.07, 0.25, False, -8_000_000, 'short 3m 25dP'),
        OptionPosition('EURUSD', 1.10, 1.0, True, 12_000_000, 'long 1y ATM C'),
    ])


def test_residual_is_small_on_normal_move():
    """A clean Greek explain reconciles to within a few percent on a 1-day move."""
    m0 = MarketState(1.10, 0.03, 0.01, _surf(1.10))
    m1 = MarketState(1.1044, 0.03, 0.01, _surf(1.1044, bump=0.005))
    res = explain(_book(), m0, m1)
    rel = abs(res.totals.residual) / abs(res.totals.actual)
    assert rel < 0.05  # within 5%


def test_pure_decay_when_nothing_moves():
    """If spot/rates/surface are unchanged, there is no delta or spot P&L.
    (Vega P&L can still be nonzero from rolling DOWN the term structure as
    expiry shortens - a real 'theta of the vol surface' effect, not an error.)"""
    m0 = MarketState(1.10, 0.03, 0.01, _surf(1.10))
    m1 = MarketState(1.10, 0.03, 0.01, _surf(1.10))  # identical market
    res = explain(_book(), m0, m1)
    assert abs(res.totals.delta_pnl) < 1e-6
    assert abs(res.totals.gamma_pnl) < 1e-6
    # residual stays small relative to the decay-driven P&L
    assert abs(res.totals.residual) < abs(res.totals.actual) * 0.5 + 50.0


def test_attribution_sums_to_explained():
    """The six Greek terms must sum to 'explained' exactly (accounting identity)."""
    m0 = MarketState(1.10, 0.03, 0.01, _surf(1.10))
    m1 = MarketState(1.105, 0.03, 0.01, _surf(1.105, bump=0.003))
    res = explain(_book(), m0, m1)
    t = res.totals
    s = (t.delta_pnl + t.gamma_pnl + t.vega_pnl + t.volga_pnl
         + t.vanna_pnl + t.theta_pnl)
    assert abs(s - t.explained) < 1e-6
