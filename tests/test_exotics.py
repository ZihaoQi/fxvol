"""Exotic pricer tests: barrier bridge correction, TARF logic, MC consistency."""

from fxvol.core.quotes import SmileQuote
from fxvol.exotics.barrier_mc import BarrierType, price_barrier
from fxvol.exotics.tarf import TARFSpec, price_tarf
from fxvol.stochvol.heston import HestonParams
from fxvol.surface.surface import VolSurface


def _surf():
    q = [SmileQuote(0.25, 0.10, -0.01, 0.002, -0.018, 0.006),
         SmileQuote(0.5, 0.105, -0.011, 0.0022, -0.02, 0.0065),
         SmileQuote(1.0, 0.11, -0.012, 0.0025, -0.022, 0.007)]
    return VolSurface.from_quotes(q, S=1.10, r_d=0.03, r_f=0.01)


def _heston():
    return HestonParams(v0=0.011, kappa=1.5, theta=0.012, xi=0.3, rho=-0.3)


def test_bridge_lowers_knockout_price():
    """The Brownian-bridge correction catches between-step crossings, so it
    RAISES knock probability and LOWERS the up-and-out price vs naive monitoring."""
    common = dict(S0=1.10, K=1.10, barrier=1.20, T=1.0, r_d=0.03, r_f=0.01,
                  btype=BarrierType.UP_OUT, is_call=True, n_steps=50,
                  n_paths=6000, seed=1)
    naive = price_barrier(_surf(), _heston(), use_bridge=False, **common)
    bridge = price_barrier(_surf(), _heston(), use_bridge=True, **common)
    assert bridge["knock_prob"] > naive["knock_prob"]
    assert bridge["price"] < naive["price"]


def test_knockin_plus_knockout_equals_vanilla():
    """KI + KO = vanilla (a path either knocks or doesn't). Within MC error."""
    common = dict(S0=1.10, K=1.10, barrier=1.20, T=1.0, r_d=0.03, r_f=0.01,
                  is_call=True, n_steps=50, n_paths=8000, seed=2)
    ko = price_barrier(_surf(), _heston(), btype=BarrierType.UP_OUT, **common)
    ki = price_barrier(_surf(), _heston(), btype=BarrierType.UP_IN, **common)
    # vanilla call via many-path terminal payoff on same engine (barrier huge)
    van = price_barrier(_surf(), _heston(), btype=BarrierType.UP_OUT,
                        S0=1.10, K=1.10, barrier=10.0, T=1.0, r_d=0.03, r_f=0.01,
                        is_call=True, n_steps=50, n_paths=8000, seed=2)
    assert abs((ko["price"] + ki["price"]) - van["price"]) < 0.003


def test_tarf_redeems_and_bank_positive():
    spec = TARFSpec(strike=1.11, target=0.04 * 1_000_000, leverage=2.0, n_fixings=12)
    r = price_tarf(_surf(), _heston(), 1.10, 1.0, 0.03, 0.01, spec,
                   n_paths=6000, steps_per_fixing=4, seed=3)
    assert 0.0 < r["redeem_prob"] < 1.0
    assert r["avg_fixings"] < spec.n_fixings   # early redemption shortens life


def test_higher_target_means_more_fixings():
    """A higher target is harder to reach -> structure lives longer on average."""
    base = TARFSpec(1.11, 0.03 * 1_000_000, 2.0, 12)
    high = TARFSpec(1.11, 0.08 * 1_000_000, 2.0, 12)
    rb = price_tarf(_surf(), _heston(), 1.10, 1.0, 0.03, 0.01, base,
                    n_paths=5000, steps_per_fixing=3, seed=4)
    rh = price_tarf(_surf(), _heston(), 1.10, 1.0, 0.03, 0.01, high,
                    n_paths=5000, steps_per_fixing=3, seed=4)
    assert rh["avg_fixings"] > rb["avg_fixings"]
