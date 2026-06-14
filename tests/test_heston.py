"""Heston pricing sanity tests."""
import pytest

from fxvol.core.black_scholes import garman_kohlhagen
from fxvol.stochvol.heston import HestonParams, heston_price


def test_heston_put_call_parity():
    S, K, T, r_d, r_f = 1.10, 1.12, 1.0, 0.03, 0.01
    p = HestonParams(v0=0.01, kappa=1.5, theta=0.012, xi=0.3, rho=-0.3)
    c = heston_price(S, K, T, r_d, r_f, p, True)
    put = heston_price(S, K, T, r_d, r_f, p, False)
    import math
    rhs = S * math.exp(-r_f * T) - K * math.exp(-r_d * T)
    assert (c - put) == pytest.approx(rhs, abs=1e-3)


def test_heston_low_volvol_approaches_bs():
    """As xi->0, Heston collapses to BS at vol sqrt(v0)."""
    S, K, T, r_d, r_f = 1.10, 1.10, 1.0, 0.02, 0.01
    p = HestonParams(v0=0.04, kappa=2.0, theta=0.04, xi=1e-3, rho=0.0)
    h = heston_price(S, K, T, r_d, r_f, p, True)
    bs = garman_kohlhagen(S, K, T, r_d, r_f, 0.20, True).price
    assert h == pytest.approx(bs, abs=2e-3)
