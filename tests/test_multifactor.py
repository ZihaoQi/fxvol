"""Three-factor MC sanity: zero rate-vol limit recovers GK price."""
import numpy as np
import pytest

from fxvol.core.black_scholes import garman_kohlhagen
from fxvol.multifactor.three_factor import ThreeFactorParams, price_long_dated_call


def test_zero_ratevol_recovers_gk():
    corr = np.eye(3)
    p = ThreeFactorParams(sigma_S=0.12, a_d=0.01, sigma_d=1e-8,
                          a_f=0.01, sigma_f=1e-8, corr=corr)
    S0, r_d0, r_f0, K, T = 1.10, 0.0, 0.0, 1.10, 1.0
    mc = price_long_dated_call(S0, r_d0, r_f0, K, T, p,
                               n_paths=60000, n_steps=50, seed=1)
    gk = garman_kohlhagen(S0, K, T, 0.0, 0.0, 0.12, True).price
    assert mc == pytest.approx(gk, abs=5e-3)
