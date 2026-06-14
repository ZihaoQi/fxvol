"""Delta-convention round-trip tests."""

import pytest

from fxvol.core.black_scholes import garman_kohlhagen
from fxvol.core.conventions import DeltaType, strike_from_delta, forward


def test_strike_from_delta_roundtrip():
    """A strike recovered from a 0.25 spot-delta should reprice to ~0.25 delta."""
    S, T, r_d, r_f, sigma = 1.10, 0.5, 0.03, 0.01, 0.10
    K = strike_from_delta(0.25, S, T, r_d, r_f, sigma, True, DeltaType.SPOT)
    delta = garman_kohlhagen(S, K, T, r_d, r_f, sigma, True).delta
    assert delta == pytest.approx(0.25, abs=1e-3)


def test_atm_dns_pa_below_forward():
    """Premium-adjusted DNS (USDCNH convention) sits BELOW the forward."""
    from fxvol.core.conventions import ATMConvention, atm_strike
    S, T, r_d, r_f, sigma = 1.10, 1.0, 0.03, 0.01, 0.20
    k_pa = atm_strike(S, T, r_d, r_f, sigma, ATMConvention.DNS, premium_adjusted=True)
    k_npa = atm_strike(S, T, r_d, r_f, sigma, ATMConvention.DNS, premium_adjusted=False)
    F = forward(S, T, r_d, r_f)
    assert k_pa < F < k_npa
