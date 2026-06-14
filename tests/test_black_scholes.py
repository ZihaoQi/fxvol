"""Tests for Garman-Kohlhagen. These define 'correct' - make them green.

Run with:  pytest tests/test_black_scholes.py -v

Why these specific tests? Each one pins down a property that MUST hold for any
correct implementation, independent of the exact numbers:
  - put-call parity is a no-arbitrage identity; if it fails, the model is wrong
  - delta bounds catch sign/factor errors
  - the zero-rate case collapses to plain Black-Scholes, a known reference
"""
import math

import pytest

from fxvol.core.black_scholes import garman_kohlhagen


def test_put_call_parity():
    """C - P must equal S*exp(-r_f*T) - K*exp(-r_d*T). A no-arb identity."""
    S, K, T, r_d, r_f, sigma = 1.10, 1.12, 0.5, 0.03, 0.01, 0.10
    call = garman_kohlhagen(S, K, T, r_d, r_f, sigma, is_call=True).price
    put = garman_kohlhagen(S, K, T, r_d, r_f, sigma, is_call=False).price
    lhs = call - put
    rhs = S * math.exp(-r_f * T) - K * math.exp(-r_d * T)
    assert lhs == pytest.approx(rhs, abs=1e-10)


def test_call_delta_bounds():
    """A call's spot delta lives in (0, exp(-r_f*T)), not (0, 1)."""
    res = garman_kohlhagen(1.10, 1.10, 1.0, 0.02, 0.01, 0.12, is_call=True)
    assert 0.0 < res.delta < math.exp(-0.01 * 1.0)


def test_collapses_to_black_scholes():
    """With r_f = 0, GK is just Black-Scholes on a non-dividend asset.
    ATM-forward 1y option, known closed form ~ S*(2*N(sigma*sqrt(T)/2) - 1)
    when discounting is stripped out. We check the call is positive & sane."""
    S, K, T, r_d, r_f, sigma = 100.0, 100.0, 1.0, 0.0, 0.0, 0.20
    res = garman_kohlhagen(S, K, T, r_d, r_f, sigma, is_call=True)
    # Approx ATM BS price: S * sigma * sqrt(T) / sqrt(2*pi)
    approx = S * sigma * math.sqrt(T) / math.sqrt(2 * math.pi)
    assert res.price == pytest.approx(approx, rel=0.02)


def test_vega_positive():
    res = garman_kohlhagen(1.10, 1.15, 0.75, 0.02, 0.005, 0.11)
    assert res.vega > 0
