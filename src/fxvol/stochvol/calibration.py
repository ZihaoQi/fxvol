"""Calibrate Heston to a market vol surface.

THE LOSS
We minimise squared error between Heston model vols and the surface's implied
vols across a grid of (strike, tenor). We fit in IMPLIED-VOL space, not price
space, because vega differs wildly across strikes - matching prices would
over-weight ATM and ignore the wings that carry the skew/convexity info.

scipy.optimize.least_squares (trust-region) does the work, as agreed. The art
is in the bounds and the starting point: v0 and theta near ATM variance, kappa
moderate, rho negative (typical FX skew), xi giving realistic convexity.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares

from ..core.black_scholes import garman_kohlhagen
from ..surface.surface import VolSurface
from .heston import HestonParams, heston_price


def _implied_vol_from_price(price, S, K, T, r_d, r_f, is_call) -> float:
    from scipy.optimize import brentq

    def diff(sig):
        return garman_kohlhagen(S, K, T, r_d, r_f, sig, is_call).price - price

    try:
        return brentq(diff, 1e-4, 5.0, xtol=1e-8)
    except ValueError:
        return np.nan


def calibrate_heston(surface: VolSurface,
                     strikes_per_tenor: int = 5) -> HestonParams:
    S, r_d, r_f = surface.spot, surface.r_dom, surface.r_for

    targets = []  # (K, T, market_vol)
    for T in surface.tenors:
        F = S * np.exp((r_d - r_f) * T)
        for k in np.linspace(-0.15, 0.15, strikes_per_tenor):
            K = F * np.exp(k)
            targets.append((K, T, surface.implied_vol(K, T)))

    def resid(x):
        p = HestonParams(v0=x[0], kappa=x[1], theta=x[2], xi=x[3], rho=x[4])
        out = []
        for K, T, mkt_vol in targets:
            price = heston_price(S, K, T, r_d, r_f, p, is_call=True)
            model_vol = _implied_vol_from_price(price, S, K, T, r_d, r_f, True)
            out.append((model_vol - mkt_vol) if np.isfinite(model_vol) else 1.0)
        return out

    atm_var = surface.implied_vol(S * np.exp((r_d - r_f) * surface.tenors[0]),
                                  surface.tenors[0]) ** 2
    x0 = [atm_var, 1.5, atm_var, 0.4, -0.3]
    lb = [1e-4, 0.1, 1e-4, 1e-2, -0.95]
    ub = [1.0, 10.0, 1.0, 2.0, 0.95]
    res = least_squares(resid, x0, bounds=(lb, ub), method="trf", max_nfev=200)
    return HestonParams(v0=res.x[0], kappa=res.x[1], theta=res.x[2],
                        xi=res.x[3], rho=res.x[4])
