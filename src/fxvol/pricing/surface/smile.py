"""Single-tenor smile via the SVI (Stochastic Volatility Inspired) parametrization.

WHY SVI
A smile is 5 noisy points; you need a smooth, arbitrage-controllable curve
through them. Gatheral's SVI parametrizes TOTAL IMPLIED VARIANCE w = sigma^2 * T
as a function of log-moneyness k = ln(K/F):

    w(k) = a + b * ( rho*(k - m) + sqrt((k - m)^2 + s^2) )

Five parameters with clean meanings:
  a : overall variance level        b : wing slope (smile steepness)
  rho : skew (left/right tilt)       m : horizontal shift of the minimum
  s : smoothness near the minimum (how rounded the bottom is)

It's the desk standard because (i) it fits FX/equity smiles well with 5 params
and (ii) there are KNOWN inequalities on (a,b,rho,m,s) guaranteeing no butterfly
arbitrage - checked below in no_butterfly_arb.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares


@dataclass
class SVIParams:
    a: float
    b: float
    rho: float
    m: float
    s: float

    def total_variance(self, k: np.ndarray) -> np.ndarray:
        return self.a + self.b * (
            self.rho * (k - self.m) + np.sqrt((k - self.m) ** 2 + self.s**2)
        )


def fit_svi(k: np.ndarray, total_var: np.ndarray) -> SVIParams:
    """Calibrate SVI to observed (log-moneyness, total-variance) points.

    least_squares with bounds keeps params in the economically sensible region
    (b>0, |rho|<1, s>0). Initial guess: a at the min observed variance, modest
    slope, zero skew - robust enough for 5 well-behaved points.
    """
    def resid(p):
        a, b, rho, m, s = p
        model = a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + s**2))
        return model - total_var

    x0 = [float(np.min(total_var)), 0.1, 0.0, 0.0, 0.1]
    lb = [-1.0, 1e-6, -0.999, -1.0, 1e-6]
    ub = [1.0, 5.0, 0.999, 1.0, 2.0]
    res = least_squares(resid, x0, bounds=(lb, ub), method="trf", max_nfev=10000)
    return SVIParams(*res.x)


def no_butterfly_arb(p: SVIParams, k_grid: np.ndarray) -> bool:
    """Gatheral-Jacquier g(k) >= 0 condition for no butterfly arbitrage.

    If the risk-neutral density implied by the smile ever goes negative, the
    smile admits arbitrage. g(k) is that density up to positive scaling; we
    check it stays non-negative across the grid.
    """
    w = p.total_variance(k_grid)
    dk = k_grid[1] - k_grid[0]
    wp = np.gradient(w, dk)
    wpp = np.gradient(wp, dk)
    g = (1 - 0.5 * k_grid * wp / w) ** 2 - 0.25 * (wp**2) * (1 / w + 0.25) + 0.5 * wpp
    return bool(np.all(g >= -1e-6))
