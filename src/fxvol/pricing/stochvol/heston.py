"""Heston stochastic volatility: characteristic function + Fourier pricing.

THE MODEL
Variance v_t is itself stochastic, mean-reverting:
  dS_t = (r_d - r_f) S_t dt + sqrt(v_t) S_t dW_S
  dv_t = kappa (theta - v_t) dt + xi sqrt(v_t) dW_v,   d<W_S, W_v> = rho dt

Five params:
  kappa : mean-reversion speed     theta : long-run variance
  xi    : vol-of-vol (smile convexity)   rho : spot/vol corr (skew)
  v0    : initial variance

WHY A CHARACTERISTIC FUNCTION
Heston has no closed-form density, but its CHARACTERISTIC FUNCTION is known in
closed form. Carr-Madan / Gatheral pricing integrates the CF to get option
prices. We use the "little Heston trap" formulation of the CF, which keeps the
complex logarithm on the principal branch and avoids the discontinuities that
plague the original 1993 formula. This branch issue is the single most common
Heston-implementation bug - hence the explicit comment.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import quad


@dataclass
class HestonParams:
    v0: float
    kappa: float
    theta: float
    xi: float
    rho: float

    def feller_ok(self) -> bool:
        """2*kappa*theta > xi^2 keeps variance strictly positive (Feller)."""
        return 2 * self.kappa * self.theta > self.xi**2


def _cf(u, T, r_d, r_f, p: HestonParams):
    """Heston characteristic function (little Heston trap form)."""
    xi, kappa, theta, rho, v0 = p.xi, p.kappa, p.theta, p.rho, p.v0
    xi2 = xi * xi
    beta = kappa - rho * xi * 1j * u
    d = np.sqrt(beta**2 + xi2 * (1j * u + u**2))
    g = (beta - d) / (beta + d)          # trap form: (beta - d)/(beta + d)
    exp_dt = np.exp(-d * T)
    C = (r_d - r_f) * 1j * u * T + (kappa * theta / xi2) * (
        (beta - d) * T - 2.0 * np.log((1 - g * exp_dt) / (1 - g))
    )
    D = ((beta - d) / xi2) * ((1 - exp_dt) / (1 - g * exp_dt))
    return np.exp(C + D * v0)


def heston_price(S, K, T, r_d, r_f, p: HestonParams, is_call=True) -> float:
    """European vanilla via the Gatheral two-integral (P1, P2) representation.

    Priced on the FORWARD measure: we evaluate the characteristic function with
    zero rates (so it carries only the stochastic-vol dynamics around the
    forward), work in log-moneyness ln(K/F), and discount exactly once at the
    end with exp(-r_d*T). Putting the (r_d - r_f) drift inside the CF AND
    discounting spot/strike separately double-counts the drift - the classic
    Heston pricing bug.
    """
    F = S * np.exp((r_d - r_f) * T)
    ln_m = np.log(K / F)

    def cf0(u):
        return _cf(u, T, 0.0, 0.0, p)

    def integrand(u, j):
        if j == 1:
            cf = cf0(u - 1j) / cf0(-1j)
        else:
            cf = cf0(u)
        return np.real(np.exp(-1j * u * ln_m) * cf / (1j * u))

    P1 = 0.5 + (1 / np.pi) * quad(lambda u: integrand(u, 1), 1e-8, 200, limit=200)[0]
    P2 = 0.5 + (1 / np.pi) * quad(lambda u: integrand(u, 2), 1e-8, 200, limit=200)[0]

    call = np.exp(-r_d * T) * (F * P1 - K * P2)
    if is_call:
        return float(call)
    # put-call parity
    return float(call - S * np.exp(-r_f * T) + K * np.exp(-r_d * T))
