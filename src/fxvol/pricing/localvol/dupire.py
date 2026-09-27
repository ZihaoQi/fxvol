"""Dupire local volatility extracted from the implied vol surface.

THE KEY INSIGHT (and the lesson)
Local vol sigma_loc(K,T) is the UNIQUE deterministic vol function that reprices
every European vanilla on the surface exactly. Dupire's formula gives it from
the surface's derivatives. We use the implied-total-variance form (numerically
far kinder than the call-price form):

  sigma_loc^2 = (dw/dT) /
       [ 1 - (k/w)*(dw/dk) + 0.25*(-0.25 - 1/w + k^2/w^2)*(dw/dk)^2 + 0.5*d2w/dk2 ]

where w = total implied variance, k = log-moneyness.

WHY IT MATTERS (the thing to understand, not just run):
Local vol fits today's surface perfectly but predicts that the smile FLATTENS
as spot moves - real smiles don't. That wrong forward smile dynamic is exactly
why exotic desks moved to stochastic and local-stochastic vol. Building this
yourself is how you feel that limitation rather than just reading about it.
"""
from __future__ import annotations

import numpy as np

from ..surface.surface import VolSurface


def local_vol(surface: VolSurface, K: float, T: float,
              dT: float = 1e-4, dk: float = 1e-4) -> float:
    """Local vol at (K, T) by finite-differencing the surface in (k, T)."""
    from ..core.conventions import forward
    F = forward(surface.spot, T, surface.r_dom, surface.r_for)
    k = np.log(K / F)

    def w_of(k_val: float, t_val: float) -> float:
        K_val = F * np.exp(k_val)
        return surface.total_variance(K_val, t_val)

    w = w_of(k, T)
    dw_dT = (w_of(k, T + dT) - w_of(k, T - dT)) / (2 * dT)
    dw_dk = (w_of(k + dk, T) - w_of(k - dk, T)) / (2 * dk)
    d2w_dk2 = (w_of(k + dk, T) - 2 * w + w_of(k - dk, T)) / (dk**2)

    denom = (1.0
             - (k / w) * dw_dk
             + 0.25 * (-0.25 - 1.0 / w + k**2 / w**2) * dw_dk**2
             + 0.5 * d2w_dk2)
    if denom <= 0:
        raise ValueError("Non-positive Dupire denominator: surface has arbitrage")
    return float(np.sqrt(max(dw_dT, 1e-12) / denom))
