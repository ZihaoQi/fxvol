"""Local-Stochastic Volatility: the leverage function bridging Heston to exact fit.

THE IDEA
Heston has realistic DYNAMICS but doesn't fit the surface exactly. Local vol
fits EXACTLY but has wrong dynamics. LSV takes the best of both: keep Heston's
stochastic variance, then multiply by a deterministic LEVERAGE function L(S,t)
that "corrects" the fit back to exact. The SDE becomes:

  dS_t = (r_d - r_f) S_t dt + L(S_t, t) sqrt(v_t) S_t dW_S

The calibration identity (Gyongyo / Markovian projection):

  L(S,t)^2 = sigma_Dupire(S,t)^2 / E[ v_t | S_t = S ]

i.e. the leverage squared is the Dupire local variance divided by the
conditional expectation of the stochastic variance given spot. The hard part is
that conditional expectation; the rigorous route is a particle method. As agreed
(time-boxed), we implement the simpler DETERMINISTIC PROJECTION approximation:
approximate E[v_t | S_t] by the unconditional mean reversion path of v_t, which
is exact in the zero-correlation / low-vol-of-vol limit and a sound first cut.
"""
from __future__ import annotations

import numpy as np

from ..localvol.dupire import local_vol
from ..stochvol.heston import HestonParams
from ..surface.surface import VolSurface


def expected_variance_path(p: HestonParams, t: float) -> float:
    """E[v_t] under Heston: theta + (v0 - theta) e^{-kappa t}. Unconditional."""
    return p.theta + (p.v0 - p.theta) * np.exp(-p.kappa * t)


def leverage(surface: VolSurface, p: HestonParams, S: float, t: float) -> float:
    """Leverage L(S,t) under the deterministic-projection approximation.

    L^2 = sigma_Dupire^2 / E[v_t].  In the full model E[v_t | S_t=S] would
    replace E[v_t]; that refinement is the particle-method extension noted above.
    """
    if t <= 0:
        return 1.0
    sig_loc = local_vol(surface, S, t)
    ev = expected_variance_path(p, t)
    return float(sig_loc / np.sqrt(max(ev, 1e-12)))
