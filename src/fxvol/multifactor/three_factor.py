"""Three-factor model: FX spot + Hull-White short rates in each currency.

WHY THREE FACTORS
Long-dated FX options (e.g. PRDC notes, 5-30y) are sensitive to BOTH currencies'
interest rates, not just spot vol. The classic setup makes rates stochastic too:

  dS_t   = (r_d(t) - r_f(t)) S_t dt + sigma_S S_t dW_S
  dr_d   = (theta_d(t) - a_d r_d) dt + sigma_d dW_d     (Hull-White, domestic)
  dr_f   = (theta_f(t) - a_f r_f - rho_fS sigma_f sigma_S) dt + sigma_f dW_f

with a full 3x3 correlation among (W_S, W_d, W_f). The quanto-style drift
adjustment on r_f (the -rho_fS sigma_f sigma_S term) comes from changing measure
to the domestic risk-neutral world - a real effect that makes FX-rates
correlation a FIRST-CLASS risk, exactly what the JD's 'multi-factor' refers to.

We price by Monte Carlo (Euler), correlating the three Brownian increments via
Cholesky. As agreed this is the time-boxed 'reach the top rung' layer: correct
in structure and in the limits, not production-tuned for variance reduction.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ThreeFactorParams:
    sigma_S: float       # FX vol
    a_d: float           # domestic mean reversion
    sigma_d: float       # domestic rate vol
    a_f: float           # foreign mean reversion
    sigma_f: float       # foreign rate vol
    corr: np.ndarray     # 3x3 correlation matrix, order (S, d, f)


def simulate(S0, r_d0, r_f0, T, p: ThreeFactorParams,
             n_paths=20000, n_steps=100, seed=0):
    """Euler MC of (S, r_d, r_f). Returns terminal spot and mean discount factor."""
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    sqrt_dt = np.sqrt(dt)
    L = np.linalg.cholesky(p.corr)

    S = np.full(n_paths, S0)
    rd = np.full(n_paths, r_d0)
    rf = np.full(n_paths, r_f0)
    disc = np.zeros(n_paths)   # integral of r_d dt for domestic discounting

    for _ in range(n_steps):
        z = rng.standard_normal((3, n_paths))
        dW = (L @ z) * sqrt_dt
        disc += rd * dt
        S *= np.exp((rd - rf - 0.5 * p.sigma_S**2) * dt + p.sigma_S * dW[0])
        rd += (-p.a_d * rd) * dt + p.sigma_d * dW[1]
        # quanto drift adjustment on the foreign rate (domestic measure)
        rf += (-p.a_f * rf - p.corr[2, 0] * p.sigma_f * p.sigma_S) * dt \
            + p.sigma_f * dW[2]

    return S, disc


def price_long_dated_call(S0, r_d0, r_f0, K, T, p: ThreeFactorParams,
                          **mc) -> float:
    S_T, disc = simulate(S0, r_d0, r_f0, T, p, **mc)
    payoff = np.maximum(S_T - K, 0.0)
    return float(np.mean(np.exp(-disc) * payoff))
