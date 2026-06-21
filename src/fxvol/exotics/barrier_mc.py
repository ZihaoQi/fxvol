"""Monte Carlo exotic pricer on the LSV layer, with Brownian-bridge barriers.

WHY THIS MODULE
The vol surface prices vanillas. Exotics depend on the PATH spot takes, so they
need a dynamic model. We simulate Local-Stochastic Volatility paths (Heston
variance x leverage function) and price path-dependent payoffs on them.

THE BARRIER SUBTLETY (the teachable hard part)
Discrete time steps MISS barrier touches that happen BETWEEN steps. If spot is
1.09 at step i and 1.11 at step i+1 with a knock-out at 1.10, naive monitoring
never sees the crossing - so you systematically UNDER-knock and OVERPRICE
knock-outs. The Brownian-bridge correction analytically computes the probability
that the continuous path crossed the barrier between the two simulated points
and applies it. This is a classic exotic-desk correctness issue.

For a step from S_i to S_{i+1} over dt with local vol sigma, the probability of
NOT crossing an upper barrier B (both points below B) is:
    p_survive = 1 - exp( -2 * ln(B/S_i) * ln(B/S_{i+1}) / (sigma^2 * dt) )
(the analogous expression holds for a lower barrier). We knock out a path with
probability (1 - p_survive) even when neither endpoint breached.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from ..lsv.leverage import leverage
from ..stochvol.heston import HestonParams
from ..surface.surface import VolSurface


class BarrierType(Enum):
    UP_OUT = "up_out"
    DOWN_OUT = "down_out"
    UP_IN = "up_in"
    DOWN_IN = "down_in"


@dataclass
class LSVPaths:
    spot: np.ndarray      # (n_paths, n_steps+1)
    var: np.ndarray       # (n_paths, n_steps+1) variance
    dt: float
    times: np.ndarray


def simulate_lsv(
    surface: VolSurface, p: HestonParams, S0: float, T: float,
    r_d: float, r_f: float, n_steps: int = 100, n_paths: int = 20000,
    seed: int = 0, leverage_grid: int = 40,
) -> LSVPaths:
    """Simulate LSV paths (vectorized across paths).

    Variance follows Heston; spot follows dS = (r_d-r_f)S dt + L(S,t)*sqrt(v)*S dW
    with the two Brownians correlated by rho. The leverage L(S,t) is precomputed
    on a (time, spot) grid and interpolated, since calling the Dupire-based
    leverage per path per step would be far too slow.
    """
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    sqrt_dt = np.sqrt(dt)
    times = np.linspace(0, T, n_steps + 1)

    # Precompute leverage on a grid: spans a few vols around the forward each step
    s_lo, s_hi = S0 * 0.5, S0 * 1.8
    s_grid = np.linspace(s_lo, s_hi, leverage_grid)
    lev_grid = np.zeros((n_steps + 1, leverage_grid))
    for i, t in enumerate(times):
        if t <= 1e-6:
            lev_grid[i, :] = 1.0
            continue
        for j, s in enumerate(s_grid):
            try:
                lev_grid[i, j] = leverage(surface, p, s, t)
            except Exception:
                lev_grid[i, j] = 1.0
    lev_grid = np.clip(lev_grid, 0.1, 5.0)

    def lev_at(i, S):
        return np.interp(S, s_grid, lev_grid[i])

    S = np.full(n_paths, S0)
    v = np.full(n_paths, p.v0)
    spot = np.zeros((n_paths, n_steps + 1))
    var = np.zeros((n_paths, n_steps + 1))
    spot[:, 0] = S
    var[:, 0] = v

    for i in range(n_steps):
        z1 = rng.standard_normal(n_paths)
        z2 = rng.standard_normal(n_paths)
        dW_S = z1 * sqrt_dt
        dW_v = (p.rho * z1 + np.sqrt(1 - p.rho**2) * z2) * sqrt_dt

        L = lev_at(i, S)
        vol = L * np.sqrt(np.maximum(v, 0))
        S = S * np.exp((r_d - r_f - 0.5 * vol**2) * dt + vol * dW_S)
        # Heston variance, full-truncation Euler (keep v >= 0)
        v = v + p.kappa * (p.theta - np.maximum(v, 0)) * dt \
            + p.xi * np.sqrt(np.maximum(v, 0)) * dW_v
        v = np.maximum(v, 0)

        spot[:, i + 1] = S
        var[:, i + 1] = v

    return LSVPaths(spot=spot, var=var, dt=dt, times=times)


def _bridge_survival(S0, S1, barrier, vol, dt, upper: bool):
    """Prob. the continuous path did NOT cross `barrier` between S0 and S1.

    Both endpoints assumed on the surviving side. Brownian-bridge formula in
    log-space with local vol `vol`. Returns array of survival probabilities.
    """
    var = (vol**2) * dt
    var = np.maximum(var, 1e-16)
    if upper:
        # distance to barrier in log space (positive when below barrier)
        a = np.log(barrier / S0)
        b = np.log(barrier / S1)
    else:
        a = np.log(S0 / barrier)
        b = np.log(S1 / barrier)
    # crossing prob = exp(-2 a b / var) when both a,b > 0; else already crossed
    crossed_endpoint = (a <= 0) | (b <= 0)
    # clamp the exponent argument to avoid overflow when var is tiny; a*b>0 here
    expo = np.clip(-2.0 * a * b / var, -700.0, 0.0)
    p_cross = np.where(crossed_endpoint, 1.0, np.exp(expo))
    return 1.0 - p_cross


def price_barrier(
    surface: VolSurface, p: HestonParams, S0: float, K: float, barrier: float,
    T: float, r_d: float, r_f: float, btype: BarrierType,
    is_call: bool = True, rebate: float = 0.0,
    n_steps: int = 100, n_paths: int = 20000, seed: int = 0,
    use_bridge: bool = True,
) -> dict:
    """Price a single-barrier option under LSV by Monte Carlo.

    Returns price plus the knock probability and standard error. `use_bridge`
    toggles the Brownian-bridge correction so you can SEE its effect (turning it
    off overprices knock-outs - a good demonstration).
    """
    paths = simulate_lsv(surface, p, S0, T, r_d, r_f, n_steps, n_paths, seed)
    spot = paths.spot
    dt = paths.dt
    upper = btype in (BarrierType.UP_OUT, BarrierType.UP_IN)

    # survival probability per path (product over steps of per-step survival)
    surv = np.ones(spot.shape[0])
    for i in range(spot.shape[1] - 1):
        S0i, S1i = spot[:, i], spot[:, i + 1]
        # discrete monitoring: knocked if either endpoint breached
        if upper:
            breached = (S0i >= barrier) | (S1i >= barrier)
        else:
            breached = (S0i <= barrier) | (S1i <= barrier)
        if use_bridge:
            vol_i = np.sqrt(np.maximum(paths.var[:, i], 1e-12))
            step_surv = _bridge_survival(S0i, S1i, barrier, vol_i, dt, upper)
            step_surv = np.where(breached, 0.0, step_surv)
        else:
            step_surv = np.where(breached, 0.0, 1.0)
        surv *= step_surv

    knocked = 1.0 - surv
    ST = spot[:, -1]
    payoff = np.maximum(ST - K, 0.0) if is_call else np.maximum(K - ST, 0.0)

    if btype in (BarrierType.UP_OUT, BarrierType.DOWN_OUT):
        # survive -> get payoff; knock -> get rebate
        val = surv * payoff + knocked * rebate
    else:  # knock-IN: payoff only if knocked in
        val = knocked * payoff

    disc = np.exp(-r_d * T)
    pv = disc * val
    return {
        "price": float(np.mean(pv)),
        "stderr": float(np.std(pv) / np.sqrt(len(pv))),
        "knock_prob": float(np.mean(knocked)),
    }
