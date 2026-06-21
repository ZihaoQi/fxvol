"""TARF (Target Redemption Forward) pricing under LSV by Monte Carlo.

WHAT A TARF IS
A schedule of (usually weekly/monthly) fixings. On each fixing the CLIENT
typically buys the foreign currency at a favourable strike vs spot - but with
LEVERAGE on the unfavourable side. Crucially the structure carries a TARGET:
once the client's CUMULATIVE gain reaches the target, the whole structure
KNOCKS OUT (redeems). So the client's upside is capped at the target while the
downside (the leveraged leg) is open until redemption.

WHY IT'S PATH-DEPENDENT IN A NASTY WAY
The state at each fixing includes "how much has accumulated so far", because that
determines whether the target is hit. So you must carry the accumulated amount
along each path - it's an extra state variable. There is no clean PDE; Monte
Carlo over the fixing schedule is standard.

THE BANK'S RISK (when the bank SELLS the TARF to a client)
The bank's worst case is the path where the client NEVER hits the target and
keeps accumulating on the leveraged downside. The target acts as a knock-out on
the bank's *short-gain* exposure but NOT on the client's leveraged downside -
an asymmetry that makes the TARF's Greeks sharp and path-sensitive.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..stochvol.heston import HestonParams
from ..surface.surface import VolSurface
from .barrier_mc import simulate_lsv


@dataclass
class TARFSpec:
    strike: float            # client buys foreign ccy at this strike each fixing
    target: float            # cumulative client gain that triggers redemption
    leverage: float          # multiplier on the unfavourable (client-loss) leg
    n_fixings: int           # number of fixing dates
    notional: float = 1_000_000   # per-fixing notional (foreign ccy)
    # client is long the structure: gains when spot < strike (buys cheap),
    # loses leveraged when spot > strike. (A common USDCNH-style accumulator.)


def price_tarf(
    surface: VolSurface, p: HestonParams, S0: float, T: float,
    r_d: float, r_f: float, spec: TARFSpec,
    n_paths: int = 20000, steps_per_fixing: int = 5, seed: int = 0,
) -> dict:
    """Price a TARF from the CLIENT's perspective (bank value = -client value).

    Returns client PV, expected number of fixings before redemption, and the
    probability the target is reached (structure redeems early).
    """
    n_steps = spec.n_fixings * steps_per_fixing
    paths = simulate_lsv(surface, p, S0, T, r_d, r_f, n_steps, n_paths, seed)
    spot = paths.spot
    # fixing indices along the path
    fix_idx = [(k + 1) * steps_per_fixing for k in range(spec.n_fixings)]
    dt_fix = T / spec.n_fixings

    n = spot.shape[0]
    accumulated = np.zeros(n)        # cumulative client gain
    alive = np.ones(n, dtype=bool)   # not yet redeemed
    pv = np.zeros(n)
    n_fix_done = np.zeros(n)
    redeemed = np.zeros(n, dtype=bool)

    for k, idx in enumerate(fix_idx):
        Sfix = spot[:, idx]
        t = (k + 1) * dt_fix
        disc = np.exp(-r_d * t)

        # client payoff this fixing (only for still-alive paths)
        gain = (spec.strike - Sfix) * spec.notional          # >0 if spot below strike
        # leverage applies on the loss side
        leg = np.where(gain >= 0, gain, gain * spec.leverage)

        active = alive.copy()
        pv += np.where(active, disc * leg, 0.0)
        accumulated += np.where(active & (gain > 0), gain, 0.0)
        n_fix_done += active.astype(float)

        # check redemption: cumulative POSITIVE gain hits target
        hit = active & (accumulated >= spec.target)
        redeemed |= hit
        alive &= ~hit   # redeemed paths take no further fixings

    return {
        "client_pv": float(np.mean(pv)),
        "bank_pv": float(-np.mean(pv)),
        "redeem_prob": float(np.mean(redeemed)),
        "avg_fixings": float(np.mean(n_fix_done)),
        "stderr": float(np.std(pv) / np.sqrt(n)),
    }
