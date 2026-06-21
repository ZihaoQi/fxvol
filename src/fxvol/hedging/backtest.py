"""Delta-hedging strategy comparison: the economics of hedging discipline.

THE QUESTION THIS ANSWERS
You sold an option. You must delta-hedge it to expiry. HOW OFTEN you re-hedge
is a real optimization:
  - hedge continuously  -> you bleed transaction costs (spread paid every trade)
  - hedge rarely        -> you carry gamma risk (delta drifts, P&L gets noisy)
The right answer is a NO-TRADE BAND around target delta whose width depends on
the ratio of transaction cost to gamma. This module simulates spot paths and
compares hedging rules on P&L *and its variance net of costs* - the efficient
frontier of hedging discipline.

KEY RESULT TO REPRODUCE (Whalley-Wilmott asymptotics)
Optimal band half-width ~ (1.5 * cost * S * gamma_cash / risk_aversion)^(1/3).
Long-gamma books want WIDE bands (gamma works for you); short-gamma books need
TIGHT bands (every move hurts). This is the cube-root law - derive it empirically
here, don't just cite it.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from ..core.black_scholes import gk_greeks


class HedgeRule(Enum):
    FIXED_TIME = "fixed_time"        # re-hedge every N steps
    FIXED_BAND = "fixed_band"        # re-hedge when |delta drift| > fixed threshold
    WW_BAND = "ww_band"              # Whalley-Wilmott gamma-scaled band


@dataclass
class HedgeConfig:
    rule: HedgeRule
    # FIXED_TIME: hedge every `time_steps` steps
    time_steps: int = 1
    # FIXED_BAND: hedge when delta moves more than `band` (in delta units)
    band: float = 0.05
    # WW_BAND: risk-aversion (higher -> tighter bands); cost in spot terms
    risk_aversion: float = 1.0


@dataclass
class HedgeResult:
    final_pnl: np.ndarray            # P&L per path
    n_hedges: np.ndarray             # number of re-hedges per path
    txn_cost: np.ndarray             # total transaction cost per path

    @property
    def mean_pnl(self) -> float:
        return float(np.mean(self.final_pnl))

    @property
    def pnl_std(self) -> float:
        return float(np.std(self.final_pnl))

    @property
    def mean_cost(self) -> float:
        return float(np.mean(self.txn_cost))


def simulate_paths(S0: float, sigma: float, r_d: float, r_f: float,
                   T: float, n_steps: int, n_paths: int,
                   seed: int = 0) -> np.ndarray:
    """GBM spot paths under the real-world drift = r_d - r_f (FX forward drift)."""
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    drift = (r_d - r_f - 0.5 * sigma**2) * dt
    diff = sigma * np.sqrt(dt)
    shocks = rng.standard_normal((n_paths, n_steps))
    log_paths = np.cumsum(drift + diff * shocks, axis=1)
    S = S0 * np.exp(log_paths)
    return np.hstack([np.full((n_paths, 1), S0), S])   # include S0 column


def backtest(
    cfg: HedgeConfig,
    S0: float, K: float, T: float, r_d: float, r_f: float, sigma: float,
    is_call: bool = True, notional: float = 1_000_000,
    cost_bps: float = 0.5, n_steps: int = 252, n_paths: int = 5000,
    seed: int = 0,
) -> HedgeResult:
    """Sell one option, delta-hedge to expiry under `cfg`, across many paths.

    P&L = -(option payout) + (premium received) + (hedge P&L) - (txn costs).
    We track the SHORT option position (market-maker convention). The hedge is
    long delta*notional of spot, rebalanced per the rule. Transaction cost is
    cost_bps of the notional traded at each rebalance.
    """
    dt = T / n_steps
    paths = simulate_paths(S0, sigma, r_d, r_f, T, n_steps, n_paths, seed)
    cost_frac = cost_bps / 1e4

    premium = gk_greeks(S0, K, T, r_d, r_f, sigma, is_call).price * notional

    final_pnl = np.zeros(n_paths)
    n_hedges = np.zeros(n_paths)
    txn_cost = np.zeros(n_paths)

    for p in range(n_paths):
        path = paths[p]
        cash = premium               # we received the premium
        hedge_units = 0.0            # current spot holding (in notional units)

        for i in range(n_steps):
            t_rem = T - i * dt
            if t_rem <= 0:
                break
            S = path[i]
            g = gk_greeks(S, K, T - i * dt, r_d, r_f, sigma, is_call)
            # target hedge: long g.delta * notional of spot to offset short option
            target = g.delta * notional

            do_hedge = False
            if cfg.rule == HedgeRule.FIXED_TIME:
                do_hedge = (i % cfg.time_steps == 0)
            elif cfg.rule == HedgeRule.FIXED_BAND:
                drift = abs(target - hedge_units)
                do_hedge = drift > cfg.band * notional
            elif cfg.rule == HedgeRule.WW_BAND:
                # band half-width from WW cube-root law, in delta(notional) units
                gamma_cash = abs(g.gamma) * notional * S
                hw = (1.5 * cost_frac * S * gamma_cash /
                      max(cfg.risk_aversion, 1e-9)) ** (1 / 3)
                do_hedge = abs(target - hedge_units) > hw

            if do_hedge or i == 0:
                trade = target - hedge_units
                cost = abs(trade) * S * cost_frac
                cash -= trade * S          # buy `trade` units of spot
                cash -= cost
                txn_cost[p] += cost
                hedge_units = target
                n_hedges[p] += 1

        # settle at expiry
        ST = path[-1]
        payout = max(ST - K, 0.0) if is_call else max(K - ST, 0.0)
        cash -= payout * notional          # we are short the option
        cash += hedge_units * ST           # liquidate hedge
        final_pnl[p] = cash

    return HedgeResult(final_pnl=final_pnl, n_hedges=n_hedges, txn_cost=txn_cost)
