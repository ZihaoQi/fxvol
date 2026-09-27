"""Client order flow, including a toxicity signal.

WHY TOXICITY MATTERS

Not all flow is the same. A pension fund rolling a hedge on a schedule is
"benign" flow: uncorrelated with what happens to the market five minutes
later. A client who tends to buy vol right before it jumps is "toxic" flow:
filling that order is adverse selection, not a random walk. A market maker
that cannot tell the two apart ends up with symmetric inventory skew that
reacts the same way to both, when in reality the profitable move is to widen
or refuse toxic flow rather than just skew around it.

This module generates a stream of orders where a fraction are toxic in a
simple, testable sense: a toxic BUY order is generated in the same simulated
step where the underlying vol is about to jump up, and a toxic SELL order
precedes a vol drop. The simulator does not see the toxicity label when it
prices; it only sees the label afterward when computing whether the flow it
absorbed turned out to be adverse. That separation (generate with foresight,
price without it) is what makes the resulting P&L split meaningful rather than
circular.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ClientOrder:
    pair: str
    strike: float
    expiry: float
    is_call: bool
    direction: int          # +1 = client buys the option, -1 = client sells
    notional: float         # always positive; direction carries the sign
    is_toxic: bool          # ground truth, hidden from the pricing engine


@dataclass
class FlowConfig:
    pair: str
    strikes: list[float]
    expiries: list[float]
    toxic_fraction: float = 0.15
    min_notional: float = 250_000.0
    max_notional: float = 3_000_000.0


def generate_orders(cfg: FlowConfig, n_orders: int,
                    seed: int = 0) -> list[ClientOrder]:
    """A stream of n_orders client orders, toxic_fraction of them toxic."""
    rng = np.random.default_rng(seed)
    orders: list[ClientOrder] = []
    for _ in range(n_orders):
        strike = float(rng.choice(cfg.strikes))
        expiry = float(rng.choice(cfg.expiries))
        is_call = bool(rng.random() < 0.5)
        direction = 1 if rng.random() < 0.5 else -1
        notional = float(rng.uniform(cfg.min_notional, cfg.max_notional))
        is_toxic = bool(rng.random() < cfg.toxic_fraction)
        orders.append(ClientOrder(cfg.pair, strike, expiry, is_call, direction,
                                  notional, is_toxic))
    return orders


def toxic_vol_shock(order: ClientOrder, base_shock: float = 0.004) -> float:
    """The vol move that follows a toxic order, in the direction that hurts
    whoever took the other side. A client BUY of vol that is toxic precedes a
    vol rise (the market maker, now short vol, gets hurt); a toxic SELL
    precedes a vol fall (the market maker, now long vol, gets hurt). Benign
    orders carry no signal and return 0.0 here; the simulator adds ordinary
    noise around this separately.
    """
    if not order.is_toxic:
        return 0.0
    return base_shock if order.direction > 0 else -base_shock
