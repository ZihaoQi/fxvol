"""Run a market making book through a stream of client orders and split the
resulting P&L into spread capture versus adverse selection.

WHY THE SPLIT MATTERS

A market maker who only looks at total P&L cannot tell a good week from a
lucky one. Two P&L sources look identical in the total but mean opposite
things for how the desk should adjust its pricing:

  spread P&L     : locked in at the moment of the trade, the gap between the
                   quoted price and fair value at that instant. This is the
                   compensation the desk is SUPPOSED to earn for making a
                   market. It should be positive on average by construction
                   and its size tells you whether your spreads are wide
                   enough for the flow you are seeing.
  inventory P&L   : the mark to market change of everything the desk is
                   carrying, realized after the trade, as the market moves.
                   This is the part that toxic flow attacks: a client who
                   buys right before a jump hands the desk a position that
                   loses money as soon as it moves. A desk that is skewing
                   correctly should see this component shrink as inventory
                   skew improves, even while spread P&L stays roughly flat.

If total P&L is healthy only because spread P&L is large while inventory P&L
is a persistent drag, the fix is not "widen everything," it is "figure out
which flow is toxic and skew or spread specifically against it." This module
exists to make that distinction visible and measurable rather than a
qualitative desk narrative.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from fxvol.pricing.core.black_scholes import gk_greeks

from ..pnl.book import Book, OptionPosition
from ..pnl.engine import MarketState
from ..pnl.valuation import ShockedMarketState, reprice_book
from .flow import ClientOrder, FlowConfig, generate_orders, toxic_vol_shock
from .quoting import QuoteEngine


@dataclass
class SimulationResult:
    orders: list[ClientOrder]
    final_book: Book
    spread_pnl: float
    inventory_pnl: float
    n_toxic: int
    n_filled: int
    n_declined: int
    history: list[dict] = field(default_factory=list)

    @property
    def total_pnl(self) -> float:
        return self.spread_pnl + self.inventory_pnl

    def report(self) -> str:
        n = len(self.orders)
        fill_rate = self.n_filled / n if n else 0.0
        lines = [
            "Market making simulation",
            "=" * 40,
            f"orders seen         : {n}",
            f"orders filled       : {self.n_filled} ({fill_rate:.1%} fill rate, "
            f"{self.n_toxic} toxic)",
            f"orders declined     : {self.n_declined} (client walked, quote too far from fair)",
            f"spread P&L          : {self.spread_pnl:+,.0f}",
            f"inventory P&L       : {self.inventory_pnl:+,.0f}",
            "-" * 40,
            f"total P&L           : {self.total_pnl:+,.0f}",
            f"open positions      : {len(self.final_book.positions)}",
        ]
        return "\n".join(lines)


def fill_probability(quote, order: ClientOrder, max_deviation: float = 0.01) -> float:
    """Probability the client accepts the quoted price, given how far it sits
    from fair value on the side the client is trading.

    A skewed or wide quote does not just change the price a fill happens at,
    it changes whether a fill happens at all: a real client walks away from a
    price that has moved too far from where they think fair value is. This is
    the mechanism that makes inventory skew actually reduce risk rather than
    just repricing the same risk. Linear falloff to zero at max_deviation vol
    points away from fair; clipped to [0, 1].
    """
    if order.direction > 0:                 # client buys -> looks at the ask
        deviation = quote.ask_vol - quote.fair_vol
    else:                                    # client sells -> looks at the bid
        deviation = quote.fair_vol - quote.bid_vol
    return float(np.clip(1.0 - deviation / max_deviation, 0.0, 1.0))


@dataclass
class MarketMakingSimulator:
    pair: str
    quote_engine: QuoteEngine
    base_market: MarketState
    flow_cfg: FlowConfig

    def run(self, n_orders: int, seed: int = 0, toxic_shock: float = 0.004,
            benign_vol_noise_std: float = 0.0008,
            max_fill_deviation: float = 0.01) -> SimulationResult:
        """Fill n_orders client orders in sequence, re-quoting after every
        fill so later quotes reflect the inventory just taken on.

        toxic_shock            : vol move (absolute) that follows a toxic
            order, applied in the direction that hurts the side the desk just
            took (see flow.toxic_vol_shock).
        benign_vol_noise_std   : ordinary background vol noise applied after
            every order regardless of toxicity, so inventory P&L is not
            purely a toxicity detector but also reflects everyday carry risk.
        """
        rng = np.random.default_rng(seed + 1)   # separate stream from order flow's own seed
        orders = generate_orders(self.flow_cfg, n_orders, seed=seed)

        book = Book([])
        vol_offset = 0.0
        spread_pnl_total = 0.0
        inventory_pnl_total = 0.0
        n_toxic = 0
        n_filled = 0
        n_declined = 0
        history: list[dict] = []

        for i, order in enumerate(orders):
            market_before = ShockedMarketState(self.base_market, vol_add=vol_offset)
            quote = self.quote_engine.quote(book, market_before, self.pair,
                                            order.strike, order.expiry,
                                            order.is_call, order.notional)

            if order.direction > 0:            # client buys -> desk sells, at the ask
                trade_vol, mm_sign = quote.ask_vol, -1
            else:                               # client sells -> desk buys, at the bid
                trade_vol, mm_sign = quote.bid_vol, 1

            filled = bool(rng.random() < fill_probability(
                quote, order, max_deviation=max_fill_deviation))

            spread_pnl = 0.0
            if filled:
                fair_price = gk_greeks(market_before.spot, order.strike,
                                       order.expiry, market_before.r_dom,
                                       market_before.r_for, quote.fair_vol,
                                       order.is_call).price
                trade_price = gk_greeks(market_before.spot, order.strike,
                                        order.expiry, market_before.r_dom,
                                        market_before.r_for, trade_vol,
                                        order.is_call).price
                spread_pnl = mm_sign * order.notional * (fair_price - trade_price)
                spread_pnl_total += spread_pnl

                new_position = OptionPosition(
                    self.pair, order.strike, order.expiry, order.is_call,
                    mm_sign * order.notional, label=f"order{i}")
                book = Book(book.positions + [new_position])
                n_filled += 1
                n_toxic += int(order.is_toxic)
            else:
                n_declined += 1

            # The vol move that follows a toxic order is a market wide event,
            # not something this desk caused by filling or declining the
            # order, so it hits the EXISTING book either way. Declining a
            # toxic order avoids taking ON new toxic risk, but does not undo
            # the desk's prior inventory being exposed to the same move.
            value_before = reprice_book(book, market_before)
            shock = toxic_vol_shock(order, base_shock=toxic_shock) \
                + float(rng.normal(0.0, benign_vol_noise_std))
            vol_offset += shock
            market_after = ShockedMarketState(self.base_market, vol_add=vol_offset)
            value_after = reprice_book(book, market_after)

            inv_pnl = value_after - value_before
            inventory_pnl_total += inv_pnl

            history.append(dict(
                step=i, filled=filled, is_toxic=order.is_toxic,
                spread_pnl=spread_pnl, inventory_pnl=inv_pnl,
                vol_offset=vol_offset,
                cum_pnl=spread_pnl_total + inventory_pnl_total))

        return SimulationResult(orders=orders, final_book=book,
                                spread_pnl=spread_pnl_total,
                                inventory_pnl=inventory_pnl_total,
                                n_toxic=n_toxic, n_filled=n_filled,
                                n_declined=n_declined, history=history)
