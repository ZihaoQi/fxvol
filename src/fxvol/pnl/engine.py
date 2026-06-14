"""Greek P&L explain engine.

WHAT THIS IS
The nightly process on every options desk: take yesterday's book and market,
take today's market, and DECOMPOSE the change in book value into the Greek
contributions plus an unexplained residual:

  dPnL ~= delta*dS + 0.5*gamma*dS^2          (spot)
        + vega*dSigma + 0.5*volga*dSigma^2    (vol level + vol convexity)
        + vanna*dS*dSigma                     (spot-vol cross)
        + theta*dt                            (time decay)
        + residual                            (model error / higher order)

WHY IT MATTERS
The residual is the alarm. If it is large, either the risk numbers are wrong or
the model is - both are urgent and both cost money. A clean explain (small
residual) is how a desk gains confidence that its hedges reflect its true risk.
The cross terms (vanna, volga) are non-negotiable: on any day where spot AND vol
both move, leaving them out makes the residual structurally non-zero, especially
on skewed books.

This engine marks each position with the project's own Garman-Kohlhagen Greeks
and the project's own vol surface, so it exercises the whole stack end to end.
"""
from __future__ import annotations

from dataclasses import dataclass

from ..core.black_scholes import gk_greeks
from ..surface.surface import VolSurface
from .book import Book


@dataclass
class MarketState:
    """Market as of one valuation date: spot, rates, and a vol surface."""
    spot: float
    r_dom: float
    r_for: float
    surface: VolSurface

    def vol(self, strike: float, expiry: float) -> float:
        return self.surface.implied_vol(strike, expiry)


@dataclass
class AttributionRow:
    label: str
    actual: float
    delta_pnl: float
    gamma_pnl: float
    vega_pnl: float
    volga_pnl: float
    vanna_pnl: float
    theta_pnl: float
    explained: float
    residual: float


@dataclass
class PnLExplain:
    rows: list[AttributionRow]
    totals: AttributionRow

    def report(self) -> str:
        h = (f"{'position':>14} {'actual':>10} {'delta':>10} {'gamma':>9} "
             f"{'vega':>10} {'volga':>8} {'vanna':>8} {'theta':>9} "
             f"{'explain':>10} {'resid':>9}")
        lines = ["Greek P&L explain", "=" * len(h), h]
        for r in self.rows + [self.totals]:
            if r is self.totals:
                lines.append("-" * len(h))
            lines.append(
                f"{r.label:>14} {r.actual:>10.2f} {r.delta_pnl:>10.2f} "
                f"{r.gamma_pnl:>9.2f} {r.vega_pnl:>10.2f} {r.volga_pnl:>8.2f} "
                f"{r.vanna_pnl:>8.2f} {r.theta_pnl:>9.2f} {r.explained:>10.2f} "
                f"{r.residual:>9.2f}")
        rr = abs(self.totals.residual)
        denom = abs(self.totals.actual) or 1.0
        lines += ["=" * len(h),
                  f"residual / |actual P&L| = {rr / denom:.2%}  "
                  f"(small = clean explain)"]
        return "\n".join(lines)


def explain(book: Book, m0: MarketState, m1: MarketState,
            dt: float = 1 / 365) -> PnLExplain:
    """Decompose book P&L from market state m0 -> m1.

    Greeks are taken at m0 (start-of-day risk), which is the standard desk
    convention: you are explaining today's P&L using the risk you were CARRYING
    coming into today.
    """
    dS = m1.spot - m0.spot
    rows: list[AttributionRow] = []

    agg = dict(actual=0.0, delta=0.0, gamma=0.0, vega=0.0, volga=0.0,
               vanna=0.0, theta=0.0)

    book1 = book.roll_one_day(dt)
    for p0, p1 in zip(book.positions, book1.positions):
        v0 = m0.vol(p0.strike, p0.expiry)
        v1 = m1.vol(p1.strike, p1.expiry)
        dSigma = v1 - v0

        g = gk_greeks(m0.spot, p0.strike, p0.expiry, m0.r_dom, m0.r_for,
                      v0, p0.is_call)
        price0 = g.price
        price1 = gk_greeks(m1.spot, p1.strike, p1.expiry, m1.r_dom, m1.r_for,
                           v1, p1.is_call).price

        n = p0.notional
        actual = (price1 - price0) * n
        delta_pnl = g.delta * dS * n
        gamma_pnl = 0.5 * g.gamma * dS * dS * n
        vega_pnl = g.vega * dSigma * n
        volga_pnl = 0.5 * g.volga * dSigma * dSigma * n
        vanna_pnl = g.vanna * dS * dSigma * n
        theta_pnl = -g.theta * dt * n   # theta is dPrice/dt; decay over dt
        explained = (delta_pnl + gamma_pnl + vega_pnl + volga_pnl
                     + vanna_pnl + theta_pnl)
        residual = actual - explained

        rows.append(AttributionRow(
            label=p0.label or f"{p0.pair}{'C' if p0.is_call else 'P'}",
            actual=actual, delta_pnl=delta_pnl, gamma_pnl=gamma_pnl,
            vega_pnl=vega_pnl, volga_pnl=volga_pnl, vanna_pnl=vanna_pnl,
            theta_pnl=theta_pnl, explained=explained, residual=residual))

        agg["actual"] += actual
        agg["delta"] += delta_pnl
        agg["gamma"] += gamma_pnl
        agg["vega"] += vega_pnl
        agg["volga"] += volga_pnl
        agg["vanna"] += vanna_pnl
        agg["theta"] += theta_pnl

    explained = (agg["delta"] + agg["gamma"] + agg["vega"] + agg["volga"]
                 + agg["vanna"] + agg["theta"])
    totals = AttributionRow(
        label="TOTAL", actual=agg["actual"], delta_pnl=agg["delta"],
        gamma_pnl=agg["gamma"], vega_pnl=agg["vega"], volga_pnl=agg["volga"],
        vanna_pnl=agg["vanna"], theta_pnl=agg["theta"], explained=explained,
        residual=agg["actual"] - explained)
    return PnLExplain(rows=rows, totals=totals)
