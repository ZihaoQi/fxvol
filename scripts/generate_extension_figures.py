"""Generate the three README figures for the market making, risk dashboard,
and surface maintenance additions.

Run:  python scripts/generate_extension_figures.py
Writes into docs/img/, alongside the existing figures.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from fxvol.desk.marketmaking.flow import FlowConfig
from fxvol.desk.marketmaking.quoting import InventorySkewModel, QuoteEngine
from fxvol.desk.marketmaking.simulator import MarketMakingSimulator
from fxvol.desk.pnl.book import Book, OptionPosition
from fxvol.desk.pnl.engine import MarketState
from fxvol.desk.risk.dashboard import run_stress_suite
from fxvol.pricing.surface.maintenance import RecalibrationPipeline
from fxvol.data.synthetic.g10 import build_surface, simulate_quote_history

ROOT = Path(__file__).resolve().parents[1]
IMG_DIR = ROOT / "docs" / "img"

NAVY = "#1f3b57"
TEAL = "#1f9e89"
CORAL = "#d1495b"
GREY = "#8c8c8c"

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#444444",
    "axes.grid": True,
    "grid.color": "#e6e6e6",
    "grid.linewidth": 0.8,
    "font.size": 10,
})


def market_making_figure() -> None:
    surface = build_surface("EURUSD")
    market = MarketState(spot=surface.spot, r_dom=surface.r_dom,
                         r_for=surface.r_for, surface=surface)
    flow_cfg = FlowConfig(
        pair="EURUSD",
        strikes=[market.spot * m for m in (0.97, 1.0, 1.03)],
        expiries=[1 / 12, 0.25, 0.5], toxic_fraction=0.15)

    sim = MarketMakingSimulator(pair="EURUSD",
                                quote_engine=QuoteEngine(InventorySkewModel()),
                                base_market=market, flow_cfg=flow_cfg)
    result = sim.run(n_orders=500, seed=42)

    steps = [h["step"] for h in result.history]
    cum_spread = np.cumsum([h["spread_pnl"] for h in result.history])
    cum_inventory = np.cumsum([h["inventory_pnl"] for h in result.history])
    cum_total = cum_spread + cum_inventory

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(steps, cum_spread, color=TEAL, lw=2, label="Spread P&L (cumulative)")
    ax.plot(steps, cum_inventory, color=CORAL, lw=2,
           label="Inventory P&L (cumulative)")
    ax.plot(steps, cum_total, color=NAVY, lw=2.5, label="Total P&L (cumulative)")
    ax.axhline(0, color="#444444", lw=0.8)
    ax.set_xlabel("Order sequence")
    ax.set_ylabel("Cumulative P&L (USD)")
    ax.set_title("Market making: spread capture vs. adverse selection")
    ax.legend(loc="upper left", frameon=False)
    fig.tight_layout()
    fig.savefig(IMG_DIR / "market_making_pnl.png", dpi=150)
    plt.close(fig)


def stress_scenario_figure() -> None:
    pairs = ["EURUSD", "USDJPY", "GBPUSD"]
    markets = {}
    for pair in pairs:
        s = build_surface(pair)
        markets[pair] = MarketState(spot=s.spot, r_dom=s.r_dom, r_for=s.r_for,
                                    surface=s)
    book = Book([
        OptionPosition("EURUSD", markets["EURUSD"].spot, 1 / 12, True,
                       40_000_000, "short-dated EURUSD call"),
        OptionPosition("EURUSD", markets["EURUSD"].spot * 1.01, 0.25, True,
                       25_000_000, "3m EURUSD call"),
        OptionPosition("USDJPY", markets["USDJPY"].spot, 0.5, False,
                       6_000_000, "6m USDJPY put"),
        OptionPosition("GBPUSD", markets["GBPUSD"].spot * 0.98, 1.0, False,
                       4_000_000, "1y GBPUSD put"),
    ])
    report = run_stress_suite(book, markets)
    items = sorted(report.results.items(), key=lambda kv: kv[1])
    names = [name for name, _ in items]
    values = [v / 1e6 for _, v in items]
    colors = [CORAL if v < 0 else TEAL for v in values]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.barh(names, values, color=colors)
    ax.axvline(0, color="#444444", lw=0.8)
    ax.set_xlabel("Scenario P&L (USD millions)")
    ax.set_title("Stress scenario suite across a 3-pair book")
    fig.tight_layout()
    fig.savefig(IMG_DIR / "stress_scenarios.png", dpi=150)
    plt.close(fig)


def surface_health_figure() -> None:
    history = simulate_quote_history(
        "EURUSD", n_days=30, seed=11, daily_noise_std=0.0010,
        freeze_days={14: 0}, outlier_days={21: (2, 0.06)})
    result = RecalibrationPipeline().run(history)

    days = [r.day for r in result.records]
    rmse = [r.fit_rmse_bp for r in result.records]
    clean = [r.is_clean for r in result.records]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(days, rmse, color=NAVY, lw=1.5, zorder=1)
    clean_days = [d for d, c in zip(days, clean) if c]
    clean_rmse = [r for r, c in zip(rmse, clean) if c]
    flagged_days = [d for d, c in zip(days, clean) if not c]
    flagged_rmse = [r for r, c in zip(rmse, clean) if not c]
    ax.scatter(clean_days, clean_rmse, color=TEAL, s=35, zorder=2, label="Clean day")
    ax.scatter(flagged_days, flagged_rmse, color=CORAL, s=55, zorder=3,
              marker="x", label="Flagged (stale quote or arbitrage break)")
    ax.set_xlabel("Day")
    ax.set_ylabel("Fit RMSE (vol bp)")
    ax.set_title("Surface maintenance: fit quality with a frozen quote (day 14)\n"
                "and a bad print (day 21) injected")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.32), ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(IMG_DIR / "surface_health.png", dpi=150)
    plt.close(fig)


def main() -> None:
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    market_making_figure()
    stress_scenario_figure()
    surface_health_figure()
    print(f"Wrote figures to {IMG_DIR}")


if __name__ == "__main__":
    main()
