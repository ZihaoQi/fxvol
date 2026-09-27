"""Market making demo: quote a book under inventory skew, feed it a stream of
client orders (some toxic), and see the P&L split into spread capture versus
adverse selection.

Run:  python scripts/demo_market_making.py
"""
from fxvol.desk.marketmaking.flow import FlowConfig
from fxvol.desk.marketmaking.quoting import InventorySkewModel, QuoteEngine
from fxvol.desk.marketmaking.simulator import MarketMakingSimulator
from fxvol.desk.pnl.engine import MarketState
from fxvol.data.synthetic.g10 import build_surface


def main() -> None:
    surface = build_surface("EURUSD")
    market = MarketState(spot=surface.spot, r_dom=surface.r_dom,
                         r_for=surface.r_for, surface=surface)

    flow_cfg = FlowConfig(
        pair="EURUSD",
        strikes=[market.spot * m for m in (0.97, 1.0, 1.03)],
        expiries=[1 / 12, 0.25, 0.5],
        toxic_fraction=0.15,
    )

    print("Skewed engine (skew_per_vega = 0.00006, the package default)")
    print("=" * 60)
    skewed = MarketMakingSimulator(
        pair="EURUSD",
        quote_engine=QuoteEngine(InventorySkewModel()),
        base_market=market, flow_cfg=flow_cfg)
    skewed_result = skewed.run(n_orders=500, seed=42)
    print(skewed_result.report())

    print("\nNaive engine for comparison (skew_per_vega = 0, spread only)")
    print("=" * 60)
    naive = MarketMakingSimulator(
        pair="EURUSD",
        quote_engine=QuoteEngine(InventorySkewModel(skew_per_vega=0.0)),
        base_market=market, flow_cfg=flow_cfg)
    naive_result = naive.run(n_orders=500, seed=42)
    print(naive_result.report())

    skewed_net = sum(p.notional for p in skewed_result.final_book.positions)
    naive_net = sum(p.notional for p in naive_result.final_book.positions)
    print("\nNet notional carried at the end of the run (same order flow):")
    print(f"  skewed engine : {skewed_net:+,.0f}")
    print(f"  naive engine  : {naive_net:+,.0f}")
    print("Same order flow into both engines. Skewing away from an already")
    print("large position is what keeps the net position smaller here, not")
    print("a change in the flow itself.")


if __name__ == "__main__":
    main()
