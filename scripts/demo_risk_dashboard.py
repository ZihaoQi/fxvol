"""Cross pair risk dashboard demo: aggregate a multi-currency options book by
pair and tenor bucket, flag concentration, and run a named stress scenario
suite.

Run:  python scripts/demo_risk_dashboard.py
"""
from fxvol.desk.pnl.book import Book, OptionPosition
from fxvol.desk.pnl.engine import MarketState
from fxvol.desk.risk.dashboard import (aggregate_by_pair, check_concentration,
                                  run_stress_suite)
from fxvol.data.synthetic.g10 import build_surface


def main() -> None:
    pairs = ["EURUSD", "USDJPY", "GBPUSD"]
    markets: dict[str, MarketState] = {}
    for pair in pairs:
        s = build_surface(pair)
        markets[pair] = MarketState(spot=s.spot, r_dom=s.r_dom, r_for=s.r_for,
                                    surface=s)

    # Deliberately built to LOOK EURUSD-heavy by notional. The point of the
    # dashboard is that vega, not notional, is what concentration should be
    # measured in, and the two disagree here: USDJPY's much higher spot level
    # means the same style of position carries far more vega per unit of
    # notional, so the real concentration turns out to be in USDJPY.
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

    print("Risk by pair")
    print("=" * 60)
    by_pair = aggregate_by_pair(book, markets)
    for pair, g in by_pair.items():
        print(f"{pair:>7}  delta {g.delta:+14,.0f}  gamma {g.gamma:+12,.0f}"
             f"  vega {g.vega:+14,.0f}")
    print("Note: EURUSD carries the largest notional in this book, but")
    print("USDJPY's higher spot level means the same style of position")
    print("carries far more vega per unit notional, which is exactly why")
    print("risk is tracked in vega, not in notional.")

    print("\nConcentration check")
    print("=" * 60)
    flags = check_concentration(book, markets, pair_threshold=0.6,
                                bucket_threshold=0.6)
    if flags:
        for f in flags:
            print(f"  {f}")
    else:
        print("  no concentration above threshold")

    print("\n" + run_stress_suite(book, markets).report())


if __name__ == "__main__":
    main()
