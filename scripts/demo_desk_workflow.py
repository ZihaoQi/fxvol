"""End-to-end desk workflow demo: surface -> P&L explain -> risk grid + limits.

Run:  python scripts/demo_desk_workflow.py
Shows the three production pieces working together on one option book.
"""
from fxvol.core.quotes import SmileQuote
from fxvol.pnl.book import Book, OptionPosition
from fxvol.pnl.engine import MarketState, explain
from fxvol.risk.grid import Limit, LimitMonitor, bucketed_vega, build_risk_grid
from fxvol.surface.surface import VolSurface


def _surface(S, bump=0.0):
    q = [SmileQuote(0.25, 0.10 + bump, -0.01, 0.002, -0.018, 0.006),
         SmileQuote(0.5, 0.105 + bump, -0.011, 0.0022, -0.02, 0.0065),
         SmileQuote(1.0, 0.11 + bump, -0.012, 0.0025, -0.022, 0.007)]
    return VolSurface.from_quotes(q, S=S, r_d=0.03, r_f=0.01)


def main() -> None:
    book = Book([
        OptionPosition('EURUSD', 1.10, 0.5, True, -10_000_000, 'short 6m ATM C'),
        OptionPosition('EURUSD', 1.13, 0.5, True, 6_000_000, 'long 6m 25dC'),
        OptionPosition('EURUSD', 1.07, 0.25, False, -8_000_000, 'short 3m 25dP'),
        OptionPosition('EURUSD', 1.10, 1.0, True, 12_000_000, 'long 1y ATM C'),
    ])

    m0 = MarketState(1.10, 0.03, 0.01, _surface(1.10))
    m1 = MarketState(1.1044, 0.03, 0.01, _surface(1.1044, bump=0.005))

    print(explain(book, m0, m1).report())
    print("\n" + build_risk_grid(book, m0).report())
    print("\nBucketed vega (per 1.00 vol):")
    for k, v in bucketed_vega(book, m0).items():
        if abs(v) > 1e-6:
            print(f"  {k:>3}: {v:+,.0f}")

    mon = LimitMonitor([
        Limit('net delta', 'delta', 5_000_000),
        Limit('net vega', 'vega', 300_000),
        Limit('6m vega bucket', 'vega_bucket', 150_000, bucket='6m'),
        Limit('grid worst-case', 'grid_worst', 400_000),
    ])
    print("\n" + mon.report(book, m0))


if __name__ == "__main__":
    main()
