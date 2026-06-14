"""Regenerate the visual report (presentation/fxvol_report.html) from live outputs.

Runs the validation, P&L engine, and risk grid, captures the real numbers, and
writes a self-contained interactive HTML report. Re-run after any model change
to refresh the figures.
"""
import json
from pathlib import Path

import numpy as np

from fxvol.core.black_scholes import gk_greeks
from fxvol.core.quotes import SmileQuote
from fxvol.pnl.book import Book, OptionPosition
from fxvol.pnl.engine import MarketState, explain
from fxvol.risk.grid import Limit, LimitMonitor, bucketed_vega, build_risk_grid
from fxvol.surface.surface import VolSurface

ROOT = Path(__file__).resolve().parents[1]


def validation_data():
    import sys
    sys.path.insert(0, str(ROOT / "scripts"))
    from run_bloomberg_real import (  # noqa: E402
        TENORS, WITHIN_10D, build_surface, load_target, rates_for,
    )
    target = load_target(ROOT / "data" / "bloomberg")
    surf = build_surface(target)
    vol_rows = []
    for t, T in TENORS.items():
        for lab, (kb, vb) in target[t].items():
            vo = float(np.sqrt(max(surf.total_variance(kb, T), 1e-12) / T))
            vol_rows.append({"tenor": t, "point": lab, "err_bp": abs(vo - vb) * 1e4,
                             "within10d": lab in WITHIN_10D})
    ten_plot = [("1M", 1 / 12), ("3M", .25), ("6M", .5), ("1Y", 1.), ("2Y", 2.), ("5Y", 5.)]
    smile_curves = {}
    for t, T in ten_plot:
        r_d, r_f = rates_for(t)
        pts = sorted(target[t].items(), key=lambda kv: kv[1][0])
        smile_curves[t] = {
            "strikes": [kv[1][0] for kv in pts],
            "vols_bbg": [kv[1][1] * 100 for kv in pts],
            "vols_ours": [float(np.sqrt(max(surf.total_variance(kv[1][0], T), 1e-12) / T)) * 100
                          for kv in pts]}
    atm_ts = [{"tenor": t, "atm": target[t]["ATM"][1] * 100} for t in TENORS]
    rrbf = []
    for t in TENORS:
        c25, p25, atm = target[t]["25D C"][1], target[t]["25D P"][1], target[t]["ATM"][1]
        rrbf.append({"tenor": t, "rr25": (c25 - p25) * 100, "bf25": ((c25 + p25) / 2 - atm) * 100})
    return {"vol_rows": vol_rows, "smile_curves": smile_curves, "atm_ts": atm_ts, "rrbf": rrbf,
            "summary": {"n_points": len(vol_rows),
                        "vol_avg_bp": float(np.mean([r["err_bp"] for r in vol_rows])),
                        "vol_max_bp": float(np.max([r["err_bp"] for r in vol_rows]))}}


def desk_data():
    def surf(S, bump=0.0):
        q = [SmileQuote(0.25, 0.10 + bump, -0.01, 0.002, -0.018, 0.006),
             SmileQuote(0.5, 0.105 + bump, -0.011, 0.0022, -0.02, 0.0065),
             SmileQuote(1.0, 0.11 + bump, -0.012, 0.0025, -0.022, 0.007)]
        return VolSurface.from_quotes(q, S=S, r_d=0.03, r_f=0.01)
    book = Book([
        OptionPosition('EURUSD', 1.10, 0.5, True, -10_000_000, 'short 6m ATM C'),
        OptionPosition('EURUSD', 1.13, 0.5, True, 6_000_000, 'long 6m 25dC'),
        OptionPosition('EURUSD', 1.07, 0.25, False, -8_000_000, 'short 3m 25dP'),
        OptionPosition('EURUSD', 1.10, 1.0, True, 12_000_000, 'long 1y ATM C')])
    m0 = MarketState(1.10, 0.03, 0.01, surf(1.10))
    m1 = MarketState(1.1044, 0.03, 0.01, surf(1.1044, bump=0.005))
    pnl = explain(book, m0, m1)
    t = pnl.totals
    pnl_d = {"rows": [{"label": r.label, "actual": r.actual, "delta": r.delta_pnl,
                       "gamma": r.gamma_pnl, "vega": r.vega_pnl, "volga": r.volga_pnl,
                       "vanna": r.vanna_pnl, "theta": r.theta_pnl, "residual": r.residual}
                      for r in pnl.rows],
             "total": {"actual": t.actual, "delta": t.delta_pnl, "gamma": t.gamma_pnl,
                       "vega": t.vega_pnl, "volga": t.volga_pnl, "vanna": t.vanna_pnl,
                       "theta": t.theta_pnl, "explained": t.explained, "residual": t.residual}}
    grid = build_risk_grid(book, m0, spot_steps=7, vol_steps=7)
    vb = {k: v for k, v in bucketed_vega(book, m0).items() if abs(v) > 1e-6}
    mon = LimitMonitor([Limit('net delta', 'delta', 5_000_000), Limit('net vega', 'vega', 300_000),
                        Limit('6m vega bucket', 'vega_bucket', 150_000, bucket='6m'),
                        Limit('grid worst', 'grid_worst', 400_000)])
    breaches = [{"name": b.limit.name, "value": b.value} for b in mon.check(book, m0)]
    agg = {"delta": 0., "gamma": 0., "vega": 0., "theta": 0.}
    for p in book.positions:
        v = m0.surface.implied_vol(p.strike, p.expiry)
        gg = gk_greeks(m0.spot, p.strike, p.expiry, m0.r_dom, m0.r_for, v, p.is_call)
        agg["delta"] += gg.delta * p.notional
        agg["gamma"] += gg.gamma * p.notional
        agg["vega"] += gg.vega * p.notional
        agg["theta"] += -gg.theta * (1 / 365) * p.notional
    return {"pnl": pnl_d, "grid": {"spot": [s * 100 for s in grid.spot_shocks],
            "vol": [v * 100 for v in grid.vol_shocks], "pnl": grid.pnl.tolist()},
            "vega_buckets": vb, "breaches": breaches, "greeks": agg}


def main():
    data = {"validation": validation_data(), "desk": desk_data()}
    tmpl = (ROOT / "presentation" / "fxvol_report.html").read_text()
    import re
    new = re.sub(r"const DATA = .*?;</script>",
                 "const DATA = " + json.dumps(data) + ";</script>", tmpl, count=1, flags=re.S)
    (ROOT / "presentation" / "fxvol_report.html").write_text(new)
    print("report refreshed:", data["validation"]["summary"])


if __name__ == "__main__":
    main()
