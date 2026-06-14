# FX Volatility Surface Modelling + P&L Engine + Risk Grid & Limit Monitor

FX option pricing models, built up the full modeling ladder a real desk uses:
**from volatility surfaces to multi-factor exotic models.**

This is a learning-by-building project. Each layer depends on the one below it,
which is also the order to build them in.

## The ladder

| Layer | Module | What it does |
|-------|--------|--------------|
| Foundation | `core/` | Garman-Kohlhagen (FX Black-Scholes), Greeks, delta conventions |
| Surface | `surface/` | ATM/RR/BF quotes → SVI smile → 2D surface with no-arb checks |
| Local vol | `localvol/` | Dupire local volatility from the surface |
| Stochastic vol | `stochvol/` | Heston via characteristic function + calibration |
| LSV | `lsv/` | Leverage function bridging Heston to an exact surface fit |
| Multi-factor | `multifactor/` | Spot FX + Hull-White rates, Monte Carlo, FX-rates correlation |

## Design principle

Every model implements the same interface (`PricingModel` in `base.py`):
`calibrate(market)` then `price(strike, expiry)`. That substitutability is the
point — a risk system swaps one model for another without knowing the difference.

## Setup

```bash
pip install -e ".[dev]"
pytest
```

## Status

Built incrementally. Tests define "correct" for each layer — make them green.

## Desk tooling (built on the model stack)

Three production modules sit on top of the pricing ladder:

### `validation/` — Bloomberg OVML surface validation
Replicates [Mathema](#reference)'s Bloomberg OVDV comparison methodology for USDCNY: builds
the surface under Bloomberg's exact conventions (DNS ATM, premium-adjusted,
spot delta <1Y / forward delta >=1Y), backs out strike+vol at 11 delta points
per tenor, and reports the within-10D vs total error split.

Run against the included real USDCNY snapshot:
`python scripts/run_bloomberg_real.py`.

**Result**: vol agrees with Bloomberg to
3.2bp avg (15.8bp max) and strikes to 0.0006 avg across 187 points.


| Metric | Within 10D | Total (all points) |
|---|---|---|
| Vol abs error (avg) | 3.19 bp | 3.16 bp |
| Vol abs error (max) | 15.8 bp | 15.8 bp |
| Strike abs error (avg) | 0.00051 | 0.00062 |
| Strike abs error (max) | 0.0053 | 0.0072 |

187 points across 17 tenors × 11 delta strikes.

The largest vol differences are all at the 1-day tenor (no 1D forward on the
rate screen — 1W rates are reused — and the 1D smile is noisiest). Everything
from 1W out agrees to better than ~11 bp, wings included.

### `pnl/` — Greek P&L explain engine
The nightly desk process: decomposes daily book P&L into
delta / gamma / vega / volga / vanna / theta + an unexplained residual.

 `python scripts/demo_desk_workflow.py`.

### `risk/` — risk grid + limit monitor
Full spot x vol revaluation grid, bucketed vega by tenor, and a configurable
limit monitor (per-Greek, per-bucket, and worst-cell) with breach flags.

Run the combined demo: `python scripts/demo_desk_workflow.py`


## Visual report

`presentation/fxvol_report.html` is a self-contained interactive report
(validation results, vol surface, smiles, P&L attribution, risk grid, limits).
Open it in any browser. Regenerate from live model outputs with:

```bash
python scripts/generate_report.py
```


## References

https://help.mathema.com.cn/latest/docs/toolbox/bbg_ovml.html