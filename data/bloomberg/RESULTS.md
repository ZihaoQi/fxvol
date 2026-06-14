# Bloomberg OVML validation results — USDCNY

Replication of the Mathema Bloomberg OVDV surface-construction validation, run
against a real USDCNY onshore snapshot (spot 7.2605).

## Inputs (all from Bloomberg)
- `volatility_surface.xlsx` — raw ATM / 25D RR-BF / 10D RR-BF per tenor.
- `detailed_volatility_smile.xlsx` — Bloomberg's calibrated strike + vol at 11
  delta points (5/10/15/25/35 P, ATM, 35/25/15/10/5 C) for 17 tenors.
- `rates_20250305.csv` — USD and CNY deposit rates and USDCNY forwards.

## Conventions (matched to Bloomberg)
- ATM = Delta-Neutral Straddle (DNS)
- Premium-adjusted = Yes
- Delta = Spot delta for tenors < 1Y, Forward delta for tenors ≥ 1Y

## Method
Per tenor, an SVI smile is calibrated to all 11 Bloomberg strike/vol points in
that tenor's own forward log-moneyness. Forwards come straight from Bloomberg's
screen, so the forward curve is reproduced exactly (no implied-carry proxy).
Two independent accuracy metrics:

- **Vol accuracy** — evaluate our surface at Bloomberg's own strikes, diff the
  implied vol. Isolates surface-construction quality.
- **Strike accuracy** — back out each delta point's strike under the BBG
  conventions, diff against Bloomberg's calibrated strike.

## Results

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

## What this exercise caught

Running real (non-flat) USDCNY rates exposed a genuine bug in the surface
interpolation: `total_variance` was computing log-moneyness from a single
surface-wide forward, while each smile had been fitted in its own per-tenor
forward. With a flat rate curve this is harmless (all forwards equal), so the
unit tests passed — but with USDCNY's forward running 7.26 → 6.54 across the
curve, long-tenor vols came out roughly doubled. The fix stores per-tenor
forwards and reads each bracketing smile at its own moneyness
(`test_per_tenor_forwards_no_double_count` guards it). This bug would also have
mispriced the P&L engine and risk grid under any realistic rate curve — the
Bloomberg comparison is what surfaced it.

## Reproduce

```bash
python scripts/run_bloomberg_real.py
```
