# Bloomberg OVML validation data (USDCNH)

**The data files in this folder are licensed Bloomberg terminal output and are
NOT committed to the repository** (they are git-ignored). To reproduce the
validation, supply your own Bloomberg OVML snapshot in the schemas below.

## Files (you provide these)
- `volatility_surface.csv` — raw ATM / 25D RR-BF / 10D RR-BF per tenor.
- `detailed_volatility_smile.csv` — Bloomberg's calibrated strike + vol at each
  delta point (one row per `tenor, delta_point, strike, vol_pct`).
- `rates_20250305.csv` — `tenor, usd_deposit_pct, cny_deposit_pct, usdcnh_forward`.

The loaders also accept the original `.xlsx` exports if present.

## Conventions (wired into the harness)
- ATM = Delta-Neutral Straddle (DNS)
- Premium-adjusted = Yes
- Delta = Spot (<1Y) / Forward (>=1Y)

## Run
```bash
python scripts/run_bloomberg_real.py
```
Without the data files the script prints an explanatory message and exits
cleanly; the rest of the project runs on the synthetic G10 surfaces.

## Result on the reference snapshot (05 Mar 2025)
Vol agrees with Bloomberg to 3.2bp average (15.8bp max) across 187 points;
strikes to 0.0006 average. See `RESULTS.md`.

## Methodology reference
Follows Mathema's Bloomberg OVML comparison:
https://help.mathema.com.cn/latest/docs/toolbox/bbg_ovml.html
