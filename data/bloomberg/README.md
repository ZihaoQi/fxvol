# Bloomberg OVML validation data (USDCNY, 2025-03-05)

Transcribe the numbers from the Bloomberg OVML screenshots into these two CSVs.

## `USDCNY_20250305_market.csv` — raw market data, one row per tenor
- `spot` is constant (7.2605 for 2025-03-05); repeat it on each row.
- `r_dom`, `r_for`: continuously-compounded CNH and USD rates at that tenor
  (derive from the interest-rate / forward-points screen).
- `atm_vol`, `rr25`, `bf25`, `rr10`, `bf10`: from the volatility-data screen,
  in DECIMAL (5% -> 0.05, -0.30% RR -> -0.003).
- Add/remove tenor rows to match the screenshot (1W,2W,1M,2M,3M,6M,9M,1Y,18M,2Y...).

## `USDCNY_20250305_bbg_target.csv` — Bloomberg's calibrated output
- One row per (tenor, delta point). `delta_label` from:
  5DP,10DP,15DP,25DP,35DP,ATM,35DC,25DC,15DC,10DC,5DC.
- `strike`, `vol` are Bloomberg's calibrated numbers (vol in decimal).

## Conventions (already wired into the harness)
- ATM = Delta-Neutral Straddle (DNS)
- Premium-adjusted = YES
- Delta = Spot (<1Y) / Forward (>=1Y)

Then run:  `python scripts/run_bloomberg_validation.py`
