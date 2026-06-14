"""Bloomberg OVDV volatility-surface validation harness.

Replicates the comparison methodology from Mathema's Bloomberg OVML validation:
build a surface from the same raw market data under the SAME conventions
Bloomberg uses, back out strike + vol at a wide set of delta points across all
tenors, and compare against Bloomberg's own calibrated output - reporting the
error split within-10-delta vs beyond-10-delta exactly as the reference does.

USDCNH conventions for this data set (from the Bloomberg OVML screens):
  - ATM        : Delta-Neutral Straddle (DNS)
  - Premium adj: YES
  - Delta      : Spot delta < 1Y, Forward delta >= 1Y

DATA: you transcribe the Bloomberg grids from the screenshots into two CSVs
(schemas below). Until then the loader validates structure and the comparison
runs the moment real numbers are present.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..core.conventions import (
    ATMConvention,
    atm_strike,
    delta_type_for_tenor,
    forward,
    strike_from_delta,
)
from ..core.quotes import SmileQuote
from ..surface.surface import VolSurface


# Standard Bloomberg OVML delta points (signed: negative = put wing).
# 5DP,10DP,15DP,25DP,35DP, ATM, 35DC,25DC,15DC,10DC,5DC
DELTA_POINTS: list[tuple[str, float, bool]] = [
    ("5DP", -0.05, False), ("10DP", -0.10, False), ("15DP", -0.15, False),
    ("25DP", -0.25, False), ("35DP", -0.35, False),
    ("ATM", 0.0, True),
    ("35DC", 0.35, True), ("25DC", 0.25, True), ("15DC", 0.15, True),
    ("10DC", 0.10, True), ("5DC", 0.05, True),
]

WITHIN_10D_LABELS = {"10DP", "15DP", "25DP", "35DP", "ATM", "35DC", "25DC", "15DC", "10DC"}


@dataclass
class MarketData:
    """Raw Bloomberg market data for one valuation date."""
    spot: float
    # tenor (years) -> (r_dom, r_for) continuously-compounded
    rates: dict[float, tuple[float, float]]
    # tenor (years) -> SmileQuote (ATM/RR/BF at 25d and 10d)
    quotes: dict[float, SmileQuote]


@dataclass
class BloombergTarget:
    """Bloomberg's calibrated strike+vol output to compare against.

    target[tenor][delta_label] = (strike, vol)
    """
    grid: dict[float, dict[str, tuple[float, float]]] = field(default_factory=dict)


def load_market_data(path: str | Path) -> MarketData:
    """Load raw market data CSV.

    Schema (one row per tenor):
      tenor_years,spot,r_dom,r_for,atm_vol,rr25,bf25,rr10,bf10
    spot repeated each row (constant); vols in decimal (0.05 = 5%).
    """
    rows = list(csv.DictReader(Path(path).open()))
    if not rows:
        raise ValueError(f"{path} is empty - transcribe the Bloomberg grid first.")
    spot = float(rows[0]["spot"])
    rates: dict[float, tuple[float, float]] = {}
    quotes: dict[float, SmileQuote] = {}
    for r in rows:
        T = float(r["tenor_years"])
        rates[T] = (float(r["r_dom"]), float(r["r_for"]))
        quotes[T] = SmileQuote(
            T=T, atm_vol=float(r["atm_vol"]),
            rr_25=float(r["rr25"]), bf_25=float(r["bf25"]),
            rr_10=float(r["rr10"]), bf_10=float(r["bf10"]),
        )
    return MarketData(spot=spot, rates=rates, quotes=quotes)


def load_bloomberg_target(path: str | Path) -> BloombergTarget:
    """Load Bloomberg's calibrated output CSV.

    Schema (one row per tenor x delta point):
      tenor_years,delta_label,strike,vol
    """
    tgt = BloombergTarget()
    for r in csv.DictReader(Path(path).open()):
        T = float(r["tenor_years"])
        tgt.grid.setdefault(T, {})[r["delta_label"]] = (
            float(r["strike"]), float(r["vol"]),
        )
    return tgt


def build_surface(md: MarketData) -> VolSurface:
    """Build our surface from the raw data (per-tenor rates honored)."""
    # VolSurface.from_quotes assumes flat rates; here rates vary by tenor, so we
    # build per-tenor smiles directly with each tenor's own (r_dom, r_for).
    from ..surface.smile import fit_svi

    tenors = sorted(md.quotes)
    smiles = []
    # Use the shortest tenor's rates as the surface-level default for interpolation
    r_d0, r_f0 = md.rates[tenors[0]]
    for T in tenors:
        r_d, r_f = md.rates[T]
        q = md.quotes[T]
        svs = _quotes_to_strikevols_usdcnh(q, md.spot, r_d, r_f)
        F = forward(md.spot, T, r_d, r_f)
        k = np.array([np.log(sv[0] / F) for sv in svs])
        tv = np.array([sv[1] ** 2 * T for sv in svs])
        smiles.append(fit_svi(k, tv))
    return VolSurface(md.spot, r_d0, r_f0, tenors, smiles)


def _quotes_to_strikevols_usdcnh(q: SmileQuote, S, r_d, r_f):
    """Strike-vol anchors under USDCNH conventions (DNS-PA, tenor-switched delta)."""
    dt = delta_type_for_tenor(q.T)
    c25 = q.atm_vol + q.bf_25 + 0.5 * q.rr_25
    p25 = q.atm_vol + q.bf_25 - 0.5 * q.rr_25
    c10 = q.atm_vol + q.bf_10 + 0.5 * q.rr_10
    p10 = q.atm_vol + q.bf_10 - 0.5 * q.rr_10
    k_atm = atm_strike(S, q.T, r_d, r_f, q.atm_vol, ATMConvention.DNS, True)
    return [
        (strike_from_delta(-0.10, S, q.T, r_d, r_f, p10, False, dt), p10),
        (strike_from_delta(-0.25, S, q.T, r_d, r_f, p25, False, dt), p25),
        (k_atm, q.atm_vol),
        (strike_from_delta(0.25, S, q.T, r_d, r_f, c25, True, dt), c25),
        (strike_from_delta(0.10, S, q.T, r_d, r_f, c10, True, dt), c10),
    ]


@dataclass
class ComparisonResult:
    rows: list[dict]
    strike_err_within_10d: float
    strike_err_total: float
    vol_err_within_10d: float
    vol_err_total: float

    def report(self) -> str:
        lines = [
            "Bloomberg OVML surface validation",
            "=" * 60,
            f"{'tenor':>6} {'point':>5} {'K_ours':>9} {'K_bbg':>9} "
            f"{'dK':>8} {'v_ours':>8} {'v_bbg':>8} {'dv(bp)':>8}",
        ]
        for r in self.rows:
            lines.append(
                f"{r['tenor']:>6.3f} {r['label']:>5} {r['k_ours']:>9.4f} "
                f"{r['k_bbg']:>9.4f} {r['dk']:>8.4f} {r['v_ours']:>8.4f} "
                f"{r['v_bbg']:>8.4f} {r['dv_bp']:>8.1f}"
            )
        lines += [
            "=" * 60,
            f"Strike abs-error  within-10D: {self.strike_err_within_10d:.4f}",
            f"Strike abs-error  total:      {self.strike_err_total:.4f}",
            f"Vol abs-error(%)  within-10D: {self.vol_err_within_10d * 100:.4f}",
            f"Vol abs-error(%)  total:      {self.vol_err_total * 100:.4f}",
        ]
        return "\n".join(lines)


def compare(md: MarketData, target: BloombergTarget) -> ComparisonResult:
    """Back out our strike+vol at each delta point and diff against Bloomberg."""
    surf = build_surface(md)
    rows: list[dict] = []
    for T in sorted(target.grid):
        r_d, r_f = md.rates[T]
        dt = delta_type_for_tenor(T)
        for label, delta, is_call in DELTA_POINTS:
            if label not in target.grid[T]:
                continue
            k_bbg, v_bbg = target.grid[T][label]
            if label == "ATM":
                k_ours = atm_strike(md.spot, T, r_d, r_f,
                                    md.quotes[T].atm_vol, ATMConvention.DNS, True)
            else:
                # iterate: vol depends on strike via the surface; 2 passes suffice
                v_guess = md.quotes[T].atm_vol
                for _ in range(8):
                    k_ours = strike_from_delta(delta, md.spot, T, r_d, r_f,
                                               v_guess, is_call, dt)
                    v_new = surf.implied_vol(k_ours, T)
                    if abs(v_new - v_guess) < 1e-8:
                        break
                    v_guess = v_new
            v_ours = surf.implied_vol(k_ours, T)
            rows.append({
                "tenor": T, "label": label, "k_ours": k_ours, "k_bbg": k_bbg,
                "dk": k_ours - k_bbg, "v_ours": v_ours, "v_bbg": v_bbg,
                "dv_bp": (v_ours - v_bbg) * 1e4,
                "within10d": label in WITHIN_10D_LABELS,
            })

    def _sum(field_, mask):
        return float(sum(abs(r[field_]) for r in rows if mask(r)))

    return ComparisonResult(
        rows=rows,
        strike_err_within_10d=_sum("dk", lambda r: r["within10d"]),
        strike_err_total=_sum("dk", lambda r: True),
        vol_err_within_10d=_sum("dv_bp", lambda r: r["within10d"]) / 1e4,
        vol_err_total=_sum("dv_bp", lambda r: True) / 1e4,
    )
