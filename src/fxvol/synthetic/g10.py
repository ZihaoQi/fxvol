"""Synthetic-but-realistic G10 FX vol surfaces, generated from the model.

PURPOSE
The Bloomberg USDCNH data used for validation is licensed and not redistributed.
To keep the repo self-contained and runnable by anyone who clones it, this
module GENERATES illustrative surfaces for the major pairs from realistic ATM /
RR / BF term structures. These are NOT market quotes - they are model-generated
and clearly labelled as such - but they are calibrated to resemble each pair's
characteristic smile so the surface, P&L, and risk tooling can be demonstrated.

Per-pair stylized facts encoded below (typical, not point-in-time):
  EURUSD : low vol, mild slightly-negative skew, modest convexity
  USDJPY : negative risk reversal (JPY calls / USD puts bid - the carry-unwind
           crash hedge), term structure of skew steepening with tenor
  USDCHF : low-to-mid vol, negative skew (CHF safe-haven bid), like EURUSD's mirror
  GBPUSD : higher vol, two-sided skew, event-sensitive front end
"""
from __future__ import annotations

from dataclasses import dataclass

from ..core.quotes import SmileQuote
from ..surface.surface import VolSurface

# Standard tenor grid (year fractions)
TENORS = [1 / 12, 0.25, 0.5, 1.0, 2.0]
TENOR_LABELS = ["1M", "3M", "6M", "1Y", "2Y"]


@dataclass
class PairSpec:
    """Stylized vol parameters for one pair. Vols/quotes in DECIMAL (0.08 = 8%)."""
    pair: str
    spot: float
    r_dom: float
    r_for: float
    atm: list[float]      # ATM vol per tenor
    rr25: list[float]     # 25d risk reversal per tenor (skew)
    bf25: list[float]     # 25d butterfly per tenor (convexity)
    rr10: list[float]     # 10d risk reversal
    bf10: list[float]     # 10d butterfly


# Realistic-looking specs (illustrative, model-generated - NOT market quotes).
SPECS: dict[str, PairSpec] = {
    "EURUSD": PairSpec(
        "EURUSD", spot=1.0850, r_dom=0.041, r_for=0.025,
        atm=[0.072, 0.075, 0.078, 0.082, 0.085],
        rr25=[-0.004, -0.005, -0.006, -0.007, -0.008],
        bf25=[0.0015, 0.0017, 0.0019, 0.0021, 0.0023],
        rr10=[-0.008, -0.010, -0.011, -0.013, -0.015],
        bf10=[0.0050, 0.0055, 0.0060, 0.0066, 0.0072]),
    "USDJPY": PairSpec(
        "USDJPY", spot=152.30, r_dom=0.055, r_for=0.005,
        atm=[0.092, 0.098, 0.103, 0.110, 0.118],
        rr25=[-0.012, -0.014, -0.016, -0.019, -0.023],  # JPY-call skew, moderated
        bf25=[0.0030, 0.0034, 0.0038, 0.0044, 0.0051],  # higher convexity -> real smile
        rr10=[-0.022, -0.026, -0.030, -0.036, -0.044],
        bf10=[0.0100, 0.0112, 0.0125, 0.0143, 0.0164]),
    "USDCHF": PairSpec(
        "USDCHF", spot=0.8820, r_dom=0.055, r_for=0.015,
        atm=[0.068, 0.071, 0.074, 0.079, 0.083],
        rr25=[-0.005, -0.006, -0.008, -0.010, -0.012],  # CHF safe-haven bid
        bf25=[0.0020, 0.0023, 0.0026, 0.0030, 0.0034],
        rr10=[-0.009, -0.011, -0.014, -0.017, -0.020],
        bf10=[0.0066, 0.0073, 0.0081, 0.0090, 0.0100]),
    "GBPUSD": PairSpec(
        "GBPUSD", spot=1.2710, r_dom=0.041, r_for=0.048,
        atm=[0.082, 0.086, 0.090, 0.096, 0.102],
        rr25=[-0.005, -0.006, -0.007, -0.009, -0.011],
        bf25=[0.0018, 0.0020, 0.0022, 0.0025, 0.0028],
        rr10=[-0.010, -0.012, -0.014, -0.017, -0.021],
        bf10=[0.0058, 0.0064, 0.0070, 0.0078, 0.0087]),
}


def build_quotes(spec: PairSpec) -> list[SmileQuote]:
    return [
        SmileQuote(T=T, atm_vol=spec.atm[i], rr_25=spec.rr25[i], bf_25=spec.bf25[i],
                   rr_10=spec.rr10[i], bf_10=spec.bf10[i])
        for i, T in enumerate(TENORS)
    ]


def build_surface(pair: str) -> VolSurface:
    """Construct a synthetic VolSurface for a major pair."""
    spec = SPECS[pair]
    quotes = build_quotes(spec)
    return VolSurface.from_quotes(quotes, S=spec.spot, r_d=spec.r_dom, r_f=spec.r_for)


def all_pairs() -> list[str]:
    return list(SPECS.keys())
