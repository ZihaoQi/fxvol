"""Convert FX market quotes (ATM, RR, BF) into (strike, vol) anchor points.

ATM: the at-the-money vol (delta-neutral straddle)
RR: sigma_call - sigma_put     (smile SKEW)
BF: 0.5*(sigma_call + sigma_put) - ATM   (smile CONVEXITY)

Invert the algebra to recover the wing vols:
  sigma_call = ATM + BF + 0.5*RR
  sigma_put  = ATM + BF - 0.5*RR

Then map each wing vol+delta to a strike via conventions.strike_from_delta.

Result: a handful of (strike, vol) points per tenor which is what the SVI
smile is fitted to.
"""
from __future__ import annotations

from dataclasses import dataclass

from .conventions import DeltaType, atm_strike, strike_from_delta


@dataclass(frozen=True)
class SmileQuote:
    """Market quotes for ONE tenor."""
    T: float
    atm_vol: float
    rr_25: float
    bf_25: float
    rr_10: float
    bf_10: float


@dataclass(frozen=True)
class StrikeVol:
    strike: float
    vol: float
    label: str


def quotes_to_strikevols(
    q: SmileQuote, S: float, r_d: float, r_f: float,
    delta_type: DeltaType = DeltaType.SPOT,
) -> list[StrikeVol]:
    """Five anchor points: 10dP, 25dP, ATM, 25dC, 10dC."""
    c25 = q.atm_vol + q.bf_25 + 0.5 * q.rr_25
    p25 = q.atm_vol + q.bf_25 - 0.5 * q.rr_25
    c10 = q.atm_vol + q.bf_10 + 0.5 * q.rr_10
    p10 = q.atm_vol + q.bf_10 - 0.5 * q.rr_10

    k_atm = atm_strike(S, q.T, r_d, r_f, q.atm_vol)
    k_c25 = strike_from_delta(0.25, S, q.T, r_d, r_f, c25, True, delta_type)
    k_p25 = strike_from_delta(-0.25, S, q.T, r_d, r_f, p25, False, delta_type)
    k_c10 = strike_from_delta(0.10, S, q.T, r_d, r_f, c10, True, delta_type)
    k_p10 = strike_from_delta(-0.10, S, q.T, r_d, r_f, p10, False, delta_type)

    return [
        StrikeVol(k_p10, p10, "10dP"),
        StrikeVol(k_p25, p25, "25dP"),
        StrikeVol(k_atm, q.atm_vol, "ATM"),
        StrikeVol(k_c25, c25, "25dC"),
        StrikeVol(k_c10, c10, "10dC"),
    ]
