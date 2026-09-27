"""FX delta conventions and quote-to-strike conversion.

WHY THIS MODULE EXISTS
In equities a strike is a strike. In FX, smiles are quoted against DELTA, not
strike - a "25-delta risk reversal" is the vol difference between the call and
put whose deltas are +/-0.25. To build a surface you must invert: given a target
delta and a vol, find the strike. That inversion lives here.

"Delta" is ambiguous in FX. Four conventions coexist:
  - spot vs forward delta (is the exp(-r_f*T) discount factor included?)
  - unadjusted vs premium-adjusted (PA): when the option premium is paid in the
    FOREIGN currency it is itself an FX exposure, so it is netted out of the
    delta. PA is standard for many pairs.

WHICH convention applies depends on the pair AND the tenor. The Bloomberg USDCNY
setup this module supports:
  - ATM convention : Delta-Neutral Straddle (DNS)
  - Premium adjusted: YES
  - Delta          : SPOT delta for tenors < 1Y, FORWARD delta for tenors >= 1Y

Getting any of these wrong shifts every strike on the surface, so they are
explicit, testable inputs - never hard-coded assumptions.
"""
from __future__ import annotations

from enum import Enum

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm


class DeltaType(Enum):
    SPOT = "spot"                      # delta = exp(-r_f*T) * N(d1)
    FORWARD = "forward"                # delta = N(d1)
    SPOT_PA = "spot_pa"                # premium-adjusted spot
    FORWARD_PA = "forward_pa"          # premium-adjusted forward


class ATMConvention(Enum):
    DNS = "dns"                        # delta-neutral straddle (Bloomberg USDCNH)
    FORWARD = "forward"                # ATM = forward
    SPOT = "spot"                      # ATM = spot


def forward(S: float, T: float, r_d: float, r_f: float) -> float:
    return float(S * np.exp((r_d - r_f) * T))


def delta_type_for_tenor(
    T: float,
    short: DeltaType = DeltaType.SPOT_PA,
    long: DeltaType = DeltaType.FORWARD_PA,
    cutoff_years: float = 1.0,
) -> DeltaType:
    """Tenor-dependent delta convention switch.

    Bloomberg USDCNH uses spot delta below 1Y and forward delta at 1Y+. The
    cutoff is INCLUSIVE on the long side (T == 1Y uses the long convention),
    matching Bloomberg's '1Y and above' wording.
    """
    return long if T >= cutoff_years - 1e-9 else short


def _delta_of_strike(
    K: float, S: float, T: float, r_d: float, r_f: float,
    sigma: float, is_call: bool, delta_type: DeltaType,
) -> float:
    """Signed delta produced by a given strike under a chosen convention."""
    F = forward(S, T, r_d, r_f)
    vol_sqrt_t = sigma * np.sqrt(T)
    d1 = (np.log(F / K) + 0.5 * sigma**2 * T) / vol_sqrt_t
    d2 = d1 - vol_sqrt_t

    if delta_type in (DeltaType.SPOT, DeltaType.FORWARD):
        base = norm.cdf(d1) if is_call else norm.cdf(d1) - 1.0
        if delta_type == DeltaType.SPOT:
            base *= np.exp(-r_f * T)
        return float(base)

    # premium-adjusted: the premium (in foreign ccy) is netted out. The PA delta
    # is (K/F) * N(d2)-type, carrying K on both sides - hence the root-find in
    # strike_from_delta. Sign handled per call/put.
    disc_f = np.exp(-r_f * T) if delta_type == DeltaType.SPOT_PA else 1.0
    if is_call:
        return float(disc_f * (K / F) * norm.cdf(d2))
    return float(-disc_f * (K / F) * norm.cdf(-d2))


def strike_from_delta(
    delta: float,
    S: float,
    T: float,
    r_d: float,
    r_f: float,
    sigma: float,
    is_call: bool,
    delta_type: DeltaType = DeltaType.SPOT,
) -> float:
    """Invert a (signed) delta to the strike that produces it.

    One numerical path (brentq) for all four conventions - the PA ones have no
    closed form (K appears on both sides), and using the same path everywhere
    keeps the code honest and uniform.
    """
    F = forward(S, T, r_d, r_f)

    def obj(K: float) -> float:
        return _delta_of_strike(K, S, T, r_d, r_f, sigma, is_call, delta_type) - delta

    # The premium-adjusted CALL delta is NOT monotone in K: (K/F)*N(d2) rises
    # from 0, peaks, then decays to 0, so a naive [tiny, 100F] bracket can share
    # a sign at both ends. We scan a dense grid of candidate strikes and pick the
    # first sign change, then refine with brentq. For the (monotone) non-PA cases
    # this finds the single root immediately too.
    grid = F * np.exp(np.linspace(-3.0, 3.0, 400))  # +/- ~3 vols of log-moneyness span
    vals = np.array([obj(K) for K in grid])
    sign_changes = np.where(np.sign(vals[:-1]) != np.sign(vals[1:]))[0]
    if len(sign_changes) == 0:
        raise ValueError(
            f"No strike reproduces delta={delta} (conv={delta_type.value}, T={T}); "
            "target delta may be unreachable for this convention/vol."
        )
    # For an OTM option we want the economically correct root: calls -> the
    # higher-strike (further OTM) branch, puts -> the lower-strike branch.
    idx = sign_changes[-1] if is_call else sign_changes[0]
    return float(brentq(obj, grid[idx], grid[idx + 1], xtol=1e-12))


def atm_strike(
    S: float, T: float, r_d: float, r_f: float, sigma: float,
    convention: ATMConvention = ATMConvention.DNS,
    premium_adjusted: bool = True,
) -> float:
    """ATM strike under the chosen convention.

    DNS (delta-neutral straddle) is the strike where a straddle has zero delta.
    - Non-PA DNS:  K = F * exp(0.5 * sigma^2 * T)
    - PA DNS:      K = F * exp(-0.5 * sigma^2 * T)
    The sign flip in the exponent is the premium-adjustment correction: netting
    the foreign-currency premium out of the delta moves the zero-delta strike to
    the OTHER side of the forward. Missing this is a classic ATM mismark.
    """
    F = forward(S, T, r_d, r_f)
    if convention == ATMConvention.FORWARD:
        return F
    if convention == ATMConvention.SPOT:
        return S
    # DNS
    if premium_adjusted:
        return float(F * np.exp(-0.5 * sigma**2 * T))
    return float(F * np.exp(0.5 * sigma**2 * T))
