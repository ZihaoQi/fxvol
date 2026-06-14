"""Bloomberg OVML surface validation - USDCNH, real Bloomberg rates & forwards.

Two comparisons:
  STRIKE accuracy : back out each delta point's strike under BBG conventions
                    (DNS / premium-adjusted / spot<1Y forward>=1Y) and diff vs
                    Bloomberg's calibrated strike.
  VOL accuracy    : evaluate our SVI surface at Bloomberg's OWN strikes and diff
                    the vol. This isolates surface-construction quality from
                    delta-round-trip noise and is the like-for-like fit test.

Surfaces are calibrated to all 11 Bloomberg strike/vol points per tenor.
Rates/forwards are Bloomberg's actual screen values, so forwards reproduce
exactly and no implied-carry approximation enters.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from fxvol.core.conventions import (
    ATMConvention, atm_strike, delta_type_for_tenor, forward, strike_from_delta,
)
from fxvol.surface.smile import fit_svi
from fxvol.surface.surface import VolSurface

SPOT = 7.2605
TENORS = {
    "1D": 1 / 365, "1W": 7 / 365, "2W": 14 / 365, "3W": 21 / 365,
    "1M": 1 / 12, "2M": 2 / 12, "3M": 3 / 12, "4M": 4 / 12, "5M": 5 / 12,
    "6M": 0.5, "9M": 0.75, "1Y": 1.0, "18M": 1.5, "2Y": 2.0,
    "3Y": 3.0, "4Y": 4.0, "5Y": 5.0,
}
# Bloomberg rates screen: tenor -> (USD deposit %, CNY deposit %, USDCNY forward)
BBG = {
    "1W": (4.463, 1.998, 7.2571), "2W": (4.480, 1.998, 7.2536),
    "3W": (4.475, 1.999, 7.2502), "1M": (4.464, 1.999, 7.2454),
    "2M": (4.514, 2.001, 7.2302), "3M": (4.561, 1.995, 7.2131),
    "4M": (4.564, 1.947, 7.1980), "5M": (4.586, 1.915, 7.1808),
    "6M": (4.618, 1.894, 7.1626), "9M": (4.654, 1.827, 7.1106),
    "1Y": (4.654, 1.779, 7.0600), "18M": (4.492, 1.713, 6.9758),
    "2Y": (4.348, 1.683, 6.9040), "3Y": (4.217, 1.655, 6.7646),
    "4Y": (4.152, 1.662, 6.6400), "5Y": (4.067, 1.681, 6.5405),
}
BBG["1D"] = BBG["1W"]
WITHIN_10D = {"10D P", "15D P", "25D P", "35D P", "ATM",
              "35D C", "25D C", "15D C", "10D C"}
DELTA_MAP = {
    "5D P": (-0.05, False), "10D P": (-0.10, False), "15D P": (-0.15, False),
    "25D P": (-0.25, False), "35D P": (-0.35, False), "ATM": (0.0, True),
    "35D C": (0.35, True), "25D C": (0.25, True), "15D C": (0.15, True),
    "10D C": (0.10, True), "5D C": (0.05, True),
}


def rates_for(tenor: str) -> tuple[float, float]:
    """Continuous (r_dom=CNY, r_for=USD) reproducing BBG's forward exactly."""
    usd, _cny, fwd = BBG[tenor]
    T = TENORS[tenor]
    r_for = np.log(1 + (usd / 100) * T) / T
    r_dom = r_for + np.log(fwd / SPOT) / T
    return float(r_dom), float(r_for)


def load_target(updir: Path):
    raw = pd.read_excel(updir / "detailed_volatility_smile.xlsx", header=None)
    labels = list(raw.iloc[2, 4:].values)
    target, r = {}, 3
    while r < raw.shape[0]:
        exp = raw.iloc[r, 0]
        if pd.isna(exp):
            r += 1
            continue
        vols = raw.iloc[r, 4:].astype(float).values
        strikes = raw.iloc[r + 1, 4:].astype(float).values
        target[str(exp)] = {lab: (float(k), float(v) / 100.0)
                            for lab, k, v in zip(labels, strikes, vols)}
        r += 2
    return target


def build_surface(target):
    tenors = sorted(TENORS.values())
    name_by_T = {v: k for k, v in TENORS.items()}
    smiles = []
    fwds = []
    for T in tenors:
        t = name_by_T[T]
        r_d, r_f = rates_for(t)
        F = forward(SPOT, T, r_d, r_f)
        fwds.append(F)
        ks, tvs = [], []
        for _lab, (strike, vol) in target[t].items():
            ks.append(np.log(strike / F))
            tvs.append(vol ** 2 * T)
        o = np.argsort(ks)
        smiles.append(fit_svi(np.array(ks)[o], np.array(tvs)[o]))
    r_d0, r_f0 = rates_for(name_by_T[tenors[0]])
    return VolSurface(SPOT, r_d0, r_f0, tenors, smiles, forwards=fwds)


def main():
    updir = Path(__file__).resolve().parents[1] / "data" / "bloomberg"
    target = load_target(updir)
    surf = build_surface(target)

    vol_err_w, vol_err_t = [], []
    strike_err_w, strike_err_t = [], []
    detail = []
    for t, T in TENORS.items():
        r_d, r_f = rates_for(t)
        dt = delta_type_for_tenor(T)
        for lab, (strike_bbg, vol_bbg) in target[t].items():
            # VOL: our surface at BBG's own strike
            vol_ours = float(np.sqrt(max(surf.total_variance(strike_bbg, T), 1e-12) / T))
            ve = abs(vol_ours - vol_bbg) * 1e4
            # STRIKE: back out the delta point under BBG conventions
            delta, is_call = DELTA_MAP[lab]
            if lab == "ATM":
                strike_ours = atm_strike(SPOT, T, r_d, r_f, vol_bbg,
                                         ATMConvention.DNS, True)
            else:
                strike_ours = strike_from_delta(delta, SPOT, T, r_d, r_f,
                                                vol_bbg, is_call, dt)
            se = abs(strike_ours - strike_bbg)
            vol_err_t.append(ve)
            strike_err_t.append(se)
            if lab in WITHIN_10D:
                vol_err_w.append(ve)
                strike_err_w.append(se)
            detail.append((t, lab, strike_ours, strike_bbg, se, vol_ours, vol_bbg, ve))

    print("USDCNH Bloomberg OVML surface validation")
    print(f"Spot {SPOT} | real BBG rates+forwards | DNS, premium-adj, "
          f"spot<1Y/forward>=1Y | SVI on 11 points/tenor")
    print("=" * 72)
    print(f"{'VOL accuracy (surface @ BBG strike)':<40}")
    print(f"  within-10D: avg {np.mean(vol_err_w):.2f}bp  "
          f"max {np.max(vol_err_w):.1f}bp  ({len(vol_err_w)} pts)")
    print(f"  total:      avg {np.mean(vol_err_t):.2f}bp  "
          f"max {np.max(vol_err_t):.1f}bp  ({len(vol_err_t)} pts)")
    print(f"{'STRIKE accuracy (delta back-out @ BBG vol)':<40}")
    print(f"  within-10D: avg {np.mean(strike_err_w):.5f}  "
          f"max {np.max(strike_err_w):.4f}  ({len(strike_err_w)} pts)")
    print(f"  total:      avg {np.mean(strike_err_t):.5f}  "
          f"max {np.max(strike_err_t):.4f}  ({len(strike_err_t)} pts)")
    print("=" * 72)
    # worst few vol points
    detail.sort(key=lambda x: -x[7])
    print("Largest vol diffs:")
    for t, lab, ko, kb, se, vo, vb, ve in detail[:6]:
        print(f"  {t:>4} {lab:>6}: ours {vo*100:.3f}%  bbg {vb*100:.3f}%  "
              f"diff {ve:.1f}bp")


if __name__ == "__main__":
    main()
