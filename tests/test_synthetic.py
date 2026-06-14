"""Synthetic G10 surface tests: well-formed, arb-free, realistic ordering."""
import numpy as np

from fxvol.synthetic.g10 import SPECS, TENORS, all_pairs, build_surface


def test_all_pairs_build_and_no_calendar_arb():
    for p in all_pairs():
        surf = build_surface(p)
        assert surf.no_calendar_arb(), f"{p} has calendar arbitrage"


def test_smiles_are_two_sided():
    """Each 1Y smile must curve up on both wings (a real smile, not monotone)."""
    for p in all_pairs():
        spec = SPECS[p]
        surf = build_surface(p)
        F = spec.spot * np.exp((spec.r_dom - spec.r_for) * 1.0)
        ks = np.array([-0.15, -0.075, 0, 0.075, 0.15])
        v = [surf.implied_vol(F * np.exp(k), 1.0) for k in ks]
        assert v[0] > min(v) and v[-1] > min(v), f"{p} smile not two-sided"


def test_negative_risk_reversal():
    """All four specs encode downside skew (puts bid) -> negative 25d RR."""
    for p in all_pairs():
        assert all(rr < 0 for rr in SPECS[p].rr25), f"{p} should have negative RR"


def test_usdjpy_highest_vol():
    """USDJPY should carry the highest ATM vol of the four (carry-pair feature)."""
    atm_6m = {p: build_surface(p).implied_vol(SPECS[p].spot, 0.5) for p in all_pairs()}
    assert atm_6m["USDJPY"] == max(atm_6m.values())


def test_tenor_count():
    for p in all_pairs():
        assert len(SPECS[p].atm) == len(TENORS)
