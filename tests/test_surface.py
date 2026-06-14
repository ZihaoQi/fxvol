"""Surface construction and no-arbitrage tests."""
from fxvol.core.quotes import SmileQuote
from fxvol.surface.surface import VolSurface


def _demo_surface():
    quotes = [
        SmileQuote(T=0.25, atm_vol=0.10, rr_25=-0.01, bf_25=0.002, rr_10=-0.018, bf_10=0.006),
        SmileQuote(T=0.50, atm_vol=0.105, rr_25=-0.011, bf_25=0.0022, rr_10=-0.02, bf_10=0.0065),
        SmileQuote(T=1.00, atm_vol=0.11, rr_25=-0.012, bf_25=0.0025, rr_10=-0.022, bf_10=0.007),
    ]
    return VolSurface.from_quotes(quotes, S=1.10, r_d=0.03, r_f=0.01)


def test_surface_builds_and_reprices_atm():
    surf = _demo_surface()
    iv = surf.implied_vol(1.10, 0.5)
    assert 0.08 < iv < 0.14


def test_no_calendar_arb():
    assert _demo_surface().no_calendar_arb(k=0.0)


def test_per_tenor_forwards_no_double_count():
    """With a real FX forward term structure (forwards differing a lot across
    tenors), reading a smile at a single surface-wide forward doubles long-tenor
    vols. The surface must use each tenor's OWN forward. Regression for the
    USDCNH validation bug."""
    import numpy as np
    from fxvol.surface.smile import fit_svi
    from fxvol.surface.surface import VolSurface

    # Two tenors with very different forwards (like USDCNH 6M vs 5Y).
    spot = 7.2605
    fwds = [7.16, 6.54]
    tenors = [0.5, 5.0]
    smiles = []
    for T, F in zip(tenors, fwds):
        # flat 5% smile in this tenor's own moneyness
        k = np.linspace(-0.2, 0.2, 5)
        tv = np.full(5, 0.05 ** 2 * T)
        smiles.append(fit_svi(k, tv))
    surf = VolSurface(spot, 0.0, 0.0, tenors, smiles, forwards=fwds)
    # At each tenor's ATM (== its forward), implied vol must be ~5%, not ~10%.
    assert abs(surf.implied_vol(fwds[1], 5.0) - 0.05) < 0.005
    assert abs(surf.implied_vol(fwds[0], 0.5) - 0.05) < 0.005
