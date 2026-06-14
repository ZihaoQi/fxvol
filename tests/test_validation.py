"""Bloomberg validation harness: structure + self-consistency tests."""

from fxvol.core.conventions import (
    ATMConvention, atm_strike, delta_type_for_tenor, strike_from_delta,
)
from fxvol.core.quotes import SmileQuote
from fxvol.validation.bloomberg import (
    BloombergTarget, DELTA_POINTS, MarketData, build_surface, compare,
)


def _synthetic_md():
    S = 7.2605
    rates = {0.5: (0.018, 0.026), 1.0: (0.019, 0.025), 1.5: (0.020, 0.024)}
    quotes = {
        0.5: SmileQuote(0.5, 0.050, -0.004, 0.0012, -0.008, 0.004),
        1.0: SmileQuote(1.0, 0.054, -0.005, 0.0014, -0.010, 0.0048),
        1.5: SmileQuote(1.5, 0.057, -0.006, 0.0016, -0.011, 0.0052),
    }
    return MarketData(spot=S, rates=rates, quotes=quotes)


def test_self_consistency_zero_error():
    """If Bloomberg target == our own back-out, errors must be ~0.
    This proves the comparison plumbing is correct (no systematic offset)."""
    md = _synthetic_md()
    surf = build_surface(md)
    tgt = BloombergTarget()
    T = 1.0
    r_d, r_f = md.rates[T]
    dt = delta_type_for_tenor(T)
    for label, delta, is_call in DELTA_POINTS:
        if label in ("5DP", "5DC", "15DP", "15DC", "35DP", "35DC"):
            continue
        if label == "ATM":
            K = atm_strike(md.spot, T, r_d, r_f, md.quotes[T].atm_vol,
                           ATMConvention.DNS, True)
        else:
            v = md.quotes[T].atm_vol
            for _ in range(8):
                K = strike_from_delta(delta, md.spot, T, r_d, r_f, v, is_call, dt)
                v = surf.implied_vol(K, T)
        tgt.grid.setdefault(T, {})[label] = (K, surf.implied_vol(K, T))
    res = compare(md, tgt)
    assert res.strike_err_total < 1e-6
    assert res.vol_err_total < 1e-6


def test_tenor_delta_switch():
    """Convention must switch spot->forward delta across the 1Y boundary."""
    assert delta_type_for_tenor(0.5).value.startswith("spot")
    assert delta_type_for_tenor(1.0).value.startswith("forward")
    assert delta_type_for_tenor(2.0).value.startswith("forward")
