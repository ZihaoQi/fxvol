"""Tests for cross pair risk aggregation, concentration flags, and stress
scenarios."""
from __future__ import annotations

from fxvol.desk.pnl.book import Book, OptionPosition
from fxvol.desk.pnl.engine import MarketState
from fxvol.desk.risk.dashboard import (StressScenario, aggregate_by_pair,
                                  apply_scenario, check_concentration,
                                  run_stress_suite, standard_scenarios)
from fxvol.data.synthetic.g10 import build_surface


def _markets(pairs: list[str]) -> dict[str, MarketState]:
    out = {}
    for pair in pairs:
        s = build_surface(pair)
        out[pair] = MarketState(spot=s.spot, r_dom=s.r_dom, r_for=s.r_for,
                                surface=s)
    return out


def test_aggregate_by_pair_splits_correctly():
    markets = _markets(["EURUSD", "USDJPY"])
    book = Book([
        OptionPosition("EURUSD", markets["EURUSD"].spot, 0.25, True, 10_000_000),
        OptionPosition("USDJPY", markets["USDJPY"].spot, 0.25, True, -5_000_000),
    ])
    agg = aggregate_by_pair(book, markets)
    assert set(agg) == {"EURUSD", "USDJPY"}
    assert agg["EURUSD"].vega > 0          # long the option -> long vega
    assert agg["USDJPY"].vega < 0          # short the option -> short vega


def test_concentration_flag_fires_on_single_pair_book():
    # USDJPY carries far more vega per unit notional than EURUSD at this spot
    # level (higher spot, so a small notional still produces large vega), so
    # the notionals below are chosen to actually concentrate vega in EURUSD,
    # not just to look concentrated in notional terms.
    markets = _markets(["EURUSD", "USDJPY"])
    book = Book([
        OptionPosition("EURUSD", markets["EURUSD"].spot, 0.25, True, 50_000_000),
        OptionPosition("USDJPY", markets["USDJPY"].spot, 0.25, True, 10_000),
    ])
    flags = check_concentration(book, markets, pair_threshold=0.6)
    assert any(f.category == "pair" and f.name == "EURUSD" for f in flags)


def test_concentration_flag_silent_on_balanced_book():
    # Notionals scaled so the two pairs contribute roughly equal vega, not
    # equal notional (see note above).
    markets = _markets(["EURUSD", "USDJPY"])
    book = Book([
        OptionPosition("EURUSD", markets["EURUSD"].spot, 0.25, True, 10_000_000),
        OptionPosition("USDJPY", markets["USDJPY"].spot, 0.25, True, 73_000),
    ])
    flags = check_concentration(book, markets, pair_threshold=0.6)
    assert not any(f.category == "pair" for f in flags)


def test_spot_down_scenario_hurts_long_call_book():
    markets = _markets(["EURUSD"])
    m = markets["EURUSD"]
    book = Book([OptionPosition("EURUSD", m.spot, 0.25, True, 10_000_000)])
    pnl = apply_scenario(book, markets,
                         StressScenario("down", spot_mult={"EURUSD": 0.95}))
    assert pnl < 0


def test_correlation_break_scenario_present_for_multi_pair_book():
    markets = _markets(["EURUSD", "USDJPY"])
    scenarios = standard_scenarios(list(markets))
    assert any("Correlation break" in s.name for s in scenarios)


def test_stress_suite_report_lists_worst_scenario_first():
    markets = _markets(["EURUSD", "USDJPY"])
    book = Book([
        OptionPosition("EURUSD", markets["EURUSD"].spot, 0.25, True, 10_000_000),
        OptionPosition("USDJPY", markets["USDJPY"].spot, 0.25, False, 10_000_000),
    ])
    report = run_stress_suite(book, markets)
    worst_name = min(report.results, key=lambda name: report.results[name])
    text_lines = [ln for ln in report.report().splitlines() if ln.strip()][2:]
    assert text_lines[0].startswith(worst_name)
