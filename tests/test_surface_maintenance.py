"""Tests for the daily surface recalibration and quote quality pipeline."""
from __future__ import annotations

from fxvol.pricing.surface.maintenance import RecalibrationPipeline, StalenessDetector
from fxvol.data.synthetic.g10 import simulate_quote_history


def test_clean_history_has_no_flags_and_passes_arb_checks():
    history = simulate_quote_history("EURUSD", n_days=15, seed=1,
                                     daily_noise_std=0.0008)
    result = RecalibrationPipeline().run(history)
    assert len(result.records) == 15
    assert all(r.butterfly_arb_ok for r in result.records)
    assert all(r.calendar_arb_ok for r in result.records)
    assert all(not r.staleness_flags for r in result.records)


def test_frozen_quote_is_detected():
    history = simulate_quote_history("EURUSD", n_days=15, seed=2,
                                     daily_noise_std=0.0008,
                                     freeze_days={8: 0})   # freeze the 1M tenor on day 8
    result = RecalibrationPipeline().run(history)
    day8 = next(r for r in result.records if r.day == 8)
    assert any(f.kind == "frozen" for f in day8.staleness_flags)


def test_outlier_quote_is_detected():
    history = simulate_quote_history("EURUSD", n_days=20, seed=3,
                                     daily_noise_std=0.0008,
                                     outlier_days={12: (1, 0.05)})   # big shock, 3M tenor
    result = RecalibrationPipeline().run(history)
    day12 = next(r for r in result.records if r.day == 12)
    assert any(f.kind == "outlier" for f in day12.staleness_flags)


def test_fit_rmse_is_near_zero_for_a_well_behaved_surface():
    history = simulate_quote_history("EURUSD", n_days=5, seed=4,
                                     daily_noise_std=0.0008)
    result = RecalibrationPipeline().run(history)
    # SVI fits its own five anchor points almost exactly by construction, so
    # this is really a check that the RMSE computation itself is wired up
    # correctly, not a claim about real market fit quality.
    assert all(r.fit_rmse_bp < 5.0 for r in result.records)


def test_detector_needs_two_days_before_it_can_flag_anything():
    history = simulate_quote_history("EURUSD", n_days=1, seed=5)
    flags = StalenessDetector().check(history)
    assert flags == []
