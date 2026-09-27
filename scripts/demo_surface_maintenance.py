"""Surface maintenance demo: run 30 days of quotes through the recalibration
pipeline, with a frozen quote and a bad print injected on purpose, and see
the pipeline catch both while tracking fit quality over the whole run.

Run:  python scripts/demo_surface_maintenance.py
"""
from fxvol.pricing.surface.maintenance import RecalibrationPipeline
from fxvol.data.synthetic.g10 import TENOR_LABELS, simulate_quote_history


def main() -> None:
    history = simulate_quote_history(
        "EURUSD", n_days=30, seed=11, daily_noise_std=0.0010,
        freeze_days={14: 0},              # 1M quote freezes on day 14
        outlier_days={21: (2, 0.06)})     # 6M quote gets a bad print on day 21

    result = RecalibrationPipeline().run(history)

    print(result.report())

    print("\nDay by day flags:")
    print("=" * 60)
    for record in result.records:
        if record.staleness_flags or not record.is_clean:
            for flag in record.staleness_flags:
                tenor_label = TENOR_LABELS[
                    [1 / 12, 0.25, 0.5, 1.0, 2.0].index(flag.tenor)]
                print(f"  day {record.day:>2} [{tenor_label}] {flag.kind}: "
                     f"{flag.detail}")
            if not (record.butterfly_arb_ok and record.calendar_arb_ok):
                print(f"  day {record.day:>2} arbitrage check failed "
                     f"(butterfly_ok={record.butterfly_arb_ok}, "
                     f"calendar_ok={record.calendar_arb_ok})")

    print("\nFit RMSE by day (bp):")
    print("=" * 60)
    for record in result.records:
        marker = " <-- flagged" if not record.is_clean else ""
        print(f"  day {record.day:>2}: {record.fit_rmse_bp:6.2f}{marker}")


if __name__ == "__main__":
    main()
