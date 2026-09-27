"""Daily vol surface recalibration, quote quality checks, and fit tracking.

WHY A SURFACE NEEDS MAINTENANCE, NOT JUST CONSTRUCTION

surface.surface.VolSurface.from_quotes builds a surface once, from one set of
quotes. A real desk repeats that process every single day, off a live feed,
and a live feed fails in ways a one off calibration exercise never has to
deal with:

  stale quotes  : a broker screen freezes and keeps reporting yesterday's
                  number, unchanged, while the rest of the market moves.
  outlier quotes: a fat fingered print or a feed glitch produces a number
                  that is simply wrong, and calibrating straight through it
                  distorts the whole smile for that tenor.
  quiet drift    : the parametric fit that tracked the market closely last
                  month slowly stops tracking it as well, which is a signal
                  worth watching even when no single day looks alarming.

This module treats "build today's surface" as one step in a pipeline that
also screens the incoming quotes, calibrates, checks the result for
arbitrage, scores how well the fit reproduces the input quotes, and keeps a
running health record across days so a slow drift is visible even when no
single day would have tripped an alarm on its own.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..core.quotes import SmileQuote, quotes_to_strikevols
from .smile import no_butterfly_arb
from .surface import VolSurface


@dataclass(frozen=True)
class DailyQuotes:
    """One day's full quote set for one pair: one SmileQuote per tenor."""
    day: int
    pair: str
    spot: float
    r_dom: float
    r_for: float
    quotes: list[SmileQuote]


@dataclass(frozen=True)
class StalenessFlag:
    day: int
    tenor: float
    kind: str          # 'frozen' or 'outlier'
    detail: str

    def __str__(self) -> str:
        return f"day {self.day}, T={self.tenor:.3f}: {self.kind} ({self.detail})"


class StalenessDetector:
    """Screens one day's ATM quotes against a rolling window of recent daily
    changes, per tenor.

    frozen  : today's ATM vol is IDENTICAL to yesterday's for that tenor,
              bit for bit. A live market almost never prints exactly flat;
              an identical repeat is what a frozen feed looks like.
    outlier : today's day over day change is more than `z_threshold` rolling
              standard deviations away from the recent mean change. The
              window is the trailing `window` days BEFORE today, so the
              threshold reflects how much that tenor normally moves, not a
              single fixed vol-point cutoff that would be too tight for a
              volatile pair and too loose for a calm one.
    """

    def __init__(self, window: int = 10, z_threshold: float = 4.0):
        self.window = window
        self.z_threshold = z_threshold

    def check(self, history: list[DailyQuotes]) -> list[StalenessFlag]:
        """Flags for the LAST day in `history`, using the days before it as
        context. Returns an empty list until enough history has accumulated
        to judge an outlier (`window` + 1 days)."""
        if len(history) < 2:
            return []
        today = history[-1]
        yesterday = history[-2]
        flags: list[StalenessFlag] = []

        for tenor_idx, q_today in enumerate(today.quotes):
            q_yday = yesterday.quotes[tenor_idx]
            if q_today.atm_vol == q_yday.atm_vol:
                flags.append(StalenessFlag(
                    today.day, q_today.T, "frozen",
                    f"ATM unchanged at {q_today.atm_vol:.4f}"))
                continue

            if len(history) >= self.window + 2:
                window_days = history[-(self.window + 2):-1]
                changes = np.array([
                    window_days[i + 1].quotes[tenor_idx].atm_vol
                    - window_days[i].quotes[tenor_idx].atm_vol
                    for i in range(len(window_days) - 1)
                ])
                mu, sigma = float(np.mean(changes)), float(np.std(changes))
                today_change = q_today.atm_vol - q_yday.atm_vol
                if sigma > 1e-9:
                    z = abs(today_change - mu) / sigma
                    if z > self.z_threshold:
                        flags.append(StalenessFlag(
                            today.day, q_today.T, "outlier",
                            f"ATM moved {today_change:+.4f}, z={z:.1f} vs "
                            f"trailing {self.window}d"))
        return flags


@dataclass
class DailyHealthRecord:
    day: int
    fit_rmse_bp: float
    butterfly_arb_ok: bool
    calendar_arb_ok: bool
    staleness_flags: list[StalenessFlag] = field(default_factory=list)

    @property
    def is_clean(self) -> bool:
        return (self.butterfly_arb_ok and self.calendar_arb_ok
                and not self.staleness_flags)


def _fit_rmse_bp(quotes: list[SmileQuote], surface: VolSurface) -> float:
    """RMSE, in vol basis points, between the input anchor quotes and what the
    calibrated surface reproduces at those same strikes. This is the number
    that catches quiet drift: a surface can pass every arbitrage check and
    still be fitting the market progressively worse."""
    errors = []
    for q in quotes:
        for sv in quotes_to_strikevols(q, surface.spot, surface.r_dom, surface.r_for):
            model_vol = surface.implied_vol(sv.strike, q.T)
            errors.append((model_vol - sv.vol) * 1e4)
    return float(np.sqrt(np.mean(np.square(errors))))


def _surface_arb_ok(surface: VolSurface) -> tuple[bool, bool]:
    """(butterfly_ok, calendar_ok) across the whole surface."""
    butterfly_ok = True
    for smile in surface.smiles:
        k_grid = np.linspace(-2.0, 2.0, 200)
        if not no_butterfly_arb(smile, k_grid):
            butterfly_ok = False
            break
    calendar_ok = surface.no_calendar_arb()
    return butterfly_ok, calendar_ok


@dataclass
class SurfaceHealthHistory:
    records: list[DailyHealthRecord] = field(default_factory=list)
    surfaces: dict[int, VolSurface] = field(default_factory=dict)

    def report(self) -> str:
        n = len(self.records)
        clean_days = sum(1 for r in self.records if r.is_clean)
        arb_breaks = sum(1 for r in self.records
                         if not (r.butterfly_arb_ok and r.calendar_arb_ok))
        stale_events = sum(len(r.staleness_flags) for r in self.records)
        avg_rmse = float(np.mean([r.fit_rmse_bp for r in self.records])) if n else 0.0
        worst = max(self.records, key=lambda r: r.fit_rmse_bp) if n else None

        lines = [
            "Surface maintenance history",
            "=" * 40,
            f"days processed      : {n}",
            f"clean days          : {clean_days} ({clean_days / n:.0%})" if n else "clean days          : 0",
            f"arbitrage breaks    : {arb_breaks}",
            f"staleness events    : {stale_events}",
            f"average fit RMSE    : {avg_rmse:.2f} bp",
        ]
        if worst is not None:
            lines.append(f"worst fit day       : day {worst.day} "
                        f"({worst.fit_rmse_bp:.2f} bp)")
        return "\n".join(lines)


class RecalibrationPipeline:
    """Runs the daily maintenance loop: screen quotes, calibrate, check for
    arbitrage, score the fit, and accumulate a health history."""

    def __init__(self, detector: StalenessDetector | None = None):
        self.detector = detector or StalenessDetector()

    def run(self, daily_quotes: list[DailyQuotes]) -> SurfaceHealthHistory:
        history_out = SurfaceHealthHistory()
        seen: list[DailyQuotes] = []
        for day_quotes in daily_quotes:
            seen.append(day_quotes)
            flags = self.detector.check(seen)

            surface = VolSurface.from_quotes(
                day_quotes.quotes, S=day_quotes.spot, r_d=day_quotes.r_dom,
                r_f=day_quotes.r_for)
            butterfly_ok, calendar_ok = _surface_arb_ok(surface)
            rmse = _fit_rmse_bp(day_quotes.quotes, surface)

            history_out.records.append(DailyHealthRecord(
                day=day_quotes.day, fit_rmse_bp=rmse,
                butterfly_arb_ok=butterfly_ok, calendar_arb_ok=calendar_ok,
                staleness_flags=flags))
            history_out.surfaces[day_quotes.day] = surface

        return history_out
