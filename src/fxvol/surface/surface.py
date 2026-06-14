"""Full 2D volatility surface: SVI smiles tiled across tenors, plus no-arb checks.

A surface is a list of per-tenor SVI smiles plus an interpolation rule across
tenors. Two arbitrage families to police:
  - butterfly (within a tenor): handled per-smile in smile.no_butterfly_arb
  - calendar (across tenors): total variance must be NON-DECREASING in T at
    fixed log-moneyness. If a longer-dated option had less total variance than
    a shorter one, you could arbitrage the calendar spread.

We interpolate in TOTAL VARIANCE linearly in T (the standard choice - it keeps
calendar-arbitrage checks simple and is what keeps forward variance positive).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..core.conventions import forward
from ..core.quotes import SmileQuote, quotes_to_strikevols
from .smile import SVIParams, fit_svi


@dataclass
class VolSurface:
    spot: float
    r_dom: float
    r_for: float
    tenors: list[float]
    smiles: list[SVIParams]
    forwards: list[float] | None = None   # per-tenor forwards (FX term structure)

    @classmethod
    def from_quotes(cls, quotes: list[SmileQuote], S: float,
                    r_d: float, r_f: float) -> "VolSurface":
        tenors, smiles = [], []
        for q in sorted(quotes, key=lambda x: x.T):
            svs = quotes_to_strikevols(q, S, r_d, r_f)
            F = forward(S, q.T, r_d, r_f)
            k = np.array([np.log(sv.strike / F) for sv in svs])
            tv = np.array([sv.vol**2 * q.T for sv in svs])
            smiles.append(fit_svi(k, tv))
            tenors.append(q.T)
        return cls(S, r_d, r_f, tenors, smiles)

    def _forward_at(self, T: float) -> float:
        """Forward at tenor T. Uses per-tenor forwards if supplied (a real FX
        term structure where forwards vary a lot across tenors), else falls back
        to the single-rate forward (flat-rate case)."""
        if self.forwards is not None:
            # interpolate the forward curve linearly in T
            ts = self.tenors
            if T <= ts[0]:
                return self.forwards[0]
            if T >= ts[-1]:
                return self.forwards[-1]
            i = np.searchsorted(ts, T)
            t0, t1 = ts[i - 1], ts[i]
            f0, f1 = self.forwards[i - 1], self.forwards[i]
            return f0 + (f1 - f0) * (T - t0) / (t1 - t0)
        return forward(self.spot, T, self.r_dom, self.r_for)

    def total_variance(self, K: float, T: float) -> float:
        # Each smile was fitted in ITS OWN forward's log-moneyness. With a real
        # FX forward term structure the forward moves a lot across tenors, so the
        # same strike K is at different moneyness per tenor. We therefore read
        # each bracketing smile at the moneyness implied by THAT tenor's forward,
        # not a single surface-wide forward (which doubles long-tenor vols when
        # forwards differ materially - a real bug under non-flat rates).
        if T <= self.tenors[0]:
            k = np.log(K / self._forward_at(self.tenors[0]))
            return float(self.smiles[0].total_variance(np.array([k]))[0])
        if T >= self.tenors[-1]:
            k = np.log(K / self._forward_at(self.tenors[-1]))
            return float(self.smiles[-1].total_variance(np.array([k]))[0])
        i = np.searchsorted(self.tenors, T)
        t0, t1 = self.tenors[i - 1], self.tenors[i]
        k0 = np.log(K / self._forward_at(t0))
        k1 = np.log(K / self._forward_at(t1))
        w0 = self.smiles[i - 1].total_variance(np.array([k0]))[0]
        w1 = self.smiles[i].total_variance(np.array([k1]))[0]
        frac = (T - t0) / (t1 - t0)
        return float(w0 + frac * (w1 - w0))

    def implied_vol(self, K: float, T: float) -> float:
        return float(np.sqrt(self.total_variance(K, T) / T))

    def no_calendar_arb(self, k: float = 0.0) -> bool:
        """Total variance non-decreasing in T at fixed log-moneyness."""
        prev = -np.inf
        for T, smile in zip(self.tenors, self.smiles):
            w = smile.total_variance(np.array([k]))[0]
            if w < prev - 1e-8:
                return False
            prev = w
        return True
