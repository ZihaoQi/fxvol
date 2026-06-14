"""Garman-Kohlhagen: Black-Scholes for FX.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm


@dataclass
class GKResult:
    price: float
    delta: float
    vega: float
    gamma: float


def _d1_d2(S: float, K: float, T: float, r_d: float, r_f: float, sigma: float):
    vol_sqrt_t = sigma * np.sqrt(T)
    d1 = (np.log(S / K) + (r_d - r_f + 0.5 * sigma**2) * T) / vol_sqrt_t
    d2 = d1 - vol_sqrt_t
    return d1, d2


def garman_kohlhagen(
    S: float,
    K: float,
    T: float,
    r_d: float,
    r_f: float,
    sigma: float,
    is_call: bool = True,
) -> GKResult:
    """FX option price and Greeks under Garman-Kohlhagen."""
    if T <= 0 or sigma <= 0:
        raise ValueError("T and sigma must be positive")

    d1, d2 = _d1_d2(S, K, T, r_d, r_f, sigma)
    disc_f = np.exp(-r_f * T)
    disc_d = np.exp(-r_d * T)

    if is_call:
        price = S * disc_f * norm.cdf(d1) - K * disc_d * norm.cdf(d2)
        delta = disc_f * norm.cdf(d1)
    else:
        price = K * disc_d * norm.cdf(-d2) - S * disc_f * norm.cdf(-d1)
        delta = -disc_f * norm.cdf(-d1)

    vega = S * disc_f * norm.pdf(d1) * np.sqrt(T)
    gamma = disc_f * norm.pdf(d1) / (S * sigma * np.sqrt(T))

    return GKResult(price=float(price), delta=float(delta),
                    vega=float(vega), gamma=float(gamma))


def gk_price(S, K, T, r_d, r_f, sigma, is_call=True) -> float:
    """Thin price-only wrapper: convenient for calibration inner loops."""
    return garman_kohlhagen(S, K, T, r_d, r_f, sigma, is_call).price


@dataclass
class GKGreeks:
    """Full Greek set for P&L attribution. Spot conventions throughout.

    Units chosen for a P&L-explain engine:
      delta : dPrice/dS            (per 1.0 of spot)
      gamma : d2Price/dS2
      vega  : dPrice/dSigma        (per 1.00 vol, i.e. 100 vol points)
      theta : dPrice/dt            (per year; negate for calendar decay)
      vanna : d2Price/dS/dSigma    (cross spot-vol)
      volga : d2Price/dSigma2      (vol convexity / 'vomma')
    """
    price: float
    delta: float
    gamma: float
    vega: float
    theta: float
    vanna: float
    volga: float


def gk_greeks(S, K, T, r_d, r_f, sigma, is_call=True) -> GKGreeks:
    """Full Greeks including vanna and volga, needed to close P&L attribution.

    Vanna and volga are what make a Greek P&L 'explain' actually reconcile: in
    any move where BOTH spot and vol shift (i.e. every real day), the cross term
    vanna*dS*dSigma and the vol-convexity term 0.5*volga*dSigma^2 carry P&L that
    delta/gamma/vega/theta alone cannot account for. Omit them and the residual
    is structurally non-zero on skewed books.
    """
    if T <= 0 or sigma <= 0:
        raise ValueError("T and sigma must be positive")
    d1, d2 = _d1_d2(S, K, T, r_d, r_f, sigma)
    disc_f = np.exp(-r_f * T)
    disc_d = np.exp(-r_d * T)
    sqrtT = np.sqrt(T)
    pdf_d1 = norm.pdf(d1)

    if is_call:
        price = S * disc_f * norm.cdf(d1) - K * disc_d * norm.cdf(d2)
        delta = disc_f * norm.cdf(d1)
        theta = (-S * disc_f * pdf_d1 * sigma / (2 * sqrtT)
                 + r_f * S * disc_f * norm.cdf(d1)
                 - r_d * K * disc_d * norm.cdf(d2))
    else:
        price = K * disc_d * norm.cdf(-d2) - S * disc_f * norm.cdf(-d1)
        delta = -disc_f * norm.cdf(-d1)
        theta = (-S * disc_f * pdf_d1 * sigma / (2 * sqrtT)
                 - r_f * S * disc_f * norm.cdf(-d1)
                 + r_d * K * disc_d * norm.cdf(-d2))

    gamma = disc_f * pdf_d1 / (S * sigma * sqrtT)
    vega = S * disc_f * pdf_d1 * sqrtT
    vanna = -disc_f * pdf_d1 * d2 / sigma
    volga = vega * d1 * d2 / sigma

    return GKGreeks(price=float(price), delta=float(delta), gamma=float(gamma),
                    vega=float(vega), theta=float(theta), vanna=float(vanna),
                    volga=float(volga))
