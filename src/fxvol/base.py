"""Common interface every pricing model implements.

The whole point: a desk's risk system shouldn't care whether it's talking to a
Heston model or an LSV model. They expose the same two verbs - price() and
calibrate() - so they're interchangeable behind the same risk plumbing. That
substitutability is what 'production architecture' means here, not cleverness.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class MarketSnapshot:
    """Everything a model needs to know about 'today' to calibrate/price.

    Immutable on purpose: a snapshot is a fact about a moment in time. If you
    want a new market, you make a new snapshot - you never mutate one, because
    a model calibrated to a snapshot must stay paired with the exact data it saw.
    """
    spot: float
    r_dom: float          # domestic continuously-compounded rate
    r_for: float          # foreign continuously-compounded rate
    # quotes filled in once you build core/quotes.py; left open for now.


class PricingModel(ABC):
    """Abstract base every model in this repo inherits from."""

    @abstractmethod
    def calibrate(self, market: MarketSnapshot) -> None:
        """Fit model parameters to observed market data. Mutates self."""
        ...

    @abstractmethod
    def price(self, strike: float, expiry: float, is_call: bool = True) -> float:
        """Price a European vanilla. Exotics extend this per-model."""
        ...

    @property
    @abstractmethod
    def is_calibrated(self) -> bool:
        """Guard: refuse to price before calibration. Cheap, saves hours."""
        ...
