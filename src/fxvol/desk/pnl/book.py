"""Option book representation for the P&L engine."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class OptionPosition:
    """One vanilla FX option line in the book.

    notional is in FOREIGN (base) currency units; sign = long(+)/short(-).
    A market maker who SOLD a client option holds a negative position.
    """
    pair: str
    strike: float
    expiry: float          # year-fraction to expiry as of the valuation date
    is_call: bool
    notional: float        # signed; foreign-ccy notional
    label: str = ""

    def with_expiry(self, new_expiry: float) -> "OptionPosition":
        return OptionPosition(self.pair, self.strike, new_expiry, self.is_call,
                              self.notional, self.label)


@dataclass
class Book:
    positions: list[OptionPosition]

    def roll_one_day(self, day_count: float = 1 / 365) -> "Book":
        """Advance every position's expiry by one day (time decay)."""
        return Book([p.with_expiry(max(p.expiry - day_count, 1e-6))
                     for p in self.positions])
