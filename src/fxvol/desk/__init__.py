"""Desk operations built on top of the pricing library: holding a book of
positions, aggregating and stress-testing its risk, quoting a two-way market
under inventory skew, and backtesting a hedging strategy. Everything here
takes a VolSurface or a MarketState as an input; it never builds one.
"""
