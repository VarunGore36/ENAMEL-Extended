"""Measurement backends and repeat aggregation.

Everything that touches a clock lives here so that the metric layer stays pure
and testable, and so the timing backend can be swapped for a deterministic one
(instruction counts) without any change to scoring.
"""

from enamel_ext.measure.timing import AGGREGATORS, aggregate_repeats, hodges_lehmann

__all__ = ["AGGREGATORS", "aggregate_repeats", "hodges_lehmann"]
