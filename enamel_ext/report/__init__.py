"""Reporting: uncertainty, hyperparameter sensitivity, rank stability.

Everything here is post-processing over recorded per-problem or per-level
numbers. Nothing in this package executes code or touches a clock, which is why
it can be built and tested without the benchmark's data.
"""

from enamel_ext.report.hyperparams import (
    HComparison,
    attainable_range,
    compare_under_h,
    eff_at_h,
    reorderable_pairs,
    rescore_at_alpha,
)
from enamel_ext.report.stats import (
    Interval,
    bootstrap_ci,
    kendall_tau,
    paired_bootstrap_diff_ci,
    paired_sign_test,
)

__all__ = [
    "HComparison",
    "Interval",
    "attainable_range",
    "bootstrap_ci",
    "compare_under_h",
    "eff_at_h",
    "kendall_tau",
    "paired_bootstrap_diff_ci",
    "paired_sign_test",
    "reorderable_pairs",
    "rescore_at_alpha",
]
