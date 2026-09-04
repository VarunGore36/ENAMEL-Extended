"""Plain-text report over a run record: leaderboard, levels, and stability.

Every number here is recomputed from the recorded times, so the report cannot
disagree with the record. See docs/decisions/0006-run-record.md.
"""

from __future__ import annotations

from typing import Sequence

from enamel_ext.data.naming import rename
from enamel_ext.data.published import COLUMNS, table
from enamel_ext.pipeline.record import Environment, RunRecord
from enamel_ext.report.hyperparams import attainable_range, eff_at_h, reorderable_pairs
from enamel_ext.report.levels import (
    PAPER_SLOWDOWNS,
    describe_levels,
    limit_level_counts,
    sensitivity_shares,
)
from enamel_ext.report.parity import compare, format_parity
from enamel_ext.report.stats import (
    bootstrap_ci,
    kendall_tau,
    paired_bootstrap_diff_ci,
    paired_sign_test,
)

__all__ = ["ALPHA_SWEEP", "format_summary"]

#: Alphas reported alongside the measured one, filtered to those <= it.
ALPHA_SWEEP = (1.25, 1.5, 2.0, 3.0, 4.0)

#: Beyond this many models, only adjacent leaderboard pairs are tested.
_ALL_PAIRS_LIMIT = 6


def _num(x: float, places: int = 3) -> str:
    return f"{x:.{places}f}"


def _table(
    header: Sequence[str], rows: Sequence[Sequence[str]], indent: str = "  "
) -> list[str]:
    """Left-align the first column, right-align the rest."""
    if not rows:
        return []
    widths = [
        max(len(header[i]), max(len(row[i]) for row in rows)) for i in range(len(header))
    ]

    def line(cells: Sequence[str]) -> str:
        out = [cells[0].ljust(widths[0])]
        out += [cells[i].rjust(widths[i]) for i in range(1, len(cells))]
        return (indent + "  ".join(out)).rstrip()

    return [line(header)] + [line(row) for row in rows]

def _machine(env: Environment) -> str:
    load = (
        "load " + "/".join(_num(x, 2) for x in env.load_average) + " at start"
        if env.load_average
        else "load unknown"
    )
    return f"{env.python}, {env.platform}, {env.cpu_count} cores, {load}"


def _header(record: RunRecord) -> list[str]:
    out = [
        "ENAMEL-Extended run",
        f"  measured {record.started} .. {record.finished}",
        f"  {_machine(record.environment)}",
        f"  alpha {record.metric.alpha}, h {record.metric.level_weights}, "
        f"{record.metric.normalization} normalization, R = {record.repeats}, "
        f"{record.aggregator}",
        f"  problems: {len(record.problems)} scored, {len(record.failures)} without a reference",
        f"  data: {record.data.name} ({record.data.url}) {record.data_fingerprint[:12]}",
        f"  solutions: {record.solutions.name} ({record.solutions.url}) "
        f"{record.solutions_fingerprint[:12]}",
    ]
    if record.resumed:
        out.append(f"  sessions: {len(record.segments)}")
        for index, segment in enumerate(record.segments, start=1):
            count = len(segment.problem_ids)
            out.append(
                f"    {index}. {count} problem{'' if count == 1 else 's'}, "
                f"{segment.started} .. {segment.finished}"
            )
            out.append(f"       {_machine(segment.environment)}")
    caveats = record.caveats()
    if caveats:
        out.append("  caveats:")
        out += [f"    - {c}" for c in caveats]
    if record.failures:
        out.append("  no reference measured for problems: " + ", ".join(
            str(pid) for pid, _ in record.failures
        ))
    return out


def _leaderboard(
    record: RunRecord, k: int, level: float, resamples: int, seed: int
) -> tuple[list[str], list[tuple[str, float, float]]]:
    """The table, and (model, eff, pass) for the sections that follow it."""
    rows = []
    ranked: list[tuple[str, float, float]] = []
    for model in record.models:
        per_problem = record.per_problem_eff(model, k)
        interval = bootstrap_ci(
            per_problem, resamples=resamples, level=level, seed=seed
        )
        passes = record.pass_at_k(model, k)
        ranked.append((model, interval.point, passes))
        rows.append(
            [
                model,
                _num(interval.point),
                f"[{_num(interval.lo)}, {_num(interval.hi)}]",
                _num(passes),
                str(len(per_problem)),
                str(record.incorrect_samples(model)),
                str(record.censored_samples(model)),
            ]
        )
    order = sorted(range(len(ranked)), key=lambda i: -ranked[i][1])
    header = [
        "model",
        f"eff@{k}",
        f"{level:.0%} CI",
        f"pass@{k}",
        "problems",
        "wrong",
        "censored",
    ]
    lines = [f"Leaderboard ({resamples} bootstrap resamples over problems, seed {seed})"]
    lines += _table(header, [rows[i] for i in order])
    return lines, [ranked[i] for i in order]

def _levels(record: RunRecord, slowdowns: Sequence[float]) -> list[str]:
    times = [p.timed_reference() for p in record.problems]
    if not times:
        return []
    summaries = describe_levels(times, alpha=record.metric.alpha, slowdowns=slowdowns)
    shares = sensitivity_shares(
        [s.q_median for s in summaries], record.metric.level_weights, record.metric.alpha
    )
    header = ["level", "q median", "q range", "tolerated", "share"]
    header += [f"f at {s:g}x" for s in slowdowns]
    rows = []
    for summary, share in zip(summaries, shares):
        rows.append(
            [
                str(summary.level),
                _num(summary.q_median),
                f"{_num(summary.q_min)} - {_num(summary.q_max)}",
                _num(summary.tolerated, 1) + "x",
                f"{share:.0%}",
                *[_num(summary.fractions[s]) for s in slowdowns],
            ]
        )
    counts = limit_level_counts(times)
    where = ", ".join(f"level {lvl} in {n}" for lvl, n in sorted(counts.items()))
    return [
        "Level discrimination (q = this level's worst reference case over the largest)",
        *_table(header, rows),
        f"  T_i is set by {where} of {len(times)} problems",
        "  tolerated is the slowdown at which the level first scores 0 at the median q;",
        "  share is the level's part of the score's response to a uniform slowdown",
    ]


def _alpha_sweep(record: RunRecord, k: int, alphas: Sequence[float]) -> list[str]:
    usable = sorted({a for a in alphas if 1.0 < a <= record.metric.alpha})
    if len(usable) < 2:
        return []
    rows = []
    for model in record.models:
        cells = [model]
        for alpha in usable:
            cells.append(_num(record.eff_at_k(model, k, alpha=alpha)))
        rows.append(cells)
    return [
        f"Time limit sensitivity (eff@{k} rescored from the same measurements)",
        *_table(["model", *[f"alpha {a}" for a in usable]], rows),
        f"  alpha above the measured {record.metric.alpha} needs a new run: a censored "
        "sample has no observed time",
    ]

def _hardness(record: RunRecord, ranked: Sequence[tuple[str, float, float]]) -> list[str]:
    """Level means, and whether any pair swaps under another ``h``.

    An ``eff@1`` statement: above ``k = 1`` the level means no longer determine
    the score.
    """
    ids = record.aligned_ids()
    if not ids:
        return ["Hardness stability: models share no problems, so h cannot be compared"]
    means = {model: record.level_means(model, ids=ids) for model, _, _ in ranked}
    rows = []
    for model, _, _ in ranked:
        f = means[model]
        lo, hi = attainable_range(f)
        try:
            at_h = _num(eff_at_h(f, record.metric.level_weights))
        except ValueError:  # eff_at_h needs every h_l > 0, MetricConfig allows 0
            at_h = "n/a"
        rows.append(
            [
                model,
                *[_num(x) for x in f],
                at_h,
                f"{_num(lo)} - {_num(hi)}",
            ]
        )
    header = [
        "model",
        *[f"F_{i + 1}" for i in range(record.metric.n_levels)],
        "eff@1",
        "reachable by h",
    ]
    out = [
        f"Hardness weights (level means over the {len(ids)} shared problems)",
        *_table(header, rows),
    ]
    pairs = reorderable_pairs(means)
    if not pairs:
        out.append("  no pair swaps under any admissible h")
    for a, b, cmp in pairs:
        out.append(f"  {a} vs {b}: reorderable, h={cmp.witness_a} favours {a}, "
                   f"h={cmp.witness_b} favours {b}")
    return out

def _pairs_to_test(models: Sequence[str]) -> tuple[list[tuple[str, str]], str]:
    if len(models) <= _ALL_PAIRS_LIMIT:
        pairs = [(a, b) for i, a in enumerate(models) for b in models[i + 1 :]]
        return pairs, "every pair"
    return list(zip(models, models[1:])), "adjacent pairs only"


def _comparisons(
    record: RunRecord,
    ranked: Sequence[tuple[str, float, float]],
    k: int,
    level: float,
    resamples: int,
    seed: int,
) -> list[str]:
    """Paired differences on the problems both models answered."""
    names = [model for model, _, _ in ranked]
    if len(names) < 2:
        return []
    pairs, which = _pairs_to_test(names)
    rows = []
    for a, b in pairs:
        ids = record.aligned_ids([a, b])
        if not ids:
            rows.append([f"{a} - {b}", "-", "-", "-", "0"])
            continue
        left = record.per_problem_eff(a, k, ids=ids)
        right = record.per_problem_eff(b, k, ids=ids)
        diff = paired_bootstrap_diff_ci(
            left, right, resamples=resamples, level=level, seed=seed
        )
        p = paired_sign_test(left, right, resamples=resamples, seed=seed)
        rows.append(
            [
                f"{a} - {b}",
                f"{diff.point:+.3f}",
                f"[{diff.lo:+.3f}, {diff.hi:+.3f}]",
                f"{p:.3f}",
                str(len(ids)),
            ]
        )
    out = [
        f"Paired comparisons (eff@{k} difference, {which}, sign-flip p)",
        *_table(["pair", "diff", f"{level:.0%} CI", "p", "problems"], rows),
    ]
    try:
        tau = kendall_tau([e for _, e, _ in ranked], [p for _, _, p in ranked])
    except ValueError:
        return out
    out.append(f"  Kendall tau-b between the eff@{k} and pass@{k} orderings: {_num(tau)}")
    return out

def _parity(record: RunRecord, k: int, name: str) -> list[str]:
    """Empty only when no run model resolves to a published one and none looks like it.

    Names are resolved first, so a run whose models are spelled the way their
    source release spelled them is still compared. A name that looks like a
    published model but did not match is printed rather than passed over, since
    silence there is indistinguishable from having nothing to compare.
    """
    if f"eff{k}" not in COLUMNS:
        return []
    eff, report = rename({m: record.eff_at_k(m, k) for m in record.models})
    passes, _ = rename({m: record.pass_at_k(m, k) for m in record.models})
    published = set(table(name))
    if not published & set(eff) and not report.suspect:
        return []
    out = format_parity(compare(eff, passes, name=name, k=k))
    for model, candidates in report.suspect.items():
        out.append(f"    unmatched {model}: did you mean {', '.join(candidates)}?")
    return out


def format_summary(
    record: RunRecord,
    *,
    k: int = 1,
    level: float = 0.95,
    resamples: int = 10_000,
    seed: int = 0,
    alphas: Sequence[float] = ALPHA_SWEEP,
    slowdowns: Sequence[float] = PAPER_SLOWDOWNS,
    parity_table: str = "greedy",
) -> str:
    """The whole report. Sections with nothing to say are left out."""
    sections = [_header(record)]
    if record.problems and record.models:
        board, ranked = _leaderboard(record, k, level, resamples, seed)
        sections.append(board)
        sections.append(_levels(record, slowdowns))
        sections.append(_alpha_sweep(record, k, alphas))
        sections.append(_hardness(record, ranked))
        sections.append(_comparisons(record, ranked, k, level, resamples, seed))
        sections.append(_parity(record, k, parity_table))
    else:
        sections.append(["Nothing was scored."])
    return "\n\n".join("\n".join(s) for s in sections if s) + "\n"
