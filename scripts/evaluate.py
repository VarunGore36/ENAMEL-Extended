"""Measure a run and report it: the entry point that ties the layers together.

Problem-major, so one reference measurement sets ``T_i`` for every model. Writes
a run record and prints a report derived from it, never the other way round. See
docs/decisions/0006-run-record.md.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from enamel_ext.data.sources import (  # noqa: E402
    problem_set_from_json,
    synthetic_problem_set,
)
from enamel_ext.measure.calibrate import probe  # noqa: E402
from enamel_ext.measure.runner import PAPER_REPEATS, RunConfig  # noqa: E402
from enamel_ext.measure.sandbox import SandboxError  # noqa: E402
from enamel_ext.metrics.score import PAPER, MetricConfig  # noqa: E402
from enamel_ext.pipeline import (  # noqa: E402
    format_summary,
    load_record,
    resume_evaluation,
    run_evaluation,
    save_record,
    selected_ids,
    solution_set_from_json,
    synthetic_solutions,
)

OK = 0
RUN_FAILURE = 1
USAGE_FAILURE = 2

def _report_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--k", type=int, default=1, help="k in eff@k and pass@k")
    parser.add_argument("--resamples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--level", type=float, default=0.95, help="confidence level")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        epilog=(
            "Candidate code runs in a subprocess under resource limits. That is "
            "isolation, not a container: do not point this at samples you have not "
            "read on a machine you care about. Exit 1: the run did not finish. "
            "Exit 2: fix the flags."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="measure, save a record, print the report")
    run.add_argument("--problems", type=Path, default=None, help="problem set JSON")
    run.add_argument("--solutions", type=Path, default=None, help="solution set JSON")
    run.add_argument("--out", type=Path, default=None, help="where to write the run record")
    run.add_argument("--no-save", action="store_true", help="do not write a record")
    run.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="extend this record: measure only what it is missing, retrying its failures",
    )
    run.add_argument(
        "--checkpoint-every",
        type=int,
        default=1,
        metavar="N",
        help="save the record-so-far every N problems; 0 to save only at the end",
    )
    run.add_argument("--models", default="", help="comma-separated subset of models")
    run.add_argument("--ids", default="", help="comma-separated subset of problem ids")
    run.add_argument("--limit", type=int, default=0, help="attempt only the first N problems")
    run.add_argument("--repeats", type=int, default=PAPER_REPEATS, help="R per test case")
    run.add_argument("--alpha", type=float, default=PAPER.alpha)
    run.add_argument("--hardness", default="", help="level weights, default 3,3,4")
    run.add_argument(
        "--keep-going",
        action="store_true",
        help="record problems whose reference fails instead of stopping",
    )
    run.add_argument("--quiet", action="store_true", help="no per-problem progress on stderr")
    run.add_argument(
        "--no-calibrate",
        action="store_true",
        help="skip the calibration probe; comparability then rests on machine strings",
    )
    _report_flags(run)

    report = sub.add_parser("report", help="print the report for an existing record")
    report.add_argument("record", type=Path)
    _report_flags(report)
    return parser


def _ints(text: str, what: str) -> tuple[int, ...]:
    if not text.strip():
        return ()
    try:
        return tuple(int(part) for part in text.split(","))
    except ValueError:
        raise ValueError(f"{what} must be comma-separated integers, got {text!r}") from None


def _floats(text: str, what: str) -> tuple[float, ...]:
    if not text.strip():
        return ()
    try:
        return tuple(float(part) for part in text.split(","))
    except ValueError:
        raise ValueError(f"{what} must be comma-separated numbers, got {text!r}") from None

def _default_out() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return Path("runs") / f"run-{stamp}.json"


def _load_inputs(args: argparse.Namespace):
    """Problem set and solution set, synthetic only when neither is given."""
    if args.problems is None:
        if args.solutions is not None:
            raise ValueError("--solutions needs --problems: the two must describe each other")
        problems = synthetic_problem_set()
        return problems, synthetic_solutions(list(problems))
    if args.solutions is None:
        raise ValueError(
            "--problems needs --solutions: the synthetic samples only answer the "
            "synthetic problem"
        )
    problems = problem_set_from_json(args.problems.read_text())
    return problems, solution_set_from_json(args.solutions.read_text())


def _run(args: argparse.Namespace) -> int:
    problems, solutions = _load_inputs(args)
    models = [m.strip() for m in args.models.split(",") if m.strip()] or None
    hardness = _floats(args.hardness, "--hardness") or PAPER.level_weights
    metric = MetricConfig(alpha=args.alpha, level_weights=hardness)
    config = RunConfig(repeats=args.repeats)

    ids = selected_ids(problems, solutions, models, _ints(args.ids, "--ids") or None)
    if args.limit > 0:
        ids = ids[: args.limit]
    if not ids:
        raise ValueError("no problem is both present in the data and answered by a model")

    on_progress = None if args.quiet else lambda msg: print(msg, file=sys.stderr)
    calibration = None
    if not args.no_calibrate:
        if on_progress is not None:
            on_progress("calibrating")
        calibration = probe(repeats=config.repeats, aggregator=config.aggregator)
        if on_progress is not None:
            on_progress(
                f"calibration resolves a differential to {calibration.resolution():.3f}"
            )
    # Resolved before measuring, because a checkpoint has to know where it lands.
    destination = None if args.no_save else _out_path(args)
    if args.resume is not None:
        record = resume_evaluation(
            load_record(args.resume),
            problems,
            solutions,
            config=config,
            metric=metric,
            models=models,
            ids=ids,
            keep_going=args.keep_going,
            on_progress=on_progress,
            checkpoint=destination,
            checkpoint_every=args.checkpoint_every,
            calibration=calibration,
        )
    else:
        record = run_evaluation(
            problems,
            solutions,
            config=config,
            metric=metric,
            models=models,
            ids=ids,
            keep_going=args.keep_going,
            on_progress=on_progress,
            checkpoint=destination,
            checkpoint_every=args.checkpoint_every,
            calibration=calibration,
        )
    if destination is not None:
        print(f"record written to {save_record(record, destination)}", file=sys.stderr)
    print(_summary(record, args), end="")
    return OK


def _out_path(args: argparse.Namespace) -> Path:
    """Resume writes back over the record it extended unless told otherwise.

    Memoized into ``args.out``: checkpointing asks before the run and the crash
    hint asks after, and a default name must not restamp itself in between.
    """
    if args.out is None:
        args.out = args.resume if args.resume is not None else _default_out()
    return args.out


def _resume_hint(args: argparse.Namespace) -> str:
    """What is salvageable after a run dies, true whether or not anything is."""
    if args.command != "run" or args.no_save or args.checkpoint_every <= 0:
        return "no record was being written as it went, so this run starts over"
    path = _out_path(args)
    if not path.is_file():
        return f"nothing reached {path} yet, so this run starts over"
    return f"what was measured is in {path}; continue it with --resume {path}"


def _summary(record, args: argparse.Namespace) -> str:
    return format_summary(
        record, k=args.k, level=args.level, resamples=args.resamples, seed=args.seed
    )

def _report(args: argparse.Namespace) -> int:
    print(_summary(load_record(args.record), args), end="")
    return OK


def _message(exc: Exception) -> str:
    """KeyError stringifies to a quoted repr; every raiser here passes a sentence."""
    if isinstance(exc, KeyError) and exc.args:
        return str(exc.args[0])
    return str(exc)


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        return _run(args) if args.command == "run" else _report(args)
    except SandboxError as exc:
        print(f"the run did not finish: {exc}", file=sys.stderr)
        print(_resume_hint(args), file=sys.stderr)
        return RUN_FAILURE
    except (ValueError, KeyError, OSError) as exc:
        print(f"error: {_message(exc)}", file=sys.stderr)
        return USAGE_FAILURE


if __name__ == "__main__":
    raise SystemExit(main())
