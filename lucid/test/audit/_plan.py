"""Which stages a command line runs, and over which packages.

Two independent things are being chosen and they used to be tangled in one
function: *what* (the packages — :mod:`~lucid.test.audit._packages`) and
*which questions to ask of it* (the stages).  They stay orthogonal here:

``--audit-only`` / ``--tests-only`` / ``--doctests-only``
    select those stages and **exclude the zoo stage**.  The zoo is the
    slowest thing the audit does; running it must be something a command
    line says, not something a stage flag happens to leave in.
``--models``
    is the zoo stage, and only that unless other packages are named too.
    Combining it with a stage flag contradicts itself — a stage flag
    narrows to the sweep or the suite, ``--models`` asks for neither.

Contradictions are a :class:`PlanError` and the exit status is ``2``: the
command line was wrong, which is not the framework failing (``1``).
"""

import argparse
from dataclasses import dataclass
from pathlib import Path

from lucid.test.audit._packages import (
    DEFAULT_SUITE_PATH,
    MODELS,
    PACKAGES,
    PackageRegistry,
    PackageSelectionError,
    Selection,
)

_ROOT = Path(__file__).resolve().parents[3]


class PlanError(ValueError):
    """The command line asks for something that cannot be run as stated."""


@dataclass(frozen=True)
class RunPlan:
    """The stages one run executes, in the order it executes them."""

    selection: Selection
    self_check: bool
    sweep: bool
    suite: bool
    doctests: bool
    zoo: bool

    @classmethod
    def from_args(cls, args: argparse.Namespace, registry: PackageRegistry = PACKAGES) -> RunPlan:
        """Resolve ``args`` into a plan, or raise :class:`PlanError`."""
        stage_only = (args.audit_only, args.tests_only, args.doctests_only)
        if sum(stage_only) > 1:
            raise PlanError(
                "--audit-only, --tests-only and --doctests-only exclude each other; "
                "pass none to run everything"
            )
        try:
            selection = registry.select(args)
        except PackageSelectionError as exc:
            raise PlanError(str(exc)) from exc
        if any(stage_only) and getattr(args, registry[MODELS].dest):
            raise PlanError(
                "--models is the model-zoo stage; --audit-only, --tests-only and "
                "--doctests-only select other stages. Pass one or the other"
            )

        # Both stages by default: a gate that has to be invoked twice is a gate
        # that gets invoked once.  The doctest stage rides with the suite, and
        # alone under --doctests-only — CI's sweep ran with --audit-only, which
        # skipped the examples, and 63 of them shipped without ever running.
        do_audit = not (args.tests_only or args.doctests_only)
        do_suite = not (args.audit_only or args.doctests_only)
        sweep = do_audit and selection.has_code
        plan = cls(
            selection=selection,
            self_check=sweep and not args.no_self_check,
            sweep=sweep,
            suite=do_suite and selection.has_code,
            doctests=(do_suite or args.doctests_only)
            and selection.has_code
            and not args.no_doctests,
            zoo=not any(stage_only) and selection.runs_zoo,
        )
        _require_whole_run(args, plan)
        _reject_stray_zoo_options(args, plan)
        _reject_ignored_options(args, plan)
        return plan


def _sweep_is_narrowed(args: argparse.Namespace, selection: Selection) -> bool:
    """Whether the sweep this run would do is less than the whole, answered cell by cell.

    ``--no-models`` is not a filter here: the sweep has no zoo stage to
    narrow, and leaving the zoo out leaves the swept surface exactly whole.
    """
    return bool(
        args.axis.strip() not in ("all", "")
        or args.select
        or args.limit
        or args.fail_fast
        or args.cpu_only
        or args.quick
        or selection.sweep_keys() is not None
    )


def _require_whole_run(args: argparse.Namespace, plan: RunPlan) -> None:
    """Every ``--update-*`` records a baseline from this run, so this run must be the whole of it.

    Only ``--update-coverage`` was guarded.  ``--fft --tests-only --update-suite``
    then wrote a 26.98% floor over the 85.92% one and exited 0, and ``--fft
    --doctests-only --update-doctests`` took the recorded example count from 6,932
    to 135: a baseline written from a narrowed run is a regression gate loosened
    without anyone deciding to.
    """
    selection = plan.selection
    if args.update_coverage and _sweep_is_narrowed(args, selection):
        raise PlanError("--update-coverage requires a complete, unfiltered sweep")
    if args.update_doctests and not selection.complete:
        raise PlanError(
            "--update-doctests requires every package's examples: a narrowed run would "
            "record its smaller count over the whole floor"
        )
    if args.update_suite:
        where = selection.suite_plan(
            _ROOT, args.suite_path, zoo_stage=plan.zoo, extra_ignore=args.suite_ignore
        )
        whole = selection.complete and where.paths == (DEFAULT_SUITE_PATH,) and not where.ignore
        if not whole:
            raise PlanError(
                "--update-suite requires the whole suite, the model zoo's tests included "
                "(this run leaves some tree out, so it would record a floor from a filtered "
                "one): use `lucid-audit --tests-only --update-suite`"
            )


def _reject_stray_zoo_options(args: argparse.Namespace, plan: RunPlan) -> None:
    """A zoo option on a run with no zoo stage is a mistake, not a no-op."""
    asked = [
        flag
        for flag, given in (
            ("--family", args.family),
            ("--zoo-record", args.zoo_record),
            ("--fresh", args.fresh),
            ("--mem-limit-gb", args.mem_limit_gb),
            ("--zoo-device", args.zoo_device),
        )
        if given
    ]
    if asked and not plan.zoo:
        raise PlanError(f"{', '.join(asked)} configure the model-zoo stage, which this run skips")


def _reject_ignored_options(args: argparse.Namespace, plan: RunPlan) -> None:
    """On a zoo-only run, an option for a stage that does not run is a mistake, not a no-op.

    ``lucid-audit --models --json report.json`` writing no report, or ``--models
    --cpu-only`` quietly changing nothing, are options that look accepted.
    """
    if not plan.zoo or plan.sweep or plan.suite or plan.doctests:
        return
    sweep = (
        ("--axis", args.axis.strip() not in ("all", "")),
        ("--select", args.select),
        ("--quick", args.quick),
        ("--cpu-only", args.cpu_only),
        ("--fail-fast", args.fail_fast),
        ("--json", args.json is not None),
        ("--no-self-check", args.no_self_check),
        ("--update-coverage", args.update_coverage),
        ("--update-known", args.update_known),
    )
    suite = (
        ("--suite-path", args.suite_path != DEFAULT_SUITE_PATH),
        ("--suite-ignore", args.suite_ignore),
        ("--no-line-coverage", args.no_line_coverage),
        ("--update-suite", args.update_suite),
        ("--no-doctests", args.no_doctests),
        ("--update-doctests", args.update_doctests),
    )
    for stage, options in (("sweep", sweep), ("suite and doctest", suite)):
        asked = [flag for flag, given in options if given]
        if asked:
            raise PlanError(
                f"{', '.join(asked)} configure the {stage} stage, which a zoo-only run skips; "
                "name the packages they are for, or drop them"
            )


__all__ = ["PlanError", "RunPlan"]
