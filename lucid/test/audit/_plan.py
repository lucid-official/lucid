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

from lucid.test.audit._packages import (
    MODELS,
    PACKAGES,
    PackageRegistry,
    PackageSelectionError,
    Selection,
)


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
        _require_complete_sweep(args, selection)

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
        _reject_stray_zoo_options(args, plan)
        return plan


def _require_complete_sweep(args: argparse.Namespace, selection: Selection) -> None:
    """``--update-coverage`` records the whole answered set, so it needs all of it.

    ``--no-models`` is not a filter here: the sweep has no zoo stage to
    narrow, and leaving the zoo out leaves the swept surface exactly whole.
    """
    if not args.update_coverage:
        return
    narrowed = (
        args.axis.strip() not in ("all", "")
        or args.select
        or args.limit
        or args.fail_fast
        or args.cpu_only
        or args.quick
        or selection.sweep_keys() is not None
    )
    if narrowed:
        raise PlanError("--update-coverage requires a complete, unfiltered sweep")


def _reject_stray_zoo_options(args: argparse.Namespace, plan: RunPlan) -> None:
    """A zoo option on a run with no zoo stage is a mistake, not a no-op."""
    asked = [
        flag
        for flag, given in (
            ("--family", args.family),
            ("--zoo-record", args.zoo_record),
            ("--fresh", args.fresh),
            ("--mem-limit-gb", args.mem_limit_gb),
        )
        if given
    ]
    if asked and not plan.zoo:
        raise PlanError(f"{', '.join(asked)} configure the model-zoo stage, which this run skips")


__all__ = ["PlanError", "RunPlan"]
