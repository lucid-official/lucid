"""The two zoo checks that are not pytest: a real-size build and the training matrix.

Each runs in a child process of its own, started by the zoo stage, so that the
memory guard can kill it and so that one family's allocations never outlive it.
The output follows the line protocol :func:`~lucid.test.audit._zoo_stage.read_result`
reads: ``FAILED`` for a failure, ``CHECK`` for something to look at, ``SKIPPED``
for something the harness declares it cannot check (with the reason), ``ADAPTIVE``
for something this machine's budget made it skip.  Exit ``5`` means nothing was
checked, which is not the same as nothing failing.

    python -m lucid.test.audit._zoo_probes build FAMILY --budget-gb 22.4
    python -m lucid.test.audit._zoo_probes matrix FAMILY
"""

import argparse
import gc
import sys
import time
from dataclasses import dataclass, field

from lucid.test.audit._zoo_budget import fits_build, weights_gb


@dataclass
class Factories:
    """One family directory's registered factories and the registry labels they carry."""

    names: list[str] = field(default_factory=list)
    labels: set[str] = field(default_factory=set)


def factories_by_family() -> dict[str, Factories]:
    """Registered factories grouped by family directory."""
    import lucid.models  # noqa: F401 - registers every factory
    from lucid.models._registry import _REGISTRY

    grouped: dict[str, Factories] = {}
    for name, entry in _REGISTRY.items():
        parts = getattr(entry.factory, "__module__", "").split(".")
        if len(parts) < 4 or parts[1] != "models":
            continue
        slot = grouped.setdefault(parts[3], Factories())
        slot.names.append(name)
        slot.labels.add(entry.family)
    return grouped


def build_family(family: str, budget_gb: float) -> int:
    """Build every factory of ``family`` at its real size; report what the build allocated.

    A factory whose declared parameter count would not fit ``budget_gb`` as
    float32 is skipped *by name*: the build is the one check that cannot be
    shrunk, so a machine too small for it has to say which models it left out.
    """
    import lucid
    import lucid.metal
    from lucid.models import create_model
    from lucid.models._registry import _REGISTRY

    started = time.monotonic()
    names = factories_by_family().get(family, Factories()).names
    built_count = past_budget = failed = 0
    for name in names:
        declared = _REGISTRY[name].params
        if declared and not fits_build(declared, budget_gb):
            past_budget += 1
            need = weights_gb(declared)
            print(
                f"ADAPTIVE {family}/{name}: {declared:,} params need {need:.1f} GB, "
                f"past this machine's build budget of {budget_gb:.1f} GB",
                flush=True,
            )
            continue
        began = time.monotonic()
        try:
            model = create_model(name)
        except Exception as exc:  # noqa: BLE001 - reported per factory; the rest still build
            failed += 1
            print(f"FAILED {name} - {type(exc).__name__}: {str(exc)[:200]}", flush=True)
            continue
        built_count += 1
        built = sum(int(param.numel()) for param in model.parameters())
        print(
            f"built {name}: {built:,} params, declared {declared}, "
            f"{time.monotonic() - began:.1f}s",
            flush=True,
        )
        if declared and built != declared:
            print(f"CHECK {name}: built {built:,} params, declared {declared:,}", flush=True)
        del model
        gc.collect()
        lucid.metal.empty_cache()
    elapsed = time.monotonic() - started
    print(f"{built_count} built, {past_budget} past budget, {failed} failed in {elapsed:.1f}s")
    if failed:
        return 1
    # Nothing built is nothing verified, whether every factory was past the budget
    # or the family registers none; saying "pass" for it is the lie this exit avoids.
    return 0 if built_count else 5


def matrix_family(family: str) -> int:
    """The compiled-training pairs of ``family``, held to the matrix's own expectations.

    A pair the matrix itself skips (``status == "skip"``, with the reason it
    gives) is reported as skipped, not as a pair that ran: three families read
    ``pass`` for a stage that trained nothing.
    """
    from lucid.test.unit.compile import _zoo_matrix as matrix

    started = time.monotonic()
    labels = factories_by_family().get(family, Factories()).labels | {family}
    ran = bad = skipped = 0
    for task in matrix.TASKS:
        for label in matrix.families(task):
            if label not in labels:
                continue
            record = matrix.run_isolated(label, task)
            if record["status"] == "skip":
                skipped += 1
                print(f"SKIPPED [1] {task} {label}: {record.get('detail')}", flush=True)
                continue
            why = matrix.verdict(record)
            ran += 1
            bad += why is not None
            print(f"FAILED {task} {label} - {why}" if why else f"ok {task} {label}", flush=True)
    if not (ran or skipped):
        print(f"SKIPPED [1] {family}: no compiled-training pair is declared for it", flush=True)
    print(f"{ran} pairs run, {skipped} skipped, {bad} failing in {time.monotonic() - started:.1f}s")
    return 1 if bad else (0 if ran else 5)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m lucid.test.audit._zoo_probes")
    sub = parser.add_subparsers(dest="probe", required=True)
    build = sub.add_parser("build", help="build every factory of a family at its real size")
    build.add_argument("family")
    build.add_argument("--budget-gb", type=float, required=True)
    sub.add_parser("matrix", help="the compiled-training matrix").add_argument("family")
    args = parser.parse_args(argv)
    if args.probe == "build":
        return build_family(args.family, args.budget_gb)
    return matrix_family(args.family)


if __name__ == "__main__":
    sys.exit(main())
