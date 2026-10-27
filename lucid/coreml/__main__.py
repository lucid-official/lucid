"""Command line for ``lucid.coreml``.

Examples
--------
::

    # export a zoo model, check it, and report where it will run
    python -m lucid.coreml resnet_18_cls --out /tmp/r18.mlpackage --verify

    # target the Neural Engine (float16 is what it runs)
    python -m lucid.coreml resnet_18_cls --out /tmp/r18.mlpackage \
        --precision FLOAT16 --units CPU_AND_NE --verify
"""

import argparse
import sys

import lucid
import lucid.models as models
from lucid.coreml import ComputeUnits, Precision, WeightPrecision, export

__all__ = ["main"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m lucid.coreml",
        description="Export a Lucid model zoo entry to a Core ML package.",
    )
    parser.add_argument("model", help="Registered factory name, e.g. 'resnet_18_cls'.")
    parser.add_argument("--out", required=True, help="Destination .mlpackage path.")
    parser.add_argument(
        "--shape", default="1,3,224,224", help="Comma-separated input shape (default: 1,3,224,224)."
    )
    parser.add_argument(
        "--num-classes", type=int, default=None, help="Override the head's class count."
    )
    parser.add_argument(
        "--precision",
        default="FLOAT32",
        choices=[p.value for p in Precision],
        help="Body precision (default: FLOAT32, faithful to the source model; "
        "FLOAT16 is what the Neural Engine runs).",
    )
    parser.add_argument(
        "--weights",
        default="FLOAT",
        choices=[w.value for w in WeightPrecision],
        help="Weight storage (default: FLOAT). INT8 keeps eight bits per weight "
        "plus a per-channel scale, halving the package against float16.",
    )
    parser.add_argument(
        "--units",
        default="ALL",
        choices=[u.value for u in ComputeUnits],
        help="Processors Core ML may use (default: ALL).",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Compare against the eager model and report the largest difference.",
    )
    args = parser.parse_args(argv)

    shape = tuple(int(d) for d in args.shape.split(","))
    overrides = {} if args.num_classes is None else {"num_classes": args.num_classes}
    model = models.create_model(args.model, **overrides).eval()
    example = lucid.randn(*shape)

    print(f"[export] {args.model} {shape} -> {args.out}")
    exported = export(
        model,
        example,
        args.out,
        precision=Precision(args.precision),
        weights=WeightPrecision(args.weights),
        compute_units=ComputeUnits(args.units),
    )
    print(
        f"[export] saved  precision={args.precision} weights={args.weights} " f"units={args.units}"
    )

    plan = exported.compute_plan()
    if plan.total_compute:
        placement = ", ".join(f"{d}={n}" for d, n in sorted(plan.compute.items()))
        print(f"[plan]   {placement}  ({plan.constants} constants)")
        print(f"[plan]   Neural Engine takes {plan.ane_fraction:.0%} of the computation")
        if args.units == ComputeUnits.CPU_AND_NE.value and plan.ane_fraction == 0.0:
            print(
                "[plan]   nothing was scheduled on the Neural Engine — it only runs "
                "float16; re-export with --precision FLOAT16"
            )
    else:
        print("[plan]   unavailable (needs macOS 14.4+) — this is unknown, not unaccelerated")

    print(f"[export] inputs {exported.input_names} -> outputs {exported.output_names}")

    if args.verify:
        worst = exported.verify(model, example)
        print(f"[verify] worst max|lucid - coreml| over the outputs = {worst:.3e}")

    exported.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
