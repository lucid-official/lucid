"""Reproduce the two-panel benchmark figure shown in the README.

Run it directly — it is a script, not a pytest module, because it needs the
reference framework installed and takes about a minute:

    python -m lucid.test.perf.bench_readme_figure            # numbers only
    python -m lucid.test.perf.bench_readme_figure --plot     # + the SVGs

What is measured
----------------
Both panels are GPU-resident and synchronised before the clock stops, which
is the part that is easy to get wrong: MLX defers execution, so timing a
forward without flushing measures graph *construction*, not compute.  The
reference side gets the equivalent device synchronise.

* Left  — one full training step: forward, backward, and an Adam update.
* Right — forward-only latency under no-grad.

The shapes here are small-to-mid layers, where per-op dispatch cost is a
real share of the step and Lucid's short path from Python to the engine
shows up.  It does not hold everywhere, and the defaults say so: width 1024
swung between 0.95x and 1.23x across repeats, so it is left out rather than
reported as a win, and by 2048 the reference framework is ahead.  Past that
point both are waiting on the same Metal kernels and dispatch is no longer
what you are measuring.  Run ``--sweep`` to watch the crossover instead of
taking it on faith.
"""

import argparse
import gc
import json
import statistics
import time
from typing import Any, Callable

import lucid
import lucid.nn as nn
import lucid.optim as optim
from lucid.test._fixtures.ref_framework import REF_DISPLAY_NAME, require_ref

WARMUP = 8
ITERS = 40
BATCH = 128


def _time(fn: Callable[[], object], warmup: int = WARMUP, iters: int = ITERS) -> float:
    """Median wall-clock milliseconds over ``iters`` runs."""
    for _ in range(warmup):
        fn()
    samples: list[float] = []
    for _ in range(iters):
        gc.collect()
        start = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - start) * 1e3)
    return statistics.median(samples)


def _train_step(width: int, ref: Any) -> tuple[float, float]:
    """One forward + backward + Adam update on an MLP of the given width."""
    model = nn.Sequential(nn.Linear(width, width), nn.ReLU(), nn.Linear(width, width)).to("metal")
    opt = optim.Adam(model.parameters(), lr=1e-3)
    x = lucid.randn(BATCH, width, device="metal")
    y = lucid.randn(BATCH, width, device="metal")

    def lucid_step() -> None:
        loss = nn.functional.mse_loss(model(x), y)
        loss.eval()  # flush the lazy graph before it is timed
        opt.zero_grad()
        loss.backward()
        opt.step()

    ref_model = ref.nn.Sequential(
        ref.nn.Linear(width, width), ref.nn.ReLU(), ref.nn.Linear(width, width)
    ).to("mps")
    ref_opt = ref.optim.Adam(ref_model.parameters(), lr=1e-3)
    rx = ref.randn(BATCH, width, device="mps")
    ry = ref.randn(BATCH, width, device="mps")

    def ref_step() -> None:
        loss = ref.nn.functional.mse_loss(ref_model(rx), ry)
        ref_opt.zero_grad()
        loss.backward()
        ref_opt.step()
        ref.mps.synchronize()

    return _time(lucid_step), _time(ref_step)


def _infer(batch: int, ref: Any, depth: int = 8, width: int = 512) -> tuple[float, float]:
    """Forward-only latency through a stack of ``depth`` linear+ReLU blocks."""
    ours: list[nn.Module] = []
    theirs: list[Any] = []
    for _ in range(depth):
        ours += [nn.Linear(width, width), nn.ReLU()]
        theirs += [ref.nn.Linear(width, width), ref.nn.ReLU()]

    model = nn.Sequential(*ours).to("metal").eval()
    ref_model = ref.nn.Sequential(*theirs).to("mps").eval()
    x = lucid.randn(batch, width, device="metal")
    rx = ref.randn(batch, width, device="mps")

    with lucid.no_grad():
        ours_ms = _time(lambda: lucid.eval(model(x)))
    with ref.no_grad():
        ref_ms = _time(lambda: (ref_model(rx), ref.mps.synchronize()))
    return ours_ms, ref_ms


def main() -> None:
    """Run both panels and optionally render the figure."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plot", action="store_true", help="write the SVGs too")
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="extend the width sweep past the crossover instead of stopping at 1024",
    )
    parser.add_argument("--out", default="docs/assets", help="where the SVGs go")
    args = parser.parse_args()

    ref = require_ref()
    # The default set is the one that reproduces: five repeats of each stayed
    # inside a narrow band and never crossed 1.0x.  Width 1024 is deliberately
    # excluded — it measured anywhere from 0.95x to 1.23x across runs, so any
    # single number for it would be a coin flip dressed up as a result.
    widths = (128, 256, 512, 1024, 2048, 4096) if args.sweep else (128, 256, 512)
    results: dict[str, list[dict[str, float]]] = {"train": [], "infer": []}

    print(f"training step — batch {BATCH}, median of {ITERS} runs")
    for width in widths:
        ours, theirs = _train_step(width, ref)
        results["train"].append({"width": width, "lucid": ours, "ref": theirs})
        print(
            f"  width {width:<6} lucid {ours:7.3f} ms   ref {theirs:7.3f} ms"
            f"   {theirs / ours:.2f}x"
        )

    print(f"\ninference latency — 8 x 512 MLP, median of {ITERS} runs")
    for batch in (1, 8, 32, 128):
        ours, theirs = _infer(batch, ref)
        results["infer"].append({"batch": batch, "lucid": ours, "ref": theirs})
        print(
            f"  batch {batch:<6} lucid {ours:7.3f} ms   ref {theirs:7.3f} ms"
            f"   {theirs / ours:.2f}x"
        )

    if args.plot:
        _render(results, args.out)
    else:
        print("\n" + json.dumps(results, indent=2))


def _render(results: dict[str, list[dict[str, float]]], out_dir: str) -> None:
    """Write the light and dark SVGs the README embeds."""
    import os

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    themes = {
        "dark": {"fg": "#E6EDF3", "muted": "#8B949E", "ref": "#EE4C2C"},
        "light": {"fg": "#1F2328", "muted": "#59636E", "ref": "#EE4C2C"},
    }
    accent = "#7C5CFF"
    # The comparison is against the reference framework by its own brand colour,
    # so the figure reads at a glance without a legend lookup. The name comes
    # from the fixture rather than being written here: that module is the one
    # place H5 admits the literal.
    ref_label = REF_DISPLAY_NAME
    os.makedirs(out_dir, exist_ok=True)

    for theme, colour in themes.items():
        fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.0))
        fig.patch.set_alpha(0)

        def style(ax: Any) -> None:
            ax.set_facecolor("none")
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            for side in ("left", "bottom"):
                ax.spines[side].set_color(colour["muted"])
            ax.tick_params(colors=colour["muted"], labelsize=9)
            ax.yaxis.label.set_color(colour["fg"])
            ax.xaxis.label.set_color(colour["fg"])
            ax.title.set_color(colour["fg"])
            ax.grid(axis="y", color=colour["muted"], alpha=0.18, linewidth=0.7)
            ax.set_axisbelow(True)

        widths = [r["width"] for r in results["train"]]
        ours = [r["lucid"] for r in results["train"]]
        theirs = [r["ref"] for r in results["train"]]
        xs = list(range(len(widths)))
        bar = 0.36
        left.bar([v - bar / 2 for v in xs], ours, bar, label="Lucid", color=accent)
        left.bar(
            [v + bar / 2 for v in xs],
            theirs,
            bar,
            label=ref_label,
            color=colour["ref"],
        )
        for i, (a, b) in enumerate(zip(ours, theirs)):
            left.text(
                i,
                max(a, b) * 1.06,
                f"{b / a:.2f}×",
                ha="center",
                color=accent,
                fontsize=10,
                fontweight="bold",
            )
        left.set_xticks(xs)
        left.set_xticklabels([str(v) for v in widths])
        left.set_xlabel("hidden width")
        left.set_ylabel("ms / step   (lower is better)")
        left.set_title("Training step — forward + backward + Adam", fontsize=11, pad=12)
        left.set_ylim(0, max(theirs) * 1.25)
        style(left)

        batches = [r["batch"] for r in results["infer"]]
        ours2 = [r["lucid"] for r in results["infer"]]
        theirs2 = [r["ref"] for r in results["infer"]]
        xs2 = list(range(len(batches)))
        right.plot(
            xs2,
            theirs2,
            "-o",
            color=colour["ref"],
            linewidth=2,
            markersize=6,
            label=ref_label,
        )
        right.plot(xs2, ours2, "-o", color=accent, linewidth=2.4, markersize=6, label="Lucid")
        right.fill_between(xs2, ours2, theirs2, color=accent, alpha=0.13)
        for i, (a, b) in enumerate(zip(ours2, theirs2)):
            right.text(
                i,
                b * 1.04,
                f"{b / a:.2f}×",
                ha="center",
                color=accent,
                fontsize=10,
                fontweight="bold",
            )
        right.set_xticks(xs2)
        right.set_xticklabels([str(v) for v in batches])
        right.set_xlabel("batch size")
        right.set_ylabel("ms / forward   (lower is better)")
        right.set_title("Inference latency — 8-layer 512-wide MLP", fontsize=11, pad=12)
        right.set_ylim(0, max(theirs2) * 1.3)
        style(right)

        handles, labels = left.get_legend_handles_labels()
        legend = fig.legend(
            handles,
            labels,
            loc="lower center",
            ncol=2,
            frameon=False,
            bbox_to_anchor=(0.5, -0.04),
            fontsize=10,
        )
        for text in legend.get_texts():
            text.set_color(colour["fg"])

        fig.tight_layout(rect=(0, 0.04, 1, 1))
        path = os.path.join(out_dir, f"benchmark-{theme}.svg")
        fig.savefig(path, format="svg", transparent=True, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
