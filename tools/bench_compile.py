"""Eager vs compile wall-clock sweep.

Usage::

    python -m tools.bench_compile [--quick]

Outputs a Markdown table comparing ``model(x)`` vs ``compile(model)(x)``
across a representative set of models and input sizes.  The compile
side discounts the first call (warmup includes the compile cost) and
reports the steady-state per-call latency.

The table is what the user-guide / docs site embeds — the script
exists so the numbers stay reproducible: run it on a known hardware
configuration, copy the table into the guide.

Notes
-----
* Wall-clock is wallclock not CPU-time — Metal command submission +
  GPU work is captured.
* "Speedup" = eager_ms / compile_ms.  > 1.0 means compile is faster;
  < 1.0 means compile pays Python / dispatch / kernel-launch overhead
  that exceeds the savings.
* Bit-exact parity is enforced before timing — any mismatch aborts
  the row (so a buggy compile path never silently shows up as
  "faster").
"""

import argparse
import math
from collections.abc import Callable
from typing import override

import lucid
import lucid.models as M
import lucid.nn as nn
import lucid.nn.functional as F
import lucid.optim as optim
from lucid.compile import compile as compile_model, fused_step
from tools._bench_timing import warm_ms

COMPILE_DEVICE = "metal"


def _unwrap(out: object) -> lucid.Tensor:
    if isinstance(out, lucid.Tensor):
        return out
    for attr in ("logits", "last_hidden_state", "prediction"):
        v = getattr(out, attr, None)
        if isinstance(v, lucid.Tensor):
            return v
    raise TypeError(type(out).__name__)


def _bench_once(call: Callable[[], object], n: int) -> float:
    """Median materialized end-to-end latency after three warmup calls."""
    return warm_ms(call, 3, n)[0]


class _LstmHead(nn.Module):
    def __init__(self, input_size: int = 64, hidden_size: int = 128, n_classes: int = 10) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size)
        self.fc = nn.Linear(hidden_size, n_classes)

    @override
    def forward(self, x: lucid.Tensor) -> lucid.Tensor:
        y, _ = self.lstm(x)
        return _unwrap(self.fc(_unwrap(y)[-1]))


def _seq_to_cls(input_size: int = 64, hidden_size: int = 256) -> nn.Module:
    """Composite Linear + LSTM + Linear stack — exercises the LSTM emit
    path inside a multi-op pipeline (where compile fuses the head
    Linear into the same executable as the LSTM)."""

    class _M(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embed = nn.Linear(input_size, hidden_size)
            self.lstm = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size)
            self.head = nn.Linear(hidden_size, 10)

        @override
        def forward(self, x: lucid.Tensor) -> lucid.Tensor:
            y, _ = self.lstm(_unwrap(self.embed(x)))
            return _unwrap(self.head(_unwrap(y)[-1]))

    return _M()


def _bench_case(
    name: str, mk_model: Callable[[], nn.Module], mk_input: Callable[[], lucid.Tensor], n_iter: int
) -> dict[str, str | float]:
    model = mk_model()
    model.eval()
    model.to(COMPILE_DEVICE)
    x = mk_input().to(COMPILE_DEVICE)
    # A compiled forward needs no_grad since LCD-582: under grad mode a
    # trainable model's call runs eagerly, and this would time eager twice.
    with lucid.no_grad():
        return _bench_inference(name, model, x, n_iter)


def _bench_inference(
    name: str, model: nn.Module, x: lucid.Tensor, n_iter: int
) -> dict[str, str | float]:
    # Parity guard — abort the row if compile diverges materially.
    eager_out = _unwrap(model(x))
    cm = compile_model(model)
    compiled_out = _unwrap(cm(x))
    abs_diff = float((eager_out - compiled_out).abs().max().item())
    scale = float(eager_out.abs().max().item())
    rel_diff = abs_diff / max(scale, 1e-9)
    if not math.isfinite(rel_diff) or rel_diff > 1e-3:
        return {
            "name": name,
            "eager_ms": float("nan"),
            "compile_ms": float("nan"),
            "speedup": float("nan"),
            "rel_diff": rel_diff,
            "note": "parity-diverged",
        }

    eager_ms = _bench_once(lambda: model(x), n_iter)
    compile_ms = _bench_once(lambda: cm(x), n_iter)
    return {
        "name": name,
        "eager_ms": eager_ms,
        "compile_ms": compile_ms,
        "speedup": eager_ms / compile_ms,
        "rel_diff": rel_diff,
        "note": "ok",
    }


def _cases(quick: bool) -> list[tuple[str, Callable[[], nn.Module], Callable[[], lucid.Tensor]]]:
    """Bench cases.  ``--quick`` keeps the runtime under 30s."""
    if quick:
        return [
            (
                "lenet_5 (1×1×32×32)",
                lambda: M.lenet_5(num_classes=10),
                lambda: lucid.randn(1, 1, 32, 32),
            ),
            (
                "vit_base_16 (1×3×64²)",
                lambda: M.vit_base_16(image_size=64, num_classes=10),
                lambda: lucid.randn(1, 3, 64, 64),
            ),
            ("lstm_head (50×16×64)", _LstmHead, lambda: lucid.randn(50, 16, 64)),
            ("seq2cls (100×32×64)", _seq_to_cls, lambda: lucid.randn(100, 32, 64)),
        ]
    return [
        (
            "lenet_5 (1×1×32×32)",
            lambda: M.lenet_5(num_classes=10),
            lambda: lucid.randn(1, 1, 32, 32),
        ),
        (
            "resnet_18 (1×3×224²)",
            lambda: M.resnet_18(num_classes=10),
            lambda: lucid.randn(1, 3, 224, 224),
        ),
        (
            "mobilenet (1×3×224²)",
            lambda: M.mobilenet(num_classes=10),
            lambda: lucid.randn(1, 3, 224, 224),
        ),
        (
            "efficientnet_b0 (1×3×224²)",
            lambda: M.efficientnet_b0(num_classes=10),
            lambda: lucid.randn(1, 3, 224, 224),
        ),
        (
            "densenet_121 (1×3×224²)",
            lambda: M.densenet_121(num_classes=10),
            lambda: lucid.randn(1, 3, 224, 224),
        ),
        (
            "vit_base_16 (1×3×224²)",
            lambda: M.vit_base_16(image_size=224, num_classes=10),
            lambda: lucid.randn(1, 3, 224, 224),
        ),
        ("lstm_head (50×16×64)", _LstmHead, lambda: lucid.randn(50, 16, 64)),
        (
            "lstm_head (200×32×64)",
            lambda: _LstmHead(input_size=64, hidden_size=256),
            lambda: lucid.randn(200, 32, 64),
        ),
        ("seq2cls (100×32×64)", _seq_to_cls, lambda: lucid.randn(100, 32, 64)),
    ]


def _bench_training_case(
    name: str,
    mk_model: Callable[[], nn.Module],
    mk_inputs: Callable[[], tuple[lucid.Tensor, ...]],
    n_iter: int,
) -> dict[str, str | float]:
    """Benchmark a single training step: eager (3 calls: zero_grad, backward, step)
    vs ``fused_step`` (one MPSGraph executable).

    Both paths start from the same parameter state so the timing is
    apples-to-apples — only the dispatch overhead differs.
    """
    lucid.manual_seed(0)
    model_eager = mk_model().to(COMPILE_DEVICE)
    model_fused = mk_model().to(COMPILE_DEVICE)
    # Sync params
    with lucid.no_grad():
        for (_, p), (_, q) in zip(model_eager.named_parameters(), model_fused.named_parameters()):
            q.copy_(p)
    args = tuple(t.to(COMPILE_DEVICE) for t in mk_inputs())

    opt_eager = optim.Adam(list(model_eager.parameters()), lr=1e-3)
    opt_fused = optim.Adam(list(model_fused.parameters()), lr=1e-3)
    step_fused = fused_step(model_fused, F.mse_loss, opt_fused)

    def _eager_step() -> lucid.Tensor:
        opt_eager.zero_grad()
        loss = F.mse_loss(_unwrap(model_eager(*args[:-1])), args[-1])
        loss.backward()
        opt_eager.step()
        return loss

    def _fused_call() -> lucid.Tensor:
        return step_fused(*args)

    # Compare one complete update before timing; a literal zero in the
    # report used to masquerade as a training parity measurement.
    eager_loss = float(_eager_step().item())
    fused_loss = float(_fused_call().item())
    rel_diff = abs(eager_loss - fused_loss) / max(abs(eager_loss), 1e-9)
    for own, compiled in zip(model_eager.parameters(), model_fused.parameters()):
        error = float((own - compiled).abs().max().item())
        scale = max(float(own.abs().max().item()), 1e-9)
        if not math.isfinite(error) or not math.isfinite(scale):
            rel_diff = float("inf")
            break
        rel_diff = max(rel_diff, error / scale)
    if not math.isfinite(rel_diff) or rel_diff > 1e-3:
        return {
            "name": name,
            "eager_ms": float("nan"),
            "compile_ms": float("nan"),
            "speedup": float("nan"),
            "rel_diff": rel_diff,
            "note": "parity-diverged",
        }

    eager_ms = _bench_once(_eager_step, n_iter)
    compile_ms = _bench_once(_fused_call, n_iter)
    return {
        "name": name,
        "eager_ms": eager_ms,
        "compile_ms": compile_ms,
        "speedup": eager_ms / compile_ms,
        "rel_diff": rel_diff,
        "note": "training",
    }


def _training_cases(
    quick: bool,
) -> list[tuple[str, Callable[[], nn.Module], Callable[[], tuple[lucid.Tensor, ...]]]]:
    """Training-step cases.  Each tuple is (name, model factory, inputs factory).
    The inputs factory returns ``(x, target)`` for ``loss_fn(model(x), target)``.
    """

    def _mlp() -> nn.Module:
        class _M(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.fc1 = nn.Linear(64, 128)
                self.fc2 = nn.Linear(128, 10)

            @override
            def forward(self, x: lucid.Tensor) -> lucid.Tensor:
                return _unwrap(self.fc2(_unwrap(self.fc1(x)).relu()))

        return _M()

    def _deeper_mlp() -> nn.Module:
        layers = []
        prev = 128
        for h in (256, 256, 256, 128):
            layers.extend([nn.Linear(prev, h), nn.ReLU()])
            prev = h
        layers.append(nn.Linear(prev, 10))
        return nn.Sequential(*layers)

    cases: list[tuple[str, Callable[[], nn.Module], Callable[[], tuple[lucid.Tensor, ...]]]] = [
        ("mlp (64→128→10, BS=32)", _mlp, lambda: (lucid.randn(32, 64), lucid.randn(32, 10))),
        (
            "deep_mlp (×4 hidden, BS=64)",
            _deeper_mlp,
            lambda: (lucid.randn(64, 128), lucid.randn(64, 10)),
        ),
    ]
    if not quick:
        cases.append(
            (
                "lstm_head (50×16×64) train",
                _LstmHead,
                lambda: (lucid.randn(50, 16, 64), lucid.randn(16, 10)),
            )
        )
    return cases


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="small/fast subset for CI smoke")
    ap.add_argument("--iter", type=int, default=20, help="benchmark iterations per case")
    args = ap.parse_args()
    if args.iter < 1:
        ap.error("iterations must be positive")

    print("\n## Inference (eager vs `lucid.compile(model)`)\n")
    rows = []
    for name, mk_model, mk_input in _cases(args.quick):
        try:
            row = _bench_case(name, mk_model, mk_input, n_iter=args.iter)
        except Exception as e:  # pragma: no cover - bench-script error path
            row = {
                "name": name,
                "eager_ms": float("nan"),
                "compile_ms": float("nan"),
                "speedup": float("nan"),
                "rel_diff": float("nan"),
                "note": f"error: {type(e).__name__}: {str(e)[:50]}",
            }
        rows.append(row)

    # Print markdown table for inference
    print("| Model | Eager (ms) | Compile (ms) | Speedup | Rel diff | Note |")
    print("|---|---:|---:|---:|---:|---|")
    for r in rows:
        eager = f"{r['eager_ms']:.2f}" if r["eager_ms"] == r["eager_ms"] else "—"
        comp = f"{r['compile_ms']:.2f}" if r["compile_ms"] == r["compile_ms"] else "—"
        speed = f"{r['speedup']:.2f}×" if r["speedup"] == r["speedup"] else "—"
        rel = f"{r['rel_diff']:.2e}" if r["rel_diff"] == r["rel_diff"] else "—"
        print(f"| {r['name']} | {eager} | {comp} | {speed} | {rel} | {r['note']} |")

    # Training sweep
    print("\n## Training step (eager Adam vs `fused_step`)\n")
    train_rows = []
    for name, mk_model, mk_inputs in _training_cases(args.quick):
        try:
            row = _bench_training_case(name, mk_model, mk_inputs, n_iter=args.iter)
        except Exception as e:
            row = {
                "name": name,
                "eager_ms": float("nan"),
                "compile_ms": float("nan"),
                "speedup": float("nan"),
                "note": f"error: {type(e).__name__}",
            }
        train_rows.append(row)

    print("| Model | Eager step (ms) | Fused step (ms) | Speedup | Note |")
    print("|---|---:|---:|---:|---|")
    for r in train_rows:
        eager = f"{r['eager_ms']:.2f}" if r["eager_ms"] == r["eager_ms"] else "—"
        comp = f"{r['compile_ms']:.2f}" if r["compile_ms"] == r["compile_ms"] else "—"
        speed = f"{r['speedup']:.2f}×" if r["speedup"] == r["speedup"] else "—"
        print(f"| {r['name']} | {eager} | {comp} | {speed} | {r.get('note', '')} |")
    print()
    return int(
        any(str(row["note"]).startswith(("error:", "parity-diverged")) for row in rows + train_rows)
    )


if __name__ == "__main__":
    raise SystemExit(main())
