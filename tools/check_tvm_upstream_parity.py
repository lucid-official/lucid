"""Compare Lucid's TVM with the authors' implementation on random weights.

The model zoo has no other comparison of this family against its source:
the released checkpoints have not been converted yet (H12 step 11, the
published-weights check, is pending on the first real run of
``tools/convert_weights/tvm.py``), and the user's rule keeps
family-specific tests out of ``lucid/test``.  This script is that
comparison, pinned from the 2026-10-25 audit's probes.

It needs a checkout of the authors' repository (``lumalabs/tvm``; the
audit used commit ``331c01d5``) and the reference framework, and it
**skips** — exit status 0, with the reason printed — when either cannot
be imported.  It builds a small TVM in both, gives Lucid random weights,
moves them into the authors' network through the converter's key map, and
compares, for the scaled and the unscaled parameterisation:

* the network's output, conditional and unconditional;
* the terminal derivative ``df/ds`` and the point the jump lands on;
* the loss, its two terms, and every parameter's gradient, with the
  draws fixed;
* the sampler at 1, 2 and 4 steps against the authors' ``generator_fn``.

The authors run the Jacobian-vector product's attention in bf16 when no
autocast is active.  Lucid computes in the model's dtype throughout (a
documented difference, see :class:`~lucid.models.generative.tvm.TVMConfig`),
so the comparison makes that attention float32 too; ``--release-precision``
leaves the authors' cast in place and reports the gaps without judging
them.

What this does not show: each released pickle embeds its own copy of the
authors' modules, and the 512-pixel release's copy predates the
repository's (its scaled ``get_dfds`` branch scales by ``beta_scale``
where the repository scales by ``1 / beta_scale``).  Agreement with the
repository is not agreement with the code a release was trained with.

Usage::

    python -m tools.check_tvm_upstream_parity --upstream ~/src/tvm
    python -m tools.check_tvm_upstream_parity --upstream ~/src/tvm --device metal

Exit status: 0 when every quantity is within tolerance (or the run
skipped), 1 otherwise.
"""

import argparse
import copy
import dataclasses
import importlib
import sys
from collections.abc import Callable, Iterator
from contextlib import contextmanager, nullcontext
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, TypedDict
from unittest import mock

import numpy as np
import numpy.typing as npt

import lucid
from lucid.models.generative.tvm import TVMConfig, TVMForImageGeneration

if TYPE_CHECKING:
    from torch import Tensor as RefTensor


class _Network(TypedDict):
    """The network both implementations are built with."""

    patch_size: int
    hidden_size: int
    depth: int
    num_heads: int


_BATCH, _CHANNELS, _SIDE, _CLASSES = 3, 4, 8, 10
_NETWORK: _Network = {"patch_size": 2, "hidden_size": 32, "depth": 2, "num_heads": 4}
# max|lucid - ref| / max(1, max|ref|).  The audit measured at most 7.3e-6.
_TOLERANCE = 5e-5

_Array = npt.NDArray[np.float64]


@dataclasses.dataclass(frozen=True)
class _Gap:
    """One compared quantity."""

    name: str
    diff: float
    scale: float

    @property
    def relative(self) -> float:
        return self.diff / max(1.0, self.scale)


def _gap(name: str, got: object, ref: object) -> _Gap:
    a = np.asarray(got, dtype=np.float64)
    b = np.asarray(ref, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"{name}: shapes differ, {a.shape} vs {b.shape}")
    return _Gap(name, float(np.abs(a - b).max()), float(np.abs(b).max()))


def _import_upstream(root: Path) -> tuple[ModuleType, ModuleType] | str:
    """The reference framework and the authors' wrapper module, or why not."""
    if not (root / "training" / "wrapper.py").is_file():
        return f"no training/wrapper.py under {root}"
    sys.path.insert(0, str(root))
    try:
        framework = importlib.import_module("torch")
        wrapper = importlib.import_module("training.wrapper")
    except ImportError as error:
        return f"cannot import the authors' code: {error}"
    return framework, wrapper


@contextmanager
def _fixed_draws(
    model: object, t: object, s: object, s_fm: object, w: object, noise: object
) -> Iterator[None]:
    """Make Lucid's training step use the given times, guidance and noise."""
    with (
        mock.patch.object(model, "_sample_times", lambda batch, device: (t, s, s_fm)),
        mock.patch.object(model, "_sample_guidance", lambda batch, device: w),
        mock.patch.object(lucid, "randn", lambda *args, **kwargs: noise),
    ):
        yield


def _compare(
    framework: ModuleType,
    wrapper_module: ModuleType,
    *,
    scaled: bool,
    device: str,
    float32_jvp: bool,
) -> list[_Gap]:
    """Every compared quantity for one parameterisation."""
    # Imported here: the converter imports the reference framework at
    # module level, and the script must still skip cleanly without it.
    from tools.convert_weights.tvm import TVMArch

    torch = framework
    jvp_precision = (
        mock.patch.object(torch, "is_autocast_enabled", lambda *a, **k: True)
        if float32_jvp
        else nullcontext()
    )
    rng = np.random.default_rng(0)
    config = TVMConfig(
        sample_size=_SIDE,
        num_classes=_CLASSES,
        class_dropout=0.0,
        scaled_parameterization=scaled,
        eval_ema_decay=None,
        **_NETWORK,
    )
    lucid.manual_seed(0)
    model = TVMForImageGeneration(config)
    with lucid.no_grad():
        for name, param in model.tvm.named_parameters():
            shape = tuple(int(d) for d in param.shape)
            values = rng.standard_normal(shape).astype(np.float32) * 0.2
            if name.endswith("norm.weight"):
                values = 1.0 + values
            param[:] = lucid.tensor(values)
    model.to(device)

    upstream = wrapper_module.TVMWrapper(
        img_resolution=_SIDE,
        img_channels=_CHANNELS,
        label_dim=_CLASSES,
        model_type="DiT",
        data_type="fp32",
        do_scale_param=scaled,
        model_kwargs=dict(
            num_classes=_CLASSES,
            qk_norm=True,
            learn_guidance=True,
            do_mod_norm=True,
            init_type="spectral",
            time_specinit=True,
            **_NETWORK,
        ),
    )
    arch = TVMArch("tvm_imagenet256", "IMAGENET1K_256_W2")
    lucid_state = {k: v.to("cpu").numpy() for k, v in model.tvm.state_dict().items()}
    filled: set[str] = set()
    with torch.no_grad():
        for key, value in upstream.state_dict().items():
            target = arch.map_key(key)
            if target is None:
                continue
            value.copy_(torch.from_numpy(lucid_state[target].copy()))
            filled.add(target)
    if filled != set(lucid_state):
        raise RuntimeError(f"keys never filled: {sorted(set(lucid_state) - filled)}")
    upstream.train()
    teacher = copy.deepcopy(upstream).eval().requires_grad_(False)

    def ref(a: _Array) -> RefTensor:
        tensor: RefTensor = torch.from_numpy(np.asarray(a, dtype=np.float32).copy())
        return tensor

    def lu(a: _Array) -> lucid.Tensor:
        return lucid.tensor(np.asarray(a, dtype=np.float32), device=device)

    def host(x: lucid.Tensor) -> _Array:
        return np.asarray(x.detach().to("cpu").numpy(), dtype=np.float64)

    def column(a: _Array) -> _Array:
        return a.reshape(-1, 1, 1, 1)

    x = rng.standard_normal((_BATCH, _CHANNELS, _SIDE, _SIDE))
    t = np.array([0.9, 0.6, 0.5])
    s = np.array([0.3, 0.1, 0.5])
    labels = np.array([3, 7, 1])
    w = np.array([2.0, 2.0, 1.0])
    onehot = np.eye(_CLASSES)[labels]
    lucid_labels = lucid.tensor(labels, device=device)
    gaps: list[_Gap] = []

    with torch.no_grad(), jvp_precision:
        cond = upstream.forward_model(
            ref(x),
            ref(column(t)),
            ref(column(s)),
            class_labels=ref(onehot),
            beta_scale=ref(column(1 / w)),
        ) * (ref(column(w)) if scaled else 1.0)
        null = upstream.forward_model(
            ref(x),
            ref(column(t)),
            ref(column(s)),
            class_labels=torch.zeros(_BATCH, _CLASSES),
            beta_scale=torch.ones(_BATCH, 1, 1, 1),
        )
    with lucid.no_grad():
        gaps.append(
            _gap(
                "forward, conditional",
                host(model.tvm(lu(x), lu(t), lu(s), lucid_labels, lu(w)).sample),
                cond.numpy(),
            )
        )
        gaps.append(
            _gap(
                "forward, unconditional",
                host(model.tvm(lu(x), lu(t), lu(s)).sample),
                null.numpy(),
            )
        )

    with jvp_precision:
        landing_ref, dfds_ref = upstream.get_dfds(
            ref(x), ref(column(t)), ref(column(s)), ref(onehot), ref(column(1 / w))
        )
    velocity, terminal = model._terminal_velocity(
        lu(x), lu(t), lu(s), lucid_labels, lu(w)
    )
    gaps.append(
        _gap("terminal derivative df/ds", host(terminal), dfds_ref.detach().numpy())
    )
    gaps.append(
        _gap(
            "landing point",
            host(lu(x) + lu(column(s - t)) * velocity),
            landing_ref.detach().numpy(),
        )
    )

    images = rng.standard_normal((_BATCH, _CHANNELS, _SIDE, _SIDE))
    noise = rng.standard_normal((_BATCH, _CHANNELS, _SIDE, _SIDE))
    s_fm = np.array([0.2, 0.7, 0.4])
    xt = column(1 - t) * images + column(t) * noise
    xs = column(1 - s_fm) * images + column(s_fm) * noise
    upstream.zero_grad()
    with jvp_precision:
        loss_ref, logs = upstream(
            ref(xt),
            ref(xs),
            ref(column(t)),
            ref(column(s)),
            ref(column(s_fm)),
            ref(onehot),
            ref(column(1 / w)),
            ref(noise - images),
            teacher_model=teacher,
        )
        loss_ref.mean().backward()
    with _fixed_draws(model, lu(t), lu(s), lu(s_fm), lu(w), lu(noise)):
        out = model(lu(images), lucid_labels)
    for param in model.parameters():
        param.grad = None
    out.loss.backward()
    gaps.append(_gap("loss", float(out.loss.item()), float(loss_ref.mean().item())))
    gaps.append(
        _gap(
            "terminal term",
            float(out.terminal_loss.item()),
            float(logs["terminal_loss"].mean().item()),
        )
    )
    gaps.append(
        _gap(
            "flow-matching term",
            float(out.flow_matching_loss.item()),
            float(logs["fm_loss"].mean().item()),
        )
    )
    grads = {
        n: host(p.grad) for n, p in model.tvm.named_parameters() if p.grad is not None
    }
    compared: set[str] = set()
    for key, value in upstream.named_parameters():
        target = arch.map_key(key)
        if target is None or value.grad is None:
            continue
        gaps.append(_gap(f"grad {target}", grads[target], value.grad.numpy()))
        compared.add(target)
    if compared != set(grads):
        raise RuntimeError(f"gradients not compared: {sorted(set(grads) - compared)}")

    upstream.eval()
    model.eval()
    z = rng.standard_normal((_BATCH, _CHANNELS, _SIDE, _SIDE))
    for steps in (1, 2, 4):
        with jvp_precision:
            want = _authors_sampler(torch, upstream, ref(z), ref(onehot), steps, 2.0)
        got = model.generate(
            _BATCH, labels=lucid_labels, steps=steps, guidance_scale=2.0, noise=lu(z)
        ).samples
        gaps.append(_gap(f"sampler, {steps} step(s)", host(got), want))
    return gaps


def _authors_sampler(
    torch: ModuleType,
    net: object,
    latents: RefTensor,
    labels: RefTensor,
    steps: int,
    cfg_scale: float,
) -> _Array:
    """The logic of the authors' ``generate_images.generator_fn``, verbatim."""
    forward: Callable[..., RefTensor] = getattr(net, "forward_model")
    with torch.no_grad():
        t_steps = torch.linspace(1, 0, steps + 1, dtype=torch.float64)
        x = latents.to(torch.float64)
        beta = (
            1 / torch.as_tensor(cfg_scale, dtype=torch.float64).view(-1, 1, 1, 1)
        ).repeat(x.shape[0], 1, 1, 1)
        out_scale = cfg_scale if getattr(net, "do_scale_param") else 1.0
        for t_cur, t_next in zip(t_steps[:-1], t_steps[1:], strict=True):
            velocity = forward(x, t_cur, t_next, class_labels=labels, beta_scale=beta)
            x = x + (t_next - t_cur) * velocity.to(torch.float64) * out_scale
        return np.asarray(x.numpy(), dtype=np.float64)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m tools.check_tvm_upstream_parity",
        description="Compare Lucid's TVM with the authors' implementation.",
    )
    parser.add_argument(
        "--upstream", type=Path, required=True, help="checkout of lumalabs/tvm"
    )
    parser.add_argument("--device", choices=("cpu", "metal"), default="cpu")
    parser.add_argument("--tolerance", type=float, default=_TOLERANCE)
    parser.add_argument(
        "--release-precision",
        action="store_true",
        help="keep the authors' bf16 attention in the JVP and only report",
    )
    args = parser.parse_args(argv)

    imported = _import_upstream(args.upstream.expanduser())
    if isinstance(imported, str):
        print(f"skip: {imported}")
        return 0
    framework, wrapper_module = imported

    failed = 0
    for scaled in (True, False):
        gaps = _compare(
            framework,
            wrapper_module,
            scaled=scaled,
            device=args.device,
            float32_jvp=not args.release_precision,
        )
        worst = max(gaps, key=lambda gap: gap.relative)
        bad = [gap for gap in gaps if gap.relative > args.tolerance]
        label = "scaled" if scaled else "unscaled"
        print(
            f"{label}: {len(gaps)} quantities, worst {worst.relative:.2e} at {worst.name}"
        )
        for gap in gaps if args.release_precision else bad:
            print(
                f"  {gap.name:40s} max|diff| {gap.diff:.3e}  max|ref| {gap.scale:.3e}"
            )
        failed += 0 if args.release_precision else len(bad)
    if failed:
        print(
            f"FAIL: {failed} quantit{'y' if failed == 1 else 'ies'} beyond {args.tolerance:g}"
        )
        return 1
    print(
        "ok"
        if not args.release_precision
        else "reported (release precision, not judged)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
