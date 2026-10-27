"""Named state-dict alignment and weight transfer for model parity tests.

Why named transfer instead of positional
-----------------------------------------
Positional transfer (iterating ``model.parameters()`` in order) fails
silently whenever two models use the same layer shapes but in a
different traversal order (e.g. DenseNet transition-layer BN).  Named
transfer matches by key, catches order differences explicitly, and
gives actionable diagnostics.

API
---
``align(lucid_sd, timm_sd, remap)``
    Returns an ``AlignResult`` describing which keys matched, which
    were auto-remapped (common head patterns), and which are missing
    in one or the other model.

``transfer(lucid_model, timm_model, remap)``
    Copies Lucid parameters → timm model using named alignment.
    Returns the ``AlignResult`` so callers can inspect coverage.

``diagnose_forward(lucid_model, timm_model, x_np, remap)``
    Registers forward hooks on both models, runs a forward pass, and
    returns a per-layer comparison table so callers can pinpoint the
    first diverging layer.
"""

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

# ── Common head key remappings (Lucid → timm) ─────────────────────────────────
# Lucid always names the classification head ``classifier``.
# timm uses different names per architecture family.

_AUTO_HEAD_REMAP: dict[str, list[str]] = {
    "classifier.weight": [
        "fc.weight",  # ResNet, ResNeXt, SENet, SKNet, ResNeSt, Inception v3
        "head.weight",  # ViT, PVT
        "head.fc.weight",  # Swin, ConvNeXt, MaxViT
        "last_linear.weight",  # Inception v4
        "classif.weight",  # Inception-ResNet v2
        "classifier.weight",  # DenseNet, EfficientNet, MobileNetV2
    ],
    "classifier.bias": [
        "fc.bias",
        "head.bias",
        "head.fc.bias",
        "last_linear.bias",
        "classif.bias",
        "classifier.bias",
    ],
}


# ── Result type ────────────────────────────────────────────────────────────────


@dataclass
class AlignResult:
    """Outcome of aligning two state dicts."""

    matched: dict[str, str]  # lucid_key → timm_key (exact or remapped)
    unmatched_lucid: list[str]  # keys in Lucid with no timm counterpart
    unmatched_timm: list[str]  # keys in timm with no Lucid counterpart
    auto_remapped: dict[str, str]  # subset of ``matched`` that used auto-remap

    @property
    def coverage(self) -> float:
        """Fraction of Lucid keys that found a timm match."""
        total = len(self.matched) + len(self.unmatched_lucid)
        return len(self.matched) / total if total else 0.0

    def assert_full_coverage(self) -> None:
        """Raise AssertionError if any key is unmatched."""
        if self.unmatched_lucid or self.unmatched_timm:
            lines = ["State-dict alignment is incomplete:"]
            if self.unmatched_lucid:
                lines.append(f"  Lucid keys with no timm match ({len(self.unmatched_lucid)}):")
                for k in self.unmatched_lucid[:8]:
                    lines.append(f"    {k}")
                if len(self.unmatched_lucid) > 8:
                    lines.append(f"    … and {len(self.unmatched_lucid) - 8} more")
            if self.unmatched_timm:
                lines.append(f"  timm keys with no Lucid match ({len(self.unmatched_timm)}):")
                for k in self.unmatched_timm[:8]:
                    lines.append(f"    {k}")
            raise AssertionError("\n".join(lines))

    def summary(self) -> str:
        pct = self.coverage * 100
        n = len(self.matched)
        r = len(self.auto_remapped)
        u_l = len(self.unmatched_lucid)
        u_t = len(self.unmatched_timm)
        return (
            f"aligned {n} keys ({pct:.0f}% coverage), "
            f"{r} auto-remapped, "
            f"{u_l} unmatched-lucid, {u_t} unmatched-timm"
        )


# ── Core alignment logic ──────────────────────────────────────────────────────


def align(
    lucid_sd: dict[str, np.ndarray],
    timm_sd: dict[str, np.ndarray],
    remap: dict[str, str] | None = None,
    key_transform: Callable[[str], str] | None = None,
) -> AlignResult:
    """Align Lucid state dict keys to timm state dict keys.

    Priority order for each Lucid key:
    0. Apply ``key_transform(lucid_key)`` to produce a candidate timm name.
    1. Explicit ``remap`` override (lucid_key → timm_key).
    2. Exact key match (same name in both dicts, or transformed name).
    3. Auto head-remap (``_AUTO_HEAD_REMAP`` table).
    4. Unmatched — reported but not fatal.
    """
    explicit = remap or {}

    matched: dict[str, str] = {}
    auto_remapped: dict[str, str] = {}
    unmatched_lucid: list[str] = []
    remaining_timm: set[str] = set(timm_sd.keys())

    for lk in lucid_sd:
        # 0 — apply key_transform to get candidate timm key
        transformed = key_transform(lk) if key_transform else lk

        # 1 — explicit override (takes priority over transform)
        if lk in explicit:
            tk = explicit[lk]
            if tk in remaining_timm:
                matched[lk] = tk
                remaining_timm.discard(tk)
                continue
            # Override pointed to a missing timm key — fall through

        # 2 — transformed name exact match
        if transformed in remaining_timm:
            matched[lk] = transformed
            if transformed != lk:
                auto_remapped[lk] = transformed
            remaining_timm.discard(transformed)
            continue

        # 3 — auto head remap (applied to the *original* key name)
        if lk in _AUTO_HEAD_REMAP:
            for candidate in _AUTO_HEAD_REMAP[lk]:
                if candidate in remaining_timm:
                    matched[lk] = candidate
                    auto_remapped[lk] = candidate
                    remaining_timm.discard(candidate)
                    break
            else:
                unmatched_lucid.append(lk)
            continue

        unmatched_lucid.append(lk)

    return AlignResult(
        matched=matched,
        unmatched_lucid=unmatched_lucid,
        unmatched_timm=sorted(remaining_timm),
        auto_remapped=auto_remapped,
    )


# ── Weight transfer ────────────────────────────────────────────────────────────


def transfer(
    lucid_model: Any,
    timm_model: Any,
    remap: dict[str, str] | None = None,
    key_transform: Callable[[str], str] | None = None,
) -> AlignResult:
    """Copy Lucid weights → timm model using named alignment.

    Parameters
    ----------
    lucid_model:
        Lucid model (``lucid.nn.Module`` subclass).
    timm_model:
        timm / reference-framework model.
    remap:
        Optional explicit overrides ``{lucid_key: timm_key}``.

    Returns
    -------
    AlignResult
        Alignment statistics.  Caller decides whether to assert full
        coverage or accept partial alignment.
    """
    from lucid.test._fixtures.ref_framework import require_ref

    _ref = require_ref()

    # Build numpy views of both state dicts
    lucid_sd_np: dict[str, np.ndarray] = {
        k: v.data.numpy() for k, v in lucid_model.state_dict().items()
    }
    timm_sd = dict(timm_model.state_dict())

    timm_sd_np: dict[str, np.ndarray] = {k: v.detach().cpu().numpy() for k, v in timm_sd.items()}

    result = align(lucid_sd_np, timm_sd_np, remap, key_transform)

    # Copy matched weights Lucid → timm
    # Copy matched weights, handling reshape-compatible mismatches transparently.
    # Linear(n, m) ↔ Conv2d with the same total elements are mathematically
    # equivalent (e.g. VGG fc6 Linear(25088,4096) vs timm Conv2d(512,4096,7,7)).
    true_mismatches: list[str] = []
    for lk, tk in list(result.matched.items()):
        arr = lucid_sd_np[lk]
        expected_shape = tuple(timm_sd[tk].shape)
        if arr.shape != expected_shape:
            if arr.size == int(np.prod(expected_shape)):
                arr = arr.reshape(expected_shape)
            else:
                true_mismatches.append(f"{lk} → {tk}: lucid {arr.shape} vs timm {expected_shape}")
                del result.matched[lk]
                result.unmatched_lucid.append(lk)
                result.unmatched_timm.append(tk)
                continue
        timm_sd[tk].copy_(_ref.from_numpy(arr))

    if true_mismatches:
        import warnings

        warnings.warn(
            f"Shape mismatch during weight transfer ({len(true_mismatches)} keys):\n"
            + "\n".join(f"  {m}" for m in true_mismatches[:5]),
            stacklevel=2,
        )

    return result


# ── Positional fallback transfer ─────────────────────────────────────────────


def transfer_positional(lucid_model: Any, timm_model: Any) -> None:
    """Copy Lucid weights → timm by matching state positionally.

    Used as a fallback when named alignment fails (e.g. when our module
    attribute names differ from timm's but shapes / order match).
    Raises ``AssertionError`` on count or shape mismatch.

    Parameters *and* buffers.  It copied only parameters for a long
    while, which is a quiet way to be wrong: the running statistics of
    every BatchNorm stayed at whatever each side happened to hold, and
    since ``_randomise_norm_buffers`` deliberately makes those differ,
    the comparison then measured the statistics rather than the model.
    Thirty-three specs lean on this fallback; the ResNet family read as
    a numerical failure with the divergence sitting entirely in ``bn``
    layers, which is exactly what a missing buffer looks like.
    """
    from lucid.test._fixtures.ref_framework import require_ref

    _ref = require_ref()

    lparams = list(lucid_model.parameters())
    tparams = list(timm_model.parameters())

    if len(lparams) != len(tparams):
        raise AssertionError(
            f"positional transfer: param count mismatch "
            f"lucid={len(lparams)} timm={len(tparams)}"
        )

    lbuffers = list(lucid_model.buffers())
    tbuffers = list(timm_model.buffers())

    if len(lbuffers) != len(tbuffers):
        raise AssertionError(
            f"positional transfer: buffer count mismatch "
            f"lucid={len(lbuffers)} timm={len(tbuffers)}"
        )

    mismatches: list[str] = []

    def _copy(index: int, source: Any, destination: Any, what: str) -> None:
        arr = source.numpy() if hasattr(source, "numpy") else source.data.numpy()
        # A 0-d scalar and a length-1 vector are the same value here:
        # num_batches_tracked is stored one way on each side.
        trivial = arr.shape in ((), (1,)) and tuple(destination.shape) in ((), (1,))
        if arr.shape != tuple(destination.shape) and not trivial:
            mismatches.append(
                f"  [{what} {index}] lucid={arr.shape} " f"vs timm={tuple(destination.shape)}"
            )
            return
        destination.data.copy_(_ref.from_numpy(arr).reshape(destination.shape))

    for i, (lp, tp) in enumerate(zip(lparams, tparams)):
        _copy(i, lp.data, tp, "param")
    for i, (lb, tb) in enumerate(zip(lbuffers, tbuffers)):
        _copy(i, lb, tb, "buffer")

    if mismatches:
        raise AssertionError("positional transfer shape mismatches:\n" + "\n".join(mismatches[:5]))


# ── Layer-by-layer diagnostic ─────────────────────────────────────────────────


@dataclass
class LayerDiff:
    name: str
    max_abs: float
    mean_abs: float
    shape: tuple[int, ...]

    def __str__(self) -> str:
        return (
            f"{self.name:<60s} "
            f"max={self.max_abs:.2e}  mean={self.mean_abs:.2e}  "
            f"shape={self.shape}"
        )


def diagnose_forward(
    lucid_model: Any,
    timm_model: Any,
    x_np: np.ndarray,
    remap: dict[str, str] | None = None,
    key_transform: Callable[[str], str] | None = None,
    *,
    top_k: int = 10,
) -> list[LayerDiff]:
    """Register hooks, run both models, return per-layer diff table.

    Only layers that exist in BOTH models under matching names are
    compared.  Returns the ``top_k`` worst-diverging layers.

    Useful for diagnosing *where* divergence starts in a failing test.
    """
    import lucid
    from lucid.test._fixtures.ref_framework import require_ref

    _ref = require_ref()

    transfer(lucid_model, timm_model, remap, key_transform)

    lucid_acts: dict[str, np.ndarray] = {}
    timm_acts: dict[str, np.ndarray] = {}

    # --- Lucid hooks ---
    lucid_handles = []
    for name, mod in lucid_model.named_modules():
        if name == "":
            continue

        def _hook_lucid(m: Any, inp: Any, out: Any, _n: str = name) -> None:
            if hasattr(out, "numpy"):
                # .copy(): see the note on the reference hook below.
                lucid_acts[_n] = out.numpy().copy()

        lucid_handles.append(mod.register_forward_hook(_hook_lucid))

    # --- timm hooks ---
    timm_handles = []
    for name, mod in timm_model.named_modules():
        if name == "":
            continue

        def _hook_timm(m: Any, inp: Any, out: Any, _n: str = name) -> None:
            if isinstance(out, _ref.Tensor):
                # ``.numpy()`` shares memory with the tensor, and the
                # reference's residual blocks activate in place, so the
                # array recorded here was being overwritten with the
                # post-activation values a moment later.  Every layer
                # followed by an inplace ReLU then read as a large
                # divergence — which is every ``bn`` in a ResNet, and is
                # exactly what the table showed.  Copy at capture time.
                timm_acts[_n] = out.detach().cpu().numpy().copy()

        timm_handles.append(mod.register_forward_hook(_hook_timm))

    try:
        lucid_model.eval()
        timm_model.eval()

        lucid_model(lucid.from_numpy(x_np.copy()))
        with _ref.no_grad():
            timm_model(_ref.from_numpy(x_np.copy()))
    finally:
        for h in lucid_handles + timm_handles:
            h.remove()

    # Compare common layer names
    diffs: list[LayerDiff] = []
    common = set(lucid_acts) & set(timm_acts)
    for name in common:
        la = lucid_acts[name]
        ta = timm_acts[name]
        if la.shape != ta.shape:
            continue
        diff = np.abs(la - ta)
        diffs.append(
            LayerDiff(
                name=name,
                max_abs=float(diff.max()),
                mean_abs=float(diff.mean()),
                shape=tuple(la.shape),
            )
        )

    diffs.sort(key=lambda d: d.max_abs, reverse=True)
    return diffs[:top_k]
