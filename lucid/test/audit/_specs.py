"""How to call each op — the part that decides whether a census is one.

The first hand-rolled sweep of this framework could only build ``f(x)``
and ``f(x, y)`` over a single small tensor, so 335 of 487 names were never
evaluated: every convolution, every pooling layer, every loss,
``batch_norm``, attention.  Coverage was 29.5% of the op surface and the
gap was invisible in the output.  This module is the fix, and the reason
:class:`~lucid.test.audit._result.Status` has a SKIP that gets counted
and listed rather than dropped.

A spec yields **candidate invocations in order**, not one.  Signatures
differ in ways that are tedious to encode exactly — ``dim`` versus
``axis``, ``keepdim`` versus ``keepdims``, a kernel size that may be an
int or a tuple — and a spec that guesses wrong should degrade to trying
the next form rather than reporting a defect that is really a mis-call.
When every candidate fails, the symbol is reported SKIP with the last
error, which is honest and is what tells you a spec needs adding.
"""

import annotationlib
import inspect
import re
from typing import TYPE_CHECKING, Any

import numpy as np

import lucid
import lucid.fft
import lucid.linalg
import lucid.nn.utils.rnn
from lucid.test.audit import _probe, _surface

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator


def _rebuild(original: Any, array: Any, *, keep_integer: bool = False) -> Any:
    """``array`` as a tensor at ``original``'s dtype family.

    A float32 or float16 operand comes back float32, a complex one at its
    width, anything else float64 — or, with ``keep_integer``, an integer
    operand (a class index) at its own integer dtype.
    """
    dtype = getattr(original, "dtype", None)
    text = str(dtype) if dtype is not None else ""
    if "complex" in text:
        return (_probe.as_complex64 if "64" in text else _probe.as_complex)(array)
    if "float32" in text or "float16" in text:
        return _probe.as_f32(array)
    if keep_integer and "int" in text:
        return _probe.tensor_of(np.asarray(array, dtype=np.int64), dtype)
    return _probe.as_f64(array)


def _own(value: Any) -> Any:
    """``value`` on storage of its own when it is a tensor, else ``value``."""
    if not isinstance(value, lucid.Tensor):
        return value
    copy = value.detach().clone()
    return copy.requires_grad_() if value.requires_grad else copy


def _strided_leaf(value: Any) -> Any:
    """``value``'s numbers as a non-contiguous *leaf*: every second element
    of a buffer twice as wide, detached so a gradient stops at it."""
    if not isinstance(value, lucid.Tensor) or value.ndim == 0:
        return value
    values = _probe.to_numpy(value)
    padded = np.zeros((*values.shape[:-1], 2 * values.shape[-1]), dtype=values.dtype)
    padded[..., ::2] = values
    wide = _probe.tensor_of(padded, dtype=value.dtype, device=value.device.type)
    leaf = wide[..., ::2].detach()
    return leaf.requires_grad_() if value.requires_grad else leaf


class Call:
    """One concrete invocation of an op.

    Attributes
    ----------
    args : list
        Positional arguments, already built.
    kwargs : dict
        Keyword arguments.
    primary : int
        Index into ``args`` of the tensor the numeric axes differentiate
        and perturb.  Not always zero — ``embedding`` differentiates its
        weight, not its indices.
    note : str
        How this invocation was chosen, for the SKIP message.
    same_shape : tuple of int
        Indices into ``args`` of the operands that must keep the
        primary's shape — a loss's target, a variance — so that an axis
        which *resizes* the primary (:meth:`resized`) resizes them with
        it.  An axis that only changes values never looks at it.
    domain : (float, float) or None
        The closed interval the primary's values are part of the op's
        contract over — ``binary_cross_entropy``'s probabilities, which
        the reference's CPU kernel (and Lucid's) refuses outside
        ``[0, 1]``.  An axis that rebuilds the primary at another dtype
        keeps the call's own values rather than moving them to
        ``|x| + 1``, out of the domain; and a ``ValueError`` for a
        non-finite probe is the op's answer, not a gap in it.
    joint : callable or None
        For an op whose operands constrain each other by more than a
        shared shape, the rule that rebuilds the whole call around a
        resized primary — ``(call, array) -> Call``.  Set from
        :data:`_JOINT_RESIZE` by :func:`invocations`; see there.
    strided : bool
        The primary is handed over as a non-contiguous leaf
        (:func:`_strided_leaf`) however the call is rebuilt.  For an op
        whose layout decides its path: ``contiguous`` returns a packed
        input itself, with no node, so only a strided one reaches the copy
        that has a backward.
    """

    __slots__ = (
        "args",
        "kwargs",
        "primary",
        "note",
        "same_shape",
        "domain",
        "joint",
        "strided",
    )

    def __init__(
        self,
        args: list[Any],
        kwargs: dict[str, Any] | None = None,
        primary: int = 0,
        note: str = "",
        same_shape: tuple[int, ...] = (),
        domain: tuple[float, float] | None = None,
        joint: Callable[[Call, np.ndarray], Call] | None = None,
        strided: bool = False,
    ) -> None:
        if strided and 0 <= primary < len(args):
            args = list(args)
            args[primary] = _strided_leaf(args[primary])
        self.args = args
        self.kwargs = kwargs or {}
        self.primary = primary
        self.note = note
        self.same_shape = same_shape
        self.domain = domain
        self.joint = joint
        self.strided = strided

    def with_primary(self, array: np.ndarray) -> Call:
        """A copy whose differentiated argument is replaced by ``array``.

        At the *original argument's* dtype, not unconditionally float64.
        The axes mean "the same call with different values", and rebuilding
        at float64 changed the call instead: the whole ``fft`` family
        accepts only ``F16/F32/C64`` and every substituting axis —
        nonfinite, layout, grad, creategraph, grad2 — handed it a float64
        probe and was told ``fftn requires F16/F32/C64 input, got
        float64``.  Sixty cells filed as "the op refused" about a refusal
        the probe had provoked.
        """
        args = list(self.args)
        original = args[self.primary] if 0 <= self.primary < len(args) else None
        args[self.primary] = _rebuild(original, array)
        return Call(
            args,
            dict(self.kwargs),
            self.primary,
            self.note,
            self.same_shape,
            self.domain,
            self.joint,
            self.strided,
        )

    def fresh(self) -> Call:
        """A copy whose tensor operands are copies too.

        For an axis that runs one call more than once and compares the
        answers.  An in-place op writes into its operand, so a second run
        on the same arguments reads what the first one wrote: the device
        axis compared ``x.bitwise_xor_(y)`` on the CPU against
        ``x ^ y ^ y`` on Metal, because ``.to("cpu")`` of a CPU tensor is
        that tensor (LCD-532), and the entry axis's second in-place spelling
        was refused for writing into storage the first answer was still
        read through.
        """
        return Call(
            [_own(arg) for arg in self.args],
            {name: _own(value) for name, value in self.kwargs.items()},
            self.primary,
            self.note,
            self.same_shape,
            self.domain,
            self.joint,
            self.strided,
        )

    def resized(self, array: np.ndarray) -> Call:
        """:meth:`with_primary` for an ``array`` of *another shape*.

        Every operand declared in ``same_shape`` follows the primary to the
        new shape, filled with its own first value at its own dtype — a
        probability stays a probability, a class index stays an index.
        Emptying only the primary of ``bce(input, target)`` asked whether
        the loss takes an empty input *against a full target*, and a loss
        that refuses a target of another shape (as the reference does)
        was filed as refusing empty input.

        A call with a ``joint`` rule is rebuilt by that rule instead: its
        operands are related by more than one shape.
        """
        if self.joint is not None:
            return self.joint(self, array)
        call = self.with_primary(array)
        for index in self.same_shape:
            peer = self.args[index]
            values = _probe.to_numpy(peer)
            fill = values.flat[0] if values is not None and values.size else 0
            call.args[index] = _rebuild(
                peer, np.full(array.shape, fill), keep_integer=True
            )
        return call

    @property
    def base(self) -> np.ndarray:
        """The differentiated argument, as an array.

        Raises ``TypeError`` — never ``IndexError`` — when there is no
        such argument.  A keyword-only op like ``affine_matrix(*, cx, cy)``
        has an empty ``args``, and the numeric axes already read a
        TypeError here as "nothing to differentiate, skip"; an IndexError
        would escape as a harness ERROR instead.
        """
        if not 0 <= self.primary < len(self.args):
            raise TypeError("no positional argument to differentiate")
        got = _probe.to_numpy(self.args[self.primary])
        if got is None:
            raise TypeError("primary argument is not a tensor")
        return np.asarray(got, dtype=np.float64)


# ── shape vocabulary ─────────────────────────────────────────────────────────
# Small on purpose: every numeric axis costs one forward per element of
# the differentiated tensor, so the probes stay tiny while still having
# more than one channel, more than one batch element and a spatial extent
# that survives a stride-2 convolution.

_N, _CIN, _COUT = 2, 3, 4
_L, _H, _W = 8, 6, 6
_D = 4


#: How many times each ``(shape, domain)`` has been drawn while building
#: the current invocation.  Reset by :func:`invocations`.
_DRAWN: dict[tuple[Any, ...], int] = {}


def _f(shape: tuple[int, ...], domain: str = "moderate") -> Any:
    """One float operand.  Repeat draws of the same shape differ.

    Two operands built the same way used to be the *same numbers*, and
    that is the one input where several checks cannot fail:

    * every regression loss was probed as ``loss(x, x)``, which sits at
      its exact minimum, so the gradient is identically zero and the grad
      axis had nothing to compare.  Thirteen losses, all reported vacuous
      — honestly, but that meant the whole family's derivative was going
      unmeasured;
    * ``where(cond, a, a)`` cannot tell the two branches apart;
    * ``lerp(a, a, w)`` is constant in ``w``;
    * ``maximum(a, a)`` sits exactly on its tie, where the analytic
      gradient reports a convention and a central difference reports the
      average of two one-sided limits.

    The variant counter is keyed on ``(shape, domain)``, so a spec that
    asks for one operand of a given shape gets exactly what it got
    before; only the second and later repeats move.  Still fully
    deterministic — the variant selects another fixed seed.

    Where a spec passes the *same variable* into two slots, this cannot
    help; those are fixed at the call site.
    """
    key = (tuple(shape), domain)
    variant = _DRAWN.get(key, 0)
    _DRAWN[key] = variant + 1
    return _probe.as_f64(_probe.sample(domain, shape, variant))


def _int(shape: tuple[int, ...], high: int) -> Any:
    return _probe.as_int(_probe.rng(_probe.SEED_B).integers(0, high, shape))


def _spatial(rank: int) -> tuple[int, ...]:
    return {1: (_L,), 2: (_H, _W), 3: (_D, _H, _W)}[rank]


def _rank_of(name: str) -> int:
    match = re.search(r"([123])d$", name)
    return int(match.group(1)) if match else 2


# ── family builders ──────────────────────────────────────────────────────────


def _conv(name: str, domain: str) -> Iterator[Call]:
    rank = _rank_of(name)
    spatial = _spatial(rank)
    k = (3,) * rank
    transposed = "transpose" in name
    w_shape = (_CIN, _COUT, *k) if transposed else (_COUT, _CIN, *k)
    x = _f((_N, _CIN, *spatial), domain)
    w = _f(w_shape, domain)
    b = _f((_COUT,), domain)
    yield Call([x, w, b], {"stride": 1, "padding": 1}, 0, "conv(x, weight, bias)")
    yield Call([x, w, b], {}, 0, "conv(x, weight, bias) defaults")
    yield Call([x, w], {}, 0, "conv(x, weight)")


def _pool(name: str, domain: str) -> Iterator[Call]:
    rank = _rank_of(name)
    x = _f((_N, _CIN, *_spatial(rank)), domain)
    yield Call([x, 2], {}, 0, "pool(x, kernel_size=2)")
    yield Call([x], {"kernel_size": 2}, 0, "pool(x, kernel_size=2) kw")
    yield Call([x, (2,) * rank], {}, 0, "pool(x, kernel_size tuple)")


def _adaptive_pool(name: str, domain: str) -> Iterator[Call]:
    rank = _rank_of(name)
    x = _f((_N, _CIN, *_spatial(rank)), domain)
    out = (2,) * rank if rank > 1 else 2
    yield Call([x, out], {}, 0, "adaptive_pool(x, output_size)")
    yield Call([x], {"output_size": out}, 0, "adaptive_pool kw")


def _fractional_pool(name: str, domain: str) -> Iterator[Call]:
    rank = _rank_of(name)
    x = _f((_N, _CIN, *_spatial(rank)), domain)
    yield Call([x, 2], {"output_ratio": 0.5}, 0, "fractional_pool(x, k, output_ratio)")
    yield Call(
        [x, 2], {"output_size": (2,) * rank}, 0, "fractional_pool(x, k, output_size)"
    )


def _norm(name: str, domain: str) -> Iterator[Call]:
    x = _f((_N, _CIN, _H, _W), domain)
    weight, bias = _f((_CIN,), "positive"), _f((_CIN,), domain)
    if "batch" in name:
        mean, var = _f((_CIN,), domain), _f((_CIN,), "positive")
        yield Call(
            [x, mean, var, weight, bias], {"training": True}, 0, "batch_norm full"
        )
        yield Call(
            [x, None, None, weight, bias], {"training": True}, 0, "batch_norm no stats"
        )
        yield Call([x, mean, var], {}, 0, "batch_norm(x, mean, var)")
    elif "layer" in name:
        flat = _f((_N, _CIN * 2), domain)
        yield Call(
            [flat, (_CIN * 2,), _f((_CIN * 2,), "positive"), _f((_CIN * 2,), domain)],
            {},
            0,
            "layer_norm(x, shape, w, b)",
        )
        yield Call([flat, (_CIN * 2,)], {}, 0, "layer_norm(x, shape)")
    elif "group" in name:
        yield Call([x, 1, weight, bias], {}, 0, "group_norm(x, groups, w, b)")
        yield Call([x, 1], {}, 0, "group_norm(x, groups)")
    elif "instance" in name:
        yield Call([x], {}, 0, "instance_norm(x)")
        yield Call([x, None, None, weight, bias], {}, 0, "instance_norm full")
    elif "local_response" in name:
        yield Call([x, 3], {}, 0, "local_response_norm(x, size)")
    else:
        yield Call([x], {}, 0, "norm(x)")


#: Losses grouped by what their target has to look like.  Getting this
#: wrong is not a defect in the loss, so each group is tried in turn.
_CLASS_LOSSES = ("cross_entropy", "nll_loss", "multi_margin", "multilabel_margin")
_PROB_LOSSES = ("binary_cross_entropy", "bce", "kl_div", "poisson_nll")
_PAIR_LOSSES = ("cosine_embedding", "margin_ranking", "hinge_embedding", "triplet")


def _loss(name: str, domain: str) -> Iterator[Call]:
    classes = 4
    logits = _f((_N, classes), "moderate")
    target_idx = _int((_N,), classes)
    same = _f((_N, classes), domain)
    probs = _f((_N, classes), "small_pos")

    # ``same_shape`` names each operand that has to keep the input's shape:
    # a resizing axis resizes it with the input (see ``Call.resized``).
    if any(k in name for k in _CLASS_LOSSES):
        yield Call([logits, target_idx], {}, 0, "loss(logits, class index)")
    if any(k in name for k in _PROB_LOSSES):
        yield Call(
            [probs, _f((_N, classes), "small_pos")],
            {},
            0,
            "loss(prob, prob)",
            same_shape=(1,),
            # A probability is binary_cross_entropy's contract; it refuses
            # one outside [0, 1] on the CPU, as the reference does.
            domain=(0.0, 1.0) if name == "binary_cross_entropy" else None,
        )
    if any(k in name for k in _PAIR_LOSSES):
        sign = _probe.as_f64(np.array([1.0, -1.0]))
        # The negative was the anchor.  A triplet loss whose anchor and
        # negative coincide is at a corner of its own hinge, and its
        # gradient there is neither what the formula gives nor stable
        # under a finite difference — the check could not fail.
        negative = _f((_N, classes), domain)
        yield Call(
            [logits, same, sign], {}, 0, "loss(a, b, target sign)", same_shape=(1,)
        )
        yield Call(
            [logits, same, negative],
            {},
            0,
            "loss(anchor, positive, negative)",
            same_shape=(1, 2),
        )
    # The elementwise regressions, and the fallback for anything unmatched.
    yield Call([logits, same], {}, 0, "loss(input, target) same shape", same_shape=(1,))
    yield Call([logits, target_idx], {}, 0, "loss(logits, class index)")


def _embedding(name: str, domain: str) -> Iterator[Call]:
    vocab, dim = 6, 4
    weight = _f((vocab, dim), domain)
    idx = _int((_N, 3), vocab)
    if "bag" in name:
        offsets = _probe.as_int(np.array([0, 3]))
        yield Call(
            [idx.reshape(-1), weight, offsets],
            {},
            1,
            "embedding_bag(idx, weight, offsets)",
        )
        yield Call([idx, weight], {}, 1, "embedding_bag(idx2d, weight)")
    else:
        yield Call([idx, weight], {}, 1, "embedding(idx, weight)")


def _attention(name: str, domain: str) -> Iterator[Call]:
    heads, seq, head_dim = 2, 4, 4
    q = _f((_N, heads, seq, head_dim), domain)
    k = _f((_N, heads, seq, head_dim), domain)
    v = _f((_N, heads, seq, head_dim), domain)
    yield Call([q, k, v], {}, 0, "sdpa(q, k, v)")
    yield Call([q, k, v], {"is_causal": True}, 0, "sdpa causal")


def _resample(name: str, domain: str) -> Iterator[Call]:
    x = _f((_N, _CIN, _H, _W), domain)
    if "grid_sample" in name:
        grid = _f((_N, _H, _W, 2), "unit")
        yield Call([x, grid], {}, 0, "grid_sample(x, grid)")
    elif "affine_grid" in name:
        theta = _f((_N, 2, 3), domain)
        yield Call([theta, (_N, _CIN, _H, _W)], {}, 0, "affine_grid(theta, size)")
    elif "interpolate" in name or "upsample" in name:
        yield Call(
            [x], {"scale_factor": 2.0, "mode": "nearest"}, 0, "interpolate nearest"
        )
        yield Call([x], {"size": (_H * 2, _W * 2)}, 0, "interpolate size")
    elif "pixel_shuffle" in name:
        yield Call([_f((_N, 4, _H, _W), domain), 2], {}, 0, "pixel_shuffle(x, 2)")
    elif "pixel_unshuffle" in name:
        yield Call([x, 2], {}, 0, "pixel_unshuffle(x, 2)")
    elif "channel_shuffle" in name:
        yield Call([x, 3], {}, 0, "channel_shuffle(x, groups)")
    elif name.endswith("unfold"):
        yield Call([x, 3], {"padding": 1}, 0, "unfold(x, kernel)")
    elif name.endswith("fold"):
        cols = _f((_N, _CIN * 9, _H * _W), domain)
        yield Call([cols, (_H, _W), 3], {"padding": 1}, 0, "fold(cols, size, kernel)")
    elif name.endswith("pad"):
        yield Call([x, (1, 1, 1, 1)], {}, 0, "pad(x, padding)")
        yield Call([x, (1, 1)], {}, 0, "pad(x, padding pair)")


def _linalg(name: str, domain: str) -> Iterator[Call]:
    n = 4
    square = _probe.rng(_probe.SEED_X).standard_normal((n, n)) * 0.4 + np.eye(n) * 2.0
    rect = _probe.sample(domain, (n, 3))
    psd = square @ square.T + np.eye(n) * 0.5
    if name == "pinv":
        # Rectangular first: the axes read the first call, and a square
        # input is the one case a pseudo-inverse shares with ``inv`` — the
        # rectangular path dropped its gradient for months behind it.
        yield Call([_probe.as_f64(rect)], {}, 0, "pinv(rectangular)")
    if any(k in name for k in ("cholesky", "eigh", "inv_psd")):
        yield Call([_probe.as_f64(psd)], {}, 0, "linalg(PSD)")
    if any(k in name for k in ("solve", "lstsq")):
        yield Call(
            [_probe.as_f64(square), _probe.as_f64(_probe.sample(domain, (n, 2)))],
            {},
            0,
            "linalg(A, B)",
        )
    if "matrix_power" in name:
        yield Call([_probe.as_f64(square), 2], {}, 0, "matrix_power(A, 2)")
    if any(k in name for k in ("matmul", "dot", "outer", "kron", "cross")):
        yield Call(
            [_probe.as_f64(rect), _probe.as_f64(rect.T.copy())], {}, 0, "linalg(A, B)"
        )
    yield Call([_probe.as_f64(square)], {}, 0, "linalg(square)")
    yield Call([_probe.as_f64(rect)], {}, 0, "linalg(rectangular)")


def _fft(name: str, domain: str) -> Iterator[Call]:
    x = _probe.as_f64(_probe.sample(domain, (_N, 8)))
    if name.endswith(("shift",)):
        yield Call([x], {}, 0, "fftshift(x)")
        return
    if "2" in name:
        yield Call([_probe.as_f64(_probe.sample(domain, (4, 8)))], {}, 0, "fft2(x)")
    yield Call([x], {}, 0, "fft(x)")
    yield Call([x], {"n": 8}, 0, "fft(x, n)")


def _einops(name: str, domain: str) -> Iterator[Call]:
    x = _f((_N, _CIN, _H, _W), domain)
    if "rearrange" in name:
        yield Call([x, "b c h w -> b h w c"], {}, 0, "rearrange")
    elif "reduce" in name:
        yield Call([x, "b c h w -> b c", "mean"], {}, 0, "reduce")
    elif "repeat" in name:
        yield Call(
            [
                x,
                "b c h w -> b c h w r",
            ],
            {"r": 2},
            0,
            "repeat",
        )
    elif "pack" in name or "unpack" in name:
        yield Call([[x, x], "b c h *"], {}, 0, "pack")


def _reduction(name: str, domain: str) -> Iterator[Call]:
    x = _f(_probe.SHAPE, domain)
    yield Call([x], {}, 0, "reduce(x)")
    yield Call([x], {"dim": -1}, 0, "reduce(x, dim=-1)")
    yield Call([x, -1], {}, 0, "reduce(x, -1)")


def _binary(name: str, domain: str) -> Iterator[Call]:
    a = _f(_probe.SHAPE, domain)
    b = _f(_probe.SHAPE, "positive")
    yield Call([a, b], {}, 0, "binary(a, b)")


def _unary(name: str, domain: str) -> Iterator[Call]:
    yield Call([_f(_probe.SHAPE, domain)], {}, 0, "unary(x)")


# ── the groups a first depth probe could not call ────────────────────────────
# 189 of 716 symbols had no invocation the harness could build.  They were
# not scattered: every one belonged to a family with a shape convention,
# and each block below closes a whole family.  The numeric axes ride on
# this, so a symbol that cannot be called here is a symbol whose gradient
# is never checked.


def _ternary(name: str, domain: str) -> Iterator[Call]:
    """``add*`` fused forms: an accumulator plus two operands."""
    v = _f((_COUT,), domain)
    m = _f((_COUT, _CIN), domain)
    n = _f((_CIN, _COUT), domain)
    batch_a = _f((_N, _COUT, _CIN), domain)
    batch_b = _f((_N, _CIN, _COUT), domain)
    square = _f((_COUT, _COUT), domain)
    same = _f(_probe.SHAPE, domain)
    if name in ("addbmm",):
        yield Call([square, batch_a, batch_b], {}, 0, "addbmm(bias, batch1, batch2)")
    if name in ("baddbmm",):
        yield Call([_f((_N, _COUT, _COUT), domain), batch_a, batch_b], {}, 0, "baddbmm")
    if name in ("addmm",):
        yield Call([square, m, n], {}, 0, "addmm(bias, mat1, mat2)")
    if name in ("addmv",):
        yield Call([v, m, _f((_CIN,), domain)], {}, 0, "addmv(bias, mat, vec)")
    if name in ("addr",):
        yield Call([m, v, _f((_CIN,), domain)], {}, 0, "addr(mat, vec1, vec2)")
    if name in ("addcmul", "addcdiv"):
        # ``t`` and ``a`` were the same variable, so the accumulate and
        # the multiplicand were the same numbers — an operand pair the
        # gradient cannot tell apart.
        addend = _f(_probe.SHAPE, domain)
        yield Call(
            [same, addend, _f(_probe.SHAPE, "positive")],
            {"value": 0.5},
            0,
            "addc*(t, a, b)",
        )
        yield Call(
            [same, addend, _f(_probe.SHAPE, "positive")], {}, 0, "addc*(t, a, b)"
        )
    if name == "lerp":
        yield Call([same, _f(_probe.SHAPE, domain), 0.3], {}, 0, "lerp(a, b, weight)")


def _bitwise(name: str, domain: str) -> Iterator[Call]:
    """Integer-only ops.  A float probe is rejected before the op is reached."""
    a = _probe.as_int(_probe.rng(1).integers(1, 30, _probe.SHAPE))
    b = _probe.as_int(_probe.rng(2).integers(1, 6, _probe.SHAPE))
    if "not" in name or "invert" in name:
        yield Call([a], {}, 0, "bitwise_not(int)")
        return
    yield Call([a, b], {}, 0, "bitwise(int, int)")
    yield Call([a, 2], {}, 0, "bitwise(int, scalar)")


def _factory(name: str, domain: str) -> Iterator[Call]:
    """Constructors take a shape or a range, never an input tensor.

    ``primary`` still points at argument zero, so the numeric axes will
    report SKIP — correctly, since there is no input to differentiate.
    """
    shape = (2, 3)
    if name in ("arange",):
        yield Call([0.0, 6.0, 1.0], {}, 0, "arange(start, stop, step)")
        yield Call([6], {}, 0, "arange(n)")
    elif name in ("linspace", "logspace"):
        yield Call([0.0, 1.0, 5], {}, 0, "linspace(start, stop, num)")
    elif name in ("eye",):
        yield Call([3], {}, 0, "eye(n)")
        yield Call([3, 4], {}, 0, "eye(n, m)")
    elif name in ("full", "new_full"):
        yield Call([shape, 0.5], {}, 0, "full(shape, value)")
    elif name in ("randint",):
        yield Call([0, 6, shape], {}, 0, "randint(low, high, shape)")
    elif name in ("randperm",):
        yield Call([6], {}, 0, "randperm(n)")
    elif name in ("meshgrid",):
        yield Call([_f((3,), domain), _f((4,), domain)], {}, 0, "meshgrid(a, b)")
    elif name in ("bernoulli",):
        # ``bernoulli(input, p=None)``: the input is the probabilities, or
        # with a scalar ``p`` only the shape, dtype and device to draw at.
        # The probabilities are the op's contract, so an axis that rebuilds
        # them at another dtype keeps them in [0, 1] (``domain``) — moved to
        # ``|x| + 1`` every dtype was refused for an invalid probability.
        probs = _probe.as_f32(_probe.rng(9).uniform(0.2, 0.8, _probe.SHAPE))
        yield Call([probs], {}, 0, "bernoulli(probs)", domain=(0.0, 1.0))
        yield Call([_f(_probe.SHAPE, domain), 0.5], {}, 0, "bernoulli(like, p)")
    elif name in ("normal",):
        yield Call([0.0, 1.0, shape], {}, 0, "normal(mean, std, shape)")
        yield Call(
            [_f(_probe.SHAPE, domain), _f(_probe.SHAPE, "positive")],
            {},
            0,
            "normal(mu, sigma)",
        )
    elif name in ("cartesian_prod",):
        yield Call(
            [_f((3,), domain), _f((2,), domain)], {}, 0, "cartesian_prod(1-D, 1-D)"
        )
    elif name in ("vander",):
        yield Call([_f((4,), "positive")], {}, 0, "vander(1-D)")
    else:
        yield Call([shape], {}, 0, "factory(shape)")
        yield Call([2, 3], {}, 0, "factory(*shape)")


#: Ops that write a source at the positions an index names.
_OVERWRITES = frozenset(
    {"scatter", "scatter_", "index_copy", "index_copy_", "put", "put_"}
)


def _from_zero(positions: np.ndarray) -> np.ndarray:
    """The same number of distinct positions, the first of them 0."""
    return np.concatenate([[0], positions[positions != 0]])[: len(positions)]


def _overwrite(name: str, domain: str) -> Iterator[Call]:
    """Overwrites by index, each naming a position once.

    A position named twice keeps the last write on the CPU and whichever
    write lands last on Metal — the reference documents the order as
    undefined — so a repeated index made the device axis compare a race,
    not an answer.  It is not a chip difference: on one M1 Pro, fifteen
    runs of the signature-derived ``(2, 3, 6, 6)`` ``scatter_`` gave
    fifteen different results and ``put_`` ten.  The M4's FAILs on
    ``index_copy``, ``put_`` and ``scatter_`` were that race losing the
    other way.  The accumulating forms (``scatter_add``, ``index_add``)
    keep their repeats.

    Fewer positions than the receiver has, so the rest keep ``x``'s value
    and its gradient: a full permutation overwrote all of ``x`` and left
    the gradient and NaN axes nothing to see.  The index and the source
    follow a resized receiver (``same_shape``) — an empty one takes an
    empty index, a size-1 one position 0, which is why each index starts
    there.
    """
    rng = _probe.rng(3)
    x = _f((4, 5), domain)
    if name in ("scatter", "scatter_"):
        # Three of five columns per row.
        rows = [rng.permutation(5)[:3] for _ in range(4)]
        rows[0] = _from_zero(rows[0])
        yield Call(
            [x, 1, _probe.as_int(np.stack(rows)), _f((4, 3), domain)],
            {},
            0,
            "scatter(x, dim, index, src)",
            same_shape=(2, 3),
        )
    elif name in ("index_copy", "index_copy_"):
        index = _probe.as_int(_from_zero(rng.permutation(5)[:3]))
        yield Call(
            [_f((5,), domain), 0, index, _f((3,), domain)],
            {},
            0,
            "index_copy(x, dim, index, source)",
            same_shape=(2, 3),
        )
    else:
        yield Call(
            [x, _probe.as_int(_from_zero(rng.permutation(20)[:6])), _f((6,), domain)],
            {},
            0,
            "put(x, flat index, source)",
            same_shape=(1, 2),
        )


def _indexing(name: str, domain: str) -> Iterator[Call]:
    """Gather / scatter shapes, which need an index tensor of the right rank."""
    x = _f((4, 5), domain)
    idx_full = _probe.as_int(_probe.rng(3).integers(0, 5, (4, 5)))
    idx_row = _probe.as_int(_probe.rng(3).integers(0, 4, (3,)))
    src = _f((4, 5), domain)
    if name in ("gather", "take_along_dim", "take_along_axis"):
        yield Call([x, 1, idx_full], {}, 0, "gather(x, dim, index)")
        yield Call([x, idx_full], {"dim": 1}, 0, "gather(x, index, dim=)")
    elif name in _OVERWRITES:
        yield from _overwrite(name, domain)
    elif name in ("scatter_add", "scatter_reduce"):
        yield Call([x, 1, idx_full, src], {}, 0, "scatter(x, dim, index, src)")
    elif name in ("index_select",):
        yield Call([x, 0, idx_row], {}, 0, "index_select(x, dim, index)")
    elif name in ("take",):
        yield Call(
            [x, _probe.as_int(_probe.rng(3).integers(0, 20, (5,)))],
            {},
            0,
            "take(x, flat idx)",
        )
    elif name in ("masked_fill", "masked_fill_"):
        mask = lucid.tensor((_probe.rng(4).random((4, 5)) > 0.5))
        yield Call([x, mask, 0.0], {}, 0, "masked_fill(x, mask, value)")
    elif name in ("searchsorted", "bucketize"):
        sorted_1d = _probe.as_f64(np.linspace(0.0, 1.0, 6))
        values = _probe.as_f64(_probe.rng(5).uniform(0.0, 1.0, (4,)))
        yield Call([sorted_1d, values], {}, 0, "searchsorted(sorted, values)")
        yield Call([values, sorted_1d], {}, 0, "bucketize(values, boundaries)")
    elif name in ("narrow",):
        yield Call([x, 1, 1, 2], {}, 0, "narrow(x, dim, start, length)")
    elif name in ("where",):
        cond = lucid.tensor((_probe.rng(6).random((4, 5)) > 0.5))
        yield Call([cond, x, src], {}, 1, "where(cond, a, b)")
    elif name in ("index_put", "index_put_"):
        rows = _probe.as_int(_probe.rng(3).permutation(4)[:3])
        yield Call(
            [x, (rows,), _f((3, 5), domain)],
            {},
            0,
            "index_put(x, (rows,), values)",
        )


def _shape_with_args(name: str, domain: str) -> Iterator[Call]:
    """Reshapes and permutations whose target has to be spelled out."""
    x = _f((2, 3, 4), domain)
    flat = _f((2, 3), domain)
    if name in ("broadcast_to",):
        yield Call([_f((1, 3), domain), (2, 3)], {}, 0, "broadcast_to(x, shape)")
    elif name in ("expand",):
        yield Call([_f((1, 3), domain), (2, 3)], {}, 0, "expand(x, shape)")
    elif name in ("as_strided",):
        # Positions 0, 1, 3, 4 of the six elements.
        yield Call([flat, (2, 2), (3, 1)], {}, 0, "as_strided(x, size, stride)")
    elif name in ("permute",):
        yield Call([x, (2, 0, 1)], {}, 0, "permute(x, dims)")
    elif name in ("moveaxis", "movedim", "swapaxes", "swapdims", "transpose"):
        yield Call([x, 0, 2], {}, 0, "moveaxis(x, src, dst)")
    elif name in ("tile", "repeat"):
        yield Call([flat, (2, 2)], {}, 0, "tile(x, reps)")
    elif name in ("roll",):
        # The binding takes sequences, not bare ints.
        yield Call([flat, (1,), (0,)], {}, 0, "roll(x, shifts, dims)")
        yield Call([flat, (1, 1), (0, 1)], {}, 0, "roll(x, shifts, dims)")
        yield Call([flat, 1], {}, 0, "roll(x, shift)")
    elif name in ("unflatten",):
        yield Call([_f((2, 6), domain), 1, (2, 3)], {}, 0, "unflatten(x, dim, sizes)")
    elif name in ("unfold",):
        yield Call([x, 1, 2, 1], {}, 0, "unfold(x, dim, size, step)")
    elif name in ("dsplit", "hsplit", "vsplit", "split", "chunk", "tensor_split"):
        yield Call([_f((2, 3, 4), domain), 2], {}, 0, "split(x, n)")
    elif name in ("narrow",):
        yield Call([flat, 1, 0, 2], {}, 0, "narrow(x, dim, start, length)")
    elif name in ("view",):
        # A shape every layout of ``x`` can be viewed as, so the strided
        # half of the layout axis still asks something.  ``view`` never
        # copies: it refuses a shape the strides cannot read in place, and
        # ``(2, 3) -> (3, 2)`` regroups a strided slice's rows.  A size-1
        # axis moves no element, so inserting one is a view of any layout
        # (LCD-367).
        yield Call([x, (2, 1, 3, 4)], {}, 0, "view(x, shape with a size-1 axis)")
    elif name in ("flatten", "ravel"):
        yield Call([x], {}, 0, "flatten(x)")


def _matmul(name: str, domain: str) -> Iterator[Call]:
    """Products, each with its own rank convention."""
    vec = _f((4,), domain)
    mat = _f((3, 4), domain)
    other = _f((4, 3), domain)
    batch_a = _f((2, 3, 4), domain)
    batch_b = _f((2, 4, 3), domain)
    if name in ("bmm", "baddbmm"):
        yield Call([batch_a, batch_b], {}, 0, "bmm(batched, batched)")
    elif name in ("mm",):
        yield Call([mat, other], {}, 0, "mm(a, b)")
    elif name in ("dot", "vdot", "inner"):
        yield Call([vec, _f((4,), domain)], {}, 0, "dot(1-D, 1-D)")
    elif name in ("outer", "ger"):
        yield Call([vec, _f((3,), domain)], {}, 0, "outer(1-D, 1-D)")
    elif name in ("multi_dot",):
        yield Call([[mat, other, mat]], {}, 0, "multi_dot([a, b, c])")
    elif name in ("einsum",):
        yield Call(["ij,jk->ik", mat, other], {}, 1, "einsum(subscripts, a, b)")
    elif name in ("float_power",):
        yield Call([_f(_probe.SHAPE, "positive"), 2.0], {}, 0, "float_power(x, p)")
    else:
        yield Call([mat, other], {}, 0, "matmul(a, b)")


def _sequences(name: str, domain: str) -> Iterator[Call]:
    """Ops whose argument is a *list* of tensors, not a tensor.

    The signature says ``list[Tensor]`` and the autospec builds one
    tensor, so every one of these reported "no argument shape worked" or
    "no dtype accepted" — a refusal the probe had provoked.  ``primary``
    stays 0 even though args[0] is a list: the numeric axes read
    ``Call.base`` and get a TypeError, which they already treat as
    "nothing to differentiate here".
    """
    if name == "multi_dot":
        yield Call(
            [[_f((4, 3), domain), _f((3, 5), domain), _f((5, 2), domain)]],
            {},
            0,
            "multi_dot([A, B, C])",
        )
        return
    # pad_sequence / pack_sequence / pad_packed_sequence want *ragged*
    # sequences; equal lengths would make the padding a no-op.
    seqs = [_f((5, _CIN), domain), _f((3, _CIN), domain), _f((2, _CIN), domain)]
    if name == "pad_sequence":
        yield Call([seqs], {"batch_first": True}, 0, "pad_sequence(ragged)")
        yield Call([seqs], {}, 0, "pad_sequence(ragged) defaults")
    elif name == "pack_sequence":
        yield Call([seqs], {"enforce_sorted": True}, 0, "pack_sequence(sorted)")
    elif name == "pad_packed_sequence":
        packed = lucid.nn.utils.rnn.pack_sequence(seqs, enforce_sorted=True)
        yield Call([packed], {"batch_first": True}, 0, "pad_packed_sequence(packed)")


def _fused_linear(name: str, domain: str) -> Iterator[Call]:
    """``fused_linear_*`` takes the three operands a Linear layer holds."""
    x = _f((_N, _CIN), domain)
    w = _f((_COUT, _CIN), domain)
    b = _f((_COUT,), domain)
    yield Call([x, w, b], {}, 0, "fused_linear(x, weight, bias)")


def _linalg_extra(name: str, domain: str) -> Iterator[Call]:
    """Decompositions, each with the matrix property it requires."""
    n = 4
    gen = _probe.rng(_probe.SEED_X)
    square = gen.standard_normal((n, n)) * 0.35 + np.eye(n) * 2.5
    psd = square @ square.T + np.eye(n) * 0.75
    sym = (square + square.T) / 2.0
    rhs = gen.standard_normal((n, 2))
    upper = np.triu(square)
    if name in ("cholesky", "cholesky_ex"):
        yield Call([_probe.as_f64(psd)], {}, 0, "cholesky(PSD)")
    elif name in ("eigh", "eigvalsh"):
        yield Call([_probe.as_f64(sym)], {}, 0, "eigh(symmetric)")
    elif name in ("eig", "eigvals"):
        yield Call([_probe.as_f64(square)], {}, 0, "eig(square)")
    elif name in (
        "inv",
        "det",
        "slogdet",
        "matrix_exp",
        "lu",
        "lu_factor",
        "ldl_factor",
    ):
        yield Call([_probe.as_f64(square)], {}, 0, "linalg(well-conditioned square)")
    elif name in ("matrix_power",):
        yield Call([_probe.as_f64(square), 2], {}, 0, "matrix_power(A, 2)")
    elif name in ("solve", "lstsq"):
        yield Call([_probe.as_f64(square), _probe.as_f64(rhs)], {}, 0, "solve(A, B)")
    elif name in ("solve_triangular",):
        yield Call(
            [_probe.as_f64(upper), _probe.as_f64(rhs)],
            {"upper": True},
            0,
            "solve_triangular",
        )
    elif name in ("lu_solve", "ldl_solve"):
        yield Call(
            [_probe.as_f64(square), _probe.as_f64(rhs)], {}, 0, "lu_solve(LU, B)"
        )
    elif name in ("householder_product",):
        # ``H`` is a single 2-D reflector block and ``tau`` a *vector* of
        # min(m, n) coefficients.  Derived independently, the two came out
        # the same shape — a 1-D H with a 1-D tau, or a batched H with a
        # batched tau — and the op answered both instead of refusing:
        # rank-1 came back as an empty ``(0, 0)`` and a batch came back as
        # its first matrix alone.  Every axis was therefore recording a
        # verdict about a call no caller would make.
        m, n = 4, 3
        yield Call(
            [
                _probe.as_f64(gen.uniform(-1.0, 1.0, (m, n))),
                _probe.as_f64(gen.uniform(-0.5, 0.5, (min(m, n),))),
            ],
            {},
            0,
            "householder_product(H(4,3), tau(3,))",
        )
    elif name in ("tensorinv",):
        yield Call(
            [_probe.as_f64(square.reshape(2, 2, 2, 2))], {}, 0, "tensorinv((2,2,2,2))"
        )
    elif name in ("tensorsolve",):
        yield Call(
            [
                _probe.as_f64(square.reshape(2, 2, 2, 2)),
                _probe.as_f64(rhs[:2, :2]),
            ],
            {},
            0,
            "tensorsolve(A(2,2,2,2), B(2,2))",
        )
    elif name in ("vander",):
        yield Call([_probe.as_f64(gen.uniform(0.5, 1.5, (4,)))], {}, 0, "vander(1-D)")


def _fft_full(name: str, domain: str) -> Iterator[Call]:
    """Every transform in the family, at the precision the engine accepts.

    The transforms reject float64 outright (``fftn requires F16/F32/C64``),
    which is why a float64 probe reported all twenty of them as
    uncallable.  The inverse and Hermitian forms want a *spectrum*, so
    they are fed one produced by their own forward partner rather than
    raw samples.
    """
    real_1d = _probe.as_f32(_probe.sample(domain, (8,)))
    real_2d = _probe.as_f32(_probe.sample(domain, (4, 8)))
    if name in ("fftfreq", "rfftfreq"):
        yield Call([8], {}, 0, "fftfreq(n)")
        yield Call([8, 0.5], {}, 0, "fftfreq(n, d)")
        return
    if name in ("fftshift", "ifftshift"):
        yield Call([real_1d], {}, 0, "fftshift(x)")
        return

    rank = 2 if name.endswith("2") else (3 if name.endswith("n") else 1)
    source = real_2d if rank >= 2 else real_1d
    spectrum = None
    forward_name = {1: "fft", 2: "fft2", 3: "fftn"}[rank]
    forward = getattr(lucid.fft, forward_name, None)
    if callable(forward):
        try:
            spectrum = forward(source)
        except Exception:  # noqa: BLE001
            spectrum = None

    # hfft / ihfft and the inverse transforms consume a complex spectrum.
    if spectrum is not None and (name.startswith("i") or "hfft" in name):
        yield Call([spectrum], {}, 0, f"{name}(spectrum)")
    yield Call([source], {}, 0, f"{name}(real)")
    if spectrum is not None:
        yield Call([spectrum], {}, 0, f"{name}(spectrum)")
    if rank == 1:
        yield Call([source], {"n": 8}, 0, f"{name}(x, n=8)")


def _complex(name: str, domain: str) -> Iterator[Call]:
    """Ops that only mean anything on a complex tensor."""
    real = _probe.as_f64(_probe.sample(domain, (2, 4)))
    pair = _probe.as_f64(_probe.sample(domain, (2, 4)))
    built = None
    maker = getattr(lucid, "complex", None)
    if callable(maker):
        try:
            built = maker(real, pair)
        except Exception:  # noqa: BLE001
            built = None
    if name == "view_as_complex":
        yield Call(
            [_probe.as_f64(_probe.sample(domain, (2, 4, 2)))],
            {},
            0,
            "view_as_complex(...,2)",
        )
        return
    if built is not None:
        yield Call([built], {}, 0, f"{name}(complex)")
    yield Call([real], {}, 0, f"{name}(real)")


def _dtype_util(name: str, domain: str) -> Iterator[Call]:
    """Type-system helpers: they take dtypes, not tensors."""
    if name in ("can_cast", "promote_types"):
        yield Call([lucid.float32, lucid.float64], {}, 0, f"{name}(dtype, dtype)")
    elif name in ("result_type",):
        yield Call(
            [_f(_probe.SHAPE, domain), _f(_probe.SHAPE, domain)],
            {},
            0,
            "result_type(a, b)",
        )
    elif name in ("type", "astype"):
        # ``Tensor.type`` takes the *name*, not the dtype object.
        yield Call([_f(_probe.SHAPE, domain), "float32"], {}, 0, "type(x, 'float32')")
        yield Call([_f(_probe.SHAPE, domain), lucid.float32], {}, 0, "astype(x, dtype)")


def _nn_leftover(name: str, domain: str) -> Iterator[Call]:
    """The functional entries with signatures no other family covers."""
    if name in ("lp_pool1d", "lp_pool2d", "lp_pool3d"):
        rank = _rank_of(name)
        yield Call(
            [_f((_N, _CIN, *_spatial(rank)), "positive"), 2.0, 2],
            {},
            0,
            "lp_pool(x, p, k)",
        )
    elif name in ("max_unpool1d", "max_unpool2d", "max_unpool3d"):
        rank = _rank_of(name)
        spatial = _spatial(rank)
        pooled = tuple(v // 2 for v in spatial)
        x = _f((_N, _CIN, *pooled), domain)
        idx = _probe.as_int(
            _probe.rng(7).integers(0, int(np.prod(spatial)), (_N, _CIN, *pooled))
        )
        yield Call(
            [x, idx, 2], {"output_size": spatial}, 0, "max_unpool(x, indices, k)"
        )
        yield Call([x, idx, 2], {}, 0, "max_unpool(x, indices, k)")
    elif name in ("gaussian_nll_loss",):
        yield Call(
            [_f((_N, 4), domain), _f((_N, 4), domain), _f((_N, 4), "positive")],
            {},
            0,
            "gaussian_nll(input, target, var)",
            same_shape=(1, 2),
        )
    elif name in ("multilabel_margin_loss",):
        yield Call(
            [
                _f((_N, 4), domain),
                _probe.as_int(np.array([[0, -1, -1, -1], [1, -1, -1, -1]])),
            ],
            {},
            0,
            "multilabel_margin(input, target)",
            same_shape=(1,),
        )
    elif name in ("ctc_loss",):
        t, n, c = 6, 2, 4
        log_probs = _f((t, n, c), domain)
        targets = _probe.as_int(_probe.rng(8).integers(1, c, (n, 3)))
        yield Call(
            [
                log_probs,
                targets,
                _probe.as_int(np.array([t, t])),
                _probe.as_int(np.array([3, 3])),
            ],
            {},
            0,
            "ctc_loss(log_probs, targets, input_lengths, target_lengths)",
        )
    elif name in ("threshold", "threshold_"):
        yield Call(
            [_f(_probe.SHAPE, domain), 0.0, 0.0],
            {},
            0,
            "threshold(x, threshold, value)",
        )
    elif name in ("fused_linear_relu", "fused_linear_gelu"):
        w, b = _f((_COUT, _CIN), domain), _f((_COUT,), domain)
        yield Call(
            [_f((_N, _CIN), domain), w, b], {}, 0, "fused_linear(x, weight, bias)"
        )
        yield Call(
            [_f((_N, _CIN), domain)],
            {"weight": w, "bias": b},
            0,
            "fused_linear(x, weight=, bias=)",
        )
    elif name in ("sinusoidal_embedding",):
        yield Call(
            [_probe.as_f64(np.arange(4.0)), 8], {}, 0, "sinusoidal_embedding(pos, dim)"
        )
    elif name in ("sinusoidal_embedding_2d",):
        yield Call([8, 4, 4], {}, 0, "sinusoidal_embedding_2d(dim, h, w)")
    elif name in ("multi_head_attention_forward",):
        seq, batch, embed = 4, 2, 8
        q = _f((seq, batch, embed), domain)
        yield Call(
            [
                q,
                q,
                q,
                embed,
                2,
                _f((3 * embed, embed), domain),
                _f((3 * embed,), domain),
            ],
            {},
            0,
            "mha_forward(q, k, v, embed, heads, in_proj_w, in_proj_b)",
        )
    elif name in ("polygamma",):
        yield Call([1, _f(_probe.SHAPE, "positive")], {}, 1, "polygamma(n, x)")
    elif name in ("one_hot",):
        yield Call([_int((_N,), 4), 4], {}, 0, "one_hot(idx, n)")


def _accessor(name: str, domain: str) -> Iterator[Call]:
    """Shape, dtype and residency predicates — they answer with a scalar."""
    if name in ("grad", "grad_fn"):
        # Both are ``None`` on a fresh tensor, so the probe has to make a
        # gradient exist before asking for one.
        leaf = _f((2, 3), domain)
        leaf.requires_grad_(True)
        (leaf * leaf).sum().backward()
        yield Call([leaf], {}, 0, f"{name} after a backward")
    x = _f((2, 3), domain)
    yield Call([x], {}, 0, "accessor(x)")
    yield Call([x, 0], {}, 0, "accessor(x, dim)")
    yield Call([x, x], {}, 0, "accessor(x, other)")


# ── resolution ───────────────────────────────────────────────────────────────
# Ordered: the first predicate that matches wins.  Patterns are matched
# against the short name, so ``F.conv2d`` and ``lucid.conv2d`` share a
# builder.

_FAMILIES: list[tuple[str, Callable[[str, str], Iterator[Call]]]] = [
    (r"^conv(_transpose)?[123]d$", _conv),
    (r"^max_unpool[123]d$|^lp_pool[123]d$", _nn_leftover),
    (
        r"^(ctc_loss|gaussian_nll_loss|multilabel_margin_loss|threshold_?|"
        r"fused_linear_(relu|gelu)|sinusoidal_embedding(_2d)?|"
        r"multi_head_attention_forward|polygamma|one_hot)$",
        _nn_leftover,
    ),
    (r"^(addbmm|baddbmm|addmm|addmv|addr|addcmul|addcdiv|lerp)$", _ternary),
    (r"^bitwise_|^invert$", _bitwise),
    (
        r"^(arange|linspace|logspace|eye|full|new_full|empty|zeros|ones|rand|randn|"
        r"randint|randperm|normal|bernoulli|meshgrid|cartesian_prod|vander|"
        r"empty_like|full_like)$",
        _factory,
    ),
    (
        r"^(gather|scatter_?|scatter_add|scatter_reduce|index_select|index_put_?|"
        r"index_copy_?|put_?|take|take_along_dim|take_along_axis|masked_fill_?|"
        r"searchsorted|bucketize|narrow|where)$",
        _indexing,
    ),
    (
        r"^(broadcast_to|expand|as_strided|permute|moveaxis|movedim|swapaxes|swapdims|tile|"
        r"repeat|roll|unflatten|unfold|dsplit|hsplit|vsplit|tensor_split|view)$",
        _shape_with_args,
    ),
    (
        r"^(bmm|mm|dot|vdot|inner|outer|ger|multi_dot|einsum|float_power|matmul)$",
        _matmul,
    ),
    (
        r"^(cholesky(_ex)?|eig|eigh|eigvals|eigvalsh|inv|det|slogdet|matrix_exp|"
        r"matrix_power|lu|lu_factor|lu_solve|ldl_factor|ldl_solve|solve|"
        r"solve_triangular|lstsq|householder_product|tensorinv|tensorsolve|vander)$",
        _linalg_extra,
    ),
    (
        r"^(multi_dot|pad_sequence|pack_sequence|pad_packed_sequence)$",
        _sequences,
    ),
    (
        r"^fused_linear_(relu|gelu)$",
        _fused_linear,
    ),
    (
        r"^i?(rfft|hfft|fft)[2n]?$|^ifft[2n]?$|^irfft[2n]?$|^ihfft[2n]?$|"
        r"^r?fftfreq$|^i?fftshift$",
        _fft_full,
    ),
    (r"^(angle|imag|real|view_as_complex|view_as_real|conj|resolve_conj)$", _complex),
    (r"^(can_cast|promote_types|result_type|type|astype)$", _dtype_util),
    (
        r"^(dim|ndim|shape|size|numel|nbytes|element_size|stride|is_[a-z_]+|"
        r"grad|grad_fn|impl|untyped_storage)$",
        _accessor,
    ),
    (r"^(fractional_max_pool)[23]d$", _fractional_pool),
    (r"^adaptive_(avg|max)_pool[123]d$", _adaptive_pool),
    (r"^(avg|max|lp)_pool[123]d$", _pool),
    (
        r"norm$|^batch_norm|^layer_norm|^group_norm|^instance_norm|^local_response",
        _norm,
    ),
    # ``binary_cross_entropy`` ends in neither: it fell through to the
    # signature-derived call, whose target of another shape the loss
    # refuses, and every axis filed it unanswered.
    (r"loss$|^cross_entropy$|^kl_div$|^bce|^binary_cross_entropy", _loss),
    (r"^embedding", _embedding),
    (r"attention|^sdpa$", _attention),
    (
        r"^(interpolate|upsample.*|grid_sample|affine_grid|pixel_(un)?shuffle|"
        r"channel_shuffle|unfold|fold|pad)$",
        _resample,
    ),
    (r"^(rearrange|reduce|repeat|pack|unpack|einsum)$", _einops),
    (
        r"^(sum|mean|prod|max|min|var|std|median|nanmean|nansum|nanmedian|"
        r"amax|amin|logsumexp|cumsum|cumprod|cummax|cummin|norm|argmax|argmin|"
        r"any|all|count_nonzero)$",
        _reduction,
    ),
    (
        r"^(add|sub|mul|div|pow|maximum|minimum|fmod|remainder|floor_divide|"
        r"true_divide|arctan2|atan2|hypot|logaddexp|logaddexp2|copysign|nextafter|"
        r"equal|eq|ne|not_equal|greater|gt|ge|greater_equal|less|lt|le|less_equal|"
        r"logical_and|logical_or|logical_xor|bitwise_.*|matmul|dot|inner|outer|kron)$",
        _binary,
    ),
]

#: Exact overrides, for the handful whose family cannot be inferred.
_EXACT: dict[str, Callable[[str, str], Iterator[Call]]] = {
    # A contiguous input is returned itself, with no node, as the reference
    # does — the identity, which the gradient axis declines to measure.  The
    # copy a strided input takes is the path with a backward of its own.
    "contiguous": lambda n, d: iter(
        [Call([_f(_probe.SHAPE, d)], {}, 0, "contiguous(strided leaf)", strided=True)]
    ),
    "one_hot": lambda n, d: iter([Call([_int((_N,), 4), 4], {}, 0, "one_hot(idx, n)")]),
    # Class scores against class indices.  The blind ladder's ``op(x, dim)``
    # used to be the first call that ran; once ``==`` promoted its operands
    # the signature-derived call at rank 1 ran first instead, and the rank
    # axis had nothing to take away from it.
    **{
        metric: lambda n, d: iter(
            [
                Call(
                    [_f((_N, 4), d), _int((_N,), 4)],
                    {},
                    0,
                    "metric(logits, class index)",
                )
            ]
        )
        for metric in ("accuracy", "correct_count")
    },
    "linear": lambda n, d: iter(
        [
            Call(
                [_f((_N, _CIN), d), _f((_COUT, _CIN), d), _f((_COUT,), d)],
                {},
                0,
                "linear",
            )
        ]
    ),
    "bilinear": lambda n, d: iter(
        [
            Call(
                [_f((_N, _CIN), d), _f((_N, _CIN), d), _f((_COUT, _CIN, _CIN), d)],
                {},
                0,
                "bilinear",
            )
        ]
    ),
    "softmax": lambda n, d: iter(
        [Call([_f(_probe.SHAPE, d)], {"dim": -1}, 0, "softmax dim=-1")]
    ),
    "log_softmax": lambda n, d: iter(
        [Call([_f(_probe.SHAPE, d)], {"dim": -1}, 0, "log_softmax dim=-1")]
    ),
    "softmin": lambda n, d: iter(
        [Call([_f(_probe.SHAPE, d)], {"dim": -1}, 0, "softmin dim=-1")]
    ),
    "glu": lambda n, d: iter([Call([_f((_N, 4), d)], {"dim": -1}, 0, "glu")]),
    "normalize": lambda n, d: iter(
        [Call([_f(_probe.SHAPE, d)], {"dim": -1}, 0, "normalize")]
    ),
    "repeat_kv": lambda n, d: iter(
        [Call([_f((_N, 2, 4, 4), d), 2], {}, 0, "repeat_kv(x, n_rep)")]
    ),
}


def _autograd_grad(domain: str) -> Iterator[Call]:
    """``autograd.grad(outputs, inputs)`` over a graph the probe builds.

    The short name ``grad`` is the tensor accessor's, which hands it one
    tensor, and the signature derivation hands it two unrelated leaves.
    Neither is a graph, and since ``grad`` refuses a non-scalar output
    without its seed, as the reference does, every candidate was refused
    (LCD-367).  The graph is retained because an axis calls the same
    invocation more than once; the primary is ``inputs``, the tensor the
    gradient is taken with respect to.
    """
    x = _f((2, 3), domain)
    x.requires_grad_(True)
    y = (x * x).sum()
    yield Call(
        [y, x], {"retain_graph": True}, 1, "grad(outputs, inputs) of (x * x).sum()"
    )


def _input_tangent(op: str) -> Callable[[str], Iterator[Call]]:
    """``autograd.<op>(func, inputs, v)`` with ``v`` the shape of ``inputs``.

    ``jvp``, ``hvp`` and ``vhp`` take a vector in the input space, unlike
    ``vjp``'s, which is shaped like the output.  The signature tier fills
    only parameters without a default, and with ``v`` optional (``None``,
    as in the reference) it handed ``jvp`` no tangent for a ``(2, 3)``
    input — which ``jvp`` refuses, as the reference does, so the smoke
    cell went unanswered.  ``v`` follows a resized ``inputs`` through
    ``same_shape``; the scalar ``func`` is what ``hvp`` and ``vhp`` need.
    """

    def build(domain: str) -> Iterator[Call]:
        yield Call(
            [lambda t: (t * t).sum(), _f((2, 3), domain), _f((2, 3), domain)],
            {},
            1,
            f"{op}(func, inputs, v) of (x * x).sum()",
            same_shape=(2,),
        )

    return build


def _retain_grad(domain: str) -> Iterator[Call]:
    """``retain_grad()`` on an interior node of a graph.

    A tensor that does not require grad is refused — no gradient will
    ever reach it — so a fresh probe is the one input it cannot take.
    """
    leaf = _f((2, 3), domain)
    leaf.requires_grad_(True)
    yield Call([leaf * 2.0], {}, 0, "retain_grad() on a non-leaf")


#: Keyed by the *qualified* name, because a short name is not unique:
#: ``einops.repeat`` takes a pattern string while ``lucid.repeat`` takes
#: a repeat count, and ``Tensor.unfold`` slides a window while
#: ``F.unfold`` extracts image patches.  Matching on the short name alone
#: sent four of these to the wrong builder.
_QUALIFIED: dict[str, Callable[[str], Iterator[Call]]] = {
    "lucid.autograd.grad": _autograd_grad,
    # A name not on the surface yet costs nothing: lookup is by qualname.
    **{f"lucid.autograd.{op}": _input_tangent(op) for op in ("jvp", "hvp", "vhp")},
    "Tensor.retain_grad": _retain_grad,
    "lucid.einops.repeat": lambda d: iter(
        [
            Call(
                [_f((2, 3), d), "a b -> a b r"],
                {"r": 2},
                0,
                "einops.repeat(x, pattern, **axes)",
            )
        ]
    ),
    "lucid.einops.rearrange": lambda d: iter(
        [Call([_f((2, 3, 4), d), "a b c -> a c b"], {}, 0, "einops.rearrange")]
    ),
    "lucid.einops.reduce": lambda d: iter(
        [Call([_f((2, 3, 4), d), "a b c -> a b", "mean"], {}, 0, "einops.reduce")]
    ),
    "Tensor.where": lambda d: iter(
        [
            Call(
                [
                    _f((3, 4), d),
                    lucid.tensor(_probe.rng(6).random((3, 4)) > 0.5),
                    _f((3, 4), d),
                ],
                {},
                0,
                "x.where(cond, other)",
            )
        ]
    ),
    "Tensor.unfold": lambda d: iter(
        [Call([_f((2, 3, 8), d), 2, 3, 1], {}, 0, "x.unfold(dim, size, step)")]
    ),
    "F.unfold": lambda d: iter(
        [Call([_f((2, 3, 6, 6), d), 3], {"padding": 1}, 0, "F.unfold(x, kernel_size)")]
    ),
    "Tensor.new_full": lambda d: iter(
        [Call([_f((2, 3), d), (2, 2), 0.5], {}, 0, "x.new_full(shape, value)")]
    ),
    "lucid.linalg.vander": lambda d: iter(
        [
            Call(
                [_probe.as_f64(_probe.rng(3).uniform(0.5, 1.5, (4,)))],
                {},
                0,
                "vander(1-D)",
            )
        ]
    ),
    "lucid.normal": lambda d: iter(
        [Call([0.0, 1.0], {"size": (2, 3)}, 0, "normal(mean, std, size=)")]
    ),
    "lucid.initial_seed": lambda d: iter([Call([], {}, 0, "initial_seed()")]),
    "lucid.is_grad_enabled": lambda d: iter([Call([], {}, 0, "is_grad_enabled()")]),
    "lucid.is_nonzero": lambda d: iter(
        [Call([_probe.as_f64(np.array([1.0]))], {}, 0, "is_nonzero(single element)")]
    ),
    "F.multilabel_margin_loss": lambda d: iter(
        [
            Call(
                [
                    _f((2, 4), d),
                    # int32, not the int64 a plain ``lucid.tensor`` of ints
                    # produces — the loss rejects int64, reported separately.
                    lucid.tensor(
                        np.array([[3, 0, -1, -1], [2, -1, -1, -1]]), dtype=lucid.int32
                    ),
                ],
                {},
                0,
                "multilabel_margin_loss(input, target)",
                same_shape=(1,),
            )
        ]
    ),
    "F.sinusoidal_embedding": lambda d: iter(
        [
            Call(
                [_probe.as_f64(np.arange(4.0)), 8],
                {},
                0,
                "sinusoidal_embedding(t, dim)",
            ),
            Call([4, 8], {}, 0, "sinusoidal_embedding(n, dim)"),
        ]
    ),
}


# ── joint resizing ───────────────────────────────────────────────────────────
# ``Call.same_shape`` covers a peer that keeps the primary's shape.  These
# ops relate their operands by more than that, and an axis that resizes
# the primary alone hands them a problem whose parts disagree.  Since the
# ops refuse such a problem, as the reference does, the edge axis read
# the refusal as "rejects empty input" (LCD-275 for ``ctc_loss``,
# LCD-367 for these).  Each rule rebuilds the whole problem around the
# requested primary, and answers an empty request with an empty *batch*
# of the original problem when emptying the requested axis leaves no
# coherent problem at all.


def _all_tensors(call: Call) -> bool:
    return all(hasattr(a, "shape") and hasattr(a, "dtype") for a in call.args)


def _linear_system_resized(call: Call, array: np.ndarray) -> Call:
    """``solve(A, B)``, ``solve_triangular`` and the factor solves.

    The right-hand side shares the matrix's batch and row count, and a
    factor solve's pivots (``lu_solve(LU, pivots, B)``, ``ldl_solve``)
    are a permutation of its order — the identity at any size.  Emptying
    the first axis of an unbatched ``(n, n)`` matrix leaves ``(0, n)``,
    which no solver can be handed, so that request becomes an empty
    batch ``(0, n, n)``.
    """
    if call.primary != 0 or len(call.args) not in (2, 3) or not _all_tensors(call):
        return call.with_primary(array)
    matrix, rhs = call.args[0], call.args[-1]
    if array.ndim < 2 or array.shape[-1] != array.shape[-2]:
        if array.size:
            return call.with_primary(array)
        array = np.zeros((0, *tuple(matrix.shape)[-2:]))
    rebuilt = call.with_primary(array)
    first = _probe.to_numpy(rhs)
    fill = first.flat[0] if first is not None and first.size else 0.0
    columns = tuple(rhs.shape)[len(matrix.shape) - 1 :]
    rebuilt.args[-1] = _rebuild(rhs, np.full((*array.shape[:-1], *columns), fill))
    if len(call.args) == 3:
        order = array.shape[-1]
        identity = np.broadcast_to(np.arange(1, order + 1), array.shape[:-1])
        rebuilt.args[1] = _rebuild(call.args[1], identity, keep_integer=True)
    return rebuilt


def _padded_batch_resized(call: Call, array: np.ndarray) -> Call:
    """``pack_padded_sequence(input, lengths)``: one length per sequence.

    Every length is at least 1 and at most the padded length, so a
    resized input takes ``lengths`` of its own batch, each the full
    padded length.  An empty time axis admits no length at all; the
    empty problem is an empty batch, which takes no lengths.
    """
    if call.primary != 0 or len(call.args) < 2 or array.ndim < 2:
        return call.with_primary(array)
    batch_first = bool(
        call.args[2] if len(call.args) > 2 else call.kwargs.get("batch_first", False)
    )
    time, batch = (1, 0) if batch_first else (0, 1)
    if array.shape[time] == 0:
        shape = list(array.shape)
        shape[time] = int(call.args[0].shape[time])
        shape[batch] = 0
        array = np.zeros(shape)
    lengths = [int(array.shape[time])] * int(array.shape[batch])
    rebuilt = call.with_primary(array)
    peer = call.args[1]
    rebuilt.args[1] = (
        _probe.as_int(np.array(lengths, dtype=np.int64))
        if hasattr(peer, "shape")
        else lengths
    )
    return rebuilt


def _ctc_resized(call: Call, array: np.ndarray) -> Call:
    """``ctc_loss(log_probs, targets, input_lengths, target_lengths)``.

    ``log_probs`` is ``(T, N, C)``: the lengths are one per sequence, every
    input length at most ``T``, and a target is ``L`` labels from
    ``1 .. C-1`` (0 is the blank) with ``L <= T``.  The targets are the
    labels ``1 .. L`` — no repeat, so ``T >= L`` suffices — with ``L`` the
    original target length capped by ``C - 1`` and ``T``.  An empty ``T``
    is still a problem (every length 0) and is asked as such.
    """
    if call.primary != 0 or len(call.args) != 4 or array.ndim != 3:
        return call.with_primary(array)
    if not _all_tensors(call):
        return call.with_primary(array)
    steps, batch, classes = (int(n) for n in array.shape)
    width = int(call.args[1].shape[-1]) if len(call.args[1].shape) == 2 else 0
    length = max(0, min(width, classes - 1, steps))
    rebuilt = call.with_primary(array)
    labels = np.broadcast_to(np.arange(1, length + 1), (batch, length))
    rebuilt.args[1] = _rebuild(call.args[1], labels, keep_integer=True)
    rebuilt.args[2] = _rebuild(call.args[2], np.full(batch, steps), keep_integer=True)
    rebuilt.args[3] = _rebuild(call.args[3], np.full(batch, length), keep_integer=True)
    return rebuilt


def _bounds_resized(call: Call, array: np.ndarray) -> Call:
    """``x.op_(peer, ...)``: a tensor peer of the receiver's shape follows it.

    An in-place op cannot grow its receiver, so a resized ``x`` against
    peers of the old shape is refused for a result that no longer fits:
    the probe's refusal, not the op's (LCD-532, LCD-536).  The peers the
    call itself declares (``same_shape``) follow it too.
    """
    if not 0 <= call.primary < len(call.args):
        return call.with_primary(array)
    shape = tuple(call.args[call.primary].shape)
    bounds = tuple(
        index
        for index, arg in enumerate(call.args)
        if index != call.primary
        and (index in call.same_shape or tuple(getattr(arg, "shape", (None,))) == shape)
    )
    # Everything the call declares survives, as in every other rebuild
    # (``with_primary``), except ``joint``: this function *is* the call's
    # joint rule, and carried, ``resized`` would hand the call back to it.
    peers = Call(
        call.args,
        call.kwargs,
        call.primary,
        call.note,
        bounds,
        call.domain,
        None,
        call.strided,
    )
    return peers.resized(array)


#: Keyed by qualified name; consulted by :func:`invocations`, which sets
#: the rule on every candidate it yields for that symbol.
_JOINT_RESIZE: dict[str, Callable[[Call, np.ndarray], Call]] = {
    "lucid.linalg.solve": _linear_system_resized,
    "lucid.linalg.solve_ex": _linear_system_resized,
    "lucid.linalg.solve_triangular": _linear_system_resized,
    "lucid.linalg.lu_solve": _linear_system_resized,
    "lucid.linalg.ldl_solve": _linear_system_resized,
    "lucid.nn.utils.pack_padded_sequence": _padded_batch_resized,
    "F.ctc_loss": _ctc_resized,
}


def invocations(
    name: str,
    domain: str,
    qualname: str | None = None,
    fn: Any = None,
) -> Iterator[Call]:
    """Every candidate call for ``name``, best guess first.

    Four tiers, narrowing from "someone wrote this down for this exact
    symbol" to "guess":

    ``_QUALIFIED`` / ``_EXACT``
        Hand-written for one symbol, where the name is ambiguous across
        subsystems or the op needs something no rule would infer.
    ``_FAMILIES``
        Hand-written for a group — every convolution, every pooling layer.
    :mod:`~lucid.test.audit._autospec`
        Derived from the signature.  Covers the long tail that used to
        fall straight to the ladder and SKIP there.
    the ladder
        ``f(x)``, ``f(x, y)``, ``f(x, dim)`` — the original sweep's whole
        vocabulary, kept as the floor rather than the ceiling.

    Parameters
    ----------
    name : str
        Short symbol name — ``"conv2d"``, not ``"F.conv2d"``.
    domain : str
        Key into :data:`~lucid.test.audit._probe.DOMAINS`.
    qualname : str, optional
        Full spelling, used to disambiguate names that exist in more than
        one subsystem.
    fn : callable, optional
        The resolved callable.  Without it the signature tier is skipped
        — there is nothing to introspect — and the symbol falls to the
        ladder as it did before.

    Yields
    ------
    Call
        Tried in order until one runs.
    """
    # Fresh draw counters per invocation, so a run stays reproducible and
    # a finding can still be re-derived by hand.
    _DRAWN.clear()
    joint = _JOINT_RESIZE.get(qualname) if qualname is not None else None
    if (
        joint is None
        and qualname is not None
        and qualname.startswith("Tensor.")
        and name.endswith("_")
    ):
        joint = _bounds_resized
    drawing = _draws_only_when_training(name, fn)
    for call in _candidates(name, domain, qualname, fn):
        if joint is not None and call.joint is None:
            call.joint = joint
        twin = None
        if drawing and "training" not in call.kwargs:
            # Copied before ``call`` is handed out: an axis that runs an
            # in-place sampler (``rrelu_``) writes into ``call``'s operands.
            twin = call.fresh()
            twin.kwargs["training"] = True
            twin.note = f"{call.note}, training=True"
        yield call
        if twin is not None:
            yield twin


def _draws_only_when_training(name: str, fn: Any) -> bool:
    """Whether ``fn`` is a sampler whose default, ``training=False``, does not draw.

    ``rrelu``, ``alpha_dropout`` and ``feature_alpha_dropout`` take
    ``training=False`` by default, as the reference does, and then answer
    deterministically (``rrelu`` with the *expectation* of its slope).
    Called with the default only, the determinism axis saw the same numbers
    under two seeds and reported, correctly, that it could not tell whether
    the op respects a seed — of an op that was not being asked to draw.
    So every candidate is followed by its ``training=True`` twin: the axes
    that measure the op keep the deterministic default, and the one that
    measures the draw moves on to the twin.  Read from the signature:
    ``dropout`` defaults to ``training=True`` and needs nothing, and a mode
    switch on an op that never draws (``batch_norm``) is not a sampler's.
    """
    if fn is None or not re.search(_surface.STOCHASTIC_PATTERN, name):
        return False
    try:
        signature = inspect.signature(
            fn, annotation_format=annotationlib.Format.FORWARDREF
        )
    except TypeError, ValueError:
        return False
    training = signature.parameters.get("training")
    return training is not None and training.default is False


def _candidates(
    name: str, domain: str, qualname: str | None, fn: Any
) -> Iterator[Call]:
    """The tiers of :func:`invocations`, in order."""
    if qualname is not None and qualname in _QUALIFIED:
        yield from _QUALIFIED[qualname](domain)
    if name in _EXACT:
        yield from _EXACT[name](name, domain)
    for pattern, build in _FAMILIES:
        if re.search(pattern, name):
            yield from build(name, domain)
            break
    if fn is not None:
        yield from _autospec.invocations(fn, name, domain)
    yield from _unary(name, domain)
    yield from _binary(name, domain)
    x = _f(_probe.SHAPE, domain)
    yield Call([x, -1], {}, 0, "op(x, dim)")
    yield Call([x, 2], {}, 0, "op(x, int)")
    yield Call([[x, x]], {}, 0, "op([x, x])")


def has_spec(name: str, fn: Any = None) -> bool:
    """Whether ``name`` gets a real invocation rather than the ladder.

    ``fn`` opts in the signature tier: a symbol nobody wrote a spec for is
    still specified if its own signature says enough.
    """
    if name in _EXACT:
        return True
    if any(re.search(p, name) for p, _ in _FAMILIES):
        return True
    if fn is None:
        return False
    return next(_autospec.invocations(fn, name, "moderate"), None) is not None


def spec_families() -> list[str]:
    """The family patterns, for ``--list-specs``."""
    return [p for p, _ in _FAMILIES]


__all__ = ["Call", "has_spec", "invocations", "spec_families"]

# Imported last, not at the top: ``_autospec`` needs :class:`Call` from
# here, so the two import each other.  By this line ``Call`` is bound, and
# the partially-initialised module it sees is complete enough.  Same
# arrangement as ``_axes`` and its sub-axis modules.
from lucid.test.audit import _autospec  # noqa: E402
