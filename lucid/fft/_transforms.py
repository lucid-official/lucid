"""The four base transforms behind :mod:`lucid.fft` and their autograd.

Everything the public transforms share: ``norm`` / ``dim`` / ``s``
canonicalisation, the raw engine entry points, and the
``lucid.autograd.Function`` classes for the base transforms — one for the
complex pair ``fftn`` / ``ifftn``, one each for ``rfftn`` and ``irfftn`` —
whose backward is the dual transform.
"""

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import final, override

import lucid
from lucid._C import engine as _C_engine
from lucid._dispatch import _unwrap, _wrap
from lucid._ops._autocast import _autocast_entry
from lucid._ops._dims import _wrap_dim
from lucid._tensor.tensor import Tensor
from lucid.autograd.function import Function as _AutogradFunction
from lucid.autograd.function import FunctionCtx

_fft = _C_engine.fft

# ── Normalisation ────────────────────────────────────────────────────────────

_VALID_NORMS = ("backward", "forward", "ortho")


def _check_norm(norm: str | None) -> str:
    if norm is None:
        return "backward"
    if norm not in _VALID_NORMS:
        raise ValueError(f"norm must be one of {_VALID_NORMS} or None, got {norm!r}")
    return norm


def _dual_norm(norm: str) -> str:
    if norm == "ortho":
        return "ortho"
    return "forward" if norm == "backward" else "backward"


def _scale_after_fft(N: int, norm: str) -> float:
    """Post-multiplier applied to MLX's fft (which is 'backward', i.e. unscaled)."""
    if norm == "backward":
        return 1.0
    if norm == "forward":
        return 1.0 / N
    return 1.0 / math.sqrt(N)  # ortho


def _scale_after_ifft(N: int, norm: str) -> float:
    """Post-multiplier applied to MLX's ifft (which already includes 1/N)."""
    if norm == "backward":
        return 1.0  # MLX already divides by N
    if norm == "forward":
        return float(N)  # cancel MLX's 1/N
    return math.sqrt(float(N))  # ortho: turn 1/N into 1/sqrt(N)


# ── Axis / shape canonicalisation ────────────────────────────────────────────


def _as_axis_list(dim: int | Sequence[int] | None, rank: int, default_all: bool) -> list[int]:
    if dim is None:
        if default_all:
            return list(range(rank))
        return [-1]
    if isinstance(dim, int):
        return [dim]
    return [int(d) for d in dim]


def _as_size_list(s: int | Sequence[int] | None) -> list[int]:
    if s is None:
        return []
    if isinstance(s, int):
        return [int(s)]
    return [int(v) for v in s]


def _normalise_axes(axes: list[int], rank: int) -> list[int]:
    if rank == 0 and axes:
        raise IndexError(f"FFT axis {axes[0]} out of range for rank-0 tensor")
    return [_wrap_dim(ax, rank, "fft") for ax in axes]


def _validate_axes_and_s(axes: list[int], s: list[int], op: str) -> None:
    if s and len(s) != len(axes):
        raise ValueError(f"{op}: len(s)={len(s)} must match len(dim)={len(axes)}")


def _promote(x: Tensor) -> Tensor:
    """An integer or boolean input in the default floating dtype.

    The engine transforms take only floating and complex input; the
    reference reads an integer signal as the real signal it denotes, so
    ``fft(arange(4))`` is a ``complex64`` spectrum rather than an error.
    """
    if x.is_floating_point() or x.is_complex():
        return x
    return x.to(lucid.get_default_dtype())


def _prepare(
    input: Tensor,
    s: int | Sequence[int] | None,
    dim: int | Sequence[int] | None,
    op: str,
) -> tuple[Tensor, list[int], list[int]]:
    """The promoted input, its normalised transform axes, and ``s`` as a list.

    With ``dim`` omitted the transform covers every axis — or, when ``s``
    is given, the last ``len(s)`` axes, which is what ``s`` then names.
    """
    input = _promote(input)
    rank = input.ndim
    s_list = _as_size_list(s)
    if dim is None and s_list:
        if len(s_list) > rank:
            raise ValueError(f"{op}: len(s)={len(s_list)} exceeds the input's {rank} dimensions")
        axes = list(range(rank - len(s_list), rank))
    else:
        axes = _normalise_axes(_as_axis_list(dim, rank, default_all=True), rank)
    _validate_axes_and_s(axes, s_list, op)
    return input, axes, s_list


def _input_sizes_along_axes(x_shape: tuple[int, ...], axes: list[int]) -> list[int]:
    return [int(x_shape[a]) for a in axes]


def _resize_axes(x: Tensor, axes: list[int], sizes: list[int]) -> Tensor:
    """``x`` cropped or zero-padded at the end of each axis to ``sizes``.

    The adjoint of what ``s`` does to a transform's input.  The engine
    crops an axis longer than its ``s`` entry and zero-pads a shorter one
    before transforming, so a gradient computed at the transform size has
    to be brought back to the input's: the cropped tail of the input
    received nothing (zero-pad the gradient), and the padding was never an
    input (crop it away).
    """
    from lucid.nn.functional import pad

    for axis, size in zip(axes, sizes, strict=True):
        have = int(x.shape[axis])
        if have > size:
            x = x.narrow(axis, 0, size)
        elif have < size:
            # ``pad`` takes (low, high) pairs from the last axis backwards.
            widths = [0] * (2 * x.ndim)
            widths[2 * (x.ndim - 1 - axis) + 1] = size - have
            x = pad(x, tuple(widths))
    return x


def _transform_size(s: list[int], in_sizes: list[int]) -> int:
    """Total length N transformed = product of per-axis lengths."""
    sizes = s if s else in_sizes
    N = 1
    for v in sizes:
        N *= int(v)
    return N


# ── Engine wrappers (raw, no normalisation) ──────────────────────────────────


def _engine_fftn(x: Tensor, s: list[int], axes: list[int]) -> Tensor:
    return _wrap(_fft.fftn(_unwrap(x), s, axes))


def _engine_ifftn(x: Tensor, s: list[int], axes: list[int]) -> Tensor:
    return _wrap(_fft.ifftn(_unwrap(x), s, axes))


def _engine_rfftn(x: Tensor, s: list[int], axes: list[int]) -> Tensor:
    return _wrap(_fft.rfftn(_unwrap(x), s, axes))


def _engine_irfftn(x: Tensor, s: list[int], axes: list[int]) -> Tensor:
    """The inverse real transform, promoting a real input first.

    The engine kernel takes a complex spectrum and refused anything else,
    so ``irfft`` / ``irfft2`` / ``irfftn`` / ``hfft`` of a real array
    raised ``irfftn requires C64 input, got float32``.  The reference
    accepts one — a real array *is* a spectrum with no imaginary part,
    and reading it as one is the only sensible interpretation.
    """
    if not x.is_complex():
        x = lucid.complex(x, lucid.zeros_like(x))
    return _wrap(_fft.irfftn(_unwrap(x), s, axes))


def _conj(x: Tensor) -> Tensor:
    """Conjugate, through the op that carries a gradient.

    This called ``_fft._conj_complex`` directly, which is a bare engine
    entry point with no autograd node, so it was where ``ihfft``'s chain
    ended: the forward was right and ``x.grad`` came back ``None``.
    ``lucid.conj`` is the same computation with a backward attached.
    """
    return lucid.conj(x)


def _scale(x: Tensor, s: float) -> Tensor:
    """Multiply by a real scalar, lane-wise when ``x`` is complex.

    ``x * s`` is enough for a real tensor.  For a complex one it is not:
    a scalar operand goes through the *broadcast* path, and that path has
    no complex branch — only equal-shape ``mul(C64, C64)`` does.  This
    used to say both were supported, and the mistake was invisible while
    nothing differentiated through a complex tensor.  The moment ``real``
    and ``imag`` had backwards, every FFT backward reached here and
    raised ``broadcast: dtype not supported``.

    Scaling a complex number by a real one is scaling each lane by it, so
    the projections do the work and no complex arithmetic is needed.
    """
    if s == 1.0:
        return x
    if not x.is_complex():
        return x * s
    return lucid.complex(lucid.real(x) * s, lucid.imag(x) * s)


# ── Autograd Function classes (the base transforms) ─────────────────────────
#
# Backward derivation for a complex DFT y = F(x):
#   J[k,n] = exp(-2πi kn/N).  The conjugate transpose is J^H[n,k] = exp(+2πi kn/N).
#   Under norm='backward', grad_x = J^H · grad_y = N · ifft(grad_y, 'backward')
#                                                = ifft(grad_y, 'forward').
# More generally grad_x = ifft(grad_y, dual(norm)) for any norm.  The same
# argument with the role of fft/ifft swapped gives the ifft backward.
#
# ``s`` makes every transform the composition of a resize — crop or
# zero-pad each transformed axis to its ``s`` entry — and a transform at
# that size.  The adjoint is the transposed composition: the dual transform
# at the transform size, then the opposite resize back to the input's own
# sizes (``_resize_axes``).  Every ``forward`` therefore saves the input's
# sizes along the transformed axes, and every ``backward`` ends at them —
# what a gradient for that input has to be shaped as.
#
# For the real variants the transform size along the last axis is needed
# as well, because rfft → n//2+1 is not invertible from the gradient shape
# alone.
#
# Every backward reaches its dual transform through that transform's own
# ``Function.apply`` rather than the raw engine entry point.  The raw call
# records nothing, so under ``create_graph`` the gradient came back
# detached and the engine stood a ZeroDerivativeBackward in for its
# derivative: every second derivative through a spectrum was a silent
# zero.  ``apply`` records a node only when grad mode is on and the
# incoming gradient requires grad — that is, only under ``create_graph``
# — and otherwise runs the same engine call as before.
#
# The dual is taken in the engine's native, unscaled ``'backward'`` form and
# the dual normalisation applied afterwards by ``_input_grad``, as before:
# for a real input the real part is taken first, so the scale multiplies
# a real tensor rather than both lanes of a complex one, and a first-order
# backward dispatches exactly the ops it did.  ``N`` is still the true
# transform size — the dual's own backward reads it.


@dataclass(frozen=True, slots=True)
class _Plan:
    """What a transform's backward needs to know about its forward.

    ``in_sizes`` are the input's sizes along ``axes`` — what the gradient
    is resized back to.  ``sizes`` are the transform sizes the real
    variants need beyond that: the full signal lengths ``rfftn`` read, or
    the output lengths ``irfftn`` produced.  Empty for the complex pair,
    which records instead whether it ran inverted.
    """

    axes: list[int]
    in_sizes: list[int]
    real_input: bool
    norm: str
    N: int
    sizes: list[int]
    inverse: bool

    @classmethod
    def of(
        cls,
        x: Tensor,
        axes: list[int],
        norm: str,
        N: int,
        sizes: list[int],
        inverse: bool = False,
    ) -> _Plan:
        in_sizes = _input_sizes_along_axes(tuple(x.shape), axes)
        return cls(axes, in_sizes, not x.is_complex(), norm, N, sizes, inverse)


def _forward(
    ctx: FunctionCtx,
    raw: Callable[[Tensor, list[int], list[int]], Tensor],
    x: Tensor,
    s: list[int],
    plan: _Plan,
    scale: Callable[[int, str], float],
) -> Tensor:
    """Record ``plan`` on ``ctx`` and run the raw transform, normalised."""
    ctx.plan = plan
    return _scale(raw(x, s, plan.axes), scale(plan.N, plan.norm))


def _plan(ctx: FunctionCtx) -> _Plan:
    plan = ctx.plan
    if not isinstance(plan, _Plan):
        raise TypeError(f"an FFT backward expected its forward's plan, got {plan!r}")
    return plan


def _transform(
    fn: type[_AutogradFunction],
    x: Tensor,
    s: list[int],
    axes: list[int],
    norm: str,
    N: int,
    extra: list[int] | bool,
) -> Tensor:
    """One base transform of ``x``, through its ``Function`` so it is tracked.

    The single entry point to the transform ``Function`` classes, for the
    public transforms and for each backward's dual transform alike — so a
    backward differentiates exactly as the forward it mirrors does.
    ``extra`` is the class's last argument: the direction of the complex
    pair, or the transform sizes of a real one.
    """
    if fn is _FftnAutograd:
        op_name = "ifftn" if extra else "fftn"
    elif fn is _RfftnAutograd:
        op_name = "rfftn"
    else:
        op_name = "irfftn"
    with _autocast_entry(op_name, _unwrap(x)) as (x_impl,):
        out = fn.apply(_wrap(x_impl), s, axes, norm, N, extra)
    if not isinstance(out, Tensor):
        raise TypeError(f"{fn.__name__} returned {type(out).__name__}")
    return out


def _input_grad(g: Tensor, plan: _Plan, scale: float) -> Tensor:
    """The dual transform's result as the gradient for ``forward``'s input.

    Kept real for a real input — the gradient of a real signal is the real
    part of the complex one — then scaled by the dual normalisation and
    resized from the transform size back to the input's sizes.
    """
    if plan.real_input and g.is_complex():
        g = lucid.real(g)
    # Rebound rather than nested, so the unscaled gradient is freed before
    # the resize allocates.
    g = _scale(g, scale)
    return _resize_axes(g, plan.axes, plan.in_sizes)


@final
class _FftnAutograd(_AutogradFunction):
    """``fftn``, or ``ifftn`` with ``inverse`` — the two are each other's dual."""

    @override
    @staticmethod
    def forward(
        ctx: FunctionCtx,
        x: Tensor,
        s: list[int],
        axes: list[int],
        norm: str,
        N: int,
        inverse: bool,
    ) -> Tensor:
        plan = _Plan.of(x, axes, norm, N, [], inverse)
        if inverse:
            return _forward(ctx, _engine_ifftn, x, s, plan, _scale_after_ifft)
        return _forward(ctx, _engine_fftn, x, s, plan, _scale_after_fft)

    @override
    @staticmethod
    def backward(ctx: FunctionCtx, grad_out: Tensor) -> Tensor:
        # grad_x = ifft(grad_out, dual(norm)) at the output's size for fft,
        # and the mirror for ifft, then resized to the input's.
        p = _plan(ctx)
        g = _transform(_FftnAutograd, grad_out, [], p.axes, "backward", p.N, not p.inverse)
        scale = _scale_after_fft if p.inverse else _scale_after_ifft
        return _input_grad(g, p, scale(p.N, _dual_norm(p.norm)))


def _hermitian_weights(like: Tensor, axis: int, bins: int, full: int) -> Tensor:
    """``[1, 2, 2, ..., 2, 1?]`` shaped to broadcast along ``axis``.

    The adjoint of a transform that reads a half-spectrum has to count
    each interior bin twice, because ``irfft`` reads each one twice: once
    as itself and once as the conjugate mirror it reconstructs.  DC is
    its own mirror, and so is Nyquist when the full length is even, so
    those two are counted once.

    Built from ``cat`` rather than an array literal because this is a
    compute path and may not import numpy (H4).
    """
    ones_head = lucid.ones(1, dtype=like.dtype, device=like.device)
    has_nyquist = full % 2 == 0 and bins > 1
    middle = bins - 1 - (1 if has_nyquist else 0)
    parts = [ones_head]
    if middle > 0:
        parts.append(lucid.full((middle,), 2.0, dtype=like.dtype, device=like.device))
    if has_nyquist:
        parts.append(lucid.ones(1, dtype=like.dtype, device=like.device))
    weights = lucid.cat(parts) if len(parts) > 1 else ones_head

    shape = [1] * like.ndim
    shape[axis % like.ndim] = bins
    return weights.reshape(tuple(shape))


@final
class _RfftnAutograd(_AutogradFunction):
    @override
    @staticmethod
    def forward(
        ctx: FunctionCtx,
        x: Tensor,
        s: list[int],
        axes: list[int],
        norm: str,
        N: int,
        full_sizes: list[int],
    ) -> Tensor:
        plan = _Plan.of(x, axes, norm, N, full_sizes)
        return _forward(ctx, _engine_rfftn, x, s, plan, _scale_after_fft)

    @override
    @staticmethod
    def backward(ctx: FunctionCtx, grad_out: Tensor) -> Tensor:
        """The adjoint of ``rfft``, which is not ``irfft``.

        This used to call ``irfftn``, and ``irfft`` is the *inverse* of
        ``rfft`` rather than its adjoint.  The two differ by exactly the
        thing that makes ``irfft`` cheap: it assumes its input is the
        half-spectrum of a real signal, so it counts every bin between DC
        and Nyquist twice, reconstructing the conjugate half it was not
        given.  The adjoint counts each bin once, because those
        conjugate entries are not inputs the gradient can flow to.

        Every interior bin was therefore doubled.  On a length-6 signal:

            d/dx sum|rfft(x)|   was  [-2.73, 0, 0.73, 1.27, 2, 4.73]
                                 is  [-1.37, 1, 0.37, 1.63, 1, 3.37]

        DC and Nyquist were right, which is the worst way for it to be
        wrong: the two bins a quick check looks at.

        Unreachable until complex tensors carried a gradient at all, so
        this is newly visible rather than newly broken.

        Zero-padding the half-spectrum back to full length and taking the
        ordinary inverse gives the adjoint directly: the padded bins
        contribute nothing, which is precisely "count each once".  The
        real part is taken because the input to ``rfft`` was real.

        That gradient is at the transform size; with ``s`` it is then
        resized to the input's, as for every transform here.
        """
        p = _plan(ctx)
        # Only the last transformed axis is halved by ``rfft``; the
        # others are full complex transforms already.
        padded = _resize_axes(grad_out, [p.axes[-1]], [p.sizes[-1]])
        g = _transform(_FftnAutograd, padded, [], p.axes, "backward", p.N, True)
        return _input_grad(g, p, _scale_after_ifft(p.N, _dual_norm(p.norm)))


@final
class _IrfftnAutograd(_AutogradFunction):
    @override
    @staticmethod
    def forward(
        ctx: FunctionCtx,
        x: Tensor,
        s: list[int],
        axes: list[int],
        norm: str,
        N: int,
        out_sizes: list[int],
    ) -> Tensor:
        # The input's sizes are in bins along the last axis — what the
        # gradient comes back to, whatever ``s`` asked the output to be.
        plan = _Plan.of(x, axes, norm, N, out_sizes)
        return _forward(ctx, _engine_irfftn, x, s, plan, _scale_after_ifft)

    @override
    @staticmethod
    def backward(ctx: FunctionCtx, grad_out: Tensor) -> Tensor:
        """The adjoint of ``irfft``, which is not ``rfft``.

        The mirror of the mistake in ``_RfftnAutograd``: ``rfft`` is the
        *inverse* of ``irfft`` and not its adjoint.  ``irfft`` reads each
        interior bin of its half-spectrum twice — once as itself and once
        as the conjugate mirror it reconstructs — so the adjoint sends
        the gradient back twice.  Plain ``rfft`` sends it once, leaving
        every bin between DC and Nyquist short by a factor of two.

        DC and Nyquist were right, being their own mirrors, which is
        again the worst way to be wrong: they are the entries a spot
        check reads first.

        The weighted spectrum has ``out_sizes[-1] // 2 + 1`` bins — the
        count ``irfft`` read, after cropping or zero-padding its input to
        fit ``s``.  It is resized to the input's own bin count (and its
        other axes to the input's sizes), or the engine would read a
        gradient of the wrong length as the input's.
        """
        p = _plan(ctx)
        g = _transform(_RfftnAutograd, grad_out, [], p.axes, "backward", p.N, p.sizes)

        last_axis = p.axes[-1]
        bins = int(g.shape[last_axis])
        weights = _hermitian_weights(lucid.real(g), last_axis, bins, int(p.sizes[-1]))
        # Lane-wise, because a broadcast multiply has no complex branch.
        g = lucid.complex(lucid.real(g) * weights, lucid.imag(g) * weights)
        return _input_grad(g, p, _scale_after_fft(p.N, _dual_norm(p.norm)))
