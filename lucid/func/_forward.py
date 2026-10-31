"""Forward-mode transforms: ``jvp``, ``linearize`` and ``jacfwd``.

A tangent here is the exact directional derivative
:func:`lucid.autograd.jvp` computes, and these decide whether it joins the
caller's autograd graph the way the reference framework's does — that
logic is :func:`_tangent_pass` and :func:`attachment`, shared by all three.
"""

import warnings
from collections.abc import Callable
from pathlib import Path
from typing import override

import lucid
from lucid._C import engine as _C_engine
from lucid._ctx import _RandomnessGuard, check_randomness
from lucid._factories.random import get_rng_state, set_rng_state
from lucid._tensor.tensor import Tensor
from lucid.autograd._functional import jvp as _jvp_by_double_backward
from lucid.func._function import FlatFunction, Selection, detach_aux, keeps_graph, pack_jacobian
from lucid.func._transform import Isolated, PyTree, Transform

__all__ = ["Jacfwd", "Linearization", "jvp_at", "linearize_at"]

#: Frames in this package are not where a warning is the caller's to fix.
_PACKAGE_DIR = str(Path(__file__).parent)


def _requires_grad(out: tuple[Tensor, ...]) -> bool:
    """Whether any tensor of ``out`` is attached to a graph."""
    return any(o.requires_grad for o in out)


def attachment(primals: tuple[Tensor, ...], out: tuple[Tensor, ...]) -> tuple[bool, bool]:
    """``(attach, required)`` for tangents at ``primals``, under the current
    grad mode, given ``func``'s output there evaluated in that mode.

    Required when a primal requires grad; attached then, and also when
    ``out`` reaches the caller's graph through tensors ``func`` captures.

    Parameters
    ----------
    primals : tuple of Tensor
        The tensors the tangents are taken at.
    out : tuple of Tensor
        ``func``'s output at ``primals``, evaluated in the caller's grad mode.

    Returns
    -------
    tuple of (bool, bool)
        Whether the tangents join the caller's graph, and whether they have
        to (an op without a graph-mode backward is then an error).
    """
    if not lucid.is_grad_enabled():
        return False, False
    required = any(p.requires_grad for p in primals)
    return required or _requires_grad(out), required


def _replay_draws() -> Callable[[], None]:
    """A rewind of the default generator to where it stands now.

    Calling the rewind before ``func`` is evaluated again makes that
    evaluation draw exactly what the previous one drew — the same dropout
    mask — and leaves the stream where one evaluation leaves it, so a
    transform that has to run ``func`` twice still looks like one call.
    """
    state = get_rng_state()

    def rewind() -> None:
        set_rng_state(state)

    return rewind


def _flat(value: Tensor | tuple[Tensor, ...]) -> tuple[Tensor, ...]:
    """``value`` as a tuple — a flat function's results always are one."""
    return value if isinstance(value, tuple) else (value,)


def _tangent_pass(
    fn: FlatFunction,
    primals: tuple[Tensor, ...],
    tangents: tuple[Tensor, ...],
    *,
    attach: bool,
    required: bool,
    strict: bool,
) -> tuple[tuple[Tensor, ...], tuple[Tensor, ...]]:
    """``fn``'s outputs and tangents at ``primals``, the tangents attached
    to the caller's graph when ``attach``.

    The attachment is ``required`` when a primal or a tangent requires grad:
    the caller differentiates through the tangent, and an op on the path
    with no graph-mode backward is then an error.  Otherwise only tensors
    ``fn`` captures would attach it, and through such an op the tangent is
    taken first-order and returned detached, with a warning — ``fn`` runs
    once more for it, on the same random draws.
    """
    rewind = _replay_draws() if attach and not required else None
    try:
        out, tan = _jvp_by_double_backward(
            fn, primals, tangents, create_graph=attach, strict=strict
        )
        return _flat(out), _flat(tan)
    except NotImplementedError:
        if rewind is None:
            raise
    warnings.warn(
        "lucid.func: an op on the path has no graph-mode backward, so the "
        "tangent cannot reach the tensors func captures and is returned "
        "detached",
        skip_file_prefixes=(_PACKAGE_DIR,),
    )
    rewind()
    out, tan = _jvp_by_double_backward(fn, primals, tangents, create_graph=False, strict=strict)
    return _flat(out), _flat(tan)


def _jvp_attached(
    fn: FlatFunction, primals: tuple[Tensor, ...], tangents: tuple[Tensor, ...], strict: bool
) -> tuple[tuple[Tensor, ...], tuple[Tensor, ...]]:
    """:func:`jvp_at` at tensor primals, its results attached to the caller's
    graph exactly where the reference attaches them."""
    grad_on = lucid.is_grad_enabled()
    if not grad_on or any(p.requires_grad for p in primals):
        return _tangent_pass(fn, primals, tangents, attach=grad_on, required=grad_on, strict=strict)
    # Nothing passed in requires grad, yet ``fn`` can still reach the
    # caller's graph through the tensors it captures — a model's parameters.
    # The backward passes run ``fn`` at copies of the primals that require
    # grad, so their output is attached whatever ``fn`` captures; only an
    # evaluation at the primals themselves tells.
    replay = _replay_draws()
    primals_out = fn.observe(*primals)
    replay()
    required = any(t.requires_grad for t in tangents)
    _, tangents_out = _tangent_pass(
        fn,
        primals,
        tangents,
        attach=required or _requires_grad(primals_out),
        required=required,
        strict=strict,
    )
    return primals_out, tangents_out


def jvp_at(
    func: Callable[..., object],
    primals: tuple[object, ...],
    tangents: tuple[object, ...],
    strict: bool,
    has_aux: bool,
) -> tuple[PyTree, PyTree] | tuple[PyTree, PyTree, object]:
    """``(func(*primals), J · tangents)``, both in ``func``'s output structure
    — and, for a ``has_aux`` function, ``func``'s side output after them.

    Parameters
    ----------
    func : Callable
        The function to differentiate, of the primals as positional arguments.
    primals : tuple
        Pytrees of tensors, the points to differentiate at; a leaf that is
        not a tensor is held fixed.
    tangents : tuple
        Pytrees of exactly the structure of ``primals``.
    strict : bool
        Refuse what :func:`lucid.autograd.jvp` refuses under ``strict``.
    has_aux : bool
        Whether ``func`` returns ``(output, aux)``.

    Returns
    -------
    tuple
        ``(primals_out, tangents_out)``, or ``(primals_out, tangents_out,
        aux)`` with ``aux`` detached unless the results keep the graph.
    """
    if len(primals) != len(tangents):
        raise RuntimeError(
            f"jvp: expected one tangent per primal, got {len(primals)} primals "
            f"and {len(tangents)} tangents"
        )
    selection = Selection(primals, tuple(range(len(primals))), "jvp", allow_fixed=True)
    flat = FlatFunction(func, primals, {}, selection, has_aux=has_aux, name="jvp")
    primals_out, tangents_out = _jvp_attached(
        flat, selection.leaves, selection.tangents(tangents), strict
    )
    pair = (flat.out_spec.unflatten(primals_out), flat.out_spec.unflatten(tangents_out))
    if not has_aux:
        return pair
    return (*pair, detach_aux(flat.aux, keep=keeps_graph(selection.leaves)))


class Linearization(Isolated):
    """The ``linear_fn`` that :func:`linearize_at` returns: ``t ↦ J t`` at the
    primals it was made at.

    Whether the tangent joins the caller's graph was decided by the primal
    output, once, when the linearization was made.
    """

    __slots__ = ("_flat", "_selection", "_n_primals", "_primal_attached", "_attached")

    def __init__(
        self,
        flat: FlatFunction,
        selection: Selection,
        n_primals: int,
        primals_out: tuple[Tensor, ...],
    ) -> None:
        self._flat = flat
        self._selection = selection
        self._n_primals = n_primals
        self._attached, self._primal_attached = attachment(selection.leaves, primals_out)

    def __call__(self, *tangents: object) -> PyTree:
        if len(tangents) != self._n_primals:
            raise RuntimeError(
                f"linearize: expected one tangent per primal, got {self._n_primals} "
                f"primals and {len(tangents)} tangents"
            )
        picked = self._selection.tangents(tangents)
        on = lucid.is_grad_enabled()
        required = on and (self._primal_attached or any(t.requires_grad for t in picked))
        _, tangents_out = _tangent_pass(
            self._flat,
            self._selection.leaves,
            picked,
            attach=required or (on and self._attached),
            required=required,
            strict=False,
        )
        return self._flat.out_spec.unflatten(tangents_out)


def linearize_at(
    func: Callable[..., object], primals: tuple[object, ...]
) -> tuple[PyTree, Linearization]:
    """``(func(*primals), linear_fn)`` — the primal output and the map
    from tangents to ``J · tangents`` at ``primals``, as :func:`lucid.func.linearize`
    returns them.

    Parameters
    ----------
    func : Callable
        The function to linearise.
    primals : tuple
        Pytrees of tensors, the point to linearise at; a leaf that is not a
        tensor is held fixed.

    Returns
    -------
    tuple
        ``(primals_out, linear_fn)``.
    """
    selection = Selection(primals, tuple(range(len(primals))), "linearize", allow_fixed=True)
    flat = FlatFunction(func, primals, {}, selection, has_aux=False, name="linearize")
    primals_out = flat.observe(*selection.leaves)
    return (
        flat.out_spec.unflatten(primals_out),
        Linearization(flat, selection, len(primals), primals_out),
    )


def _one_hot_tangent(x: Tensor, col: int) -> Tensor:
    x_numel = x.numel()
    x_shape = list(x.shape)
    dev = x._impl.device
    dtype = x._impl.dtype
    t_flat = _C_engine.zeros([x_numel], dtype, dev)
    one = _C_engine.ones([1], dtype, dev)
    idx = _C_engine.full([1], float(col), _C_engine.I32, dev)
    t_flat = _C_engine.scatter_add(t_flat, idx, one, 0)
    return Tensor.__new_from_impl__(_C_engine.reshape(t_flat, x_shape))


def _forward_blocks(
    fn: FlatFunction,
    primals: tuple[Tensor, ...],
    sample: tuple[Tensor, ...],
    *,
    attach: bool,
    required: bool,
) -> list[list[Tensor]]:
    """The Jacobian of every output of ``fn`` in every primal: ``blocks[i][j]``
    is output ``i`` in primal ``j``, of shape ``out.shape + primal.shape``.

    One :func:`jvp` along a unit tangent per column of each primal gives that
    column of every output's block at once.
    """
    blocks: list[list[Tensor]] = [[] for _ in sample]
    for target, x in enumerate(primals):
        columns: list[list[Tensor]] = [[] for _ in sample]
        for col in range(x.numel()):
            tangents = tuple(
                _one_hot_tangent(x, col) if k == target else lucid.zeros_like(p)
                for k, p in enumerate(primals)
            )
            _, tangents_out = _tangent_pass(
                fn, primals, tangents, attach=attach, required=required, strict=False
            )
            for column, tangent in zip(columns, tangents_out, strict=True):
                column.append(tangent.reshape(-1))
        for block, column, out in zip(blocks, columns, sample, strict=True):
            shape = list(out.shape) + list(x.shape)
            block.append(
                lucid.stack(column, dim=-1).reshape(shape)
                if column
                else lucid.zeros(tuple(shape), dtype=out.dtype, device=out.device)
            )
    return blocks


class Jacfwd[**P](Transform[P, PyTree]):
    """``jacfwd(func, argnums, has_aux, randomness=)``: the Jacobian of
    ``func``, a column at a time by forward-mode products."""

    __slots__ = ("argnums", "has_aux", "randomness")

    def __init__(
        self,
        func: Callable[P, object],
        argnums: int | tuple[int, ...],
        has_aux: bool,
        randomness: str,
    ) -> None:
        super().__init__(func)
        self.argnums = argnums
        self.has_aux = has_aux
        self.randomness = check_randomness(randomness)

    @override
    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> PyTree:
        selection = Selection(args, self.argnums, "jacfwd")
        flat = FlatFunction(self.func, args, kwargs, selection, has_aux=self.has_aux, name="jacfwd")
        primals = selection.leaves
        with _RandomnessGuard(self.randomness):
            # At the arguments, in the caller's grad mode: the outputs' shapes,
            # the aux, and the attachment — decided once, not per column.
            sample = flat.observe(*primals)
            attach, required = attachment(primals, sample)
            blocks = _forward_blocks(flat, primals, sample, attach=attach, required=required)
        packed = pack_jacobian(flat.out_spec, selection.spec, blocks)
        return (packed, detach_aux(flat.aux, keep=attach)) if self.has_aux else packed
