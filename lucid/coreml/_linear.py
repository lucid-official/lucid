"""The fully connected layer's translations: ``linear`` and the fused kernels.

``nn.FusedLinear`` runs ``_fused_linear_relu`` / ``_fused_linear_gelu`` at
inference, which export records as one op each. They are the same GEMM as
``linear`` with an activation after it, so they share its translation and
add MIL's ``relu``, or its ``gelu`` in the tanh mode the kernel computes
(LCD-450). :mod:`lucid.coreml._emit` registers all three.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lucid.coreml._build import Builder
    from lucid.coreml._emit import Bindings, EmitResult, TracedOp

__all__ = ["fused_linear_gelu", "fused_linear_relu", "linear", "matmul"]


def matmul(b: Builder, _op: TracedOp, ins: list[str]) -> EmitResult:
    """``x @ y``, de-aliasing ``x @ x`` so Core ML's BNNS compiler terminates.

    Apple's CPU MIL-to-BNNS graph compiler (``bnns::GraphCompile``) enters an
    infinite loop when a self-aliased ``matmul(x=v, y=v)`` feeds the right-hand
    side of a downstream ``matmul(x=v, y=...)`` such as ``x @ (x @ x)`` or
    ``x @ relu(x @ x)`` (LCD-20).  MIL's ``noop_elimination`` and
    ``fuse_transpose_matmul`` passes canonicalise ``identity`` and transposed
    operands back into the self-aliased form, whereas a full-extent
    ``slice_by_size`` along every axis is preserved as a distinct SSA value
    with identical shape (including dynamic ``shape_range`` axes) and bits.

    Parameters
    ----------
    b : Builder
        Program being written.
    _op : TracedOp
        The traced ``matmul`` operation.
    ins : list of str
        Value names of ``x`` and ``y``.

    Returns
    -------
    EmitResult
        MIL's ``matmul`` operation and its bindings.
    """
    lhs, rhs = ins[0], ins[1]
    if lhs == rhs:
        dims = b.shape_of(rhs)
        rank = len(dims)
        rhs = b.emit(
            "slice_by_size",
            [("x", rhs), ("begin", b.const_ints([0] * rank)), ("size", b.const_ints([-1] * rank))],
            dims,
        )
    return "matmul", [
        ("x", lhs),
        ("y", rhs),
        ("transpose_x", b.const_bool(False)),
        ("transpose_y", b.const_bool(False)),
    ]


def linear(b: Builder, op: TracedOp, ins: list[str]) -> EmitResult:
    """``x @ weight.T + bias``, as one operation where Core ML allows it.

    MIL's ``linear`` takes its weight and bias as constants only, and
    says so at parse time rather than at emission: a model that computes
    either of them — DETR builds a decoder bias from its query embedding
    — produces a package that loads nowhere, with ``Param 'bias' must be
    const`` and an internal value name to go on.

    So the fused operation is used when both operands really are
    constants, and otherwise the same arithmetic goes out as a transposed
    ``matmul`` and an ``add``. Two operations instead of one, only for
    the models that need it.

    Parameters
    ----------
    b : Builder
        Program being written.
    op : TracedOp
        The traced ``linear`` (or fused kernel) operation.
    ins : list of str
        Value names of ``x``, ``weight`` and, when present, ``bias``.

    Returns
    -------
    EmitResult
        The last MIL operation and its bindings, for the driver to append.
    """
    const_operands = all(b.is_const(name) for name in ins[1:])
    if const_operands:
        bindings: Bindings = [("x", ins[0]), ("weight", ins[1])]
        if len(ins) > 2:
            bindings.append(("bias", ins[2]))
        return "linear", bindings

    product = b.emit(
        "matmul",
        [
            ("x", ins[0]),
            ("y", ins[1]),
            ("transpose_x", b.const_bool(False)),
            ("transpose_y", b.const_bool(True)),
        ],
        b.result_dims(op),
    )
    if len(ins) == 2:
        return "identity", [("x", product)]
    return "add", [("x", product), ("y", ins[2])]


def _pre_activation(b: Builder, op: TracedOp, ins: list[str]) -> str:
    """A fused kernel's ``x @ weight.T + bias``, placed as a value of its own."""
    mil_type, bindings = linear(b, op, ins)
    return b.emit(mil_type, bindings, b.result_dims(op))


def fused_linear_relu(b: Builder, op: TracedOp, ins: list[str]) -> EmitResult:
    """``relu(x @ weight.T + bias)``: the linear translation, then MIL's ``relu``.

    Parameters
    ----------
    b : Builder
        Program being written.
    op : TracedOp
        The traced ``_fused_linear_relu`` operation.
    ins : list of str
        Value names of ``x``, ``weight`` and ``bias``.

    Returns
    -------
    EmitResult
        MIL's ``relu`` reading the pre-activation.
    """
    return "relu", [("x", _pre_activation(b, op, ins))]


def fused_linear_gelu(b: Builder, op: TracedOp, ins: list[str]) -> EmitResult:
    """``gelu(x @ weight.T + bias)`` in the tanh form the kernel computes.

    Parameters
    ----------
    b : Builder
        Program being written.
    op : TracedOp
        The traced ``_fused_linear_gelu`` operation.
    ins : list of str
        Value names of ``x``, ``weight`` and ``bias``.

    Returns
    -------
    EmitResult
        MIL's ``gelu`` in ``TANH_APPROXIMATION`` mode reading the
        pre-activation, as plain ``gelu`` translates.
    """
    return "gelu", [("x", _pre_activation(b, op, ins)), ("mode", b.const_str("TANH_APPROXIMATION"))]
