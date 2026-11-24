"""A forward that builds its own result leaves storage that matches its shape (LCD-1072).

LCD-1062 made a GPU result of another extent than its shape an error, in the forwards of the three
kernel families.  The guard is in ``helpers::fresh`` now (``core/ResultExtent.h``), the function
every forward builds its result through, and the forwards that built it by hand — bitwise,
compare, scans, sort, concat, linalg, fft, the nn layers, the factories — go through it.  What is
held here is the class, not the ops it was found in: for each family, every result *and* every
gradient holds exactly the bytes its shape names, over scalar, empty and regular shapes, on each
stream available (the CPU stream is the control).

A view shares its parent's buffer on purpose, so ``diag``, ``diagonal``, ``split``, ``chunk``,
``unbind`` and what is built on them are not in the table.
"""

from collections.abc import Callable
from typing import TYPE_CHECKING

import pytest

import lucid
import lucid.fft as fft
import lucid.linalg as la
import lucid.nn as nn
import lucid.nn.functional as F

if TYPE_CHECKING:
    from _pytest.mark import ParameterSet

Shape = tuple[int, ...]
Build = Callable[[str, lucid.dtype, Shape], object]

_FLOATS = (lucid.float32, lucid.float16, lucid.bfloat16)
_INTS = (lucid.int32, lucid.int64, lucid.int8)
_REAL = _FLOATS + _INTS
_ANY = _REAL + (lucid.bool,)

#: scalar, a vector, a matrix, and two empty shapes
_ALL_SHAPES: tuple[Shape, ...] = ((), (5,), (3, 4), (0,), (2, 0, 3))
#: the shapes a call along an axis needs
_AXIS_SHAPES: tuple[Shape, ...] = ((5,), (3, 4), (0,), (2, 0, 3))
#: (a, b): equal, a 0 against a 1 either side, a scalar partner, a broadcast over a matrix
_PAIRS: tuple[tuple[Shape, Shape], ...] = (
    ((5,), (5,)),
    ((3, 4), (4,)),
    ((3, 4), (3, 1)),
    ((0,), (1,)),
    ((1,), (0,)),
    ((0, 3), (1, 3)),
    ((2, 0, 3), (1, 3)),
    ((0,), ()),
)
#: square matrices: single, batched, and the empty ones a factorisation must still answer
_SQUARES: tuple[Shape, ...] = ((3, 3), (2, 3, 3), (0, 3, 3), (2, 0, 0), (0, 0))


def _t(shape: Shape, dtype: lucid.dtype, device: str, *, grad: bool = False) -> lucid.Tensor:
    """Distinct small positive values of ``shape`` and ``dtype``."""
    n = 1
    for d in shape:
        n *= d
    base = lucid.arange(n, dtype=lucid.float32, device=device).reshape(shape) * 0.1 + 0.1
    if dtype == lucid.bool:
        out = base > 0.35
    elif dtype in _INTS:
        out = (base * 10).to(dtype=dtype)
    else:
        out = base.to(dtype=dtype)
    return out.requires_grad_(grad) if grad and dtype in _FLOATS else out


def _square(shape: Shape, dtype: lucid.dtype, device: str, *, grad: bool = False) -> lucid.Tensor:
    """A well-conditioned (batched) square matrix that every factorisation accepts."""
    if 0 in shape:
        return lucid.zeros(shape, dtype=dtype, device=device)
    eye = lucid.eye(shape[-1], dtype=dtype, device=device)
    out = (_t(shape, lucid.float32, device) * 0.01 + eye * 3.0).to(dtype=dtype)
    return out.requires_grad_(grad) if grad else out


def _tensors(out: object) -> list[lucid.Tensor]:
    if isinstance(out, lucid.Tensor):
        return [out]
    if isinstance(out, (tuple, list)):
        return [t for item in out for t in _tensors(item)]
    return []


def _consistent(t: lucid.Tensor, what: str) -> None:
    """The storage holds exactly the elements the shape names."""
    assert t._impl.storage_nbytes() == t._impl.nbytes(), (
        f"{what}: shape {tuple(t.shape)} {t.dtype}: storage {t._impl.storage_nbytes()} B, "
        f"shape needs {t._impl.nbytes()} B"
    )


def _unary(fn: Callable[[lucid.Tensor], object], *, grad: bool = False) -> Build:
    def build(device: str, dtype: lucid.dtype, shape: Shape) -> object:
        x = _t(shape, dtype, device, grad=grad)
        out = fn(x)
        if grad and x.requires_grad:
            for t in _tensors(out):
                if t.requires_grad and t.dtype in _FLOATS:
                    t.sum().backward(retain_graph=True)
            if x.grad is not None:
                _consistent(x.grad, "grad")
        return out

    return build


def _cases(
    table: dict[str, tuple[Build, tuple[lucid.dtype, ...], tuple[Shape, ...]]],
) -> list[ParameterSet]:
    return [
        pytest.param(name, build, dt, sh, id=f"{name}-{dt}-{'x'.join(map(str, sh)) or 'scalar'}")
        for name, (build, dtypes, shapes) in table.items()
        for dt in dtypes
        for sh in shapes
    ]


# ── one operand ─────────────────────────────────────────────────────────────────────────────

_ONE_OPERAND = {
    # predicates and reductions with an output of their own dtype
    "isnan": (_unary(lucid.isnan), _FLOATS, _ALL_SHAPES),
    "isinf": (_unary(lucid.isinf), _FLOATS, _ALL_SHAPES),
    "isfinite": (_unary(lucid.isfinite), _FLOATS, _ALL_SHAPES),
    "all": (_unary(lucid.all), _ANY, _ALL_SHAPES),
    "any": (_unary(lucid.any), _ANY, _ALL_SHAPES),
    "bitwise_not": (_unary(lucid.bitwise_not), (lucid.int32, lucid.int8, lucid.bool), _ALL_SHAPES),
    "var": (_unary(lucid.var, grad=True), _FLOATS, ((5,), (3, 4))),
    "std": (_unary(lucid.std, grad=True), _FLOATS, ((5,), (3, 4))),
    # scans, sorts, selections
    "cumsum": (_unary(lambda x: lucid.cumsum(x, dim=0), grad=True), _REAL, _AXIS_SHAPES),
    "cumprod": (_unary(lambda x: lucid.cumprod(x, dim=0), grad=True), _REAL, _AXIS_SHAPES),
    "cummax": (_unary(lambda x: lucid.cummax(x, dim=0), grad=True), _REAL, _AXIS_SHAPES),
    "cummin": (_unary(lambda x: lucid.cummin(x, dim=0), grad=True), _REAL, _AXIS_SHAPES),
    "logcumsumexp": (
        _unary(lambda x: lucid.logcumsumexp(x, dim=0), grad=True),
        _FLOATS,
        _AXIS_SHAPES,
    ),
    "sort": (_unary(lambda x: lucid.sort(x, dim=-1), grad=True), _REAL, _AXIS_SHAPES),
    "argsort": (_unary(lambda x: lucid.argsort(x, dim=-1)), _REAL, _AXIS_SHAPES),
    "msort": (_unary(lucid.msort, grad=True), _REAL, _AXIS_SHAPES),
    "topk": (_unary(lambda x: lucid.topk(x, 1), grad=True), _REAL, ((5,), (3, 4))),
    "nonzero": (_unary(lucid.nonzero), _ANY, _ALL_SHAPES),
    "unique": (_unary(lucid.unique), _REAL, _AXIS_SHAPES),
    "masked_select": (
        _unary(lambda x: lucid.masked_select(x, x > 0.3), grad=True),
        _FLOATS,
        _ALL_SHAPES,
    ),
    "masked_fill": (
        _unary(lambda x: lucid.masked_fill(x, x > 0.3, 1), grad=True),
        _REAL,
        _ALL_SHAPES,
    ),
    "where": (_unary(lambda x: lucid.where(x > 0.3, x, x * 2), grad=True), _FLOATS, _ALL_SHAPES),
    "gather": (
        _unary(
            lambda x: lucid.gather(x, 0, lucid.zeros(x.shape, dtype=lucid.int64, device=x.device)),
            grad=True,
        ),
        _FLOATS,
        _AXIS_SHAPES,
    ),
    # layout
    "tril": (_unary(lucid.tril, grad=True), _REAL, ((3, 4), (0, 3), (3, 0))),
    "triu": (_unary(lucid.triu, grad=True), _REAL, ((3, 4), (0, 3), (3, 0))),
    "flip": (_unary(lambda x: lucid.flip(x, (0,)), grad=True), _REAL, _AXIS_SHAPES),
    "roll": (_unary(lambda x: lucid.roll(x, 1, 0), grad=True), _REAL, _AXIS_SHAPES),
    "cat": (_unary(lambda x: lucid.cat([x, x]), grad=True), _REAL, _AXIS_SHAPES),
    "stack": (_unary(lambda x: lucid.stack([x, x]), grad=True), _REAL, _ALL_SHAPES),
    "tile": (_unary(lambda x: lucid.tile(x, (2,)), grad=True), _REAL, _ALL_SHAPES),
    "repeat_interleave": (
        _unary(lambda x: lucid.repeat_interleave(x, 2), grad=True),
        _REAL,
        _ALL_SHAPES,
    ),
    "pad": (_unary(lambda x: F.pad(x, (1, 1)), grad=True), _FLOATS, _AXIS_SHAPES),
    "meshgrid": (_unary(lambda x: lucid.meshgrid(x, x, indexing="ij")), _REAL, ((5,), (0,))),
    "histc": (_unary(lambda x: lucid.histc(x, bins=4, min=0, max=3)), _FLOATS, _ALL_SHAPES),
    "bincount": (_unary(lambda x: lucid.bincount(x)), (lucid.int32, lucid.int64), _AXIS_SHAPES[:1]),
    "trace": (_unary(lucid.trace, grad=True), _FLOATS, ((3, 3), (3, 4), (0, 3))),
    "conj": (_unary(lucid.conj), (lucid.float32, lucid.complex64), _ALL_SHAPES),
}


@pytest.mark.parametrize(("name", "build", "dtype", "shape"), _cases(_ONE_OPERAND))
def test_a_one_operand_forward_result_has_matching_storage(
    device: str, name: str, build: Build, dtype: lucid.dtype, shape: Shape
) -> None:
    for t in _tensors(build(device, dtype, shape)):
        _consistent(t, name)


# ── two operands ────────────────────────────────────────────────────────────────────────────


def _binary(
    fn: Callable[[lucid.Tensor, lucid.Tensor], object], *, grad: bool = False
) -> Callable[..., object]:
    def build(device: str, dtype: lucid.dtype, a_shape: Shape, b_shape: Shape) -> object:
        a, b = _t(a_shape, dtype, device, grad=grad), _t(b_shape, dtype, device, grad=grad)
        out = fn(a, b)
        if grad and a.requires_grad:
            for t in _tensors(out):
                if t.requires_grad and t.dtype in _FLOATS:
                    t.sum().backward(retain_graph=True)
            for operand in (a, b):
                if operand.grad is not None:
                    _consistent(operand.grad, "grad")
        return out

    return build


_COMPARES = ("eq", "ne", "lt", "le", "gt", "ge")
_TWO_OPERANDS: dict[str, tuple[Callable[..., object], tuple[lucid.dtype, ...]]] = {
    **{name: (_binary(getattr(lucid, name)), _ANY) for name in _COMPARES},
    "bitwise_and": (_binary(lucid.bitwise_and), (lucid.int32, lucid.int8, lucid.bool)),
    "bitwise_or": (_binary(lucid.bitwise_or), (lucid.int32, lucid.int8, lucid.bool)),
    "bitwise_xor": (_binary(lucid.bitwise_xor), (lucid.int32, lucid.int8, lucid.bool)),
    "bitwise_left_shift": (_binary(lucid.bitwise_left_shift), _INTS),
    "bitwise_right_shift": (_binary(lucid.bitwise_right_shift), _INTS),
    "logical_and": (_binary(lucid.logical_and), _ANY),
    "logical_or": (_binary(lucid.logical_or), _ANY),
    "maximum": (_binary(lucid.maximum, grad=True), _REAL),
    "minimum": (_binary(lucid.minimum, grad=True), _REAL),
}


def _pair_cases() -> list[ParameterSet]:
    return [
        pytest.param(
            name,
            build,
            dt,
            a,
            b,
            id=f"{name}-{dt}-{'x'.join(map(str, a)) or 's'}_{'x'.join(map(str, b)) or 's'}",
        )
        for name, (build, dtypes) in _TWO_OPERANDS.items()
        for dt in dtypes
        for a, b in _PAIRS
    ]


@pytest.mark.parametrize(("name", "build", "dtype", "a_shape", "b_shape"), _pair_cases())
def test_a_two_operand_forward_result_has_matching_storage(
    device: str,
    name: str,
    build: Callable[..., object],
    dtype: lucid.dtype,
    a_shape: Shape,
    b_shape: Shape,
) -> None:
    for t in _tensors(build(device, dtype, a_shape, b_shape)):
        _consistent(t, name)


@pytest.mark.parametrize("dtype", _REAL, ids=str)
@pytest.mark.parametrize("values", ((), (3,), (2, 3), (0,)), ids=str)
@pytest.mark.parametrize("n", (0, 5))
def test_searchsorted_result_has_matching_storage(
    device: str, dtype: lucid.dtype, n: int, values: Shape
) -> None:
    _consistent(
        lucid.searchsorted(_t((n,), dtype, device), _t(values, dtype, device)), "searchsorted"
    )


@pytest.mark.parametrize("dtype", _FLOATS + _INTS, ids=str)
@pytest.mark.parametrize("n", (0, 1, 5))
def test_the_products_of_vectors_have_matching_storage(
    device: str, dtype: lucid.dtype, n: int
) -> None:
    a, b = _t((n,), dtype, device), _t((n,), dtype, device)
    for name, out in (
        ("outer", la.outer(a, b)),
        ("dot", la.dot(a, b)),
        ("inner", la.inner(a, b)),
        ("tensordot", lucid.tensordot(a, b, dims=0)),
    ):
        _consistent(out, name)


# ── factorisations, solves and transforms ────────────────────────────────────────────────────

_MATRIX = {
    name: getattr(la, name)
    for name in (
        "cholesky",
        "inv",
        "det",
        "slogdet",
        "eig",
        "eigh",
        "eigvals",
        "eigvalsh",
        "svd",
        "svdvals",
        "pinv",
        "qr",
        "lu",
        "lu_factor",
        "ldl_factor",
        "matrix_exp",
        "matrix_rank",
        "matrix_norm",
        "norm",
        "cholesky_inverse",
    )  # fmt: skip
}


@pytest.mark.parametrize("shape", _SQUARES, ids=str)
@pytest.mark.parametrize("dtype", (lucid.float32,), ids=str)
@pytest.mark.parametrize("name", sorted(_MATRIX))
def test_a_factorisation_result_and_its_gradient_have_matching_storage(
    device: str, name: str, dtype: lucid.dtype, shape: Shape
) -> None:
    a = _square(shape, dtype, device, grad=True)
    out = _MATRIX[name](a)
    for t in _tensors(out):
        _consistent(t, name)
        if t.requires_grad and t.dtype in _FLOATS:
            t.sum().backward(retain_graph=True)
    if a.grad is not None:
        _consistent(a.grad, f"{name} grad")


#: (A, right-hand side): a matrix of the same shape, a wider one, and — for a single A — a vector
_SOLVES = [(s, r) for s in _SQUARES for r in ("same", "wide")] + [
    ((3, 3), "vector"),
    ((0, 0), "vector"),
]


@pytest.mark.parametrize(("shape", "rhs"), _SOLVES, ids=str)
def test_a_solve_result_and_its_gradients_have_matching_storage(
    device: str, shape: Shape, rhs: str
) -> None:
    a = _square(shape, lucid.float32, device, grad=True)
    b_shape = {"same": shape, "vector": shape[:-1], "wide": (*shape[:-1], 2)}[rhs]
    b = _t(b_shape, lucid.float32, device, grad=True)
    lu, pivots = la.lu_factor(_square(shape, lucid.float32, device))
    for name, out in (
        ("solve", la.solve(a, b)),
        ("solve_triangular", la.solve_triangular(a, b, upper=True)),
        ("lu_solve", la.lu_solve(lu, pivots, b)),
    ):
        for t in _tensors(out):
            _consistent(t, name)
            if t.requires_grad:
                t.sum().backward(retain_graph=True)
    for operand in (a, b):
        if operand.grad is not None:
            _consistent(operand.grad, "solve grad")


_COMPLEX = (lucid.float32, lucid.complex64)
_RANKS_1_TO_3: tuple[Shape, ...] = ((8,), (2, 4), (3, 4, 4))
_RANKS_2_TO_3: tuple[Shape, ...] = ((2, 4), (3, 4, 4))
#: name -> (dtypes it takes, shapes it takes).  A transform of an empty extent is refused by
#: design ("invalid number of data points"), so the empty shapes are not in the table.
_FFT: dict[str, tuple[tuple[lucid.dtype, ...], tuple[Shape, ...]]] = {
    "fft": (_COMPLEX, _RANKS_1_TO_3),
    "ifft": (_COMPLEX, _RANKS_1_TO_3),
    "irfft": (_COMPLEX, _RANKS_1_TO_3),
    "hfft": (_COMPLEX, _RANKS_1_TO_3),
    "fftn": (_COMPLEX, _RANKS_1_TO_3),
    "ifftn": (_COMPLEX, _RANKS_1_TO_3),
    "rfft": ((lucid.float32,), _RANKS_1_TO_3),
    "ihfft": ((lucid.float32,), _RANKS_1_TO_3),
    "rfftn": ((lucid.float32,), _RANKS_1_TO_3),
    "fft2": (_COMPLEX, _RANKS_2_TO_3),
    "ifft2": (_COMPLEX, _RANKS_2_TO_3),
    "irfft2": (_COMPLEX, _RANKS_2_TO_3),
    "rfft2": ((lucid.float32,), _RANKS_2_TO_3),
    "fftshift": (_COMPLEX, (*_ALL_SHAPES, (3, 4, 4))),
}


@pytest.mark.parametrize(
    ("name", "dtype", "shape"),
    [(n, dt, sh) for n, (dtypes, shapes) in _FFT.items() for dt in dtypes for sh in shapes],
    ids=lambda v: str(v).replace(" ", ""),
)
def test_a_transform_result_has_matching_storage(
    device: str, name: str, dtype: lucid.dtype, shape: Shape
) -> None:
    for t in _tensors(getattr(fft, name)(_t(shape, dtype, device))):
        _consistent(t, name)


# ── layers, losses and factories ─────────────────────────────────────────────────────────────

_LEADS: tuple[Shape, ...] = ((), (2,), (0,), (2, 0))  # in front of a feature axis of 4
_IMAGES: tuple[Shape, ...] = ((2, 4, 6), (2, 4, 6, 6), (0, 4, 6, 6), (2, 4, 3, 4, 5))
_ACTIVATIONS = ("relu", "gelu", "silu", "elu", "leaky_relu", "softplus", "softmax", "log_softmax")
_REDUCTIONS = ("mean", "sum", "none")
_LOSSES = ("mse_loss", "l1_loss", "smooth_l1_loss", "huber_loss", "bce_logits", "kl_div")


def _features(device: str, dtype: lucid.dtype, lead: Shape) -> lucid.Tensor:
    return _t((*lead, 4), dtype, device, grad=True)


def _backward(out: object, *operands: lucid.Tensor) -> object:
    """Run the backward of ``out`` and hold every gradient it leaves to the same count."""
    for t in _tensors(out):
        if t.requires_grad and t.dtype in _FLOATS:
            t.sum().backward(retain_graph=True)
    for operand in operands:
        if operand.grad is not None:
            _consistent(operand.grad, "grad")
    return out


def _linear(device: str, dtype: lucid.dtype, lead: Shape) -> object:
    x, w = _features(device, dtype, lead), _t((3, 4), dtype, device, grad=True)
    return _backward(F.linear(x, w, _t((3,), dtype, device)), x, w)


def _matrix_product(device: str, dtype: lucid.dtype, lead: Shape) -> object:
    x, w = _features(device, dtype, lead), _t((3, 4), dtype, device, grad=True)
    return _backward(lucid.matmul(x, w.T), x, w)


def _normalised(fn: Callable[[lucid.Tensor], lucid.Tensor]) -> Build:
    def build(device: str, dtype: lucid.dtype, lead: Shape) -> object:
        x = _features(device, dtype, lead)
        return _backward(fn(x), x)

    return build


def _activation(name: str) -> Build:
    fn = getattr(F, name)
    return _normalised(lambda x: fn(x, dim=-1) if name.endswith("softmax") else fn(x))


def _image_norm(name: str) -> Build:
    def build(device: str, dtype: lucid.dtype, shape: Shape) -> object:
        x = _t(shape, dtype, device, grad=True)
        gamma, beta = _t((4,), dtype, device, grad=True), _t((4,), dtype, device, grad=True)
        mean, var = _t((4,), dtype, device), _t((4,), dtype, device)
        out = {
            "batch_norm": lambda: F.batch_norm(x, None, None, gamma, beta, training=True),
            "batch_norm_running": lambda: F.batch_norm(x, mean, var, gamma, beta, training=True),
            "batch_norm_eval": lambda: F.batch_norm(x, mean, var, gamma, beta, training=False),
            "group_norm": lambda: F.group_norm(x, 2, gamma, beta),
            "instance_norm": lambda: F.instance_norm(x),
        }[name]()
        return _backward(out, x, gamma, beta)

    return build


def _conv(device: str, dtype: lucid.dtype, shape: Shape) -> object:
    nd = len(shape) - 2
    x, w = _t(shape, dtype, device, grad=True), _t((4, 4, *(3,) * nd), dtype, device, grad=True)
    return _backward(getattr(F, f"conv{nd}d")(x, w, None, padding=1), x, w)


def _conv_transpose(device: str, dtype: lucid.dtype, shape: Shape) -> object:
    nd = len(shape) - 2
    x, w = _t(shape, dtype, device, grad=True), _t((4, 4, *(3,) * nd), dtype, device, grad=True)
    return _backward(getattr(F, f"conv_transpose{nd}d")(x, w), x, w)


def _pool(kind: str) -> Build:
    def build(device: str, dtype: lucid.dtype, shape: Shape) -> object:
        x = _t(shape, dtype, device, grad=True)
        return _backward(getattr(F, f"{kind}_pool{len(shape) - 2}d")(x, 2), x)

    return build


def _interpolate(device: str, dtype: lucid.dtype, shape: Shape) -> object:
    mode = {3: "linear", 4: "bilinear", 5: "trilinear"}[len(shape)]
    x = _t(shape, dtype, device, grad=True)
    return _backward(F.interpolate(x, scale_factor=2.0, mode=mode, align_corners=False), x)


def _nearest(device: str, dtype: lucid.dtype, shape: Shape) -> object:
    x = _t(shape, dtype, device, grad=True)
    return _backward(F.interpolate(x, scale_factor=2.0, mode="nearest"), x)


def _loss(name: str) -> Build:
    def build(device: str, dtype: lucid.dtype, shape: Shape) -> object:
        x, y = _t(shape, dtype, device, grad=True), _t(shape, dtype, device).clip(0.1, 0.9)
        outs: list[object] = []
        for reduction in _REDUCTIONS:
            if name == "bce_logits":
                out = F.binary_cross_entropy_with_logits(x, y, reduction=reduction)
            elif name == "kl_div":
                out = F.kl_div(F.log_softmax(x, dim=-1), F.softmax(y, dim=-1), reduction=reduction)
            else:
                out = getattr(F, name)(x, y, reduction=reduction)
            outs.append(_backward(out, x))
        return outs

    return build


def _classification(name: str) -> Build:
    def build(device: str, dtype: lucid.dtype, shape: Shape) -> object:
        x = _t(shape, dtype, device, grad=True)
        target = lucid.arange(shape[0], dtype=lucid.int64, device=device) % shape[1]
        outs: list[object] = []
        for reduction in _REDUCTIONS:
            if name == "cross_entropy":
                out = F.cross_entropy(x, target, reduction=reduction)
            else:
                out = F.nll_loss(F.log_softmax(x, dim=-1), target, reduction=reduction)
            outs.append(_backward(out, x))
        return outs

    return build


def _attention(device: str, dtype: lucid.dtype, shape: Shape) -> object:
    q = _t(shape, dtype, device, grad=True)
    k, v = _t((2, 3, 4), dtype, device, grad=True), _t((2, 3, 4), dtype, device, grad=True)
    return [
        _backward(F.scaled_dot_product_attention(q, k, v), q, k, v),
        _backward(F.scaled_dot_product_attention(q, k, v, is_causal=True), q, k, v),
    ]


def _embedding(device: str, dtype: lucid.dtype, lead: Shape) -> object:
    w = _t((6, 4), dtype, device, grad=True)
    n = 1
    for d in lead:
        n *= d
    ids = lucid.arange(n, dtype=lucid.int64, device=device).reshape(lead) % 6
    return _backward(F.embedding(ids, w), w)


def _lstm(device: str, dtype: lucid.dtype, shape: Shape) -> object:
    layer = nn.LSTM(4, 5, batch_first=True).to(dtype=dtype, device=device)
    x = _t(shape, dtype, device, grad=True)
    return _backward(layer(x), x)


def _factory(name: str) -> Build:
    def build(device: str, dtype: lucid.dtype, shape: Shape) -> object:
        if name == "full":
            return lucid.full(shape, 2, dtype=dtype, device=device)
        return getattr(lucid, name)(shape, dtype=dtype, device=device)

    return build


def _scatter(name: str) -> Build:
    def build(device: str, dtype: lucid.dtype, shape: Shape) -> object:
        x, src = _t(shape, dtype, device, grad=True), _t(shape, dtype, device, grad=True)
        index = lucid.zeros(shape, dtype=lucid.int64, device=device)
        return _backward(getattr(lucid, name)(x, 0, index, src), x, src)

    return build


_LAYERS: dict[str, tuple[Build, tuple[lucid.dtype, ...], tuple[Shape, ...]]] = {
    "linear": (_linear, _FLOATS, _LEADS),
    "matmul": (_matrix_product, _FLOATS, _LEADS),
    "layer_norm": (_normalised(lambda x: F.layer_norm(x, (4,))), _FLOATS, _LEADS),
    "rms_norm": (_normalised(lambda x: F.rms_norm(x, (4,))), _FLOATS, _LEADS),
    "normalize": (_normalised(lambda x: F.normalize(x, dim=-1)), _FLOATS, _LEADS),
    "dropout": (_normalised(lambda x: F.dropout(x, 0.5, True)), _FLOATS, _LEADS),
    "dropout_eval": (_normalised(lambda x: F.dropout(x, 0.5, False)), _FLOATS, _LEADS),
    "alpha_dropout": (_normalised(lambda x: F.alpha_dropout(x, 0.5, True)), _FLOATS, _LEADS),
    "pow": (_normalised(lambda x: x**2.0), _FLOATS, _LEADS),
    "rpow": (_normalised(lambda x: 2.0**x), _FLOATS, _LEADS),
    "clip": (_normalised(lambda x: lucid.clip(x, 0.2, 0.5)), _FLOATS, _LEADS),
    "nan_to_num": (_normalised(lucid.nan_to_num), _FLOATS, _LEADS),
    "astype": (_normalised(lambda x: x.to(dtype=lucid.float32)), _FLOATS, _LEADS),
    "permute": (_normalised(lambda x: x.permute(*reversed(range(x.ndim)))), _FLOATS, _LEADS),
    "reshape": (_normalised(lambda x: x.reshape(-1)), _FLOATS, _LEADS),
    "contiguous": (_normalised(lambda x: x.contiguous()), _FLOATS, _LEADS),
    "embedding": (_embedding, _FLOATS, ((3,), (2, 3), (0,))),
    **{name: (_activation(name), _FLOATS, _LEADS) for name in _ACTIVATIONS},
    "batch_norm": (_image_norm("batch_norm"), _FLOATS, _IMAGES),
    "batch_norm_running": (_image_norm("batch_norm_running"), _FLOATS, _IMAGES),
    "batch_norm_eval": (_image_norm("batch_norm_eval"), _FLOATS, _IMAGES),
    "group_norm": (_image_norm("group_norm"), _FLOATS, _IMAGES),
    "instance_norm": (_image_norm("instance_norm"), _FLOATS, _IMAGES),
    "conv": (_conv, _FLOATS, _IMAGES),
    "conv_transpose": (_conv_transpose, _FLOATS, _IMAGES),
    "max_pool": (_pool("max"), _FLOATS, _IMAGES),
    "avg_pool": (_pool("avg"), _FLOATS, _IMAGES),
    "interpolate": (_interpolate, _FLOATS, _IMAGES),
    "interpolate_nearest": (_nearest, _FLOATS, _IMAGES),
    **{name: (_loss(name), _FLOATS, ((4, 5), (0, 5), (3,))) for name in _LOSSES},
    "cross_entropy": (_classification("cross_entropy"), _FLOATS, ((4, 5), (0, 5))),
    "nll_loss": (_classification("nll_loss"), _FLOATS, ((4, 5), (0, 5))),
    "attention": (_attention, _FLOATS, ((2, 3, 4), (2, 0, 4))),
    "lstm": (_lstm, _FLOATS, ((2, 3, 4), (2, 1, 4))),
    "scatter_add": (_scatter("scatter_add"), _FLOATS, ((3, 4), (3,))),
    "scatter": (_scatter("scatter"), _FLOATS, ((3, 4), (3,))),
    **{
        name: (_factory(name), (*_FLOATS, lucid.int32), _ALL_SHAPES)
        for name in ("zeros", "ones", "full", "empty")
    },
    **{name: (_factory(name), _FLOATS, _ALL_SHAPES) for name in ("rand", "randn")},
}


@pytest.mark.parametrize(("name", "build", "dtype", "shape"), _cases(_LAYERS))
def test_a_layer_forward_result_and_its_gradients_have_matching_storage(
    device: str, name: str, build: Build, dtype: lucid.dtype, shape: Shape
) -> None:
    for t in _tensors(build(device, dtype, shape)):
        _consistent(t, name)
