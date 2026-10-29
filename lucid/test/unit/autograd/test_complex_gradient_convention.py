"""Complex gradients follow the conjugate convention for every op (LCD-661).

For a real loss L of z = x + iy, ``z.grad`` is dL/dx + i dL/dy, as in the
reference.  Through a holomorphic w = f(z) that is grad_w * conj(f'(z)).  The
engine's backward formulas are written as the real chain rule,
grad_w * f'(z), and the kernel bases conjugate around them
(``lucid/_C/autograd/ComplexGrad.h``).  Before that only ``mul`` and ``div``
conjugated by hand: ``exp``, ``log``, ``sqrt``, ``sin``, ``pow``, ``matmul``,
``linear``, ``dot``, ``outer`` … returned g * f'(z) for a complex input, with
the forward correct and no error.

Two guards:

* closed forms for a few ops, which need nothing but NumPy;
* a sweep over every op of the op registry and every public function of
  ``lucid``, ``lucid.linalg``, ``lucid.special`` and ``lucid.nn.functional``
  that takes one or two tensors, against the reference wherever both accept a
  complex input — first and second derivatives.  A coverage floor keeps the
  sweep from passing by refusing everything.
"""

import annotationlib
from collections.abc import Callable, Sequence
from dataclasses import dataclass
import functools
import inspect
import re
from types import ModuleType

import numpy as np
import pytest

import lucid
from lucid._ops._registry import _REGISTRY
from lucid._tensor.tensor import Tensor
import lucid.nn.functional as F
from lucid.test._fixtures.devices import metal_available

_METAL = pytest.mark.skipif(not metal_available(), reason="needs Metal")

_Z = np.array([1 + 2j, -0.5 + 0.3j, 0.7 - 1.1j])
_W = np.array([0.6 - 0.4j, 1.3 + 0.2j, -0.8 + 0.9j])
_A = np.array([[1 + 2j, -0.5 + 0.3j], [0.7 - 1.1j, 0.2 + 0.4j]])
_B = np.array([[0.3 - 0.4j, 1.1 + 0.2j], [-0.6 + 0.5j, 0.9 - 0.7j]])

# No point sits on a branch cut: every one has a non-zero real and imaginary part.
_SAMPLES: dict[int, tuple[tuple[np.ndarray, ...], ...]] = {
    1: ((_Z,), (_A,)),
    2: ((_Z, _W), (_A, _B), (_A[None], _B[None])),
}

#: What "accepts a complex input" fails with when it does not.
_REFUSALS = (ArithmeticError, AttributeError, LookupError, RuntimeError, TypeError, ValueError)


# ── closed forms ──────────────────────────────────────────────────────────────

#: name -> (lucid f, NumPy f, NumPy f').  L = |f(z)|^2 gives z.grad = 2 f conj(f').
_HOLOMORPHIC: dict[
    str, tuple[Callable[..., Tensor], Callable[..., np.ndarray], Callable[..., np.ndarray]]
] = {
    "exp": (lucid.exp, np.exp, np.exp),
    "log": (lucid.log, np.log, lambda z: 1 / z),
    "sqrt": (lucid.sqrt, np.sqrt, lambda z: 0.5 / np.sqrt(z)),
    "sin": (lucid.sin, np.sin, np.cos),
    "tanh": (lucid.tanh, np.tanh, lambda z: 1 - np.tanh(z) ** 2),
    "asinh": (lucid.asinh, np.arcsinh, lambda z: 1 / np.sqrt(z * z + 1)),
    "z**3": (lambda z: z**3, lambda z: z**3, lambda z: 3 * z**2),
    "2**z": (lambda z: 2**z, lambda z: 2**z, lambda z: np.log(2) * 2**z),
}


def _placements(cpu_accepts: bool) -> list[object]:
    cpu_marks = (
        ()
        if cpu_accepts
        else pytest.mark.xfail(
            strict=True,
            raises=NotImplementedError,
            reason="LCD-667: the CPU stream has no complex kernel for this op",
        )
    )
    return [
        pytest.param("cpu", id="cpu", marks=cpu_marks),
        pytest.param("metal", id="metal", marks=_METAL),
    ]


def _leaf(values: np.ndarray, device: str) -> Tensor:
    return lucid.tensor(values, dtype=lucid.complex64, device=device).requires_grad_(True)


def _norm_sq(y: Tensor) -> Tensor:
    return lucid.real(y * lucid.conj(y)).sum()


@pytest.mark.parametrize("device", _placements(cpu_accepts=False))
@pytest.mark.parametrize("name", list(_HOLOMORPHIC))
def test_a_holomorphic_unary_conjugates_its_derivative(name: str, device: str) -> None:
    f, f_np, df_np = _HOLOMORPHIC[name]
    z = _leaf(_Z, device)
    _norm_sq(f(z)).backward()
    want = 2 * f_np(_Z) * np.conj(df_np(_Z))
    np.testing.assert_allclose(z.grad.numpy(), want, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("device", _placements(cpu_accepts=False))
def test_pow_conjugates_both_partial_derivatives(device: str) -> None:
    a, b = _leaf(_Z, device), _leaf(_W, device)
    _norm_sq(a**b).backward()
    w = _Z**_W
    np.testing.assert_allclose(
        a.grad.numpy(), 2 * w * np.conj(_W * _Z ** (_W - 1)), rtol=1e-4, atol=1e-5
    )
    np.testing.assert_allclose(
        b.grad.numpy(), 2 * w * np.conj(np.log(_Z) * w), rtol=1e-4, atol=1e-5
    )


@pytest.mark.parametrize("device", _placements(cpu_accepts=False))
def test_matmul_takes_the_conjugate_transpose(device: str) -> None:
    a, b = _leaf(_A, device), _leaf(_B, device)
    _norm_sq(a @ b).backward()
    c = _A @ _B
    np.testing.assert_allclose(a.grad.numpy(), 2 * c @ _B.conj().T, rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(b.grad.numpy(), 2 * _A.conj().T @ c, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("device", _placements(cpu_accepts=True))
def test_dot_conjugates_the_other_operand(device: str) -> None:
    a, b = _leaf(_Z, device), _leaf(_W, device)
    _norm_sq(a.dot(b)).backward()
    w = _Z @ _W
    np.testing.assert_allclose(a.grad.numpy(), 2 * w * _W.conj(), rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(b.grad.numpy(), 2 * w * _Z.conj(), rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("device", _placements(cpu_accepts=False))
def test_outer_conjugates_the_other_operand(device: str) -> None:
    a, b = _leaf(_Z, device), _leaf(_W[:2], device)
    _norm_sq(a.outer(b)).backward()
    c = np.outer(_Z, _W[:2])
    np.testing.assert_allclose(a.grad.numpy(), 2 * c @ _W[:2].conj(), rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(b.grad.numpy(), 2 * c.T @ _Z.conj(), rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("device", [pytest.param("metal", marks=_METAL)])
def test_a_real_derivative_is_left_alone(device: str) -> None:
    """add / sub skip the conjugation (their derivatives are real); the
    result must be the same as for the ops that do not."""
    a, b = _leaf(_Z, device), _leaf(_W, device)
    _norm_sq((a + b) * (a - b)).backward()
    c = (_Z + _W) * (_Z - _W)
    np.testing.assert_allclose(a.grad.numpy(), 2 * c * np.conj(2 * _Z), rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(b.grad.numpy(), 2 * c * np.conj(-2 * _W), rtol=1e-4, atol=1e-5)


# ── the sweep ────────────────────────────────────────────────────────────────

_STOCHASTIC = re.compile(
    r"dropout|^rand|bernoulli|^poisson$|multinomial|^normal$|gumbel|rrelu|fractional"
)

#: label -> why it differs from the reference.  Strict: an entry that starts
#: matching fails the sweep until it is removed.
_KNOWN: dict[tuple[str, str], str] = {
    ("cpu", "lucid.float_power"): "float_power drops the imaginary part (works in float64)"
}

#: Must be compared on Metal — a sweep that skipped them would pass vacuously.
_FIRST_ORDER_FLOOR = frozenset(
    {
        *(f"_ops.{n}" for n in ("exp", "log", "log1p", "sqrt", "rsqrt", "square", "reciprocal")),
        *(f"_ops.{n}" for n in ("sin", "cos", "tan", "sinh", "cosh", "tanh", "atan")),
        *(f"_ops.{n}" for n in ("pow", "mul", "div", "matmul", "dot", "outer")),
        "lucid.asinh",
        "lucid.expm1",
        "lucid.ger",
        "nn.functional.linear",
    }
)
_SECOND_ORDER_FLOOR = frozenset(
    f"_ops.{n}"
    for n in ("exp", "log", "sqrt", "sin", "tanh", "pow", "mul", "div", "matmul", "dot", "outer")
) | {"nn.functional.linear"}


@dataclass(frozen=True)
class _Op:
    label: str
    arity: int
    fn: Callable[..., object]
    ref_path: tuple[str, ...]
    method: str | None

    def reference(self, ref: ModuleType) -> Callable[..., object] | None:
        target: object = ref
        for part in self.ref_path:
            target = getattr(target, part, None)
        if callable(target) and not inspect.ismodule(target):
            return target
        return None if self.method is None else functools.partial(_call_method, self.method)


def _call_method(name: str, *xs: object) -> object:
    return getattr(xs[0], name)(*xs[1:])


def _registry_ops() -> list[_Op]:
    ops = []
    for e in _REGISTRY:
        if e.inplace or not e.returns_tensor or e.n_tensor_args not in (1, 2):
            continue
        if e.free_fn_name is not None:
            fn: Callable[..., object] = getattr(lucid._ops, e.free_fn_name)
        else:
            fn = functools.partial(_call_method, e.method_name)
        name = e.free_fn_name or e.name
        ops.append(_Op(f"_ops.{name}", e.n_tensor_args, fn, (name,), e.method_name))
    return ops


def _takes_a_tensor(annotation: object) -> bool:
    text = (
        annotation.__forward_arg__
        if isinstance(annotation, annotationlib.ForwardRef)
        else str(annotation)
    )
    return "Tensor" in text and not re.search(r"Sequence|Iterable|list|tuple|dict", text)


def _tensor_arity(fn: Callable[..., object]) -> int | None:
    try:
        sig = inspect.signature(fn, annotation_format=annotationlib.Format.FORWARDREF)
    except (TypeError, ValueError):
        return None
    params = sig.parameters.values()
    if any(p.kind is p.VAR_POSITIONAL for p in params):
        return None
    required = [p for p in params if p.default is p.empty and p.kind is not p.KEYWORD_ONLY]
    return len(required) if all(_takes_a_tensor(p.annotation) for p in required) else None


def _namespace_ops() -> list[_Op]:
    ops = []
    for module, path in (
        (lucid, ()),
        (lucid.linalg, ("linalg",)),
        (lucid.special, ("special",)),
        (F, ("nn", "functional")),
    ):
        prefix = ".".join(path) or "lucid"
        for name in getattr(module, "__all__", ()):
            fn = getattr(module, name, None)
            if name.endswith("_") or not callable(fn) or inspect.isclass(fn):
                continue
            arity = _tensor_arity(fn)
            if arity in (1, 2):
                ops.append(_Op(f"{prefix}.{name}", arity, fn, (*path, name), None))
    return ops


def _sweep_ops() -> list[_Op]:
    ops = [
        op
        for op in (*_registry_ops(), *_namespace_ops())
        if not _STOCHASTIC.search(op.label.rsplit(".", 1)[-1])
    ]
    return sorted(ops, key=lambda op: op.label)


_OPS = _sweep_ops()


def _first_tensor(y: object, tensor_type: type) -> object | None:
    if isinstance(y, (tuple, list)):
        y = y[0] if y else None
    if not isinstance(y, tensor_type) or not (y.is_complex() or y.is_floating_point()):
        return None
    return y


def _loss(mod: ModuleType, y: object, kind: str) -> object:
    if not y.is_complex():
        return y.sum() if kind == "abs" else (y * y).sum()
    return mod.abs(y).sum() if kind == "abs" else mod.real(y * mod.conj(y)).sum()


@dataclass(frozen=True)
class _Side:
    """One framework: how to make leaves, run autograd, and read a tensor out."""

    mod: ModuleType
    tensor_type: type
    leaves: Callable[[Sequence[np.ndarray]], list[object]]
    numpy: Callable[[object], np.ndarray]

    def grads(
        self, fn: Callable[..., object], arrays: Sequence[np.ndarray], order: int
    ) -> list[np.ndarray] | None:
        out = []
        for kind in ("abs", "norm_sq"):
            xs = self.leaves(arrays)
            y = _first_tensor(fn(*xs), self.tensor_type)
            if y is None or not y.requires_grad:
                return None
            gs = self.mod.autograd.grad(
                _loss(self.mod, y, kind), xs, create_graph=order == 2, allow_unused=True
            )
            if order == 2:
                gs = self._second(xs, gs)
            out.extend(
                np.zeros(a.shape, np.complex128) if g is None else self.numpy(g)
                for a, g in zip(arrays, gs, strict=True)
            )
        return out

    def _second(self, xs: list[object], gs: Sequence[object]) -> Sequence[object]:
        live = [g for g in gs if g is not None]
        total = sum(
            self.mod.real(g * self.mod.conj(g)).sum() + self.mod.imag(g).sum() for g in live
        )
        return self.mod.autograd.grad(total, xs, allow_unused=True)


def _lucid_side(device: str, dtype: object) -> _Side:
    return _Side(
        lucid,
        Tensor,
        lambda arrays: [
            lucid.tensor(a, dtype=dtype, device=device).requires_grad_(True) for a in arrays
        ],
        lambda t: np.asarray(t.detach().numpy(), np.complex128),
    )


def _ref_side(ref: ModuleType) -> _Side:
    return _Side(
        ref,
        ref.Tensor,
        lambda arrays: [ref.tensor(a.astype(np.complex128)).requires_grad_(True) for a in arrays],
        lambda t: np.asarray(t.detach().resolve_conj().resolve_neg().numpy(), np.complex128),
    )


def _try(
    side: _Side, fn: Callable[..., object], arrays: Sequence[np.ndarray], order: int
) -> list[np.ndarray] | None:
    try:
        return side.grads(fn, arrays, order)
    except _REFUSALS:
        return None


def _compare(
    op: _Op, ref: ModuleType, lucid_side: _Side, order: int, tol: dict[str, float]
) -> str | None:
    """``"compared"``, ``None`` (nothing both accept), or what differed."""
    ref_fn = op.reference(ref)
    if ref_fn is None:
        return None
    verdict = None
    for arrays in _SAMPLES[op.arity]:
        want = _try(_ref_side(ref), ref_fn, arrays, order)
        got = None if want is None else _try(lucid_side, op.fn, arrays, order)
        if got is None:
            continue
        for k, (g, w) in enumerate(zip(got, want, strict=True)):
            if not np.allclose(g, w, **tol):
                return f"sample {arrays[0].shape} grad {k}: lucid {np.round(g.ravel()[:2], 4)} ref {np.round(w.ravel()[:2], 4)}"
        verdict = "compared"
    return verdict


_SWEEP_PLACEMENTS = [
    pytest.param("metal", "complex64", id="metal-complex64", marks=_METAL),
    pytest.param("cpu", "complex64", id="cpu-complex64"),
    pytest.param("cpu", "complex128", id="cpu-complex128"),
]
_TOL = {"complex64": {"rtol": 3e-3, "atol": 3e-4}, "complex128": {"rtol": 1e-6, "atol": 1e-8}}


def _sweep(ref: ModuleType, device: str, dtype: str, order: int) -> tuple[set[str], list[str]]:
    lucid_side = _lucid_side(device, getattr(lucid, dtype))
    compared: set[str] = set()
    problems: list[str] = []
    for op in _OPS:
        verdict = _compare(op, ref, lucid_side, order, _TOL[dtype])
        known = _KNOWN.get((device, op.label))
        if verdict == "compared":
            compared.add(op.label)
            if known is not None:
                problems.append(f"{op.label}: matches now — remove it from _KNOWN ({known})")
        elif verdict is not None and known is None:
            problems.append(f"{op.label}: {verdict}")
    return compared, problems


@pytest.mark.parity
@pytest.mark.parametrize(("device", "dtype"), _SWEEP_PLACEMENTS)
def test_every_first_derivative_matches_the_reference(
    device: str, dtype: str, ref: ModuleType
) -> None:
    compared, problems = _sweep(ref, device, dtype, order=1)
    assert not problems, "\n".join(problems)
    if device == "metal":
        assert _FIRST_ORDER_FLOOR <= compared, sorted(_FIRST_ORDER_FLOOR - compared)


@pytest.mark.parity
@pytest.mark.parametrize(("device", "dtype"), _SWEEP_PLACEMENTS)
def test_every_second_derivative_matches_the_reference(
    device: str, dtype: str, ref: ModuleType
) -> None:
    compared, problems = _sweep(ref, device, dtype, order=2)
    assert not problems, "\n".join(problems)
    if device == "metal":
        assert _SECOND_ORDER_FLOOR <= compared, sorted(_SECOND_ORDER_FLOOR - compared)
