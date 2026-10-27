"""Regression tests: ``cast(Tensor, ...)`` where ``Tensor`` is not defined.

Found 2026-08-02 by the audit's depth probe — the pass that asks not "is
this op correct" but "can the harness call it at all".

``typing.cast`` evaluates its first argument at run time even though it
does nothing with it, and these modules import ``Tensor`` only under
``TYPE_CHECKING``.  Every one of the four sites therefore raised
``NameError: name 'Tensor' is not defined`` the moment it was reached.
Two of them are on paths a caller always takes:

* ``lucid.float_power(x, y)`` — the final ``return``, so the function had
  **never worked for any input**;
* ``x.to(other_tensor)`` — the "match another tensor's device and dtype"
  form of ``.to``, while ``.to("cpu")`` and ``.to(dtype)`` were fine.

Neither was covered by a test, which is how a public function could be
entirely broken and stay that way.  The fix is the string form the
quantized and QAT modules already use in 46 places; the guard below is
static rather than behavioural, so it catches the next one before it
ships rather than after someone calls it.
"""

import ast
import importlib
import pathlib

import numpy as np
import pytest

import lucid

# ── the two APIs that were dead ──────────────────────────────────────────────


def test_float_power_runs_at_all() -> None:
    """It raised NameError for every input, in every form."""
    x = lucid.tensor(np.array([2.0, 3.0], dtype=np.float32))
    assert np.allclose(lucid.float_power(x, 2.0).numpy(), [4.0, 9.0])
    assert np.allclose(lucid.float_power(x, x).numpy(), [4.0, 27.0])
    assert np.allclose(lucid.float_power(2.0, x).numpy(), [4.0, 8.0])


def test_float_power_promotes_to_float64() -> None:
    """The reason the op exists: no integer overflow, no domain error."""
    x = lucid.tensor(np.array([-8.0], dtype=np.float32))
    out = lucid.float_power(x, 2.0)
    assert str(out.dtype).endswith("float64")
    assert np.allclose(out.numpy(), [64.0])


def test_to_accepts_another_tensor() -> None:
    """``x.to(other)`` copies the other tensor's device and dtype."""
    x = lucid.tensor(np.array([1.0, 2.0], dtype=np.float32))
    reference = lucid.tensor(np.array([1.0]), dtype=lucid.float64)
    moved = x.to(reference)
    assert str(moved.dtype) == str(reference.dtype)
    assert str(moved.device) == str(reference.device)
    assert np.allclose(moved.numpy(), [1.0, 2.0])


@pytest.mark.parametrize("target", ["cpu", lucid.float64, lucid.float32])
def test_the_other_to_forms_still_work(target: object) -> None:
    """These were never broken; pinned so the fix cannot regress them."""
    x = lucid.tensor(np.array([1.0, 2.0], dtype=np.float32))
    assert x.to(target) is not None  # type: ignore[arg-type]


def test_bernoulli_inplace_accepts_a_probability_tensor() -> None:
    """``x.bernoulli_(p_tensor)`` must not reference a TYPE_CHECKING-only ``Tensor``."""
    x = lucid.zeros(4)
    p = lucid.ones(4)
    out = x.bernoulli_(p)
    assert out is x
    assert np.allclose(x.numpy(), [1.0, 1.0, 1.0, 1.0])


# ── the static guard ─────────────────────────────────────────────────────────


def _type_checking_only(tree: ast.Module) -> set[str]:
    """Names this module imports under ``TYPE_CHECKING`` and nowhere else."""
    deferred: set[str] = set()
    at_runtime: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.If) and "TYPE_CHECKING" in ast.unparse(node.test):
            for sub in ast.walk(node):
                if isinstance(sub, ast.ImportFrom):
                    deferred.update(a.asname or a.name for a in sub.names)
    for node in tree.body:
        if isinstance(node, ast.ImportFrom):
            at_runtime.update(a.asname or a.name for a in node.names)
    return deferred - at_runtime


def _locally_imported(function: ast.AST) -> set[str]:
    names: set[str] = set()
    for sub in ast.walk(function):
        if isinstance(sub, ast.ImportFrom):
            names.update(a.asname or a.name for a in sub.names)
    return names


def _unsafe_sites(path: pathlib.Path) -> list[str]:
    """``cast``/``isinstance`` sites naming a type that is not bound at run time."""
    try:
        tree = ast.parse(path.read_text())
    except SyntaxError:
        return []
    risky = _type_checking_only(tree)
    if not risky:
        return []
    functions = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef)]
    found: list[str] = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)):
            continue
        if node.func.id not in ("cast", "isinstance"):
            continue
        # cast() evaluates its first argument; isinstance() its second.
        arguments = node.args[:1] if node.func.id == "cast" else node.args[1:]
        used = {
            sub.id for arg in arguments for sub in ast.walk(arg) if isinstance(sub, ast.Name)
        } & risky
        if not used:
            continue
        enclosing = [f for f in functions if f.lineno <= node.lineno <= (f.end_lineno or f.lineno)]
        if any(used <= _locally_imported(f) for f in enclosing):
            continue  # a function-local import binds it
        found.append(f"{path}:{node.lineno} {node.func.id}({sorted(used)[0]}, ...)")
    return found


def test_no_runtime_use_of_a_type_checking_only_name() -> None:
    """The guard the four defects needed.

    ``cast`` and ``isinstance`` both evaluate a type argument at run time.
    Naming something that only exists for the type checker is a latent
    ``NameError`` on whichever branch reaches it — invisible until a
    caller does.
    """
    root = pathlib.Path(lucid.__file__).parent
    offenders: list[str] = []
    for path in sorted(root.rglob("*.py")):
        if "/test/" in str(path):
            continue
        offenders.extend(_unsafe_sites(path))
    assert not offenders, "runtime use of a TYPE_CHECKING-only name:\n  " + "\n  ".join(offenders)


def test_the_guard_can_actually_fail(tmp_path: pathlib.Path) -> None:
    """Guard the instrument: a check that cannot fail proves nothing."""
    bad = tmp_path / "offender.py"
    bad.write_text(
        "from typing import TYPE_CHECKING, cast\n"
        "if TYPE_CHECKING:\n"
        "    from lucid._tensor.tensor import Tensor\n"
        "def f(x):\n"
        "    return cast(Tensor, x)\n"
    )
    assert _unsafe_sites(bad), "the guard failed to flag a known offender"

    good = tmp_path / "clean.py"
    good.write_text(
        "from typing import TYPE_CHECKING, cast\n"
        "if TYPE_CHECKING:\n"
        "    from lucid._tensor.tensor import Tensor\n"
        "def f(x):\n"
        '    return cast("Tensor", x)\n'
    )
    assert not _unsafe_sites(good)


def test_every_module_still_imports() -> None:
    """A NameError at import time would be louder, but check anyway."""
    for name in ("lucid._ops.composite.elementwise", "lucid._tensor._to", "lucid._tensor._methods"):
        assert importlib.import_module(name) is not None


class TestWholeTensorAssignmentKeepsGrad:
    """``x[:] = v`` must not quietly demote a parameter.

    The whole-slice fast path rebinds ``_impl`` outright instead of
    scattering into the existing one, and ``requires_grad`` rides on the
    impl — so the flag went with it. Nothing about the object looked
    wrong afterwards: still a Parameter, still a leaf, still in
    ``parameters()`` and ``state_dict()``, and never trained again. The
    general path (``x[0] = v``) always kept the flag, which is what the
    fast path now matches.

    The assignments sit under ``no_grad`` — the way a parameter is
    overwritten.  Outside it, writing into a leaf that requires grad is
    refused, as the reference framework refuses it; before, the write went
    through and turned the parameter into a non-leaf that never received
    ``.grad``.
    """

    def test_slice_assignment_preserves_requires_grad(self) -> None:
        p = lucid.nn.Linear(4, 4).weight
        with lucid.no_grad():
            p[:] = lucid.randn((4, 4))
        assert p.requires_grad and p.is_leaf

    def test_ellipsis_assignment_preserves_requires_grad(self) -> None:
        p = lucid.nn.Linear(4, 4).weight
        with lucid.no_grad():
            p[...] = lucid.zeros((4, 4))
        assert p.requires_grad and p.is_leaf

    def test_scalar_assignment_preserves_requires_grad(self) -> None:
        p = lucid.nn.Linear(4, 4).weight
        with lucid.no_grad():
            p[:] = 0.5
        assert p.requires_grad and p.is_leaf

    def test_the_parameter_still_trains(self) -> None:
        layer = lucid.nn.Linear(4, 4)
        with lucid.no_grad():
            layer.weight[:] = lucid.randn((4, 4))
        layer(lucid.randn((3, 4))).sum().backward()
        assert layer.weight.grad is not None
        assert float(abs(layer.weight.grad).sum()) > 0.0

    def test_assignment_outside_no_grad_is_refused(self) -> None:
        layer = lucid.nn.Linear(4, 4)
        layer(lucid.randn((3, 4)))  # a used parameter is still a leaf
        for write in (
            lambda p: p.__setitem__(slice(None), 0.0),
            lambda p: p.__setitem__(0, lucid.randn((4,))),
        ):
            with pytest.raises(RuntimeError, match="leaf tensor that requires grad"):
                write(layer.weight)
        assert layer.weight.is_leaf

    def test_a_plain_tensor_is_unaffected(self) -> None:
        x = lucid.randn((3, 3))
        x[:] = lucid.ones((3, 3))
        assert not x.requires_grad
        assert float((x - 1.0).abs().max().item()) == 0.0

    def test_the_general_path_still_agrees(self) -> None:
        p = lucid.nn.Linear(4, 4).weight
        with lucid.no_grad():
            p[0] = lucid.randn((4,))
        assert p.requires_grad
