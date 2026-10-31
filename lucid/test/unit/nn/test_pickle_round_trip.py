"""A pickled tensor or module comes back as itself (LCD-792).

``Tensor.__reduce__`` rebuilt every instance as a plain ``Tensor``, so an
unpickled ``Module`` held plain Tensors in ``_parameters``: ``isinstance(p,
Parameter)`` failed, ``prune.remove`` raised ``TypeError``, and the next
attribute rebind evicted the weight from the registry.

The defect class is "a ``Tensor`` subclass does not survive the pickle
boundary as its own type".  The guards below are enumerated, not hand-listed:
every subclass found by walking ``Tensor.__subclasses__()``, and every module
the audit can construct (the whole module surface plus the hand-written
``BUILDERS``).
"""

import copy
import pickle
import subprocess
import sys

import numpy as np
import pytest

import lucid
import lucid.nn as nn
from lucid.nn.utils import prune
from lucid.nn.parameter import Buffer, Parameter, UninitializedParameter


def _all_tensor_subclasses() -> list[type[lucid.Tensor]]:
    found: list[type[lucid.Tensor]] = []
    todo = [lucid.Tensor]
    while todo:
        for sub in todo.pop().__subclasses__():
            if sub not in found:
                found.append(sub)
                todo.append(sub)
    return found


_SUBCLASSES = _all_tensor_subclasses()


class _Tagged(lucid.Tensor):
    """A user subclass: it needs no hook to round-trip."""


def test_the_enumeration_sees_the_framework_subclasses() -> None:
    assert {Parameter, Buffer, UninitializedParameter} <= set(_SUBCLASSES)


@pytest.mark.parametrize("cls", _SUBCLASSES, ids=lambda c: c.__qualname__)
@pytest.mark.parametrize("requires_grad", [True, False])
def test_every_tensor_subclass_round_trips_as_itself(cls, requires_grad) -> None:
    src = lucid.arange(6, dtype=lucid.float32).reshape(2, 3)
    t = cls.__new_from_impl__(src.requires_grad_(requires_grad)._impl)
    t.__dict__["extra"] = {"tag": 7}
    back = pickle.loads(pickle.dumps(t))
    assert type(back) is cls
    assert back.requires_grad is requires_grad
    assert back.dtype == t.dtype and tuple(back.shape) == (2, 3)
    assert back.__dict__["extra"] == {"tag": 7}
    np.testing.assert_array_equal(back.detach().numpy(), t.detach().numpy())
    assert back._impl is not t._impl


def test_a_plain_tensor_is_still_a_plain_tensor() -> None:
    t = lucid.ones(2, 2, requires_grad=True)
    back = pickle.loads(pickle.dumps(t))
    assert type(back) is lucid.Tensor and back.requires_grad


def test_the_rebuild_hook_is_the_one_override_point() -> None:
    class _Counted(lucid.Tensor):
        built = 0

        @classmethod
        def _from_pickle_impl(cls, impl):
            cls.built += 1
            return super()._from_pickle_impl(impl)

    globals()["_Counted"] = _Counted
    _Counted.__qualname__ = "_Counted"
    t = _Counted.__new_from_impl__(lucid.ones(3)._impl)
    back = pickle.loads(pickle.dumps(t))
    assert type(back) is _Counted and _Counted.built == 1


def test_a_buffer_keeps_its_persistence_through_pickle_and_deepcopy() -> None:
    b = Buffer(lucid.ones(2), persistent=False)
    for back in (pickle.loads(pickle.dumps(b)), copy.deepcopy(b)):
        assert type(back) is Buffer and back.persistent is False


def test_a_deep_copied_parameter_is_unchanged_by_the_pickle_path() -> None:
    p = Parameter(lucid.ones(2), requires_grad=False)
    p.grad = None
    q = copy.deepcopy(p)
    assert type(q) is Parameter and q.requires_grad is False and q._impl is not p._impl


def test_an_unpickled_module_can_be_pruned_and_made_permanent() -> None:
    m = pickle.loads(pickle.dumps(nn.Linear(4, 3)))
    assert all(type(p) is Parameter for p in m._parameters.values())
    prune.l1_unstructured(m, "weight", amount=0.5)
    prune.remove(m, "weight")
    assert type(m.weight) is Parameter


def test_an_unpickled_module_keeps_a_rebound_parameter_registered() -> None:
    m = pickle.loads(pickle.dumps(nn.Linear(4, 3)))
    w = m.weight
    m.weight = w
    assert "weight" in m._parameters


_SWEEP = """
import pickle, warnings
warnings.simplefilter("ignore")
import numpy as np
import lucid
from lucid.test.audit import _surface
from lucid.test.audit._axes import _forward_inputs, _module_input_shapes, _try_construct
from lucid.test.audit._builders import BUILDERS


def tensors(out):
    if hasattr(out, "shape"):
        return [out]
    if isinstance(out, (tuple, list)):
        return [t for o in out for t in tensors(o)]
    for attr in ("logits", "sample", "last_hidden_state"):
        got = getattr(out, attr, None)
        if got is not None:
            return tensors(got)
    return []


def working_args(module):
    for shape in _module_input_shapes(module):
        for args in _forward_inputs(module, shape):
            try:
                lucid.manual_seed(0)
                module(*args)
            except Exception:
                continue
            return args
    return None


def run(module, args):
    lucid.manual_seed(0)
    return tensors(module(*args))


def registries(module):
    for path, sub in module.named_modules():
        for kind in ("_parameters", "_buffers"):
            for name, t in getattr(sub, kind).items():
                yield f"{path}.{name}", kind, t


def check(name, module):
    problems = []
    try:
        copy_ = pickle.loads(pickle.dumps(module))
    except Exception as exc:
        return [f"{name}: does not pickle ({type(exc).__name__}: {str(exc)[:80]})"]
    ours, theirs = list(registries(module)), list(registries(copy_))
    if [(p, k) for p, k, _ in ours] != [(p, k) for p, k, _ in theirs]:
        return [f"{name}: registries differ"]
    for (path, kind, a), (_, _, b) in zip(ours, theirs, strict=True):
        if a is None or b is None:
            if a is not b:
                problems.append(f"{name}.{path}: None became a tensor or back")
            continue
        if type(a) is not type(b):
            problems.append(f"{name}.{path}: {type(a).__name__} became {type(b).__name__}")
        elif a.requires_grad != b.requires_grad:
            problems.append(f"{name}.{path}: requires_grad changed")
        elif tuple(a.shape) != tuple(b.shape) or a.dtype != b.dtype:
            problems.append(f"{name}.{path}: shape or dtype changed")
    if module._non_persistent_buffers != copy_._non_persistent_buffers:
        problems.append(f"{name}: non-persistent buffer set changed")
    if problems:
        return problems
    module.eval()
    copy_.eval()
    args = working_args(module)
    if args is not None:
        try:
            for a, b in zip(run(module, args), run(copy_, args), strict=True):
                if not np.array_equal(a.detach().numpy(), b.detach().numpy(), equal_nan=True):
                    problems.append(f"{name}: output differs after the round trip")
        except Exception as exc:
            problems.append(f"{name}: copy fails where the original runs ({type(exc).__name__})")
    return problems


modules = [(s.qualname, _try_construct(s.obj)) for s in _surface.enumerate_surface() if s.kind == "module"]
modules += [(cls.__name__ + " (builder)", build()) for cls, build in BUILDERS.items()]
failed, checked = [], 0
for name, module in modules:
    if module is None:
        continue
    checked += 1
    failed += check(name, module)
print(checked)
for line in failed:
    print(line)
"""


def test_every_module_the_audit_builds_survives_a_pickle_round_trip() -> None:
    """Types, ``requires_grad`` and outputs, over the whole module surface.

    In a subprocess for the reason ``test_every_nn_module_class_can_be_built``
    gives: building every layer leaves global state behind.
    """
    done = subprocess.run(
        [sys.executable, "-W", "ignore", "-c", _SWEEP], capture_output=True, text=True, check=True
    )
    lines = [ln for ln in done.stdout.splitlines() if ln.strip()]
    assert int(lines[0]) > 300, f"only {lines[0]} modules were checked"
    assert not lines[1:], "\n".join(lines[1:])
