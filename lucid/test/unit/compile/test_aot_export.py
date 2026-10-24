"""AOT export: what ``save_compiled`` / ``load_compiled`` actually promise.

The pair had no test at all, and the one probe that touched it — the
audit's ``CompiledAxis`` — ran a closure over a single tensor.  One feed
has exactly one ordering, so the probe could not see the half of the
contract that was broken: a reloaded ``.mpsgraphpackage`` does not report
the feed order it expects, and that order only survives while every feed
shares a shape and dtype.  Anything else (any module, since weights and
activations rarely share a shape) used to be bound under the wrong slot's
shape, and MetalPerformanceShaders aborted the process — uncatchable, so
no assertion could have run afterwards.

What is pinned here:

* a uniform-feed artefact round-trips bit-exactly, and the feed *order*
  survives — checked with an expression no permutation leaves alone;
* an artefact whose feeds differ is refused in Python, never bound;
* a feed of the wrong shape or dtype raises before it reaches MPSGraph.

The last one is why these tests can exist: every assertion below is
reachable only because a mis-bound feed now raises instead of killing the
interpreter.
"""

import os

import pytest

import lucid
import lucid.compile as lc
import lucid.nn as nn
from lucid._C import engine as _C_engine
from lucid.test.unit.compile._helpers import COMPILE_DEVICE, metal_tensor, to_metal

# Compiled forwards under grad mode run eagerly (LCD-582); these test them.
pytestmark = [
    pytest.mark.skipif(
        not lucid.metal.is_available(),
        reason="the compile path emits MPSGraph; there is no CPU equivalent",
    ),
    pytest.mark.usefixtures("inference"),
]


def _saved(tmp_path: object, compiled: object, name: str) -> str:
    path = os.path.join(str(tmp_path), name)
    assert lc.save_compiled(compiled, path) is True
    assert os.path.exists(path + ".mpsgraphpackage")
    assert os.path.exists(path + ".meta")
    return path


class TestUniformFeedRoundTrip:
    def test_single_feed_matches_bit_for_bit(self, tmp_path: object) -> None:
        compiled = lc.compile(lambda a: a * 2.0 + 1.0)
        x = metal_tensor(3, 4)
        want = compiled(x)

        restored = lc.load_compiled(_saved(tmp_path, compiled, "one"))

        assert float((restored(x) - want).abs().max().item()) == 0.0

    def test_feed_order_survives(self, tmp_path: object) -> None:
        # Distinct weights per feed: any permutation of the three moves
        # the result, which a commutative expression would have hidden.
        compiled = lc.compile(lambda a, b, c: a * 1.0 + b * 10.0 + c * 100.0)
        feeds = tuple(metal_tensor(2, 8) for _ in range(3))
        want = compiled(*feeds)

        restored = lc.load_compiled(_saved(tmp_path, compiled, "three"))

        assert restored.num_inputs == 3
        assert float((restored(*feeds) - want).abs().max().item()) == 0.0

    def test_the_wrapper_publishes_its_feed_contract(self, tmp_path: object) -> None:
        compiled = lc.compile(lambda a, b: a * b + 1.0)
        x = metal_tensor(2, 8)
        y = metal_tensor(2, 8)
        compiled(x, y)

        restored = lc.load_compiled(_saved(tmp_path, compiled, "pair"))

        assert restored.input_shapes == [[2, 8], [2, 8]]
        assert restored.input_dtypes == ["float32", "float32"]


class TestUnbindableArtefactIsRefused:
    """A module's feeds differ in shape, so its artefact cannot be bound.

    Saving still succeeds — the package and the meta sidecar are written
    and are correct.  It is the load that has to say no.
    """

    def test_module_artefact_refuses_to_load(self, tmp_path: object) -> None:
        model = to_metal(nn.Linear(8, 4))
        compiled = lc.compile(model)
        compiled(metal_tensor(2, 8))

        path = _saved(tmp_path, compiled, "linear")

        with pytest.raises(RuntimeError, match="differing"):
            lc.load_compiled(path)

    def test_the_engine_loader_refuses_too(self, tmp_path: object) -> None:
        # The same loader backs ``LUCID_COMPILE_DISK_CACHE=1``, where a
        # refusal has to read as a cache miss rather than a crash.
        model = to_metal(nn.Linear(8, 4))
        compiled = lc.compile(model)
        compiled(metal_tensor(2, 8))

        path = _saved(tmp_path, compiled, "linear_engine")

        with pytest.raises(RuntimeError, match="feed order"):
            _C_engine.compile.load_executable(path)


class TestMisboundFeedRaises:
    """Every one of these used to abort the interpreter inside MPS."""

    @pytest.fixture
    def restored(self, tmp_path: object) -> object:
        compiled = lc.compile(lambda a, b: a - b)
        x = metal_tensor(2, 8)
        y = metal_tensor(2, 8)
        compiled(x, y)
        return lc.load_compiled(_saved(tmp_path, compiled, "sub"))

    def test_wrong_shape(self, restored: object) -> None:
        good = metal_tensor(2, 8)
        wrong = metal_tensor(4, 8)

        with pytest.raises(ValueError, match=r"feed slot 1 expects shape"):
            restored(good, wrong)

    def test_wrong_dtype(self, restored: object) -> None:
        good = metal_tensor(2, 8)
        wrong = metal_tensor(2, 8).half()

        with pytest.raises(ValueError, match=r"feed slot 1 expects dtype float32"):
            restored(good, wrong)

    def test_wrong_device(self, restored: object) -> None:
        good = metal_tensor(2, 8)

        with pytest.raises(ValueError, match="Device::GPU"):
            restored(good, lucid.randn(2, 8))

    def test_arity_error_names_every_slot(self, restored: object) -> None:
        good = metal_tensor(2, 8)

        with pytest.raises(ValueError, match=r"slot 1: shape \(2, 8\)"):
            restored(good)


class TestSaveRequiresAWarmCache:
    def test_uninvoked_module_cannot_be_saved(self, tmp_path: object) -> None:
        compiled = lc.compile(to_metal(nn.Linear(8, 4)))

        with pytest.raises(RuntimeError, match="no compiled entries"):
            lc.save_compiled(compiled, os.path.join(str(tmp_path), "cold"))


def test_compile_device_is_metal() -> None:
    # Guards the helper the rest of this file leans on.
    assert COMPILE_DEVICE == "metal"
