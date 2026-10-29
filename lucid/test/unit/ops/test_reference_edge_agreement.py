"""Edge cases where Lucid and the reference framework once disagreed (LCD-653).

Each test is one defect class held against the ``ref`` fixture:

* a DLPack export of a tensor that requires grad is refused (``BufferError``),
  whichever entry reaches it;
* ``dropout1d`` / ``dropout2d`` / ``dropout3d`` take the ranks the reference
  takes and draw their mask over the channel axes it draws it over -- an
  input with no batch axis has its channels on axis 0;
* a scaled or complementary special function has a first derivative of ``-0.0``
  (``erfc``) or ``0`` (``i0e`` / ``i1e``) at ``±inf``, and the Bessel ones a
  second derivative of ``0`` there, where differentiating the formula meets
  ``0 * inf``; ``i0e`` keeps its curvature at ``0``.
"""

import math
import warnings
from collections.abc import Callable
from types import ModuleType

import pytest

import lucid
import lucid.nn.functional as F
import lucid.special as S


def _outcome(call: Callable[[], object]) -> str:
    """``"ok"``, ``"BufferError"``, or ``"error"`` for any other refusal ``call`` raised."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            call()
    except BufferError:
        return "BufferError"
    except (RuntimeError, ValueError, TypeError):
        return "error"
    return "ok"


# ── DLPack ────────────────────────────────────────────────────────────────────


@pytest.mark.parity
@pytest.mark.parametrize("entry", ["from_dlpack", "to_dlpack", "__dlpack__"])
@pytest.mark.parametrize("requires_grad", [True, False])
def test_a_dlpack_export_of_a_tensor_requiring_grad_is_refused(
    entry: str, requires_grad: bool, device: str, ref: ModuleType
) -> None:
    x = lucid.ones(3, device=device, requires_grad=requires_grad)
    entries: dict[str, Callable[[lucid.Tensor], object]] = {
        "from_dlpack": lucid.from_dlpack,
        "to_dlpack": lucid.to_dlpack,
        "__dlpack__": lambda t: t.__dlpack__(),
    }
    want = _outcome(lambda: ref.from_dlpack(ref.ones(3, requires_grad=requires_grad)))
    assert want == ("BufferError" if requires_grad else "ok")
    assert _outcome(lambda: entries[entry](x)) == want


# ── dropout1d / 2d / 3d ───────────────────────────────────────────────────────


def _shape(ndim: int) -> tuple[int, ...]:
    """Wide on the two leading axes, so a channel mask is distinguishable."""
    return (24,) * min(ndim, 2) + (2,) * max(ndim - 2, 0)


def _varying_axes(fw: ModuleType, y: object, ndim: int) -> set[int]:
    """The axes along which ``y`` is not constant."""
    return {
        axis for axis in range(ndim) if bool((fw.amax(y, dim=axis) != fw.amin(y, dim=axis)).any())
    }


@pytest.mark.parity
@pytest.mark.parametrize("training", [True, False])
@pytest.mark.parametrize("ndim", range(7))
@pytest.mark.parametrize("name", ["dropout1d", "dropout2d", "dropout3d"])
def test_channel_dropout_takes_the_references_ranks_and_masks_its_channels(
    name: str, ndim: int, training: bool, device: str, ref: ModuleType
) -> None:
    shape = _shape(ndim)
    want = _outcome(lambda: getattr(ref.nn.functional, name)(ref.ones(shape), 0.5, training))
    got = _outcome(lambda: getattr(F, name)(lucid.ones(shape, device=device), 0.5, training))
    assert got == want, f"{name} on {ndim}-D, training={training}"
    if want != "ok" or not training:
        return
    lucid.manual_seed(0)
    ref.manual_seed(0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        expected = getattr(ref.nn.functional, name)(ref.ones(shape), 0.5, True)
        actual = getattr(F, name)(lucid.ones(shape, device=device), 0.5, True)
    assert _varying_axes(ref, expected, ndim) == _varying_axes(lucid, actual, ndim)


# ── erfc / i0e / i1e at ±inf and 0 ────────────────────────────────────────────


def _derivatives(
    fw: ModuleType, fn: Callable[[object], object], x: float, dtype: str
) -> list[float]:
    t = fw.tensor(x, dtype=getattr(fw, dtype), requires_grad=True)
    (g1,) = fw.autograd.grad(fn(t), t, create_graph=True)
    (g2,) = fw.autograd.grad(g1, t, allow_unused=True)
    return [g1.item(), g2.item()]


#: (name, Lucid function, reference-framework function path)
_SCALED: list[tuple[str, Callable[[object], object], str]] = [
    ("erfc", lucid.erfc, "erfc"),
    ("i0e", S.i0e, "special.i0e"),
    ("i1e", S.i1e, "special.i1e"),
]


@pytest.mark.parity
@pytest.mark.parametrize("dtype", ["float64", "float32"])
@pytest.mark.parametrize("x", [math.inf, -math.inf, 0.0])
@pytest.mark.parametrize(("name", "fn", "ref_name"), _SCALED, ids=[s[0] for s in _SCALED])
def test_the_special_derivatives_at_inf_and_zero_match_the_reference(
    name: str, fn: Callable[[object], object], ref_name: str, x: float, dtype: str, ref: ModuleType
) -> None:
    ref_fn = ref
    for part in ref_name.split("."):
        ref_fn = getattr(ref_fn, part)
    want = _derivatives(ref, ref_fn, x, dtype)
    got = _derivatives(lucid, fn, x, dtype)
    # float32 i0e has a fitted curvature at 0 (0.4999997); every other pair
    # agrees to the dtype's rounding.
    tol = 1e-5 if dtype == "float32" else 1e-12
    for g, w in zip(got, want, strict=True):
        assert math.isnan(g) == math.isnan(w), (name, dtype, x, got, want)
        if not math.isnan(w):
            assert math.isclose(g, w, rel_tol=tol, abs_tol=tol), (name, dtype, x, got, want)
    if name == "erfc" and math.isinf(x):
        # the slope of erfc is negative, so its zero is a negative one
        assert math.copysign(1.0, got[0]) == -1.0 and got[0] == 0.0
