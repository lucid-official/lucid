"""Complex tensors on the compile path, and the buffer sizes they exposed.

``complex`` was stubbed for needing "a 2-storage backing path the
real-input pipeline doesn't model". Lucid's complex is not two storages:
``Dtype.h`` defines C64 as an interleaved pair of float32 lanes in one
storage, eight bytes per element — the same bytes MPSGraph reads as
``MPSDataTypeComplexFloat32``. Nothing needed repacking; the converters
were missing the case.

Enabling it surfaced two sizing bugs that had nothing to do with complex
and everything to do with eight-byte dtypes, both covered below.
"""

import pytest

import lucid
import lucid.nn as nn
from lucid._C import engine as _C_engine


def _metal_ok() -> bool:
    try:
        lucid.zeros((1,)).to("metal")
        return True
    except Exception:  # noqa: BLE001 — any failure means no Metal here
        return False


pytestmark = pytest.mark.skipif(not _metal_ok(), reason="Metal unavailable")


class _Apply(nn.Module):
    def __init__(self, fn: object) -> None:
        super().__init__()
        self.fn = fn

    def forward(self, x: lucid.Tensor) -> lucid.Tensor:
        return self.fn(x)  # type: ignore[operator]


def _run(fn: object, x: lucid.Tensor) -> tuple[lucid.Tensor, lucid.Tensor, bool]:
    model = _Apply(fn).eval()
    reference = model(x)
    _C_engine.compile.session_cache_clear()
    compiled = lucid.compile.compile(model.to("metal"))
    got = compiled(x.to("metal")).to("cpu")
    return reference, got, _C_engine.compile.session_cache_size() > 0


def _lane_error(a: lucid.Tensor, b: lucid.Tensor) -> float:
    """Compare per lane — the CPU backend has no complex subtraction."""
    if str(a.dtype).endswith("complex64"):
        return max(
            float((lucid.real(a) - lucid.real(b)).abs().max().item()),
            float((lucid.imag(a) - lucid.imag(b)).abs().max().item()),
        )
    return float((a - b).abs().max().item())


class TestTheLaneOperations:
    """These used to answer for a real input whatever they were given.

    ``real`` and ``conj`` emitted the identity and ``imag`` emitted
    zeros, which is right for a real tensor and wrong for a complex one.
    It was safe only because no complex tensor could reach them — the
    converter threw on C64, so such a graph never got built. Turning the
    dtype on without rewriting them would have turned three refusals
    into three wrong answers.
    """

    @pytest.mark.parametrize(
        ("name", "fn"),
        [
            ("real", lambda x: lucid.real(lucid.complex(x, x * 2))),
            ("imag", lambda x: lucid.imag(lucid.complex(x, x * 2))),
            ("conj", lambda x: lucid.imag(lucid.conj(lucid.complex(x, x * 2)))),
        ],
        ids=["real", "imag", "conj-then-imag"],
    )
    def test_the_lane_that_comes_back_is_the_one_asked_for(self, name, fn):
        lucid.manual_seed(0)
        reference, got, compiled = _run(fn, lucid.randn(2, 4))
        assert compiled, "fell back to eager — the emitter declined this node"
        # Selecting a lane copies bytes; anything but exact means the
        # wrong lane came back.
        assert _lane_error(reference, got) == 0.0

    def test_imag_is_not_just_zeros(self):
        """The old emitter returned zeros here and would still 'pass'."""
        lucid.manual_seed(0)
        x = lucid.randn(2, 4)
        _, got, _ = _run(lambda t: lucid.imag(lucid.complex(t, t * 2)), x)
        assert float(got.abs().max().item()) > 0.0

    def test_conj_on_a_real_tensor_is_the_identity(self):
        """The engine keeps the node so the gradient still flows."""
        lucid.manual_seed(0)
        reference, got, compiled = _run(lambda t: lucid.conj(t) * 3, lucid.randn(2, 4))
        assert compiled
        assert _lane_error(reference, got) == 0.0


class TestComplexCrossesTheGraphBoundary:
    def test_a_complex_output_comes_back(self):
        lucid.manual_seed(0)
        reference, got, compiled = _run(lambda t: lucid.complex(t, t * 2), lucid.randn(2, 4))
        assert compiled
        assert str(got.dtype).endswith("complex64")
        assert _lane_error(reference, got) == 0.0

    def test_a_complex_input_goes_in(self):
        lucid.manual_seed(0)
        z = lucid.complex(lucid.randn(2, 4), lucid.randn(2, 4))
        reference, got, compiled = _run(lambda t: lucid.real(t) + lucid.imag(t), z)
        assert compiled
        assert _lane_error(reference, got) == 0.0


class TestTheBufferSizesEightByteDtypesNeed:
    """Output buffers were sized by a two-case guess: 2 for F16, else 4.

    Every eight-byte dtype therefore got half the buffer it needed, and
    MPSGraph answers that with a failed assertion inside MPSNDArray — a
    process abort, not an exception anything can catch or fall back
    from. Complex found it; ``argmax`` had been walking into it all
    along, which is as ordinary as a classifier gets.
    """

    def test_an_int64_output_is_allocated_for_int64(self):
        lucid.manual_seed(0)
        reference, got, compiled = _run(lambda t: lucid.argmax(t, dim=1), lucid.randn(4, 16))
        assert compiled
        assert str(reference.dtype).endswith("int64")
        assert bool((got == reference).all().item())

    def test_a_bool_output_comes_back(self):
        """A different hole in the same table: the readback had no Bool.

        It raised at call time rather than aborting, but a compiled
        model returning a comparison still could not be called.
        """
        lucid.manual_seed(0)
        reference, got, compiled = _run(lambda t: t > 0, lucid.randn(2, 4))
        assert compiled
        assert str(got.dtype).endswith("bool")
        assert bool((got == reference).all().item())


class TestTheBoundaryIsStated:
    def test_complex128_says_why_rather_than_failing_obscurely(self):
        """Neither MPSGraph nor MLX has a double-precision complex.

        The refusal comes from the device move, before compilation, and
        it names the dtype and the way out.
        """
        lucid.manual_seed(0)
        z = lucid.complex(lucid.randn(2, 2).to(lucid.float64), lucid.randn(2, 2).to(lucid.float64))
        assert str(z.dtype).endswith("complex128")
        with pytest.raises(Exception, match="complex128"):
            z.to("metal")
