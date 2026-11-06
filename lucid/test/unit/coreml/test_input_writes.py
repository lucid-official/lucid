"""A forward that writes into its own input, or its buffers, while exported.

The trace runs the model once, and an in-place write to an input moved the
value that input's tensor resolved to: the export then found the *written*
value where it looked for the input, and wrote the value the input arrived
with into the package as a constant. The package ignored the cache it was
given — ``[[5, 5, 0, 0]]`` for ``[[5, 5, 7, 7]]`` — and the example tensor
kept the trace's write, so exporting changed the caller's data.

The input is now bound to the value it arrived with, so the package
computes the write from whatever it is handed, as eager does; and whatever
the trace wrote to those tensors — inputs and buffers alike — is put back
afterwards, whether the export succeeded or not.  That goes for a write through
a *view* of one (``x[:1].add_(1.0)``) too: what counts is the bytes it lands in,
and the view is how to reach them again (LCD-854).

A value the trace wrote stays what the program folds, after the tensor has been
put back: ``held.add_(1.0); return x * held`` used to store ``held`` from before
the write as the constant, and the package answered zeros.
"""

from collections.abc import Callable

import pytest

import lucid
import lucid.coreml as cml
import lucid.nn as nn
from lucid._C import engine as _C_engine
from lucid.coreml import _build as cml_build

pytestmark = pytest.mark.skipif(
    not hasattr(_C_engine, "coreml"), reason="the engine was built without the Core ML writer"
)


class _FillsCache(nn.Module):
    def forward(self, k: lucid.Tensor, cache: lucid.Tensor) -> lucid.Tensor:
        cache[:, 0:2] = k
        return cache * 1.0


class _AddsIntoCache(nn.Module):
    def forward(self, k: lucid.Tensor, cache: lucid.Tensor) -> lucid.Tensor:
        cache.add_(lucid.cat([k, k], dim=1))
        return cache


class _CountsIntoBuffer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("total", lucid.zeros(1, 4))

    def forward(self, x: lucid.Tensor) -> lucid.Tensor:
        self.total += x
        return self.total + x


class TestAWriteIntoAnInput:
    def test_the_package_reads_the_input_it_is_given(self, tmp_path: object) -> None:
        k, cache = lucid.ones(1, 2), lucid.zeros(1, 4)
        exported = cml.export(_FillsCache().eval(), (k, cache), f"{tmp_path}/m.mlpackage")
        try:
            got = exported.predict((k * 5.0, lucid.full((1, 4), 7.0)))
            assert got.tolist() == [[5.0, 5.0, 7.0, 7.0]]
        finally:
            exported.close()

    def test_the_example_is_left_as_it_was(self, tmp_path: object) -> None:
        k, cache = lucid.ones(1, 2), lucid.zeros(1, 4)
        cml.export(_FillsCache().eval(), (k, cache), f"{tmp_path}/m.mlpackage").close()
        assert cache.tolist() == [[0.0, 0.0, 0.0, 0.0]]

    def test_an_in_place_op_verifies_against_eager(self, tmp_path: object) -> None:
        """``verify`` runs the package first, or eager's write is counted twice."""
        k, cache = lucid.ones(1, 2), lucid.zeros(1, 4)
        exported = cml.export(_AddsIntoCache().eval(), (k, cache), f"{tmp_path}/m.mlpackage")
        try:
            fresh = (k * 5.0, lucid.full((1, 4), 7.0))
            assert exported.predict(fresh).tolist() == [[12.0, 12.0, 12.0, 12.0]]
            assert exported.verify(_AddsIntoCache().eval(), fresh) == 0.0
        finally:
            exported.close()

    def test_an_input_that_requires_grad_is_written_into_too(self, tmp_path: object) -> None:
        """A write into an input that requires grad is recorded like any other.

        Assigning into a tensor that requires grad used to rebind it to a new
        impl instead of writing into it.  The rebinding happened in Python,
        where the recording could not follow it, so the export refused such an
        input by name.  The assignment now writes in place (``_rebind``), as
        it does for every other tensor, and the package reads the cache it is
        handed.
        """
        k, cache = lucid.ones(1, 2), lucid.zeros(1, 4, requires_grad=True)
        held = cache._impl
        exported = cml.export(_FillsCache().eval(), (k, cache), f"{tmp_path}/m.mlpackage")
        try:
            got = exported.predict((k * 5.0, lucid.full((1, 4), 7.0)))
            assert got.tolist() == [[5.0, 5.0, 7.0, 7.0]]
        finally:
            exported.close()
        assert cache._impl is held and cache.requires_grad
        assert cache.tolist() == [[0.0, 0.0, 0.0, 0.0]]


class _AccumulatesInPlace(nn.Module):
    def forward(self, x: lucid.Tensor, cache: lucid.Tensor) -> tuple[lucid.Tensor, lucid.Tensor]:
        cache.add_(x)
        return cache, cache * 2.0


class TestACacheWrittenInPlaceCarriedAsState:
    def test_it_accumulates_across_predictions(self, tmp_path: object) -> None:
        """The decoder's own spelling of a cache, now that the input is read.

        Written in place and returned, the cache is the value the package
        keeps: each prediction reads what the last one wrote.
        """
        x, cache = lucid.ones(1, 4) * 3.0, lucid.zeros(1, 4)
        exported = cml.export(
            _AccumulatesInPlace().eval(),
            {"x": x, "cache": cache},
            f"{tmp_path}/m.mlpackage",
            precision=cml.Precision.FLOAT16,
            state=[cml.State(input="cache", output="output_0")],
        )
        try:
            got = [float(exported.predict(x)[0, 0].item()) for _ in range(3)]
            assert got == [6.0, 12.0, 18.0]
            assert cache.tolist() == [[0.0, 0.0, 0.0, 0.0]]
        finally:
            exported.close()


class TestAWriteIntoABuffer:
    def test_it_is_still_refused_and_the_buffer_is_put_back(self, tmp_path: object) -> None:
        model = _CountsIntoBuffer().eval()
        with pytest.raises(cml.StatefulModel):
            cml.export(model, lucid.ones(1, 4), f"{tmp_path}/m.mlpackage")
        assert model.total.tolist() == [[0.0, 0.0, 0.0, 0.0]]

    @pytest.mark.parametrize("make", [lambda: lucid.zeros(1, 4), lambda: lucid.zeros(2, 4)[1:]])
    def test_a_write_through_a_view_of_it_is_too(
        self, make: Callable[[], lucid.Tensor], tmp_path: object
    ) -> None:
        class _WritesAPart(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.register_buffer("total", make())

            def forward(self, x: lucid.Tensor) -> lucid.Tensor:
                self.total[:, :2].add_(1.0)
                return x * 1.0

        model = _WritesAPart().eval()
        with pytest.raises(cml.StatefulModel):
            cml.export(model, lucid.ones(1, 4), f"{tmp_path}/m.mlpackage")
        assert model.total.tolist() == [[0.0, 0.0, 0.0, 0.0]]


class _Apply(nn.Module):
    def __init__(self, fn: Callable[[lucid.Tensor], lucid.Tensor]) -> None:
        super().__init__()
        self.fn = fn

    def forward(self, x: lucid.Tensor) -> lucid.Tensor:
        return self.fn(x)


def _answer_after_export(
    fn: Callable[[lucid.Tensor], lucid.Tensor], tmp_path: object
) -> list[list[float]]:
    """The package's answer for ``ones * 3``, exported from an example of ones."""
    exported = cml.export(_Apply(fn).eval(), lucid.ones(1, 4), f"{tmp_path}/m.mlpackage")
    try:
        return exported.predict(lucid.ones(1, 4) * 3.0).tolist()
    finally:
        exported.close()


class TestAWriteThroughAView:
    """What the trace wrote through a view is put back, wherever it landed."""

    def test_of_the_input_leaves_the_example_alone(self) -> None:
        def fn(x: lucid.Tensor) -> lucid.Tensor:
            a = x * 2
            x[:, :1].add_(1.0)
            return a

        example = lucid.ones(1, 4)
        cml_build.trace(_Apply(fn).eval(), example)
        assert example.tolist() == [[1.0, 1.0, 1.0, 1.0]]

    def test_of_the_input_keeps_the_input_as_the_programs_input(self, tmp_path: object) -> None:
        def fn(x: lucid.Tensor) -> lucid.Tensor:
            a = x * 2
            x[:, :1].add_(1.0)
            return a

        assert _answer_after_export(fn, tmp_path) == [[6.0, 6.0, 6.0, 6.0]]

    @pytest.mark.parametrize(
        "view",
        [
            lambda t: t[:2],
            lambda t: t[1],
            lambda t: t[:, 1],
            lambda t: t.mT,
            lambda t: lucid.diagonal(t),
            lambda t: t.real,
            lambda t: t.imag,
        ],
        ids=["slice", "index", "column", "mT", "diagonal", "real", "imag"],
    )
    @pytest.mark.parametrize("to", ["input", "closure"])
    def test_of_any_kind_is_put_back(
        self, to: str, view: Callable[[lucid.Tensor], lucid.Tensor]
    ) -> None:
        """Every view a write can go through, on the trace's own (CPU) path."""
        complex_part = view(lucid.zeros(3, 3, dtype=lucid.complex64)).dtype == lucid.float32
        dtype = lucid.complex64 if complex_part else lucid.float32
        closure = lucid.ones(3, 3, dtype=dtype)
        example = lucid.ones(3, 3, dtype=dtype)
        want = closure.tolist()

        def fn(x: lucid.Tensor) -> lucid.Tensor:
            result = x * 1
            view(x if to == "input" else closure).add_(1.0)
            return result

        cml_build.trace(_Apply(fn).eval(), example)
        assert closure.tolist() == want
        assert example.tolist() == want

    @pytest.mark.parametrize(
        "view", [lambda t: t[0:1].expand(4), lambda t: t.unfold(0, 2, 1)], ids=["expand", "unfold"]
    )
    def test_with_overlapping_elements_is_refused_and_put_back(
        self, view: Callable[[lucid.Tensor], lucid.Tensor]
    ) -> None:
        """Nothing can be copied into such a view, so the tensor it came from is
        what goes back, and the export says why it stopped."""
        held = lucid.zeros(4)

        def fn(x: lucid.Tensor) -> lucid.Tensor:
            view(held).fill_(2.0)
            return x * 1.0

        with pytest.raises(ValueError, match="elements overlap"):
            cml_build.trace(_Apply(fn).eval(), lucid.ones(1, 4))
        assert held.tolist() == [0.0, 0.0, 0.0, 0.0]

    def test_a_tensor_written_whole_after_a_view_of_it_is_put_back_too(self) -> None:
        def fn(x: lucid.Tensor) -> lucid.Tensor:
            x[:, :2].add_(1.0)
            x.copy_(x.detach() * 0.0 + 9.0)
            return x * 1.0

        example = lucid.ones(1, 4)
        with pytest.raises(ValueError, match="after an in-place write"):
            cml_build.trace(_Apply(fn).eval(), example)
        assert example.tolist() == [[1.0, 1.0, 1.0, 1.0]]

    def test_of_a_tensor_it_closes_over_leaves_that_tensor_alone(self) -> None:
        held = lucid.zeros(4)

        def fn(x: lucid.Tensor) -> lucid.Tensor:
            held[:2].add_(1.0)
            return x * 1.0

        cml_build.trace(_Apply(fn).eval(), lucid.ones(1, 4))
        assert held.tolist() == [0.0, 0.0, 0.0, 0.0]

    def test_a_refused_read_leaves_that_tensor_alone_too(self, tmp_path: object) -> None:
        held = lucid.zeros(4)

        def fn(x: lucid.Tensor) -> lucid.Tensor:
            held[:2].add_(1.0)
            return x * held

        with pytest.raises(ValueError, match="after an in-place write"):
            cml.export(_Apply(fn).eval(), lucid.ones(1, 4), f"{tmp_path}/m.mlpackage")
        assert held.tolist() == [0.0, 0.0, 0.0, 0.0]

    def test_the_program_computes_the_write_once(self, tmp_path: object) -> None:
        held = lucid.zeros(4)

        def fn(x: lucid.Tensor) -> lucid.Tensor:
            part = held[:2]
            part.add_(1.0)
            return x[:, :2] * part

        assert _answer_after_export(fn, tmp_path) == [[3.0, 3.0]]
        assert held.tolist() == [0.0, 0.0, 0.0, 0.0]


class TestAConstantTheTraceWrote:
    """The package holds the value after the write, whatever became of the tensor."""

    def test_a_tensor_it_closes_over(self, tmp_path: object) -> None:
        held = lucid.zeros(4)

        def fn(x: lucid.Tensor) -> lucid.Tensor:
            held.add_(1.0)
            return x * held

        assert _answer_after_export(fn, tmp_path) == [[3.0, 3.0, 3.0, 3.0]]
        assert held.tolist() == [0.0, 0.0, 0.0, 0.0]

    def test_a_literal_written_with_a_constant(self, tmp_path: object) -> None:
        def fn(x: lucid.Tensor) -> lucid.Tensor:
            y = lucid.tensor([[1.0, 2.0, 3.0, 4.0]])
            y.add_(1.0)
            return x * y

        assert _answer_after_export(fn, tmp_path) == [[6.0, 9.0, 12.0, 15.0]]

    def test_a_literal_written_with_the_input(self, tmp_path: object) -> None:
        def fn(x: lucid.Tensor) -> lucid.Tensor:
            y = lucid.tensor([[1.0, 2.0, 3.0, 4.0]])
            y.add_(x)
            return y * 2.0

        assert _answer_after_export(fn, tmp_path) == [[8.0, 10.0, 12.0, 14.0]]


class TestTheTraceKeepsNoGraph:
    def test_autograd_is_off_while_tracing(self, tmp_path: object) -> None:
        """An export saves no activations for a backward that never comes.

        With autograd on, every intermediate of a model with trainable
        parameters was held for the whole build — gigabytes at video
        resolutions — for a graph nothing ever differentiated.
        """
        seen: list[bool] = []

        class _Records(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.lin = nn.Linear(4, 4)

            def forward(self, x: lucid.Tensor) -> lucid.Tensor:
                seen.append(lucid.is_grad_enabled())
                out = self.lin(x)
                seen.append(out.requires_grad)
                return out

        model = _Records().eval()
        assert all(p.requires_grad for p in model.parameters())
        cml.export(model, lucid.randn(1, 4), f"{tmp_path}/m.mlpackage").close()
        assert seen == [False, False]
        assert lucid.is_grad_enabled()
