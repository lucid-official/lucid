"""Values the package keeps between predictions.

A decoder's key-value cache is the case: the caller should not be handing
the whole cache in and getting it back every step. Core ML holds it, and
each prediction sees what the last one wrote.

Lucid's side has to be an input/output pair rather than a mutated buffer.
The tracer records a pure graph, so an in-place buffer write does not
appear in it — and the package built from that agrees on the first call
and stops accumulating on every one after, which `verify` would not
notice because it runs one call. That is refused separately.
"""

import pytest

import lucid
import lucid.coreml as cml
import lucid.nn as nn
from lucid._C import engine as _C_engine

pytestmark = pytest.mark.skipif(
    not hasattr(_C_engine, "coreml"),
    reason="the engine was built without the Core ML writer",
)


class Accumulate(nn.Module):
    """Reads the carried value, returns the new one beside a result."""

    def forward(self, x: lucid.Tensor, cache: lucid.Tensor) -> tuple[lucid.Tensor, lucid.Tensor]:
        carried = cache + x
        return carried, carried * 2.0


class MutatesItsBuffer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("total", lucid.zeros(1, 4))

    def forward(self, x: lucid.Tensor) -> lucid.Tensor:
        self.total += x
        return self.total + x


def _stateful(tmp_path: object, name: str) -> object:
    return cml.export(
        Accumulate().eval(),
        {"x": lucid.ones(1, 4) * 3.0, "cache": lucid.zeros(1, 4)},
        f"{tmp_path}/{name}.mlpackage",
        precision=cml.Precision.FLOAT16,
        state=[cml.State(input="cache", output="output_0")],
    )


class TestCarriedState:
    def test_it_accumulates_across_predictions(self, tmp_path: object) -> None:
        exported = _stateful(tmp_path, "accumulate")
        x = lucid.ones(1, 4) * 3.0
        try:
            assert exported.carries_state
            # The caller passes only x; the cache is not an input any more.
            assert exported.input_names == ["x"]
            assert exported.output_names == ["output_1"]

            # Eager, threaded through the same sequence by hand.
            model, cache = Accumulate().eval(), lucid.zeros(1, 4)
            for _ in range(3):
                cache, expected = model(x, cache)
                got = exported.predict(x)
                scale = float(expected.abs().max().item())
                assert float((got - expected).abs().max().item()) / scale < 1e-2
        finally:
            exported.close()

    def test_reset_returns_it_to_the_start(self, tmp_path: object) -> None:
        exported = _stateful(tmp_path, "reset")
        x = lucid.ones(1, 4) * 3.0
        try:
            first = exported.predict(x).reshape(-1).tolist()[0]
            exported.predict(x)
            exported.predict(x)
            exported.reset_state()
            assert abs(exported.predict(x).reshape(-1).tolist()[0] - first) < 1e-2
        finally:
            exported.close()

    def test_verify_refuses_because_one_call_proves_nothing(self, tmp_path: object) -> None:
        exported = _stateful(tmp_path, "verify")
        try:
            with pytest.raises(TypeError, match="carries state"):
                exported.verify(Accumulate().eval(), lucid.ones(1, 4))
        finally:
            exported.close()


class TestStateRefusals:
    def test_a_model_that_writes_its_own_buffer_is_refused(self, tmp_path: object) -> None:
        """The failure this guards against passes ``verify``.

        The buffer's value is read into the package after the trace, so
        the first prediction agrees exactly. Every one after it does not,
        because the package is a pure function and the eager model is
        still accumulating.
        """
        with pytest.raises(cml.StatefulModel) as excinfo:
            cml.export(
                MutatesItsBuffer().eval(),
                lucid.ones(1, 4),
                f"{tmp_path}/buffer.mlpackage",
            )
        assert excinfo.value.names == ["total"]

    def test_a_model_that_leaves_its_buffers_alone_still_exports(self, tmp_path: object) -> None:
        """BatchNorm's running statistics must not trip the guard."""
        import lucid.models as M

        model = M.create_model("resnet_18_cls", num_classes=10).eval()
        exported = cml.export(model, lucid.randn(1, 3, 224, 224), f"{tmp_path}/bn.mlpackage")
        exported.close()

    def test_state_needs_float16(self, tmp_path: object) -> None:
        with pytest.raises(ValueError, match="float16"):
            cml.export(
                Accumulate().eval(),
                {"x": lucid.ones(1, 4), "cache": lucid.zeros(1, 4)},
                f"{tmp_path}/fp32.mlpackage",
                state=[cml.State(input="cache", output="output_0")],
            )

    def test_a_name_that_is_not_there_is_refused(self, tmp_path: object) -> None:
        with pytest.raises(ValueError, match="not an input"):
            cml.export(
                Accumulate().eval(),
                {"x": lucid.ones(1, 4), "cache": lucid.zeros(1, 4)},
                f"{tmp_path}/name.mlpackage",
                precision=cml.Precision.FLOAT16,
                state=[cml.State(input="nope", output="output_0")],
            )

    def test_a_state_whose_shape_does_not_come_back_is_refused(self, tmp_path: object) -> None:
        class Shrinks(nn.Module):
            def forward(
                self, x: lucid.Tensor, cache: lucid.Tensor
            ) -> tuple[lucid.Tensor, lucid.Tensor]:
                return (cache + x).mean(dim=1, keepdim=True), cache + x

        with pytest.raises(ValueError, match="written back"):
            cml.export(
                Shrinks().eval(),
                {"x": lucid.ones(1, 4), "cache": lucid.zeros(1, 4)},
                f"{tmp_path}/shape.mlpackage",
                precision=cml.Precision.FLOAT16,
                state=[cml.State(input="cache", output="output_0")],
            )

    def test_making_every_input_state_is_refused(self, tmp_path: object) -> None:
        class OnlyCache(nn.Module):
            def forward(self, cache: lucid.Tensor) -> tuple[lucid.Tensor, ...]:
                return cache + 1.0, cache + 1.0

        with pytest.raises(ValueError, match="nothing for the caller"):
            cml.export(
                OnlyCache().eval(),
                {"cache": lucid.zeros(1, 4)},
                f"{tmp_path}/all.mlpackage",
                precision=cml.Precision.FLOAT16,
                state=[cml.State(input="cache", output="output_0")],
            )
