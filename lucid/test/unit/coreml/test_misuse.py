"""How a package answers a caller who drives it wrongly.

Every other file here asks whether a correct call produces a correct
answer. This one asks what happens to an incorrect one, which is the
half a person actually meets: the shapes are close but not equal, a
tensor is passed where the model wanted two, the handle was closed in a
``finally`` further up.

Core ML itself is good about most of it — a wrong shape, a wrong rank and
a wrong channel count all come back naming both sides. The gaps were on
the Lucid side of the boundary, and one of them did not raise at all.

Passing a sequence one element too long dropped the extra silently.
``zip`` stops at the shorter side, and the count that was supposed to
catch a mismatch was taken *after* the pairing — so it compared the
truncated list against the input names and found them equal. A caller
who added an argument to their model and re-exported it, but fed the old
package, got a prediction that ignored their new input and no
indication of it.

The rest is about error text. A list of floats used to surface
``'float' object has no attribute 'dtype'``, which names an internal
attribute and not the mistake; and a float64 input was told that Core ML
has no int64, which is true and has nothing to do with what it was
given.
"""

import pytest

import lucid
import lucid.nn as nn
import lucid.coreml as cml
from lucid._C import engine as _C_engine

pytestmark = pytest.mark.skipif(
    not hasattr(_C_engine, "coreml"),
    reason="the engine was built without the Core ML writer",
)


class _Adds(nn.Module):
    def forward(self, a: lucid.Tensor, b: lucid.Tensor) -> lucid.Tensor:
        return a + b


class _Small(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.body = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.ReLU())

    def forward(self, x: lucid.Tensor) -> lucid.Tensor:
        return self.body(x)


@pytest.fixture
def two_inputs(tmp_path):
    lucid.manual_seed(0)
    exported = cml.export(
        _Adds().eval(),
        {"a": lucid.ones(1, 4), "b": lucid.ones(1, 4)},
        str(tmp_path / "two.mlpackage"),
    )
    yield exported
    exported.close()


@pytest.fixture
def one_input(tmp_path):
    lucid.manual_seed(0)
    exported = cml.export(_Small().eval(), lucid.randn(1, 3, 8, 8), str(tmp_path / "one.mlpackage"))
    yield exported
    exported.close()


class TestTheCountIsTakenBeforeThePairing:
    """The defect, and both sides of the boundary it sits on."""

    def test_one_tensor_too_many_is_refused(self, two_inputs) -> None:
        """It was accepted, and the third tensor was never fed."""
        with pytest.raises(ValueError, match="2 input"):
            two_inputs.predict((lucid.ones(1, 4), lucid.ones(1, 4) * 2, lucid.ones(1, 4) * 100))

    def test_the_message_says_how_many_were_given(self, two_inputs) -> None:
        """Three, not the two that survived being zipped.

        Reporting the truncated count would be the bug written into the
        error: a caller told "2 were given" when they passed three has
        been handed a contradiction instead of a diagnosis.
        """
        with pytest.raises(ValueError) as excinfo:
            two_inputs.predict([lucid.ones(1, 4)] * 3)
        assert "3 were given" in str(excinfo.value)

    def test_one_tensor_too_few_is_still_refused(self, two_inputs) -> None:
        """This half always worked — the check has not been loosened."""
        with pytest.raises(ValueError, match="2 input"):
            two_inputs.predict((lucid.ones(1, 4),))

    def test_a_lone_tensor_to_a_two_input_package(self, two_inputs) -> None:
        with pytest.raises(ValueError, match="2 input"):
            two_inputs.predict(lucid.ones(1, 4))

    def test_the_right_count_still_predicts(self, two_inputs) -> None:
        """As narrow as the defect: the correct call is untouched."""
        got = two_inputs.predict((lucid.ones(1, 4), lucid.ones(1, 4) * 2))
        assert float(got.sum().item()) == pytest.approx(12.0)


class TestSomethingThatIsNotATensor:
    def test_a_list_of_numbers_names_the_mistake(self, one_input) -> None:
        """``'float' object has no attribute 'dtype'`` named an attribute."""
        with pytest.raises(TypeError, match="must be a Tensor"):
            one_input.predict([1.0])

    def test_a_mapping_holding_a_list(self, one_input) -> None:
        with pytest.raises(TypeError, match="must be a Tensor"):
            one_input.predict({"input": [1.0, 2.0]})

    def test_the_feature_is_named_too(self, two_inputs) -> None:
        """With two inputs, which one is wrong is the useful half."""
        with pytest.raises(TypeError) as excinfo:
            two_inputs.predict({"a": lucid.ones(1, 4), "b": 3.0})
        assert "'b'" in str(excinfo.value)

    def test_something_that_is_not_a_container_either(self, one_input) -> None:
        with pytest.raises(TypeError, match="expected a Tensor"):
            one_input.predict(object())


class TestADtypeCoreMLCannotRead:
    def test_the_refusal_names_what_was_given(self, one_input) -> None:
        """It named int64 and token ids, whatever it was handed."""
        with pytest.raises(ValueError) as excinfo:
            one_input.predict(lucid.randn(1, 3, 8, 8).to(lucid.float64))
        assert "float64" in str(excinfo.value)

    def test_a_boolean_input_says_boolean(self, one_input) -> None:
        with pytest.raises(ValueError) as excinfo:
            one_input.predict(lucid.ones(1, 3, 8, 8).to(lucid.bool_))
        assert "bool" in str(excinfo.value)

    def test_int64_is_narrowed_rather_than_refused(self, one_input) -> None:
        """The one integer type Core ML lacks is the one Lucid defaults to.

        ``lucid.tensor([1, 2])`` is int64, so token ids arrive that way
        from any ordinary call site. Narrowing them here rather than
        refusing is the difference between a package that takes ids and
        one that makes every caller convert.
        """
        got = one_input.predict(lucid.ones(1, 3, 8, 8).to(lucid.int64))
        assert tuple(got.shape) == (1, 8, 8, 8)


class TestAClosedHandle:
    def test_predicting_through_it_is_refused(self, tmp_path) -> None:
        """Not a crash — the handle owns a compiled model that is gone."""
        lucid.manual_seed(0)
        exported = cml.export(
            _Small().eval(), lucid.randn(1, 3, 8, 8), str(tmp_path / "closed.mlpackage")
        )
        exported.close()
        with pytest.raises(RuntimeError, match="closed"):
            exported.predict(lucid.randn(1, 3, 8, 8))

    def test_closing_twice_is_allowed(self, tmp_path) -> None:
        """``close`` in a ``finally`` beside a ``with`` is ordinary."""
        lucid.manual_seed(0)
        exported = cml.export(
            _Small().eval(), lucid.randn(1, 3, 8, 8), str(tmp_path / "twice.mlpackage")
        )
        exported.close()
        exported.close()


class TestWhatCoreMLItselfCatches:
    """Kept because it is the boundary, and boundaries move.

    None of these are Lucid's checks — they are Core ML refusing at
    prediction time — so they are here to record that the errors reach
    the caller as exceptions rather than as a crash or a wrong answer.
    """

    @pytest.mark.parametrize(
        ("name", "shape"),
        [
            ("spatial", (1, 3, 16, 16)),
            ("channels", (1, 8, 8, 8)),
            ("batch", (4, 3, 8, 8)),
            ("rank", (3, 8, 8)),
        ],
    )
    def test_a_shape_the_package_did_not_declare(self, one_input, name, shape) -> None:
        with pytest.raises(RuntimeError, match="prediction failed|rank"):
            one_input.predict(lucid.randn(*shape))


class TestAHandleIsAContextManager:
    """It owns a compiled model, so closing it is not optional.

    Every use of one in this codebase was already a ``try``/``finally``
    around ``close``, which is the shape a context manager exists for.
    """

    def test_the_block_closes_it(self, tmp_path) -> None:
        lucid.manual_seed(0)
        x = lucid.randn(1, 3, 8, 8)
        with cml.export(_Small().eval(), x, f"{tmp_path}/managed.mlpackage") as package:
            assert tuple(package.predict(x).shape) == (1, 8, 8, 8)
        with pytest.raises(RuntimeError, match="closed"):
            package.predict(x)

    def test_it_closes_when_the_block_raises(self, tmp_path) -> None:
        """The half that matters — a failure mid-block still releases it."""
        lucid.manual_seed(0)
        x = lucid.randn(1, 3, 8, 8)
        opened = cml.export(_Small().eval(), x, f"{tmp_path}/raising.mlpackage")
        with pytest.raises(ZeroDivisionError):
            with opened as package:
                package.predict(x)
                raise ZeroDivisionError
        with pytest.raises(RuntimeError, match="closed"):
            opened.predict(x)


class TestTheOtherFourAxesReachingThisOne:
    """Where the rest of Lucid meets the exporter, and what it used to say.

    Both of these sit on ordinary paths — you train on Metal, and you
    compile before you measure — and both used to fail somewhere below
    the exporter, naming neither the cause nor the way out. Neither is a
    translation gap: the export works fine once you know the two moves.
    """

    def test_a_model_on_the_gpu_says_what_to_do(self, tmp_path) -> None:
        """It raised from the blob writer, several layers down.

        "weights must be on the CPU to be written into the blob" is true
        and says nothing about the model, the export, or `.to('cpu')`.
        """
        lucid.manual_seed(0)
        model = _Small().eval().to("metal")
        with pytest.raises(ValueError, match="parameters are on the GPU"):
            cml.export(
                model,
                lucid.randn(1, 3, 8, 8, device="metal"),
                f"{tmp_path}/gpu.mlpackage",
            )

    def test_and_the_move_is_all_it_takes(self, tmp_path) -> None:
        """The refusal has to be worth following."""
        lucid.manual_seed(0)
        model = _Small().eval().to("metal").to("cpu")
        x = lucid.randn(1, 3, 8, 8)
        exported = cml.export(model, x, f"{tmp_path}/moved.mlpackage")
        try:
            assert tuple(exported.predict(x).shape) == (1, 8, 8, 8)
        finally:
            exported.close()

    def test_a_compile_wrapper_is_refused_by_name(self, tmp_path) -> None:
        """Tracing followed the wrapper and blamed the model.

        "the model ignored its input" is the opposite of what happened —
        the input reaches the network fine, and the trace never got to
        the network.
        """
        lucid.manual_seed(0)
        compiled = lucid.compile(_Small().eval().to("metal"))
        with pytest.raises(TypeError, match="lucid.compile wrapper"):
            cml.export(
                compiled,
                lucid.randn(1, 3, 8, 8, device="metal"),
                f"{tmp_path}/compiled.mlpackage",
            )

    def test_a_model_whose_weights_are_buffers_is_caught_too(self, tmp_path) -> None:
        """The guard first looked only at parameters, and that was a hole.

        A quantized linear registers its packed weight, its scales and
        even its bias as buffers, so a model made of them has no GPU
        parameter at all and would have reached the blob writer — the
        message this guard exists to replace. Reproduced with a plain
        module carrying a GPU buffer, since that is the shape rather than
        anything about quantization.
        """

        class Carries(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.register_buffer("held", lucid.ones(4, 4).to("metal"))

            def forward(self, x: lucid.Tensor) -> lucid.Tensor:
                return x + self.held

        lucid.manual_seed(0)
        with pytest.raises(ValueError, match="on the GPU"):
            cml.export(Carries().eval(), lucid.randn(4, 4), f"{tmp_path}/buffers.mlpackage")

    def test_pack_bits_refuses_gpu_keys_with_value_error(self) -> None:
        """LCD-263 / LCD-521: ``pack_bits`` must check ``device() == CPU``
        before ``std::get<CpuStorage>(keys->storage())`` so a Metal tensor
        raises ``ValueError`` (``InvalidArgument``) instead of
        ``std::bad_variant_access``."""
        gpu_keys = lucid.tensor([0, 1, 2, 3], dtype=lucid.int32, device="metal")
        with pytest.raises(ValueError, match="keys must be on the CPU"):
            _C_engine.coreml.pack_bits(gpu_keys._impl, 2)
