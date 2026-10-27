"""``lucid.coreml`` — writing Core ML packages, running them, proving it.

Three properties, each with its own failure mode:

* **the bytes are right.**  Lucid writes the ``.mlpackage`` itself — MIL
  protobuf, weight blob, bundle — with no protobuf library and no
  coremltools.  A wrong field number or blob offset yields a file that
  still parses and still loads; only comparing against the reference
  reader, and against the eager model's numbers, catches it.
* **the numbers are right.**  A package missing a layer has the correct
  output shape and returns plausible values.  Only a value comparison
  finds that, which is why every model test verifies rather than asserting
  a shape.
* **the accelerator is actually used.**  Asking for the Neural Engine and
  not getting it is silent: a float32 program requested with
  ``CPU_AND_NE`` reports zero ANE operations, runs at CPU speed, and warns
  about nothing.  ``compute_plan`` turns that into an assertion.
"""

import os

import pytest

import lucid
import lucid.coreml as cml
import lucid.models as M
import lucid.nn as nn
from lucid._C import engine as _C_engine
from lucid.test.unit.coreml._helpers import (
    has_neural_engine,
    require_neural_engine,
    under_hypervisor,
)

pytestmark = pytest.mark.skipif(
    not hasattr(_C_engine, "coreml"), reason="the engine was built without the Core ML writer"
)


class _Tiny(nn.Module):
    """One of each op class the classification tier covers."""

    def __init__(self, num_classes: int = 5) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 8, 3, padding=1)
        self.bn = nn.BatchNorm2d(8)
        self.pool = nn.MaxPool2d(2)
        self.fc = nn.Linear(8 * 4 * 4, num_classes)

    def forward(self, x: lucid.Tensor) -> lucid.Tensor:
        h = nn.functional.relu(self.bn(self.conv(x)))
        return self.fc(self.pool(h).reshape(x.shape[0], -1))


def _tiny() -> tuple[nn.Module, lucid.Tensor]:
    return _Tiny().eval(), lucid.randn(1, 3, 8, 8)


class TestPackageFormat:
    """The engine writes the bundle; these check its shape on disk."""

    def test_a_package_has_the_three_parts(self, tmp_path: object) -> None:
        import os

        model, x = _tiny()
        cm = cml.export(model, x, str(tmp_path / "t.mlpackage"))

        root = cm.path
        assert os.path.exists(os.path.join(root, "Manifest.json"))
        assert os.path.exists(os.path.join(root, "Data/com.apple.CoreML/model.mlmodel"))
        assert os.path.exists(os.path.join(root, "Data/com.apple.CoreML/weights/weight.bin"))
        cm.close()

    def test_a_hand_built_program_loads_and_runs(self, tmp_path: object) -> None:
        # The smallest possible exercise of writer + runtime, with no
        # tracer involved: if this fails the format is wrong, not the
        # translation.
        engine = _C_engine.coreml
        f32 = engine.DTYPE_FLOAT32
        program = engine.MilProgram([("x", (f32, [1, 4]))])
        program.add_op("relu", [("x", ["x"])], "y", (f32, [1, 4]))
        program.add_output("y", (f32, [1, 4]))

        paths = engine.prepare_package(str(tmp_path / "relu.mlpackage"))
        blob = engine.BlobWriter(paths.weight_bin)
        blob.finalize()
        engine.finish_package(paths, program.serialize())

        handle = engine.load_model(paths.root, engine.ComputeUnits.CPU_ONLY)
        out = lucid.Tensor(
            handle.predict([("x", lucid.tensor([[-1.0, 2.0, -3.0, 4.0]])._impl)], ["y"])[0]
        )

        assert out.numpy().ravel().tolist() == [0.0, 2.0, 0.0, 4.0]
        handle.close()


def _with_an_opinion(factory: str) -> object:
    """A zoo factory whose zero-initialised parameters have been perturbed.

    Several heads start at exactly zero, which is ordinary practice and
    makes an untrained model answer every input with almost the same
    almost-nothing. A comparison against that reports agreement whatever
    the package does.
    """
    model = M.create_model(factory, num_classes=10).eval()
    replaced = {}
    for name, value in model.state_dict().items():
        zeroed = (
            hasattr(value, "shape")
            and int(value.numel()) > 0
            and float(value.abs().max().item()) == 0.0
        )
        replaced[name] = lucid.randn(*value.shape) * 0.05 if zeroed else value
    perturbed = M.create_model(factory, num_classes=10).eval()
    perturbed.load_state_dict(replaced)
    return perturbed


class TestAgreementWithEager:
    def test_a_tiny_model_matches(self, tmp_path: object) -> None:
        model, x = _tiny()

        cm = cml.export(model, x, str(tmp_path / "tiny.mlpackage"))

        assert cm.verify(model, x) < 1e-5
        cm.close()

    @pytest.mark.parametrize(
        "factory", ["resnet_18_cls", "mobilenet_v2_cls", "densenet_121_cls", "convnext_tiny_cls"]
    )
    def test_zoo_classifiers_match(self, factory: str, tmp_path: object) -> None:
        """Unmodified factories, output dataclass and all.

        With one modification that the comparison needs: a
        zero-initialised head answers a 64-square input with logits near
        1e-12, and flooring the scale to divide by then compares noise
        against noise — MobileNet passed this way while measuring
        nothing. Perturbing the zeroed parameters gives the model an
        opinion to check.
        """
        model = _with_an_opinion(factory)
        x = lucid.randn(1, 3, 64, 64)
        scale = float(model(x).logits.abs().max().item())
        assert scale > 1e-6, f"{factory} still answers with nothing to compare"

        cm = cml.export(model, x, str(tmp_path / f"{factory}.mlpackage"))

        assert cm.verify(model, x) / scale < 1e-4
        cm.close()

    def test_float16_costs_precision_and_says_so(self, tmp_path: object) -> None:
        model, x = _tiny()

        half = cml.export(
            model, x, str(tmp_path / "half.mlpackage"), precision=cml.Precision.FLOAT16
        )

        assert half.precision == "FLOAT16"
        assert half.verify(model, x) < 1e-2
        half.close()


class TestTheAcceleratorIsActuallyUsed:
    """The claim this package exists for, asserted rather than timed."""

    def test_float32_reaches_no_neural_engine(self, tmp_path: object) -> None:
        # The silent failure: asking for the ANE with a float32 program
        # succeeds, returns correct numbers, and uses none of it.  Without
        # a Neural Engine 0.0 is the only answer there is, so asserting it
        # would pass without being able to fail.
        require_neural_engine()
        model = M.create_model("resnet_18_cls", num_classes=10).eval()
        x = lucid.randn(1, 3, 64, 64)

        cm = cml.export(
            model,
            x,
            str(tmp_path / "f32.mlpackage"),
            precision=cml.Precision.FLOAT32,
            compute_units=cml.ComputeUnits.CPU_AND_NE,
        )

        plan = cm.compute_plan()
        if plan.total_compute:  # macOS < 14.4 reports nothing
            assert plan.ane_fraction == 0.0
        cm.close()

    def test_float16_runs_the_computation_on_the_neural_engine(self, tmp_path: object) -> None:
        require_neural_engine()
        model = M.create_model("resnet_18_cls", num_classes=10).eval()
        x = lucid.randn(1, 3, 64, 64)

        cm = cml.export(
            model,
            x,
            str(tmp_path / "f16.mlpackage"),
            precision=cml.Precision.FLOAT16,
            compute_units=cml.ComputeUnits.CPU_AND_NE,
        )

        plan = cm.compute_plan()
        if plan.total_compute:
            # The only operations left on the CPU are the two casts that
            # keep the interface float32.
            assert plan.ane_fraction > 0.9
            assert plan.compute.get("CPU", 0) <= 2
        cm.close()

    def test_constants_are_not_counted_as_computation(self, tmp_path: object) -> None:
        # ResNet-18 emits ~250 const operations against ~70 real ones;
        # counting them would report ~20% ANE for a fully accelerated model.
        model, x = _tiny()
        cm = cml.export(model, x, str(tmp_path / "counts.mlpackage"))

        plan = cm.compute_plan()
        if plan.total_compute:
            assert plan.constants > plan.total_compute
        cm.close()


class TestVerificationCannotBeVacuous:
    def test_an_all_zero_reference_is_refused(self, tmp_path: object) -> None:
        """Comparing against zeros would score a broken exporter perfectly.

        Not theoretical: several zoo models zero-initialise their head, so
        an untrained factory returns exactly zero and every difference is
        zero with it. ViT is one, which is how this was found.
        """

        class ZeroHead(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.fc = nn.Linear(4, 3)
                with lucid.no_grad():
                    state = self.state_dict()
                    state["fc.weight"] = lucid.zeros(*state["fc.weight"].shape)
                    state["fc.bias"] = lucid.zeros(*state["fc.bias"].shape)
                    self.load_state_dict(state)

            def forward(self, x: lucid.Tensor) -> lucid.Tensor:
                return self.fc(x)

        model = ZeroHead().eval()
        x = lucid.randn(1, 4)
        cm = cml.export(model, x, str(tmp_path / "zero.mlpackage"))

        with pytest.raises(ValueError, match="below the"):
            cm.verify(model, x)
        cm.close()

    def test_a_transformer_matches(self, tmp_path: object) -> None:
        # Attention, split and broadcast all at once.  The zero-initialised
        # parameters are perturbed first, or the comparison would be the
        # vacuous one above.
        model = M.create_model("vit_base_16_cls", num_classes=10).eval()
        state = model.state_dict()
        for key, value in list(state.items()):
            if "running" not in key and float(value.abs().max().item()) == 0.0:
                state[key] = lucid.randn(*value.shape) * 0.02
        model.load_state_dict(state)
        x = lucid.randn(1, 3, 224, 224)
        scale = float(model(x).logits.abs().max().item())

        cm = cml.export(model, x, str(tmp_path / "vit.mlpackage"))

        assert cm.verify(model, x) / scale < 1e-4
        cm.close()


class TestTextModels:
    """Integer inputs, embeddings and attention — a different shape of graph.

    Vision models are float end to end. A language model arrives as token
    ids, which are integers, and Core ML's multi-array has int32 and no
    int64. The interface narrows on the way in; what must *not* happen is
    the float16 path casting those indices to half, which would turn them
    into approximations of themselves.
    """

    @pytest.mark.parametrize("factory", ["bert_base", "gpt2_small"])
    def test_a_language_model_matches(self, factory: str, tmp_path: object) -> None:
        model = M.create_model(factory).eval()
        ids = lucid.tensor([[101, 7592, 2088, 102]]).long()
        reference = model(ids).last_hidden_state
        scale = float(reference.abs().max().item())

        cm = cml.export(
            model, ids, str(tmp_path / f"{factory}.mlpackage"), output_field="last_hidden_state"
        )

        got = cm.predict(ids)
        assert float((got - reference).abs().max().item()) / scale < 1e-4
        cm.close()

    def test_token_ids_survive_the_float16_path(self, tmp_path: object) -> None:
        model = M.create_model("bert_base").eval()
        ids = lucid.tensor([[101, 7592, 2088, 102]]).long()
        reference = model(ids).last_hidden_state
        scale = float(reference.abs().max().item())

        cm = cml.export(
            model,
            ids,
            str(tmp_path / "bert16.mlpackage"),
            output_field="last_hidden_state",
            precision=cml.Precision.FLOAT16,
            compute_units=cml.ComputeUnits.CPU_AND_NE,
        )

        # Indices cast to half would not merely lose precision, they would
        # look up different rows; the error would be enormous, not ~1e-2.
        assert float((cm.predict(ids) - reference).abs().max().item()) / scale < 0.05
        # Where it ran is a separate claim, and only hardware with a Neural
        # Engine can answer it; the numbers above hold either way.
        plan = cm.compute_plan()
        if plan.total_compute and has_neural_engine():
            assert plan.ane_fraction > 0.9
        cm.close()

    def test_an_output_dataclass_exports_every_field_it_declares(self, tmp_path: object) -> None:
        """A model's outputs are all of them, named as the model names them.

        Taking one field and calling the package the model is the same
        failure as dropping a layer, a level up: it loads, it runs, and it
        answers for a part of the network.
        """
        model = M.create_model("bert_base").eval()
        ids = lucid.tensor([[101, 7592, 2088, 102]]).long()
        reference = model(ids)

        cm = cml.export(model, ids, str(tmp_path / "fields.mlpackage"))
        got = cm.predict(ids)

        assert cm.output_names == ["last_hidden_state", "pooler_output"]
        for field in cm.output_names:
            wanted = getattr(reference, field)
            scale = float(wanted.abs().max().item())
            assert float((got[field] - wanted).abs().max().item()) / scale < 1e-4
        cm.close()

    def test_a_model_returning_no_tensor_says_so(self, tmp_path: object) -> None:
        class ReturnsNothingUseful(nn.Module):
            def forward(self, x: lucid.Tensor) -> object:
                return {"not": "a tensor", "x": x}

        with pytest.raises(TypeError, match="no tensor to export"):
            cml.export(
                ReturnsNothingUseful().eval(), lucid.randn(1, 4), str(tmp_path / "none.mlpackage")
            )


class TestSegmentationAndDetection:
    @pytest.mark.parametrize("factory,channels", [("unet", 1), ("attention_unet", 1)])
    def test_a_segmentation_model_matches(
        self, factory: str, channels: int, tmp_path: object
    ) -> None:
        # These default to one input channel — the medical-imaging
        # convention — which is worth stating: feeding three produces a
        # shape error that looks like a tracer bug and is not.
        model = M.create_model(factory).eval()
        x = lucid.randn(1, channels, 64, 64)
        reference = model(x)
        reference = reference if isinstance(reference, lucid.Tensor) else reference.logits
        scale = float(reference.abs().max().item())

        cm = cml.export(model, x, str(tmp_path / f"{factory}.mlpackage"))

        assert float((cm.predict(x) - reference).abs().max().item()) / scale < 1e-5
        cm.close()

    def test_a_detector_exports_all_three_of_its_heads(self, tmp_path: object) -> None:
        """A detector is boxes and objectness, not only class scores.

        Exporting one field of a several-field output produces a package
        that loads, runs, and returns plausible numbers for a third of
        the model. Nothing about it looks wrong from the outside.
        """
        if under_hypervisor():
            # Inside the hosted runner's virtual machine Core ML runs this
            # package on BNNS's CPU path whatever compute units are asked
            # for, and BNNS traps there at prediction — SIGTRAP, exit 133,
            # nothing printed — taking the rest of the suite with it. It
            # passes on hardware. The conftest would refuse the load anyway;
            # skipping here saves building YOLOv3 first. Where the trap is:
            # .github/workflows/coreml-vm-probe.yml.
            pytest.skip("Core ML traps running YOLOv3 inside a VM; runs on hardware")
        model = M.create_model("yolo_v3").eval()
        x = lucid.randn(1, 3, 416, 416)
        reference = model(x)

        cm = cml.export(model, x, str(tmp_path / "yolo.mlpackage"))
        got = cm.predict(x)

        assert set(cm.output_names) == {"logits", "pred_boxes", "objectness"}
        for field in cm.output_names:
            wanted = getattr(reference, field)
            scale = float(wanted.abs().max().item())
            # Tight on purpose. A leaky-ReLU slope read from the wrong
            # attribute name put this at 1.8e-03 while everything still
            # ran; a loose bound would have called that a pass.
            assert float((got[field] - wanted).abs().max().item()) / scale < 1e-6
        cm.close()


class TestFormatLimits:
    def test_window_attention_exports_despite_the_rank_cap(self, tmp_path: object) -> None:
        """Core ML caps tensors at rank five; window attention wants six.

        This used to assert the refusal, because a window partition lifts
        ``(B, H, W, C)`` to ``(B, H/w, w, W/w, w, C)`` for the two
        operations it takes to permute and collapse again. The tall shape
        is transient, so it is now staged under the cap instead — the
        rank-6 tensor is never asked for — and Swin exports.

        The cap itself has not moved: a tall value that something else
        reads is still refused by name, which
        ``test_op_parity.py::TestTheWindowPartitionGetsPastTheRankCap``
        covers alongside the staging.
        """
        lucid.manual_seed(0)
        model = M.create_model("swin_tiny_cls").eval()
        x = lucid.randn(1, 3, 224, 224)
        scale = max(float(model(x).logits.abs().max().item()), 1e-6)

        exported = cml.export(model, x, str(tmp_path / "swin.mlpackage"))
        try:
            assert exported.verify(model, x) / scale < 1e-4
        finally:
            exported.close()


class TestRoundTrip:
    def test_a_written_package_can_be_loaded_back(self, tmp_path: object) -> None:
        model, x = _tiny()
        path = str(tmp_path / "again.mlpackage")
        first = cml.export(model, x, path)
        want = first.predict(x)
        first.close()

        again = cml.load(path)

        assert float((again.predict(x) - want).abs().max().item()) == 0.0
        again.close()


class TestRefusals:
    """Each of these would otherwise be a quietly wrong model."""

    def test_an_unmapped_op_names_itself(self, tmp_path: object) -> None:
        # A matrix inverse, because Core ML's program dialect carries no
        # linear-algebra solver and so this one stays unmapped however far
        # the emitter table grows.
        class UsesAnUnmappedOp(nn.Module):
            def forward(self, x: lucid.Tensor) -> lucid.Tensor:
                return lucid.linalg.inv(x)

        with pytest.raises(cml.UnsupportedOp) as excinfo:
            cml.export(
                UsesAnUnmappedOp().eval(),
                lucid.eye(4).reshape(1, 4, 4) * 4.0,
                str(tmp_path / "no.mlpackage"),
            )

        assert excinfo.value.op_name == "inv"

    def test_a_training_mode_model_is_refused(self, tmp_path: object) -> None:
        with pytest.raises(ValueError, match="training mode"):
            cml.export(_Tiny().train(), lucid.randn(1, 3, 8, 8), str(tmp_path / "t.mlpackage"))

    def test_float64_has_no_core_ml_equivalent(self, tmp_path: object) -> None:
        class Passthrough(nn.Module):
            def forward(self, x: lucid.Tensor) -> lucid.Tensor:
                return x * 2.0

        with pytest.raises(TypeError, match="float64"):
            cml.export(
                Passthrough().eval(),
                lucid.ones(2, dtype=lucid.float64),
                str(tmp_path / "d.mlpackage"),
            )

    def test_a_metal_input_is_not_silently_downloaded(self, tmp_path: object) -> None:
        if not lucid.metal.is_available():
            pytest.skip("needs Metal")
        model, x = _tiny()
        cm = cml.export(model, x, str(tmp_path / "dev.mlpackage"))

        with pytest.raises(ValueError, match="CPU tensor"):
            cm.predict(x.to("metal"))
        cm.close()


class TestQuantizedWeights:
    """Eight bits per weight, and what that buys and costs.

    Quantization is a storage decision: the codes are dequantized on the
    way into each operation and the arithmetic still runs at the body's
    precision. So the two things worth asserting are that the package
    actually got smaller and that the numbers did not fall apart.
    """

    def _weight_bytes(self, package: str) -> int:
        return os.path.getsize(f"{package}/Data/com.apple.CoreML/weights/weight.bin")

    def test_int8_halves_the_weights_and_still_agrees(self, tmp_path: object) -> None:
        model = M.create_model("resnet_18_cls", num_classes=10).eval()
        # From a stream of its own: drawn after the factory's initialisation
        # on the default one, the input changed with every draw that
        # initialisation made, and the error is measured on it (LCD-615).
        x = lucid.randn(1, 3, 224, 224, generator=lucid.Generator(1))
        reference = model(x).logits
        scale = float(reference.abs().max().item())

        plain = str(tmp_path / "float.mlpackage")
        packed = str(tmp_path / "int8.mlpackage")
        a = cml.export(model, x, plain, precision=cml.Precision.FLOAT16)
        b = cml.export(
            model, x, packed, precision=cml.Precision.FLOAT16, weights=cml.WeightPrecision.INT8
        )
        try:
            # Halved, near enough: the per-channel scales and the weights
            # below the threshold are still stored in full.
            assert self._weight_bytes(packed) < self._weight_bytes(plain) * 0.6
            error = float((b.predict(x) - reference).abs().max().item()) / scale
            # Looser than float16 on purpose. Eight bits cannot hold what
            # sixteen did, and pretending otherwise would make this test
            # pass for a package that quantized nothing.
            assert error < 0.2
        finally:
            a.close()
            b.close()

    def test_a_quantized_package_still_reaches_the_neural_engine(self, tmp_path: object) -> None:
        require_neural_engine()
        model = M.create_model("resnet_18_cls", num_classes=10).eval()
        x = lucid.randn(1, 3, 224, 224)
        exported = cml.export(
            model,
            x,
            str(tmp_path / "q.mlpackage"),
            precision=cml.Precision.FLOAT16,
            weights=cml.WeightPrecision.INT8,
            compute_units=cml.ComputeUnits.CPU_AND_NE,
        )
        try:
            plan = exported.compute_plan()
            if plan.total_compute:
                assert plan.ane_fraction > 0.9
        finally:
            exported.close()


class TestReadingSomeoneElsesOutputs:
    """A loaded package chooses its own element type; we do not.

    Every package Lucid writes casts its outputs to float32, so this path
    is only reachable through ``load`` — which is exactly why it went
    unnoticed. A reference package returning float16 loaded, reported
    itself correctly, and could not hand back its result.
    """

    def _package(self, tmp_path: object, name: str, dtype: int) -> str:
        engine = _C_engine.coreml
        program = engine.MilProgram([("x", (engine.DTYPE_FLOAT16, [1, 4]))])
        program.add_string_const("target", "fp32")
        if dtype == engine.DTYPE_FLOAT16:
            program.add_op("relu", [("x", ["x"])], "y", (dtype, [1, 4]))
        else:
            program.add_op("relu", [("x", ["x"])], "h", (engine.DTYPE_FLOAT16, [1, 4]))
            program.add_op("cast", [("x", ["h"]), ("dtype", ["target"])], "y", (dtype, [1, 4]))
        program.add_output("y", (dtype, [1, 4]))
        paths = engine.prepare_package(f"{tmp_path}/{name}.mlpackage")
        engine.BlobWriter(paths.weight_bin).finalize()
        engine.finish_package(paths, program.serialize())
        return str(paths.root)

    @pytest.mark.parametrize("kind", ["float16", "float32"])
    def test_an_output_is_read_at_the_type_it_declares(self, kind: str, tmp_path: object) -> None:
        engine = _C_engine.coreml
        wanted = engine.DTYPE_FLOAT16 if kind == "float16" else engine.DTYPE_FLOAT32
        handle = engine.load_model(
            self._package(tmp_path, kind, wanted), engine.ComputeUnits.CPU_ONLY
        )
        try:
            x = lucid.tensor([[-1.0, 2.0, -3.0, 4.0]]).half()
            got = lucid.Tensor(handle.predict([("x", x._impl)], list(handle.output_names), [])[0])
            assert str(got.dtype).endswith(kind)
            assert got.reshape(-1).tolist() == [0.0, 2.0, 0.0, 4.0]
        finally:
            handle.close()


class TestAPlanningFailureSaysWhichHalfFailed:
    """ "Error in building plan" does not say what to do about it.

    A package can be well formed and still be refused by Core ML's
    accelerator planner. Malformed bytes and a planner that will not take
    the graph need opposite responses from the caller, so the runtime
    asks the one question that separates them — does it load on the
    CPU? — and answers it in the message.

    ``roformer`` was the model that provoked this: it loaded on the CPU
    and nowhere else. It plans everywhere now — constant folding took
    its graph from something the planner refused to something it
    accepts — so no model in the zoo reaches that message any more, and
    what is checked here is the improvement instead. The message lives
    in ``CoreMLRuntime.mm`` and is exercised the next time a package
    provokes it.
    """

    def test_a_model_that_used_to_be_refused_now_plans(self, tmp_path: object) -> None:
        """What folding bought, stated as a fact rather than a hope."""
        lucid.manual_seed(0)
        model = M.create_model("roformer").eval()
        ids = lucid.zeros(1, 16).to(lucid.int64)

        exported = cml.export(
            model, ids, str(tmp_path / "roformer_all.mlpackage"), compute_units=cml.ComputeUnits.ALL
        )
        try:
            assert exported.verify(model, ids, relative=True) < 1e-4
        finally:
            exported.close()

    def test_and_the_cpu_configuration_still_works(self, tmp_path: object) -> None:
        """The advice is checked, not offered."""
        lucid.manual_seed(0)
        model = M.create_model("roformer").eval()
        ids = lucid.zeros(1, 16).to(lucid.int64)
        raw = model(ids)
        reference = raw if isinstance(raw, lucid.Tensor) else raw.last_hidden_state
        scale = max(float(reference.abs().max().item()), 1e-6)

        exported = cml.export(
            model,
            ids,
            str(tmp_path / "roformer_cpu.mlpackage"),
            compute_units=cml.ComputeUnits.CPU_ONLY,
        )
        try:
            assert exported.verify(model, ids) / scale < 1e-4
        finally:
            exported.close()


class TestThePublicSurfaceCanBeIntrospected:
    """``inspect.signature`` on ``export`` used to raise ``NameError``.

    Annotations are evaluated lazily (PEP 649) but they are still
    evaluated, against the defining module's globals, the moment anything
    asks — and ``inspect.signature`` asks. Names that live only inside a
    ``TYPE_CHECKING`` block are absent then, so ``help(cml.export)``,
    Sphinx, and every IDE hit the same wall on this package's main entry
    point.

    The names are imported for real now. This is not only about tidiness:
    a documentation build that cannot read a signature produces a page
    that silently omits it.
    """

    @pytest.mark.parametrize("name", ["export", "export_functions", "load"])
    def test_a_public_entry_point_has_a_readable_signature(self, name: str) -> None:
        import inspect

        signature = inspect.signature(getattr(cml, name))
        assert signature.parameters

    def test_the_annotations_resolve_to_the_real_classes(self) -> None:
        import inspect

        from lucid.nn.module import Module

        assert inspect.signature(cml.export).parameters["model"].annotation is Module


class TestWeightQuantizationInEitherBody:
    """``INT8`` weights used to work only alongside a float16 body.

    Asked for on its own — the first thing to try, since it is a storage
    decision and the body's precision is a separate one — the package
    failed to parse: "Tensor storage and type have different number of
    elements". The scale went out as raw bytes, which is the encoding
    MIL needs for float16 because it has no immediate list for it; for
    float32 there is one, and the reader counts elements against the
    declared type rather than guessing.

    Sizes are asserted as ratios rather than absolutes so the test says
    what quantizing is for: four times smaller than float32, and the
    accuracy cost is measured rather than assumed.
    """

    @staticmethod
    def _bytes(root: str) -> int:
        total = 0
        for base, _dirs, files in os.walk(root):
            for name in files:
                total += os.path.getsize(os.path.join(base, name))
        return total

    @pytest.mark.parametrize(
        ("name", "precision"),
        [("float32", cml.Precision.FLOAT32), ("float16", cml.Precision.FLOAT16)],
    )
    def test_int8_weights_hold_in_either_body(
        self, name: str, precision: object, tmp_path: object
    ) -> None:
        lucid.manual_seed(0)
        model = M.create_model("resnet_18_cls", num_classes=10).eval()
        x = lucid.randn(1, 3, 64, 64)
        scale = max(float(model(x).logits.abs().max().item()), 1e-6)

        plain = cml.export(model, x, str(tmp_path / f"{name}.mlpackage"), precision=precision)
        quantized = cml.export(
            model,
            x,
            str(tmp_path / f"{name}_int8.mlpackage"),
            precision=precision,
            weights=cml.WeightPrecision.INT8,
        )
        try:
            assert quantized.verify(model, x) / scale < 5e-2
            # Eight bits against thirty-two or sixteen: smaller, and by
            # enough that a silently-unquantized export would fail here.
            assert self._bytes(quantized.path) < self._bytes(plain.path) * 0.75
        finally:
            plain.close()
            quantized.close()


class TestVerifyingAnImageExport:
    """``verify`` on an image package used to report the pixel rounding.

    A pixel buffer is eight bits per channel. Handed ``randn`` values —
    what any test reaches for — Core ML quantises them on the way in, the
    two sides see different pixels, and the comparison came back at
    3e-1. That number reads as a broken exporter and is nothing of the
    sort, so the call now refuses rather than answering it, the way it
    already refuses for a classifier and for a stateful package.
    """

    @staticmethod
    def _image_model(tmp_path: object, name: str) -> tuple[object, object, object]:
        lucid.manual_seed(0)
        model = M.create_model("resnet_18_cls", num_classes=10).eval()
        pixels = (lucid.rand(1, 3, 64, 64) * 255).round()
        exported = cml.export(
            model,
            pixels,
            str(tmp_path / name),
            image_input=cml.ImageInput(color=cml.ColorSpace.RGB),
        )
        return model, pixels, exported

    def test_pixel_input_compares_cleanly(self, tmp_path: object) -> None:
        model, pixels, exported = self._image_model(tmp_path, "pixels.mlpackage")
        try:
            scale = max(float(model(pixels).logits.abs().max().item()), 1e-6)
            assert exported.verify(model, pixels) / scale < 1e-5
        finally:
            exported.close()

    def test_non_pixel_input_is_refused_with_the_reason(self, tmp_path: object) -> None:
        model, _pixels, exported = self._image_model(tmp_path, "noise.mlpackage")
        try:
            with pytest.raises(TypeError, match="pixel data"):
                exported.verify(model, lucid.randn(1, 3, 64, 64))
        finally:
            exported.close()


class TestASingleResultEmitter:
    """An emitter that builds one value names its operation's first result."""

    def test_an_operation_with_several_results_binds_the_first(
        self, tmp_path: object, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Stands in for the next emitter that answers a multi-result
        # operation with one MIL op: the driver names the first result and
        # leaves the rest for a consumer to ask after by name.  Pairing the
        # one value with every result raised once ``zip`` turned strict.
        from lucid.coreml import _emit
        from lucid.coreml._build import Builder

        def first_section(b: Builder, op: _emit.TracedOp, ins: list[str]) -> _emit.EmitResult:
            size = [int(d) for d in op.outputs[0].shape]
            return "slice_by_size", [
                ("x", ins[0]),
                ("begin", b.const_ints([0] * len(size))),
                ("size", b.const_ints(size)),
            ]

        class FirstHalf(nn.Module):
            def forward(self, x: lucid.Tensor) -> lucid.Tensor:
                return lucid.split(x, 2, dim=1)[0] * 3.0

        monkeypatch.setitem(_emit.EMITTERS, "split", first_section)
        model, x = FirstHalf().eval(), lucid.randn(3, 4)
        cm = cml.export(model, x, str(tmp_path / "first.mlpackage"))
        try:
            got = cm.predict(x)
            assert isinstance(got, lucid.Tensor)
            assert float((got - model(x)).abs().max().item()) < 1e-6
        finally:
            cm.close()
