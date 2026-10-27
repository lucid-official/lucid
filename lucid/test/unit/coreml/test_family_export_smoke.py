"""One model per architecture family, exported and compared.

The rest of this directory tests operations one at a time and models
three at a time. That is what let a `gather` emitted for the wrong
opset, a rank cap that refused every windowed transformer, and an
operation listed as "never traced" that LP pooling traces, all sit in
the tree at once: each was reachable only from an architecture nothing
exported.

So this walks families rather than operations. It is deliberately not
the whole zoo — 440 factories at inference size is not a unit test — but
one representative per family, at a small input, is enough to keep the
class of defect from returning silently. Everything here passed when it
was written; a new entry that fails is a real gap, not a flaky test.

Families whose input is not a single image tensor (detection heads with
multiple inputs, generative samplers, world models) are named in
``NOT_SINGLE_IMAGE`` rather than left out, so the list says what it does
not cover.
"""

import pytest

import lucid
import lucid.coreml as cml
import lucid.models as M
from lucid._C import engine as _C_engine

pytestmark = pytest.mark.skipif(
    not hasattr(_C_engine, "coreml"),
    reason="the engine was built without the Core ML writer",
)

#: The bound every single-image family is held to.  There is no exception
#: for the hosted runner: inside its virtual machine the conftest refuses
#: to load any package, because Core ML there runs everything on BNNS's CPU
#: path.  ZFNet, the zoo's one model with local response normalisation,
#: landed 1.67e-3 from eager on that path under every compute-unit
#: setting, against 1.7e-6 on an M1 Pro.
_BOUND = 1e-4

#: One factory per family, with the input it takes: ``img`` for pixels
#: and ``ids`` for token indices. Small on purpose — this is checking
#: that the translation holds, not how fast it runs.
FAMILIES = [
    ("alexnet", (1, 3, 224, 224)),
    ("convnext_tiny", (1, 3, 224, 224)),
    ("crossvit_9", (1, 3, 224, 224)),
    ("cspdarknet_53", (1, 3, 224, 224)),
    ("coatnet_0", (1, 3, 224, 224)),
    ("cspresnet_50", (1, 3, 224, 224)),
    ("cspresnext_50", (1, 3, 224, 224)),
    ("cvt_13", (1, 3, 224, 224)),
    ("densenet_121", (1, 3, 224, 224)),
    ("efficientformer_l1", (1, 3, 224, 224)),
    ("efficientnet_b0", (1, 3, 224, 224)),
    ("googlenet", (1, 3, 224, 224)),
    ("inception_v3", (1, 3, 224, 224)),
    ("lenet_5", (1, 1, 32, 32)),
    # Segmentation: one channel in, a map out.
    ("unet", (1, 1, 64, 64)),
    ("res_unet_2d", (1, 1, 64, 64)),
    ("attention_unet", (1, 1, 64, 64)),
    ("maxvit_tiny", (1, 3, 224, 224)),
    ("mobilenet", (1, 3, 224, 224)),
    # The hybrid, so Mobile MQA's shared-head attention is translated too.
    ("mobilenet_v4_hybrid_medium", (1, 3, 224, 224)),
    ("pvt_v2_b0", (1, 3, 224, 224)),
    ("resnest_14", (1, 3, 224, 224)),
    ("resnet_18", (1, 3, 224, 224)),
    ("resnext_50_32x4d", (1, 3, 224, 224)),
    ("se_resnet_18", (1, 3, 224, 224)),
    ("sk_resnet_18", (1, 3, 224, 224)),
    ("swin_tiny", (1, 3, 224, 224)),
    ("vgg_11", (1, 3, 224, 224)),
    ("vit_base_16", (1, 3, 224, 224)),
    # The probe, not the pretraining model: the latter samples its masks
    # and answers with a loss.
    ("ijepa_base_16_cls", (1, 3, 224, 224)),
    ("wide_resnet_50", (1, 3, 224, 224)),
    ("xception", (1, 3, 224, 224)),
    ("zfnet", (1, 3, 224, 224)),
]

#: Families this file used to name in ``NOT_SINGLE_IMAGE`` on grounds
#: that turned out to be false. Each takes a single tensor and exports;
#: they are separated only because they return several outputs — boxes
#: beside scores, a reconstruction beside a latent — so the comparison
#: goes through ``verify`` rather than through one tensor.
MULTI_OUTPUT = [
    ("fcn_resnet101", (1, 3, 224, 224)),
    ("yolo", (1, 3, 448, 448)),
    ("detr_resnet101", (1, 3, 224, 224)),
    ("vqvae", (1, 3, 64, 64)),
    ("nice_cifar", (1, 3072)),
    # Segmentation and detection heads that were excluded with the
    # two-stage detectors and are not two-stage: one pass, no proposals.
    ("maskformer_resnet101", (1, 3, 224, 224)),
    # Smaller than its native 512: the translation is what is under
    # test, and the suite has to fit in memory beside everything else.
    ("efficientdet_d0", (1, 3, 256, 256)),
    # Deformable attention computes at rank 6; the driver emits that
    # component a rank lower, since every value in it carries a batch
    # axis of one.
    ("mask2former_swin_base", (1, 3, 224, 224)),
]

#: Families whose input is token indices rather than pixels.
TOKEN_FAMILIES = [
    ("gpt", (1, 16)),
    # Two heads, so ``predict`` answers with a dict keyed by field name
    # rather than one tensor — the comparison goes through ``verify``.
    ("bert_tiny", (1, 16)),
    # Was excluded for loading CPU_ONLY: Core ML's planner refused the
    # graph on the accelerators. Constant folding left it a graph the
    # planner accepts, everywhere.
    ("roformer", (1, 16)),
]

#: Families whose forward takes more than one tensor: a diffusion network
#: is a function of a sample *and* a time, and a sequence-to-sequence
#: model of two token streams. The sampler loop around them is Python and
#: does not export — the network it calls is what a deployment runs, and
#: that is what these check.
#:
#: Several of these zero-initialise their output projection, which is
#: ordinary practice and makes an untrained model answer with exact
#: zeros; ``verify`` refuses that comparison, correctly, so the fixture
#: perturbs the zeroed parameters first.
MULTI_INPUT = [
    ("ddpm_cifar", lambda: (lucid.randn(1, 3, 32, 32), lucid.zeros(1).to(lucid.int64))),
    ("flow_matching_cifar", lambda: (lucid.randn(1, 3, 32, 32), lucid.zeros(1))),
    (
        "ncsn_celeba",
        lambda: (lucid.randn(1, 3, 32, 32), lucid.zeros(1).to(lucid.int64)),
    ),
    ("dit_base_2", lambda: (lucid.randn(1, 4, 32, 32), lucid.zeros(1).to(lucid.int64))),
    (
        "mean_flow_base_2",
        lambda: (lucid.randn(1, 4, 32, 32), lucid.zeros(1), lucid.zeros(1)),
    ),
    (
        "transformer_base",
        lambda: (
            lucid.zeros(1, 16).to(lucid.int64),
            lucid.zeros(1, 16).to(lucid.int64),
        ),
    ),
    # The same two streams; T5's position bias depends on their lengths
    # only, so the trace fixes it along with the shapes.
    (
        "t5_small",
        lambda: (
            lucid.zeros(1, 16).to(lucid.int64),
            lucid.zeros(1, 16).to(lucid.int64),
        ),
    ),
    # Pixels beside token ids, and a mixed-dtype multiply Core ML will
    # not promote — the export inserts the cast now.
    (
        "clip_vit_base_16",
        lambda: (lucid.randn(1, 3, 224, 224), lucid.zeros(1, 77).to(lucid.int64)),
    ),
    # Upsamples by splitting an axis, padding zeros into it and merging
    # back, which wants rank 6 for two axes at once. Staged one axis at a
    # time so no step passes the cap. At 64 rather than its native 256,
    # like efficientdet above: at 256 this one export peaked near 7 GB on
    # an M1 Pro, and at 64 it peaks at 1.65 GB through the same staged
    # upsample (verify 1.5e-08).
    ("rectified_flow_afhq_cat", lambda: (lucid.randn(1, 3, 64, 64), lucid.zeros(1))),
    # A latent, the two times and a label. Built small through ``SMALL``:
    # the paper's DiT-XL/2 is its only size.
    (
        "tvm_imagenet256",
        lambda: (
            lucid.randn(1, 4, 8, 8),
            lucid.ones(1),
            lucid.zeros(1),
            lucid.zeros(1).to(lucid.int64),
        ),
    ),
]

#: Factories whose only sizes are too large to export in a unit test, built
#: through config overrides on the real factory — never an invented variant.
SMALL: dict[str, dict[str, object]] = {
    # 678M parameters at the paper's size: a package in gigabytes.
    "tvm_imagenet256": {
        "sample_size": 8,
        "hidden_size": 32,
        "depth": 1,
        "num_heads": 4,
        "num_classes": 4,
    },
}

#: Families whose ``forward`` draws random numbers, exported with the
#: draw lifted to an input the caller fills — see
#: :class:`~lucid.coreml.Draws`. Every one of these refused outright
#: before that existed, and they were the largest single group of
#: families this file could not reach.
#:
#: The comparison is against what the trace answered rather than against
#: a fresh eager run, because a fresh run draws different numbers; the
#: package is fed the sample the trace used.
SAMPLING = [
    ("vae", lambda: lucid.randn(1, 3, 32, 32)),
    ("hvae", lambda: lucid.randn(1, 3, 32, 32)),
    ("score_sde_ve", lambda: (lucid.randn(1, 3, 32, 32), lucid.zeros(1))),
    # World models: a rollout of observations beside the actions taken,
    # with a stochastic latent at every step — eight draws for dreamer
    # and planet, which is what made them look like a boundary.
    ("dreamer", lambda: (lucid.randn(1, 4, 3, 64, 64), lucid.randn(1, 4, 1))),
    ("planet", lambda: (lucid.randn(1, 4, 3, 64, 64), lucid.randn(1, 4, 1))),
    (
        "diamond",
        lambda: (
            lucid.randn(1, 4, 3, 64, 64),
            lucid.zeros(1, 4).to(lucid.int64),
            lucid.randn(1, 3, 64, 64),
        ),
    ),
]

#: Detectors that take their region proposals rather than making them.
#: Their forward reads each box's corners on the host (``.item()``) to
#: crop it, so a traced program would carry the example's boxes as
#: constants and crop the same windows for every input. They exported
#: and verified — on the very proposals they were traced with — until
#: export refused host reads (LCD-337); now the refusal is the contract.
WITH_PROPOSALS = [
    (
        "fast_rcnn",
        lambda: (
            lucid.randn(1, 3, 64, 64),
            lucid.tensor([[[0.0, 0.0, 32.0, 32.0], [8.0, 8.0, 48.0, 48.0]]]),
        ),
    ),
    (
        "rcnn",
        lambda: (
            lucid.randn(1, 3, 64, 64),
            lucid.tensor([[[0.0, 0.0, 32.0, 32.0], [8.0, 8.0, 48.0, 48.0]]]),
        ),
    ),
]

#: Families this file does not reach, and why. Listed rather than
#: omitted: an absence with no reason attached reads as an oversight
#: later, and some of these are boundaries rather than gaps.
#:
#: The list used to be twice this size, and most of what left it was
#: never a boundary — ``fcn`` was excluded for a shape it does not want,
#: ``yolo`` and ``detr`` for post-processing that is not in their
#: forward at all, and the diffusion families for a sampler loop that
#: wraps the network rather than being inside it. Each entry below now
#: names something that was observed, not assumed.
NOT_SINGLE_IMAGE = {
    # Keys, not factory names: the family of ``mask2former_swin_base`` is
    # ``mask``, the same as ``mask_rcnn``'s. Crude on purpose — the point
    # is that nothing drops out of sight, not that the taxonomy is exact.
    #
    # ``faster_rcnn`` and ``mask_rcnn`` do not finish tracing — over four
    # minutes on this input. Their eager forward returns in seconds; it
    # is the recording that does not scale, because non-maximum
    # suppression is a Python loop over thousands of small indexing
    # operations and every one of them becomes a traced node. A graph
    # built from it would be that one input's suppression unrolled.
    #
    # ``mask_rcnn`` is the same story, but its family key is ``mask``,
    # which ``mask2former_swin_base`` covers above — so it does not
    # appear here and this comment is the only record of it.
    #
    # ``fast_rcnn`` and ``rcnn`` used to be here on the grounds that they
    # trace to a graph holding an empty constant. That was the call, not
    # the model: passing only an image left ``proposals`` at its default
    # and sent them down a path the trace could not follow. Given
    # proposals they read the boxes on the host, so export refuses them —
    # ``WITH_PROPOSALS`` above holds that refusal.
    "faster",
    # ``neural_ode`` exports its draws now — that was what stopped it —
    # and stops on ``rk_combine`` instead, which is the solver's own
    # step and has no emitter. A translation gap, and a named one.
    "neural",
    # ``stable_diffusion`` takes an image, a text context and a
    # timestep, and its default configuration is 512 pixels square:
    # covering it needs an agreed small configuration first, not a
    # translation it is missing.
    "stable",
    # ``vjepa`` exports cleanly at a small configuration — a 64-pixel,
    # four-frame probe traces and verifies to 2e-07 — but the factories
    # here are built with their defaults, and V-JEPA's smallest is a pair
    # of ViT-L encoders over a 16-frame clip: 630M parameters, and a
    # package in gigabytes.  Covering it needs an agreed small
    # configuration first, as ``stable`` does.
    "vjepa",
    # ``genie``'s ``forward`` is its three training objectives, and a
    # tiny configuration stops on ``mse_loss``, which has no Core ML
    # translation.  A packaged training loss has no caller; the playable
    # rollout is a MaskGIT loop in Python around the dynamics model and
    # was not tried.
    "genie",
    # Covered above by a representative whose family key differs:
    # ``nice_cifar`` stands for the flow family, ``vqvae`` for itself,
    # ``ddpm``/``flow``/``ncsn``/``dit``/``mean`` and
    # ``transformer``/``bert`` are in the lists above.
    #
    # ``realnvp`` exports too, but an untrained one sits at the edge of
    # its coupling's ``exp`` and a small perturbation puts it over — a
    # property of the architecture untrained, not of the export.
    "realnvp",
    "bert",
    # ``roformer`` exports and is a well-formed package, but Core ML
    # will not build an execution plan for it on the GPU or the Neural
    # Engine — it loads with CPU_ONLY.  That is the accelerator planner
    # refusing the graph, not a translation gap, and the runtime now
    # says so by name rather than reporting "Error in building plan".
    "roformer",
}


def _tensor_of(result: object) -> lucid.Tensor:
    """Zoo factories return output dataclasses as often as tensors."""
    if isinstance(result, lucid.Tensor):
        return result
    for field in ("logits", "out", "output", "last_hidden_state"):
        value = getattr(result, field, None)
        if isinstance(value, lucid.Tensor):
            return value
    for value in vars(result).values():
        if isinstance(value, lucid.Tensor):
            return value
    raise AssertionError(f"no tensor in {type(result).__name__}")


@pytest.mark.parametrize(("factory", "shape"), FAMILIES, ids=[f for f, _s in FAMILIES])
def test_a_family_representative_exports_and_matches(factory, shape, tmp_path):
    lucid.manual_seed(0)
    model = M.create_model(factory).eval()
    x = lucid.randn(*shape)
    reference = _tensor_of(model(x))

    # Some untrained factories answer with almost nothing —
    # EfficientNet's logits come out around 1e-12, MobileNet's and
    # GoogLeNet's around 1e-10 — because their heads are
    # zero-initialised. Dividing by a floored scale then compares noise
    # against noise and passes whatever the package does, which is how
    # three families sat in this list for weeks proving nothing.
    if float(reference.abs().max().item()) < 1e-6:
        model = _without_zero_initialised_parameters(factory)
        reference = _tensor_of(model(x))
    scale = float(reference.abs().max().item())
    assert scale > 1e-6, (
        f"{factory} answers with values too small to compare even after its "
        "zero-initialised parameters were perturbed — this test cannot see "
        "anything about it"
    )

    exported = cml.export(model, x, str(tmp_path / f"{factory}.mlpackage"))
    try:
        got = exported.predict(x)
        assert tuple(got.shape) == tuple(reference.shape)
        assert float((got - reference).abs().max().item()) / scale < _BOUND
    finally:
        exported.close()


@pytest.mark.parametrize(("factory", "shape"), TOKEN_FAMILIES, ids=[f for f, _s in TOKEN_FAMILIES])
def test_a_token_model_exports_and_matches(factory, shape, tmp_path):
    """Indices in, not pixels — a different feed dtype end to end."""
    lucid.manual_seed(0)
    model = M.create_model(factory).eval()
    x = lucid.zeros(*shape).to(lucid.int64)
    reference = _tensor_of(model(x))
    scale = max(float(reference.abs().max().item()), 1e-6)

    exported = cml.export(model, x, str(tmp_path / f"{factory}.mlpackage"))
    try:
        # ``verify`` rather than ``predict``: a model with two heads
        # answers with a dict, and the comparison should not depend on
        # how many the factory happens to have.
        assert exported.verify(model, x) / scale < 1e-4
    finally:
        exported.close()


def _create(factory: str) -> lucid.nn.Module:
    return M.create_model(factory, **SMALL.get(factory, {})).eval()


def _without_zero_initialised_parameters(factory: str) -> object:
    """The same model with its zeroed parameters given small values.

    Diffusion networks zero-initialise their output projection so that
    the residual branch starts as the identity. An untrained one
    therefore answers with exact zeros, and ``verify`` refuses that
    comparison — rightly, since an export that dropped every layer would
    score the same. Perturbing first is what the refusal asks for.
    """
    model = _create(factory)
    replaced = {}
    for name, value in model.state_dict().items():
        zeroed = (
            hasattr(value, "shape")
            and int(value.numel()) > 0
            and float(value.abs().max().item()) == 0.0
        )
        replaced[name] = lucid.randn(*value.shape) * 0.05 if zeroed else value
    perturbed = _create(factory)
    perturbed.load_state_dict(replaced)
    return perturbed


@pytest.mark.parametrize(("factory", "shape"), MULTI_OUTPUT, ids=[f for f, _s in MULTI_OUTPUT])
def test_a_multi_output_family_exports_and_matches(factory, shape, tmp_path):
    """One tensor in, several out — compared through ``verify``.

    ``relative=True`` because the outputs differ in magnitude: a flow
    returns a latent of order one beside a log-probability of order
    ten thousand, and the absolute worst across them is set by the
    second whatever the first did.
    """
    lucid.manual_seed(0)
    model = M.create_model(factory).eval()
    x = lucid.randn(*shape)

    exported = cml.export(model, x, str(tmp_path / f"{factory}.mlpackage"))
    try:
        assert exported.verify(model, x, relative=True) < 1e-4
    finally:
        exported.close()


@pytest.mark.parametrize(("factory", "make_inputs"), MULTI_INPUT, ids=[f for f, _m in MULTI_INPUT])
def test_a_multi_input_family_exports_and_matches(factory, make_inputs, tmp_path):
    """A network of a sample and a time, or of two token streams.

    ``relative=True`` because these return outputs of different
    magnitudes — a latent beside a log-probability — and the absolute
    worst across them says nothing about either.
    """
    lucid.manual_seed(0)
    model = _without_zero_initialised_parameters(factory)
    inputs = make_inputs()

    exported = cml.export(model, inputs, str(tmp_path / f"{factory}.mlpackage"))
    try:
        assert exported.verify(model, inputs, relative=True) < 1e-4
    finally:
        exported.close()


@pytest.mark.parametrize(("factory", "make_inputs"), SAMPLING, ids=[f for f, _m in SAMPLING])
def test_a_sampling_family_exports_with_its_draws_lifted(factory, make_inputs, tmp_path):
    """A draw is not part of the network, so it can be an input."""
    lucid.manual_seed(0)
    model = _without_zero_initialised_parameters(factory)
    inputs = make_inputs()

    exported = cml.export(
        model,
        inputs,
        str(tmp_path / f"{factory}.mlpackage"),
        draws=cml.Draws.AS_INPUT,
    )
    try:
        assert exported.noise_inputs
        assert exported.verify(model, inputs, relative=True) < 1e-4
    finally:
        exported.close()


@pytest.mark.parametrize(
    ("factory", "make_inputs"), WITH_PROPOSALS, ids=[f for f, _m in WITH_PROPOSALS]
)
def test_a_detector_that_reads_its_proposals_is_refused(factory, make_inputs, tmp_path):
    lucid.manual_seed(0)
    model = M.create_model(factory).eval()
    with pytest.raises(ValueError, match="read on the host"):
        cml.export(model, make_inputs(), str(tmp_path / f"{factory}.mlpackage"))


def test_every_family_is_either_covered_or_named():
    """No family drops out of sight.

    A representative here, or an entry in ``NOT_SINGLE_IMAGE`` saying
    what shape it wants instead. Adding a family to the zoo without
    doing either fails this.
    """
    import re

    covered = {
        re.split(r"[_\d]", f)[0]
        for f, _s in (
            *FAMILIES,
            *TOKEN_FAMILIES,
            *MULTI_OUTPUT,
            *MULTI_INPUT,
            *SAMPLING,
            *WITH_PROPOSALS,
        )
    }
    families = {re.split(r"[_\d]", n)[0] for n in M.list_models()}
    missing = families - covered - NOT_SINGLE_IMAGE
    assert not missing, (
        f"architecture families with no export coverage and no reason given: " f"{sorted(missing)}"
    )
