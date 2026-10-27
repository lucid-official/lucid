"""Every option, crossed with every other option.

Each feature of this subsystem has tests. Their combinations did not,
and that is where the last three defects of this kind came from: a
float16 body with an integer output, constant folding beside the rank-6
rewrite, constant folding beside carried state. Each was a pair that
worked separately.

Writing this file found a fourth. Palettizing a float16 export failed
outright — the weight is rounded to half before the palette is fitted,
and the edge table's sentinel is single precision, so Lloyd's algorithm
tried to concatenate the two. Nothing in the single-feature tests could
see it, because neither feature is wrong on its own.

The matrix is small on purpose. Thirty-two exports of a six-layer model
run in about a minute; a matrix that took ten would be one nobody runs.
What matters is that the axes are the ones that interact — how the body
computes, how the weights are stored, and what shape of interface the
package presents.
"""

import dataclasses
import itertools

import pytest

import lucid
import lucid.nn as nn
import lucid.coreml as cml
from lucid._C import engine as _C_engine

pytestmark = pytest.mark.skipif(
    not hasattr(_C_engine, "coreml"), reason="the engine was built without the Core ML writer"
)

#: How the body computes.
PRECISIONS = [("fp32", cml.Precision.FLOAT32), ("fp16", cml.Precision.FLOAT16)]

#: How the weights are stored. Each takes a different path through the
#: writer: plain blob, int8 codes with per-channel scales, sub-byte keys
#: into grouped tables, survivors plus a bit mask.
WEIGHTS = [
    ("float", cml.WeightPrecision.FLOAT),
    ("int8", cml.WeightPrecision.INT8),
    ("palette", cml.Palettize(bits=4)),
    ("sparse", cml.Sparsify(ratio=0.5)),
]

#: What the package presents. A flexible input, a ranged one and a
#: classifier each rewrite the interface after the body is built.
INTERFACES = [
    ("fixed", {}),
    ("enumerated", {"shapes": [(1, 3, 32, 32), (2, 3, 32, 32)]}),
    ("ranged", {"shape_range": {0: (1, 4)}}),
    ("classifier", {"classifier": cml.Classifier(labels=[f"c{i}" for i in range(5)])}),
]

CASES = list(itertools.product(PRECISIONS, WEIGHTS, INTERFACES))


class _Net(nn.Module):
    """Deep enough to compress, small enough to export thirty-two times.

    A convolution stack and a linear head: between them they reach the
    weight paths that differ — a four-dimensional weight with an output
    channel axis, and a two-dimensional one.
    """

    def __init__(self) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
        )
        self.head = nn.Linear(32, 5)

    def forward(self, x: lucid.Tensor) -> lucid.Tensor:
        return self.head(self.body(x).mean(dim=(2, 3)))


@pytest.mark.parametrize(
    ("precision", "weights", "interface"), CASES, ids=[f"{p[0]}-{w[0]}-{i[0]}" for p, w, i in CASES]
)
def test_a_combination_exports_and_runs(precision, weights, interface, tmp_path):
    """Built, loaded, and asked for an answer.

    The check is deliberately that it *runs*, not that it agrees closely:
    four-bit palettization and half sparsity are lossy by design, and a
    tolerance loose enough to admit them would admit anything. What the
    combinations break is the writing and the loading, which either
    happen or do not.
    """
    _precision_name, precision_value = precision
    _weights_name, weights_value = weights
    interface_name, interface_kwargs = interface

    lucid.manual_seed(0)
    model = _Net().eval()
    x = lucid.randn(1, 3, 32, 32)

    exported = cml.export(
        model,
        x,
        str(tmp_path / "combination.mlpackage"),
        precision=precision_value,
        weights=weights_value,
        **interface_kwargs,
    )
    try:
        if interface_name == "classifier":
            label, probabilities = exported.classify(x)
            assert label in {f"c{i}" for i in range(5)}
            assert len(probabilities) == 5
        else:
            got = exported.predict(x)
            assert tuple(got.shape) == (1, 5)
    finally:
        exported.close()


class _Carries(nn.Module):
    """Reads the carried value, returns the new one beside a result."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(64, 64)

    def forward(self, x: lucid.Tensor, cache: lucid.Tensor) -> tuple[lucid.Tensor, lucid.Tensor]:
        carried = cache + self.fc(x)
        return carried, carried * 2.0


@pytest.mark.parametrize(("name", "weights"), WEIGHTS, ids=[name for name, _w in WEIGHTS])
def test_carried_state_crossed_with_weight_storage(name, weights, tmp_path):
    """State raises the opset, and the weight encodings have to follow.

    Sparsity is written one way in a ``CoreML7`` program and another in
    a ``CoreML8`` one, and carrying state is enough to make it the
    second. Core ML's loader does not reject the older spelling there —
    it segfaults on it, which is how this pair was found: a process that
    disappears with no exception and no message.
    """
    lucid.manual_seed(0)
    model = _Carries().eval()
    example = {"x": lucid.ones(1, 64), "cache": lucid.zeros(1, 64)}

    exported = cml.export(
        model,
        example,
        str(tmp_path / f"state_{name}.mlpackage"),
        precision=cml.Precision.FLOAT16,
        weights=weights,
        state=[cml.State(input="cache", output="output_0")],
    )
    try:
        assert exported.deployment_target is cml.DeploymentTarget.IOS18
        assert tuple(exported.predict({"x": lucid.ones(1, 64)}).shape) == (1, 64)
    finally:
        exported.close()


@pytest.mark.parametrize(
    ("precision", "weights"),
    list(itertools.product(PRECISIONS, WEIGHTS)),
    ids=[f"{p[0]}-{w[0]}" for p, w in itertools.product(PRECISIONS, WEIGHTS)],
)
def test_an_image_input_crossed_with_the_rest(precision, weights, tmp_path):
    """Pixels in, and the body still compressed however it was asked.

    An image input rewrites the interface after the body is built, which
    is the same place the interface axis above reaches — but through a
    different path, since Core ML refuses a multi-array for it.
    """
    _precision_name, precision_value = precision
    weights_name, weights_value = weights

    lucid.manual_seed(0)
    model = _Net().eval()
    pixels = (lucid.rand(1, 3, 32, 32) * 255).round()

    exported = cml.export(
        model,
        pixels,
        str(tmp_path / f"image_{weights_name}.mlpackage"),
        precision=precision_value,
        weights=weights_value,
        image_input=cml.ImageInput(scale=1 / 255.0),
    )
    try:
        assert tuple(exported.predict(pixels).shape) == (1, 5)
    finally:
        exported.close()


@dataclasses.dataclass
class _TwoFields:
    """A dataclass output, so ``output_field`` has something to choose."""

    logits: lucid.Tensor
    features: lucid.Tensor


class _Branching(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(), nn.Conv2d(64, 64, 3, padding=1), nn.ReLU()
        )
        self.head = nn.Linear(64, 5)

    def forward(self, x: lucid.Tensor) -> _TwoFields:
        features = self.body(x).mean(dim=(2, 3))
        return _TwoFields(logits=self.head(features), features=features)


@pytest.mark.parametrize(
    ("precision", "weights"),
    list(itertools.product(PRECISIONS, WEIGHTS)),
    ids=[f"{p[0]}-{w[0]}" for p, w in itertools.product(PRECISIONS, WEIGHTS)],
)
def test_pixels_in_and_labels_out_at_every_storage(precision, weights, tmp_path):
    """Three rewrites of the interface at once.

    An image input replaces the feed, a classifier replaces the outputs,
    and ``output_field`` chooses which of a dataclass's tensors either of
    them acts on. All three happen after the body is built, and the body
    is compressed differently in each case.
    """
    _precision_name, precision_value = precision
    weights_name, weights_value = weights

    lucid.manual_seed(0)
    model = _Branching().eval()
    pixels = (lucid.rand(1, 3, 32, 32) * 255).round()

    exported = cml.export(
        model,
        pixels,
        str(tmp_path / f"three_{weights_name}.mlpackage"),
        precision=precision_value,
        weights=weights_value,
        output_field="logits",
        image_input=cml.ImageInput(scale=1 / 255.0),
        classifier=cml.Classifier(labels=[f"c{i}" for i in range(5)]),
    )
    try:
        label, probabilities = exported.classify(pixels)
        assert label in {f"c{i}" for i in range(5)}
        assert len(probabilities) == 5
    finally:
        exported.close()


@pytest.mark.parametrize(
    ("precision", "weights"),
    list(itertools.product(PRECISIONS, WEIGHTS)),
    ids=[f"{p[0]}-{w[0]}" for p, w in itertools.product(PRECISIONS, WEIGHTS)],
)
def test_a_chosen_field_beside_metadata(precision, weights, tmp_path):
    """The other field, and the two axes nothing else crosses.

    Metadata is inert by design — it changes the description and not the
    program — which is exactly why it is worth one pass: an axis assumed
    to be inert is one nobody checks.
    """
    _precision_name, precision_value = precision
    weights_name, weights_value = weights

    lucid.manual_seed(0)
    model = _Branching().eval()
    x = lucid.randn(1, 3, 32, 32)

    exported = cml.export(
        model,
        x,
        str(tmp_path / f"field_{weights_name}.mlpackage"),
        precision=precision_value,
        weights=weights_value,
        output_field="features",
        metadata=cml.Metadata(
            description="a branching model", author="lucid", license="MIT", version="1"
        ),
    )
    try:
        assert tuple(exported.predict(x).shape) == (1, 64)
    finally:
        exported.close()


def test_a_lossless_combination_still_agrees(tmp_path: object) -> None:
    """One case where closeness is meaningful, to keep the rest honest.

    The matrix above only asks whether each combination runs, which
    would pass for a package that runs and returns nonsense. Float32
    with plain weights has no reason to differ from the model at all, so
    it is checked against it — if the writing were broken in a way the
    matrix cannot see, this is where it shows.
    """
    lucid.manual_seed(0)
    model = _Net().eval()
    x = lucid.randn(1, 3, 32, 32)

    exported = cml.export(model, x, str(tmp_path / "exact.mlpackage"))
    try:
        assert exported.verify(model, x, relative=True) < 1e-5
    finally:
        exported.close()


def test_palettizing_a_half_precision_export(tmp_path: object) -> None:
    """The defect this file was written to catch, kept by name.

    The weight is rounded to half before its palette is fitted, and the
    edge table's sentinel is single precision — so the fit tried to
    concatenate the two and the export failed outright. Palettes are
    fitted in single precision now, which the weight has already been
    rounded past, so nothing is lost by it.
    """
    lucid.manual_seed(0)
    model = _Net().eval()
    x = lucid.randn(1, 3, 32, 32)

    errors = {}
    for bits in (2, 4, 8):
        exported = cml.export(
            model,
            x,
            str(tmp_path / f"half_{bits}.mlpackage"),
            precision=cml.Precision.FLOAT16,
            weights=cml.Palettize(bits=bits),
        )
        try:
            errors[bits] = exported.verify(model, x, relative=True)
        finally:
            exported.close()
    # More bits, less error — a fit that had gone wrong would not be
    # ordered.
    assert errors[8] < errors[4] < errors[2]


# ── the axes added after this file was written ───────────────────────────────
#
# Lifting a draw to an input and training against the compression are
# both new, and neither had been crossed with anything. Running the
# crossing found three defects in the first, all of the same shape: a
# lifted draw is an input of the *package* and the interface logic counts
# the inputs of the *model*.
#
#   - a flexible shape, a range and an image input each refused, saying
#     the model took two inputs when it took one
#   - the image declaration was applied to every input, so the noise was
#     declared as the picture and the export refused, naming the noise's
#     shape as though the caller had passed it
#   - the flexible-shape probe traces at several sizes and compares each
#     operation's attributes; a draw records its seed and had drawn
#     again, so every flexible export of a model that samples was refused
#     for a seed that changed with nothing
#
# None of the three is reachable without crossing two features. That is
# the argument for this file existing.

_DRAW_INTERFACES = [
    ("fixed", {}),
    ("enumerated", {"shapes": [(1, 3, 32, 32), (2, 3, 32, 32)]}),
    ("ranged", {"shape_range": {0: (1, 4)}}),
    ("classifier", {"classifier": cml.Classifier(labels=[f"c{i}" for i in range(5)])}),
    ("image", {"image_input": cml.ImageInput(scale=1 / 255.0)}),
]

_DRAW_CASES = list(itertools.product(PRECISIONS, WEIGHTS, _DRAW_INTERFACES))


class _Samples(nn.Module):
    """Large enough to compress, and it draws — a variational head."""

    def __init__(self) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.Conv2d(32, 32, 3, padding=1), nn.ReLU()
        )
        self.mu = nn.Linear(32, 5)
        self.logvar = nn.Linear(32, 5)

    def forward(self, x: lucid.Tensor) -> lucid.Tensor:
        pooled = self.body(x).mean(dim=(2, 3))
        spread = (self.logvar(pooled) * 0.5).exp()
        return self.mu(pooled) + spread * lucid.randn(1, 5)


@pytest.mark.parametrize(
    ("precision", "weights", "interface"),
    _DRAW_CASES,
    ids=[f"{p[0]}-{w[0]}-{i[0]}" for p, w, i in _DRAW_CASES],
)
def test_a_lifted_draw_crossed_with_the_interface(precision, weights, interface, tmp_path):
    """The draw survives every way the package can present itself."""
    _precision_name, precision_value = precision
    _weights_name, weights_value = weights
    interface_name, interface_kwargs = interface

    lucid.manual_seed(0)
    model = _Samples().eval()
    x = (
        (lucid.rand(1, 3, 32, 32) * 255).round()
        if interface_name == "image"
        else lucid.randn(1, 3, 32, 32)
    )

    exported = cml.export(
        model,
        x,
        str(tmp_path / "drawn.mlpackage"),
        precision=precision_value,
        weights=weights_value,
        draws=cml.Draws.AS_INPUT,
        **interface_kwargs,
    )
    try:
        assert exported.noise_inputs == [("noise_0", (1, 5))]
        if interface_name == "classifier":
            label, probabilities = exported.classify(x)
            assert label in {f"c{i}" for i in range(5)}
            first = probabilities["c0"]
            second = exported.classify(x)[1]["c0"]
        else:
            assert tuple(exported.predict(x).shape) == (1, 5)
            first = float(exported.predict(x).sum().item())
            second = float(exported.predict(x).sum().item())
        # Still a sampler: two predictions of one input must differ, or
        # the draw was folded after all and the package is frozen.
        assert first != second
    finally:
        exported.close()


class _CarriesAndDraws(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(64, 64)

    def forward(self, x: lucid.Tensor, cache: lucid.Tensor) -> tuple[lucid.Tensor, lucid.Tensor]:
        carried = cache + self.fc(x) + lucid.randn(1, 64)
        return carried, carried * 2.0


@pytest.mark.parametrize(("name", "weights"), WEIGHTS, ids=[name for name, _w in WEIGHTS])
def test_a_lifted_draw_beside_carried_state(name, weights, tmp_path):
    """Both raise the opset, and a stochastic decoder wants both.

    The caller supplies a sample per step and the package keeps its cache
    between them, which is what sampling from a recurrent latent is.
    """
    lucid.manual_seed(0)
    exported = cml.export(
        _CarriesAndDraws().eval(),
        {"x": lucid.ones(1, 64), "cache": lucid.zeros(1, 64)},
        str(tmp_path / f"drawn_state_{name}.mlpackage"),
        precision=cml.Precision.FLOAT16,
        weights=weights,
        draws=cml.Draws.AS_INPUT,
        state=[cml.State(input="cache", output="output_0")],
    )
    try:
        assert exported.noise_inputs == [("noise_0", (1, 64))]
        assert exported.deployment_target is cml.DeploymentTarget.IOS18
        assert tuple(exported.predict({"x": lucid.ones(1, 64)}).shape) == (1, 64)
    finally:
        exported.close()


_AWARE_CASES = list(itertools.product(PRECISIONS, WEIGHTS[1:]))


@pytest.mark.parametrize(
    ("precision", "weights"), _AWARE_CASES, ids=[f"{p[0]}-{w[0]}" for p, w in _AWARE_CASES]
)
def test_training_against_the_compression_then_exporting_with_it(precision, weights, tmp_path):
    """A settled model has to export as the compression it was settled to.

    Float32 is where this is exact — the weights are already on the
    palette — so the tolerance is tight there and loose at float16, whose
    own rounding is the larger term. INT8 is loose at both because the
    export re-derives its scale on a half-step grid, which is measured in
    ``test_compression_aware``.
    """
    precision_name, precision_value = precision
    weights_name, weights_value = weights

    lucid.manual_seed(0)
    aware = cml.CompressionAware(_Net().eval(), weights=weights_value)
    aware(lucid.randn(2, 3, 32, 32)).sum().backward()
    aware.refit()
    settled = aware.settle()

    x = lucid.randn(1, 3, 32, 32)
    exported = cml.export(
        settled,
        x,
        str(tmp_path / f"aware_{weights_name}.mlpackage"),
        precision=precision_value,
        weights=weights_value,
    )
    try:
        loose = weights_name == "int8" or precision_name == "fp16"
        assert exported.verify(settled, x, relative=True) < (5e-2 if loose else 1e-4)
    finally:
        exported.close()


@pytest.mark.parametrize(("name", "weights"), WEIGHTS[1:], ids=[name for name, _w in WEIGHTS[1:]])
def test_a_compressed_model_that_also_draws(name, weights, tmp_path):
    """The two new axes against each other.

    A variational encoder is the model that wants both: it samples, and
    it is the kind of thing shipped compressed.
    """
    lucid.manual_seed(0)
    aware = cml.CompressionAware(_Samples().eval(), weights=weights)
    aware(lucid.randn(2, 3, 32, 32)).sum().backward()
    settled = aware.settle()

    x = lucid.randn(1, 3, 32, 32)
    exported = cml.export(
        settled,
        x,
        str(tmp_path / f"aware_drawn_{name}.mlpackage"),
        weights=weights,
        draws=cml.Draws.AS_INPUT,
    )
    try:
        assert exported.noise_inputs == [("noise_0", (1, 5))]
        assert exported.verify(settled, x, relative=True) < (5e-2 if name == "int8" else 1e-3)
    finally:
        exported.close()
