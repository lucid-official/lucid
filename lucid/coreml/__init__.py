"""``lucid.coreml`` — export Lucid models to Core ML, and run them on the ANE.

Lucid computes on Accelerate (CPU) and MLX (Metal).  Neither targets the
Neural Engine, so a whole processor on every machine Lucid supports was
unreachable; the same gap meant a model trained here could not ship inside
an iOS or macOS app.  Core ML is the only public route to either.

**No third-party dependency.**  The package format is written by
``lucid/_C/coreml`` — the MIL protobuf, the weight blob, the bundle —
and executed through Apple's own CoreML.framework, which stands beside
Accelerate and Metal rather than beside a pip package.  Nothing under
``lucid/`` imports anything external (H4).

**Two things worth knowing before trusting an export.**

*The Neural Engine only runs float16.*  A float32 program asked for with
``ComputeUnits.CPU_AND_NE`` does not warn, does not error, and runs at CPU
speed — measured: zero of its operations are scheduled on the ANE.  Pass
``Precision.FLOAT16`` to actually reach it, and expect the ~1e-4 that
half precision costs.  :meth:`CoreMLModel.compute_plan` reports where the
operations really landed, so this is checkable rather than inferred from a
stopwatch.

*Coverage is narrow and loud.*  The mapped operations are the ones real
models were measured to emit; anything else raises
:class:`UnsupportedOp` naming the gap, because a package quietly missing a
layer still loads and still returns plausible numbers.

**Smaller packages.**  ``weights=`` stores the weights compressed —
:class:`WeightPrecision` for eight bits with a scale per output channel,
:class:`Palettize` for one to eight bits through a lookup table,
:class:`Sparsify` for the survivors plus a mask.  Measured on a trained
ResNet-50, against its top-1 over five inputs: int8 is 3.9x smaller and
keeps 4/5, six-bit palettization 4.3x and 4/5.  Below six bits
post-training palettization changes every prediction, and no export
setting recovers it — which is what :class:`CompressionAware` is for.

**Models from** ``lucid.quantization`` **export.**  A model prepared for
quantization-aware training carries observers that record what passes
through them, and tracing is not calibration, so they are paused around
it and put back; its weights were trained onto their grid, so storing
them there costs nothing — measured at 0.00e+00 against the eager model
where the same network quantized afterwards lands at 7e-04.  A converted
model — ``quantize_dynamic``, or ``convert`` — holds a packed MLX linear
whose Metal-only kernel the tracer cannot follow, and is exported through
that form's own dequantize-to-float reference path, which is the same
arithmetic to 5e-07 rather than an approximation of it.
:class:`Activations` decides whether a prepared model's *activation*
quantization is carried into the package as arithmetic or left out; Core
ML quantizes weights only, so carrying it is faithful and is work the
accelerator did not ask for.

**A package is compiled once.**  Loading one makes Core ML write a bundle
into its own cache — for a GPU segment a full copy of the weights — keyed
by where the compiled model is.  The compiled model is therefore kept, one
per package content, in ``~/.cache/lucid/coreml`` (bounded by
``LUCID_COREML_CACHE_LIMIT``, default 4G), and every later load opens that
one; see :func:`empty_cache`.  A load that would run the disk out raises
``OSError`` first, because Core ML ends the process instead.

Examples
--------
::

    import lucid, lucid.coreml as cml

    model = lucid.models.create_model("resnet_18_cls").eval()
    x = lucid.zeros(1, 3, 224, 224)

    cm = cml.export(model, x, "/tmp/r18.mlpackage",
                    precision=cml.Precision.FLOAT16,
                    compute_units=cml.ComputeUnits.CPU_AND_NE)
    cm.verify(model, x)        # max|coreml - eager|
    cm.compute_plan()          # PlacementSummary(ANE=69, CPU=2, constants=..., ane=97%)
    y = cm.predict(x)
"""

from typing import TYPE_CHECKING, cast

from lucid.coreml._build import (
    _DRAWN_KEY,
    _IO_PRECISION_KEY,
    _PRECISION_KEY,
    _packed_linears_unpacked,
    commit_staging,
    discard_staging,
    staging_path,
    _Shared,
    StatefulModel,
    UnsupportedOp,
    UnsupportedRank,
    build_package,
)
from lucid.coreml._flex import ShapeNotFlexible
from lucid.coreml._model import CoreMLModel, Latency, PlacementSummary
from lucid.coreml._optimize import CompressionAware
from lucid.coreml._spec import (
    Activations,
    DeploymentTarget,
    Draws,
    Palettize,
    Sparsify,
    Classifier,
    State,
    ColorSpace,
    ComputeUnits,
    ImageInput,
    Metadata,
    Precision,
    WeightPrecision,
)

# Imported for real, not only for the type checker.  Annotations are
# evaluated lazily (PEP 649) but they are still evaluated — against this
# module's globals — the moment anything asks for them, and
# ``inspect.signature`` asks.  Behind a ``TYPE_CHECKING`` guard the names
# are absent at runtime, so ``inspect.signature(export)``, ``help``, and
# every documentation tool raised ``NameError`` on this package's main
# entry point.
#
# Safe here because ``lucid.coreml`` is a lazily loaded subpackage: by
# the time anything reaches it, ``lucid.nn`` has been importable for a
# while.  ``TYPE_CHECKING`` stays for the checker's benefit and costs
# nothing.
from lucid._tensor.tensor import Tensor
from lucid.nn.module import Module

if TYPE_CHECKING:
    from lucid._tensor.tensor import Tensor
    from lucid.nn.module import Module

__all__ = [
    "Classifier",
    "State",
    "ColorSpace",
    "ComputeUnits",
    "ImageInput",
    "Metadata",
    "WeightPrecision",
    "Activations",
    "CompressionAware",
    "DeploymentTarget",
    "Draws",
    "Palettize",
    "Sparsify",
    "CoreMLModel",
    "Latency",
    "PlacementSummary",
    "ShapeNotFlexible",
    "StatefulModel",
    "Precision",
    "UnsupportedOp",
    "UnsupportedRank",
    "empty_cache",
    "export",
    "export_functions",
    "load",
    "precision_cost",
]


def _features(value: object) -> list[tuple[str, tuple[int, ...]]]:
    """The feature names and shapes as ``build_package`` reports them."""
    if not isinstance(value, list):
        raise TypeError(f"lucid.coreml: expected a feature list, got {type(value).__name__}")
    return [(str(name), tuple(int(d) for d in shape)) for name, shape in value]


def export(
    model: Module,
    example: object,
    path: str,
    *,
    precision: Precision = Precision.FLOAT32,
    weights: WeightPrecision | Palettize | Sparsify = WeightPrecision.FLOAT,
    shapes: list[tuple[int, ...]] | None = None,
    shape_range: dict[int, tuple[int, int]] | None = None,
    state: list[State] | None = None,
    image_input: ImageInput | None = None,
    classifier: Classifier | None = None,
    metadata: Metadata | None = None,
    compute_units: ComputeUnits = ComputeUnits.ALL,
    output_field: str | None = None,
    minimum_deployment_target: DeploymentTarget | None = None,
    draws: Draws = Draws.REFUSED,
    activations: Activations = Activations.SIMULATED,
    io_precision: Precision = Precision.FLOAT32,
) -> CoreMLModel:
    """Trace ``model``, write a ``.mlpackage`` at ``path``, and load it.

    Parameters
    ----------
    model : nn.Module
        Must be in ``eval()`` mode: an exported graph is an inference
        graph, and a training-mode dropout is refused rather than
        silently turned into an identity.
    example : Tensor or tuple of Tensor or dict of str to Tensor
        Supplies each input's shape and dtype; the values are irrelevant.
        A tuple is passed to the model positionally, a mapping by
        keyword. The exported model's input shapes are fixed to these.
    path : str
        Destination ``.mlpackage``. Replaced if it exists.
    precision : Precision, optional, keyword-only, default=FLOAT32
        Precision of the program body. ``FLOAT32`` keeps the export
        faithful to the model it came from; ``FLOAT16`` is what the
        Neural Engine runs. Inputs and outputs stay float32 either way,
        unless ``io_precision`` asks for float16.
    weights : WeightPrecision or Palettize or Sparsify, optional, keyword-only, default=WeightPrecision.FLOAT
        How weights are stored. ``INT8`` keeps eight bits per weight plus
        one scale per output channel and lets Core ML dequantize on the
        way in — the package halves against float16 and the accelerator
        moves half as much memory, at a real cost in agreement that
        :meth:`~CoreMLModel.verify` will quantify.
    shapes : list of tuple of int or None, optional, keyword-only, default=None
        Every input shape the package should accept — a range of batch
        sizes, or of resolutions — with the example's own among them.
        Found by tracing at each shape and comparing, so an operation
        whose configuration came from the input size (an adaptive pool
        bakes its kernel that way) is refused by name rather than fixed
        to one shape and wrong at the others.
    shape_range : dict of int to tuple of int or None, optional, keyword-only, default=None
        Axis to ``(lowest, highest)``, when the sizes are a range rather
        than a short list — a variable sequence length, a camera whose
        resolution changes. Axes left out keep the example's size.
        Mutually exclusive with ``shapes``.
    state : list of State or None, optional, keyword-only, default=None
        Input/output pairs Core ML should carry between predictions
        rather than exchange with the caller — a decoder's key-value
        cache. Needs iOS 18 / macOS 15, which a package asks for only
        when it uses one. The state begins at zero.
    image_input : ImageInput or None, optional, keyword-only, default=None
        Present the sole input as an image, with the normalisation it
        expects, so an app can hand over a pixel buffer instead of
        converting pixels itself.
    classifier : Classifier or None, optional, keyword-only, default=None
        Declare the model a classifier over these labels, so Core ML —
        and Vision through it — returns the winning label and a
        label-to-probability map instead of a score array. Read with
        :meth:`~CoreMLModel.classify`.
    metadata : Metadata or None, optional, keyword-only, default=None
        Description, author, licence and version to record in the package.
    compute_units : ComputeUnits, optional, keyword-only, default=ALL
        Which processors Core ML may schedule on.
    activations : Activations, optional, keyword-only, default=SIMULATED
        What to do with a quantization-aware model's activation
        fake-quantization. The default carries it as arithmetic, so the
        package reproduces the model exactly; ``DROPPED`` removes it,
        leaving quantized weights and float activations — which is what
        the Neural Engine computes anyway. Inert for a model that carries
        none.
    draws : Draws, optional, keyword-only, default=REFUSED
        What to do about a model that draws random numbers in
        ``forward``. Core ML folds a draw at build time, so the default
        refuses rather than writing a package that returns one fixed
        sample forever. ``AS_INPUT`` declares each draw as an input the
        caller fills; the handle draws for a caller who passes nothing,
        so the package still samples the way the model does.
    minimum_deployment_target : DeploymentTarget or None, optional, keyword-only, default=None
        Oldest system the package must run on. State, palettization and
        several entry points each raise that floor to ``IOS18``; naming a
        lower one refuses the export rather than producing a package that
        loads nowhere the caller intended. ``None`` accepts whatever the
        features require, and the result is reported on the model.
    output_field : str or None, optional, keyword-only, default=None
        Single attribute to export when the model returns an output
        dataclass. ``None`` exports every tensor field it declares —
        a detector's boxes and objectness as well as its class scores.
    io_precision : Precision, optional, keyword-only, default=FLOAT32
        Element type of the package's float inputs and outputs. The
        default keeps them float32 around any body, so a caller never
        has to convert. ``FLOAT16`` — which needs
        ``precision=Precision.FLOAT16`` — declares them half precision
        and drops the two casts that bracket the body: an app that
        already holds float16 buffers hands them over as they are, and
        :meth:`~CoreMLModel.predict` returns float16 tensors (a float32
        input is converted on the way in). An image input stays a pixel
        buffer, and a classifier's scores stay behind its labels.

    Returns
    -------
    CoreMLModel
        Loaded and ready to :meth:`~CoreMLModel.predict`.

    Raises
    ------
    ValueError
        The model is in training mode, or ``io_precision`` is float16
        around a float32 body.
    UnsupportedOp
        The trace contains an operation with no MIL translation.
    OSError
        ``ENOSPC``: the package is written, but there is not the room to
        compile and load it — see :func:`load`.

    Examples
    --------
    The whole of it, for a classifier that should run on the accelerator:

    >>> import shutil, tempfile, lucid, lucid.nn as nn, lucid.models as M, lucid.coreml as cml
    >>> lucid.manual_seed(1)       # the error bounds below are of these weights
    >>> model = M.create_model(
    ...     "resnet_18_cls", num_classes=10,
    ...     stem_channels=8, hidden_sizes=(8, 16, 32, 64),   # narrowed, to be quick
    ... ).eval()
    >>> x, room = lucid.randn(1, 3, 32, 32), tempfile.mkdtemp()
    >>> package = cml.export(
    ...     model, x, f"{room}/resnet18.mlpackage",
    ...     precision=cml.Precision.FLOAT16,
    ...     compute_units=cml.ComputeUnits.CPU_AND_NE,
    ... )
    >>> package.verify(model, x, relative=True) < 1e-2
    True
    >>> package.benchmark(x).median_ms > 0.0       # once it has settled
    True

    A model of several inputs is given them the way its ``forward``
    takes them — a tuple positionally, a mapping by name:

    >>> class Pair(nn.Module):
    ...     def __init__(self):
    ...         super().__init__()
    ...         self.image = nn.Linear(16, 4)
    ...         self.text = nn.Linear(8, 4)
    ...     def forward(self, image, text):
    ...         return self.image(image) + self.text(text)
    >>> pair, image, text = Pair().eval(), lucid.randn(1, 16), lucid.randn(1, 8)
    >>> with cml.export(pair, (image, text), f"{room}/pair.mlpackage") as two:
    ...     print(two.input_names, two.predict((image, text)).shape)
    ['input_0', 'input_1'] (1, 4)
    >>> feed = {"image": image, "text": text}
    >>> with cml.export(pair, feed, f"{room}/named.mlpackage") as two:
    ...     print(two.input_names, two.predict(feed).shape)
    ['image', 'text'] (1, 4)

    Smaller on disk, at a cost worth measuring before shipping it:

    >>> with cml.export(model, x, f"{room}/int8.mlpackage",
    ...                 weights=cml.WeightPrecision.INT8) as small:
    ...     print(small.verify(model, x, relative=True) < 1e-2)
    True

    Pixels in and labels out:

    >>> pixels = (lucid.rand(1, 3, 32, 32) * 255).round()
    >>> names = tuple(f"class_{i}" for i in range(10))
    >>> with cml.export(model, pixels, f"{room}/cls.mlpackage",
    ...                 image_input=cml.ImageInput(scale=1 / 255.0),
    ...                 classifier=cml.Classifier(labels=names)) as labelled:
    ...     label, scores = labelled.classify(pixels)
    >>> label in names, len(scores)
    (True, 10)

    A batch axis the caller may vary, and a model that samples:

    >>> with cml.export(model, x, f"{room}/flex.mlpackage",
    ...                 shape_range={0: (1, 16)}) as flexible:
    ...     print(flexible.predict(lucid.randn(4, 3, 32, 32)).shape)
    (4, 10)
    >>> class Encoder(nn.Module):
    ...     def __init__(self):
    ...         super().__init__()
    ...         self.mu = nn.Linear(8, 4)
    ...         self.logvar = nn.Linear(8, 4)
    ...     def forward(self, h):
    ...         mu, logvar = self.mu(h), self.logvar(h)
    ...         return mu + (logvar * 0.5).exp() * lucid.randn(1, 4)
    >>> with cml.export(Encoder().eval(), lucid.randn(1, 8), f"{room}/vae.mlpackage",
    ...                 draws=cml.Draws.AS_INPUT) as vae:
    ...     print(vae.noise_inputs)
    [('noise_0', (1, 4))]

    A model that came out of ``lucid.quantization`` needs nothing said
    about it — the export recognises what it is carrying:

    >>> import lucid.quantization as q
    >>> def small_net():
    ...     return nn.Sequential(
    ...         nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(),
    ...         nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
    ...         nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(32, 10),
    ...     )
    >>> aware = q.prepare_qat(small_net(), q.get_default_qat_qconfig_mapping())
    >>> _ = aware(x)                              # fine-tune, then
    >>> with cml.export(aware.eval(), x, f"{room}/qat.mlpackage",
    ...                 weights=cml.WeightPrecision.INT8,
    ...                 activations=cml.Activations.DROPPED) as qat:
    ...     print(qat.predict(x).shape)
    (1, 10)

    Leave a converted model's weights alone, though. They already sit on
    MLX's grid, and asking for another stacks two of them — 3.8e-06
    against the quantized model becomes 3.4e-02:

    >>> dynamic = q.quantize_dynamic(small_net().eval())
    >>> with cml.export(dynamic, x, f"{room}/dynamic.mlpackage") as converted:
    ...     print(converted.verify(dynamic, x, relative=True) < 1e-4)
    True
    >>> package.close()
    >>> shutil.rmtree(room)

    """
    if getattr(model, "training", False):
        raise ValueError("lucid.coreml: model is in training mode; call model.eval() first")

    inner = getattr(model, "model", None)
    if inner is not None and type(model).__name__ == "CompiledModule":
        # A compiled module is a wrapper around the model, and tracing it
        # follows the wrapper rather than the network: the export ends up
        # reporting that the model ignored its input, which is not what
        # happened and sends the reader looking in the wrong place.
        raise TypeError(
            "lucid.coreml: this is a lucid.compile wrapper, not a model — the "
            "trace would follow the wrapper and report that the input reached "
            "nothing. Export the module it wraps (`compiled.model`); the two "
            "compute the same thing, and Core ML compiles the package itself "
            "so nothing is lost by exporting the uncompiled one"
        )

    # A packed MLX linear keeps its weight on the GPU because its kernel
    # is Metal-only, and none of that kernel reaches the tracer. It is
    # stood in for by the same layer computed from a reconstructed dense
    # weight — the packed form's own reference path, which matches the
    # kernel to 5e-07 — so the swap happens before the check below, or a
    # dynamically quantized model would be turned away for holding
    # exactly the weights it is meant to hold.
    with _packed_linears_unpacked(model):
        return _export_prepared(
            model,
            example,
            path,
            precision=precision,
            weights=weights,
            shapes=shapes,
            shape_range=shape_range,
            state=state,
            image_input=image_input,
            classifier=classifier,
            metadata=metadata,
            output_field=output_field,
            compute_units=compute_units,
            minimum_deployment_target=minimum_deployment_target,
            draws=draws,
            activations=activations,
            io_precision=io_precision,
        )


def _export_prepared(
    model: Module,
    example: object,
    path: str,
    *,
    precision: Precision,
    weights: WeightPrecision | Palettize | Sparsify,
    shapes: list[tuple[int, ...]] | None,
    shape_range: dict[int, tuple[int, int]] | None,
    state: list[State] | None,
    image_input: ImageInput | None,
    classifier: Classifier | None,
    metadata: Metadata | None,
    output_field: str | None,
    compute_units: ComputeUnits,
    minimum_deployment_target: DeploymentTarget | None,
    draws: Draws,
    activations: Activations,
    io_precision: Precision,
) -> CoreMLModel:
    """Write the package, once the model is in a shape the trace can follow.

    Split from :func:`export` so the substitutions that make a model
    traceable happen around the whole of it — the checks below included,
    since one of them would otherwise reject a model for weights the
    substitution has already dealt with.

    Parameters
    ----------
    model : nn.Module
        Model to export, already prepared.
    example : Tensor or tuple of Tensor or dict of str to Tensor
        Input the trace runs on.
    path : str
        Destination package.
    precision, weights, shapes, shape_range, state, image_input, classifier, metadata, output_field, compute_units, minimum_deployment_target, draws, activations, io_precision
        As :func:`export` documents them.

    Returns
    -------
    CoreMLModel
        The written package, loaded.
    """
    # Buffers as well as parameters. A quantized linear keeps its packed
    # weight, its scales and even its bias as buffers, so a check that
    # looked only at parameters would miss a model made entirely of them
    # and let it reach the blob writer — which is the message this exists
    # to replace.
    on_gpu = [
        name
        for source in ("named_parameters", "named_buffers")
        for name, held in getattr(model, source, lambda: [])()
        if getattr(getattr(held, "device", None), "type", "cpu") != "cpu"
    ]
    if on_gpu:
        # Reachable from the ordinary path — train on Metal, export — and
        # what it used to raise came from the blob writer several layers
        # down, naming neither the model nor the way out.
        raise ValueError(
            f"lucid.coreml: this model's parameters are on the GPU "
            f"({len(on_gpu)} of them, {on_gpu[0]!r} among the first), and the "
            f"package's weights are written from host memory. Move it first — "
            f"`model.to('cpu')` — and export from there; the exported package "
            f"runs on whichever units you open it with, so training on Metal "
            f"and shipping to the Neural Engine is the expected path, not a "
            f"contradiction"
        )

    # Built beside the destination and moved onto it only once it is
    # whole. Writing in place means clearing first, and an export that
    # then fails — an unsupported operation, a shape the trace could not
    # know — used to leave the caller with neither the package they had
    # nor the one they asked for.
    staging = staging_path(path)
    try:
        info = build_package(
            model,
            example,
            staging,
            precision=precision,
            weights=weights,
            shapes=shapes,
            shape_range=shape_range,
            state=state,
            image_input=image_input,
            classifier=classifier,
            metadata=metadata,
            output_field=output_field,
            minimum_deployment_target=minimum_deployment_target,
            draws=draws,
            activations=activations,
            io_precision=io_precision,
        )
    except BaseException:
        discard_staging(path)
        raise
    commit_staging(path)
    info["path"] = path
    outputs = _features(info["outputs"])
    return CoreMLModel(
        str(info["path"]),
        [name for name, _shape in _features(info["inputs"])],
        [name for name, _shape in outputs],
        compute_units=compute_units,
        precision=precision.value,
        io_precision=io_precision.value,
        # ``-1`` marks an axis that follows the input.
        output_shapes=dict(outputs),
        image_input=image_input,
        classifier=classifier,
        deployment_target=info["deployment_target"],
        noise=cast("list[tuple[str, tuple[int, ...], str, Tensor | None]]", info["noise"]),
        traced_outputs=cast("dict[str, Tensor]", info["traced_outputs"]),
    )


def export_functions(
    functions: dict[str, tuple[Module, object]],
    path: str,
    *,
    default: str | None = None,
    precision: Precision = Precision.FLOAT32,
    weights: WeightPrecision | Palettize | Sparsify = WeightPrecision.FLOAT,
    metadata: Metadata | None = None,
    compute_units: ComputeUnits = ComputeUnits.ALL,
    io_precision: Precision = Precision.FLOAT32,
) -> dict[str, CoreMLModel]:
    """Write several entry points into one package, sharing its weights.

    A decoder wants two: one that reads a whole prompt and one that reads
    a single token. They are the same network, and shipping them as two
    packages ships the weights twice. Here a parameter is written once and
    every function that uses it points at the same bytes.

    Needs iOS 18 / macOS 15, which is what several entry points costs.

    Parameters
    ----------
    functions : dict[str, tuple[nn.Module, object]]
        Name to the model and the example input to trace it with. The
        models may be the same object; that is the case worth having.
    path : str
        Destination ``.mlpackage``. Replaced if it exists.
    default : str or None, optional, keyword-only, default=None
        Entry point a caller gets without asking. The first, if unnamed.
    precision : Precision, optional, keyword-only, default=FLOAT32
        Body precision, for every function.
    weights : WeightPrecision or Palettize or Sparsify, optional, keyword-only, default=WeightPrecision.FLOAT
        Weight storage, for every function.
    metadata : Metadata or None, optional, keyword-only, default=None
        What the package says about itself. One package, one set.
    compute_units : ComputeUnits, optional, keyword-only, default=ALL
        Which processors Core ML may schedule on.
    io_precision : Precision, optional, keyword-only, default=FLOAT32
        Element type of every function's float inputs and outputs, as
        :func:`export` takes it.

    Returns
    -------
    dict[str, CoreMLModel]
        One handle per function, each pinned to its own entry point.

    Raises
    ------
    ValueError
        No functions, or ``default`` names one that is not there.

    Examples
    --------
    A decoder wants two entry points: one that reads a whole prompt and
    one that reads a single token. They are the same network, so the
    weights are written once and both point at the same bytes.

    >>> import shutil, tempfile
    >>> import lucid, lucid.nn as nn, lucid.coreml as cml
    >>> class Decoder(nn.Module):
    ...     def __init__(self):
    ...         super().__init__()
    ...         self.embed = nn.Embedding(100, 16)
    ...         self.head = nn.Linear(16, 100)
    ...     def forward(self, ids):
    ...         return self.head(self.embed(ids))
    >>> model, room = Decoder().eval(), tempfile.mkdtemp()
    >>> handles = cml.export_functions(
    ...     {
    ...         "prompt": (model, lucid.zeros(1, 8).to(lucid.int64)),
    ...         "step": (model, lucid.zeros(1, 1).to(lucid.int64)),
    ...     },
    ...     f"{room}/decoder.mlpackage",
    ...     default="step",
    ... )
    >>> prompt_ids = lucid.randint(0, 100, (1, 8))
    >>> handles["prompt"].predict(prompt_ids).shape
    (1, 8, 100)
    >>> handles["step"].predict(prompt_ids[:, -1:]).shape
    (1, 1, 100)
    >>> for handle in handles.values():
    ...     handle.close()
    >>> shutil.rmtree(room)

    """
    if not functions:
        raise ValueError("lucid.coreml: a package needs at least one function")
    chosen = default if default is not None else next(iter(functions))
    if chosen not in functions:
        raise ValueError(
            f"lucid.coreml: the default function {chosen!r} is not one of " f"{sorted(functions)}"
        )

    from lucid._C import engine as _C_engine

    cm = _C_engine.coreml
    # Same discipline as ``export``: built beside the destination and
    # moved onto it whole, so a failure part way through several
    # functions does not take an already-deployed package with it.
    staging = staging_path(path)
    try:
        paths = cm.prepare_package(staging)
        shared = _Shared(paths=paths, blob=cm.BlobWriter(paths.weight_bin))

        built: dict[str, dict[str, object]] = {}
        programs: list[tuple[str, object]] = []
        for name, (model, example) in functions.items():
            info = build_package(
                model,
                example,
                staging,
                precision=precision,
                weights=weights,
                metadata=metadata if name == chosen else None,
                into=shared,
                io_precision=io_precision,
            )
            built[name] = info
            programs.append((name, info["program"]))
        shared.blob.finalize()
        cm.finish_package(paths, cm.serialize_functions(programs, chosen))
    except BaseException:
        discard_staging(path)
        raise
    commit_staging(path)

    handles: dict[str, CoreMLModel] = {}
    for name, info in built.items():
        outputs = _features(info["outputs"])
        handles[name] = CoreMLModel(
            path,
            [feature for feature, _shape in _features(info["inputs"])],
            [feature for feature, _shape in outputs],
            compute_units=compute_units,
            precision=precision.value,
            io_precision=io_precision.value,
            output_shapes=dict(outputs),
            function_name=name,
            # Several entry points is itself an IOS18 feature, whatever
            # else the package uses.
            deployment_target=DeploymentTarget.IOS18,
        )
    return handles


def precision_cost(
    model: Module,
    example: object,
    *,
    weights: WeightPrecision | Palettize | Sparsify = WeightPrecision.FLOAT,
    output_field: str | None = None,
) -> dict[str, float]:
    """What each precision costs this model, measured rather than assumed.

    The Neural Engine runs float16 and nothing else, so reaching the
    accelerator means accepting whatever float16 does to a particular
    network — and that varies by more than an order of magnitude between
    architectures that look alike. Measured across fourteen families on
    an untrained forward: convolutional stacks land near 1e-3, ViT at
    1e-2, and MaxViT at 1.6e-1. Nothing in an export says which kind a
    given model is.

    So this exports it twice and compares both against the eager model,
    relative to each output's own magnitude. The packages are written to
    a temporary directory and removed: the answer is the point, not the
    artifacts.

    The float16 figure is Core ML's, which is better than running the
    same model in half precision throughout — a ViT does that at 1.5
    relative, against 1e-2 here — because the runtime keeps the parts
    that need range in float32. It is not a bound on your own half
    precision arithmetic elsewhere.

    Parameters
    ----------
    model : nn.Module
        Model to measure, in ``eval()`` mode.
    example : Tensor or tuple of Tensor or dict of str to Tensor
        Input to trace and compare with, as ``export`` takes it.
    weights : WeightPrecision or Palettize or Sparsify, optional, keyword-only, default=WeightPrecision.FLOAT
        How weights are stored, held the same across both exports so the
        difference is the body's precision alone.
    output_field : str or None, optional, keyword-only, default=None
        Which field of a dataclass output to export, as ``export`` takes
        it.

    Returns
    -------
    dict[str, float]
        ``{"float32": err, "float16": err}`` — the worst relative
        disagreement with the eager model, per precision.

    Raises
    ------
    ValueError
        When the model answers with all zeros, since comparing against
        that proves nothing about either precision.

    Examples
    --------
    >>> import lucid, lucid.coreml, lucid.nn as nn
    >>> model = nn.Sequential(nn.Conv2d(3, 16, 3, padding=1), nn.ReLU()).eval()
    >>> cost = lucid.coreml.precision_cost(model, lucid.randn(1, 3, 32, 32))
    >>> sorted(cost)
    ['float16', 'float32']
    >>> cost["float16"] < 1e-2
    True
    """
    import tempfile

    measured: dict[str, float] = {}
    with tempfile.TemporaryDirectory() as room:
        for name, precision in (("float32", Precision.FLOAT32), ("float16", Precision.FLOAT16)):
            exported = export(
                model,
                example,
                f"{room}/{name}.mlpackage",
                precision=precision,
                weights=weights,
                output_field=output_field,
            )
            try:
                measured[name] = exported.verify(model, example, relative=True)
            finally:
                exported.close()
    return measured


def load(path: str, *, compute_units: ComputeUnits = ComputeUnits.ALL) -> CoreMLModel:
    """Load an existing ``.mlpackage``.

    Feature names are read from the package, so this works on packages
    Lucid did not write. So is the precision a Lucid export recorded —
    the body's and the interface's — which a package from elsewhere, or
    from a release that did not record it, reports as ``"UNKNOWN"``.

    The package is compiled once per content and the compiled model kept
    in ``~/.cache/lucid/coreml`` (``LUCID_COREML_CACHE_DIR``, or
    ``$LUCID_HOME/coreml``), so loading it again — in this process or the
    next — skips the compile and lets Core ML reuse the bundle it
    specialised the model into, instead of writing a new copy of the
    weights into its own cache each time. The cache keeps to
    ``LUCID_COREML_CACHE_LIMIT`` bytes (default 4G, least recently used
    first; ``0`` turns it off) and :func:`empty_cache` clears it.

    Parameters
    ----------
    path : str
        Package to load.
    compute_units : ComputeUnits, optional, keyword-only, default=ALL
        Which processors Core ML may schedule on. Reaching the Neural
        Engine also needs the package to be float16 — check with
        :meth:`CoreMLModel.compute_plan`.

    Returns
    -------
    CoreMLModel
        Loaded and ready to predict.

    Raises
    ------
    RuntimeError
        Core ML could not compile or load the package; its own message
        names the offending layer.
    OSError
        ``ENOSPC``: a volume the load writes to — the temporary space
        Core ML compiles into, the cache, Core ML's own caches — has less
        free than the package's size plus a margin. Core ML ends the
        process when it runs out of disk part way, so the load refuses
        before it starts.

    Examples
    --------
    >>> import shutil, tempfile
    >>> import lucid, lucid.nn as nn, lucid.coreml as cml
    >>> model = nn.Sequential(
    ...     nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(),
    ...     nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(16, 10),
    ... ).eval()
    >>> x, room = lucid.randn(1, 3, 32, 32), tempfile.mkdtemp()
    >>> cml.export(model, x, f"{room}/model.mlpackage").close()
    >>> package = cml.load(f"{room}/model.mlpackage")
    >>> package.predict(x).shape
    (1, 10)
    >>> package.close()

    The handle recovers what the export knew — that an input is a
    picture, that the outputs are labels, which inputs stand in for a
    draw — because the package declares all of it:

    >>> brightest = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten()).eval()
    >>> red = lucid.zeros(1, 3, 8, 8)
    >>> red[:, 0] = 255.0
    >>> cml.export(brightest, red, f"{room}/cls.mlpackage",
    ...            image_input=cml.ImageInput(scale=1 / 255.0),
    ...            classifier=cml.Classifier(labels=("red", "green", "blue")),
    ...            ).close()
    >>> reopened = cml.load(f"{room}/cls.mlpackage")
    >>> reopened.classify(red)[0]
    'red'
    >>> reopened.close()
    >>> shutil.rmtree(room)

    """
    # Opened once, and everything below is read off that handle. Opening
    # a throwaway handle first to learn the feature names cost a second
    # load — and, with each load compiling afresh, a second full copy of
    # the weights in Core ML's cache.
    model = CoreMLModel(
        path, [], [], compute_units=compute_units, precision="UNKNOWN", io_precision="UNKNOWN"
    )
    try:
        handle = model._handle
        model.input_names = list(handle.input_names)
        model.output_names = list(handle.output_names)
        # An exported handle carries what the export knew: that an input is
        # a picture and not an array, that the outputs are labels and not
        # scores. Reopening the file used to lose both, so the same package
        # answered differently depending on how it was opened — and in a
        # deployment, reopening is the ordinary path. The package declares
        # all of it, so it is read back rather than asked for again.
        images = list(handle.image_input_names)
        labels = list(handle.class_labels)
        # ``predict`` needs only to know the input is a picture; ``verify``
        # needs the scale and bias, which the program applies and the file
        # does not declare. Left unset so the comparison refuses rather than
        # measuring a normalisation it guessed.
        model.image_input = ImageInput() if images else None
        model.image_normalisation = None
        model.classifier = Classifier(labels=tuple(labels)) if labels else None
        # Which inputs stand in for a random draw, written by the export
        # into the creator-defined metadata because nothing else in the file
        # distinguishes them from an ordinary input.
        declared = dict(handle.user_metadata)
        noise: list[tuple[str, tuple[int, ...], str, Tensor | None]] = []
        for entry in declared.get(_DRAWN_KEY, "").split(","):
            if not entry:
                continue
            name, kind, extent = entry.split(":")
            noise.append((name, tuple(int(d) for d in extent.split("x")), kind, None))
        model._declare_noise(noise)
        # The precision the export recorded. A package Lucid did not write —
        # or wrote before it recorded one — says nothing, and a guess would
        # be repeated back as a diagnosis: the float32 advice a compute plan
        # gives is wrong for a float16 program.
        model.precision = declared.get(_PRECISION_KEY, "UNKNOWN")
        model.io_precision = declared.get(_IO_PRECISION_KEY, "UNKNOWN")
    except BaseException:
        # Not left holding the model, or its cache entry, until collected.
        model.close()
        raise
    return model


def empty_cache() -> int:
    """Remove the compiled models Lucid keeps for packages it has loaded.

    Core ML runs a *compiled* model, and loading one makes Core ML write a
    specialised bundle — for a GPU segment, a full copy of the weights —
    into its own cache, keyed by where the compiled model is. So Lucid
    compiles each package once, keeps the compiled model, and opens that
    same one on every later load; the bundle is then read again rather
    than written again. Before, every load compiled to a fresh path and
    left Core ML a bundle nothing would read: a development machine's
    cache reached 121 GB.

    The kept models live in a versioned directory (``v1``) under
    ``LUCID_COREML_CACHE_DIR`` — by default ``$LUCID_HOME/coreml`` or
    ``~/.cache/lucid/coreml`` — and keep to ``LUCID_COREML_CACHE_LIMIT``
    bytes (``4G`` unless set; ``0`` turns the cache off), dropping the
    least recently used first. This empties it now. A model an open
    handle holds, in this process or another, stays, and so does anything
    in the directory that Lucid did not put there.

    The bundles Core ML made from the removed models stay in Core ML's
    own cache, which macOS purges when the disk runs low; a package loaded
    again after this is compiled again and gets a new one.

    Returns
    -------
    int
        Bytes freed.

    Examples
    --------
    >>> import os, shutil, tempfile
    >>> import lucid, lucid.nn as nn, lucid.coreml as cml
    >>> room = tempfile.mkdtemp()
    >>> saved = os.environ.get("LUCID_COREML_CACHE_DIR")
    >>> os.environ["LUCID_COREML_CACHE_DIR"] = f"{room}/cache"   # one of its own
    >>> model = nn.Sequential(nn.Conv2d(3, 8, 3), nn.ReLU()).eval()
    >>> x = lucid.randn(1, 3, 16, 16)
    >>> cml.export(model, x, f"{room}/m.mlpackage").close()    # compiles, keeps it
    >>> with cml.load(f"{room}/m.mlpackage") as again:        # opens the kept one
    ...     print(again.predict(x).shape)
    (1, 8, 14, 14)
    >>> cml.empty_cache() > 0
    True
    >>> cml.empty_cache()                                     # nothing left
    0
    >>> if saved is None:
    ...     del os.environ["LUCID_COREML_CACHE_DIR"]
    ... else:
    ...     os.environ["LUCID_COREML_CACHE_DIR"] = saved
    >>> shutil.rmtree(room)
    """
    from lucid.coreml import _cache

    return _cache.empty_cache()
