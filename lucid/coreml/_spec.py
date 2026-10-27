"""Core ML's vocabulary: dtypes, compute units, precisions, targets.

Everything Core ML numbers or names lives here rather than being spelled
inline, so a specification bump is a change in one file.  The values that
end up in the emitted bytes come from the engine (which got them from the
generated schema header); what this module adds is the Lucid-facing names
and the mapping from Lucid's own dtypes.
"""

import dataclasses
import enum

import lucid
from lucid._C import engine as _C_engine

__all__ = [
    "Classifier",
    "State",
    "ColorSpace",
    "Activations",
    "ComputeUnits",
    "Draws",
    "ImageInput",
    "Metadata",
    "Precision",
    "WeightPrecision",
]


class ComputeUnits(enum.Enum):
    """Processors Core ML may schedule the model on.

    ``CPU_AND_NE`` is the reason this package exists: it is the only way
    to reach the Neural Engine, which neither of Lucid's own backends
    (Accelerate, MLX) targets.

    Examples
    --------
    >>> import shutil, tempfile
    >>> import lucid, lucid.nn as nn, lucid.coreml as cml
    >>> model = nn.Sequential(nn.Conv2d(3, 16, 3, padding=1), nn.ReLU()).eval()
    >>> x, room = lucid.randn(1, 3, 32, 32), tempfile.mkdtemp()
    >>> package = cml.export(model, x, f"{room}/m.mlpackage",
    ...                      precision=cml.Precision.FLOAT16,
    ...                      compute_units=cml.ComputeUnits.CPU_AND_NE)
    >>> package.compute_units
    <ComputeUnits.CPU_AND_NE: 'CPU_AND_NE'>
    >>> set(package.compute_plan().compute) <= {"CPU", "ANE"}   # never the GPU
    True
    >>> package.close()
    >>> shutil.rmtree(room)
    """

    ALL = "ALL"
    CPU_ONLY = "CPU_ONLY"
    CPU_AND_GPU = "CPU_AND_GPU"
    CPU_AND_NE = "CPU_AND_NE"


class Precision(enum.Enum):
    """Weight and activation precision of the exported program.

    ``FLOAT32`` is the default because an export should agree with the
    model it came from; Core ML's own default is ``FLOAT16``, which the
    Neural Engine wants and which costs roughly 1e-4 against the eager
    model.

    Examples
    --------
    >>> import shutil, tempfile
    >>> import lucid, lucid.nn as nn, lucid.coreml as cml
    >>> model = nn.Sequential(nn.Conv2d(3, 16, 3, padding=1), nn.ReLU()).eval()
    >>> x, room = lucid.randn(1, 3, 32, 32), tempfile.mkdtemp()
    >>> with cml.export(model, x, f"{room}/half.mlpackage",
    ...                 precision=cml.Precision.FLOAT16) as package:
    ...     print(package.precision, package.verify(model, x, relative=True) < 1e-2)
    FLOAT16 True
    >>> shutil.rmtree(room)
    """

    FLOAT32 = "FLOAT32"
    FLOAT16 = "FLOAT16"


class Activations(enum.Enum):
    """What to do with a quantization-aware model's activation quantization.

    A model prepared for quantization-aware training carries two kinds of
    fake-quantization: one on each weight, which puts the weight onto the
    grid it will be stored on, and one on each activation, which
    simulates a runtime that computes in integers.

    The first is what the export wants — a weight trained onto its grid
    is written on that grid exactly. The second is a simulation of a
    runtime this is not. Core ML's quantization is weight-only, and the
    Neural Engine computes in float16 whatever the package declares, so
    the simulation is carried as real arithmetic: rounding, clipping and
    scaling at every activation, work the accelerator did not ask for.

    ``SIMULATED`` is the default and keeps it, because dropping it
    changes what the package computes and that is not a decision to make
    on a caller's behalf. ``DROPPED`` removes it, leaving the weights
    quantized and the activations in float — which is what Core ML would
    do with the same network quantized after training, except that the
    weights were trained for their grid.

    Measured on a small classifier at eight bits, against the same
    network quantized after training rather than during it:

    ==================================  ========  =========  =======
    export                              accuracy  vs eager   ops
    ==================================  ========  =========  =======
    post-training int8                     0.986   7.0e-04         6
    QAT int8, activations simulated        0.997   0.00e+00       24
    QAT int8, activations dropped          0.997   6.4e-06         6
    ==================================  ========  =========  =======

    Inert for a model that carries no fake-quantization, which is every
    model that did not come out of ``lucid.quantization``.

    Examples
    --------
    >>> import shutil, tempfile
    >>> import lucid, lucid.nn as nn, lucid.coreml as cml
    >>> import lucid.quantization as q
    >>> lucid.manual_seed(0)
    >>> x = lucid.randn(1, 3, 16, 16)
    >>> float_model = nn.Sequential(
    ...     nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(),
    ...     nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
    ...     nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(32, 10),
    ... )
    >>> qat_model = q.prepare_qat(float_model, q.get_default_qat_qconfig_mapping())
    >>> _ = qat_model(x)                 # fine-tune here; this only calibrates
    >>> qat_model = qat_model.eval()
    >>> room = tempfile.mkdtemp()
    >>> simulated = cml.export(qat_model, x, f"{room}/simulated.mlpackage",
    ...                        weights=cml.WeightPrecision.INT8)
    >>> small = cml.export(qat_model, x, f"{room}/small.mlpackage",
    ...                    weights=cml.WeightPrecision.INT8,
    ...                    activations=cml.Activations.DROPPED)
    >>> simulated.verify(qat_model, x, relative=True) < 1e-5   # the model, exactly
    True
    >>> small.verify(qat_model, x, relative=True) < 1e-2       # close to it
    True
    >>> ops = [p.compute_plan().total_compute for p in (simulated, small)]
    >>> ops[1] < ops[0]                 # and the simulation's work is gone
    True
    >>> simulated.close()
    >>> small.close()
    >>> shutil.rmtree(room)
    """

    SIMULATED = "SIMULATED"
    DROPPED = "DROPPED"


class Draws(enum.Enum):
    """What to do about a model that draws random numbers in ``forward``.

    Core ML has no random operation. A draw whose inputs are all
    constants — which every ``randn`` is — folds at build time, so a
    package written the obvious way returns one fixed sample for the life
    of the file. A variational encoder exported that way has a latent
    that never moves, and nothing about the package looks wrong.

    ``REFUSED`` is the default and says so rather than shipping that.

    ``AS_INPUT`` lifts each draw to an input the caller fills. The
    reparameterisation ``mu + sigma * eps`` becomes a function of ``eps``
    instead of a function that makes one, which is what the model always
    was — the draw was never part of the network. Seven families in the
    zoo are exportable only this way: variational encoders, world models
    with a stochastic latent, and score-based samplers.

    The handle draws for the caller who does not, so a package still
    answers differently on each prediction the way the eager model does.

    Examples
    --------
    >>> import shutil, tempfile
    >>> import lucid, lucid.nn as nn, lucid.coreml as cml
    >>> class Encoder(nn.Module):
    ...     def __init__(self):
    ...         super().__init__()
    ...         self.mu = nn.Linear(8, 4)
    ...         self.logvar = nn.Linear(8, 4)
    ...     def forward(self, x):
    ...         mu, logvar = self.mu(x), self.logvar(x)
    ...         return mu + (logvar * 0.5).exp() * lucid.randn(1, 4)
    >>> vae, x, room = Encoder().eval(), lucid.randn(1, 8), tempfile.mkdtemp()
    >>> package = cml.export(vae, x, f"{room}/vae.mlpackage",
    ...                      draws=cml.Draws.AS_INPUT)
    >>> package.noise_inputs
    [('noise_0', (1, 4))]

    Pass nothing for it and the handle draws, so the package samples the
    way the model does:

    >>> float((package.predict(x) - package.predict(x)).abs().max()) > 0
    True

    Pass one and the package is a deterministic function of it:

    >>> eps = lucid.randn(1, 4)
    >>> a = package.predict({"input": x, "noise_0": eps})
    >>> b = package.predict({"input": x, "noise_0": eps})
    >>> float((a - b).abs().max())
    0.0
    >>> package.close()
    >>> shutil.rmtree(room)
    """

    REFUSED = "REFUSED"
    AS_INPUT = "AS_INPUT"


class DeploymentTarget(enum.Enum):
    """Oldest operating system the exported package will run on.

    Three features raise this floor, and until now they raised it
    silently: carrying state, palettizing weights, and writing several
    entry points into one package all move the program from the
    ``CoreML7`` opset to ``CoreML8``. A package built with any of them
    will not load on iOS 17, and nothing said so at the time — the
    caller found out from a device.

    Naming a target makes the trade explicit in the direction that
    matters: ask for ``IOS17`` and a feature that cannot be expressed
    there is refused, by name, while the export is still a Python call.

    There is no entry below ``IOS17`` because the emitters do not write
    below it: this package's ``gather`` and ``scatter`` carry
    ``validate_indices``, which iOS 16 has no field for. Supporting an
    older floor is a change to the emitters, not a value in this enum.

    Attributes
    ----------
    IOS17 : str
        The opset this writer emits by default (macOS 14, iPadOS 17).
    IOS18 : str
        Needed for state, palettization and multi-function packages
        (macOS 15, iPadOS 18).

    Examples
    --------
    Read back what the package ended up needing:

    >>> import shutil, tempfile
    >>> import lucid, lucid.nn as nn, lucid.coreml as cml
    >>> model = nn.Sequential(
    ...     nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(), nn.Conv2d(64, 64, 3, padding=1)
    ... ).eval()
    >>> x, room = lucid.randn(1, 3, 16, 16), tempfile.mkdtemp()
    >>> package = cml.export(model, x, f"{room}/m.mlpackage")
    >>> package.deployment_target
    <DeploymentTarget.IOS17: 'CoreML7'>
    >>> package.close()

    Or refuse a floor the package cannot meet, while it is still a
    Python call rather than a device:

    >>> cml.export(model, x, f"{room}/p.mlpackage", weights=cml.Palettize(bits=4),
    ...            minimum_deployment_target=cml.DeploymentTarget.IOS17)
    Traceback (most recent call last):
    ValueError: lucid.coreml: IOS17 was asked for, ... palettization ... only from IOS18...
    >>> shutil.rmtree(room)
    """

    IOS17 = "CoreML7"
    IOS18 = "CoreML8"

    @property
    def opset(self) -> str:
        """MIL opset string the program declares."""
        return self.value


#: What each feature costs in deployment floor, for the message that
#: refuses it. Keyed by the words a caller would recognise from their own
#: call, not by internal names.
_NEEDS_IOS18 = {
    "state": "state=[State(...)]",
    "palettization": "weights=Palettize(...)",
    "multiple functions": "export_functions",
}


class WeightPrecision(enum.Enum):
    """How a weight is stored, as distinct from how the body computes.

    ``INT8`` stores each weight as an integer code plus one scale per
    output channel, and Core ML dequantizes it on the way into the
    operation that uses it. The arithmetic still runs at the body's
    precision — this is a storage decision, not a different network — so
    the package halves in size against float16 and the Neural Engine
    moves half as much memory to do the same work.

    The cost is real and one-directional: eight bits per weight cannot
    represent what sixteen did, so a quantized export is further from the
    eager model than a float16 one. ``verify`` will say by how much.

    Examples
    --------
    >>> import shutil, tempfile
    >>> import lucid, lucid.nn as nn, lucid.coreml as cml
    >>> model = nn.Sequential(
    ...     nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(), nn.Conv2d(64, 64, 3, padding=1)
    ... ).eval()
    >>> x, room = lucid.randn(1, 3, 16, 16), tempfile.mkdtemp()
    >>> with cml.export(model, x, f"{room}/int8.mlpackage",
    ...                 weights=cml.WeightPrecision.INT8) as package:
    ...     print(package.verify(model, x, relative=True) < 1e-2)   # by how much
    True
    >>> shutil.rmtree(room)
    """

    FLOAT = "FLOAT"
    INT8 = "INT8"


@dataclasses.dataclass(frozen=True)
class Palettize:
    """Store each weight as an index into a small table of values.

    A layer's weights are clustered into ``2 ** bits`` representative
    values; the package keeps the table and one key per weight, and Core
    ML expands it on the way into the operation that uses it. At four
    bits that is a quarter of the space float16 takes and an eighth of
    float32, for weights that were never using their full range to begin
    with.

    Unlike ``WeightPrecision.INT8``, which spaces its levels evenly and
    spends them wherever the range happens to be, the table is fitted to
    the weights — so a layer whose values crowd around zero keeps its
    resolution there. The cost is the same in kind: fewer distinct values
    than the model was trained with, and ``verify`` will say how much.

    Attributes
    ----------
    bits : int
        One of ``1, 2, 3, 4, 6, 8`` — palette sizes ``2, 4, 8, 16, 64,
        256``. Anything else is refused rather than rounded, since the
        choice is a size/accuracy trade the caller is making
        deliberately.

        At eight bits, ``WeightPrecision.INT8`` is usually the better
        instrument: it stores the same byte per weight but carries a
        scale per output channel, and on a trained ResNet-50 it measured
        both smaller and closer to the model.

    Examples
    --------
    >>> import shutil, tempfile
    >>> import lucid, lucid.nn as nn, lucid.coreml as cml
    >>> model = nn.Sequential(
    ...     nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(), nn.Conv2d(64, 64, 3, padding=1)
    ... ).eval()
    >>> x, room = lucid.randn(1, 3, 16, 16), tempfile.mkdtemp()
    >>> with cml.export(model, x, f"{room}/6bit.mlpackage",
    ...                 weights=cml.Palettize(bits=6)) as package:
    ...     print(package.palettized, package.deployment_target)
    True DeploymentTarget.IOS18
    >>> shutil.rmtree(room)

    Below six bits, fit the palette during a fine-tune rather than
    afterwards — see :class:`~lucid.coreml.CompressionAware`:

    >>> aware = cml.CompressionAware(model, weights=cml.Palettize(bits=2))
    >>> aware.covered              # the first layer is too small to palettize
    ['2.weight']
    """

    bits: int = 4

    def __post_init__(self) -> None:
        if self.bits not in (1, 2, 3, 4, 6, 8):
            raise ValueError(
                f"lucid.coreml: palettization takes 1, 2, 3, 4, 6 or 8 bits "
                f"(palette sizes 2, 4, 8, 16, 64, 256); got {self.bits}"
            )


@dataclasses.dataclass(frozen=True)
class Sparsify:
    """Keep only the largest weights, and a bit saying where they were.

    The smallest ``ratio`` of each weight tensor by magnitude is set to
    zero, and the package stores the survivors plus one bit per element.
    Below about half sparsity that costs more than it saves — the mask is
    an eighth of a byte per weight whether the weight survives or not —
    so the useful settings start around ``0.5`` and the saving grows from
    there.

    This is magnitude pruning applied at export, not training: the model
    is not fine-tuned afterwards, so accuracy falls faster than it would
    for a network pruned and then retrained. ``verify`` measures it.

    Attributes
    ----------
    ratio : float
        Fraction of each weight set to zero, in ``[0, 1)``.

    Examples
    --------
    >>> import shutil, tempfile
    >>> import lucid, lucid.nn as nn, lucid.coreml as cml
    >>> model = nn.Sequential(
    ...     nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(), nn.Conv2d(64, 64, 3, padding=1)
    ... ).eval()
    >>> x, room = lucid.randn(1, 3, 16, 16), tempfile.mkdtemp()
    >>> with cml.export(model, x, f"{room}/sparse.mlpackage",
    ...                 weights=cml.Sparsify(ratio=0.5)) as package:
    ...     cost = package.verify(model, x, relative=True)
    >>> cost > 1e-3                 # pruned and not retrained: not free
    True
    >>> shutil.rmtree(room)
    """

    ratio: float = 0.5

    def __post_init__(self) -> None:
        if not 0.0 <= self.ratio < 1.0:
            raise ValueError(f"lucid.coreml: sparsity is a fraction in [0, 1); got {self.ratio}")


class ColorSpace(enum.Enum):
    """Pixel layout Core ML should hand the model.

    Names the order the model's own channel 0, 1, 2 mean, so that the
    runtime writes a pixel buffer the right way round. Getting this wrong
    is silent: the model runs and answers badly, which is the same
    failure as feeding it an image with the red and blue channels
    swapped, because that is exactly what it is.

    Examples
    --------
    >>> import shutil, tempfile
    >>> import lucid, lucid.nn as nn, lucid.coreml as cml
    >>> gray = cml.ImageInput(color=cml.ColorSpace.GRAYSCALE, scale=1 / 255.0)
    >>> gray.color
    <ColorSpace.GRAYSCALE: 'GRAYSCALE'>
    >>> model = nn.Sequential(nn.Conv2d(1, 4, 3, padding=1), nn.ReLU()).eval()
    >>> pixels = (lucid.rand(1, 1, 16, 16) * 255).round()     # one channel
    >>> room = tempfile.mkdtemp()
    >>> with cml.export(model, pixels, f"{room}/gray.mlpackage",
    ...                 image_input=gray) as package:
    ...     print(package.predict(pixels).shape)
    (1, 4, 16, 16)
    >>> shutil.rmtree(room)
    """

    GRAYSCALE = "GRAYSCALE"
    RGB = "RGB"
    BGR = "BGR"


@dataclasses.dataclass(frozen=True)
class ImageInput:
    """Present an input as an image, with the normalisation it expects.

    An app holding a ``CVPixelBuffer`` cannot feed a multi-array without
    converting the pixels itself, and getting that conversion subtly wrong
    — a missed scale, the wrong channel order — produces a model that runs
    and answers badly. Declaring the input as an image moves both the
    conversion and the normalisation into the package.

    ``scale`` and ``bias`` are applied as ``pixel * scale + bias``, which
    is where a mean-and-standard-deviation normalisation lands: a channel
    normalised by ``(p/255 - m) / s`` has ``scale = 1/(255 * s)`` and
    ``bias = -m / s``.

    Attributes
    ----------
    scale : float
        Multiplier applied to every pixel, before ``bias``.
    bias : tuple[float, ...]
        One offset per channel. Empty adds nothing.
    color : ColorSpace
        Pixel layout. ``GRAYSCALE`` expects one channel, the others three.

    Examples
    --------
    Pixels are whole numbers in ``[0, 255]``; the scale is applied inside
    the package, so the caller feeds pixels and not normalised values:

    >>> import shutil, tempfile
    >>> import lucid, lucid.nn as nn, lucid.coreml as cml
    >>> model = nn.Sequential(
    ...     nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(),
    ...     nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(16, 10),
    ... ).eval()
    >>> pixels = (lucid.rand(1, 3, 32, 32) * 255).round()
    >>> room = tempfile.mkdtemp()
    >>> package = cml.export(model, pixels, f"{room}/img.mlpackage",
    ...                      image_input=cml.ImageInput(scale=1 / 255.0))
    >>> package.predict(pixels).shape
    (1, 10)
    >>> package.verify(model, pixels) < 1e-4   # the eager side is scaled to match
    True
    >>> package.close()
    >>> shutil.rmtree(room)
    """

    scale: float = 1.0
    bias: tuple[float, ...] = ()
    color: ColorSpace = ColorSpace.RGB


@dataclasses.dataclass(frozen=True)
class Classifier:
    """Turn the exported scores into labels Core ML knows how to name.

    Without this a package returns a score array and the app does its own
    argmax and label lookup. Vision's ``VNCoreMLRequest`` does not even
    get that far: it reads the package's ``predictedFeatureName``, and an
    unset one means it returns nothing.

    The two feature names default to what the reference tooling emits,
    because that is what Xcode's preview and most sample code look for.

    ``classify`` does **not** normalise: whatever the model produces is
    what the map contains, so a network ending in a linear layer yields
    raw scores under a name that says "probabilities". Add a softmax to
    the model if the values need to be probabilities — Core ML will not
    do it, and the name will not tell you it did not.

    Attributes
    ----------
    labels : tuple[str, ...]
        One label per score, in the order the model produces them.
    label_name : str
        Feature the winning label is returned under.
    probabilities_name : str
        Feature the label-to-probability map is returned under.

    Examples
    --------
    A linear layer that passes its input through, so the scores are the
    input itself:

    >>> import shutil, tempfile
    >>> import lucid, lucid.nn as nn, lucid.coreml as cml
    >>> model = nn.Linear(2, 2).eval()
    >>> _ = nn.init.eye_(model.weight), nn.init.zeros_(model.bias)
    >>> x, room = lucid.tensor([[3.0, 1.0]]), tempfile.mkdtemp()
    >>> package = cml.export(
    ...     model, x, f"{room}/cls.mlpackage",
    ...     classifier=cml.Classifier(labels=("cat", "dog")),
    ... )
    >>> label, probabilities = package.classify(x)
    >>> label
    'cat'
    >>> sorted(probabilities.items())      # the raw scores, not normalised
    [('cat', 3.0), ('dog', 1.0)]
    >>> package.close()
    >>> shutil.rmtree(room)
    """

    labels: tuple[str, ...]
    label_name: str = "classLabel"
    probabilities_name: str = "classLabel_probs"


@dataclasses.dataclass(frozen=True)
class State:
    """A value the package carries from one prediction to the next.

    Core ML keeps it: the caller neither passes it in nor gets it back,
    and each prediction sees what the last one wrote. A decoder's
    key-value cache is the case this exists for.

    Lucid's side has to be a pair — an input the model reads and an output
    it returns — rather than a buffer it mutates. The tracer records a
    pure graph, so an in-place buffer write does not appear in it at all;
    a package built from that would agree on the first call and stop
    accumulating on every one after, which ``export`` refuses.

    The state begins at zero. Nothing carries the example's values into
    it, so a model that needs a different starting point has to be given
    one through an ordinary input.

    Attributes
    ----------
    input : str
        Feature name of the input the state replaces.
    output : str
        Output field whose value is written back into it.

    Examples
    --------
    The cache stops being an input the caller passes; Core ML keeps it
    and each prediction sees what the last one wrote:

    >>> import shutil, tempfile
    >>> import lucid, lucid.nn as nn, lucid.coreml as cml
    >>> class Decoder(nn.Module):
    ...     def forward(self, x, cache):
    ...         cache = cache + x          # what this step writes back
    ...         return cache, cache * 2.0
    >>> token, room = lucid.ones(1, 4) * 3.0, tempfile.mkdtemp()
    >>> package = cml.export(
    ...     Decoder().eval(), {"x": token, "cache": lucid.zeros(1, 4)},
    ...     f"{room}/decoder.mlpackage", precision=cml.Precision.FLOAT16,
    ...     state=[cml.State(input="cache", output="output_0")],
    ... )
    >>> package.input_names
    ['x']
    >>> package.predict(token).tolist(), package.predict(token).tolist()
    ([[6.0, 6.0, 6.0, 6.0]], [[12.0, 12.0, 12.0, 12.0]])
    >>> package.reset_state()      # start a fresh sequence
    >>> package.predict(token).tolist()
    [[6.0, 6.0, 6.0, 6.0]]
    >>> package.close()
    >>> shutil.rmtree(room)
    """

    input: str
    output: str


@dataclasses.dataclass(frozen=True)
class Metadata:
    """What the package says about itself.

    Empty fields are left out of the description rather than written
    blank, so a package carries only what someone actually stated.

    Examples
    --------
    >>> import shutil, tempfile
    >>> import lucid, lucid.nn as nn, lucid.coreml as cml
    >>> model = nn.Sequential(nn.Conv2d(3, 16, 3, padding=1), nn.ReLU()).eval()
    >>> x, room = lucid.randn(1, 3, 32, 32), tempfile.mkdtemp()
    >>> package = cml.export(model, x, f"{room}/m.mlpackage", metadata=cml.Metadata(
    ...     description="One convolution, for the example",
    ...     author="me", license="MIT", version="1.0",
    ... ))
    >>> package.close()
    >>> shutil.rmtree(room)
    """

    description: str = ""
    author: str = ""
    license: str = ""
    version: str = ""


_COLOR_SPACES: dict[ColorSpace, int] = {
    ColorSpace.GRAYSCALE: _C_engine.coreml.COLOR_GRAYSCALE,
    ColorSpace.RGB: _C_engine.coreml.COLOR_RGB,
    ColorSpace.BGR: _C_engine.coreml.COLOR_BGR,
}


def color_space(color: ColorSpace) -> int:
    """Engine constant for a colour space.

    Parameters
    ----------
    color : ColorSpace
        Layout to translate.

    Returns
    -------
    int
        The ``ImageFeatureType.ColorSpace`` value.
    """
    return _COLOR_SPACES[color]


# Lucid dtype -> (MIL dtype, blob dtype).  The two numberings disagree —
# MIL calls float32 11, the blob calls it 2 — so both are carried rather
# than derived from each other.
_DTYPES: dict[object, tuple[int, int]] = {
    lucid.float32: (_C_engine.coreml.DTYPE_FLOAT32, _C_engine.coreml.BLOB_FLOAT32),
    lucid.float16: (_C_engine.coreml.DTYPE_FLOAT16, _C_engine.coreml.BLOB_FLOAT16),
    # Integer inputs (token ids, masks) reach the interface but never the
    # weight blob, so they have a MIL type and no blob type.
    lucid.int64: (_C_engine.coreml.DTYPE_INT32, -1),
    lucid.int32: (_C_engine.coreml.DTYPE_INT32, -1),
}

INT8: int = _C_engine.coreml.DTYPE_INT8
UINT8: int = _C_engine.coreml.MIL_SUBBYTE["8"]
BLOB_INT8: int = _C_engine.coreml.BLOB_INT8
BLOB_UINT8: int = _C_engine.coreml.BLOB_UINT8
INT32: int = _C_engine.coreml.DTYPE_INT32
#: The values an int32 constant or program-body integer can hold.
INT32_MIN, INT32_MAX = -(2**31), 2**31 - 1


def as_int32(values: list[int], name: str) -> list[int]:
    """``values`` for an int32 constant, refusing one Core ML would wrap.

    The writer stores each integer as given and Core ML keeps its low 32
    bits, so ``2**40`` became ``0`` with no error anywhere.

    Parameters
    ----------
    values : list of int
        The constant's elements.
    name : str
        The constant's value name, for the error.

    Returns
    -------
    list of int
        The same values as Python ints.

    Raises
    ------
    ValueError
        A value lies outside ``[INT32_MIN, INT32_MAX]``.
    """
    payload = [int(v) for v in values]
    for value in payload:
        if not INT32_MIN <= value <= INT32_MAX:
            raise ValueError(
                f"lucid.coreml: integer constant {name!r} holds {value}, outside "
                f"int32 [{INT32_MIN}, {INT32_MAX}]; Core ML integers are int32 "
                "and a wider value would be truncated"
            )
    return payload


def int32_clip_bounds(low: int, high: int | None) -> tuple[int, int | None]:
    """An integer clamp's bounds for a program whose integers are int32.

    A bound past int32 saturates exactly, since no int32 value lies beyond
    it — unless it forces every result past int32, which cannot be held.

    Parameters
    ----------
    low : int
        Lower bound, already truncated to an integer.
    high : int or None
        Upper bound, or ``None`` for a one-sided clamp.

    Returns
    -------
    tuple of (int, int or None)
        The bounds, saturated to int32.

    Raises
    ------
    ValueError
        ``low`` is above ``INT32_MAX`` or ``high`` below ``INT32_MIN``.
    """
    if low > INT32_MAX or (high is not None and high < INT32_MIN):
        raise ValueError(
            f"lucid.coreml: clamping to [{low}, {high}] puts every value outside "
            "int32, the only integer type a Core ML program computes in"
        )
    low = max(low, INT32_MIN)
    return low, None if high is None else min(high, INT32_MAX)


BOOL: int = _C_engine.coreml.DTYPE_BOOL
STRING: int = _C_engine.coreml.DTYPE_STRING
FLOAT32: int = _C_engine.coreml.DTYPE_FLOAT32


def mil_dtype(dtype: object) -> int:
    """MIL element-type number for a Lucid dtype.

    Raises
    ------
    TypeError
        Core ML has no storage for this dtype. ``float64`` is the common
        case and it is genuine — the format has no double.
    """
    entry = _DTYPES.get(dtype)
    if entry is None:
        raise TypeError(
            f"lucid.coreml: dtype {dtype} has no Core ML equivalent "
            f"(supported: {', '.join(str(d) for d in _DTYPES)})"
        )
    return entry[0]


FLOAT16: int = _C_engine.coreml.DTYPE_FLOAT16
BLOB_FLOAT16: int = _C_engine.coreml.BLOB_FLOAT16
BLOB_FLOAT32: int = _C_engine.coreml.BLOB_FLOAT32


def body_dtypes(precision: Precision) -> tuple[int, int]:
    """Element types the program body computes in.

    Parameters
    ----------
    precision : Precision
        Requested body precision. ``FLOAT16`` is what the Neural Engine
        runs; ``FLOAT32`` keeps the export faithful to the source model.

    Returns
    -------
    tuple[int, int]
        ``(MIL dtype, blob dtype)`` — the two numberings disagree.
    """
    if precision is Precision.FLOAT16:
        return FLOAT16, BLOB_FLOAT16
    return FLOAT32, BLOB_FLOAT32


def trace_dtype(name: str, body_mil: int) -> int:
    """MIL element type for a value the tracer produced.

    Float results follow the program body — an fp16 export wants fp16
    intermediates — but integer and boolean ones must not. Declaring an
    index tensor as float is not a rounding difference; MIL rejects the
    program, and where it does not the operation reads the wrong values.

    Parameters
    ----------
    name : str
        Lucid dtype name as the trace spells it (``"F32"``, ``"I64"``…).
    body_mil : int
        MIL dtype the body computes in, used for float results.

    Returns
    -------
    int
        MIL element-type number.

    Raises
    ------
    TypeError
        Core ML has no equivalent — ``float64`` is the reachable case.
    """
    if name == "F16":
        return FLOAT16
    if name == "F32":
        return body_mil
    if name in ("I64", "I32", "I16", "I8"):
        # Core ML's program dialect works in int32; a vocabulary index
        # never approaches its range.
        return INT32
    if name == "Bool":
        return BOOL
    raise TypeError(f"lucid.coreml: no Core ML element type for a {name} intermediate")
