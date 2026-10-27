"""The exported model: run it, check it, and see where it actually runs.

``CoreMLModel`` wraps a written ``.mlpackage`` plus the compiled handle
Core ML loaded from it.  Two of its methods exist because of how this
subsystem fails rather than what it does:

* :meth:`verify` compares against the eager model, because an exporter
  that drops a layer still writes a valid package and still returns
  plausible numbers.
* :meth:`compute_plan` asks Core ML which device each operation landed
  on, because asking for the Neural Engine and not getting it is silent.
  A float32 program requested with ``CPU_AND_NE`` reports **zero** ANE
  operations, runs at CPU speed, and warns about nothing.
"""

import weakref
from typing import TYPE_CHECKING, NamedTuple, Self, override

import lucid
from lucid._C import engine as _C_engine
from lucid._dispatch import _unwrap, _wrap
from lucid.coreml import _cache, _spec
from lucid.coreml._build import (
    _floor_of,
    _apply_image_normalisation,
    _named_examples,
    _select_outputs,
)
from lucid.coreml._spec import Classifier, ComputeUnits, ImageInput

if TYPE_CHECKING:
    from lucid._C.engine import TensorImpl
    from lucid._tensor.tensor import Tensor
    from lucid.nn.module import Module

__all__ = ["CoreMLModel", "PlacementSummary"]

#: Smallest reference magnitude a comparison can say anything about.
#:
#: The export path computes in single precision, so a reference below
#: this is being compared at the level of its own rounding: the
#: difference comes out tiny whether the package is right or not. An
#: untrained EfficientNet lands at 1e-12 and used to report agreement to
#: 1e-19, which reads as a flawless export and is a blind probe.
_COMPARABLE = 1e-6

_UNITS = {
    ComputeUnits.ALL: _C_engine.coreml.ComputeUnits.ALL,
    ComputeUnits.CPU_ONLY: _C_engine.coreml.ComputeUnits.CPU_ONLY,
    ComputeUnits.CPU_AND_GPU: _C_engine.coreml.ComputeUnits.CPU_AND_GPU,
    ComputeUnits.CPU_AND_NE: _C_engine.coreml.ComputeUnits.CPU_AND_NE,
}


def _reads_a_palette(path: str) -> bool:
    """Whether the package expands weights through a lookup table.

    Read off the serialized program rather than remembered from the
    build, so a package opened by ``load`` is judged the same way as one
    that has just been written. The operation's type is a plain string in
    the protobuf, so finding it needs no schema — and a false positive
    would only cost a compute unit, not correctness.
    """
    try:
        with open(f"{path}/Data/com.apple.CoreML/model.mlmodel", "rb") as handle:
            return b"constexpr_lut_to_dense" in handle.read()
    except OSError:
        return False


def _palettized_units(units: ComputeUnits, palettized: bool) -> ComputeUnits:
    """The units a palettized model may actually be run on.

    Core ML's GPU path expands ``constexpr_lut_to_dense`` incorrectly for
    the palette sizes 4, 16 and 256 — measured against the package's own
    tables, on macOS 26: a stack of eight 128-channel convolutions comes
    back with 15% error at four bits, while the same package on the CPU
    or the Neural Engine is exact to the last bit. Palette sizes 2, 8 and
    64 are unaffected.

    The failure is silent, and ``ALL`` is the default, so a palettized
    model would otherwise return plausible wrong numbers on the compute
    unit nobody chose. ``ALL`` therefore becomes ``CPU_AND_NE`` — the
    fast path on this hardware anyway — and an explicit request for the
    GPU is refused rather than quietly redirected, because a caller who
    named the GPU is owed an answer about the GPU.
    """
    if not palettized:
        return units
    if units is ComputeUnits.CPU_AND_GPU:
        raise ValueError(
            "lucid.coreml: a palettized model cannot be run on Core ML's GPU "
            "path — it expands the lookup table incorrectly for palette sizes "
            "4, 16 and 256, and does so without reporting an error. Use "
            "ComputeUnits.CPU_AND_NE (the default for these models) or "
            "CPU_ONLY, or export with weights=WeightPrecision.INT8, which the "
            "GPU handles correctly."
        )
    return ComputeUnits.CPU_AND_NE if units is ComputeUnits.ALL else units


class PlacementSummary:
    """Where a model's operations are scheduled.

    ``const`` operations carry no device assignment — they are data, not
    computation — so they are counted separately and kept out of the
    fraction. Reporting 21% ANE for a model whose every computation runs
    on the ANE would be true of the raw operation list and useless.

    Examples
    --------
    >>> import shutil, tempfile
    >>> import lucid, lucid.nn as nn, lucid.coreml as cml
    >>> model = nn.Sequential(nn.Conv2d(3, 16, 3, padding=1), nn.ReLU()).eval()
    >>> x, room = lucid.randn(1, 3, 32, 32), tempfile.mkdtemp()
    >>> package = cml.export(model, x, f"{room}/half.mlpackage",
    ...                      precision=cml.Precision.FLOAT16,
    ...                      compute_units=cml.ComputeUnits.CPU_ONLY)
    >>> plan = package.compute_plan()
    >>> plan.constants > 0 and plan.total_compute > 0   # counted apart
    True
    >>> plan.note                      # the CPU was asked for: nothing to say
    ''

    A float32 program asked for the Neural Engine is the case it speaks up
    for, since that device does not run float32:

    >>> single = cml.export(model, x, f"{room}/single.mlpackage",
    ...                     compute_units=cml.ComputeUnits.CPU_AND_NE)
    >>> single.compute_plan().ane_fraction
    0.0
    >>> print(single.compute_plan().note)
    no operation reached the Neural Engine because the program is float32, ...
    >>> package.close()
    >>> single.close()
    >>> shutil.rmtree(room)
    """

    def __init__(
        self,
        placements: list[tuple[str, str]],
        *,
        precision: str = "",
        units: ComputeUnits | None = None,
    ) -> None:
        self.placements = placements
        # What the model was built and opened as, so a plan of all-CPU
        # can say whether that was the request or the consequence of one.
        self.precision = precision
        self.units = units
        self.compute: dict[str, int] = {}
        self.constants = 0
        for op, device in placements:
            if device == "unknown":
                self.constants += 1
                continue
            self.compute[device] = self.compute.get(device, 0) + 1

    @property
    def total_compute(self) -> int:
        return sum(self.compute.values())

    @property
    def ane_fraction(self) -> float:
        """Share of computation the Neural Engine takes, 0.0 to 1.0.

        ``0.0`` on a model asked to use the ANE is the signal that the
        request did not take — most often because the program is float32,
        which the Neural Engine does not run.
        """
        total = self.total_compute
        return 0.0 if total == 0 else self.compute.get("ANE", 0) / total

    @property
    def note(self) -> str:
        """Why the Neural Engine took none of the work, when it took none.

        A float32 program cannot run on the Neural Engine at all — it is
        a float16 device — so an export that keeps the default precision
        lands entirely on the CPU however the compute units were asked
        for. Measured on a ResNet-18: 66 of 66 operations on the CPU at
        float32, 66 of 68 on the Neural Engine at float16.

        Nothing about that is an error, and Core ML reports no problem,
        so the only place it can surface is here — where somebody is
        already asking where the work went.

        The float32 explanation is given only for a program known to be
        float32. A float16 one that still lands elsewhere has another
        cause, and so may a package that does not record its precision —
        one Lucid did not write, or wrote before it recorded one — so
        those get the causes Core ML leaves unsaid instead of advice
        that would send the reader to a setting they already have.
        """
        wanted = self.units in (ComputeUnits.ALL, ComputeUnits.CPU_AND_NE)
        if not wanted or self.total_compute == 0 or self.ane_fraction > 0.0:
            return ""
        precision = self.precision.upper()
        if precision == "FLOAT32":
            return (
                "no operation reached the Neural Engine because the program is "
                "float32, which that device does not run — export with "
                "precision=Precision.FLOAT16 to reach it"
            )
        unrecorded = (
            ""
            if precision == "FLOAT16"
            else (
                "; this package does not record its precision, and a float32 "
                "program never reaches that device"
            )
        )
        return (
            "no operation reached the Neural Engine, and Core ML does not say "
            "why — the usual causes are a program too small for its planner to "
            "dispatch, one too large for the Neural Engine's compiler (splitting "
            "it into smaller packages avoids that), operations or shapes that "
            "device does not take, such as flexible shapes or tensors above "
            f"rank 4, and a machine without one, as a virtual machine is{unrecorded}"
        )

    @override
    def __repr__(self) -> str:
        parts = ", ".join(f"{d}={n}" for d, n in sorted(self.compute.items()))
        summary = (
            f"PlacementSummary({parts}, constants={self.constants}, "
            f"ane={self.ane_fraction:.0%})"
        )
        return f"{summary} — {self.note}" if self.note else summary


class Latency(NamedTuple):
    """What one prediction costs, with the settings that produced it.

    Carries the compute units and precision because a latency without
    them says nothing: the same package is three times slower with the
    accelerator withheld, and float32 forfeits the accelerator entirely.

    Examples
    --------
    >>> import shutil, statistics, tempfile
    >>> import lucid, lucid.nn as nn, lucid.coreml as cml
    >>> model = nn.Sequential(nn.Conv2d(3, 16, 3, padding=1), nn.ReLU()).eval()
    >>> x, room = lucid.randn(1, 3, 32, 32), tempfile.mkdtemp()
    >>> package = cml.export(model, x, f"{room}/m.mlpackage",
    ...                      precision=cml.Precision.FLOAT16,
    ...                      compute_units=cml.ComputeUnits.CPU_AND_NE)
    >>> package.benchmark(x)          # the first call after an export
    Latency(median=...ms, best=...ms, n=30, CPU_AND_NE, FLOAT16)
    >>> settled = statistics.median(   # and again, once it has settled
    ...     package.benchmark(x).median_ms for _ in range(3)
    ... )
    >>> settled > 0.0
    True
    >>> package.close()
    >>> shutil.rmtree(room)

    A package is slower on its first measured runs than it will be after
    a few — Core ML is still warming its own caches — so a single reading
    taken right after an export is not the number to publish. Measure a
    few times and take the median of those.
    """

    median_ms: float
    best_ms: float
    repeats: int
    compute_units: ComputeUnits
    precision: str

    @override
    def __repr__(self) -> str:
        return (
            f"Latency(median={self.median_ms:.2f}ms, best={self.best_ms:.2f}ms, "
            f"n={self.repeats}, {self.compute_units.name}, {self.precision})"
        )


class CoreMLModel:
    """A Core ML package written by Lucid, loaded and ready to run.

    Holds the ``.mlpackage`` on disk plus the compiled model Core ML
    produced from it. Compilation happens once per package content, not
    once per handle: the compiled model is kept in Lucid's cache and every
    later handle on the same content — in this process or another — opens
    it, which also lets Core ML reuse the bundle it specialised the model
    into rather than writing another (see :func:`lucid.coreml.empty_cache`).

    Attributes
    ----------
    path : str
        The ``.mlpackage`` this handle opened.
    input_names, output_names : list of str
        Features the package declares, in the order the program names
        them. ``predict`` accepts a tuple in this order or a mapping.
    noise_inputs : list of tuple
        ``(name, shape)`` for each input that stands in for a random
        draw, when the export lifted one — see
        :class:`~lucid.coreml.Draws`. Empty otherwise. A caller who
        passes nothing for these gets a fresh sample.
    deployment_target : DeploymentTarget
        Oldest system the package runs on. Three features raise it:
        carrying state, palettizing weights, several entry points.
    compute_units : ComputeUnits
        What the handle actually opened with, which is not always what
        was asked for — a palettized package is opened ``CPU_AND_NE``
        whatever the request, because Core ML's GPU path unpacks small
        palettes incorrectly.
    precision : str
        ``"FLOAT32"`` or ``"FLOAT16"``, as the package was written — the
        body's precision. ``"UNKNOWN"`` for a package that does not
        record it: one Lucid did not write, or wrote before it did.
    io_precision : str
        Element type of the float inputs and outputs, in the same
        spelling. ``"FLOAT16"`` makes :meth:`predict` convert float32
        inputs on the way in and return float16 tensors.
    palettized : bool
        Whether the program reads a palette.
    image_input : ImageInput or None
        Present when an input is declared as a picture.
    classifier : Classifier or None
        Present when the package answers with a label; read it through
        :meth:`classify` rather than :meth:`predict`.

    Examples
    --------
    >>> import shutil, tempfile
    >>> import lucid, lucid.nn as nn, lucid.coreml as cml
    >>> model = nn.Sequential(
    ...     nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(),
    ...     nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(16, 10),
    ... ).eval()
    >>> x, room = lucid.randn(1, 3, 32, 32), tempfile.mkdtemp()
    >>> package = cml.export(model, x, f"{room}/m.mlpackage")
    >>> package.predict(x).shape
    (1, 10)
    >>> package.verify(model, x) < 1e-5   # float32; a float16 package is nearer 1e-3
    True
    >>> package.close()

    A handle owns a compiled model, so close it — or use it as a context
    manager if you only need it for one call.

    >>> with cml.load(f"{room}/m.mlpackage") as reopened:
    ...     print(reopened.predict(x).shape)
    (1, 10)
    >>> shutil.rmtree(room)
    """

    def __init__(
        self,
        path: str,
        input_names: list[str],
        output_names: list[str],
        *,
        compute_units: ComputeUnits = ComputeUnits.ALL,
        precision: str = "FLOAT32",
        io_precision: str = "FLOAT32",
        output_shapes: dict[str, tuple[int, ...]] | None = None,
        image_input: ImageInput | None = None,
        classifier: Classifier | None = None,
        function_name: str = "",
        deployment_target: object = None,
        noise: list[tuple[str, tuple[int, ...], str, Tensor | None]] | None = None,
        traced_outputs: dict[str, Tensor] | None = None,
    ) -> None:
        self.path = path
        self.input_names = input_names
        self.output_names = output_names
        # A palettized package computes the wrong answer on Core ML's GPU
        # path — see ``_palettized_units`` — so the units it is opened
        # with are not always the units that were asked for.
        self.palettized = _reads_a_palette(path)
        self.compute_units = _palettized_units(compute_units, self.palettized)
        # Looked up before anything is opened, so a value that is not a
        # ComputeUnits fails here rather than with a cache entry held.
        units = _UNITS[self.compute_units]
        self.precision = precision
        # What the float features are declared as. Only a known float16
        # interface changes what ``predict`` does; anything else is fed
        # as it is given, which is what a foreign package has always had.
        self.io_precision = io_precision
        # Core ML's multi-array has no rank-0 form, so a model whose output
        # is a scalar comes back shaped (1,).  Keeping the traced shapes
        # lets ``predict`` hand back what the eager model would.
        self.output_shapes = output_shapes or {}
        # An image input changes two things: Core ML refuses a multi-array
        # for it, so the tensor has to reach the runtime as a pixel buffer;
        # and the normalisation now lives inside the package, so a
        # comparison against the eager model has to apply it on that side.
        self.image_input = image_input
        # How the package normalises those pixels, when it is known. It
        # is written into the program rather than declared, so a handle
        # that reopened the file knows *that* an input is an image and
        # not *what* was done to it — enough to predict, not enough to
        # compare against the eager model.
        self.image_normalisation = image_input if image_input is not None else None
        # A classifier returns a string and a dictionary, not arrays, so
        # it is read through ``classify`` rather than ``predict``.
        self.classifier = classifier
        self._declare_noise(noise or [])
        # What the traced run answered with those samples. A model that
        # draws cannot be re-run for a reference — it would draw again —
        # so this is the only comparison that has both sides computing
        # the same thing.
        self._traced_outputs = dict(traced_outputs or {})
        # Empty takes whichever entry point the package names as default.
        self.function_name = function_name
        # The oldest system this package runs on. Three features raise it
        # — state, palettization, several entry points — and a caller who
        # did not name a target is told what they ended up with rather
        # than finding out from a device.
        self.deployment_target = (
            deployment_target if deployment_target is not None else _floor_of(path)
        )
        # The compiled model is shared by every handle on the same package
        # content, in this process and others, and stays after this one
        # closes: Core ML keys the bundle it specialises a model into by
        # the compiled model's path, so reopening that path is what lets
        # the bundle be read again instead of written again. See _cache.
        self._lease = _cache.open_compiled(path)
        self._release = weakref.finalize(self, self._lease.release)
        try:
            # The engine holds a duplicate of the lease's lock for as long
            # as the model it loads lives — past ``close`` while another
            # thread's prediction is still running — so this side's lock
            # can go at ``close`` without the files going with it.
            self._handle = _C_engine.coreml.load_model(
                self._lease.path, units, function_name, self._lease.lock_fd
            )
        except RuntimeError as exc:
            # Whatever ends the load, the entry must not stay locked for
            # the rest of the process: a lock nobody will release is an
            # entry no eviction can ever remove.
            self._release()
            # Named after the package the caller gave, not the entry.
            raise RuntimeError(str(exc).replace(self._lease.path, path)) from None
        except BaseException:
            self._release()
            raise

    def _declare_noise(self, noise: list[tuple[str, tuple[int, ...], str, Tensor | None]]) -> None:
        """Record which inputs stand in for a draw, and what the trace drew."""
        # Inputs that stand in for a draw the model used to make itself.
        # A caller who passes nothing for them gets a fresh sample, so
        # the package behaves like the model it came from; a caller who
        # passes one gets a deterministic function, which is the other
        # reason to want this.
        self.noise_inputs = [(name, shape) for name, shape, _k, _s in noise]
        self._noise_kind = {name: kind for name, _shape, kind, _s in noise}
        # The samples the trace itself drew. A comparison against the
        # eager model needs the numbers that model actually used — it
        # cannot be asked to draw them again — and a handle that reopened
        # the file does not have them, so it refuses instead.
        # Empty for a handle that reopened the file: the samples are
        # not in it, which is why a comparison there refuses.
        self._traced_noise = {
            name: sample for name, _shape, _k, sample in noise if sample is not None
        }

    def _feed(self, x: object) -> list[tuple[str, TensorImpl]]:
        """Pair each input feature with its tensor.

        Accepts the same three shapes ``export`` did — a lone tensor, a
        tuple in the model's argument order, or a mapping — so a caller
        drives the package the way they built it.
        """
        # A lifted draw is an input of the package and not one of the
        # model, so a caller who names none of them is asking for what the
        # eager model did: a fresh sample. One who names some is asking
        # for a deterministic function, and gets it.
        supplied: dict[str, object] = {}
        if isinstance(x, dict):
            supplied = {k: v for k, v in x.items() if k in self._noise_kind}
            x = {k: v for k, v in x.items() if k not in self._noise_kind}
        asked = [name for name in self.input_names if name not in self._noise_kind]

        if isinstance(x, lucid.Tensor):
            given: list[tuple[str, object]] = [(asked[0], x)] if asked else []
            offered = 1
        elif isinstance(x, dict):
            given = list(x.items())
            offered = len(x)
        elif isinstance(x, (tuple, list)):
            given = list(zip(asked, x, strict=False))
            # Counted from what was handed over, not from what the pairing
            # kept: ``zip`` stops at the shorter side, so a caller who
            # passes one tensor too many would otherwise get a list that
            # agrees with itself and a prediction that quietly ignored it.
            offered = len(x)
        else:
            raise TypeError(
                f"lucid.coreml: expected a Tensor, a tuple, or a mapping — got "
                f"{type(x).__name__}"
            )
        if offered != len(asked):
            raise ValueError(
                f"lucid.coreml: this package takes {len(asked)} input(s) "
                f"{asked}, and {offered} were given"
            )
        for name, shape in self.noise_inputs:
            drawn = supplied.get(name)
            if drawn is None:
                drawn = (
                    lucid.randn(*shape) if self._noise_kind[name] == "randn" else lucid.rand(*shape)
                )
            given.append((name, drawn))

        images = {image for image, _color in self._images()}
        fed: list[tuple[str, TensorImpl]] = []
        for name, tensor in given:
            if name not in self.input_names:
                raise KeyError(
                    f"lucid.coreml: {name!r} is not an input of this package " f"{self.input_names}"
                )
            if not isinstance(tensor, lucid.Tensor):
                raise TypeError(
                    f"lucid.coreml: input {name!r} must be a Tensor — got "
                    f"{type(tensor).__name__}"
                )
            if tensor.dtype in (lucid.int64, lucid.int32):
                # Core ML's multi-array has int32 and no int64, so an
                # integer input is narrowed here rather than at every call
                # site. Token ids and masks are nowhere near the range
                # where that loses anything.
                tensor = tensor if tensor.dtype == lucid.int32 else tensor.to(lucid.int32)
            elif (
                tensor.dtype == lucid.float32
                and self.io_precision == "FLOAT16"
                and name not in images
            ):
                # The package reads half precision here, and the model the
                # caller compares it with takes single — so the same tensor
                # serves both, converted where the package needs it. A
                # lifted draw is sampled at float32 and arrives this way.
                tensor = tensor.half()
            fed.append((name, tensor._impl))
        return fed

    @property
    def carries_state(self) -> bool:
        """Whether the package keeps values between predictions."""
        return bool(self._handle.carries_state)

    def reset_state(self) -> None:
        """Forget everything the package has accumulated.

        A state persists across predictions by design, so starting a fresh
        sequence has to be asked for; there is no other way back to the
        value it began at.

        Raises
        ------
        ValueError
            The package carries no state.
        """
        self._handle.reset_state()

    @property
    def state_names(self) -> tuple[str, ...]:
        """Names of the states the package carries between predictions, sorted."""
        return tuple(self._handle.state_names)

    def read_state(self, name: str) -> Tensor:
        """A state's current value, as a CPU tensor of its element type.

        Parameters
        ----------
        name : str
            One of :attr:`state_names`.

        Returns
        -------
        Tensor
            A copy; writing to it does not change the state.

        Raises
        ------
        ValueError
            The package carries no state, or none by that name.
        """
        return _wrap(self._handle.read_state(name))

    def write_state(self, name: str, value: Tensor) -> None:
        """Overwrite a state — to seed a stream, or to restore a saved one.

        A state was write-only from outside the package: a stream whose
        first step differs from the rest (a video decoder that skips its
        temporal upsampling on frame 0) could not be seeded, and what one
        had accumulated could not be saved or inspected.

        Parameters
        ----------
        name : str
            One of :attr:`state_names`.
        value : Tensor
            The new value, of the state's shape.  It is converted to the
            state's element type and copied to the host.

        Raises
        ------
        ValueError
            The package carries no state, no state by that name, or the
            value's shape differs from the state's.
        """
        current = _wrap(self._handle.read_state(name))
        if tuple(value.shape) != tuple(current.shape):
            raise ValueError(
                f"write_state: state {name!r} has shape {tuple(current.shape)}, "
                f"the value has {tuple(value.shape)}"
            )
        host = value.detach().to("cpu").to(current.dtype).contiguous()
        self._handle.write_state(name, _unwrap(host))

    def _images(self) -> list[tuple[str, int]]:
        if self.image_input is None:
            return []
        return [(self.input_names[0], _spec.color_space(self.image_input.color))]

    def classify(self, x: object) -> tuple[str, dict[str, float]]:
        """Run a classifier package and read back what it names.

        Parameters
        ----------
        x : Tensor or tuple of Tensor or dict of str to Tensor
            Input, in the same shapes :meth:`predict` accepts.

        Returns
        -------
        tuple[str, dict[str, float]]
            The winning label, and every label with its probability.

        Raises
        ------
        TypeError
            The package was not exported with a classifier.
        """
        if self.classifier is None:
            raise TypeError(
                "lucid.coreml: this package returns scores, not labels — export it "
                "with classifier=Classifier(labels=...) to get labels"
            )
        label, scores = self._handle.classify(
            self._feed(x),
            self._images(),
            self.classifier.label_name,
            self.classifier.probabilities_name,
        )
        return label, {name: float(value) for name, value in scores}

    def predict(self, x: object) -> Tensor | dict[str, Tensor]:
        """Run the model.

        Inputs must be host tensors: Core ML reads host memory, and moving
        a Metal tensor here would hide a copy the caller did not ask for.
        Move it explicitly with ``.to("cpu")``.

        Parameters
        ----------
        x : Tensor or tuple of Tensor or dict of str to Tensor
            One tensor for a single-input package; otherwise a tuple in
            the package's input order, or a mapping by feature name.

        Returns
        -------
        Tensor or dict[str, Tensor]
            The output for a single-output package; otherwise every
            output, keyed by the field the model declared it as.
        """
        if self.classifier is not None:
            raise TypeError(
                "lucid.coreml: this package returns a label and a probability map, "
                "not arrays — use classify()"
            )
        raw = self._handle.predict(self._feed(x), self.output_names, self._images())
        produced: dict[str, Tensor] = {}
        for name, impl in zip(self.output_names, raw, strict=True):
            out = _wrap(impl)
            declared = self.output_shapes.get(name)
            # Core ML returns a 0-d result as ``(1,)``. An axis that follows
            # the input is ``-1`` here and keeps whatever size came back.
            if declared is not None and (
                len(out.shape) != len(declared) if -1 in declared else out.shape != declared
            ):
                out = out.reshape(*declared)
            produced[name] = out
        if len(self.output_names) == 1:
            return produced[self.output_names[0]]
        return produced

    def verify(self, model: Module, x: object, *, relative: bool = False) -> float:
        """Largest difference against the eager model.

        Shapes agreeing is not evidence: a package missing a layer has
        the right shape and returns plausible numbers. This runs both and
        compares values — every output, not just the first, since a
        detector that exported its class scores and dropped its boxes
        would otherwise pass.

        The default is an **absolute** difference, which is only
        interpretable against outputs of a known size. A model whose
        outputs differ in magnitude makes that trap easy to fall into:
        RealNVP returns a latent of order 1 beside a log-probability of
        order 1e4, so the absolute worst is set by the second, and
        dividing it by the first reads as a 4% error when every output
        agrees to 1e-6.

        ``relative=True`` scales each output's difference by that
        output's own magnitude — but only down to one, which is the
        other half of the same trap: VQ-VAE's latent has values around
        2e-3, and dividing float32 noise by that reads as 3e-4 when the
        difference is 5e-7. Dividing by ``max(scale, 1)`` is relative
        where relative means something and absolute where it does not,
        which is the same bargain a tolerance pair makes.

        Parameters
        ----------
        model : nn.Module
            The eager model this package was exported from.
        x : Tensor or tuple of Tensor or dict of str to Tensor
            Input to run through both. Host tensors.
        relative : bool, optional, default=False
            Scale each output's difference by that output's own largest
            magnitude, floored at one, before taking the worst.

        Returns
        -------
        float
            The worst ``max|coreml - eager|`` across the outputs, or the
            worst of those divided by each output's own scale when
            ``relative``. Expect ~1e-7 for a float32 export and ~1e-3
            relative for float16.

        Notes
        -----
        For an image export the pixel buffer is eight bits per channel,
        so anything but whole numbers in ``[0, 255]`` is rounded on the
        way in and the two sides see different pixels. That is refused
        rather than reported: for ``randn`` values the rounding is most
        of the signal and the answer would be around 3e-1, which reads
        as a broken export. Feed pixels and the comparison is the usual
        one.
        """
        if self.classifier is not None:
            raise TypeError(
                "lucid.coreml: a classifier's output is a label and a probability "
                "map; compare them with classify() rather than verify()"
            )
        if self.carries_state:
            raise TypeError(
                "lucid.coreml: this package carries state, so one prediction says "
                "nothing about whether it agrees — the eager model would have to "
                "be threaded through the same sequence. Run both over several "
                "steps and compare, with reset_state() between runs"
            )
        if self.noise_inputs and not self._traced_noise:
            raise ValueError(
                "lucid.coreml: this package takes its random draws as inputs, and "
                "the samples the export used are not written into the file — so a "
                "handle from load() has nothing to compare with. Running the eager "
                "model would draw different numbers and measure that instead of "
                "the export. Verify the handle export returned"
            )
        examples, by_keyword = _named_examples(x)
        if self.image_input is not None:
            # A pixel buffer is eight bits per channel, so an input that
            # is not already whole numbers in [0, 255] is rounded on the
            # way in and the two sides genuinely see different pixels.
            # For ``randn`` values that is most of the signal, and the
            # number this would return — around 3e-1 — reads as a broken
            # export rather than as the round-trip it measures.
            for name, tensor in examples:
                low = float(tensor.min().item())
                high = float(tensor.max().item())
                integral = bool(((tensor - tensor.round()).abs().max() < 1e-6).item())
                if low < 0.0 or high > 255.0 or not integral:
                    raise TypeError(
                        f"lucid.coreml: {name!r} is not pixel data (range "
                        f"[{low:.3g}, {high:.3g}], "
                        f"{'non-integral' if not integral else 'integral'}), and this "
                        "package takes an image. Core ML would quantise it to eight "
                        "bits per channel, so the comparison would measure that "
                        "rounding rather than the network. Feed whole numbers in "
                        "[0, 255] to compare the two models."
                    )
            # The package normalises the pixels itself, so the eager model
            # has to be shown the same normalised values or the comparison
            # is between two different inputs.
            if self.image_normalisation is None:
                raise ValueError(
                    "lucid.coreml: this package takes an image, and the "
                    "normalisation it applies is written into the program "
                    "rather than into anything the file declares — so a "
                    "handle from load() cannot recover it. Comparing "
                    "without it would measure the missing scale and bias "
                    "instead of the export. Pass the ImageInput the "
                    "package was written with, or verify the handle export "
                    "returned"
                )
            examples = [
                (name, _apply_image_normalisation(tensor, self.image_normalisation))
                for name, tensor in examples
            ]
        # The package runs first. A model may write into its own input — a
        # cache it fills in place — and running it eagerly first would
        # hand the package an input the write had already been applied
        # to, so the comparison measured the write twice.
        fed = x
        if self.noise_inputs:
            feed = dict(_named_examples(x)[0])
            feed.update(self._traced_noise)
            fed = feed
        got = self.predict(fed)
        if by_keyword:
            reference = model(**dict(examples))
        else:
            reference = model(*(tensor for _, tensor in examples))
        expected = dict(_select_outputs(reference, None))

        if self.noise_inputs:
            # The eager model was run for its shape and its field names,
            # which are what would change if the model moved out from
            # under the package. Its *values* cannot be the reference: it
            # drew its own sample and the package was given the trace's,
            # so comparing them would measure two different draws. The
            # traced answer is what the package is being asked to
            # reproduce, and it was computed from exactly the numbers the
            # package was fed.
            for name in self.output_names:
                if name not in expected:
                    raise KeyError(
                        f"lucid.coreml: the eager model no longer returns {name!r}, "
                        "which this package exports"
                    )
            expected = dict(self._traced_outputs)

        produced = got if isinstance(got, dict) else {self.output_names[0]: got}

        worst = 0.0
        carries_signal = False
        for name in self.output_names:
            wanted = expected.get(name)
            if wanted is None:
                raise KeyError(
                    f"lucid.coreml: the eager model no longer returns {name!r}, "
                    "which this package exports"
                )
            extreme = float(wanted.abs().max().item())
            if extreme != extreme:  # NaN: the reference has no finite answer
                raise ValueError(
                    f"lucid.coreml: the eager model's {name!r} contains NaN, so "
                    "there is nothing to compare the package against — the model "
                    "produces it before any export is involved, and a difference "
                    "against NaN is NaN whatever the package computed"
                )
            carries_signal = carries_signal or extreme > _COMPARABLE
            answer = produced[name]
            if answer.dtype != wanted.dtype and answer.dtype == lucid.float16:
                # A float16 interface answers in half precision; the gap is
                # measured at the eager model's, where the reference is.
                answer = answer.to(wanted.dtype)
            gap = float((answer - wanted).abs().max().item())
            worst = max(worst, gap / max(extreme, 1.0) if relative else gap)
        if not carries_signal:
            # Comparing against a reference with no magnitude proves
            # nothing: an exporter that dropped every layer would score
            # just as well. Several zoo models zero-initialise their
            # head, so this is reachable with an untrained factory
            # rather than being a theoretical case.
            #
            # Not only exact zeros. An untrained EfficientNet answers
            # with logits around 1e-12, and the comparison then reports
            # agreement to 1e-19 — a number that reads as a perfect
            # export and is really a blind probe. Below ``_COMPARABLE``
            # the difference cannot separate a correct package from a
            # broken one at single precision, so the refusal names the
            # magnitude rather than pretending.
            #
            # Only when *every* output is that small, though. A model can
            # have one output that is legitimately zero and others that
            # are not — NICE's log-determinant is exactly zero because
            # the transform preserves volume, which is a fact about the
            # architecture and not a missing weight.
            largest = max(
                (
                    float(expected[name].abs().max().item())
                    for name in self.output_names
                    if expected.get(name) is not None
                ),
                default=0.0,
            )
            raise ValueError(
                f"lucid.coreml: the eager model's outputs "
                f"({', '.join(self.output_names)}) reach only {largest:.2e}, "
                f"which is below the {_COMPARABLE:.0e} this comparison needs to "
                "tell a correct package from a broken one — it would report "
                "agreement either way. Load weights, or perturb the "
                "zero-initialised parameters, before verifying"
            )
        return worst

    def benchmark(self, x: object, *, repeats: int = 30, warmup: int = 5) -> Latency:
        """How long one prediction takes, measured the way it should be.

        The first calls are not the model: Core ML defers work to them —
        specialising for the units it was given, laying out weights the
        accelerator wants — and a timing that includes them reports the
        setup. Hence a warmup that is thrown away, and a median over
        repeats rather than a mean, since a scheduling hiccup on a shared
        machine moves a mean and not a median.

        Measured on an M1 Pro with a ResNet-18 at 224 square: 18.6 ms
        eager, 4.5 ms as a float32 package on the CPU, 2.7 ms at float16
        on the CPU, and 1.5 ms with the Neural Engine allowed — so the
        accelerator is worth about 12x against eager and 3x against the
        same package on the CPU. Those are this machine's numbers and
        will not be yours, which is why this exists rather than a
        documented figure.

        Parameters
        ----------
        x : Tensor or tuple of Tensor or dict of str to Tensor
            Input to run, in the shape the package was built for.
        repeats : int, optional, keyword-only, default=30
            Timed calls.
        warmup : int, optional, keyword-only, default=5
            Calls made and discarded first.

        Returns
        -------
        Latency
            Median and best of the timed calls, in milliseconds.

        Raises
        ------
        ValueError
            When ``repeats`` is not positive.
        """
        import statistics
        import time

        if repeats < 1:
            raise ValueError(
                f"lucid.coreml: benchmark needs at least one timed call, got " f"{repeats}"
            )
        for _ in range(max(warmup, 0)):
            self.predict(x)
        timings: list[float] = []
        for _ in range(repeats):
            started = time.perf_counter()
            self.predict(x)
            timings.append((time.perf_counter() - started) * 1000.0)
        return Latency(
            median_ms=statistics.median(timings),
            best_ms=min(timings),
            repeats=repeats,
            compute_units=self.compute_units,
            precision=self.precision,
        )

    def compute_plan(self) -> PlacementSummary:
        """Which device Core ML assigns each operation to.

        Requires macOS 14.4+; an empty plan there means *unknown*, not
        *unaccelerated*.
        """
        # The model this handle opened, rather than the package compiled
        # again; a closed handle no longer holds it, and compiles afresh.
        placements = _C_engine.coreml.compute_plan(
            self._lease.path if self._lease.held else self.path,
            _UNITS[self.compute_units],
        )
        return PlacementSummary(
            [(op, device) for op, device in placements],
            precision=self.precision,
            units=self.compute_units,
        )

    def close(self) -> None:
        """Release the compiled model.

        The compiled model itself stays in Lucid's cache for the next
        handle on the same package — see :func:`lucid.coreml.empty_cache`.
        A prediction still running on another thread finishes with the
        model it took, which stays loaded, and protected from eviction,
        until it returns. Safe to call twice, so a ``finally`` beside a
        ``with`` is fine.
        """
        self._handle.close()
        self._release()

    def __enter__(self) -> Self:
        """Return the handle, so a package can be opened in a ``with``.

        A handle owns a compiled model and a directory Core ML wrote it
        into, and every use of one in this codebase was already a
        ``try``/``finally`` around :meth:`close`.

        Returns
        -------
        CoreMLModel
            This handle.
        """
        return self

    def __exit__(self, kind: object, value: object, trace: object) -> None:
        """Close the handle, whether the block ended well or not.

        Parameters
        ----------
        kind : type or None
            Exception class, when the block raised.
        value : BaseException or None
            The exception itself.
        trace : TracebackType or None
            Its traceback.
        """
        self.close()

    @override
    def __repr__(self) -> str:
        # The interface is named only when it is not the float32 default,
        # so the common case reads as it always has.
        io = f", io={self.io_precision}" if self.io_precision == "FLOAT16" else ""
        return (
            f"CoreMLModel({self.path!r}, precision={self.precision}{io}, "
            f"units={self.compute_units.value})"
        )
