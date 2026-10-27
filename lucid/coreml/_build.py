"""Trace a Lucid model and build the Core ML package it describes.

The tracer that :mod:`lucid.compile` uses to lower a forward pass into
MPSGraph produces a complete graph IR.  Core ML is a *second backend* for
that IR — MIL instead of MPSGraph — not a second front end, so nothing
here re-walks modules or re-derives shapes.

Two things the trace hands over that the driver has to sort out:

* **which external feed is the input.**  Everything a traced graph reads
  from outside is an "external feed": the model's parameters and buffers
  as well as the argument.  The argument is identified by object
  identity, the rest become constants.
* **which value is the output.**  Zoo models return an output dataclass
  (``ImageClassificationOutput`` and friends) rather than a tensor, so a
  bare ``isinstance`` check would reject every model in the zoo.
"""

from collections.abc import Callable, Iterator, Sequence
from typing import TYPE_CHECKING, Any, NoReturn, cast, override

import dataclasses
import math
import contextlib
import struct

import lucid
import lucid.compile as _compile
import lucid.nn as _nn
import lucid.nn.functional as _F
import lucid.quantization as _quant
from lucid._C import engine as _C_engine
from lucid._dispatch import _unwrap, _wrap
from lucid.coreml import _flex, _fold, _spec, _staging
from lucid.coreml._convert import MIL_CAST_NAMES, emit_cast
from lucid.coreml._flex import ShapeNotFlexible
from lucid.coreml._emit import (
    EMITTERS,
    Bindings,
    Bound,
    Constant,
    MultiOutput,
    TracedOp,
    _as_float,
    _as_int,
)
from lucid.coreml._spec import (
    Classifier,
    State,
    ColorSpace,
    ImageInput,
    Metadata,
    Precision,
    WeightPrecision,
)

if TYPE_CHECKING:
    from lucid._C.engine import BlobWriter, MilProgram, PackagePaths
    from lucid._C.engine.compile import TraceGraph, TraceNode, Tracer
    from lucid._tensor.tensor import Tensor
    from lucid.nn.module import Module

__all__ = ["Builder", "UnsupportedOp", "build_package", "trace"]


class UnsupportedRank(NotImplementedError):
    """A tensor exceeds the rank Core ML's program dialect allows.

    Not something the writer can work around: the limit is the format's.
    Window-attention models hit it — Swin partitions into
    ``(B, H/w, w, W/w, w, C)``, which is rank six — and the honest answer
    is to say which operation and which shape rather than to reshape
    around it and hope the semantics survive.

    Examples
    --------
    >>> import shutil, tempfile
    >>> import lucid, lucid.nn as nn, lucid.coreml as cml
    >>> class Tall(nn.Module):
    ...     def forward(self, x):
    ...         return x.reshape(1, 4, 2, 4, 2, 3).sum(dim=5)
    >>> room = tempfile.mkdtemp()
    >>> try:
    ...     cml.export(Tall().eval(), lucid.randn(1, 8, 8, 3), f"{room}/m.mlpackage")
    ... except cml.UnsupportedRank as refusal:
    ...     print(refusal)
    lucid.coreml: op 'reshape' produces a rank-6 tensor (1, 4, 2, 4, 2, 3), ...
    >>> shutil.rmtree(room)
    """

    def __init__(self, op_name: str, shape: tuple[int, ...]) -> None:
        super().__init__(
            f"lucid.coreml: op {op_name!r} produces a rank-{len(shape)} tensor "
            f"{shape}, and Core ML's program dialect allows at most rank "
            f"{_staging.MAX_RANK}. This is a format limit, not a missing emitter."
        )
        self.op_name = op_name
        self.shape = shape


class UnsupportedOp(NotImplementedError):
    """A traced op has no MIL translation, or a refused one.

    Named rather than generic because the alternative failure — emitting
    a package that quietly lacks a layer — produces a model that loads
    and returns plausible numbers.

    Parameters
    ----------
    op_name : str
        Lucid operation that could not be emitted.
    reason : str or None, optional, default=None
        Why, when the answer is something other than "nobody has written
        the emitter yet". An operation Core ML translates *badly* — one
        whose package would load and be wrong — is refused deliberately,
        and pointing the reader at ``_emit.py`` to add a mapping would
        send them to write the very thing that was rejected.

    Examples
    --------
    A matrix inverse, which Core ML's program dialect has no operation for:

    >>> import shutil, tempfile
    >>> import lucid, lucid.nn as nn, lucid.coreml as cml
    >>> class Inverts(nn.Module):
    ...     def forward(self, x):
    ...         return lucid.linalg.inv(x)
    >>> room = tempfile.mkdtemp()
    >>> try:
    ...     cml.export(
    ...         Inverts().eval(), lucid.eye(4).reshape(1, 4, 4), f"{room}/m.mlpackage"
    ...     )
    ... except cml.UnsupportedOp as refusal:
    ...     print(refusal)
    lucid.coreml: no Core ML translation for Lucid op 'inv'. Mapped ops: ...
    >>> shutil.rmtree(room)
    """

    def __init__(self, op_name: str, reason: str | None = None) -> None:
        super().__init__(
            f"lucid.coreml: {op_name} — {reason}"
            if reason is not None
            else (
                f"lucid.coreml: no Core ML translation for Lucid op "
                f"{op_name!r}. Mapped ops: {', '.join(sorted(EMITTERS))}. "
                f"Add an emitter in lucid/coreml/_emit.py."
            )
        )
        self.op_name = op_name
        self.reason = reason


def _select_outputs(output: object, field: str | None) -> list[tuple[str, Tensor]]:
    """Every tensor the model returns, named, in the order it declares them.

    Exporting one field of a several-field output and calling the result
    the model is the same failure as dropping a layer, one level up: a
    detector reduced to its class scores loads, runs, and returns
    plausible numbers with no boxes in them. So the default is all of
    them, and ``field`` is how a caller asks for less.
    """
    if isinstance(output, lucid.Tensor):
        return [("output", output)]
    if field is not None:
        picked = getattr(output, field, None)
        if not isinstance(picked, lucid.Tensor):
            raise TypeError(
                f"lucid.coreml: {type(output).__name__}.{field} is "
                f"{type(picked).__name__}, not a Tensor"
            )
        return [(field, picked)]

    named: list[tuple[str, Tensor]] = []
    if isinstance(output, tuple):
        # A named tuple brings its own names; a plain one is positional,
        # and a state pair has to be able to refer to one of them.
        labels = getattr(output, "_fields", None)
        for position, value in enumerate(output):
            if not isinstance(value, lucid.Tensor):
                continue
            named.append((labels[position] if labels is not None else f"output_{position}", value))
        if named:
            return named
    if dataclasses.is_dataclass(output) and not isinstance(output, type):
        for spec in dataclasses.fields(output):
            value = getattr(output, spec.name)
            if isinstance(value, lucid.Tensor):
                named.append((spec.name, value))
    if named:
        return named
    raise TypeError(
        f"lucid.coreml: the model returned {type(output).__name__}, which carries "
        "no tensor to export. Name the field with output_field=..., or return a "
        "Tensor."
    )


def _named_examples(example: object) -> tuple[list[tuple[str, Tensor]], bool]:
    """The example input(s), named, and whether they are passed by keyword.

    A model with more than one input is the common case, not the exotic
    one — a transformer takes an attention mask, an encoder-decoder takes
    both halves — so a tuple is passed positionally and a mapping by
    keyword, which is how the model would be called anyway.
    """
    if isinstance(example, lucid.Tensor):
        return [("input", example)], False
    if isinstance(example, dict):
        for name, value in example.items():
            if not isinstance(value, lucid.Tensor):
                raise TypeError(
                    f"lucid.coreml: example {name!r} is {type(value).__name__}, " "not a Tensor"
                )
        return list(example.items()), True
    if isinstance(example, (tuple, list)):
        named: list[tuple[str, Tensor]] = []
        for position, value in enumerate(example):
            if not isinstance(value, lucid.Tensor):
                raise TypeError(
                    f"lucid.coreml: example {position} is {type(value).__name__}, " "not a Tensor"
                )
            named.append((f"input_{position}", value))
        if not named:
            raise ValueError("lucid.coreml: no example inputs given")
        return named, False
    raise TypeError(
        f"lucid.coreml: example must be a Tensor, a tuple of Tensors, or a "
        f"mapping of name to Tensor — got {type(example).__name__}"
    )


class _DequantizedLinear(_nn.Module):
    """A packed MLX linear, computed the way an exported graph can follow.

    ``QuantizedLinearMLX`` runs a Metal-only grouped GEMM over a packed
    weight and moves the activation to the GPU and back to do it. None of
    that reaches the tracer, so a model holding one exported to a graph
    its own output had fallen out of.

    The module's own documentation names the way through: the packed form
    has a dequantize-to-float reference path, and it is not an
    approximation of the kernel but the same arithmetic — measured at
    4.8e-07 between them. So the weight is reconstructed once, on the
    host, and the layer becomes the ordinary ``linear`` the exporter
    already writes.

    What is lost is the packing, not the quantization: the reconstructed
    weight carries the values MLX's grid gave it, and asking the export
    for ``INT8`` stores them again on a grid of its own.

    Parameters
    ----------
    source : nn.Module
        The ``QuantizedLinearMLX`` to stand in for.
    """

    def __init__(self, source: Module) -> None:
        super().__init__()
        from lucid.quantization import _qgemm

        dense = _qgemm.dequantize(
            cast("Tensor", source.packed_weight),
            cast("Tensor", source.scales),
            cast("Tensor", source.biases),
            group_size=int(cast("int", source.group_size)),
            bits=int(cast("int", source.bits)),
        )
        self.weight = _nn.Parameter(dense.cpu())
        held = cast("Tensor | None", getattr(source, "bias", None))
        self.bias = _nn.Parameter(held.cpu()) if held is not None else None
        self.relu = bool(getattr(source, "relu", False))

    @override
    def forward(self, *args: Tensor, **kwargs: object) -> Tensor:
        """Apply the reconstructed weight, and the fused ReLU if there was one.

        Parameters
        ----------
        *args : Tensor
            The activation, as the layer this stands in for takes it.
        **kwargs : object
            Accepted and unused, to match the base signature.

        Returns
        -------
        Tensor
            ``x @ weightᵀ + bias``, through a ReLU when one was fused.
        """
        out = _F.linear(args[0], self.weight, self.bias)
        return _F.relu(out) if self.relu else out


@contextlib.contextmanager
def _packed_linears_unpacked(model: Module) -> Iterator[None]:
    """Stand in for every packed MLX linear while the model is traced.

    Swapped on the parent by attribute name and put back afterwards,
    whether the trace succeeded or not — the caller's model is not the
    export's to change.

    Recognised by the attributes the packing needs rather than by class,
    which keeps this from importing ``lucid.nn.quantized`` to name it.

    Parameters
    ----------
    model : nn.Module
        Model about to be traced.

    Yields
    ------
    None
        For the duration of the trace.
    """
    swapped: list[tuple[Module, str, Module]] = []
    for _name, parent in model.named_modules():
        for attribute, child in list(vars(parent).get("_modules", {}).items()):
            if child is None:
                continue
            if not all(
                hasattr(child, field)
                for field in ("packed_weight", "scales", "biases", "group_size", "bits")
            ):
                continue
            swapped.append((parent, attribute, child))
    for parent, attribute, child in swapped:
        setattr(parent, attribute, _DequantizedLinear(child))
    try:
        yield
    finally:
        for parent, attribute, child in swapped:
            setattr(parent, attribute, child)


@contextlib.contextmanager
def _activation_quantization_paused(model: Module) -> Iterator[None]:
    """Take a model's activation fake-quantization out of the trace.

    Only the activation one. A quantization-aware module holds its
    weight's fake-quantize as ``weight_fake_quant`` and its output's as
    ``activation_post_process``, and dropping the first would throw away
    the grid the weights were trained onto — the whole reason to export
    such a model at all.

    Told apart by the attribute they hang from rather than by anything
    about the module, since both are the same class. That is a name to
    depend on, and it is the one ``lucid.quantization`` uses.

    Parameters
    ----------
    model : nn.Module
        Model about to be traced.

    Yields
    ------
    None
        For the duration of the trace.
    """
    paused = [
        module
        for name, module in model.named_modules()
        if name.rsplit(".", 1)[-1] == "activation_post_process"
        and hasattr(module, "disable_fake_quant")
        and hasattr(module, "enable_fake_quant")
    ]
    was_on = [bool(getattr(module, "_fake_quant_enabled", False)) for module in paused]
    for module in paused:
        module.disable_fake_quant()
    try:
        yield
    finally:
        for module, on in zip(paused, was_on, strict=True):
            if on:
                module.enable_fake_quant()


@contextlib.contextmanager
def _observers_paused(model: Module) -> Iterator[None]:
    """Stop calibration machinery from writing while the model is traced.

    A quantization-aware model carries observers that record the range of
    every activation they see, and ``eval()`` does not stop them —
    correctly, because post-training calibration runs in eval mode and is
    exactly that recording. Tracing is not calibration: it runs the model
    once to learn its shape, and letting the observers write turns that
    single pass into a buffer change the export then refuses as a model
    that mutates itself.

    So they are paused around the trace and put back the way they were
    found, whether it succeeded or not. Nothing about the exported
    program changes — an observer contributes no operation, only a
    record of what passed through it.

    Recognised by shape rather than by class: anything carrying
    ``disable_observer`` and ``enable_observer`` is treated as one, which
    keeps this from importing the quantization subsystem to name it.

    Parameters
    ----------
    model : nn.Module
        Model about to be traced.

    Yields
    ------
    None
        For the duration of the trace.
    """
    paused = [
        module
        for _name, module in model.named_modules()
        if hasattr(module, "disable_observer") and hasattr(module, "enable_observer")
    ]
    was_on = [bool(getattr(module, "_observer_enabled", False)) for module in paused]
    for module in paused:
        module.disable_observer()
    try:
        yield
    finally:
        for module, on in zip(paused, was_on, strict=True):
            if on:
                module.enable_observer()


def _bind_traced_inputs(
    examples: list[tuple[str, Tensor]],
    held: list[_C_engine.TensorImpl],
    written: set[int],
    feeds: dict[int, _C_engine.TensorImpl],
    tracer: Tracer,
) -> list[tuple[str, int, Tensor]]:
    """Bind each example input to its trace value id."""
    feed_of = {id(impl): int(tid) for tid, impl in feeds.items()}
    inputs: list[tuple[str, int, Tensor]] = []
    for (name, tensor), impl in zip(examples, held, strict=True):
        value_id = feed_of.get(id(impl)) if id(impl) in written else tracer.lookup_id(impl)
        if value_id is None:
            raise ValueError(
                f"lucid.coreml: example input {name!r} never reached an op — the "
                "model ignored it, or copied it before use"
            )
        inputs.append((name, value_id, tensor))
    return inputs


def _bind_traced_outputs(
    selected: list[tuple[str, Tensor]], answered: dict[str, Tensor], tracer: Tracer
) -> list[tuple[str, int, Tensor]]:
    """Bind each selected output to its trace value id."""
    outputs: list[tuple[str, int, Tensor]] = []
    for name, tensor in selected:
        value_id = tracer.lookup_id(_unwrap(tensor))
        if value_id is None:
            raise ValueError(
                f"lucid.coreml: output {name!r} did not come out of the traced "
                "graph. A constant the trace holds as a buffer (``arange``, a "
                "tensor from a list) does not depend on the input: return it from "
                f"outside the model. Otherwise it was computed on the host. {_HOST_READS}"
            )
        outputs.append((name, value_id, answered.get(name, tensor)))
    return outputs


def _restore_traced_examples(
    tracer: Tracer | None, examples: list[tuple[str, Tensor]], held: list[_C_engine.TensorImpl]
) -> list[tuple[str, bool]]:
    if tracer is not None:
        tracer.restore_outside_writes()
    replaced: list[tuple[str, bool]] = []
    for (name, tensor), impl in zip(examples, held, strict=True):
        if _unwrap(tensor) is not impl:
            replaced.append((name, bool(impl.requires_grad)))
            tensor._impl = impl
    return replaced


def trace(model: Module, example: object, *, output_field: str | None = None) -> Any:
    """Run one traced forward pass.

    Parameters
    ----------
    model : nn.Module
        Model to trace. Should already be in ``eval()`` mode.
    example : Tensor or tuple of Tensor or dict of str to Tensor
        Input(s) to trace with; their identity is how the driver tells the
        arguments apart from the parameters, which are external feeds too.
        A tuple is passed positionally, a mapping by keyword.
    output_field : str or None, optional, keyword-only, default=None
        Single attribute to take when the model returns an output
        dataclass. ``None`` takes every tensor field it declares.

    Returns
    -------
    tuple
        ``(graph, external_feeds, inputs, outputs, retained_values)``,
        where ``inputs`` is a list of ``(feature name, value id, tensor)``
        and ``outputs`` a list of ``(feature name, value id, tensor)``. An
        input the forward wrote in place is bound to the value it arrived
        with, so the program computes the write from what it is given.
    """
    examples, by_keyword = _named_examples(example)
    held = [_unwrap(tensor) for _name, tensor in examples]
    tracer: Tracer | None = None
    try:
        with _observers_paused(model):
            before = _buffer_marks(model)
            with _compile._tracing(watch=[t for _n, t in examples]) as tracer:
                tracer.redraw_rng = False
                with lucid.no_grad():
                    result = (
                        model(**dict(examples))
                        if by_keyword
                        else model(*(tensor for _, tensor in examples))
                    )
                selected = _select_outputs(result, output_field)
            after = _buffer_marks(model)

        writes = list(tracer.outside_writes())
        written = {id(impl) for _tid, impl in writes}
        answered = {
            name: tensor.clone() for name, tensor in selected if id(_unwrap(tensor)) in written
        }
    finally:
        replaced = _restore_traced_examples(tracer, examples, held)

    moved = sorted(name for name, mark in after.items() if before.get(name, mark) != mark)
    if moved:
        raise StatefulModel(moved)
    if replaced:
        _refuse_replaced(*replaced[0])
    if tracer.graph.unsupported:
        raise ValueError(f"lucid.coreml: {tracer.graph.unsupported}. {_HOST_READS}")

    feeds = tracer.external_feeds
    inputs = _bind_traced_inputs(examples, held, written, feeds, tracer)
    outputs = _bind_traced_outputs(selected, answered, tracer)
    return tracer.graph, feeds, inputs, outputs, tracer.retained_values


def _refuse_replaced(name: str, grad: bool) -> NoReturn:
    """Refused rather than followed: the replacement is a Python-level
    rebinding the recording cannot see, so the program would read the
    example's values as a constant and the input would go unread."""
    why = (
        " — an example that requires grad is replaced rather than written "
        "into, so passing it detached (`x.detach()`) is enough"
        if grad
        else ""
    )
    raise ValueError(
        f"lucid.coreml: forward writes into input {name!r} by replacing its "
        "tensor rather than writing into it, so the program would read the "
        "example's values as a constant instead of the input. Clone it "
        "first (`x = x.clone()` at the top of forward) and write into the "
        f"copy{why}"
    )


_HOST_READS = (
    "A model that reads tensor values while it runs computes that part on the host, "
    "where an exported graph cannot follow: ``.item()``, ``.tolist()``, ``float(t)``, "
    "an ``if`` on a comparison, or an index computed from the data. Whatever reads "
    "values has to become tensor operations before this model can be exported."
)


def _flatten[T](tensor: Tensor, convert: Callable[[object], T]) -> list[T]:
    """Each element of ``tensor``, row-major, through ``convert`` (numpy-free, H4)."""
    flat: list[T] = []

    def walk(value: object) -> None:
        if isinstance(value, list):
            for item in value:
                walk(item)
        else:
            flat.append(convert(value))

    walk(tensor.tolist())
    return flat


# Below this many elements a weight is not worth quantizing: the
# per-channel scale is itself stored uncompressed, so a small tensor can
# come out larger, and the error it adds buys nothing. The same threshold
# the reference tooling uses, stated here rather than inherited.
_QUANTIZE_MIN_ELEMENTS = 2048


def _settle_target(
    wanted: _spec.DeploymentTarget | None,
    *,
    state: object,
    weights: object,
    functions: bool = False,
) -> _spec.DeploymentTarget:
    """The floor this package will actually have, refusing a lower ask.

    Three features move a program from ``CoreML7`` to ``CoreML8``, and
    until this existed they moved it without saying so: a caller who
    needed iOS 17 found out from a device. Asked for a floor the package
    cannot meet, the export stops here and names which feature raised
    it and how the call asked for it.

    Parameters
    ----------
    wanted : DeploymentTarget or None
        What the caller asked for. ``None`` accepts whatever the
        features require.
    state : list of State or None
        The state specification, if any.
    weights : WeightPrecision or Palettize or Sparsify
        How weights are stored; palettization needs the newer opset.
    functions : bool, optional, default=False
        Whether the package carries several entry points.

    Returns
    -------
    DeploymentTarget
        The floor the package has.

    Raises
    ------
    ValueError
        When a requested floor is lower than the features allow.
    """
    reasons: list[str] = []
    if state:
        reasons.append("state")
    if isinstance(weights, _spec.Palettize):
        reasons.append("palettization")
    if functions:
        reasons.append("multiple functions")

    if not reasons:
        return wanted or _spec.DeploymentTarget.IOS17
    if wanted is None or wanted is _spec.DeploymentTarget.IOS18:
        return _spec.DeploymentTarget.IOS18

    named = ", ".join(f"{r} ({_spec._NEEDS_IOS18[r]})" for r in reasons)
    raise ValueError(
        f"lucid.coreml: {wanted.name} was asked for, and this export uses "
        f"{named}, which Core ML expresses only from IOS18. The package "
        "would load on iOS 18 and nowhere earlier — drop the feature or "
        "raise the target, rather than finding out on a device"
    )


def staging_path(path: str) -> str:
    """Where a package is built before it is allowed to replace anything.

    Writing straight to the destination means clearing it first, and a
    build that then fails leaves the caller with neither the package
    they had nor the one they asked for. Re-exporting a deployed model
    and hitting an unsupported operation should not cost them the
    deployed model.

    A sibling of the destination, so that moving it into place is a
    rename within one filesystem rather than a copy.

    Parameters
    ----------
    path : str
        Destination the caller asked for.

    Returns
    -------
    str
        Path to build at.
    """
    return f"{path}.building"


def discard_staging(path: str) -> None:
    """Remove a half-built package after a failed export.

    Best effort on purpose: the export is already raising, and a failure
    to clean up must not replace the reason it failed with a filesystem
    complaint. A leftover directory is a smaller problem than losing the
    exception that explains what went wrong.

    Parameters
    ----------
    path : str
        Destination the caller asked for; the staging directory is
        derived from it.
    """
    import shutil

    shutil.rmtree(staging_path(path), ignore_errors=True)


def commit_staging(path: str) -> None:
    """Move a finished package onto the destination.

    The old package is moved aside first and removed after the new one
    is in place, so the window in which neither exists is a rename wide
    rather than a build wide.

    Parameters
    ----------
    path : str
        Destination the caller asked for.

    Raises
    ------
    OSError
        The move failed; the staging directory is left for inspection
        rather than deleted, since it holds the only copy of the work.
    """
    import os
    import shutil

    staging = staging_path(path)
    previous = f"{path}.replaced"
    shutil.rmtree(previous, ignore_errors=True)
    had_previous = os.path.exists(path)
    if had_previous:
        os.rename(path, previous)
    try:
        os.rename(staging, path)
    except OSError:
        if had_previous:
            os.rename(previous, path)
        raise
    shutil.rmtree(previous, ignore_errors=True)


def _floor_of(path: str) -> _spec.DeploymentTarget | None:
    """Oldest system a written package will run on, read back off it.

    A model that was just exported knows this because the export settled
    it; one reopened with ``load`` would otherwise not, and the two
    should not disagree about the same file. The opset is a plain string
    in the serialized program, so finding it needs no schema.

    Parameters
    ----------
    path : str
        The ``.mlpackage`` to read.

    Returns
    -------
    DeploymentTarget or None
        ``None`` when the file cannot be read, since a guess about a
        deployment floor is worse than an absence.
    """
    try:
        with open(f"{path}/Data/com.apple.CoreML/model.mlmodel", "rb") as handle:
            program = handle.read()
    except OSError:
        return None
    if _spec.DeploymentTarget.IOS18.opset.encode() in program:
        return _spec.DeploymentTarget.IOS18
    if _spec.DeploymentTarget.IOS17.opset.encode() in program:
        return _spec.DeploymentTarget.IOS17
    return None


def _refuse_if_empty(tensor: Tensor) -> None:
    """Stop on a constant with no elements, and say where it came from.

    An empty constant is not a weight a model happens to leave blank; it
    is a shape the trace could not know. A two-stage detector sizes its
    region features by how many proposals survived, and on an untrained
    model with random input that is often none — so the package would be
    built around this one run's answer and give it for every input.

    The blob writer refuses it too, with "the tensor has no host
    storage", which is true and says nothing about why.

    Parameters
    ----------
    tensor : Tensor
        Constant about to be written.

    Raises
    ------
    UnsupportedOp
        When the tensor has no elements.
    """
    if int(tensor.numel()) != 0:
        return
    raise UnsupportedOp(
        "a constant with no elements",
        f"the traced graph contains an empty tensor of shape "
        f"{tuple(int(d) for d in tensor.shape)}. That shape came from the "
        "example input rather than from the model — a region proposal count, "
        "a non-maximum suppression result — so the package would be built "
        "around this run's answer and give it for every input. Export the "
        "part of the model whose shapes are fixed, and keep the "
        "data-dependent stage in the caller",
    )


def _worth_storing_compressed(
    tensor: Tensor, weights: WeightPrecision | _spec.Palettize | _spec.Sparsify
) -> bool:
    """Whether a folded constant should go through the weight writer.

    The same rule the compressors apply themselves — rank at least two,
    enough elements to pay for the table — asked before the value is
    routed, so a small folded constant keeps travelling inline rather
    than acquiring a blob entry it does not need.
    """
    if weights is WeightPrecision.FLOAT:
        return False
    return len(tensor.shape) >= 2 and int(tensor.numel()) >= _QUANTIZE_MIN_ELEMENTS


#: A channel is tested for an existing grid on at most this many of its
#: elements. An ordinary float weight has as many distinct values as
#: elements, so a sample rejects it immediately; one that really sits on
#: an 8-bit grid cannot have more than 256 however much is looked at.
_GRID_SAMPLE = 1 << 12


def _grid_already_there(flat: Tensor) -> Tensor | None:
    """The step a weight is already quantized on, or ``None``.

    A quantization-aware model's weights have been trained onto a grid,
    and deriving a fresh one throws that away: this writer's scale is
    ``max / 127.5``, so re-quantizing a weight whose extreme is
    ``127 * step`` lands it on a grid 0.4% finer that its values do not
    sit on. Measured on a QAT classifier, that cost 1.4e-03 against a
    model the export could have reproduced exactly — the whole claim of
    training with quantization, given away at the last step.

    Rejected cheaply: a sample of each channel is sorted and its distinct
    values counted, and an ordinary weight has as many as it has
    elements. Only a weight that survives that is examined in full.

    Parameters
    ----------
    flat : Tensor
        ``(channels, elements)`` — the weight, one row per output
        channel.

    Returns
    -------
    Tensor or None
        ``(channels, 1)`` steps, or ``None`` when any channel holds more
        distinct values than eight bits can index, or when the values are
        not multiples of a common step.
    """
    width = int(flat.shape[1])
    sample = flat[:, :: (width // _GRID_SAMPLE + 1)] if width > _GRID_SAMPLE else flat
    ordered = lucid.sort(sample, dim=-1).values
    fresh = (ordered[:, 1:] != ordered[:, :-1]).to(lucid.float32)
    if float(fresh.sum(dim=1).max().item()) + 1 > 256:
        return None

    ordered = lucid.sort(flat, dim=-1).values
    gaps = ordered[:, 1:] - ordered[:, :-1]
    # The step is the smallest gap that is not two identical values. A
    # grid with a code missing in the middle would give a multiple of the
    # step instead, which the integrality check below then rejects.
    positive = lucid.where(gaps > 0, gaps, lucid.full_like(gaps, float("inf")))
    step = positive.min(dim=1).values.reshape(-1, 1)
    if not bool((step < float("inf")).all().item()):
        return None

    codes = (flat / step).round()
    # Measured against the step rather than against the weight's range:
    # the fake-quantize that produced these computed ``round(w / s) * s``
    # in float32, so a value sits within an ulp of its lattice point, and
    # an ulp of the largest weight is far larger than an ulp of the
    # smallest. A thousandth of a step is decisively on the lattice.
    off = (flat - codes * step).abs() / step
    # A hundredth of a step. The discriminating work is done by the
    # distinct-value count above — an ordinary weight has as many values
    # as elements and never gets here — so this only has to tolerate the
    # float32 round trip that produced the lattice, which on a trained
    # weight runs to about a thousandth of a step and on an untrained one
    # to a tenth of that.
    if float(off.max().item()) > 1e-2:
        return None
    # int8 runs from -128 to 127, and a symmetric fake-quantize reaches
    # the negative end — rejecting on magnitude alone would throw away
    # the grid for a code that is perfectly representable.
    if float(codes.max().item()) > 127.0 or float(codes.min().item()) < -128.0:
        return None
    return step


def _quantize_weight(tensor: Tensor, scale_mil: int) -> tuple[Tensor, bytes, bytes, int] | None:
    """A weight as int8 codes with one scale per output channel.

    Returns ``None`` when the tensor should stay in floating point — a
    bias, a norm's parameters, anything below the threshold, or a rank-1
    value that has no channel axis to scale along.

    Symmetric rather than affine: a symmetric grid puts zero exactly on a
    lattice point, which matters because padding and masking write real
    zeros that an affine grid would round to something else.

    Parameters
    ----------
    tensor : Tensor
        Float weight to quantize.
    scale_mil : int
        MIL dtype the scale is stored as — the body's precision, so the
        dequantized value lands in the type the operation wants.

    Returns
    -------
    tuple or None
        ``(codes, scale bytes, zero-point bytes, channels)``.
    """
    if len(tensor.shape) < 2:
        return None
    count = 1
    for dim in tensor.shape:
        count *= int(dim)
    if count < _QUANTIZE_MIN_ELEMENTS:
        return None

    channels = int(tensor.shape[0])
    flat = tensor.reshape(channels, -1)
    existing = _grid_already_there(flat)
    if existing is not None:
        # Already on a grid, so it is written on that one. Deriving a new
        # one would round values that were trained to sit exactly where
        # they are.
        scale = existing.reshape(-1)
        zero_point = lucid.zeros_like(scale)
        codes = (flat / existing).round().reshape(*[int(d) for d in tensor.shape]).to(lucid.int8)
    else:
        scale, zero_point = _quant.calculate_qparams(
            flat.min(dim=1)[0], flat.max(dim=1)[0], _quant.per_channel_symmetric, _quant.qint8
        )
        codes = _quant.quantize(tensor, scale, zero_point, _quant.qint8, ch_axis=0)

    # MIL has no float16 or int8 immediate list; both travel as raw
    # little-endian bytes, so they are packed here rather than handed to
    # the engine as numbers it would have to re-encode.
    fmt = "<e" if scale_mil == _spec.FLOAT16 else "<f"
    scale_bytes = b"".join(struct.pack(fmt, v) for v in _flatten(scale, _as_float))
    zero_bytes = b"".join(struct.pack("<b", int(v)) for v in _flatten(zero_point, _as_float))
    return codes, scale_bytes, zero_bytes, channels


#: A palette is fitted from at most this many weights.  The table has at
#: most 256 entries, so past a certain sample the centroids stop moving
#: and every extra element is time spent for nothing — an AlexNet
#: classifier weight has 37 million of them.
_PALETTE_SAMPLE = 1 << 16


def _assign(rows: Tensor, table: Tensor, count: int) -> Tensor:
    """Which palette entry each value falls in, a row at a time.

    ``lucid.bucketize`` answers this for one shared edge list, but it
    walks the whole list per element — a 256-entry table costs 255 passes
    over the weight — and it cannot give each row its own edges, which is
    the whole point of a per-channel palette. The edges are sorted and
    ``count`` is a power of two, so bisection answers in ``log2(count)``
    passes and the running index needs no bounds check: the steps sum to
    ``count - 1`` exactly.

    ``table`` is ``(groups, count)``: column zero is a placeholder that
    nothing probes (the first candidate is already ``count // 2``), and
    the rest are the midpoints between neighbouring palette entries. The
    convention matches ``bucketize`` — the entry is the number of edges
    strictly below the value — which is what the unit test compares
    against.
    """
    found = lucid.zeros_like(rows).to(lucid.int64)
    step = count // 2
    while step >= 1:
        candidate = found + step
        picked = lucid.gather(table, 1, candidate)
        found = lucid.where(picked < rows, candidate, found)
        step //= 2
    return found


def _edge_table(centres: Tensor) -> Tensor:
    """Midpoints between neighbouring palette entries, per row.

    Column zero is the placeholder ``_assign`` never reads, kept so the
    table indexes the same way the palette does.
    """
    groups = int(centres.shape[0])
    lower = centres[:, :-1]
    upper = centres[:, 1:]
    head = lucid.full((groups, 1), -math.inf)
    return lucid.concat([head, (lower + upper) / 2], dim=1)


#: A palette is fitted from at most this many values per group.  The
#: table has at most 256 entries, so past a certain sample the centroids
#: stop moving and every extra element is time spent for nothing.
_PALETTE_SAMPLE = 1 << 14

#: Lloyd passes.  Well past where the movement stops mattering at these
#: sizes, and a fixed count keeps export time predictable.
_PALETTE_PASSES = 10


def _values_already_there(ordered: Tensor, count: int) -> Tensor | None:
    """The exact palette of a weight that already sits on few values.

    ``None`` when any row holds more distinct values than the table has
    entries, which is every ordinary weight.

    Lloyd's algorithm does not reliably find this palette even though it
    is the optimum with zero error. Two initial centres can land inside
    one repeated value, leaving another value unclaimed — and an entry no
    value landed on keeps the centre it had and never moves again, so the
    iteration cannot recover. Measured on a fine-tuned model whose rows
    held exactly sixteen values, the refit came back 9.3e-03 away from
    the palette the model had been trained against.

    That case is not exotic. It is what a compression-aware fine-tune
    produces by construction, and the whole point of that fine-tune is
    that the export writes the palette the model learned rather than a
    nearby one.

    Parameters
    ----------
    ordered : Tensor
        ``(groups, span)``, each row sorted ascending.
    count : int
        Entries the table has.

    Returns
    -------
    Tensor or None
        ``(groups, count)`` palettes, ascending, with any unused tail
        entry filled by the row's largest value so the table stays
        monotone for the bisection.
    """
    groups = int(ordered.shape[0])
    fresh = lucid.concat(
        [lucid.ones(groups, 1), (ordered[:, 1:] != ordered[:, :-1]).to(lucid.float32)], dim=1
    )
    slot = (lucid.cumsum(fresh, dim=1) - 1.0).to(lucid.int64)
    if int(slot.max().item()) >= count:
        return None

    empty = lucid.zeros(groups, count)
    totals = lucid.scatter_add(empty, 1, slot, ordered)
    hits = lucid.scatter_add(empty, 1, slot, lucid.ones_like(ordered))
    safe = lucid.where(hits > 0, hits, lucid.ones_like(hits))
    return lucid.where(hits > 0, totals / safe, ordered[:, -1:])


def _palettes_for(rows: Tensor, count: int) -> Tensor:
    """One palette per row, by Lloyd's algorithm in one dimension.

    Every row is fitted at once rather than in a Python loop over
    channels: a bisection for the assignment and two ``scatter_add``
    passes for the means, all batched. A per-channel fit over a 2048
    channel weight is 2048 independent clusterings, and doing them one at
    a time is the difference between an export that takes seconds and one
    that takes an hour.

    Started from evenly spaced order statistics rather than from the
    range, so a row whose mass sits near zero does not spend most of its
    table on tails it barely uses.
    """
    groups = int(rows.shape[0])
    width = int(rows.shape[1])
    sample = rows[:, :: (width // _PALETTE_SAMPLE + 1)] if width > _PALETTE_SAMPLE else rows
    span = int(sample.shape[1])

    ordered = lucid.sort(sample, dim=-1).values
    settled = _values_already_there(ordered, count)
    if settled is not None:
        # Exact, and the iteration below would not find it — see the
        # helper. Every ordinary weight falls straight through this.
        return settled
    marks = [float(min(span - 1, int((index + 0.5) * span / count))) for index in range(count)]
    centres = lucid.gather(
        ordered, 1, (lucid.zeros(groups, 1) + lucid.tensor(marks)).to(lucid.int64)
    )

    ones = lucid.ones_like(sample)
    empty = lucid.zeros(groups, count)
    for _ in range(_PALETTE_PASSES):
        assigned = _assign(sample, _edge_table(centres), count)
        totals = lucid.scatter_add(empty, 1, assigned, sample)
        weights = lucid.scatter_add(empty, 1, assigned, ones)
        # An entry no value landed on keeps the centre it had; dividing by
        # its zero population would put a NaN in the table and take the
        # whole row's bisection with it.
        moved = lucid.where(
            weights > 0, totals / lucid.where(weights > 0, weights, ones[:, :1]), centres
        )
        # Re-sorted because the bisection needs a monotone table, and a
        # row whose entries crossed would otherwise answer nonsense.
        centres = lucid.sort(moved, dim=-1).values
    return centres


def _palette_groups(channels: int, count: int, elements: int) -> int:
    """How many palettes to fit along the output-channel axis.

    One per channel is what makes palettization usable — a convolution's
    channels differ in magnitude by orders of magnitude, and a single
    shared table spends nearly all of its resolution on the loudest ones.
    But the tables are stored, so at 256 entries and 2048 channels they
    would cost more than the keys they index. Groups are halved until the
    table is a small fraction of the payload it serves.
    """
    key_bytes = elements * (count.bit_length() - 1) // 8
    groups = channels
    while groups > 1 and groups * count * 2 > key_bytes // 4:
        groups //= 2
    # The group count has to divide the channel count for the operation
    # to lay the tables out over the axis.
    while groups > 1 and channels % groups:
        groups -= 1
    return max(groups, 1)


def _palettize_weight(tensor: Tensor, bits: int) -> tuple[bytes, Tensor, list[int], int] | None:
    """A weight as packed keys into one palette per group of channels.

    ``None`` when the weight is too small to be worth the tables, or has
    no channel axis — the same two refusals the int8 path makes, for the
    same reasons: the tables cost something fixed, and a bias or a norm's
    parameters have no output channel to group along and are ruinous to
    approximate.

    Returns the packed keys, the tables, the shape the operation wants
    them in, and the width the keys were packed at — the last so the
    caller picks its dtype tags from the same number that did the
    packing rather than from the request.
    """
    if len(tensor.shape) < 2 or int(tensor.numel()) < _QUANTIZE_MIN_ELEMENTS:
        return None
    count = 1 << bits
    channels = int(tensor.shape[0])
    elements = int(tensor.numel())
    groups = _palette_groups(channels, count, elements)
    # Fitted in single precision even when the body is half. The weight
    # has already been rounded to float16 by then, so this loses nothing
    # — and Lloyd's algorithm on half-precision centroids both converges
    # worse and mixes dtypes against the edge table's sentinel, which is
    # how a float16 palettized export used to fail outright.
    rows = tensor.to(lucid.float32).reshape(groups, -1)

    palettes = _palettes_for(rows, count)
    keys = _assign(rows, _edge_table(palettes), count)
    # ``[groups, 1, ..., 1, count, 1]``: one table per group along the
    # output-channel axis, then the entries, then the vector width — one,
    # because these are scalar palettes.
    lut_shape = [groups] + [1] * (len(tensor.shape) - 1) + [count, 1]
    return (
        _C_engine.coreml.pack_bits(_unwrap(keys.reshape(-1)), bits),
        palettes.reshape(-1),
        lut_shape,
        bits,
    )


def _sparsify_weight(tensor: Tensor, ratio: float) -> tuple[Tensor, bytes, int] | None:
    """A weight as its surviving values plus one bit per element.

    ``None`` when nothing would be dropped, or when the weight is small
    enough that the mask costs more than the zeros save.
    """
    if len(tensor.shape) < 2 or ratio <= 0.0:
        return None
    if int(tensor.numel()) < _QUANTIZE_MIN_ELEMENTS:
        return None
    flat = tensor.reshape(-1)
    threshold = float(lucid.quantile(flat.abs(), ratio).item())
    keep = flat.abs() > threshold
    kept = int(keep.to(lucid.int64).sum().item())
    if kept == int(flat.numel()) or kept == 0:
        # Nothing to drop, or nothing to keep. The second happens for
        # real: a constant whose elements are all the same value has no
        # element above its own median, and a folded batch norm leaves
        # whole zero vectors behind. Such a constant would be a sparse
        # const with an empty payload, which is not something to write —
        # it goes out dense, at a cost of one small tensor.
        return None

    # Both of these are tensor work rather than Python loops: a weight
    # carries tens of millions of elements and iterating them here cost
    # minutes per layer.
    values = lucid.masked_select(flat, keep)
    return values, _C_engine.coreml.pack_bits(_unwrap(keep), 1), kept


def _operands(bindings: Bindings) -> list[tuple[str, list[str]]]:
    """Parameter bindings in the shape the engine takes.

    An emitter may bind a parameter to one operand or to several —
    ``concat`` is variadic — and the engine wants a list either way.

    Parameters
    ----------
    bindings : Bindings
        Pairs of parameter name and operand name, or names.

    Returns
    -------
    list[tuple[str, list[str]]]
        The same pairs, with every operand side a list.
    """
    return [
        (param, [value] if isinstance(value, str) else list(value)) for param, value in bindings
    ]


class Builder:
    """Mints the constants MIL requires and appends operations.

    Every scalar operand in a MIL program is its own ``const`` operation —
    a convolution's ``groups`` is a value, not an attribute — so emitters
    ask the builder for names rather than writing literals.
    """

    def __init__(
        self,
        program: MilProgram,
        blob: BlobWriter,
        body_mil: int = 0,
        body_blob: int = 0,
        half: bool = False,
    ) -> None:
        self._program = program
        self._blob = blob
        self._body_mil = body_mil
        self._body_blob = body_blob
        self._half = half
        self._counter = 0
        # Value name -> declared shape, ``-1`` on axes a flexible export
        # leaves open. Private: emitters read it through ``shape_of``,
        # whose open axes cannot be read as numbers.
        self._shapes: dict[str, list[int]] = {}
        # Traced id -> value name, the driver's own map, so an operand's
        # name leads back to the probes that say which of its axes vary.
        self.traced_names: dict[int, str] = {}
        # Values that are constants.  Core ML rejects a computed operand
        # where it wants a constant, so emitters check before binding.
        self.consts: set[str] = set()
        # Values emitted one rank lower than the trace records, because
        # their leading axis is one and Core ML stops at rank five. Keyed
        # by MIL value name so an emitter can ask about its operand.
        self.unit_axis_dropped: set[str] = set()
        # The same set keyed by traced value id, for an emitter asking
        # about its own result rather than about an operand.
        self.thinned: set[int] = set()
        # Value name -> MIL element type. Core ML does not promote: an
        # elementwise operation whose two operands differ in dtype is
        # refused when the package is parsed, naming an internal value
        # and nothing else. Recorded so an emitter can insert the cast.
        self.dtypes: dict[str, int] = {}
        # Which axes of which traced values a flexible export leaves open,
        # so an emitter that writes a shape into a constant can leave them
        # open too rather than baking the default.
        self.flex = _flex.Flexibility()
        # The operation being emitted, named by any refusal a shape read
        # raises; and the flexible input, which run-time sizes are read off.
        self.current = "<driver>"
        self.flex_input: str | None = None
        self._run_time: dict[str, str] = {}

    def declare(self, name: str, shape: Sequence[int]) -> None:
        """Record a value's declared shape, ``-1`` on its open axes."""
        self._shapes[name] = shape.declared() if isinstance(shape, _flex.Dims) else list(shape)

    def result_dims(self, op: TracedOp, index: int = 0) -> _flex.Dims:
        """One of an operation's result shapes, open axes guarded.

        The form to declare an intermediate with, and the only way an
        emitter reads a result's size: an axis a flexible export leaves
        open raises :class:`ShapeNotFlexible` naming ``op`` when read as a
        number — the traced size is the only number there is, and writing
        it down is the failure. ``.declared()`` spells it with ``-1``.

        Parameters
        ----------
        op : TracedOp
            Operation whose result to describe.
        index : int, optional, default=0
            Which result.
        """
        out = op.outputs[index]
        recorded = out.shape
        shape = (
            recorded.declared() if isinstance(recorded, _flex.Dims) else [int(d) for d in recorded]
        )
        # A value the driver emits a rank lower loses its leading axis
        # here too, or an emitter that writes the shape into a constant
        # would ask for the rank Core ML refused.
        if int(out.id) in self.thinned:
            shape = shape[1:]
        flexed = _flex.flex(shape, self.flex.varying.get(int(out.id)))
        return _flex.Dims(flexed, op.name)

    def shape_operand(
        self,
        shape: Sequence[int],
        value_id: int = -1,
        operand: str | None = None,
        known: dict[int, tuple[str, int]] | None = None,
    ) -> str:
        """A ``reshape`` target for ``shape``, open axes and all.

        One open axis is a ``-1`` in a constant. More are read at run
        time as :meth:`sizes_operand` reads them, and one of them may still
        be left for ``reshape`` to infer.

        Parameters
        ----------
        shape : sequence of int
            Declared shape, ``-1`` on open axes.
        value_id : int, optional, default=-1
            Traced value the shape is, ``-1`` for none.
        operand : str or None, optional, default=None
            The value being reshaped, whose own axes are read where the
            shape keeps them.
        known : dict or None, optional, default=None
            Axis to ``(value, axis)`` it is known to equal.
        """
        dims = shape.declared() if isinstance(shape, _flex.Dims) else list(shape)
        if sum(d < 0 for d in dims) <= 1:
            return self.const_ints(dims)
        return self._sizes(dims, value_id, operand, known, infer=True)

    def sizes_operand(self, sizes: Sequence[int], value_id: int, operand: str | None = None) -> str:
        """``sizes`` as an int32 vector, each open one read at run time.

        An open axis the operand carries — ``value_id`` keeps it from
        ``operand`` — is the operand's own size, exactly. Any other is the
        input axis it equalled at every probe, and refused when there is
        none, since then there is nothing to read.

        Parameters
        ----------
        sizes : sequence of int
            One size per axis, ``-1`` where it follows the input.
        value_id : int
            Traced value the sizes describe.
        operand : str or None, optional, default=None
            A value whose axes may be the open ones.
        """
        dims = sizes.declared() if isinstance(sizes, _flex.Dims) else list(sizes)
        return self._sizes(dims, value_id, operand, None, infer=False)

    def _carried_where(
        self,
        dims: list[int],
        value_id: int,
        operand: str | None,
        known: dict[int, tuple[str, int]] | None,
    ) -> dict[int, tuple[str, int]]:
        where = dict(known or {})
        traced = None if operand is None else self.traced_id(operand)
        if operand is not None and traced is not None:
            mine = (value_id, dims)
            theirs = (traced, self._shapes[operand])
            for axis, other in self.flex.carried(mine, theirs).items():
                where.setdefault(axis, (operand, other))
        return where

    def _sizes(
        self,
        dims: list[int],
        value_id: int,
        operand: str | None,
        known: dict[int, tuple[str, int]] | None,
        infer: bool,
    ) -> str:
        where = self._carried_where(dims, value_id, operand, known)
        pieces: list[str] = []
        run: list[int] = []
        for axis, size in enumerate(dims):
            if size < 0 and axis not in where:
                source = self.flex.source(value_id, axis)
                if source is not None and self.flex_input is not None:
                    where[axis] = (self.flex_input, source)
                elif infer:
                    infer = False
                else:
                    raise ShapeNotFlexible(
                        self.current,
                        f"axis {axis} of its result follows the input, and "
                        "neither an operand nor a single input axis has its "
                        "size at every traced shape, so there is nothing to "
                        "read it from at run time",
                    )
            if axis not in where:
                run.append(size)
                continue
            if run:
                pieces.append(self.const_ints(run))
                run = []
            pieces.append(self.run_time_dim(*where[axis]))
        if run:
            pieces.append(self.const_ints(run))
        if len(pieces) == 1:
            return pieces[0]
        return self.emit(
            "concat",
            [
                ("values", pieces),
                ("axis", self.const_int(0)),
                ("interleave", self.const_bool(False)),
            ],
            [len(dims)],
            dtype=_C_engine.coreml.DTYPE_INT32,
        )

    def traced_id(self, name: str) -> int | None:
        """The traced value a program value stands for, the first if several."""
        return next((tid for tid, held in self.traced_names.items() if held == name), None)

    def run_time_dim(self, name: str, axis: int) -> str:
        """``name``'s size along ``axis``, read at run time, as ``[n]``."""
        key = f"{name}:{axis}"
        if key in self._run_time:
            return self._run_time[key]
        int32 = _C_engine.coreml.DTYPE_INT32
        if name not in self._run_time:
            rank = len(self._shapes[name])
            self._run_time[name] = self.emit("shape", [("x", name)], [rank], dtype=int32)
        self._run_time[key] = self.emit(
            "gather",
            [
                ("x", self._run_time[name]),
                ("indices", self.const_ints([axis])),
                ("axis", self.const_int(0)),
                ("validate_indices", self.const_bool(False)),
            ],
            [1],
            dtype=int32,
        )
        return self._run_time[key]

    def result_mil_dtype(self, op: TracedOp, index: int = 0) -> int:
        """MIL element type the driver will declare for a result.

        An emitter that has to build a constant of the *result's* type —
        ``one_hot``'s two fill values must match it, and Core ML says so
        at parse time — needs the same answer the driver reaches, not a
        guess from the operation's name.

        Parameters
        ----------
        op : TracedOp
            Operation whose result to describe.
        index : int, optional, default=0
            Which result.

        Returns
        -------
        int
            MIL element-type number.
        """
        out = op.outputs[index]
        return _spec.trace_dtype(str(out.dtype).split(".")[-1], self._body_mil)

    def leading_axis_dropped(self, name: str) -> bool:
        """Whether this value is a rank lower than the trace recorded.

        An operation that names an axis has to move it when its operand
        lost the leading one. Everything else — arithmetic, and the shape
        operations that reshape to their recorded output — adapts by
        itself.

        Parameters
        ----------
        name : str
            Value to ask about.

        Returns
        -------
        bool
            True when the leading axis was dropped.
        """
        return name in self.unit_axis_dropped

    def dtype_of(self, name: str) -> int | None:
        """MIL element type of a value, when the builder recorded one.

        ``None`` for a value nothing declared — an emitter should leave
        such an operand alone rather than guess at a cast.

        Parameters
        ----------
        name : str
            Value to look up.

        Returns
        -------
        int or None
            MIL element-type number, or ``None`` if unrecorded.
        """
        return self.dtypes.get(name)

    def agree_on_dtype(self, names: list[str], target: int | None = None) -> list[str]:
        """The same values, cast so an elementwise operation accepts them.

        Core ML does not promote. ``mul`` of an int32 by a float32
        constant is refused when the package is parsed — with an
        internal value name and no indication of which operand is
        wrong — where Lucid promoted and moved on. CLIP builds its
        position indices that way.

        Pass the operation's declared result type as ``target``: an
        arithmetic operation must have operands of the type its output
        was declared as, and guessing a promotion instead is how an
        ``int64 * 3`` ends up with float operands feeding an int32
        result. Leave ``target`` out for a comparison, whose result is
        boolean and whose operands only have to match each other.

        Values whose type was never recorded are left alone: an unforced
        cast can be as wrong as a missing one.

        Parameters
        ----------
        names : list of str
            Operands of one elementwise operation.
        target : int or None, optional, default=None
            MIL element type to cast to. ``None`` promotes among the
            operands, with the float side winning.

        Returns
        -------
        list of str
            The operands, some possibly replaced by a cast of themselves.
        """
        kinds = {kind for kind in (self.dtype_of(name) for name in names) if kind is not None}
        if len(kinds) < 2 and (target is None or kinds <= {target}):
            return names
        if target is None:
            floats = {_C_engine.coreml.DTYPE_FLOAT16, _C_engine.coreml.DTYPE_FLOAT32}
            settled = self._body_mil if kinds & floats else max(kinds)
        else:
            settled = target
        spelling = MIL_CAST_NAMES.get(settled)
        if spelling is None:
            return names
        agreed = []
        for name in names:
            kind = self.dtype_of(name)
            if kind is None or kind == settled:
                agreed.append(name)
                continue
            agreed.append(
                self.emit(
                    *emit_cast(self, name, spelling),
                    # A scalar constant has no recorded shape, and none
                    # is the right answer for it.
                    self._shapes.get(name, []),
                    dtype=settled,
                )
            )
        return agreed

    def result_cast_spelling(self, op: TracedOp, index: int = 0) -> str | None:
        """MIL's name for the type a result is *declared* as.

        Not the type the trace asked for. A float16 program declares
        every float intermediate as float16, so a model that casts to
        float32 in the middle — VQ-VAE does, around its codebook — has a
        ``cast`` whose requested type and whose declared output type
        disagree, and Core ML refuses the package with "Specified dtype
        of cast does not match that of output tensor".

        Parameters
        ----------
        op : TracedOp
            Operation whose result to name.
        index : int, optional, default=0
            Which result.

        Returns
        -------
        str or None
            MIL's spelling, or ``None`` for a type ``cast`` cannot name.
        """
        return MIL_CAST_NAMES.get(self.result_mil_dtype(op, index))

    def narrow_to_int32(self, name: str) -> str:
        """The same value as int32, for an operand that takes nothing wider.

        This opset's ``gather`` accepts int32 indices and smaller, and
        refuses int64 when the package is parsed rather than converting
        it. Casting int32 to int32 is a no-op the compiler drops, so the
        emitter does not have to know which case it is in.
        """
        return self.emit(
            *emit_cast(self, name, "int32"), self.shape_of(name), dtype=_C_engine.coreml.DTYPE_INT32
        )

    def result_is_float(self, op: TracedOp, index: int = 0) -> bool:
        """Whether a result is declared as a floating type."""
        return self.result_mil_dtype(op, index) in (
            _C_engine.coreml.DTYPE_FLOAT16,
            _C_engine.coreml.DTYPE_FLOAT32,
        )

    def shape_of(self, name: str) -> _flex.Dims:
        """An operand's shape; reading an open axis refuses the operation."""
        shape = self._shapes.get(name)
        if shape is None:
            raise KeyError(
                f"lucid.coreml: no shape recorded for value {name!r} — an emitter "
                "asked about an operand the driver never registered"
            )
        return _flex.Dims(shape, self.current)

    def _next(self, kind: str) -> str:
        self._counter += 1
        return f"_c{self._counter}_{kind}"

    def _const_name(self, kind: str) -> str:
        """Allocate a name and remember that it names a constant.

        Core ML refuses some operands that are not constants —
        ``linear``'s weight and bias, a convolution's weight — so an
        emitter has to be able to ask. Recorded rather than inferred from
        the name: ``_next`` also names the results of operations, and a
        prefix check would call those constants too.
        """
        name = self._next(kind)
        self.consts.add(name)
        return name

    def mark_const(self, name: str) -> None:
        """Record a constant the driver named itself, such as a weight."""
        self.consts.add(name)

    def is_const(self, name: str) -> bool:
        """Whether this value is a constant Core ML will accept as one."""
        return name in self.consts

    def const_ints(self, values: list[int]) -> str:
        name = self._const_name("ints")
        self.dtypes[name] = _C_engine.coreml.DTYPE_INT32
        self._program.add_int_const(name, _spec.as_int32(values, name), False)
        return name

    def const_ints_shaped(self, values: list[int], shape: list[int]) -> str:
        """An integer constant of arbitrary shape, carried inline.

        ``const_from_tensor`` is the blob path and the blob holds float
        payloads only, so an integer tensor — a gather's indices, say —
        cannot go through it without changing dtype.

        Parameters
        ----------
        values : list[int]
            Flattened values, in row-major order.
        shape : list[int]
            Shape to declare.

        Returns
        -------
        str
            Name of the constant.
        """
        name = self._const_name("intsn")
        self._program.add_int_const_shaped(name, _spec.as_int32(values, name), shape)
        self._shapes[name] = list(shape)
        self.dtypes[name] = _C_engine.coreml.DTYPE_INT32
        return name

    def const_int(self, value: int) -> str:
        name = self._const_name("int")
        self.dtypes[name] = _C_engine.coreml.DTYPE_INT32
        self._program.add_int_const(name, _spec.as_int32([value], name), True)
        return name

    def const_float(self, value: float) -> str:
        """A float scalar, at the body's precision.

        In an fp16 program this has to be fp16 too — MIL rejects a
        ``batch_norm`` whose ``epsilon`` is float32 while its statistics
        are float16, and that rejection happens at load time, not at
        export. Core ML writes fp16 constants into the weight blob rather
        than inline (a reference fp16 package contains no fp16 immediate
        values at all), so this follows it there.
        """
        name = self._const_name("float")
        self.dtypes[name] = _C_engine.coreml.DTYPE_FLOAT32
        if not self._half:
            self._program.add_float_const(name, [float(value)], True)
            return name
        payload = lucid.tensor([float(value)]).half()
        offset = self._blob.append_tensor(_unwrap(payload), self._body_blob)
        # Declared scalar (rank 0) even though the payload is one element:
        # MIL types ``epsilon`` as a scalar and rejects tensor<fp16, [1]>.
        self._program.add_blob_const(name, (self._body_mil, []), offset)
        return name

    def const_float32_shaped(self, values: list[float], shape: list[int]) -> str:
        """A float32 constant of arbitrary shape, carried inline.

        Image preprocessing needs a per-channel bias shaped
        ``(1, C, 1, 1)``, which neither the scalar nor the rank-1 helper
        can express, and which must stay float32 because it is applied
        before the body's cast.

        Parameters
        ----------
        values : list[float]
            Flattened values, row-major.
        shape : list[int]
            Shape to declare.

        Returns
        -------
        str
            Name of the constant.
        """
        name = self._const_name("f32n")
        self._program.add_float_const_shaped(name, [float(v) for v in values], shape)
        self._shapes[name] = list(shape)
        self.dtypes[name] = _C_engine.coreml.DTYPE_FLOAT32
        return name

    def const_float32(self, value: float) -> str:
        """A float scalar that stays float32 whatever the body's precision.

        Not every float parameter is data. ``epsilon`` is added to the
        activations and must share their dtype — MIL rejects a float32
        epsilon beside float16 statistics. A resampling ``scale_factor``
        is the opposite: it configures the operator, and MIL accepts only
        int32 or float32 there, rejecting the float16 that following the
        body would produce.

        Parameters
        ----------
        value : float
            Scalar value.

        Returns
        -------
        str
            Name of the constant.
        """
        name = self._const_name("f32")
        self.dtypes[name], self._shapes[name] = _spec.FLOAT32, []
        self._program.add_float_const(name, [float(value)], True)
        return name

    def const_str(self, value: str) -> str:
        name = self._const_name("str")
        self._program.add_string_const(name, value)
        return name

    def is_0d_float32(self, name: str) -> bool:
        """Whether ``name`` is a 0-d float32 scalar or tensor."""
        return self.dtypes.get(name) == _spec.FLOAT32 and not self._shapes.get(name)

    def const_from_tensor(self, tensor: Tensor) -> str:
        """A constant of arbitrary shape, carried in the weight blob.

        The inline constant helpers make rank-1 values, which is enough for
        the small integer lists MIL wants for axes and strides. An op that
        *produces* a tensor out of nothing — ``zeros``, ``full``,
        ``arange`` — needs its real shape, and the blob path already
        handles shape and dtype for weights.

        Parameters
        ----------
        tensor : Tensor
            Host tensor holding the constant's value.

        Returns
        -------
        str
            Name of the constant.
        """
        _refuse_if_empty(tensor)
        payload = tensor.half() if self._half else tensor
        is_f16 = payload.dtype == lucid.float16
        mil_dt = _spec.FLOAT16 if is_f16 else self._body_mil
        blob_dt = _spec.BLOB_FLOAT16 if is_f16 else self._body_blob
        name = self._const_name("blobconst")
        offset = self._blob.append_tensor(_unwrap(payload), blob_dt)
        shape = [int(d) for d in tensor.shape]
        self._program.add_blob_const(name, (mil_dt, shape), offset)
        self._shapes[name] = shape
        self.dtypes[name] = mil_dt
        return name

    def emit(
        self, mil_type: str, bindings: Bindings, shape: Sequence[int], dtype: int | None = None
    ) -> str:
        """Append an intermediate operation and return its value name.

        Some Lucid ops are several MIL ops — scaled dot-product attention
        is a matmul, a scale, a softmax and another matmul — so an emitter
        needs to place the ones before the last itself. The driver names
        and types the final one from the trace.

        Parameters
        ----------
        mil_type : str
            MIL operator name.
        bindings : list
            ``(parameter, value name)`` pairs, as an emitter returns.
        shape : sequence of int
            Result shape; a :class:`~lucid.coreml._flex.Dims` declares its
            open axes as such.
        dtype : int or None, optional, default=None
            MIL element type of the result. ``None`` is the type of the
            ``x`` operand when that is an integer or boolean, and the
            program body's otherwise; a comparison has to say ``BOOL``
            or Core ML rejects the operation for producing the wrong type.

        Returns
        -------
        str
            Name of the value this operation produces.
        """
        name = self._next(mil_type)
        normalised = _operands(bindings)
        declared = self._result_dtype(normalised) if dtype is None else int(dtype)
        dims = shape.declared() if isinstance(shape, _flex.Dims) else list(shape)
        self._program.add_op(mil_type, normalised, name, (declared, dims))
        self._shapes[name] = dims
        self.dtypes[name] = declared
        return name

    def _result_dtype(self, operands: list[tuple[str, list[str]]]) -> int:
        """An unnamed result type: MIL does not promote, so an int32, bool, or
        explicit float16 ``x`` keeps its type; other floats follow the body."""
        for param, values in operands:
            if param == "x" and len(values) == 1:
                kind = self.dtypes.get(values[0])
                if kind in (_spec.INT32, _spec.BOOL, _spec.FLOAT16):
                    return kind
        return self._body_mil

    def emit_multi(self, mil_type: str, bindings: Bindings, shapes: list[list[int]]) -> list[str]:
        """Append an intermediate operation with several results.

        ``split`` is the one that needs it — decomposing a ``roll`` into a
        split and a swapped concat keeps the translation on operations
        that are already verified, instead of reaching for a slice whose
        masks would be a second thing to get right.

        Parameters
        ----------
        mil_type : str
            MIL operator name.
        bindings : list
            ``(parameter, value name)`` pairs.
        shapes : list[list[int]]
            One shape per result.

        Returns
        -------
        list[str]
            Names of the values produced, in order.
        """
        normalised = _operands(bindings)
        declared = self._result_dtype(normalised)
        names = [self._next(f"{mil_type}{i}") for i in range(len(shapes))]
        self._program.add_op_multi(
            mil_type,
            normalised,
            [(n, (declared, list(sh))) for n, sh in zip(names, shapes, strict=True)],
        )
        for n, sh in zip(names, shapes, strict=True):
            self._shapes[n] = list(sh)
            self.dtypes[n] = declared
        return names

    def const_bool(self, value: bool) -> str:
        name = self._const_name("bool")
        self._program.add_bool_const(name, bool(value))
        return name


#: Operations that a leading axis of one can be taken away from without
#: changing what they compute. Elementwise arithmetic follows its
#: operands; shape ops reshape to their recorded output; ``split_at``
#: needs its axis moved and knows to ask.
_UNIT_AXIS_SAFE = frozenset(
    {
        "add",
        "div",
        "floordiv",
        "fmod",
        "mul",
        "remainder",
        "sub",
        "maximum",
        "minimum",
        "pow",
        "reshape",
        "squeeze",
        "unsqueeze",
        "split_at",
    }
)


def _named(names: dict[int, str], tid: int, consumer: str) -> str:
    """The value a graph identifier stands for, or a diagnosis.

    A bare ``KeyError`` out of this lookup has been the visible end of
    two unrelated defects: an operation that never recorded its operands,
    so its consumers refer to something nothing produced; and a model
    called with fewer arguments than its forward takes, so a branch ran
    that the trace could not follow. Both arrive here as a number with
    nothing attached to it, several thousand operations after the cause.

    Neither is worth guessing between, but naming the operation and the
    identifier turns an unreadable failure into a place to look.

    Parameters
    ----------
    names : dict of int to str
        Graph identifier to the value the program calls it.
    tid : int
        Identifier being read.
    consumer : str
        What wanted it, for the message.

    Returns
    -------
    str
        The value's name in the program.

    Raises
    ------
    ValueError
        The identifier names nothing the graph produced.
    """
    try:
        return names[tid]
    except KeyError:
        raise ValueError(
            f"lucid.coreml: {consumer} reads value {tid}, which nothing in the "
            f"traced graph produces. Two things do this: an operation that did "
            f"not record its operands (the trace holds it with none, and its "
            f"consumers point past it), and a model called with fewer arguments "
            f"than its forward takes, so a default sent it down a path the trace "
            f"could not follow. Check the call first — it is the likelier one"
        ) from None


#: Key under which the package records which of its inputs stand in for
#: a draw. Core ML carries the creator-defined dictionary untouched, so
#: this is the one channel a fact about the *program* has into a handle
#: that reopened the file.
_DRAWN_KEY = "lucid.coreml.draws"

#: Keys under which the package records the precision it was built at —
#: the body's, and the interface's. Nothing else in the file states
#: either as one fact: the body is a scatter of casts and typed
#: constants, and a handle that reopened the package used to assume
#: float32, then told the owner of a float16 program that float32 was
#: why the Neural Engine took none of it.
_PRECISION_KEY = "lucid.coreml.precision"
_IO_PRECISION_KEY = "lucid.coreml.io_precision"

#: Operations that make a fresh sample out of nothing. Core ML folds
#: them at build time, so they are either refused or lifted to an input
#: the caller fills — see :class:`~lucid.coreml.Draws`.
_DRAW_OPS = frozenset({"randn", "rand"})


def _lift_draws(graph: Any, traced_values: dict[int, Any]) -> list[tuple[str, int, Tensor, str]]:
    """Each random draw in the graph, as an input the caller supplies.

    The draw was never part of the network. ``mu + sigma * eps`` is a
    function of ``eps``; that the eager model happens to make its own
    ``eps`` on the way past is an implementation detail of running in
    Python, and Core ML cannot do it — a folded draw returns one sample
    forever.

    Numbered in graph order, so the same model exported twice presents
    the same interface.

    Parameters
    ----------
    graph : TraceGraph
        Traced operations.
    traced_values : dict of int to TensorImpl
        Values the tracer kept, which give each draw its shape and dtype.

    Returns
    -------
    list of tuple
        ``(input name, graph identifier, the sample the trace drew,
        which operation drew it)``. The sample is kept so a comparison
        against the eager model has the numbers that model actually used,
        and the operation's name is kept because a handle drawing for a
        caller has to draw from the same distribution — filling a uniform
        draw with a normal one is a package that runs and is wrong.
    """
    lifted: list[tuple[str, int, Tensor, str]] = []
    for op in graph.ops:
        if op.name not in _DRAW_OPS or not op.outputs:
            continue
        tid = int(op.outputs[0].id)
        sample = traced_values.get(tid)
        if sample is None:
            raise ValueError(
                f"lucid.coreml: {op.name} was drawn and not kept, so there is no "
                f"shape to declare an input from. This is a tracer defect rather "
                f"than anything about the model"
            )
        lifted.append((f"noise_{len(lifted)}", tid, _wrap(sample), op.name))
    return lifted


def _tall_unit_ids(graph: TraceGraph, handled: set[int]) -> set[int]:
    tall: set[int] = set()
    for op in graph.ops:
        for out in op.outputs:
            if len(out.shape) > _staging.MAX_RANK and out.id not in handled:
                if int(out.shape[0]) != 1:
                    return set()
                tall.add(out.id)
    return tall


def _unit_axis_rewrites(graph: Any, handled: set[int], folded: set[int] | None = None) -> set[int]:
    """Rank-6 values that can be emitted without their leading axis."""
    tall = _tall_unit_ids(graph, handled)
    if not tall:
        return set()

    gone = folded or set()
    for op in graph.ops:
        if op.outputs and int(op.outputs[0].id) in gone:
            continue
        touches = any(o.id in tall for o in op.outputs) or any(i in tall for i in op.inputs)
        if touches and op.name not in _UNIT_AXIS_SAFE:
            return set()
    return tall


def _reachable_ops(graph: Any, output_ids: list[int]) -> list[Any]:
    """The traced ops the output actually depends on, in trace order.

    A trace can carry operations nothing consumes. Lucid's ``norm``
    computes a guard against a zero scale eagerly, so the comparison
    collapses to a constant feed and the ``bitwise_and`` that would have
    combined it is left behind with no operands and no consumer. Emitting
    such an op means asking an emitter for something the graph never
    computed; refusing to emit it is both correct and smaller.

    Parameters
    ----------
    graph : trace graph
        Graph to walk.
    output_ids : list[int]
        Values the package returns.

    Returns
    -------
    list
        Ops in their original order, minus the unreachable ones.
    """
    producer = {out.id: op for op in graph.ops for out in op.outputs}
    wanted = set(output_ids)
    pending = list(output_ids)
    while pending:
        op = producer.get(pending.pop())
        if op is None:
            continue
        for value in op.inputs:
            if value not in wanted:
                wanted.add(value)
                pending.append(value)
    return [op for op in graph.ops if any(out.id in wanted for out in op.outputs)]


def _apply_image_normalisation(tensor: Tensor, spec: ImageInput) -> Tensor:
    """``pixel * scale + bias``, the way the exported package applies it.

    Used to put the eager model on the same footing when verifying an
    image export: the package normalises internally, so comparing it
    against a model fed raw pixels would compare two different inputs.

    Parameters
    ----------
    tensor : Tensor
        Pixel values, shaped ``(1, C, H, W)``.
    spec : ImageInput
        The normalisation the package carries.

    Returns
    -------
    Tensor
        Normalised input.
    """
    out = tensor * spec.scale if spec.scale != 1.0 else tensor
    if spec.bias:
        offsets = lucid.tensor([[[[b]] for b in spec.bias]])
        out = out + offsets
    return out


class StatefulModel(NotImplementedError):
    """The model changed its own buffers while being traced.

    An exported package is a pure function of its inputs: whatever the
    buffers held is written into it as constants. A model that updates a
    buffer as it runs — a counter, a cache, a running statistic — keeps
    accumulating in eager and stops the moment it is exported.

    The first call still agrees, because the constants were read after
    the trace, which is what makes this worth refusing rather than
    warning about: ``verify`` runs one prediction and passes.

    Examples
    --------
    >>> import shutil, tempfile
    >>> import lucid, lucid.nn as nn, lucid.coreml as cml
    >>> class Counts(nn.Module):
    ...     def __init__(self):
    ...         super().__init__()
    ...         self.register_buffer("total", lucid.zeros(1, 4))
    ...     def forward(self, x):
    ...         self.total += x
    ...         return self.total + x
    >>> room = tempfile.mkdtemp()
    >>> try:
    ...     cml.export(Counts().eval(), lucid.ones(1, 4), f"{room}/m.mlpackage")
    ... except cml.StatefulModel as refusal:
    ...     print(refusal)
    lucid.coreml: tracing changed total, so this model is not a pure function ...
    >>> shutil.rmtree(room)
    """

    def __init__(self, names: list[str]) -> None:
        super().__init__(
            f"lucid.coreml: tracing changed {', '.join(names)}, so this model is "
            "not a pure function of its input and an exported package would stop "
            "accumulating after the first call — while still agreeing on that "
            "call. Express the state as an input the model reads and an output "
            "it returns, and name the pair with state=..., or export a model "
            "that does not write to its own buffers."
        )
        self.names = names


def _buffer_marks(model: Module) -> dict[str, float]:
    """A cheap fingerprint of every buffer, to notice one changing.

    A sum rather than a copy: buffers are read once before the trace and
    once after, and copying them all would double the memory an export
    already spends on weights. A mutation that leaves the sum unchanged
    would escape, which is a trade this check states rather than hides.

    Parameters
    ----------
    model : nn.Module
        Model whose buffers to fingerprint.

    Returns
    -------
    dict[str, float]
        Buffer name to the sum of its values.
    """
    marks: dict[str, float] = {}
    for name, buffer in model.named_buffers():
        if buffer is None:
            continue
        marks[name] = float(buffer.sum().item())
    return marks


def _probe(
    model: Module, example: Tensor, probes: list[tuple[int, ...]], output_field: str | None
) -> _flex.Flexibility:
    """Which axes of which traced values change with the input shape.

    Found by tracing the model once per shape and comparing, rather than
    propagating symbols through the graph. The tracer already knows every
    value's shape; asking it twice is cheaper than teaching it algebra,
    and it cannot disagree with itself about how an operation behaves.
    """
    records: list[_flex.Record] = []
    for shape in probes:
        record = _record(model, lucid.zeros(*shape).to(example.dtype), output_field)
        if records and _names(record) != _names(records[0]):
            # A cache the model fills on its first call at a shape — Swin's
            # attention mask — makes that call's graph longer than every
            # later one's; the later one is what an export runs.
            again = _record(model, lucid.zeros(*shape).to(example.dtype), output_field)
            record = again._replace(unsettled=_names(again) != _names(record))
        records.append(record)
    # A draw records the seed it used, and each probe drew again — so its
    # attributes always differ and never because of the size. The
    # operation is either refused outright or lifted to an input, so
    # nothing downstream reads them.
    return _flex.compare(records, probes, _DRAW_OPS)


def _names(record: _flex.Record) -> list[str]:
    return [name for name, _ids in record.signature]


def _record(model: Module, probe: Tensor, output_field: str | None) -> _flex.Record:
    """One probe trace, reduced to what the comparison reads."""
    graph, feeds, inputs, _outputs, _values = trace(model, probe, output_field=output_field)
    ops = list(graph.ops)
    given = int(inputs[0][1])
    return _flex.Record(
        [
            (op.name, dict(op.attrs), [tuple(int(d) for d in out.shape) for out in op.outputs])
            for op in ops
        ],
        [out.id for op in ops for out in op.outputs],
        given,
        _flex.signature(graph),
        # Non-input feed shapes (parameters, buffers, constants).
        {
            int(tid): tuple(int(d) for d in impl.shape)
            for tid, impl in feeds.items()
            if int(tid) != given
        },
    )


def _declare_image(
    program: Any, builder: Builder, name: str, shape: list[int], spec: ImageInput
) -> str:
    """Mark an input as an image and prepend its normalisation.

    Core ML puts the image *type* in the model description and the
    normalisation in the program: ``pixel * scale + bias`` is an ordinary
    ``mul`` and ``add`` at the head, before any cast to the body's
    precision, which is where a reference package puts them too.

    Parameters
    ----------
    program : MilProgram
        Program being written.
    builder : Builder
        Builder minting the constants.
    name : str
        Input feature name.
    shape : list[int]
        Input shape, which must be ``(1, C, H, W)``.
    spec : ImageInput
        Colour layout and normalisation.

    Returns
    -------
    str
        Value name the rest of the program should read.
    """
    if len(shape) != 4 or shape[0] != 1:
        raise ValueError(
            f"lucid.coreml: an image input must be (1, C, H, W), and this one is " f"{tuple(shape)}"
        )
    channels = shape[1]
    wanted = 1 if spec.color is ColorSpace.GRAYSCALE else 3
    if channels != wanted:
        raise ValueError(
            f"lucid.coreml: {spec.color.value} needs {wanted} channel(s) and the "
            f"input has {channels}"
        )
    if spec.bias and len(spec.bias) != channels:
        raise ValueError(f"lucid.coreml: bias has {len(spec.bias)} entries for {channels} channels")

    program.set_image_input(name, shape[3], shape[2], _spec.color_space(spec.color))

    source = name
    interface = (_spec.FLOAT32, shape)
    if spec.scale != 1.0:
        scaled = "_image_scaled"
        program.add_op(
            "mul",
            _operands([("x", source), ("y", builder.const_float32(spec.scale))]),
            scaled,
            interface,
        )
        builder.declare(scaled, shape)
        source = scaled
    if spec.bias:
        biased = "_image_biased"
        offsets = builder.const_float32_shaped(list(spec.bias), [1, channels, 1, 1])
        program.add_op("add", _operands([("x", source), ("y", offsets)]), biased, interface)
        builder.declare(biased, shape)
        source = biased
    return source


@dataclasses.dataclass
class _Shared:
    """A package and blob several functions are being written into."""

    paths: PackagePaths
    blob: BlobWriter
    offsets: dict[int, int] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class _BuildCtx:
    """Shared state across the stages of :func:`build_package`."""

    model: Module
    example: object
    path: str
    precision: Precision
    io_precision: Precision
    weights: WeightPrecision | _spec.Palettize | _spec.Sparsify
    shapes: list[tuple[int, ...]] | None
    shape_range: dict[int, tuple[int, int]] | None
    state: list[State] | None
    image_input: ImageInput | None
    classifier: Classifier | None
    metadata: Metadata | None
    output_field: str | None
    draws: _spec.Draws
    activations: _spec.Activations
    into: _Shared | None
    target: _spec.DeploymentTarget
    half: bool
    half_io: bool
    body_mil: int
    body_blob: int
    paths: PackagePaths = dataclasses.field(init=False)
    blob: BlobWriter = dataclasses.field(init=False)
    program: MilProgram = dataclasses.field(init=False)
    builder: Builder = dataclasses.field(init=False)
    graph: TraceGraph = dataclasses.field(init=False)
    feeds: dict[int, _C_engine.TensorImpl] = dataclasses.field(default_factory=dict)
    inputs: list[tuple[str, int, Tensor]] = dataclasses.field(default_factory=list)
    outputs: list[tuple[str, int, Tensor]] = dataclasses.field(default_factory=list)
    traced_values: dict[int, _C_engine.TensorImpl] = dataclasses.field(default_factory=dict)
    flexibility: _flex.Flexibility = dataclasses.field(default_factory=_flex.Flexibility)
    ordered_shapes: list[tuple[int, ...]] = dataclasses.field(default_factory=list)
    bounds: list[tuple[int, int]] = dataclasses.field(default_factory=list)
    varying: dict[int, set[int]] = dataclasses.field(default_factory=dict)
    lifted: list[tuple[str, int, Tensor, str]] = dataclasses.field(default_factory=list)
    lifted_heads: set[int] = dataclasses.field(default_factory=set)
    lifted_names: set[str] = dataclasses.field(default_factory=set)
    asked_for: int = 0
    carried: dict[str, State] = dataclasses.field(default_factory=dict)
    plain_inputs: list[tuple[str, int, Tensor]] = dataclasses.field(default_factory=list)
    names: dict[int, str] = dataclasses.field(default_factory=dict)
    input_ids: set[int] = dataclasses.field(default_factory=set)
    builder_shapes_state: dict[str, list[int]] = dataclasses.field(default_factory=dict)
    weight_shapes: dict[str, list[int]] = dataclasses.field(default_factory=dict)
    weight_dtypes: dict[str, int] = dataclasses.field(default_factory=dict)
    quantized_count: int = 0
    f32_opmath_scalars: set[int] = dataclasses.field(default_factory=set)
    rewrites: dict[int, _staging.WindowRewrite] = dataclasses.field(default_factory=dict)
    stuffing: dict[int, _staging.StuffRewrite] = dataclasses.field(default_factory=dict)
    absorbed: set[int] = dataclasses.field(default_factory=set)
    precomputed: set[int] = dataclasses.field(default_factory=set)
    folded: set[int] = dataclasses.field(default_factory=set)
    thinned: set[int] = dataclasses.field(default_factory=set)
    wanted_name: dict[int, str] = dataclasses.field(default_factory=dict)
    cast_out: bool = False


def _trace_and_probe(ctx: _BuildCtx) -> None:
    """Run the initial trace and any flexibility shape probes."""
    with (
        _activation_quantization_paused(ctx.model)
        if ctx.activations is _spec.Activations.DROPPED
        else contextlib.nullcontext()
    ):
        graph, feeds, inputs, outputs, traced_values = trace(
            ctx.model, ctx.example, output_field=ctx.output_field
        )
        probes, ordered, bounds = _flex.plan_probes(
            tuple(int(d) for d in inputs[0][2].shape) if inputs else (),
            len(inputs),
            ctx.shapes,
            ctx.shape_range,
        )
        flexibility = _flex.Flexibility()
        if probes:
            flexibility = _probe(ctx.model, inputs[0][2], probes, ctx.output_field)
            if not _flex.same_graph(flexibility, graph, inputs[0][1]):
                graph, feeds, inputs, outputs, traced_values = trace(
                    ctx.model, ctx.example, output_field=ctx.output_field
                )
            if not _flex.same_graph(flexibility, graph, inputs[0][1]):
                raise ShapeNotFlexible(
                    "<graph>",
                    "the model traces a different graph at the example's shape "
                    "on different calls, so which values follow the input "
                    "cannot be told from comparing them",
                )
    ctx.graph = graph
    ctx.feeds = feeds
    ctx.inputs = inputs
    ctx.outputs = outputs
    ctx.traced_values = traced_values
    ctx.flexibility = flexibility
    ctx.ordered_shapes = ordered
    ctx.bounds = bounds
    ctx.varying = flexibility.varying


def _validate_state_and_draws(ctx: _BuildCtx) -> None:
    """Validate carried state specs and lifted random draws."""
    ctx.lifted = (
        _lift_draws(ctx.graph, ctx.traced_values) if ctx.draws is _spec.Draws.AS_INPUT else []
    )
    ctx.asked_for = len(ctx.inputs)
    ctx.carried = {spec.input: spec for spec in (ctx.state or [])}
    if ctx.carried and not ctx.half:
        raise ValueError(
            "lucid.coreml: a carried state is stored as float16 — Core ML accepts "
            "no other type for one — so the body has to be float16 too, or the "
            "value written back would not be the value read. Pass "
            "precision=Precision.FLOAT16"
        )
    if ctx.carried:
        _check_carried_state(ctx.state or [], ctx.inputs, ctx.outputs)
    for name, tid, sample, kind in ctx.lifted:
        if tid in ctx.varying:
            raise ShapeNotFlexible(
                kind,
                f"its result is shaped {tuple(int(d) for d in sample.shape)} at one "
                f"input size and differently at another, so the input {name!r} it "
                "becomes would have to be flexible too — and which of the package's "
                "inputs a caller may vary is one choice, not two. Export it at a "
                "single shape",
            )
        ctx.lifted_heads.add(tid)
        ctx.lifted_names.add(name)
        ctx.inputs.append((name, tid, sample))
    ctx.plain_inputs = [entry for entry in ctx.inputs if entry[0] not in ctx.carried]
    if not ctx.plain_inputs:
        raise ValueError(
            "lucid.coreml: every input was declared state, leaving nothing for the "
            "caller to pass"
        )


def _check_carried_state(
    state: list[State],
    inputs: list[tuple[str, int, Tensor]],
    outputs: list[tuple[str, int, Tensor]],
) -> None:
    by_input = {name for name, _tid, _t in inputs}
    by_output = {field for field, _tid, _t in outputs}
    for spec in state:
        if spec.input not in by_input:
            raise ValueError(
                f"lucid.coreml: {spec.input!r} is not an input of this model " f"{sorted(by_input)}"
            )
        if spec.output not in by_output:
            raise ValueError(
                f"lucid.coreml: {spec.output!r} is not an output of this model "
                f"{sorted(by_output)}"
            )
        held = next(t for n, _i, t in inputs if n == spec.input)
        written = next(t for f, _i, t in outputs if f == spec.output)
        if tuple(held.shape) != tuple(written.shape):
            raise ValueError(
                f"lucid.coreml: state {spec.input!r} is shaped "
                f"{tuple(int(d) for d in held.shape)} and {spec.output!r} is "
                f"{tuple(int(d) for d in written.shape)}; what is written back "
                "has to be what was read"
            )


def _half_interface(ctx: _BuildCtx, name: str, tensor: Tensor) -> bool:
    """Whether ``name`` is declared float16 and read without a cast."""
    is_image = ctx.image_input is not None and name not in ctx.lifted_names
    is_float = tensor.dtype in (lucid.float32, lucid.float16)
    return ctx.half_io and is_float and not is_image


def _declare_carried_state(ctx: _BuildCtx) -> None:
    for name, tid, tensor in ctx.inputs:
        ctx.names[tid] = name
        ctx.input_ids.add(tid)
        if name not in ctx.carried:
            continue
        shape = [int(d) for d in tensor.shape]
        carried_type = (ctx.body_mil, shape)
        ctx.program.add_state(name, carried_type)
        held = f"_state_{name}"
        ctx.program.read_state(name, held, carried_type)
        ctx.names[tid] = held
        ctx.builder_shapes_state[held] = shape


def _init_program_and_state(ctx: _BuildCtx) -> None:
    """Create the ``MilProgram``, configure shapes, and declare state."""
    cm = _C_engine.coreml
    ctx.paths = ctx.into.paths if ctx.into is not None else cm.prepare_package(ctx.path)
    ctx.program = cm.MilProgram(
        [
            (
                name,
                (
                    (
                        _spec.FLOAT16
                        if _half_interface(ctx, name, tensor)
                        else _spec.mil_dtype(tensor.dtype)
                    ),
                    _flex.flex([int(d) for d in tensor.shape], ctx.varying.get(tid)),
                ),
            )
            for name, tid, tensor in ctx.plain_inputs
        ]
    )
    if ctx.shapes is not None:
        ctx.program.set_enumerated_shapes(ctx.inputs[0][0], [list(s) for s in ctx.ordered_shapes])
        ctx.program.set_default_shape(ctx.inputs[0][0], list(ctx.ordered_shapes[0]))
    if ctx.shape_range is not None:
        ctx.program.set_shape_range(ctx.inputs[0][0], ctx.bounds)
        ctx.program.set_default_shape(ctx.inputs[0][0], [int(d) for d in ctx.inputs[0][2].shape])
    _declare_carried_state(ctx)
    ctx.blob = ctx.into.blob if ctx.into is not None else cm.BlobWriter(ctx.paths.weight_bin)


def _write_palettized_const(
    ctx: _BuildCtx, name: str, shape: list[int], palettized: tuple[bytes, Tensor, list[int], int]
) -> None:
    keys, palettes, lut_shape, key_bits = palettized
    offset = ctx.blob.append_bytes(keys, _C_engine.coreml.BLOB_SUBBYTE[str(key_bits)])
    lut_offset = ctx.blob.append_tensor(
        _unwrap(palettes.half() if ctx.half else palettes), ctx.body_blob
    )
    ctx.program.add_grouped_lut_const(
        name,
        (ctx.body_mil, shape),
        offset,
        _C_engine.coreml.MIL_SUBBYTE[str(key_bits)],
        lut_offset,
        ctx.body_mil,
        lut_shape,
    )
    ctx.quantized_count += 1


def _write_sparse_const(
    ctx: _BuildCtx, name: str, shape: list[int], sparse: tuple[Tensor, bytes, int]
) -> None:
    values, mask, kept = sparse
    if ctx.half:
        values = values.half()
    value_offset = ctx.blob.append_tensor(_unwrap(values), ctx.body_blob)
    if ctx.target is _spec.DeploymentTarget.IOS18:
        mask_offset = ctx.blob.append_bytes(mask, _C_engine.coreml.BLOB_SUBBYTE["1"])
        ctx.program.add_sparse_const_extended(
            name, (ctx.body_mil, shape), value_offset, kept, ctx.body_mil, mask_offset
        )
    else:
        mask_offset = ctx.blob.append_bytes(mask, _spec.BLOB_UINT8)
        ctx.program.add_sparse_const(
            name, (ctx.body_mil, shape), value_offset, kept, mask_offset, len(mask)
        )
    ctx.quantized_count += 1


def _write_float_const(ctx: _BuildCtx, name: str, tensor: Tensor, dedup: int | None) -> None:
    """Write one float constant, compressed the way ``ctx.weights`` asks."""
    shape = [int(d) for d in tensor.shape]
    if isinstance(ctx.weights, _spec.Palettize):
        palettized = _palettize_weight(tensor, ctx.weights.bits)
        if palettized is not None:
            _write_palettized_const(ctx, name, shape, palettized)
            return
    elif isinstance(ctx.weights, _spec.Sparsify):
        sparse = _sparsify_weight(tensor, ctx.weights.ratio)
        if sparse is not None:
            _write_sparse_const(ctx, name, shape, sparse)
            return
    elif ctx.weights is WeightPrecision.INT8:
        quantized = _quantize_weight(tensor, ctx.body_mil)
        if quantized is not None:
            codes, scale_bytes, zero_bytes, channels = quantized
            offset = ctx.blob.append_tensor(_unwrap(codes), _spec.BLOB_INT8)
            ctx.program.add_quantized_const(
                name,
                (ctx.body_mil, shape),
                offset,
                scale_bytes,
                ctx.body_mil,
                zero_bytes,
                channels,
                0,
            )
            ctx.quantized_count += 1
            return

    is_f16 = tensor.dtype == lucid.float16
    mil_dt = _spec.FLOAT16 if is_f16 else _spec.FLOAT32
    blob_dt = _spec.BLOB_FLOAT16 if is_f16 else _spec.BLOB_FLOAT32
    blob_offset = ctx.into.offsets.get(dedup) if ctx.into is not None and dedup else None
    if blob_offset is None:
        blob_offset = ctx.blob.append_tensor(_unwrap(tensor), blob_dt)
        if ctx.into is not None and dedup:
            ctx.into.offsets[dedup] = blob_offset
    ctx.program.add_blob_const(name, (mil_dt, shape), blob_offset)


def _write_feed_constants(ctx: _BuildCtx) -> None:
    """Write non-input external feeds as constants in the program and blob."""
    ctx.f32_opmath_scalars = _fold.f32_opmath_scalars(ctx.graph, ctx.feeds)
    for tid, impl in ctx.feeds.items():
        if tid in ctx.input_ids:
            continue
        tensor = _wrap(impl)
        _refuse_if_empty(tensor)
        is_float = tensor.dtype in (lucid.float32, lucid.float16)
        keep_f32 = tid in ctx.f32_opmath_scalars
        if ctx.half and is_float and not keep_f32:
            tensor = tensor.half()
        shape = [int(d) for d in tensor.shape]
        name = f"_w{tid}"
        if is_float:
            _write_float_const(ctx, name, tensor, None if keep_f32 else id(impl))
            ctx.weight_dtypes[name] = (
                _spec.FLOAT16 if tensor.dtype == lucid.float16 else _spec.FLOAT32
            )
        elif tensor.dtype == lucid.bool_:
            ctx.program.add_bool_const_shaped(name, _flatten(tensor, bool), shape)
            ctx.weight_dtypes[name] = _C_engine.coreml.DTYPE_BOOL
        else:
            values = _spec.as_int32(_flatten(tensor, _as_int), name)
            ctx.program.add_int_const_shaped(name, values, shape)
            ctx.weight_dtypes[name] = _C_engine.coreml.DTYPE_INT32
        ctx.names[tid] = name
        ctx.weight_shapes[name] = shape


def _verify_unthinned_ranks(ctx: _BuildCtx, staged_or_absorbed: set[int]) -> None:
    if ctx.thinned & set(ctx.varying):
        raise ShapeNotFlexible(
            "<rank-6 value>",
            "a value computed at rank 6 is emitted without its leading axis of "
            "one, and its shape changes with the input",
        )
    for op in ctx.graph.ops:
        head = int(op.outputs[0].id) if op.outputs else -1
        if head in staged_or_absorbed:
            continue
        for out in op.outputs:
            if len(out.shape) > _staging.MAX_RANK and out.id not in ctx.thinned:
                raise UnsupportedRank(op.name, tuple(int(d) for d in out.shape))


def _plan_rewrites_and_folding(ctx: _BuildCtx) -> None:
    """Plan window/stuffing rewrites, constant folding, and rank-6 thinning."""
    ctx.rewrites = _staging.window_rewrites(ctx.graph)
    ctx.stuffing = _staging.stuff_rewrites(ctx.graph)
    absorbed = {oid for r in ctx.rewrites.values() for oid in r.absorbs}
    absorbed |= {oid for r in ctx.stuffing.values() for oid in r.absorbs}
    ctx.absorbed = absorbed
    output_ids = {tid for _n, tid, _t in ctx.outputs}
    ctx.precomputed, ctx.folded = _fold.foldable(
        ctx.graph, ctx.feeds, set(ctx.input_ids), ctx.traced_values, set(ctx.varying), output_ids
    )
    staged_or_absorbed = set(ctx.rewrites) | set(ctx.stuffing) | absorbed
    ctx.thinned = _unit_axis_rewrites(ctx.graph, staged_or_absorbed, ctx.folded)
    _verify_unthinned_ranks(ctx, staged_or_absorbed)


def _prepare_builder_and_inputs(ctx: _BuildCtx) -> None:
    """Initialise the :class:`Builder`, register weights, and declare inputs."""
    builder = Builder(ctx.program, ctx.blob, ctx.body_mil, ctx.body_blob, ctx.half)
    builder.flex = ctx.flexibility
    builder.traced_names = ctx.names
    builder.thinned = set(ctx.thinned)
    for held_name, held_shape in {**ctx.weight_shapes, **ctx.builder_shapes_state}.items():
        builder.declare(held_name, held_shape)
    for weight_name in ctx.weight_shapes:
        builder.mark_const(weight_name)
        builder.dtypes[weight_name] = ctx.weight_dtypes[weight_name]
    if ctx.image_input is not None and ctx.asked_for != 1:
        raise ValueError(
            f"lucid.coreml: image_input needs a single-input model, and this one "
            f"takes {ctx.asked_for} — which of them is the image would be a guess"
        )
    for name, tid, tensor in ctx.plain_inputs:
        shape = _flex.flex(tensor.shape, ctx.varying.get(tid))
        builder.declare(name, shape)
        if tid == ctx.flexibility.input_id:
            builder.flex_input = name
        half_in = _half_interface(ctx, name, tensor)
        builder.dtypes[name] = _spec.FLOAT16 if half_in else _spec.mil_dtype(tensor.dtype)
        source = name
        if ctx.image_input is not None and name not in ctx.lifted_names:
            source = _declare_image(
                ctx.program, builder, name, [int(d) for d in tensor.shape], ctx.image_input
            )
            builder.declare(source, shape)
            ctx.names[tid] = source
        if (
            ctx.half
            and not half_in
            and tensor.dtype == lucid.float32
            and tid not in ctx.f32_opmath_scalars
        ):
            cast_name = f"_cast_in_{name}"
            mil_type, raw = emit_cast(builder, source, "fp16")
            ctx.program.add_op(mil_type, _operands(raw), cast_name, (ctx.body_mil, shape))
            builder.declare(cast_name, shape)
            builder.dtypes[cast_name] = ctx.body_mil
            ctx.names[tid] = cast_name
    ctx.builder = builder


def _emit_one_folded_constant(ctx: _BuildCtx, tid: int) -> bool:
    builder = ctx.builder
    folded_value = _wrap(ctx.traced_values[tid])
    if tid in ctx.thinned:
        folded_value = folded_value.reshape(*[int(d) for d in folded_value.shape][1:])
    if tid in ctx.f32_opmath_scalars:
        ctx.names[tid] = builder.const_float32(float(folded_value.item()))
        return True
    if folded_value.dtype in (lucid.float32, lucid.float16):
        written = folded_value.half() if ctx.half else folded_value
        if _worth_storing_compressed(written, ctx.weights):
            name = f"_f{tid}"
            _write_float_const(ctx, name, written, None)
            builder.mark_const(name)
            is_f16 = written.dtype == lucid.float16
            builder.dtypes[name] = _spec.FLOAT16 if is_f16 else ctx.body_mil
            builder.declare(name, [int(d) for d in written.shape])
            ctx.names[tid] = name
        else:
            ctx.names[tid] = builder.const_from_tensor(written)
        return True
    if folded_value.dtype in (lucid.int8, lucid.int16, lucid.int32, lucid.int64):
        ctx.names[tid] = builder.const_ints_shaped(
            [int(v) for v in _flatten(folded_value, _as_int)], [int(d) for d in folded_value.shape]
        )
        return True
    return False


def _emit_folded_constants(ctx: _BuildCtx) -> None:
    """Write precomputed constant-folded values into the program."""
    carried_out = {spec.output for spec in (ctx.state or [])}
    ctx.cast_out = ctx.half and not (ctx.half_io and ctx.classifier is None)
    ctx.wanted_name = {
        tid: field
        for field, tid, t in ctx.outputs
        if field not in carried_out and not (ctx.cast_out and t.dtype == lucid.float32)
    }
    for tid in ctx.precomputed:
        if not _emit_one_folded_constant(ctx, tid):
            ctx.folded, ctx.precomputed = set(), set()
            break


def _try_emit_staged_rewrite(ctx: _BuildCtx, op: TraceNode, head: int) -> bool:
    """Emit a window or zero-stuffing rewrite if ``head`` starts one."""
    plan = ctx.rewrites.get(head)
    if plan is not None:
        plan = plan._replace(
            mid_shape=_flex.flex(plan.mid_shape, ctx.varying.get(head)),
            out_shape=_flex.flex(plan.out_shape, ctx.varying.get(plan.out_id)),
        )
        ctx.names[plan.out_id] = _staging.stage_permutation(
            ctx.builder, ctx.names[op.inputs[0]], plan
        )
        return True
    stuff = ctx.stuffing.get(head)
    if stuff is not None:
        wide = _flex.flex(stuff.out_shape, ctx.varying.get(stuff.out_id))
        stuff = dataclasses.replace(
            stuff,
            source_shape=[
                -1 if w < 0 else d for d, w in zip(stuff.source_shape, wide, strict=True)
            ],
            out_shape=wide,
        )
        ctx.names[stuff.out_id] = _staging.stage_zero_stuff(
            ctx.builder, ctx.names[op.inputs[0]], stuff
        )
        return True
    return False


def _emit_single_op(ctx: _BuildCtx, op: TraceNode) -> None:
    """Emit one reachable traced operation into ``ctx.program``."""
    builder = ctx.builder
    emitter = EMITTERS.get(op.name)
    if emitter is None:
        raise UnsupportedOp(op.name)
    operands = [_named(ctx.names, i, f"operation {op.name!r}") for i in op.inputs]
    result = emitter(builder, _flex.GuardedOp(op, ctx.varying) if ctx.varying else op, operands)
    if isinstance(result, Constant):
        ctx.names[op.outputs[0].id] = result.name
        return
    if isinstance(result, Bound):
        for out, bound in zip(op.outputs, result.names, strict=True):
            ctx.names[out.id] = bound
        return
    outs = op.outputs if isinstance(result, MultiOutput) else op.outputs[:1]
    mil_type, raw_bindings = (
        (result.mil_type, result.bindings) if isinstance(result, MultiOutput) else result
    )
    bindings = _operands(raw_bindings)
    produced = [
        (
            ctx.wanted_name.get(o.id, f"_v{o.id}_{op.name}"),
            (
                _spec.trace_dtype(str(o.dtype).split(".")[-1], ctx.body_mil),
                _flex.flex(
                    [int(d) for d in o.shape][1 if o.id in ctx.thinned else 0 :],
                    ctx.varying.get(o.id),
                ),
            ),
        )
        for o in outs
    ]
    if isinstance(result, MultiOutput):
        ctx.program.add_op_multi(mil_type, bindings, produced)
    else:
        ctx.program.add_op(mil_type, bindings, produced[0][0], produced[0][1])
    for (out_name, (out_dtype, shape)), o in zip(produced, outs, strict=True):
        ctx.names[o.id] = out_name
        builder.declare(out_name, shape)
        builder.dtypes[out_name] = out_dtype
        if o.id in ctx.thinned:
            builder.unit_axis_dropped.add(out_name)


def _emit_reachable_ops(ctx: _BuildCtx) -> None:
    """Emit every reachable non-folded op in topological trace order."""
    for op in _reachable_ops(ctx.graph, [tid for _n, tid, _t in ctx.outputs]):
        if op.outputs and int(op.outputs[0].id) in ctx.folded:
            continue
        head = int(op.outputs[0].id) if op.outputs else -1
        if head in ctx.lifted_heads or head in ctx.absorbed:
            continue
        ctx.builder.current = op.name
        if _try_emit_staged_rewrite(ctx, op, head):
            continue
        _emit_single_op(ctx, op)
    if ctx.into is None:
        ctx.blob.finalize()


def _check_classifier(classifier: Classifier, outputs: list[tuple[str, int, Tensor]]) -> None:
    if len(outputs) != 1:
        raise ValueError(
            f"lucid.coreml: a classifier needs one output and this model "
            f"returns {len(outputs)} ({[name for name, _t, _x in outputs]})"
        )
    scores = outputs[0][2]
    score_shape = tuple(int(d) for d in scores.shape)
    if len(score_shape) != 2 or score_shape[0] != 1:
        raise ValueError(
            f"lucid.coreml: a classifier's output must be (1, classes) and this "
            f"one is {score_shape}"
        )
    if score_shape[1] != len(classifier.labels):
        raise ValueError(
            f"lucid.coreml: {len(classifier.labels)} labels for " f"{score_shape[1]} scores"
        )


def _set_package_metadata(ctx: _BuildCtx) -> None:
    if ctx.metadata is not None:
        ctx.program.set_metadata(
            ctx.metadata.description,
            ctx.metadata.author,
            ctx.metadata.license,
            ctx.metadata.version,
        )
    ctx.program.set_user_metadata(_PRECISION_KEY, ctx.precision.value)
    ctx.program.set_user_metadata(_IO_PRECISION_KEY, ctx.io_precision.value)
    if ctx.lifted:
        ctx.program.set_user_metadata(
            _DRAWN_KEY,
            ",".join(
                f"{name}:{kind}:{'x'.join(str(int(d)) for d in sample.shape)}"
                for name, _tid, sample, kind in ctx.lifted
            ),
        )
    if ctx.into is None:
        _C_engine.coreml.finish_package(ctx.paths, ctx.program.serialize())


def _declare_outputs_and_metadata(ctx: _BuildCtx) -> list[tuple[str, tuple[int, ...]]]:
    """Declare program outputs, state write-backs, classifier head, and metadata."""
    if ctx.classifier is not None:
        _check_classifier(ctx.classifier, ctx.outputs)
    written = {spec.output: spec.input for spec in (ctx.state or [])}
    declared: list[tuple[str, tuple[int, ...]]] = []
    for field, tid, tensor in ctx.outputs:
        if field in written:
            ctx.program.write_state(
                written[field], _named(ctx.names, tid, f"the state written from {field!r}")
            )
            continue
        value = _named(ctx.names, tid, f"output {field!r}")
        is_float = tensor.dtype in (lucid.float32, lucid.float16)
        flexible_type = (
            (
                _spec.FLOAT16
                if is_float and not ctx.cast_out and ctx.half_io
                else _spec.mil_dtype(tensor.dtype)
            ),
            _flex.flex([int(d) for d in tensor.shape], ctx.varying.get(tid)),
        )
        if ctx.cast_out and tensor.dtype == lucid.float32:
            mil_type, raw = emit_cast(ctx.builder, value, "fp32")
            ctx.program.add_op(mil_type, _operands(raw), field, flexible_type)
            value = field
        if value != field:
            ctx.program.add_op("identity", [("x", [value])], field, flexible_type)
        if ctx.classifier is not None:
            ctx.program.set_classifier(
                value,
                list(ctx.classifier.labels),
                ctx.classifier.label_name,
                ctx.classifier.probabilities_name,
            )
            declared.append((ctx.classifier.label_name, ()))
            declared.append((ctx.classifier.probabilities_name, ()))
            continue
        ctx.program.add_output(field, flexible_type)
        declared.append((field, tuple(flexible_type[1])))
    _set_package_metadata(ctx)
    return declared


def _weights_summary_name(weights: WeightPrecision | _spec.Palettize | _spec.Sparsify) -> str:
    if isinstance(weights, WeightPrecision):
        return weights.value
    if isinstance(weights, _spec.Palettize):
        return f"PALETTIZED_{weights.bits}BIT"
    return f"SPARSE_{weights.ratio:g}"


def _build_package_summary(
    ctx: _BuildCtx, declared: list[tuple[str, tuple[int, ...]]]
) -> dict[str, object]:
    return {
        "inputs": [
            (name, tuple(int(d) for d in tensor.shape)) for name, _tid, tensor in ctx.plain_inputs
        ],
        "outputs": declared,
        "ops": int(ctx.program.op_count),
        "precision": ctx.precision.value,
        "io_precision": ctx.io_precision.value,
        "weights": _weights_summary_name(ctx.weights),
        "classifier": ctx.classifier is not None,
        "noise": [
            (name, tuple(int(d) for d in sample.shape), kind, sample)
            for name, _tid, sample, kind in ctx.lifted
        ],
        "traced_outputs": (
            {field: tensor for field, _tid, tensor in ctx.outputs} if ctx.lifted else {}
        ),
        "quantized_weights": ctx.quantized_count,
        "flexible": ctx.shapes is not None or ctx.shape_range is not None,
        "state": [(spec.input, spec.output) for spec in (ctx.state or [])],
        "deployment_target": _floor_of(str(ctx.paths.root)) or ctx.target,
        "program": ctx.program,
        "path": ctx.paths.root,
    }


def _run_build_pipeline(ctx: _BuildCtx) -> dict[str, object]:
    _trace_and_probe(ctx)
    _validate_state_and_draws(ctx)
    _init_program_and_state(ctx)
    _write_feed_constants(ctx)
    _plan_rewrites_and_folding(ctx)
    _prepare_builder_and_inputs(ctx)
    _emit_folded_constants(ctx)
    _emit_reachable_ops(ctx)
    declared = _declare_outputs_and_metadata(ctx)
    return _build_package_summary(ctx, declared)


def build_package(
    model: Module,
    example: object,
    path: str,
    *,
    precision: Precision = Precision.FLOAT32,
    weights: WeightPrecision | _spec.Palettize | _spec.Sparsify = WeightPrecision.FLOAT,
    shapes: list[tuple[int, ...]] | None = None,
    shape_range: dict[int, tuple[int, int]] | None = None,
    state: list[State] | None = None,
    image_input: ImageInput | None = None,
    classifier: Classifier | None = None,
    metadata: Metadata | None = None,
    output_field: str | None = None,
    minimum_deployment_target: _spec.DeploymentTarget | None = None,
    draws: _spec.Draws = _spec.Draws.REFUSED,
    activations: _spec.Activations = _spec.Activations.SIMULATED,
    into: _Shared | None = None,
    io_precision: Precision = Precision.FLOAT32,
) -> dict[str, object]:
    """Trace ``model`` and write a complete ``.mlpackage`` at ``path``.

    Parameters
    ----------
    model : nn.Module
        Model to export, in ``eval()`` mode.
    example : Tensor or tuple of Tensor or dict of str to Tensor
        Supplies each input's shape and dtype (tuple positionally, dict by key).
    path : str
        Destination package path, replaced if it already exists.
    precision : Precision, optional, keyword-only, default=FLOAT32
        Body precision (``FLOAT32`` or ``FLOAT16``).
    io_precision : Precision, optional, keyword-only, default=FLOAT32
        Element type of float inputs and outputs; ``FLOAT16`` requires a float16 body.
    weights : WeightPrecision, optional, keyword-only, default=FLOAT
        How weights are stored (``FLOAT``, ``INT8``, ``Palettize``, or ``Sparsify``).
    image_input : ImageInput or None, optional, keyword-only, default=None
        Present the sole input as an image with its expected normalisation.
    shapes : list of tuple of int or None, optional, keyword-only, default=None
        Enumerated input shapes the package should accept.
    shape_range : dict of int to tuple of int or None, optional, keyword-only, default=None
        Axis to ``(lowest, highest)`` bounds, mutually exclusive with ``shapes``.
    state : list of State or None, optional, keyword-only, default=None
        Input/output state pairs carried across predictions.
    classifier : Classifier or None, optional, keyword-only, default=None
        Declare the model a classifier over these labels.
    metadata : Metadata or None, optional, keyword-only, default=None
        Package metadata (author, license, description, version).
    activations : _spec.Activations, optional, keyword-only, default=SIMULATED
        How QAT activation fake-quantization is lowered (``SIMULATED`` or ``DROPPED``).
    draws : _spec.Draws, optional, keyword-only, default=REFUSED
        How random draws in ``forward`` are handled (``REFUSED`` or ``AS_INPUT``).
    minimum_deployment_target : DeploymentTarget or None, optional, keyword-only, default=None
        Oldest deployment target the package must run on.
    output_field : str or None, optional, keyword-only, default=None
        Single attribute to export from an output dataclass.
    into : _Shared or None, optional, keyword-only, default=None
        Open package and weight blob when building one function of several.

    Returns
    -------
    dict
        Summary of ``inputs``, ``outputs``, ``ops``, ``precision``, and ``path``.

    Raises
    ------
    UnsupportedOp
        The trace contains an op with no MIL translation.
    ValueError
        A float16 interface was asked for around a float32 body.
    """
    half_io = io_precision is Precision.FLOAT16
    if half_io and precision is not Precision.FLOAT16:
        raise ValueError(
            "lucid.coreml: io_precision=Precision.FLOAT16 declares float16 "
            "inputs and outputs, and the body is float32 — the interface would "
            "only round what the program computed. Pass "
            "precision=Precision.FLOAT16 as well"
        )
    target = _settle_target(
        minimum_deployment_target, state=state, weights=weights, functions=into is not None
    )
    body_mil, body_blob = _spec.body_dtypes(precision)
    return _run_build_pipeline(
        _BuildCtx(
            model=model,
            example=example,
            path=path,
            precision=precision,
            io_precision=io_precision,
            weights=weights,
            shapes=shapes,
            shape_range=shape_range,
            state=state,
            image_input=image_input,
            classifier=classifier,
            metadata=metadata,
            output_field=output_field,
            draws=draws,
            activations=activations,
            into=into,
            target=target,
            half=precision is Precision.FLOAT16,
            half_io=half_io,
            body_mil=body_mil,
            body_blob=body_blob,
        )
    )


# Bound at runtime, not only for type checking: PEP 649 evaluates a
# function's annotations in its own module globals when something asks
# for them, so a name that exists under ``TYPE_CHECKING`` alone makes
# ``inspect.signature`` raise NameError — which is what help(), an IDE
# and the docs build all call.
from lucid._tensor.tensor import Tensor
from lucid.nn.module import Module
