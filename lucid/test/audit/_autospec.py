"""Deriving the invocation from the signature instead of writing it out.

:mod:`~lucid.test.audit._specs` is a hand-written ladder: a family
pattern per group of ops, and a generic ``f(x)`` / ``f(x, y)`` floor for
everything else.  446 of 827 callable symbols reached only that floor,
which is fine for a plain unary op and useless for anything taking a
second required argument of a shape the floor cannot guess.  Those became
SKIP — counted honestly, and still not verified.

That is the same failure mode as the module list which held the audit's
reach at 73.8%: a hand-maintained enumeration of something the
interpreter already knows.  A signature is machine-readable, so read it.
With this module the figure is 825 of 827; the two left take no arguments
at all, and calling them with none is what the smoke axis already does.

    def conv2d(input, weight, bias=None, stride=1, ...)
              ^tensor ^tensor  ^skipped, it has a default

Three rules, in order of how much they know:

1.  **Relations** — a parameter whose value depends on one already
    chosen.  ``weight`` has to agree with ``input`` on channels, a
    class-index ``target`` has to be bounded by the logits' width.  This
    is the only part that encodes semantics, and it is a dozen entries
    rather than one per op.
2.  **Name** — ``dim``, ``eps``, ``groups``, ``kernel_size``.  A name
    says more than ``int`` does.
3.  **Annotation text** — the fallback: ``Tensor``, ``bool``,
    ``tuple[int, int]``, ``Literal["mean", "sum"]``.

Only parameters *without* a default are filled.  A default is the
author's own statement of a value that works, and overriding it would
test this module's opinion rather than the op.

Annotations are read with ``Format.STRING``.  Lucid puts tensor types in
a ``TYPE_CHECKING`` block and leaves the annotation bare (H1 / H7 plus
PEP 649), so resolving them raises ``NameError`` for a large part of the
framework — the unevaluated text is both sufficient and always available.
"""

import annotationlib
import atexit
import inspect
import re
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

import lucid
import lucid.nn.utils.rnn
from lucid.test.audit import _probe
from lucid.test.audit._specs import Call

if TYPE_CHECKING:
    from collections.abc import Iterator

#: Probe extents, matching :mod:`_specs` so a hand spec and a derived one
#: produce comparably sized work.
_N, _CIN, _COUT = 2, 3, 4
_L, _H, _W = 8, 6, 6

#: How many derived candidates a single symbol may yield.  The variants
#: exist to survive a wrong guess about rank or ``dim``, not to search;
#: an unbounded product would multiply the sweep by the arity.
_MAX_CANDIDATES = 6


#: Parameters whose probe is built in **float32**, not the sweep's
#: float64 working precision.  ``utils.transforms.functional`` builds its
#: sampling grids and its masks single-precision, so a float64 image
#: reaches an internal ``grid_sample`` or ``where`` that refuses it —
#: ``DtypeMismatch (grid_sample): expected float64, got float32``, which
#: names the *probe's* dtype as the expectation and reads as a defect in
#: a function that works.  An image transform's domain is float32 images
#: and that is what it should be handed.
_F32_PARAMS = frozenset({"img", "image", "pts", "points", "matrix", "field", "dx", "dy"})


def _tensor(shape: tuple[int, ...], domain: str, variant: int = 0, name: str = "") -> Any:
    """One tensor argument.

    ``variant`` must differ between the operands of a binary op.  Built
    the same way they were identical — the draw is seeded — and every
    comparison was then probed exactly on its tie, which is the one input
    where it has no derivative: ``maximum(a, a)`` reports a convention
    rather than a slope, and no finite difference can agree with it.
    """
    build = _probe.as_f32 if name in _F32_PARAMS else _probe.as_f64
    return build(_probe.sample(domain, shape, variant))


def _factorization(kind: str, order: int) -> tuple[Any, Any] | None:
    """A **real** ``(factor, pivots)`` pair from the framework's own factoriser.

    ``lu_solve`` and ``ldl_solve`` document that they take a
    factorization, and the derivation was handing them a draw of random
    numbers with a plausible pivot vector beside it.  LAPACK is entitled
    to assume the structure it was promised — ``dsytrs`` walks the
    triangle and the pivot sequence together — and a matrix that is not
    one sends it off the end of the buffer: the sweep died inside
    ``ldl_solve``'s backward pass roughly one run in three, always
    somewhere else, always later.

    Factorising a well-conditioned symmetric matrix costs one call and
    removes the whole class.  Seeded, so a finding stays reproducible.
    """
    matrix = _probe.rng(_probe.SEED_X).standard_normal((order, order))
    matrix = matrix + matrix.T + np.eye(order) * (order + 2.0)
    factorise = {
        "LU": getattr(lucid.linalg, "lu_factor", None),
        "LD": getattr(lucid.linalg, "ldl_factor", None),
    }.get(kind)
    if factorise is None:
        return None
    try:
        factor, pivots = factorise(_probe.as_f64(matrix))
    except Exception:  # noqa: BLE001 - surveying, not asserting
        return None
    return factor, pivots


class _AuditToyDataset:
    """Six samples, for the loader helpers that take a dataset."""

    def __len__(self) -> int:
        return 6

    def __getitem__(self, index: int) -> tuple[Any, int]:
        if index >= 6:
            raise IndexError(index)
        return _probe.as_f32(np.full((2,), float(index))), index % 2


def _toy_dataset() -> Any:
    return _AuditToyDataset()


def _indices(shape: tuple[int, ...], high: int) -> Any:
    return _probe.as_int(_probe.rng(_probe.SEED_B).integers(0, high, shape))


def _mask(shape: tuple[int, ...]) -> Any:
    """An all-true boolean mask — masking everything out hides the op."""
    return lucid.tensor(np.ones(shape, dtype=bool), dtype=lucid.bool)


def _clean(annotation: Any) -> str:
    """The annotation as comparable text, whatever format it arrived in."""
    if annotation is inspect.Parameter.empty:
        return ""
    return str(annotation).replace("'", "").replace('"', "")


def _is_tensor_annotation(text: str) -> bool:
    if not re.search(r"\bTensor\b|\bTensorOrScalar\b|\b_ArrayOrScalar\b", text):
        return False
    return not re.search(r"\b(list|Sequence|tuple|Iterable)\b", text)


def _is_tensor_sequence(text: str) -> bool:
    # A ``Callable`` that *returns* tensors is not a sequence of them.
    #
    # ``Callable[..., Tensor | tuple[Tensor, ...]]`` contains both
    # ``Tensor`` and ``tuple``, so ``func`` — the first parameter of
    # every higher-order function in the framework — was filled with a
    # list and then called: ``TypeError: 'list' object is not callable``,
    # for ``vjp``, ``jvp``, ``linearize``, ``gradcheck``,
    # ``gradgradcheck``, ``hessian`` and ``checkpoint`` alike.  The
    # return annotation describes what comes out, not what goes in.
    if re.search(r"\bCallable\b", text):
        return False
    return bool(re.search(r"\bTensor\b", text)) and bool(
        re.search(r"\b(list|Sequence|tuple|Iterable)\b", text)
    )


def _names_only_tuple(text: str) -> bool:
    """Whether a sequence annotation allows a tuple and no other container."""
    return bool(re.search(r"\btuple\b", text)) and not re.search(
        r"\b(list|Sequence|Iterable)\b", text
    )


#: Parameters that want a live module.  ``fuse_conv_bn_eval`` needs a
#: convolution and a norm layer specifically, so the name picks the class
#: rather than everything getting a ``Linear``.
_MODULE_BY_NAME: dict[str, Any] = {
    # ``fuse_modules(model, [["0", "1"]])`` names children by index, so
    # the model it is handed has to *have* children — a bare ``Linear``
    # answered "'Linear' has no attribute '0'".
    "model": lambda: lucid.nn.Sequential(lucid.nn.Linear(_COUT, _COUT), lucid.nn.ReLU()),
    "conv": lambda: lucid.nn.Conv2d(_CIN, _COUT, 3, padding=1),
    "bn": lambda: lucid.nn.BatchNorm2d(_COUT),
    "norm": lambda: lucid.nn.BatchNorm2d(_COUT),
    "linear": lambda: lucid.nn.Linear(_COUT, _COUT),
}


def _packed_sequence(domain: str) -> Any:
    """A real ``PackedSequence`` — the only thing ``pad_packed_sequence`` takes."""
    try:
        return lucid.nn.utils.rnn.pack_sequence(
            [_tensor((3, _COUT), domain), _tensor((2, _COUT), domain)], enforce_sorted=True
        )
    except Exception:  # noqa: BLE001 - surveying, not asserting
        return None


def _module_for(name: str) -> Any:
    """A small module for a parameter that takes one."""
    build = _MODULE_BY_NAME.get(name)
    if build is not None:
        return build()
    return lucid.nn.Linear(_COUT, _COUT)


#: Parameters that take a module, by name.  ``prune`` and the fusion
#: helpers are the users; without these they were the only group of
#: ``nn.utils`` left on the ladder.
_MODULE_PARAMS = frozenset(
    {"module", "mod", "model", "source", "dest", "conv", "bn", "norm", "linear"}
)

#: Parameters that take a sequence of tensors.  ``cat`` and ``stack``
#: annotate theirs as bare ``tensors`` with no annotation at all, so the
#: name is the only signal there is.
_TENSOR_SEQUENCE_PARAMS = frozenset(
    {"tensors", "sequences", "arrays", "inputs", "seq", "primals", "tangents"}
)


#: Parameter names that mean "a place on disk" rather than "a word".  The
#: annotation still has to look like a path for these to apply — ``f`` is
#: also the conventional name for a *function*, and handing ``lucid.func``
#: a filename would be a worse guess than the one this replaces.
_PATH_NAMES: frozenset[str] = frozenset(
    {
        "f",
        "fp",
        "file",
        "filename",
        "filepath",
        "path",
        "dst",
        "src",
        "dir",
        "folder",
        "directory",
        "save_dir",
        "out_dir",
        "ckpt_dir",
        "checkpoint",
    }
)

_SCRATCH: tempfile.TemporaryDirectory[str] | None = None


def _scratch_dir() -> Path:
    """A process-lifetime directory for probes that write to disk.

    Created once, removed at interpreter exit.  The sweep must leave the
    checkout exactly as it found it, and a derived path is the one place
    a probe can write without meaning to.
    """
    global _SCRATCH
    if _SCRATCH is None:
        _SCRATCH = tempfile.TemporaryDirectory(prefix="lucid-audit-")
        atexit.register(_SCRATCH.cleanup)
    return Path(_SCRATCH.name)


def _is_path_param(name: str, text: str) -> bool:
    """Whether this parameter names a location rather than a value."""
    if re.search(r"PathLike|IOBase", text):
        return True
    return name in _PATH_NAMES and bool(re.search(r"\bstr\b|\bbytes\b", text))


def _scratch_path(op_name: str, name: str) -> str:
    """A concrete path for ``op_name``'s ``name`` parameter.

    Every ``str`` parameter used to fall through to the generic ``"mean"``,
    which for a *path* meant the probe wrote a file called ``mean`` into
    the working directory — the sweep's one unexplained artefact, and the
    reason ``smoke::lucid.load`` was flaky: ``load("mean")`` succeeded
    exactly when a previous run's ``save("mean")`` had left the file
    behind, so the cell passed on a dirty checkout and skipped on a clean
    one.

    Readers are pointed at a file that exists, writers at a fresh name in
    the same directory, so a save/load pair is answerable in either order
    instead of depending on one having run first.
    """
    folder = _scratch_dir()
    if re.search(r"load|read|open|restore|from_file", op_name):
        seeded = folder / "seeded.checkpoint"
        if not seeded.exists():
            try:
                lucid.save(lucid.zeros((2, 2)), str(seeded))
            except Exception:  # noqa: BLE001 - a probe input, not a check
                return str(folder / f"{op_name}.{name}")
        return str(seeded)
    return str(folder / f"{op_name}.{name}")


def _literal_choice(text: str) -> str | None:
    """The first option of a ``Literal[...]`` annotation, if it is one."""
    match = re.search(r"Literal\[([^\]]+)\]", text)
    if match is None:
        return None
    first = match.group(1).split(",")[0].strip()
    return first or None


#: Names that mean "the tensor being operated on", beyond the ones the
#: annotation already identifies.  ``self`` is here because a Tensor
#: method arrives unbound and its receiver is an ordinary parameter.
#: Parameter names that are an operand whatever the annotation says.
#:
#: The second and third arguments of the binary and ternary free
#: functions.  Several carry no annotation, and Tensor methods spell the
#: receiver ``Self``, so the annotation-driven rules cannot see them; the
#: name is the only evidence there is.
_OPERAND_NAMES = frozenset({"other", "input", "mat1", "mat2", "condition", "reps", "base_impl"})

_TENSOR_PARAMS = frozenset(
    {"input", "x", "tensor", "a", "b", "self", "real", "imag", "abs", "angle"}
)

#: Values keyed on the parameter *name*, tried before the annotation.
#: ``int`` describes ``dim``, ``groups``, ``num_classes`` and
#: ``kernel_size`` equally badly.
_BY_NAME: dict[str, Any] = {
    "dim": -1,
    "M": 8,
    "nonlinearity": "relu",
    "clip_value": 1.0,
    "max_norm": 1.0,
    "cx": 0.0,
    "cy": 0.0,
    "axis": -1,
    "dim0": 0,
    "dim1": 1,
    "axis0": 0,
    "axis1": 1,
    "dims": (0, 1),
    "axes": (0, 1),
    "keepdim": False,
    "keepdims": False,
    "eps": 1e-5,
    "epsilon": 1e-5,
    "groups": 1,
    "num_groups": 1,
    "num_classes": _COUT,
    "num_embeddings": 8,
    "embedding_dim": _COUT,
    "num_heads": 2,
    "kernel_size": 3,
    "stride": 1,
    "padding": 0,
    "dilation": 1,
    "output_size": 2,
    "num_samples": 2,
    "p": 0.5,
    "q": 2,
    "n": 2,
    "k": 1,
    "diagonal": 0,
    "repeats": 2,
    "chunks": 2,
    "sections": 2,
    "split_size": 2,
    # An int every axis divides into.  The annotation fallback gave the
    # sequence ``(2, 2)``, which sums to the axis of a 1-D probe only; since
    # LCD-490 split refuses sizes that do not, and the first call left to
    # run was rank 1, so the rank axis had nothing to take away.
    "split_size_or_sections": 1,
    "start": 0,
    "end": 4,
    "step": 1,
    "min": 0.0,
    "max": 1.0,
    "alpha": 1.0,
    "beta": 1.0,
    "gamma": 1.0,
    "lambd": 0.5,
    "margin": 1.0,
    "reduction": "mean",
    "mode": "constant",
    "value": 0.0,
    "shape": (_N, _COUT),
    "size": (_N, _COUT),
    "shift": 1,
    "shifts": 1,
    "decimals": 2,
    "rounding_mode": None,
    "generator": None,
    "out": None,
    # normalisation and the geometric transforms, whose required
    # parameters are plain tuples the annotation describes only as
    # ``tuple[int, int]`` — which the fallback fills with ``(2, 2)``,
    # a valid pair and the wrong one for an image extent.
    "normalized_shape": (_COUT,),
    "mean": (0.5, 0.5, 0.5),
    "std": (0.5, 0.5, 0.5),
    "in_hw": (_L, _L),
    "out_hw": (_L, _L),
    "canvas_hw": (_L, _L),
    "tile_grid_size": (2, 2),
    "kernel2d": [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]],
    "src": [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
    "dst": [[0.0, 0.0], [1.0, 0.1], [0.9, 1.0], [0.1, 0.9]],
    # ``pack_padded_sequence(input, lengths)`` reads ``input`` as
    # ``(T, B, *)`` and every length has to be ``<= T``.  Against the
    # ``(2, 4)`` probe the list ``[3, 2]`` asks for three timesteps out
    # of two, and the engine answered with ``item()``'s error rather
    # than its own.
    "lengths": [3, 2],
    "argnums": 0,
    "in_dims": 0,
    "tensor_name": "weight",
    # The ODE solvers: an initial state and a time grid.  ``y0`` and
    # ``t`` carry no annotation the rules can read, so all four
    # ``odeint`` entry points reported "missing 2 required positional
    # arguments" while their signature said plainly what they take.
    "t0": 0.0,
    "t1": 1.0,
    "rtol": 1e-6,
    "atol": 1e-8,
}

#: Names whose value must agree with a tensor already chosen.  The only
#: semantic knowledge here, and the reason a derived spec can reach
#: ``conv2d`` and ``cross_entropy`` at all.
_RELATED = (
    "weight",
    "bias",
    "target",
    "labels",
    "other",
    "mat2",
    "tensor2",
    "key",
    "value",
    "mask",
    "attn_mask",
    "index",
    "indices",
    "running_mean",
    "running_var",
    # ``pivots`` had a rule in :func:`_related_value` and was missing from
    # this tuple, which is the only thing that reaches it — so the branch
    # was unreachable and ``lu_solve`` and ``ldl_solve`` were handed a
    # float tensor for an integer pivot vector on every candidate,
    # refused it every time, and answered none of their 22 cells.  A
    # dispatch table and its dispatch list are one fact written twice.
    "pivots",
    "B",
    "matrix",
    "pts",
    "dx",
    "dy",
    "field",
)


class _Plan:
    """The arguments chosen for one derived invocation."""

    __slots__ = ("args", "kwargs", "primary", "unknown", "note")

    def __init__(self) -> None:
        self.args: list[Any] = []
        self.kwargs: dict[str, Any] = {}
        self.primary: int = 0
        self.unknown: list[str] = []
        self.note: str = ""


def _related_value(
    name: str, text: str, reference: Any, op_name: str, domain: str
) -> tuple[bool, Any]:
    """A value for ``name`` that agrees with the tensor already chosen."""
    if reference is None:
        return False, None
    shape = tuple(np.shape(reference))
    if not shape:
        return False, None

    if name in ("other", "mat2", "tensor2"):
        return True, _tensor(shape, domain, variant=1)
    if name in ("key", "value"):
        return True, _tensor(shape, domain, variant=1 if name == "key" else 2)
    if name in ("mask", "attn_mask"):
        return True, _mask(shape)
    if name in ("index", "indices"):
        return True, _indices(shape, max(shape[-1], 1))
    if name == "pivots":
        # Annotated plainly as ``Tensor``, so nothing but the name says
        # this is an index vector.  Filling it as a float tensor is what
        # found the ``lu_solve`` bus error — and then killed the sweep at
        # the same symbol every run, which is the reason it matters here
        # rather than only in the framework.  LAPACK pivots are 1-based
        # over the order of the factorization, and int32 wide.
        #
        # One row per batch element, not one vector for the batch.  A
        # flat ``(order,)`` against a batched ``LU`` is accepted, runs,
        # returns plausible numbers — and reads pivots past the end of
        # the buffer for every element after the first.  LAPACK's
        # ``dgetrs`` then swaps rows of the right-hand side by those
        # garbage indices, which writes outside it: the process kept
        # going and died later, somewhere else, with a corrupted
        # ObjC method cache.  Recorded in ``known.json``; the probe has
        # to be right regardless, or the sweep cannot finish.
        order = max(shape[-2] if len(shape) >= 2 else shape[0], 1)
        # The pivots that belong to the factor the primary was built
        # from, where there is one.  A pivot sequence is not independent
        # of the triangle it permutes.
        pair = _factorization("LD" if re.search(r"ldl", op_name) else "LU", order)
        if pair is not None and not shape[:-2]:
            return True, pair[1]
        batch = tuple(shape[:-2])
        return True, _indices((*batch, order), order).to(lucid.int32) + 1
    if name in ("running_mean", "running_var"):
        channels = shape[1] if len(shape) > 1 else shape[0]
        fill = 0.0 if name == "running_mean" else 1.0
        return True, _probe.as_f64(np.full((channels,), fill))

    if name == "B":
        # The right-hand side of a triangular solve, square with the
        # factorization it is solved against.
        order = shape[-1]
        return True, _tensor((*shape[:-2], order, 1) if len(shape) >= 2 else (order, 1), domain)
    if name == "matrix":
        # A 3x3 *homogeneous* affine, not the 2x3 the shorthand suggests.
        # Every one of these transforms inverts the matrix internally and
        # refuses a non-square one — ``inv.a: last two dims must be equal``
        # — so the 2x3 form reached the op and was rejected by it, which
        # reads as a defect and is a probe that handed over the wrong
        # convention.  Identity plus a small shift: invertible, and not
        # the degenerate map a random draw produces about one time in six.
        return True, _probe.as_f32(np.array([[1.0, 0.0, 0.1], [0.0, 1.0, -0.1], [0.0, 0.0, 1.0]]))
    if name == "pts":
        return True, _probe.as_f32(np.array([[1.0, 2.0], [3.0, 4.0], [0.5, 1.5]]))
    if name in ("dx", "dy"):
        # A displacement field the size of the image it resamples.
        spatial = shape[-2:] if len(shape) >= 2 else shape
        return True, _tensor(tuple(spatial), "unit", variant=1 if name == "dx" else 2, name=name)
    if name == "field":
        # ``(H, W)``: a *scalar* field, as the docstring says.  Read as
        # ``(2, H, W)`` it reached ``grid_sample`` with twice the values
        # it expected and answered "reshape: total numel mismatch".
        return True, _tensor((_L, _L), domain, name=name)

    if name in ("target", "labels"):
        # A class-index target is bounded by the width of the logits; a
        # regression target has the logits' shape.  Getting this wrong is
        # not a soft failure — an out-of-range index reads past the table.
        if re.search(r"cross_entropy|nll|multi_margin|_class", op_name):
            return True, _indices(shape[:1] or (1,), max(shape[-1], 1))
        # A regression target drawn the same way as the prediction *is*
        # the prediction — the draw is seeded — so every such loss was
        # evaluated at its own minimum, where the loss is 0 and so is its
        # gradient.  The check then passed by measuring nothing, which is
        # what VACUOUS exists to say and what the grad axis reported for
        # the whole loss family.
        return True, _tensor(shape, domain, variant=1)

    if name == "weight":
        if re.search(r"conv", op_name):
            rank = _rank_from(op_name, len(shape))
            spatial = (3,) * rank
            return True, _tensor((_COUT, shape[1] if len(shape) > 1 else 1, *spatial), domain)
        if re.search(r"linear|bilinear", op_name):
            return True, _tensor((_COUT, shape[-1]), domain)
        if re.search(r"embedding", op_name):
            return True, _tensor((8, _COUT), domain)
        # A per-element or per-channel weight, as norms and losses use.
        return True, _tensor((shape[-1],), domain)
    if name == "bias":
        return True, _tensor((_COUT,), domain)
    return False, None


def _rank_from(op_name: str, tensor_rank: int) -> int:
    match = re.search(r"([123])d\b", op_name)
    if match is not None:
        return int(match.group(1))
    return max(tensor_rank - 2, 1)


#: What the *first parameter's name* says the probe should look like.
#: A geometric transform takes an image, not a matrix of unrelated
#: numbers, and ``(2, 4)`` is neither — the whole of
#: ``utils.transforms.functional`` was probed with it and answered
#: ShapeMismatch, which reads as a defect in nine working functions.
_SHAPE_BY_FIRST_PARAM: dict[str, tuple[tuple[int, ...], ...]] = {
    "img": ((_CIN, _L, _L), (_N, _CIN, _L, _L)),
    "image": ((_CIN, _L, _L), (_N, _CIN, _L, _L)),
    "pts": ((3, 2),),
    "points": ((3, 2),),
    "matrix": ((3, 3),),
    "field": ((_L, _L),),
}


def _first_parameter(fn: Any) -> str:
    try:
        signature = inspect.signature(fn, annotation_format=annotationlib.Format.STRING)
    except (TypeError, ValueError, NameError):
        return ""
    return next(iter(signature.parameters), "")


#: Ops defined only on a single element.  ``Tensor.item`` says so in its
#: own error — "item() can only be called on a 1-element tensor" — and the
#: probe handed it ``(2, 4)`` on every axis, so a method whose contract is
#: one line went unchecked on all ten of them.
_SINGLE_ELEMENT_OPS = frozenset({"item", "__index__", "__float__", "__int__"})


def _tensor_shapes(op_name: str, fn: Any = None) -> tuple[tuple[int, ...], ...]:
    """Candidate shapes for the first tensor, most likely first."""
    if op_name in _SINGLE_ELEMENT_OPS:
        return ((1,), (1, 1))
    if re.search(r"pack_padded|pad_packed|pack_sequence", op_name):
        # ``(T, B, *)`` — a padded batch, which is the only thing the
        # packing helpers read.  Against a 2-D probe the time axis is
        # missing and the length list indexes past the end of the shape.
        return ((3, _N, _COUT),)
    if fn is not None:
        by_name = _SHAPE_BY_FIRST_PARAM.get(_first_parameter(fn))
        if by_name is not None:
            return by_name
    match = re.search(r"([123])d\b", op_name)
    if match is not None:
        rank = int(match.group(1))
        spatial = {1: (_L,), 2: (_H, _W), 3: (4, _H, _W)}[rank]
        return ((_N, _CIN, *spatial),)
    if re.search(r"matmul|bmm|mm\b|matrix|linalg|inv|det|solve|cholesky|qr|svd", op_name):
        return ((_N, _COUT, _COUT), (_COUT, _COUT))
    return ((_N, _COUT), (_N, _CIN, _H, _W), (_COUT,))


def _value_for(
    param: inspect.Parameter,
    op_name: str,
    domain: str,
    reference: Any,
    shape: tuple[int, ...],
    variant: int = 0,
) -> tuple[bool, Any, bool]:
    """``(found, value, is_tensor)`` for one required parameter.

    ``variant`` is how many tensor arguments have already been built, so
    each operand of a binary op gets a different fixed draw.  Given the
    same one they were *identical*, and every comparison was probed
    exactly on its tie.
    """
    name = param.name
    text = _clean(param.annotation)

    # An unambiguous scalar annotation outranks every name rule below.
    #
    # ``condition`` is in :data:`_OPERAND_NAMES` because ``lucid.where``
    # spells its predicate that way, and ``unsupported_if(condition:
    # bool, ...)`` spells a plain bool the same — so it was handed a 2x4
    # tensor and answered ``RuntimeError: Boolean value of Tensor is
    # ambiguous`` on all eleven axes.  A name is evidence; an annotation
    # that admits exactly one type is proof.
    if re.fullmatch(r"bool|int|float|str", text.strip()) and name not in _BY_NAME:
        return (True, {"bool": True, "int": 2, "float": 0.5, "str": "mean"}[text.strip()], False)

    if name in _RELATED:
        found, value = _related_value(name, text, reference, op_name, domain)
        if found:
            return True, value, _probe.to_numpy(value) is not None

    if name in _MODULE_PARAMS and not _is_tensor_annotation(text):
        return True, _module_for(name), False
    # The ODE solvers.  ``t`` is annotated ``Tensor | Sequence[float]``,
    # which reads as a sequence *of tensors* to the rule below and gave
    # ``odeint`` a list of probes where it wanted a time grid.
    # The quantisation helpers.  Their required arguments are a scale, a
    # zero point, a scheme and a dtype — none of which any annotation
    # describes ("_ScaleLike", "_ZeroPointLike", "QDtype") and none of
    # which the ladder can invent, so eight public functions reported
    # "no argument shape worked" on the smoke floor while the quant axis
    # checked them properly one axis over.
    if name in ("scale", "scales"):
        return True, _probe.as_f32(np.array(0.02)), False
    if name in ("zero_point", "biases"):
        return True, _probe.as_int(np.array(0)), False
    if name == "qdtype" or name == "dtype" and re.search(r"quant", op_name):
        return True, lucid.quantization.quint8, False
    if name == "qscheme":
        return True, lucid.quantization.QScheme.PER_TENSOR_AFFINE, False
    if name == "min_val":
        return True, _probe.as_f32(np.array(-2.0)), False
    if name == "max_val":
        return True, _probe.as_f32(np.array(2.0)), False
    if name == "quant_min":
        return True, 0, False
    if name == "quant_max":
        return True, 255, False
    if name == "modules_to_fuse":
        return True, [["0", "1"]], False

    if name == "vec":
        # As many values as the parameters it is about to overwrite.
        # ``Linear(4, 4)`` holds 20, and a ``(2, 4)`` probe made
        # ``vector_to_parameters`` refuse a vector of the wrong length —
        # correctly, and on all ten axes.
        # ...and at the parameters' dtype.  A freshly built layer is
        # float32 and ``copy_from`` refuses the sweep's float64.
        total = sum(
            int(np.prod(tuple(t.shape))) for t in lucid.nn.Linear(_COUT, _COUT).parameters()
        )
        return True, _probe.as_f32(_probe.sample(domain, (total,))), True
    # The remaining named arguments the ladder cannot invent.  Each is
    # one entry and each was costing its symbol every cell it had.
    if name in ("p", "q") and re.search(r"kl_divergence", op_name):
        return True, lucid.distributions.Normal(0.0, 1.0), False
    if name == "dataset":
        return True, _toy_dataset(), False
    if name == "v" and re.search(r"vjp", op_name):
        # ``autograd.vjp(func, inputs, v)`` — the cotangent has the shape
        # of the *output*, and the probe function reduces to a scalar.
        # Built directly: ``_probe.as_f64`` goes through
        # ``np.ascontiguousarray``, which turns a 0-d array into shape
        # ``(1,)``, and ``vjp`` refuses a seed of another shape (LCD-367).
        return True, lucid.tensor(1.0, dtype=lucid.float64), False
    if name == "v" and re.fullmatch(r"jvp|hvp|vhp", op_name):
        # ``jvp`` / ``hvp`` / ``vhp`` take ``v`` in the *input* space: a
        # tensor shaped like each of ``inputs``, a different draw from it.
        if not isinstance(reference, list | tuple):
            return True, _tensor(shape, domain, variant + 1), True
        tangents = [
            _tensor(tuple(t.shape), domain, variant + i + 1) for i, t in enumerate(reference)
        ]
        # In the container ``inputs`` came in: a tuple of inputs takes a
        # tuple of tangents.
        if isinstance(reference, tuple):
            return True, tuple(tangents), True
        return True, tangents, True
    if name == "name" and re.search(r"weight|param", op_name):
        return True, "weight", False
    if name == "lengths":
        # ``Tensor | list[int]`` reads as a sequence of tensors to the
        # rule below, so the packing helpers were handed two float
        # probes where they wanted two integers.
        #
        # ``random_split`` spells its partition sizes the same way and
        # requires them to *sum to the dataset length*, which the
        # sequence-length reading cannot satisfy.
        if re.search(r"random_split|split", op_name):
            return True, [4, 2], False
        return True, [int(shape[0]), max(int(shape[0]) - 1, 1)], False
    if name == "y0":
        return True, _tensor((2,), "positive"), True
    if name in ("t", "t_span") and re.search(r"odeint|solve_ivp", op_name):
        return True, [0.0, 0.5, 1.0], False
    if name == "event_fn":
        return True, (lambda _t, y: y.reshape(-1)[0] - 1e9), False
    if name in _TENSOR_SEQUENCE_PARAMS or _is_tensor_sequence(text):
        # Marked differentiable: for ``cat`` and ``stack`` the sequence is
        # the operand, and the generic ladder's ``op([x, x])`` form treats
        # it the same way.  Left un-marked, ``_build`` finds no tensor to
        # differentiate and discards a perfectly good invocation.
        pair = (_tensor(shape, domain, variant), _tensor(shape, domain, variant=1))
        if re.search(r"gradcheck|gradgradcheck", op_name):
            # These differentiate what they are given, so what they are
            # given has to be differentiable — "an input has no gradient
            # after backward()" is the check correctly refusing a leaf
            # nobody asked to track.
            for operand in pair:
                operand.requires_grad_(True)
        # The container the annotation names.  ``inputs: Tensor |
        # tuple[Tensor, ...]`` — ``hessian``, ``jacobian``, ``vjp`` —
        # refuses a list, as the reference does, and the list this built
        # cost each its smoke cell.
        return True, (pair if _names_only_tuple(text) else list(pair)), True
    if name == "parameters" or re.search(r"Iterable\[Parameter\]", text):
        return True, list(lucid.nn.Linear(_COUT, _COUT).parameters()), False
    if re.search(r"\bCallable\b", text) and re.search(r"odeint|solve_ivp", op_name):
        # A right-hand side, not a loss: ``dy/dt`` has to have ``y``'s
        # shape, and the scalar probe below made every solver report
        # "func returned shape () but y0 has shape (2,)".
        return True, (lambda _t, y: y * 0.5), False
    if re.search(r"\bCallable\b", text):
        # ``gradcheck``, ``jacrev``, ``vmap`` — the function under test is
        # the argument.  Squaring is differentiable twice and has a second
        # derivative that is not zero, so it exercises what they compute.
        #
        # Reduced to a scalar, because ``gradcheck`` and ``hessian``
        # require one and say so — "gradcheck requires a scalar-valued
        # function" — while everything else here is happy either way.
        # Every operand, not just the first.  ``gradcheck`` is handed a
        # *sequence* of inputs and requires each to receive a gradient —
        # a probe function that reads only ``operands[0]`` makes it
        # report "an input has no gradient after backward()", which is
        # the check working correctly on a function that ignores half
        # its arguments.
        return (True, (lambda *operands: sum((x * x).sum() for x in operands)), False)
    if name in ("LU", "LD") and len(shape) == 2:
        pair = _factorization(name, shape[-1])
        if pair is not None:
            return True, pair[0], True

    if _is_tensor_annotation(text) or name in _TENSOR_PARAMS:
        # An op that reads a table wants the *row numbers*, and the
        # annotation says ``Tensor`` for both.  ``check_embedding_indices``
        # is named after what it takes and was still handed floats.
        if variant == 0 and re.search(r"embedding_indices", op_name):
            return True, _indices(shape, 8), True
        return True, _tensor(shape, domain, variant, name=name), True
    # ``Self`` on a Tensor method, and the operand names the free
    # functions use.  ``lucid.atan2(input, other)`` carries no annotation
    # at all and ``Tensor.addmm(self, mat1, mat2)`` spells its operands
    # ``Self``; neither reads as a tensor to the rules above, and one
    # unresolvable parameter discards the whole symbol.
    if re.fullmatch(r"Self|Tensor", text.strip()) or name in _OPERAND_NAMES:
        return True, _tensor(shape, domain, variant), True
    if name in ("t", "obj") and text in ("object", "", "Any"):
        # ``is_complex(t)``, ``is_floating_point(t)``, ``is_tensor(obj)``
        # — a predicate over a tensor, annotated as widely as possible.
        return True, _tensor(shape, domain, variant), True
    if re.search(r"_DType\b", text) or name.endswith("_dtype"):
        return True, lucid.float32, False
    if re.search(r"TensorImpl", text):
        # An engine-level entry point: it wants the impl, not the wrapper.
        return True, _tensor(shape, domain, variant)._impl, False
    if name in ("p_cls", "q_cls"):
        # ``register_kl(p_cls, q_cls)`` refuses a class that is not a
        # Distribution, as the reference does; the ``type`` rule below
        # handed it ``nn.Linear``.  Calling it only builds the decorator,
        # so nothing is registered.
        return True, lucid.distributions.Normal, False
    if name == "module_cls" or re.search(r"^type$", text):
        return True, lucid.nn.Linear, False
    if re.search(r"PackedSequence", text):
        packed = _packed_sequence(domain)
        if packed is not None:
            return True, packed, False
    if name in ("data", "arr", "ext_tensor", "array"):
        # The bridge functions — ``tensor``, ``as_tensor``, ``from_numpy``.
        # Annotated ``object`` or ``np.ndarray``, so only the name says
        # what they want, and what they want is not a Lucid tensor.
        return True, np.asarray(_probe.sample(domain, shape, variant)), True

    if name in _BY_NAME:
        return True, _BY_NAME[name], False

    choice = _literal_choice(text)
    if choice is not None:
        return True, choice, False

    if re.search(r"\bbool\b", text):
        return True, False, False
    if re.search(r"tuple\[int|Sequence\[int\]|list\[int\]|_IntTuple|_Shape", text):
        return True, (2, 2), False
    if re.search(r"\bint\b", text):
        return True, 2, False
    if re.search(r"\bfloat\b", text):
        return True, 0.5, False
    if _is_path_param(name, text):
        return True, _scratch_path(op_name, name), False
    if re.search(r"\bstr\b", text):
        return True, "mean", False
    if re.search(r"\bdtype\b|Dtype", text):
        return True, lucid.float32, False
    if re.search(r"\bdevice\b|Device", text):
        return True, "cpu", False
    if "None" in text:
        # ``T | None`` with no default still accepts None, and a None is a
        # truthful "not supplied" rather than a guess.
        return True, None, False
    return False, None, False


def _build(fn: Any, op_name: str, domain: str, shape: tuple[int, ...]) -> _Plan | None:
    try:
        signature = inspect.signature(fn, annotation_format=annotationlib.Format.STRING)
    except (TypeError, ValueError, NameError):
        return None

    plan = _Plan()
    first_tensor: int | None = None
    reference: Any = None
    tensors_built = 0
    for name, param in signature.parameters.items():
        # ``self`` is *not* skipped.  A Tensor method is enumerated with
        # ``getattr_static`` and so arrives unbound — the receiver is a
        # real positional parameter and has to be supplied, or 68 of the
        # 253 methods look like they take an argument nothing can fill.
        if name == "cls":
            continue
        if param.kind is param.VAR_KEYWORD:
            continue
        if param.kind is param.VAR_POSITIONAL:
            # ``f(*tensors)`` — two of the primary shape is the reading
            # that makes ``stack`` and ``cat`` mean something.  The draws
            # continue after any tensor already built: a method such as
            # ``Tensor.clamp(self, *args)`` has its receiver built first,
            # and starting again at variant 0 handed it ``x.clamp(x, y)`` —
            # every element exactly on its lower bound, the one input where
            # the derivative is a convention, so the gradient axes reported
            # a corner and a missing double backward (e2420050d let tensor
            # bounds through, which made this the first call that ran).
            plan.args.append(_tensor(shape, domain, variant=tensors_built))
            plan.args.append(_tensor(shape, domain, variant=tensors_built + 1))
            tensors_built += 2
            if first_tensor is None:
                first_tensor, reference = len(plan.args) - 2, plan.args[-2]
            continue
        if param.default is not param.empty:
            continue

        found, value, is_tensor = _value_for(
            param, op_name, domain, reference, shape, variant=tensors_built
        )
        if not found:
            plan.unknown.append(f"{name}: {_clean(param.annotation) or '<no annotation>'}")
            return None
        if param.kind is param.KEYWORD_ONLY:
            plan.kwargs[name] = value
            if is_tensor:
                tensors_built += 1
        else:
            plan.args.append(value)
            if is_tensor:
                tensors_built += 1
                if first_tensor is None:
                    first_tensor, reference = len(plan.args) - 1, value

    if first_tensor is None and not signature.parameters:
        # A zero-argument query — ``get_default_dtype``,
        # ``is_grad_enabled``, ``initial_seed``.  ``_build`` discarded
        # these for having no operand, which left thirteen symbols
        # reported as unreachable when calling them is the entire test.
        plan.primary = -1
        plan.note = "derived from signature(), no arguments"
        return plan

    if first_tensor is None:
        # No tensor *input* is not the same as nothing to check.
        # ``windows.bartlett(8)`` builds a tensor from an int,
        # ``calculate_gain("relu")`` returns a float, ``fuse_conv_bn_eval``
        # takes two modules — all callable, all worth running, none of
        # them differentiable with respect to an argument.  Discarding
        # these left 35 symbols on the ladder for want of an operand they
        # do not have.  The gradient axes ask ``Call.base`` for a tensor,
        # get a TypeError and skip, which is the truthful outcome; the
        # smoke, edge and dtype axes still get their answer.
        if not plan.args and not plan.kwargs:
            return None
        plan.primary = 0 if plan.args else -1
        plan.note = f"derived from signature{signature}, no tensor operand"
        return plan
    plan.primary = first_tensor
    plan.note = f"derived from signature{signature} at {shape}"
    return plan


def invocations(fn: Any, op_name: str, domain: str) -> Iterator[Call]:
    """Candidate calls for ``fn``, derived from its signature.

    Yields nothing when the signature cannot be read or leaves a required
    parameter this module has no value for — that is the honest answer,
    and :mod:`_specs` falls through to its generic ladder behind it.

    Parameters
    ----------
    fn : callable
        The resolved callable, not the :class:`~lucid.test.audit.
        _surface.Symbol`.
    op_name : str
        Short name, used for the handful of shape decisions a signature
        cannot express — ``conv2d`` wants a 4-D input, ``cross_entropy``
        wants an integer target.
    domain : str
        Key into :data:`~lucid.test.audit._probe.DOMAINS`.
    """
    emitted = 0
    for shape in _tensor_shapes(op_name, fn):
        plan = _build(fn, op_name, domain, shape)
        if plan is None:
            continue
        yield Call(plan.args, plan.kwargs, plan.primary, plan.note)
        emitted += 1
        if emitted >= _MAX_CANDIDATES:
            return


def explain(fn: Any, op_name: str) -> str:
    """Why a symbol gets no derived invocation, for ``--list-uncovered``."""
    try:
        signature = inspect.signature(fn, annotation_format=annotationlib.Format.STRING)
    except (TypeError, ValueError, NameError) as exc:
        return f"unreadable signature: {type(exc).__name__}"
    for shape in _tensor_shapes(op_name, fn):
        plan = _Plan()
        probe = _build(fn, op_name, "moderate", shape)
        if probe is not None:
            return "derived"
        del plan
    required = [
        f"{p.name}: {_clean(p.annotation) or '<no annotation>'}"
        for p in signature.parameters.values()
        if p.default is p.empty and p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)
    ]
    return f"no value for {required}" if required else "no tensor parameter"


__all__ = ["explain", "invocations"]
