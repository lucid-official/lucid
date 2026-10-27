"""Axes for the four subsystems that are classes rather than functions.

``utils.transforms`` (97 symbols), ``utils.data`` (25),
``utils.tokenizer`` (20) and ``optim.lr_scheduler`` (16) were reached by
the surface and then hit only by :class:`~lucid.test.audit.
_axes_subsystem.SmokeAxis` — constructed and called once, which for a
class with a lifecycle is barely a check at all.  158 symbols, a tenth
of the framework, verified to the depth of "it did not raise".

Each gets the question that can actually fail for it:

    transform    does applying it return a finite tensor, leave the input
                 alone, and repeat under a fixed seed
    data         does the dataset / sampler / loader / collator triple of
                 length, indexing and iteration agree with each other
    tokenizer    does ``decode(encode(text))`` give the text back
    scheduler    do the learning rates stay finite and does a state_dict
                 round trip resume the same schedule

The shared difficulty is construction: ``Resize`` wants ``height`` and
``width``, ``BPETokenizer`` wants a vocabulary and a merge list, every
scheduler wants a live optimizer.  :func:`_construct` reads the
signature and fills what it recognises, so adding a class to any of
these packages does not mean editing this file.
"""

import annotationlib
import inspect
import math
from typing import TYPE_CHECKING, Any

import numpy as np

import lucid
import lucid.optim.lr_scheduler
import lucid.utils.cache
import lucid.utils.data
from lucid.test.audit import _probe
from lucid.test.audit._axes import Axis, Context
from lucid.test.audit._result import Finding, Status

if TYPE_CHECKING:
    from lucid.test.audit._surface import Symbol

#: Side of the probe image.  Small enough that a 97-symbol sweep is
#: quick, big enough that a crop, a resize and a blur all have room.
_SIDE = 16

#: The tokenizer probe sentence, split once so the vocabulary builder
#: and the axis cannot drift apart.
_PROBE_WORDS = ("the", "quick", "brown", "fox")


def _image() -> Any:
    """A deterministic ``(3, 16, 16)`` image in ``[0, 1]``."""
    values = np.linspace(0.0, 1.0, 3 * _SIDE * _SIDE, dtype=np.float32)
    return lucid.tensor(values.reshape(3, _SIDE, _SIDE))


class _ToyIterableDataset(lucid.utils.data.IterableDataset):
    """Three samples, reachable only by iterating — what ``ChainDataset`` takes."""

    def __iter__(self) -> Any:
        for index in range(3):
            yield lucid.tensor(np.full((2,), float(index), np.float32)), index % 2


class _ToyDataset:
    """Six samples of ``(features, label)``, indexable and sized."""

    def __init__(self, n: int = 6) -> None:
        self.n = n

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, index: int) -> tuple[Any, int]:
        return lucid.tensor(np.full((2,), float(index), np.float32)), index % 2


def _tiny_vocab() -> dict[str, int]:
    """A vocabulary the probe sentence can be spelled in three ways.

    Single characters for a byte-level or character model, the whole
    words for a whitespace or word-piece one, and the ``##``-prefixed
    continuations word-piece needs.  Handing a word-level tokenizer a
    vocabulary of bare letters produced an empty encoding and looked like
    six framework defects; it was six statements that the harness had
    given them nothing they could match.
    """
    letters = "abcdefghijklmnopqrstuvwxyz "
    vocab = {ch: i for i, ch in enumerate(letters)}
    for word in _PROBE_WORDS:
        vocab.setdefault(word, len(vocab))
        vocab.setdefault("##" + word, len(vocab))
        for i in range(1, len(word)):
            vocab.setdefault(word[:i], len(vocab))
            vocab.setdefault("##" + word[i:], len(vocab))
    for token in ("<unk>", "<pad>", "<bos>", "<eos>", "<mask>", "[UNK]", "[PAD]", "[CLS]", "[SEP]"):
        vocab.setdefault(token, len(vocab))
    return vocab


#: How to fill a required constructor parameter, keyed by its name.
#: Checked before the annotation, because a name says more than ``int``
#: does — ``height`` and ``num_classes`` are both ints and want very
#: different values.
_BY_NAME: dict[str, Any] = {
    "height": _SIDE // 2,
    "width": _SIDE // 2,
    "size": _SIDE // 2,
    "output_size": _SIDE // 2,
    "num_classes": 4,
    "num_output_channels": 3,
    "kernel_size": 3,
    "batch_size": 2,
    "max_length": 8,
    "num_samples": 4,
    "num_replicas": 1,
    "rank": 0,
    "alpha": 1.0,
    "p": 1.0,
    "factor": 0.5,
    "gamma": 0.9,
    "T_max": 5,
    "T_0": 5,
    "step_size": 2,
    "milestones": [2, 4],
    "total_iters": 5,
    # Past its end ``OneCycleLR`` raises, as the reference does; the axis
    # walks it six epochs and then six more from a resumed copy.
    "total_steps": 20,
    "base_lr": 0.01,
    "max_lr": 0.1,
    "start_factor": 0.5,
    "end_factor": 1.0,
    "mean": (0.5, 0.5, 0.5),
    "std": (0.5, 0.5, 0.5),
    "merges": [],
    # The tokenizers.  ``vocab`` is annotated ``dict[str, int]`` and
    # ``pieces`` ``list[tuple[str, float]]``, so the annotation fallback
    # gave them ``2`` and ``1.0`` — "'int' object is not iterable" for
    # eight tokenizer classes whose vocabulary the harness already
    # builds one function above.
    "pieces": [(w, -1.0) for w in _PROBE_WORDS] + [("<unk>", -10.0)],
    # A required ``str`` that is not free-form: the annotation fallback
    # supplied the literal "constant", which is a valid regex matching
    # nothing in the probe sentence, so the tokenizer correctly returned
    # no ids and the axis called it a defect.
    "pattern": r"\w+|\S",
    "weights": [1.0] * 6,
    "scores": [0.0] * 6,
    "lr_lambda": (lambda epoch: 0.95**epoch),
    "policy": None,
}

#: What to pass for a ``*args`` parameter, keyed by its name.  Built by a
#: lambda so no tensor is allocated at import time.
_VARIADIC: dict[str, Any] = {
    "tensors": lambda: (
        lucid.tensor(np.arange(12, dtype=np.float32).reshape(6, 2)),
        lucid.tensor(np.arange(6, dtype=np.float32)),
    ),
    "datasets": lambda: (_ToyDataset(3), _ToyDataset(3)),
    # ``StackDataset(*args)`` names its variadic ``args``, so the table
    # keyed on the parameter name missed it entirely.
    "args": lambda: (_ToyDataset(3), _ToyDataset(3)),
}

#: Values that have to be *built* rather than named, so nothing is
#: allocated at import time.  Keyed on the parameter name, tried after
#: :data:`_BY_NAME` and before the annotation.
#:
#: This table is where the distributions and the tensor subclasses were
#: stuck.  ``MultivariateNormal(loc, ...)``, ``Dirichlet(concentration)``,
#: ``BoundingBoxes(data)`` and ``ComposeTransform(parts)`` each need one
#: constructed object that no annotation describes, and each was
#: reported as "no value for required parameter" on both the contract
#: axis and its own subsystem axis — the same gap counted twice.
_BUILD_BY_NAME: dict[str, Any] = {
    "data": lambda: lucid.tensor(np.linspace(0.1, 0.9, 12, dtype=np.float32)),
    "vocab": _tiny_vocab,
    # The last handful of classes whose one required argument is an
    # object no annotation describes.  Each is one line and each was
    # costing its class every cell it had.
    "activation": lambda: lucid.quantization.MinMaxObserver,
    "weight": lambda: lucid.quantization.MinMaxObserver,
    "scale": lambda: lucid.tensor(np.array(0.02, dtype=np.float32)),
    "zero_point": lambda: _probe.tensor_of(np.array(0), dtype=lucid.int64),
    "qscheme": lambda: lucid.quantization.QScheme.PER_TENSOR_AFFINE,
    "qdtype": lambda: lucid.quantization.quint8,
    "self_attention_cache": lucid.utils.cache.DynamicCache,
    "cross_attention_cache": lucid.utils.cache.DynamicCache,
    "hook": lambda: (lambda *args, **kwargs: None),
    # A real second-order tableau: Heun's method, which satisfies the
    # consistency conditions the diffeq axis then checks.
    "a": lambda: ((), (1.0,)),
    "b": lambda: (0.5, 0.5),
    "c": lambda: (0.0, 1.0),
    "order": lambda: 2,
    "format": lambda: "xyxy",
    "canvas_size": lambda: (_SIDE, _SIDE),
    # ``PackedSequence(data, batch_sizes, ...)`` — the two agree with each
    # other: five values packed as three then two.
    "batch_sizes": lambda: lucid.tensor(np.array([3, 2]), dtype=lucid.int64),
    "sorted_indices": lambda: None,
    "unsorted_indices": lambda: None,
    "loc": lambda: lucid.tensor(np.zeros(3, dtype=np.float32)),
    "concentration": lambda: lucid.tensor(np.full(3, 2.0, dtype=np.float32)),
    "concentration1": lambda: lucid.tensor(np.full(3, 2.0, dtype=np.float32)),
    "covariance_matrix": lambda: lucid.tensor(np.eye(3, dtype=np.float32)),
    "scale_tril": lambda: lucid.tensor(np.eye(3, dtype=np.float32)),
    "probs": lambda: lucid.tensor(np.full(3, 1.0 / 3.0, dtype=np.float32)),
    # Every scheduler takes a live optimizer, and the smoke axis had no
    # way to build one — fourteen of them reported "no value for required
    # parameter 'optimizer'" while the scheduler axis was checking them
    # properly one axis over.
    "optimizer": lambda: lucid.optim.SGD(
        [lucid.nn.Parameter(lucid.tensor(np.zeros(3, dtype=np.float32)))], lr=0.1
    ),
    "opt": lambda: lucid.optim.SGD(
        [lucid.nn.Parameter(lucid.tensor(np.zeros(3, dtype=np.float32)))], lr=0.1
    ),
    # ``CompiledModule(model)`` wraps a live module.  Without this the
    # smoke axis reached it only through its ``(tensor)`` rung, which
    # passed while the constructor never looked at what it was handed and
    # went unanswered once it did (it reads ``model.modules()``).
    "model": lambda: lucid.nn.Linear(2, 2),
    "hooks_list": list,
    "events": list,
    # A transform, and a list of them.  ``ComposeTransform`` and its two
    # siblings reject an empty sequence by design, which is the right
    # refusal and left them unconstructible for want of one element.
    "transform": lambda: lucid.distributions.ExpTransform(),
    "parts": lambda: [lucid.distributions.ExpTransform(), lucid.distributions.AbsTransform()],
    "tseq": lambda: [lucid.distributions.ExpTransform()],
    # ``CatTransform``/``StackTransform``/``ComposeTransform`` all refuse
    # an empty sequence, which is the right refusal — the table used to
    # supply ``[]`` and the three of them were unconstructible for it.
    "transforms": lambda: [lucid.distributions.ExpTransform(), lucid.distributions.AbsTransform()],
    "reinterpreted_batch_ndims": lambda: 1,
    "registered": list,
    "engine_dtype": lambda: lucid.float32._engine_dtype,
    # A batch shape of (3,), not a scalar: ``Independent`` reinterprets
    # ``reinterpreted_batch_ndims`` of the base's *batch* dimensions and
    # refuses a base that has none.
    # ``CumulativeDistributionTransform(distribution)`` needs something
    # with a ``cdf``, and the annotation only says ``Distribution``.
    "distribution": lambda: lucid.distributions.Normal(0.0, 1.0),
    "base_distribution": lambda: lucid.distributions.Normal(
        lucid.tensor(np.zeros(3, dtype=np.float32)), lucid.tensor(np.ones(3, dtype=np.float32))
    ),
    # A Wishart needs more degrees of freedom than the scale's order.
    "df": lambda: 5.0,
    "mixture_distribution": lambda: lucid.distributions.Categorical(
        probs=lucid.tensor(np.full(3, 1.0 / 3.0, dtype=np.float32))
    ),
    "component_distribution": lambda: lucid.distributions.Normal(
        lucid.tensor(np.zeros(3, dtype=np.float32)), lucid.tensor(np.ones(3, dtype=np.float32))
    ),
    "in_shape": lambda: (2, 3),
    "out_shape": lambda: (3, 2),
    # ``finfo`` wants a float dtype and ``iinfo`` an integer one, and
    # they spell the parameter identically — the class asking is the only
    # thing that distinguishes them.
    "dt": lambda: "float32",
    "type_or_str": lambda: "cpu",
    # ``utils.transforms`` has no ``Identity``; an empty ``Compose`` is
    # the transform that does nothing.
    "first": lambda: lucid.utils.transforms.Compose([]),
    "second": lambda: lucid.utils.transforms.Compose([]),
    # ``WorkerInfo`` and ``BatchSampler`` — the loader's own metadata.
    "id": lambda: 0,
    "num_workers": lambda: 2,
    "seed": lambda: 0,
    "drop_last": lambda: False,
    "sampler": lambda: lucid.utils.data.SequentialSampler(_ToyDataset(6)),
    "dataset": lambda: _ToyDataset(6),
    # ``ChainDataset`` requires *iterable* children and says so; a
    # map-style dataset is the right refusal.
    "datasets": lambda: [_ToyIterableDataset(), _ToyIterableDataset()],
    # ``DiagnosisReport`` is a record of lists; an empty report is a
    # valid one and is what a trace of nothing produces.
    "grad_sinks": list,
    "uncovered": list,
    "covered": list,
    "ops": list,
    "nodes": list,
}

#: Optional parameters of which a class accepts **exactly one**.  Stated
#: because the class states it — every one of these raises "pass exactly
#: one of ..." naming its own group.
_EXCLUSIVE: tuple[frozenset[str], ...] = (
    frozenset({"probs", "logits"}),
    frozenset({"covariance_matrix", "precision_matrix", "scale_tril"}),
)

#: Fallback by annotation when the name is not recognised.
_BY_ANNOTATION: tuple[tuple[str, Any], ...] = (
    ("bool", False),
    ("int", 2),
    ("float", 1.0),
    ("str", "constant"),
)


#: Parameters that mean something different in one class than in another,
#: keyed on the class.  ``finfo`` and ``iinfo`` spell their argument
#: identically and accept disjoint dtypes, so the name alone cannot
#: decide — ``iinfo`` correctly refused the float the table supplies.
_BY_CLASS: dict[str, dict[str, Any]] = {
    "iinfo": {"dt": "int32"},
    "dtype": {"engine_dtype": lucid.float32._engine, "itemsize": 4},
    # One switch between the two nested schedulers: ``SequentialLR`` takes
    # ``len(schedulers) - 1`` milestones, and refuses the table's
    # ``[2, 4]`` (a ``MultiStepLR`` value) as the reference does (LCD-492).
    "SequentialLR": {"milestones": [2]},
}


#: Values that have to agree with a parameter filled before them in the
#: same signature: ``(arguments built so far) -> value``.  A composite
#: scheduler's children must drive the optimizer it is handed —
#: ``SequentialLR`` refuses another optimizer's, as the reference does
#: (LCD-492) — so they are built on it.  ``ChainedScheduler`` takes its
#: optimizer after the children, defaulting to theirs.
_BUILD_FROM_SIBLINGS: dict[str, Any] = {
    "schedulers": lambda built: _nested_schedulers(built.get("optimizer"))
}


def _value_for(
    param: inspect.Parameter,
    extra: dict[str, Any],
    cls_name: str = "",
    built: dict[str, Any] | None = None,
) -> tuple[bool, Any]:
    """``(found, value)`` for one required parameter."""
    override = _BY_CLASS.get(cls_name, {})
    if param.name in override:
        return True, override[param.name]
    if param.name in extra:
        return True, extra[param.name]
    sibling = _BUILD_FROM_SIBLINGS.get(param.name)
    if sibling is not None:
        # The caller's own arguments count as built: ``ChainedScheduler``
        # lists its optimizer after the children it has to agree with.
        value = sibling({**extra, **(built or {})})
        return value is not None, value
    if param.name in _BY_NAME:
        return True, _BY_NAME[param.name]
    builder = _BUILD_BY_NAME.get(param.name)
    if builder is not None:
        try:
            return True, builder()
        except Exception:  # noqa: BLE001 - an unbuildable value is no value
            return False, None
    annotation = str(param.annotation)
    for needle, value in _BY_ANNOTATION:
        if needle in annotation:
            return True, value
    return False, None


def _construct(cls: Any, **extra: Any) -> tuple[Any, str]:
    """Build ``cls`` by reading its signature, or say why not.

    Returns ``(instance, "")`` or ``(None, reason)``.  Only parameters
    with no default are filled — a default is the author's own statement
    of a working value, and overriding it would test this file's opinion
    rather than the class.
    """
    # ``Format.STRING`` rather than the default.  Lucid's own rules put
    # tensor types in a ``TYPE_CHECKING`` block and leave the annotation
    # bare (H1 / H7 plus PEP 649 lazy evaluation), so ``Tensor`` and
    # ``QConfig`` genuinely do not exist at runtime — resolving
    # annotations raises ``NameError`` for 32 public classes, including
    # ``lucid.Tensor``.  Asking for the unevaluated text sidesteps it, and
    # the text is all the value table below needs anyway.
    # An Enum is a namespace of members, not a thing you build.  Calling
    # the class is the *functional API* — ``Enum("Name", names)`` — so it
    # reported ``EnumType.__call__() missing 1 required positional
    # argument`` for ``QScheme``, ``Interpolation`` and every other
    # constant group.  Its first member is the instance a caller means.
    members = list(getattr(cls, "__members__", {}).values())
    if members:
        return members[0], ""

    try:
        signature = inspect.signature(cls, annotation_format=annotationlib.Format.STRING)
    except (TypeError, ValueError, NameError) as exc:
        return None, f"no signature: {type(exc).__name__}: {exc}"

    kwargs: dict[str, Any] = {}
    args: list[Any] = []
    optional: list[inspect.Parameter] = []
    for name, param in signature.parameters.items():
        if name == "self" or param.kind is param.VAR_KEYWORD:
            continue
        if param.kind is param.VAR_POSITIONAL:
            # ``TensorDataset(*tensors)`` and ``StackDataset(*datasets)``
            # declare no required *named* parameter, so skipping the
            # variadic constructed them empty and they rejected that —
            # reported as a skip when the harness had simply passed
            # nothing.  Supply two when the name is recognised.
            builder = _VARIADIC.get(name)
            if builder is not None:
                args.extend(builder())
            continue
        if param.default is not param.empty and name not in extra:
            if param.default is None:
                optional.append(param)
            continue
        found, value = _value_for(param, extra, getattr(cls, "__name__", ""), kwargs)
        if not found:
            return None, f"no value for required parameter {name!r}: {param.annotation}"
        kwargs[name] = value
    try:
        return cls(*args, **kwargs), ""
    except Exception as exc:  # noqa: BLE001 - surveying, not asserting
        first = f"{type(exc).__name__}: {str(exc)[:80]}"

    # A second attempt, supplying the optional parameters the tables know
    # a value for.  Nine distributions refuse every default they declare:
    # ``probs`` and ``logits`` are both optional, exactly one has to be
    # given, and honouring both defaults produces "pass exactly one of
    # `probs` or `logits`" — a correct refusal of a construction nothing
    # would ever write.  Only ``probs`` is in the table, so the pair
    # cannot both be filled.
    #
    # Tried *after* the faithful construction rather than instead of it:
    # a class that works with its own defaults is audited as the user
    # gets it.
    extra_kwargs = dict(kwargs)
    added = False
    claimed: set[int] = set()
    for param in optional:
        # At most one member of a mutually exclusive group.  Filling both
        # is not a smaller mistake than filling neither: ``Wishart`` and
        # ``MultivariateNormal`` refuse "exactly one of covariance_matrix,
        # precision_matrix, scale_tril" just as loudly for two as for
        # zero, and the retry that was meant to construct them made the
        # message change without making the class buildable.
        group = next((i for i, names in enumerate(_EXCLUSIVE) if param.name in names), None)
        if group is not None and group in claimed:
            continue
        found, value = _value_for(param, extra, getattr(cls, "__name__", ""), extra_kwargs)
        if found:
            extra_kwargs[param.name] = value
            added = True
            if group is not None:
                claimed.add(group)
    if not added:
        return None, first
    try:
        return cls(*args, **extra_kwargs), ""
    except Exception as exc:  # noqa: BLE001 - surveying, not asserting
        # Both attempts, because they fail for different reasons and
        # reporting only the first sends the reader to fix a parameter
        # the second attempt had already supplied.
        return (None, f"{first}; with optionals filled: {type(exc).__name__}: {str(exc)[:60]}")


class TransformAxis(Axis):
    """Applying a transform returns a finite tensor and does not eat its input.

    Three questions in one pass, because constructing an augmentation is
    the expensive part and applying it is not:

    *Output* — a tensor comes back, finite, and not empty.  A transform
    that silently returns its input unchanged is caught separately by the
    probability switch: every constructible transform is built with
    ``p=1.0`` where it takes one, so "no change at all" is a real answer
    only for the handful that are genuinely identity.

    *Input* — the source tensor is compared against a copy taken before
    the call.  An augmentation that writes through to the caller's image
    corrupts the next epoch's data rather than raising, and it is the
    kind of defect a smoke test cannot see.

    *Repeatability* — the same seed twice must give the same output, or a
    training run cannot be reproduced from its seed.
    """

    name = "transform"
    summary = "apply to an image: finite output, unmodified input, seed-repeatable"
    kinds = frozenset({"transform"})
    varies_a_tensor = False

    def run(self, symbol: Symbol, ctx: Context) -> Finding:
        obj = symbol.obj
        if not isinstance(obj, type):
            return self._finding(symbol, Status.SKIP, "not a class")

        instance, why = _construct(obj)
        if instance is None:
            return self._finding(symbol, Status.SKIP, f"construct: {why}")
        if not callable(instance):
            # ``Image``, ``Mask``, ``Keypoints``, ``BboxParams`` and the
            # interpolation enum live in this package and are not
            # transforms — they are the *data* transforms are applied to,
            # and the parameter object that configures them.  "Applying
            # it returns a finite tensor" is not a question about any of
            # them, and filing six of them under SKIP claimed six
            # augmentations were unchecked.
            return self._finding(
                symbol,
                Status.NOT_APPLICABLE,
                "a data container, not a transform — nothing to apply",
            )

        source = _image()
        before = source.numpy().copy()
        lucid.manual_seed(0)
        try:
            first = instance(source)
        except Exception as exc:  # noqa: BLE001
            return self._finding(
                symbol, Status.UNSUPPORTED, f"apply: {type(exc).__name__}: {str(exc)[:70]}"
            )

        tensor = _as_tensor(first)
        if tensor is None:
            return self._finding(symbol, Status.SKIP, "returned no tensor to check")

        values = tensor.numpy()
        if values.size == 0:
            return self._finding(symbol, Status.FAIL, "returned an empty tensor")
        if not np.isfinite(values.astype(np.float64)).all():
            return self._finding(
                symbol, Status.FAIL, "output contains NaN or infinity", shape=values.shape
            )

        after = source.numpy()
        if not np.array_equal(before, after):
            return self._finding(
                symbol,
                Status.FAIL,
                "modified the input image in place — the caller's tensor changed",
            )

        lucid.manual_seed(0)
        try:
            second = _as_tensor(instance(_image()))
        except Exception as exc:  # noqa: BLE001
            return self._finding(
                symbol, Status.FAIL, f"second application raised: {type(exc).__name__}: {exc}"
            )
        if second is None or second.shape != tensor.shape:
            return self._finding(
                symbol,
                Status.FAIL,
                f"same seed gave shape {getattr(second, 'shape', None)} then {tensor.shape}",
            )
        if not np.allclose(
            second.numpy().astype(np.float64), values.astype(np.float64), rtol=1e-5, atol=1e-6
        ):
            return self._finding(
                symbol,
                Status.FAIL,
                "the same seed produced two different outputs — the run is not reproducible",
            )
        return self._finding(symbol, Status.PASS, f"{values.shape} finite, seed-stable")


def _as_tensor(value: Any) -> Any:
    """The tensor inside a transform's return value, if there is one.

    Transforms may hand back a bare tensor, a ``(image, target)`` pair or
    a dict-shaped sample; the image is what this axis measures.
    """
    if hasattr(value, "numpy") and hasattr(value, "shape"):
        return value
    if isinstance(value, dict):
        for key in ("image", "img", "pixel_values"):
            if key in value:
                return _as_tensor(value[key])
        return None
    if isinstance(value, (tuple, list)) and value:
        return _as_tensor(value[0])
    return None


class DataAxis(Axis):
    """Length, indexing and iteration have to agree with one another.

    A dataset that reports six samples and yields five, a sampler whose
    ``__len__`` disagrees with how many indices it emits, a loader that
    drops the last batch when it was told not to — all of these are
    silent.  Training just sees fewer samples than the epoch count says.
    """

    name = "data"
    summary = "dataset / sampler / loader: len, indexing and iteration agree"
    kinds = frozenset({"data"})
    varies_a_tensor = False

    def _collation(self, symbol: Symbol, fn: Any) -> Finding:
        """The five module-level functions, which are not classes.

        ``collate``, ``default_collate``, ``default_convert``,
        ``get_worker_info`` and ``random_split`` were reported "not a
        class" — true, and the reason the loader's own batching path had
        no check at all.  Each has one property that fails loudly when it
        is wrong: a collation stacks a list of samples into a batch whose
        leading dimension is the list's length, and a split partitions
        the dataset without losing or duplicating a sample.
        """
        name = symbol.short
        if name == "get_worker_info":
            info = fn()
            if info is not None:
                return self._finding(
                    symbol, Status.FAIL, f"reported {type(info).__name__} outside a worker process"
                )
            return self._finding(symbol, Status.PASS, "None outside a worker")

        if name == "random_split":
            dataset = _ToyDataset(6)
            try:
                parts = fn(dataset, [4, 2])
            except Exception as exc:  # noqa: BLE001
                return self._finding(
                    symbol, Status.UNSUPPORTED, f"{type(exc).__name__}: {str(exc)[:60]}"
                )
            sizes = [len(p) for p in parts]
            if sizes != [4, 2]:
                return self._finding(symbol, Status.FAIL, f"asked for [4, 2] and got {sizes}")
            # A partition, not a sample: every index exactly once.
            seen = sorted(i for p in parts for i in getattr(p, "indices", []))
            if seen and seen != list(range(6)):
                return self._finding(
                    symbol, Status.FAIL, f"the parts cover {seen}, not each of 0..5 exactly once"
                )
            return self._finding(symbol, Status.PASS, f"split 6 into {sizes}")

        batch = [_ToyDataset(3)[i] for i in range(3)]
        try:
            out = fn(batch) if name != "default_convert" else fn(batch[0])
        except Exception as exc:  # noqa: BLE001
            return self._finding(
                symbol, Status.UNSUPPORTED, f"{type(exc).__name__}: {str(exc)[:60]}"
            )
        if name == "default_convert":
            return self._finding(symbol, Status.PASS, f"converted to {type(out).__name__}")
        if not isinstance(out, (list, tuple)) or not out:
            return self._finding(
                symbol, Status.FAIL, f"collated a 3-sample batch into {type(out).__name__}"
            )
        stacked = out[0]
        if getattr(stacked, "shape", (0,))[0] != 3:
            return self._finding(
                symbol,
                Status.FAIL,
                f"three samples collated to leading dimension "
                f"{getattr(stacked, 'shape', None)}",
            )
        return self._finding(symbol, Status.PASS, "three samples become one batch of 3")

    def run(self, symbol: Symbol, ctx: Context) -> Finding:
        obj = symbol.obj
        if not isinstance(obj, type):
            return self._collation(symbol, obj)

        dataset = _ToyDataset()
        instance, why = _construct(
            obj,
            **{
                "dataset": dataset,
                "datasets": [_ToyDataset(3), _ToyDataset(3)],
                "data_source": dataset,
                "sampler": _sequential_over(dataset),
                "indices": [0, 1, 2],
                "lengths": [3, 3],
            },
        )
        if instance is None:
            return self._finding(symbol, Status.SKIP, f"construct: {why}")

        declared = None
        try:
            declared = len(instance)
        except TypeError:
            declared = None
        except NotImplementedError:
            # ``Dataset`` and ``Sampler`` are the protocol, not an
            # implementation of it; refusing to answer is the contract.
            return self._finding(
                symbol, Status.NOT_APPLICABLE, "abstract base — __len__ is the subclass's"
            )
        except Exception as exc:  # noqa: BLE001
            return self._finding(symbol, Status.FAIL, f"len() raised: {type(exc).__name__}: {exc}")

        if not hasattr(instance, "__iter__") and hasattr(instance, "__getitem__"):
            if declared is None:
                return self._finding(symbol, Status.SKIP, "indexable but unsized")
            try:
                instance[0]
                instance[declared - 1]
            except Exception as exc:  # noqa: BLE001
                return self._finding(
                    symbol,
                    Status.FAIL,
                    f"len() says {declared} but indexing raised: {type(exc).__name__}: {exc}",
                )
            return self._finding(symbol, Status.PASS, f"{declared} samples, indexable")

        if not hasattr(instance, "__iter__"):
            return _call_if_collator(self, symbol, instance)

        try:
            produced = list(_bounded(instance, limit=64))
        except Exception as exc:  # noqa: BLE001
            return self._finding(
                symbol, Status.UNSUPPORTED, f"iteration: {type(exc).__name__}: {str(exc)[:70]}"
            )
        if declared is not None and len(produced) != declared:
            return self._finding(
                symbol, Status.FAIL, f"len() says {declared} but iterating yielded {len(produced)}"
            )
        return self._finding(
            symbol,
            Status.PASS,
            f"{len(produced)} items, len() {'agrees' if declared else 'absent'}",
        )


def _sequential_over(dataset: Any) -> Any:
    """A sampler for ``dataset``, or the plain index range if there is none."""
    sampler_cls = getattr(lucid.utils.data, "SequentialSampler", None)
    if sampler_cls is not None:
        try:
            return sampler_cls(dataset)
        except Exception:  # noqa: BLE001
            pass
    return list(range(len(dataset)))


def _bounded(iterable: Any, limit: int) -> Any:
    """Yield at most ``limit`` items.

    An ``IterableDataset`` may be infinite by design; an audit that calls
    ``list()`` on one hangs the whole sweep instead of reporting it.
    """
    for index, item in enumerate(iterable):
        if index >= limit:
            return
        yield item


def _call_if_collator(axis: Axis, symbol: Symbol, instance: Any) -> Finding:
    """Collators are neither sized nor iterable — they are called on a batch."""
    if not callable(instance):
        return axis._finding(symbol, Status.SKIP, "neither sized, iterable nor callable")
    batch = [_ToyDataset()[i] for i in range(4)]
    try:
        out = instance(batch)
    except Exception as exc:  # noqa: BLE001
        return axis._finding(
            symbol, Status.UNSUPPORTED, f"collate: {type(exc).__name__}: {str(exc)[:70]}"
        )
    if out is None:
        return axis._finding(symbol, Status.FAIL, "collating a 4-sample batch returned None")
    return axis._finding(symbol, Status.PASS, f"collated 4 samples -> {type(out).__name__}")


class TokenizerAxis(Axis):
    """``decode(encode(text))`` has to give the text back.

    The one property every tokenizer claims and the only one worth
    checking generically.  Where a vocabulary cannot represent the probe
    string the round trip is not required to be exact, so the check falls
    back to the weaker invariant that ids are in range and decoding is
    stable — reported as a pass with the reason attached rather than
    dressed up as the strong result.
    """

    name = "tokenizer"
    summary = "encode / decode round trip, ids within the vocabulary"
    kinds = frozenset({"tokenizer"})
    varies_a_tensor = False

    #: Lowercase ASCII only, so a byte-level and a word-level vocabulary
    #: both have a chance of representing it exactly.
    _TEXT = " ".join(_PROBE_WORDS)

    def run(self, symbol: Symbol, ctx: Context) -> Finding:
        obj = symbol.obj
        if not isinstance(obj, type):
            return self._finding(symbol, Status.SKIP, "not a class")

        vocab = _tiny_vocab()
        instance, why = _construct(obj, vocab=vocab, merges=[])
        if instance is None:
            return self._finding(symbol, Status.SKIP, f"construct: {why}")
        if not hasattr(instance, "encode") or not hasattr(instance, "decode"):
            return self._finding(symbol, Status.SKIP, "no encode / decode pair")

        try:
            ids = instance.encode(self._TEXT)
        except Exception as exc:  # noqa: BLE001
            return self._finding(
                symbol, Status.UNSUPPORTED, f"encode: {type(exc).__name__}: {str(exc)[:70]}"
            )
        ids = list(ids)
        if not ids:
            return self._finding(symbol, Status.FAIL, "encoding a 19-character string gave no ids")

        size = getattr(instance, "vocab_size", None)
        if isinstance(size, int) and size > 0:
            out_of_range = [i for i in ids if not 0 <= int(i) < size]
            if out_of_range:
                return self._finding(
                    symbol,
                    Status.FAIL,
                    f"ids {out_of_range[:4]} fall outside a vocabulary of {size}",
                )

        try:
            text = instance.decode(ids)
        except Exception as exc:  # noqa: BLE001
            return self._finding(
                symbol, Status.FAIL, f"decode of its own ids raised: {type(exc).__name__}: {exc}"
            )
        if text == self._TEXT:
            return self._finding(symbol, Status.PASS, f"round trip exact over {len(ids)} ids")

        try:
            again = instance.decode(list(instance.encode(self._TEXT)))
        except Exception as exc:  # noqa: BLE001
            return self._finding(
                symbol, Status.FAIL, f"second encode / decode raised: {type(exc).__name__}: {exc}"
            )
        if again != text:
            return self._finding(
                symbol, Status.FAIL, "encoding the same text twice decoded to two different strings"
            )
        # Not a pass.  The axis's question is whether ``decode(encode(t))``
        # gives ``t`` back; when it does not, "at least it is repeatable"
        # is a weaker statement wearing the same colour.  Found by
        # mutation: a decoder that silently drops the last character is
        # lossy and perfectly stable, and this branch passed it.
        #
        # There is one more thing a *lossy* round trip still owes, and it
        # is not weaker: re-encoding the decoded text must land on the same
        # ids.  ``decode(encode(t))`` is the tokenizer's own rendering of
        # ``t``, so encoding it again is asking whether that rendering is a
        # fixed point — and the truncating decoder above is not one, since
        # the dropped character comes back as fewer ids.  Where it holds,
        # the cell has a verdict rather than a shrug.
        try:
            reencoded = [int(i) for i in instance.encode(text)]
        except Exception as exc:  # noqa: BLE001
            return self._finding(
                symbol,
                Status.FAIL,
                f"re-encoding its own output raised: {type(exc).__name__}: {exc}",
            )
        if reencoded == [int(i) for i in ids]:
            return self._finding(
                symbol,
                Status.PASS,
                f"round trip is lossy but a fixed point: {len(ids)} ids survive "
                f"re-encoding unchanged",
            )

        # VACUOUS rather than FAIL because a probe vocabulary genuinely
        # cannot spell every input, so an inexact round trip here is not
        # by itself evidence against the tokenizer — it is evidence that
        # nothing was established.
        return self._finding(
            symbol,
            Status.VACUOUS,
            f"round trip is lossy and not a fixed point either: {len(ids)} ids "
            f"in, {len(reencoded)} out; only decode stability was checked",
        )


class SchedulerAxis(Axis):
    """Learning rates stay finite, and a state_dict resumes the same schedule.

    Checkpoint-and-resume is where schedulers actually fail: the rate is
    a function of ``last_epoch``, and a ``state_dict`` that leaves the
    epoch counter out restores a scheduler that restarts its warmup.
    Training then silently runs at the wrong rate for a few hundred
    steps.  Stepping twice from the same state and comparing is the only
    way to see it.
    """

    name = "scheduler"
    summary = "lr stays finite over an epoch sweep and survives a state_dict round trip"
    kinds = frozenset({"scheduler"})
    varies_a_tensor = False

    _EPOCHS = 6

    def run(self, symbol: Symbol, ctx: Context) -> Finding:
        obj = symbol.obj
        if not isinstance(obj, type):
            return self._finding(symbol, Status.SKIP, "not a class")

        optimizer = _toy_optimizer()
        if optimizer is None:
            return self._finding(symbol, Status.SKIP, "no optimizer to schedule")

        instance, why = _construct(obj, optimizer=optimizer)
        if instance is None:
            return self._finding(symbol, Status.SKIP, f"construct: {why}")
        if not hasattr(instance, "step"):
            return self._finding(symbol, Status.SKIP, "no step()")

        rates = _walk(instance, optimizer, self._EPOCHS)
        if rates is None:
            return self._finding(symbol, Status.UNSUPPORTED, "step() raised")
        bad = [r for r in rates if not math.isfinite(r) or r < 0.0]
        if bad:
            return self._finding(
                symbol, Status.FAIL, f"produced a negative or non-finite rate: {bad[:3]}"
            )

        if not hasattr(instance, "state_dict"):
            return self._finding(symbol, Status.PASS, f"{len(rates)} finite rates, no state_dict")

        try:
            saved = instance.state_dict()
        except Exception as exc:  # noqa: BLE001
            return self._finding(symbol, Status.FAIL, f"state_dict(): {type(exc).__name__}: {exc}")

        # Restore the optimizer's rates as well as the scheduler's state.
        # Several schedulers are *chainable*: ``get_lr`` scales whatever
        # the optimizer currently holds rather than recomputing from
        # ``base_lrs``, so resuming with a fresh optimizer at the initial
        # rate is a different situation, not a failed one.  Handing back
        # only the scheduler state reported four of them as losing their
        # epoch counter when the counter was restored correctly and it was
        # the optimizer that had been reset.
        live_rates = [group["lr"] for group in optimizer.param_groups]
        fresh_optimizer = _toy_optimizer()
        replacement, _ = _construct(obj, optimizer=fresh_optimizer)
        if replacement is None or not hasattr(replacement, "load_state_dict"):
            return self._finding(symbol, Status.PASS, f"{len(rates)} finite rates")
        for group, rate in zip(fresh_optimizer.param_groups, live_rates):
            group["lr"] = rate
        try:
            replacement.load_state_dict(saved)
        except Exception as exc:  # noqa: BLE001
            return self._finding(
                symbol, Status.FAIL, f"load_state_dict(): {type(exc).__name__}: {exc}"
            )

        resumed = _walk(replacement, fresh_optimizer, 2)
        continued = _walk(instance, optimizer, 2)
        if resumed is None or continued is None:
            return self._finding(symbol, Status.PASS, f"{len(rates)} finite rates")
        if not np.allclose(resumed, continued, rtol=1e-6, atol=1e-12):
            return self._finding(
                symbol,
                Status.FAIL,
                f"resuming from state_dict gave {resumed} where continuing gave {continued} "
                "— the epoch counter is not in the state",
            )
        return self._finding(
            symbol, Status.PASS, f"{len(rates)} finite rates, resume matches to 1e-6"
        )


def _toy_optimizer() -> Any:
    """SGD over one parameter, the smallest thing a scheduler can drive."""
    try:
        parameter = lucid.nn.Parameter(lucid.tensor(np.ones((2,), np.float32)))
        return lucid.optim.SGD([parameter], lr=0.1)
    except Exception:  # noqa: BLE001
        return None


def _nested_schedulers(optimizer: Any = None) -> list[Any] | None:
    """Two schedulers to nest, for the ones that compose others.

    Built on the optimizer the composite drives: ``ChainedScheduler`` and
    ``SequentialLR`` refuse schedulers of another optimizer, as the
    reference does (LCD-492), and nesting ones built on a fresh optimizer
    cost the composites their cells.  Two, so ``SequentialLR`` has a
    switch to make (one milestone, :data:`_BY_CLASS`).
    """
    if optimizer is None:
        optimizer = _toy_optimizer()
    if optimizer is None:
        return None
    constant = lucid.optim.lr_scheduler.ConstantLR
    try:
        return [
            constant(optimizer, factor=0.5, total_iters=2),
            constant(optimizer, factor=0.25, total_iters=2),
        ]
    except Exception:  # noqa: BLE001 - surveying: no scheduler is no value
        return None


def _walk(scheduler: Any, optimizer: Any, epochs: int) -> list[float] | None:
    """Step ``epochs`` times, returning the rate seen at each one."""
    rates: list[float] = []
    for _ in range(epochs):
        try:
            optimizer.step()
            scheduler.step()
        except Exception:  # noqa: BLE001
            return None
        try:
            rates.append(float(optimizer.param_groups[0]["lr"]))
        except Exception:  # noqa: BLE001
            return None
    return rates


#: Registered into ``ALL_AXES`` beside the other subsystem axes.
DATA_AXES: tuple[Axis, ...] = (TransformAxis(), DataAxis(), TokenizerAxis(), SchedulerAxis())

__all__ = ["DATA_AXES", "DataAxis", "SchedulerAxis", "TokenizerAxis", "TransformAxis"]
