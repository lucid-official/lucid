"""Bind the names annotations mention but a module imports only for type checkers.

Under PEP 649 a function's annotations are evaluated in its own module
globals when something asks for them — ``inspect.signature``, ``help()``, an
editor's tooltip, ``typing.get_type_hints``.  The package imports most of the
types it annotates with inside ``if TYPE_CHECKING:``, to keep import cycles
out, so those names are absent at that moment and the asking raises
``NameError``: 297 public callables did, across ``nn``, ``compile``,
``quantization``, ``models`` and the rest.

Once a module has finished executing, a cycle can no longer bite, so the
names are bound then, from the ``NAMES`` table at the foot of this file,
which ``tools/gen_annotation_names.py`` writes.  How eagerly depends on when:

* a module imported by ``import lucid`` itself binds only what is already
  loaded and waits for the rest — importing a source for it would make
  ``import lucid`` pay for subpackages it deliberately defers (``lucid.nn``
  alone doubles it, 0.18 s to 0.39 s);
* a module imported later — a subpackage the user asked for — also imports
  its own Lucid and standard-library sources, which that import is paying
  for anyway: once the outermost Lucid import in progress has finished, and
  never inside one.  Importing a source while an enclosing Lucid module is
  still executing reaches that module half-built — ``lucid.nn`` loading
  ``lucid.utils`` loading the rollout driver, whose ``RSSM`` source needs
  ``lucid.nn.GELU`` — and the failed import leaves ``lucid.models`` out of
  ``sys.modules`` with its subpackages still in it, which the next plain
  ``import`` of a model family trips over.

A third-party source is never imported for an annotation.  numpy is a
dependency of the bridge, not of the package, and making it required to
read a signature would be the worse trade; those names bind only if numpy
is already loaded.
"""

import importlib
import importlib.abc
import importlib.machinery
import sys
from types import ModuleType
from typing import Any, override

_MISSING = object()

#: source module -> (target module, bound name, attribute) waiting for it.
_waiting: dict[str, list[tuple[str, str, str | None]]] = {}

#: Set once ``import lucid`` has finished: from then on a module binds its
#: sources eagerly.
_eager = False

#: Modules whose code has run to the end.  The import system keeps a module
#: flagged as initialising until ``exec_module`` returns — which is after
#: :func:`_resolve` runs inside it — so the flag alone would make a module
#: wait for itself: ``lucid.weights`` never received ``Module`` from
#: ``lucid.nn``, which imports it.
_done: set[str] = set()

#: Lucid modules executing right now, one inside another.
_depth = 0

#: Sources to import once ``_depth`` is back to zero, in the order asked for.
_deferred: dict[str, None] = {}


def _importable(source: str) -> bool:
    top = source.split(".")[0]
    return top == "lucid" or top in sys.stdlib_module_names


def _ready(name: str) -> ModuleType | None:
    """``name`` if it is loaded and has finished executing."""
    module = sys.modules.get(name)
    if module is None:
        return None
    if name == "lucid" or name in _done:
        # ``lucid`` is still initialising while :func:`install` runs — as its
        # last statement, so everything it defines is already there.
        return module
    spec = getattr(module, "__spec__", None)
    if spec is not None and getattr(spec, "_initializing", False):
        return None
    return module


def _bind(target: ModuleType, bound: str, source: str, attr: str | None) -> None:
    if bound in target.__dict__:
        return
    module = _ready(source)
    if module is None:
        _waiting.setdefault(source, []).append((target.__name__, bound, attr))
        if _eager and _importable(source):
            _deferred[source] = None
        return
    value: Any = module if attr is None else getattr(module, attr, _MISSING)
    if value is _MISSING:
        # ``from package import submodule`` before the submodule is loaded.
        _waiting.setdefault(f"{source}.{attr}", []).append(
            (target.__name__, bound, None)
        )
        return
    target.__dict__[bound] = value


def _resolve(module: ModuleType) -> None:
    for bound, source, attr in NAMES.get(module.__name__, ()):
        _bind(module, bound, source, attr)
    for target_name, bound, attr in _waiting.pop(module.__name__, ()):
        target = sys.modules.get(target_name)
        if target is not None:
            _bind(target, bound, module.__name__, attr)


def _import_deferred() -> None:
    """Import the sources modules asked for, now that none is half-built.

    Each import binds its waiting names as the source finishes; one that
    asks for more sources adds them here, and they are taken in turn.
    """
    while _deferred:
        source = next(iter(_deferred))
        del _deferred[source]
        if source in sys.modules:
            continue
        try:
            importlib.import_module(source)
        except ImportError:
            # A source this install cannot import leaves its names unbound;
            # an annotation must never break the import that asked for it.
            continue


class _ThenResolve(importlib.abc.Loader):
    """The module's own loader, followed by :func:`_resolve`.

    Everything else — ``get_source`` for ``inspect.getsource``, resource
    readers, ``is_package`` — is the wrapped loader's.
    """

    def __init__(self, inner: Any) -> None:
        self._inner = inner

    @override
    def create_module(self, spec: importlib.machinery.ModuleSpec) -> ModuleType | None:
        created: ModuleType | None = self._inner.create_module(spec)
        return created

    @override
    def exec_module(self, module: ModuleType) -> None:
        global _depth
        _depth += 1
        try:
            self._inner.exec_module(module)
        finally:
            _depth -= 1
        _done.add(module.__name__)
        _resolve(module)
        if _depth == 0:
            _import_deferred()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


class _Finder(importlib.abc.MetaPathFinder):
    """Wraps the loader of every ``lucid`` module but the test suite's."""

    @override
    def find_spec(
        self, fullname: str, path: Any = None, target: ModuleType | None = None
    ) -> importlib.machinery.ModuleSpec | None:
        if not fullname.startswith("lucid.") or fullname.startswith("lucid.test."):
            return None
        for finder in sys.meta_path:
            if finder is self:
                continue
            find_spec = getattr(finder, "find_spec", None)
            if find_spec is None:
                continue
            spec: importlib.machinery.ModuleSpec | None = find_spec(
                fullname, path, target
            )
            if spec is None:
                continue
            if spec.loader is not None and hasattr(spec.loader, "exec_module"):
                spec.loader = _ThenResolve(spec.loader)
            return spec
        return None


def install() -> None:
    """Bind annotation names now, and for every ``lucid`` module loaded later.

    Called once, as the last statement of ``lucid/__init__.py``.  Modules
    already loaded bind what their sources already provide and wait for the
    rest; a finder placed first on :data:`sys.meta_path` then runs the same
    binding as each later ``lucid`` module finishes executing.  A second
    call does nothing.

    Returns
    -------
    None
    """
    global _eager
    if any(isinstance(finder, _Finder) for finder in sys.meta_path):
        return
    for name in list(NAMES):
        module = _ready(name)
        if module is not None:
            _resolve(module)
    for name in list(_waiting):
        module = _ready(name)
        if module is not None:
            _resolve(module)
    _eager = True
    sys.meta_path.insert(0, _Finder())


# ── generated by tools/gen_annotation_names.py — do not edit below this line ──
#: Every name a module imports only under ``if TYPE_CHECKING:``, as
#: ``module -> ((bound name, source module, attribute or None), ...)``.
NAMES: dict[str, tuple[tuple[str, str, str | None], ...]] = {
    "lucid._audit_cli": (("Sequence", "collections.abc", "Sequence"),),
    "lucid._dispatch": (("Tensor", "lucid._tensor.tensor", "Tensor"),),
    "lucid._factories._generator": (("Tensor", "lucid._tensor.tensor", "Tensor"),),
    "lucid._factories.converters": (
        ("Tensor", "lucid._tensor.tensor", "Tensor"),
        ("np", "numpy", None),
    ),
    "lucid._factories.creation": (("Tensor", "lucid._tensor.tensor", "Tensor"),),
    "lucid._factories.random": (("Tensor", "lucid._tensor.tensor", "Tensor"),),
    "lucid._ops": (("Tensor", "lucid._tensor.tensor", "Tensor"),),
    "lucid._ops._adapters": (("Tensor", "lucid._tensor.tensor", "Tensor"),),
    "lucid._ops.composite._bessel": (("Tensor", "lucid._tensor.tensor", "Tensor"),),
    "lucid._ops.composite.blas": (("Tensor", "lucid._tensor.tensor", "Tensor"),),
    "lucid._ops.composite.complex": (("Tensor", "lucid._tensor.tensor", "Tensor"),),
    "lucid._ops.composite.dtype": (("Tensor", "lucid._tensor.tensor", "Tensor"),),
    "lucid._ops.composite.elementwise": (("Tensor", "lucid._tensor.tensor", "Tensor"),),
    "lucid._ops.composite.indexing": (("Tensor", "lucid._tensor.tensor", "Tensor"),),
    "lucid._ops.composite.integer": (("Tensor", "lucid._tensor.tensor", "Tensor"),),
    "lucid._ops.composite.predicates": (("Tensor", "lucid._tensor.tensor", "Tensor"),),
    "lucid._ops.composite.reductions": (("Tensor", "lucid._tensor.tensor", "Tensor"),),
    "lucid._ops.composite.shape": (("Tensor", "lucid._tensor.tensor", "Tensor"),),
    "lucid._ops.composite.statistics": (("Tensor", "lucid._tensor.tensor", "Tensor"),),
    "lucid._tensor._dunders": (
        ("Tensor", "lucid._tensor.tensor", "Tensor"),
        ("TensorOrScalar", "lucid._types", "TensorOrScalar"),
        ("_IndexType", "lucid._types", "_IndexType"),
    ),
    "lucid._tensor._indexing": (
        ("Tensor", "lucid._tensor.tensor", "Tensor"),
        ("TensorOrScalar", "lucid._types", "TensorOrScalar"),
        ("_IndexType", "lucid._types", "_IndexType"),
    ),
    "lucid._tensor._methods": (
        ("DTypeLike", "lucid._types", "DTypeLike"),
        ("Tensor", "lucid._tensor.tensor", "Tensor"),
        ("TensorOrScalar", "lucid._types", "TensorOrScalar"),
    ),
    "lucid._tensor._repr": (("Tensor", "lucid._tensor.tensor", "Tensor"),),
    "lucid._tensor._to": (("Tensor", "lucid._tensor.tensor", "Tensor"),),
    "lucid._tensor.tensor": (
        ("NpDTypeLike", "numpy.typing", "DTypeLike"),
        ("_RemovableHandle", "lucid.autograd._hooks", "RemovableHandle"),
        ("np", "numpy", None),
    ),
    "lucid._types": (
        ("Tensor", "lucid._tensor.tensor", "Tensor"),
        ("_C_engine", "lucid._C", "engine"),
        ("np", "numpy", None),
    ),
    "lucid.amp.grad_scaler": (
        ("Optimizer", "lucid.optim.optimizer", "Optimizer"),
        ("Tensor", "lucid._tensor.tensor", "Tensor"),
    ),
    "lucid.autograd._backward": (("Tensor", "lucid._tensor.tensor", "Tensor"),),
    "lucid.autograd._functional": (("Tensor", "lucid._tensor.tensor", "Tensor"),),
    "lucid.autograd._hooks": (("Tensor", "lucid._tensor.tensor", "Tensor"),),
    "lucid.compile": (
        ("CompiledModule", "lucid.compile._entry.module", "CompiledModule"),
        ("DiagnosisReport", "lucid.compile._debug.diagnose", "DiagnosisReport"),
        ("Module", "lucid.nn.module", "Module"),
        ("OpInfo", "lucid.compile._debug.diagnose", "OpInfo"),
        ("Tracer", "lucid._C.engine.compile", "Tracer"),
        ("compile_optimizer", "lucid.compile._optim.compiler", "compile_optimizer"),
        ("compiled_step", "lucid.compile._entry.function", "compiled_step"),
        ("diagnose", "lucid.compile._debug.diagnose", "diagnose"),
        ("fused_step", "lucid.compile._entry.fused_step", "fused_step"),
        ("make_step", "lucid.compile._entry.step", "make_step"),
    ),
    "lucid.compile._core.attention_probe": (
        ("TraceGraph", "lucid._C.engine.compile", "TraceGraph"),
    ),
    "lucid.compile._core.bn_runstats": (
        ("Module", "lucid.nn.module", "Module"),
        ("TensorImpl", "lucid._C.engine", "TensorImpl"),
    ),
    "lucid.compile._core.buffer_writes": (
        ("Tracer", "lucid._C.engine.compile", "Tracer"),
    ),
    "lucid.compile._core.custom_backward": (
        ("TraceGraph", "lucid._C.engine.compile", "TraceGraph"),
        ("Tracer", "lucid._C.engine.compile", "Tracer"),
    ),
    "lucid.compile._core.fallback": (
        ("CacheKey", "lucid.compile._core.signature", "CacheKey"),
        ("Module", "lucid.nn.module", "Module"),
    ),
    "lucid.compile._core.signature": (
        ("Module", "lucid.nn.module", "Module"),
        ("Tensor", "lucid._tensor.tensor", "Tensor"),
        ("Tracer", "lucid._C.engine.compile", "Tracer"),
    ),
    "lucid.compile._core.trace_context": (
        ("Module", "lucid.nn.module", "Module"),
        ("Tracer", "lucid._C.engine.compile", "Tracer"),
    ),
    "lucid.compile._debug.diagnose": (("Tensor", "lucid._tensor.tensor", "Tensor"),),
    "lucid.compile._debug.trace_dump": (
        ("OpNode", "lucid._C.engine.compile", "OpNode"),
        ("TensorMeta", "lucid._C.engine.compile", "TensorMeta"),
        ("TraceGraph", "lucid._C.engine.compile", "TraceGraph"),
    ),
    "lucid.compile._entry.decode_step": (
        ("Module", "lucid.nn.module", "Module"),
        ("StaticCache", "lucid.utils.cache", "StaticCache"),
        ("Tensor", "lucid._tensor.tensor", "Tensor"),
        ("Tracer", "lucid._C.engine.compile", "Tracer"),
    ),
    "lucid.compile._entry.function": (
        ("Module", "lucid.nn.module", "Module"),
        ("Parameter", "lucid.nn.parameter", "Parameter"),
    ),
    "lucid.compile._entry.fused_step": (
        ("CompiledExecutable", "lucid._C.engine.compile", "CompiledExecutable"),
        ("GradScaler", "lucid.amp.grad_scaler", "GradScaler"),
        ("Module", "lucid.nn.module", "Module"),
        ("Optimizer", "lucid.optim.optimizer", "Optimizer"),
        ("Tensor", "lucid._tensor.tensor", "Tensor"),
        ("TraceGraph", "lucid._C.engine.compile", "TraceGraph"),
        ("Tracer", "lucid._C.engine.compile", "Tracer"),
        ("_Flags", "lucid.compile._optim.compiler", "_Flags"),
        ("_ScalarLayout", "lucid.compile._optim.compiler", "_ScalarLayout"),
    ),
    "lucid.compile._entry.module": (
        ("Module", "lucid.nn.module", "Module"),
        ("Tensor", "lucid._tensor.tensor", "Tensor"),
        ("Tracer", "lucid._C.engine.compile", "Tracer"),
    ),
    "lucid.compile._entry.segmented_step": (
        ("CompiledExecutable", "lucid._C.engine.compile", "CompiledExecutable"),
        ("Module", "lucid.nn.module", "Module"),
        ("ParamFingerprint", "lucid.compile._core.signature", "ParamFingerprint"),
        ("Sequential", "lucid.nn.modules.container", "Sequential"),
        ("Tracer", "lucid._C.engine.compile", "Tracer"),
        ("_Slot", "lucid.compile._core.signature", "_Slot"),
    ),
    "lucid.compile._entry.step": (
        ("Module", "lucid.nn.module", "Module"),
        ("Parameter", "lucid.nn.parameter", "Parameter"),
        ("Tensor", "lucid._tensor.tensor", "Tensor"),
        ("Tracer", "lucid._C.engine.compile", "Tracer"),
    ),
    "lucid.compile._optim.compiler": (
        ("Optimizer", "lucid.optim.optimizer", "Optimizer"),
    ),
    "lucid.coreml": (
        ("Module", "lucid.nn.module", "Module"),
        ("Tensor", "lucid._tensor.tensor", "Tensor"),
    ),
    "lucid.coreml._build": (
        ("BlobWriter", "lucid._C.engine", "BlobWriter"),
        ("MilProgram", "lucid._C.engine", "MilProgram"),
        ("Module", "lucid.nn.module", "Module"),
        ("PackagePaths", "lucid._C.engine", "PackagePaths"),
        ("Tensor", "lucid._tensor.tensor", "Tensor"),
        ("TraceGraph", "lucid._C.engine.compile", "TraceGraph"),
        ("TraceNode", "lucid._C.engine.compile", "TraceNode"),
        ("Tracer", "lucid._C.engine.compile", "Tracer"),
    ),
    "lucid.coreml._convert": (
        ("Bindings", "lucid.coreml._emit", "Bindings"),
        ("Builder", "lucid.coreml._build", "Builder"),
        ("EmitResult", "lucid.coreml._emit", "EmitResult"),
        ("TracedOp", "lucid.coreml._emit", "TracedOp"),
    ),
    "lucid.coreml._emit": (("Builder", "lucid.coreml._build", "Builder"),),
    "lucid.coreml._linear": (
        ("Bindings", "lucid.coreml._emit", "Bindings"),
        ("Builder", "lucid.coreml._build", "Builder"),
        ("EmitResult", "lucid.coreml._emit", "EmitResult"),
        ("TracedOp", "lucid.coreml._emit", "TracedOp"),
    ),
    "lucid.coreml._model": (
        ("Module", "lucid.nn.module", "Module"),
        ("Tensor", "lucid._tensor.tensor", "Tensor"),
        ("TensorImpl", "lucid._C.engine", "TensorImpl"),
    ),
    "lucid.coreml._sized": (
        ("Bindings", "lucid.coreml._emit", "Bindings"),
        ("Builder", "lucid.coreml._build", "Builder"),
        ("EmitResult", "lucid.coreml._emit", "EmitResult"),
        ("TracedOp", "lucid.coreml._emit", "TracedOp"),
    ),
    "lucid.fft": (("Tensor", "lucid._tensor.tensor", "Tensor"),),
    "lucid.func": (("Tensor", "lucid._tensor.tensor", "Tensor"),),
    "lucid.metal": (("lucid", "lucid", None),),
    "lucid.models._auto": (
        ("ModelConfig", "lucid.models._base", "ModelConfig"),
        ("PretrainedModel", "lucid.models._base", "PretrainedModel"),
    ),
    "lucid.models._mixins": (
        (
            "DiffusionScheduler",
            "lucid.models.generative._common._schedulers",
            "DiffusionScheduler",
        ),
        ("GenerationOutput", "lucid.models._output", "GenerationOutput"),
    ),
    "lucid.models._output": (
        ("Cache", "lucid.utils.cache", "Cache"),
        ("EncoderDecoderCache", "lucid.utils.cache", "EncoderDecoderCache"),
        ("Tensor", "lucid._tensor.tensor", "Tensor"),
    ),
    "lucid.models._registry": (
        ("ModelConfig", "lucid.models._base", "ModelConfig"),
        ("PretrainedModel", "lucid.models._base", "PretrainedModel"),
    ),
    "lucid.models._utils._common": (
        ("DataclassInstance", "_typeshed", "DataclassInstance"),
        ("Tensor", "lucid._tensor.tensor", "Tensor"),
    ),
    "lucid.nn._module_backward_state": (("Module", "lucid.nn.module", "Module"),),
    "lucid.nn.functional._grid_sampling": (
        ("Tensor", "lucid._tensor.tensor", "Tensor"),
    ),
    "lucid.nn.functional._index_checks": (
        ("Tensor", "lucid._tensor.tensor", "Tensor"),
    ),
    "lucid.nn.functional._inplace": (
        ("Tensor", "lucid._tensor.tensor", "Tensor"),
        ("_C_engine", "lucid._C", "engine"),
    ),
    "lucid.nn.functional._interpolate": (("Tensor", "lucid._tensor.tensor", "Tensor"),),
    "lucid.nn.functional._loss_checks": (("Tensor", "lucid._tensor.tensor", "Tensor"),),
    "lucid.nn.functional.activations": (("Tensor", "lucid._tensor.tensor", "Tensor"),),
    "lucid.nn.functional.attention": (
        ("DTypeLike", "lucid._types", "DTypeLike"),
        ("DeviceLike", "lucid._types", "DeviceLike"),
        ("Tensor", "lucid._tensor.tensor", "Tensor"),
    ),
    "lucid.nn.functional.conv": (("Tensor", "lucid._tensor.tensor", "Tensor"),),
    "lucid.nn.functional.dropout": (("Tensor", "lucid._tensor.tensor", "Tensor"),),
    "lucid.nn.functional.linear": (("Tensor", "lucid._tensor.tensor", "Tensor"),),
    "lucid.nn.functional.loss": (("Tensor", "lucid._tensor.tensor", "Tensor"),),
    "lucid.nn.functional.metrics": (("Tensor", "lucid._tensor.tensor", "Tensor"),),
    "lucid.nn.functional.normalization": (
        ("Tensor", "lucid._tensor.tensor", "Tensor"),
    ),
    "lucid.nn.functional.pooling": (("Tensor", "lucid._tensor.tensor", "Tensor"),),
    "lucid.nn.functional.positional": (("Tensor", "lucid._tensor.tensor", "Tensor"),),
    "lucid.nn.functional.sampling": (("Tensor", "lucid._tensor.tensor", "Tensor"),),
    "lucid.nn.functional.sparse": (("Tensor", "lucid._tensor.tensor", "Tensor"),),
    "lucid.nn.init": (("Tensor", "lucid._tensor.tensor", "Tensor"),),
    "lucid.nn.intrinsic.qat.modules": (
        ("FakeQuantize", "lucid.quantization._fake_quantize", "FakeQuantize"),
        ("QConfig", "lucid.quantization.qconfig", "QConfig"),
        ("Tensor", "lucid._tensor.tensor", "Tensor"),
    ),
    "lucid.nn.module": (("_HasForward", "lucid._types", "_HasForward"),),
    "lucid.nn.qat.conv": (
        ("FakeQuantize", "lucid.quantization._fake_quantize", "FakeQuantize"),
        ("QConfig", "lucid.quantization.qconfig", "QConfig"),
        ("Tensor", "lucid._tensor.tensor", "Tensor"),
    ),
    "lucid.nn.qat.linear": (
        ("FakeQuantize", "lucid.quantization._fake_quantize", "FakeQuantize"),
        ("QConfig", "lucid.quantization.qconfig", "QConfig"),
        ("Tensor", "lucid._tensor.tensor", "Tensor"),
    ),
    "lucid.nn.qat.sparse": (
        ("FakeQuantize", "lucid.quantization._fake_quantize", "FakeQuantize"),
        ("QConfig", "lucid.quantization.qconfig", "QConfig"),
        ("Tensor", "lucid._tensor.tensor", "Tensor"),
    ),
    "lucid.nn.quantized._utils": (
        ("FakeQuantize", "lucid.quantization._fake_quantize", "FakeQuantize"),
        ("ObserverBase", "lucid.quantization.observer", "ObserverBase"),
        ("Parameter", "lucid.nn.parameter", "Parameter"),
        ("QConfig", "lucid.quantization.qconfig", "QConfig"),
        ("QDtype", "lucid.quantization._qscheme", "QDtype"),
        ("Tensor", "lucid._tensor.tensor", "Tensor"),
        ("nn", "lucid.nn", None),
    ),
    "lucid.nn.quantized.activation": (("Tensor", "lucid._tensor.tensor", "Tensor"),),
    "lucid.nn.quantized.conv": (("Tensor", "lucid._tensor.tensor", "Tensor"),),
    "lucid.nn.quantized.conv_transpose": (
        ("Tensor", "lucid._tensor.tensor", "Tensor"),
    ),
    "lucid.nn.quantized.dynamic.linear": (
        ("Tensor", "lucid._tensor.tensor", "Tensor"),
    ),
    "lucid.nn.quantized.dynamic.rnn": (("Tensor", "lucid._tensor.tensor", "Tensor"),),
    "lucid.nn.quantized.functional_module": (
        ("Tensor", "lucid._tensor.tensor", "Tensor"),
    ),
    "lucid.nn.quantized.intrinsic": (("Tensor", "lucid._tensor.tensor", "Tensor"),),
    "lucid.nn.quantized.linear": (("Tensor", "lucid._tensor.tensor", "Tensor"),),
    "lucid.nn.quantized.mlx_linear": (("Tensor", "lucid._tensor.tensor", "Tensor"),),
    "lucid.nn.quantized.modules": (
        ("ObserverBase", "lucid.quantization.observer", "ObserverBase"),
        ("Tensor", "lucid._tensor.tensor", "Tensor"),
    ),
    "lucid.nn.quantized.sparse": (("Tensor", "lucid._tensor.tensor", "Tensor"),),
    "lucid.nn.utils.clip_grad": (
        ("Parameter", "lucid.nn.parameter", "Parameter"),
        ("Tensor", "lucid._tensor.tensor", "Tensor"),
    ),
    "lucid.nn.utils.fusion": (("Tensor", "lucid._tensor.tensor", "Tensor"),),
    "lucid.nn.utils.skip_init": (("Module", "lucid.nn.module", "Module"),),
    "lucid.optim._step_counter": (("Optimizer", "lucid.optim.optimizer", "Optimizer"),),
    "lucid.quantization._fake_quantize": (
        ("Tensor", "lucid._tensor.tensor", "Tensor"),
    ),
    "lucid.quantization._functional": (
        ("QDtype", "lucid.quantization._qscheme", "QDtype"),
    ),
    "lucid.quantization._qgemm": (("Tensor", "lucid._tensor.tensor", "Tensor"),),
    "lucid.quantization._quantize": (("Tensor", "lucid._tensor.tensor", "Tensor"),),
    "lucid.quantization._quantize_fx": (
        ("QConfig", "lucid.quantization.qconfig", "QConfig"),
        ("QConfigMapping", "lucid.quantization.qconfig", "QConfigMapping"),
        ("Tensor", "lucid._tensor.tensor", "Tensor"),
        ("nn", "lucid.nn", None),
    ),
    "lucid.quantization.fuse_modules": (
        ("QConfig", "lucid.quantization.qconfig", "QConfig"),
        ("Sequence", "collections.abc", "Sequence"),
    ),
    "lucid.quantization.observer": (("Tensor", "lucid._tensor.tensor", "Tensor"),),
    "lucid.signal.windows": (("Tensor", "lucid._tensor.tensor", "Tensor"),),
    "lucid.special": (("Tensor", "lucid._tensor.tensor", "Tensor"),),
    "lucid.utils.cache._base": (("Tensor", "lucid._tensor.tensor", "Tensor"),),
    "lucid.utils.cache._dynamic": (("Tensor", "lucid._tensor.tensor", "Tensor"),),
    "lucid.utils.cache._encoder_decoder": (
        ("Tensor", "lucid._tensor.tensor", "Tensor"),
    ),
    "lucid.utils.data.dataset": (("Tensor", "lucid._tensor.tensor", "Tensor"),),
    "lucid.utils.rollout._driver": (
        ("RSSM", "lucid.models.generative._common._rssm", "RSSM"),
        ("RSSMState", "lucid.models.generative._common._rssm", "RSSMState"),
    ),
    "lucid.weights._base": (
        ("Transform", "lucid.utils.transforms._base", "TransformLike"),
    ),
    "lucid.weights._loading": (
        ("Module", "lucid.nn", "Module"),
        ("Tensor", "lucid._tensor", "Tensor"),
    ),
}
# ── end generated ──
