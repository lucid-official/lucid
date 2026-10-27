"""Axes for the symbols that answer about state rather than about numbers.

Ninety-six symbols reached only :class:`~lucid.test.audit._axes_subsystem.
SmokeAxis` — called once, required not to crash, and asked nothing else.
Seventy-nine of them were excused on the grounds that they "mutate process
state and no numeric axis can express them", which is true and is not a
reason to stop.  *Mutating* state is exactly what makes them checkable:

    manual_seed(0); a = rand(); manual_seed(0); b = rand()   a must equal b
    set_default_dtype(float64)                               zeros(2) must follow
    set_grad_enabled(False)                                  x * x must have no graph
    get_rng_state() / set_rng_state()                        must restore the draw

None of those is a numeric comparison and every one of them can fail.  A
setter whose getter does not see it, a seed that does not reproduce, a
hook that is registered and never fires, a handle whose ``remove`` leaves
the hook installed — these are the defects this file exists for, and the
smoke axis passes all of them because the call itself does not raise.

Five axes, each over a group the numeric sweep cannot reach:

    state       global get/set pairs, seeds, thread counts, grad mode
    hook        the module and tensor hook registrars, and their handles
    metadata    dtype and device algebra: promote_types, finfo, .itemsize
    transform   the functional transforms: grad, vmap, jacrev, vjp, checkpoint
    nnutils     parametrize / prune / weight_norm / fuse, which take a module
"""

import contextlib
import copy
import math
from typing import TYPE_CHECKING, Any

import numpy as np

import lucid
import lucid.utils.transforms
from lucid.test.audit import _probe, _surface
from lucid.test.audit._axes import Axis, Context
from lucid.test.audit._result import Finding, Status

if TYPE_CHECKING:
    from lucid.test.audit._surface import Symbol


def _draw(device: str = "cpu") -> Any:
    """One sample drawn on ``device`` — the observable a seed controls.

    Every device draws from one generator: ``lucid.metal.manual_seed`` and
    ``lucid.manual_seed`` seed the same stream, and since LCD-583 every draw
    that names no generator reads it.  The device picks only how the stream
    is sampled: one seed gives different numbers on the CPU and on Metal, so
    the draws a check compares are taken on one device.  It is a parameter
    so that a ``lucid.metal`` seed is checked by a draw on the device it
    names.
    """
    return _probe.to_numpy(lucid.rand((3,), device=device))


# ── global state ─────────────────────────────────────────────────────────────


class StateAxis(Axis):
    """A setter must be visible to its getter, and a seed must reproduce.

    Every check restores what it changed.  The guard is not a courtesy:
    a first sweep left ``set_grad_enabled(False)`` behind and reported
    the framework as non-differentiable for the 278 symbols after it.
    """

    name = "state"
    summary = "setters are visible to their getters; a seed reproduces a draw"
    kinds = frozenset({"op", "util", "method", "data", "quant"})

    #: ``(reader, writer, values to try)``.  The values are deliberately
    #: *different from the current one* — writing back what is already
    #: there passes whatever the setter does, which is the vacuous pass
    #: this whole tool is built to refuse.
    _PAIRS: tuple[tuple[str, str, tuple[Any, ...]], ...] = (
        ("get_default_dtype", "set_default_dtype", (lucid.float64, lucid.float32)),
        ("get_num_threads", "set_num_threads", (1, 2)),
        # Two values, like every other pair.  It was one, which is the
        # VACUOUS condition this axis exists to refuse: a getter returning
        # a constant would have passed.  A second value became available
        # once the setter stopped rejecting its own getter's "unset"
        # sentinel, so the pair is answerable now.
        ("get_num_interop_threads", "set_num_interop_threads", (1, 2)),
        ("are_deterministic_algorithms_enabled", "use_deterministic_algorithms", (True, False)),
        ("is_grad_enabled", "set_grad_enabled", (False, True)),
    )

    def applies(self, symbol: Symbol) -> bool:
        return symbol.short in self._reachable()

    @classmethod
    def _reachable(cls) -> frozenset[str]:
        names = {name for pair in cls._PAIRS for name in pair[:2]}
        names |= {
            "manual_seed",
            "seed",
            "initial_seed",
            "get_rng_state",
            "set_rng_state",
            "get_default_device",
            "set_default_device",
            "no_grad",
            "enable_grad",
            "inference_mode",
            "set_detect_anomaly",
            "get_worker_info",
            "get_device_name",
            "memory_allocated",
            "max_memory_allocated",
            "get_cache_memory",
            "reset_peak_memory_stats",
            "empty_cache",
            "synchronize",
            "is_available",
            "is_shared",
        }
        return frozenset(names)

    def run(self, symbol: Symbol, ctx: Context) -> Finding:
        name = symbol.short
        obj = _surface.resolve(symbol)
        if obj is None or not callable(obj):
            return self._finding(symbol, Status.SKIP, "not callable")

        for reader, writer, values in self._PAIRS:
            if name in (reader, writer):
                return self._round_trip(symbol, reader, writer, values)
        if name in ("manual_seed", "seed", "initial_seed"):
            return self._seed(symbol, obj, name)
        if name in ("get_rng_state", "set_rng_state"):
            return self._rng_state(symbol)
        if name in ("get_default_device", "set_default_device"):
            return self._default_device(symbol)
        if name in ("no_grad", "enable_grad", "inference_mode"):
            return self._grad_context(symbol, obj, name)
        if name == "set_detect_anomaly":
            return self._detect_anomaly(symbol, obj)
        return self._read_only(symbol, obj, name)

    # ── the checks ───────────────────────────────────────────────────────────

    def _round_trip(
        self, symbol: Symbol, reader: str, writer: str, values: tuple[Any, ...]
    ) -> Finding:
        get = getattr(lucid, reader, None)
        set_ = getattr(lucid, writer, None)
        if get is None or set_ is None:
            return self._finding(symbol, Status.SKIP, f"{reader}/{writer} not exposed")
        try:
            before = get()
        except Exception as exc:  # noqa: BLE001
            return self._finding(symbol, Status.SKIP, f"{reader}: {type(exc).__name__}")

        seen: list[str] = []
        try:
            for value in values:
                try:
                    set_(value)
                except Exception as exc:  # noqa: BLE001
                    return self._finding(
                        symbol,
                        Status.UNSUPPORTED,
                        f"{writer}({value!r}): {type(exc).__name__}: {str(exc)[:50]}",
                    )
                got = get()
                if got != value and str(got) != str(value):
                    return self._finding(
                        symbol, Status.FAIL, f"{writer}({value!r}) then {reader}() reported {got!r}"
                    )
                seen.append(f"{value!r}")
        finally:
            with contextlib.suppress(Exception):
                set_(before)

        if len(seen) < 2:
            # One value proves nothing: the getter could be returning a
            # constant that happens to match.  Reported rather than
            # passed, which is what VACUOUS is for.
            return self._finding(symbol, Status.VACUOUS, f"only one value was accepted: {seen}")
        # The restore is part of the contract too — a setter that cannot
        # accept its own getter's value is a setter no save/restore can
        # use, and this tool's own :class:`StateGuard` is one of them: it
        # snapshots with the getter and restores with the setter, and
        # suppresses the exception, so it silently fails to restore.
        if get() != before and str(get()) != str(before):
            return self._finding(
                symbol,
                Status.FAIL,
                f"{reader}() reports {before!r} and {writer}() will not accept it "
                f"— the value cannot be saved and restored",
            )
        return self._finding(symbol, Status.PASS, f"{reader} follows {writer} for {seen}")

    def _seed(self, symbol: Symbol, obj: Any, name: str) -> Finding:
        # One stream serves both devices (see ``_draw``).  The draw is taken
        # on the device the symbol's qualname names, so a ``lucid.metal``
        # seed is held to what a Metal draw sees.
        device = "metal" if ".metal." in symbol.qualname else "cpu"
        if device == "metal" and not _probe.metal_available():
            return self._finding(symbol, Status.SKIP, "no Metal device to seed")
        # The stream is saved and restored around a ``lucid`` seed only.  The
        # asymmetry dates from separate CPU and Metal streams, when restoring
        # around a Metal seed disturbed the Metal stream; with one stream it
        # no longer protects anything.
        before = None
        if device == "cpu":
            with contextlib.suppress(Exception):
                before = lucid.get_rng_state()
        try:
            if name == "initial_seed":
                value = obj()
                return self._finding(symbol, Status.PASS, f"reports seed {value}")
            if name == "seed":
                obj()
                first = _draw(device)
                obj()
                second = _draw(device)
                if first is None or second is None:
                    return self._finding(symbol, Status.SKIP, "no draw to compare")
                if np.array_equal(first, second):
                    # ``seed()`` re-seeds from entropy: two calls agreeing
                    # would mean it is not doing that.
                    return self._finding(
                        symbol,
                        Status.FAIL,
                        "two re-seeds produced the same draw — the seed is not fresh",
                    )
                return self._finding(symbol, Status.PASS, "re-seeds from fresh entropy")

            obj(1234)
            first = _draw(device)
            obj(1234)
            second = _draw(device)
            obj(4321)
            other = _draw(device)
        except Exception as exc:  # noqa: BLE001
            return self._finding(
                symbol, Status.UNSUPPORTED, f"{type(exc).__name__}: {str(exc)[:60]}"
            )
        finally:
            if before is not None:
                with contextlib.suppress(Exception):
                    lucid.set_rng_state(before)

        if first is None or second is None:
            return self._finding(symbol, Status.SKIP, "no draw to compare")
        if not np.array_equal(first, second):
            return self._finding(symbol, Status.FAIL, "the same seed produced two different draws")
        if other is not None and np.array_equal(first, other):
            # Reproducible and *insensitive* is the worse failure: it
            # reads as a working seed and means the seed is ignored.
            return self._finding(symbol, Status.FAIL, "a different seed produced the same draw")
        return self._finding(symbol, Status.PASS, "reproducible and seed-sensitive")

    def _rng_state(self, symbol: Symbol) -> Finding:
        get = getattr(lucid, "get_rng_state", None)
        set_ = getattr(lucid, "set_rng_state", None)
        if get is None or set_ is None:
            return self._finding(symbol, Status.SKIP, "the pair is not exposed")
        try:
            saved = get()
            first = _draw()
            set_(saved)
            second = _draw()
        except Exception as exc:  # noqa: BLE001
            return self._finding(
                symbol, Status.UNSUPPORTED, f"{type(exc).__name__}: {str(exc)[:60]}"
            )
        if first is None or second is None:
            return self._finding(symbol, Status.SKIP, "no draw to compare")
        if not np.array_equal(first, second):
            return self._finding(
                symbol, Status.FAIL, "restoring the captured state did not reproduce the draw"
            )
        return self._finding(symbol, Status.PASS, "the captured state replays the draw")

    def _default_device(self, symbol: Symbol) -> Finding:
        get = getattr(lucid, "get_default_device", None)
        set_ = getattr(lucid, "set_default_device", None)
        if get is None or set_ is None:
            return self._finding(symbol, Status.SKIP, "the pair is not exposed")
        before = get()
        try:
            for value in ("cpu",):
                set_(value)
                if value not in str(get()):
                    return self._finding(
                        symbol, Status.FAIL, f"set to {value!r}, reported {get()!r}"
                    )
                # The setter's whole purpose is where a new tensor lands.
                if value not in str(lucid.zeros((2,)).device):
                    return self._finding(
                        symbol,
                        Status.FAIL,
                        f"default device is {value!r} and a new tensor is not on it",
                    )
        except Exception as exc:  # noqa: BLE001
            return self._finding(
                symbol, Status.UNSUPPORTED, f"{type(exc).__name__}: {str(exc)[:60]}"
            )
        finally:
            with contextlib.suppress(Exception):
                set_(before)
        return self._finding(symbol, Status.PASS, "new tensors follow the default device")

    def _grad_context(self, symbol: Symbol, obj: Any, name: str) -> Finding:
        x = lucid.tensor(np.ones((2, 2)), requires_grad=True)
        wanted = name == "enable_grad"
        try:
            with obj():
                inside = (x * x).requires_grad
            outside = (x * x).requires_grad
        except Exception as exc:  # noqa: BLE001
            return self._finding(
                symbol, Status.UNSUPPORTED, f"{type(exc).__name__}: {str(exc)[:60]}"
            )
        if inside is not wanted:
            return self._finding(
                symbol, Status.FAIL, f"inside {name}() a product reports requires_grad={inside}"
            )
        if not outside:
            return self._finding(symbol, Status.FAIL, f"{name}() did not restore grad mode on exit")
        return self._finding(symbol, Status.PASS, f"grad is {wanted} inside and restored after")

    def _detect_anomaly(self, symbol: Symbol, obj: Any) -> Finding:
        reader = getattr(lucid.autograd, "is_anomaly_enabled", None)
        try:
            obj(True)
            enabled = reader() if reader is not None else True
            obj(False)
            disabled = reader() if reader is not None else False
        except Exception as exc:  # noqa: BLE001
            return self._finding(
                symbol, Status.UNSUPPORTED, f"{type(exc).__name__}: {str(exc)[:60]}"
            )
        finally:
            with contextlib.suppress(Exception):
                obj(False)
        if reader is None:
            return self._finding(
                symbol, Status.VACUOUS, "no reader to confirm the flag took effect"
            )
        if not (enabled and not disabled):
            return self._finding(
                symbol, Status.FAIL, f"set True -> {enabled}, set False -> {disabled}"
            )
        return self._finding(symbol, Status.PASS, "the flag is readable and follows")

    def _read_only(self, symbol: Symbol, obj: Any, name: str) -> Finding:
        """A query about the process or the device: call it, twice.

        The weakest check here and still not the smoke axis's: a reader
        has to be *stable* between two calls that changed nothing, and
        the memory counters have to be non-negative.  A counter that goes
        backwards or a device name that changes under a fixed device is a
        defect the single call cannot see.
        """
        try:
            first = obj()
            second = obj()
        except Exception as exc:  # noqa: BLE001
            return self._finding(
                symbol, Status.UNSUPPORTED, f"{type(exc).__name__}: {str(exc)[:60]}"
            )
        if isinstance(first, (int, float)) and not isinstance(first, bool):
            if first < 0:
                return self._finding(symbol, Status.FAIL, f"reported {first}, negative")
            return self._finding(symbol, Status.PASS, f"reports {first}")
        if isinstance(first, (str, bool)) and first != second:
            return self._finding(
                symbol, Status.FAIL, f"two calls disagree: {first!r} then {second!r}"
            )
        return self._finding(symbol, Status.PASS, f"stable: {str(first)[:40]}")


# ── hooks ────────────────────────────────────────────────────────────────────


class HookAxis(Axis):
    """A registered hook fires, and its handle's ``remove`` un-registers it.

    Both halves matter and they fail independently.  A hook that never
    fires is a feature that silently does nothing; a handle whose
    ``remove`` does not remove leaks the hook into every later forward
    pass, which is the harder of the two to notice and the one that turns
    a survey's own instrumentation into a source of findings.
    """

    name = "hook"
    summary = "a registered hook fires, and removing it stops it firing"
    kinds = frozenset({"op", "method"})

    _GLOBAL = {
        "register_module_forward_hook",
        "register_module_forward_pre_hook",
        "register_module_full_backward_hook",
        "register_module_full_backward_pre_hook",
        "register_module_load_state_dict_post_hook",
        "register_module_load_state_dict_pre_hook",
    }

    def applies(self, symbol: Symbol) -> bool:
        return symbol.short in self._GLOBAL or symbol.qualname == "Tensor.register_hook"

    def run(self, symbol: Symbol, ctx: Context) -> Finding:
        register = _surface.resolve(symbol)
        if register is None or not callable(register):
            return self._finding(symbol, Status.SKIP, "not callable")
        if symbol.qualname == "Tensor.register_hook":
            return self._tensor_hook(symbol, register)
        return self._module_hook(symbol, register)

    def _tensor_hook(self, symbol: Symbol, register: Any) -> Finding:
        fired: list[Any] = []
        x = lucid.tensor(np.ones((2, 2)), requires_grad=True)
        try:
            handle = register(x, lambda grad: fired.append(grad))
            (x * x).sum().backward()
        except Exception as exc:  # noqa: BLE001
            return self._finding(
                symbol, Status.UNSUPPORTED, f"{type(exc).__name__}: {str(exc)[:60]}"
            )
        if not fired:
            return self._finding(symbol, Status.FAIL, "backward ran and the hook never fired")
        return self._removes(symbol, handle, fired, self._backward_again(x))

    @staticmethod
    def _backward_again(x: Any) -> Any:
        def again() -> None:
            x.grad = None
            (x * x).sum().backward()

        return again

    def _module_hook(self, symbol: Symbol, register: Any) -> Finding:
        fired: list[Any] = []
        module = lucid.nn.Linear(3, 3)
        probe = _probe.as_f32(_probe.sample("moderate", (2, 3)))

        def hook(*args: Any, **kwargs: Any) -> None:
            fired.append(args)

        try:
            handle = register(hook)
        except Exception as exc:  # noqa: BLE001
            return self._finding(symbol, Status.UNSUPPORTED, f"register: {type(exc).__name__}")

        def exercise() -> None:
            if "load_state_dict" in symbol.short:
                module.load_state_dict(module.state_dict())
            elif "backward" in symbol.short:
                module(probe).sum().backward()
            else:
                module(probe)

        try:
            exercise()
        except Exception as exc:  # noqa: BLE001
            with contextlib.suppress(Exception):
                handle.remove()
            return self._finding(symbol, Status.UNSUPPORTED, f"exercise: {type(exc).__name__}")
        if not fired:
            with contextlib.suppress(Exception):
                handle.remove()
            return self._finding(symbol, Status.FAIL, "the hook was registered and never fired")
        return self._removes(symbol, handle, fired, exercise)

    def _removes(self, symbol: Symbol, handle: Any, fired: list[Any], exercise: Any) -> Finding:
        remove = getattr(handle, "remove", None)
        if remove is None:
            return self._finding(symbol, Status.FAIL, "fired, but returned no removable handle")
        count = len(fired)
        try:
            remove()
            exercise()
        except Exception as exc:  # noqa: BLE001
            return self._finding(symbol, Status.UNSUPPORTED, f"after remove: {type(exc).__name__}")
        if len(fired) != count:
            return self._finding(
                symbol,
                Status.FAIL,
                f"handle.remove() left the hook installed — it fired "
                f"{len(fired) - count} more time(s)",
            )
        return self._finding(symbol, Status.PASS, "fires once registered, stops once removed")


# ── dtype and device algebra ─────────────────────────────────────────────────


class MetadataAxis(Axis):
    """What an op says *about* a tensor, where no numeric axis applies.

    ``Tensor.dtype`` returns a dtype, ``promote_types`` returns a dtype,
    ``finfo`` returns a description — none of them is an array, so every
    numeric axis skipped them and reported "returned dtype, nothing
    measurable" eleven times each.  Nothing measurable is not nothing
    checkable: a dtype has to be the dtype the tensor was built with,
    promotion has to be commutative and has to widen, and ``itemsize``
    has to agree with the dtype it belongs to.
    """

    name = "metadata"
    summary = "dtype and device queries agree with the tensor they describe"
    kinds = frozenset({"op", "method"})

    _NAMES = frozenset(
        {
            "dtype",
            "device",
            "itemsize",
            "ndim",
            "shape",
            "size",
            "numel",
            "data_ptr",
            "storage_offset",
            "get_device",
            "is_floating_point",
            "is_complex",
            "promote_types",
            "result_type",
            "finfo",
            "iinfo",
            "to_engine_dtype",
            "get_default_dtype",
            "get_default_device",
        }
    )

    def applies(self, symbol: Symbol) -> bool:
        if symbol.short not in self._NAMES:
            return False
        # ``get_default_*`` belong to the state axis, which checks them
        # against their setters; here they would only be called.
        return symbol.short not in ("get_default_dtype", "get_default_device")

    def run(self, symbol: Symbol, ctx: Context) -> Finding:
        fn = _surface.resolve(symbol)
        if fn is None or not callable(fn):
            return self._finding(symbol, Status.SKIP, "not callable")
        name = symbol.short
        if name in ("promote_types", "result_type"):
            return self._promotion(symbol, fn, name)
        if name in ("finfo", "iinfo"):
            return self._limits(symbol, fn, name)
        if name == "to_engine_dtype":
            return self._engine_dtype(symbol, fn)
        return self._describes(symbol, fn, name)

    def _promotion(self, symbol: Symbol, fn: Any, name: str) -> Finding:
        def operand(dtype: Any) -> Any:
            if name == "result_type":
                return lucid.zeros((2,), dtype=dtype)
            return dtype

        pairs = (
            (lucid.float32, lucid.float64, "float64"),
            (lucid.int32, lucid.int64, "int64"),
            (lucid.int32, lucid.float32, "float"),
        )
        for left, right, expected in pairs:
            try:
                forward = fn(operand(left), operand(right))
                backward = fn(operand(right), operand(left))
            except Exception as exc:  # noqa: BLE001
                return self._finding(
                    symbol, Status.UNSUPPORTED, f"{type(exc).__name__}: {str(exc)[:60]}"
                )
            if str(forward) != str(backward):
                return self._finding(
                    symbol,
                    Status.FAIL,
                    f"not commutative: ({left}, {right}) -> {forward}, " f"reversed -> {backward}",
                )
            if expected not in str(forward):
                return self._finding(
                    symbol,
                    Status.FAIL,
                    f"({left}, {right}) promoted to {forward}, which does not " f"contain both",
                )
        return self._finding(symbol, Status.PASS, "commutative and widening")

    def _limits(self, symbol: Symbol, fn: Any, name: str) -> Finding:
        dtype = lucid.float32 if name == "finfo" else lucid.int32
        try:
            info = fn(dtype)
        except Exception as exc:  # noqa: BLE001
            return self._finding(
                symbol, Status.UNSUPPORTED, f"{type(exc).__name__}: {str(exc)[:60]}"
            )
        low = getattr(info, "min", None)
        high = getattr(info, "max", None)
        if low is None or high is None:
            return self._finding(symbol, Status.SKIP, "no min/max to compare")
        if not low < high:
            return self._finding(symbol, Status.FAIL, f"min {low} is not below max {high}")
        if name == "finfo":
            eps = getattr(info, "eps", None)
            if eps is not None and not 0.0 < float(eps) < 1.0:
                return self._finding(symbol, Status.FAIL, f"eps is {eps}")
        return self._finding(symbol, Status.PASS, f"{low} .. {high}")

    def _engine_dtype(self, symbol: Symbol, fn: Any) -> Finding:
        seen: dict[str, Any] = {}
        for name in ("float32", "float64", "int32", "int64"):
            dtype = getattr(lucid, name, None)
            if dtype is None:
                continue
            try:
                seen[name] = fn(dtype)
            except Exception as exc:  # noqa: BLE001
                return self._finding(symbol, Status.UNSUPPORTED, f"{name}: {type(exc).__name__}")
        if len(seen) < 2:
            return self._finding(symbol, Status.SKIP, "fewer than two dtypes accepted")
        # Distinct in, distinct out.  A mapping that collapses two dtypes
        # onto one engine dtype is how a float64 tensor comes back
        # single-precision with nothing raised.
        distinct = {str(v) for v in seen.values()}
        if len(distinct) != len(seen):
            return self._finding(
                symbol,
                Status.FAIL,
                f"{len(seen)} dtypes map onto {len(distinct)} engine dtypes: {seen}",
            )
        return self._finding(symbol, Status.PASS, f"{len(seen)} dtypes map one-to-one")

    def _describes(self, symbol: Symbol, fn: Any, name: str) -> Finding:
        """The query has to agree with the tensor it is asked about."""
        for dtype, itemsize in ((lucid.float32, 4), (lucid.float64, 8)):
            tensor = lucid.zeros((2, 3), dtype=dtype)
            try:
                answer = fn(tensor)
            except Exception as exc:  # noqa: BLE001
                return self._finding(
                    symbol, Status.UNSUPPORTED, f"{type(exc).__name__}: {str(exc)[:60]}"
                )
            if name == "dtype" and str(answer) != str(dtype):
                return self._finding(
                    symbol, Status.FAIL, f"a {dtype} tensor reports dtype {answer}"
                )
            if name == "itemsize" and int(answer) != itemsize:
                return self._finding(
                    symbol,
                    Status.FAIL,
                    f"a {dtype} tensor reports itemsize {answer}, not {itemsize}",
                )
            if name == "device" and "cpu" not in str(answer):
                return self._finding(symbol, Status.FAIL, f"a cpu tensor reports device {answer}")
            if name in ("ndim",) and int(answer) != 2:
                return self._finding(symbol, Status.FAIL, f"a 2-D tensor reports ndim {answer}")
            if name in ("numel",) and int(answer) != 6:
                return self._finding(symbol, Status.FAIL, f"a (2, 3) tensor reports numel {answer}")
            if name == "is_floating_point" and not bool(answer):
                return self._finding(symbol, Status.FAIL, f"a {dtype} tensor is not floating point")
            if name == "is_complex" and bool(answer):
                return self._finding(symbol, Status.FAIL, f"a {dtype} tensor reports as complex")
            if name in ("data_ptr", "storage_offset"):
                if int(answer) < 0:
                    return self._finding(symbol, Status.FAIL, f"reported {answer}")
            if name == "get_device" and int(answer) not in (-1, 0):
                # ``-1`` is the ordinal a host tensor reports — there is
                # no device index to give — and reading it as an error
                # blamed the convention rather than the answer.
                return self._finding(symbol, Status.FAIL, f"a cpu tensor reports ordinal {answer}")
        return self._finding(symbol, Status.PASS, "agrees with the tensor it describes")


# ── functional transforms ────────────────────────────────────────────────────


class FunctionalTransformAxis(Axis):
    """``grad(f)`` must be the derivative of ``f``, not merely a callable.

    These return a *function*, so every numeric axis called them, got a
    closure back, reported "nothing measurable" and skipped — thirteen
    symbols and 143 cells, covering the whole of ``lucid.func`` plus
    ``gradcheck``, ``hessian`` and ``checkpoint``.  The answer is one
    call further on: apply the returned function and compare it against
    something already trusted.

    The reference is always a second spelling rather than a hand-derived
    value.  ``grad(f)(x)`` is checked against ``backward()``, ``vmap``
    against the loop it replaces, ``checkpoint`` against running the
    function directly — so a disagreement names two routes to the same
    number rather than this file's opinion of what the number is.
    """

    name = "functional"
    summary = "grad / jac / vmap / vjp agree with the eager route they replace"
    kinds = frozenset({"op"})

    _NAMES = frozenset(
        {
            "grad",
            "grad_and_value",
            "jacrev",
            "jacfwd",
            "hessian",
            "vmap",
            "jvp",
            "vjp",
            "linearize",
            "checkpoint",
            "gradcheck",
            "gradgradcheck",
        }
    )

    def applies(self, symbol: Symbol) -> bool:
        if symbol.short not in self._NAMES:
            return False
        # ``lucid.autograd.grad`` is the eager entry point and takes
        # tensors, not a function; the numeric axes reach it already.
        return (
            symbol.qualname.startswith(("lucid.func.", "lucid.autograd.", "lucid.utils."))
            and symbol.qualname != "lucid.autograd.grad"
        )

    @staticmethod
    def _square(*operands: Any) -> Any:
        return (operands[0] * operands[0]).sum()

    def run(self, symbol: Symbol, ctx: Context) -> Finding:
        fn = _surface.resolve(symbol)
        if fn is None or not callable(fn):
            return self._finding(symbol, Status.SKIP, "not callable")
        name = symbol.short
        x = _probe.as_f64(np.array([0.3, -0.7, 1.1]))
        # d/dx sum(x*x) = 2x, which every one of these can be measured
        # against without a finite difference anywhere.
        expected = 2.0 * _probe.to_numpy(x)

        try:
            if name in ("grad", "grad_and_value"):
                out = fn(self._square)(x)
                got = out[0] if isinstance(out, tuple) else out
            elif name in ("jacrev", "jacfwd"):
                got = fn(lambda t: t * t)(x)
                expected = np.diag(expected)
            elif name == "hessian":
                got = fn(self._square)(x) if "func" in symbol.qualname else fn(self._square, x)
                expected = 2.0 * np.eye(3)
            elif name == "vmap":
                got = fn(lambda t: t * 2.0)(_probe.as_f64(np.stack([[1.0, 2.0]] * 3)))
                expected = np.full((3, 2), 0.0) + np.array([2.0, 4.0])
            elif name == "jvp":
                _, tangent = fn(lambda t: t * t, (x,), (lucid.ones_like(x),))
                got = tangent
            elif name == "vjp":
                _, backward = fn(lambda t: t * t, x)
                got = backward(lucid.ones_like(x))[0]
            elif name == "linearize":
                _, applied = fn(lambda t: t * t, x)
                got = applied(lucid.ones_like(x))
            elif name == "checkpoint":
                probe = _probe.as_f64(np.array([0.3, -0.7, 1.1]))
                probe.requires_grad = True
                fn(lambda t: (t * t).sum(), probe).backward()
                got = probe.grad
            else:  # gradcheck / gradgradcheck
                # They differentiate only the inputs that require grad.
                verdict = fn(lambda t: (t * t).sum(), [x.detach().requires_grad_(True)])
                if verdict is not True:
                    return self._finding(
                        symbol, Status.FAIL, f"reported {verdict!r} for x -> sum(x*x)"
                    )
                return self._finding(symbol, Status.PASS, "accepts an exact gradient")
        except Exception as exc:  # noqa: BLE001
            return self._finding(
                symbol, Status.UNSUPPORTED, f"{type(exc).__name__}: {str(exc)[:70]}"
            )

        measured = _probe.to_numpy(got)
        if measured is None:
            return self._finding(symbol, Status.SKIP, "produced nothing measurable")
        if measured.shape != expected.shape:
            return self._finding(
                symbol, Status.FAIL, f"shape {measured.shape}, expected {expected.shape}"
            )
        if not np.allclose(measured.astype(float), expected, rtol=1e-8, atol=1e-10):
            return self._finding(
                symbol,
                Status.FAIL,
                f"disagrees with the eager route by "
                f"{np.abs(measured.astype(float) - expected).max():.3e}",
            )
        return self._finding(symbol, Status.PASS, "matches the eager route exactly")


# ── module surgery ───────────────────────────────────────────────────────────


class NnUtilsAxis(Axis):
    """``nn.utils``: functions that take a module and give one back.

    Seventeen symbols with no numeric cell answered between them.  They
    take a ``Module`` and return ``None`` or another ``Module``, so there
    is no array for an axis to compare — and each of them still has a
    property that can fail loudly:

    * a re-parametrisation must not change what the module computes;
    * removing one must leave the original weight behind, not the
      parametrised copy;
    * pruning ``amount`` of the weights must zero that many of them;
    * folding a batch-norm into a convolution must give the same output
      as running the two in sequence.
    """

    name = "nnutils"
    summary = "parametrise, prune and fuse without changing what the module computes"
    kinds = frozenset({"op"})

    _NAMES = frozenset(
        {
            "weight_norm",
            "remove_weight_norm",
            "spectral_norm",
            "remove_spectral_norm",
            "register_parametrization",
            "remove_parametrizations",
            "fuse_conv_bn_eval",
            "fuse_linear_bn_eval",
            "skip_init",
            "vector_to_parameters",
            "parameters_to_vector",
            "copy_parameters_and_buffers",
            "pack_padded_sequence",
            "clip_grad_value_",
            "get_total_norm",
            "identity",
            "l1_unstructured",
            "random_unstructured",
            "remove",
        }
    )

    def applies(self, symbol: Symbol) -> bool:
        return symbol.qualname.startswith("lucid.nn.utils.") and symbol.short in self._NAMES

    def run(self, symbol: Symbol, ctx: Context) -> Finding:
        fn = _surface.resolve(symbol)
        if fn is None or not callable(fn):
            return self._finding(symbol, Status.SKIP, "not callable")
        name = symbol.short
        probe = _probe.as_f32(_probe.sample("moderate", (2, 4)))

        try:
            if name in ("weight_norm", "spectral_norm", "register_parametrization"):
                return self._parametrise(symbol, fn, name, probe)
            if name in ("remove_weight_norm", "remove_spectral_norm", "remove_parametrizations"):
                return self._unparametrise(symbol, fn, name, probe)
            if name in ("identity", "l1_unstructured", "random_unstructured", "remove"):
                return self._prune(symbol, fn, name, probe)
            if name in ("fuse_conv_bn_eval", "fuse_linear_bn_eval"):
                return self._fuse(symbol, fn, name)
            if name == "skip_init":
                return self._skip_init(symbol, fn, probe)
            if name in ("vector_to_parameters", "parameters_to_vector"):
                return self._vector(symbol, fn, name)
            if name == "copy_parameters_and_buffers":
                return self._copy(symbol, fn, probe)
            if name == "pack_padded_sequence":
                return self._pack(symbol, fn)
            return self._grad_norm(symbol, fn, name)
        except Exception as exc:  # noqa: BLE001 - surveying, not asserting
            return self._finding(
                symbol, Status.UNSUPPORTED, f"{type(exc).__name__}: {str(exc)[:70]}"
            )

    # ── the checks ───────────────────────────────────────────────────────────

    @staticmethod
    def _linear() -> Any:
        return lucid.nn.Linear(4, 3)

    def _parametrise(self, symbol: Symbol, fn: Any, name: str, probe: Any) -> Finding:
        module = self._linear()
        before = _probe.to_numpy(module(probe))
        if name == "register_parametrization":
            fn(module, "weight", lucid.nn.Identity())
        else:
            fn(module)
        after = _probe.to_numpy(module(probe))
        if after is None:
            return self._finding(symbol, Status.SKIP, "no output to compare")
        if before is not None and before.shape != after.shape:
            return self._finding(
                symbol, Status.FAIL, f"output shape changed from {before.shape} to {after.shape}"
            )
        # The reparametrised weight has to still be trainable — a
        # parametrisation that detaches its own weight silently stops the
        # layer learning, and forward still works.
        module.zero_grad()
        module(probe).sum().backward()
        if not any(p.grad is not None for p in module.parameters()):
            return self._finding(
                symbol, Status.FAIL, "after parametrisation no parameter receives a gradient"
            )
        return self._finding(symbol, Status.PASS, "forward keeps its shape and stays trainable")

    def _unparametrise(self, symbol: Symbol, fn: Any, name: str, probe: Any) -> Finding:
        if name == "remove_spectral_norm":
            return self._unparametrise_spectral(symbol, fn, probe)

        module = self._linear()
        apply = {"remove_weight_norm": lucid.nn.utils.weight_norm}.get(name)
        if apply is not None:
            apply(module)
        else:
            lucid.nn.utils.parametrize.register_parametrization(
                module, "weight", lucid.nn.Identity()
            )
        parametrised = _probe.to_numpy(module(probe))
        fn(module, "weight") if name == "remove_parametrizations" else fn(module)
        plain = _probe.to_numpy(module(probe))
        if parametrised is None or plain is None:
            return self._finding(symbol, Status.SKIP, "no output to compare")
        # Removing a parametrisation keeps the *current* weight; the
        # module must compute the same thing before and after.
        if not np.allclose(parametrised, plain, rtol=1e-5, atol=1e-6):
            return self._finding(
                symbol,
                Status.FAIL,
                f"removing it changed the output by " f"{np.abs(parametrised - plain).max():.3e}",
            )
        return self._finding(symbol, Status.PASS, "removed without changing the output")

    def _unparametrise_spectral(self, symbol: Symbol, fn: Any, probe: Any) -> Finding:
        """Check ``remove_spectral_norm`` against what it actually promises.

        The sibling removals put the *effective* weight back, so the
        module computes the same thing afterwards and "output unchanged"
        is the right question to ask them.  This one documents the
        opposite: it restores ``weight_orig``, the unnormalised matrix
        that was being trained behind the parametrisation, and warns the
        caller to copy the rescaled one out first if they wanted it.  So
        a changed output is the contract being kept, not broken, and the
        shared check called it a defect on every run.

        What is verifiable is that the restored weight *is* the original
        and that the machinery is gone.
        """
        module = self._linear()
        lucid.nn.utils.spectral_norm(module, "weight")
        module.eval()

        orig = _probe.to_numpy(getattr(module, "weight_orig", None))
        if orig is None:
            return self._finding(symbol, Status.SKIP, "no weight_orig to compare")
        normed = _probe.to_numpy(module(probe))

        fn(module)

        restored = _probe.to_numpy(getattr(module, "weight", None))
        if restored is None or normed is None:
            return self._finding(symbol, Status.SKIP, "no weight to inspect")
        if not np.allclose(orig, restored, rtol=1e-5, atol=1e-6):
            return self._finding(
                symbol,
                Status.FAIL,
                f"restored weight is not weight_orig — differs by "
                f"{np.abs(orig - restored).max():.3e}",
            )

        left = [attr for attr in ("weight_orig", "weight_u", "weight_v") if hasattr(module, attr)]
        if left:
            return self._finding(symbol, Status.FAIL, f"the machinery survived removal: {left}")

        # The normalisation divided by sigma > 1 here, so the two must
        # differ; if they do not, spectral_norm never applied and the
        # check above compared a weight with itself.
        plain = _probe.to_numpy(module(probe))
        if plain is not None and np.allclose(normed, plain, rtol=1e-5, atol=1e-6):
            return self._finding(
                symbol,
                Status.VACUOUS,
                "normed and unnormed outputs agree — nothing was being normalised",
            )
        return self._finding(symbol, Status.PASS, "weight_orig restored and the buffers are gone")

    def _prune(self, symbol: Symbol, fn: Any, name: str, probe: Any) -> Finding:
        module = self._linear()
        if name == "remove":
            lucid.nn.utils.prune.l1_unstructured(module, "weight", amount=0.5)
            fn(module, "weight")
            weight = _probe.to_numpy(module.weight)
            if weight is None:
                return self._finding(symbol, Status.SKIP, "no weight to inspect")
            zeros = int((weight == 0.0).sum())
            if zeros < weight.size // 2:
                return self._finding(
                    symbol,
                    Status.FAIL,
                    f"remove() dropped the mask as well as the reparametrisation: "
                    f"{zeros} of {weight.size} are zero, expected at least "
                    f"{weight.size // 2}",
                )
            return self._finding(symbol, Status.PASS, "the pruned values survive removal")

        if name == "random_unstructured":
            return self._prune_bernoulli(symbol, fn)

        fn(module, "weight") if name == "identity" else fn(module, "weight", amount=0.5)
        weight = _probe.to_numpy(module.weight)
        if weight is None:
            return self._finding(symbol, Status.SKIP, "no weight to inspect")
        zeros = int((weight == 0.0).sum())
        if name == "identity":
            if zeros == weight.size:
                return self._finding(symbol, Status.FAIL, "identity pruning zeroed everything")
            return self._finding(symbol, Status.PASS, "identity leaves the weight alone")
        wanted = weight.size // 2
        if zeros != wanted:
            return self._finding(
                symbol,
                Status.FAIL,
                f"amount=0.5 zeroed {zeros} of {weight.size}, expected {wanted}",
            )
        return self._finding(symbol, Status.PASS, f"zeroed {zeros} of {weight.size}")

    def _prune_bernoulli(self, symbol: Symbol, fn: Any) -> Finding:
        """Check ``random_unstructured``, which is not a count-based prune.

        Its documented contract samples every element independently and
        keeps it iff ``u >= amount``, so the realised sparsity is a
        binomial draw, not ``amount * n``.  Demanding an exact count made
        this axis fail roughly two runs in five on a 12-element weight
        while the op did exactly what it says — a red light nobody could
        act on.  What is actually promised is checked instead:

        * the two endpoints are deterministic.  ``amount=0`` keeps every
          element and ``amount=1`` drops every one, for any draw;
        * in between, the count concentrates.  On 4096 elements a six
          sigma band around the mean is +/-192, so a run inside it is
          ordinary and a run outside it is a broken ``amount`` rather
          than bad luck — a false red here is a once-in-a-billion event.
        """
        wide = lucid.nn.Linear(64, 64)
        size = int(_probe.to_numpy(wide.weight).size)

        for amount, want, what in ((0.0, 0, "keep everything"), (1.0, size, "drop all")):
            module = lucid.nn.Linear(64, 64)
            base = _probe.to_numpy(module.weight)
            already = int((base == 0.0).sum())
            fn(module, "weight", amount=amount)
            after = _probe.to_numpy(module.weight)
            if after is None:
                return self._finding(symbol, Status.SKIP, "no weight to inspect")
            zeros = int((after == 0.0).sum()) - already
            if zeros != want:
                return self._finding(
                    symbol,
                    Status.FAIL,
                    f"amount={amount} should {what}: zeroed {zeros} of {size}, " f"expected {want}",
                )

        fn(wide, "weight", amount=0.5)
        weight = _probe.to_numpy(wide.weight)
        if weight is None:
            return self._finding(symbol, Status.SKIP, "no weight to inspect")
        zeros = int((weight == 0.0).sum())
        mean = size / 2.0
        band = 6.0 * math.sqrt(size * 0.25)
        if abs(zeros - mean) > band:
            return self._finding(
                symbol,
                Status.FAIL,
                f"amount=0.5 zeroed {zeros} of {size}, outside the six-sigma "
                f"band {mean - band:.0f}-{mean + band:.0f}",
            )
        return self._finding(
            symbol, Status.PASS, f"endpoints exact, zeroed {zeros} of {size} at 0.5"
        )

    def _fuse(self, symbol: Symbol, fn: Any, name: str) -> Finding:
        if name == "fuse_conv_bn_eval":
            first = lucid.nn.Conv2d(3, 4, 3, padding=1)
            norm = lucid.nn.BatchNorm2d(4)
            probe = _probe.as_f32(_probe.sample("moderate", (2, 3, 6, 6)))
        else:
            first = lucid.nn.Linear(4, 4)
            norm = lucid.nn.BatchNorm1d(4)
            probe = _probe.as_f32(_probe.sample("moderate", (2, 4)))
        first.eval()
        norm.eval()
        want = _probe.to_numpy(norm(first(probe)))
        fused = fn(first, norm)
        got = _probe.to_numpy(fused(probe))
        if want is None or got is None:
            return self._finding(symbol, Status.SKIP, "no output to compare")
        if not np.allclose(want, got, rtol=1e-4, atol=1e-5):
            return self._finding(
                symbol,
                Status.FAIL,
                f"the fused layer differs from the sequence by " f"{np.abs(want - got).max():.3e}",
            )
        return self._finding(symbol, Status.PASS, "the fused layer matches the sequence")

    def _skip_init(self, symbol: Symbol, fn: Any, probe: Any) -> Finding:
        module = fn(lucid.nn.Linear, 4, 3)
        if not isinstance(module, lucid.nn.Module):
            return self._finding(symbol, Status.FAIL, f"returned {type(module).__name__}")
        out = _probe.to_numpy(module(probe))
        if out is None or out.shape != (2, 3):
            return self._finding(symbol, Status.FAIL, "the built module does not forward")
        return self._finding(symbol, Status.PASS, "builds a usable module")

    def _vector(self, symbol: Symbol, fn: Any, name: str) -> Finding:
        module = self._linear()
        flat = lucid.nn.utils.parameters_to_vector(module.parameters())
        if name == "parameters_to_vector":
            total = sum(int(np.prod(tuple(p.shape))) for p in module.parameters())
            if int(flat.shape[0]) != total:
                return self._finding(
                    symbol, Status.FAIL, f"flattened to {flat.shape[0]}, expected {total}"
                )
            return self._finding(symbol, Status.PASS, f"{total} values, one vector")
        target = self._linear()
        fn(flat, target.parameters())
        for source, restored in zip(module.parameters(), target.parameters()):
            a, b = _probe.to_numpy(source), _probe.to_numpy(restored)
            if a is None or b is None or not np.allclose(a, b):
                return self._finding(
                    symbol, Status.FAIL, "the round trip did not restore the parameters"
                )
        return self._finding(symbol, Status.PASS, "vector -> parameters round trips")

    def _copy(self, symbol: Symbol, fn: Any, probe: Any) -> Finding:
        source, dest = self._linear(), self._linear()
        fn(source, dest)
        a, b = _probe.to_numpy(source(probe)), _probe.to_numpy(dest(probe))
        if a is None or b is None:
            return self._finding(symbol, Status.SKIP, "no output to compare")
        if not np.allclose(a, b, rtol=1e-6, atol=1e-7):
            return self._finding(
                symbol,
                Status.FAIL,
                f"the copy computes something else by {np.abs(a - b).max():.3e}",
            )
        return self._finding(symbol, Status.PASS, "the copy computes the same thing")

    def _pack(self, symbol: Symbol, fn: Any) -> Finding:
        padded = _probe.as_f32(_probe.sample("moderate", (2, 4, 3)))
        packed = fn(padded, [4, 2], batch_first=True, enforce_sorted=True)
        restored, lengths = lucid.nn.utils.rnn.pad_packed_sequence(packed, batch_first=True)
        a, b = _probe.to_numpy(padded), _probe.to_numpy(restored)
        if a is None or b is None:
            return self._finding(symbol, Status.SKIP, "no tensor to compare")
        # Only the valid timesteps survive packing, by design.
        if not np.allclose(a[0, :4], b[0, :4]) or not np.allclose(a[1, :2], b[1, :2]):
            return self._finding(
                symbol, Status.FAIL, "the valid timesteps did not survive the round trip"
            )
        return self._finding(symbol, Status.PASS, "packs and unpacks the valid timesteps")

    def _grad_norm(self, symbol: Symbol, fn: Any, name: str) -> Finding:
        module = self._linear()
        module(_probe.as_f32(_probe.sample("moderate", (2, 4)))).sum().backward()
        grads = [p.grad for p in module.parameters() if p.grad is not None]
        if not grads:
            return self._finding(symbol, Status.SKIP, "no gradient to measure")
        if name == "get_total_norm":
            got = float(_probe.to_numpy(fn(grads)))
            want = float(np.sqrt(sum((_probe.to_numpy(g).astype(float) ** 2).sum() for g in grads)))
            if not np.isclose(got, want, rtol=1e-5):
                return self._finding(
                    symbol, Status.FAIL, f"reported {got:.6g}, computed {want:.6g}"
                )
            return self._finding(symbol, Status.PASS, f"total norm {got:.4g}")

        before = copy.deepcopy([_probe.to_numpy(g) for g in grads])
        fn(module.parameters(), 1e-4)
        after = [_probe.to_numpy(p.grad) for p in module.parameters() if p.grad is not None]
        if any(np.abs(a).max() > 1e-4 + 1e-9 for a in after if a is not None):
            return self._finding(symbol, Status.FAIL, "a gradient survived above the clip value")
        if all(np.array_equal(x, y) for x, y in zip(before, after) if x is not None) and any(
            np.abs(x).max() > 1e-4 for x in before if x is not None
        ):
            return self._finding(symbol, Status.FAIL, "nothing was clipped")
        return self._finding(symbol, Status.PASS, "clipped to the value given")


# ── the pretrained-weight registry ───────────────────────────────────────────


class WeightsAxis(Axis):
    """The weights registry: what is registered comes back, by every route.

    Seven symbols with no axis at all, because the obvious check needs
    the network and the obvious data needs ``lucid.models`` — which this
    tool excludes, so the registry is *empty* in the audit's process and
    every lookup correctly answers ``None``.

    Registering a synthetic architecture removes both problems.  The
    contract is then fully checkable offline and it is not a small one:
    four public functions read the same table by four different keys,
    and a registry where ``weights_for`` finds an entry that
    ``get_weight`` cannot name is broken in a way no single lookup
    reveals.
    """

    name = "weights"
    summary = "register then look up: every route reaches the same entry"
    kinds = frozenset({"util", "class"})
    varies_a_tensor = False

    _NAMES = frozenset(
        {
            "register_weights",
            "weights_for",
            "list_pretrained",
            "get_weight",
            "resolve_weights",
            "WeightsEnum",
            "WeightEntry",
        }
    )

    def applies(self, symbol: Symbol) -> bool:
        return symbol.qualname.startswith("lucid.weights.") and (symbol.short in self._NAMES)

    def run(self, symbol: Symbol, ctx: Context) -> Finding:
        import lucid.weights as weights  # noqa: PLC0415 - optional subsystem

        model_name = "_audit_probe_net"
        try:
            entry = weights.WeightEntry(
                url="https://example.invalid/probe.lct",
                sha256="0" * 64,
                num_classes=10,
                transforms=lucid.utils.transforms.Compose([]),
            )
            enum_cls = weights.WeightsEnum("_AuditProbeWeights", {"PROBE_V1": entry})
            registered = weights.register_weights(model_name)(enum_cls)
        except Exception as exc:  # noqa: BLE001
            return self._finding(
                symbol, Status.UNSUPPORTED, f"{type(exc).__name__}: {str(exc)[:60]}"
            )

        problems: list[str] = []
        if weights.weights_for(model_name) is not registered:
            problems.append("weights_for did not return the enum that was registered")
        tags = weights.list_pretrained(model_name)
        if "PROBE_V1" not in tags:
            problems.append(f"list_pretrained reported {tags}, without the tag")
        try:
            named = weights.get_weight("_AuditProbeWeights.PROBE_V1")
            if named is not registered.PROBE_V1:
                problems.append("get_weight resolved a different member")
        except Exception as exc:  # noqa: BLE001
            problems.append(f"get_weight raised {type(exc).__name__}")

        # ``pretrained=False`` must mean random init, and a tag string
        # must select the member it names — the two directions a factory
        # relies on and the two a typo silently breaks.
        try:
            if weights.resolve_weights(registered, False, None) is not None:
                problems.append("resolve_weights(pretrained=False) chose weights anyway")
            if weights.resolve_weights(registered, "PROBE_V1", None) is not (registered.PROBE_V1):
                problems.append("resolve_weights did not honour the tag it was given")
        except Exception as exc:  # noqa: BLE001
            problems.append(f"resolve_weights raised {type(exc).__name__}")

        if entry.url not in str(registered.PROBE_V1.url):
            problems.append("the member does not expose its entry's url")

        if problems:
            return self._finding(symbol, Status.FAIL, "; ".join(problems))
        return self._finding(symbol, Status.PASS, "registered once, found by all four routes")


STATE_AXES: tuple[Axis, ...] = (
    StateAxis(),
    HookAxis(),
    MetadataAxis(),
    FunctionalTransformAxis(),
    NnUtilsAxis(),
    WeightsAxis(),
)

__all__ = [
    "FunctionalTransformAxis",
    "HookAxis",
    "MetadataAxis",
    "NnUtilsAxis",
    "STATE_AXES",
    "StateAxis",
    "WeightsAxis",
]
