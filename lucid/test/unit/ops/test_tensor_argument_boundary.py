"""A tensor parameter refuses a non-tensor with ``TypeError``, before anything reads it (LCD-731).

``None`` reached a pybind11 ``TensorImpl`` parameter, which takes it as a null
pointer: ``lucid.cummax(None, 0)``, ``lucid.ravel(None)`` and the like ended
the process with SIGSEGV.  A composite met it at its first ``x.ndim`` and
raised ``AttributeError``; others let the engine raise ``LucidError: null
input tensor``.  The reference raises ``TypeError`` for every one of them.

One batch of calls runs per child process, since a failure here can be a
signal that takes the interpreter with it.  The child reports each call before
making it, so a crash names the call that caused it.  Four batches, four
children, so the whole file costs a few seconds:

* ``table`` — an explicit table of functions (free functions, ``nn.functional``,
  ``linalg``, ``special``, ``einops``) with a bad value in each tensor slot;
* ``registry-functions`` / ``registry-methods`` — every op the registry turns
  into a free function or a Tensor method;
* ``guarded`` — every function the namespaces guard from its annotations, a bad
  value in each of its tensor parameters.

A refusal is a ``TypeError`` that names ``Tensor`` and the type it got.
"""

import inspect
import json
import pickle
import re
import subprocess
import sys
import types
from collections.abc import Sequence
from pathlib import Path

import pytest

import lucid
import lucid.einops
import lucid.nn
from lucid import Tensor
from lucid._dispatch import _unwrap_or_scalar
from lucid._tensor_args import _slots_of, guard_functions
from lucid._types import TensorLike

#: A bad value for a tensor slot: the one that crashes, and one wrong type.
_BAD = (None, "bad")

#: What a refusal reads like, whichever layer raised it.
_SHAPE = re.compile(
    r"must be Tensor|expected Tensor|must be a sequence of Tensors|Expected Tensor"
    r"|must be Tensor or array-like"
)

#: ``$f`` a float matrix, ``$v`` a vector, ``$i`` an index vector, ``$b`` a mask,
#: ``$sq`` a square matrix, ``$x4`` an image batch, ``$w4`` conv weights;
#: ``$list`` a list of two float matrices, ``$like`` a float matrix where
#: array-likes are accepted too (only ``None`` is refused).  A token that
#: begins ``$`` is a tensor slot, and each is made bad in turn; ``=f`` is a
#: float matrix that is left alone.
_TABLE: list[tuple[str, list[object], dict[str, object]]] = [
    # the four the card names, and what the fork sweep found beside them
    ("lucid.cummax", ["$f", 0], {}),
    ("lucid.cummin", ["$f", 0], {}),
    ("lucid.ravel", ["$f"], {}),
    ("lucid.prod", ["$f"], {}),
    ("lucid.sum", ["$f"], {}),
    ("lucid.square", ["$f"], {}),
    ("lucid.exp", ["$f"], {}),
    ("lucid.abs", ["$f"], {}),
    ("lucid.sqrt", ["$f"], {}),
    ("lucid.neg", ["$f"], {}),
    ("lucid.relu", ["$f"], {}),
    ("lucid.sigmoid", ["$f"], {}),
    ("lucid.tanh", ["$f"], {}),
    ("lucid.flatten", ["$f"], {}),
    ("lucid.reshape", ["$f", [3, 2]], {}),
    ("lucid.transpose", ["$f", 0, 1], {}),
    ("lucid.squeeze", ["$f"], {}),
    ("lucid.cumsum", ["$f", 0], {}),
    ("lucid.cumprod", ["$f", 0], {}),
    ("lucid.mean", ["$f"], {}),
    ("lucid.std", ["$f"], {}),
    ("lucid.var", ["$f"], {}),
    ("lucid.max", ["$f"], {}),
    ("lucid.min", ["$f"], {}),
    ("lucid.amax", ["$f", 0], {}),
    ("lucid.argmax", ["$f"], {}),
    ("lucid.sort", ["$f"], {}),
    ("lucid.topk", ["$f", 1], {}),
    ("lucid.softmax", ["$f", -1], {}),
    ("lucid.log_softmax", ["$f", -1], {}),
    ("lucid.add", ["$f", "$f"], {}),
    ("lucid.mul", ["$f", "$f"], {}),
    ("lucid.maximum", ["$f", "$f"], {}),
    ("lucid.matmul", ["$f", "$sq"], {}),
    ("lucid.mm", ["$f", "$sq"], {}),
    ("lucid.bmm", ["$f", "$f"], {}),
    ("lucid.kron", ["$f", "$f"], {}),
    ("lucid.tensordot", ["$f", "$f"], {}),
    ("lucid.where", ["$b", "=f", "=f"], {}),
    ("lucid.cat", ["$list"], {}),
    ("lucid.stack", ["$list"], {}),
    ("lucid.dstack", ["$list"], {}),
    ("lucid.hstack", ["$list"], {}),
    ("lucid.vstack", ["$list"], {}),
    ("lucid.column_stack", ["$list"], {}),
    ("lucid.block_diag", ["$f", "$f"], {}),
    ("lucid.meshgrid", ["$v", "$v"], {}),
    ("lucid.cartesian_prod", ["$v", "$v"], {}),
    ("lucid.atleast_2d", ["$f"], {}),
    ("lucid.take_along_dim", ["$f", "$i", 0], {}),
    ("lucid.lerp", ["$f", "$f", 0.5], {}),
    ("lucid.addcmul", ["$f", "$f", "$f"], {}),
    ("lucid.addcdiv", ["$f", "$f", "$f"], {}),
    ("lucid.masked_select", ["$f", "$b"], {}),
    ("lucid.masked_fill", ["$f", "$b", 1.0], {}),
    ("lucid.masked_scatter", ["$f", "$b", "$f"], {}),
    ("lucid.scatter", ["$f", 0, "$i", "$f"], {}),
    ("lucid.scatter_add", ["$f", 0, "$i", "$f"], {}),
    ("lucid.scatter_reduce", ["$f", 0, "$i", "$f", "sum"], {}),
    ("lucid.index_add", ["$f", 0, "$i", "$f"], {}),
    ("lucid.index_select", ["$f", 0, "$i"], {}),
    ("lucid.gather", ["$f", 0, "$i"], {}),
    ("lucid.isin", ["$like", "$like"], {}),
    ("lucid.equal", ["$f", "$f"], {}),
    ("lucid.allclose", ["$f", "$f"], {}),
    ("lucid.roll", ["$f", 1], {}),
    ("lucid.tril", ["$f"], {}),
    ("lucid.trace", ["$sq"], {}),
    ("lucid.cov", ["$f"], {}),
    ("lucid.quantile", ["$f", 0.5], {}),
    ("lucid.bincount", ["$i"], {}),
    ("lucid.histc", ["$f"], {}),
    ("lucid.unique", ["$f"], {}),
    ("lucid.nonzero", ["$f"], {}),
    ("lucid.clip", ["$f", 0.0, 1.0], {}),
    ("lucid.real", ["$f"], {}),
    # nn.functional
    ("lucid.nn.functional.relu", ["$f"], {}),
    ("lucid.nn.functional.gelu", ["$f"], {}),
    ("lucid.nn.functional.softmax", ["$f"], {}),
    ("lucid.nn.functional.linear", ["$f", "$sq"], {}),
    ("lucid.nn.functional.conv1d", ["$x3", "$w3"], {}),
    ("lucid.nn.functional.conv2d", ["$x4", "$w4"], {}),
    ("lucid.nn.functional.conv_transpose2d", ["$x4", "$w4"], {}),
    ("lucid.nn.functional.max_pool2d", ["$x4", 2], {}),
    ("lucid.nn.functional.avg_pool2d", ["$x4", 2], {}),
    ("lucid.nn.functional.layer_norm", ["$f", [3]], {}),
    ("lucid.nn.functional.group_norm", ["$x4", 1], {}),
    ("lucid.nn.functional.batch_norm", ["$f"], {}),
    ("lucid.nn.functional.dropout", ["$f"], {}),
    ("lucid.nn.functional.mse_loss", ["$f", "$f"], {}),
    ("lucid.nn.functional.l1_loss", ["$f", "$f"], {}),
    ("lucid.nn.functional.cross_entropy", ["$f", "$i"], {}),
    ("lucid.nn.functional.nll_loss", ["$f", "$i"], {}),
    ("lucid.nn.functional.binary_cross_entropy", ["$f", "$f"], {}),
    ("lucid.nn.functional.kl_div", ["$f", "$f"], {}),
    ("lucid.nn.functional.embedding", ["$i", "$f"], {}),
    ("lucid.nn.functional.pad", ["$f", [1, 1]], {}),
    ("lucid.nn.functional.interpolate", ["$x4"], {"scale_factor": 2.0}),
    ("lucid.nn.functional.pixel_shuffle", ["$x4", 1], {}),
    ("lucid.nn.functional.unfold", ["$x4", 2], {}),
    ("lucid.nn.functional.one_hot", ["$i"], {}),
    ("lucid.nn.functional.scaled_dot_product_attention", ["$f", "$f", "$f"], {}),
    ("lucid.nn.functional.cosine_similarity", ["$f", "$f"], {}),
    ("lucid.nn.functional.pdist", ["$f"], {}),
    # linalg, special, einops
    ("lucid.linalg.inv", ["$sq"], {}),
    ("lucid.linalg.det", ["$sq"], {}),
    ("lucid.linalg.pinv", ["$f"], {}),
    ("lucid.linalg.matrix_exp", ["$sq"], {}),
    ("lucid.linalg.matrix_rank", ["$f"], {}),
    ("lucid.linalg.cond", ["$sq"], {}),
    ("lucid.special.expit", ["$f"], {}),
    ("lucid.special.entr", ["$f"], {}),
    ("lucid.special.bessel_j0", ["$f"], {}),
    ("lucid.einops.rearrange", ["$f", "a b -> b a"], {}),
    ("lucid.einops.parse_shape", ["$f", "a b"], {}),
    # a method's operands after ``self``
    ("Tensor.masked_select", ["$f", "$b"], {}),
    ("Tensor.mm", ["$f", "$sq"], {}),
    ("Tensor.masked_scatter", ["$f", "$b", "$f"], {}),
    ("Tensor.take_along_dim", ["$f", "$i", 0], {}),
]


def _resolve(path: str) -> object:
    import lucid

    head, *rest = path.split(".")
    obj: object = lucid.Tensor if head == "Tensor" else lucid
    for name in rest:
        obj = getattr(obj, name)
    return obj


def _value(token: object) -> object:
    import lucid

    if not (isinstance(token, str) and token[:1] in "$="):
        return token
    return {
        "$f": lambda: lucid.randn(2, 3),
        "$v": lambda: lucid.randn(3),
        "$i": lambda: lucid.tensor([0, 1]),
        "$b": lambda: lucid.ones(2, 3).to(dtype=lucid.bool),
        "$sq": lambda: lucid.randn(3, 3),
        "$x3": lambda: lucid.randn(1, 2, 5),
        "$w3": lambda: lucid.randn(2, 2, 2),
        "$x4": lambda: lucid.randn(1, 2, 4, 4),
        "$w4": lambda: lucid.randn(2, 2, 2, 2),
        "$list": lambda: [lucid.randn(2, 3), lucid.randn(2, 3)],
    }[token.replace("=", "$").replace("$like", "$f")]()


def _bad_forms(token: object) -> list[object]:
    """The bad values for the slot ``token`` stands for."""
    if token == "$list":
        import lucid

        return [[lucid.randn(2, 3), None], "bad"]
    if token == "$like":
        return [None]  # anything else is for the converter to judge
    return list(_BAD)


def _emit(out: object, **record: object) -> None:
    print(json.dumps(record), file=out, flush=True)  # type: ignore[call-overload]


def _attempt(out: object, case: str, call: object) -> None:
    """Make ``call()`` and report whether it refused as a ``TypeError`` that names a Tensor."""
    _emit(out, kind="begin", case=case)
    try:
        call()  # type: ignore[operator]
    except TypeError as exc:
        text = str(exc)
        ok = type(exc) is TypeError and bool(_SHAPE.search(text))
        _emit(out, kind="done", case=case, ok=ok, got=f"TypeError: {text[:140]}")
    except BaseException as exc:  # a report, whatever it is
        _emit(out, kind="done", case=case, ok=False, got=f"{type(exc).__name__}: {str(exc)[:140]}")
    else:
        _emit(out, kind="done", case=case, ok=False, got="returned normally")


def _batch_table(out: object) -> None:
    for path, args, kwargs in _TABLE:
        fn = _resolve(path)
        slots = [i for i, a in enumerate(args) if isinstance(a, str) and a.startswith("$")]
        if path.startswith("Tensor."):
            slots = slots[1:]  # ``self`` is a tensor by construction
        for slot in slots:
            for bad in _bad_forms(args[slot]):
                call_args = [_value(a) for a in args]
                call_args[slot] = bad
                _attempt(out, f"{path} slot {slot} <- {bad!r}"[:90], lambda: fn(*call_args, **kwargs))  # type: ignore[operator]


def _batch_registry_functions(out: object) -> None:
    import lucid
    from lucid._ops._registry import _REGISTRY

    seen: set[str] = set()
    for entry in _REGISTRY:
        name = entry.free_fn_name or entry.name
        if name in seen or not hasattr(lucid, name) or entry.n_tensor_args == 0:
            continue
        seen.add(name)
        fn = getattr(lucid, name)
        if entry.n_tensor_args == -1:
            for bad in _bad_forms("$list"):
                _attempt(out, f"lucid.{name}(list) <- {bad!r}"[:90], lambda: fn(bad))
            continue
        for slot in range(entry.n_tensor_args):
            for bad in _BAD:
                args = [lucid.randn(2, 3) for _ in range(slot)] + [bad]
                _attempt(out, f"lucid.{name} slot {slot} <- {bad!r}", lambda: fn(*args))


def _batch_registry_methods(out: object) -> None:
    import lucid
    from lucid._ops._registry import _REGISTRY

    seen: set[str] = set()
    for entry in _REGISTRY:
        name = entry.method_name
        if not name or name in seen or not hasattr(lucid.Tensor, name) or entry.n_tensor_args < 2:
            continue
        seen.add(name)
        for slot in range(1, entry.n_tensor_args):
            for bad in _BAD:
                x = lucid.randn(2, 3)
                args = [lucid.randn(2, 3) for _ in range(slot - 1)] + [bad]
                _attempt(
                    out, f"Tensor.{name} slot {slot} <- {bad!r}", lambda: getattr(x, name)(*args)
                )


def _batch_guarded(out: object) -> None:
    import importlib

    import lucid
    from lucid import _tensor_args as guard

    namespaces = ("lucid", "lucid.nn.functional", "lucid.linalg", "lucid.special", "lucid.einops")
    for ns_name in namespaces:
        ns = importlib.import_module(ns_name)
        for name in sorted(dir(ns)):
            fn = getattr(ns, name)
            if not getattr(fn, guard._MARK, False):
                continue
            for slot in guard._slots_of(fn.__wrapped__):
                if isinstance(slot, guard._One) and slot.optional:
                    continue
                if isinstance(slot, guard._Like) or (
                    isinstance(slot, guard._Many) and slot.optional
                ):
                    continue
                bads = _bad_forms("$list") if isinstance(slot, guard._Many) else list(_BAD)
                for bad in bads:
                    filler = lucid.randn(2, 3)
                    if slot.position is None:
                        args: list[object] = []
                        kwargs: dict[str, object] = {slot.name: bad}
                    else:
                        args = [filler] * slot.position + [bad]
                        kwargs = {}
                    _attempt(
                        out,
                        f"{ns_name}.{name} {slot.name} <- {bad!r}"[:90],
                        lambda: fn(*args, **kwargs),
                    )


_BATCHES = {
    "table": _batch_table,
    "registry-functions": _batch_registry_functions,
    "registry-methods": _batch_registry_methods,
    "guarded": _batch_guarded,
}

#: Fewer attempts than this and the batch is not sweeping what it claims to.
_FLOOR = {"table": 300, "registry-functions": 400, "registry-methods": 60, "guarded": 400}


def _run_batch(name: str) -> tuple[int, list[dict[str, object]], str]:
    done = subprocess.run(
        [sys.executable, str(Path(__file__)), name], capture_output=True, text=True, timeout=300
    )
    records = [json.loads(line) for line in done.stdout.splitlines() if line.startswith("{")]
    return done.returncode, records, done.stderr[-400:]


@pytest.mark.parametrize("batch", list(_BATCHES))
def test_a_bad_tensor_argument_is_a_type_error(batch: str) -> None:
    code, records, err = _run_batch(batch)
    began = [r["case"] for r in records if r["kind"] == "begin"]
    finished = {r["case"]: r for r in records if r["kind"] == "done"}

    crashed = [c for c in began if c not in finished]
    assert not crashed, f"the process died (exit {code}) in {crashed[-1]}\n{err}"
    assert code == 0, f"the batch exited {code}\n{err}"

    wrong = [f"{c}: {r['got']}" for c, r in finished.items() if not r["ok"]]
    assert not wrong, f"{len(wrong)} call(s) did not refuse with a TypeError naming Tensor:\n" + (
        "\n".join(wrong[:40])
    )
    assert len(finished) >= _FLOOR[batch], f"only {len(finished)} calls were made"


def test_the_crashing_ops_the_card_names_are_in_the_table() -> None:
    names = {path for path, _, _ in _TABLE}
    assert {"lucid.cummax", "lucid.cummin", "lucid.ravel", "lucid.prod", "lucid.sum"} <= names


def _warning_sites() -> list[tuple[str, object]]:
    import lucid
    import lucid.nn.functional as F

    x = lucid.zeros(1, 1, 2, 2)
    return [
        ("softmax-implicit-dim", lambda: F.softmax(lucid.zeros(2, 3))),
        ("upsample-alias", lambda: F.upsample_nearest(x, scale_factor=2)),
        ("gumbel-eps", lambda: F.gumbel_softmax(lucid.zeros(2, 3), eps=1e-3)),
        ("mse-broadcast", lambda: F.mse_loss(lucid.zeros(2, 3), lucid.zeros(3))),
    ]


@pytest.mark.parametrize("site", [name for name, _ in _warning_sites()])
def test_the_guard_does_not_move_a_warning_off_the_caller(site: str) -> None:
    """The guard is a frame between the caller and the function, so a warning
    counted by ``stacklevel`` must skip it (``GUARD_FRAMES``)."""
    call = dict(_warning_sites())[site]
    with pytest.warns(UserWarning) as record:
        call()  # type: ignore[operator]
    assert record[0].filename == __file__


def _guarded_functions() -> list[types.FunctionType]:
    """Every guarded function the public namespaces hold, and the Tensor class's."""
    import importlib

    import lucid
    from lucid import _tensor_args as guard

    found: dict[int, types.FunctionType] = {}
    for ns_name in (
        "lucid",
        "lucid.nn.functional",
        "lucid.linalg",
        "lucid.special",
        "lucid.einops",
    ):
        ns = importlib.import_module(ns_name)
        for name in getattr(ns, "__all__", ()):
            fn = getattr(ns, name, None)
            if isinstance(fn, types.FunctionType) and getattr(fn, guard._MARK, False):
                found[id(fn)] = fn
    for fn in vars(lucid.Tensor).values():
        if isinstance(fn, types.FunctionType) and getattr(fn, guard._MARK, False):
            found[id(fn)] = fn
    return list(found.values())


def _found_by_name(fn: types.FunctionType) -> object:
    """What ``pickle`` finds when it looks ``fn`` up by ``__module__`` and ``__qualname__``."""
    found: object = sys.modules[fn.__module__]
    for part in fn.__qualname__.split("."):
        found = getattr(found, part, None)
    return found


def test_a_guarded_function_is_the_one_its_module_names() -> None:
    """One function, one guard: the defining module holds the very object a
    package re-exports, so ``pickle`` (which checks exactly that) accepts it."""
    guarded = _guarded_functions()
    assert len(guarded) > 400
    assert [f.__qualname__ for f in guarded if _found_by_name(f) is not f] == []


def test_a_guarded_function_survives_a_pickle_round_trip() -> None:
    guarded = _guarded_functions()
    broken = []
    for fn in guarded:
        try:
            ok = pickle.loads(pickle.dumps(fn)) is fn
        except pickle.PicklingError:
            ok = False
        if not ok:
            broken.append(f"{fn.__module__}.{fn.__qualname__}")
    assert broken == []


class _HoldsAFunction(lucid.nn.Module):
    """A module that keeps a public function as an attribute."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = lucid.nn.Linear(2, 2)
        self.act = lucid.nn.functional.relu

    def forward(self, x: lucid.Tensor) -> lucid.Tensor:
        return self.act(self.lin(x))


def test_a_module_holding_a_public_function_saves_and_loads(tmp_path: Path) -> None:
    module = _HoldsAFunction()
    lucid.save(module, tmp_path / "m.lucid")
    loaded = lucid.load(tmp_path / "m.lucid", weights_only=False)
    assert loaded.act is lucid.nn.functional.relu
    x = lucid.randn(3, 2)
    assert loaded(x).tolist() == module(x).tolist()


# ── the in-process half: what a child process cannot report to coverage ──────


def _probe_rest(*tensors: Tensor) -> int:
    return len(tensors)


def _probe_many(group: Sequence[Tensor] | None = None) -> int:
    return 0 if group is None else len(group)


def _probe_like(first: Tensor | TensorLike, second: Tensor | TensorLike | None = None) -> int:
    return 1


def _probe_keyword_only(*, mask: Tensor) -> int:
    return 1


_PROBES = ("_probe_rest", "_probe_many", "_probe_like", "_probe_keyword_only")
guard_functions(sys.modules[__name__], _PROBES)


def test_a_variadic_tensor_parameter_names_the_element_it_refused() -> None:
    x = lucid.ones(2)
    assert _probe_rest(x, x) == 2
    with pytest.raises(TypeError, match="expected Tensor as element 1 in argument 'tensors'"):
        _probe_rest(x, None)  # type: ignore[arg-type]  # the refusal under test


def test_a_sequence_parameter_is_refused_unless_it_is_a_sequence_of_tensors() -> None:
    x = lucid.ones(2)
    assert _probe_many() == 0
    assert _probe_many(None) == 0
    assert _probe_many((x, x)) == 2
    with pytest.raises(TypeError, match="must be a sequence of Tensors, not str"):
        _probe_many("abc")  # type: ignore[arg-type]  # the refusal under test
    with pytest.raises(TypeError, match="expected Tensor as element 1 in argument 0"):
        _probe_many([x, None])  # type: ignore[list-item]  # the refusal under test


def test_a_tensor_like_parameter_refuses_none_only_by_position_or_by_name() -> None:
    x = lucid.ones(2)
    assert _probe_like(x) == 1
    assert _probe_like([1.0, 2.0], None) == 1
    with pytest.raises(TypeError, match=r"argument 'first' \(position 1\) must be Tensor or array"):
        _probe_like(None)  # type: ignore[arg-type]  # the refusal under test
    with pytest.raises(TypeError, match=r"argument 'first' must be Tensor or array-like"):
        _probe_like(first=None)  # type: ignore[arg-type]  # the refusal under test


def test_a_keyword_only_tensor_parameter_is_found_by_its_name() -> None:
    assert _probe_keyword_only(mask=lucid.ones(2)) == 1
    with pytest.raises(TypeError, match=r"argument 'mask' must be Tensor, not str"):
        _probe_keyword_only(mask="x")  # type: ignore[arg-type]  # the refusal under test


def test_a_function_without_a_findable_home_is_still_guarded() -> None:
    def local(x: Tensor) -> Tensor:
        return x

    home = types.SimpleNamespace(local=local)
    guard_functions(home, ["local"])
    assert home.local is not local
    with pytest.raises(TypeError, match="must be Tensor"):
        home.local(None)


def test_a_signature_that_cannot_be_read_guards_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(*args: object, **kwargs: object) -> inspect.Signature:
        raise ValueError("no signature")

    monkeypatch.setattr(inspect, "signature", refuse)
    assert _slots_of(print) == ()


def test_meshgrid_takes_only_the_two_indexings_the_reference_names() -> None:
    a, b = lucid.arange(3.0), lucid.arange(2.0)
    assert lucid.meshgrid(a, b, indexing="xy")[0].shape == (2, 3)
    assert lucid.meshgrid(a, b)[0].shape == (3, 2)
    for bad in ("zz", "XY", ""):
        with pytest.raises(ValueError, match="indexing must be one of 'xy' or 'ij'"):
            lucid.meshgrid(a, b, indexing=bad)
    for wrong in (None, 3):
        with pytest.raises(TypeError, match="argument 'indexing' must be str"):
            lucid.meshgrid(a, b, indexing=wrong)  # type: ignore[arg-type]  # the refusal under test


def test_block_diag_refuses_more_than_two_dimensions() -> None:
    with pytest.raises(ValueError, match="input 1 has 3 dimensions"):
        lucid.block_diag(lucid.ones(2, 2), lucid.ones(2, 2, 2))


def test_einsum_takes_a_list_of_operands_and_a_string_equation() -> None:
    x, y = lucid.randn(2, 3), lucid.randn(3, 4)
    assert lucid.einops.einsum("ij,jk->ik", [x, y]).shape == (2, 4)
    with pytest.raises(TypeError, match="argument 'equation' must be str, not NoneType"):
        lucid.einops.einsum(None, x)  # type: ignore[arg-type]  # the refusal under test


def test_a_method_alias_names_the_operand_it_refused() -> None:
    with pytest.raises(
        TypeError, match=r"lerp\(\): argument 'input' \(position 1\) must be Tensor"
    ):
        lucid.lerp(None, lucid.ones(2), 0.5)  # type: ignore[arg-type]  # the refusal under test


def test_an_operand_that_is_neither_tensor_nor_number_is_refused_by_the_unwrap() -> None:
    with pytest.raises(TypeError, match="expected Tensor or Number, but got NoneType"):
        _unwrap_or_scalar(None)


def test_engine_callables_refuse_null_tensor_impl_without_crashing() -> None:
    """Direct _C.engine calls with None raise LucidError instead of segfaulting (LCD-683)."""
    from lucid._C import engine as _C_engine

    x = _C_engine.ones([2, 2])
    for call in (
        lambda: _C_engine.cummax(None, 0),
        lambda: _C_engine.cummin(None, 0),
        lambda: _C_engine.ravel(None),
        lambda: _C_engine.sum(None, [], False),
        lambda: _C_engine.prod(None, [], False),
        lambda: _C_engine.square(None),
        lambda: _C_engine.cube(None),
        lambda: _C_engine.concatenate([None, x], 0),
        lambda: _C_engine.stack([None, x], 0),
        lambda: _C_engine.meshgrid([None, x], False),
        lambda: _C_engine.einops.einsum("ij,jk->ik", [None, x]),
    ):
        with pytest.raises(_C_engine.LucidError):
            call()


if __name__ == "__main__":
    _BATCHES[sys.argv[1]](sys.stdout)
