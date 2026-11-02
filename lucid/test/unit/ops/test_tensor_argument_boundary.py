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

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

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


if __name__ == "__main__":
    _BATCHES[sys.argv[1]](sys.stdout)
