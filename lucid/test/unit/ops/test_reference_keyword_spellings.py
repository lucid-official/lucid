"""The reference's keyword spellings reach the op, not the engine binding.

``tril(x, diagonal=1)``, ``flatten(x, start_dim=1)``, ``tile(x, dims=...)``
and ``all(x, dim=1)`` went straight through to a pybind11 binding that
takes ``k`` / ``start_axis`` / ``reps`` and no ``dim`` at all, and failed
with "incompatible function arguments" (LCD-386).  The reference's names
are the only ones now — Lucid's own were dropped without a deprecation
(LCD-410) — and each answer is checked against the reference over every
spelling — the free function, the Tensor method, positional and keyword —
on both devices.
"""

from collections.abc import Callable
from types import ModuleType

import pytest

import lucid

_MATRIX = [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]]


@pytest.mark.parametrize("op", ["tril", "triu"])
@pytest.mark.parametrize("offset", [-1, 0, 2])
def test_tril_triu_take_diagonal(ref: ModuleType, device: str, op: str, offset: int) -> None:
    x = lucid.tensor(_MATRIX, device=device)
    want = getattr(ref, op)(ref.tensor(_MATRIX), diagonal=offset).tolist()
    for got in (
        getattr(lucid, op)(x, diagonal=offset),
        getattr(lucid, op)(x, offset),
        getattr(lucid, op)(input=x, diagonal=offset),
        getattr(x, op)(diagonal=offset),
        getattr(x, op)(offset),
    ):
        assert got.tolist() == want


def test_flatten_takes_start_and_end_dim(ref: ModuleType, device: str) -> None:
    x = lucid.zeros(2, 3, 4, 5, device=device)
    r = ref.zeros(2, 3, 4, 5)
    for start, end in [(0, -1), (1, -1), (1, 2), (0, 1)]:
        want = tuple(ref.flatten(r, start_dim=start, end_dim=end).shape)
        assert lucid.flatten(x, start_dim=start, end_dim=end).shape == want
        assert x.flatten(start_dim=start, end_dim=end).shape == want
        assert x.flatten(start, end_dim=end).shape == want
        assert lucid.flatten(x, start, end).shape == want


def test_tile_takes_dims(ref: ModuleType, device: str) -> None:
    x = lucid.tensor([[1, 2], [3, 4]], device=device)
    for dims in [(2,), (2, 3), (2, 1, 2)]:
        want = ref.tile(ref.tensor([[1, 2], [3, 4]]), dims=dims).tolist()
        assert lucid.tile(x, dims=dims).tolist() == want
        assert x.tile(dims=dims).tolist() == want
        assert lucid.tile(x, dims).tolist() == want
        assert x.tile(*dims).tolist() == want


@pytest.mark.parametrize(
    "call",
    [
        lambda m: m.tril(m.ones(2, 2), k=1),
        lambda m: m.ones(2, 2).triu(k=1),
        lambda m: m.flatten(m.ones(2, 2), start_axis=0),
        lambda m: m.ones(2, 2).flatten(end_axis=-1),
        lambda m: m.tile(m.ones(2), reps=(2,)),
        lambda m: m.tile(m.ones(2)),
        lambda m: m.tile(m.ones(2), 2, 2),
    ],
    ids=[
        "tril-k",
        "triu-method-k",
        "flatten-start_axis",
        "flatten-method-end_axis",
        "tile-reps",
        "tile-neither",
        "tile-free-spread",
    ],
)
def test_the_old_lucid_spellings_are_refused(
    call: Callable[[ModuleType], object],
) -> None:
    # The reference's free ``tile`` takes the counts as one list, as its
    # method does not need to.
    with pytest.raises(TypeError):
        call(lucid)


_LOGICAL_DATA = {
    "bool": [[True, False, True], [True, True, True]],
    "int64": [[1, 0, 3], [2, 5, -1]],
    "float32": [[0.5, 0.0, float("nan")], [1.0, -2.0, 3.0]],
}


def _dtypes(m: ModuleType, name: str) -> object:
    return getattr(m, name)


@pytest.mark.parametrize("op", ["any", "all"])
@pytest.mark.parametrize("dtype", sorted(_LOGICAL_DATA))
@pytest.mark.parametrize("dim", [None, 0, 1, -1, (0, 1)], ids=lambda d: f"dim={d}")
@pytest.mark.parametrize("keepdim", [False, True], ids=["drop", "keep"])
def test_any_all_reduce_over_dim(
    ref: ModuleType, device: str, op: str, dtype: str, dim: object, keepdim: bool
) -> None:
    data = _LOGICAL_DATA[dtype]
    x = lucid.tensor(data, dtype=_dtypes(lucid, dtype), device=device)
    r = ref.tensor(data, dtype=_dtypes(ref, dtype))
    if dim is None:
        want = getattr(ref, op)(r, dim=None, keepdim=keepdim) if keepdim else getattr(ref, op)(r)
    else:
        want = getattr(ref, op)(r, dim=dim, keepdim=keepdim)
    calls = [
        getattr(lucid, op)(x, dim=dim, keepdim=keepdim),
        getattr(x, op)(dim=dim, keepdim=keepdim),
        getattr(x, op)(dim, keepdim),
    ]
    for got in calls:
        assert got.dtype == lucid.bool
        assert got.shape == tuple(want.shape)
        assert got.tolist() == want.tolist()


@pytest.mark.parametrize("op", ["any", "all"])
def test_any_all_over_an_empty_axis(ref: ModuleType, device: str, op: str) -> None:
    x = lucid.zeros(0, 3, device=device)
    for dim in (0, 1):
        want = getattr(ref, op)(ref.zeros(0, 3), dim=dim)
        got = getattr(lucid, op)(x, dim=dim)
        assert got.shape == tuple(want.shape)
        assert got.tolist() == want.tolist()


def test_any_all_read_both_parts_of_a_complex_entry(device: str) -> None:
    # 2j is non-zero although its real part is 0: Metal's complex-to-bool
    # cast reads the real part only.
    z = lucid.complex(
        lucid.tensor([[0.0, 0.0], [1.0, 0.0]], device=device),
        lucid.tensor([[2.0, 0.0], [0.0, 0.0]], device=device),
    )
    assert lucid.any(z, dim=1).tolist() == [True, True]
    assert lucid.all(z, dim=0).tolist() == [True, False]


def test_tile_takes_a_single_count(ref: ModuleType, device: str) -> None:
    # ``x.tile(2)`` used to fail with "'int' object is not iterable".
    data = [[1, 2], [3, 4]]
    x = lucid.tensor(data, device=device)
    want = ref.tensor(data).tile(2).tolist()
    assert x.tile(2).tolist() == want
    assert lucid.tile(x, 2).tolist() == want
    assert x.tile(dims=2).tolist() == want


# ── LCD-410 / LCD-421: the reference's spellings are the only ones ───────────


@pytest.mark.parametrize(
    "spread, listed",
    [
        (lambda x: x.permute(1, 0), lambda m, x: m.permute(x, (1, 0))),
        (lambda x: x.reshape(3, 2), lambda m, x: m.reshape(x, (3, 2))),
        (lambda x: x.view(3, 2), lambda m, x: m.view(x, (3, 2))),
        (lambda x: x[:1].expand(4, 3), lambda m, x: m.expand(x[:1], (4, 3))),
        (lambda x: x.broadcast_to(2, 2, 3), lambda m, x: m.broadcast_to(x, (2, 2, 3))),
        (lambda x: x.tile(2, 1), lambda m, x: m.tile(x, (2, 1))),
        (lambda x: x.flip(0, 1), lambda m, x: m.flip(x, (0, 1))),
    ],
    ids=["permute", "reshape", "view", "expand", "broadcast_to", "tile", "flip"],
)
def test_methods_spread_an_int_list_and_free_functions_take_it_whole(
    device: str, spread: Callable[..., object], listed: Callable[..., object]
) -> None:
    # As in the reference: ``x.permute(1, 0)`` and the free function with
    # ``(x, (1, 0))``, while the free function refuses ``(x, 1, 0)``.
    x = lucid.arange(6.0, device=device).reshape(2, 3)
    assert spread(x).tolist() == listed(lucid, x).tolist()  # type: ignore[attr-defined]
    name = listed.__code__.co_names[0]
    with pytest.raises(TypeError):
        getattr(lucid, name)(x, 1, 0)


def test_operands_take_the_reference_names(device: str) -> None:
    a = lucid.tensor([[1.0, 2.0], [3.0, 4.0]], device=device)
    v = lucid.tensor([1.0, 2.0], device=device)
    want = (a @ a).tolist()
    assert lucid.mm(a, mat2=a).tolist() == want
    assert a.mm(mat2=a).tolist() == want
    assert lucid.pow(a, exponent=2).tolist() == a.pow(exponent=2).tolist()
    assert v.dot(tensor=v).item() == 5.0
    assert v.outer(vec2=v).shape == (2, 2)
    assert lucid.where(condition=a > 2, input=a, other=0.0).tolist() == [
        [0.0, 0.0],
        [3.0, 4.0],
    ]
    assert lucid.take(a, index=lucid.tensor([3], device=device)).tolist() == [4.0]
    for old in (
        lambda: lucid.mm(a, other=a),
        lambda: lucid.pow(a, other=2),
        lambda: lucid.where(cond=a > 2, x=a, y=a),
        lambda: lucid.take(a, indices=lucid.tensor([0], device=device)),
        lambda: lucid.xlogy(x=a, y=a),
        lambda: lucid.zeros_like(t=a),
    ):
        with pytest.raises(TypeError):
            old()


def test_max_pool_reads_ceil_mode_before_return_indices(ref: ModuleType) -> None:
    # The sixth positional argument is ``ceil_mode`` (LCD-421): it used to be
    # ``return_indices``, so a positional caller's two flags were swapped.
    import lucid.nn.functional as F

    data = [[[[float(i * 5 + j) for j in range(5)] for i in range(5)]]]
    want = ref.nn.functional.max_pool2d(ref.tensor(data), 2, 2, 0, 1, True, False)
    got = F.max_pool2d(lucid.tensor(data), 2, 2, 0, 1, True, False)
    assert got.tolist() == want.tolist()
    out, idx = F.max_pool2d(lucid.tensor(data), 2, 2, 0, 1, False, True)
    assert out.shape == (1, 1, 2, 2) and idx.shape == (1, 1, 2, 2)


def test_conv_transpose_takes_only_zeros_padding_mode() -> None:
    import lucid.nn as nn

    assert nn.ConvTranspose2d(2, 2, 3, padding_mode="zeros").padding_mode == "zeros"
    with pytest.raises(ValueError, match="zeros"):
        nn.ConvTranspose2d(2, 2, 3, padding_mode="reflect")


#: ``.to`` calls, each run on Lucid and on the reference with ``(m, x)``:
#: ``m`` the module, ``x`` a float32 CPU tensor of three zeros.
_TO_CALLS: dict[str, Callable[[ModuleType, object], object]] = {
    # the reference's shapes
    "no-args": lambda m, x: x.to(),
    "device": lambda m, x: x.to("cpu"),
    "device-dtype": lambda m, x: x.to("cpu", m.float64),
    "device-dtype-flags": lambda m, x: x.to("cpu", m.float64, False, True),
    "device-dtype-flag": lambda m, x: x.to("cpu", m.float64, False),
    "device-none-dtype": lambda m, x: x.to(None, m.float64),
    "none": lambda m, x: x.to(None),
    "dtype": lambda m, x: x.to(m.float64),
    "dtype-flags": lambda m, x: x.to(m.float64, False, True),
    "dtype-flag-copy-kw": lambda m, x: x.to(m.float64, False, copy=True),
    "dtype-copy-kw": lambda m, x: x.to(m.float32, copy=True),
    "same-dtype": lambda m, x: x.to(m.float32),
    "tensor": lambda m, x: x.to(x),
    "tensor-flags": lambda m, x: x.to(x, False, True),
    "keywords": lambda m, x: x.to(device="cpu", dtype=m.float64),
    "device-dtype-kw": lambda m, x: x.to("cpu", dtype=m.float64),
    "device-kw-none": lambda m, x: x.to(device=None),
    "dtype-kw-none": lambda m, x: x.to(dtype=None),
    "non-blocking-kw": lambda m, x: x.to("cpu", non_blocking=True),
    "bool-as-dtype": lambda m, x: x.to(True),
    "false-as-dtype": lambda m, x: x.to(False),
    "bool-dtype-flags": lambda m, x: x.to(True, False, True),
    # refused by both
    "unknown-keyword": lambda m, x: x.to(dtyp=m.float64),
    "object": lambda m, x: x.to(object()),
    "list": lambda m, x: x.to([1]),
    "dtype-then-device": lambda m, x: x.to(m.float64, "cpu"),
    "tensor-then-dtype": lambda m, x: x.to(x, m.float64),
    "two-dtypes": lambda m, x: x.to(m.float64, m.float64),
    "device-twice": lambda m, x: x.to("cpu", device="cpu"),
    "copy-twice": lambda m, x: x.to(m.float64, False, True, copy=True),
    "non-blocking-twice": lambda m, x: x.to(m.float64, False, non_blocking=True),
    "flag-in-dtype-slot": lambda m, x: x.to("cpu", True),
    "third-flag": lambda m, x: x.to("cpu", m.float64, False, True, False),
    "dtype-then-device-kw": lambda m, x: x.to(m.float64, device="cpu"),
    "tensor-then-dtype-kw": lambda m, x: x.to(x, dtype=m.float64),
}


@pytest.mark.parametrize("name", list(_TO_CALLS))
def test_to_binds_as_the_reference_binds(ref: ModuleType, name: str) -> None:
    call = _TO_CALLS[name]

    def run(m: ModuleType) -> tuple[str, bool] | None:
        x = m.zeros(3)
        try:
            out = call(m, x)
        except TypeError:
            return None
        return str(out.dtype).rsplit(".", 1)[-1].rstrip("_"), out is x

    assert run(lucid) == run(ref)


def test_to_refuses_a_device_index() -> None:
    # The reference reads ``x.to(3)`` as device index 3; Lucid has one
    # device of each kind, so an index is refused rather than guessed at.
    with pytest.raises(TypeError):
        lucid.zeros(3).to(3)


_CONV_CLASSES = [
    f"{lazy}Conv{kind}{rank}d"
    for lazy in ("", "Lazy")
    for kind in ("", "Transpose")
    for rank in (1, 2, 3)
]


def _size_cases() -> list[object]:
    cases = []
    for name in _CONV_CLASSES:
        rank = int(name[-2])
        args = ["kernel_size", "stride", "padding", "dilation"]
        if "Transpose" in name:
            args.append("output_padding")
        for arg in args:
            value = 2 if arg == "dilation" else 1
            for form, given in (
                ("int", value),
                ("one", (value,)),
                ("rank", (value,) * rank),
                ("wrong", (value,) * (rank + 1)),
            ):
                cases.append(pytest.param(name, arg, given, id=f"{name}-{arg}-{form}"))
    return cases


def _build(m: ModuleType, name: str, arg: str, given: object) -> object:
    kwargs: dict[str, object] = {"kernel_size": 3, arg: given}
    if arg == "output_padding":
        kwargs["stride"] = 2
    cls = getattr(m.nn, name)
    return cls(2, **kwargs) if name.startswith("Lazy") else cls(2, 2, **kwargs)


@pytest.mark.parametrize("name,arg,given", _size_cases())
def test_conv_sizes_read_as_the_reference_reads_them(
    ref: ModuleType, name: str, arg: str, given: object
) -> None:
    # One int, or a one-element sequence, stands for every spatial dim —
    # except a kernel_size of one element on a 2-D / 3-D window, and a
    # sequence of the wrong length, which the reference refuses (at the
    # forward; Lucid at construction).  A 1-D layer takes ``(3,)`` too.
    rank = int(name[-2])
    shape = (1, 2) + (9,) * rank
    try:
        want = tuple(_build(ref, name, arg, given)(ref.zeros(shape)).shape)
    except RuntimeError:
        want = None
    if want is None:
        with pytest.raises(ValueError, match=arg):
            _build(lucid, name, arg, given)(lucid.zeros(shape))
        return
    assert _build(lucid, name, arg, given)(lucid.zeros(shape)).shape == want


def test_a_lazy_transposed_conv_pickled_without_padding_mode_still_reprs() -> None:
    import lucid.nn as nn

    layer = nn.LazyConvTranspose2d(4, 3)
    del layer.__dict__["padding_mode"]
    assert "padding_mode" not in repr(layer)
