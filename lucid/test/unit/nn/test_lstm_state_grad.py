"""The LSTM's final state is a differentiable output, in every float dtype.

``nn.LSTM`` returns ``(output, (h_n, c_n))``.  The engine node used to give
``output`` its ``grad_fn`` and hand ``h_n`` / ``c_n`` back detached, seeding
BPTT with zeros for them.  Every loss that reaches the final state — an
encoder's ``h_n`` seeding a decoder, a classifier on ``h_n`` — then trained
nothing behind it, and a packed batch, which carries the state from one run
of timesteps into the next, lost the gradient of every step but the last
run's.  The CPU kernels also existed for float32 only.

Each case runs the same weights through Lucid (CPU and Metal) and through
the reference and compares every gradient: loss on ``output``, ``h_n``,
``c_n`` or all three (``h_n`` twice, so a barrier slot has to sum) ×
packed / dense × device × float32 / float64.
"""

from types import ModuleType

import numpy as np
import pytest

import lucid
import lucid.nn as nn
from lucid._C import engine as _C_engine
from lucid.nn.utils.rnn import PackedSequence, pack_padded_sequence
from lucid.test._fixtures.devices import device_dtype_params, metal_available

_T, _B, _I, _H = 5, 3, 4, 6
_LENGTHS = [3, 5, 2]  # uneven and unsorted: the packed path permutes too
_LOSSES = ("output", "h_n", "c_n", "all")
_TOL = {lucid.float32: 1e-5, lucid.float64: 1e-10}


def _np_dtype(dtype: lucid.dtype) -> type:
    return np.float64 if dtype is lucid.float64 else np.float32


def _make_pair(
    ref: ModuleType, dtype: lucid.dtype, device: str, **kwargs: object
) -> tuple[nn.LSTM, object]:
    """A Lucid LSTM and a reference LSTM holding the same weights."""
    ours = nn.LSTM(_I, _H, dtype=dtype, **kwargs)
    theirs = ref.nn.LSTM(_I, _H, **kwargs).to(
        ref.float64 if dtype is lucid.float64 else ref.float32
    )
    named = dict(theirs.named_parameters())
    for name, p in ours.named_parameters():
        named[name].data = ref.tensor(p.numpy())
    return ours.to(device), theirs


def _inputs(
    dtype: lucid.dtype, layers: int = 1, dirs: int = 1, rec: int = _H
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(7)
    dt = _np_dtype(dtype)
    x = rng.standard_normal((_T, _B, _I)).astype(dt)
    for b, n in enumerate(_LENGTHS):
        x[n:, b] = 0.0
    return {
        "x": x,
        "h0": rng.standard_normal((layers * dirs, _B, rec)).astype(dt),
        "c0": rng.standard_normal((layers * dirs, _B, _H)).astype(dt),
        "w_out": rng.standard_normal((_T, _B, rec * dirs)).astype(dt),
        "w_packed": rng.standard_normal((sum(_LENGTHS), rec * dirs)).astype(dt),
        "w_h": rng.standard_normal((layers * dirs, _B, rec)).astype(dt),
        "w_c": rng.standard_normal((layers * dirs, _B, _H)).astype(dt),
    }


def _loss(out: object, h: object, c: object, which: str, w: dict, packed: bool) -> object:
    """Weighted so no two gradient entries coincide; works on either library.

    A packed output is compared through its ``.data`` — both libraries lay it
    out in the same descending-length order.
    """
    terms = {
        "output": lambda: (out * w["w_packed" if packed else "w_out"]).sum(),
        "h_n": lambda: (h * w["w_h"]).sum(),
        "c_n": lambda: (c * w["w_c"]).sum(),
    }
    if which != "all":
        return terms[which]()
    return terms["output"]() + terms["h_n"]() + terms["c_n"]() + (h * h).sum()


def _pack(x: lucid.Tensor, device: str) -> PackedSequence:
    """Pack on CPU and move the data: unsorted packing does not run on Metal
    (its permutation index stays on CPU), and that is not what is tested here."""
    p = pack_padded_sequence(x, _LENGTHS, enforce_sorted=False)
    return PackedSequence(p.data.to(device), p.batch_sizes, p.sorted_indices, p.unsorted_indices)


def _run_ours(
    model: nn.LSTM, data: dict, device: str, packed: bool, which: str
) -> dict[str, np.ndarray]:
    x = lucid.tensor(data["x"], device="cpu" if packed else device, requires_grad=True)
    h0 = lucid.tensor(data["h0"], device=device, requires_grad=True)
    c0 = lucid.tensor(data["c0"], device=device, requires_grad=True)
    feed = _pack(x, device) if packed else x
    out, (h, c) = model(feed, (h0, c0))
    if packed:
        assert isinstance(out, PackedSequence)
        out = out.data
    w = {k: lucid.tensor(v, device=device) for k, v in data.items() if k.startswith("w_")}
    _loss(out, h, c, which, w, packed).backward()
    grads = {"x": x.grad, "h0": h0.grad, "c0": c0.grad}
    grads.update({n: p.grad for n, p in model.named_parameters()})
    return {k: (None if g is None else g.to("cpu").numpy()) for k, g in grads.items()}


def _run_ref(
    ref: ModuleType, model: object, data: dict, packed: bool, which: str
) -> dict[str, np.ndarray]:
    x = ref.tensor(data["x"], requires_grad=True)
    h0 = ref.tensor(data["h0"], requires_grad=True)
    c0 = ref.tensor(data["c0"], requires_grad=True)
    rnn = ref.nn.utils.rnn
    feed = rnn.pack_padded_sequence(x, _LENGTHS, enforce_sorted=False) if packed else x
    out, (h, c) = model(feed, (h0, c0))
    if packed:
        out = out.data
    w = {k: ref.tensor(v) for k, v in data.items() if k.startswith("w_")}
    _loss(out, h, c, which, w, packed).backward()
    grads = {"x": x.grad, "h0": h0.grad, "c0": c0.grad}
    grads.update({n: p.grad for n, p in model.named_parameters()})
    return {k: (None if g is None else g.detach().numpy()) for k, g in grads.items()}


def _assert_grads_match(
    ours: dict[str, np.ndarray], theirs: dict[str, np.ndarray], tol: float
) -> None:
    assert ours.keys() == theirs.keys()
    for name, want in theirs.items():
        got = ours[name]
        assert got is not None, f"{name}: no gradient"
        assert want is not None
        err = float(np.max(np.abs(got - want) / np.maximum(1.0, np.abs(want))))
        assert err <= tol, f"{name}: max rel err {err:.3e} > {tol:.0e}"


# ── the matrix the card asks for ─────────────────────────────────────────────


@pytest.mark.parametrize("device,dtype", device_dtype_params((lucid.float32, lucid.float64)))
@pytest.mark.parametrize("packed", [False, True], ids=["dense", "packed"])
@pytest.mark.parametrize("which", _LOSSES)
def test_every_output_carries_the_reference_gradient(
    ref: ModuleType, device: str, dtype: lucid.dtype, packed: bool, which: str
) -> None:
    ours, theirs = _make_pair(ref, dtype, device)
    data = _inputs(dtype)
    got = _run_ours(ours, data, device, packed, which)
    want = _run_ref(ref, theirs, data, packed, which)
    _assert_grads_match(got, want, _TOL[dtype])


@pytest.mark.parametrize("device,dtype", device_dtype_params((lucid.float32, lucid.float64)))
@pytest.mark.parametrize("packed", [False, True], ids=["dense", "packed"])
@pytest.mark.parametrize(
    "config",
    [{"num_layers": 2, "bidirectional": True}, {"proj_size": 3}],
    ids=["stacked-bidirectional", "projected"],
)
def test_the_state_gradient_survives_stacking_directions_and_projection(
    ref: ModuleType, device: str, dtype: lucid.dtype, packed: bool, config: dict
) -> None:
    ours, theirs = _make_pair(ref, dtype, device, **config)
    layers = int(config.get("num_layers", 1))
    dirs = 2 if config.get("bidirectional") else 1
    rec = int(config.get("proj_size", 0)) or _H
    data = _inputs(dtype, layers, dirs, rec)
    got = _run_ours(ours, data, device, packed, "all")
    want = _run_ref(ref, theirs, data, packed, "all")
    _assert_grads_match(got, want, _TOL[dtype])


@pytest.mark.parametrize("device,dtype", device_dtype_params((lucid.float32, lucid.float64)))
def test_an_encoder_learns_through_the_state_it_hands_a_decoder(
    ref: ModuleType, device: str, dtype: lucid.dtype
) -> None:
    enc, ref_enc = _make_pair(ref, dtype, device)
    dec, ref_dec = _make_pair(ref, dtype, device)
    data = _inputs(dtype)

    x = lucid.tensor(data["x"], device=device)
    _, state = enc(x)
    out, _ = dec(x, state)
    (out * lucid.tensor(data["w_out"], device=device)).sum().backward()

    _, ref_state = ref_enc(ref.tensor(data["x"]))
    ref_out, _ = ref_dec(ref.tensor(data["x"]), ref_state)
    (ref_out * ref.tensor(data["w_out"])).sum().backward()

    for (name, p), (_, q) in zip(enc.named_parameters(), ref_enc.named_parameters(), strict=True):
        assert p.grad is not None, f"encoder {name} got no gradient"
        np.testing.assert_allclose(
            p.grad.to("cpu").numpy(), q.grad.numpy(), rtol=_TOL[dtype], atol=_TOL[dtype]
        )


# ── backward-pass hygiene of the barrier ─────────────────────────────────────


@pytest.mark.parametrize("device", ["cpu", "metal"])
def test_a_retained_graph_gives_the_same_gradient_twice(device: str) -> None:
    if device == "metal" and not metal_available():
        pytest.skip("Metal device not available on this host")
    m = nn.LSTM(_I, _H).to(device)
    x = lucid.randn(_T, _B, _I, device=device, requires_grad=True)
    out, (h, c) = m(x)
    loss = out.sum() + h.sum() + c.sum()
    loss.backward(retain_graph=True)
    first = x.grad.to("cpu").numpy().copy()
    x.grad = None
    loss.backward()
    np.testing.assert_allclose(x.grad.to("cpu").numpy(), first, rtol=1e-6, atol=1e-6)


def test_a_graph_that_was_not_retained_is_refused_the_second_time() -> None:
    m = nn.LSTM(_I, _H)
    x = lucid.randn(_T, _B, _I, requires_grad=True)
    _, (h, _) = m(x)
    loss = h.sum()
    loss.backward()
    with pytest.raises(RuntimeError):
        loss.backward()


def test_autograd_grad_reaches_through_the_final_state() -> None:
    m = nn.LSTM(_I, _H)
    x = lucid.randn(_T, _B, _I, requires_grad=True)
    _, (h, c) = m(x)
    (gx,) = lucid.autograd.grad((h * c).sum(), x)
    assert float(gx.abs().sum().item()) > 0.0


@pytest.mark.parametrize("device,dtype", device_dtype_params((lucid.float32, lucid.float64)))
def test_hooks_on_the_outputs_see_and_replace_each_slots_gradient(
    ref: ModuleType, device: str, dtype: lucid.dtype
) -> None:
    """Each output's hook runs on its own slot's summed gradient, before BPTT.

    ``h`` is read twice so its slot sums two arrivals before the hook sees
    it; the ``h`` and ``out`` hooks replace the gradient, the ``c`` hook only
    looks.  What every hook saw and every gradient behind the node must match.
    """
    ours, theirs = _make_pair(ref, dtype, device)
    data = _inputs(dtype)

    def run(model: object, x: object, w: dict) -> list[np.ndarray]:
        seen: list[object] = []
        out, (h, c) = model(x)

        def keep(g: object) -> None:
            seen.append(g)

        def doubled(g: object) -> object:
            seen.append(g)
            return g * 2.0

        out.register_hook(lambda g: g * 0.5)
        h.register_hook(doubled)
        c.register_hook(keep)
        loss = (out * w["w_out"]).sum() + (h * w["w_h"]).sum() + (h * h).sum()
        (loss + (c * w["w_c"]).sum()).backward()
        return [_numpy(g) for g in seen]

    x = lucid.tensor(data["x"], device=device, requires_grad=True)
    w = {k: lucid.tensor(v, device=device) for k, v in data.items() if k.startswith("w_")}
    got_seen = run(ours, x, w)
    rx = ref.tensor(data["x"], requires_grad=True)
    want_seen = run(theirs, rx, {k: ref.tensor(v) for k, v in data.items()})

    tol = _TOL[dtype]
    assert len(got_seen) == len(want_seen) == 2
    for g, want in zip(sorted(got_seen, key=np.size), sorted(want_seen, key=np.size)):
        np.testing.assert_allclose(g, want, rtol=tol, atol=tol)
    np.testing.assert_allclose(_numpy(x.grad), _numpy(rx.grad), rtol=tol, atol=tol)
    for (name, p), (_, q) in zip(ours.named_parameters(), theirs.named_parameters(), strict=True):
        np.testing.assert_allclose(_numpy(p.grad), _numpy(q.grad), rtol=tol, atol=tol, err_msg=name)


@pytest.mark.parametrize("proj_size", [0, 3])
def test_create_graph_through_the_node_is_refused_by_name(proj_size: int) -> None:
    m = nn.LSTM(_I, _H, proj_size=proj_size)
    x = lucid.randn(_T, _B, _I, requires_grad=True)
    out, (h, c) = m(x)
    with pytest.raises(NotImplementedError, match="create_graph"):
        lucid.autograd.grad((out.sum() + h.sum() + c.sum()), x, create_graph=True)


def _numpy(t: object) -> np.ndarray:
    """A Lucid or reference tensor as a host array."""
    if isinstance(t, lucid.Tensor):
        return t.to("cpu").numpy()
    return t.detach().numpy()


# ── dtypes ────────────────────────────────────────────────────────────────────


def test_float64_matches_the_reference_forward_to_double_precision(
    ref: ModuleType,
) -> None:
    ours, theirs = _make_pair(ref, lucid.float64, "cpu")
    data = _inputs(lucid.float64)
    out, (h, c) = ours(lucid.tensor(data["x"]))
    ref_out, (ref_h, ref_c) = theirs(ref.tensor(data["x"]))
    for got, want in ((out, ref_out), (h, ref_h), (c, ref_c)):
        assert got.dtype is lucid.float64
        np.testing.assert_allclose(got.numpy(), want.detach().numpy(), rtol=1e-13, atol=1e-13)


def _as_f32(t: lucid.Tensor) -> np.ndarray:
    return t.to(device="cpu", dtype=lucid.float32).numpy()


@pytest.mark.parametrize("dtype", [lucid.float16, lucid.bfloat16], ids=["f16", "bf16"])
@pytest.mark.parametrize("device", ["cpu", "metal"])
def test_half_precision_trains_and_tracks_float32(device: str, dtype: lucid.dtype) -> None:
    if device == "metal" and not metal_available():
        pytest.skip("Metal device not available on this host")
    tol = 2e-2 if dtype is lucid.float16 else 1e-1
    wide = nn.LSTM(_I, _H)
    narrow = nn.LSTM(_I, _H)
    narrow.load_state_dict(wide.state_dict())
    wide, narrow = wide.to(device), narrow.to(device=device, dtype=dtype)
    x = lucid.randn(_T, _B, _I, device=device)

    def run(model: nn.LSTM, dt: lucid.dtype) -> tuple[np.ndarray, np.ndarray]:
        xi = x.to(dtype=dt).detach().requires_grad_(True)
        out, (h, c) = model(xi)
        assert out.dtype is dt and h.dtype is dt and c.dtype is dt
        (out.sum() + h.sum() + c.sum()).backward()
        assert xi.grad is not None and xi.grad.dtype is dt
        return _as_f32(out), _as_f32(xi.grad)

    out32, gx32 = run(wide, lucid.float32)
    out16, gx16 = run(narrow, dtype)
    np.testing.assert_allclose(out16, out32, atol=tol, rtol=tol)
    np.testing.assert_allclose(gx16, gx32, atol=tol * 5, rtol=tol)


def test_bias_free_layers_run_in_float64() -> None:
    m = nn.LSTM(_I, _H, bias=False, dtype=lucid.float64)
    x = lucid.randn(_T, _B, _I, dtype=lucid.float64, requires_grad=True)
    out, _ = m(x)
    out.sum().backward()
    assert x.grad is not None and x.grad.dtype is lucid.float64


# ── operands are checked at the engine's door ────────────────────────────────


def test_weights_on_another_device_raise_device_mismatch() -> None:
    if not metal_available():
        pytest.skip("Metal device not available on this host")
    m = nn.LSTM(_I, _H)
    with pytest.raises(_C_engine.DeviceMismatch, match="weight_ih"):
        m(lucid.randn(_T, _B, _I, device="metal"))


def test_an_input_of_another_dtype_raises_dtype_mismatch() -> None:
    m = nn.LSTM(_I, _H)
    with pytest.raises(TypeError):
        m(lucid.randn(_T, _B, _I, dtype=lucid.float64))


def test_a_misshapen_initial_state_raises_shape_mismatch() -> None:
    m = nn.LSTM(_I, _H)
    x = lucid.randn(_T, _B, _I)
    bad_h0 = lucid.zeros(1, _B, _H + 1)
    with pytest.raises(ValueError, match="h0"):
        m(x, (bad_h0, lucid.zeros(1, _B, _H)))
