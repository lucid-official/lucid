"""Every FFT differentiates twice (LCD-169).

Each transform's backward is its dual transform.  That dual was computed
with the raw engine entry point, which records no graph, so under
``create_graph`` the first gradient came back detached and the engine
attached a ``ZeroDerivativeBackward`` in its place: every second
derivative through a spectrum was a silent zero.  On the reported case,
``fft`` of a length-6 signal under ``|y|^2`` gave a Hessian-vector product
of ``[0, 0, 0, 0, 0, 0]`` where the answer is
``[86.4, -345.6, 230.4, 576, -144, 316.8]``.

Each backward now runs the dual transform through its own ``Function``,
so the gradient records a graph and the derivative of a derivative is
the transform it should be.  The sweep holds the Hessian-vector product
of a cubic loss to the reference for every base transform, each composed
one (``fft2`` / ``hfft`` / ``ihfft`` …) and ``fftshift``, at sizes that
crop and pad, for each norm, on each device.  Metal has no double complex,
so the comparison uses single-precision tolerances throughout.
"""

import numpy as np
import pytest

import lucid
from lucid.autograd import grad

#: (transform, input shape, kwargs, input kinds).  The input kind is which
#: signal the transform takes: the reference refuses complex input to the
#: real-signal transforms, and the half-spectrum readers are fed a spectrum.
CASES: list[tuple[str, tuple[int, ...], dict[str, object], tuple[bool, ...]]] = [
    ("fft", (6,), {}, (False, True)),
    ("fft", (6,), {"n": 4}, (False, True)),
    ("ifft", (5,), {"n": 7}, (False, True)),
    ("rfft", (7,), {}, (False,)),
    ("rfft", (5,), {"n": 8}, (False,)),
    ("irfft", (4,), {"n": 6}, (True,)),
    ("irfft", (4,), {}, (True,)),
    ("fft2", (3, 4), {}, (False, True)),
    ("rfftn", (3, 4), {}, (False,)),
    ("rfftn", (2, 3, 5), {"s": (4, 6)}, (False,)),
    ("irfftn", (3, 3), {"s": (3, 5)}, (True,)),
    ("hfft", (4,), {"n": 6}, (True,)),
    ("ihfft", (6,), {}, (False,)),
    ("fftshift", (5,), {}, (False, True)),
    ("fftshift", (3, 4), {"dim": (0, 1)}, (False,)),
]
NORMS = [None, "forward", "ortho"]


def _params() -> list[object]:
    out: list[object] = []
    for name, shape, kwargs, kinds in CASES:
        # ``fftshift`` is a permutation and takes no ``norm``.
        norms = [None] if name == "fftshift" else NORMS
        for norm in norms:
            for complex_in in kinds:
                kw = dict(kwargs)
                if norm is not None:
                    kw["norm"] = norm
                kind = "complex" if complex_in else "real"
                ident = f"{name}-{shape}-{kwargs}-{norm}-{kind}"
                out.append(pytest.param(name, shape, kw, complex_in, id=ident))
    return out


def _arrays(seed: int, shape: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    return (
        rng.standard_normal(shape).astype(np.float32),
        rng.standard_normal(shape).astype(np.float32),
    )


def _split(g) -> np.ndarray:
    g = g.to("cpu")
    if g.is_complex():
        return np.asarray(lucid.real(g).numpy()) + 1j * np.asarray(lucid.imag(g).numpy())
    return np.asarray(g.numpy())


def _lucid_hvp(name, kwargs, re, im, complex_in, weights, u, device):
    """The first gradient (recording) and the Hessian-vector product along ``u``."""
    if complex_in:
        x = lucid.complex(lucid.tensor(re), lucid.tensor(im)).to(device)
        x.requires_grad_(True)
    else:
        x = lucid.tensor(re.copy(), device=device, requires_grad=True)
    y = getattr(lucid.fft, name)(x, **kwargs)
    w1, w2 = (lucid.tensor(w, device=device) for w in weights(tuple(y.shape)))
    if y.is_complex():
        r, i = lucid.real(y), lucid.imag(y)
        loss = (w1 * r * r * r + w2 * i * i).sum()
    else:
        loss = (w1 * y * y * y).sum()
    (g,) = grad(loss, [x], create_graph=True)
    u_re, u_im = (lucid.tensor(a, device=device) for a in u)
    if g.is_complex():
        projected = (lucid.real(g) * u_re + lucid.imag(g) * u_im).sum()
    else:
        projected = (g * u_re).sum()
    (hvp,) = grad(projected, [x])
    return g, _split(hvp)


def _ref_hvp(ref, name, kwargs, re, im, complex_in, weights, u):
    re64, im64 = re.astype(np.float64), im.astype(np.float64)
    if complex_in:
        x = ref.complex(ref.tensor(re64), ref.tensor(im64)).requires_grad_(True)
    else:
        x = ref.tensor(re64, requires_grad=True)
    y = getattr(ref.fft, name)(x, **kwargs)
    w1, w2 = (ref.tensor(w.astype(np.float64)) for w in weights(tuple(y.shape)))
    if y.is_complex():
        loss = (w1 * y.real**3 + w2 * y.imag**2).sum()
    else:
        loss = (w1 * y**3).sum()
    (g,) = ref.autograd.grad(loss, [x], create_graph=True)
    u_re, u_im = (ref.tensor(a.astype(np.float64)) for a in u)
    if g.is_complex():
        projected = (g.real * u_re + g.imag * u_im).sum()
    else:
        projected = (g * u_re).sum()
    (hvp,) = ref.autograd.grad(projected, [x])
    return hvp.detach().resolve_conj().numpy()


def _run(name, shape, kwargs, complex_in, device, ref=None):
    seed = sum(map(ord, name)) + len(shape)
    re, im = _arrays(seed, shape)
    u = _arrays(seed + 2, shape)

    def weights(out_shape: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray]:
        # Every output lane weighted differently, so a misplaced term
        # cannot cancel.
        return _arrays(seed + 1, out_shape)

    g, got = _lucid_hvp(name, kwargs, re, im, complex_in, weights, u, device)
    if ref is None:
        return g, got, None
    want = _ref_hvp(ref, name, kwargs, re, im, complex_in, weights, u)
    return g, got, want


@pytest.mark.parity
@pytest.mark.parametrize("name,shape,kwargs,complex_in", _params())
def test_the_hessian_vector_product_matches_the_reference(
    name: str, shape: tuple[int, ...], kwargs: dict[str, object], complex_in: bool, device: str, ref
) -> None:
    _, got, want = _run(name, shape, kwargs, complex_in, device, ref)
    assert got.shape == want.shape, (got.shape, want.shape)
    scale = float(np.abs(want).max())
    assert scale > 0.0, "the case has no curvature, so it checks nothing"
    np.testing.assert_allclose(got, want, rtol=2e-3, atol=2e-4 * scale, err_msg=f"{name} {kwargs}")


@pytest.mark.parametrize("name,shape,kwargs,complex_in", _params())
def test_no_transform_hands_back_a_zero_derivative(
    name: str, shape: tuple[int, ...], kwargs: dict[str, object], complex_in: bool, device: str
) -> None:
    """The first gradient is recorded, not a stand-in with a zero derivative.

    The engine attaches ``ZeroDerivativeBackward`` to a gradient that a
    backward returned detached — the signature of a backward built from
    ops that record nothing.  On a cubic loss the gradient depends on the
    input, so a recorded graph is the only right answer, and the second
    derivative it gives is not zero.
    """
    g, got, _ = _run(name, shape, kwargs, complex_in, device)
    assert g.grad_fn is not None
    assert g.grad_fn.name() != "ZeroDerivativeBackward"
    assert np.abs(got).max() > 0.0


def test_the_reported_case() -> None:
    """The issue's numbers, so a regression is recognisable."""
    x = lucid.tensor([0.3, -1.2, 0.8, 2.0, -0.5, 1.1], requires_grad=True)
    y = lucid.fft.fft(x)
    (g,) = grad((lucid.real(y) ** 2 + lucid.imag(y) ** 2).sum(), [x], create_graph=True)
    (gg,) = grad((g * g).sum(), [x])
    np.testing.assert_allclose(
        np.asarray(gg.numpy()), [86.4, -345.6, 230.4, 576.0, -144.0, 316.8], rtol=1e-5, atol=1e-3
    )


def test_first_order_backward_records_nothing() -> None:
    """Without ``create_graph`` the dual transform builds no graph.

    The backward goes through the dual transform's ``Function``, which
    records a node only when grad mode is on and the incoming gradient
    requires grad — so a plain ``backward`` costs what it did.
    """
    x = lucid.randn(8, requires_grad=True)
    (g,) = grad(lucid.abs(lucid.fft.rfft(x)).sum(), [x])
    assert g.grad_fn is None
    assert not g.requires_grad
