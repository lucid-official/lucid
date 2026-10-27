"""Reference parity for ``lucid.signal.windows``."""

from collections.abc import Callable
from typing import Any

import pytest

import lucid.signal.windows as W
from lucid.test._helpers.compare import assert_close

# ── parametric value parity ───────────────────────────────────────────────────
# (name, lucid_fn, ref_attr, kwargs_lucid, kwargs_ref, atol)
_WINDOWS: list[tuple[str, Callable, str, dict, dict, float]] = [
    ("hann", W.hann, "hann", {}, {}, 1e-6),
    ("hamming", W.hamming, "hamming", {}, {}, 1e-6),
    ("blackman", W.blackman, "blackman", {}, {}, 1e-6),
    ("bartlett", W.bartlett, "bartlett", {}, {}, 1e-6),
    ("nuttall", W.nuttall, "nuttall", {}, {}, 1e-6),
    ("cosine", W.cosine, "cosine", {}, {}, 1e-6),
    ("gaussian", W.gaussian, "gaussian", {"std": 2.0}, {"std": 2.0}, 1e-6),
    ("kaiser", W.kaiser, "kaiser", {"beta": 4.0}, {"beta": 4.0}, 1e-4),
    ("exponential", W.exponential, "exponential", {"tau": 2.0}, {"tau": 2.0}, 1e-6),
    (
        "general_cosine",
        W.general_cosine,
        "general_cosine",
        {"a": [0.54, 0.46]},
        {"a": [0.54, 0.46]},
        1e-6,
    ),
    (
        "general_hamming",
        W.general_hamming,
        "general_hamming",
        {"alpha": 0.54},
        {"alpha": 0.54},
        1e-6,
    ),
]


@pytest.mark.parity
@pytest.mark.parametrize(
    "name,lucid_fn,ref_attr,kw_l,kw_r,atol", _WINDOWS, ids=[w[0] for w in _WINDOWS]
)
def test_window_parity(
    name: str, lucid_fn: Callable, ref_attr: str, kw_l: dict, kw_r: dict, atol: float, ref: Any
) -> None:
    for M in [8, 16, 33]:
        l = lucid_fn(M, **kw_l).numpy()
        r = getattr(ref.signal.windows, ref_attr)(M, **kw_r).detach().cpu().numpy()
        assert_close(l, r, atol=atol, msg=f"{name}(M={M})")


@pytest.mark.parity
def test_hann_sym_vs_periodic(ref: Any) -> None:
    l_sym = W.hann(8, sym=True).numpy()
    l_per = W.hann(8, sym=False).numpy()
    r_sym = ref.signal.windows.hann(8, sym=True).detach().cpu().numpy()
    r_per = ref.signal.windows.hann(8, sym=False).detach().cpu().numpy()
    assert_close(l_sym, r_sym, atol=1e-6, msg="hann sym")
    assert_close(l_per, r_per, atol=1e-6, msg="hann periodic")


@pytest.mark.parity
def test_kaiser_beta_sweep(ref: Any) -> None:
    for beta in [0.0, 2.0, 6.0, 14.0]:
        l = W.kaiser(16, beta=beta).numpy()
        r = ref.signal.windows.kaiser(16, beta=beta).detach().cpu().numpy()
        assert_close(l, r, atol=1e-4, msg=f"kaiser beta={beta}")


@pytest.mark.parity
def test_gaussian_std_sweep(ref: Any) -> None:
    for std in [0.5, 1.0, 3.0]:
        l = W.gaussian(16, std=std).numpy()
        r = ref.signal.windows.gaussian(16, std=std).detach().cpu().numpy()
        assert_close(l, r, atol=1e-6, msg=f"gaussian std={std}")
