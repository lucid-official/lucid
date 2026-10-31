"""A recurrent layer's weights can be reparametrized (LCD-780).

``LSTM.forward`` and the GRU / RNN ``_weights`` read ``self._parameters[...]``
directly.  Weight norm, an orthogonal parametrization, spectral norm and a
prune mask all take a weight out of that registry (it becomes a derived
attribute), so the forward raised ``KeyError: 'weight_ih_l0'`` the first
time it ran.  Every weight of every layer and direction — and the LSTM
projection — is covered here, forward and backward, against the reference.

The second half is the class-wide guard: every ``nn`` module class that owns
a parameter still runs its forward when that parameter is replaced by an
identity parametrization.  A forward that reaches into the registry instead
of the attribute fails there, whichever class it lives in.
"""

import importlib
import warnings
from types import ModuleType

import numpy as np
import pytest

import lucid
import lucid.nn as nn
from lucid.nn.utils import parametrize
from lucid.test.audit._axes import _forward_inputs, _module_candidates, _module_input_shapes
from lucid.test.audit._surface import enumerate_surface

_MODULE_SUBSYSTEMS = frozenset({"nn", "nn.quantized", "nn.qat", "nn.intrinsic"})


def _prune_of(utils: ModuleType) -> ModuleType:
    """``utils.prune``, which a framework only binds once somebody imports it."""
    return importlib.import_module(f"{utils.__name__}.prune")


#: name of the reparametrization -> how it is applied to ``module.<name>``,
#: given the ``nn.utils`` of the framework in use.
_KINDS = {
    "weight_norm_legacy": lambda U, m, n: U.weight_norm(m, n),
    "weight_norm": lambda U, m, n: U.parametrizations.weight_norm(m, n),
    "orthogonal": lambda U, m, n: U.parametrizations.orthogonal(m, n),
    "spectral_norm_legacy": lambda U, m, n: U.spectral_norm(m, n),
    "prune": lambda U, m, n: _prune_of(U).l1_unstructured(m, n, amount=0.3),
}

#: (class name, bidirectional, proj_size)
_CONFIGS = [
    ("RNN", False, 0),
    ("RNN", True, 0),
    ("GRU", False, 0),
    ("GRU", True, 0),
    ("LSTM", False, 0),
    ("LSTM", True, 0),
    ("LSTM", False, 3),
    ("LSTM", True, 3),
]


def _weight_names(cls: str, bidirectional: bool, proj: int) -> list[str]:
    """One weight of each kind: layer 0 and 1, both directions, the projection."""
    names = ["weight_ih_l0", "weight_hh_l1", "weight_ih_l1"]
    if bidirectional:
        names.append("weight_hh_l0_reverse")
    if proj:
        names.append("weight_hr_l0")
    return names


_CASES = [
    pytest.param(kind, cls, bi, proj, name, id=f"{kind}-{cls}-bi{int(bi)}-proj{proj}-{name}")
    for kind in _KINDS
    for cls, bi, proj in _CONFIGS
    for name in _weight_names(cls, bi, proj)
]


def _as_lucid(state: dict) -> dict[str, lucid.Tensor]:
    return {k: lucid.tensor(v.detach().numpy()) for k, v in state.items()}


def _loss(out, hidden):
    return (out**2).sum() + sum(h.sum() for h in hidden)


@pytest.mark.parity
@pytest.mark.parametrize("kind, cls, bidirectional, proj, name", _CASES)
def test_forward_and_backward_match_the_reference(ref, kind, cls, bidirectional, proj, name):
    kwargs = dict(num_layers=2, bidirectional=bidirectional)
    if proj:
        kwargs["proj_size"] = proj
    ref_m = getattr(ref.nn, cls)(3, 4, **kwargs)
    our_m = getattr(nn, cls)(3, 4, **kwargs)
    our_m.load_state_dict(_as_lucid(ref_m.state_dict()))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # the reference deprecates its legacy hooks
        _KINDS[kind](ref.nn.utils, ref_m, name)
    _KINDS[kind](nn.utils, our_m, name)
    # The reference's own state, so every derived piece (u / v vectors, the
    # prune mask, the orthogonal base) starts from the same values.
    our_m.load_state_dict(_as_lucid(ref_m.state_dict()))

    assert sorted(n for n, _ in our_m.named_parameters()) == sorted(
        n for n, _ in ref_m.named_parameters()
    )

    x = np.random.default_rng(0).standard_normal((4, 2, 3)).astype("float32")
    rx, ox = ref.tensor(x, requires_grad=True), lucid.tensor(x, requires_grad=True)
    r_out, r_hid = ref_m(rx)
    o_out, o_hid = our_m(ox)
    r_hid = r_hid if isinstance(r_hid, tuple) else (r_hid,)
    o_hid = o_hid if isinstance(o_hid, tuple) else (o_hid,)
    np.testing.assert_allclose(o_out.numpy(), r_out.detach().numpy(), rtol=1e-4, atol=1e-4)
    for o, r in zip(o_hid, r_hid, strict=True):
        np.testing.assert_allclose(o.numpy(), r.detach().numpy(), rtol=1e-4, atol=1e-4)

    _loss(r_out, r_hid).backward()
    _loss(o_out, o_hid).backward()
    np.testing.assert_allclose(ox.grad.numpy(), rx.grad.numpy(), rtol=1e-3, atol=1e-4)
    ref_grads = {n: p.grad for n, p in ref_m.named_parameters() if p.grad is not None}
    our_grads = {n: p.grad for n, p in our_m.named_parameters() if p.grad is not None}
    assert sorted(our_grads) == sorted(ref_grads)
    for n, g in our_grads.items():
        np.testing.assert_allclose(g.numpy(), ref_grads[n].numpy(), rtol=1e-3, atol=1e-4, err_msg=n)


class _Identity(nn.Module):
    """A parametrization that changes nothing: any difference is the registry's doing."""

    def forward(self, x: lucid.Tensor) -> lucid.Tensor:
        return x


def _tensors(out: object) -> list[lucid.Tensor]:
    if isinstance(out, (tuple, list)):
        return [t for item in out for t in _tensors(item)]
    return [out] if isinstance(out, lucid.Tensor) else []


def _first_forward(cls: type) -> tuple[nn.Module, list[object], object] | None:
    """A built instance of ``cls`` with arguments its forward accepts, and the result.

    The audit's own ladder: its constructor candidates and the input shapes
    each module suggests.  ``eval`` so dropout and sampling are not the
    difference between two calls.
    """
    for candidate in _module_candidates(cls):
        candidate.eval()
        for shape in _module_input_shapes(candidate):
            for args in _forward_inputs(candidate, shape):
                try:
                    out = candidate(*args)
                except Exception:  # noqa: BLE001 - a probe: the next shape may fit
                    continue
                return candidate, args, out
    return None


def test_every_module_class_runs_forward_under_an_identity_parametrization() -> None:
    """The class-wide guard: parametrizing a parameter must not break the forward.

    Every ``nn`` module class the audit can build and run has each direct
    parameter of every (sub)module replaced by an identity parametrization,
    and must produce what it produced before.  A forward that reads
    ``self._parameters[...]`` instead of the attribute fails here whatever
    class it lives in.
    """
    exercised = 0
    failures: list[str] = []
    for symbol in enumerate_surface():
        if symbol.kind != "module" or symbol.subsystem not in _MODULE_SUBSYSTEMS:
            continue
        built = _first_forward(symbol.obj)
        if built is None:
            continue
        module, args, expect = built
        names = [
            (sub, name)
            for sub in list(module.modules())
            for name, p in sub._parameters.items()
            if p is not None
        ]
        if not names:
            continue
        for sub, name in names:
            parametrize.register_parametrization(sub, name, _Identity())
        try:
            got = module(*args)
        except Exception as exc:  # noqa: BLE001 - reported with the class name
            failures.append(f"{symbol.qualname}: {type(exc).__name__}: {str(exc)[:80]}")
            continue
        exercised += 1
        for g, e in zip(_tensors(got), _tensors(expect), strict=True):
            if not np.allclose(g.numpy(), e.numpy(), rtol=1e-4, atol=1e-4, equal_nan=True):
                failures.append(f"{symbol.qualname}: output changed")
                break
    assert not failures, failures
    # Not vacuous: 47 classes qualified when this was written.
    assert exercised >= 40, exercised
