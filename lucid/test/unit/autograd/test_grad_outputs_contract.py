"""Seed gradients are matched to their outputs at the boundary (LCD-251).

``autograd.grad`` indexed ``grad_outputs`` as a list whatever it was: a bare
tensor was read row by row ("grad_outputs[0] shape (3,) does not match output
shape (2, 3)"), a short list raised IndexError and a long one had its extras
dropped.  ``autograd.backward`` zipped roots with ``grad_tensors``, so a
missing gradient silently dropped its root.  A non-scalar output with no seed
was differentiated as if its seed were ones, and a seed of another real dtype
promoted the output (float64 seed, float32 output) and lost the gradient.

One normaliser in ``autograd/_backward.py`` now serves ``grad``,
``backward`` and ``Tensor.backward``: a single tensor or ``None`` stands for a
one-element sequence, the counts must agree, and each seed must fit its
output.
"""

from types import ModuleType

import pytest

import lucid
from lucid._C import engine as _C_engine

_FORMS = ["bare", "list", "none"]
_COUNTS = ["short", "exact", "long"]


def _call(lib: ModuleType, device: str, api: str, form: str, count: str) -> object:
    """``x``'s gradient, or the class of what was raised."""
    x = lib.ones(2, 3, device=device, requires_grad=True)
    y = x * 2
    outs = [y] if form == "bare" else [y, (x * 3).sum()]
    seeds: list[object] = [lib.ones(2, 3, device=device), None][: len(outs)]
    if count == "short":
        seeds = seeds[:-1]
    elif count == "long":
        seeds = [*seeds, lib.ones(2, 3, device=device)]
    if form == "bare":
        root: object = outs[0]
        given: object = seeds[0] if len(seeds) == 1 else seeds
    elif form == "list":
        root, given = outs, seeds
    else:
        root, given = outs, None
    try:
        if api == "grad":
            (g,) = lib.autograd.grad(root, x, grad_outputs=given)
            return g.tolist()
        lib.autograd.backward(root, given)
        return x.grad.tolist()
    except (RuntimeError, TypeError, IndexError) as exc:
        return type(exc)


def _kind(result: object) -> object:
    """A list of values as it is; any refusal as one class — the reference's
    and Lucid's exception classes are not the same objects."""
    if isinstance(result, type):
        return (
            "IndexError"
            if issubclass(result, IndexError)
            else ("TypeError" if issubclass(result, TypeError) else "RuntimeError")
        )
    return result


@pytest.mark.parity
@pytest.mark.parametrize("count", _COUNTS)
@pytest.mark.parametrize("form", _FORMS)
@pytest.mark.parametrize("api", ["grad", "backward"])
def test_matches_the_reference(
    api: str, form: str, count: str, device: str, ref: ModuleType
) -> None:
    got = _kind(_call(lucid, device, api, form, count))
    if count == "long":
        # Stricter than the reference on purpose: it drops the extra seeds
        # without a word, which hides a caller that lined its seeds up
        # with the wrong outputs.
        assert got == "RuntimeError"
        return
    want = _kind(_call(ref, "cpu", api, form, count))
    assert got == want
    assert got != "IndexError"


def test_a_bare_tensor_is_one_seed(device: str) -> None:
    x = lucid.ones(2, 3, device=device, requires_grad=True)
    y = x * 2
    (g,) = lucid.autograd.grad(y, x, grad_outputs=lucid.ones(2, 3, device=device))
    assert g is not None and g.tolist() == [[2.0] * 3] * 2


@pytest.mark.parametrize("api", ["grad", "backward"])
@pytest.mark.parametrize("n_seeds", [0, 2])
def test_the_counts_must_agree(api: str, n_seeds: int, device: str) -> None:
    x = lucid.ones(2, device=device, requires_grad=True)
    y = x * 2
    seeds = [lucid.ones(2, device=device)] * n_seeds
    with pytest.raises(RuntimeError, match=f"got 1 tensors and {n_seeds} gradients"):
        if api == "grad":
            lucid.autograd.grad([y], x, grad_outputs=seeds)
        else:
            lucid.autograd.backward([y], seeds)
    assert x.grad is None


def test_backward_does_not_drop_a_root_without_a_gradient(device: str) -> None:
    x = lucid.ones(2, device=device, requires_grad=True)
    y = x * 2
    with pytest.raises(RuntimeError, match="got 2 tensors and 1 gradients"):
        lucid.autograd.backward([y, y * 3], [lucid.ones(2, device=device)])
    assert x.grad is None


@pytest.mark.parametrize("api", ["grad", "backward", "method"])
def test_a_non_scalar_output_needs_a_seed(api: str, device: str) -> None:
    x = lucid.ones(2, device=device, requires_grad=True)
    y = x * 2
    with pytest.raises(RuntimeError, match="only for scalar outputs"):
        if api == "grad":
            lucid.autograd.grad(y, x)
        elif api == "backward":
            lucid.autograd.backward(y)
        else:
            y.backward()


@pytest.mark.parametrize("api", ["grad", "backward"])
def test_a_seed_that_is_not_a_tensor_is_a_type_error(api: str) -> None:
    x = lucid.ones(2, requires_grad=True)
    y = x * 2
    with pytest.raises(TypeError, match="must be a Tensor or None"):
        if api == "grad":
            lucid.autograd.grad([y], x, grad_outputs=[[1.0, 1.0]])
        else:
            lucid.autograd.backward([y], [[1.0, 1.0]])


@pytest.mark.parametrize("api", ["grad", "backward", "method"])
def test_a_seed_of_another_shape_is_refused(api: str, device: str) -> None:
    x = lucid.ones(2, 3, device=device, requires_grad=True)
    y = x * 2
    seed = lucid.ones(3, device=device)
    with pytest.raises(_C_engine.ShapeMismatch, match="mismatch in shape"):
        if api == "grad":
            lucid.autograd.grad(y, x, grad_outputs=seed)
        elif api == "backward":
            lucid.autograd.backward(y, seed)
        else:
            y.backward(seed)


def test_a_seed_on_another_device_is_refused(device_gpu_only: str) -> None:
    x = lucid.ones(2, device="cpu", requires_grad=True)
    y = x * 2
    with pytest.raises(_C_engine.DeviceMismatch):
        lucid.autograd.grad(y, x, grad_outputs=lucid.ones(2, device=device_gpu_only))


@pytest.mark.parametrize("api", ["grad", "method"])
def test_a_seed_of_another_real_dtype_is_cast_to_the_output(api: str, device: str) -> None:
    if device == "metal":
        pytest.skip("float64 is CPU-only")
    x = lucid.ones(2, device=device, requires_grad=True)
    y = x * 2
    seed = lucid.full((2,), 3.0, dtype=lucid.float64, device=device)
    if api == "grad":
        (g,) = lucid.autograd.grad(y, x, grad_outputs=seed)
    else:
        y.backward(seed)
        g = x.grad
    assert g is not None and g.dtype == lucid.float32
    assert g.tolist() == [6.0, 6.0]


def test_backward_with_a_seed_runs_under_no_grad(device: str) -> None:
    # The reference runs backward whatever the grad mode; the seed product
    # built here used to be untracked under ``no_grad`` and reached nothing.
    x = lucid.ones(2, device=device, requires_grad=True)
    y = x * 2
    with lucid.no_grad():
        y.backward(lucid.ones(2, device=device))
    assert x.grad is not None and x.grad.tolist() == [2.0, 2.0]


def test_backward_does_not_accumulate_into_a_seed_that_requires_grad(device: str) -> None:
    x = lucid.ones(2, device=device, requires_grad=True)
    seed = lucid.ones(2, device=device, requires_grad=True)
    (x * 2).backward(seed, create_graph=True)
    assert seed.grad is None
