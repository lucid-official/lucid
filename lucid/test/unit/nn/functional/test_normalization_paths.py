"""Normalisation branches the suite never entered.

Mostly rank dispatch and argument validation — the parts of a normaliser
that only run when someone passes the unusual shape, and so the parts a
refactor silently breaks.  The one with teeth is ``rms_norm`` over more
than one trailing axis: the engine normalises over the last axis only, so
multi-axis support is a reshape the module does itself, and before it
existed ``normalized_shape`` was accepted and ignored — every call
normalised over the final axis whatever was asked for.  A shape assertion
would have passed then, so these compare against the definition.
"""

import numpy as np
import pytest

import lucid
import lucid.nn.functional as F

# ── batch_norm rank dispatch ─────────────────────────────────────────────────


def test_batch_norm_accepts_two_dimensional_input() -> None:
    """``(N, C)`` is routed through the 1-D kernel and comes back ``(N, C)``.

    Compared against the definition rather than against itself: the
    unsqueeze/squeeze round trip could return the input untouched and a
    shape check would not notice.
    """
    lucid.manual_seed(0)
    x = lucid.randn(8, 4)
    out = F.batch_norm(x, None, None, training=True).numpy()
    assert out.shape == (8, 4)

    xn = x.numpy()
    want = (xn - xn.mean(0)) / np.sqrt(xn.var(0) + 1e-5)
    np.testing.assert_allclose(out, want, rtol=1e-4, atol=1e-4)


def test_batch_norm_matches_across_ranks() -> None:
    """The same channel statistics, whichever rank carries them."""
    lucid.manual_seed(0)
    flat = lucid.randn(8, 4)
    two = F.batch_norm(flat, None, None, training=True).numpy()
    three = F.batch_norm(flat.reshape(8, 4, 1), None, None, training=True).numpy()
    np.testing.assert_allclose(two, three.reshape(8, 4), rtol=1e-5, atol=1e-5)


def test_batch_norm_refuses_an_unsupported_rank() -> None:
    lucid.manual_seed(0)
    with pytest.raises(ValueError, match="expected 2–5D input"):
        F.batch_norm(lucid.randn(2, 3, 4, 5, 6, 7), None, None, training=True)


def test_batch_norm_functional_rejects_one_value_per_channel_in_training() -> None:
    x = lucid.tensor(np.array([[1.0, 2.0, 3.0]], dtype=np.float32))
    with pytest.raises(ValueError, match="more than one value per channel"):
        F.batch_norm(x, None, None, training=True)


def test_batch_norm_functional_allows_single_batch_with_spatial_values() -> None:
    x = lucid.tensor(np.array([[[1.0, 2.0], [3.0, 4.0]]], dtype=np.float32))
    out = F.batch_norm(x, None, None, training=True)
    assert out.shape == x.shape


def test_batch_norm_functional_allows_single_value_per_channel_in_eval() -> None:
    x = lucid.tensor(np.array([[1.0, 2.0, 3.0]], dtype=np.float32))
    out = F.batch_norm(x, lucid.zeros(3), lucid.ones(3), training=False)
    assert out.shape == x.shape


# ── rms_norm over more than one axis ─────────────────────────────────────────


def _rms_reference(x: np.ndarray, axes: tuple[int, ...], eps: float) -> np.ndarray:
    ms = np.mean(x.astype(np.float64) ** 2, axis=axes, keepdims=True)
    return (x / np.sqrt(ms + eps)).astype(np.float32)


def test_rms_norm_over_two_axes_uses_both() -> None:
    """``normalized_shape=(H, W)`` must normalise over H *and* W.

    The failure this is here for is the one the module documents: the
    argument accepted, the reduction still over the last axis alone.  So
    the check is against the two-axis definition, and — below — against
    the one-axis answer being *different*.
    """
    lucid.manual_seed(0)
    x = lucid.randn(2, 3, 4, 5)
    eps = 1e-8
    got = F.rms_norm(x, (4, 5), eps=eps).numpy()
    want = _rms_reference(x.numpy(), (-2, -1), eps)
    np.testing.assert_allclose(got, want, rtol=1e-4, atol=1e-5)

    last_only = F.rms_norm(x, (5,), eps=eps).numpy()
    assert not np.allclose(
        got, last_only, atol=1e-4
    ), "two-axis and one-axis rms_norm agree — normalized_shape is being ignored"


def test_rms_norm_over_three_axes() -> None:
    lucid.manual_seed(0)
    x = lucid.randn(2, 3, 4, 5)
    got = F.rms_norm(x, (3, 4, 5), eps=1e-8).numpy()
    want = _rms_reference(x.numpy(), (-3, -2, -1), 1e-8)
    np.testing.assert_allclose(got, want, rtol=1e-4, atol=1e-5)


def test_rms_norm_multi_axis_applies_the_weight() -> None:
    """The weight is flattened to match the collapsed axes, not dropped."""
    lucid.manual_seed(0)
    x = lucid.randn(2, 3, 4, 5)
    plain = F.rms_norm(x, (4, 5), eps=1e-8).numpy()
    scaled = F.rms_norm(x, (4, 5), weight=lucid.ones(20) * 2.0, eps=1e-8).numpy()
    np.testing.assert_allclose(scaled, plain * 2.0, rtol=1e-4, atol=1e-5)


def test_rms_norm_rejects_an_empty_shape() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        F.rms_norm(lucid.ones(2, 3), ())


def test_rms_norm_rejects_a_mismatched_shape() -> None:
    with pytest.raises(ValueError, match="does not match the trailing"):
        F.rms_norm(lucid.ones(2, 3, 4), (7, 4))


# ── instance_norm ────────────────────────────────────────────────────────────


def test_instance_norm_requires_a_spatial_axis() -> None:
    with pytest.raises(ValueError, match="at least 3-D"):
        F.instance_norm(lucid.ones(4, 3))


def test_instance_norm_can_use_running_stats_instead_of_the_batch() -> None:
    """``use_input_stats=False`` normalises by the buffers, not by ``x``.

    Proven by giving it running statistics that are deliberately *wrong*
    for the input: if the batch were used the output would be centred,
    and it must not be.
    """
    lucid.manual_seed(0)
    x = lucid.randn(2, 3, 8, 8) + 5.0
    running_mean = lucid.zeros(3)
    running_var = lucid.ones(3)

    out = F.instance_norm(
        x, running_mean=running_mean, running_var=running_var, use_input_stats=False
    ).numpy()

    # mean 0 / var 1 buffers make this the identity, so the +5 offset has
    # to survive; per-instance statistics would have removed it.
    np.testing.assert_allclose(out, x.numpy(), rtol=1e-4, atol=1e-4)

    centred = F.instance_norm(x, use_input_stats=True).numpy()
    assert abs(float(centred.mean())) < 1e-3, "input stats did not centre the input"


# ── local_response_norm ──────────────────────────────────────────────────────


@pytest.mark.parametrize("shape", [(), (3,), (2, 4)], ids=str)
def test_local_response_norm_refuses_inputs_under_3d(ref, shape) -> None:
    """Inputs under 3-D have no spatial axes and raise ValueError, matching the reference."""
    with pytest.raises(ValueError, match="Expected 3D or higher dimensionality"):
        ref.nn.functional.local_response_norm(ref.ones(shape), size=2)
    with pytest.raises(ValueError, match="Expected 3D or higher dimensionality"):
        F.local_response_norm(lucid.ones(shape), size=2)


@pytest.mark.parametrize("size", [0, -1, -3])
def test_local_response_norm_refuses_non_positive_size(ref, size) -> None:
    with pytest.raises(RuntimeError):
        ref.nn.functional.local_response_norm(ref.ones(1, 4, 3), size=size)
    with pytest.raises(RuntimeError, match="kernel size should be greater than zero"):
        F.local_response_norm(lucid.ones(1, 4, 3), size=size)
