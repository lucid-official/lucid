"""The module kinds LoHa and LoKr share: geometry, applying a weight, and the Tucker contraction.

A kind turns a base module into a weight-shaped geometry and applies a
replacement weight to an input the way the module applies its own, so the
update of an adapter is a convolution (or a matrix product) with the module's
stride, padding, dilation, groups and padding mode.  The contraction is checked
against the einsum it stands for.
"""

import pytest

import lucid
import lucid.einops as einops
import lucid.nn as nn
from lucid.peft.tuners._kinds import _Geometry, _Kind, _tucker


def _close(a: lucid.Tensor, b: lucid.Tensor, atol: float = 1e-5) -> bool:
    return bool(lucid.allclose(a, b, atol=atol))


class TestGeometry:
    def test_a_linear_weight_is_out_by_in_with_no_kernel(self) -> None:
        geometry = _Kind.of(nn.Linear).geometry(nn.Linear(8, 5))
        assert geometry == _Geometry(5, 8, ())
        assert geometry.weight_shape == (5, 8) and geometry.flat_in == 8
        assert not geometry.has_kernel and geometry.pointwise

    def test_a_convolution_counts_input_channels_per_group(self) -> None:
        geometry = _Kind.of(nn.Conv2d).geometry(nn.Conv2d(4, 6, (3, 2), groups=2))
        assert geometry == _Geometry(6, 2, (3, 2))
        assert geometry.weight_shape == (6, 2, 3, 2) and geometry.flat_in == 2 * 3 * 2
        assert geometry.has_kernel and not geometry.pointwise
        assert _Kind.of(nn.Conv2d).geometry(nn.Conv2d(4, 6, 1)).pointwise
        assert _Kind.of(nn.Conv1d).geometry(nn.Conv1d(3, 4, 5)) == _Geometry(4, 3, (5,))

    def test_a_kind_refuses_a_module_of_another_type(self) -> None:
        with pytest.raises(TypeError, match="expected an nn.Conv2d"):
            _Kind.of(nn.Conv2d).geometry(nn.Linear(2, 2))
        with pytest.raises(TypeError, match="expected an nn.Linear"):
            _Kind.of(nn.Linear).geometry(nn.Conv2d(2, 2, 1))

    def test_a_type_with_no_kind_is_named(self) -> None:
        with pytest.raises(TypeError, match="no LyCORIS kind for Conv3d"):
            _Kind.of(nn.Conv3d)


class TestApply:
    def test_linear_applies_the_weight_as_a_matrix_product(self) -> None:
        base = nn.Linear(8, 5)
        x = lucid.randn(3, 8)
        weight = lucid.randn(5, 8)
        assert _close(_Kind.of(nn.Linear).apply(base, x, weight), x @ weight.mT)

    @pytest.mark.parametrize("mode", ["zeros", "reflect", "replicate", "circular"])
    def test_a_convolution_follows_the_base_s_padding_mode(self, mode: str) -> None:
        lucid.manual_seed(0)
        base = nn.Conv2d(3, 4, 3, padding=1, padding_mode=mode, bias=False)  # type: ignore[arg-type]
        x = lucid.randn(2, 3, 6, 6)
        weight = lucid.randn(4, 3, 3, 3)
        probe = nn.Conv2d(3, 4, 3, padding=1, padding_mode=mode, bias=False)  # type: ignore[arg-type]
        with lucid.no_grad():
            probe.weight.copy_(weight)
        assert _close(_Kind.of(nn.Conv2d).apply(base, x, weight), probe(x), atol=1e-5)

    def test_a_convolution_follows_stride_dilation_and_groups(self) -> None:
        lucid.manual_seed(0)
        base = nn.Conv1d(4, 6, 3, stride=2, padding=2, dilation=2, groups=2, bias=False)
        x = lucid.randn(2, 4, 9)
        weight = lucid.randn(6, 2, 3)
        probe = nn.Conv1d(4, 6, 3, stride=2, padding=2, dilation=2, groups=2, bias=False)
        with lucid.no_grad():
            probe.weight.copy_(weight)
        assert _close(_Kind.of(nn.Conv1d).apply(base, x, weight), probe(x), atol=1e-5)


class TestTucker:
    @pytest.mark.parametrize("spatial", [(3, 3), (2, 1), (1, 4)])
    def test_it_is_the_einsum_it_stands_for(self, spatial: tuple[int, int]) -> None:
        lucid.manual_seed(0)
        core = lucid.randn(3, 3, *spatial)
        rows, columns = lucid.randn(3, 5), lucid.randn(3, 6)
        expected = einops.einsum("ijkl,ip,jq->pqkl", core, rows, columns)
        assert _close(_tucker(core, rows, columns), expected, atol=1e-5)

    def test_it_is_differentiable_in_all_three_arguments(self) -> None:
        core = lucid.randn(2, 2, 3, 3, requires_grad=True)
        rows = lucid.randn(2, 4, requires_grad=True)
        columns = lucid.randn(2, 5, requires_grad=True)
        _tucker(core, rows, columns).sum().backward()
        for tensor in (core, rows, columns):
            assert tensor.grad is not None and float(tensor.grad.abs().max().item()) > 0
