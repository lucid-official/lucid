"""Every way of copying a model must leave the copy's storage its own (LCD-686).

``Parameter.__deepcopy__`` re-wrapped ``self`` as ``Parameter(self)``, which
binds the source's buffer rather than cloning it, so ``copy.deepcopy(model)``
returned parameters sharing storage with the original: an optimizer step on
the "copy" moved the source.  Every non-inplace entry of the quantization and
fusion toolkits copies through ``copy.deepcopy``, so they all inherited it.

The defect class is "a copy entry point returns tensors that alias the source",
and the invariant is checked at the one place all of them reach: write into
every parameter and buffer of the copy in place, then require the source to be
bit-for-bit what it was.  The same models are built on the reference, where the
copy keeps ``requires_grad``, the Parameter type and the weight ties, and drops
``.grad`` — pinned here so that a "fix" cannot copy more than the reference does.
"""

import copy
from collections.abc import Callable
from types import ModuleType

import pytest

import lucid
import lucid.nn as nn
import lucid.nn.utils.fusion as fusion
import lucid.quantization as Q


def _conv_bn(lib: ModuleType, device: str):
    net = lib.nn.Sequential(lib.nn.Conv2d(2, 3, 3), lib.nn.BatchNorm2d(3), lib.nn.ReLU())
    net.to(device)
    net(lib.ones(2, 2, 5, 5, device=device))  # moves the running statistics
    return net


def _linear(lib: ModuleType, device: str):
    return lib.nn.Linear(4, 3).to(device)


def _tied(lib: ModuleType, device: str):
    net = lib.nn.Sequential(lib.nn.Linear(3, 3), lib.nn.Linear(3, 3)).to(device)
    net[1].weight = net[0].weight
    return net


def _shared_child(lib: ModuleType, device: str):
    shared = lib.nn.Linear(3, 3)
    return lib.nn.Sequential(shared, lib.nn.ReLU(), shared).to(device)


_MODELS: dict[str, Callable[[ModuleType, str], object]] = {
    "linear": _linear,
    "conv-bn": _conv_bn,
    "tied-weight": _tied,
    "shared-child": _shared_child,
}


def _tensors(module) -> list:
    return [t for _, t in module.named_parameters()] + [t for _, t in module.named_buffers()]


def _values(module) -> list[list[float]]:
    return [t.detach().cpu().reshape(-1).tolist() for t in _tensors(module)]


def _overwrite(lib: ModuleType, module) -> None:
    with lib.no_grad():
        for t in _tensors(module):
            t.fill_(7)


def _tie_groups(module) -> list[list[str]]:
    seen: dict[int, list[str]] = {}
    for name, p in module.named_parameters(remove_duplicate=False):
        seen.setdefault(id(p), []).append(name)
    return sorted(seen.values())


@pytest.mark.parametrize("name", list(_MODELS))
def test_deepcopy_owns_its_storage(name: str, device: str) -> None:
    source = _MODELS[name](lucid, device)
    before = _values(source)
    twin = copy.deepcopy(source)
    _overwrite(lucid, twin)
    assert _values(source) == before
    assert _values(twin) != before


def test_an_optimizer_step_on_the_copy_leaves_the_source(device: str) -> None:
    source = lucid.nn.Linear(4, 3).to(device)
    before = _values(source)
    twin = copy.deepcopy(source)
    twin(lucid.ones(2, 4, device=device)).sum().backward()
    lucid.optim.SGD(twin.parameters(), lr=1.0).step()
    assert _values(source) == before
    assert _values(twin) != before
    assert source.weight.grad is None


@pytest.mark.parity
@pytest.mark.parametrize("name", list(_MODELS))
def test_deepcopy_keeps_what_the_reference_keeps(name: str, ref: ModuleType, device: str) -> None:
    mine = copy.deepcopy(_MODELS[name](lucid, device))
    theirs = copy.deepcopy(_MODELS[name](ref, "cpu"))
    assert _tie_groups(mine) == _tie_groups(theirs)
    assert [n for n, _ in mine.named_parameters()] == [n for n, _ in theirs.named_parameters()]
    assert [p.requires_grad for p in mine.parameters()] == [
        p.requires_grad for p in theirs.parameters()
    ]
    assert all(isinstance(p, nn.Parameter) for p in mine.parameters())


@pytest.mark.parity
@pytest.mark.parametrize("requires_grad", [True, False])
def test_parameter_deepcopy_matches_the_reference(
    requires_grad: bool, ref: ModuleType, device: str
) -> None:
    def build(lib: ModuleType, dev: str):
        p = lib.nn.Parameter(lib.ones(3, device=dev), requires_grad=requires_grad)
        if requires_grad:
            (p * 2).sum().backward()
        return p

    mine = copy.deepcopy(build(lucid, device))
    theirs = copy.deepcopy(build(ref, "cpu"))
    assert type(mine) is nn.Parameter
    assert mine.requires_grad == theirs.requires_grad == requires_grad
    assert (mine.grad is None) == (theirs.grad is None)
    assert mine.tolist() == theirs.tolist()


def test_a_parameter_subclass_and_its_attributes_survive(device: str) -> None:
    class Tagged(nn.Parameter):
        pass

    source = Tagged(lucid.ones(3, device=device))
    source.tags = ["a"]
    twin = copy.deepcopy(source)
    assert type(twin) is Tagged and twin.tags == ["a"] and twin.tags is not source.tags
    with lucid.no_grad():
        twin.fill_(5)
    assert source.tolist() == [1.0, 1.0, 1.0]


def test_a_pending_lazy_parameter_stays_pending() -> None:
    twin = copy.deepcopy(nn.LazyLinear(3))
    assert twin.has_uninitialized_parameters()
    assert type(twin.weight) is nn.parameter.UninitializedParameter


def test_a_parameter_in_a_cycle_is_copied_once() -> None:
    source = nn.Parameter(lucid.ones(2))
    source.me = source
    twin = copy.deepcopy(source)
    assert twin.me is twin


# ── the toolkit entry points that copy through deepcopy ─────────────────────


def _qconfig() -> object:
    return Q.get_default_qconfig_mapping()


def _qat_config() -> object:
    return Q.get_default_qat_qconfig_mapping()


def _mlp() -> nn.Module:
    return nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 2)).eval()


def _conv_bn_relu() -> nn.Module:
    net = nn.Sequential(nn.Conv2d(2, 3, 3), nn.BatchNorm2d(3), nn.ReLU())
    net(lucid.ones(2, 2, 5, 5))
    return net.eval()


def _prepared() -> nn.Module:
    return Q.prepare(_mlp(), _qconfig(), inplace=True)


def _calibrated() -> nn.Module:
    net = _prepared()
    net(lucid.ones(2, 4))
    return net


_ENTRIES: dict[str, tuple[Callable[[], nn.Module], Callable[[nn.Module], nn.Module]]] = {
    "prepare": (_mlp, lambda m: Q.prepare(m, _qconfig())),
    "prepare_qat": (_mlp, lambda m: Q.prepare_qat(m, _qat_config())),
    "convert": (_calibrated, lambda m: Q.convert(m)),
    "quantize_dynamic": (_mlp, lambda m: Q.quantize_dynamic(m)),
    "fuse_modules": (_conv_bn_relu, lambda m: Q.fuse_modules(m, [["0", "1", "2"]])),
    "fuse_modules_qat": (
        _conv_bn_relu,
        lambda m: Q.fuse_modules_qat(m, [["0", "1", "2"]], Q.get_default_qat_qconfig()),
    ),
}


@pytest.mark.parametrize("entry", list(_ENTRIES))
def test_a_non_inplace_toolkit_entry_does_not_alias_its_source(entry: str) -> None:
    build, run = _ENTRIES[entry]
    source = build()
    before = _values(source)
    result = run(source)
    assert result is not source
    _overwrite(lucid, result)
    assert _values(source) == before


def test_fuse_conv_bn_eval_does_not_alias_the_conv() -> None:
    net = _conv_bn_relu()
    conv, bn = net[0], net[1]
    before = _values(conv)
    fused = fusion.fuse_conv_bn_eval(conv, bn)
    _overwrite(lucid, fused)
    assert _values(conv) == before


def test_fuse_linear_bn_eval_does_not_alias_the_linear() -> None:
    linear, bn = nn.Linear(4, 3), nn.BatchNorm1d(3)
    linear(lucid.ones(2, 4))
    before = _values(linear)
    fused = fusion.fuse_linear_bn_eval(linear.eval(), bn.eval())
    _overwrite(lucid, fused)
    assert _values(linear) == before
