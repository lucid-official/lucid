"""Every weight average in the zoo pairs each parameter with its namesake.

An EMA teacher, a target encoder and a slow critic are all
``average ← d·average + (1 − d)·live``, through
``lucid.models._utils._common.update_averages``.  They used to zip the two
networks' ``parameters()`` by position (LCD-635, TVM in LCD-552), and
deleting a child and assigning it back moves it to the end of its
registry: the orders come apart, and where two neighbours agree in shape
the average mixes one into the other without an error.

Each case here re-orders the live network that way at a child with a
same-shaped sibling — so a positional pairing would pass silently — and
checks the average name by name.
"""

from dataclasses import dataclass

import pytest

import lucid
import lucid.nn as nn
from lucid.models import create_model
from lucid.models._utils._common import update_averages

_DREAMER = dict(
    action_dim=2,
    cnn_depth=4,
    stoch_size=4,
    discrete=5,
    deter_size=16,
    actor_hidden=16,
    value_hidden=16,
    reward_hidden=16,
    horizon=4,
    pcont=False,
)


@dataclass(frozen=True)
class _Case:
    factory: str
    overrides: dict[str, object]
    live: str
    # Each average by attribute path, with the weight its next update keeps.
    averages: tuple[tuple[str, float], ...]
    # The update method, called once more before the re-order where its
    # first call copies outright instead of averaging (Dreamer).
    update: str
    update_args: tuple[float, ...] = ()
    prime: bool = False


_CASES = {
    "ijepa": _Case(
        "ijepa_base_16",
        dict(
            image_size=32,
            patch_size=8,
            dim=16,
            depth=2,
            num_heads=2,
            predictor_dim=8,
            predictor_depth=1,
            min_keep=1,
            target_scale=(0.1, 0.2),
            context_scale=(0.8, 1.0),
        ),
        live="encoder",
        averages=(("target_encoder", 0.25),),
        update="update_target",
        update_args=(0.25,),
    ),
    "vjepa": _Case(
        "vjepa_large_16",
        dict(
            image_size=32,
            patch_size=8,
            tubelet_size=2,
            num_frames=4,
            dim=24,
            depth=2,
            num_heads=2,
            predictor_dim=12,
            predictor_depth=1,
            short_range_blocks=2,
            long_range_blocks=1,
        ),
        live="encoder",
        averages=(("target_encoder", 0.25),),
        update="update_target",
        update_args=(0.25,),
    ),
    "vjepa2": _Case(
        "vjepa2_vit_large",
        dict(
            image_size=16,
            patch_size=8,
            tubelet_size=2,
            num_frames=4,
            dim=24,
            depth=2,
            num_heads=2,
            predictor_dim=12,
            predictor_depth=1,
            predictor_heads=2,
            predictor_num_mask_tokens=2,
        ),
        live="encoder",
        averages=(("target_encoder", 0.25),),
        update="update_target",
        update_args=(0.25,),
    ),
    "dreamer_v2": _Case(
        "dreamer_v2_world_model",
        dict(_DREAMER, hidden_size=16, slow_target_update=1, slow_target_fraction=0.75),
        live="dreamer_v2.value_head",
        averages=(("dreamer_v2.target_value_head", 0.25),),
        update="update_slow_target",
        prime=True,
    ),
    "dreamer_v3": _Case(
        "dreamer_v3_12m_world_model",
        dict(_DREAMER, hidden_size=8, num_bins=7, blocks=4, critic_ema=0.75),
        live="dreamer_v3.value_head",
        averages=(("dreamer_v3.slow_value_head", 0.25),),
        update="update_slow_critic",
        prime=True,
    ),
    "tvm": _Case(
        "tvm_imagenet256_gen",
        dict(
            sample_size=8,
            hidden_size=32,
            depth=2,
            num_heads=4,
            num_classes=4,
            target_ema_decay=0.25,
            eval_ema_decay=0.5,
        ),
        live="tvm",
        averages=(("target_network", 0.25), ("ema_network", 0.5)),
        update="update_ema",
    ),
}


def _module(model: nn.Module, path: str) -> nn.Module:
    found: object = model
    for part in path.split("."):
        found = getattr(found, part)
    assert isinstance(found, nn.Module), path
    return found


def _update(model: nn.Module, case: _Case) -> None:
    getattr(model, case.update)(*case.update_args)


def _shapes(module: nn.Module) -> list[tuple[int, ...]]:
    return [tuple(p.shape) for p in module.parameters()]


def _move_a_twin_to_the_end(live: nn.Module) -> None:
    """Delete a child and assign it back, where that only rotates same shapes.

    The child moves past every later sibling, so the shapes stay aligned
    position by position only when it and all of those siblings agree.
    """
    for _, parent in live.named_modules():
        children = list(parent.named_children())
        for i, (name, child) in enumerate(children[:-1]):
            shapes = _shapes(child)
            if shapes and all(_shapes(other) == shapes for _, other in children[i:]):
                delattr(parent, name)
                setattr(parent, name, child)
                return
    raise AssertionError("the config has no trailing run of same-shaped children")


def _snapshot(module: nn.Module) -> dict[str, lucid.Tensor]:
    return {name: p.detach().clone() for name, p in module.named_parameters()}


@pytest.mark.parametrize("family", list(_CASES))
def test_the_average_pairs_parameters_by_name(family: str) -> None:
    case = _CASES[family]
    lucid.manual_seed(0)
    model = create_model(case.factory, **case.overrides)
    if case.prime:
        _update(model, case)
    live = _module(model, case.live)
    averages = [(_module(model, path), keep) for path, keep in case.averages]
    built = [name for name, _ in averages[0][0].named_parameters()]

    _move_a_twin_to_the_end(live)
    order = [name for name, _ in live.named_parameters()]
    assert order != built and sorted(order) == sorted(built)
    # A positional pairing would meet no shape error: every displaced
    # position still holds a same-shaped parameter.
    by_name = dict(live.named_parameters())
    named = dict(averages[0][0].named_parameters())
    assert all(
        tuple(by_name[a].shape) == tuple(named[b].shape)
        for a, b in zip(order, built, strict=True)
    )

    with lucid.no_grad():
        for parameter in live.parameters():
            parameter[:] = parameter + lucid.randn(tuple(parameter.shape))
    fresh = _snapshot(live)
    starts = [_snapshot(average) for average, _ in averages]

    _update(model, case)

    for (average, keep), start in zip(averages, starts, strict=True):
        for name, value in average.named_parameters():
            expected = keep * start[name] + (1.0 - keep) * fresh[name]
            assert float((value - expected).abs().max().item()) < 1e-5, name


def test_a_tree_that_lost_a_parameter_is_refused_before_any_write() -> None:
    live = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2))
    average = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2, bias=False))
    before = _snapshot(average)
    with pytest.raises(RuntimeError, match=r"missing from the average: \['1.bias'\]"):
        update_averages(live, [(average, 0.5)])
    for name, value in average.named_parameters():
        assert float((value - before[name]).abs().max().item()) == 0.0


def test_a_namesake_of_another_shape_is_refused() -> None:
    live = nn.Sequential(nn.Linear(2, 3))
    average = nn.Sequential(nn.Linear(2, 4))
    with pytest.raises(RuntimeError, match=r"'0.weight' has shape \(4, 2\)"):
        update_averages(live, [(average, 0.5)])
