"""A zoo model writes its buffers through the module, so a compiled step keeps them.

RealNVP's flow batch norm folded each batch into its running statistics by
writing ``self._buffers[...]`` directly (LCD-600).  That skipped
``Module.__setattr__``, which under a compile trace writes a buffer in
place so the compiled call carries the update back
(``Module._write_traced_buffer``).  A compiled RealNVP therefore read the
statistics it was traced with on every later step: four steps in, its
running statistics were 0.16 away from eager's.

Two guards.  No module in the zoo writes a registry (``_buffers``,
``_parameters``, ``_modules``) itself: every write goes through the
``Module`` API, which is what the tracer, the compiled cache and a
checkpoint all see.  And a compiled RealNVP training step, in both of its
batch-norm modes, ends with the running statistics eager ends with.
"""

import ast
from collections.abc import Callable
from pathlib import Path

import pytest

import lucid
import lucid.models as models
from lucid.compile import fused_step, make_step

_REGISTRIES = frozenset({"_buffers", "_parameters", "_modules", "_non_persistent_buffers"})
_MUTATORS = frozenset(
    {
        "__setitem__",
        "__delitem__",
        "clear",
        "move_to_end",
        "pop",
        "popitem",
        "setdefault",
        "update",
    }
)


def _registry_writes(tree: ast.AST) -> list[int]:
    """Lines that store into, delete from, or mutate a module registry."""
    lines = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.Subscript, ast.Attribute)) and isinstance(
            node.ctx, (ast.Store, ast.Del)
        ):
            target = node.value if isinstance(node, ast.Subscript) else node
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr not in _MUTATORS:
                continue
            target = node.func.value
        else:
            continue
        if isinstance(target, ast.Attribute) and target.attr in _REGISTRIES:
            lines.append(node.lineno)
    return lines


def test_no_zoo_module_writes_a_module_registry_itself() -> None:
    root = Path(models.__file__).parent
    found = []
    for path in sorted(root.rglob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        found += [f"{path.relative_to(root)}:{line}" for line in _registry_writes(tree)]
    assert not found, (
        "these write a module registry directly, past Module.__setattr__ — "
        f"a compiled step will not see the write: {found}"
    )


def test_the_registry_scan_sees_each_kind_of_write() -> None:
    source = (
        "self._buffers['a'] = x\n"
        "del self._parameters['b']\n"
        "self._modules.update(c=m)\n"
        "self._buffers = {}\n"
        "y = self._buffers['a']\n"
        "self._buffers.get('a')\n"
    )
    assert _registry_writes(ast.parse(source)) == [1, 2, 3, 4]


_REALNVP = {"sample_size": 8, "num_scales": 2, "residual_blocks": 1, "base_dim": 8}


def _negative_log_likelihood(out: models.NormalizingFlowOutput) -> lucid.Tensor:
    return -out.log_prob.mean()


def _running(model: lucid.nn.Module) -> dict[str, lucid.Tensor]:
    return {
        name: buffer.detach().clone() for name, buffer in model.named_buffers() if "running" in name
    }


#: A compile entry's step, and a check that it compiled what it should have.
_Entry = tuple[Callable[[lucid.Tensor], lucid.Tensor], Callable[[], bool]]


def _make(model: lucid.nn.Module) -> _Entry:
    step = make_step(model, _negative_log_likelihood)
    return step, lambda: not step.eager_only


def _fused(model: lucid.nn.Module) -> _Entry:
    # A flow batch norm writes its buffers in place, which fused_step cannot
    # carry back, so it runs the step eagerly — correct, and not compiled.
    optimizer = lucid.optim.SGD(model.parameters(), lr=0.0)
    return fused_step(model, _negative_log_likelihood, optimizer), lambda: True


#: The compile entries a training loop would reach for, and whether each is
#: expected to have compiled the step rather than run it eagerly.
_ENTRIES = [
    pytest.param(_make, id="make_step"),
    pytest.param(_fused, id="fused_step"),
]


@pytest.mark.filterwarnings("ignore::lucid.compile._core.fallback.CompileFallbackWarning")
@pytest.mark.parametrize("entry", _ENTRIES)
@pytest.mark.parametrize("running_in_training", [False, True], ids=["batch", "running"])
def test_a_compiled_realnvp_step_keeps_the_running_statistics(
    device_gpu_only: str,
    running_in_training: bool,
    entry: Callable[[lucid.nn.Module], _Entry],
) -> None:
    def build() -> lucid.nn.Module:
        return (
            models.create_model(
                "realnvp_cifar",
                batch_norm_running_stats_in_training=running_in_training,
                **_REALNVP,
            )
            .to(device_gpu_only)
            .train()
        )

    lucid.manual_seed(0)
    compiled = build()
    lucid.manual_seed(0)
    eager = build()
    step, compiled_it = entry(compiled)
    before = _running(eager)

    for index in range(3):
        x = lucid.rand(2, 3, 8, 8, device=device_gpu_only)
        lucid.manual_seed(10 + index)
        got = float(step(x).item())
        lucid.manual_seed(10 + index)
        want = float(_negative_log_likelihood(eager(x)).item())
        assert got == pytest.approx(want, rel=1e-4, abs=1e-4), f"step {index}"

    assert compiled_it(), "the step ran eagerly, so nothing compiled was checked"
    ran, reference = _running(compiled), _running(eager)
    assert ran.keys() == reference.keys() and ran
    for name, value in reference.items():
        assert float((value - before[name]).abs().max().item()) > 0.0, name
        drift = float((ran[name] - value).abs().max().item())
        assert drift <= 1e-5, f"{name}: compiled is {drift} away from eager"
