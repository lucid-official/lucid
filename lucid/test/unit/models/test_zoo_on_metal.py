"""The zoo, written for Metal: every (family, task) pair trains and answers there (LCD-852).

Models are trained on Metal.  The rest of this directory follows the
process's default device — the CPU, unless ``lucid-audit --models`` says
otherwise — so on its own it leaves Metal untested.  These tests are
written for Metal and need no environment variable: they run wherever
Metal is, the nightly zoo job included, and skip where it is not.

Every (family, task) pair of the compiled-training matrix (``_zoo_matrix``:
``TASKS``, ``families`` and the ``representative`` factory, shrunk as
``test_zoo_dtype_follows_input`` shrinks it) reaches Metal both ways a
caller puts it there:

* **built** under ``lucid.set_default_device("metal")`` and never moved, so
  a table a constructor leaves on the CPU fails here;
* **moved**: built with the CPU as the default and sent with
  ``.to("metal")``, the case running on with the CPU as the default, so a
  tensor ``Module.to`` does not reach — or one a forward makes without its
  input's device — fails here.

Either way the model must hold every parameter and buffer on Metal, answer
an eval forward on Metal with every floating tensor in float32, and take a
training step: a finite float32 loss on Metal, every gradient on Metal, an
optimizer step that moves the weights and leaves them, and every buffer,
on Metal.  Image generators sample, language models generate and world
models plan or imagine, the same two ways.

The three families whose task forward is a rollout — Stable Diffusion,
DIAMOND and Genie, which the matrix skips — train through their own losses
and answer through that rollout, at the sizes the dtype sweep drives them.
"""

import dataclasses
from collections.abc import Callable, Iterator, Mapping
from typing import Any

import pytest

import lucid
import lucid.models as models
import lucid.nn as nn
import lucid.optim as optim
from lucid.models._registry import _REGISTRY
from lucid.test._default_device import default_device, metal_available
from lucid.test.unit.compile import _zoo_matrix as zoo
from lucid.test.unit.models.test_zoo_dtype_follows_input import (
    _DIAMOND,
    _GENIE,
    _SD,
    _generate,
    _history,
    _like,
    _matrix_case,
    _sampler_case,
    _tensors,
)

pytestmark = pytest.mark.skipif(not metal_available(), reason="needs Metal")

_METAL = "metal"
_F32 = lucid.float32


# -- the two ways onto Metal ------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class _Way:
    """How a model reaches Metal, and the default device the case runs under."""

    default: str
    moved: bool

    def build(self, factory: str, overrides: Mapping[str, Any]) -> nn.Module:
        lucid.manual_seed(0)
        model = models.create_model(factory, **overrides)
        return model.to(_METAL) if self.moved else model


WAYS: dict[str, _Way] = {
    "built-on-metal": _Way(default="metal", moved=False),
    "moved-from-the-cpu": _Way(default="cpu", moved=True),
}


@pytest.fixture(params=list(WAYS))
def way(request: pytest.FixtureRequest) -> Iterator[_Way]:
    chosen = WAYS[str(request.param)]
    with default_device(chosen.default):
        yield chosen


def _off_metal(model: nn.Module) -> list[str]:
    held = [*model.named_parameters(), *model.named_buffers()]
    return [name for name, tensor in held if tensor.device.type != _METAL]


def _answered_on_metal(answer: Any, what: str) -> None:
    """Every tensor ``answer`` holds is on Metal, and every floating one float32."""
    held = _tensors(answer)
    assert held, f"{what} answered with no tensor"
    off = [f"{n}: {t.device.type}" for n, t in held if t.device.type != _METAL]
    assert not off, f"{what} answered off Metal: {off[:6]}"
    wide = [f"{n}: {t.dtype}" for n, t in held if t.is_floating_point() and t.dtype != _F32]
    assert not wide, f"{what} answered outside float32: {wide[:6]}"


# -- what each pair is driven with -------------------------------------------------

Step = Callable[[nn.Module], tuple[Any, lucid.Tensor]]


@dataclasses.dataclass(frozen=True)
class _Drive:
    """One pair: the factory, its overrides, and how it is run on Metal."""

    factory: str
    overrides: Mapping[str, Any]
    #: An eval forward on inputs drawn on Metal.
    answer: Callable[[nn.Module], Any]
    #: A training forward: its answer and the scalar it is trained on.
    step: Step


def _matrix_drive(family: str, task: str, monkeypatch: pytest.MonkeyPatch) -> _Drive:
    """The matrix's inputs (drawn on its device, Metal) and its objective."""
    name, overrides, cfg = _matrix_case(family, task, monkeypatch)

    def answer(model: nn.Module) -> Any:
        args, kwargs = zoo.default_inputs(model, cfg, family, 1)
        return model(*args, **kwargs)

    def step(model: nn.Module) -> tuple[Any, lucid.Tensor]:
        out = answer(model)
        return out, zoo.objective(out)

    return _Drive(name, overrides, answer, step)


def _sd_inputs() -> tuple[lucid.Tensor, lucid.Tensor]:
    lucid.manual_seed(1)
    return _like(_METAL, _F32, 1, 3, 32, 32), _like(_METAL, _F32, 1, 4, 16)


def _sd_step(model: nn.Module) -> tuple[Any, lucid.Tensor]:
    out = model(*_sd_inputs())
    return out, out.loss


def _diamond_answer(model: nn.Module) -> Any:
    lucid.manual_seed(1)
    return model(*_history(model.diamond, _METAL, _F32), horizon=2)


def _diamond_step(model: nn.Module) -> tuple[Any, lucid.Tensor]:
    """The task forward trains the actor-critic in imagination."""
    out = _diamond_answer(model)
    return out, out.policy_loss + out.value_loss


def _genie_answer(model: nn.Module) -> Any:
    """The task forward: play two frames from a one-frame prompt."""
    lucid.manual_seed(1)
    prompt = lucid.rand(1, 1, 3, 8, 8).to(_METAL)
    return model(prompt, lucid.zeros(1, 2, dtype=lucid.int64).to(_METAL))


def _genie_step(model: nn.Module) -> tuple[Any, lucid.Tensor]:
    """One backward trains the tokenizer, the action model and the dynamics."""
    lucid.manual_seed(1)
    out = model.genie(lucid.rand(2, 4, 3, 8, 8).to(_METAL))
    return out, out.loss


#: The families whose task forward is a rollout the matrix skips.
_ROLLOUTS: dict[str, _Drive] = {
    "stable_diffusion": _Drive("stable_diffusion_gen", _SD, lambda m: m(*_sd_inputs()), _sd_step),
    "diamond": _Drive("diamond_world_model", _DIAMOND, _diamond_answer, _diamond_step),
    "genie": _Drive(
        "genie_world_model", {**_GENIE, "maskgit_steps": 2}, _genie_answer, _genie_step
    ),
}


def _drive(family: str, task: str, monkeypatch: pytest.MonkeyPatch) -> _Drive:
    return _ROLLOUTS.get(family) or _matrix_drive(family, task, monkeypatch)


#: Past 100M parameters at their smallest, the ``heavy`` marker's line: no
#: config field narrows VGG-11's classifier (129M) or Fast R-CNN's VGG-16 (134M).
_HEAVY = frozenset({"vgg", "fast_rcnn"})


def _pair(family: str, task: str) -> Any:
    marks = [pytest.mark.heavy] if family in _HEAVY else []
    return pytest.param(family, task, id=f"{family}-{task}", marks=marks)


_PAIRS = [(family, task) for task in zoo.TASKS for family in zoo.families(task)]


def test_every_pair_of_the_registry_is_driven() -> None:
    """The matrix's tasks cover every task but backbones, and every family."""
    registered = {(e.family, e.task) for e in _REGISTRY.values() if e.task != "base"}
    assert set(_PAIRS) == registered, sorted(set(_PAIRS) ^ registered)
    assert {f for f, _ in _PAIRS} == {e.family for e in _REGISTRY.values()}
    assert set(_ROLLOUTS) == {f for f, spec in zoo.SPECS.items() if spec.skip is not None}


# -- train and answer ---------------------------------------------------------------


@pytest.mark.parametrize(("family", "task"), [_pair(f, t) for f, t in _PAIRS])
def test_a_pair_answers_and_trains_a_step_on_metal(
    family: str, task: str, way: _Way, monkeypatch: pytest.MonkeyPatch
) -> None:
    drive = _drive(family, task, monkeypatch)
    model = way.build(drive.factory, drive.overrides)
    assert not _off_metal(model), f"built off Metal: {_off_metal(model)[:6]}"

    model.eval()
    with lucid.no_grad():
        _answered_on_metal(drive.answer(model), "an eval forward")

    model.train()
    trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    out, loss = drive.step(model)
    _answered_on_metal(out, "a training forward")
    assert loss.ndim == 0, f"the loss has shape {tuple(loss.shape)}"
    assert (loss.device.type, loss.dtype) == (_METAL, _F32), (loss.device, loss.dtype)
    assert bool(loss.isfinite().item()), f"the loss is {float(loss.item())}"
    loss.backward()

    grads = [(n, p.grad) for n, p in trainable if p.grad is not None]
    assert grads, "no parameter received a gradient"
    off = [n for n, g in grads if g.device.type != _METAL]
    assert not off, f"gradients off Metal: {off[:6]}"
    # The step must move the weight it pushes hardest.  One copy, not the
    # model's: a second copy of every weight is what fills a hosted runner.
    steep = max(grads, key=lambda named: float(named[1].abs().max().item()))[0]
    held = dict(trainable)[steep]
    before = held.detach().clone()
    optim.SGD([p for _, p in trainable], lr=1e-2).step()
    assert float((held.detach() - before).abs().max().item()) > 0, f"the step left {steep}"
    assert not _off_metal(model), f"off Metal after the step: {_off_metal(model)[:6]}"


# -- sample, generate, plan, imagine ----------------------------------------------


Sample = Callable[[nn.Module], Any]


def _image_sampler(family: str) -> tuple[str, dict[str, Any], Sample]:
    name, overrides = _sampler_case(family)
    return name, overrides, lambda m: _generate(family, m, _METAL, _F32)


def _text_sampler(family: str, monkeypatch: pytest.MonkeyPatch) -> tuple[str, Any, Sample]:
    name, overrides, cfg = _matrix_case(family, "language-modeling", monkeypatch)

    def sample(model: nn.Module) -> Any:
        lucid.manual_seed(1)
        prompt = lucid.randint(0, int(cfg.vocab_size), (1, 4)).to(_METAL)
        return model.generate(prompt, max_length=6)

    return name, overrides, sample


def _imagination(family: str, monkeypatch: pytest.MonkeyPatch) -> tuple[str, Any, Sample]:
    """A Dreamer's policy, rolled out in its own latent space."""
    name, overrides, _cfg = _matrix_case(family, "world-modeling", monkeypatch)

    def sample(model: nn.Module) -> Any:
        inner = getattr(model, type(model).base_model_prefix)
        return inner.imagine(inner.rssm.initial(2), 3)

    return name, overrides, sample


def _planet_plan(_family: str, monkeypatch: pytest.MonkeyPatch) -> tuple[str, Any, Sample]:
    name, overrides, _cfg = _matrix_case("planet", "world-modeling", monkeypatch)

    def sample(model: nn.Module) -> Any:
        start = model.planet.rssm.initial(1)
        return model.plan(start, horizon=3, iterations=2, candidates=8, elites=2)

    return name, overrides, sample


def _diamond_dream(_family: str, _monkeypatch: pytest.MonkeyPatch) -> tuple[str, Any, Sample]:
    def sample(model: nn.Module) -> Any:
        lucid.manual_seed(1)
        return model.diamond.imagine_frame(*_history(model.diamond, _METAL, _F32), steps=2)

    return "diamond_world_model", _DIAMOND, sample


_TEXT_GENERATORS = ("gpt", "gpt2", "t5", "transformer")

#: (family, task) -> what that pair samples with, beside its forward.
_SAMPLERS: dict[tuple[str, str], Callable[[str, pytest.MonkeyPatch], tuple[str, Any, Sample]]] = {
    **{
        (f, "image-generation"): lambda fam, _mp: _image_sampler(fam)
        for f in zoo.families("image-generation")
    },
    **{(f, "language-modeling"): _text_sampler for f in _TEXT_GENERATORS},
    **{(f, "world-modeling"): _imagination for f in ("dreamer", "dreamer_v2", "dreamer_v3")},
    ("planet", "world-modeling"): _planet_plan,
    ("diamond", "world-modeling"): _diamond_dream,
}


@pytest.mark.parametrize(("family", "task"), [_pair(f, t) for f, t in _SAMPLERS])
def test_a_pair_samples_on_metal(
    family: str, task: str, way: _Way, monkeypatch: pytest.MonkeyPatch
) -> None:
    factory, overrides, sample = _SAMPLERS[(family, task)](family, monkeypatch)
    model = way.build(factory, overrides).eval()
    assert not _off_metal(model), f"built off Metal: {_off_metal(model)[:6]}"
    with lucid.no_grad():
        answer = sample(model)
    _answered_on_metal(answer, "sampling")
    for name, tensor in _tensors(answer):
        if tensor.is_floating_point():
            assert bool(tensor.isfinite().all().item()), f"sampling: {name} is not finite"
