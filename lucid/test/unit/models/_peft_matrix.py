"""The zoo-wide LoRA coverage matrix: one declarative row per ``model_type``.

``test_peft_zoo.py`` proves the deep behaviours on a handful of families.
This module is the wide table behind its ``test_the_matrix_*`` tests: every
model type the zoo registers has a :class:`ZooRow` — a real factory, a tiny
config override (H11), the inputs of one forward — and a :class:`Trial`
runs the same steps on it:

1. ``get_peft_model`` with a :class:`LoraConfig` whose ``target_modules`` is
   ``None``, so the model type's default applies;
2. a forward, which every later step compares against;
3. a backward that reaches every adapter parameter and no other;
4. ``save_pretrained`` then ``PeftModel.from_pretrained`` onto a fresh base,
   giving the same outputs;
5. ``merge_and_unload``, giving the same outputs from the plain model.

A family LoRA cannot adapt yet carries a :class:`Gap` instead of a
workaround: the row runs as a strict ``xfail`` whose reason names the exact
missing piece, so the table is the progress meter and a fix shows up as an
unexpected pass.

The rows test one factory per type at a tiny size.  :func:`default_problems`
is the other half: it builds *every* registered factory without allocating
and checks that the type's default names reach layers LoRA can adapt and
merge, so a default that holds for the tiny factory but not for a sibling
fails too.
"""

import dataclasses
import tempfile
from collections.abc import Callable, Iterator, Mapping
from types import MappingProxyType
from typing import Any, Final, Self

import pytest

import lucid
import lucid.models as models
import lucid.nn as nn
from lucid.models._registry import _REGISTRY
from lucid.nn._shadow import shadow_alloc
from lucid.peft import LoraConfig, PeftModel, get_peft_model
from lucid.peft._target_defaults import (
    DEFAULT_LORA_TARGET_MODULES,
    DEFAULT_LORA_TARGET_PARAMETERS,
)
from lucid.peft._targeting import _suffix_match

type Call = tuple[tuple[Any, ...], dict[str, Any]]

_SEED: Final = 7
_RANK: Final = 4
_ALPHA: Final = 8

#: ``merged`` against ``adapted``: float32 merge arithmetic, scaled by the output's size.
_RTOL: Final = 1e-4
_ATOL: Final = 1e-4


def _wake_zero_initialised(model: nn.Module) -> None:
    """Give every all-zero weight matrix a value.

    The zoo zero-initialises on purpose (a ViT head, a DiT's adaLN-zero
    modulation, a flow's output layer), which leaves the output constant
    whatever the layers below it do — and an adapter under it, however
    wired, changes nothing a forward can show.
    """
    with lucid.no_grad():
        for parameter in model.parameters():
            if parameter.ndim >= 2 and not bool((parameter != 0).any().item()):
                nn.init.normal_(parameter, std=0.2)


@dataclasses.dataclass(frozen=True, slots=True)
class Gap:
    """What stops a family being adapted, in the words of the failing step.

    ``raises`` narrows the strict ``xfail`` to the failure named, so a row
    cannot hide a different break behind its reason.
    """

    reason: str
    raises: tuple[type[BaseException], ...]


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class ZooRow:
    """One zoo model type: how to build it tiny, feed it, and what it takes of an adapter."""

    factory: str
    overrides: Mapping[str, Any]
    inputs: Callable[[], Call]
    #: Module-name parts whose adapters get no gradient by design: an EMA target network, or a
    #: sub-network the row's forward does not call.
    unreached: tuple[str, ...] = ()
    gap: Gap | None = None
    #: Over 100M parameters at its paper size: left to ``-m heavy``.
    heavy: bool = False

    @property
    def model_type(self) -> str:
        return _REGISTRY[self.factory].model_type

    def build(self) -> nn.Module:
        lucid.manual_seed(0)
        model = models.create_model(self.factory, **self.overrides)
        _wake_zero_initialised(model)
        return model

    def marks(self) -> list[pytest.MarkDecorator]:
        found = [pytest.mark.heavy] if self.heavy else []
        if self.gap is not None:
            found.append(
                pytest.mark.xfail(strict=True, raises=self.gap.raises, reason=self.gap.reason)
            )
        return found

    @classmethod
    def from_case(cls, case: Any) -> Self:
        """A row from a ``test_models_train_step`` case: its factory, config and inputs."""
        # A ``pytest.param`` wraps the tuple and carries the marks.
        _family, factory, overrides, make_inputs = getattr(case, "values", case)
        heavy = any(mark.name == "heavy" for mark in getattr(case, "marks", ()))
        return cls(
            factory=factory,
            overrides=MappingProxyType(overrides),
            inputs=make_inputs,
            heavy=heavy,
        )


def collect(rows: list[ZooRow]) -> Mapping[str, ZooRow]:
    """The rows by ``model_type``, read-only; two rows for one type is an error."""
    table: dict[str, ZooRow] = {}
    for row in rows:
        if row.model_type in table:
            raise ValueError(f"two rows for model type {row.model_type!r}")
        table[row.model_type] = row
    return MappingProxyType(table)


def zoo_model_types() -> frozenset[str]:
    """Every ``model_type`` the registry holds a factory for."""
    return frozenset(entry.model_type for entry in _REGISTRY.values())


# ── outputs ──────────────────────────────────────────────────────────────────


def float_tensors(output: object) -> Iterator[lucid.Tensor]:
    """Every floating tensor a forward answered with, however deeply it is nested."""
    if isinstance(output, lucid.Tensor):
        if output.is_floating_point():
            yield output
    elif isinstance(output, Mapping):
        for value in output.values():
            yield from float_tensors(value)
    elif isinstance(output, (list, tuple)):
        for value in output:
            yield from float_tensors(value)
    elif dataclasses.is_dataclass(output) and not isinstance(output, type):
        for field in dataclasses.fields(output):
            yield from float_tensors(getattr(output, field.name, None))


def scalar(output: object) -> lucid.Tensor:
    """One scalar that depends on every floating output: the sum of their means."""
    total: lucid.Tensor | None = None
    for tensor in float_tensors(output):
        term = tensor.mean()
        total = term if total is None else total + term
    if total is None:
        raise AssertionError("the forward answered with no floating tensor")
    return total


def owned_by_adapter(name: str) -> bool:
    return any(
        part in name
        for part in (
            ".lora_A.",
            ".lora_B.",
            ".out_proj_lora_",  # an attention's output projection
            ".lora_embedding_",  # an embedding's factors are bare parameters
            ".modules_to_save.",
        )
    )


# ── the trial ────────────────────────────────────────────────────────────────


class Trial:
    """The steps on one row, in order; each states what it proves."""

    def __init__(self, row: ZooRow) -> None:
        self.row = row
        lucid.manual_seed(_SEED)
        self.args, self.kwargs = row.inputs()
        config = LoraConfig(r=_RANK, lora_alpha=_ALPHA, init_lora_weights=False)
        self.peft: PeftModel = get_peft_model(row.build(), config)
        self.peft.eval()

    def forward(self, model: nn.Module) -> object:
        """The row's forward, with the generator seeded so a stochastic layer repeats itself."""
        lucid.manual_seed(_SEED)
        return model(*self.args, **self.kwargs)

    def _outputs(self, model: nn.Module) -> list[lucid.Tensor]:
        with lucid.no_grad():
            return list(float_tensors(self.forward(model.eval())))

    @staticmethod
    def _agree(got: list[lucid.Tensor], want: list[lucid.Tensor], what: str) -> None:
        assert len(got) == len(want), f"{what}: {len(got)} outputs, expected {len(want)}"
        for index, (a, b) in enumerate(zip(got, want, strict=True)):
            assert a.shape == b.shape, f"{what}: output {index} is {a.shape}, expected {b.shape}"
            scale = float(b.abs().max().item())
            gap = float((a - b).abs().max().item())
            assert gap <= _ATOL + _RTOL * scale, f"{what}: output {index} is off by {gap}"

    def adapted(self) -> list[lucid.Tensor]:
        """Steps 1-2: something took an adapter, and the adapted forward is finite and moved."""
        tuner = self.peft.base_model
        assert tuner.targeted_module_names or tuner.targeted_parameter_names, "nothing adapted"
        want = self._outputs(self.row.build())
        got = self._outputs(self.peft)
        for tensor in got:
            assert bool(tensor.isfinite().all().item()), "the adapted forward is not finite"
        moved = any(float((a - b).abs().max().item()) > 0 for a, b in zip(got, want, strict=True))
        assert moved, "the adapters change no output: they are off the forward path"
        return got

    def backward_reaches_only_the_adapter(self) -> None:
        """Step 3: every trainable parameter is an adapter's, and each got a finite gradient."""
        named = list(self.peft.named_parameters())
        trainable = [n for n, p in named if p.requires_grad]
        assert trainable and all(owned_by_adapter(f".{n}.") for n in trainable), trainable[:4]
        scalar(self.forward(self.peft)).backward()
        unreached = self.row.unreached
        missing = [n for n, p in named if p.requires_grad and p.grad is None]
        stray = [n for n in missing if not any(part in n for part in unreached)]
        assert not stray, f"{len(stray)} adapter parameters got no gradient: {stray[:4]}"
        loose = [n for n, p in named if not p.requires_grad and p.grad is not None]
        assert not loose, f"base parameters got a gradient: {loose[:4]}"
        broken = [
            n for n, p in named if p.grad is not None and not bool(p.grad.isfinite().all().item())
        ]
        assert not broken, f"non-finite gradients in {broken[:4]}"

    def reload_agrees(self, want: list[lucid.Tensor]) -> None:
        """Step 4: the adapter file loads onto a fresh base and gives the same outputs."""
        with tempfile.TemporaryDirectory() as directory:
            self.peft.save_pretrained(directory)
            fresh = PeftModel.from_pretrained(self.row.build(), directory)
        self._agree(self._outputs(fresh), want, "the reloaded adapter")

    def merge_agrees(self, want: list[lucid.Tensor]) -> None:
        """Step 5: the merged plain model gives the adapted outputs."""
        merged = self.peft.merge_and_unload()
        assert not any(".lora_" in n for n, _ in merged.named_modules()), "an adapter is left"
        self._agree(self._outputs(merged), want, "the merged model")

    def run(self) -> None:
        want = self.adapted()
        self.backward_reaches_only_the_adapter()
        self.reload_agrees(want)
        self.merge_agrees(want)


# ── the defaults, against every factory ──────────────────────────────────────

_ADAPTABLE: Final = (
    nn.Linear,
    nn.Conv1d,
    nn.Conv2d,
    nn.Conv3d,
    nn.Embedding,
    nn.MultiheadAttention,
)


def _refusal(module: nn.Module) -> str | None:
    """Why LoRA cannot adapt and merge ``module``, or ``None`` if it can."""
    if not isinstance(module, _ADAPTABLE):
        return type(module).__name__
    if getattr(module, "groups", 1) > 1:
        return f"{type(module).__name__} with groups > 1 cannot be merged"
    if isinstance(module, nn.MultiheadAttention) and not module._qkv_same_embed_dim:
        return "MultiheadAttention with kdim or vdim apart from embed_dim"
    return None


def default_problems(factory: str) -> list[str]:
    """What is wrong with the model type's default targets on ``factory``, without allocating it.

    The factory is built under ``shadow_alloc`` at its paper size.  A default
    is sound when it reaches at least one module (or names raw parameters) and
    every module it reaches is one LoRA adapts and can merge.
    """
    entry = _REGISTRY[factory]
    names = DEFAULT_LORA_TARGET_MODULES.get(entry.model_type, frozenset())
    with shadow_alloc():
        model = entry.factory()
    reached = [(n, m) for n, m in model.named_modules() if n and _suffix_match(n, names)]
    problems = [f"{n}: {why}" for n, m in reached if (why := _refusal(m)) is not None]
    if not reached and entry.model_type not in DEFAULT_LORA_TARGET_PARAMETERS:
        problems.append(f"the default {sorted(names)} reaches no module")
    return problems
