"""Flexible exports that used to be refused, and a constant output (LCD-365).

* **A sequence range.** A text model adds a position table sliced to the
  input's length, ``table[:T]``; the trace records it as a ``split_at``
  whose cut point is ``T``, and the export refused it ("indices changed
  with the input size"). The sections are now sliced at run time.
* **A plane range under a resample.** ``Upsample(size=)`` keeps its output
  size whatever the input; ``Upsample(scale_factor=)`` keeps its ratio.
  Both were refused because the output size is an attribute of the trace.
  The first is now a resize to the size, the second an upsample by the
  ratio every traced shape agrees on.
* **A constant output.** ``return x * 2, lucid.zeros(2, 3)`` failed saying
  nothing in the graph produced the zeros.

Each exported case is run at shapes across the range and compared with
eager on ``CPU_ONLY``.
"""

from collections.abc import Callable

import pytest

import lucid
import lucid.coreml as cml
import lucid.models as M
import lucid.nn as nn
from lucid._C import engine as _C_engine

pytestmark = pytest.mark.skipif(
    not hasattr(_C_engine, "coreml"), reason="the engine was built without the Core ML writer"
)


def _first(value: object) -> lucid.Tensor:
    """The leading tensor of a model output or a package's answer."""
    if isinstance(value, dict):
        return next(iter(value.values()))
    for field in ("logits", "last_hidden_state"):
        if hasattr(value, field):
            return getattr(value, field)
    assert isinstance(value, lucid.Tensor)
    return value


def _agrees(got: lucid.Tensor, want: lucid.Tensor) -> None:
    assert tuple(got.shape) == tuple(want.shape)
    scale = max(float(want.abs().max().item()), 1.0)
    assert float((got - want).abs().max().item()) / scale < 1e-4


def _export(
    model: nn.Module, example: lucid.Tensor, path: str, ranges: dict[int, tuple[int, int]]
) -> cml.CoreMLModel:
    return cml.export(
        model, example, path, shape_range=ranges, compute_units=cml.ComputeUnits.CPU_ONLY
    )


# ── a sequence range over a sliced position table ────────────────────

_TEXT = {
    "vocab_size": 100,
    "hidden_size": 32,
    "num_hidden_layers": 1,
    "num_attention_heads": 2,
    "intermediate_size": 64,
    "max_position_embeddings": 256,
}
_LENGTHS = (4, 7, 16, 50, 128)


@pytest.mark.parametrize(
    "name", ["gpt2_small_lm", "gpt_lm", "bert_tiny", "roformer", "transformer_base_cls"]
)
def test_a_text_model_takes_a_sequence_range(name: str, tmp_path: object) -> None:
    lucid.manual_seed(0)
    model = M.create_model(name, **_TEXT).eval()
    exported = _export(
        model, lucid.randint(0, 100, (1, 16)), str(tmp_path / "text.mlpackage"), {1: (4, 128)}
    )
    try:
        for length in _LENGTHS:
            ids = lucid.randint(0, 100, (1, length))
            _agrees(_first(exported.predict(ids)), _first(model(ids)))
    finally:
        exported.close()


class _Table(nn.Module):
    """``table[:T]`` and ``table[T:]`` both used: a cut at the input's length."""

    def __init__(self) -> None:
        super().__init__()
        self.table = nn.Parameter(lucid.randn(40, 8))

    def forward(self, x: lucid.Tensor) -> lucid.Tensor:
        head = self.table[: x.shape[1]]
        tail = self.table[x.shape[1] :]
        return x + head + tail.sum(dim=0)


def test_both_sides_of_a_cut_follow_the_input(tmp_path: object) -> None:
    lucid.manual_seed(0)
    model = _Table().eval()
    exported = _export(model, lucid.randn(1, 8, 8), str(tmp_path / "t.mlpackage"), {1: (2, 30)})
    try:
        for length in (2, 3, 8, 17, 30):
            x = lucid.randn(1, length, 8)
            _agrees(exported.predict(x), model(x))
    finally:
        exported.close()


class _UnknownCut(nn.Module):
    """A cut at half the input's length: no input axis has it."""

    def __init__(self) -> None:
        super().__init__()
        self.table = nn.Parameter(lucid.randn(40, 8))

    def forward(self, x: lucid.Tensor) -> lucid.Tensor:
        half = x.shape[1] // 2
        return x[:, :half] + self.table[:half] + self.table[half:].sum(dim=0)


def test_a_cut_no_input_axis_explains_is_refused(tmp_path: object) -> None:
    with pytest.raises(cml.ShapeNotFlexible):
        _export(
            _UnknownCut().eval(), lucid.randn(1, 8, 8), str(tmp_path / "u.mlpackage"), {1: (4, 16)}
        )


# ── a plane range under a resample ───────────────────────────────────

_PLANES = [(4, 4), (2, 2), (16, 16), (3, 7), (12, 5)]

RESAMPLES: dict[str, Callable[[], nn.Module]] = {
    "nearest-size": lambda: nn.Upsample(size=(8, 8)),
    "bilinear-size": lambda: nn.Upsample(size=(8, 6), mode="bilinear"),
    "bilinear-size-aligned": lambda: nn.Upsample(size=(8, 8), mode="bilinear", align_corners=True),
    "nearest-scale": lambda: nn.Upsample(scale_factor=2),
    "bilinear-scale": lambda: nn.Upsample(scale_factor=2, mode="bilinear"),
}


@pytest.mark.parametrize("case", sorted(RESAMPLES))
def test_a_resample_takes_a_plane_range(case: str, tmp_path: object) -> None:
    lucid.manual_seed(0)
    model = RESAMPLES[case]().eval()
    exported = _export(
        model, lucid.randn(1, 3, 4, 4), str(tmp_path / "up.mlpackage"), {2: (2, 16), 3: (2, 16)}
    )
    try:
        for h, w in _PLANES:
            x = lucid.randn(1, 3, h, w)
            _agrees(exported.predict(x), model(x))
    finally:
        exported.close()


# ── a constant output ────────────────────────────────────────────────


class _WithZeros(nn.Module):
    def forward(self, x: lucid.Tensor) -> tuple[lucid.Tensor, lucid.Tensor]:
        return x * 2, lucid.zeros(2, 3)


def test_a_constant_output_is_exported(tmp_path: object) -> None:
    model = _WithZeros().eval()
    exported = cml.export(
        model,
        lucid.randn(2, 3),
        str(tmp_path / "zeros.mlpackage"),
        compute_units=cml.ComputeUnits.CPU_ONLY,
    )
    try:
        x = lucid.randn(2, 3)
        got = exported.predict(x)
        assert isinstance(got, dict)
        doubled, zeros = (got[name] for name in sorted(got))
        _agrees(doubled, model(x)[0])
        _agrees(zeros, lucid.zeros(2, 3))
    finally:
        exported.close()


class _WithArange(nn.Module):
    def forward(self, x: lucid.Tensor) -> tuple[lucid.Tensor, lucid.Tensor]:
        return x * 2, lucid.arange(4)


def test_an_untraced_constant_output_is_named(tmp_path: object) -> None:
    """``arange`` reaches the trace as a buffer, not an operation, so it is
    refused — by a message that says it may be a constant."""
    with pytest.raises(ValueError, match="constant"):
        cml.export(_WithArange().eval(), lucid.randn(2, 3), str(tmp_path / "a.mlpackage"))
