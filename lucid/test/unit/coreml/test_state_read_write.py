"""A Core ML package's state can be seeded, read back and restored.

A stateful package kept its carried values out of reach: they could only be
reset, never seeded or read.  A stream whose first step differs from the
rest — a video decoder that skips its temporal upsampling on frame 0 — could
not start from a prepared state, and nothing a stream had accumulated could
be saved or inspected (reported while porting Self-Forcing).
"""

import pytest

import lucid
import lucid.coreml as cml
import lucid.nn as nn


class _Accumulates(nn.Module):
    def forward(self, x: lucid.Tensor, cache: lucid.Tensor) -> tuple[lucid.Tensor, lucid.Tensor]:
        cache.add_(x)
        return cache, cache * 2.0


@pytest.fixture
def stateful(tmp_path: object) -> cml.CoreMLModel:  # type: ignore[name-defined]
    exported = cml.export(
        _Accumulates().eval(),
        {"x": lucid.ones(1, 4), "cache": lucid.zeros(1, 4)},
        f"{tmp_path}/m.mlpackage",
        precision=cml.Precision.FLOAT16,
        state=[cml.State(input="cache", output="output_0")],
    )
    yield exported
    exported.close()


def test_a_state_is_seeded_read_and_restored(stateful) -> None:  # type: ignore[no-untyped-def]
    (name,) = stateful.state_names
    assert stateful.read_state(name).to(lucid.float32).tolist() == [[0.0] * 4]

    stateful.write_state(name, lucid.tensor([[10.0, 20.0, 30.0, 40.0]]))
    out = stateful.predict(lucid.ones(1, 4))
    assert out.to(lucid.float32).tolist() == [[22.0, 42.0, 62.0, 82.0]]
    saved = stateful.read_state(name)
    assert saved.to(lucid.float32).tolist() == [[11.0, 21.0, 31.0, 41.0]]

    stateful.predict(lucid.ones(1, 4))
    stateful.write_state(name, saved)  # back to the saved point
    assert stateful.read_state(name).to(lucid.float32).tolist() == [[11.0, 21.0, 31.0, 41.0]]


def test_a_value_at_an_offset_is_read_from_its_first_element(stateful) -> None:  # type: ignore[no-untyped-def]
    # The handle took the element-0 buffer a CPU view hands its readers and
    # advanced it by the view's offset into its own allocation as well: the
    # write read two elements past the view and past the end of its copy
    # (LCD-517).  The public write_state hands it a copy today, so the handle
    # is called directly with the view; once ``contiguous()`` returns a
    # contiguous view itself, the public path hands it the view as well.
    (name,) = stateful.state_names
    view = lucid.arange(4 + 2.0, dtype=lucid.float16).as_strided((1, 4), (4, 1), 2)
    assert view.is_contiguous() and view.storage_offset() != 0
    stateful._handle.write_state(name, view._impl)
    assert stateful.read_state(name).tolist() == view.tolist() == [[2.0, 3.0, 4.0, 5.0]]

    stateful.write_state(name, lucid.zeros(1, 4, dtype=lucid.float16))
    stateful.write_state(name, view)
    assert stateful.read_state(name).tolist() == [[2.0, 3.0, 4.0, 5.0]]


def test_a_wrong_name_or_shape_is_refused(stateful) -> None:  # type: ignore[no-untyped-def]
    (name,) = stateful.state_names
    with pytest.raises(ValueError):
        stateful.read_state("no_such_state")
    with pytest.raises(ValueError, match="shape"):
        stateful.write_state(name, lucid.zeros(2, 4))
