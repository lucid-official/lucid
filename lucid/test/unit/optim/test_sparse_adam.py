"""SparseAdam updates only the rows a gradient touches, and checkpoints them (LCD-416).

SparseAdam used to run dense Adam: a row absent from a batch kept moving on
its old momentum, so an embedding table drifted wherever it was not looked
up, and its moments were not in ``state_dict()`` at all — a resumed run
restarted them from zero.  The reference's SparseAdam is lazy: per touched
row it updates the moments and the value, with one step count per
parameter.  These pin that rule:

* against the reference over batches whose rows are disjoint, overlap and
  repeat, comparing values, moments and step counts after every step;
* rows outside a batch keep their value and moments exactly;
* a checkpoint resumes bit-identically, through ``state_dict``,
  ``copy.deepcopy`` and ``lucid.save`` / ``lucid.load``;
* a malformed checkpoint is refused and leaves the live state as it was.
"""

import copy
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

import lucid
import lucid.nn as nn
import lucid.optim as optim
from lucid._C import engine as _C_engine

_ROWS, _DIM = 6, 3
_W0 = (np.arange(_ROWS * _DIM, dtype=np.float32).reshape(_ROWS, _DIM) - 7.0) / 5.0

# Each case is a run of batches of embedding ids.
_BATCHES: dict[str, list[list[int]]] = {
    "disjoint": [[0, 1], [2, 3], [4], [5, 0]],
    "overlapping": [[0, 1, 2], [1, 2, 3], [2, 5], [0, 2]],
    "repeated": [[1, 1, 1], [1], [3, 3, 0], [1, 3]],
}
_LR = 0.1


def _loss(emb: Any, ids: Any) -> Any:
    # Not linear in the rows, so each step's gradient depends on the values.
    return ((emb(ids) - 0.5) ** 2).sum()


def _lucid_run(
    device: str, batches: list[list[int]], *, maximize: bool = False
) -> tuple[list[dict[str, np.ndarray]], optim.SparseAdam]:
    emb = nn.Embedding(_ROWS, _DIM)
    emb.weight = nn.Parameter(lucid.tensor(_W0.copy()))
    emb.to(device)
    opt = optim.SparseAdam(emb.parameters(), lr=_LR, maximize=maximize)
    out = []
    for ids in batches:
        opt.zero_grad()
        _loss(emb, lucid.tensor(ids).to(device)).backward()
        opt.step()
        st = opt.state[0]
        out.append(
            {
                "weight": emb.weight.numpy().copy(),
                "exp_avg": _as_array(st["exp_avg"]),
                "exp_avg_sq": _as_array(st["exp_avg_sq"]),
                "step": np.asarray(st["step"]),
            }
        )
    return out, opt


def _as_array(v: object) -> np.ndarray:
    numpy = getattr(v, "numpy", None)
    return np.array(numpy() if callable(numpy) else v, copy=True)


@pytest.mark.parity
@pytest.mark.parametrize("maximize", [False, True])
@pytest.mark.parametrize("case", sorted(_BATCHES))
def test_rows_follow_the_reference(case: str, maximize: bool, device: str, ref: Any) -> None:
    batches = _BATCHES[case]
    ours, _ = _lucid_run(device, batches, maximize=maximize)

    emb = ref.nn.Embedding(_ROWS, _DIM, sparse=True)
    emb.weight = ref.nn.Parameter(ref.from_numpy(_W0.copy()))
    opt = ref.optim.SparseAdam(emb.parameters(), lr=_LR, maximize=maximize)
    for k, ids in enumerate(batches):
        opt.zero_grad()
        _loss(emb, ref.tensor(ids)).backward()
        opt.step()
        st = opt.state[emb.weight]
        theirs = {
            "weight": emb.weight.detach().numpy(),
            "exp_avg": st["exp_avg"].numpy(),
            "exp_avg_sq": st["exp_avg_sq"].numpy(),
            "step": np.asarray(st["step"]),
        }
        for key, want in theirs.items():
            np.testing.assert_allclose(
                ours[k][key],
                want,
                rtol=1e-5,
                atol=1e-6,
                err_msg=f"{case} (maximize={maximize}): {key} after step {k + 1}",
            )


def test_rows_outside_a_batch_keep_their_value_and_moments(device: str) -> None:
    out, _ = _lucid_run(device, [[0, 1], [2], [0]])
    first, second, third = out
    for key in ("weight", "exp_avg", "exp_avg_sq"):
        # Row 1 was looked up only in the first batch; rows 3-5 never.
        np.testing.assert_array_equal(second[key][1], first[key][1])
        np.testing.assert_array_equal(third[key][1], first[key][1])
        np.testing.assert_array_equal(third[key][2], second[key][2])
        np.testing.assert_array_equal(third[key][3:], first[key][3:])
    np.testing.assert_array_equal(third["weight"][3:], _W0[3:])
    assert not np.any(third["exp_avg"][3:])
    # One count per parameter, advanced by every step with a gradient.
    assert [int(o["step"]) for o in out] == [1, 2, 3]


def test_a_vector_and_a_scalar_parameter_update_their_nonzero_entries(device: str) -> None:
    vec = nn.Parameter(lucid.tensor([1.0, 2.0, 3.0]).to(device))
    scalar = nn.Parameter(lucid.tensor(4.0).to(device))
    opt = optim.SparseAdam([vec, scalar], lr=_LR)
    vec.grad = lucid.tensor([0.5, 0.0, -0.5]).to(device)
    scalar.grad = lucid.tensor(1.0).to(device)
    opt.step()
    # Each moved by lr against its gradient's sign: the first Adam step.
    np.testing.assert_allclose(vec.numpy(), [0.9, 2.0, 3.1], rtol=1e-6)
    np.testing.assert_allclose(scalar.numpy(), 3.9, rtol=1e-6)
    assert opt.state[0]["exp_avg"].shape == (3,)
    assert opt.state[1]["exp_avg"].shape == ()


def test_a_missing_gradient_skips_and_a_zero_gradient_only_counts(device: str) -> None:
    a = nn.Parameter(lucid.tensor([[1.0, 2.0]]).to(device))
    b = nn.Parameter(lucid.tensor([[3.0, 4.0]]).to(device))
    opt = optim.SparseAdam([a, b], lr=_LR)
    a.grad = lucid.zeros(1, 2).to(device)
    opt.step()
    np.testing.assert_array_equal(a.numpy(), [[1.0, 2.0]])
    assert opt.state[0]["step"] == 1
    assert not np.any(_as_array(opt.state[0]["exp_avg"]))
    assert 1 not in opt.state  # b had no gradient


def test_a_group_added_after_a_step_trains(device: str) -> None:
    a = nn.Parameter(lucid.ones(2, 2).to(device))
    b = nn.Parameter(lucid.ones(3).to(device))
    opt = optim.SparseAdam([a], lr=_LR)
    a.grad = lucid.ones(2, 2).to(device)
    opt.step()
    opt.add_param_group({"params": [b]})
    a.grad = lucid.ones(2, 2).to(device)
    b.grad = lucid.ones(3).to(device)
    opt.step()
    assert opt.state[0]["step"] == 2 and opt.state[1]["step"] == 1
    np.testing.assert_allclose(b.numpy(), [0.9, 0.9, 0.9], rtol=1e-6)


# ── checkpointing ────────────────────────────────────────────────────────────


def _through_disk(tmp_path: Any) -> Callable[[dict[str, Any]], Any]:
    def transfer(sd: dict[str, Any]) -> Any:
        path = str(tmp_path / "optim.lucid")
        lucid.save(sd, path)
        with pytest.warns(UserWarning, match="weights_only=False"):
            return lucid.load(path, weights_only=False)

    return transfer


@pytest.mark.parametrize("how", ["state_dict", "deepcopy", "disk"])
def test_a_restored_optimizer_continues_bit_identically(
    how: str, device: str, tmp_path: Any
) -> None:
    transfer: Callable[[dict[str, Any]], Any] = {
        "state_dict": lambda sd: sd,
        "deepcopy": copy.deepcopy,
        "disk": _through_disk(tmp_path),
    }[how]
    batches = _BATCHES["overlapping"] + _BATCHES["repeated"]
    live = nn.Embedding(_ROWS, _DIM)
    live.weight = nn.Parameter(lucid.tensor(_W0.copy()))
    live.to(device)
    opt = optim.SparseAdam(live.parameters(), lr=_LR)

    def step(emb: nn.Embedding, o: optim.SparseAdam, ids: list[int]) -> None:
        o.zero_grad()
        _loss(emb, lucid.tensor(ids).to(device)).backward()
        o.step()

    for ids in batches[:4]:
        step(live, opt, ids)
    saved = opt.state_dict()
    entry = saved["state"][0]
    assert sorted(entry) == ["exp_avg", "exp_avg_sq", "step"]
    assert entry["step"] == 4
    assert entry["exp_avg"].shape == (_ROWS, _DIM)

    resumed = nn.Embedding(_ROWS, _DIM)
    resumed.weight = nn.Parameter(lucid.tensor(live.weight.numpy().copy()))
    resumed.to(device)
    opt2 = optim.SparseAdam(resumed.parameters(), lr=_LR)
    opt2.load_state_dict(transfer(saved))
    assert opt2.state[0]["exp_avg"].device == live.weight.device

    for k, ids in enumerate(batches[4:]):
        step(live, opt, ids)
        step(resumed, opt2, ids)
        np.testing.assert_array_equal(
            resumed.weight.numpy(), live.weight.numpy(), err_msg=f"step {k + 5}"
        )
    for key in ("exp_avg", "exp_avg_sq", "step"):
        np.testing.assert_array_equal(_as_array(opt2.state[0][key]), _as_array(opt.state[0][key]))


def test_state_dict_leaves_the_live_state_alone(device: str) -> None:
    _, opt = _lucid_run(device, [[0, 1]])
    before = opt.state[0]["exp_avg"]
    opt.state_dict()
    assert opt.state[0]["exp_avg"] is before
    assert isinstance(before, lucid.Tensor)


def _saved(device: str) -> tuple[optim.SparseAdam, dict[str, Any]]:
    _, opt = _lucid_run(device, [[0, 1], [2]])
    return opt, opt.state_dict()


def _break_shape(sd: dict[str, Any]) -> None:
    sd["state"][0]["exp_avg_sq"] = np.zeros((_ROWS + 1, _DIM), np.float32)


def _break_index(sd: dict[str, Any]) -> None:
    sd["state"][1] = sd["state"][0]


def _break_step(sd: dict[str, Any]) -> None:
    sd["state"][0]["step"] = 2.5


@pytest.mark.parametrize(
    ("corrupt", "error"),
    [
        pytest.param(_break_shape, _C_engine.ShapeMismatch, id="shape"),
        pytest.param(_break_index, IndexError, id="index"),
        pytest.param(_break_step, TypeError, id="step"),
    ],
)
def test_a_malformed_checkpoint_is_refused_and_the_state_kept(
    corrupt: Callable[[dict[str, Any]], None], error: type[Exception], device: str
) -> None:
    opt, sd = _saved(device)
    sd = copy.deepcopy(sd)
    corrupt(sd)
    before = {k: _as_array(v) for k, v in opt.state[0].items()}
    with pytest.raises(error):
        opt.load_state_dict(sd)
    for key, want in before.items():
        np.testing.assert_array_equal(_as_array(opt.state[0][key]), want)
