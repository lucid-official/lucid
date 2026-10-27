"""Engine nn entries refuse an operand their kernel does not handle (LCD-396).

Every ``lucid._C.engine.nn`` entry is reachable on its own, without the
checks the public ``lucid.nn`` layer adds on top.  Several of them used to
take an input whose rank, per-channel length or window geometry their
kernel never handles and run anyway: ``engine.nn.batch_norm`` on a 3-D
input read a fourth axis that was not there and segfaulted on the CPU while
Metal raised MLX's own ``IndexError``; a zero kernel or stride divided by
zero; a ``running_mean`` without its ``running_var`` dereferenced a null
buffer on both devices.

The invariant is that each door — the op's entry point — checks its operands
with the shared ``require_*`` helpers in ``core/Validate.h`` and refuses with
a typed ``ValueError`` (``ShapeMismatch`` for a rank or a length,
``InvalidArgument`` for a geometry value), identically on both devices,
before anything is dispatched.

Each case runs in a child interpreter, so a crash fails one case instead of
killing the test run: one child per device runs every case in order between
markers, and a child that dies is blamed on the case it had begun and is
restarted past it.
"""

import json
import os
import signal
import subprocess
import sys

import pytest

SHAPE = "ShapeMismatch"
ARG = "InvalidArgument"
OK = "ok"

# (name, code, expected outcome, a fragment the message must contain).
# ``code`` is evaluated in the child with the helpers of ``_CHILD`` in
# scope; whatever it returns is forced, so a lazy Metal failure surfaces
# inside its own case.

_BATCH_NORM = [
    ("bn2d_rank3", "E.nn.batch_norm(R(2, 3, 7), O(3), Z(3))", SHAPE, "must be 4-D"),
    (
        "bn2d_rank5",
        "E.nn.batch_norm(R(2, 3, 4, 4, 4), O(3), Z(3))",
        SHAPE,
        "must be 4-D",
    ),
    ("bn2d_rank2", "E.nn.batch_norm(R(2, 3), O(3), Z(3))", SHAPE, "must be 4-D"),
    ("bn1d_rank2", "E.nn.batch_norm1d(R(2, 3), O(3), Z(3))", SHAPE, "must be 3-D"),
    (
        "bn1d_rank4",
        "E.nn.batch_norm1d(R(2, 3, 4, 4), O(3), Z(3))",
        SHAPE,
        "must be 3-D",
    ),
    (
        "bn3d_rank4",
        "E.nn.batch_norm3d(R(2, 3, 4, 4), O(3), Z(3))",
        SHAPE,
        "must be 5-D",
    ),
    ("bn2d_weight_len", "E.nn.batch_norm(R(2, 3, 4, 4), O(5), Z(3))", SHAPE, "weight"),
    ("bn2d_bias_len", "E.nn.batch_norm(R(2, 3, 4, 4), O(3), Z(5))", SHAPE, "bias"),
    (
        "bn2d_running_mean_len",
        "E.nn.batch_norm(R(2, 3, 4, 4), O(3), Z(3), 1e-5, Z(5), O(3), 0.1)",
        SHAPE,
        "running_mean",
    ),
    (
        "bn2d_running_var_len",
        "E.nn.batch_norm(R(2, 3, 4, 4), O(3), Z(3), 1e-5, Z(3), O(5), 0.1)",
        SHAPE,
        "running_var",
    ),
    (
        "bn2d_running_mean_only",
        "E.nn.batch_norm(R(2, 3, 4, 4), O(3), Z(3), 1e-5, Z(3), None, 0.1)",
        ARG,
        "together",
    ),
    (
        "bn2d_running_var_only",
        "E.nn.batch_norm(R(2, 3, 4, 4), O(3), Z(3), 1e-5, None, O(3), 0.1)",
        ARG,
        "together",
    ),
    (
        "bn_backward_rank3",
        "BW(E.nn.batch_norm(G(2, 3, 7), O(3), Z(3)))",
        SHAPE,
        "must be 4-D",
    ),
    (
        "bn_eval_rank1",
        "E.nn.batch_norm_eval(R(3), Z(3), O(3), O(3), Z(3))",
        SHAPE,
        "at least 2-D",
    ),
    (
        "bn_eval_var_len",
        "E.nn.batch_norm_eval(R(2, 3, 4), Z(3), O(5), O(3), Z(3))",
        SHAPE,
        "running_var",
    ),
    (
        "bn_eval_weight_len",
        "E.nn.batch_norm_eval(R(2, 3, 4), Z(3), O(3), O(5), Z(3))",
        SHAPE,
        "weight",
    ),
    (
        "bn_running_stats_rank1",
        "E.nn.batch_norm_update_running_stats(Z(3), O(3), R(3), [0], 0.1)",
        SHAPE,
        "at least 2-D",
    ),
    (
        "bn_running_stats_rank0",
        "E.nn.batch_norm_update_running_stats(Z(3), O(3), R(), [], 0.1)",
        SHAPE,
        "at least 2-D",
    ),
    (
        "bn_running_stats_len",
        "E.nn.batch_norm_update_running_stats(Z(5), O(5), R(2, 3, 4), [0, 2], 0.1)",
        SHAPE,
        "running_mean",
    ),
    (
        "bn_running_stats_keeps_channel",
        "E.nn.batch_norm_update_running_stats(Z(3), O(3), R(2, 3, 4), [0], 0.1)",
        ARG,
        "channel axis",
    ),
    (
        "bn_running_stats_reduces_channel",
        "E.nn.batch_norm_update_running_stats(Z(3), O(3), R(2, 3, 4), [0, 1, 2], 0.1)",
        ARG,
        "channel axis",
    ),
]

_OTHER_NORMS = [
    ("gn_rank1", "E.nn.group_norm(R(6), O(6), Z(6), 2)", SHAPE, "at least 2-D"),
    ("gn_weight_len", "E.nn.group_norm(R(2, 6, 4), O(3), Z(6), 3)", SHAPE, "weight"),
    ("gn_bias_len", "E.nn.group_norm(R(2, 6, 4), O(6), Z(3), 3)", SHAPE, "bias"),
    ("gn_indivisible", "E.nn.group_norm(R(2, 6, 4), O(6), Z(6), 4)", ARG, "divisible"),
    ("gn_groups_zero", "E.nn.group_norm(R(2, 6, 4), O(6), Z(6), 0)", ARG, "num_groups"),
    (
        "gn_groups_negative",
        "E.nn.group_norm(R(2, 6, 4), O(6), Z(6), -2)",
        ARG,
        "num_groups",
    ),
    (
        "ln_weight_trailing",
        "E.nn.layer_norm(R(2, 3, 4), O(5), Z(5))",
        SHAPE,
        "layer_norm",
    ),
    ("ln_bias_shape", "E.nn.layer_norm(R(2, 3, 4), O(4), Z(3))", SHAPE, "same shape"),
    (
        "ln_bias_backward",
        "BW(E.nn.layer_norm(G(2, 3, 4), O(4), Z(3)))",
        SHAPE,
        "same shape",
    ),
    ("rms_weight_trailing", "E.nn.rms_norm(R(2, 3, 4), O(5))", SHAPE, "rms_norm"),
    (
        "grn_rank3",
        "E.nn.global_response_norm(R(2, 3, 4), O(1, 1, 1, 4), Z(1, 1, 1, 4))",
        SHAPE,
        "must be 4-D",
    ),
]

_CONV = [
    ("conv1d_x_rank2", "E.nn.conv1d(R(3, 8), R(4, 3, 3), None)", SHAPE, "must be 3-D"),
    (
        "conv1d_weight_rank2",
        "E.nn.conv1d(R(2, 3, 8), R(4, 3), None)",
        SHAPE,
        "must be 3-D",
    ),
    ("conv1d_weight_rank0", "E.nn.conv1d(R(2, 3, 8), R(), None)", SHAPE, "weight"),
    ("conv1d_in_channels", "E.nn.conv1d(R(2, 3, 8), R(4, 5, 3), None)", SHAPE, "C_in"),
    ("conv1d_bias_len", "E.nn.conv1d(R(2, 3, 8), R(4, 3, 3), R(7))", SHAPE, "bias"),
    ("conv1d_stride0", "E.nn.conv1d(R(2, 3, 8), R(4, 3, 3), None, 0)", ARG, "stride"),
    (
        "conv1d_pad_negative",
        "E.nn.conv1d(R(2, 3, 8), R(4, 3, 3), None, 1, -1)",
        ARG,
        "padding",
    ),
    (
        "conv2d_x_rank3",
        "E.nn.conv2d(R(3, 8, 8), R(4, 3, 3, 3), None)",
        SHAPE,
        "must be 4-D",
    ),
    (
        "conv2d_x_rank5",
        "E.nn.conv2d(R(2, 3, 8, 8, 8), R(4, 3, 3, 3), None)",
        SHAPE,
        "4-D",
    ),
    (
        "conv2d_in_channels",
        "E.nn.conv2d(R(2, 3, 8, 8), R(4, 5, 3, 3), None)",
        SHAPE,
        "C_in",
    ),
    (
        "conv2d_stride0",
        "E.nn.conv2d(R(2, 3, 8, 8), R(4, 3, 3, 3), None, 0, 0)",
        ARG,
        "stride",
    ),
    (
        "conv3d_x_rank4",
        "E.nn.conv3d(R(3, 4, 4, 4), R(4, 3, 3, 3, 3), None)",
        SHAPE,
        "5-D",
    ),
    (
        "conv3d_in_channels",
        "E.nn.conv3d(R(2, 3, 4, 4, 4), R(4, 5, 3, 3, 3), None)",
        SHAPE,
        "C_in",
    ),
    (
        "convT1d_x_rank2",
        "E.nn.conv_transpose1d(R(3, 8), R(3, 4, 3), None)",
        SHAPE,
        "3-D",
    ),
    (
        "convT1d_in_channels",
        "E.nn.conv_transpose1d(R(2, 3, 8), R(5, 4, 3), None)",
        SHAPE,
        "C_in",
    ),
    (
        "convT1d_weight_rank1",
        "E.nn.conv_transpose1d(R(2, 3, 8), R(3), None)",
        SHAPE,
        "weight",
    ),
    (
        "convT1d_stride0",
        "E.nn.conv_transpose1d(R(2, 3, 8), R(3, 4, 3), None, 0)",
        ARG,
        "stride",
    ),
    (
        "convT1d_output_padding",
        "E.nn.conv_transpose1d(R(2, 3, 8), R(3, 4, 3), None, 1, 0, 3)",
        ARG,
        "output_padding",
    ),
    (
        "convT1d_groups0",
        "E.nn.conv_transpose1d(R(2, 4, 8), R(4, 2, 3), None, 1, 0, 0, 1, 0)",
        ARG,
        "groups",
    ),
    (
        "convT2d_x_rank3",
        "E.nn.conv_transpose2d(R(3, 8, 8), R(3, 4, 3, 3), None)",
        SHAPE,
        "4-D",
    ),
    (
        "convT2d_bias_len",
        "E.nn.conv_transpose2d(R(2, 3, 8, 8), R(3, 4, 3, 3), R(7))",
        SHAPE,
        "bias",
    ),
    (
        "convT3d_in_channels",
        "E.nn.conv_transpose3d(R(2, 3, 4, 4, 4), R(5, 4, 3, 3, 3), None)",
        SHAPE,
        "C_in",
    ),
    (
        "unfold_rank2",
        "E.nn.unfold(R(8, 8), [2, 2], [1, 1], [0, 0], [1, 1])",
        SHAPE,
        "4-D",
    ),
    (
        "unfold_kernel0",
        "E.nn.unfold(R(1, 3, 8, 8), [0, 0], [1, 1], [0, 0], [1, 1])",
        ARG,
        "kernel",
    ),
    (
        "unfold_stride0",
        "E.nn.unfold(R(1, 3, 8, 8), [2, 2], [0, 0], [0, 0], [1, 1])",
        ARG,
        "stride",
    ),
    (
        "fold_rank2",
        "E.nn.fold(R(12, 9), [4, 4], [2, 2], [1, 1], [0, 0], [1, 1])",
        SHAPE,
        "3-D",
    ),
    (
        "fold_channels",
        "E.nn.fold(R(1, 7, 9), [4, 4], [2, 2], [1, 1], [0, 0], [1, 1])",
        ARG,
        "kH*kW",
    ),
    (
        "fold_kernel_len",
        "E.nn.fold(R(1, 4, 9), [4, 4], [2], [1, 1], [0, 0], [1, 1])",
        ARG,
        "2 entries",
    ),
]

_POOL = [
    ("max_pool1d_rank2", "E.nn.max_pool1d(R(3, 8), 2)", SHAPE, "must be 3-D"),
    ("max_pool2d_rank3", "E.nn.max_pool2d(R(3, 8, 8), 2, 2)", SHAPE, "must be 4-D"),
    (
        "max_pool2d_rank5",
        "E.nn.max_pool2d(R(2, 3, 8, 8, 8), 2, 2)",
        SHAPE,
        "must be 4-D",
    ),
    (
        "max_pool3d_rank4",
        "E.nn.max_pool3d(R(3, 4, 4, 4), 2, 2, 2)",
        SHAPE,
        "must be 5-D",
    ),
    ("avg_pool1d_rank2", "E.nn.avg_pool1d(R(3, 8), 2)", SHAPE, "must be 3-D"),
    ("avg_pool2d_rank3", "E.nn.avg_pool2d(R(3, 8, 8), 2, 2)", SHAPE, "must be 4-D"),
    (
        "avg_pool3d_rank4",
        "E.nn.avg_pool3d(R(3, 4, 4, 4), 2, 2, 2)",
        SHAPE,
        "must be 5-D",
    ),
    ("max_pool1d_kernel0", "E.nn.max_pool1d(R(2, 3, 8), 0)", ARG, "kernel_size"),
    ("max_pool2d_kernel0", "E.nn.max_pool2d(R(2, 3, 8, 8), 0, 0)", ARG, "kernel_size"),
    (
        "avg_pool3d_kernel0",
        "E.nn.avg_pool3d(R(2, 3, 4, 4, 4), 0, 0, 0)",
        ARG,
        "kernel_size",
    ),
    (
        "max_pool1d_pad_negative",
        "E.nn.max_pool1d(R(2, 3, 8), 2, 2, -1)",
        ARG,
        "padding",
    ),
    ("avg_pool1d_stride_negative", "E.nn.avg_pool1d(R(2, 3, 8), 2, -1)", ARG, "stride"),
    ("adaptive_max_pool1d_rank2", "E.nn.adaptive_max_pool1d(R(3, 8), 4)", SHAPE, "3-D"),
    (
        "adaptive_max_pool2d_rank3",
        "E.nn.adaptive_max_pool2d(R(3, 8, 8), 4, 4)",
        SHAPE,
        "4-D",
    ),
    (
        "adaptive_avg_pool2d_rank5",
        "E.nn.adaptive_avg_pool2d(R(2, 3, 8, 8, 8), 4, 4)",
        SHAPE,
        "4-D",
    ),
    (
        "adaptive_avg_pool3d_rank4",
        "E.nn.adaptive_avg_pool3d(R(3, 4, 4, 4), 2, 2, 2)",
        SHAPE,
        "5-D",
    ),
]

_RESAMPLE_AND_DROPOUT = [
    (
        "interp_bilinear_rank3",
        "E.nn.interpolate_bilinear(R(3, 8, 8), 4, 4)",
        SHAPE,
        "4-D",
    ),
    (
        "interp_nearest2d_negative",
        "E.nn.interpolate_nearest_2d(R(2, 3, 8, 8), -2, 4)",
        ARG,
        "output",
    ),
    (
        "interp_nearest2d_empty_source",
        "E.nn.interpolate_nearest_2d(R(1, 3, 0, 4), 4, 4)",
        ARG,
        "input spatial",
    ),
    (
        "interp_trilinear_negative",
        "E.nn.interpolate_trilinear(R(2, 3, 4, 4, 4), -2, 2, 2)",
        ARG,
        "output",
    ),
    (
        "interp_nearest3d_rank4",
        "E.nn.interpolate_nearest_3d(R(3, 4, 4, 4), 2, 2, 2)",
        SHAPE,
        "5-D",
    ),
    (
        "grid_sample_rank3",
        "E.nn.grid_sample(R(3, 8, 8), R(1, 4, 4, 2))",
        SHAPE,
        "(N, C, H_in, W_in)",
    ),
    (
        "grid_sample_mode_bicubic",
        "E.nn.grid_sample(R(1, 3, 4, 4), R(1, 4, 4, 2), 2, 0, False)",
        ARG,
        "mode",
    ),
    (
        "grid_sample_pad_reflection",
        "E.nn.grid_sample(R(1, 3, 4, 4), R(1, 4, 4, 2), 0, 2, False)",
        ARG,
        "padding_mode",
    ),
    (
        "affine_grid_negative",
        "E.nn.affine_grid(R(1, 2, 3), 1, -4, 4)",
        ARG,
        "affine_grid",
    ),
    ("dropoutnd_rank1", "E.nn.dropoutnd(R(8), 0.5)", SHAPE, "at least 2-D"),
    ("drop_block_rank3", "E.nn.drop_block(R(3, 8, 8), 3, 0.5)", SHAPE, "4-D"),
]

# Valid ranks still answer, forward and backward, on both devices.
_STILL_ANSWERS = [
    ("bn1d_valid", "BW(E.nn.batch_norm1d(G(2, 3, 7), O(3), Z(3)))", OK, ""),
    (
        "bn2d_valid_running_stats",
        "E.nn.batch_norm(R(2, 3, 4, 4), O(3), Z(3), 1e-5, Z(3), O(3), 0.1)",
        OK,
        "",
    ),
    ("bn3d_valid", "BW(E.nn.batch_norm3d(G(2, 3, 2, 2, 2), O(3), Z(3)))", OK, ""),
    (
        "bn_running_stats_valid",
        "E.nn.batch_norm_update_running_stats(Z(3), O(3), R(2, 3, 4), [2, 0], 0.1)",
        OK,
        "",
    ),
    ("gn_rank2_valid", "BW(E.nn.group_norm(G(2, 6), O(6), Z(6), 2))", OK, ""),
    (
        "conv_transpose_dilated_opad",
        "E.nn.conv_transpose1d(R(2, 3, 8), R(3, 4, 3), None, 1, 0, 1, 2)",
        OK,
        "",
    ),
    ("max_pool_stride_sentinel", "E.nn.max_pool2d(R(2, 3, 8, 8), 2, 2, 0, 0)", OK, ""),
]

CASES = _BATCH_NORM + _OTHER_NORMS + _CONV + _POOL + _RESAMPLE_AND_DROPOUT + _STILL_ANSWERS

_CHILD = r"""
import json
import sys

import lucid
from lucid._C import engine as E

D = sys.argv[1]
lucid.manual_seed(0)


def R(*shape):
    return lucid.randn(*shape, device=D)._impl


def G(*shape):
    return lucid.randn(*shape, device=D, requires_grad=True)._impl


def O(*shape):
    return lucid.ones(*shape, device=D)._impl


def Z(*shape):
    return lucid.zeros(*shape, device=D)._impl


def BW(impl):
    lucid.Tensor.__new_from_impl__(impl).sum().backward()
    return impl


def force(out):
    for impl in out if isinstance(out, (tuple, list)) else [out]:
        if impl is not None:
            lucid.Tensor.__new_from_impl__(impl).sum().item()


for name, code in json.loads(sys.stdin.read()):
    print("@@BEGIN " + name, flush=True)
    try:
        force(eval(code))
        verdict = ["ok", "", False]
    except Exception as e:
        verdict = [
            type(e).__module__ + "." + type(e).__qualname__,
            str(e)[:400],
            isinstance(e, ValueError) and isinstance(e, E.LucidError),
        ]
    print("@@END " + name + " " + json.dumps(verdict), flush=True)
"""

_CHILD_TIMEOUT_S = 600


def _describe_exit(code: int) -> str:
    if code < 0:
        try:
            return f"killed by {signal.Signals(-code).name}"
        except ValueError:
            return f"killed by signal {-code}"
    return f"exit {code}"


def _run_cases(device: str) -> dict[str, list[object]]:
    """Run every case on ``device``; a dead child is restarted past its case."""
    results: dict[str, list[object]] = {}
    pending = [(name, code) for name, code, _, _ in CASES]
    env = dict(os.environ, PYTHONWARNINGS="ignore")
    while pending:
        done = subprocess.run(
            [sys.executable, "-c", _CHILD, device],
            input=json.dumps(pending),
            capture_output=True,
            text=True,
            timeout=_CHILD_TIMEOUT_S,
            env=env,
        )
        current = None
        for line in done.stdout.splitlines():
            if line.startswith("@@BEGIN "):
                current = line[len("@@BEGIN ") :]
            elif line.startswith("@@END "):
                name, _, verdict = line[len("@@END ") :].partition(" ")
                results[name] = json.loads(verdict)
                current = None
        if done.returncode == 0:
            break
        tail = done.stderr.strip()[-600:]
        if current is None:
            for name, _ in pending:
                results.setdefault(
                    name, ["crash", f"{_describe_exit(done.returncode)}: {tail}", False]
                )
            break
        results[current] = [
            "crash",
            f"{_describe_exit(done.returncode)}: {tail}",
            False,
        ]
        pending = [(n, c) for n, c in pending if n not in results]
    return results


_RESULTS: dict[str, dict[str, list[object]]] = {}


def _result(device: str, name: str) -> list[object]:
    if device not in _RESULTS:
        _RESULTS[device] = _run_cases(device)
    return _RESULTS[device].get(name, ["missing", "the child never reached this case", False])


@pytest.mark.parametrize(
    "name,expect,needle",
    [(n, e, s) for n, _, e, s in CASES],
    ids=[n for n, _, _, _ in CASES],
)
def test_engine_door_refuses_unhandled_operand(
    device: str, name: str, expect: str, needle: str
) -> None:
    kind, detail, typed_value_error = _result(device, name)
    assert kind != "crash", f"{name} on {device} killed the interpreter — {detail}"
    if expect == OK:
        assert kind == "ok", f"{name} on {device}: {kind}: {detail}"
        return
    assert kind == f"lucid._C.engine.{expect}", f"{name} on {device}: {kind}: {detail}"
    assert typed_value_error, f"{name} on {device}: {kind} is not a ValueError LucidError"
    assert needle in str(detail), f"{name} on {device}: {needle!r} not in {detail!r}"


def test_case_names_are_unique() -> None:
    names = [n for n, _, _, _ in CASES]
    assert len(names) == len(set(names))
