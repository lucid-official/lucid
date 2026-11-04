"""Generate ``lucid/_C/backend/PhantomBackend.h`` from ``IBackend.h``.

The phantom backend is the backend the dispatcher hands out while phantom
mode is on (``lucid/_C/core/PhantomMode.h``): it computes nothing and
allocates nothing, and answers every method with a phantom storage — the
size of which the :class:`TensorImpl` that adopts it fills in, since the op
computed the shape before it dispatched.

Every :class:`IBackend` method needs an answer, and the interface has some
250 of them, so the class is generated rather than written.  It overrides
*every* virtual, the ones with a default body included: a default runs real
code (``to_shared_storage`` hands back its input, the LSTM defaults raise
"not supported"), and a phantom inheriting one would run it on phantom
storage.  So a new virtual of either kind needs a rerun — a pure one fails
to compile without it, and a defaulted one is caught by ``--check``, which
pre-commit and CI run.  A ``virtual`` declaration this parser cannot read
(one defined out of line, say) fails the run rather than being skipped, and
so does a method whose answer cannot be derived from its return type (a
multi-output method whose output count this file does not know).

Answers, by return type:

* ``Storage`` — one phantom storage.  ``diag`` also fills its ``out_shape``
  (the one method whose output shape the backend computes), and the
  data-dependent ``masked_select_count`` refuses.
* ``std::vector<Storage>`` — as many phantoms as the method returns, from
  ``OUTPUT_COUNTS``: a constant, or an expression of the arguments.
* ``std::pair<Storage, Storage>`` / ``StoragePair`` — two phantoms.
* ``CpuStorage`` (a host copy) and ``ClassLossForwardResult`` (a loss with a
  data-dependent count of valid targets) — refuse.

Usage
-----
::

    python tools/gen_phantom_backend.py            # rewrite the header
    python tools/gen_phantom_backend.py --check    # exit 1 if it would change

The output goes through the repository's pinned ``clang-format`` (``$CLANG_FORMAT``,
else the main checkout's ``.venv/bin/clang-format``), so the header passes the
format gate byte for byte and ``--check`` compares formatted text.
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
IBACKEND = ROOT / "lucid/_C/backend/IBackend.h"
HEADER_PATH = ROOT / "lucid/_C/backend/PhantomBackend.h"
HEADER_REL = "lucid/_C/backend/PhantomBackend.h"

# How many storages each multi-output method returns, as a C++ expression of
# its parameters.  The count is the one the CPU and Metal backends return and
# the op indexes: a phantom that returned fewer would read past the end.
OUTPUT_COUNTS: dict[str, str] = {
    "meshgrid": "xs.size()",
    "split_equal": "static_cast<std::size_t>(num_splits)",
    "split_at": "indices.size() + 1",
    "linear_backward": "3",
    "layer_norm_forward": "3",
    "layer_norm_backward": "3",
    # y, mean, rstd.  A fourth slot (MLX's xnorm) is optional: the op reads
    # it only when the backend returns it.
    "batch_norm_forward": "3",
    "batch_norm_backward": "3",
    "group_norm_forward": "3",
    "group_norm_backward": "3",
    "linalg_svd": "compute_uv ? 3 : 1",
    "linalg_lu_factor": "3",
    "linalg_lstsq": "1",
    "linalg_ldl_factor": "3",
    "bce_loss_backward": "3",
    "bce_with_logits_backward": "4",
    "sdpa_forward": "2",
    "sdpa_backward": "4",
    "conv_transpose_nd_backward": "3",
    "conv_nd_backward": "3",
    "rope_forward": "3",
    "max_pool_nd_forward": "2",
    "batch_norm_eval_forward": "2",
    "batch_norm_eval_backward": "5",
    "lp_normalize_forward": "2",
    "global_response_norm_forward": "2",
    "global_response_norm_backward": "3",
    "grid_sample_backward": "2",
    # dx1, dx2, dW and db — an empty db when there is no bias.
    "bilinear_layer_backward": "4",
    "lstm_forward": "3",
    "lstm_forward_train": "5",
    # A gradient for the input, h0, c0 and each weight.
    "lstm_backward": "3 + weights.size()",
}

# Methods whose answer depends on the values: a phantom has none to give.
REFUSE = {"masked_select_count"}

# The parameter a phantom storage takes its dtype from, in order of
# preference.  The dtype only matters for a storage no tensor adopts (a
# gradient in flight); the TensorImpl that adopts one sets it from its own.
DTYPE_PARAMS = ("dst_dt", "out_dtype", "output_dtype", "dt", "dtype")

PAIR_TYPES = {"std::pair<Storage, Storage>", "StoragePair"}
REFUSED_TYPES = {"CpuStorage", "ClassLossForwardResult"}


@dataclass(frozen=True)
class Param:
    type: str
    name: str


@dataclass(frozen=True)
class Method:
    ret: str
    name: str
    params: tuple[Param, ...]
    qualifiers: str
    defaulted: bool  # IBackend gives it a body; overridden all the same


class GenerationError(Exception):
    """IBackend.h holds a method this generator has no answer for."""


def _strip_comments(src: str) -> str:
    src = re.sub(r"/\*.*?\*/", "", src, flags=re.S)
    return "\n".join(re.sub(r"//.*$", "", line) for line in src.split("\n"))


def _split_top_level(text: str) -> list[str]:
    """Split on commas outside <>, (), [] and {}."""
    parts: list[str] = []
    depth = 0
    current: list[str] = []
    for ch in text:
        if ch in "<([{":
            depth += 1
        elif ch in ">)]}":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append("".join(current))
            current = []
            continue
        current.append(ch)
    if "".join(current).strip():
        parts.append("".join(current))
    return [p.strip() for p in parts]


def _parse_param(text: str) -> Param:
    text = re.sub(r"\s*=.*$", "", text).strip()  # an override needs no default
    m = re.match(r"^(.*?[\s&*>])(\w+)$", text)
    if m is None:
        return Param(type=text, name="")  # already unnamed: ``const Shape&``
    return Param(type=" ".join(m.group(1).split()), name=m.group(2))


def parse(src: str) -> list[Method]:
    body = _strip_comments(src)
    methods: list[Method] = []
    destructors = 0
    for m in re.finditer(r"\bvirtual\s+([^;]*?\))\s*([^;{()]*?)\s*(=\s*0\s*;|\{)", body, re.S):
        decl = " ".join(m.group(1).split())
        if decl.startswith("~"):
            destructors += 1
            continue
        dm = re.match(r"^(.*?)\b(\w+)\s*\((.*)\)$", decl)
        if dm is None:
            raise GenerationError(f"cannot parse declaration: {decl}")
        ret, name, params = dm.group(1).strip(), dm.group(2), dm.group(3)
        qualifiers = " ".join(m.group(2).split())
        methods.append(
            Method(
                ret=ret,
                name=name,
                params=tuple(_parse_param(p) for p in _split_top_level(params)),
                qualifiers=qualifiers,
                defaulted=m.group(3) == "{",
            )
        )
    # Every ``virtual`` is a method above or the destructor: one this parser
    # did not read (declared here, defined out of line) would be inherited
    # by the phantom backend and run real code on phantom storage.
    destructors += len(re.findall(r"\bvirtual\s+~\w+\s*\(\s*\)\s*=\s*default\s*;", body))
    seen = len(re.findall(r"\bvirtual\b", body))
    if seen != len(methods) + destructors:
        raise GenerationError(
            f"IBackend.h declares {seen} virtuals and this parser read "
            f"{len(methods) + destructors} — teach tools/gen_phantom_backend.py the new form"
        )
    return methods


def _dtype_expr(method: Method) -> str:
    names = {p.name for p in method.params if p.type == "Dtype"}
    for candidate in DTYPE_PARAMS:
        if candidate in names:
            return candidate
    if method.name == "from_cpu":
        return "cpu.dtype"
    return "Dtype::F32"


def _signature(method: Method, body: str) -> str:
    # Only the parameters the body reads keep their names; the rest stay
    # unnamed, which keeps most overrides on one line.
    used = set(re.findall(r"\b\w+\b", body))
    params = ", ".join(f"{p.type} {p.name}" if p.name in used else p.type for p in method.params)
    quals = f" {method.qualifiers}" if method.qualifiers else ""
    return f"{method.ret} {method.name}({params}){quals} override"


def _body(method: Method) -> str:
    if method.name == "device":
        return "return device_;"
    if method.name in REFUSE or method.ret in REFUSED_TYPES:
        return f'refuse_phantom_read("{method.name}");'
    dt = _dtype_expr(method)
    if method.ret == "Storage":
        if method.name == "diag":
            return f"out_shape = diag_shape(v_shape, k);\nreturn phantom({dt});"
        return f"return phantom({dt});"
    if method.ret in PAIR_TYPES:
        return f"return {{phantom({dt}), phantom({dt})}};"
    if method.ret == "std::vector<Storage>":
        count = OUTPUT_COUNTS.get(method.name)
        if count is None:
            raise GenerationError(
                f"IBackend::{method.name} returns several storages and OUTPUT_COUNTS "
                "does not say how many — add it to tools/gen_phantom_backend.py"
            )
        return f"return phantoms({count}, {dt});"
    raise GenerationError(
        f"IBackend::{method.name} returns {method.ret}, which the phantom backend has no "
        "answer for — teach tools/gen_phantom_backend.py"
    )


PROLOGUE = """\
// lucid/_C/backend/PhantomBackend.h
//
// GENERATED by tools/gen_phantom_backend.py from IBackend.h — do not edit by
// hand; rerun the generator after changing IBackend.h (``--check`` says
// whether it is current).
//
// The backend the dispatcher hands out while phantom mode is on (see
// :file:`../core/PhantomMode.h`).  It computes nothing and allocates
// nothing: every method answers with phantom storages, which the tensor that
// adopts one sizes from the shape its op computed before dispatching.  A
// method whose answer would be a value — a host copy, a data-dependent count
// — refuses with ``LucidError``.
//
// Outside Apple Accelerate and MLX (H3) because it computes nothing; it is
// not a compute stream.

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "../core/PhantomMode.h"
#include "IBackend.h"

namespace lucid {
namespace backend {

class PhantomBackend final : public IBackend {
public:
    explicit PhantomBackend(Device device) noexcept : device_(device) {}

"""

EPILOGUE = """\

private:
    // An unsized phantom on this backend's device: the tensor that adopts it
    // gives it its size, and a gradient handed out as a bare Storage still
    // says where it lives (storage_device).
    Storage phantom(Dtype dt) const {
        refuse_what_device_cannot_hold(dt, device_);
        PhantomStorage p;
        p.dtype = dt;
        p.device = device_;
        return Storage{p};
    }

    std::vector<Storage> phantoms(std::size_t n, Dtype dt) const {
        std::vector<Storage> out;
        out.reserve(n);
        for (std::size_t i = 0; i < n; ++i)
            out.push_back(phantom(dt));
        return out;
    }

    // The shape the CPU and Metal ``diag`` give: a square matrix with ``v``
    // on diagonal ``k`` for a vector, the k-th diagonal for a matrix.
    static Shape diag_shape(const Shape& v_shape, std::int64_t k) {
        const std::int64_t off = k < 0 ? -k : k;
        if (v_shape.size() == 1)
            return Shape{v_shape[0] + off, v_shape[0] + off};
        const std::int64_t rows = v_shape[0] - (k < 0 ? off : 0);
        const std::int64_t cols = v_shape[1] - (k >= 0 ? off : 0);
        const std::int64_t len = rows < cols ? rows : cols;
        return Shape{len > 0 ? len : 0};
    }

    Device device_;
};

}  // namespace backend
}  // namespace lucid
"""


def render(methods: list[Method]) -> str:
    out = [PROLOGUE]
    for method in methods:
        body = _body(method)
        lines = "\n".join(f"        {line}" for line in body.split("\n"))
        out.append(f"    {_signature(method, body)} {{\n{lines}\n    }}\n")
    out.append(EPILOGUE)
    return "".join(out)


def _clang_format() -> str:
    env = os.environ.get("CLANG_FORMAT")
    candidates = [env] if env else []
    common = subprocess.run(
        ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    if common:
        candidates.append(str(Path(common).parent / ".venv/bin/clang-format"))
    candidates.append(str(ROOT / ".venv/bin/clang-format"))
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return candidate
    found = shutil.which("clang-format")
    if found is None:
        raise GenerationError(
            "clang-format not found — set $CLANG_FORMAT (the repository pins one in the "
            "main checkout's .venv)"
        )
    return found


def formatted(text: str) -> str:
    proc = subprocess.run(
        [_clang_format(), "--style=file", f"--assume-filename={HEADER_REL}"],
        cwd=ROOT,
        input=text,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise GenerationError(f"clang-format failed: {proc.stderr.strip()}")
    return proc.stdout


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python tools/gen_phantom_backend.py")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Do not write; exit 1 if the committed header is out of date.",
    )
    args = parser.parse_args(argv)

    try:
        methods = parse(IBACKEND.read_text())
        rendered = formatted(render(methods))
    except GenerationError as exc:
        print(f"gen_phantom_backend: {exc}", file=sys.stderr)
        return 2

    if args.check:
        current = HEADER_PATH.read_text() if HEADER_PATH.exists() else ""
        if current != rendered:
            print(
                "gen_phantom_backend: PhantomBackend.h is out of date with IBackend.h — "
                "run `python tools/gen_phantom_backend.py`",
                file=sys.stderr,
            )
            return 1
        defaulted = sum(m.defaulted for m in methods)
        print(
            f"gen_phantom_backend: PhantomBackend.h matches IBackend.h ({len(methods)} methods "
            f"overridden, {defaulted} of them defaulted in IBackend)"
        )
        return 0

    HEADER_PATH.write_text(rendered)
    print(f"[gen_phantom_backend] {HEADER_PATH.relative_to(ROOT)}  ({len(methods)} methods)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
