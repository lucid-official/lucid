"""Every binding that makes or exposes a tensor outside the ops tells the tracer.

The op dispatcher records each op for a compile trace or a Core ML export.
Bindings that work on a tensor directly do not go through it. A copy made
there (``transfer_to_device``, a fused kernel, a custom Metal kernel) was a
tensor the trace had never seen, so a host read of it went unnoticed, and an
op reading it took it as a feed pinned at its first value. A binding that
handed out bytes (``to_bytes``, which ``pickle``, ``deepcopy`` and
``lucid.save`` use) or a DLPack capsule read the values without saying so.
Either way every later compiled call answered with the first call's values
(LCD-345).

Two hooks in ``lucid/_C/compile/Tracer.h`` close this: ``traced_derive``
for a binding that returns a new tensor made from existing ones, and
``note_host_read`` for one that exposes values. This scan holds every
binding in ``lucid/_C/bindings/bind*.cpp`` to them, so a new binding cannot
skip them by accident. The behaviour itself is tested in
``test_host_reads.py``.
"""

import re
from pathlib import Path

import pytest

import lucid

_BINDINGS = Path(lucid.__file__).parent / "_C" / "bindings"

#: Files that bind TensorImpl itself. There a binding returning a tensor is
#: never an op (ops are bound elsewhere and record themselves), so any
#: TensorImpl return is held to ``traced_derive``, even one the scan cannot
#: see being built.
_TENSOR_FILES = frozenset({"bind_tensor.cpp", "bind_interop.cpp"})

#: Bindings that make a TensorImpl not from an existing tensor's values. The
#: trace treats each as a constant, which it is.
_FRESH = {
    "from_bytes": "built from a Python bytes object",
    "make_shared_tensor": "a new zero-filled buffer",
    "from_dlpack_metal": "adopts a buffer from outside Lucid",
    "grad_to_tensor": "the gradient, which no forward trace computes",
    "grad_as_impl": "the graph-mode gradient, which no forward trace computes",
    "lstm_forward": "builds only the zero initial state; the LSTM op records itself",
    "custom_function_view": "records a reshape inside TensorImpl::custom_function_view",
}

#: Bindings that hand values to Python but are not host reads of a value the
#: result depends on.
_NOT_A_READ = {
    "to_string": "display only (repr); nothing it returns reaches a result",
    "grad_as_python": "the gradient, which no forward trace computes",
}

#: Bindings to a TensorImpl member function by pointer, whose body the scan
#: cannot see. ``transfer_to_device`` was one of these. Each is reviewed: none
#: returns a tensor or its values.
_BARE_MEMBERS = {
    "numel",
    "nbytes",
    "is_contiguous",
    "is_dense",
    "storage_offset_bytes",
    "storage_nbytes",
    "storage_data_ptr",
    "family_id",
    "grad_as_python",
    "from_bytes",
    "to_string",
    "copy_from",
    "zero_grad",
    "is_aliased",
    "detach_inplace",
    "rebind_version_from",
}

_START = re.compile(
    r'\.def(?:_static|_property|_property_readonly|_readonly|_readwrite)?\(\s*(?:"(\w+)"|py::init)'
)
_IMPL = r"(?:std::shared_ptr<(?:lucid::)?TensorImpl>|(?:lucid::)?TensorImplPtr)"
_RETURNS_IMPL = re.compile(rf"->\s*{_IMPL}\s*\{{")
_CREATES = re.compile(
    r"make_shared<(?:lucid::)?TensorImpl>|make_view\(|->transfer_to_device\("
)
_EXPOSES = re.compile(
    r"(?:->|\.)(?:tolist|item|data_as_python|to_bytes)\(\)|metal_to_dlpack\("
)
_BARE = re.compile(r'^\.def(?:_static)?\(\s*"\w*",\s*&(?:lucid::)?TensorImpl::')


def _bindings(root: Path = _BINDINGS) -> list[tuple[str, str, str]]:
    """``(file, name, code)`` for every binding in ``root/bind*.cpp``.

    A binding's code runs to the next one's start.
    """
    files = sorted(root.glob("bind*.cpp"))
    if not files:
        pytest.skip("the engine sources are not installed")
    found: list[tuple[str, str, str]] = []
    for path in files:
        text = path.read_text()
        starts = list(_START.finditer(text))
        ends = [m.start() for m in starts[1:]] + [len(text)] if starts else []
        for m, end in zip(starts, ends, strict=True):
            name = m.group(1) or "__init__"
            found.append((path.name, name, _code(text[m.start() : end])))
    return found


def _code(src: str) -> str:
    """``src`` without comments or the text of string literals.

    A block runs to the next binding, so it carries the comment above that
    one, which may well mention ``.item()``.
    """
    src = re.sub(r"//[^\n]*", "", src)
    return re.sub(r'"(?:[^"\\\n]|\\.)*"', '""', src)


def underived(root: Path = _BINDINGS) -> list[str]:
    """Bindings that return a tensor made outside the ops without recording it."""
    return [
        f"{file}: {name}"
        for file, name, src in _bindings(root)
        if (
            _CREATES.search(src)
            or (file in _TENSOR_FILES and _RETURNS_IMPL.search(src))
        )
        and name not in _FRESH
        and "traced_derive(" not in src
    ]


def unnoted(root: Path = _BINDINGS) -> list[str]:
    """Bindings that hand a tensor's values to Python without noting the read."""
    return [
        f"{file}: {name}"
        for file, name, src in _bindings(root)
        if _EXPOSES.search(src)
        and name not in _NOT_A_READ
        and "note_host_read(" not in src
    ]


def unreviewed(root: Path = _BINDINGS) -> list[str]:
    """TensorImpl members bound by pointer that nobody checked."""
    return [
        f"{file}: {name}"
        for file, name, src in _bindings(root)
        if _BARE.match(src) and name not in _BARE_MEMBERS
    ]


def test_the_scan_finds_the_bindings() -> None:
    names = {(file, name) for file, name, _src in _bindings()}
    assert {
        ("bind_tensor.cpp", "transfer_to_device"),
        ("bind_tensor.cpp", "to_bytes"),
        ("bind_interop.cpp", "to_dlpack_metal"),
        ("bind.cpp", "_fused_linear_relu"),
        ("bind.cpp", "_run_metal_kernel"),
        ("bind_nn.cpp", "lstm_forward"),
    } <= names


def test_a_binding_returning_a_derived_tensor_records_it() -> None:
    missing = underived()
    assert not missing, (
        "return the new tensor through compile::traced_derive, or list the "
        f"binding in _FRESH with the reason it is a constant: {missing}"
    )


def test_a_binding_exposing_values_notes_the_read() -> None:
    missing = unnoted()
    assert not missing, (
        "call compile::note_host_read before handing values to Python, or "
        f"list the binding in _NOT_A_READ with the reason: {missing}"
    )


def test_a_binding_by_member_pointer_is_reviewed() -> None:
    missing = unreviewed()
    assert not missing, (
        "a member bound by pointer may return a tensor or its values without "
        "the trace hooks; bind a lambda that calls them, or add it to "
        f"_BARE_MEMBERS once checked: {missing}"
    )
