"""The engine nodes that still refuse ``create_graph=True`` — a ledger that
only shrinks (LCD-436).

A node without a graph-mode backward raises "create_graph=True is not yet
supported" the first time a second derivative passes through it: eval
``dropout`` and ``batch_norm`` did, though one is the identity and the other
affine.  The scan reads the engine sources and finds every autograd node
class that neither overrides ``apply_for_graph`` nor supplies
``grad_formula_impl``, and every graph-mode backward that refuses some of its
inputs.  Both sets must equal the ledgers below: a new refusal fails here
until it is either implemented or recorded with the reference's own
behaviour, and an entry that is implemented has to leave the ledger.
"""

import re
from pathlib import Path

import pytest

import lucid
from lucid._C import engine as _C_engine
from lucid._dispatch import _unwrap, _wrap

_ENGINE = Path(lucid.__file__).parent / "_C"

# Node class -> why it still refuses, and what the reference does there.
_REFUSING_NODES = {
    "BCELossBackward": "engine-only (F.binary_cross_entropy is a Python composite with "
    "double backward); the reference supports it",
    "BCEWithLogitsBackward": "engine-only (F.binary_cross_entropy_with_logits is a "
    "composite); the reference supports it",
    "CrossEntropyBackward": "engine-only (F.cross_entropy is a composite); the reference "
    "supports it",
    "NLLLossBackward": "engine-only (F.nll_loss is a composite); the reference supports it",
    "CtcLossBackward": "the reference refuses too: derivative for _ctc_loss_backward is "
    "not implemented",
    "GlobalResponseNormBackward": "engine-only, no Python caller; no reference built-in",
}

# Class whose graph-mode backward refuses some cases -> which, and why.
_PARTIAL_REFUSALS = {
    "AvgPoolNdBackward": "overlapping or padded windows",
    "BatchNormEvalBackward": "a running_var that requires grad: the node keeps only rstd; "
    "the reference refuses any running statistic that requires grad",
    "CofactorBackward": "a third derivative of det at a singular matrix",
    "EmbeddingBagBackward": "an index dtype the graph path does not cover",
    "GridSampleBackward": "inputs that are not 4-D",
    "InterpolateNearestBackward2D": "a non-integer scale factor",
    "InterpolateNearestBackward3D": "a non-integer scale factor",
    "LpNormalizeBackward": "an infinite ord or one below 1",
    "LstmBackward": "every case; the reference's LSTM double-backward goes through its "
    "composite cell",
    "PythonBackwardNode": "a Function with several outputs, or a backward that is not "
    "differentiable",
}

_NODE_BASES = {
    "FuncOp",
    "UnaryOp",
    "BinaryOp",
    "ReduceOp",
    "AutogradNode",
    "Node",
    "UnaryKernel",
    "BinaryKernel",
    "ReduceKernel",
    "NaryKernel",
    "VariadicKernel",
}
# Reductions inherit a graph-mode backward from their kernel.
_GRAPH_BY_BASE = {"ReduceOp", "ReduceKernel"}
_GRAPH_MEMBERS = ("apply_for_graph", "grad_formula_impl", "accumulate_barrier_grad_for_graph")
_DECL = re.compile(
    r"\b(?:class|struct)\s+(?:LUCID_API\s+|LUCID_INTERNAL\s+)?(\w+)\s*(?:final\s*)?"
    r":\s*public\s+([\w:]+)"
)
# A graph-mode entry point with a body: in the class, or out of line on a
# class (template arguments included).
_GRAPH_DEF = re.compile(
    r"(?:(\w+)(?:<[^>]*>)?::)?(?:apply_for_graph|accumulate_barrier_grad_for_graph"
    r"|grad_formula_impl)\s*\([^)]*\)\s*(?:const\s*)?(?:override\s*)?\{"
)
_REFUSES = re.compile(r"not_implemented|\brefuse_create_graph\(")
_COMMENT = re.compile(r"//[^\n]*")


def _sources():
    files = [p for p in _ENGINE.rglob("*") if p.suffix in (".h", ".cpp", ".mm")]
    if not any(p.suffix == ".h" for p in files):
        pytest.skip("engine sources are not shipped with this install")
    return {p: _COMMENT.sub("", p.read_text(errors="ignore")) for p in files}


def _body(text, open_brace):
    depth = 0
    for j in range(open_brace, len(text)):
        depth += {"{": 1, "}": -1}.get(text[j], 0)
        if depth == 0:
            return text[open_brace:j]
    return text[open_brace:]


def _node_classes(sources):
    """Every concrete autograd node: name -> (base, class body)."""
    nodes = {}
    for text in sources.values():
        for m in _DECL.finditer(text):
            name, base = m.group(1), m.group(2).split("::")[-1]
            if base not in _NODE_BASES or name in _NODE_BASES:
                continue
            body = _body(text, text.find("{", m.end()))
            if re.search(r"kHasGradient\s*=\s*false", body):
                continue
            nodes[name] = (base, body)
    return nodes


def _refusing_nodes(sources):
    everything = "\n".join(sources.values())
    refusing = set()
    for name, (base, body) in _node_classes(sources).items():
        if base in _GRAPH_BY_BASE or any(k in body for k in _GRAPH_MEMBERS):
            continue
        if re.search(rf"\b{name}::(apply_for_graph|grad_formula_impl)\b", everything):
            continue
        refusing.add(name)
    return refusing


def _partial_refusals(sources):
    partial = set()
    for text in sources.values():
        for m in _GRAPH_DEF.finditer(text):
            body = _body(text, m.end() - 1)
            if not _REFUSES.search(body):
                continue
            owner = m.group(1)
            if owner is None:
                enclosing = [d.group(1) for d in _DECL.finditer(text, 0, m.start())]
                owner = enclosing[-1] if enclosing else "?"
            partial.add(owner)
    return partial


def test_refusing_nodes_match_the_ledger():
    found = _refusing_nodes(_sources())
    new = sorted(found - set(_REFUSING_NODES))
    gone = sorted(set(_REFUSING_NODES) - found)
    assert not new, f"nodes without a graph-mode backward, not in the ledger: {new}"
    assert not gone, f"ledger entries that now have a graph-mode backward — remove them: {gone}"


def test_partial_refusals_match_the_ledger():
    found = _partial_refusals(_sources())
    new = sorted(found - set(_PARTIAL_REFUSALS))
    gone = sorted(set(_PARTIAL_REFUSALS) - found)
    assert not new, f"graph-mode backwards that refuse a case, not in the ledger: {new}"
    assert not gone, f"ledger entries that no longer refuse — remove them: {gone}"


def test_the_scan_sees_the_nodes_it_should():
    """Guard the scanner itself: it must find the graph-capable nodes this
    change added, or an empty result would pass the ledger vacuously."""
    sources = _sources()
    nodes = _node_classes(sources)
    for name in (
        "DropoutBackward",
        "BatchNormEvalBackward",
        "MaximumBackward",
        "CubeBackward",
        "RotaryPosEmbeddingBackward",
    ):
        assert name in nodes, name
    assert len(nodes) > 100
    assert not {"DropoutBackward", "BatchNormEvalBackward"} & _refusing_nodes(sources)


# ── the nodes this ledger lost, held to their second derivatives ────────────

_X = [0.5, -1.0, 2.0, 0.25, -3.0, 1.5, 0.75, -0.5, 1.25, 0.1, -2.0, 0.3, 0.6, -0.7, 0.9, 1.1]

# name -> (Lucid call, reference call) on a 16-element x.  Engine-only ops
# are reached through the binding and held to the reference's composite.
_ENGINE_OPS = {
    "cube": (lambda x: _wrap(_C_engine.cube(_unwrap(x))), lambda x: x**3),
    "cube_root": (
        lambda x: _wrap(_C_engine.cube_root(_unwrap(x.abs() + 0.5))),
        lambda x: (x.abs() + 0.5) ** (1.0 / 3.0),
    ),
    "rpow_scalar": (lambda x: _wrap(_C_engine.rpow_scalar(2.0, _unwrap(x))), lambda x: 2.0**x),
    "norm_2": (
        lambda x: _wrap(_C_engine.linalg.norm(_unwrap(x.reshape(4, 4)), 2.0, [1], False)),
        lambda x: x.reshape(4, 4).norm(2, 1),
    ),
    "norm_1": (
        lambda x: _wrap(_C_engine.linalg.norm(_unwrap(x.reshape(4, 4)), 1.0, [1], False)),
        lambda x: x.reshape(4, 4).norm(1, 1),
    ),
    "complex_abs": (lambda x: lucid.abs(lucid.complex(x, x * x)), lambda x: (x + 1j * x * x).abs()),
}


def _second_derivative(lib, call, x):
    y = call(x)
    (g,) = lib.autograd.grad((y * y).sum() + y.sum(), x, create_graph=True)
    (h,) = lib.autograd.grad(g.sum(), x)
    return g, h


@pytest.mark.parametrize("name", list(_ENGINE_OPS))
def test_engine_op_second_derivative_matches_reference(ref, device, name):
    ours, theirs = _ENGINE_OPS[name]
    want_g, want_h = _second_derivative(ref, theirs, ref.tensor(_X, requires_grad=True))
    got_g, got_h = _second_derivative(
        lucid, ours, lucid.tensor(_X, device=device, requires_grad=True)
    )
    assert got_g.requires_grad == want_g.requires_grad
    assert got_g.tolist() == pytest.approx(want_g.tolist(), rel=1e-4, abs=1e-5)
    assert got_h.tolist() == pytest.approx(want_h.tolist(), rel=1e-4, abs=1e-5)


@pytest.mark.parametrize("interleaved", [False, True], ids=["halves", "pairs"])
def test_rope_second_derivative(device, interleaved):
    """RoPE is a rotation, so ``sum(y**2) == sum(x**2)``: the gradient is
    ``2x`` and its derivative 2 everywhere, in either pair layout."""
    x = lucid.tensor(_X, device=device, requires_grad=True)
    y = _wrap(_C_engine.nn.rotary_pos_embedding(_unwrap(x.reshape(2, 2, 4)), None, interleaved))
    (g,) = lucid.autograd.grad((y * y).sum(), x, create_graph=True)
    assert g.tolist() == pytest.approx([2 * v for v in _X], rel=1e-5, abs=1e-5)
    (h,) = lucid.autograd.grad(g.sum(), x)
    assert h.tolist() == pytest.approx([2.0] * len(_X), rel=1e-5)


_MASKED = {
    "dropout": lambda x: lucid.nn.functional.dropout(x, 0.5, training=True),
    "dropout2d": lambda x: lucid.nn.functional.dropout2d(x.reshape(2, 4, 2, 1), 0.5, True),
    "alpha_dropout": lambda x: lucid.nn.functional.alpha_dropout(x, 0.5, training=True),
    "drop_path": lambda x: _wrap(_C_engine.nn.drop_path(_unwrap(x.reshape(4, 4)), 0.5, True)),
}


@pytest.mark.parametrize("name", list(_MASKED))
def test_masked_second_derivative(device, name):
    """``y = a * m * x + b`` with the drawn mask: ``sum(y**2)`` has the
    gradient ``2 a m y`` and the second derivative ``2 (a m)**2``, which
    the slope ``a m`` read back from ``grad(y.sum())`` predicts."""
    lucid.manual_seed(436)
    x = lucid.tensor(_X, device=device, requires_grad=True)
    y = _MASKED[name](x)
    (slope,) = lucid.autograd.grad(y.sum(), x, retain_graph=True)
    assert not slope.requires_grad
    (g,) = lucid.autograd.grad((y * y).sum(), x, create_graph=True)
    (h,) = lucid.autograd.grad(g.sum(), x)
    assert h.tolist() == pytest.approx([2 * s * s for s in slope.tolist()], rel=1e-5)
