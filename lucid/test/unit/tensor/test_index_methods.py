"""``gather`` in the reference order, and the ``Tensor.index_*`` methods.

``gather`` took ``(input, indices, dim=-1)``; the reference takes
``(input, dim, index)`` and ``x.gather(1, idx)`` failed with a binding
error.  The reference order is the signature now; the old one is refused.

``index_add`` / ``index_copy`` / ``index_fill`` / ``index_put`` and their
in-place forms existed only as ``lucid.*`` functions.  The methods are
held here to a NumPy oracle on both devices, and the in-place forms to
what an in-place op owes: the values land in ``self``'s storage (a CPU view
sees them), ``self`` takes the write's place in the graph, and a leaf that
requires grad is refused while autograd records.
"""

import ast
from pathlib import Path

import numpy as np
import pytest

import lucid
import lucid.nn as nn
from lucid._C import engine as _C_engine
import lucid.nn.functional as F
from lucid.test._helpers.compare import assert_close

METHODS = (
    "index_add",
    "index_add_",
    "index_copy",
    "index_copy_",
    "index_fill",
    "index_fill_",
    "index_put",
    "index_put_",
)

# ``(shape, dim, index)`` — repeated indices only where the op allows them.
CASES = [
    ((5,), 0, [0, 3, 3]),
    ((4, 3), 0, [2, 0]),
    ((4, 3), 1, [1, 1, 0, 2]),
    ((2, 3, 4), -1, [3, 0]),
    ((2, 3, 4), 1, [2]),
]


def _rng() -> np.random.Generator:
    return np.random.default_rng(21)


def _along(dim: int, ndim: int, j: int) -> tuple[slice | int, ...]:
    dim %= ndim
    return (slice(None),) * dim + (j,)


def _source(shape: tuple[int, ...], dim: int, m: int) -> np.ndarray:
    sshape = list(shape)
    sshape[dim % len(shape)] = m
    return _rng().standard_normal(sshape).astype(np.float32)


def _ref_index_add(
    x: np.ndarray, dim: int, index: list[int], src: np.ndarray, alpha: float
) -> np.ndarray:
    out = x.copy()
    for i, j in enumerate(index):
        out[_along(dim, x.ndim, j)] += alpha * src[_along(dim, x.ndim, i)]
    return out


def _ref_index_copy(
    x: np.ndarray, dim: int, index: list[int], src: np.ndarray
) -> np.ndarray:
    out = x.copy()
    for i, j in enumerate(index):
        out[_along(dim, x.ndim, j)] = src[_along(dim, x.ndim, i)]
    return out


def _ref_index_fill(
    x: np.ndarray, dim: int, index: list[int], value: float
) -> np.ndarray:
    out = x.copy()
    for j in index:
        out[_along(dim, x.ndim, j)] = value
    return out


def _unique(index: list[int]) -> list[int]:
    return list(dict.fromkeys(index))


# ── gather ────────────────────────────────────────────────────────────────────


class TestGatherOrder:
    X = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    IDX = [[0, 2], [1, 1]]
    WANT = [[1.0, 3.0], [5.0, 5.0]]

    def _operands(self, device: str) -> tuple[lucid.Tensor, lucid.Tensor]:
        return lucid.tensor(self.X, device=device), lucid.tensor(
            self.IDX, device=device
        )

    def test_the_reference_order_answers(self, device: str) -> None:
        x, idx = self._operands(device)
        for out in (
            x.gather(1, idx),
            x.gather(dim=1, index=idx),
            lucid.gather(x, 1, idx),
            lucid.gather(x, 1, index=idx),
            lucid.gather(x, dim=1, index=idx),
            lucid.gather(x, -1, idx),
        ):
            assert_close(out, np.array(self.WANT, dtype=np.float32))

    def test_a_zero_d_integer_tensor_is_a_dim(self, device: str) -> None:
        # The reference reads a 0-d integer tensor there as the dim, so it
        # is the reference order even though a tensor sits where the old
        # order put its index.
        x, idx = self._operands(device)
        dim = lucid.tensor(1, device=device)
        for out in (
            lucid.gather(x, dim, idx),
            x.gather(dim, idx),
            lucid.gather(x, dim, index=idx),
            lucid.gather(x, dim=dim, index=idx),
        ):
            assert_close(out, np.array(self.WANT, dtype=np.float32))

    def test_input_binds_by_keyword(self, device: str) -> None:
        x, idx = self._operands(device)
        out = lucid.gather(input=x, dim=1, index=idx)
        assert_close(out, np.array(self.WANT, dtype=np.float32))

    def test_it_matches_take_along_axis(self, device: str) -> None:
        a = _rng().standard_normal((3, 4, 5)).astype(np.float32)
        for dim in (0, 1, 2, -1):
            shape = list(a.shape)
            shape[dim] = 2
            index = _rng().integers(0, a.shape[dim], shape)
            want = np.take_along_axis(a, index, axis=dim)
            x = lucid.tensor(a, device=device)
            assert_close(x.gather(dim, lucid.tensor(index, device=device)), want)
            assert_close(lucid.gather(x, dim, lucid.tensor(index, device=device)), want)

    @pytest.mark.parametrize(
        "call",
        [
            lambda x, i: lucid.gather(x, i, 1),
            lambda x, i: lucid.gather(x, i, dim=1),
            lambda x, i: lucid.gather(x, indices=i, dim=1),
            lambda x, i: x.gather(i, 1),
            lambda x, i: x.gather(i, dim=1),
            lambda x, i: lucid.gather(x, i),
        ],
        ids=[
            "free-positional",
            "free-dim-kw",
            "free-indices-kw",
            "method",
            "method-dim-kw",
            "no-dim",
        ],
    )
    def test_the_old_lucid_order_is_refused(self, call) -> None:  # type: ignore[no-untyped-def]
        # ``gather(input, indices, dim=-1)`` was Lucid's order until 3.16;
        # it answered with a warning until the reference's became the only
        # one (LCD-410).
        x, idx = self._operands("cpu")
        with pytest.raises(TypeError):
            call(x, idx)

    @pytest.mark.parametrize(
        "call,message",
        [
            (lambda x, i: lucid.gather(x, 1), "required positional argument: 'index'"),
            (lambda x, i: lucid.gather(x, 1.5, i), "dim must be an int"),
            (lambda x, i: lucid.gather(x, 1, [0, 1]), "index must be a Tensor"),
            (
                lambda x, i: lucid.gather(x, 1, i, axis=1),
                "unexpected keyword argument 'axis'",
            ),
            (
                lambda x, i: lucid.gather(x, 1, dim=1, index=i),
                "multiple values for argument 'dim'",
            ),
            (
                lambda x, i: lucid.gather(x, i, index=i),
                "0-d integer tensor",
            ),
            (lambda x, i: x.gather(1, i, 0), "takes 3 positional arguments"),
            (lambda x, i: lucid.gather(x, lucid.tensor(1.0), i), "0-d integer tensor"),
            (lambda x, i: lucid.gather(x, lucid.tensor(True), i), "0-d integer tensor"),
            (lambda x, i: lucid.gather(x, i, i), "0-d integer tensor"),
        ],
        ids=[
            "no-index",
            "float-dim",
            "list-index",
            "bad-kw",
            "dim-twice",
            "index-twice",
            "too-many",
            "float-tensor-dim",
            "bool-tensor-dim",
            "two-index-tensors",
        ],
    )
    def test_a_call_neither_order_binds_is_a_type_error(self, call, message: str) -> None:  # type: ignore[no-untyped-def]
        x, idx = self._operands("cpu")
        with pytest.raises(TypeError, match=message):
            call(x, idx)

    def test_it_differentiates_in_the_reference_order(self, device: str) -> None:
        x = lucid.tensor(self.X, device=device, requires_grad=True)
        _, idx = self._operands(device)
        x.gather(1, idx).sum().backward()
        assert x.grad is not None
        assert_close(
            x.grad, np.array([[1.0, 0.0, 1.0], [0.0, 2.0, 0.0]], dtype=np.float32)
        )


class TestLibraryUsesTheReferenceOrder:
    """No caller inside Lucid spells the old order."""

    def test_internal_callers_run(self) -> None:
        rng = _rng()
        logits = lucid.tensor(rng.standard_normal((4, 5)).astype(np.float32))
        target = lucid.tensor([0, 3, 1, 4])
        weight = lucid.tensor(rng.random(5).astype(np.float32))
        x = lucid.tensor(rng.standard_normal((2, 3, 6)).astype(np.float32))
        F.cross_entropy(logits, target, weight=weight)
        F.nll_loss(F.log_softmax(logits, dim=1), target, weight=weight)
        F.pad(x, (2, 2), mode="reflect")
        F.pdist(lucid.tensor(rng.standard_normal((4, 3)).astype(np.float32)))
        lucid.take_along_dim(x, lucid.tensor([[[0, 5]]]), dim=2)
        lucid.view_as_complex(lucid.tensor([[1.0, 2.0], [3.0, 4.0]]))
        nn.RNN(6, 4, bidirectional=True)(x.transpose(0, 1))
        lucid.distributions.Categorical(logits=logits).log_prob(target)

    def test_no_library_call_spells_the_old_order(self) -> None:
        """The callers the test above cannot reach — the zoo, Core ML export,
        ``tools/`` — are held statically.

        A ``gather`` with the dim as a keyword and only two positionals, the
        old ``indices=``, or an integer literal third is the index-first
        order.  Not caught: the old order with the dim in a variable,
        ``gather(x, idx, dim)``, which reads like ``gather(x, dim, idx)``.
        """
        package = Path(lucid.__file__).parent
        sources = [
            p
            for p in package.rglob("*.py")
            if "test" not in p.relative_to(package).parts
        ]
        tools = package.parent / "tools"
        if tools.is_dir():  # a source checkout; an installed wheel has none
            sources += tools.rglob("*.py")
        offenders = []
        for path in sorted(sources):
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                if not (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "gather"
                ):
                    continue
                owner = node.func.value
                if isinstance(owner, ast.Name) and owner.id.endswith("engine"):
                    continue  # the engine binding keeps (a, indices, dim)
                keywords = {k.arg for k in node.keywords}
                is_method = not (
                    isinstance(owner, ast.Name) and owner.id in ("lucid", "_lucid")
                )
                positionals = len(node.args) + (1 if is_method else 0)
                third = (
                    node.args[2 - (1 if is_method else 0)] if positionals == 3 else None
                )
                old = (
                    "indices" in keywords
                    or (positionals == 2 and "index" not in keywords)
                    or isinstance(third, ast.Constant)
                    or (
                        isinstance(third, ast.UnaryOp)
                        and isinstance(third.operand, ast.Constant)
                    )
                )
                if old:
                    offenders.append(
                        f"{path.relative_to(package.parent)}:{node.lineno}"
                    )
        assert offenders == []


# ── index_* methods ───────────────────────────────────────────────────────────


def test_the_eight_methods_exist() -> None:
    assert [m for m in METHODS if not hasattr(lucid.Tensor, m)] == []


class TestOutOfPlace:
    @pytest.mark.parametrize("shape,dim,index", CASES)
    def test_index_add(self, device: str, shape, dim: int, index: list[int]) -> None:  # type: ignore[no-untyped-def]
        a = _rng().standard_normal(shape).astype(np.float32)
        src = _source(shape, dim, len(index))
        x = lucid.tensor(a, device=device)
        out = x.index_add(
            dim,
            lucid.tensor(index, device=device),
            lucid.tensor(src, device=device),
            alpha=1.5,
        )
        assert_close(out, _ref_index_add(a, dim, index, src, 1.5))
        assert_close(x, a)  # untouched

    @pytest.mark.parametrize("shape,dim,index", CASES)
    def test_index_copy(self, device: str, shape, dim: int, index: list[int]) -> None:  # type: ignore[no-untyped-def]
        index = _unique(index)
        a = _rng().standard_normal(shape).astype(np.float32)
        src = _source(shape, dim, len(index))
        x = lucid.tensor(a, device=device)
        out = x.index_copy(
            dim, lucid.tensor(index, device=device), lucid.tensor(src, device=device)
        )
        assert_close(out, _ref_index_copy(a, dim, index, src))
        assert_close(x, a)

    @pytest.mark.parametrize("shape,dim,index", CASES)
    def test_index_fill(self, device: str, shape, dim: int, index: list[int]) -> None:  # type: ignore[no-untyped-def]
        a = _rng().standard_normal(shape).astype(np.float32)
        x = lucid.tensor(a, device=device)
        out = x.index_fill(dim, lucid.tensor(index, device=device), -2.5)
        assert_close(out, _ref_index_fill(a, dim, index, -2.5))
        assert_close(x, a)

    @pytest.mark.parametrize("accumulate", [False, True])
    def test_index_put(self, device: str, accumulate: bool) -> None:
        a = _rng().standard_normal((4, 3)).astype(np.float32)
        rows, cols = [0, 2, 2], [1, 0, 0]
        if not accumulate:
            rows, cols = rows[:2], cols[:2]
        values = _rng().standard_normal(len(rows)).astype(np.float32)
        want = a.copy()
        for r, c, v in zip(rows, cols, values):
            want[r, c] = want[r, c] + v if accumulate else v
        x = lucid.tensor(a, device=device)
        index = (lucid.tensor(rows, device=device), lucid.tensor(cols, device=device))
        out = x.index_put(
            index, lucid.tensor(values, device=device), accumulate=accumulate
        )
        assert_close(out, want)
        assert_close(x, a)

    def test_index_put_takes_trailing_dimensions_whole(self, device: str) -> None:
        a = np.zeros((4, 3), dtype=np.float32)
        values = _rng().standard_normal((2, 3)).astype(np.float32)
        out = lucid.tensor(a, device=device).index_put(
            [lucid.tensor([3, 1], device=device)], lucid.tensor(values, device=device)
        )
        want = a.copy()
        want[[3, 1]] = values
        assert_close(out, want)

    @pytest.mark.parity
    def test_index_put_refuses_0d_narrow_integer_indices(
        self, ref: object, device: str
    ) -> None:
        x = lucid.zeros(3, device=device)
        v = lucid.tensor(5.0, device=device)
        rx = ref.zeros(3)  # type: ignore[attr-defined]
        rv = ref.tensor(5.0)  # type: ignore[attr-defined]
        for dtype_name in ("int8", "int16"):
            idx = lucid.tensor(1, dtype=getattr(lucid, dtype_name), device=device)
            ridx = ref.tensor(1, dtype=getattr(ref, dtype_name))  # type: ignore[attr-defined]
            with pytest.raises(IndexError):
                rx.index_put((ridx,), rv)
            with pytest.raises(IndexError, match="index_put"):
                x.index_put((idx,), v)
            with pytest.raises(IndexError, match="index_put_"):
                x.clone().index_put_((idx,), v)
            assert x[idx].item() == 0.0
        for dtype_name in ("int32", "int64"):
            idx = lucid.tensor(1, dtype=getattr(lucid, dtype_name), device=device)
            assert x.index_put((idx,), v).tolist() == [0.0, 5.0, 0.0]

    def test_a_source_of_another_dtype_is_refused_not_reinterpreted(
        self, device: str
    ) -> None:
        # The engine's scatter read an int source's bits as float32; then the
        # composites cast it, where the reference refuses (LCD-324).
        x = lucid.zeros(3, device=device)
        src = lucid.tensor([1, 2], dtype=lucid.int32, device=device)
        index = lucid.tensor([0, 2], device=device)
        with pytest.raises(_C_engine.DtypeMismatch, match="index_add"):
            x.index_add(0, index, src)
        with pytest.raises(_C_engine.DtypeMismatch, match="index_copy"):
            x.index_copy(0, index, src)


class TestInPlace:
    @pytest.mark.parametrize("shape,dim,index", CASES)
    def test_each_writes_self_and_returns_it(self, device: str, shape, dim: int, index: list[int]) -> None:  # type: ignore[no-untyped-def]
        a = _rng().standard_normal(shape).astype(np.float32)
        unique = _unique(index)
        src = _source(shape, dim, len(index))
        usrc = _source(shape, dim, len(unique))
        idx = lucid.tensor(index, device=device)
        uidx = lucid.tensor(unique, device=device)
        for method, args, want in [
            (
                "index_add_",
                (dim, idx, lucid.tensor(src, device=device)),
                _ref_index_add(a, dim, index, src, 1.0),
            ),
            (
                "index_copy_",
                (dim, uidx, lucid.tensor(usrc, device=device)),
                _ref_index_copy(a, dim, unique, usrc),
            ),
            ("index_fill_", (dim, idx, 7.0), _ref_index_fill(a, dim, index, 7.0)),
        ]:
            x = lucid.tensor(a, device=device)
            assert getattr(x, method)(*args) is x
            assert_close(x, want, msg=method)

    def test_index_add_takes_alpha_by_keyword(self, device: str) -> None:
        x = lucid.zeros(3, device=device)
        x.index_add_(
            0,
            lucid.tensor([1], device=device),
            lucid.ones(1, device=device),
            alpha=-2.0,
        )
        assert_close(x, np.array([0.0, -2.0, 0.0], dtype=np.float32))

    @pytest.mark.parametrize("accumulate", [False, True])
    def test_index_put_(self, device: str, accumulate: bool) -> None:
        x = lucid.zeros(4, device=device)
        index = (lucid.tensor([1, 1, 3], device=device),)
        values = lucid.tensor([1.0, 2.0, 3.0], device=device)
        if not accumulate:
            index, values = (lucid.tensor([1, 3], device=device),), values[1:]
        assert x.index_put_(index, values, accumulate=accumulate) is x
        want = [0.0, 3.0, 0.0, 3.0] if accumulate else [0.0, 2.0, 0.0, 3.0]
        assert_close(x, np.array(want, dtype=np.float32))

    def test_a_write_through_a_view_reaches_its_base(self) -> None:
        index = lucid.tensor([0, 2])
        for method, args in [
            ("index_add_", (0, index, lucid.ones(2))),
            ("index_copy_", (0, index, lucid.ones(2))),
            ("index_fill_", (0, index, 1.0)),
            ("index_put_", ((index,), lucid.ones(2))),
        ]:
            base = lucid.zeros(2, 4)
            getattr(base[1], method)(*args)
            assert_close(
                base,
                np.array([[0, 0, 0, 0], [1, 0, 1, 0]], dtype=np.float32),
                msg=method,
            )

    def test_a_view_sees_a_write_to_its_base(self) -> None:
        base = lucid.zeros(2, 3)
        row = base[1]
        base.index_fill_(1, lucid.tensor([2]), 4.0)
        assert_close(row, np.array([0.0, 0.0, 4.0], dtype=np.float32))
        lucid.index_put_(
            base, (lucid.tensor([1]), lucid.tensor([0])), lucid.tensor([9.0])
        )
        assert_close(row, np.array([9.0, 0.0, 4.0], dtype=np.float32))


class TestInPlaceAutograd:
    @pytest.mark.parametrize("shape,dim,index", CASES[1:4])
    def test_a_non_leaf_carries_the_gradient(self, device: str, shape, dim: int, index: list[int]) -> None:  # type: ignore[no-untyped-def]
        a = _rng().standard_normal(shape).astype(np.float32)
        unique = _unique(index)
        for method in ("index_add_", "index_copy_", "index_fill_"):
            use = index if method == "index_add_" else unique
            src_np = _source(shape, dim, len(use))
            x = lucid.tensor(a, device=device, requires_grad=True)
            src = lucid.tensor(src_np, device=device, requires_grad=True)
            y = x * 2.0
            idx = lucid.tensor(use, device=device)
            if method == "index_add_":
                y.index_add_(dim, idx, src, alpha=0.5)
                final = _ref_index_add(2 * a, dim, use, src_np, 0.5)
                keep = np.ones_like(a)
            elif method == "index_copy_":
                y.index_copy_(dim, idx, src)
                final = _ref_index_copy(2 * a, dim, use, src_np)
                keep = _ref_index_fill(np.ones_like(a), dim, use, 0.0)
            else:
                y.index_fill_(dim, idx, 0.25)
                final = _ref_index_fill(2 * a, dim, use, 0.25)
                keep = _ref_index_fill(np.ones_like(a), dim, use, 0.0)
            assert y.requires_grad and not y.is_leaf
            (y * y).sum().backward()
            assert_close(y.detach(), final, msg=method)
            # loss = sum(y²): dL/dy = 2y, and y = 2x where the write kept x.
            assert x.grad is not None
            assert_close(x.grad, 4 * final * keep, msg=method)
            if method != "index_fill_":
                scale = 0.5 if method == "index_add_" else 1.0
                want = np.stack(
                    [scale * 2 * final[_along(dim, a.ndim, j)] for j in use],
                    axis=dim % a.ndim,
                )
                assert src.grad is not None
                assert_close(src.grad, want, msg=method)

    def test_a_leaf_that_requires_grad_is_refused(self, device: str) -> None:
        for method, args in [
            (
                "index_add_",
                (0, lucid.tensor([0], device=device), lucid.ones(1, device=device)),
            ),
            (
                "index_copy_",
                (0, lucid.tensor([0], device=device), lucid.ones(1, device=device)),
            ),
            ("index_fill_", (0, lucid.tensor([0], device=device), 1.0)),
            (
                "index_put_",
                ((lucid.tensor([0], device=device),), lucid.ones(1, device=device)),
            ),
        ]:
            p = lucid.zeros(3, device=device, requires_grad=True)
            with pytest.raises(RuntimeError, match="leaf tensor that requires grad"):
                getattr(p, method)(*args)

    def test_a_parameter_written_under_no_grad_stays_a_trainable_leaf(
        self, device: str
    ) -> None:
        p = nn.Parameter(lucid.zeros(3, device=device))
        with lucid.no_grad():
            p.index_add_(
                0, lucid.tensor([0, 2], device=device), lucid.ones(2, device=device)
            )
            p.index_fill_(0, lucid.tensor([1], device=device), 5.0)
        assert p.requires_grad and p.is_leaf and isinstance(p, nn.Parameter)
        assert_close(p.detach(), np.array([1.0, 5.0, 1.0], dtype=np.float32))

    def test_a_tensor_written_from_a_differentiable_source_joins_the_graph(
        self, device: str
    ) -> None:
        buf = lucid.zeros(3, 2, device=device)
        src = lucid.ones(2, 2, device=device, requires_grad=True)
        buf.index_copy_(0, lucid.tensor([0, 2], device=device), src)
        assert buf.requires_grad and not buf.is_leaf
        # Not a leaf, so a later in-place write is allowed.
        buf.index_fill_(1, lucid.tensor([1], device=device), 0.0)
        weights = lucid.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], device=device)
        (buf * weights).sum().backward()
        assert src.grad is not None
        assert_close(src.grad, np.array([[1.0, 0.0], [5.0, 0.0]], dtype=np.float32))

    def test_second_derivatives_pass_through(self, device: str) -> None:
        x = lucid.tensor([1.0, 2.0, 3.0], device=device, requires_grad=True)
        y = x * x
        y.index_add_(0, lucid.tensor([0], device=device), x[:1] * x[:1])
        (g,) = lucid.autograd.grad(y.sum(), [x], create_graph=True)
        assert_close(g, np.array([4.0, 4.0, 6.0], dtype=np.float32))
        (g2,) = lucid.autograd.grad(g.sum(), [x])
        assert_close(g2, np.array([4.0, 2.0, 2.0], dtype=np.float32))

    def test_a_write_through_a_view_carries_the_gradient(self) -> None:
        x = lucid.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True)
        y = x * 1.0
        y[1].index_fill_(0, lucid.tensor([0]), 9.0)
        (y * y).sum().backward()
        assert_close(y.detach(), np.array([[1, 2, 3], [9, 5, 6]], dtype=np.float32))
        assert x.grad is not None
        assert_close(x.grad, np.array([[2, 4, 6], [0, 10, 12]], dtype=np.float32))

    def test_recorded_setitem_and_inplace_bump_old_version(self, device: str) -> None:
        w = lucid.tensor([1.0, 2.0, 3.0], device=device, requires_grad=True)
        x = w * 2.0
        y = x * x
        x[0] = 0.0
        with pytest.raises(RuntimeError, match="version"):
            y.sum().backward()

        w2 = lucid.tensor([1.0, 2.0, 3.0], device=device, requires_grad=True)
        x2 = w2 * 2.0
        x2[0] = 0.0
        x2.sum().backward()
        assert w2.grad is not None
        assert_close(w2.grad, np.array([0.0, 2.0, 2.0], dtype=np.float32))

        w3 = lucid.tensor([1.0, 2.0, 3.0], device=device, requires_grad=True)
        x3 = w3 * 2.0
        y3 = x3 * x3
        x3.masked_fill_(lucid.tensor([True, False, False], device=device), 0.0)
        with pytest.raises(RuntimeError, match="version"):
            y3.sum().backward()


class TestNewTensorMethodsLCD340:
    def test_masked_fill_inplace(self, device: str) -> None:
        x = lucid.tensor([1.0, 2.0, 3.0], device=device, requires_grad=True)
        y = x * 2.0
        mask = lucid.tensor([False, True, False], device=device)
        ret = y.masked_fill_(mask, -1.0)
        assert ret is y
        assert_close(y.detach(), np.array([2.0, -1.0, 6.0], dtype=np.float32))
        y.sum().backward()
        assert x.grad is not None
        assert_close(x.grad, np.array([2.0, 0.0, 2.0], dtype=np.float32))

        v = lucid.tensor(5.0, device=device, requires_grad=True)
        z = lucid.zeros(3, device=device)
        z.masked_fill_(mask, v)
        z.sum().backward()
        assert v.grad is not None and v.grad.item() == 1.0

        with pytest.raises(RuntimeError, match="0-dimensional"):
            lucid.zeros(3, device=device).masked_fill_(
                mask, lucid.ones(1, device=device)
            )

    def test_scatter_and_scatter_add_inplace(self, device: str) -> None:
        x = lucid.zeros(4, device=device)
        idx = lucid.tensor([1, 3], device=device)
        src = lucid.tensor([2.0, 5.0], device=device, requires_grad=True)
        ret = x.scatter_(0, idx, src)
        assert ret is x
        assert_close(x.detach(), np.array([0.0, 2.0, 0.0, 5.0], dtype=np.float32))
        (x * 3.0).sum().backward()
        assert src.grad is not None
        assert_close(src.grad, np.array([3.0, 3.0], dtype=np.float32))

        a = lucid.ones(4, device=device)
        a.scatter_(0, idx, 10.0, reduce="add")
        assert_close(a, np.array([1.0, 11.0, 1.0, 11.0], dtype=np.float32))

        b = lucid.ones(4, device=device, requires_grad=True)
        c = b * 2.0
        s = lucid.tensor([3.0, 4.0], device=device, requires_grad=True)
        ret2 = c.scatter_add_(0, idx, s)
        assert ret2 is c
        assert_close(c.detach(), np.array([2.0, 5.0, 2.0, 6.0], dtype=np.float32))
        c.sum().backward()
        assert b.grad is not None and s.grad is not None
        assert_close(b.grad, np.array([2.0, 2.0, 2.0, 2.0], dtype=np.float32))
        assert_close(s.grad, np.array([1.0, 1.0], dtype=np.float32))

    def test_scatter_reduce_and_inplace(self, device: str) -> None:
        x = lucid.tensor([1.0, 2.0, 3.0], device=device)
        idx = lucid.tensor([0, 0, 2], device=device)
        src = lucid.tensor([4.0, 6.0, 5.0], device=device)
        out = x.scatter_reduce(0, idx, src, "mean", include_self=False)
        assert_close(out, np.array([5.0, 2.0, 5.0], dtype=np.float32))

        y = x.clone()
        ret = y.scatter_reduce_(0, idx, src, "sum", include_self=True)
        assert ret is y
        assert_close(y, np.array([11.0, 2.0, 8.0], dtype=np.float32))

    def test_put_inplace(self, device: str) -> None:
        x = lucid.zeros(2, 2, device=device)
        idx = lucid.tensor([0, 3], device=device)
        src = lucid.tensor([7.0, 9.0], device=device, requires_grad=True)
        ret = x.put_(idx, src)
        assert ret is x
        assert_close(x.detach(), np.array([[7.0, 0.0], [0.0, 9.0]], dtype=np.float32))
        x.sum().backward()
        assert src.grad is not None
        assert_close(src.grad, np.array([1.0, 1.0], dtype=np.float32))

    def test_masked_scatter_and_inplace(self, device: str) -> None:
        x = lucid.zeros(2, 2, device=device)
        mask = lucid.tensor([[True, False], [False, True]], device=device)
        src = lucid.tensor([3.0, 8.0, 9.0], device=device, requires_grad=True)
        out = x.masked_scatter(mask, src)
        assert_close(out.detach(), np.array([[3.0, 0.0], [0.0, 8.0]], dtype=np.float32))

        y = lucid.ones(2, 2, device=device)
        ret = y.masked_scatter_(mask, src)
        assert ret is y
        assert_close(y.detach(), np.array([[3.0, 1.0], [1.0, 8.0]], dtype=np.float32))
        y.sum().backward()
        assert src.grad is not None
        assert_close(src.grad, np.array([1.0, 1.0, 0.0], dtype=np.float32))

    def test_bincount_and_take_along_dim(self, device: str) -> None:
        idx = lucid.tensor([0, 1, 1, 3], dtype=lucid.int64, device=device)
        assert idx.bincount().tolist() == [1, 2, 0, 1]
        w = lucid.tensor([0.5, 1.0, 2.0, 4.0], device=device)
        assert_close(
            idx.bincount(weights=w, minlength=5),
            np.array([0.5, 3.0, 0.0, 4.0, 0.0], dtype=np.float32),
        )

        mat = lucid.tensor([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]], device=device)
        take_idx = lucid.tensor([[2, 0], [1, 2]], dtype=lucid.int64, device=device)
        assert_close(
            mat.take_along_dim(take_idx, dim=1),
            np.array([[30.0, 10.0], [50.0, 60.0]], dtype=np.float32),
        )
