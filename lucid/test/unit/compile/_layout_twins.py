"""The rewritten layouts and explicit low-rank versions of the same operations."""

from collections.abc import Callable
import lucid
import lucid.nn.functional as F
from lucid.test.unit.compile._rank_twins import RankTwin, Side, Shape, _filled


def _window_partition(x: lucid.Tensor) -> lucid.Tensor:
    b, h, w, c = x.shape
    ws = 7
    t = x.reshape(b, h // ws, ws, w // ws, ws, c)
    return t.permute(0, 1, 3, 2, 4, 5)


def _operand(device: str, shape: Shape, phase: float = 0.0) -> lucid.Tensor:
    return _filled(shape, phase).to(device)


Case = tuple[Callable[..., lucid.Tensor], tuple[lucid.Tensor, ...]]


# Each case reads its rank-5+ result through a reshape, as a model does (Swin merges the
# window axes, attention merges heads): a reduction reads the recorded rank-5+ result by
# itself, which is the reduction rule's to fix (LCD-920e).
def _window_cases(device: str) -> tuple[Case, Case]:
    x = _operand(device, (256, 56, 56, 4))
    return (
        (lambda t: _window_partition(t).reshape(256 * 64, 49, 4).sum(dim=1), (x,)),
        (
            lambda t: (
                t.reshape(2048, 7, 8, 28).permute(0, 2, 1, 3).reshape(256 * 64, 49, 4).sum(dim=1)
            ),
            (x,),
        ),
    )


def _narrow_cases(device: str) -> tuple[Case, Case]:
    # Swap the last two axes of a rank-5 tensor whose last axis is 2 wide: 250 ms unfixed.
    x = _operand(device, (65536 * 4 * 2 * 8 * 2,))
    return (
        (
            lambda t: (
                t.reshape(65536, 4, 2, 8, 2)
                .permute(0, 1, 2, 4, 3)
                .reshape(65536 * 8 * 2, 8)
                .sum(dim=1)
            ),
            (x,),
        ),
        (
            lambda t: (
                t.reshape(65536 * 8, 8, 2).permute(0, 2, 1).reshape(65536 * 8 * 2, 8).sum(dim=1)
            ),
            (x,),
        ),
    )


def _softmax_cases(device: str) -> tuple[Case, Case]:
    def fn(t: lucid.Tensor) -> lucid.Tensor:
        return F.softmax(t, dim=-1).reshape(1024 * 8 * 4 * 16, 16).sum(dim=-1)

    return (fn, (_operand(device, (1024, 8, 4, 16, 16)),)), (
        fn,
        (_operand(device, (1024 * 8 * 4, 16, 16)),),
    )


def _matmul_cases(device: str) -> tuple[Case, Case]:
    def fn(a: lucid.Tensor, b: lucid.Tensor) -> lucid.Tensor:
        return (a @ b).reshape(1024 * 8 * 16, 16).sum(dim=-1)

    return (
        (fn, (_operand(device, (1024, 4, 2, 16, 32)), _operand(device, (1024, 4, 2, 32, 16), 1.0))),
        (fn, (_operand(device, (1024 * 8, 16, 32)), _operand(device, (1024 * 8, 32, 16), 1.0))),
    )


#: Each row: the rank-5+ graph and the same elements and the same work at rank <= 4.
LAYOUT_CASES: dict[str, Callable[[str], tuple[Case, Case]]] = {
    "window_partition": _window_cases,
    "narrow_last_axis_permute": _narrow_cases,
    "softmax_lead_1024": _softmax_cases,
    "matmul_lead_1024": _matmul_cases,
}


def layout_twin(name: str, device: str) -> RankTwin:
    wide, twin = LAYOUT_CASES[name](device)
    return RankTwin(name, Side(*wide), Side(*twin))
