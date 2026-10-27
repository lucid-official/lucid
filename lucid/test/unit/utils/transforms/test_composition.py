"""Composition transforms (B7) — OneOf / SomeOf / Sequential / OneOrOther."""

import pytest

import lucid
import lucid.utils.transforms as T


class TestComposition:
    def test_oneof_shape(self) -> None:
        lucid.manual_seed(0)
        out = T.OneOf([T.HorizontalFlip(p=1.0), T.VerticalFlip(p=1.0)], p=1.0)(
            lucid.rand(3, 16, 16)
        )
        assert tuple(out.shape) == (3, 16, 16)

    def test_oneof_forces_selected_child(self) -> None:
        # single child with p=0 is still applied (its p is only a weight)
        x = lucid.rand(3, 8, 8)
        out = T.OneOf([T.InvertImg(p=0.0)], p=1.0)(x)
        assert float((out - (1.0 - x)).abs().max().item()) < 1e-6

    def test_oneof_p0_passthrough(self) -> None:
        x = lucid.rand(3, 8, 8)
        out = T.OneOf([T.InvertImg(p=1.0)], p=0.0)(x)
        assert float((out - x).abs().max().item()) < 1e-6

    def test_someof_applies_n(self) -> None:
        lucid.manual_seed(1)
        out = T.SomeOf([T.Blur(p=1.0), T.InvertImg(p=1.0), T.ToGray(p=1.0)], n=2, p=1.0)(
            lucid.rand(3, 16, 16)
        )
        assert tuple(out.shape) == (3, 16, 16)

    def test_sequential_applies_all(self) -> None:
        x = lucid.rand(3, 8, 8)
        # hflip then hflip = identity
        out = T.Sequential([T.HorizontalFlip(p=1.0), T.HorizontalFlip(p=1.0)], p=1.0)(x)
        assert float((out - x).abs().max().item()) < 1e-6

    def test_oneorother(self) -> None:
        x = lucid.rand(3, 8, 8)
        # switch p=1 -> always first
        out = T.OneOrOther(T.InvertImg(p=1.0), T.ToGray(p=1.0), p=1.0)(x)
        assert float((out - (1.0 - x)).abs().max().item()) < 1e-6
        # switch p=0 -> always second
        out2 = T.OneOrOther(T.InvertImg(p=1.0), T.ToGray(p=1.0), p=0.0)(x)
        assert float((out2 - T.ToGray(p=1.0)(x)).abs().max().item()) < 1e-6

    def test_oneof_multitarget(self) -> None:
        s = {
            "image": T.Image(lucid.rand(3, 20, 20)),
            "boxes": T.BoundingBoxes(lucid.tensor([[2.0, 2.0, 10.0, 10.0]]), "xyxy", (20, 20)),
        }
        out = T.OneOf([T.HorizontalFlip(p=1.0)], p=1.0)(s)
        assert out["boxes"].canvas_size == (20, 20)

    def test_seeded(self) -> None:
        lucid.manual_seed(9)
        a = T.OneOf([T.Blur(p=1.0), T.InvertImg(p=1.0)], p=1.0)(lucid.rand(3, 16, 16))
        lucid.manual_seed(9)
        b = T.OneOf([T.Blur(p=1.0), T.InvertImg(p=1.0)], p=1.0)(lucid.rand(3, 16, 16))
        assert float((a - b).abs().max().item()) == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
