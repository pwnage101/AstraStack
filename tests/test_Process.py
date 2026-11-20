"""Tests for Process.py."""
import numpy as np
import pytest

from Process import MAX_AUTO_BLACK_LEVEL, autoBlackLevel, blackLevelFromPosition, positionFromBlackLevel


def test_black_level_slider_curve() -> None:
    """The slider sets 0 at its start, 15% in the middle and 100% at its end, and its inverse round-trips."""
    assert [blackLevelFromPosition(position) for position in (0, 0.5, 1)] == pytest.approx([0, 0.15, 1], abs=1e-3)
    assert blackLevelFromPosition(positionFromBlackLevel(0.0625)) == pytest.approx(0.0625)


def test_autoBlackLevel() -> None:
    """The estimate is the darkest channel's, capped."""
    img = np.full((50, 50, 3), [0.05, 0.06, 0.07], dtype=np.float32)
    assert autoBlackLevel(img=img) == pytest.approx(0.05)
    assert autoBlackLevel(img=img + 0.5) == MAX_AUTO_BLACK_LEVEL
