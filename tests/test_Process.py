"""Tests for Process.py."""
import numpy as np
import pytest

import ColorSpace
from Process import MAX_AUTO_BLACK_LEVEL, autoBlackLevel, blackLevelFromPosition, positionFromBlackLevel


def test_black_level_slider_curve() -> None:
    """The slider sets 0 at its start, 15% in the middle and 100% at its end, and its inverse round-trips."""
    assert [blackLevelFromPosition(position) for position in (0, 0.5, 1)] == pytest.approx([0, 0.15, 1], abs=1e-3)
    assert blackLevelFromPosition(positionFromBlackLevel(0.0625)) == pytest.approx(0.0625)


def test_autoBlackLevel() -> None:
    """The stated black level comes first; else the darkest channel's estimate, capped."""
    img = np.full((50, 50, 3), [0.05, 0.06, 0.07], dtype=np.float32)
    stated = ColorSpace.DecodeParams(blackLevel=(64.0, 63.0, 65.0), whiteLevel=1023.0)
    assert autoBlackLevel(img=img, params=stated) == pytest.approx(63/1023)
    assert autoBlackLevel(img=img, params=ColorSpace.DecodeParams(whiteLevel=255.0)) == pytest.approx(0.05)
    assert autoBlackLevel(img=img + 0.5, params=None) == MAX_AUTO_BLACK_LEVEL
