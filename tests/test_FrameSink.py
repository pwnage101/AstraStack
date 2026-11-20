"""Tests for FrameSink.py."""
import pathlib

import numpy as np

import ColorSpace
from FrameSink import FrameSink
from FrameSource import ImageSequenceSource, linearize


def test_png_reads_back_as_srgb(tmp_path: pathlib.Path) -> None:
    """A saved PNG is tagged sRGB, and reads back as the same colors."""
    path = str(tmp_path/"saved.png")
    img = np.array([[[0.0, 0.5, 1.0], [0.2, 0.2, 0.2]]], dtype=np.float32)

    FrameSink(path=path, bits=16).write(img=img, workingSpace=ColorSpace.SRGB)

    frame = ImageSequenceSource(path).readStored(0)
    assert frame.workingSpace == ColorSpace.SRGB
    linear = linearize(img=frame.image[:, :, ::-1], linearizeParams=frame.linearizeParams)
    assert np.allclose(linear, img, atol=1e-4)
