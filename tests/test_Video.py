"""Tests for Video.py."""
import pathlib
from unittest import mock

import cv2
import numpy as np
import pytest

import ColorSpace
from Constants import AUTO, DemosaicAlgorithm, DemosaicParams, MosaicPattern
from FrameSource import CameraRawSource, FitsSource, ImageSequenceSource, LinearizeParams, VideoSource
from tests.inputs import (
    INPUTS_DIR,
    MP4_YUV8,
    PNG8_MIXED_SIZE_SEQUENCE,
    PNG8_MIXED_SIZE_SEQUENCE_DIMENSIONS,
    PNG8_SEQUENCE,
    PNG8_SEQUENCE_DIMENSIONS,
)
from Video import Video, loadFrames


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16])
def test_single_image_getFrame_scales_to_0_1(tmp_path: pathlib.Path, dtype: type) -> None:
    """An untagged image's frame is float32 0-1, decoded with the sRGB curve over its bit depth's full range."""
    maxVal = np.iinfo(dtype).max
    image = np.zeros((4, 4, 3), dtype=dtype)
    image[0, 0] = maxVal
    image[1, 1] = maxVal//3
    path = str(tmp_path / "image.png")
    cv2.imwrite(path, image)

    frame = Video(ImageSequenceSource([path])).getFrame(index=0)

    assert frame.dtype == np.float32
    assert frame.min() == 0
    assert frame.max() == 1
    assert np.allclose(frame, ColorSpace.transferToLinear(x=image/maxVal, transfer=ColorSpace.SRGB_CURVE), atol=1e-6)


def test_loadFrames_leaves_out_unloadable_frames() -> None:
    """Frames of another size, and indices past the last frame, are left out."""
    frames, sharps, deltas, mismatch = loadFrames(
        source=ImageSequenceSource(PNG8_MIXED_SIZE_SEQUENCE),
        indices=range(0, 6),
        frameSize=PNG8_MIXED_SIZE_SEQUENCE_DIMENSIONS,
        progress=mock.Mock(),
    )

    assert frames == [0, 2, 3]
    assert len(sharps) == len(deltas) == len(frames)


def test_loadFrames_uses_frame_before_chunk() -> None:
    """A chunk's first delta is taken from the frame before the chunk, as it is when loading all frames."""
    allFrames = loadFrames(
        source=ImageSequenceSource(PNG8_SEQUENCE),
        indices=range(0, 4),
        frameSize=PNG8_SEQUENCE_DIMENSIONS,
        progress=mock.Mock(),
    )

    frames, sharps, deltas, mismatch = loadFrames(
        source=ImageSequenceSource(PNG8_SEQUENCE),
        indices=range(3, 4),
        frameSize=PNG8_SEQUENCE_DIMENSIONS,
        progress=mock.Mock(),
    )

    assert frames == [3]
    assert sharps == allFrames[1][-1:]
    assert np.array_equal(deltas[0], allFrames[2][-1])


def test_measurementCopy_is_gamma_2_2_per_channel() -> None:
    """The measurement copy is a plain gamma 2.2 of each channel, clipped to 0-1, with no primaries conversion."""
    image = np.array([[[0.0, 0.18, 1.0], [-0.5, 2.0, 0.5]]], dtype=np.float64)

    copy = Video.measurementCopy(image)

    assert copy.dtype == np.float32
    assert np.allclose(copy, np.array([[[0.0, 0.18**(1/2.2), 1.0], [0.0, 1.0, 0.5**(1/2.2)]]]), atol=1e-6)
    assert np.array_equal(Video.encodedCopy(image), np.around(copy*255).astype(np.uint8))


COLOR = INPUTS_DIR/"color"


def test_loadFrames_flags_mismatched_color_tags() -> None:
    """Images tagged differently from the first are flagged, since every image is decoded like the first."""
    source = ImageSequenceSource([str(COLOR/"png16_srgb_linear.png"), str(COLOR/"png16_srgb_srgb.png")])
    mismatch = loadFrames(
        source=source,
        indices=range(0, 2),
        frameSize=(192, 160),
        progress=mock.Mock(),
    )[3]
    assert mismatch


def test_jupiter_is_untagged_full_range_bt709() -> None:
    """Jupiter.mp4 (yuvj420p, only its range tagged) is BT.709 and full range."""
    first = VideoSource(MP4_YUV8).firstFrameFacts
    assert first.linearizeParams == LinearizeParams(transfer=("H273", 1))
    assert first.workingSpace == ColorSpace.SRGB


def test_video_frame_times_match_opencv() -> None:
    """PyAV's presentation times address the same frames as cv2's millisecond positions did."""
    source = VideoSource(MP4_YUV8)
    times = [float(timestamp*source.stream.time_base*1000) for timestamp in source.frameTimestamps()]
    vidcap = cv2.VideoCapture(MP4_YUV8)
    cvTimes = []
    while(vidcap.read()[0]):
        cvTimes.append(vidcap.get(cv2.CAP_PROP_POS_MSEC))
    assert len(times) == len(cvTimes) == 60
    assert np.allclose(times, cvTimes, atol=1e-6)


def test_raw_dng_is_balanced_to_d65_reference() -> None:
    """The Pixel 8a crop decodes with its black subtracted, and its neutral gravel has the raw ratios times daylight."""
    import rawpy
    file = str(INPUTS_DIR/"raw"/"pixel_crop_1.dng")
    first = CameraRawSource(file).firstFrameFacts
    assert first.linearizeParams.blackLevel == (64.0, 64.0, 64.0)
    assert first.linearizeParams.whiteLevel == 1023.0
    assert np.allclose(first.workingSpace.toXYZ @ np.ones(3), ColorSpace.D65_XYZ, atol=1e-6)
    linear = Video(CameraRawSource(file)).getFrame(index=0)
    assert np.percentile(linear, 0.1) == pytest.approx(0, abs=0.01)
    region = linear[256:336, 224:320].reshape(-1, 3).mean(axis=0)[::-1]
    with rawpy.imread(file) as raw:
        gains = np.array(raw.camera_whitebalance[:3])/raw.camera_whitebalance[1]
        daylight = np.array(raw.daylight_whitebalance[:3])/raw.daylight_whitebalance[1]
    ratios = region/region[1]
    assert np.allclose(ratios, daylight/gains, rtol=0.1)


def test_fits_bayer_pattern_and_levels() -> None:
    """The Bayer FITS states GBRG (RGGB shifted one row) and keeps one scale for both frames."""
    first = FitsSource(str(INPUTS_DIR/"fits"/"bayer_1.fits")).readStored(0)
    second = FitsSource(str(INPUTS_DIR/"fits"/"bayer_2.fits")).readStored(0)
    expected = LinearizeParams(bayerPattern=MosaicPattern.GBRG, whiteLevel=60000.0)
    assert first.linearizeParams == second.linearizeParams == expected
    video = Video(FitsSource([str(INPUTS_DIR/"fits"/"bayer_1.fits"), str(INPUTS_DIR/"fits"/"bayer_2.fits")]))
    assert video.source.setDemosaicChoices(mode=AUTO, pattern=AUTO, algorithm=AUTO) == DemosaicParams(
        MosaicPattern.GBRG,
        DemosaicAlgorithm.EDGE_AWARE,
    )
    frames = [video.getFrame(index=index) for index in range(2)]
    assert frames[0].min() == pytest.approx(2000/60000, abs=1e-3)
    assert frames[1].min() == pytest.approx(2000/60000, abs=1e-3)
    assert (frames[1].max() - 2000/60000) == pytest.approx(0.6*(frames[0].max() - 2000/60000), rel=0.02)
