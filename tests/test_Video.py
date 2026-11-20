"""Tests for Video.py."""
import pathlib
from unittest import mock

import cv2
import numpy as np
import pytest

import ColorSpace
from Constants import DemosaicAlgorithm, DemosaicIntent, DemosaicMode, MosaicPattern
from FrameSource import CameraRawSource, DecodeParams, FitsSource, FrameSource, ImageSequenceSource, VideoSource
from tests.inputs import AVI8_RGGB, INPUTS_DIR, MP4_YUV8, PNG8_MIXED_SIZE_SEQUENCE, PNG8_MIXED_SIZE_SEQUENCE_DIMENSIONS
from Globals import g
from Video import Video, calculateSharpness, loadFrames


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


@pytest.mark.parametrize("mode", [DemosaicMode.AUTO, DemosaicMode.OFF])
@pytest.mark.parametrize("name", ["image.png", "image.jpg"])
def test_single_image_getFrame_matches_readFrame(tmp_path: pathlib.Path, mode: DemosaicMode, name: str) -> None:
    """A color image is never demosaiced, whatever the demosaic mode: it reads as its frame with sRGB's curve."""
    rng = np.random.default_rng(seed=0)
    path = str(tmp_path / name)
    cv2.imwrite(path, rng.integers(low=0, high=256, size=(24, 32, 3), dtype=np.uint8))
    source = FrameSource.open(path)
    expected = ColorSpace.transferToLinear(x=source.readStored(0).image/255, transfer=ColorSpace.SRGB_CURVE)

    video = Video(source)
    video.setDemosaic(mode=mode, pattern=None, algorithm=None)

    frame = video.getFrame(index=0)

    assert np.allclose(frame, expected, atol=1e-6)


@pytest.mark.parametrize("mode, pattern, algorithm, expected",
                         [(DemosaicMode.AUTO, None, None, DemosaicIntent(MosaicPattern.RGGB, DemosaicAlgorithm.EDGE_AWARE)),
                          (DemosaicMode.OFF, None, None, None),
                          (DemosaicMode.ON, MosaicPattern.GBRG, None,
                           DemosaicIntent(MosaicPattern.GBRG, DemosaicAlgorithm.EDGE_AWARE)),
                          (DemosaicMode.ON, MosaicPattern.BGGR, DemosaicAlgorithm.VNG,
                           DemosaicIntent(MosaicPattern.BGGR, DemosaicAlgorithm.VNG))])
def test_setDemosaic_resolves_the_choice_through_the_source(
    mode: DemosaicMode,
    pattern: MosaicPattern | None,
    algorithm: DemosaicAlgorithm | None,
    expected: DemosaicIntent | None,
) -> None:
    """The demosaic choice resolves as Auto, Off, or On with its pattern and algorithm, and getFrame uses it."""
    video = Video(FrameSource.open(AVI8_RGGB))

    assert video.setDemosaic(mode=mode, pattern=pattern, algorithm=algorithm) == expected
    video.getFrame(index=0)

    assert video.intent == expected


@pytest.mark.parametrize("stored, expected",
                         [("GBRG", MosaicPattern.GBRG), (5, None), ("", None), ("LibRaw", None)])
def test_rememberedPattern_reads_a_pattern_name_and_ignores_the_rest(stored: object, expected: MosaicPattern | None) -> None:
    """An integer stored by the old color mode preference is ignored."""
    with mock.patch.object(g, "ui") as ui:
        ui.preferences.get.return_value = stored

        assert Video.rememberedPattern() == expected


class FakeProgress:
    """Records the calls loadFrames makes to its progress counter."""

    def __init__(self) -> None:
        self.counts: list[tuple[int, int]] = []
        self.extras = 0

    def count(self, i: int, size: int) -> None:
        """Records a counted frame.

        Args:
            i: The frame's position in the chunk.
            size: The chunk's size.
        """
        self.counts.append((i, size))

    def countExtra(self) -> None:
        """Records the end of the chunk."""
        self.extras += 1


def test_loadFrames_skips_other_sizes() -> None:
    """Frames whose size differs are skipped, and every index is counted."""
    progress = FakeProgress()

    frames, sharps, deltas, mismatch = loadFrames(source=ImageSequenceSource(PNG8_MIXED_SIZE_SEQUENCE),
                                        indices=range(0, 4),
                                        frameSize=PNG8_MIXED_SIZE_SEQUENCE_DIMENSIONS,
                                        intent=None,
                                        progress=progress)

    assert frames == [0, 2, 3]
    assert len(sharps) == 3
    assert len(deltas) == 3
    assert np.array_equal(deltas[0], np.identity(3))
    assert progress.counts == [(i, 4) for i in range(4)]
    assert progress.extras == 1


def test_loadFrames_uses_frame_before_chunk() -> None:
    """A chunk's first delta is taken from the frame before the chunk, as it is when loading all frames."""
    allFrames = loadFrames(source=ImageSequenceSource(PNG8_MIXED_SIZE_SEQUENCE),
                           indices=range(0, 4),
                           frameSize=PNG8_MIXED_SIZE_SEQUENCE_DIMENSIONS,
                           intent=None,
                           progress=FakeProgress())

    frames, sharps, deltas, mismatch = loadFrames(source=ImageSequenceSource(PNG8_MIXED_SIZE_SEQUENCE),
                                        indices=range(3, 4),
                                        frameSize=PNG8_MIXED_SIZE_SEQUENCE_DIMENSIONS,
                                        intent=None,
                                        progress=FakeProgress())

    assert frames == [3]
    assert sharps == allFrames[1][-1:]
    assert np.array_equal(deltas[0], allFrames[2][-1])


def test_loadFrames_past_end() -> None:
    """Indices past the last frame are counted but load nothing."""
    progress = FakeProgress()

    frames, sharps, deltas, mismatch = loadFrames(source=ImageSequenceSource(PNG8_MIXED_SIZE_SEQUENCE),
                                        indices=range(3, 7),
                                        frameSize=PNG8_MIXED_SIZE_SEQUENCE_DIMENSIONS,
                                        intent=None,
                                        progress=progress)

    assert frames == [3]
    assert len(sharps) == 1
    assert len(deltas) == 1
    assert len(progress.counts) == 4


def test_loadFrames_measures_16bit_frames_at_8_bits(tmp_path: pathlib.Path) -> None:
    """A 16-bit sequence loads with the same sharpness and deltas as the same values saved at 8 bits."""
    paths16 = []
    for index, path in enumerate(PNG8_MIXED_SIZE_SEQUENCE):
        image = cv2.imread(path, cv2.IMREAD_UNCHANGED).astype(np.uint16)*257
        path16 = str(tmp_path / f"seq16_{index}.png")
        cv2.imwrite(path16, image)
        paths16.append(path16)

    loaded8 = loadFrames(source=ImageSequenceSource(PNG8_MIXED_SIZE_SEQUENCE),
                         indices=range(0, 4),
                         frameSize=PNG8_MIXED_SIZE_SEQUENCE_DIMENSIONS,
                         intent=None,
                         progress=FakeProgress())
    loaded16 = loadFrames(source=ImageSequenceSource(paths16),
                          indices=range(0, 4),
                          frameSize=PNG8_MIXED_SIZE_SEQUENCE_DIMENSIONS,
                          intent=None,
                          progress=FakeProgress())

    assert loaded16[0] == loaded8[0]
    assert loaded16[1] == loaded8[1]
    for delta16, delta8 in zip(loaded16[2], loaded8[2]):
        assert np.array_equal(delta16, delta8)


@mock.patch.object(FrameSource, "demosaic", wraps=FrameSource.demosaic)
def test_loadFrames_demosaics_once(demosaic: mock.MagicMock) -> None:
    """The intent is applied once to each frame read, including the frame before the chunk."""
    intent = DemosaicIntent(MosaicPattern.GRBG, DemosaicAlgorithm.EDGE_AWARE)

    loadFrames(source=ImageSequenceSource(PNG8_MIXED_SIZE_SEQUENCE),
               indices=range(3, 4),
               frameSize=PNG8_MIXED_SIZE_SEQUENCE_DIMENSIONS,
               intent=intent,
               progress=FakeProgress())

    assert [call.kwargs["intent"] for call in demosaic.call_args_list] == [intent]*2


@pytest.mark.parametrize("intent", [None, DemosaicIntent(MosaicPattern.GRBG, DemosaicAlgorithm.EDGE_AWARE)])
def test_loadFrames_measures_sharpness_after_demosaic(intent: DemosaicIntent | None) -> None:
    """Sharpness is measured on the gamma 2.2 frame after the intent is applied."""
    source = ImageSequenceSource(PNG8_MIXED_SIZE_SEQUENCE)
    frames, sharps, deltas, mismatch = loadFrames(source=source,
                                                  indices=range(0, 3),
                                                  frameSize=PNG8_MIXED_SIZE_SEQUENCE_DIMENSIONS,
                                                  intent=intent,
                                                  progress=FakeProgress())

    for index, sharp in zip(frames, sharps):
        assert sharp == calculateSharpness(Video.encodedCopy(source.readFrame(index=index, intent=intent).image))


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
    mismatch = loadFrames(source=source,
                          indices=range(0, 2),
                          frameSize=(192, 160),
                          intent=None,
                          progress=FakeProgress())[3]
    assert mismatch


def test_jupiter_is_untagged_full_range_bt709() -> None:
    """Jupiter.mp4 (yuvj420p, only its range tagged) is BT.709 and full range."""
    first = VideoSource(MP4_YUV8).firstFrame()
    assert first.params == DecodeParams(transfer=("H273", 1))
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
    first = CameraRawSource(file).firstFrame()
    assert first.params.blackLevel == (64.0, 64.0, 64.0)
    assert first.params.whiteLevel == 1023.0
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
    assert first.params == second.params == DecodeParams(bayerPattern=MosaicPattern.GBRG, whiteLevel=60000.0)
    video = Video(FitsSource([str(INPUTS_DIR/"fits"/"bayer_1.fits"), str(INPUTS_DIR/"fits"/"bayer_2.fits")]))
    assert video.setDemosaic(mode=DemosaicMode.AUTO, pattern=None, algorithm=None) == DemosaicIntent(
        MosaicPattern.GBRG,
        DemosaicAlgorithm.EDGE_AWARE,
    )
    frames = [video.getFrame(index=index) for index in range(2)]
    assert frames[0].min() == pytest.approx(2000/60000, abs=1e-3)
    assert frames[1].min() == pytest.approx(2000/60000, abs=1e-3)
    assert (frames[1].max() - 2000/60000) == pytest.approx(0.6*(frames[0].max() - 2000/60000), rel=0.02)
