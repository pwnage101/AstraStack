"""Tests for Video.py."""
import pathlib
from unittest import mock

import cv2
import numpy as np
import pytest

import ColorSpace
from FrameSource import FrameSource, ImageSequenceSource, VideoSource
from tests.inputs import INPUTS_DIR, MP4_YUV8, PNG8_MIXED_SIZE_SEQUENCE, PNG8_MIXED_SIZE_SEQUENCE_DIMENSIONS
from Video import Video, calculateSharpness, loadFrames


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16])
def test_single_image_getFrame_scales_to_0_1(tmp_path: pathlib.Path, dtype: type) -> None:
    """A single image's frame is float32 on a 0-1 scale, using the full range of the input's bit depth."""
    maxVal = np.iinfo(dtype).max
    image = np.zeros((4, 4, 3), dtype=dtype)
    image[0, 0] = maxVal
    image[1, 1] = maxVal//3
    path = str(tmp_path / "image.png")
    cv2.imwrite(path, image)

    frame = Video(ImageSequenceSource([path])).getFrame(index=0,
                                                        colorMode=Video.COLOR_RGB)

    assert frame.dtype == np.float32
    assert frame.min() == 0
    assert frame.max() == 1
    assert np.array_equal(np.around(frame*maxVal).astype(dtype), image)


@pytest.mark.parametrize("colorMode", [Video.COLOR_AUTO, Video.COLOR_RGB, Video.COLOR_GRAYSCALE, Video.COLOR_RGGB])
@pytest.mark.parametrize("name", ["image.png", "image.jpg"])
def test_single_image_getFrame_matches_readFrame(tmp_path: pathlib.Path, colorMode: int, name: str) -> None:
    """An untagged single image reads as its decoded frame with the color mode applied, unchanged."""
    rng = np.random.default_rng(seed=0)
    path = str(tmp_path / name)
    cv2.imwrite(path, rng.integers(low=0, high=256, size=(24, 32, 3), dtype=np.uint8))
    image = ImageSequenceSource(path).readFrame(0).image
    expectedMode = Video.COLOR_RGB if colorMode == Video.COLOR_AUTO else colorMode
    expected = Video._bgrFromSourceFrame(
        image=image,
        colorMode=expectedMode,
        toDtype=np.float32,
    )

    frame = Video(FrameSource.open(path)).getFrame(index=0,
                                                   colorMode=colorMode)

    assert np.array_equal(frame, expected)


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
                                        colorMode=Video.COLOR_RGB,
                                        description=ImageSequenceSource(PNG8_MIXED_SIZE_SEQUENCE).description(),
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
                           colorMode=Video.COLOR_RGB,
                           description=ImageSequenceSource(PNG8_MIXED_SIZE_SEQUENCE).description(),
                           progress=FakeProgress())

    frames, sharps, deltas, mismatch = loadFrames(source=ImageSequenceSource(PNG8_MIXED_SIZE_SEQUENCE),
                                        indices=range(3, 4),
                                        frameSize=PNG8_MIXED_SIZE_SEQUENCE_DIMENSIONS,
                                        colorMode=Video.COLOR_RGB,
                                        description=ImageSequenceSource(PNG8_MIXED_SIZE_SEQUENCE).description(),
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
                                        colorMode=Video.COLOR_RGB,
                                        description=ImageSequenceSource(PNG8_MIXED_SIZE_SEQUENCE).description(),
                                        progress=progress)

    assert frames == [3]
    assert len(sharps) == 1
    assert len(deltas) == 1
    assert len(progress.counts) == 4


def test_bgrFromSourceFrame_rejects_auto() -> None:
    """COLOR_AUTO must be resolved before converting a frame."""
    with pytest.raises(ValueError):
        Video._bgrFromSourceFrame(image=np.zeros((4, 4, 3), dtype=np.uint8), colorMode=Video.COLOR_AUTO)


def test_guessColorMode_guesses_from_first_frame() -> None:
    """Without a frame given, Auto is guessed from the first frame's pixels."""
    assert Video(ImageSequenceSource(PNG8_MIXED_SIZE_SEQUENCE)).guessColorMode() == Video.COLOR_RGB


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
                         colorMode=Video.COLOR_RGB,
                         description=ImageSequenceSource(PNG8_MIXED_SIZE_SEQUENCE).description(),
                         progress=FakeProgress())
    loaded16 = loadFrames(source=ImageSequenceSource(paths16),
                          indices=range(0, 4),
                          frameSize=PNG8_MIXED_SIZE_SEQUENCE_DIMENSIONS,
                          colorMode=Video.COLOR_RGB,
                          # The copies lose the sRGB tag, so describe them like the originals.
                          description=ImageSequenceSource(PNG8_MIXED_SIZE_SEQUENCE).description(),
                          progress=FakeProgress())

    assert loaded16[0] == loaded8[0]
    assert loaded16[1] == loaded8[1]
    for delta16, delta8 in zip(loaded16[2], loaded8[2]):
        assert np.array_equal(delta16, delta8)


@mock.patch.object(Video, "_applyColorMode", wraps=Video._applyColorMode)
def test_loadFrames_applies_color_mode_once(applyColorMode: mock.MagicMock) -> None:
    """The color mode is applied once to each frame read, including the frame before the chunk."""
    loadFrames(source=ImageSequenceSource(PNG8_MIXED_SIZE_SEQUENCE),
               indices=range(3, 4),
               frameSize=PNG8_MIXED_SIZE_SEQUENCE_DIMENSIONS,
               colorMode=Video.COLOR_GRBG,
               description=ImageSequenceSource(PNG8_MIXED_SIZE_SEQUENCE).description(),
               progress=FakeProgress())

    assert [call.kwargs["colorMode"] for call in applyColorMode.call_args_list] == [Video.COLOR_GRBG]*2


@pytest.mark.parametrize("colorMode", [Video.COLOR_RGB, Video.COLOR_GRAYSCALE, Video.COLOR_GRBG])
def test_loadFrames_measures_sharpness_after_color_mode(colorMode: int) -> None:
    """Sharpness is measured on the sRGB-encoded frame after the color mode is applied."""
    frames, sharps, deltas, mismatch = loadFrames(source=ImageSequenceSource(PNG8_MIXED_SIZE_SEQUENCE),
                                        indices=range(0, 3),
                                        frameSize=PNG8_MIXED_SIZE_SEQUENCE_DIMENSIONS,
                                        colorMode=colorMode,
                                        description=ImageSequenceSource(PNG8_MIXED_SIZE_SEQUENCE).description(),
                                        progress=FakeProgress())

    video = Video(ImageSequenceSource(PNG8_MIXED_SIZE_SEQUENCE))
    for index, sharp in zip(frames, sharps):
        linear = video.getFrame(index=index,
                                colorMode=colorMode)
        assert sharp == calculateSharpness(Video.encodedCopy(linear, video.workingSpace))


COLOR = INPUTS_DIR/"color"

# The user's real camera files, used when present
LOCAL_RAWS = {"ARW": "/home/sankey/photos/2022-09-21_pq_canyon/raw/DSC02636.ARW",
              "RAF": "/home/sankey/photos/2024-10-26_boston_trip_3/raw/DSCF5293.RAF",
              "DNG": "/home/sankey/photos/2026-07-10_san_diego_pq_canyon/raw/IMG_20260710_182014_0.dng"}


@pytest.mark.parametrize("file, transfer, workingSpace",
                         [("linear.png", None, ColorSpace.UNTAGGED),
                          ("gama100000.png", None, ColorSpace.UNTAGGED),
                          ("gama100000_chrm_srgb.png", None, ColorSpace.SRGB),
                          ("srgb_chunk.png", ("sRGB", None), ColorSpace.SRGB),
                          ("gama45455.png", ("sRGB", None), ColorSpace.SRGB),
                          ("cicp.png", ("sRGB", None), ColorSpace.SRGB),
                          ("exif_srgb.jpg", ("sRGB", None), ColorSpace.SRGB),
                          ("p3.png", "ICC", ColorSpace.named("Display P3")),
                          ("p3.jpg", "ICC", ColorSpace.named("Display P3")),
                          ("gray_icc.png", "ICC", ColorSpace.UNTAGGED),
                          ("lut_srgb.png", ("Gamma", ColorSpace.LUT_GAMMA), ColorSpace.named("ITU-R BT.2020"))])
def test_describe_fixtures(file: str, transfer: object, workingSpace: ColorSpace.WorkingSpace) -> None:
    """Each image fixture is described by its tags; untagged ones have no transfer and no primaries."""
    image, params, space = Video.decodeImage(str(COLOR/file))
    if(transfer == "ICC"):
        assert params.transfer[0] == "ICC"
    else:
        assert params.transfer == transfer
    assert space == workingSpace
    if(file == "lut_srgb.png"):
        assert params.icc is not None and params.lut is not None


@pytest.mark.parametrize("file", ["srgb_chunk.png", "gama45455.png", "cicp.png", "gama100000.png"])
def test_tagged_encodings_linearize_alike(file: str) -> None:
    """Every encoding of the linear source decodes to the same linear values."""
    reference = Video(ImageSequenceSource(str(COLOR/"linear.png"))).getFrame(index=0,
                                                                             colorMode=Video.COLOR_RGB)
    frame = Video(ImageSequenceSource(str(COLOR/file))).getFrame(index=0,
                                                                 colorMode=Video.COLOR_RGB)
    assert np.abs(frame - reference).max() < 2e-5


@pytest.mark.parametrize("file, transfer, matrix, colorRange, primaries",
                         [("bt709.mp4", ("BT.709", None), "BT.709", "full", ColorSpace.SRGB),
                          ("bt601.mp4", ("BT.601", None), "BT.601", "full", ColorSpace.named("SMPTE C")),
                          ("srgb.mp4", ("sRGB", None), "BT.709", "full", ColorSpace.SRGB),
                          ("bt709_unspecified_trc.mp4", None, "BT.709", "full", ColorSpace.SRGB),
                          ("limited_range.mp4", None, "BT.709", "limited", ColorSpace.UNTAGGED),
                          ("rgb.avi", None, None, "full", ColorSpace.UNTAGGED),
                          ("gray.avi", None, None, "full", ColorSpace.UNTAGGED)])
def test_video_tags(file: str, transfer: object, matrix: str | None, colorRange: str, primaries: object) -> None:
    """Video streams are described from their color tags."""
    params, workingSpace = VideoSource(str(COLOR/file)).description()
    assert params == ColorSpace.DecodeParams(transfer=transfer, yuvMatrix=matrix, range=colorRange)
    assert workingSpace == primaries


def test_jupiter_is_untagged_full_range_bt709() -> None:
    """Jupiter.mp4 (yuvj420p, only its range tagged) uses BT.709, full range and no curve."""
    params, workingSpace = VideoSource(MP4_YUV8).description()
    assert params == ColorSpace.DecodeParams(yuvMatrix="BT.709", range="full")
    assert workingSpace.toXYZ is None


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


def test_limited_range_is_expanded_on_load() -> None:
    """The limited range re-encode decodes like the full range original (within 8-bit limited quantization)."""
    limited = Video(VideoSource(str(COLOR/"limited_range.mp4"))).getFrame(index=0,
                                                                          colorMode=Video.COLOR_RGB)
    full = Video(VideoSource(str(COLOR/"full_range.mp4"))).getFrame(index=0,
                                                                    colorMode=Video.COLOR_RGB)
    assert abs(float((limited - full).mean())) < 0.002
    assert np.abs(limited - full).max() < 0.02


@pytest.mark.parametrize("kind", ["rgb", "gray"])
def test_untagged_rgb_and_gray_video_load_unchanged(kind: str) -> None:
    """Untagged RGB/gray video frames decode to exactly the values of the matching untagged PNGs."""
    frame = Video(VideoSource(str(COLOR/(kind + ".avi")))).getFrame(index=2,
                                                                    colorMode=Video.COLOR_RGB)
    png = Video(ImageSequenceSource(str(COLOR/(kind + "_sequence")/"frame_002.png"))).getFrame(index=0,
                                                                                               colorMode=Video.COLOR_RGB)
    assert np.abs(frame - png).max() < 1e-4


def test_raw_postprocess_arguments(monkeypatch: pytest.MonkeyPatch) -> None:
    """RAW files are decoded in camera RGB, unscaled, with their black level in and no white balance."""
    calls = []

    class FakeRaw:
        color_matrix = np.zeros((3, 4))
        rgb_xyz_matrix = np.vstack([np.eye(3), np.zeros(3)])
        black_level_per_channel = [64, 64, 64, 64]
        white_level = 1023
        num_colors = 3

        def __enter__(self) -> "FakeRaw":
            return self

        def __exit__(self, *args: object) -> None:
            pass

        def postprocess(self, **kwargs: object) -> np.ndarray:
            calls.append(kwargs)
            return np.full((2, 2, 3), 100, dtype=np.uint16)

    import rawpy
    monkeypatch.setattr(rawpy, "imread", lambda file: FakeRaw())
    image, params, workingSpace = Video.decodeImage("camera.nef", thumbnail=True)
    assert calls[0]["no_auto_scale"] is True
    assert calls[0]["user_black"] == 0
    assert calls[0]["user_wb"] == [1, 1, 1, 1]
    assert calls[0]["output_color"] == rawpy.ColorSpace.raw
    assert calls[0]["gamma"] == (1, 1)
    assert calls[0]["no_auto_bright"] is True
    assert calls[0]["output_bps"] == 16
    assert calls[0]["half_size"] is True
    assert params == ColorSpace.DecodeParams(blackLevel=(64.0, 64.0, 64.0), whiteLevel=1023.0)
    assert image.dtype == np.uint16


def test_raw_dng_has_no_white_balance() -> None:
    """The Pixel 8a crop decodes with its levels and matrix, and its neutral gravel keeps the raw ratios."""
    import rawpy
    file = str(INPUTS_DIR/"raw"/"pixel_crop_1.dng")
    image, params, workingSpace = Video.decodeImage(file)
    assert params == ColorSpace.DecodeParams(blackLevel=(64.0, 64.0, 64.0), whiteLevel=1023.0)
    assert np.allclose(workingSpace.toXYZ @ np.ones(3), ColorSpace.D65_XYZ, atol=1e-6)
    linear = Video(ImageSequenceSource(file)).getFrame(index=0,
                                                       colorMode=Video.COLOR_RGB)
    region = linear[256:336, 224:320].reshape(-1, 3).mean(axis=0)[::-1]
    with rawpy.imread(file) as raw:
        gains = np.array(raw.camera_whitebalance[:3])/raw.camera_whitebalance[1]
    ratios = region/region[1]
    assert np.allclose(ratios, 1/gains, rtol=0.1)
    assert linear.min() < 0


@pytest.mark.parametrize("kind", sorted(LOCAL_RAWS))
def test_local_camera_raws(kind: str) -> None:
    """The user's full-size camera files decode linear with their levels and a matrix mapping white to D65."""
    if(not pathlib.Path(LOCAL_RAWS[kind]).exists()):
        pytest.skip("Camera file not available")
    image, params, workingSpace = Video.decodeImage(LOCAL_RAWS[kind], thumbnail=True)
    assert params.whiteLevel in (1023.0, 16383.0)
    assert workingSpace.toXYZ is not None
    assert np.allclose(workingSpace.toXYZ @ np.ones(3), ColorSpace.D65_XYZ, atol=1e-6)
    linear = ColorSpace.linearize(img=image, params=params)
    assert np.percentile(linear, 1) < 0.02
    assert linear.max() <= 1.01


def test_fits_bayer_pattern_and_levels() -> None:
    """The Bayer FITS states GBRG (RGGB shifted one row) and keeps one scale for both frames."""
    first, params1, space = Video.decodeImage(str(INPUTS_DIR/"fits"/"bayer_1.fits"))
    second, params2, space = Video.decodeImage(str(INPUTS_DIR/"fits"/"bayer_2.fits"))
    assert params1 == params2 == ColorSpace.DecodeParams(bayerPattern="GBRG",
                                                         blackLevel=(0.0, 0.0, 0.0),
                                                         whiteLevel=60000.0)
    video = Video(ImageSequenceSource([str(INPUTS_DIR/"fits"/"bayer_1.fits"), str(INPUTS_DIR/"fits"/"bayer_2.fits")]))
    assert video.guessColorMode(image=first, params=params1) == Video.COLOR_GBRG
    frames = [video.getFrame(index=index, colorMode=Video.COLOR_GBRG) for index in range(2)]
    assert frames[0].min() == pytest.approx(2000/60000, abs=1e-3)
    assert frames[1].min() == pytest.approx(2000/60000, abs=1e-3)
    assert (frames[1].max() - 2000/60000) == pytest.approx(0.6*(frames[0].max() - 2000/60000), rel=0.02)
