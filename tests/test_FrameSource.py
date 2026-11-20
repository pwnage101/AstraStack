"""Tests for FrameSource.py."""
import pickle
import struct
import unittest
from collections.abc import Sequence
from types import SimpleNamespace
from unittest import mock

import cv2
import ddt
import numpy as np
import pytest

import ColorSpace
from Constants import AUTO, UNSPECIFIED, DemosaicAlgorithm, DemosaicMode, DemosaicParams, MosaicPattern
from FrameSource import (
    CameraRawSource,
    FitsSource,
    FrameSource,
    ImageSequenceSource,
    LinearizeParams,
    OpenCVDemosaicSource,
    VideoSource,
    linearize,
)
from tests.inputs import (
    AVI8_PAL8,
    AVI8_RGGB,
    DNG16_RGGB_SEQUENCE,
    DNG16_RGGB_SEQUENCE_DIMENSIONS,
    DROPPED_FRAMES_AVI,
    FITS16_RGGB_SEQUENCE,
    FITS16_RGGB_SEQUENCE_DIMENSIONS,
    INPUTS_DIR,
    JPG8,
    JPG8_DIMENSIONS,
    JPG8_NIKON,
    MP4_YUV8,
    MP4_YUV8_DIMENSIONS,
    MP4_YUV8_FRAME_COUNT,
    MP4_YUV8_LONG_GOP,
    NOT_MEDIA,
    PNG16,
    PNG8_SEQUENCE,
    PNG8_SEQUENCE_DIMENSIONS,
)


@ddt.ddt
class TestImageSequenceSource(unittest.TestCase):
    """ImageSequenceSource."""

    def test_is_a_frameSource(self) -> None:
        """Image sequences are frame sources that keep their paths."""
        source = ImageSequenceSource(PNG8_SEQUENCE)

        assert source.paths == PNG8_SEQUENCE
        assert isinstance(source, FrameSource)

    def test_frameCount_and_size(self) -> None:
        """The count is the number of files and the size is the first file's (width, height)."""
        source = ImageSequenceSource(PNG8_SEQUENCE)
        height, width = cv2.imread(PNG8_SEQUENCE[0]).shape[:2]

        assert source.frameCount() == len(PNG8_SEQUENCE)
        assert source.frameSize() == (width, height)

    def test_readStored_by_index(self) -> None:
        """Frames are read by index in any order, as stored."""
        source = ImageSequenceSource(PNG8_SEQUENCE)

        for index in [2, 0, 3, 1, 1, 2]:
            assert np.array_equal(source.readStored(index).image,
                                  cv2.imread(PNG8_SEQUENCE[index], cv2.IMREAD_UNCHANGED))

    @ddt.data(
        # An 8-bit PNG reads as 8 bits.
        {"path": PNG8_SEQUENCE[0], "dtype": np.uint8},
        # A 16-bit PNG reads as 16 bits, not reduced to 8.
        {"path": PNG16, "dtype": np.uint16},
        # An 8-bit JPEG reads as 8 bits.
        {"path": JPG8, "dtype": np.uint8},
    )
    @ddt.unpack
    def test_readStored_color_native_depth(self, path: str, dtype: type) -> None:
        """Color frames are read with three channels at their stored depth, and frameSize agrees with them."""
        source = ImageSequenceSource([path])

        frame = source.readStored(0).image

        assert frame.dtype == dtype
        assert frame.ndim == 3 and frame.shape[2] == 3
        assert source.frameSize() == (frame.shape[1], frame.shape[0])

    def test_readStored_past_end(self) -> None:
        """There is no frame past the last file."""
        assert ImageSequenceSource(PNG8_SEQUENCE).readStored(len(PNG8_SEQUENCE)) is None

    def test_pickles(self) -> None:
        """A source survives a pickle round trip and reads the same frames."""
        source = ImageSequenceSource(PNG8_SEQUENCE)
        before = source.readStored(1).image

        source = pickle.loads(pickle.dumps(source))

        assert source.paths == PNG8_SEQUENCE
        assert np.array_equal(source.readStored(1).image, before)


@ddt.ddt
class TestVideoSource(unittest.TestCase):
    """VideoSource."""

    @staticmethod
    def _decodeFrames(path: str) -> list[np.ndarray]:
        """Reads every frame of a video in order with a fresh VideoSource, so it never seeks.

        Args:
            path: The video file.

        Returns:
            The frames, in order.
        """
        source = VideoSource(path)
        try:
            return [source.readStored(index).image for index in range(source.frameCount())]
        finally:
            source.close()

    def test_frameCount_and_size(self) -> None:
        """The count is the number of frames in the video and the size is the container's."""
        source = VideoSource(MP4_YUV8)

        assert source.frameCount() == MP4_YUV8_FRAME_COUNT
        assert source.frameSize() == MP4_YUV8_DIMENSIONS

    def test_frameCount_is_counted_once(self) -> None:
        """The frames are counted once, the first time the count is needed."""
        source = VideoSource(MP4_YUV8)
        assert source.timestamps is None
        # Reading just the first frame does not trigger a full read. This
        # behavior is what makes thumbnail generation quick.
        source.readStored(0)
        assert source.timestamps is None

        assert source.frameCount() == MP4_YUV8_FRAME_COUNT
        source.close()
        source.path = "missing.avi"
        assert source.frameCount() == MP4_YUV8_FRAME_COUNT

    def test_readStored_in_any_order(self) -> None:
        """Frames are read in any order and repeatedly."""
        source = VideoSource(MP4_YUV8)
        sequential = [source.readStored(index).image for index in range(source.frameCount())]

        for index in [0, 1, 2, 40, 3, 59, 0, 40, 40, 2, 3]:
            assert np.array_equal(source.readStored(index).image, sequential[index]), index

    def test_readStored_past_end(self) -> None:
        """There is no frame past the end of the video, whether read in order or by seeking."""
        source = VideoSource(MP4_YUV8)
        for index in range(MP4_YUV8_FRAME_COUNT):
            source.readStored(index)

        assert source.readStored(MP4_YUV8_FRAME_COUNT) is None
        assert VideoSource(MP4_YUV8).readStored(MP4_YUV8_FRAME_COUNT) is None

    def test_palette_video_after_seeks(self) -> None:
        """A palette video's frames keep their palette when read after a seek, as OpenCV decodes them."""
        capture = cv2.VideoCapture(AVI8_PAL8)
        expectedFrames = [capture.read()[1].astype(np.uint16)*257 for _ in range(3)]
        capture.release()
        source = VideoSource(AVI8_PAL8)
        for index in (2, 0, 1):
            assert np.array_equal(source.readStored(index).image, expectedFrames[index]), f"frame {index} is wrong"

    def test_pickles_without_capture(self) -> None:
        """An open source pickles as its path and timestamps, and the copy reads the same frames in any order."""
        source = VideoSource(MP4_YUV8)
        sequential = [source.readStored(index).image for index in range(source.frameCount())]
        timestamps = source.timestamps

        source = pickle.loads(pickle.dumps(source))

        assert source.path == MP4_YUV8
        assert source.timestamps == timestamps
        assert source.container is None
        for index in [2, 0, 5, 5, 3, 1, 4]:
            assert np.array_equal(source.readStored(index).image, sequential[index])

    @ddt.data(
        # The first read is frame 1, which used to read frame 0.
        {"order": [1, 2, 3]},
        # The same frame twice, then back to the start.
        {"order": [1, 1, 0]},
        # The first read is past frame 1, then steps back to it.
        {"order": [2, 3, 1]},
    )
    @ddt.unpack
    def test_fresh_copy_reads_by_index(self, order: list[int]) -> None:
        """A copy that hasn't read anything yet still reads the frames asked for, starting with any frame.

        A fresh capture reports a time of 0 before its first read, which used to look like frame 0
        had just been read, so asking for frame 1 read frame 0 instead.
        """
        expectedFrames = self._decodeFrames(MP4_YUV8)
        source = VideoSource(MP4_YUV8)
        source.frameCount()
        source = pickle.loads(pickle.dumps(source))

        for index in order:
            assert np.array_equal(source.readStored(index).image, expectedFrames[index]), index

    def test_dropped_frames_avi_is_missing_frames(self) -> None:
        """Make sure the dropped frames test video has gaps.

        Other tests rely on this video having missing frames.
        """
        capture = cv2.VideoCapture(DROPPED_FRAMES_AVI)
        headerCount = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        capture.release()
        source = VideoSource(DROPPED_FRAMES_AVI)
        assert headerCount > source.frameCount()

    @ddt.data(
        # Frames missing from the container, every frame a keyframe.
        {"path": DROPPED_FRAMES_AVI},
        # One keyframe at the start, so every seek must decode forward from frame 0.
        {"path": MP4_YUV8},
        # Several keyframes, so seeks must decode forward from an earlier one.
        {"path": MP4_YUV8_LONG_GOP},
    )
    @ddt.unpack
    def test_reads_correct_frame_however_reached(self, path: str) -> None:
        """A video always fetches the correct frame no matter where we start.

        Seeking doesn't always land on the frame asked for, so VideoSource seeks to an earlier
        keyframe and decodes forward from it.

        Check that it works by seeking to many points in the video and doing many
        different lengths of sequential reads. This hopefully simulates both a user
        scrubbing frames and worker pool processing frames.
        """
        # First, decode the frames cleanly and make sure they're unique, so reading the
        # wrong frame can't go unnoticed.
        expectedFrames = self._decodeFrames(path)
        assert len({image.tobytes() for image in expectedFrames}) == len(expectedFrames)

        # Runs of consecutive frames.
        runs: list[tuple[str, Sequence[int]]] = []
        # Full read without any seeks prior to this run.
        runs += [
            ("sequentially read entire file before a bunch of seeks", range(len(expectedFrames)))
        ]
        # Full read, but in reverse direction.
        runs += [
            (f"backward seek to {index} and read 1 frame", [index])
            for index in reversed(range(len(expectedFrames)))
        ]
        # Short runs of 4 frames each, evenly distributed across the file.
        # Different from a full sequential read because there are frame gaps between each run.
        runs += [
            (f"forward seek to {start} and read 4 frames", range(start, min(start + 4, len(expectedFrames))))
            for start in range(0, len(expectedFrames), 5)
        ]
        # Final full read with a potentially unclean internal state.
        runs += [
            ("sequentially read entire file again after a bunch of seeks", range(len(expectedFrames)))
        ]

        # Now, simulate all the runs above through VideoSource.
        source = VideoSource(path)
        actualFrameCount = source.frameCount()
        # pickle + unpickle the VideoSource to simulate being passed into a worker process.
        source = pickle.loads(pickle.dumps(source))
        actualFrames = [
            (name, index, source.readStored(index).image)
            for name, run in runs for index in run
        ]

        assert actualFrameCount == len(expectedFrames)
        for name, index, image in actualFrames:
            assert np.array_equal(image, expectedFrames[index]), f"{name}: frame {index} is wrong"


@ddt.ddt
class TestOpen(unittest.TestCase):
    """FrameSource.open and canOpen."""

    @ddt.data(
        # A single lossless image.
        {"path": PNG8_SEQUENCE[0],
         "expected_class": ImageSequenceSource,
         "expected_frame_count": 1,
         "expected_frame_size": PNG8_SEQUENCE_DIMENSIONS},
        # A single lossy image.
        {"path": JPG8,
         "expected_class": ImageSequenceSource,
         "expected_frame_count": 1,
         "expected_frame_size": JPG8_DIMENSIONS},
        # A single FITS file, which OpenCV can't read.
        {"path": FITS16_RGGB_SEQUENCE[0],
         "expected_class": FitsSource,
         "expected_frame_count": 1,
         "expected_frame_size": FITS16_RGGB_SEQUENCE_DIMENSIONS},
        # A single raw camera file.
        {"path": DNG16_RGGB_SEQUENCE[0],
         "expected_class": CameraRawSource,
         "expected_frame_count": 1,
         "expected_frame_size": DNG16_RGGB_SEQUENCE_DIMENSIONS},
        # A list of one image opens like the image alone.
        {"path": PNG8_SEQUENCE[:1],
         "expected_class": ImageSequenceSource,
         "expected_frame_count": 1,
         "expected_frame_size": PNG8_SEQUENCE_DIMENSIONS},
        # Several images are a sequence in the order given.
        {"path": PNG8_SEQUENCE,
         "expected_class": ImageSequenceSource,
         "expected_frame_count": len(PNG8_SEQUENCE),
         "expected_frame_size": PNG8_SEQUENCE_DIMENSIONS},
        # Several raw camera files are a sequence.
        {"path": DNG16_RGGB_SEQUENCE,
         "expected_class": CameraRawSource,
         "expected_frame_count": len(DNG16_RGGB_SEQUENCE),
         "expected_frame_size": DNG16_RGGB_SEQUENCE_DIMENSIONS},
        # A video.
        {"path": MP4_YUV8,
         "expected_class": VideoSource,
         "expected_frame_count": MP4_YUV8_FRAME_COUNT,
         "expected_frame_size": MP4_YUV8_DIMENSIONS},
        # An image sequence can't contain a video.
        {"path": PNG8_SEQUENCE + [MP4_YUV8], "expected_error": "can't be opened"},
        # A file that is neither an image nor a video.
        {"path": NOT_MEDIA, "expected_error": "can't be opened"},
        # Every file of an image sequence must be an image.
        {"path": PNG8_SEQUENCE + [NOT_MEDIA], "expected_error": "can't be opened"},
        # Nothing chosen.
        {"path": [], "expected_error": "No file was chosen"},
    )
    @ddt.unpack
    def test_open(self,
                  path: str | list[str],
                  expected_class: type[FrameSource] | None = None,
                  expected_frame_count: int | None = None,
                  expected_frame_size: tuple[int, int] | None = None,
                  expected_error: str | None = None) -> None:
        """The chosen files open as the expected kind of source, or fail with expected_error."""
        if(expected_error is not None):
            with self.assertRaisesRegex(ValueError, expected_error):
                FrameSource.open(path)
            return

        source = FrameSource.open(path)

        assert type(source) is expected_class
        assert source.filePaths() == ([path] if isinstance(path, str) else path)
        assert source.frameCount() == expected_frame_count
        assert source.frameSize() == expected_frame_size
        assert source.readFrame(0) is not None
        assert source.readFrame(source.frameCount() - 1) is not None
        assert source.readFrame(source.frameCount()) is None

    def test_jpeg_libraw_opens_is_read_by_opencv(self) -> None:
        """A camera-generated JPEG loads, rather than being mistaken for a raw file because LibRaw opens it."""
        source = FrameSource.open(JPG8_NIKON)

        assert type(source) is ImageSequenceSource
        assert source.readFrame(0) is not None

    @ddt.data(
        # An untagged image sequence.
        {"path": PNG8_SEQUENCE},
        # An untagged video.
        {"path": MP4_YUV8},
    )
    @ddt.unpack
    def test_firstFrameFacts_describes_the_source(self, path: str | list[str]) -> None:
        """A source is described like its first frame, and untagged images and videos state no Bayer pattern."""
        source = FrameSource.open(path)
        frame = source.readStored(0)

        assert source.firstFrameFacts.linearizeParams == frame.linearizeParams
        assert source.firstFrameFacts.workingSpace == frame.workingSpace
        assert source.firstFrameFacts.linearizeParams.bayerPattern is None

    @ddt.data(
        # A single image is a one-frame sequence.
        {"choice": PNG8_SEQUENCE[0], "expected_openers": [ImageSequenceSource]},
        # Several images are a sequence.
        {"choice": PNG8_SEQUENCE, "expected_openers": [ImageSequenceSource]},
        # Raw camera files are opened by their own source, though OpenCV reads some as TIFF.
        {"choice": DNG16_RGGB_SEQUENCE, "expected_openers": [CameraRawSource]},
        # FITS files are opened by their own source.
        {"choice": FITS16_RGGB_SEQUENCE, "expected_openers": [FitsSource]},
        # A single video is a video.
        {"choice": DROPPED_FRAMES_AVI, "expected_openers": [VideoSource]},
        # A video in a list isn't opened, since only images can be chosen as a list.
        {"choice": [DROPPED_FRAMES_AVI], "expected_openers": []},
        # A video can't be part of an image sequence.
        {"choice": PNG8_SEQUENCE + [DROPPED_FRAMES_AVI], "expected_openers": []},
        # A file that is neither an image nor a video.
        {"choice": NOT_MEDIA, "expected_openers": []},
        # Nothing chosen.
        {"choice": [], "expected_openers": []},
    )
    @ddt.unpack
    def test_canOpen(self, choice: str | list[str], expected_openers: list[type[FrameSource]]) -> None:
        """Each kind of source opens the choices it recognizes by their contents."""
        kinds = (CameraRawSource, FitsSource, ImageSequenceSource, VideoSource)
        openers = [source for source in kinds if source.canOpen(choice)]

        assert openers == expected_openers


AVI8_GRAY = str(INPUTS_DIR/"color"/"avi8_gray.avi")


@ddt.ddt
class TestDemosaicChoices(unittest.TestCase):
    """setDemosaicChoices and the supported choices, per kind of source."""

    @ddt.data(
        # A gray mosaic (an 8-bit video) is guessed as RGGB under Auto.
        {"path": AVI8_RGGB,
         "chosenMode": AUTO,
         "expectedResolvedParams": DemosaicParams(MosaicPattern.RGGB, DemosaicAlgorithm.EDGE_AWARE)},
        # A gray mosaic isn't demosaiced when Off.
        {"path": AVI8_RGGB,
         "chosenMode": DemosaicMode.OFF,
         "expectedResolvedParams": None},
        # A pattern and an algorithm are honored under On.
        {"path": AVI8_RGGB,
         "chosenMode": DemosaicMode.ON,
         "chosenPattern": MosaicPattern.GBRG,
         "chosenAlgorithm": DemosaicAlgorithm.VNG,
         "expectedResolvedParams": DemosaicParams(MosaicPattern.GBRG, DemosaicAlgorithm.VNG)},
        # A FITS file states its pattern, so the pixel guess isn't used.
        {"path": FITS16_RGGB_SEQUENCE,
         "chosenMode": AUTO,
         "expectedResolvedParams": DemosaicParams(MosaicPattern.RGGB, DemosaicAlgorithm.EDGE_AWARE)},
        # A gray image that isn't a mosaic isn't demosaiced under Auto.
        {"path": AVI8_GRAY,
         "chosenMode": AUTO,
         "expectedResolvedParams": None},
        # On demosaics a gray image as the first member of the first family.
        {"path": AVI8_GRAY,
         "chosenMode": DemosaicMode.ON,
         "expectedResolvedParams": DemosaicParams(MosaicPattern.RGGB, DemosaicAlgorithm.EDGE_AWARE)},
        # A color input isn't demosaiced.
        {"path": PNG8_SEQUENCE,
         "chosenMode": AUTO,
         "expectedResolvedParams": None},
        # A Bayer RAW is demosaiced by LibRaw, which picks the algorithm under Auto.
        {"path": DNG16_RGGB_SEQUENCE,
         "chosenMode": AUTO,
         "expectedResolvedParams": DemosaicParams(MosaicPattern.RGGB, UNSPECIFIED)},
        # A Bayer RAW is demosaiced with the algorithm chosen under On.
        {"path": DNG16_RGGB_SEQUENCE,
         "chosenMode": DemosaicMode.ON,
         "chosenAlgorithm": DemosaicAlgorithm.DHT,
         "expectedResolvedParams": DemosaicParams(MosaicPattern.RGGB, DemosaicAlgorithm.DHT)},
    )
    @ddt.unpack
    def test_setDemosaicChoices(
        self,
        path: str | list[str],
        chosenMode: DemosaicMode | AUTO,
        expectedResolvedParams: DemosaicParams | None,
        chosenPattern: MosaicPattern | AUTO = AUTO,
        chosenAlgorithm: DemosaicAlgorithm | AUTO = AUTO,
    ) -> None:
        """The choices resolve as the source supports them, and the source keeps the demosaic params."""
        source = FrameSource.open(path)

        resolvedParams = source.setDemosaicChoices(mode=chosenMode, pattern=chosenPattern, algorithm=chosenAlgorithm)

        assert resolvedParams == expectedResolvedParams
        assert source.demosaicParams == expectedResolvedParams

    @ddt.data(
        {"path": AVI8_RGGB, "chosenMode": AUTO, "expectedDisplayNames": ("On", "RGGB", "Edge-aware")},
        {"path": AVI8_RGGB, "chosenMode": DemosaicMode.OFF, "expectedDisplayNames": ("Off", "Off", "Off")},
        # LibRaw picks the algorithm.
        {"path": DNG16_RGGB_SEQUENCE, "chosenMode": AUTO, "expectedDisplayNames": ("On", "RGGB", "LibRaw")},
    )
    @ddt.unpack
    def test_demosaicParamsDisplayNames(
        self,
        path: str | list[str],
        chosenMode: DemosaicMode | AUTO,
        expectedDisplayNames: tuple[str, str, str],
    ) -> None:
        """The source names the demosaic params it resolved."""
        source = FrameSource.open(path)
        source.setDemosaicChoices(mode=chosenMode, pattern=AUTO, algorithm=AUTO)

        assert source.demosaicParamsDisplayNames() == expectedDisplayNames

    @ddt.data(
        # The remembered pattern is used when it is in the guessed family.
        {"remembered": MosaicPattern.BGGR, "expectedPattern": MosaicPattern.BGGR},
        {"remembered": MosaicPattern.RGGB, "expectedPattern": MosaicPattern.RGGB},
        # Otherwise the family's first member is.
        {"remembered": MosaicPattern.GBRG, "expectedPattern": MosaicPattern.RGGB},
        {"remembered": MosaicPattern.GRBG, "expectedPattern": MosaicPattern.RGGB},
    )
    @ddt.unpack
    def test_guess_tie_break(self, remembered: MosaicPattern, expectedPattern: MosaicPattern) -> None:
        """The pixel guess knows the family, and the remembered pattern picks the member."""
        demosaicParams = FrameSource.open(AVI8_RGGB).setDemosaicChoices(
            mode=AUTO,
            pattern=AUTO,
            algorithm=AUTO,
            rememberedPattern=remembered,
        )

        assert demosaicParams.pattern == expectedPattern

    @ddt.data(
        {"path": AVI8_RGGB, "modes": 2, "patterns": 4, "algorithms": 3},
        # VNG isn't offered for 16 bits.
        {"path": FITS16_RGGB_SEQUENCE, "modes": 2, "patterns": 4, "algorithms": 2},
        {"path": PNG8_SEQUENCE, "modes": 0, "patterns": 0, "algorithms": 0},
        # A RAW can't be left mosaiced, and its pattern comes from the file.
        {"path": DNG16_RGGB_SEQUENCE, "modes": 1, "patterns": 0, "algorithms": 7},
    )
    @ddt.unpack
    def test_supported_choices(self, path: str | list[str], modes: int, patterns: int, algorithms: int) -> None:
        """Each kind of source offers its own choices besides Auto."""
        source = FrameSource.open(path)

        assert len(source.supportedDemosaicModes()) == modes
        assert len(source.supportedMosaicPatterns()) == patterns
        assert len(source.supportedDemosaicAlgorithms()) == algorithms

    def test_opencv_demosaic_codes(self) -> None:
        """Every pattern has a cv2 code for each of OpenCV's algorithms."""
        for name in OpenCVDemosaicSource.OPENCV_DEMOSAIC_CODES.values():
            assert isinstance(getattr(cv2, name), int), name
        assert len(OpenCVDemosaicSource.OPENCV_DEMOSAIC_CODES) == 4*3

    @ddt.data(
        # A 2×2 Bayer pattern is named by its colors.
        {"colorDesc": b"RGBG",
         "rawPattern": np.array([[2, 3], [1, 0]]),
         "expectedPattern": MosaicPattern.BGGR},
        # X-Trans isn't a 2×2 pattern.
        {"colorDesc": b"RGBG",
         "rawPattern": np.zeros((6, 6), dtype=int),
         "expectedPattern": UNSPECIFIED},
    )
    @ddt.unpack
    def test_raw_pattern(
        self,
        colorDesc: bytes,
        rawPattern: np.ndarray,
        expectedPattern: MosaicPattern | UNSPECIFIED,
    ) -> None:
        """A RAW's pattern is read from LibRaw's description of it."""
        raw = mock.Mock(color_desc=colorDesc, raw_pattern=rawPattern)

        assert CameraRawSource.readMosaicPattern(raw) == expectedPattern


def chrm(white: tuple, primaries: tuple) -> bytes:
    """Returns a PNG cHRM chunk's data.

    Args:
        white: CIE xy of the white point.
        primaries: CIE xy of the red, green and blue primaries.

    Returns:
        The chunk data.
    """
    values = list(white) + [v for xy in primaries for v in xy]
    return struct.pack(">8I", *[int(round(v*100000)) for v in values])


SRGB_CHRM = chrm(white=(0.3127, 0.3290), primaries=((0.64, 0.33), (0.30, 0.60), (0.15, 0.06)))


@pytest.mark.parametrize("chunks, expectedTransfer, expectedSpace",
                         [([(b"cICP", bytes([1, 13, 0, 1])), (b"sRGB", b"\0")],
                           ColorSpace.SRGB_CURVE, ColorSpace.SRGB),
                          ([(b"cICP", bytes([9, 16, 0, 1])), (b"sRGB", b"\0")], ("H273", 16),
                           ColorSpace.cicpPrimaries(9)),
                          ([(b"sRGB", b"\0"), (b"gAMA", struct.pack(">I", 100000))], ColorSpace.SRGB_CURVE,
                           ColorSpace.SRGB),
                          ([(b"gAMA", struct.pack(">I", 45455))], ColorSpace.SRGB_CURVE, ColorSpace.SRGB),
                          ([(b"gAMA", struct.pack(">I", 45455)), (b"cHRM", SRGB_CHRM)], ColorSpace.SRGB_CURVE,
                           ColorSpace.SRGB),
                          ([(b"gAMA", struct.pack(">I", 50000))], ("Gamma", 2.0), ColorSpace.UNTAGGED),
                          ([(b"gAMA", struct.pack(">I", 100000))], None, ColorSpace.UNTAGGED),
                          ([(b"gAMA", struct.pack(">I", 100000)), (b"cHRM", SRGB_CHRM)], None, ColorSpace.SRGB),
                          ([(b"cHRM", SRGB_CHRM)], ColorSpace.SRGB_CURVE, ColorSpace.SRGB)],
                         ids=["cICP over sRGB", "cICP PQ", "sRGB over gAMA", "gAMA 45455",
                              "gAMA 45455 with sRGB cHRM", "gAMA 50000", "gAMA 100000", "gAMA 100000 with sRGB cHRM",
                              "cHRM without gAMA"])
def test_png_chunk_precedence(chunks: list,
                              expectedTransfer: tuple | None,
                              expectedSpace: ColorSpace.WorkingSpace) -> None:
    """PNG chunks follow cICP > sRGB > cHRM/gAMA."""
    transfer, workingSpace = ImageSequenceSource.fromPNGChunks(chunks=chunks)
    assert transfer == expectedTransfer
    assert workingSpace == expectedSpace


def test_png_untagged() -> None:
    """A PNG without color chunks is untagged."""
    assert ImageSequenceSource.fromPNGChunks(chunks=[(b"IHDR", bytes(13)), (b"tEXt", b"a\0b")]) is None


@pytest.mark.parametrize("colorRange, isYUV, fullRangeFormat, expected",
                         [(None, True, False, True),
                          (0, True, False, True),
                          (2, True, False, False),
                          (1, False, False, True),
                          (None, True, True, False),
                          (None, False, False, False)],
                         ids=["untagged YUV", "unspecified YUV", "tagged full", "tagged limited RGB", "yuvj",
                              "untagged RGB"])
def test_isLimitedRange(colorRange: int | None, isYUV: bool, fullRangeFormat: bool, expected: bool) -> None:
    """Each missing range tag falls back on its own: limited for YUV, full for RGB and gray."""
    assert VideoSource.isLimitedRange(colorRange=colorRange, isYUV=isYUV, fullRangeFormat=fullRangeFormat) == expected


def test_pngChunks_stop_at_idat() -> None:
    """Only the chunks before the first IDAT are read, and a file that isn't a PNG has none."""
    chunks = ImageSequenceSource.pngChunks(str(INPUTS_DIR/"color"/"png16_srgb_linear.png"))
    assert chunks[0][0] == b"IHDR"
    assert b"IDAT" not in [chunkType for chunkType, data in chunks]
    assert ImageSequenceSource.pngChunks(str(INPUTS_DIR/"color"/"jpg_none_none.jpg")) == []


@mock.patch("rawpy.imread")
def test_raw_postprocess_arguments(imread: mock.MagicMock) -> None:
    """RAW files are decoded in camera RGB with their as-shot white balance, and their black level still in."""
    raw = imread.return_value.__enter__.return_value
    raw.postprocess.return_value = np.full((2, 2, 3), 100, dtype=np.uint16)
    raw.color_matrix = np.zeros((3, 4))
    raw.rgb_xyz_matrix = np.vstack([np.eye(3), np.zeros(3)])
    raw.black_level_per_channel = [64, 64, 64, 64]
    raw.white_level = 1023
    raw.num_colors = 3
    raw.camera_whitebalance = [2.0, 1.0, 1.6, 0.0]
    raw.daylight_whitebalance = [2.2, 1.1, 1.5, 0.0]

    image, linearizeParams, workingSpace = CameraRawSource.decodeRaw(file="camera.nef")
    CameraRawSource.decodeRaw(file="camera.nef", algorithm=DemosaicAlgorithm.AAHD)

    import rawpy
    assert raw.postprocess.call_count == 2
    # LibRaw picks the algorithm unless one is chosen
    assert "demosaic_algorithm" not in raw.postprocess.call_args_list[0].kwargs
    assert raw.postprocess.call_args_list[1].kwargs["demosaic_algorithm"] == rawpy.DemosaicAlgorithm.AAHD
    raw.postprocess.assert_any_call(
        user_black=0,
        user_cblack=[0, 0, 0, 0],
        user_wb=[2.0, 1.0, 1.6, 1.0],
        adjust_maximum_thr=0,
        output_color=rawpy.ColorSpace.raw,
        gamma=(1, 1),
        no_auto_bright=True,
        output_bps=16,
    )
    assert linearizeParams == LinearizeParams(
        blackLevel=(64.0, 64.0, 64.0),
        whiteLevel=1023.0,
        asShotMultipliers=pytest.approx((2.0*64.0, 64.0, 1.6*64.0), rel=1e-3),
        daylightMultipliers=pytest.approx((2.0, 1.0, 1.5/1.1)),
    )
    assert image.dtype == np.uint16


# Measured with rawpy 0.25.1 (LibRaw 0.21.4)
PIXEL_8A_COLOR_MATRIX = np.array([[1.6143, -0.5798, -0.0345, 0.0],
                                  [-0.1447, 1.3474, -0.2027, 0.0],
                                  [0.0291, -0.8361, 1.8070, 0.0]])
A7III_RGB_XYZ_MATRIX = np.array([[0.7374, -0.2389, -0.0551],
                                 [-0.5435, 1.3162, 0.2519],
                                 [-0.1006, 0.1795, 0.6552],
                                 [0.0, 0.0, 0.0]])
A7III_RGB_CAM = np.array([[1.6354, -0.4220, -0.2134],
                          [-0.1311, 1.5427, -0.4116],
                          [0.0114, -0.4196, 1.4082]])


def fakeRaw(**overrides: object) -> SimpleNamespace:
    """Returns a stand-in for rawpy.RawPy with a Sony A7 III's color data.

    Args:
        **overrides: Attributes to replace.

    Returns:
        The fake RawPy.
    """
    raw = SimpleNamespace(color_matrix=np.zeros((3, 4)),
                          rgb_xyz_matrix=A7III_RGB_XYZ_MATRIX,
                          black_level_per_channel=[512, 510, 514, 512],
                          white_level=16383,
                          num_colors=3,
                          camera_whitebalance=[2.21, 1.0, 1.81, 0.0],
                          daylight_whitebalance=[2.3, 1.0, 1.7, 0.0])
    for name, value in overrides.items():
        setattr(raw, name, value)
    return raw



@pytest.mark.parametrize("header, dtype, expectedWhite",
                         [({"DATAMAX": 60000}, np.uint16, 60000.0),
                          ({"SATURATE": 4095}, np.uint16, 4095.0),
                          ({"DATAMAX": 50000, "SATURATE": 60000}, np.uint16, 50000.0),
                          ({}, np.uint16, 65535.0),
                          ({}, np.uint8, 255.0),
                          ({}, np.int32, 2147483647.0)])
def test_fits_levels(header: dict, dtype: type, expectedWhite: float) -> None:
    """FITS white comes from DATAMAX or SATURATE when present, else the integer type's maximum; no black is stated."""
    linearizeParams = FitsSource.fromFITS(header=header, data=np.zeros((4, 4), dtype=dtype))
    assert linearizeParams.blackLevel is None
    assert linearizeParams.whiteLevel == expectedWhite
    assert linearizeParams.bayerPattern is None


def test_fits_float_levels() -> None:
    """Float FITS without levels keywords: white 1.0 if the data fits in 0-1, else 65535."""
    assert FitsSource.fromFITS(header={}, data=np.array([[0.0, 0.5, 1.0]])).whiteLevel == 1.0
    assert FitsSource.fromFITS(header={}, data=np.array([[0.0, 2.0, 4000.0]])).whiteLevel == 65535.0


def test_fits_levels_dont_depend_on_brightness() -> None:
    """Two frames of different brightness get the same scale (no per-frame min/max)."""
    header = {"SATURATE": 60000}
    bright = np.array([[2000, 50000]], dtype=np.uint16)
    dim = np.array([[2000, 30000]], dtype=np.uint16)
    brightParams = FitsSource.fromFITS(header=header, data=bright)
    dimParams = FitsSource.fromFITS(header=header, data=dim)
    assert brightParams == dimParams
    assert linearize(img=dim, linearizeParams=dimParams)[0, 1] == pytest.approx(0.5)


@pytest.mark.parametrize("header, height, expected",
                         [({"BAYERPAT": "RGGB"}, 4, "RGGB"),
                          ({"BAYERPAT": "rggb "}, 4, "RGGB"),
                          ({"BAYERPAT": "RGGB", "YBAYROFF": 1}, 4, "GBRG"),
                          ({"BAYERPAT": "RGGB", "XBAYROFF": 1}, 4, "GRBG"),
                          ({"BAYERPAT": "RGGB", "XBAYROFF": 1, "YBAYROFF": 1}, 4, "BGGR"),
                          ({"BAYERPAT": "GRBG", "YBAYROFF": 2}, 4, "GRBG"),
                          ({"BAYERPAT": "RGGB", "ROWORDER": "BOTTOM-UP"}, 4, "GBRG"),
                          ({"BAYERPAT": "RGGB", "ROWORDER": "BOTTOM-UP"}, 5, "RGGB"),
                          ({"BAYERPAT": "RGGB", "ROWORDER": "TOP-DOWN"}, 4, "RGGB"),
                          ({"BAYERPAT": "CMYG"}, 4, None)])
def test_fits_bayer_pattern(header: dict, height: int, expected: str | None) -> None:
    """BAYERPAT is normalized to the stored data's first pixel with its offsets and row order."""
    linearizeParams = FitsSource.fromFITS(header=header, data=np.zeros((height, 4), dtype=np.uint16))
    assert linearizeParams.bayerPattern == (None if expected is None else MosaicPattern(expected))


def test_fromRAW_levels() -> None:
    """RAW levels come from black_level_per_channel (greens averaged) and white_level."""
    linearizeParams, workingSpace = CameraRawSource.fromRAW(fakeRaw())
    assert linearizeParams.blackLevel == (512.0, 511.0, 514.0)
    assert linearizeParams.whiteLevel == 16383.0


def test_fromRAW_rgb_xyz_matrix() -> None:
    """A per-model table matrix (A7 III) gives the expected rgbCam."""
    linearizeParams, workingSpace = CameraRawSource.fromRAW(fakeRaw())
    rgbCam = np.linalg.solve(ColorSpace.SRGB.toXYZ, workingSpace.toXYZ)
    assert np.allclose(rgbCam, A7III_RGB_CAM, atol=5e-4)
    assert np.allclose(rgbCam.sum(axis=1), 1)
    assert np.allclose(workingSpace.toXYZ @ np.ones(3), ColorSpace.D65_XYZ, atol=1e-6)


def test_fromRAW_color_matrix() -> None:
    """A DNG's color_matrix (Pixel 8a) is used as rgbCam."""
    raw = fakeRaw(rgb_xyz_matrix=np.zeros((4, 3)), color_matrix=PIXEL_8A_COLOR_MATRIX)
    linearizeParams, workingSpace = CameraRawSource.fromRAW(raw)
    assert np.allclose(workingSpace.toXYZ, ColorSpace.SRGB.toXYZ @ PIXEL_8A_COLOR_MATRIX[:, :3])
    assert np.allclose(workingSpace.toXYZ @ np.ones(3), ColorSpace.D65_XYZ, atol=1e-6)


def test_fromRAW_without_matrix() -> None:
    """A camera LibRaw doesn't know gets no primaries and a warning."""
    with pytest.warns(ColorSpace.ColorWarning):
        linearizeParams, workingSpace = CameraRawSource.fromRAW(fakeRaw(rgb_xyz_matrix=np.zeros((4, 3))))
    assert workingSpace.toXYZ is None
    assert linearizeParams.whiteLevel == 16383.0
    assert linearizeParams.asShotMultipliers is not None
    assert linearizeParams.daylightMultipliers is None


@pytest.mark.parametrize("asShot, expected",
                         [([2.0, 1.0, 1.6, 0.0], [2.0, 1.0, 1.6, 1.0]),
                          ([512.0, 256.0, 400.0, 258.0], [2.0, 1.0, 1.5625, 258/256]),
                          ([0.0, 1.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0])])
def test_asShotMultipliers(asShot: list[float], expected: list[float]) -> None:
    """As-shot multipliers are normalized to green, G2 = G when absent, and all 1 when the file states none."""
    assert CameraRawSource.asShotMultipliers(fakeRaw(camera_whitebalance=asShot)) == pytest.approx(expected)


def test_fromRAW_daylight_multipliers() -> None:
    """The D65 reference multipliers are LibRaw's daylight multipliers normalized to green."""
    linearizeParams, workingSpace = CameraRawSource.fromRAW(fakeRaw(daylight_whitebalance=[512.0, 256.0, 384.0, 0.0]))
    assert linearizeParams.daylightMultipliers == pytest.approx((2.0, 1.0, 1.5))


def test_raw_daylight_white_is_d65() -> None:
    """A DNG's sensor response to D65 white, once balanced by its daylight multipliers, is D65 in its working space."""
    import rawpy
    with rawpy.imread(str(INPUTS_DIR/"raw"/"pixel_crop_1.dng")) as raw:
        linearizeParams, workingSpace = CameraRawSource.fromRAW(raw)
    daylight = np.array(linearizeParams.daylightMultipliers)
    black = np.array(linearizeParams.blackLevel)
    # Raw values of a D65 white as LibRaw stores them: black plus a response inversely proportional to daylight
    sensor = black + 0.5*(linearizeParams.whiteLevel - black)/daylight
    stored = (sensor*np.array(linearizeParams.asShotMultipliers)).reshape(1, 1, 3)[:, :, ::-1]
    linear = linearize(img=stored.astype(np.float32), linearizeParams=linearizeParams)[0, 0, ::-1]
    assert linear == pytest.approx(np.full(3, 0.5), rel=1e-5)
    assert workingSpace.toXYZ @ linear == pytest.approx(0.5*np.array(ColorSpace.D65_XYZ), rel=1e-4)


def test_linearize_sensor_data_maps_black_to_0_and_white_to_1() -> None:
    """Sensor data maps black to 0 and white to 1 per channel (levels stated RGB, image BGR), unclipped."""
    img = np.array([[[1016, 1000, 16383], [1000, 1000, 1000]]], dtype=np.uint16)
    linearizeParams = LinearizeParams(blackLevel=(1000.0, 1008.0, 1016.0), whiteLevel=16383.0)
    linear = linearize(img=img, linearizeParams=linearizeParams)
    assert linear.dtype == np.float32
    assert linear[0, 0] == pytest.approx([0, -8/15375, 1])
    assert linear[0, 1, 0] < 0
    assert linear[0, 1, 2] == 0


def sensorLinearizeParams(**fields: object) -> LinearizeParams:
    """Returns linearize params for sensor data with black 0 and white 1000, with the given fields."""
    return LinearizeParams(blackLevel=(0.0, 0.0, 0.0), whiteLevel=1000.0, **fields)


def test_linearize_swaps_as_shot_for_daylight() -> None:
    """Data balanced as shot (each channel multiplied by 2, 1, 4 and a common 3) ends balanced to the D65 reference."""
    # A gray card as shot is (2, 1, 4) of the sensor's white; the reference sees it as (1.5, 1, 2)
    raw = np.array([[[100, 100, 100]]], dtype=np.float32)
    stored = raw*np.array([4, 1, 2], dtype=np.float32)*3
    linearizeParams = sensorLinearizeParams(asShotMultipliers=(6.0, 3.0, 12.0), daylightMultipliers=(1.5, 1.0, 2.0))
    assert linearize(img=stored, linearizeParams=linearizeParams)[0, 0] == pytest.approx([0.2, 0.1, 0.15])


def test_linearize_keeps_as_shot_without_daylight() -> None:
    """Without reference multipliers (no camera matrix) the data stays balanced as shot."""
    raw = np.array([[[100, 100, 100]]], dtype=np.float32)
    stored = raw*np.array([4, 1, 2], dtype=np.float32)*3
    linearizeParams = sensorLinearizeParams(asShotMultipliers=(6.0, 3.0, 12.0))
    assert linearize(img=stored, linearizeParams=linearizeParams)[0, 0] == pytest.approx([0.4, 0.1, 0.2])


def test_linearize_multiplies_nothing_without_multipliers() -> None:
    """Sensor data with no multipliers (generic Bayer, FITS) is only scaled by its levels."""
    img = np.array([[[100, 100, 100]]], dtype=np.float32)
    assert linearize(img=img, linearizeParams=sensorLinearizeParams())[0, 0] == pytest.approx([0.1, 0.1, 0.1])


class TestCameraRawPreview(unittest.TestCase):
    """CameraRawSource.thumbnail."""

    # A 2×3 BGR image
    IMAGE = np.arange(18, dtype=np.uint8).reshape((2, 3, 3))*10

    @mock.patch("rawpy.imread")
    def test_without_embedded_preview_decodes_at_half_size(self, imread: mock.MagicMock) -> None:
        """A file without a preview is LibRaw's default rendering at half size."""
        import rawpy
        raw = imread.return_value.__enter__.return_value
        raw.extract_thumb.side_effect = rawpy.LibRawNoThumbnailError()
        raw.postprocess.return_value = self.IMAGE[:, :, ::-1]

        preview = CameraRawSource("camera.nef").thumbnail()

        assert np.array_equal(preview, self.IMAGE[:, :, ::-1])
        raw.postprocess.assert_called_once_with(half_size=True, output_bps=8)
