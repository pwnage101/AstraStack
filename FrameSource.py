from __future__ import annotations
from lazy import lazy
cv2 = lazy("cv2")
np = lazy("numpy")
rawpy = lazy("rawpy")
av = lazy("av")
pypng = lazy("png")
import os
from abc import ABC, abstractmethod
import dataclasses
import struct
import warnings
from dataclasses import dataclass
from functools import cached_property
from typing import NamedTuple, override
from natsort import natsorted, ns
import ColorSpace
from Constants import (
    AUTO,
    LIBRAW_EXTENSIONS,
    UNSPECIFIED,
    DemosaicAlgorithm,
    DemosaicMode,
    DemosaicParams,
    MosaicPattern,
)

@dataclass(frozen=True)
class LinearizeParams:
    """What a source still has to undo after reading a frame to get linear 0-1 working data. Private to this module."""
    transfer: ColorSpace.Transfer | None = None  # ("H273", 1)
    bayerPattern: MosaicPattern | None = None  # MosaicPattern.RGGB, as the file states it
    # black level for each output channel (1-tuple for grayscale, 3-tuple for RGB).
    blackLevel: tuple[float] | tuple[float, float, float] | None = None
    whiteLevel: float | None = None  # 16383.0
    # What the source multiplied the sensor data by before demosaicing (as-shot white balance, and LibRaw's scale
    # to 16 bits), one per channel (RGB); (125.2, 64.1, 108.4)
    asShotMultipliers: tuple[float, float, float] | None = None
    # The D65 reference multipliers, normalized to green; the sensor data is rebalanced to them after demosaicing
    daylightMultipliers: tuple[float, float, float] | None = None  # (2.17, 1.0, 1.49)

class StoredFrame(NamedTuple):
    """A frame as stored, and how to linearize it."""
    image: np.ndarray
    linearizeParams: LinearizeParams
    workingSpace: ColorSpace.WorkingSpace

class Frame(NamedTuple):
    """A frame as working data."""
    image: np.ndarray  # Linear 0-1 float32 BGR
    mismatch: bool  # Whether the frame is described differently from the source's first, which it is linearized like

def linearize(img: np.ndarray, linearizeParams: LinearizeParams) -> np.ndarray:
    """Converts stored image values to linear 0-1 working data.

    Args:
        img: The stored (integer, or float for FITS) image, gray or with 3 channels (BGR).
        linearizeParams: How the values are encoded.

    Returns:
        The linear float32 image. Sensor data has its black level mapped to 0 and its white level to 1 per channel,
        unclipped: noise below black stays negative. Sensor data with multipliers stays balanced as shot, or is
        balanced to the D65 reference when the daylight multipliers are given.
    """
    if(linearizeParams.whiteLevel is not None):
        black = (
            np.float32(0)
            if linearizeParams.blackLevel is None
            else np.array(linearizeParams.blackLevel[::-1], dtype=np.float32)
        )
        if(img.ndim == 2):
            black = np.float32(np.mean(black))
        img = img.astype(np.float32)
        if(linearizeParams.asShotMultipliers is None):
            return ((img - black)/(np.float32(linearizeParams.whiteLevel) - black)).astype(np.float32)
        # The image is in the as-shot balance: its black and white levels carry each channel's multiplier
        multipliers = np.array(linearizeParams.asShotMultipliers[::-1], dtype=np.float32)
        asShot = multipliers/multipliers[1]
        linear = (img/multipliers[1] - black*asShot)/(np.float32(linearizeParams.whiteLevel) - black)
        if(linearizeParams.daylightMultipliers is not None):
            daylight = np.array(linearizeParams.daylightMultipliers[::-1], dtype=np.float32)
            linear = linear*(daylight/asShot)
        return linear.astype(np.float32)
    if(np.issubdtype(img.dtype, np.floating)):
        return ColorSpace.transferToLinear(x=img.astype(np.float32), transfer=linearizeParams.transfer)
    return ColorSpace.linearizeInteger(img=img, transfer=linearizeParams.transfer)

class FrameSource(ABC):
    """A sequence of frames stored in one or more files, addressed by integer index.

    Sources are sent to worker processes, so they must pickle as just what identifies their
    frames and open files lazily wherever they are read.
    """

    # The display name of the library that demosaics the frames.
    DEMOSAIC_LIBRARY_NAME: str

    # The demosaic params to actually send to the demosaicing library. None to disable demosaicing.
    demosaicParams: DemosaicParams | None = None

    @dataclass(frozen=True)
    class FrameFacts:
        """Hints about the inputs derived from one frame. Every frame is linearized like the first one, so the
        images of a sequence that are tagged differently are all linearized alike.
        """
        linearizeParams: LinearizeParams
        workingSpace: ColorSpace.WorkingSpace

    @classmethod
    def open(cls, path: str | list[str]) -> FrameSource:
        """Opens the chosen file or files as the one kind of source that can open them.

        Args:
            path: A file, or a list of files in any order.

        Returns:
            The source.

        Raises:
            ValueError: If no file was chosen or no kind of source can open the files; the
                message is meant for the user.
        """
        if(len(path) == 0):
            raise ValueError("No file was chosen.")
        for source in (CameraRawSource, FitsSource, ImageSequenceSource, VideoSource):
            if(source.canOpen(path)):
                return source(path)
        raise ValueError("The chosen files can't be opened. Choose one video, or one or more images.")

    @staticmethod
    @abstractmethod
    def canOpen(path: str | list[str]) -> bool:
        """Returns whether this kind of source can open the chosen file or files.

        Files are recognized by their contents rather than their extensions. Kinds of source
        don't overlap, so at most one can open any choice.

        Args:
            path: A file, or a list of files in any order.

        Returns:
            Whether this kind of source can open them.
        """

    @abstractmethod
    def filePaths(self) -> list[str]:
        """Returns the files the source reads.

        Returns:
            The files, in frame order.
        """

    @abstractmethod
    def frameCount(self) -> int:
        """Returns the number of frames in the source.

        Returns:
            The number of frames; indices run from 0.
        """

    @abstractmethod
    def frameSize(self) -> tuple[int, int]:
        """Returns the size of the frames.

        Returns:
            The (width, height) of the first frame.
        """

    @abstractmethod
    def thumbnail(self) -> np.ndarray:
        """Makes a display-referred thumbnail of the source, leaving its demosaic params as they are.

        The thumbnail is shown as sRGB as it is, so it only needs to look close to the first frame, as a camera's own
        rendering does.

        Returns:
            The thumbnail as uint8 RGB, possibly at a reduced size.
        """

    @abstractmethod
    def readFrame(self, index: int, atRecordedDepth: bool = False) -> Frame | None:
        """Reads one frame as linear working data, demosaiced as the source's demosaic params say.

        Every frame is linearized like the first one.

        Args:
            index: The frame's index.
            atRecordedDepth: Whether to round frames the source widened back to the depth they were recorded at,
                which the thresholds on measurements were tuned on. Ignored by sources that don't widen frames.

        Returns:
            The frame, or None if the source has no readable frame at index.
        """

    @property
    @abstractmethod
    def firstFrameFacts(self) -> FrameSource.FrameFacts:
        """What the first frame says about the source, read from it the first time it is needed."""

    def differsFromFirst(self, stored: StoredFrame) -> bool:
        """Returns whether a stored frame is described differently from the source's first."""
        first = self.firstFrameFacts
        return (stored.linearizeParams, stored.workingSpace) != (first.linearizeParams, first.workingSpace)

    def frameWorkingSpace(self) -> ColorSpace.WorkingSpace:
        """Returns the working space of the frames readFrame returns, demosaiced as the source's demosaic params say."""
        return self._workingSpaceFor(self.demosaicParams)

    @abstractmethod
    def _workingSpaceFor(self, demosaicParams: DemosaicParams | None) -> ColorSpace.WorkingSpace:
        """Returns the working space of the frames demosaiced as some demosaic params say.

        Args:
            demosaicParams: The demosaic params.

        Returns:
            The working space.
        """

    @abstractmethod
    def supportedDemosaicModes(self) -> list[DemosaicMode]:
        """Returns the demosaic modes the source supports besides AUTO, which every source does."""

    @abstractmethod
    def supportedMosaicPatterns(self) -> list[MosaicPattern]:
        """Returns the patterns the source supports besides AUTO, which every source does."""

    @abstractmethod
    def supportedDemosaicAlgorithms(self) -> list[DemosaicAlgorithm]:
        """Returns the algorithms the source supports besides AUTO, which every source does."""

    @abstractmethod
    def autoDemosaicParams(self, rememberedPattern: MosaicPattern | None = None) -> DemosaicParams | None:
        """Returns the demosaic params Auto resolves to.

        Args:
            rememberedPattern: The pattern the user last picked, which breaks a pixel guess's tie; without one, the
                guessed family's first member is used.

        Returns:
            The demosaic params, or None if Auto doesn't demosaic the frames.
        """

    def setDemosaicChoices(
        self,
        mode: DemosaicMode | AUTO,
        pattern: MosaicPattern | AUTO,
        algorithm: DemosaicAlgorithm | AUTO,
        rememberedPattern: MosaicPattern | None = None,
    ) -> DemosaicParams | None:
        """Configures this source to read frames with the user's demosaic choices.

        A choice the source doesn't support is taken as AUTO.

        Args:
            mode: The demosaic mode, or AUTO.
            pattern: The pattern, or AUTO. Ignored unless the mode is On.
            algorithm: The algorithm, or AUTO. Ignored unless the mode is On.
            rememberedPattern: The pattern the user last picked, which breaks a pixel guess's tie.

        Returns:
            The resolved demosaic params; None if the frames aren't demosaiced.
        """
        if(mode not in self.supportedDemosaicModes()):
            mode = AUTO
        demosaicParams = None
        if(mode is AUTO):
            demosaicParams = self.autoDemosaicParams(rememberedPattern)
        elif(mode == DemosaicMode.ON):
            fallbackParams = (
                self.autoDemosaicParams(rememberedPattern)
                # If we absolutely MUST demosaic but auto detection failed, then hard-code RGGB + Edge Aware.
                or DemosaicParams(MosaicPattern.RGGB, DemosaicAlgorithm.EDGE_AWARE)
            )
            demosaicParams = DemosaicParams(
                pattern if pattern in self.supportedMosaicPatterns() else fallbackParams.pattern,
                algorithm if algorithm in self.supportedDemosaicAlgorithms() else fallbackParams.algorithm,
            )
        self.demosaicParams = demosaicParams
        return demosaicParams

    def demosaicParamsDisplayNames(self) -> tuple[str, str, str]:
        """Names the demosaic mode, pattern and algorithm the frames are read with, for display.

        Returns:
            The names of the mode, pattern and algorithm, e.g. ("On", "RGGB", "Edge-aware"). All three are "Off" if
            the frames aren't demosaiced. The library's name stands for a choice it makes itself.
        """
        if(self.demosaicParams is None):
            return (DemosaicMode.OFF.value, "Off", "Off")
        pattern, algorithm = (
            self.DEMOSAIC_LIBRARY_NAME if choice is UNSPECIFIED else choice.value
            for choice in (self.demosaicParams.pattern, self.demosaicParams.algorithm)
        )
        return (DemosaicMode.ON.value, pattern, algorithm)

    def close(self) -> None:
        """Releases any open file handles. The source reopens them if it is read again."""

class OpenCVDemosaicSource(FrameSource):
    """A source whose stored frames may be a mosaic, which OpenCV demosaics.

    Auto demosaics gray frames whose file states a pattern, or that the pixel guess finds a mosaic in, with
    Edge-aware.
    """

    DEMOSAIC_LIBRARY_NAME = "OpenCV"

    # The cv2 color conversion code of each pattern and algorithm OpenCV
    # demosaics. Stored as strings to avoid triggering the lazy cv2 import.
    OPENCV_DEMOSAIC_CODES = {
        DemosaicParams(MosaicPattern.RGGB, DemosaicAlgorithm.BILINEAR): "COLOR_BAYER_BG2BGR",
        DemosaicParams(MosaicPattern.GRBG, DemosaicAlgorithm.BILINEAR): "COLOR_BAYER_GB2BGR",
        DemosaicParams(MosaicPattern.GBRG, DemosaicAlgorithm.BILINEAR): "COLOR_BAYER_GR2BGR",
        DemosaicParams(MosaicPattern.BGGR, DemosaicAlgorithm.BILINEAR): "COLOR_BAYER_RG2BGR",
        DemosaicParams(MosaicPattern.RGGB, DemosaicAlgorithm.EDGE_AWARE): "COLOR_BAYER_BG2BGR_EA",
        DemosaicParams(MosaicPattern.GRBG, DemosaicAlgorithm.EDGE_AWARE): "COLOR_BAYER_GB2BGR_EA",
        DemosaicParams(MosaicPattern.GBRG, DemosaicAlgorithm.EDGE_AWARE): "COLOR_BAYER_GR2BGR_EA",
        DemosaicParams(MosaicPattern.BGGR, DemosaicAlgorithm.EDGE_AWARE): "COLOR_BAYER_RG2BGR_EA",
        DemosaicParams(MosaicPattern.RGGB, DemosaicAlgorithm.VNG): "COLOR_BAYER_BG2BGR_VNG",
        DemosaicParams(MosaicPattern.GRBG, DemosaicAlgorithm.VNG): "COLOR_BAYER_GB2BGR_VNG",
        DemosaicParams(MosaicPattern.GBRG, DemosaicAlgorithm.VNG): "COLOR_BAYER_GR2BGR_VNG",
        DemosaicParams(MosaicPattern.BGGR, DemosaicAlgorithm.VNG): "COLOR_BAYER_RG2BGR_VNG",
    }

    # The pixel guess can tell a mosaic's family but not which member: swapping red and blue looks the same. Unless
    # the user last picked the other member, each family's first is used.
    MOSAIC_FAMILIES = ((MosaicPattern.RGGB, MosaicPattern.BGGR), (MosaicPattern.GRBG, MosaicPattern.GBRG))

    @dataclass(frozen=True)
    class FrameFacts(FrameSource.FrameFacts):
        """Hints about the inputs derived from one frame, for OpenCV's demosaic."""
        isGray: bool  # Whether its channels are equal, so it may be a mosaic
        is8Bit: bool  # Whether it was recorded at 8 bits
        # The family guessMosaicFamily finds, if the frame is gray and its file states no pattern
        mosaicFamily: tuple[MosaicPattern, MosaicPattern] | None

    @abstractmethod
    def readStored(self, index: int) -> StoredFrame | None:
        """Reads one frame as stored, with its own description.

        Args:
            index: The frame's index.

        Returns:
            The frame in BGR order, or with a single channel for gray or Bayer mosaic frames,
            at its stored depth (uint8 or uint16, or float32 for float FITS data); or None if
            the source has no readable frame at index.
        """

    @cached_property
    @override
    def firstFrameFacts(self) -> OpenCVDemosaicSource.FrameFacts:
        """The first frame is read as stored, which also tells whether it may be a mosaic and its recorded depth."""
        stored = self.readStored(0)
        if(stored is None):
            # Without a first frame to judge by, the frames are read as untagged color, as they are when no mosaic is
            # found
            return OpenCVDemosaicSource.FrameFacts(
                linearizeParams=LinearizeParams(transfer=ColorSpace.SRGB_CURVE),
                workingSpace=ColorSpace.SRGB,
                isGray=False,
                is8Bit=False,
                mosaicFamily=None,
            )
        isGray = OpenCVDemosaicSource.isGray(stored.image)
        guess = isGray and stored.linearizeParams.bayerPattern is None
        return OpenCVDemosaicSource.FrameFacts(
            linearizeParams=stored.linearizeParams,
            workingSpace=stored.workingSpace,
            isGray=isGray,
            is8Bit=stored.image.dtype == np.uint8 or self._recordedBits() == 8,
            mosaicFamily=OpenCVDemosaicSource.guessMosaicFamily(stored.image) if guess else None,
        )

    @staticmethod
    def isGray(image: np.ndarray) -> bool:
        """Returns whether a frame has equal channels."""
        return image.ndim == 2 or (np.array_equal(image[:, :, 0], image[:, :, 1]) and
                                   np.array_equal(image[:, :, 1], image[:, :, 2]))

    def _recordedBits(self) -> int | None:
        """Returns the bit depth the frames were recorded at, when readStored widens them.

        Returns:
            The recorded depth, or None if frames are read at the depth they were recorded at.
        """
        return None

    @override
    def supportedDemosaicModes(self) -> list[DemosaicMode]:
        """A gray frame may be a mosaic, a color one is not."""
        return list(DemosaicMode) if self.firstFrameFacts.isGray else []

    @override
    def supportedMosaicPatterns(self) -> list[MosaicPattern]:
        """Any 2×2 pattern for gray frames."""
        return list(MosaicPattern) if self.firstFrameFacts.isGray else []

    @override
    def supportedDemosaicAlgorithms(self) -> list[DemosaicAlgorithm]:
        """OpenCV's algorithms for gray frames; VNG only works on 8-bit recordings."""
        first = self.firstFrameFacts
        if(not first.isGray):
            return []
        vng = [DemosaicAlgorithm.VNG] if first.is8Bit else []
        return [DemosaicAlgorithm.BILINEAR, DemosaicAlgorithm.EDGE_AWARE, *vng]

    @override
    def autoDemosaicParams(self, rememberedPattern: MosaicPattern | None = None) -> DemosaicParams | None:
        """Gray frames are a mosaic of the pattern their file states, else when the pixel guess finds one, of the
        family member the user last picked or else the family's first (RGGB or GRBG); the algorithm is Edge-aware.
        """
        first = self.firstFrameFacts
        if(first.isGray and first.linearizeParams.bayerPattern is not None):
            pattern = first.linearizeParams.bayerPattern
        elif(first.mosaicFamily is not None):
            pattern = rememberedPattern if rememberedPattern in first.mosaicFamily else first.mosaicFamily[0]
        else:
            return None
        return DemosaicParams(pattern, DemosaicAlgorithm.EDGE_AWARE)

    @staticmethod
    def guessMosaicFamily(image: np.ndarray) -> tuple[MosaicPattern, MosaicPattern] | None:
        """Guesses from a gray frame's pixels whether it is a mosaic.

        A very simple guess, far from perfect: demosaicing a mosaic reveals a pattern. It is only told apart as
        RGGB/BGGR or GRBG/GBRG.

        Args:
            image: A frame with equal channels, as readStored returns it.

        Returns:
            The family the pattern is in, or None if the frame isn't a mosaic.
        """
        if(image.dtype not in (np.uint8, np.uint16)):
            # Float frames are guessed from a 16-bit copy, since demosaicing needs integers
            image = np.around(np.clip(image/max(float(image.max()), 1e-12), 0, 1)*65535).astype(np.uint16)
        rggb, grbg, gbrg, bggr = (
            OpenCVDemosaicSource.demosaic(
                img=image,
                demosaicParams=DemosaicParams(pattern, DemosaicAlgorithm.EDGE_AWARE),
            )
            for pattern in (MosaicPattern.RGGB, MosaicPattern.GRBG, MosaicPattern.GBRG, MosaicPattern.BGGR)
        )
        diff1 = np.average(cv2.absdiff(cv2.cvtColor(rggb, cv2.COLOR_BGR2GRAY), cv2.cvtColor(bggr, cv2.COLOR_BGR2GRAY)))
        diff2 = np.average(cv2.absdiff(cv2.cvtColor(grbg, cv2.COLOR_BGR2GRAY), cv2.cvtColor(gbrg, cv2.COLOR_BGR2GRAY)))
        diff = 0
        if(diff1 > 0.0001 and diff2 > 0.0001):
            diff = abs(diff1 - diff2)/min(diff1, diff2)
        if(diff <= 0.2):
            return None
        # Probably uses a bayer pattern, but which family...
        rggbDiff = np.average(cv2.absdiff(rggb[:,:,0], rggb[:,:,2]))
        grbgDiff = np.average(cv2.absdiff(grbg[:,:,0], grbg[:,:,2]))
        return OpenCVDemosaicSource.MOSAIC_FAMILIES[0 if rggbDiff >= grbgDiff else 1]

    @staticmethod
    def demosaic(img: np.ndarray, demosaicParams: DemosaicParams | None) -> np.ndarray:
        """Demosaics a frame with OpenCV, keeping its depth.

        VNG demosaicing is only done on uint8 frames; deeper frames are demosaiced edge-aware.

        Args:
            img: The frame as readStored returns it; a mosaic if demosaicParams isn't None.
            demosaicParams: The pattern and algorithm, which are not UNSPECIFIED; None to leave the frame as it is.

        Returns:
            The frame as BGR.
        """
        if(demosaicParams is None):
            return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img
        h, w = img.shape[:2]
        if(img.ndim == 3):
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if(demosaicParams.algorithm == DemosaicAlgorithm.VNG and img.dtype != np.uint8):
            demosaicParams = DemosaicParams(demosaicParams.pattern, DemosaicAlgorithm.EDGE_AWARE)
        code = getattr(cv2, OpenCVDemosaicSource.OPENCV_DEMOSAIC_CODES[demosaicParams])
        if(demosaicParams.algorithm == DemosaicAlgorithm.VNG):
            img = cv2.copyMakeBorder(img, 4, 2, 2, 2, cv2.BORDER_REPLICATE)
        else:
            img = cv2.copyMakeBorder(img, 2, 2, 2, 2, cv2.BORDER_REPLICATE)
        return cv2.cvtColor(img, code)[2:h+2, 2:w+2]

    @override
    def readFrame(self, index: int, atRecordedDepth: bool = False) -> Frame | None:
        """Demosaiced frames are sensor data: the container's curve and primaries are ignored, and the levels the file
        states are kept, else black is 0 and white is the type's maximum.
        """
        stored = self.readStored(index)
        if(stored is None):
            return None
        image = self.toWorkingData(
            image=stored.image,
            demosaicParams=self.demosaicParams,
            atRecordedDepth=atRecordedDepth,
        )
        return Frame(image, self.differsFromFirst(stored))

    @override
    def thumbnail(self) -> np.ndarray:
        """The first frame, demosaiced as Auto resolves."""
        demosaicParams = self.autoDemosaicParams()
        image = self.toWorkingData(image=self.readStored(0).image, demosaicParams=demosaicParams)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return ColorSpace.displayImage(img=image, workingSpace=self._workingSpaceFor(demosaicParams))

    def toWorkingData(
        self,
        image: np.ndarray,
        demosaicParams: DemosaicParams | None,
        atRecordedDepth: bool = False,
    ) -> np.ndarray:
        """Turns a stored image into linear working data, linearized like the first frame.

        Args:
            image: The image as readStored returns it.
            demosaicParams: The demosaic params; None if the image isn't demosaiced.
            atRecordedDepth: Whether to round an image readStored widened back to the depth it was recorded at.

        Returns:
            The linear float32 BGR image.
        """
        linearizeParams = self.firstFrameFacts.linearizeParams
        if(image.dtype == np.uint16 and self._recordedBits() == 8 and (atRecordedDepth or demosaicParams is not None)):
            # Demosaic 8-bit video as 8-bit (VNG only works on uint8)
            image = np.around(image/257).astype(np.uint8)
        if(demosaicParams is None):
            image = OpenCVDemosaicSource.demosaic(img=image, demosaicParams=None)
        else:
            image, linearizeParams = OpenCVDemosaicSource.sensorData(
                image=image,
                linearizeParams=linearizeParams,
                demosaicParams=demosaicParams,
            )
        return linearize(img=image, linearizeParams=linearizeParams)

    @staticmethod
    def sensorData(
        image: np.ndarray,
        linearizeParams: LinearizeParams,
        demosaicParams: DemosaicParams,
    ) -> tuple[np.ndarray, LinearizeParams]:
        """Demosaics stored values as a mosaic of sensor data, with no curve and no primaries.

        Black is 0, since no source OpenCV demosaics states a black level. White is the level the file states (FITS),
        else the type's maximum.

        Args:
            image: The stored image (BGR, or gray).
            linearizeParams: The stored values' linearize params.
            demosaicParams: The pattern and algorithm, which are not UNSPECIFIED.

        Returns:
            The demosaiced image (BGR) and its linearize params.
        """
        white = linearizeParams.whiteLevel
        if(white is None):
            white = float(np.iinfo(image.dtype).max) if np.issubdtype(image.dtype, np.integer) else 1.0
        if(image.dtype not in (np.uint8, np.uint16)):
            # OpenCV only demosaics integers
            image = np.around(np.clip(image*(65535/white), 0, 65535)).astype(np.uint16)
            white = 65535.0
        image = OpenCVDemosaicSource.demosaic(img=image, demosaicParams=demosaicParams)
        return (image, LinearizeParams(whiteLevel=white))

    @override
    def _workingSpaceFor(self, demosaicParams: DemosaicParams | None) -> ColorSpace.WorkingSpace:
        """The first frame's working space, or one without primaries when the frames are demosaiced, since they are
        sensor data.
        """
        return self.firstFrameFacts.workingSpace if demosaicParams is None else ColorSpace.UNTAGGED

class ImageSequenceSource(OpenCVDemosaicSource):
    """A list of image files, one frame per file."""

    def __init__(self, path: str | list[str]) -> None:
        """Initializes the source.

        Args:
            path: An image file, or a list of image files (in any order).
        """
        self.paths = [path] if isinstance(path, str) else natsorted(path, alg=ns.IGNORECASE)

    @staticmethod
    @override
    def canOpen(path: str | list[str]) -> bool:
        """Accepts an image file OpenCV reads, or a list of one or more of them, but not raw camera files.

        Each file must be recognized without decoding its pixels.
        """
        paths = [path] if isinstance(path, str) else path
        return len(paths) > 0 and all(cv2.haveImageReader(file) and not CameraRawSource.isRaw(file) for file in paths)

    @override
    def filePaths(self) -> list[str]:
        return self.paths

    @override
    def frameCount(self) -> int:
        return len(self.paths)

    @override
    def readStored(self, index: int) -> StoredFrame | None:
        """Each file is described by its own color tags."""
        if(index >= len(self.paths)):
            return None
        file = self.paths[index]
        # Any depth and channel count, without alpha and with the EXIF orientation applied
        img = cv2.imread(file, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR)
        linearizeParams, workingSpace = ImageSequenceSource.describeImage(file=file)
        return StoredFrame(ImageSequenceSource.toSupportedDtype(img), linearizeParams, workingSpace)

    @staticmethod
    def toSupportedDtype(img: np.ndarray) -> np.ndarray:
        """Returns image data as uint8, uint16 or float32, the types the pipeline handles.

        Other integer types are scaled from their maximum to 0-1 floats; floats are taken as 0-1 already.

        Args:
            img: The image as read.

        Returns:
            The image as uint8, uint16 or float32.
        """
        if(img.dtype in (np.uint8, np.uint16)):
            return img
        if(np.issubdtype(img.dtype, np.integer)):
            return (img.astype(np.float64)/np.iinfo(img.dtype).max).astype(np.float32)
        return img.astype(np.float32)

    @override
    def frameSize(self) -> tuple[int, int]:
        height, width = self.readStored(0).image.shape[:2]
        return (width, height)

    @staticmethod
    def pngChunks(file: str) -> list[tuple[bytes, bytes]]:
        """Returns the chunks of a PNG file before its first IDAT.

        Args:
            file: Path of the file.

        Returns:
            The chunks as (type, data), empty when the file isn't a PNG.
        """
        chunks = []
        try:
            for chunkType, data in pypng.Reader(filename=file).chunks():
                if(chunkType == b"IDAT"):
                    break
                chunks.append((chunkType, data))
        except pypng.Error:
            pass
        return chunks

    @staticmethod
    def fromPNGChunks(
        chunks: list[tuple[bytes, bytes]],
    ) -> tuple[ColorSpace.Transfer | None, ColorSpace.WorkingSpace] | None:
        """Describes a PNG from its color chunks, with the precedence cICP > sRGB > cHRM/gAMA; iCCP is ignored.

        Args:
            chunks: The PNG's chunks before IDAT.

        Returns:
            The transfer curve to undo and the working space, or None when the PNG is untagged. A cHRM without gAMA
            has the sRGB curve.
        """
        chunks = dict(chunks)
        if(b"cICP" in chunks):
            return ColorSpace.fromCICP(primaries=chunks[b"cICP"][0], transfer=chunks[b"cICP"][1])
        if(b"sRGB" in chunks):
            return (ColorSpace.SRGB_CURVE, ColorSpace.SRGB)
        if(b"gAMA" in chunks or b"cHRM" in chunks):
            workingSpace = ColorSpace.UNTAGGED
            if(b"cHRM" in chunks):
                v = np.array(struct.unpack(">8I", chunks[b"cHRM"][:32]))/100000
                workingSpace = ColorSpace.fromPrimaries(primaries=v[2:8].reshape(3, 2), white=v[0:2])
            transfer = ColorSpace.SRGB_CURVE
            if(b"gAMA" in chunks):
                gAMA = struct.unpack(">I", chunks[b"gAMA"][:4])[0]
                if(gAMA == 45455 and workingSpace.toXYZ is None):
                    return (ColorSpace.SRGB_CURVE, ColorSpace.SRGB)
                elif(gAMA == 45455 and workingSpace == ColorSpace.SRGB):
                    transfer = ColorSpace.SRGB_CURVE
                elif(gAMA == 100000):
                    transfer = None
                elif(gAMA > 0):
                    transfer = ("Gamma", 100000/gAMA)
            return (transfer, workingSpace)
        return None

    @staticmethod
    def describeImage(file: str) -> tuple[LinearizeParams, ColorSpace.WorkingSpace]:
        """Describes a (non RAW, non FITS) image file from its color tags.

        Only PNG color chunks are read; ICC profiles and EXIF aren't, so JPEGs and TIFFs are always untagged.

        Args:
            file: Path of the file.

        Returns:
            The linearize params and working space; untagged files are sRGB whatever their type, as PNG's
            specification defaults them, and close to TIFF's default of gamma 2.2.
        """
        transfer, workingSpace = (ImageSequenceSource.fromPNGChunks(ImageSequenceSource.pngChunks(file))
                                  or (ColorSpace.SRGB_CURVE, ColorSpace.SRGB))
        return (LinearizeParams(transfer=transfer), workingSpace)

class CameraRawSource(FrameSource):
    """A list of camera RAW files, one frame per file, decoded and demosaiced by LibRaw."""

    DEMOSAIC_LIBRARY_NAME = "LibRaw"

    # LibRaw's names of the algorithms it offers in its standard build
    LIBRAW_DEMOSAIC_ALGORITHMS = {
        DemosaicAlgorithm.BILINEAR: "LINEAR",
        DemosaicAlgorithm.VNG: "VNG",
        DemosaicAlgorithm.PPG: "PPG",
        DemosaicAlgorithm.AHD: "AHD",
        DemosaicAlgorithm.DCB: "DCB",
        DemosaicAlgorithm.DHT: "DHT",
        DemosaicAlgorithm.AAHD: "AAHD",
    }

    @dataclass(frozen=True)
    class FrameFacts(FrameSource.FrameFacts):
        """Hints about the inputs derived from one frame, for LibRaw's demosaic."""
        sensorPattern: MosaicPattern | UNSPECIFIED | None  # The sensor's pattern, as readMosaicPattern reads it

    def __init__(self, path: str | list[str]) -> None:
        """Initializes the source.

        Args:
            path: A raw camera file, or a list of raw camera files (in any order).
        """
        self.paths = [path] if isinstance(path, str) else natsorted(path, alg=ns.IGNORECASE)

    @staticmethod
    def isRaw(file: str) -> bool:
        """Returns whether LibRaw recognizes a file, without decoding its pixels.

        Only a file with a raw camera extension is tried: LibRaw opens some other files, such as JPEGs from Nikon
        cameras, then fails to decode them.

        Args:
            file: Path of the file.

        Returns:
            Whether it is a raw camera file.
        """
        if(os.path.splitext(file)[1].lower() not in LIBRAW_EXTENSIONS):
            return False
        try:
            with rawpy.imread(file):
                return True
        except Exception:
            return False

    @staticmethod
    @override
    def canOpen(path: str | list[str]) -> bool:
        """Accepts a raw camera file, or a list of one or more of them."""
        paths = [path] if isinstance(path, str) else path
        return len(paths) > 0 and all(CameraRawSource.isRaw(file) for file in paths)

    @override
    def filePaths(self) -> list[str]:
        return self.paths

    @override
    def frameCount(self) -> int:
        return len(self.paths)

    @override
    def frameSize(self) -> tuple[int, int]:
        height, width = CameraRawSource.decodeRaw(file=self.paths[0]).image.shape[:2]
        return (width, height)

    @cached_property
    @override
    def firstFrameFacts(self) -> CameraRawSource.FrameFacts:
        """What the first file says about the source, read from its metadata without decoding."""
        with rawpy.imread(self.paths[0]) as raw:
            linearizeParams, workingSpace = CameraRawSource.fromRAW(raw)
            return CameraRawSource.FrameFacts(
                linearizeParams=linearizeParams,
                workingSpace=workingSpace,
                sensorPattern=CameraRawSource.readMosaicPattern(raw),
            )

    @staticmethod
    def readMosaicPattern(raw: rawpy.RawPy) -> MosaicPattern | UNSPECIFIED | None:
        """Reads the sensor's pattern from LibRaw's own description of it.

        Args:
            raw: The opened raw camera file.

        Returns:
            None if the sensor isn't a mosaic (monochrome, linear DNG, Foveon), UNSPECIFIED if its pattern isn't a
            2×2 one (X-Trans, 4-color sensors), else the pattern, e.g. RGGB.
        """
        if(raw.raw_pattern is None):
            return None
        letters = "".join(chr(raw.color_desc[i]) for i in raw.raw_pattern.flatten())
        return MosaicPattern.__members__.get(letters, UNSPECIFIED)

    @override
    def supportedDemosaicModes(self) -> list[DemosaicMode]:
        """LibRaw demosaics any mosaic, which can't be left alone."""
        return [DemosaicMode.ON] if self.firstFrameFacts.sensorPattern is not None else []

    @override
    def supportedMosaicPatterns(self) -> list[MosaicPattern]:
        """LibRaw reads the pattern from the file."""
        return []

    @override
    def supportedDemosaicAlgorithms(self) -> list[DemosaicAlgorithm]:
        """LibRaw has the algorithms for 2×2 Bayer mosaics."""
        isBayer = isinstance(self.firstFrameFacts.sensorPattern, MosaicPattern)
        return list(CameraRawSource.LIBRAW_DEMOSAIC_ALGORITHMS) if isBayer else []

    @override
    def autoDemosaicParams(self, rememberedPattern: MosaicPattern | None = None) -> DemosaicParams | None:
        """The frames are a mosaic when LibRaw says so, demosaiced with the algorithm LibRaw picks."""
        pattern = self.firstFrameFacts.sensorPattern
        return None if pattern is None else DemosaicParams(pattern, UNSPECIFIED)

    @override
    def readFrame(self, index: int, atRecordedDepth: bool = False) -> Frame | None:
        """LibRaw demosaics the frame, with the demosaic params' algorithm if they have one, else as it chooses."""
        if(index >= len(self.paths)):
            return None
        algorithm = UNSPECIFIED if self.demosaicParams is None else self.demosaicParams.algorithm
        stored = CameraRawSource.decodeRaw(file=self.paths[index], algorithm=algorithm)
        linear = linearize(img=stored.image, linearizeParams=self.firstFrameFacts.linearizeParams)
        return Frame(linear, self.differsFromFirst(stored))

    @override
    def _workingSpaceFor(self, demosaicParams: DemosaicParams | None) -> ColorSpace.WorkingSpace:
        """The camera's working space, whatever the demosaic params: LibRaw's demosaic keeps it."""
        return self.firstFrameFacts.workingSpace

    @override
    def thumbnail(self) -> np.ndarray:
        """The first file's embedded preview, turned upright as LibRaw would turn the image; or else, for a file
        without an RGB preview, LibRaw's default rendering at half size.
        """
        with rawpy.imread(self.paths[0]) as raw:
            preview = CameraRawSource.readEmbeddedPreview(raw)
            if(preview is None):
                # sRGB with LibRaw's gamma, already upright
                return raw.postprocess(half_size=True, output_bps=8)
            flip = raw.sizes.flip
        # Map LibRaw rotation codes to OpenCV ones.
        rotation = {3: cv2.ROTATE_180, 5: cv2.ROTATE_90_COUNTERCLOCKWISE, 6: cv2.ROTATE_90_CLOCKWISE}.get(flip)
        return preview if rotation is None else cv2.rotate(preview, rotation)

    @staticmethod
    def readEmbeddedPreview(raw: rawpy.RawPy) -> np.ndarray | None:
        """Reads a raw camera file's embedded preview, as stored, without turning it upright.

        Args:
            raw: The opened raw camera file.

        Returns:
            The preview as uint8 RGB, or None if the file has no RGB preview LibRaw can extract.
        """
        try:
            thumb = raw.extract_thumb()
        except (rawpy.LibRawNoThumbnailError, rawpy.LibRawUnsupportedThumbnailError):
            return None
        if(thumb.format == rawpy.ThumbFormat.BITMAP):
            # LibRaw also extracts some gray bitmaps
            return thumb.data if thumb.data.shape[2] == 3 else None
        # Ignores any orientation tag, since not every camera tags its preview's orientation
        flags = cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
        return cv2.cvtColor(cv2.imdecode(np.frombuffer(thumb.data, np.uint8), flags), cv2.COLOR_BGR2RGB)

    @staticmethod
    def decodeRaw(file: str, algorithm: DemosaicAlgorithm | UNSPECIFIED = UNSPECIFIED) -> StoredFrame:
        """Decodes a raw camera file with LibRaw.

        Args:
            file: Path of the file.
            algorithm: The demosaic algorithm; UNSPECIFIED lets LibRaw choose.

        Returns:
            The stored image (BGR), its linearize params and its working space.
        """
        options = {}
        if(algorithm is not UNSPECIFIED):
            libRawAlgorithmName = CameraRawSource.LIBRAW_DEMOSAIC_ALGORITHMS[algorithm]
            options["demosaic_algorithm"] = rawpy.DemosaicAlgorithm[libRawAlgorithmName]
        with rawpy.imread(file) as raw:
            # user_black only keeps the common black level in; user_cblack keeps the per-channel offsets in too.
            img = raw.postprocess(user_black=0,
                                  user_cblack=[0, 0, 0, 0],
                                  user_wb=list(CameraRawSource.asShotMultipliers(raw)),
                                  adjust_maximum_thr=0,
                                  output_color=rawpy.ColorSpace.raw,
                                  gamma=(1, 1),
                                  no_auto_bright=True,
                                  output_bps=16,
                                  **options)
            linearizeParams, workingSpace = CameraRawSource.fromRAW(raw)
        return StoredFrame(cv2.cvtColor(img, cv2.COLOR_RGB2BGR), linearizeParams, workingSpace)

    @staticmethod
    def rawRGBCam(raw: rawpy.RawPy) -> np.ndarray | None:
        """Returns the camera -> linear sRGB matrix of a RAW file, whose rows sum to 1.

        Args:
            raw: The opened raw camera file.

        Returns:
            The 3×3 matrix from LibRaw's color_matrix (DNG) or rgb_xyz_matrix (its per-model table), or None when
            LibRaw has neither.
        """
        colorMatrix = np.array(raw.color_matrix, dtype=np.float64)[:3, :3]
        xyzToCam = np.array(raw.rgb_xyz_matrix, dtype=np.float64)[:3, :3]
        if(np.any(colorMatrix)):
            return colorMatrix
        elif(np.any(xyzToCam)):
            camRGB = xyzToCam @ ColorSpace.SRGB.toXYZ
            camRGB /= camRGB.sum(axis=1, keepdims=True)
            return np.linalg.inv(camRGB)
        return None

    @staticmethod
    def asShotMultipliers(raw: rawpy.RawPy) -> np.ndarray:
        """Returns the camera's as-shot white balance multipliers, normalized to green.

        Args:
            raw: The opened raw camera file.

        Returns:
            The four multipliers (R, G, B, G2), with G2 = G when the file gives none (Fuji); all 1 when the file
            states no usable white balance.
        """
        multipliers = np.array(raw.camera_whitebalance, dtype=np.float64)
        if(multipliers[3] == 0):
            multipliers[3] = multipliers[1]
        if(np.any(multipliers <= 0)):
            return np.ones(4)
        return multipliers/multipliers[1]

    @staticmethod
    def fromRAW(raw: rawpy.RawPy) -> tuple[LinearizeParams, ColorSpace.WorkingSpace]:
        """Describes a RAW file decoded linear in camera RGB, balanced as shot and with its black level in.

        Args:
            raw: The opened raw camera file.

        Returns:
            The sensor levels, the as-shot multipliers LibRaw applied and the D65 reference multipliers, and the
            camera's working space; UNTAGGED with a warning and no reference multipliers when LibRaw has no matrix.
        """
        black = [float(v) for v in raw.black_level_per_channel]
        if(raw.num_colors == 3):
            # LibRaw indexes 3-color levels by color as R, G1, B, G2 (its color_desc "RGBG"),
            # regardless of the actual mosaic pattern in the file.
            r, g1, b, g2 = black[0:4]
            # Average the two greens, they're unlikely to be very different anyway.
            blackLevel = (r, (g1 + g2)/2, b)
        else:
            # The black levels supplied by LibRaw are either unsupported, so just fallback
            # to taking the average.
            blackLevel = (float(np.mean(black)),)
        asShot = CameraRawSource.asShotMultipliers(raw)
        # LibRaw scales the multipliers so that the smallest maps the white level to 16 bits
        applied = asShot/asShot.min()*65535/float(raw.white_level)
        linearizeParams = LinearizeParams(
            blackLevel=blackLevel,
            whiteLevel=float(raw.white_level),
            asShotMultipliers=tuple(float(v) for v in applied[:3]),
        )
        rgbCam = CameraRawSource.rawRGBCam(raw)
        if(rgbCam is not None):
            daylight = np.array(raw.daylight_whitebalance[:3], dtype=np.float64)
            linearizeParams = dataclasses.replace(
                linearizeParams,
                daylightMultipliers=tuple(float(v) for v in daylight/daylight[1]),
            )
        else:
            warnings.warn(
                message="LibRaw has no color matrix for this camera, its colors are shown as sRGB",
                category=ColorSpace.ColorWarning,
            )
            return (linearizeParams, ColorSpace.UNTAGGED)
        return (linearizeParams, ColorSpace.WorkingSpace(toXYZ=ColorSpace.SRGB.toXYZ @ rgbCam))

class FitsSource(ImageSequenceSource):
    """A list of FITS files with image data, one frame per file, read by astropy."""

    @staticmethod
    @override
    def canOpen(path: str | list[str]) -> bool:
        """Accepts a FITS file with image data, or a list of one or more of them."""
        paths = [path] if isinstance(path, str) else path
        if(len(paths) == 0):
            return False
        try:
            from astropy.io import fits
            for file in paths:
                with fits.open(file) as hdus:
                    if(not any(hdu.header.get("NAXIS", 0) > 0 for hdu in hdus)):
                        return False
        except Exception:
            return False
        return True

    @override
    def readStored(self, index: int) -> StoredFrame | None:
        """Frames are stored as float32 unless they are 8 or 16 bits; the file states the Bayer pattern and levels."""
        if(index >= len(self.paths)):
            return None
        from astropy.io import fits
        with fits.open(self.paths[index]) as hdus:
            hdu = next(hdu for hdu in hdus if hdu.data is not None)
            # astropy returns big-endian data, which OpenCV and the lookup tables would misread
            img = np.asarray(hdu.data)
            img = img.astype(img.dtype.newbyteorder("="))
            header = hdu.header
            if(img.ndim == 3):
                # Color planes (RGB) first
                img = np.ascontiguousarray(np.moveaxis(img[:3], 0, -1)[:, :, ::-1])
            linearizeParams = FitsSource.fromFITS(header=header, data=img)
        if(img.dtype not in (np.uint8, np.uint16)):
            img = img.astype(np.float32)
        return StoredFrame(img, linearizeParams, ColorSpace.UNTAGGED)

    @staticmethod
    def normalizeBayerPattern(
        pattern: str,
        xOffset: int = 0,
        yOffset: int = 0,
        bottomUp: bool = False,
        height: int = 0,
    ) -> MosaicPattern | None:
        """Normalizes a FITS Bayer pattern to the pattern at the first stored pixel.

        Args:
            pattern: BAYERPAT, e.g. "RGGB".
            xOffset: XBAYROFF, columns the pattern is shifted by.
            yOffset: YBAYROFF, rows the pattern is shifted by.
            bottomUp: Whether ROWORDER is BOTTOM-UP, so the pattern describes the image flipped vertically.
            height: Image height, used for bottom-up images.

        Returns:
            The 2×2 pattern of the stored data, or None when the pattern isn't a 2×2 RGB one.
        """
        pattern = pattern.strip().upper()
        if(len(pattern) != 4 or sorted(pattern) != ["B", "G", "G", "R"]):
            return None
        rows = [pattern[0:2], pattern[2:4]]
        if(yOffset % 2):
            rows = rows[::-1]
        if(xOffset % 2):
            rows = [row[::-1] for row in rows]
        if(bottomUp and height % 2 == 0):
            rows = rows[::-1]
        result = rows[0] + rows[1]
        if(result not in ("RGGB", "GRBG", "GBRG", "BGGR")):
            return None
        return MosaicPattern(result)

    @staticmethod
    def fromFITS(header: object, data: np.ndarray) -> LinearizeParams:
        """Describes FITS data as sensor data: its Bayer pattern and levels.

        Args:
            header: The FITS header (astropy.io.fits.Header or a dict).
            data: The (first frame's) data.

        Returns:
            The linearize params: no black level, since FITS has no standard keyword for one; white from DATAMAX or
            SATURATE, else the integer type's maximum, else 1.0 for float data within 0-1 and 65535 otherwise.
        """
        bayerPattern = None
        if("BAYERPAT" in header):
            bayerPattern = FitsSource.normalizeBayerPattern(
                pattern=str(header["BAYERPAT"]),
                xOffset=int(header.get("XBAYROFF", 0)),
                yOffset=int(header.get("YBAYROFF", 0)),
                bottomUp=str(header.get("ROWORDER", "")).strip().upper() == "BOTTOM-UP",
                height=data.shape[0],
            )
        if("DATAMAX" in header and float(header["DATAMAX"]) > 0):
            white = float(header["DATAMAX"])
        elif("SATURATE" in header and float(header["SATURATE"]) > 0):
            white = float(header["SATURATE"])
        elif(np.issubdtype(data.dtype, np.integer)):
            white = float(np.iinfo(data.dtype).max)
        elif(np.nanmin(data) >= 0 and np.nanmax(data) <= 1):
            white = 1.0
        else:
            white = 65535.0
        return LinearizeParams(bayerPattern=bayerPattern, whiteLevel=white)

class VideoSource(OpenCVDemosaicSource):
    """A video file decoded with PyAV, described by its stream's color tags.

    Frames are indexed by their position in presentation order and found by seeking to their
    timestamps. Reading a frame less than a second after the last one read decodes forward
    instead of seeking. The timestamps are read from the video's packets, without decoding,
    the first time they are needed, and pickled so worker processes don't read them again.
    """

    # CICP matrix coefficients codes, by the swscale matrix (an av.video.reformatter.Colorspace value) that decodes them
    YUV_MATRICES = {1: 1, 4: 4, 5: 5, 6: 5, 7: 7, 9: 9, 10: 9}

    def __init__(self, path: str) -> None:
        """Initializes the source.

        Args:
            path: The video file.
        """
        self.path = path
        self.timestamps: list[int] | None = None
        self.container: av.container.InputContainer | None = None

    def __getstate__(self) -> dict[str, object]:
        """Pickles the path, the timestamps, the first frame facts and the demosaic params, but not the open
        container, so each process opens its own.
        """
        keep = ("path", "timestamps", "firstFrameFacts", "demosaicParams")
        return {key: value for key, value in self.__dict__.items() if key in keep}

    def __setstate__(self, state: dict[str, object]) -> None:
        """Restores a pickled source, which opens its container when first read."""
        self.__dict__.update(state)
        self.container = None

    @staticmethod
    @override
    def canOpen(path: str | list[str]) -> bool:
        """Accepts a single file PyAV demuxes at least two video frames from.

        One-frame videos are rejected, and left to other FrameSource implementations to handle.
        """
        if(not isinstance(path, str)):
            return False
        try:
            with av.open(path) as container:
                stream = container.streams.video[0]
                packets = 0
                for packet in container.demux(stream):
                    packets += packet.size > 0
                    if(packets >= 2):
                        return True
        except Exception:
            pass
        return False

    @override
    def filePaths(self) -> list[str]:
        return [self.path]

    def openContainer(self) -> None:
        """Opens the video, if needed, and describes it from its stream's color tags."""
        if(self.container is not None):
            return
        self.container = av.open(self.path)
        self.stream = self.container.streams.video[0]
        codec = self.stream.codec_context
        self.width, self.height = codec.width, codec.height
        self.videoBits = 8
        formatName = ""
        isYUV = True
        if(codec.format is not None):
            formatName = codec.format.name
            self.videoBits = VideoSource.formatBits(codec.format)
            isYUV = not codec.format.is_rgb and len(codec.format.components) >= 3
        transfer, self.workingSpace = ColorSpace.fromVideoTags(primaries=codec.color_primaries,
                                                               transfer=codec.color_trc,
                                                               isYUV=isYUV)
        self.linearizeParams = LinearizeParams(transfer=transfer)
        self.yuvMatrix = VideoSource.YUV_MATRICES.get(codec.colorspace, 1) if isYUV else None
        self.limitedRange = VideoSource.isLimitedRange(colorRange=codec.color_range,
                                                       isYUV=isYUV,
                                                       fullRangeFormat=formatName.startswith("yuvj"))
        self.palette = VideoSource.readPalette(self.path) if formatName == "pal8" else None
        self.decoder = None
        self.lastTimestamp = None
        self.lastFrame = None

    @staticmethod
    def isLimitedRange(colorRange: int | None, isYUV: bool, fullRangeFormat: bool) -> bool:
        """Returns whether a stream's values are limited range, each missing tag falling back on its own.

        Args:
            colorRange: FFmpeg color range (1 = limited, 2 = full, 0 or None = unspecified).
            isYUV: Whether the pixel format is YUV (otherwise RGB or gray).
            fullRangeFormat: Whether the pixel format itself is full range (yuvj*).

        Returns:
            Whether the range is limited. Untagged YUV is limited, as the codecs' specifications default it;
            untagged RGB/gray is full range, as Windows takes bitmaps.
        """
        return colorRange == 1 or (isYUV and colorRange != 2 and not fullRangeFormat)

    @staticmethod
    def readPalette(path: str) -> np.ndarray:
        """Returns the palette of a pal8 video.

        Demuxers like AVI's attach the palette only to the first packet read, which opening the
        video already reads and a seek discards, so frames decoded after a seek have an empty
        palette. The first frame is decoded without seeking, in a container of its own.

        Args:
            path: The video file.

        Returns:
            The 256 entries as uint8 RGB.
        """
        with av.open(path) as container:
            videoFrame = next(container.decode(video=0))
        return np.frombuffer(videoFrame.planes[1], np.uint8).reshape(256, 4)[:, [2, 1, 0]]

    @staticmethod
    def formatBits(videoFormat: av.VideoFormat) -> int:
        """Returns the bit depth of a video pixel format.

        Args:
            videoFormat: The pixel format.

        Returns:
            The depth of one sample; Bayer formats, whose components report a pseudo-depth, use
            their name.
        """
        if(videoFormat.name.startswith("bayer_")):
            return 16 if "16" in videoFormat.name else 8
        return videoFormat.components[0].bits

    def frameTimestamps(self) -> list[int]:
        """Returns the presentation timestamps of the frames, read from the packets the first time.

        Returns:
            The sorted timestamps, in the stream's time base.
        """
        if(self.timestamps is None):
            self.openContainer()
            timestamps = set()
            for packet in self.container.demux(self.stream):
                timestamp = packet.pts if packet.pts is not None else packet.dts
                if(timestamp is not None and packet.size > 0):
                    timestamps.add(timestamp)
            self.decoder = None
            self.timestamps = sorted(timestamps)
        return self.timestamps

    @override
    def frameCount(self) -> int:
        """The count is the number of distinct timestamps among the video's packets."""
        return len(self.frameTimestamps())

    def frameToImage(self, videoFrame: av.VideoFrame) -> np.ndarray:
        """Converts a decoded frame to full range RGB, YUV with the stream's matrix and range.

        Args:
            videoFrame: The decoded frame.

        Returns:
            The full range uint16 BGR image.
        """
        if(self.yuvMatrix is not None):
            image = videoFrame.reformat(format="rgb48le",
                                        src_colorspace=self.yuvMatrix,
                                        src_color_range=1 if self.limitedRange else 2).to_ndarray()
        else:
            # Converted at the source's bit depth and widened exactly, so untagged RGB/gray loads unchanged
            sourceBits = 8 if VideoSource.formatBits(videoFrame.format) <= 8 else 16
            if(self.palette is not None):
                indices = np.frombuffer(videoFrame.planes[0], np.uint8).reshape(videoFrame.height, -1)
                image = self.palette[indices[:, :videoFrame.width]]
            else:
                image = videoFrame.reformat(format="rgb24" if sourceBits == 8 else "rgb48le").to_ndarray()
            if(sourceBits == 8):
                image = image.astype(np.uint16)*257
            if(self.limitedRange):
                scaled = (image.astype(np.float32)/65535*255 - 16)/219
                image = np.around(np.clip(scaled, 0, 1)*65535).astype(np.uint16)
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    @override
    def readStored(self, index: int) -> StoredFrame | None:
        """Frames are converted to 16-bit BGR, whatever the stream's depth."""
        if(index == 0 and self.timestamps is None):
            # Decoded in a container of its own, without reading every packet's timestamp, so thumbnails are quick
            self.openContainer()
            with av.open(self.path) as container:
                videoFrame = next(container.decode(video=0), None)
            if(videoFrame is None):
                return None
            return StoredFrame(
                image=self.frameToImage(videoFrame),
                linearizeParams=self.linearizeParams,
                workingSpace=self.workingSpace,
            )
        timestamps = self.frameTimestamps()
        if(index >= len(timestamps)):
            return None
        self.openContainer()
        target = timestamps[index]
        if(self.lastFrame is None or target != self.lastTimestamp):
            if(self.decoder is None or self.lastTimestamp is None or target < self.lastTimestamp or
               target - self.lastTimestamp > 1/self.stream.time_base):
                self.container.seek(target, stream=self.stream, backward=True)
                self.decoder = self.container.decode(self.stream)
            self.lastFrame = None
            for videoFrame in self.decoder:
                timestamp = videoFrame.pts if videoFrame.pts is not None else videoFrame.dts
                if(timestamp is not None and timestamp >= target):
                    self.lastTimestamp = timestamp
                    self.lastFrame = videoFrame
                    break
            if(self.lastFrame is None):
                self.decoder = None
                return None
        if(self.lastTimestamp != target):
            # The frame couldn't be decoded
            return None
        return StoredFrame(image=self.frameToImage(self.lastFrame),
                           linearizeParams=self.linearizeParams,
                           workingSpace=self.workingSpace)

    @override
    def _recordedBits(self) -> int | None:
        self.openContainer()
        return 8 if self.videoBits <= 8 else None

    @override
    def frameSize(self) -> tuple[int, int]:
        self.openContainer()
        return (self.width, self.height)

    @override
    def close(self) -> None:
        if(self.container is not None):
            self.container.close()
            self.container = None

