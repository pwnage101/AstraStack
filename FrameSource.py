from __future__ import annotations
from lazy import lazy
cv2 = lazy("cv2")
np = lazy("numpy")
rawpy = lazy("rawpy")
av = lazy("av")
pypng = lazy("png")
from abc import ABC, abstractmethod
import dataclasses
import struct
import warnings
from dataclasses import dataclass
from typing import NamedTuple
from natsort import natsorted, ns
import ColorSpace
from Constants import DemosaicAlgorithm, DemosaicIntent, DemosaicMode, MosaicPattern

# The pixel guess can tell a mosaic's family but not which member: swapping red and blue looks the same
MOSAIC_FAMILIES = ((MosaicPattern.RGGB, MosaicPattern.BGGR), (MosaicPattern.GRBG, MosaicPattern.GBRG))
# The patterns the user can choose, which are the 2×2 ones
MOSAIC_PATTERNS = tuple(pattern for family in MOSAIC_FAMILIES for pattern in family)

# OpenCV's names of the patterns and algorithms, in its COLOR_BAYER_<pattern>2BGR<algorithm> codes
OPENCV_PATTERNS = {
    MosaicPattern.RGGB: "BG",
    MosaicPattern.GRBG: "GB",
    MosaicPattern.GBRG: "GR",
    MosaicPattern.BGGR: "RG",
}
OPENCV_ALGORITHMS = {
    DemosaicAlgorithm.BILINEAR: "",
    DemosaicAlgorithm.EDGE_AWARE: "_EA",
    DemosaicAlgorithm.VNG: "_VNG",
}

@dataclass(frozen=True)
class DecodeParams:
    """What a source still has to undo after reading a frame to get linear 0-1 working data. Private to this module."""
    transfer: ColorSpace.Transfer | None = None  # ("H273", 1)
    bayerPattern: MosaicPattern | None = None  # MosaicPattern.RGGB, as the file states it
    # (1008.0, 1008.0, 1008.0), one per output channel (RGB), as the file states it
    blackLevel: tuple[float, ...] | None = None
    whiteLevel: float | None = None  # 16383.0
    # What the source multiplied the sensor data by before demosaicing (as-shot white balance, and LibRaw's scale
    # to 16 bits), one per channel (RGB); (125.2, 64.1, 108.4)
    asShotMultipliers: tuple[float, float, float] | None = None
    # The D65 reference multipliers, normalized to green; the sensor data is rebalanced to them after demosaicing
    daylightMultipliers: tuple[float, float, float] | None = None  # (2.17, 1.0, 1.49)

class StoredFrame(NamedTuple):
    """A frame as stored, and how to linearize it."""
    image: np.ndarray
    params: DecodeParams
    workingSpace: ColorSpace.WorkingSpace

class Frame(NamedTuple):
    """A frame as working data."""
    image: np.ndarray  # Linear 0-1 float32 BGR
    mismatch: bool  # Whether the frame is described differently from the source's first, which it is linearized like

@dataclass(frozen=True)
class FirstFrame:
    """What the first frame says about the whole source. Private to this module."""
    params: DecodeParams
    workingSpace: ColorSpace.WorkingSpace
    isGray: bool  # Whether its channels are equal, so it may be a mosaic
    is8Bit: bool  # Whether it was recorded at 8 bits

def linearize(img: np.ndarray, params: DecodeParams) -> np.ndarray:
    """Converts stored image values to linear 0-1 working data.

    Args:
        img: The stored (integer, or float for FITS) image, gray or with 3 channels (BGR).
        params: How the values are encoded.

    Returns:
        The linear float32 image. Sensor data has its black level mapped to 0 and its white level to 1 per channel,
        unclipped: noise below black stays negative. Sensor data with multipliers stays balanced as shot, or is
        balanced to the D65 reference when the daylight multipliers are given.
    """
    if(params.whiteLevel is not None):
        black = np.float32(0) if params.blackLevel is None else np.array(params.blackLevel[::-1], dtype=np.float32)
        if(img.ndim == 2):
            black = np.float32(np.mean(black))
        img = img.astype(np.float32)
        if(params.asShotMultipliers is None):
            return ((img - black)/(np.float32(params.whiteLevel) - black)).astype(np.float32)
        # The image is in the as-shot balance: its black and white levels carry each channel's multiplier
        multipliers = np.array(params.asShotMultipliers[::-1], dtype=np.float32)
        asShot = multipliers/multipliers[1]
        linear = (img/multipliers[1] - black*asShot)/(np.float32(params.whiteLevel) - black)
        if(params.daylightMultipliers is not None):
            daylight = np.array(params.daylightMultipliers[::-1], dtype=np.float32)
            linear = linear*(daylight/asShot)
        return linear.astype(np.float32)
    if(np.issubdtype(img.dtype, np.floating)):
        return ColorSpace.transferToLinear(x=img.astype(np.float32), transfer=params.transfer)
    return ColorSpace.linearizeInteger(img=img, transfer=params.transfer)

class FrameSource(ABC):
    """A sequence of frames stored in one or more files, addressed by integer index.

    Sources are sent to worker processes, so they must pickle as just what identifies their
    frames and open files lazily wherever they are read.
    """

    first: FirstFrame | None = None

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
    def readStored(self, index: int) -> StoredFrame | None:
        """Reads one frame as stored, with its own description.

        Args:
            index: The frame's index.

        Returns:
            The frame in BGR order, or with a single channel for gray or Bayer mosaic frames,
            at its stored depth (uint8 or uint16, or float32 for float FITS data); or None if
            the source has no readable frame at index.
        """

    @abstractmethod
    def frameSize(self) -> tuple[int, int]:
        """Returns the size of the frames.

        Returns:
            The (width, height) of the first frame.
        """

    def fastPreview(self, intent: DemosaicIntent | None) -> np.ndarray | None:
        """Reads a preview of the first frame, if the file offers a quick way to.

        Return None unless this is much quicker than reading the first frame; the caller then
        makes the preview from readFrame(0) itself.

        Args:
            intent: The resolved demosaic choice.

        Returns:
            The first frame like readFrame returns it, possibly at a reduced size; or None.
        """
        return None

    def firstFrame(self) -> FirstFrame:
        """Returns what the first frame says about the source, reading it the first time.

        Every frame is linearized like the first one, so the images of a sequence that are tagged differently are
        all decoded alike.

        Returns:
            The first frame's decode parameters, working space and kind.
        """
        if(self.first is None):
            stored = self.readStored(0)
            self.first = FirstFrame(
                params=stored.params,
                workingSpace=stored.workingSpace,
                isGray=FrameSource.isGray(stored.image),
                is8Bit=stored.image.dtype == np.uint8 or self._recordedBits() == 8,
            )
        return self.first

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

    def supportedDemosaicModes(self) -> list[DemosaicMode]:
        """Returns the demosaic modes the source supports: a gray frame may be a mosaic, a color one is not."""
        modes = [DemosaicMode.AUTO, DemosaicMode.OFF, DemosaicMode.ON]
        return modes if self.firstFrame().isGray else modes[:1]

    def supportedMosaicPatterns(self) -> list[MosaicPattern | None]:
        """Returns the patterns the source supports, None being Auto."""
        return [None, *MOSAIC_PATTERNS] if self.firstFrame().isGray else [None]

    def supportedDemosaicAlgorithms(self) -> list[DemosaicAlgorithm | None]:
        """Returns the algorithms the source supports, None being Auto; VNG only works on 8-bit recordings."""
        first = self.firstFrame()
        if(not first.isGray):
            return [None]
        vng = [DemosaicAlgorithm.VNG] if first.is8Bit else []
        return [None, DemosaicAlgorithm.BILINEAR, DemosaicAlgorithm.EDGE_AWARE, *vng]

    def resolveDemosaic(
        self,
        mode: DemosaicMode,
        pattern: MosaicPattern | None,
        algorithm: DemosaicAlgorithm | None,
        rememberedPattern: MosaicPattern | None = None,
    ) -> DemosaicIntent | None:
        """Resolves the user's demosaic choice for this source.

        A choice the source doesn't support is taken as Auto. Auto frames are a mosaic when the file states a
        pattern, else when the pixel guess finds one; the algorithm is Edge-aware.

        Args:
            mode: The demosaic mode.
            pattern: The pattern, or None for Auto. Ignored under Auto mode.
            algorithm: The algorithm, or None for Auto. Ignored under Auto mode.
            rememberedPattern: The pattern the user last picked, which breaks the pixel guess's tie.

        Returns:
            The intent the frames are demosaiced with, or None if they aren't demosaiced.
        """
        if(mode not in self.supportedDemosaicModes()):
            mode = DemosaicMode.AUTO
        if(mode == DemosaicMode.OFF):
            return None
        autoPattern = self.autoMosaicPattern(rememberedPattern)
        if(mode == DemosaicMode.AUTO):
            return None if autoPattern is None else DemosaicIntent(autoPattern, DemosaicAlgorithm.EDGE_AWARE)
        if(pattern not in self.supportedMosaicPatterns()):
            pattern = None
        if(algorithm not in self.supportedDemosaicAlgorithms()):
            algorithm = None
        return DemosaicIntent(
            pattern=pattern or autoPattern or MOSAIC_FAMILIES[0][0],
            algorithm=algorithm or DemosaicAlgorithm.EDGE_AWARE,
        )

    def autoMosaicPattern(self, rememberedPattern: MosaicPattern | None) -> MosaicPattern | None:
        """Returns the pattern of the source's frames under Auto, if they are a mosaic.

        Args:
            rememberedPattern: The pattern the user last picked.

        Returns:
            The pattern the file states, else the pixel guess's family member the user last picked or the family's
            first; or None if the frames aren't a gray mosaic.
        """
        first = self.firstFrame()
        if(not first.isGray):
            return None
        if(first.params.bayerPattern is not None):
            return first.params.bayerPattern
        family = FrameSource.guessMosaicFamily(self.readStored(0).image)
        if(family is None):
            return None
        return rememberedPattern if rememberedPattern in family else family[0]

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
            FrameSource.demosaic(img=image, intent=DemosaicIntent(pattern, DemosaicAlgorithm.EDGE_AWARE))
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
        return MOSAIC_FAMILIES[0] if rggbDiff >= grbgDiff else MOSAIC_FAMILIES[1]

    @staticmethod
    def demosaic(img: np.ndarray, intent: DemosaicIntent) -> np.ndarray:
        """Demosaics a frame with OpenCV, keeping its depth.

        VNG demosaicing is only done on uint8 frames; deeper frames are demosaiced edge-aware.

        Args:
            img: The mosaic, gray or with equal channels.
            intent: The pattern and algorithm, which are not UNSPECIFIED.

        Returns:
            The frame as BGR.
        """
        h, w = img.shape[:2]
        if(img.ndim == 3):
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        algorithm = intent.algorithm
        if(algorithm == DemosaicAlgorithm.VNG and img.dtype != np.uint8):
            algorithm = DemosaicAlgorithm.EDGE_AWARE
        code = getattr(cv2, f"COLOR_BAYER_{OPENCV_PATTERNS[intent.pattern]}2BGR{OPENCV_ALGORITHMS[algorithm]}")
        if(algorithm == DemosaicAlgorithm.VNG):
            img = cv2.copyMakeBorder(img, 4, 2, 2, 2, cv2.BORDER_REPLICATE)
        else:
            img = cv2.copyMakeBorder(img, 2, 2, 2, 2, cv2.BORDER_REPLICATE)
        return cv2.cvtColor(img, code)[2:h+2, 2:w+2]

    def readFrame(self, index: int, intent: DemosaicIntent | None, atRecordedDepth: bool = False) -> Frame | None:
        """Reads one frame as linear working data.

        Every frame is linearized like the first one. Demosaiced frames are sensor data: the container's curve
        and primaries are ignored, and the levels the file states are kept, else black is 0 and white is the
        type's maximum.

        Args:
            index: The frame's index.
            intent: The resolved demosaic choice; None if the frames aren't demosaiced.
            atRecordedDepth: Whether to round frames the source widened back to the depth they were recorded at,
                which the thresholds on measurements were tuned on.

        Returns:
            The frame, or None if the source has no readable frame at index.
        """
        stored = self.readStored(index)
        if(stored is None):
            return None
        first = self.firstFrame()
        image, params = stored.image, first.params
        if(image.dtype == np.uint16 and self._recordedBits() == 8 and (atRecordedDepth or intent is not None)):
            # Demosaic 8-bit video as 8-bit (VNG only works on uint8)
            image = np.around(image/257).astype(np.uint8)
        if(intent is None):
            if(image.ndim == 2):
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            image, params = FrameSource.sensorData(image=image, params=params, intent=intent)
        return Frame(linearize(img=image, params=params), self.differsFromFirst(stored))

    def differsFromFirst(self, stored: StoredFrame) -> bool:
        """Returns whether a stored frame is described differently from the source's first."""
        first = self.firstFrame()
        return (stored.params, stored.workingSpace) != (first.params, first.workingSpace)

    @staticmethod
    def sensorData(
        image: np.ndarray,
        params: DecodeParams,
        intent: DemosaicIntent,
    ) -> tuple[np.ndarray, DecodeParams]:
        """Demosaics stored values as a mosaic of sensor data, with no curve and no primaries.

        The levels the file states are kept, else black is 0 and white is the type's maximum.

        Args:
            image: The stored image (BGR, or gray).
            params: The stored values' decode parameters.
            intent: The pattern and algorithm, which are not UNSPECIFIED.

        Returns:
            The demosaiced image (BGR) and its decode parameters.
        """
        if(params.whiteLevel is None):
            white = float(np.iinfo(image.dtype).max) if np.issubdtype(image.dtype, np.integer) else 1.0
            params = DecodeParams(whiteLevel=white)
        if(image.dtype not in (np.uint8, np.uint16)):
            # OpenCV only demosaics integers
            scale = 65535/params.whiteLevel
            image = np.around(np.clip(image*scale, 0, 65535)).astype(np.uint16)
            blackLevel = None if params.blackLevel is None else tuple(float(v)*scale for v in params.blackLevel)
            params = DecodeParams(blackLevel=blackLevel, whiteLevel=65535.0)
        else:
            params = DecodeParams(blackLevel=params.blackLevel, whiteLevel=params.whiteLevel)
        return (FrameSource.demosaic(img=image, intent=intent), params)

    def workingSpaceFor(self, intent: DemosaicIntent | None) -> ColorSpace.WorkingSpace:
        """Returns the working space of the frames readFrame returns for an intent.

        Args:
            intent: The resolved demosaic choice.

        Returns:
            The first frame's working space, or one without primaries when the frames are demosaiced, since they
            are sensor data.
        """
        return self.firstFrame().workingSpace if intent is None else ColorSpace.UNTAGGED

    def close(self) -> None:
        """Releases any open file handles. The source reopens them if it is read again."""

class ImageSequenceSource(FrameSource):
    """A list of image files, one frame per file."""

    def __init__(self, path: str | list[str]) -> None:
        """Initializes the source.

        Args:
            path: An image file, or a list of image files (in any order).
        """
        self.paths = [path] if isinstance(path, str) else natsorted(path, alg=ns.IGNORECASE)

    @staticmethod
    def canOpen(path: str | list[str]) -> bool:
        """Accepts an image file OpenCV reads, or a list of one or more of them, but not raw camera files.

        Each file must be recognized without decoding its pixels.
        """
        paths = [path] if isinstance(path, str) else path
        return len(paths) > 0 and all(cv2.haveImageReader(file) and not CameraRawSource.isRaw(file) for file in paths)

    def filePaths(self) -> list[str]:
        return self.paths

    def frameCount(self) -> int:
        return len(self.paths)

    def readStored(self, index: int) -> StoredFrame | None:
        """Each file is described by its own color tags."""
        if(index >= len(self.paths)):
            return None
        file = self.paths[index]
        # Any depth and channel count, without alpha and with the EXIF orientation applied
        img = cv2.imread(file, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR)
        params, workingSpace = ImageSequenceSource.describeImage(file=file)
        return StoredFrame(ImageSequenceSource.toSupportedDtype(img), params, workingSpace)

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
    def describeImage(file: str) -> tuple[DecodeParams, ColorSpace.WorkingSpace]:
        """Describes a (non RAW, non FITS) image file from its color tags.

        Only PNG color chunks are read; ICC profiles and EXIF aren't, so JPEGs and TIFFs are always untagged.

        Args:
            file: Path of the file.

        Returns:
            The decode parameters and working space; untagged files are sRGB whatever their type, as PNG's
            specification defaults them, and close to TIFF's default of gamma 2.2.
        """
        transfer, workingSpace = (ImageSequenceSource.fromPNGChunks(ImageSequenceSource.pngChunks(file))
                                  or (ColorSpace.SRGB_CURVE, ColorSpace.SRGB))
        return (DecodeParams(transfer=transfer), workingSpace)

class CameraRawSource(ImageSequenceSource):
    """A list of camera RAW files, one frame per file, decoded and demosaiced by LibRaw."""

    # LibRaw's names of the algorithms it offers in its standard build
    LIBRAW_ALGORITHMS = {
        DemosaicAlgorithm.BILINEAR: "LINEAR",
        DemosaicAlgorithm.VNG: "VNG",
        DemosaicAlgorithm.PPG: "PPG",
        DemosaicAlgorithm.AHD: "AHD",
        DemosaicAlgorithm.DCB: "DCB",
        DemosaicAlgorithm.DHT: "DHT",
        DemosaicAlgorithm.AAHD: "AAHD",
    }

    # The pattern of the sensor, None if it isn't a mosaic; set with the first frame
    mosaicPattern: MosaicPattern | None = None

    @staticmethod
    def isRaw(file: str) -> bool:
        """Returns whether LibRaw recognizes a file, without decoding its pixels.

        Args:
            file: Path of the file.

        Returns:
            Whether it is a raw camera file.
        """
        try:
            with rawpy.imread(file):
                return True
        except Exception:
            return False

    @staticmethod
    def canOpen(path: str | list[str]) -> bool:
        """Accepts a raw camera file, or a list of one or more of them."""
        paths = [path] if isinstance(path, str) else path
        return len(paths) > 0 and all(CameraRawSource.isRaw(file) for file in paths)

    def readStored(self, index: int) -> StoredFrame | None:
        """Frames are in camera RGB with no curve, balanced as shot and with their black level still in."""
        if(index >= len(self.paths)):
            return None
        return CameraRawSource.decodeRaw(self.paths[index])

    def firstFrame(self) -> FirstFrame:
        """Describes the source from its first file's metadata, without decoding it."""
        if(self.first is None):
            with rawpy.imread(self.paths[0]) as raw:
                params, workingSpace = CameraRawSource.fromRAW(raw)
                self.mosaicPattern = CameraRawSource.fromRawPattern(raw)
            self.first = FirstFrame(params=params, workingSpace=workingSpace, isGray=False, is8Bit=False)
        return self.first

    @staticmethod
    def fromRawPattern(raw: object) -> MosaicPattern | None:
        """Reads the sensor's pattern from LibRaw's own description of it.

        Args:
            raw: The rawpy.RawPy.

        Returns:
            None if the sensor isn't a mosaic (monochrome, linear DNG, Foveon), UNSPECIFIED if its pattern isn't a
            2×2 one (X-Trans, 4-color sensors), else the pattern, e.g. RGGB.
        """
        if(raw.raw_pattern is None):
            return None
        letters = "".join(chr(raw.color_desc[i]) for i in raw.raw_pattern.flatten())
        return MosaicPattern(letters) if letters in MosaicPattern.__members__ else MosaicPattern.UNSPECIFIED

    def supportedDemosaicModes(self) -> list[DemosaicMode]:
        """LibRaw demosaics a 2×2 Bayer mosaic, which can't be left alone; the others are decided by LibRaw."""
        self.firstFrame()
        if(self.mosaicPattern in MOSAIC_PATTERNS):
            return [DemosaicMode.AUTO, DemosaicMode.ON]
        return [DemosaicMode.AUTO]

    def supportedMosaicPatterns(self) -> list[MosaicPattern | None]:
        """LibRaw reads the pattern from the file."""
        return [None]

    def supportedDemosaicAlgorithms(self) -> list[DemosaicAlgorithm | None]:
        """LibRaw has the algorithms for 2×2 Bayer mosaics."""
        self.firstFrame()
        if(self.mosaicPattern in MOSAIC_PATTERNS):
            return [None, *CameraRawSource.LIBRAW_ALGORITHMS]
        return [None]

    def resolveDemosaic(
        self,
        mode: DemosaicMode,
        pattern: MosaicPattern | None,
        algorithm: DemosaicAlgorithm | None,
        rememberedPattern: MosaicPattern | None = None,
    ) -> DemosaicIntent | None:
        """The frames are a mosaic when LibRaw says so, which demosaics them; under Auto it picks the algorithm."""
        self.firstFrame()
        if(self.mosaicPattern is None):
            return None
        if(mode != DemosaicMode.ON or algorithm not in self.supportedDemosaicAlgorithms()):
            algorithm = None
        return DemosaicIntent(self.mosaicPattern, algorithm or DemosaicAlgorithm.UNSPECIFIED)

    def readFrame(self, index: int, intent: DemosaicIntent | None, atRecordedDepth: bool = False) -> Frame | None:
        """LibRaw demosaics the frame, with the algorithm of the intent if it has one."""
        if(index >= len(self.paths)):
            return None
        algorithm = DemosaicAlgorithm.UNSPECIFIED if intent is None else intent.algorithm
        stored = CameraRawSource.decodeRaw(file=self.paths[index], algorithm=algorithm)
        return Frame(linearize(img=stored.image, params=self.firstFrame().params), self.differsFromFirst(stored))

    def workingSpaceFor(self, intent: DemosaicIntent | None) -> ColorSpace.WorkingSpace:
        """The camera's working space, whatever the intent: LibRaw's demosaic keeps it."""
        return self.firstFrame().workingSpace

    def fastPreview(self, intent: DemosaicIntent | None) -> np.ndarray | None:
        """The first file is decoded at half size."""
        # TODO: Read a raw camera file's embedded JPEG preview (rawpy's extract_thumb) instead,
        # falling back to the half size decode for files without one. It is far quicker, and it
        # keeps LibRaw's OpenMP code out of the main process.
        algorithm = DemosaicAlgorithm.UNSPECIFIED if intent is None else intent.algorithm
        stored = CameraRawSource.decodeRaw(file=self.paths[0], algorithm=algorithm, thumbnail=True)
        return linearize(img=stored.image, params=self.firstFrame().params)

    @staticmethod
    def decodeRaw(
        file: str,
        algorithm: DemosaicAlgorithm = DemosaicAlgorithm.UNSPECIFIED,
        thumbnail: bool = False,
    ) -> StoredFrame:
        """Decodes a raw camera file with LibRaw.

        Args:
            file: Path of the file.
            algorithm: The demosaic algorithm; UNSPECIFIED lets LibRaw choose.
            thumbnail: Whether a half size decode is enough.

        Returns:
            The stored image (BGR), its decode parameters and its working space.
        """
        options = {}
        if(algorithm != DemosaicAlgorithm.UNSPECIFIED):
            options["demosaic_algorithm"] = rawpy.DemosaicAlgorithm[CameraRawSource.LIBRAW_ALGORITHMS[algorithm]]
        with rawpy.imread(file) as raw:
            img = raw.postprocess(half_size=thumbnail,
                                  user_black=0,
                                  user_wb=list(CameraRawSource.asShotMultipliers(raw)),
                                  adjust_maximum_thr=0,
                                  output_color=rawpy.ColorSpace.raw,
                                  gamma=(1, 1),
                                  no_auto_bright=True,
                                  output_bps=16,
                                  **options)
            params, workingSpace = CameraRawSource.fromRAW(raw)
        return StoredFrame(cv2.cvtColor(img, cv2.COLOR_RGB2BGR), params, workingSpace)

    @staticmethod
    def rawRGBCam(raw: object) -> np.ndarray | None:
        """Returns the camera -> linear sRGB matrix of a RAW file, whose rows sum to 1.

        Args:
            raw: The rawpy.RawPy.

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
    def asShotMultipliers(raw: object) -> np.ndarray:
        """Returns the camera's as-shot white balance multipliers, normalized to green.

        Args:
            raw: The rawpy.RawPy.

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
    def fromRAW(raw: object) -> tuple[DecodeParams, ColorSpace.WorkingSpace]:
        """Describes a RAW file decoded linear in camera RGB, balanced as shot and with its black level in.

        Args:
            raw: The rawpy.RawPy.

        Returns:
            The sensor levels, the as-shot multipliers LibRaw applied and the D65 reference multipliers, and the
            camera's working space; UNTAGGED with a warning and no reference multipliers when LibRaw has no matrix.
        """
        black = [float(v) for v in raw.black_level_per_channel]
        if(len(black) >= 4 and raw.num_colors == 3):
            black = [black[0], (black[1] + black[3])/2, black[2]]
        asShot = CameraRawSource.asShotMultipliers(raw)
        # LibRaw scales the multipliers so that the smallest maps the white level to 16 bits
        applied = asShot/asShot.min()*65535/float(raw.white_level)
        params = DecodeParams(
            blackLevel=tuple(black[:3]),
            whiteLevel=float(raw.white_level),
            asShotMultipliers=tuple(float(v) for v in applied[:3]),
        )
        rgbCam = CameraRawSource.rawRGBCam(raw)
        if(rgbCam is not None):
            daylight = np.array(raw.daylight_whitebalance[:3], dtype=np.float64)
            params = dataclasses.replace(params, daylightMultipliers=tuple(float(v) for v in daylight/daylight[1]))
        else:
            warnings.warn(
                message="LibRaw has no color matrix for this camera, its colors are shown as sRGB",
                category=ColorSpace.ColorWarning,
            )
            return (params, ColorSpace.UNTAGGED)
        return (params, ColorSpace.WorkingSpace(toXYZ=ColorSpace.SRGB.toXYZ @ rgbCam))

class FitsSource(ImageSequenceSource):
    """A list of FITS files with image data, one frame per file, read by astropy."""

    @staticmethod
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
            params = FitsSource.fromFITS(header=header, data=img)
        if(img.dtype not in (np.uint8, np.uint16)):
            img = img.astype(np.float32)
        return StoredFrame(img, params, ColorSpace.UNTAGGED)

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
    def fromFITS(header: object, data: np.ndarray) -> DecodeParams:
        """Describes FITS data as sensor data: its Bayer pattern and levels.

        Args:
            header: The FITS header (astropy.io.fits.Header or a dict).
            data: The (first frame's) data.

        Returns:
            The decode parameters: no black level, since FITS has no standard keyword for one; white from DATAMAX or
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
        return DecodeParams(bayerPattern=bayerPattern, whiteLevel=white)

class VideoSource(FrameSource):
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

    def __getstate__(self) -> dict[str, str | list[int] | FirstFrame | None]:
        """Returns the state to pickle: the path, the timestamps and the first frame, without the open container."""
        return {"path": self.path, "timestamps": self.timestamps, "first": self.first}

    def __setstate__(self, state: dict[str, str | list[int] | FirstFrame | None]) -> None:
        """Restores a pickled source, which opens its container when first read.

        Args:
            state: The state from __getstate__.
        """
        self.path = state["path"]
        self.timestamps = state["timestamps"]
        self.first = state["first"]
        self.container = None

    @staticmethod
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
        self.params = DecodeParams(transfer=transfer)
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

    def readStored(self, index: int) -> StoredFrame | None:
        """Frames are converted to 16-bit BGR, whatever the stream's depth."""
        if(index == 0 and self.timestamps is None):
            # Decoded in a container of its own, without reading every packet's timestamp, so thumbnails are quick
            self.openContainer()
            with av.open(self.path) as container:
                videoFrame = next(container.decode(video=0), None)
            if(videoFrame is None):
                return None
            return StoredFrame(image=self.frameToImage(videoFrame), params=self.params, workingSpace=self.workingSpace)
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
                           params=self.params,
                           workingSpace=self.workingSpace)

    def _recordedBits(self) -> int | None:
        self.openContainer()
        return 8 if self.videoBits <= 8 else None

    def frameSize(self) -> tuple[int, int]:
        self.openContainer()
        return (self.width, self.height)

    def close(self) -> None:
        if(self.container is not None):
            self.container.close()
            self.container = None

